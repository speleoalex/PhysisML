"""
AffectState — affective state of the model.

Four scalar variables updated with EWMA (alpha=0.1):

  confidence : certainty about the current response  [0, 1]
               measured as 1 - entropy(softmax) / log(V)
  ignorance  : how much the model "does not know"    [0, 1]
               measures how many embeddings have low norm
  pleasure   : memory of positive feedback           [0, 1]
  pain       : memory of negative feedback           [0, 1]
  fear       : uncertainty + pain combined           [0, 1]
               = 0.5 * pain + 0.5 * (1 - confidence)

Biological prior: the model is born ignorant (ignorance=0.9)
and neutral (pleasure=0.5, pain=0.0). No memory of past pain.

Key distinction:
  - confidence / ignorance update on every forward pass (no feedback needed)
  - pleasure / pain update ONLY with explicit external feedback
  - fear is derived from the others — no direct update
"""
import numpy as np
from dataclasses import dataclass, field
import json
import os
import re
from collections import deque
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))
import torch


@dataclass
class AffectSnapshot:
    step:       int
    confidence: float
    ignorance:  float
    pleasure:   float
    pain:       float
    fear:       float


# Word forms for the curiosity memory: letters only, accents included, so
# punctuation and digits never enter words_rewarded.
_WORD_RE = re.compile(r"[a-zàèéìòùáíóúü]+", re.IGNORECASE)


class AffectState:

    ALPHA = 0.1    # EWMA decay — slow learning, long memory
    NORM_THRESHOLD = 0.05  # minimum norm below which an embedding is "unknown"

    # Novelty drive — simulates the dopaminergic peak for novel stimuli.
    # The bonus decays with sqrt(n_encounters): first time = full,
    # 4th time = 50%, 100th time = 10%.
    NOVELTY_WEIGHT   = 0.3   # maximum novelty bonus weight on pleasure
    ACTIVATION_BONUS = 0.6   # peak when a new vocabulary slot is activated

    def __init__(self):
        # Biological prior
        self.confidence: float = 0.1
        self.ignorance:  float = 0.9
        self.pleasure:   float = 0.5
        self.pain:       float = 0.0
        self.fear:       float = 0.45
        self.step:       int   = 0
        self.history: deque = deque(maxlen=1000)
        # Novelty drive: counts how many times each token/bigram has appeared
        # in a response with positive feedback
        self.token_encounter_count: dict = {}
        self.bigram_count: dict = {}  # (id_a, id_b) -> int
        # Curiosity memory: word forms that have appeared in a target trained
        # with positive feedback. That INCLUDES the teacher's gold answer,
        # which is deliberate and is what the L12 dialogue needs — a word
        # counts as known from the moment the teacher explains it, so the model
        # is penalised for asking about it a second time. It is not a record of
        # what the model produced on its own.
        #
        # Kept SEPARATE from token_encounter_count on purpose: that dict also
        # drives the novelty reward, and persisting it would change the
        # training signal of every level, not just the one that measures
        # curiosity. This set is persisted (see save_memory), never read by the
        # reward.
        self.words_rewarded: set = set()
        # Words that carry no content and must not count as knowledge.
        # Injected by the caller (train_curriculum passes its STOP_WORDS) so
        # this module stays language-agnostic. Empty by default, which still
        # discriminates, only more weakly.
        self.function_words: set = set()

    # ------------------------------------------------------------------
    # Automatic updates (from forward pass)
    # ------------------------------------------------------------------

    def update_from_logits(self, logits: torch.Tensor, vocab_size: int) -> None:
        """
        Update confidence from a logits vector (last token).
        logits: (V,) or (T, V) — takes the last token if 2D
        """
        if logits.dim() > 1:
            logits = logits[-1]

        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            # Normalised entropy: 0 = fully certain, 1 = uniform
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            max_entropy = np.log(max(vocab_size, 2))
            new_conf = 1.0 - entropy / max_entropy

        self.confidence = (1 - self.ALPHA) * self.confidence + self.ALPHA * new_conf
        self._update_fear()

    def update_from_weights(self, embedding_weight: torch.Tensor,
                            active_vocab_size: Optional[int] = None) -> None:
        """
        Update ignorance from the norm of embedding vectors.

        embedding_weight: (V, d_model)
        active_vocab_size: rows above this are DORMANT slots, pre-allocated for
          vocabulary growth and initialised to zero. Counting them measured the
          allocated capacity, not the knowledge: on the 26 August 2026
          checkpoint (9000 slots, 2547 active) the fraction of low-norm rows
          was 0.717 over the whole matrix and 0.000 over the active ones. The
          first is a constant permanently above IGNORANCE_THRESHOLD (0.7), so
          the "don't know" gate in AffectModulator would have fired on every
          token had its token id ever been wired up.

        Both readings are useless as a per-item signal — one is a constant, the
        other is zero once training has touched every active row. What the
        model does not know about a SPECIFIC word is local_ignorance().
        """
        with torch.no_grad():
            if active_vocab_size is not None:
                embedding_weight = embedding_weight[:active_vocab_size]
            norms = embedding_weight.norm(dim=-1)
            frac_low = (norms < self.NORM_THRESHOLD).float().mean().item()

        self.ignorance = (1 - self.ALPHA) * self.ignorance + self.ALPHA * frac_low

    # ------------------------------------------------------------------
    # Local, per-item ignorance — the signal curiosity is gated on
    # ------------------------------------------------------------------

    def register_rewarded_words(self, text: str, feedback: float) -> None:
        """Remember the words of a target trained with positive feedback.

        Word forms, not token ids, and that difference is the point: see
        word_ignorance().
        """
        if feedback <= 0 or not text:
            return
        for w in _WORD_RE.findall(text.lower()):
            self.words_rewarded.add(w)

    def word_ignorance(self, text: str) -> float:
        """
        How much of THIS input the model has never been taught.

        Fraction of the distinct CONTENT words in `text` the model has never
        been taught (absent from words_rewarded). Unlike the global `ignorance` it varies per
        prompt, which is what a curiosity trigger needs: 'questo è un ragno'
        must read as unknown while 'questo è un gatto' reads as known.

        WHY WORDS AND NOT TOKEN IDS. The first version of this counted token
        ids with token_encounter_count == 0, and measurement killed it twice
        over. With one unknown noun in a prompt the fraction came out at 0.20 —
        diluted by articles, the space token and the copula, far under the
        gate's 0.5 threshold. Worse, the ordering was decided by tokenization
        rather than by knowledge: the taught unknown 'ragno' (one token) scored
        0.20 while the held-out probe 'lumaca' (three pieces: lu|ma|ca) scored
        0.43, and pieces like 'ma' and 'ca' are L0 syllables the model produces
        constantly, so the same word's score moved with the vocabulary. Words
        are stable under retraining and are also the unit LocalTeacher reasons
        in (its known/unknown sets hold nouns), so the gate and the reward now
        agree on what 'unknown' means.

        Returns 0.0 when there is no content word — nothing unknown in nothing.
        """
        if not text:
            return 0.0
        words = {w for w in _WORD_RE.findall(text.lower())
                 if len(w) > 1 and w not in self.function_words}
        if not words:
            return 0.0
        unseen = sum(1 for w in words if w not in self.words_rewarded)
        return unseen / len(words)

    def untaught_words(self, text: str) -> set:
        """The distinct content words in `text` that have never been taught."""
        if not text:
            return set()
        return {w for w in _WORD_RE.findall(text.lower())
                if len(w) > 1 and w not in self.function_words
                and w not in self.words_rewarded}

    def ask_drive(self, text: str) -> float:
        """Drive to ask about THIS input, in [0, 1] — what the gate receives.

        Presence, not proportion: 1.0 when the input holds at least one untaught
        content word, then damped by (1 - fear) because a frightened model does
        not explore.

        WHY NOT word_ignorance DIRECTLY. It is a fraction, and measured on the
        real L12 prompt 'il cane è un animale, questo è un ragno' it comes out
        at 0.25 — one untaught word among four content words — against a gate
        threshold of 0.5, so the gate would never fire however ignorant the
        model is. Worse, the value would move with prompt length: the same
        unknown noun in a shorter prompt scores higher, so the threshold would
        be calibrating sentence length rather than knowledge. The question the
        gate asks is 'is there something here I was never told about?', which
        has a yes/no answer; how MUCH of the sentence is new is a different
        question, and word_ignorance stays as the measure that answers it.
        """
        if not self.untaught_words(text):
            return 0.0
        return float(np.clip(1.0 - self.fear, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Persistence of the curiosity memory
    # ------------------------------------------------------------------

    def save_memory(self, path: str) -> None:
        """Persist the curiosity memory next to the checkpoints.

        Each build phase is a SEPARATE python process, so without this the set
        is empty at the start of every session and word_ignorance returns 1.0
        for every prompt — the gate then fires on everything, which is the
        'asks about everything, that is a tic and not curiosity' failure the
        level is supposed to avoid.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"words_rewarded":
                       sorted(self.words_rewarded)}, f,
                      ensure_ascii=False)

    def load_memory(self, path: str) -> int:
        """Restore the curiosity memory. Returns how many words were loaded.

        A missing file is not an error: the first session of a build starts
        with no memory by definition.
        """
        if not path or not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return 0
        words = data.get("words_rewarded") or []
        self.words_rewarded.update(w for w in words if isinstance(w, str))
        return len(self.words_rewarded)

    def peek_entropy(self, logits: torch.Tensor, vocab_size: int) -> float:
        """
        Normalised entropy of THIS step, without touching the EWMA.

        `confidence` is a slow average over the last many forward passes
        (ALPHA=0.1), so it cannot say whether the model is uncertain about the
        token it is about to emit. This can, and it leaves the affective state
        unchanged — a measurement, not an update.

        0.0 = fully certain, 1.0 = uniform over the vocabulary.
        """
        if logits.dim() > 1:
            logits = logits[-1]
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
        return entropy / np.log(max(vocab_size, 2))

    @property
    def curiosity(self) -> float:
        """
        Drive to ask, in [0, 1].

        Not ignorance alone: a frightened model does not explore. Multiplying
        by (1 - fear) keeps the model quiet while it is being punished, and
        `fear` already folds in low confidence and remembered pain.

        This is the GLOBAL reading, for logging and for a gate with no prompt
        in hand. The per-prompt one is word_ignorance(prompt) * (1 - fear),
        which is what the ask gate actually receives.
        """
        return float(np.clip(self.ignorance * (1.0 - self.fear), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Updates from external feedback
    # ------------------------------------------------------------------

    def update_from_feedback(self, feedback: float) -> None:
        """
        feedback: float in [-1.0, +1.0]
          +1.0 = explicit approval
           0.0 = neutral (passive observation — no update)
          -1.0 = explicit disapproval
        """
        if feedback == 0.0:
            return
        if feedback > 0:
            self.pleasure = (1 - self.ALPHA) * self.pleasure + self.ALPHA * feedback
        else:
            self.pain = (1 - self.ALPHA) * self.pain + self.ALPHA * abs(feedback)
        self._update_fear()

    def update_from_novelty(self, token_ids: list, feedback: float) -> float:
        """
        Novelty drive — dopaminergic bonus for new or rarely-seen tokens.

        Two novelty levels:
          1. Single tokens: bonus for new words (first time "cane" → full)
          2. Bigrams: bonus for new SEQUENCES (first time "il cane" → extra bonus)

        Biological: dopaminergic neurons fire stronger for unexpected rewards.
        "il cane" as a sequence is richer than "cane" alone —
        rewards correct syntactic order, not just vocabulary.

        Returns: total novelty bonus applied
        """
        if feedback <= 0 or not token_ids:
            return 0.0

        total_bonus = 0.0

        # 1. Single tokens (unchanged)
        for tid in set(token_ids):
            count = self.token_encounter_count.get(tid, 0)
            bonus = self.NOVELTY_WEIGHT * feedback / (count + 1) ** 0.5
            total_bonus += bonus
            self.token_encounter_count[tid] = count + 1

        # 2. Bigrams: reward new ordered sequences
        #    Weighted 0.5× against single tokens — an extra, not a dominant, term
        BIGRAM_WEIGHT = self.NOVELTY_WEIGHT * 0.5
        for i in range(len(token_ids) - 1):
            bg = (token_ids[i], token_ids[i + 1])
            count = self.bigram_count.get(bg, 0)
            total_bonus += BIGRAM_WEIGHT * feedback / (count + 1) ** 0.5
            self.bigram_count[bg] = count + 1

        # Cap total bonus (2× single-token cap to accommodate bigram bonus)
        total_bonus = min(total_bonus, self.NOVELTY_WEIGHT * 2)
        self.pleasure = min(1.0, self.pleasure + self.ALPHA * total_bonus)
        self._update_fear()
        return total_bonus

    def register_token_activation(self, n_new: int) -> None:
        """
        Pleasure spike when new vocabulary slots are activated during N2 dream.
        Simulates the excitement of a child discovering they can say a new word.

        Bonus is proportional to the number of new tokens (capped).
        """
        if n_new <= 0:
            return
        bonus = self.ACTIVATION_BONUS * min(1.0, n_new / 10.0)
        self.pleasure = min(1.0, self.pleasure + bonus)
        self._update_fear()

    # ------------------------------------------------------------------

    def _update_fear(self) -> None:
        self.fear = 0.5 * self.pain + 0.5 * (1.0 - self.confidence)
        self.fear = float(np.clip(self.fear, 0.0, 1.0))

    def snapshot(self) -> AffectSnapshot:
        s = AffectSnapshot(
            step=self.step,
            confidence=self.confidence,
            ignorance=self.ignorance,
            pleasure=self.pleasure,
            pain=self.pain,
            fear=self.fear,
        )
        self.history.append(s)
        self.step += 1
        return s

    def __repr__(self) -> str:
        return (f"AffectState(conf={self.confidence:.2f}, ign={self.ignorance:.2f}, "
                f"pleas={self.pleasure:.2f}, pain={self.pain:.2f}, fear={self.fear:.2f})")
