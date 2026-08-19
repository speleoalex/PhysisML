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


class AffectState:

    ALPHA = 0.1    # EWMA decay — slow learning, long memory
    NORM_THRESHOLD = 0.05  # minimum norm below which an embedding is "unknown"

    # Novelty drive — simulates the dopaminergic peak for novel stimuli.
    # The bonus decays with sqrt(n_encounters): first time = full,
    # 4th time = 50%, 100th time = 10%.
    NOVELTY_WEIGHT   = 0.3   # maximum novelty bonus weight on pleasure
    ACTIVATION_BONUS = 0.6   # picco quando un nuovo slot vocabolario viene attivato

    def __init__(self):
        # Prior biologico
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

    def update_from_weights(self, embedding_weight: torch.Tensor) -> None:
        """
        Update ignorance from the norm of embedding vectors.
        embedding_weight: (V, d_model)
        """
        with torch.no_grad():
            norms = embedding_weight.norm(dim=-1)
            frac_low = (norms < self.NORM_THRESHOLD).float().mean().item()

        self.ignorance = (1 - self.ALPHA) * self.ignorance + self.ALPHA * frac_low

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

        # 1. Token singoli (invariato)
        for tid in set(token_ids):
            count = self.token_encounter_count.get(tid, 0)
            bonus = self.NOVELTY_WEIGHT * feedback / (count + 1) ** 0.5
            total_bonus += bonus
            self.token_encounter_count[tid] = count + 1

        # 2. Bigrammi: premia sequenze ordinate nuove
        #    Peso 0.5× rispetto ai token singoli — contributo extra ma non dominante
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
