"""The epistemic trigger: reading ignorance off the weights, not off the string.

Design: docs_internal/curiosita_meccanismo.md. The autonomy loop used to
decide "the model does not know X" by looking at what the model SAID —
'non lo so.' or a question. The string generalises poorly and, when it does
appear, it names the wrong referent (the run of 2026-09-03: 63% honest
admissions on never-taught names, 0% right referent). What separates known
from unknown nouns almost perfectly is already in the weights: the margin
between the two most likely of the ten closed ontology classes on the prefix
'cos è un X? il X è ' (AUC 0.9896 before the L11-L12 rebuild, 0.9874 after).

This module is that reading and nothing else:

  * `class_posterior`  teacher-forced score of every class on the prefix,
                       length-normalised, softmaxed over the classes;
  * `verdict`          the posterior folded into a decision that NAMES the
                       referent by construction — the noun the prefix was
                       built around — instead of hoping the generated
                       question happens to mention it;
  * `pseudo_words`     syllabic reshuffles of the L0 inventory: free,
                       unlimited negatives that consume no frozen probe;
  * `calibrate`        the threshold, re-derived from the current weights
                       (known nouns vs pseudo-words), never a constant.

Contracts: no disk writes, no grading, no curriculum decisions. The scoring
goes through `model.forward` under `torch.no_grad()` and never through the
trainer's `generate`, which updates the `AffectState` at every step — the
reading must leave the state exactly as it found it (same contract as
`AffectState.peek_entropy`). The symbolic record (`words_rewarded`,
`untaught_words`) is ground truth for calibration and a log column: it is not
consulted here and must never become a decision branch upstream.

Class heads being single tokens is asserted by
tests/test_ontology_curiosity.py::test_the_class_words_are_single_tokens.
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                       # noqa: E402
from dynamic_model.ontology_oracle import article_for, guess_gender  # noqa: E402
from dynamic_model.exp_b.none_token import (              # noqa: E402
    NONE_TOKEN, SCORING_ONLY_TOKENS, mask_scoring_rows, row_id)


# ---------------------------------------------------------------------------
# The closed set and the prefix
# ---------------------------------------------------------------------------

def classes_of(lex) -> List[str]:
    """The ten closed classes, in a fixed order: keys and parents of the
    lexicon's ontology ('un animale' ... 'una cosa', 'un essere vivente')."""
    return sorted(set(lex.classes) | {v for v in lex.classes.values() if v})


def scoring_classes(lex, tok) -> List[str]:
    """The classes to score THIS tokenizer against: ten, or eleven.

    `<|NONE|>` is appended when the tokenizer has the row — and only then, so
    a checkpoint from before it existed reads exactly as it always did. The
    eleventh class changes what the separation statistic IS (see
    `separation`), which is why the choice belongs to one function and not to
    each call site.
    """
    return classes_of(lex) + ([NONE_TOKEN] if row_id(tok) is not None else [])


def separation(post: torch.Tensor, classes: Sequence[str]) -> tuple:
    """The scalar the threshold is compared against, and P(NONE) beside it.

    Returns `(score, p_none)`, `p_none` being None in ten-class mode. The
    score is always oriented the same way — HIGH means the model holds a
    class for the referent, low means it does not — so `calibrate`, the
    percentile bands, `drive.charge(tau - score)` and every log column work
    unchanged in both modes:

      * ten classes:  the margin between the top two classes. Ignorance
        shows up as two classes tying.
      * eleven:       1 - P(NONE). The margin rule INVERTS once the eleventh
        class is there — an unknown noun puts almost all the mass on
        `<|NONE|>` and so has a LARGE top-two margin (AUC 0.33 measured,
        `calibration` reporting OVERLAPPING), while 1 - P(NONE) separates
        known from never-seen nouns at AUC 0.98-1.00. Reading the margin
        after adopting the row is the one mistake that throws away a working
        detector for a broken metric.
    """
    if NONE_TOKEN in classes:
        p_none = float(post[list(classes).index(NONE_TOKEN)])
        return 1.0 - p_none, p_none
    top2 = torch.topk(post, 2).values
    return float(top2[0] - top2[1]), None


def prefix_for(noun: dict) -> str:
    """'cos è un cane? il cane è ' — the question the loop asks, followed by
    the opening of the gold answer, up to and excluding the class.

    The article follows `etp.ask_forms`: mass and unique nouns take the
    definite form ('cos è l acqua?'), the rest the indefinite one. The
    trailing space matters: the tokenizer splits on whitespace, so the class
    tokens appended after it are exactly the tokens training saw.
    """
    if noun.get("mass") or noun.get("uniq"):
        question = f"cos è {etp.phrase(noun)}?"
    else:
        question = f"cos è {etp.indef(noun)} {noun['w']}?"
    return f"{question} {etp.phrase(noun)} è "


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


@torch.no_grad()
def class_log_likelihoods(model, tok, noun: dict,
                          classes: Sequence[str]) -> torch.Tensor:
    """Length-normalised log-likelihood of each class after `prefix_for(noun)`.

    Teacher-forced: one batched forward, causal, right-padded — a class's
    tokens never see the padding that follows them. Returned on the CPU as a
    float tensor of shape (len(classes),). Dropout is switched off for the
    reading and the model's train/eval flag is restored afterwards.

    Any class string is accepted, `<|NONE|>` included — it is a single token
    and scores like the others. What the presence of that string changes is
    the vocabulary the likelihoods are normalised over: when it is NOT among
    the classes, the scoring-only rows are masked for the forward, so a
    ten-class read on a model that owns a `<|NONE|>` row gives the same
    numbers as the checkpoint that never had one. Without that mask the live
    row enters the per-position `log_softmax` and, because the classes have
    different token counts, length normalisation redistributes the shift
    unevenly: the ten-class margin degrades at frozen weights (0.942 ->
    0.925 measured), which is a measurement artefact and not a change of
    what the model knows.
    """
    prefix = prefix_for(noun)
    n_prompt = len(tok.encode(prefix))
    seqs = []
    for cls in classes:
        ids = tok.encode(prefix + cls)
        if len(ids) <= n_prompt:
            raise ValueError(f"class {cls!r} adds no token after the prefix")
        seqs.append(ids)
    max_len = getattr(model, "max_seq_len", None)
    if max_len is not None and max(len(s) for s in seqs) > max_len:
        raise ValueError(f"prefix for {noun['w']!r} exceeds the context ({max_len})")

    T = max(len(s) for s in seqs)
    device = _device_of(model)
    batch = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    batch = batch.to(device)

    scored = set(classes)
    hide = not any(t in scored for t in SCORING_ONLY_TOKENS)
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        if hide:
            with mask_scoring_rows(model, tok) as leftover:
                logits = model.forward(batch)
                if leftover:
                    logits = logits.clone()
                    logits[..., list(leftover)] = float("-inf")
        else:
            logits = model.forward(batch)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    logp = torch.log_softmax(logits.float(), dim=-1)

    out = torch.empty(len(seqs))
    for i, s in enumerate(seqs):
        # Token at position t is predicted by the logits at t-1.
        targets = torch.tensor(s[n_prompt:], dtype=torch.long, device=device)
        pred = logp[i, n_prompt - 1:len(s) - 1, :]
        out[i] = pred.gather(1, targets[:, None]).mean().cpu()
    return out


def class_posterior(model, tok, noun: dict, classes: Sequence[str]) -> torch.Tensor:
    """Softmax over the classes of their length-normalised log-likelihoods.

    A distribution over the closed set: what the weights say the noun IS,
    with no sampling and no generated text in between.
    """
    return torch.softmax(class_log_likelihoods(model, tok, noun, classes), dim=0)


@dataclass
class EpistemicVerdict:
    """The trigger's reading on one noun.

    `referent` is the noun the prefix was built around — named by
    construction, which is the whole point: a question generated by the model
    may mention another noun, this cannot. `ignorant` is `margin < threshold`
    and nothing else; the threshold comes from `calibrate` on the same weights.

    `margin` is the separation statistic of `separation`, which is the
    top-two margin in ten-class mode and 1 - P(NONE) in eleven-class mode.
    The name is kept because every consumer — the session log column, the
    drive's charge, epistemic_report — treats it as "how far this noun is
    from holding a class", and that meaning is identical in both modes.
    `p_none` is the raw eleventh-class mass, None when the row is absent.
    """
    referent: str
    margin: float
    top_class: str
    p_top: float
    entropy: float          # normalised to [0, 1] by log(len(classes))
    threshold: float
    ignorant: bool
    posterior: Dict[str, float] = field(default_factory=dict)
    p_none: Optional[float] = None

    def as_log(self) -> dict:
        """The columns the session log carries (a reading, not a decision)."""
        out = {"margin": round(self.margin, 4), "p_top": round(self.p_top, 4),
               "top_class": self.top_class, "tau": round(self.threshold, 4)}
        if self.p_none is not None:
            out["p_none"] = round(self.p_none, 4)
        return out


def verdict(model, tok, noun: dict, classes: Sequence[str],
            threshold: float) -> EpistemicVerdict:
    post = class_posterior(model, tok, noun, classes)
    top2 = torch.topk(post, 2)
    p = post.clamp_min(1e-12)
    ent = float(-(p * p.log()).sum()) / math.log(len(classes)) if len(classes) > 1 else 0.0
    score, p_none = separation(post, classes)
    return EpistemicVerdict(
        referent=noun["w"], margin=score,
        top_class=classes[int(top2.indices[0])], p_top=float(top2.values[0]),
        entropy=ent, threshold=float(threshold), ignorant=score < threshold,
        posterior={c: round(float(post[i]), 4) for i, c in enumerate(classes)},
        p_none=p_none)


def margins(model, tok, nouns: Sequence[dict], classes: Sequence[str]) -> List[float]:
    """The separation statistic for each noun — `separation`, not necessarily
    a margin. The calibration bands and every AUC are built on this list, so
    it has to switch mode with the classes exactly as `verdict` does."""
    out = []
    for n in nouns:
        post = class_posterior(model, tok, n, classes)
        out.append(separation(post, classes)[0])
    return out


# ---------------------------------------------------------------------------
# Pseudo-words: negatives that cost nothing
# ---------------------------------------------------------------------------

_SYLLABLE = re.compile(r"^[b-df-hj-np-tv-z]{1,2}[aeiou]$")


def l0_inventory(lang: str = "it") -> List[str]:
    """The syllables of the level-0 sound files ('ma', 'pa', 'bu', 'tra'...).

    Returned sorted, so the reshuffle is a function of the seed alone.
    """
    root = os.path.join(etp.ROOT, "training_files", lang, "0")
    seen = set()
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as f:
            for token in re.findall(r"[a-zàèéìòù]+", f.read().lower()):
                if _SYLLABLE.match(token):
                    seen.add(token)
    return sorted(seen)


def _known_words(lang: str) -> set:
    """Every word the curriculum could have taught, so no pseudo-word is one.

    The lexicon (nouns, verbs, adjectives, the held-out pools) plus every
    alphabetic token of the level-0 files — 'casa' is ca+sa and is in both.
    """
    d = etp.load_lexicon(lang)
    words = set()
    for key in ("nouns", "unknown_nouns", "bare_unknown_nouns"):
        words.update(n["w"] for n in d.get(key, []))
    for v in d.get("verbs", []):
        words.update(str(x) for x in v.values() if isinstance(x, str))
    for a in d.get("adjectives", []):
        words.update(str(x) for x in a.values() if isinstance(x, str))
    root = os.path.join(etp.ROOT, "training_files", lang, "0")
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.endswith(".txt"):
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    words.update(re.findall(r"[a-zàèéìòù]+", f.read().lower()))
    return words


def pseudo_words(rng, n: int, lang: str = "it",
                 avoid: Optional[set] = None) -> List[dict]:
    """`n` nouns that do not exist, shaped like the ones that do.

    Two or three syllables of the L0 inventory, ending in a gender-bearing
    vowel so `indef`/`phrase` render a well-formed prompt ('cos è una
    tacoba? la tacoba è '). Anything the curriculum could have taught is
    excluded, so a low margin on these is ignorance of the WORD, not of the
    shape. `rng` is a `random.Random`: the same seed gives the same list, and
    a different seed a fresh one — the supply is unlimited.
    """
    inv = l0_inventory(lang)
    if not inv:
        raise RuntimeError(f"no level-0 inventory for {lang!r}")
    banned = set(avoid or ()) | _known_words(lang)
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < 50 * max(1, n):
        guard += 1
        k = rng.choice((2, 3))
        w = "".join(rng.choice(inv) for _ in range(k))
        if w in seen or w in banned or len(w) < 4 or w[-1] not in "ao":
            continue
        seen.add(w)
        g = guess_gender(w) or "m"
        out.append({"w": w, "art": article_for(w, g), "g": g, "pseudo": True})
    if len(out) < n:
        raise RuntimeError(f"could not draw {n} pseudo-words (got {len(out)})")
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability that a random positive outranks a random negative
    (Mann-Whitney; ties count one half). 0.5 is chance, 1.0 is separation."""
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for q in negatives:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(positives) * len(negatives))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]; no numpy needed."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass
class Calibration:
    tau: float
    p95_pseudo: float
    p05_known: float
    auc: float
    n_known: int
    n_pseudo: int
    known: List[float] = field(default_factory=list)
    pseudo: List[float] = field(default_factory=list)

    @property
    def separated(self) -> bool:
        """True when the two bands do not overlap at the chosen percentiles."""
        return self.p95_pseudo < self.p05_known


def calibration(model, tok, lex, n_pseudo: int = 32, seed: int = 0,
                classes: Optional[Sequence[str]] = None) -> Calibration:
    """Threshold and the numbers behind it, from THESE weights.

    Positives are the lexicon's classified nouns; negatives are pseudo-words
    drawn from `seed`. τ is the midpoint between the 95th percentile of the
    pseudo-word margins and the 5th percentile of the known ones — the two
    bands' inner edges — so it moves with the weights after every dream and
    is never carried over from another checkpoint. Frozen probes never enter
    this: they are what the trigger is measured on.

    `classes` defaults to `scoring_classes(lex, tok)`, so a tokenizer that
    owns the `<|NONE|>` row is calibrated on 1 - P(NONE) and one that does
    not is calibrated on the top-two margin, without the caller choosing.
    """
    import random
    classes = list(classes or scoring_classes(lex, tok))
    known = lex.classified()
    pseudo = pseudo_words(random.Random(seed), n_pseudo, getattr(lex, "lang", "it"))
    mk = margins(model, tok, known, classes)
    mp = margins(model, tok, pseudo, classes)
    hi_pseudo, lo_known = percentile(mp, 95), percentile(mk, 5)
    return Calibration(tau=(hi_pseudo + lo_known) / 2.0,
                       p95_pseudo=hi_pseudo, p05_known=lo_known,
                       auc=auc(mk, mp), n_known=len(mk), n_pseudo=len(mp),
                       known=mk, pseudo=mp)


def calibrate(model, tok, lex, n_pseudo: int = 32, seed: int = 0) -> float:
    """The threshold alone; see `calibration` for the record behind it."""
    return calibration(model, tok, lex, n_pseudo=n_pseudo, seed=seed).tau
