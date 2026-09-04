"""
ontology_oracle — what the model gets back when it admits it does not know.

The autonomy loop needs a source of answers that is not a human. This is it:
it asks a local LLM which class an unknown noun belongs to, and turns the
verdict into curriculum material.

Two rules make it safe enough to train on, and neither is negotiable:

  1. **Closed answer set.** The class must be one of the ten the lexicon
     already uses. Free text is refused, not repaired. A class the model has
     never been taught cannot be learned from one example anyway, and an
     invented one ('un mammifero') would enter the gold as a token the
     tokenizer may not even have.

  2. **The LLM's text never becomes the gold.** Only its *choice* does. The
     sentence the model is taught is rendered from the same templates the
     curriculum uses (`g_cos_e`, `g_is_a_si`, `g_is_a_no` in
     scripts/expand_teacher_pools.py), so accepted material is
     indistinguishable in form from hand-written material. The article is
     derived by code from the gender, never copied from the answer.

What it deliberately does NOT do: propose new classes, judge whether the model
answered well, or write anything to disk. Grading and persistence belong to the
loop.

Usage:
    from dynamic_model.ontology_oracle import OntologyOracle
    o = OntologyOracle(lang="it")
    v = o.ask({"w": "falco", "art": "il", "g": "m"})
    if v.ok:
        for t in o.material_for(v.noun, v.cls):
            ...   # {"prompt": ..., "expected": ..., "noun": ...}
"""
import json
import os
import re
import sys
from typing import List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_model import llm_backend                                # noqa: E402
import expand_teacher_pools as etp                                   # noqa: E402

SYSTEM = """You classify Italian nouns into a fixed set of classes.

Reply with EXACTLY ONE of these strings, copied verbatim, and nothing else:
{classes}

Rules:
- No explanation, no punctuation, no article changes. Only the string.
- Choose the most specific class that fits.
- If the word is a living creature that is not a person, it is 'un animale'.
- If you do not recognise the word, reply: sconosciuto
"""

GENDER_SYSTEM = """You report the grammatical gender of an Italian noun.
Reply with exactly one letter: m for masculine, f for feminine. Nothing else."""

# Words that mean "I don't know" coming back from the oracle. Kept explicit:
# an oracle that declines is a legitimate outcome and must not be parsed as a
# malformed answer, because the loop treats the two differently.
_UNKNOWN_MARKERS = ("sconosciuto", "unknown", "non lo so", "i don't know")


def article_for(word: str, gender: str) -> str:
    """The definite article, derived from the word and its gender.

    Never taken from the LLM: a wrong class is caught by the closed set, but a
    wrong article would sail straight into the gold as well-formed Italian.
    Apostrophes are spaces in this corpus, so the elided form is a bare 'l'.
    """
    w = (word or "").lower()
    if not w:
        return ""
    if w[0] in "aeiou":
        return "l"
    if gender == "f":
        return "la"
    # Masculine 'lo' before s+consonant, z, gn, ps, pn, x, y — 'lo sgabello'.
    if re.match(r"^(s[^aeiou]|z|gn|ps|pn|x|y)", w):
        return "lo"
    return "il"


def guess_gender(word: str) -> Optional[str]:
    """Gender from the ending, or None when the ending does not settle it.

    '-o' is masculine and '-a' feminine with few enough exceptions to be
    usable; '-e' settles nothing ('il pettine' against 'la noce'), and that is
    exactly the case the oracle is asked about instead of guessed.
    """
    w = (word or "").lower()
    if w.endswith(("o", "ore", "ino", "etto")):
        return "m"
    if w.endswith(("a", "zione", "sione", "trice")):
        return "f"
    return None


def normalize_class(text: str, classes) -> Optional[str]:
    """An answer mapped onto the closed set, or None.

    Tolerant about the shape of the reply (case, terminator, a stray leading
    'è'), strict about its content: the head must match a known class. A bare
    head ('animale') is accepted because it identifies the class uniquely —
    the article is then the lexicon's, not the LLM's.
    """
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = t.strip(" .!?,;:\"'`")
    t = re.sub(r"^(è|e|classe|class)\s*:?\s*", "", t).strip()
    if not t:
        return None
    for c in classes:
        if t == c:
            return c
    # Bare head: 'animale' -> 'un animale', but only if no other class shares it
    heads = {}
    for c in classes:
        heads.setdefault(etp.head(c), []).append(c)
    match = heads.get(t)
    if match and len(match) == 1:
        return match[0]
    return None


class Verdict:
    """The outcome of one oracle call, including why it was refused."""

    def __init__(self, noun: dict, cls: str = None, raw: str = "",
                 reason: str = ""):
        self.noun   = noun
        self.cls    = cls
        self.raw    = raw
        self.reason = reason

    @property
    def ok(self) -> bool:
        return bool(self.cls)

    def __repr__(self) -> str:
        if self.ok:
            return f"Verdict({self.noun.get('w')} -> {self.cls})"
        return f"Verdict({self.noun.get('w')} refused: {self.reason})"


class OntologyOracle:
    """Asks a local LLM for a noun's class; renders curriculum material."""

    def __init__(self, lang: str = "it", model: str = None, backend=None,
                 lexicon: dict = None, timeout: int = 20):
        d = lexicon or etp.load_lexicon(lang)
        self.lang    = lang
        self.lex     = etp.Lex(d)
        self.timeout = timeout
        # The ten classes the curriculum says out loud: both ends of the
        # hypernym chain, because 'una cosa' and 'un essere vivente' are
        # answers too (g_iperonimo teaches them).
        self.classes = sorted(set(self.lex.classes.keys()) |
                              {v for v in self.lex.classes.values() if v})
        self.backend = backend or llm_backend.detect()
        self.model   = model or llm_backend.resolve_model(model, self.backend)
        self._system = SYSTEM.format(classes="\n".join(self.classes))

    # ── availability ──────────────────────────────────────────────────────
    def is_available(self) -> bool:
        return bool(self.backend and self.model)

    def status(self) -> str:
        return llm_backend.status_line(self.model)

    # ── the question ──────────────────────────────────────────────────────
    def ask(self, noun) -> Verdict:
        """Classify one noun. `noun` is a word or a {w, art, g} dict."""
        n = dict(noun) if isinstance(noun, dict) else {"w": str(noun)}
        w = (n.get("w") or "").strip().lower()
        if not w:
            return Verdict(n, reason="empty noun")
        n["w"] = w
        if not self.is_available():
            return Verdict(n, reason="no local LLM reachable")

        # A noun the lexicon already classifies must not come from the oracle:
        # the curriculum's answer is the one with authority, and letting the
        # LLM overwrite it is how a build acquires two golds for one prompt.
        known = next((x for x in self.lex.nouns if x["w"] == w), None)
        if known and self.lex.cls_of(known):
            return Verdict(n, reason="already classified in the lexicon")

        raw = llm_backend.generate(
            f"Noun: {w}\nClass:", system=self._system, model=self.model,
            timeout=self.timeout, max_tokens=12, temperature=0.0,
            backend=self.backend)
        if not raw:
            return Verdict(n, raw=raw, reason="oracle did not answer")
        low = raw.strip().lower()
        if any(m in low for m in _UNKNOWN_MARKERS):
            return Verdict(n, raw=raw, reason="oracle declined")
        cls = normalize_class(raw, self.classes)
        if not cls:
            return Verdict(n, raw=raw, reason=f"outside the class set: {raw!r}")

        if not n.get("g"):
            n["g"] = self.gender_of(w)
        if not n.get("art"):
            n["art"] = article_for(w, n["g"])
        return Verdict(n, cls=cls, raw=raw)

    def gender_of(self, word: str) -> str:
        """Ending first, oracle second, masculine as the last resort."""
        g = guess_gender(word)
        if g:
            return g
        raw = llm_backend.generate(
            f"Noun: {word}\nGender:", system=GENDER_SYSTEM, model=self.model,
            timeout=self.timeout, max_tokens=4, temperature=0.0,
            backend=self.backend)
        letter = (raw or "").strip().lower()[:1]
        return letter if letter in ("m", "f") else "m"

    # ── the material ──────────────────────────────────────────────────────
    def material_for(self, noun: dict, cls: str, rng=None) -> List[dict]:
        """
        The curriculum material for one acquired fact, in the curriculum's own
        shapes: the open question in every phrasing, the affirmative yes/no,
        a negative, and the two-clause introduction. Rendered from the same
        helpers the hand-written pools use, so nothing here can drift from them
        without the tests noticing.

        The set is not a matter of taste: acquiring a noun RETRACTS every shape
        that treated it as unknown (dynamic_model/retraction.py), so anything
        retracted and not replaced here is a shape the model was supervised on
        yesterday and is not supervised on today. Two of them exist for exactly
        that reason:

          * the two-clause introduction, because L12 steps A and G taught
            'questo è un falco' -> 'cos è un falco?' and once the class is
            known the answer is the assertion, which is what step B teaches
            about every noun the model knows;
          * the negative on 'un animale' specifically, because step E's yes/no
            is always 'X è un animale?' — a randomly chosen wrong class would
            leave that one prompt with no gold at all whenever the acquired
            class is not 'un animale'.

        `noun` in the returned targets is the grader's keyword, which must be a
        word of the ANSWER — the same convention as the pools, and what
        validate_teacher_configs.py checks.
        """
        import random as _rnd
        rng  = rng or _rnd.Random(0)
        gold = f"{etp.phrase(noun)} è {cls}."
        out  = [{"prompt": p, "expected": gold, "article": noun["art"],
                 "noun": etp.head(cls)}
                for p in etp.ask_forms(noun, etp.head(cls))]
        out.append({"prompt":   f"{etp.phrase(noun)} è {cls}?",
                    "expected": f"sì, {etp.phrase(noun)} è {cls}.",
                    "article":  noun["art"], "noun": etp.head(cls)})
        wrong = ("un animale" if cls != "un animale"
                 else self.lex.wrong_cls_for(rng, {"w": noun["w"], "cls": cls}))
        if wrong:
            out.append({"prompt":   f"{etp.phrase(noun)} è {wrong}?",
                        "expected": f"no, {etp.phrase(noun)} è {cls}.",
                        "article":  noun["art"], "noun": etp.head(cls)})
        anchor = self.lex.anchor_of_other_class(rng, cls)
        if anchor:
            out.append({
                "prompt": f"{etp.phrase(anchor)} è {self.lex.cls_of(anchor)}, "
                          f"{etp.intro(noun)}",
                "expected": gold, "article": noun["art"],
                "noun": etp.head(cls)})
        return out


if __name__ == "__main__":
    o = OntologyOracle()
    print("oracle:", o.status())
    print("classes:", len(o.classes))
    words = sys.argv[1:] or [n["w"] for n in o.lex.bare_unknown]
    for w in words:
        v = o.ask(w)
        print(f"  {w:14} {v}")
        if v.ok:
            for t in o.material_for(v.noun, v.cls)[:2]:
                print(f"       {t['prompt']!r} -> {t['expected']!r}")
