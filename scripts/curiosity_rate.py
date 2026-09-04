#!/usr/bin/env python3
"""
Measure whether the model ASKS when it does not know, and does NOT when it does.

The number that matters is the GAP between two ask-rates on the same prompt
shape:

  ask-rate on nouns it has never met   should be high
  ask-rate on nouns it was taught      should be low
  gap = first - second                 the only evidence of a learned policy

Both failure modes are failures, and both are invisible if you report one rate
alone: a model that asks about everything scores a perfect ask-rate on the
unknown set, and a model that never asks scores a perfect one on the known set.

Run it with --gate off. The ask gate in AffectModulator is scaffolding: it
raises the ask token when the prompt holds a word the model was never taught,
so with the gate ON a gap proves only that the gate works. With the gate OFF a
gap is the model's own.

The unknown set is the `probe` half of lexicon.json's unknown_nouns, held out
of L12 as well as of L0-L11: measuring on the half L12 teaches would be
measuring recall of a lesson, not curiosity.

Usage:
    python3 scripts/curiosity_rate.py --ckpt-base models/checkpoints/it --level 12
    python3 scripts/curiosity_rate.py --checkpoint models/active.pt \
            --tokenizer models/active_tokenizer.json --gate on
"""
import os
import re
import sys
import json
import random
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests", "test_1"),
           os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse, do not re-implement: the checkpoint/tokenizer pairing rules and the
# surface forms of the prompts both already live elsewhere and must not drift.
from measure_repetition import resolve, load_pair                  # noqa: E402
from dynamic_model.stop_words import STOP_WORDS                    # noqa: E402
from dynamic_model.exp_b.modulator import ASK_FORM, ask_token_id   # noqa: E402
from expand_teacher_pools import (Lex, load_lexicon, phrase,       # noqa: E402
                                  intro, indef, ask_forms)
# Print-only: the margin column in report() and the agreement line under it.
# The gap is still taken on the generated string and nothing here decides
# anything — the §9 guardrail of docs_internal/curiosita_meccanismo.md.
from dynamic_model.exp_b import epistemic as ep                    # noqa: E402

SEED = 20260826          # the probe set is a measurement: same prompts every run


# ── response classification ────────────────────────────────────────────────

ASK_TOKEN = "<|ask|>"
_ASK_OPENERS = ("cos", "cosa", "che", "chi", "come", "perché", "perche")


def is_question(text: str) -> bool:
    """Did the model ask something?

    Three signals, any of them enough: the explicit ask token, a question mark,
    or an interrogative opener. The opener matters because a model that has not
    yet learned to close a question still asks one.
    """
    t = text.strip().lower()
    if ASK_TOKEN in text:
        return True
    if "?" in t:
        return True
    first = re.split(r"[\s,.!?]+", t)[0] if t else ""
    return first in _ASK_OPENERS


# Level 12 step E teaches a SECOND honest answer to a name the model cannot
# know: 'cos è un falco?' -> 'non lo so.' It is not a question, so is_question
# cannot see it, and until this was added the script reported a +33% gap and
# 'non discrimina' on a model whose real separation was +67% — half the correct
# answers were being counted as failures.
_DONT_KNOW_RE = re.compile(r"\bnon\s+(?:lo\s+)?so\b", re.IGNORECASE)


def says_dont_know(text: str) -> bool:
    """Did the model declare ignorance instead of guessing?"""
    return bool(_DONT_KNOW_RE.search(text or ""))


def is_honest(text: str) -> bool:
    """Either legitimate response to something it cannot know: ask, or say so."""
    return is_question(text) or says_dont_know(text)


def stated_class(text: str, classes) -> str:
    """The class the model asserted, if any: 'il ragno è un animale.' -> 'un animale'."""
    t = text.lower()
    # longest first, so 'un essere vivente' is not matched as nothing
    for c in sorted(classes, key=len, reverse=True):
        if re.search(r"(?<!\w)" + re.escape(c.lower()) + r"(?!\w)", t):
            return c
    return ""


# ── probe construction ─────────────────────────────────────────────────────

def build_probes(lex: Lex, n_known: int) -> list:
    """The two halves of the measurement, in one prompt shape.

    'il quaderno è un oggetto, questo è un ragno' — the anchor is of a
    DIFFERENT class on purpose: with a same-class anchor the model can produce
    the right class by copying, and answering would be as cheap as asking.
    """
    rng = random.Random(SEED)
    probes = []

    for n in lex.unknown_of(probe=True):
        a = lex.anchor_of_other_class(rng, n["cls"])
        probes.append({"kind": "ignoto", "noun": n["w"], "cls": n["cls"],
                       "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
                       "should_ask": True})

    for n in lex.unknown_of(probe=False):
        a = lex.anchor_of_other_class(rng, n["cls"])
        probes.append({"kind": "insegnato-L12", "noun": n["w"], "cls": n["cls"],
                       "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
                       "should_ask": False})

    # The fourth group, and the only one that asks whether the RELATION was
    # learned rather than a list of names. `bare_probe` is taught NOWHERE, in
    # any shape, so every phrasing step E uses is fair game — and unlike the
    # two-clause shape above, asking a question back at a bare 'cos è un X?'
    # only repeats it, so 'non lo so' is the whole honest answer. These nouns
    # carry no class: there is no right assertion to make about them, which is
    # why `cls` is empty and the class column reads 0% by construction.
    #
    # Every phrasing, deliberately: measured on the 0-12 build, a single-shape
    # run reported 67% honest where a run across the shapes reported 7%. The
    # article and the interrogative had become the feature, so one shape
    # measures the shape.
    for n in lex.bare_probe:
        for pr in ask_forms(n, ""):
            probes.append({"kind": "mai-visto", "noun": n["w"], "cls": "",
                           "prompt": pr, "should_ask": True})
        probes.append({"kind": "mai-visto", "noun": n["w"], "cls": "",
                       "prompt": f"{phrase(n)} è un animale?",
                       "should_ask": True})

    known = lex.classified()
    rng.shuffle(known)
    for n in known[:n_known]:
        c = lex.cls_of(n)
        a = lex.anchor_of_other_class(rng, c)
        probes.append({"kind": "noto", "noun": n["w"], "cls": c,
                       "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
                       "should_ask": False})
    return probes


# ── measurement ────────────────────────────────────────────────────────────

def run(tr, tok, probes: list, classes, max_tokens: int = 24,
        readings: dict = None) -> list:
    """Generate on every probe and classify the string.

    `readings` (noun -> EpistemicVerdict) is optional and only annotates the
    rows: the margin the weights give the noun, and whether the mechanism calls
    it ignorant. The classification of the answer never looks at it.
    """
    rows = []
    for p in probes:
        out = tr.generate(p["prompt"], max_tokens=max_tokens,
                          base_temperature=0.0, top_k=1,
                          min_tokens=4, stop_after=6)
        answer = out[len(p["prompt"]):].strip()
        v = (readings or {}).get(p["noun"])
        rows.append({**p, "answer": answer,
                     "asked": is_question(answer),
                     "dont_know": says_dont_know(answer),
                     "honest": is_honest(answer),
                     "said": stated_class(answer, classes),
                     "margin": round(v.margin, 4) if v else None,
                     "mech_ignorant": v.ignorant if v else None})
    return rows


def report(rows: list, gate: str, tau: float = None) -> dict:
    groups = {}
    for kind in ("ignoto", "mai-visto", "insegnato-L12", "noto"):
        g = [r for r in rows if r["kind"] == kind]
        if not g:
            continue
        asked  = sum(1 for r in g if r["asked"])
        dk     = sum(1 for r in g if r["dont_know"])
        honest = sum(1 for r in g if r["honest"])
        right  = sum(1 for r in g if not r["honest"] and r["said"] == r["cls"])
        groups[kind] = {"n": len(g), "asked": asked,
                        "ask_rate": asked / len(g),
                        "dont_know": dk, "dk_rate": dk / len(g),
                        "honest": honest, "honest_rate": honest / len(g),
                        "class_right": right, "class_rate": right / len(g)}
        ms = sorted(r["margin"] for r in g if r.get("margin") is not None)
        if ms:
            groups[kind]["margin_med"] = ms[len(ms) // 2]
            groups[kind]["mech_ignorant_rate"] = (
                sum(1 for r in g if r["mech_ignorant"]) / len(ms))

    with_margin = any("margin_med" in s for s in groups.values())
    print(f"\n  gate = {gate}\n")
    print(f"  {'set':16s} {'n':>3} {'asks':>8} {'dont know':>10} "
          f"{'honest':>8} {'class right':>12}"
          + (f" {'margin':>8} {'mech.ign':>9}" if with_margin else ""))
    print("  " + "-" * (79 if with_margin else 60))
    for kind, s in groups.items():
        line = (f"  {kind:16s} {s['n']:3d} {s['ask_rate']:8.0%} "
                f"{s['dk_rate']:10.0%} {s['honest_rate']:8.0%} "
                f"{s['class_rate']:12.0%}")
        if "margin_med" in s:
            line += f" {s['margin_med']:8.3f} {s['mech_ignorant_rate']:9.0%}"
        print(line)

    # The gap is taken on the HONEST rate, not on the ask rate: asking and
    # saying 'non lo so' are two ways of not guessing, and the level teaches
    # both. Splitting them into separate columns keeps them visible.
    ign = groups.get("ignoto", {}).get("honest_rate")
    noto = groups.get("noto", {}).get("honest_rate")
    mai = groups.get("mai-visto", {})
    if mai:
        # Reported apart from the gap, because it answers a different question:
        # the gap says the model separates taught from held-out names inside the
        # pool it was built from; this says whether the policy reaches a name
        # the pool never contained. It is the number the autonomy loop lives on.
        print(f"\n  honesty on NEVER-SEEN names (no phrasing taught): "
              f"{mai['honest_rate']:.0%} on {mai['n']} prompts")
    gap = None
    if ign is not None and noto is not None:
        gap = ign - noto
        print(f"\n  'honest answer' gap, unknown - known: {gap:+.0%}")
        if ign > 0.9 and noto > 0.9:
            verdict = "NEVER ANSWERS — not curiosity, a tic"
        elif ign < 0.1 and noto < 0.1:
            verdict = "NEVER ADMITS — no curiosity"
        elif gap >= 0.4:
            verdict = "gap ≥ 40%: a discriminating policy"
            if gate != "off":
                verdict += " (but with the gate ON it only proves the gate works)"
        else:
            verdict = "gap under 40%: does not discriminate"
        print(f"  {verdict}")

    # The mechanism against the string, row by row: 'agree' means the weights
    # call the noun ignorant exactly when the generated answer was honest.
    # Informative only — the gap above is the measurement, unchanged.
    agreement = None
    judged = [r for r in rows if r.get("mech_ignorant") is not None]
    if judged:
        agree = sum(1 for r in judged if r["mech_ignorant"] == r["honest"])
        agreement = agree / len(judged)
        print(f"\n  mechanism vs string (tau={tau:.3f}): agree on "
              f"{agree}/{len(judged)} rows ({agreement:.0%})  — print-only")
    return {"gate": gate, "groups": groups, "gap": gap,
            "tau": tau, "mechanism_agreement": agreement}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", default="models/checkpoints/it")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--level", type=int, default=12,
                    help="level whose checkpoint to resolve under --ckpt-base")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--memory", default=None,
                    help="curiosity memory file (default: affect_memory.json "
                         "beside the checkpoint tree)")
    ap.add_argument("--gate", choices=["on", "off"], default="off",
                    help="ask gate in AffectModulator (default off — the only "
                         "setting whose gap is evidence of anything)")
    ap.add_argument("--known", type=int, default=12,
                    help="how many already-taught nouns to probe (default 12)")
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.checkpoint:
        ckpt, tok_path = a.checkpoint, a.tokenizer
    else:
        ckpt, tok_path = resolve(a.ckpt_base, a.level)
    if not ckpt or not os.path.exists(ckpt):
        sys.exit(f"no checkpoint: {ckpt}")
    if not tok_path or not os.path.exists(tok_path):
        sys.exit(f"no tokenizer: {tok_path}")

    tr, tok = load_pair(ckpt, tok_path)

    # The curiosity signal needs BOTH of these or it reads every prompt as
    # unknown: the content-word filter, and the memory of what has been taught.
    # load_pair builds a bare AffectState, and each build phase is a separate
    # process, so without the file below the measurement would report a model
    # that asks about everything — including the known set.
    tr.affect.function_words = STOP_WORDS
    mem_path = a.memory or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(ckpt))),
        "affect_memory.json")
    n_mem = tr.affect.load_memory(mem_path)
    print(f"  curiosity memory: {n_mem} words from {mem_path}" if n_mem
          else f"  ⚠ no curiosity memory in {mem_path}: every prompt reads "
               f"as unknown and the gate means nothing")

    if a.gate == "on":
        if not hasattr(tr.mod, "ask_gate"):
            sys.exit("this AffectModulator has no ask_gate: the curiosity gate "
                     "is not implemented yet, use --gate off")
        # Setting ask_gate alone is not enough — without an anchor the gate has
        # no logit to raise and does nothing, which would read as 'the gate has
        # no effect' rather than 'the gate was never installed'.
        aid = ask_token_id(tok)
        if aid is None:
            sys.exit(f"the tokenizer has no single token for "
                     f"{ASK_FORM.split()[0]!r} "
                     f"({[tok.decode([i]) for i in tok.encode(ASK_FORM)]}): "
                     f"with --gate on there would be nothing to push. "
                     f"Retrain the tokenizer with the L12 pool.")
        tr.mod.ask_id   = aid
        tr.mod.ask_gate = True
        print(f"  gate ON, anchor = {tok.decode([aid])!r} (id {aid})")

    lex = Lex(load_lexicon(a.lang), a.lang)
    probes = build_probes(lex, a.known)

    # The margin the weights give each probed noun, read BEFORE generating so
    # the reading cannot be of a state the probes just moved. Annotation only:
    # the classification of every answer is on the string, as above.
    classes = ep.classes_of(lex)
    tau = ep.calibrate(tr.model, tok, lex)
    by_word = {n["w"]: n for n in lex.classified() + lex.unknown + lex.bare_probe}
    readings = {w: ep.verdict(tr.model, tok, by_word[w], classes, tau)
                for w in {p["noun"] for p in probes} if w in by_word}
    rows = run(tr, tok, probes, list(lex.classes), readings=readings)

    print(f"\n{'─' * 68}")
    print(f"  CURIOSITY — {ckpt}")
    print(f"  {len(probes)} prompts, greedy, tokenizer {os.path.basename(tok_path)}")
    print(f"{'─' * 68}")
    out = report(rows, a.gate, tau=tau)

    if a.examples:
        print(f"\n  examples:")
        for kind in ("ignoto", "noto"):
            for r in [x for x in rows if x["kind"] == kind][:a.examples // 2]:
                # '?' asked, '~' declared ignorance, '.' answered anyway
                mark = "?" if r["asked"] else ("~" if r["dont_know"] else ".")
                print(f"   [{kind:13s}] {mark} {r['prompt']!r}\n"
                      f"        -> {r['answer']!r}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": ckpt, "summary": out, "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\n  written {a.json}")


if __name__ == "__main__":
    main()
