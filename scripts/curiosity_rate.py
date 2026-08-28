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
                                  intro, indef)

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

def run(tr, tok, probes: list, classes, max_tokens: int = 24) -> list:
    rows = []
    for p in probes:
        out = tr.generate(p["prompt"], max_tokens=max_tokens,
                          base_temperature=0.0, top_k=1,
                          min_tokens=4, stop_after=6)
        answer = out[len(p["prompt"]):].strip()
        rows.append({**p, "answer": answer,
                     "asked": is_question(answer),
                     "said": stated_class(answer, classes)})
    return rows


def report(rows: list, gate: str) -> dict:
    groups = {}
    for kind in ("ignoto", "insegnato-L12", "noto"):
        g = [r for r in rows if r["kind"] == kind]
        if not g:
            continue
        asked = sum(1 for r in g if r["asked"])
        right = sum(1 for r in g if not r["asked"] and r["said"] == r["cls"])
        groups[kind] = {"n": len(g), "asked": asked,
                        "ask_rate": asked / len(g),
                        "class_right": right, "class_rate": right / len(g)}

    print(f"\n  gate = {gate}\n")
    print(f"  {'insieme':16s} {'n':>3} {'ask-rate':>9} {'classe giusta':>14}")
    print("  " + "-" * 46)
    for kind, s in groups.items():
        print(f"  {kind:16s} {s['n']:3d} {s['ask_rate']:9.0%} "
              f"{s['class_rate']:14.0%}")

    ign = groups.get("ignoto", {}).get("ask_rate")
    noto = groups.get("noto", {}).get("ask_rate")
    gap = None
    if ign is not None and noto is not None:
        gap = ign - noto
        print(f"\n  divario ignoto - noto: {gap:+.0%}")
        if ign > 0.9 and noto > 0.9:
            verdict = "CHIEDE TUTTO — non è curiosità, è un tic"
        elif ign < 0.1 and noto < 0.1:
            verdict = "NON CHIEDE MAI — nessuna curiosità"
        elif gap >= 0.4:
            verdict = "divario ≥ 40%: politica discriminante"
            if gate != "off":
                verdict += " (ma a gate ON prova solo che il gate funziona)"
        else:
            verdict = "divario sotto il 40%: non discrimina"
        print(f"  {verdict}")
    return {"gate": gate, "groups": groups, "gap": gap}


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
        sys.exit(f"nessun checkpoint: {ckpt}")
    if not tok_path or not os.path.exists(tok_path):
        sys.exit(f"nessun tokenizer: {tok_path}")

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
    print(f"  memoria di curiosità: {n_mem} parole da {mem_path}" if n_mem
          else f"  ⚠ nessuna memoria di curiosità in {mem_path}: "
               f"ogni prompt risulta ignoto, il gate perde senso")

    if a.gate == "on":
        if not hasattr(tr.mod, "ask_gate"):
            sys.exit("questo AffectModulator non ha ask_gate: il gate della "
                     "curiosità non è ancora implementato, usa --gate off")
        # Setting ask_gate alone is not enough — without an anchor the gate has
        # no logit to raise and does nothing, which would read as 'the gate has
        # no effect' rather than 'the gate was never installed'.
        aid = ask_token_id(tok)
        if aid is None:
            sys.exit(f"il tokenizer non ha un token intero per "
                     f"{ASK_FORM.split()[0]!r} "
                     f"({[tok.decode([i]) for i in tok.encode(ASK_FORM)]}): "
                     f"con --gate on non ci sarebbe niente da spingere. "
                     f"Riaddestra il tokenizer col pool L12.")
        tr.mod.ask_id   = aid
        tr.mod.ask_gate = True
        print(f"  gate ON, ancora = {tok.decode([aid])!r} (id {aid})")

    lex = Lex(load_lexicon(a.lang))
    probes = build_probes(lex, a.known)
    rows = run(tr, tok, probes, list(lex.classes))

    print(f"\n{'─' * 68}")
    print(f"  CURIOSITÀ — {ckpt}")
    print(f"  {len(probes)} prompt, greedy, tokenizer {os.path.basename(tok_path)}")
    print(f"{'─' * 68}")
    out = report(rows, a.gate)

    if a.examples:
        print(f"\n  esempi:")
        for kind in ("ignoto", "noto"):
            for r in [x for x in rows if x["kind"] == kind][:a.examples // 2]:
                mark = "?" if r["asked"] else "."
                print(f"   [{kind:13s}] {mark} {r['prompt']!r}\n"
                      f"        -> {r['answer']!r}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": ckpt, "summary": out, "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\n  scritto {a.json}")


if __name__ == "__main__":
    main()
