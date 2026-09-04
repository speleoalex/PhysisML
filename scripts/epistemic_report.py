#!/usr/bin/env python3
"""Margins of the epistemic trigger on a checkpoint, group by group.

The §2 measurement of docs_internal/curiosita_meccanismo.md, replicated
through dynamic_model/exp_b/epistemic.py: the margin between the two most
likely closed classes, on the known nouns, on what L12 taught, on the frozen
probes, on the never-seen pool and on pseudo-words, with the AUC of each
contrast and the threshold `calibrate` would derive from these weights.

Print-only. Nothing here decides anything and nothing is written to disk:
this is the column the loop logs and the number the guardrail (AUC on the
frozen probes ≥ 0.95) is checked against.

    python3 scripts/epistemic_report.py \\
        --checkpoint models/checkpoints/it/level_13/plateau_best/final_dreamed.pt \\
        --tokenizer  models/checkpoints/it/level_13/plateau_best/tokenizer.json
"""
import argparse
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                       # noqa: E402
from dynamic_model.exp_b import epistemic as ep          # noqa: E402
from measure_repetition import load_pair                 # noqa: E402


def groups_of(lex):
    """The same partition scripts/curiosity_rate.py measures honesty on.

    'probe' and 'never/probe' together are the frozen set of the guardrail:
    13 names taught nowhere. 'never/other' is the pool the curriculum taught
    'non lo so' about (acquirable + reserve, retracted ones excluded).
    """
    return [
        ("known",         lex.classified()),
        ("taught-L12",    lex.unknown_of(probe=False)),
        ("probe",         lex.unknown_of(probe=True)),
        ("never/probe",   lex.bare_probe),
        ("never/other",   lex.bare_unknown),
    ]


# Groups whose margin SHOULD be high: their examples list the lowest margins
# (the candidates for a spurious ask). The others list the highest (the
# negatives the trigger would miss).
_POSITIVE = ("known", "taught-L12")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--n-pseudo", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many per-noun lines per group")
    args = ap.parse_args()

    lex = etp.Lex(etp.load_lexicon(args.lang), args.lang)
    classes = ep.classes_of(lex)
    tr, tok = load_pair(args.checkpoint, args.tokenizer)
    model = tr.model

    print(f"checkpoint: {os.path.relpath(args.checkpoint, _ROOT)}")
    print(f"classes ({len(classes)}): {', '.join(classes)}")
    print()

    table = {}
    for name, nouns in groups_of(lex) + [
            ("pseudo", ep.pseudo_words(random.Random(args.seed), args.n_pseudo, args.lang))]:
        if not nouns:
            continue
        rows = [(n["w"], ep.verdict(model, tok, n, classes, threshold=0.0)) for n in nouns]
        ms = sorted(v.margin for _, v in rows)
        table[name] = ms
        right = ""
        want = {n["w"]: (n.get("cls") or lex.cls_of(n) if "kind" in n else n.get("cls"))
                for n in nouns}
        if any(want.values()):
            hits = sum(1 for w, v in rows if want.get(w) and v.top_class == want[w])
            right = f"  argmax right {hits}/{len(rows)}"
        print(f"{name:12s} n={len(ms):3d}  margin min/med/max "
              f"{ms[0]:.3f}/{ms[len(ms) // 2]:.3f}/{ms[-1]:.3f}{right}")
        shown = sorted(rows, key=lambda r: r[1].margin, reverse=name not in _POSITIVE)
        for w, v in shown[:args.examples]:
            print(f"    {w:14s} margin {v.margin:.3f}  top {v.top_class:18s} "
                  f"p_top {v.p_top:.3f}  H {v.entropy:.3f}")
    print()

    known = table.get("known", [])
    for name in ("taught-L12", "probe", "never/probe", "never/other", "pseudo"):
        if name in table and known:
            a = ep.auc(known, table[name])
            print(f"AUC(known vs {name:11s}) = {a:.4f}")
    frozen = table.get("probe", []) + table.get("never/probe", [])
    if frozen and known:
        print(f"AUC(known vs frozen probes) = {ep.auc(known, frozen):.4f}   (guardrail ≥ 0.95)")
    print()

    cal = ep.calibration(model, tok, lex, n_pseudo=args.n_pseudo, seed=args.seed,
                         classes=classes)
    print(f"calibration: tau={cal.tau:.4f}  p95(pseudo)={cal.p95_pseudo:.4f}  "
          f"p05(known)={cal.p05_known:.4f}  AUC={cal.auc:.4f}  "
          f"{'separated' if cal.separated else 'OVERLAPPING'}")
    for name in ("known", "taught-L12", "probe", "never/probe", "never/other", "pseudo"):
        if name in table:
            below = sum(1 for m in table[name] if m < cal.tau)
            note = "  ← spurious asks" if name in _POSITIVE else ""
            print(f"  ignorant at tau: {name:11s} {below:3d}/{len(table[name])}{note}")


if __name__ == "__main__":
    main()
