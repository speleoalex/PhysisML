#!/usr/bin/env python3
"""
Retention matrix: does the curriculum accumulate, or overwrite?

Scores every level checkpoint against every level's curriculum targets. The
published per-level table is only this matrix's DIAGONAL — each level measured
on its own checkpoint. That answers "did the level train?", not "does the
model still know it?".

Measured on the validated L0->L10 build: the diagonal averages 96%, while the
final L10 checkpoint scores 20% across all levels. Reading a column downwards
shows when a competence dies; reading a row shows what one checkpoint can
still do.

Re-run this after any change meant to improve retention (rehearsal scope,
dream composition, replay weighting) and compare the same three summary
numbers: diagonal mean, final-row mean, and mean retention loss.

Usage:
    # the validated build
    python3 scripts/retention_matrix.py --ckpt-base models/checkpoints/it

    # an experiment arm, and a comparison against a saved baseline
    python3 scripts/retention_matrix.py --ckpt-base models/exp_e/rehearsal_s1 \
            --json models/analysis/retention_rehearsal.json
    python3 scripts/retention_matrix.py --compare \
            models/analysis/retention_baseline.json \
            models/analysis/retention_rehearsal.json

    # cheaper sweeps
    python3 scripts/retention_matrix.py --levels 0-5 --metric exact
"""
import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Import the sibling script by path rather than as scripts.measure_repetition:
# turning scripts/ into a package would change import semantics for every
# other script in it.
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from measure_repetition import (                  # noqa: E402
    load_pair, evaluate_level, resolve, parse_levels)

DEFAULT_JSON = "models/analysis/retention_matrix.json"


def build(ckpt_base: str, levels: list, lang: str, full: bool) -> dict:
    """Score each checkpoint level against each target level.

    Only t <= c by default: a checkpoint has never seen a higher level, so
    those cells measure zero-shot transfer, a different question. --full
    includes them.
    """
    cells = {}
    for c in levels:
        ckpt, tok_path = resolve(ckpt_base, c)
        if not ckpt or not tok_path:
            print(f"  level {c}: checkpoint or tokenizer missing — skipping",
                  flush=True)
            continue
        tr, tok = load_pair(ckpt, tok_path)
        targets = levels if full else [t for t in levels if t <= c]
        for t in targets:
            r = evaluate_level(tr, tok, lang, t)
            cells[f"{c}:{t}"] = {
                "checkpoint_level": c, "target_level": t, "n": r["n"],
                "exact_rate": r["exact_rate"],
                "repetition_rate": r["repetition_rate"],
                "examples": r["examples"],
            }
            print(f"  ckpt L{c} on target L{t}: "
                  f"exact {r['exact_rate']:.0%}  rep {r['repetition_rate']:.0%}",
                  flush=True)
        del tr, tok
    return {"ckpt_base": ckpt_base, "lang": lang, "levels": levels,
            "full": full, "cells": cells}


def _grid(data: dict, metric: str):
    cells = data["cells"]
    levels = data["levels"]
    key = "exact_rate" if metric == "exact" else "repetition_rate"
    print(f"\n  rows = checkpoint, columns = target  ({metric})\n")
    head = "  ckpt │" + "".join(f"{t:>6}" for t in levels)
    print(head)
    print("  " + "─" * (len(head) - 2))
    for c in levels:
        row = f"  L{c:<4}│"
        for t in levels:
            cell = cells.get(f"{c}:{t}")
            row += f"{cell[key]*100:>5.0f}%" if cell else "     ·"
        print(row)


def summarise(data: dict) -> dict:
    cells, levels = data["cells"], data["levels"]
    diag = [cells[f"{l}:{l}"] for l in levels if f"{l}:{l}" in cells]
    last = max((c["checkpoint_level"] for c in cells.values()), default=None)
    final = [c for c in cells.values() if c["checkpoint_level"] == last]

    def wmean(rows, key):
        n = sum(r["n"] for r in rows)
        return sum(r[key] * r["n"] for r in rows) / n if n else 0.0

    # Retention loss: for each target level, its own-checkpoint score minus
    # what the final checkpoint still scores on it.
    losses = []
    for l in levels:
        own, fin = cells.get(f"{l}:{l}"), cells.get(f"{last}:{l}")
        if own and fin:
            losses.append((l, own["exact_rate"] - fin["exact_rate"]))

    s = {
        "diagonal_exact":   wmean(diag, "exact_rate"),
        "final_row_exact":  wmean(final, "exact_rate"),
        "final_row_repetition": wmean(final, "repetition_rate"),
        "final_checkpoint_level": last,
        "retention_loss": {str(l): d for l, d in losses},
        "mean_retention_loss": sum(d for _, d in losses) / len(losses)
                              if losses else 0.0,
    }
    print(f"\n  diagonal (each level on its own checkpoint)     : "
          f"{s['diagonal_exact']:.0%}")
    print(f"  final checkpoint L{last} on every level             : "
          f"{s['final_row_exact']:.0%}"
          f"   (repetition {s['final_row_repetition']:.0%})")
    print(f"  mean retention loss                            : "
          f"{s['mean_retention_loss']:+.0%}")
    if losses:
        worst = sorted(losses, key=lambda x: -x[1])[:4]
        print("  worst: " + ", ".join(f"L{l} {d:+.0%}" for l, d in worst))
    return s


def compare(path_a: str, path_b: str) -> None:
    a, b = (json.load(open(p, encoding="utf-8")) for p in (path_a, path_b))
    sa, sb = a.get("summary") or summarise(a), b.get("summary") or summarise(b)
    print(f"\n  {'metric':<34}{'A':>8}{'B':>8}{'Δ':>8}")
    print("  " + "─" * 56)
    for key, label in (("diagonal_exact", "diagonal"),
                       ("final_row_exact", "final on every level"),
                       ("final_row_repetition", "repetition (final)"),
                       ("mean_retention_loss", "retention loss")):
        va, vb = sa.get(key, 0.0), sb.get(key, 0.0)
        print(f"  {label:<34}{va:>7.0%}{vb:>8.0%}{vb-va:>+8.0%}")
    print(f"\n  A = {path_a}\n  B = {path_b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", default="models/checkpoints/it")
    ap.add_argument("--levels", default="0-10")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--metric", default="both",
                    choices=["exact", "repetition", "both"])
    ap.add_argument("--full", action="store_true",
                    help="also score target levels above the checkpoint "
                         "(zero-shot transfer, not retention)")
    ap.add_argument("--json", default=DEFAULT_JSON,
                    help=f"where to write results (default {DEFAULT_JSON})")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two saved runs instead of measuring")
    a = ap.parse_args()

    if a.compare:
        compare(*a.compare)
        return

    levels = parse_levels(a.levels)
    print(f"  Retention matrix — {a.ckpt_base}, levels {levels}")
    data = build(a.ckpt_base, levels, a.lang, a.full)
    if not data["cells"]:
        print("  No cell measured.")
        return

    for m in (["exact", "repetition"] if a.metric == "both" else [a.metric]):
        _grid(data, m)
    data["summary"] = summarise(data)

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  → {a.json}")
    print(f"  future comparison: python3 scripts/retention_matrix.py "
          f"--compare {a.json} <new.json>")


if __name__ == "__main__":
    main()
