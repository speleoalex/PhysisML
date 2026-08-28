#!/usr/bin/env python3
"""
Write a level's curated text file from its own gold answers.

WHY
Every level needs at least one .txt in its folder: phase 0 globs `*.txt` on the
level directory, so a level without one has no text-training phase at all. For
L0-L10 those texts were written by hand. L11 and L12 teach a single relation
over a closed pool, so the honest curated text IS the set of gold answers —
writing it by hand would only be a slower way to produce the same lines.

The output is committed, so it must depend on nothing but the level's
local_teacher.json: the shuffle uses a dedicated RNG with a fixed seed, like
the qa_corpus generation in train_curriculum.py.

Usage:
    python3 scripts/gen_level_text.py --level 11
    python3 scripts/gen_level_text.py --level 11 --check   # exit 1 if stale
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260826          # fixed: the file is tracked, so the same config must
                         # always produce the same text on any machine


def build(lang: str, level: int, reps: int) -> str:
    path = os.path.join(ROOT, "training_files", lang, str(level),
                        "local_teacher.json")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    golds = []
    for sname in sorted(cfg["steps"]):          # sorted: dict order must not matter
        for t in cfg["steps"][sname]["targets"]:
            exp = t.get("expected") if isinstance(t, dict) else None
            if exp:
                golds.append(exp)
    # dict.fromkeys, not set(): dedupe while keeping first-seen order
    golds = list(dict.fromkeys(golds))

    lines = golds * reps
    random.Random(f"{SEED}-{lang}-{level}").shuffle(lines)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3,
                    help="repetitions of each gold answer (default 3)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file on disk differs")
    a = ap.parse_args()

    text = build(a.lang, a.level, a.reps)
    out = os.path.join(ROOT, "training_files", a.lang, str(a.level),
                       f"frasi_livello{a.level}.txt")

    if a.check:
        have = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
        if have == text:
            print(f"  L{a.level}: {os.path.basename(out)} allineato "
                  f"({len(text.splitlines())} righe)")
            return 0
        print(f"  L{a.level}: {os.path.basename(out)} STANTIO — rigenera con "
              f"`python3 scripts/gen_level_text.py --level {a.level}`")
        return 1

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    n_uniq = len(set(text.splitlines()))
    print(f"  L{a.level}: scritte {len(text.splitlines())} righe "
          f"({n_uniq} frasi distinte) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
