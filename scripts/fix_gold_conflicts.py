#!/usr/bin/env python3
"""
Make every prompt in the training material carry exactly ONE gold answer.

WHY
A prompt with two gold answers is contradictory supervision: whichever the
model produces, the grader marks it wrong part of the time, and exact match on
the level reads as a regression that never happened. The collisions come from
two places:

  1. Two levels asking the same question with different expected shapes.
     'cosa fa il cane?' is L3 step D with a one-verb answer and was L5 step B
     with a two-verb one; 'perché ti piace il pane?' is L8 step C with one
     reason and was L9 step B with two and a 'perciò'. Fixed in the pools by
     giving L5 and L9 a prompt that names the task ('racconta due cose del
     cane', 'due motivi: perché ti piace il pane?'), and here in the pairs
     harvested from past sessions, which still carry the old wording.

  2. A session answer that diverges from the pool's gold for the same prompt.
     'come ti chiami?' is 'physisml!' in the pool and was harvested as
     'mi chiamo physisml.'; both are true, and that is exactly the problem.

RULE
The pool (local_teacher.json) is authoritative. A harvested pair whose prompt
matches a pool prompt must carry that pool's gold, or it is dropped. Between
two harvested pairs that disagree, the one whose shape matches its own level's
pool is rewritten to the new prompt form; the other is left alone.

Usage:
    python3 scripts/fix_gold_conflicts.py --dry-run
    python3 scripts/fix_gold_conflicts.py
"""
import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from expand_teacher_pools import load_lexicon, phrase, di      # noqa: E402

N_LEVELS = 13
# Praise the teacher prepends to a prompt; the same question therefore appears
# in the harvested pairs under several prefixes.
PRAISE = r"(?:(?:bravo|ottimo|perfetto|bene|benissimo)!\s*|ancora\.\s*)*"


def pool_golds(lang: str) -> dict:
    """prompt -> gold, from every level's local_teacher.json."""
    out = {}
    for L in range(N_LEVELS):
        p = os.path.join(ROOT, "training_files", lang, str(L),
                         "local_teacher.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        for s in cfg["steps"].values():
            tmpl = s.get("prompt_template", "{prompt}")
            for t in s["targets"]:
                if not isinstance(t, dict):
                    continue
                pr = tmpl.format(prompt=t.get("prompt", ""),
                                 target=t.get("prompt", ""))
                out[pr.strip()] = t.get("expected", "").strip()
    return out


def load_pairs(lang: str) -> dict:
    """level -> list of {prompt, response}."""
    out = {}
    for L in range(N_LEVELS):
        p = os.path.join(ROOT, "training_files", lang, str(L),
                         "qa_pairs.jsonl")
        if not os.path.exists(p):
            continue
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        out[L] = rows
    return out


def rewrite_l5(prompt: str, response: str, by_phrase: dict):
    """'bene! cosa fa il cane?' + 'il cane mangia e dorme.'
       -> 'bene! racconta due cose del cane'

    Only when the answer really has L5's two-verb shape: the same prompt also
    appears in L5's pairs with a one-verb answer, harvested while rehearsing
    L3, and that one must keep L3's wording.
    """
    m = re.match(rf"^({PRAISE})cosa fa (.+?)\?$", prompt, re.IGNORECASE)
    if not m:
        return None
    pfx, ph = m.group(1), m.group(2).strip().lower()
    n = by_phrase.get(ph)
    if n is None:
        return None
    body = response.strip().lower()
    if not body.startswith(ph + " "):
        return None
    tail = body[len(ph):].strip().rstrip(".!?")
    # 'mangia e dorme' — two verbs joined by 'e', nothing else
    if not re.fullmatch(r"\w+ e \w+", tail):
        return None
    return f"{pfx}racconta due cose {di(n)}"


def rewrite_l9(prompt: str, response: str, by_phrase: dict):
    """'perché ti piace il pane?' + '... perciò mi piace.'
       -> 'due motivi: perché ti piace il pane?'"""
    m = re.match(rf"^({PRAISE})perché ti piace (.+?)\?$", prompt, re.IGNORECASE)
    if not m:
        return None
    pfx, ph = m.group(1), m.group(2).strip().lower()
    if ph not in by_phrase or "perciò" not in response.lower():
        return None
    return f"{pfx}due motivi: perché ti piace {ph}?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it")
    ap.add_argument("--levels", nargs="+", type=int, default=None,
                    help="only rewrite/drop pairs of these levels (default: "
                         "all). A level whose phase 2 is about to run will "
                         "have its qa_pairs.jsonl rewritten by "
                         "_update_qa_pairs_from_sessions, so fixing it while a "
                         "build is live is a race — restrict to the levels "
                         "that build will not touch.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lex = load_lexicon(a.lang)
    by_phrase = {phrase(n).lower(): n for n in lex["nouns"]}
    pool = pool_golds(a.lang)
    pairs = load_pairs(a.lang)
    # The collision detection always reads EVERY level — a pair only conflicts
    # relative to the whole material — but only the selected levels are edited.
    editable = set(a.levels) if a.levels is not None else set(pairs)

    # ── pass 1: bring the harvested wording in line with the new pool form ──
    rewritten = collections.Counter()
    for L, rows in pairs.items():
        fn = rewrite_l5 if L == 5 else rewrite_l9 if L == 9 else None
        if fn is None or L not in editable:
            continue
        for r in rows:
            new_p = fn(r.get("prompt", ""), r.get("response", ""), by_phrase)
            if new_p and new_p != r["prompt"]:
                print(f"  L{L} rewritten  {r['prompt']!r}\n"
                      f"              -> {new_p!r}")
                r["prompt"] = new_p
                rewritten[L] += 1

    # ── pass 2: a harvested pair may not contradict the pool ────────────────
    dropped = collections.Counter()
    deferred = []
    for L, rows in pairs.items():
        keep = []
        for r in rows:
            p, resp = r.get("prompt", "").strip(), r.get("response", "").strip()
            gold = pool.get(p)
            if gold is not None and gold != resp and L not in editable:
                deferred.append((L, p, resp))
            if gold is not None and gold != resp and L in editable:
                print(f"  L{L} dropped    {p!r}\n"
                      f"              response {resp!r} != pool gold {gold!r}")
                dropped[L] += 1
                continue
            keep.append(r)
        pairs[L] = keep

    # ── pass 3: two harvested pairs that still disagree ─────────────────────
    # Resolve by the POOL, never by level order: 'com è il sole?' is harvested
    # at L3 as 'il sole è bello.' and at L5 as 'il sole è caldo.', and the L5
    # one is the pool's gold. Dropping by whichever level came first threw away
    # the correct pair and kept the divergent one.
    first = {}                      # prompt -> (level, response, index)
    for L in sorted(pairs):
        for idx, r in enumerate(pairs[L]):
            p_, resp = r.get("prompt", "").strip(), r.get("response", "").strip()
            if p_ not in first:
                first[p_] = (L, resp, idx)
                continue
            oL, oresp, oidx = first[p_]
            if oresp == resp:
                continue
            gold = pool.get(p_)
            # the loser is the one that does not match the pool; with no pool
            # entry, the earlier level wins (it is the level that introduced it)
            if gold is not None and resp == gold:
                loser = (oL, oidx, oresp)
                first[p_] = (L, resp, idx)
            else:
                loser = (L, idx, resp)
            lL, lidx, lresp = loser
            if lL not in editable:
                deferred.append((lL, p_, lresp))
                continue
            print(f"  L{lL} dropped    {p_!r}\n"
                  f"              response {lresp!r} contradicts "
                  f"{first[p_][1]!r}")
            pairs[lL][lidx] = None
            dropped[lL] += 1
    for L in pairs:
        pairs[L] = [r for r in pairs[L] if r is not None]

    print(f"\n  rewritten: {dict(rewritten) or 'none'}")
    print(f"  dropped:   {dict(dropped) or 'none'}")
    deferred = list(dict.fromkeys(deferred))   # pass 2 and 3 can both flag one
    if deferred:
        print(f"\n  DEFERRED ({len(deferred)}): conflicting pairs on levels "
              f"not selected with --levels. They must be fixed once that "
              f"level is no longer being built:")
        for L, p_, resp in deferred:
            print(f"    L{L}  {p_!r} -> {resp!r}")

    if a.dry_run:
        print("\nDRY RUN — nothing written")
        return 0

    touched = sorted(L for L in pairs
                     if rewritten.get(L) or dropped.get(L))
    for L in touched:
        p = os.path.join(ROOT, "training_files", a.lang, str(L),
                         "qa_pairs.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in pairs[L]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {p}  ({len(pairs[L])} pairs)")
    if touched:
        print("\nNow regenerate the derived corpora:")
        print(f"  python3 scripts/generate_qa_corpus.py --levels "
              f"{' '.join(str(L) for L in touched)} --lang {a.lang}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
