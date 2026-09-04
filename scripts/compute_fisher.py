#!/usr/bin/env python3
"""
Compute the online-EWC Fisher sidecar for a finished level (exp_i).

Runs at the end-of-level boundary — after the dream top-up, on the same
final_dreamed.pt the next level's phase 0 will pick up — and writes
level_N/fisher.pt: the running Fisher diagonal (F = gamma * F_prev + F_new)
plus the anchor theta* (the checkpoint's weights). The next level's phase 1
and phase 2 load it via --anti-forgetting ewc.

The Fisher is estimated on the level's gold pairs
(training_files/<lang>/<N>/qa_pairs.jsonl) — POST-harvest, i.e. including the
pairs the dream's QA-harvest step added, which is the file the next level
actually trains on — using the exact prompt-masked loss the training steps
optimise (encode_prompt_response is shared with TrainerB.step). NEVER the
frozen probe set: that is the held-out eval.

It is a standalone script, not a --phase, for the same reason
dream_until_plateau.py is: the boundary it presides over belongs to build.sh,
and being standalone makes it backfillable offline over an existing tree.

Usage:
    python3 scripts/compute_fisher.py --level 3
    python3 scripts/compute_fisher.py --level 3 --ckpt-base models/exp_i/ewc_s1
Exit codes: 0 = sidecar written; 1 = refused (missing checkpoint/pairs, or
zero usable pairs — loud, like dream_until_plateau).
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from measure_repetition import load_pair                            # noqa: E402
from dynamic_model.exp_b.ewc import (EWC, DEFAULTS, SIDECAR,        # noqa: E402
                                     file_sha256)
from dynamic_model.train_curriculum import _find_prior_fisher       # noqa: E402


def load_pairs(path: str) -> list:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write level_N/fisher.pt for the exp_i ewc arm")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--ckpt-base", default=None)
    ap.add_argument("--gamma", type=float, default=DEFAULTS["gamma"],
                    help="online-EWC decay of the running Fisher "
                         "(default %(default)s; must match --ewc-gamma of "
                         "the training runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate and report, write nothing")
    a = ap.parse_args()

    # Absolute for the same reason dream_until_plateau insists on it: the
    # sidecar's meta records the source checkpoint path, and a relative one
    # would mean different things to different callers.
    a.ckpt_base = os.path.abspath(a.ckpt_base or
                                  os.path.join("models", "checkpoints",
                                               a.lang))
    level_dir = os.path.join(a.ckpt_base, f"level_{a.level}")
    dreamed = os.path.join(level_dir, "final_dreamed.pt")
    tok_path = None
    for lvl in range(a.level, -1, -1):
        cand = os.path.join(a.ckpt_base, f"level_{lvl}", "tokenizer.json")
        if os.path.exists(cand):
            tok_path = cand
            break
    qa_path = os.path.join("training_files", a.lang, str(a.level),
                           "qa_pairs.jsonl")

    if not os.path.exists(dreamed):
        print(f"Missing {dreamed}: the Fisher is computed on the checkpoint "
              f"the next level will inherit — run the dream first.")
        return 1
    if tok_path is None:
        print(f"No tokenizer.json in {a.ckpt_base} down to "
              f"level_{a.level}: pair not loadable.")
        return 1
    if not os.path.exists(qa_path):
        print(f"Missing {qa_path}: without gold pairs the Fisher has no data.")
        return 1

    pairs = load_pairs(qa_path)
    if not pairs:
        print(f"{qa_path} is empty: nothing to estimate.")
        return 1

    print(f"Fisher L{a.level}  checkpoint={dreamed}")
    print(f"  tokenizer={tok_path}")
    print(f"  pairs={len(pairs)} from {qa_path}  gamma={a.gamma:g}")

    tr, tok = load_pair(dreamed, tok_path)   # refuses a mismatched pair
    model = tr.model

    prior = _find_prior_fisher(a.ckpt_base, a.level)
    if prior:
        ewc = EWC.load(prior, lam=0.0, gamma=a.gamma, device=tr.device)
        print(f"  prior fisher: {prior} (level "
              f"{ewc.meta.get('level', '?')}, decayed by gamma)")
    else:
        ewc = EWC(lam=0.0, gamma=a.gamma, device=tr.device)
        print("  prior fisher: none (first anchor)")

    t0 = time.time()
    fisher_new, n_used, n_skipped = EWC.estimate_fisher(model, tok, pairs,
                                                        device=tr.device)
    print(f"  estimated on {n_used} pairs "
          f"({n_skipped} skipped, {time.time() - t0:.0f}s)")
    if n_used == 0:
        print("  zero usable pairs: sidecar NOT written.")
        return 1

    ewc.consolidate(model, fisher_new)

    if a.dry_run:
        print("  --dry-run: nothing written")
        return 0

    out = os.path.join(level_dir, SIDECAR)
    ewc.save(out,
             level=a.level,
             lang=a.lang,
             n_pairs=n_used,
             n_skipped=n_skipped,
             prior_fisher=prior,
             source_checkpoint=os.path.abspath(dreamed),
             source_sha256=file_sha256(dreamed))
    size_mb = os.path.getsize(out) / 1e6
    print(f"  → {out}  ({size_mb:.0f}MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
