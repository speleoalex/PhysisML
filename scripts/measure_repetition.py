#!/usr/bin/env python3
"""
Measure exact match AND self-repetition rate on a level's curriculum targets.

Two numbers, on the same set of prompts, in one pass:

  exact      the fraction of targets answered exactly (the build's metric)
  repetition the fraction of answers containing a repeated word bigram
             ('il cane il cane aspetta la mamma', 'la la la casa!')

Repetition needs its own metric because it is invisible to exact match when
the answer is wrong anyway, and because sampling hides it: at temperature 0.8
the same prompt gives a different answer each time. Always greedy here.

Usage:
    python3 scripts/measure_repetition.py --ckpt-base models/exp_e/ancora_s1 \
            --levels 0-3
    python3 scripts/measure_repetition.py --checkpoint models/active.pt \
            --tokenizer models/active_tokenizer.json --levels 0-10
"""
import os
import sys
import glob
import json
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physisml.torch_model import TorchGPT, TorchAdamOptimizer          # noqa: E402
from dynamic_model.core.tokenizer import DynamicBPETokenizer       # noqa: E402
from dynamic_model.exp_b.affect_state import AffectState           # noqa: E402
from dynamic_model.exp_b.modulator import AffectModulator          # noqa: E402
from dynamic_model.exp_b.axioms import AxiomRegistry               # noqa: E402
from dynamic_model.exp_b.trainer import TrainerB                   # noqa: E402
from dynamic_model.test_model import load_level_cases, is_exact    # noqa: E402


def has_repetition(text: str, n: int = 2) -> bool:
    """True if any n-word sequence occurs more than once in `text`."""
    w = text.split()
    if len(w) < 2 * n:
        return False
    seqs = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(seqs) != len(set(seqs))


def parse_levels(spec: str) -> list:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def resolve(ckpt_base: str, level: int) -> tuple:
    """Locate (checkpoint, tokenizer) for a level inside a checkpoint tree.

    Prefers final_dreamed.pt: the dream is the last supervised act, and the
    level's tokenizer.json snapshot is saved with it. Pairing the snapshot
    with final_learned.pt (pre-dream) leaves the tokenizer one token ahead of
    what the model has active — the mismatch that makes prompts re-segment.
    """
    d = os.path.join(ckpt_base, f"level_{level}")
    ckpt = next((os.path.join(d, n) for n in ("final_dreamed.pt",
                                              "final_learned.pt", "final.pt")
                 if os.path.exists(os.path.join(d, n))), None)
    # Walk back through lower levels: a level whose dream created no token
    # writes no snapshot of its own.
    tok = None
    for lvl in range(level, -1, -1):
        cand = os.path.join(ckpt_base, f"level_{lvl}", "tokenizer.json")
        if os.path.exists(cand):
            tok = cand
            break
    return ckpt, tok


def load_pair(ckpt: str, tok_path: str):
    """Load a checkpoint with its tokenizer, refusing a mismatched pair.

    Separate from evaluate_level so a caller sweeping many target levels
    against one checkpoint loads the 94MB of weights once.
    """
    model = TorchGPT.load(ckpt)
    tok = DynamicBPETokenizer()
    tok.load(tok_path)
    if len(tok) < model.active_vocab_size:
        raise RuntimeError(
            f"tokenizer has {len(tok)} tokens but the model has "
            f"{model.active_vocab_size} active — wrong pairing, results "
            f"would be meaningless (measured: a 1-token gap alone moved "
            f"exact match by tens of points)")
    af = AffectState()
    tr = TrainerB(model, tok, TorchAdamOptimizer(model.parameters(), lr=1e-5),
                  af, AffectModulator(af), AxiomRegistry())
    return tr, tok


def evaluate_level(tr, tok, lang: str, level: int) -> dict:
    """Score an already-loaded model on one level's curriculum targets."""
    cases = [(p, e) for p, e in load_level_cases(lang, level) if e]
    exact = rep = 0
    examples = []
    for prompt, expected in cases:
        n_exp = len(tok.encode(expected))
        out = tr.generate(prompt,
                          max_tokens=max(24, min(2 * n_exp + 12, 120)),
                          base_temperature=0.0, top_k=1,
                          min_tokens=max(4, min(n_exp, 40)),
                          stop_after=max(0, min(n_exp - 1, 40)))
        answer = out[len(prompt):].strip()
        if is_exact(answer, expected):
            exact += 1
        if has_repetition(answer):
            rep += 1
            if len(examples) < 3:
                examples.append(f"{prompt!r} -> {answer!r}")
    n = len(cases) or 1
    return {"level": level, "n": len(cases),
            "exact": exact, "exact_rate": exact / n,
            "repetition": rep, "repetition_rate": rep / n,
            "examples": examples}


def evaluate(ckpt: str, tok_path: str, lang: str, level: int) -> dict:
    """Convenience wrapper: load a pair and score it on one level."""
    tr, tok = load_pair(ckpt, tok_path)
    return evaluate_level(tr, tok, lang, level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", default=None,
                    help="Checkpoint tree; resolves each level automatically")
    ap.add_argument("--checkpoint", default=None,
                    help="Single checkpoint (evaluated on every --levels)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--levels", default="0-3")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--json", default=None, help="Also write results here")
    a = ap.parse_args()

    if not a.ckpt_base and not a.checkpoint:
        ap.error("pass --ckpt-base or --checkpoint")

    rows = []
    print(f"{'Lvl':>4} {'n':>4} {'exact':>7} {'repetition':>12}")
    print("-" * 32)
    for level in parse_levels(a.levels):
        if a.ckpt_base:
            ckpt, tok = resolve(a.ckpt_base, level)
        else:
            ckpt, tok = a.checkpoint, a.tokenizer
        if not ckpt or not tok:
            print(f"{level:>4}    —  (checkpoint or tokenizer missing)")
            continue
        r = evaluate(ckpt, tok, a.lang, level)
        rows.append(r)
        print(f"{level:>4} {r['n']:>4} {r['exact_rate']:>6.0%} "
              f"{r['repetition_rate']:>11.0%}")
        for ex in r["examples"]:
            print(f"       {ex}")

    if rows:
        n = sum(r["n"] for r in rows)
        print("-" * 32)
        print(f"{'tot':>4} {n:>4} "
              f"{sum(r['exact'] for r in rows)/n:>6.0%} "
              f"{sum(r['repetition'] for r in rows)/n:>11.0%}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
