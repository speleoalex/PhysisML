"""
Compare two checkpoints side by side to detect regression or improvement.

Typical use: compare level_(N-1)/final_learned.pt (after teaching)
             with level_N/final.pt (after text training on level N)
             to see whether phase 0 of level N causes regression.

Usage:
    python3 scripts/compare_checkpoints.py \\
        models/checkpoints/it/level_0/final_learned.pt \\
        models/checkpoints/it/level_1/final.pt

    python3 scripts/compare_checkpoints.py --auto --lang it --level 1
        # auto mode: compares level_(N-1)/final_learned.pt vs level_N/final.pt

    python3 scripts/compare_checkpoints.py --all --lang it
        # compares every adjacent pair available
"""
import sys, os, argparse, re, glob
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from physisml.torch_model import TorchGPT
from physisml.tokenizer   import BPETokenizer
from dynamic_model.exp_b.affect_state import AffectState
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.trainer      import TrainerB
from physisml.torch_model import TorchAdamOptimizer

torch.set_num_threads(12)

TOKENIZER = "dynamic_model/data/tokenizer_base.json"

# Prompts shared across all levels — neutral enough to test at any stage
SHARED_PROMPTS = [
    "mamma",
    "il cane",
    "buongiorno",
    "io sono",
    "la casa è grande",
    "cosa fai",
    "sì no",
    "mi chiamo",
]

VOCAB_BASELINE = [
    "mamma", "papà", "cane", "gatto", "casa", "sole", "luna",
    "sì", "no", "ciao", "bello", "grande", "piccolo",
    "il", "la", "un", "è", "sono", "ho", "vai",
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:50]


def score(text: str) -> float:
    t = text.lower()
    return sum(1 for w in VOCAB_BASELINE if w in t) / len(VOCAB_BASELINE)


def perplexity(trainer: "TrainerB", text: str, block_size: int = 64) -> float:
    """Estimate perplexity on a short text snippet."""
    tok = trainer.tokenizer
    ids = tok.encode(text)
    if len(ids) < 2:
        return float("nan")
    ids = ids[:block_size + 1]
    x   = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    y   = torch.tensor(ids[1:],  dtype=torch.long).unsqueeze(0)
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(x)
        loss   = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
    return float(torch.exp(loss))


def load_trainer(ckpt_path: str) -> "TrainerB":
    tok = BPETokenizer()
    tok.load(TOKENIZER)
    model  = TorchGPT.load(ckpt_path)
    affect = AffectState()
    mod    = AffectModulator(affect)
    opt    = TorchAdamOptimizer(model.parameters(), lr=1e-4)
    return TrainerB(model, tok, opt, affect, mod)


def evaluate(trainer: "TrainerB", label: str) -> dict:
    """Run all shared prompts and collect scores."""
    results = {"label": label, "outputs": [], "scores": [], "ppls": []}

    # Perplexity on a fixed Italian sentence
    ppl_text = "il cane dorme sul tappeto. la mamma cucina il pane."
    results["ppl"] = perplexity(trainer, ppl_text)

    for prompt in SHARED_PROMPTS:
        out       = trainer.generate(prompt, max_tokens=20,
                                     base_temperature=0.7, top_k=20)
        generated = out[len(prompt):].strip()
        sc        = score(generated)
        results["outputs"].append(clean(generated))
        results["scores"].append(sc)

    results["avg_score"] = float(np.mean(results["scores"]))
    return results


def compare(path_a: str, path_b: str) -> None:
    label_a = os.path.relpath(path_a)
    label_b = os.path.relpath(path_b)

    print(f"\n{'═'*70}")
    print(f"  CHECKPOINT COMPARISON")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"{'═'*70}")

    trainer_a = load_trainer(path_a)
    trainer_b = load_trainer(path_b)

    res_a = evaluate(trainer_a, label_a)
    res_b = evaluate(trainer_b, label_b)

    # Header
    print(f"\n  {'Prompt':<22}  {'A output':<28}  {'B output':<28}  {'Δ score':>8}")
    print(f"  {'─'*22}  {'─'*28}  {'─'*28}  {'─'*8}")

    for i, prompt in enumerate(SHARED_PROMPTS):
        sc_a = res_a["scores"][i]
        sc_b = res_b["scores"][i]
        delta = sc_b - sc_a
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        print(f"  {repr(prompt):<22}  {repr(res_a['outputs'][i]):<28}  "
              f"{repr(res_b['outputs'][i]):<28}  {arrow} {delta:+.0%}")

    # Summary
    avg_a = res_a["avg_score"]
    avg_b = res_b["avg_score"]
    delta_avg = avg_b - avg_a
    arrow_avg = "↑ IMPROVEMENT" if delta_avg > 0.02 else \
                ("↓ REGRESSION"  if delta_avg < -0.02 else "→ unchanged")

    print(f"\n  {'─'*70}")
    print(f"  Mean vocabulary     A={avg_a:.1%}   B={avg_b:.1%}   "
          f"Δ={delta_avg:+.1%}  {arrow_avg}")
    print(f"  Perplexity          A={res_a['ppl']:.1f}     B={res_b['ppl']:.1f}     "
          f"{'↓ better' if res_b['ppl'] < res_a['ppl'] else '↑ worse'}")
    print(f"{'═'*70}\n")


def find_pairs(lang: str) -> list:
    """Find all (final_learned, next_final) adjacent pairs."""
    base = f"models/checkpoints/{lang}"
    pairs = []
    for level in range(0, 20):
        learned = os.path.join(base, f"level_{level}", "final_learned.pt")
        next_final = os.path.join(base, f"level_{level+1}", "final.pt")
        if os.path.exists(learned) and os.path.exists(next_final):
            pairs.append((learned, next_final, level))
    return pairs


def ppl_timeline(lang: str, base: str = None) -> None:
    """Print perplexity for every checkpoint found, in chronological order."""
    if base is None:
        base = f"models/checkpoints/{lang}"
    ppl_text = "il cane dorme sul tappeto. la mamma cucina il pane."

    # Collect all checkpoints in order: L0/final, L0/final_learned, L0/final_dreamed, L1/final, ...
    entries = []
    for level in range(0, 20):
        d = os.path.join(base, f"level_{level}")
        for name in ("final.pt", "final_learned.pt", "final_dreamed.pt"):
            path = os.path.join(d, name)
            if os.path.exists(path):
                label = f"L{level}/{name.replace('.pt','')}"
                if name == "final.pt":         phase = "phase0"
                elif name == "final_dreamed.pt": phase = "dream "
                else:                           phase = "phase1"
                entries.append((label, path, phase))

    if not entries:
        print("No checkpoint found.")
        return

    print(f"\n{'═'*62}")
    print(f"  PERPLEXITY TRAJECTORY — {lang}")
    print(f"  Text: \"{ppl_text}\"")
    print(f"{'═'*62}")
    print(f"  {'Checkpoint':<30}  {'PPL':>8}  {'Δ':>8}  Bar")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*20}")

    prev_ppl = None
    for label, path, phase in entries:
        trainer = load_trainer(path)
        ppl = perplexity(trainer, ppl_text)

        delta_str = ""
        arrow = ""
        if prev_ppl is not None:
            delta = ppl - prev_ppl
            arrow = "↓" if delta < -1 else ("↑" if delta > 1 else "→")
            delta_str = f"{delta:+.1f}"

        # Bar: lower PPL = longer green bar (cap at 200 for display)
        bar_len = max(1, int(20 * (1 - min(ppl, 300) / 300)))
        phase_tag = "[txt]" if phase == "phase0" else "[tch]"

        print(f"  {label:<30}  {ppl:>8.1f}  {delta_str:>8}  "
              f"{arrow} {'█' * bar_len}  {phase_tag}")

        prev_ppl = ppl

    print(f"{'═'*62}\n")
    print("  [txt] = phase 0   [tch] = phase 1   [drm] = dream consolidation\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_a", nargs="?", default=None,
                        help="First checkpoint (before)")
    parser.add_argument("path_b", nargs="?", default=None,
                        help="Second checkpoint (after)")
    parser.add_argument("--auto",         action="store_true",
                        help="Auto mode: compare level_(N-1)/final_learned vs level_N/final")
    parser.add_argument("--level",        type=int, default=1,
                        help="Level N for --auto mode")
    parser.add_argument("--all",          action="store_true",
                        help="Compare all adjacent pairs found")
    parser.add_argument("--ppl-timeline", action="store_true",
                        help="Show perplexity for every checkpoint in order")
    parser.add_argument("--lang",         default="it")
    args = parser.parse_args()

    if args.ppl_timeline:
        ppl_timeline(args.lang)
        return

    if args.all:
        pairs = find_pairs(args.lang)
        if not pairs:
            print("No adjacent pairs found in models/checkpoints/")
            sys.exit(1)
        for path_a, path_b, level in pairs:
            print(f"\n  [L{level}/final_learned → L{level+1}/final]")
            compare(path_a, path_b)
        return

    if args.auto:
        base = f"models/checkpoints/{args.lang}"
        path_a = os.path.join(base, f"level_{args.level - 1}", "final_learned.pt")
        path_b = os.path.join(base, f"level_{args.level}", "final.pt")
        for p in [path_a, path_b]:
            if not os.path.exists(p):
                print(f"File not found: {p}")
                sys.exit(1)
        compare(path_a, path_b)
        return

    if not args.path_a or not args.path_b:
        parser.print_help()
        sys.exit(1)

    for p in [args.path_a, args.path_b]:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

    compare(args.path_a, args.path_b)


if __name__ == "__main__":
    main()
