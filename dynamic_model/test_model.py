"""
Quick model statistics after a build.
Runs test prompts appropriate for the level and reports quality metrics.

Usage:
    python3 dynamic_model/test_model.py --level 0
    python3 dynamic_model/test_model.py --level 1 --checkpoint models/active.pt
"""
import sys, os, argparse, re
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from splx.torch_model import TorchGPT
from splx.tokenizer   import BPETokenizer
from dynamic_model.exp_b.affect_state import AffectState
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.trainer      import TrainerB
from splx.torch_model import TorchAdamOptimizer

torch.set_num_threads(12)

# Test prompts and expected vocabulary per level
LEVEL_CONFIG = {
    0: {
        "prompts": ["ma", "pa", "oh", "sì", "na na"],
        "vocab":   ["ma", "pa", "ta", "la", "na", "ba", "da", "oh", "ah", "sì", "no"],
        "desc":    "Fonemi e sillabe",
    },
    1: {
        "prompts": ["mamma", "il cane", "ciao", "sì no", "papà"],
        "vocab":   ["mamma", "papà", "cane", "gatto", "pane", "latte", "acqua",
                    "sole", "luna", "bello", "sì", "no", "ciao", "bravo"],
        "desc":    "Parole singole e famiglia",
    },
    2: {
        "prompts": ["il cane dorme", "la casa è", "buongiorno", "il gatto"],
        "vocab":   ["il", "la", "un", "una", "è", "sono", "dorme", "mangia",
                    "beve", "bello", "grande", "piccolo", "casa", "cane", "gatto"],
        "desc":    "Frasi semplici",
    },
    3: {
        "prompts": ["io sono", "dove vai", "hai fame", "il bambino"],
        "vocab":   ["io", "tu", "lui", "sono", "sei", "è", "ho", "hai",
                    "vado", "vai", "bene", "male", "fame", "sete"],
        "desc":    "Grammatica base",
    },
}


def score_output(text: str, vocab: list) -> float:
    """Fraction of known vocabulary words found in the output."""
    text_lower = text.lower()
    words_found = sum(1 for w in vocab if w in text_lower)
    return words_found / len(vocab)


def clean_output(text: str) -> str:
    """Remove excessive whitespace for display."""
    return re.sub(r"\s+", " ", text).strip()[:60]


def main():
    parser = argparse.ArgumentParser(description="Test model quality after build")
    parser.add_argument("--level",      type=int, default=0)
    parser.add_argument("--lang",       default="it")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer",  default=None,
                        help="Tokenizer JSON (auto-detected from checkpoint dir if omitted)")
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument("--top-k",      type=int, default=20)
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt = args.checkpoint
    if not ckpt:
        for candidate in [
            f"models/checkpoints/{args.lang}/level_{args.level}/final_learned.pt",
            f"models/checkpoints/{args.lang}/level_{args.level}/final.pt",
            "models/active.pt",
        ]:
            if os.path.exists(candidate):
                ckpt = candidate
                break
    if not ckpt or not os.path.exists(ckpt):
        print("Nessun checkpoint trovato.")
        sys.exit(1)

    # Load model
    model = TorchGPT.load(ckpt)

    # Auto-detect tokenizer: prefer level-specific, then 8K base, then 500-token base
    _tok_path = args.tokenizer
    if not _tok_path:
        _ckpt_dir = os.path.dirname(ckpt)
        _candidates = [
            os.path.join(_ckpt_dir, "tokenizer.json"),
            "models/active_tokenizer.json",
            "dynamic_model/data/tokenizer_8k.json",
            "dynamic_model/data/tokenizer_base.json",
        ]
        for _c in _candidates:
            if os.path.exists(_c):
                _t = BPETokenizer(); _t.load(_c)
                if len(_t) >= model.active_vocab_size:
                    _tok_path = _c
                    break
        if not _tok_path:
            _tok_path = "dynamic_model/data/tokenizer_8k.json"
    tok = BPETokenizer()
    tok.load(_tok_path)
    print(f"  Tokenizer: {_tok_path}  (vocab={len(tok)})")
    affect = AffectState()
    mod    = AffectModulator(affect)
    opt    = TorchAdamOptimizer(model.parameters(), lr=1e-4)
    trainer = TrainerB(model, tok, opt, affect, mod)

    cfg = LEVEL_CONFIG.get(args.level, LEVEL_CONFIG[1])

    print(f"\n{'─'*58}")
    print(f"  TEST MODELLO — Livello {args.level}: {cfg['desc']}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Params: {model.num_params:,}   Vocab: {model.vocab_size}")
    print(f"{'─'*58}")

    scores = []
    confidences = []

    for prompt in cfg["prompts"]:
        # Train passively on prompt (as in real interaction)
        trainer.step("", prompt, feedback=0.0)

        # Generate
        out = trainer.generate(prompt, max_tokens=args.max_tokens,
                               base_temperature=0.8, top_k=args.top_k)
        generated = out[len(prompt):].strip()
        display   = clean_output(generated)

        # Score
        sc = score_output(generated, cfg["vocab"])
        scores.append(sc)
        confidences.append(affect.confidence)

        bar = "█" * int(sc * 20)
        bar = f"{bar:<20}"
        print(f"  {repr(prompt):18s} → {repr(display):40s}  [{bar}] {sc:.0%}")

    # Aggregate stats
    avg_score = float(np.mean(scores))
    avg_conf  = float(np.mean(confidences))

    print(f"{'─'*58}")
    print(f"  Vocabolario riconosciuto:  {avg_score:.0%}  "
          f"({'ottimo' if avg_score > 0.5 else 'buono' if avg_score > 0.25 else 'da migliorare'})")
    print(f"  Confidence media:          {avg_conf:.2f}")
    print(f"  Stato affettivo:           {affect}")
    print(f"{'─'*58}\n")

    # Return exit code based on quality (useful for CI)
    sys.exit(0 if avg_score > 0.1 else 1)


if __name__ == "__main__":
    main()
