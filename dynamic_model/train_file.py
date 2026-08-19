"""
Text file training — Experiment B (TorchGPT d=256).

Usage:
  # Train on a file, starting from scratch
  python3 dynamic_model/train_file.py text.txt

  # Continue from an existing checkpoint
  python3 dynamic_model/train_file.py text.txt --checkpoint dynamic_model/exp_b/checkpoints/pretrain_full.pt

  # More epochs, larger batch
  python3 dynamic_model/train_file.py text.txt --epochs 5 --batch 16

  # Save with a specific name
  python3 dynamic_model/train_file.py text.txt --save my_model.pt

Full flags:
  file            .txt file to use for training (required)
  --checkpoint    starting .pt checkpoint
  --tokenizer     .json tokenizer (default: it-0, vocab 2000)
  --epochs        training epochs (default: 3)
  --batch         batch size (default: 8)
  --lr            learning rate (default: 1e-4 if checkpoint, 1e-3 if from scratch)
  --block         sequence length in tokens (default: 128)
  --save          output .pt path (default: next to checkpoint)
  --log_every     print loss every N steps (default: 50)
"""
import sys, os, argparse, time, glob
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, 'tests', 'test_1')
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from splx.torch_model import TorchGPT, TorchAdamOptimizer
from splx.tokenizer   import BPETokenizer
from dynamic_model.exp_b.trainer import TrainerB

torch.set_num_threads(12)


def load_text(path: str) -> str:
    """Load text from a .txt file or from all .txt files in a directory."""
    if os.path.isdir(path):
        parts = []
        for fpath in sorted(glob.glob(os.path.join(path, "*.txt"))):
            with open(fpath, encoding="utf-8") as f:
                parts.append(f.read())
            print(f"  + {os.path.basename(fpath):40s}  {len(parts[-1]):>8,} chars")
        return "\n\n".join(parts)
    else:
        with open(path, encoding="utf-8") as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Train TorchGPT on a text file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file",
                        help=".txt file or directory with .txt files")
    parser.add_argument("--checkpoint", default="models/active.pt",
                        help="Starting .pt checkpoint (default: models/active.pt)")
    parser.add_argument("--tokenizer",
                        default="dynamic_model/data/tokenizer_base.json",
                        help=".json tokenizer")
    parser.add_argument("--epochs",  type=int,   default=3)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=None,
                        help="Learning rate (default: 1e-4 if checkpoint, 1e-3 if from scratch)")
    parser.add_argument("--block",   type=int,   default=128)
    parser.add_argument("--save",    default=None,
                        help="Output .pt path")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    # ------------------------------------------------------------------ testo
    print(f"\nLoading: {args.file}")
    text = load_text(args.file)
    print(f"  {len(text):,} total characters\n")

    # ------------------------------------------------------------------ modello
    tok = BPETokenizer()
    tok.load(args.tokenizer)
    print(f"Tokenizer: {len(tok)} token")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        model = TorchGPT.load(args.checkpoint)
        lr    = args.lr or 1e-4
    else:
        if args.checkpoint:
            print(f"Checkpoint not found ({args.checkpoint}), starting from scratch.")
        else:
            print("No checkpoint — starting from scratch.")
        model = TorchGPT(len(tok), 256, 4, 4, 1024, args.block + 1, 0.1)
        lr    = args.lr or 1e-3

    opt     = TorchAdamOptimizer(model.parameters(), lr=lr)
    trainer = TrainerB(model, tok, opt)

    print(f"Model: {model.num_params:,} params  vocab={model.vocab_size}")
    print(f"Config:  lr={lr}  batch={args.batch}  block={args.block}  epochs={args.epochs}\n")

    # ------------------------------------------------------------------ output path
    # Default path:
    #   dynamic_model/exp_b/checkpoints/{file_name}/trained.pt
    # Example: text = "pinocchio.txt" → checkpoints/pinocchio/trained.pt
    #          text = "data/it/2/"    → checkpoints/it_2/trained.pt
    _src_name = os.path.splitext(os.path.basename(args.file.rstrip("/")))[0]
    _src_name = _src_name.replace(" ", "_").replace("/", "_")
    _default_dir  = os.path.join("dynamic_model", "exp_b", "checkpoints", _src_name)
    _default_path = os.path.join(_default_dir, "trained.pt")

    if args.save:
        save_path = args.save
    elif args.checkpoint:
        # Fine-tuning: save in the same directory as the starting checkpoint
        base = os.path.splitext(args.checkpoint)[0]
        save_path = base + f"_ft_{_src_name}.pt"
    else:
        save_path = _default_path

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # ------------------------------------------------------------------ training
    baseline = float(np.log(model.vocab_size))
    print(f"Baseline loss (random): {baseline:.3f}")
    print("=" * 50)

    global_losses = []
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = trainer.train_on_text(
            text,
            block_size = args.block,
            batch_size = args.batch,
            log_every  = args.log_every,
        )
        global_losses.extend(losses)
        avg  = float(np.mean(losses[-20:])) if losses else float("nan")
        ppl  = float(np.exp(avg))
        elap = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  "
              f"loss={avg:.4f}  ppl={ppl:.1f}  time={elap:.0f}s")

        # Checkpoint per ogni epoca
        epoch_path = save_path.replace(".pt", f"_e{epoch:02d}.pt")
        model.save(epoch_path)
        print(f"  → {epoch_path}")

    # Salva il modello finale
    model.save(save_path)

    print("\n" + "=" * 50)
    print(f"Training complete in {time.time()-t_total:.0f}s")
    if global_losses:
        print(f"  Loss: {global_losses[0]:.4f} → {float(np.mean(global_losses[-20:])):.4f}")
    print(f"  Checkpoint finale: {save_path}")
    print(f"\nPer continuare la sessione interattiva:")
    print(f"  python3 dynamic_model/run.py --checkpoint {save_path}")


if __name__ == "__main__":
    main()
