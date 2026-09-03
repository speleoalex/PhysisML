"""
Training script for the PhysisML LLM — with curriculum learning support.

Usage
-----
  # First session: Italian level 0
  python3 train.py --data_dir data/it/0 --save_dir checkpoints/it-0 --epochs 5

  # Cumulative session: load checkpoint and continue at level 1
  python3 train.py --data_dir data/it/1 --save_dir checkpoints/it-1 \\
                   --resume checkpoints/it-0/best.npz --epochs 5

  # Same command twice: the second time resumes automatically from the latest
  # checkpoint in save_dir (does not overwrite previous work)

  # Single file (compatibility)
  python3 train.py --data data/corpus.txt --epochs 10

Main flags
----------
  --data_dir    directory with .txt files (concatenated in order)
  --data        single .txt file (alternative to --data_dir)
  --tokenizer   pre-trained .json tokenizer (recommended for curriculum)
  --resume      starting .npz checkpoint (used only if save_dir is empty)
  --save_dir    directory for checkpoints  (default: checkpoints)
  --vocab_size  BPE vocabulary size        (default: 2000)
  --d_model     embedding dimension        (default: 128)
  --n_heads     attention heads            (default: 4)
  --n_layers    transformer blocks         (default: 2)
  --d_ff        FFN dimension              (default: 512)
  --block_size  context window             (default: 128)
  --epochs      training epochs            (default: 10)
  --lr          learning rate              (default: 1e-3)
  --dropout     dropout probability        (default: 0.1)
  --seed        random seed                (default: 42)
  --save_every  save every N epochs        (default: 1)
  --log_every   print loss every N steps   (default: 50)
"""
import argparse
import glob
import math
import os
import re
import shutil
import time

import numpy as np

from physisml import BPETokenizer, GPT, AdamOptimizer, set_seed
from physisml.utils import clip_grad_norm


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_text(args) -> str:
    """Load and concatenate text from --data or all .txt files in --data_dir."""
    if args.data_dir:
        pattern = os.path.join(args.data_dir, "*.txt")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .txt files in {args.data_dir}")
        parts = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
            print(f"  + {os.path.basename(path):40s}  "
                  f"({len(parts[-1]):>8,} chars)")
        return "\n\n".join(parts)
    else:
        with open(args.data, "r", encoding="utf-8") as f:
            return f.read()


def make_batches(token_ids: np.ndarray, block_size: int):
    """Non-overlapping windows of length (block_size + 1)."""
    n = len(token_ids)
    for start in range(0, n - block_size, block_size):
        chunk = token_ids[start : start + block_size + 1]
        if len(chunk) < block_size + 1:
            break
        yield chunk


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(save_dir: str):
    """
    Find the most recent checkpoint in save_dir.
    Returns (path, start_epoch):
      - path        = .npz file to load (None if not found)
      - start_epoch = first epoch to train (last_number + 1)
    """
    pattern = os.path.join(save_dir, "epoch*.npz")
    files = [f for f in glob.glob(pattern) if not f.endswith("_opt.npz")]
    if files:
        def _num(f):
            m = re.search(r'epoch(\d+)', os.path.basename(f))
            return int(m.group(1)) if m else 0
        latest = max(files, key=_num)
        return latest, _num(latest) + 1
    # No epochNNN.npz — try best.npz
    best = os.path.join(save_dir, "best.npz")
    if os.path.exists(best):
        return best, 1
    return None, 1


def _load_opt(opt: AdamOptimizer, ckpt_path: str) -> None:
    """Load the optimizer state from the adjacent _opt.npz file."""
    opt_path = ckpt_path.replace(".npz", "_opt.npz")
    if os.path.exists(opt_path):
        data = np.load(opt_path, allow_pickle=True)
        opt.t  = int(data["t"])
        opt._m = data["m"].item()
        opt._v = data["v"].item()
        print(f"  Optimizer state loaded (step={opt.t})")


def _save_opt(opt: AdamOptimizer, ckpt_path: str) -> None:
    """Save the optimizer state next to the model checkpoint."""
    path = ckpt_path.replace(".npz", "_opt.npz")
    np.savez(path, t=opt.t, m=np.array(opt._m, dtype=object),
             v=np.array(opt._v, dtype=object))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ text
    src = args.data_dir if args.data_dir else args.data
    print(f"Loading data from: {src}")
    text = load_text(args)
    print(f"  Total corpus: {len(text):,} characters\n")

    # ------------------------------------------------------------ tokenizer
    tok_path = os.path.join(args.save_dir, "tokenizer.json")
    tok = BPETokenizer()

    if args.tokenizer:
        print(f"Loading tokenizer from: {args.tokenizer}")
        tok.load(args.tokenizer)
        shutil.copy(args.tokenizer, tok_path)
    elif os.path.exists(tok_path):
        # tokenizer already present in save_dir (previous run)
        print(f"Loading local tokenizer: {tok_path}")
        tok.load(tok_path)
    elif args.resume:
        resume_dir = os.path.dirname(args.resume)
        resume_tok = os.path.join(resume_dir, "tokenizer.json")
        if os.path.exists(resume_tok):
            print(f"Loading tokenizer from: {resume_tok}")
            tok.load(resume_tok)
            shutil.copy(resume_tok, tok_path)
        else:
            print("Tokenizer not found, training a new one ...")
            tok.train(text, vocab_size=args.vocab_size)
            tok.save(tok_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) ...")
        tok.train(text, vocab_size=args.vocab_size)
        tok.save(tok_path)
        print(f"  Tokenizer saved in {tok_path}")

    # ---------------------------------------------------------------- encode
    print("\nEncoding corpus ...")
    token_ids = np.array(tok.encode(text), dtype=np.int32)
    print(f"  Token: {len(token_ids):,}")

    # ----------------------------------------------------------------- model
    model = GPT(
        vocab_size  = len(tok),
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        max_seq_len = args.block_size + 1,
        dropout_p   = args.dropout,
    )

    opt = AdamOptimizer(lr=args.lr, weight_decay=1e-2)

    # ---- Resume priority:
    # 1. Local checkpoint in save_dir  (previous run at the same level)
    # 2. External --resume             (first run at this level)
    # 3. No resume                     (training from scratch)
    local_ckpt, start_epoch = _find_latest_checkpoint(args.save_dir)

    if local_ckpt:
        print(f"\nResuming from local checkpoint: {local_ckpt}  (→ epoch {start_epoch})")
        model.load(local_ckpt)
        _load_opt(opt, local_ckpt)
    elif args.resume:
        print(f"\nResuming from: {args.resume}")
        model.load(args.resume)
        _load_opt(opt, args.resume)
    else:
        print(f"\nModel parameters: {model.num_params:,}")

    print(f"Baseline loss (random): {math.log(len(tok)):.3f}\n")

    # ---- Load previous best_loss to avoid regression
    best_loss_path = os.path.join(args.save_dir, "best_loss.txt")
    if os.path.exists(best_loss_path):
        with open(best_loss_path) as f:
            best_loss = float(f.read().strip())
        print(f"  Previous best loss: {best_loss:.4f}")
    else:
        best_loss = float("inf")

    # ----------------------------------------------------------------- loop
    global_step = opt.t

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_losses = []
        t0 = time.time()

        batches = list(make_batches(token_ids, args.block_size))
        np.random.shuffle(batches)

        for chunk in batches:
            ids = chunk.astype(np.int32)

            logits = model.forward(ids, training=True)
            loss, dlogits = model.loss(logits, ids)
            model.backward(dlogits)

            grads = clip_grad_norm(model.get_grads(), max_norm=1.0)
            new_params = opt.step(model.get_params(), grads)
            model.apply_params(new_params)

            epoch_losses.append(float(loss))
            global_step += 1

            if global_step % args.log_every == 0:
                avg = np.mean(epoch_losses[-args.log_every:])
                elapsed = time.time() - t0
                print(f"  epoch {epoch:3d}  step {global_step:6d}  "
                      f"loss {avg:.4f}  ({elapsed:.1f}s)")

        avg_loss = float(np.mean(epoch_losses))
        elapsed  = time.time() - t0
        print(f"Epoch {epoch}  avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"epoch{epoch:03d}.npz")
            model.save(ckpt)
            _save_opt(opt, ckpt)
            print(f"  Checkpoint: {ckpt}")

        # Save the best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(args.save_dir, "best.npz")
            model.save(best_ckpt)
            _save_opt(opt, best_ckpt)
            with open(best_loss_path, "w") as f:
                f.write(str(best_loss))
            print(f"  New best loss={best_loss:.4f} → {best_ckpt}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Final checkpoints in: {args.save_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the PhysisML LLM with curriculum learning support",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--data",     default=None,
                     help="Single .txt file")
    grp.add_argument("--data_dir", default=None,
                     help="Directory with .txt files (concatenated)")

    parser.add_argument("--tokenizer",  default=None,
                        help="Pre-trained .json tokenizer (recommended for curriculum)")
    parser.add_argument("--resume",     default=None,
                        help="Starting .npz checkpoint (used only if save_dir is empty)")
    parser.add_argument("--save_dir",   default="checkpoints")
    parser.add_argument("--vocab_size", type=int,   default=2000)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--n_layers",   type=int,   default=2)
    parser.add_argument("--d_ff",       type=int,   default=512)
    parser.add_argument("--block_size", type=int,   default=128)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save_every", type=int,   default=1)
    parser.add_argument("--log_every",  type=int,   default=50)
    args = parser.parse_args()

    if args.data is None and args.data_dir is None:
        args.data = "data/train.txt"

    train(args)


if __name__ == "__main__":
    main()
