"""
Autoregressive text generation with a trained splx model.

Usage
-----
  python generate.py --checkpoint checkpoints/epoch005.npz \
                     --tokenizer  checkpoints/tokenizer.json \
                     --prompt "To be or not to" \
                     --max_tokens 200 \
                     --temperature 0.8 \
                     --top_k 40
"""
import argparse
import os
import numpy as np

from splx import BPETokenizer, GPT, set_seed
from splx.utils import sample_top_k, sample_top_p


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.0,
    seed: int = 0,
) -> str:
    """
    Autoregressively generate text starting from prompt.

    Parameters
    ----------
    top_k  : if > 0, restrict sampling to top-k tokens
    top_p  : if > 0, use nucleus (top-p) sampling instead of top-k
    """
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big")
    set_seed(seed)
    print(f"(seed: {seed})")

    ids = tokenizer.encode(prompt)
    block_size = model.max_seq_len - 1

    for _ in range(max_tokens):
        # Truncate context to block_size
        ctx = np.array(ids[-block_size:], dtype=np.int32)

        # Forward pass (no dropout)
        logits = model.forward(ctx, training=False)

        # Take logits at last position
        next_logits = logits[-1]   # (vocab_size,)

        # Sample next token
        if top_p > 0.0:
            next_id = sample_top_p(next_logits, p=top_p, temperature=temperature)
        else:
            next_id = sample_top_k(next_logits, k=top_k, temperature=temperature)

        ids.append(next_id)

    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text with splx LLM")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to .npz checkpoint file")
    parser.add_argument("--tokenizer",   required=True,
                        help="Path to tokenizer.json file")
    parser.add_argument("--prompt",      default="",
                        help="Initial prompt string")
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--top_p",       type=float, default=0.0,
                        help="Nucleus sampling probability (overrides top_k if > 0)")
    parser.add_argument("--seed",        type=int,   default=-1,
                        help="Seme casuale (-1 = casuale ad ogni run)")
    # Model architecture (must match checkpoint)
    parser.add_argument("--vocab_size",  type=int,   default=2000)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--d_ff",        type=int,   default=512)
    parser.add_argument("--block_size",  type=int,   default=128)
    parser.add_argument("--dropout",     type=float, default=0.0)
    args = parser.parse_args()

    # Load tokenizer
    tok = BPETokenizer()
    tok.load(args.tokenizer)
    print(f"Tokenizer loaded. Vocab size: {len(tok)}")

    # Build model and load weights
    model = GPT(
        vocab_size  = len(tok),
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        max_seq_len = args.block_size + 1,
        dropout_p   = args.dropout,
    )
    model.load(args.checkpoint)
    print(f"Model loaded from {args.checkpoint}. Parameters: {model.num_params:,}")

    # Generate
    print(f"\n--- Generating ({args.max_tokens} tokens) ---\n")
    text = generate(
        model, tok,
        prompt      = args.prompt,
        max_tokens  = args.max_tokens,
        temperature = args.temperature,
        top_k       = args.top_k,
        top_p       = args.top_p,
        seed        = args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
