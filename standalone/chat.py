"""
PhysisML — standalone chat script.

Loads the model and tokenizer from the same directory and runs
an interactive REPL: type a text prompt, get a generated continuation.

Usage:
    cd standalone/
    python3 chat.py
    python3 chat.py --temperature 0.9 --top_k 50 --max_tokens 120

Commands inside the REPL:
    /info    — show model parameters and current settings
    /quit    — exit
"""
import sys
import subprocess

# --- Auto-install missing packages ---
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "torch": "torch",
}

def install_missing_packages():
    """Check and install missing packages automatically."""
    import os as _os

    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append((module_name, pip_name))

    if not missing:
        return

    # Check if we're in a venv
    in_venv = (hasattr(sys, 'real_prefix') or
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    venv_path = _os.path.join(_HERE, ".venv")
    venv_python = _os.path.join(venv_path, "bin", "python")

    # If a local .venv exists, use it
    if _os.path.exists(venv_python) and not in_venv:
        print(f"Found .venv, restarting with {venv_python}...")
        _os.execv(venv_python, [venv_python] + sys.argv)

    print("Missing packages: " + ", ".join(p for _, p in missing))

    if not in_venv:
        # Create local venv
        print(f"Creating virtual environment in {venv_path}...")
        result = subprocess.run([sys.executable, "-m", "venv", venv_path])
        if result.returncode != 0:
            print("Error creating venv. Install manually:")
            print(f"  python3 -m venv {venv_path}")
            print(f"  {venv_python} -m pip install " + " ".join(p for _, p in missing))
            sys.exit(1)
        print(f"Restarting with {venv_python}...")
        _os.execv(venv_python, [venv_python] + sys.argv)

    # We're in a venv, install packages
    print("Installing packages...")
    for module_name, pip_name in missing:
        print(f"  - {pip_name}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
        )
        if result.returncode != 0:
            print(f"\nError installing {pip_name}.")
            sys.exit(1)
    print("Done. Restarting...\n")
    _os.execv(sys.executable, [sys.executable] + sys.argv)

install_missing_packages()
# --- End auto-install ---

import os
import argparse

import numpy as np
import torch

# Resolve paths relative to this file so the script works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from physisml.torch_model import TorchGPT
from physisml.tokenizer   import BPETokenizer
from physisml.utils       import sample_top_k

DEFAULT_MODEL     = os.path.join(_HERE, "model.pt")
DEFAULT_TOKENIZER = os.path.join(_HERE, "tokenizer.json")
CONTEXT_WINDOW    = 128


def generate(model: TorchGPT, tok: BPETokenizer,
             prompt: str, max_tokens: int,
             temperature: float, top_k: int) -> str:
    prompt_ids = tok.encode(prompt)
    if not prompt_ids:
        return ""

    ids = list(prompt_ids)

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            ctx    = torch.tensor(ids[-CONTEXT_WINDOW:], dtype=torch.long)
            logits = model.forward(ctx)   # (T, V)
            last   = logits[-1].numpy()   # (V,)

            next_id = sample_top_k(last, k=top_k, temperature=temperature)
            ids.append(next_id)

            if tok.is_special(next_id):   # stop at EOS or any special token
                break

    generated_ids = ids[len(prompt_ids):]
    return tok.decode(generated_ids)


def print_info(model: TorchGPT, args) -> None:
    print(f"  model      : {args.model}")
    print(f"  params     : {model.num_params:,}")
    print(f"  vocab      : {model.active_vocab_size} active / {model.vocab_size} total")
    print(f"  d_model    : {model.d_model}  n_heads={model.n_heads}  n_layers={model.n_layers}")
    print(f"  context    : {CONTEXT_WINDOW} tokens")
    print(f"  temperature: {args.temperature}")
    print(f"  top_k      : {args.top_k}")
    print(f"  max_tokens : {args.max_tokens}")


def main():
    parser = argparse.ArgumentParser(description="PhysisML interactive chat")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="Path to .pt checkpoint (default: model.pt)")
    parser.add_argument("--tokenizer",   default=DEFAULT_TOKENIZER,
                        help="Path to tokenizer JSON (default: tokenizer.json)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k",       type=int,   default=40,
                        help="Top-k sampling (default: 40)")
    parser.add_argument("--max_tokens",  type=int,   default=80,
                        help="Max tokens to generate per turn (default: 80)")
    args = parser.parse_args()

    # Validate paths
    for label, path in [("model", args.model), ("tokenizer", args.tokenizer)]:
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Use all available CPU threads
    torch.set_num_threads(os.cpu_count() or 4)

    # Load tokenizer and model
    tok = BPETokenizer()
    tok.load(args.tokenizer)

    model = TorchGPT.load(args.model)
    model.eval()

    print("=" * 56)
    print("  PhysisML — interactive generation")
    print("=" * 56)
    print_info(model, args)
    print("=" * 56)
    print("  /info    mostra dettagli modello")
    print("  /quit    esci")
    print("=" * 56)
    print()

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not prompt:
            continue

        if prompt in ("/quit", "/exit"):
            break

        if prompt == "/info":
            print_info(model, args)
            print()
            continue

        output = generate(model, tok, prompt,
                          max_tokens=args.max_tokens,
                          temperature=args.temperature,
                          top_k=args.top_k)
        print(f"<<< {output}\n")


if __name__ == "__main__":
    main()
