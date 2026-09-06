"""
PhysisML — standalone chat script.

Talks to a PhysisML checkpoint. With no weights on disk it downloads the
published ones from the Hugging Face Hub, so a fresh clone can hear the model
speak without training anything:

    python3 standalone/chat.py "di: cosa mangia il cane?"   # one answer
    python3 standalone/chat.py                              # interactive REPL
    python3 standalone/chat.py --download                   # fetch weights only
    python3 standalone/chat.py --lang en "say: the cat"     # another language

Weights are looked for in this order: --model if given, then standalone/model.pt
(what a local build writes), then the Hugging Face snapshot under
models/hf/<repo>/ — downloading it on first use unless --no-download.

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
import json
import argparse

import torch

# Resolve paths relative to this file so the script works from any cwd. The
# repository root goes on the path too: the affective modules live in
# dynamic_model/exp_b/ here, and are only copied into the package on export.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from physisml.torch_model import TorchGPT
from physisml.tokenizer   import BPETokenizer
from physisml.generation  import generate, undecodable_mask, load_affect
from dynamic_model import language as lang_manifest

LOCAL_MODEL    = os.path.join(_HERE, "model.pt")
LOCAL_TOKENIZER = os.path.join(_HERE, "tokenizer.json")
HF_FILES       = ["config.json", "model.safetensors", "tokenizer.json"]
HF_MB          = 95
CONTEXT_WINDOW = 128


def hf_repo_for(lang: str) -> str:
    """The Hub repo that publishes one language, from its manifest.

    Hardcoding the Italian repo here made --lang en download Italian weights
    and answer an English prompt in Italian, which looks like a broken model
    rather than the wrong download. The manifest is the one place that knows.
    """
    return lang_manifest.load(lang).hf_repo


def _in_venv() -> bool:
    return (hasattr(sys, "real_prefix")
            or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))


def _ensure(module: str, package: str, why: str) -> None:
    """
    Make sure one on-demand package is importable. Both are in
    requirements.txt; this only covers the case of someone running the script
    against a checkout that was set up before they existed. Installing is
    attempted only inside a virtualenv — a system Python is very likely
    externally managed (PEP 668) and pip would refuse anyway, with a wall of
    text that hides the one line that matters.
    """
    try:
        __import__(module)
        return
    except ImportError:
        pass
    if _in_venv():
        print(f"Installing {package} ({why})...")
        if subprocess.run(
                [sys.executable, "-m", "pip", "install", package]).returncode == 0:
            return
    print(f"Error: {package} is missing — {why}.\n"
          f"       pip install {package}\n"
          f"       or, for everything at once: pip install -r requirements.txt",
          file=sys.stderr)
    sys.exit(1)


def hf_snapshot(repo: str, download: bool = True) -> str:
    """
    Return the local folder holding the published weights, downloading them on
    first use. Lands in models/hf/<repo>/, which is gitignored like every other
    weight in this repository.
    """
    dest = os.path.join(_ROOT, "models", "hf", repo.split("/")[-1])
    if all(os.path.exists(os.path.join(dest, f)) for f in HF_FILES):
        return dest
    if not download:
        return dest
    _ensure("huggingface_hub", "huggingface_hub",
            "needed to download the published weights")
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo} (~{HF_MB} MB) → {os.path.relpath(dest, _ROOT)}/")
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=HF_FILES)
    return dest


def load_from_folder(folder: str):
    """Build the model from config.json and fill it from model.safetensors."""
    _ensure("safetensors", "safetensors",
            "needed to read model.safetensors")
    from safetensors.torch import load_file

    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    model = TorchGPT(
        vocab_size        = cfg["vocab_size"],
        d_model           = cfg["d_model"],
        n_heads           = cfg["n_heads"],
        n_layers          = cfg["n_layers"],
        d_ff              = cfg["d_ff"],
        max_seq_len       = cfg["max_seq_len"],
        dropout_p         = 0.0,
        active_vocab_size = cfg.get("active_vocab_size", cfg["vocab_size"]),
    )
    # lm_head.weight is tied to tok_emb.weight and is therefore NOT stored in
    # the file — TorchGPT.__init__ re-ties it. strict=False is required for
    # that one key and for nothing else, so check exactly that.
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(folder, "model.safetensors")), strict=False)
    tied = set(cfg.get("tied_weights", {}))
    if set(missing) - tied or unexpected:
        raise RuntimeError(
            f"checkpoint does not match config.json — missing {list(missing)}, "
            f"unexpected {list(unexpected)}")
    return model, os.path.join(folder, "tokenizer.json"), folder


def resolve(args):
    """
    Pick the weights to load and return (model, tokenizer_path, label).
    --model wins; then a local build; then the published snapshot.
    """
    if args.model:
        if os.path.isdir(args.model):
            return load_from_folder(args.model)
        if not os.path.exists(args.model):
            print(f"Error: model not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        return TorchGPT.load(args.model), args.tokenizer or LOCAL_TOKENIZER, args.model

    if os.path.exists(LOCAL_MODEL) and os.path.exists(LOCAL_TOKENIZER):
        # There is one standalone/model.pt for every language: it holds
        # whatever the last build wrote. Asking for a language explicitly and
        # being handed the other one's weights is the failure this checks for
        # -- the vocabulary beside them is the only evidence of which language
        # they speak.
        found = lang_manifest.detect(LOCAL_TOKENIZER)
        if args.lang is None or not found or found == args.lang:
            return (TorchGPT.load(LOCAL_MODEL),
                    args.tokenizer or LOCAL_TOKENIZER, LOCAL_MODEL)
        print(f"Note: {os.path.relpath(LOCAL_MODEL, _ROOT)} holds '{found}' "
              f"weights, not '{args.lang}' — using the published ones instead.")

    folder = hf_snapshot(args.hf_repo, download=not args.no_download)
    if not all(os.path.exists(os.path.join(folder, f)) for f in HF_FILES):
        print("Error: no weights found.\n"
              f"       Train one with ./build.sh, or drop --no-download to "
              f"fetch {args.hf_repo} from the Hugging Face Hub.",
              file=sys.stderr)
        sys.exit(1)
    model, tok_path, label = load_from_folder(folder)
    return model, (args.tokenizer or tok_path), label


def print_info(model, args, label, lang) -> None:
    print(f"  model      : {label}")
    print(f"  language   : {lang}")
    print(f"  params     : {model.num_params:,}")
    print(f"  vocab      : {model.active_vocab_size} active / {model.vocab_size} total")
    print(f"  d_model    : {model.d_model}  n_heads={model.n_heads}  n_layers={model.n_layers}")
    print(f"  context    : {CONTEXT_WINDOW} tokens")
    print(f"  temperature: {args.temperature}  (0 = greedy)")
    print(f"  top_k      : {args.top_k}")
    print(f"  max_tokens : {args.max_tokens}")
    print(f"  affect     : {'on' if not args.no_affect else 'off'}")


def main():
    parser = argparse.ArgumentParser(description="PhysisML interactive chat")
    parser.add_argument("prompt",        nargs="*",
                        help="prompt to answer once, then exit (omit for a REPL)")
    parser.add_argument("--model",       default=None,
                        help="checkpoint .pt, or a folder with config.json + "
                             "model.safetensors (default: standalone/model.pt, "
                             "else the published weights)")
    parser.add_argument("--tokenizer",   default=None,
                        help="tokenizer JSON (default: next to the weights)")
    parser.add_argument("--lang",        default=None,
                        help=f"which language to talk to (default: "
                             f"{lang_manifest.DEFAULT_LANG}). Picks the Hub "
                             f"repo from training_files/<lang>/language.json. "
                             f"Have: " + ", ".join(lang_manifest.available()))
    parser.add_argument("--hf-repo",     default=None,
                        help="Hugging Face model repo (default: the one --lang "
                             "declares)")
    parser.add_argument("--download",    action="store_true",
                        help="fetch the published weights and exit")
    parser.add_argument("--no-download", action="store_true",
                        help="never touch the network")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (default). Published scores are greedy")
    parser.add_argument("--top_k",       type=int,   default=40,
                        help="only used when temperature > 0")
    parser.add_argument("--max_tokens",  type=int,   default=40,
                        help="max tokens to generate per turn (default: 40)")
    parser.add_argument("--min_tokens",  type=int,   default=4,
                        help="EOS is suppressed before this many tokens")
    parser.add_argument("--no-affect",   action="store_true",
                        help="disable affective modulation of the logits")
    args = parser.parse_args()

    if args.hf_repo is None:
        lang = args.lang or lang_manifest.DEFAULT_LANG
        known = lang_manifest.available()
        if lang not in known:
            print(f"Error: no curriculum for language '{lang}' — have: "
                  + ", ".join(known), file=sys.stderr)
            sys.exit(1)
        args.hf_repo = hf_repo_for(lang)
        if not args.hf_repo:
            print(f"Error: '{lang}' publishes no weights — "
                  f"training_files/{lang}/language.json declares no "
                  f"\"hf_repo\". Train one with ./build.sh --lang {lang}, or "
                  f"pass --hf-repo explicitly.", file=sys.stderr)
            sys.exit(1)

    if args.download:
        folder = hf_snapshot(args.hf_repo)
        print(f"Weights ready in {folder}")
        return

    # Use all available CPU threads
    torch.set_num_threads(os.cpu_count() or 4)

    model, tok_path, label = resolve(args)
    model.eval()

    if not os.path.exists(tok_path):
        print(f"Error: tokenizer not found: {tok_path}", file=sys.stderr)
        sys.exit(1)
    tok = BPETokenizer()
    tok.load(tok_path)

    # Which language these weights actually speak, read off the vocabulary
    # shipped with them rather than off the flag that asked for them.
    code   = lang_manifest.detect(tok_path)
    spoken = (f"{lang_manifest.load(code).name} ({code})" if code
              else "unknown — this vocabulary matches no language on disk")

    mask = undecodable_mask(model, tok)
    modulator = affect = None
    if not args.no_affect:
        modulator, affect = load_affect(tok)

    def answer(text: str) -> str:
        return generate(model, tok, text,
                        max_tokens  = args.max_tokens,
                        temperature = args.temperature,
                        top_k       = args.top_k,
                        min_tokens  = args.min_tokens,
                        stop_after  = args.min_tokens,
                        context     = CONTEXT_WINDOW,
                        modulator   = modulator, affect = affect, mask = mask)

    if args.prompt:
        print(answer(" ".join(args.prompt)))
        return

    print("=" * 56)
    print("  PhysisML — interactive generation")
    print("=" * 56)
    print_info(model, args, label, spoken)
    print("=" * 56)
    print("  /info    show model details")
    print("  /quit    exit")
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
            print_info(model, args, label, spoken)
            print()
            continue

        print(f"<<< {answer(prompt)}\n")


if __name__ == "__main__":
    main()
