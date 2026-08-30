#!/usr/bin/env python3
"""
Assemble a Hugging Face upload folder for a PhysisML checkpoint.

The repository ships no weights (models/ and *.pt are gitignored), so this
script turns a local checkpoint into a self-contained folder that a Hub repo
can hold: safetensors weights, a plain config.json, the tokenizer, the minimal
inference code, and a manifest recording exactly which local file each artifact
came from.

Why safetensors: a .pt checkpoint is a pickle, which the Hub flags and many
users refuse to download. The tensors here are plain float32, so the
conversion is lossless.

Usage:
    # Main checkpoint only (models/active.pt by default)
    python3 scripts/export_hf.py --out hf_upload

    # Main checkpoint + the per-level ladder (post-dream snapshots)
    python3 scripts/export_hf.py --out hf_upload --levels 0-9

    # Explicit checkpoint
    python3 scripts/export_hf.py --out hf_upload --ckpt models/checkpoints/it/level_9/final_dreamed.pt

The script never uploads on its own. When the folder looks right:

    pip install huggingface_hub
    huggingface-cli login
    huggingface-cli upload <user>/<repo> hf_upload . --repo-type model

Requires: pip install safetensors
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files copied verbatim into the upload folder: the model card, the licence,
# and the standalone inference code (renamed to a package the card can import).
CARD_SRC     = os.path.join(_ROOT, "huggingface", "README.md")
GENERATE_SRC = os.path.join(_ROOT, "huggingface", "generate.py")
LICENSE_SRC  = os.path.join(_ROOT, "LICENSE")
SPLX_SRC     = os.path.join(_ROOT, "standalone", "splx")
PKG_NAME     = "physisml"

# The affective system, copied into the package so the published model can
# actually demonstrate it. These live outside standalone/ and import each other
# by absolute path, which is rewritten to a relative import on copy.
AFFECT_SRCS = [
    os.path.join(_ROOT, "dynamic_model", "exp_b", "affect_state.py"),
    os.path.join(_ROOT, "dynamic_model", "exp_b", "modulator.py"),
]

DEFAULT_CKPTS = [
    os.path.join(_ROOT, "models", "active.pt"),
    os.path.join(_ROOT, "standalone", "model.pt"),
]

# The generation window used by the standalone chat script. Stored in
# config.json so a reader does not have to guess it from max_seq_len.
CONTEXT_WINDOW = 128


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_tokenizer(ckpt_path: str) -> str:
    """
    The tokenizer that matches a checkpoint lives next to it (each level dir
    writes its own tokenizer.json). Fall back to the standalone copy.
    """
    candidates = [
        os.path.join(os.path.dirname(ckpt_path), "tokenizer.json"),
        os.path.join(_ROOT, "standalone", "tokenizer.json"),
        os.path.join(_ROOT, "dynamic_model", "data", "tokenizer_8k.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    die("no tokenizer.json found next to the checkpoint or in standalone/")


def export_one(ckpt_path: str, out_dir: str, save_file,
               root_out: str) -> dict:
    """
    Convert one .pt checkpoint into out_dir/{model.safetensors,config.json,
    tokenizer.json}. Returns the manifest entry for it.
    """
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in data or "config" not in data:
        die(f"{ckpt_path} is not a PhysisML checkpoint "
            "(expected keys 'state_dict' and 'config')")

    sd  = data["state_dict"]
    cfg = dict(data["config"])

    # The LM head is weight-TIED to the token embedding (lm_head.weight IS
    # tok_emb.weight), so the checkpoint serialises the same 4.6M values twice.
    # safetensors refuses shared storage; cloning would silently untie them and
    # inflate the file by 18MB, and summing numel() over the state dict would
    # report 28.2M parameters for a 23.6M model. Drop the duplicates instead
    # and let the loader re-tie — TorchGPT.__init__ does it unconditionally.
    seen, tensors, tied = {}, {}, []
    for k, v in sd.items():
        v = v.detach().cpu()
        ptr = v.data_ptr()
        if ptr in seen:
            tied.append((k, seen[ptr]))
            continue
        seen[ptr] = k
        tensors[k] = v.contiguous().clone()
    n_params = sum(v.numel() for v in tensors.values())
    if tied:
        print("  tied weights, stored once: "
              + ", ".join(f"{a} = {b}" for a, b in tied))

    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "model.safetensors")
    save_file(tensors, weights_path, metadata={"format": "pt"})

    tok_src = find_tokenizer(ckpt_path)
    shutil.copy2(tok_src, os.path.join(out_dir, "tokenizer.json"))

    with open(tok_src, encoding="utf-8") as f:
        tok = json.load(f)

    out_cfg = {
        "model_type":        "physisml",
        "architecture":      "decoder-only transformer, pre-LayerNorm, "
                             "LM head weight-tied to the token embedding",
        "tied_weights":      {k: v for k, v in tied},
        "vocab_size":        cfg["vocab_size"],
        "active_vocab_size": cfg.get("active_vocab_size", cfg["vocab_size"]),
        "d_model":           cfg["d_model"],
        "n_heads":           cfg["n_heads"],
        "n_layers":          cfg["n_layers"],
        "d_ff":              cfg["d_ff"],
        "max_seq_len":       cfg["max_seq_len"],
        "context_window":    CONTEXT_WINDOW,
        "dropout_p":         0.0,
        "torch_dtype":       "float32",
        "n_parameters":      n_params,
        "tokenizer":         {
            "type":           "byte-level BPE",
            "vocab_entries":  len(tok.get("vocab", {})),
            "merges":         len(tok.get("merges", [])),
            "special_tokens": tok.get("special_tokens", {}),
        },
        "language":          "it",
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "source_checkpoint": os.path.relpath(ckpt_path, _ROOT),
        "source_tokenizer":  os.path.relpath(tok_src, _ROOT),
        "weights":           os.path.relpath(weights_path, root_out),
        "n_parameters":      n_params,
        "sha256":            sha256(weights_path),
        "bytes":             os.path.getsize(weights_path),
    }


def parse_levels(spec: str) -> list:
    """'0-9' or '0,3,7' or '4' → list of ints."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble a Hugging Face upload folder for a checkpoint")
    ap.add_argument("--out", default="hf_upload",
                    help="output folder (default: hf_upload)")
    ap.add_argument("--ckpt", default=None,
                    help="main checkpoint (default: models/active.pt, "
                         "then standalone/model.pt)")
    ap.add_argument("--levels", default=None,
                    help="also export the per-level ladder, e.g. 0-9 or 0,4,9")
    ap.add_argument("--ckpt-base", default="models/checkpoints/it",
                    help="where the level_N directories live")
    ap.add_argument("--level-file", default="final_dreamed.pt",
                    help="which snapshot per level (default: final_dreamed.pt, "
                         "the one the published scores were measured on)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output folder if it exists")
    args = ap.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        die("safetensors is not installed — pip install safetensors")

    if os.path.exists(args.out):
        if not args.force:
            die(f"{args.out} already exists (use --force to overwrite)")
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    # ── main checkpoint ─────────────────────────────────────────────────────
    ckpt = args.ckpt
    if ckpt is None:
        ckpt = next((c for c in DEFAULT_CKPTS if os.path.exists(c)), None)
        if ckpt is None:
            die("no checkpoint found — pass --ckpt explicitly. Tried: "
                + ", ".join(os.path.relpath(c, _ROOT) for c in DEFAULT_CKPTS))
    if not os.path.exists(ckpt):
        die(f"checkpoint not found: {ckpt}")

    print(f"main checkpoint : {os.path.relpath(ckpt, _ROOT)}")
    manifest = {"main": export_one(ckpt, args.out, save_file, args.out),
                "levels": {}}
    print(f"  → model.safetensors  "
          f"({manifest['main']['n_parameters']/1e6:.2f}M params, "
          f"{manifest['main']['bytes']/1e6:.1f} MB)")

    # ── optional per-level ladder ───────────────────────────────────────────
    if args.levels:
        missing = []
        for lvl in parse_levels(args.levels):
            src = os.path.join(_ROOT, args.ckpt_base, f"level_{lvl}",
                               args.level_file)
            if not os.path.exists(src):
                missing.append(lvl)
                continue
            dst = os.path.join(args.out, "levels", f"level_{lvl}")
            print(f"level {lvl:>2}         : "
                  f"{os.path.relpath(src, _ROOT)}")
            manifest["levels"][str(lvl)] = export_one(src, dst, save_file,
                                                      args.out)
        if missing:
            # Say it out loud: a silently short ladder reads as a complete one.
            print(f"\n  ⚠ no {args.level_file} for level(s) "
                  f"{', '.join(map(str, missing))} — not exported.")

    # ── code, card, licence ─────────────────────────────────────────────────
    pkg_dst = os.path.join(args.out, PKG_NAME)
    shutil.copytree(SPLX_SRC, pkg_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"inference code  : standalone/splx/ → {PKG_NAME}/")

    for src in AFFECT_SRCS:
        if not os.path.exists(src):
            print(f"  ⚠ {os.path.relpath(src, _ROOT)} missing — "
                  f"the export will run without affective modulation.")
            continue
        code = open(src, encoding="utf-8").read()
        # Rewrite the repo-absolute import and drop the sys.path hack, which
        # points at a directory that does not exist outside the repository.
        code = code.replace("from dynamic_model.exp_b.affect_state import",
                            "from .affect_state import")
        code = code.replace(
            "sys.path.insert(0, os.path.join(os.path.dirname(__file__), "
            "'..', '..', 'tests', 'test_1'))\n", "")
        with open(os.path.join(pkg_dst, os.path.basename(src)), "w",
                  encoding="utf-8") as f:
            f.write(code)
    print(f"affect system   : dynamic_model/exp_b/ → {PKG_NAME}/")

    for src, dst_name in ((CARD_SRC, "README.md"),
                          (GENERATE_SRC, "generate.py"),
                          (LICENSE_SRC, "LICENSE")):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, dst_name))
        else:
            print(f"  ⚠ {os.path.relpath(src, _ROOT)} missing — "
                  f"{dst_name} not written.")

    with open(os.path.join(args.out, "MANIFEST.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    total = sum(os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(args.out) for fn in fns)
    print(f"\nDone: {args.out}/  ({total/1e6:.1f} MB)")
    print("\nCheck the card, then upload:")
    print("  pip install huggingface_hub && huggingface-cli login")
    print(f"  huggingface-cli upload <user>/<repo> {args.out} . --repo-type model")


if __name__ == "__main__":
    main()
