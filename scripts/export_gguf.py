"""
Export a TorchGPT checkpoint to GGUF format for use with llama.cpp and ollama.

Architecture: GPT-2 decoder-only (Pre-LayerNorm, causal mask, weight-tied LM head).

Weight mapping  TorchGPT → GGUF:
  tok_emb.weight                   → token_embd.weight        (active rows only;
                                       scoring-only rows such as <|NONE|> are
                                       left out — llama.cpp cannot mask them)
  pos_emb.weight                   → position_embd.weight
  blocks.i.self_attn.in_proj_weight → blk.i.attn_qkv.weight   (Q|K|V concatenated)
  blocks.i.self_attn.in_proj_bias  → blk.i.attn_qkv.bias
  blocks.i.self_attn.out_proj.weight→ blk.i.attn_output.weight
  blocks.i.self_attn.out_proj.bias → blk.i.attn_output.bias
  blocks.i.norm1.{weight,bias}     → blk.i.attn_norm.{weight,bias}
  blocks.i.linear1.{weight,bias}   → blk.i.ffn_up.{weight,bias}
  blocks.i.linear2.{weight,bias}   → blk.i.ffn_down.{weight,bias}
  blocks.i.norm2.{weight,bias}     → blk.i.ffn_norm.{weight,bias}
  ln_f.{weight,bias}               → output_norm.{weight,bias}

NOTE: llama.cpp's GPT-2 backend expects weights in the same layout as
nn.Linear (out, in) — NOT transposed like HF Conv1D. Since TorchGPT uses
nn.Linear and nn.TransformerEncoderLayer (also nn.Linear internally),
no transpositions are needed.

Usage:
    python3 scripts/export_gguf.py --checkpoint models/active.pt \\
        --tokenizer models/active_tokenizer.json \\
        --output models/physisml.gguf

    # Then quantize (optional):
    /path/to/llama.cpp/build/bin/llama-quantize models/physisml.gguf \\
        models/physisml-q4km.gguf Q4_K_M

    # Import in ollama:
    echo 'FROM ./models/physisml-q4km.gguf' > Modelfile
    ollama create physisml -f Modelfile
    ollama run physisml
"""

import sys
import os
import json
import argparse
import struct

# Add llama.cpp gguf-py to path
_LLAMA_CPP = os.path.expanduser("~/git/llama.cpp/gguf-py")
if os.path.isdir(_LLAMA_CPP):
    sys.path.insert(0, _LLAMA_CPP)

# Add project paths
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "test_1"))

try:
    import gguf
except ImportError:
    print("ERROR: gguf not found. Install with:")
    print("  pip install gguf")
    print("  or add llama.cpp/gguf-py to PYTHONPATH")
    sys.exit(1)

try:
    import torch
    import numpy as np
except ImportError as e:
    print(f"ERROR: {e}. Run from physisml_gpu conda env.")
    sys.exit(1)

from physisml.torch_model import TorchGPT
from physisml.tokenizer import BPETokenizer
from dynamic_model.exp_b.none_token import mask_scoring_rows, scoring_only_ids

# A GGUF has no active_vocab_size: every row it ships can be sampled by
# llama.cpp. The scoring-only rows (`<|NONE|>`, the eleventh class) are masked
# at decoding time inside this repo, so the export enters the same mask and
# writes what a sampler here would see — the row is simply not in the file.


# ---------------------------------------------------------------------------
# Tokenizer export
# ---------------------------------------------------------------------------

def export_tokenizer(tok: BPETokenizer, writer: gguf.GGUFWriter,
                     active_vocab: int) -> None:
    """
    Write BPE tokenizer into the GGUF file.

    GGUF BPE tokenizer fields:
      tokenizer.ggml.model        = "gpt2"
      tokenizer.ggml.tokens       = list of token strings
      tokenizer.ggml.token_type   = list of token types (0=normal, 3=control)
      tokenizer.ggml.merges       = list of merge rule strings "a b"
      tokenizer.ggml.bos_token_id = -1 (no BOS for PhysisML)
      tokenizer.ggml.eos_token_id = EOS id (or -1)
    """
    # Build vocab list: index → string representation
    # GPT-2 tokenizer format: each byte is mapped to a printable unicode char
    # using the bytes_to_unicode() mapping. This is required by llama.cpp.
    tokens = []
    token_types = []
    byte_enc = _bytes_to_unicode()  # int(0-255) → unicode char

    def _token_to_gpt2_str(raw) -> str:
        """
        Convert raw token bytes to GPT-2 unicode representation.
        Each byte maps to a unique printable unicode char via byte_enc.
        This is required for llama.cpp GPT-2 BPE encoding/decoding:
          - byte 32 (space) → 'Ġ' (chr(288))
          - byte 10 (\n)    → 'ʊ' (chr(266))
          - byte 97 ('a')   → 'a' (identity for printable ASCII)
        """
        if isinstance(raw, list): raw = bytes(raw)
        if isinstance(raw, (bytes, bytearray)):
            return "".join(byte_enc.get(b, chr(b)) for b in raw)
        return str(raw)

    for i in range(min(active_vocab, len(tok.vocab))):
        raw = tok.vocab.get(i)
        if raw is None:
            tokens.append(f"[unused{i}]")
            token_types.append(5)  # LLAMA_TOKEN_TYPE_UNUSED = 5
            continue

        special_name = (tok.special_ids.get(i)
                        if hasattr(tok, "special_ids") else None)
        if special_name:
            tokens.append(special_name)
            token_types.append(3)  # CONTROL
        else:
            text = _token_to_gpt2_str(raw)
            if not text:
                text = f"[empty{i}]"
            # Use type 1 (NORMAL) for ALL non-special tokens.
            # IMPORTANT: type=0 is UNDEFINED in llama.cpp, which returns empty
            # string in token_to_piece. We need type=1 (NORMAL) for GPT-2 BPE
            # decoding via llama_decode_text (unicode → raw bytes).
            token_types.append(1)  # LLAMA_TOKEN_TYPE_NORMAL = 1
            tokens.append(text)

    # Merges: list of "token_a token_b" strings
    # Include ALL merges within active vocab — including those starting from
    # byte tokens (<0xNN>) since those are needed to reconstruct words.
    merge_strs = []
    for merge in tok.merges:
        if len(merge) >= 3:
            a, b, new_id = merge[0], merge[1], merge[2]
            if a >= active_vocab or b >= active_vocab or new_id >= active_vocab:
                continue
            a_str = tokens[a] if a < len(tokens) else f"[{a}]"
            b_str = tokens[b] if b < len(tokens) else f"[{b}]"
            merge_strs.append(f"{a_str} {b_str}")

    # EOS token id
    eos_id = -1
    if hasattr(tok, "special_tokens") and hasattr(tok, "EOS_TOKEN") \
            and tok.EOS_TOKEN in tok.special_tokens:
        eos_id = tok.special_tokens[tok.EOS_TOKEN]
    elif hasattr(tok, "get_special_id") and hasattr(tok, "EOS_TOKEN"):
        candidate = tok.get_special_id(tok.EOS_TOKEN)
        if candidate is not None:
            eos_id = candidate

    # Fallback: EOS slot is always ID 8000 in the 8K tokenizer base.
    # Use it if within active_vocab and not already found.
    EOS_FALLBACK = 8000
    if eos_id < 0 and EOS_FALLBACK < active_vocab:
        eos_id = EOS_FALLBACK

    # Mark EOS token as CONTROL type so llama.cpp stops on it
    if 0 <= eos_id < len(token_types):
        token_types[eos_id] = 3  # CONTROL

    print(f"  Tokenizer: {len(tokens)} tokens, {len(merge_strs)} merges, EOS={eos_id}")

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    if merge_strs:
        writer.add_token_merges(merge_strs)
    # Only add special token IDs when they are valid (>=0); -1 overflows uint32
    if eos_id >= 0:
        writer.add_eos_token_id(eos_id)


def _bytes_to_unicode() -> dict:
    """GPT-2 byte-to-unicode mapping (for display purposes)."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to float32 numpy array (CPU, detached)."""
    return t.detach().cpu().float().numpy()


def export_weights(model: TorchGPT, writer: gguf.GGUFWriter) -> None:
    """Map TorchGPT weights to GGUF tensor names and write them."""
    sd = model.state_dict()
    active = model.active_vocab_size
    n_layers = model.n_layers

    def write(name: str, tensor: torch.Tensor) -> None:
        arr = tensor_to_numpy(tensor)
        writer.add_tensor(name, arr)
        print(f"    {name:50s} {tuple(arr.shape)}")

    print("  Writing tensors...")

    # Token + position embeddings (active vocab rows only)
    write("token_embd.weight", sd["tok_emb.weight"][:active])
    write("position_embd.weight", sd["pos_emb.weight"])

    # Transformer blocks
    for i in range(n_layers):
        prefix = f"blocks.{i}"

        # Attention pre-norm (norm1 in Pre-LN = applied before attention)
        write(f"blk.{i}.attn_norm.weight", sd[f"{prefix}.norm1.weight"])
        write(f"blk.{i}.attn_norm.bias",   sd[f"{prefix}.norm1.bias"])

        # QKV projection — already concatenated [Q|K|V] in in_proj_weight
        # Shape: (3*d_model, d_model) — llama.cpp GPT-2 backend expects (3*d, d)
        write(f"blk.{i}.attn_qkv.weight", sd[f"{prefix}.self_attn.in_proj_weight"])
        write(f"blk.{i}.attn_qkv.bias",   sd[f"{prefix}.self_attn.in_proj_bias"])

        # Attention output projection
        write(f"blk.{i}.attn_output.weight", sd[f"{prefix}.self_attn.out_proj.weight"])
        write(f"blk.{i}.attn_output.bias",   sd[f"{prefix}.self_attn.out_proj.bias"])

        # FFN pre-norm (norm2)
        write(f"blk.{i}.ffn_norm.weight", sd[f"{prefix}.norm2.weight"])
        write(f"blk.{i}.ffn_norm.bias",   sd[f"{prefix}.norm2.bias"])

        # FFN: linear1 = up-projection, linear2 = down-projection
        write(f"blk.{i}.ffn_up.weight",   sd[f"{prefix}.linear1.weight"])
        write(f"blk.{i}.ffn_up.bias",     sd[f"{prefix}.linear1.bias"])
        write(f"blk.{i}.ffn_down.weight", sd[f"{prefix}.linear2.weight"])
        write(f"blk.{i}.ffn_down.bias",   sd[f"{prefix}.linear2.bias"])

    # Final layer norm
    write("output_norm.weight", sd["ln_f.weight"])
    write("output_norm.bias",   sd["ln_f.bias"])

    # LM head is weight-tied with tok_emb — llama.cpp reuses token_embd
    # No need to write output.weight separately


# ---------------------------------------------------------------------------
# HF GPT-2 format export (intermediate step for convert_hf_to_gguf.py)
# ---------------------------------------------------------------------------

def export_hf_format(model: TorchGPT, tok: BPETokenizer,
                     out_dir: str, layer_norm_eps: float = 1e-5) -> None:
    """
    Save model + tokenizer in HuggingFace GPT-2 format.

    HF GPT-2 uses Conv1D (weights transposed vs nn.Linear):
      Conv1D.weight shape (in, out) = nn.Linear.weight.T

    Files written:
      config.json          GPT-2 architecture config
      pytorch_model.bin    Weight dict with HF GPT-2 key names
      vocab.json           token_string → id
      merges.txt           BPE merge rules
      tokenizer_config.json
    """
    # The mask must hold while `active` is read: everything below uses the
    # local, so the block can close right away.
    with mask_scoring_rows(model, tok) as leftover:
        if leftover:
            raise ValueError(f"scoring-only rows {leftover} sit below a trained "
                             "row; an HF export cannot mask them")
        active = model.active_vocab_size
    sd = model.state_dict()

    # --- config.json ---
    config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": active,
        "n_positions": model.max_seq_len - 1,
        "n_ctx": model.max_seq_len - 1,
        "n_embd": model.d_model,
        "n_layer": model.n_layers,
        "n_head": model.n_heads,
        "n_inner": model.d_model * 4,
        "activation_function": "gelu_new",
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "layer_norm_epsilon": layer_norm_eps,
        "initializer_range": 0.02,
        "bos_token_id": None,
        "eos_token_id": tok.get_special_id(tok.EOS_TOKEN)
            if hasattr(tok, "get_special_id") and hasattr(tok, "EOS_TOKEN") else None,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- pytorch_model.bin (HF GPT-2 weight names, Conv1D transposed) ---
    # HF GPT-2 uses Conv1D where weight.shape = (in_features, out_features)
    # i.e. TRANSPOSED compared to nn.Linear. Convert_hf_to_gguf transposes back.
    hf = {}

    # Embeddings (no transpose needed)
    hf["transformer.wte.weight"] = sd["tok_emb.weight"][:active].float()
    hf["transformer.wpe.weight"] = sd["pos_emb.weight"].float()

    for i in range(model.n_layers):
        p = f"blocks.{i}"
        h = f"transformer.h.{i}"

        # Pre-attention LayerNorm
        hf[f"{h}.ln_1.weight"] = sd[f"{p}.norm1.weight"].float()
        hf[f"{h}.ln_1.bias"]   = sd[f"{p}.norm1.bias"].float()

        # Attention QKV projection: Conv1D expects (d_model, 3*d_model)
        # nn.Linear in_proj_weight is (3*d_model, d_model) → transpose
        hf[f"{h}.attn.c_attn.weight"] = sd[f"{p}.self_attn.in_proj_weight"].T.float()
        hf[f"{h}.attn.c_attn.bias"]   = sd[f"{p}.self_attn.in_proj_bias"].float()

        # Attention output: Conv1D expects (d_model, d_model)
        hf[f"{h}.attn.c_proj.weight"] = sd[f"{p}.self_attn.out_proj.weight"].T.float()
        hf[f"{h}.attn.c_proj.bias"]   = sd[f"{p}.self_attn.out_proj.bias"].float()

        # Pre-FFN LayerNorm
        hf[f"{h}.ln_2.weight"] = sd[f"{p}.norm2.weight"].float()
        hf[f"{h}.ln_2.bias"]   = sd[f"{p}.norm2.bias"].float()

        # FFN: linear1 = c_fc (d_model→d_ff), linear2 = c_proj (d_ff→d_model)
        hf[f"{h}.mlp.c_fc.weight"]   = sd[f"{p}.linear1.weight"].T.float()
        hf[f"{h}.mlp.c_fc.bias"]     = sd[f"{p}.linear1.bias"].float()
        hf[f"{h}.mlp.c_proj.weight"] = sd[f"{p}.linear2.weight"].T.float()
        hf[f"{h}.mlp.c_proj.bias"]   = sd[f"{p}.linear2.bias"].float()

    # Final LayerNorm
    hf["transformer.ln_f.weight"] = sd["ln_f.weight"].float()
    hf["transformer.ln_f.bias"]   = sd["ln_f.bias"].float()

    # LM head is weight-tied: not saved separately (HF convention)
    torch.save(hf, os.path.join(out_dir, "pytorch_model.bin"))
    print(f"  Saved {len(hf)} weight tensors → pytorch_model.bin")

    # --- Tokenizer files ---
    # vocab.json: gpt2_unicode_str → id
    byte_enc = _bytes_to_unicode()

    def tok_to_str(raw) -> str:
        if isinstance(raw, list): raw = bytes(raw)
        return "".join(byte_enc.get(b, chr(b)) for b in raw)

    vocab_json = {}
    for i in range(active):
        raw = tok.vocab.get(i)
        if raw is not None:
            s = tok_to_str(raw) or f"[empty{i}]"
            vocab_json[s] = i

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=None)

    # Invert vocab for merge lookup
    id_to_str = {i: s for s, i in vocab_json.items()}

    # merges.txt: "token_a token_b\n"
    # Only include merges where BOTH components AND the result are in active vocab.
    # Merges producing dormant slots (id >= active_vocab) would fail HF validation.
    n_merges = 0
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in tok.merges:
            if len(merge) < 3:
                continue
            a, b, new_id = merge[0], merge[1], merge[2]
            # Skip if any token is outside active vocabulary
            if a >= active or b >= active or new_id >= active:
                continue
            a_str = id_to_str.get(a)
            b_str = id_to_str.get(b)
            if a_str is None or b_str is None:
                continue
            f.write(f"{a_str} {b_str}\n")
            n_merges += 1
    print(f"  Saved vocab ({len(vocab_json)} entries) + merges ({n_merges} rules)")

    # tokenizer_config.json
    tok_config = {
        "model_type": "gpt2",
        "tokenizer_class": "GPT2Tokenizer",
        "bos_token": None,
        "eos_token": None,
        "unk_token": None,
    }
    eos_id = (tok.get_special_id(tok.EOS_TOKEN)
              if hasattr(tok, "get_special_id") and hasattr(tok, "EOS_TOKEN") else None)
    if eos_id is not None:
        tok_config["eos_token"] = id_to_str.get(eos_id, "<|endoftext|>")
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2)

    print(f"  HF format written to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TorchGPT checkpoint to GGUF for llama.cpp / ollama")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint (e.g. models/active.pt)")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (default: auto-detect alongside checkpoint)")
    parser.add_argument("--output", default=None,
                        help="Output .gguf path (default: same dir as checkpoint)")
    parser.add_argument("--layer-norm-eps", type=float, default=1e-5)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Auto-detect tokenizer
    tok_path = args.tokenizer
    if not tok_path:
        candidates = [
            os.path.join(os.path.dirname(ckpt_path), "tokenizer.json"),
            "models/active_tokenizer.json",
            "dynamic_model/data/tokenizer_8k.json",
            "dynamic_model/data/tokenizer_base.json",
        ]
        tok_path = next((p for p in candidates if os.path.exists(p)), None)
    if not tok_path:
        print("ERROR: tokenizer not found. Specify with --tokenizer.")
        sys.exit(1)

    # Output path
    out_path = args.output
    if not out_path:
        base = os.path.splitext(ckpt_path)[0]
        out_path = base + ".gguf"

    print(f"PhysisML → GGUF export")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Tokenizer  : {tok_path}")
    print(f"  Output     : {out_path}")
    print()

    # Load model
    print("Loading model...")
    model = TorchGPT.load(ckpt_path)
    model.eval()
    print(f"  vocab={model.vocab_size}  active={model.active_vocab_size}"
          f"  d={model.d_model}  L={model.n_layers}  h={model.n_heads}"
          f"  seq={model.max_seq_len}")

    # Load tokenizer
    print("Loading tokenizer...")
    tok = BPETokenizer()
    tok.load(tok_path)
    print(f"  vocab size: {len(tok)}")

    # Write GGUF directly. The whole file is written with the scoring-only
    # rows dormant, exactly as TrainerB.generate sees the model: token list,
    # merges and embedding rows all stop at the same cut, so ids stay aligned.
    print(f"\nWriting {out_path} ...")
    with mask_scoring_rows(model, tok) as leftover:
        if leftover:
            print(f"ERROR: scoring-only rows {leftover} sit below a trained row; "
                  "a GGUF cannot mask them and llama.cpp would sample them.")
            sys.exit(1)
        dropped = sorted(i for i in scoring_only_ids(tok)
                         if i >= model.active_vocab_size)
        if dropped:
            print(f"  scoring-only rows left out of the file: {dropped} "
                  f"(active {model.active_vocab_size} for the export)")

        writer = gguf.GGUFWriter(out_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GPT2])

        # Metadata
        writer.add_name("PhysisML")
        writer.add_description(
            f"PhysisML GPT-2 {model.n_layers}L d{model.d_model} h{model.n_heads} "
            f"vocab{model.active_vocab_size}")
        writer.add_author("PhysisML project")

        # Architecture
        writer.add_context_length(model.max_seq_len)  # must match pos_emb rows
        writer.add_embedding_length(model.d_model)
        writer.add_feed_forward_length(model.d_model * 4)
        writer.add_block_count(model.n_layers)
        writer.add_head_count(model.n_heads)
        writer.add_layer_norm_eps(args.layer_norm_eps)
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)

        # Tokenizer
        print("\nExporting tokenizer...")
        export_tokenizer(tok, writer, model.active_vocab_size)

        # Weights
        print("\nExporting weights...")
        export_weights(model, writer)

        # Finalize
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nDone! {out_path}  ({size_mb:.1f} MB)")
    print()
    q_path = out_path.replace('.gguf', '-q4km.gguf')
    print(f"  # Quantize to Q4_K_M (~25% size):")
    print(f"  ~/git/llama.cpp/build/bin/llama-quantize {out_path} {q_path} Q4_K_M")
    print()
    print(f"  # Test:")
    print(f"  ~/git/llama.cpp/build/bin/llama-cli -m {out_path} -p 'di: il cane' -n 20 --no-warmup")
    print()
    out_name = os.path.basename(out_path)
    print(f"  # Import in ollama:")
    print(f"  echo 'FROM ./{out_name}' > Modelfile && ollama create physisml -f Modelfile")


if __name__ == "__main__":
    main()
