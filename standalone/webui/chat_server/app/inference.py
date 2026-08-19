"""
Model/tokenizer loading and text generation.

The model and tokenizer are held as module-level globals so the training
job can swap them atomically after a fine-tuning run. `generate_reply`
runs the blocking PyTorch forward loop in a thread to avoid stalling the
FastAPI event loop.

Generation logic mirrors standalone/chat.py::generate() — intentionally
duplicated rather than imported to avoid chat.py's top-level
install_missing_packages() side effect.
"""
import asyncio
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from app.config import settings


# Allow `import splx.*` from standalone/splx/
_STANDALONE = settings.resolve(settings.STANDALONE_DIR)
if str(_STANDALONE) not in sys.path:
    sys.path.insert(0, str(_STANDALONE))

from splx.torch_model import TorchGPT        # noqa: E402
from splx.tokenizer   import BPETokenizer    # noqa: E402
from splx.utils       import sample_top_k    # noqa: E402


CONTEXT_WINDOW = 128

# Single lock guarding model/tokenizer pointer swaps.
_state_lock = threading.Lock()

_model: TorchGPT | None = None
_tokenizer: BPETokenizer | None = None


def load() -> None:
    """Load model + tokenizer from disk into module globals."""
    global _model, _tokenizer

    model_path = settings.resolve(settings.MODEL_PATH)
    tok_path   = settings.resolve(settings.TOKENIZER_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    torch.set_num_threads(os.cpu_count() or 4)

    tok = BPETokenizer()
    tok.load(str(tok_path))

    model = TorchGPT.load(str(model_path))
    model.eval()

    with _state_lock:
        _model = model
        _tokenizer = tok


def get_state() -> tuple[TorchGPT, BPETokenizer]:
    with _state_lock:
        if _model is None or _tokenizer is None:
            raise RuntimeError("Model not loaded")
        return _model, _tokenizer


def replace_state(model: TorchGPT, tokenizer: BPETokenizer) -> None:
    """Swap the live model/tokenizer — used after Train now completes."""
    global _model, _tokenizer
    model.eval()
    with _state_lock:
        _model = model
        _tokenizer = tokenizer


def _generate_sync(prompt: str, max_tokens: int,
                   temperature: float, top_k: int) -> str:
    model, tok = get_state()

    prompt_ids = tok.encode(prompt)
    if not prompt_ids:
        return ""

    ids = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            ctx    = torch.tensor(ids[-CONTEXT_WINDOW:], dtype=torch.long)
            logits = model.forward(ctx)       # (T, V)
            last   = logits[-1].numpy()       # (V,)

            next_id = sample_top_k(last, k=top_k, temperature=temperature)
            ids.append(int(next_id))

            if tok.is_special(int(next_id)):
                break

    generated_ids = ids[len(prompt_ids):]
    return tok.decode(generated_ids)


async def generate_reply(prompt: str,
                         max_tokens: int,
                         temperature: float,
                         top_k: int) -> str:
    return await asyncio.to_thread(
        _generate_sync, prompt, max_tokens, temperature, top_k,
    )


def model_info() -> dict:
    model, tok = get_state()
    return {
        "num_params":        int(model.num_params),
        "vocab_size":        int(model.vocab_size),
        "active_vocab_size": int(model.active_vocab_size),
        "d_model":           int(model.d_model),
        "n_heads":           int(model.n_heads),
        "n_layers":          int(model.n_layers),
        "context_window":    CONTEXT_WINDOW,
        "model_path":        str(settings.resolve(settings.MODEL_PATH)),
        "tokenizer_path":    str(settings.resolve(settings.TOKENIZER_PATH)),
    }
