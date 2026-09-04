"""
TorchGPT — PyTorch backend for the GPT-2 style transformer.

Same architecture as GPT (physisml/transformer.py) but uses PyTorch + autograd.
Advantages:
  - 13× faster than NumPy at d=128 (MKL on CPU with 12 threads)
  - Efficient mini-batching: PyTorch d=256 B=8 → 64 seq/s vs NumPy 11 seq/s
  - Backward handled by autograd — no manual backward code to maintain
  - Vocabulary with dormant slots: vocab_size = total allocated capacity,
    active_vocab_size = currently active tokens. Dormant tokens have
    logit=-inf and gradient=0, and are activated one by one during the dream.

Compatibility:
  - Identical input/output to the NumPy version: ids (T,) or (B,T) → logits
  - save/load with torch.save / torch.load (not .npz)
  - Can be used as drop-in replacement in existing training loops

Example:
    from physisml.torch_model import TorchGPT, TorchAdamOptimizer
    model = TorchGPT(vocab_size=8000, d_model=256, n_heads=4, n_layers=4,
                     d_ff=1024, max_seq_len=129, dropout_p=0.1,
                     active_vocab_size=501)   # 501 active, 7499 dormant
    opt   = TorchAdamOptimizer(model.parameters(), lr=1e-3)
    loss  = model.train_step(ids_batch, opt)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Device detection — auto-selects best available device
# Priority: CUDA (NVIDIA) > XPU (Intel Arc) > MPS (Apple) > CPU
# ---------------------------------------------------------------------------

def _device_available(name: str) -> bool:
    if name == "cpu":
        return True
    if name == "cuda":
        return torch.cuda.is_available()
    if name == "xpu":
        return getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
    if name == "mps":
        return getattr(torch.backends, "mps", None) is not None \
               and torch.backends.mps.is_available()
    return False


def get_device() -> str:
    """The compute device: PHYSISML_DEVICE if set, else the best available.

    PHYSISML_DEVICE = auto (default) | cpu | xpu | cuda | mps

    The override exists because the two paths are interchangeable — a
    checkpoint written on the Arc reads back bit-identically on the CPU and
    vice versa, since load() maps to CPU and the trainer moves the model to
    whatever device is active — so which one runs is a scheduling decision, not
    an architectural one. Reasons to force CPU: the GPU is busy (llama-server
    holds VRAM on the same card), or a result has to be comparable with an
    earlier CPU run.

    A request that cannot be honoured falls back, LOUDLY. Silence here is how a
    build ends up running four times slower than intended with nothing in the
    log to say so — the same failure as the CUDA wheel that sat in the XPU env.
    """
    want = (os.environ.get("PHYSISML_DEVICE") or "auto").strip().lower()
    if want not in ("", "auto"):
        if _device_available(want):
            return want
        print(f"[device] PHYSISML_DEVICE={want} requested but not available: "
              f"falling back to the CPU", file=sys.stderr, flush=True)
        return "cpu"
    for cand in ("cuda", "xpu", "mps"):
        if _device_available(cand):
            return cand
    return "cpu"


DEVICE = get_device()


class TorchGPT(nn.Module):
    """
    GPT-2 style decoder-only transformer in PyTorch.
    Pre-LayerNorm, weight-tied LM head, causal mask.

    Supports dormant vocabulary slots: vocab_size is the total allocated
    capacity, active_vocab_size is how many tokens are currently visible.
    Dormant tokens (IDs >= active_vocab_size) receive logit=-inf and
    zero gradient — they never affect output and never update.
    Slots are activated one-by-one during the dream phase (N2) as new
    token patterns are consolidated from teaching experience.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int,
                 dropout_p: float = 0.1,
                 active_vocab_size: int = None):
        super().__init__()
        self.vocab_size        = vocab_size
        self.active_vocab_size = active_vocab_size if active_vocab_size is not None \
                                 else vocab_size
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop    = nn.Dropout(dropout_p)

        # TransformerEncoderLayer with norm_first=True (Pre-LN, as in GPT-2)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model    = d_model,
                nhead      = n_heads,
                dim_feedforward = d_ff,
                dropout    = dropout_p,
                activation = "gelu",
                batch_first = True,
                norm_first  = True,   # Pre-LN
            )
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: the LM head shares the weights of the embedding
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Dormant slots start at zero — they are invisible until activated
        if self.active_vocab_size < self.vocab_size:
            with torch.no_grad():
                nn.init.zeros_(self.tok_emb.weight[self.active_vocab_size:])

    def to_device(self, device: str = None) -> "TorchGPT":
        """Move model to specified device (or auto-detect best)."""
        d = device or DEVICE
        return self.to(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (T,) o (B, T)
        returns logits: (T, V) o (B, T, V)
        Dormant tokens (IDs >= active_vocab_size) have logit=-inf.
        """
        squeeze = ids.dim() == 1
        if squeeze:
            ids = ids.unsqueeze(0)   # (1, T)

        B, T = ids.shape
        pos  = torch.arange(T, device=ids.device)

        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))   # (B, T, d)

        # Causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'),
                                     device=ids.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=mask, is_causal=True)

        x      = self.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        # Mask dormant slots: -inf → softmax=0 → grad=0 → no update
        if self.active_vocab_size < self.vocab_size:
            logits = logits.clone()
            logits[..., self.active_vocab_size:] = float('-inf')

        return logits.squeeze(0) if squeeze else logits

    # ------------------------------------------------------------------
    # Loss e training step
    # ------------------------------------------------------------------

    def loss(self, logits: torch.Tensor, ids: torch.Tensor,
             prompt_len: int = 0):
        """
        Cross-entropy loss autoregressive.
        logits: (T,V) o (B,T,V)  |  ids: (T,) o (B,T)
        prompt_len: if > 0, the first prompt_len tokens are treated as
          conditioning context and excluded from the loss — only the
          predictions of tokens at index >= prompt_len contribute.
          Without this mask, prompt tokens dominate the average loss
          (prompts are longer than answers at high curriculum levels)
          and the model is mostly trained to reproduce the teacher's
          question instead of the answer.
        Returns: scalar loss
        """
        squeeze = logits.dim() == 2
        if squeeze:
            logits = logits.unsqueeze(0)
            ids    = ids.unsqueeze(0)
        B, T, V = logits.shape
        targets = ids[:, 1:]
        if prompt_len > 0:
            # Position t predicts token t+1: masking targets[:, :prompt_len-1]
            # excludes every prediction whose target is still a prompt token.
            targets = targets.clone()
            targets[:, :max(prompt_len - 1, 0)] = -100
            if (targets != -100).sum() == 0:
                # Nothing left to learn (response truncated away) — zero loss
                # that still participates in autograd without NaN.
                # Column 0 only: dormant-slot columns hold -inf and -inf*0=NaN.
                return logits[..., 0].sum() * 0.0
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            targets.reshape(-1),
            ignore_index=-100)

    def train_step(self, ids: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   max_norm: float = 1.0) -> float:
        """One complete gradient step. Returns the scalar loss."""
        optimizer.zero_grad()
        logits = self.forward(ids)
        loss   = self.loss(logits, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Slot activation — a vocabulary that grows biologically
    # ------------------------------------------------------------------

    def activate_slots(self, new_ids: list, init_vecs: torch.Tensor) -> None:
        """
        Activate dormant vocabulary slots for new tokens.

        Instead of resizing the embedding matrix (which invalidates the
        optimizer state and axiom hooks), we simply:
          1. Write the initial vectors into the pre-allocated rows
          2. Advance active_vocab_size

        new_ids   : list of token IDs to activate, must be contiguous
                    from active_vocab_size
        init_vecs : (N, d_model) — initial embeddings (from parent tokens)

        The optimizer, axiom hooks, and LM head are all unaffected.
        """
        N = len(new_ids)
        if N == 0:
            return
        expected_start = self.active_vocab_size
        for i, nid in enumerate(new_ids):
            if nid != expected_start + i:
                raise ValueError(
                    f"activate_slots: IDs must be contiguous from "
                    f"{expected_start}, got {nid} at position {i}")
        with torch.no_grad():
            self.tok_emb.weight[expected_start:expected_start + N] = init_vecs
        self.active_vocab_size += N

    # Legacy method kept for backward compatibility with old checkpoints
    def vocab_expand(self, init_vecs: torch.Tensor) -> None:
        """
        DEPRECATED — use activate_slots() instead.
        Kept for loading old checkpoints that used the resizing approach.
        """
        N = init_vecs.shape[0]
        with torch.no_grad():
            old_w = self.tok_emb.weight.data
            new_w = torch.cat([old_w, init_vecs], dim=0)
            new_emb  = nn.Embedding(self.vocab_size + N, self.d_model)
            new_emb.weight.data = new_w
            self.tok_emb = new_emb
            self.lm_head = nn.Linear(self.d_model, self.vocab_size + N, bias=False)
            self.lm_head.weight = self.tok_emb.weight
        self.vocab_size += N
        self.active_vocab_size += N

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        # Write to a sibling temp file and rename it over the target.  A
        # rename replaces the directory entry and never writes through the
        # old inode, so any hard link to the previous file (the autonomy
        # loop snapshots each batch that way) keeps the previous weights,
        # and a crash mid-write leaves the old checkpoint intact.
        tmp = path + ".tmp"
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "vocab_size":        self.vocab_size,
                "active_vocab_size": self.active_vocab_size,
                "d_model":           self.d_model,
                "n_heads":           self.n_heads,
                "n_layers":          self.n_layers,
                "d_ff":              self.blocks[0].linear1.out_features,
                "max_seq_len":       self.max_seq_len,
            }
        }, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "TorchGPT":
        data   = torch.load(path, map_location="cpu", weights_only=False)
        cfg    = data["config"]
        # active_vocab_size is optional for backward compatibility
        active = cfg.pop("active_vocab_size", cfg["vocab_size"])
        m      = cls(**cfg, dropout_p=0.0, active_vocab_size=active)
        m.load_state_dict(data["state_dict"])
        return m

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Optimizer wrapper with the same interface as AdamOptimizer
# ---------------------------------------------------------------------------

class TorchAdamOptimizer:
    """
    A thin wrapper around torch.optim.Adam with a compatible interface.

    weight_decay defaults to 0: with torch.optim.Adam the decay is COUPLED
    (added to the gradient), so any parameter whose true gradient is small
    receives a constant shrink toward zero at every step. Over the hundreds
    of thousands of single-sample steps of curriculum training this silently
    kills rarely-used embeddings and LayerNorm gains (measured on the May
    build: ln_f gain 0.87 at L0 → 0.008 at L10). If regularisation is ever
    needed, use decoupled AdamW and exclude LayerNorm/embedding params.
    """
    def __init__(self, parameters, lr=1e-3, weight_decay=0.0,
                 betas=(0.9, 0.999), eps=1e-8, foreach=None):
        # foreach=False on XPU. Adam's default multi-tensor path
        # (torch._foreach_lerp_) allocates temporaries for every parameter at
        # once, and on the Arc that is where the dream died:
        #   RuntimeError: level_zero backend failed with error: 38
        #   (UR_RESULT_ERROR_OUT_OF_HOST_MEMORY)   in _multi_tensor_adam
        # The single-tensor path uses a fraction of the peak memory for a
        # negligible slowdown at this model size (36 parameter tensors).
        if foreach is None:
            foreach = False if str(DEVICE) == "xpu" else None
        self._opt = torch.optim.Adam(
            parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            foreach=foreach)

    def zero_grad(self):  self._opt.zero_grad()
    def step(self):       self._opt.step()

    def state_dict(self): return self._opt.state_dict()
    def load_state_dict(self, sd): self._opt.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Helper: converts ids from numpy to torch and back
# ---------------------------------------------------------------------------

def to_torch(ids: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ids).long()

def ids_batch(all_ids: np.ndarray, starts, block_size: int) -> torch.Tensor:
    """Build a mini-batch of sequences from a pre-encoded array."""
    seqs = [all_ids[s:s + block_size + 1] for s in starts]
    return torch.from_numpy(np.stack(seqs)).long()
