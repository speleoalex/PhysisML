"""
TorchGPT — PyTorch backend for the GPT-2 style transformer.

Same architecture as GPT (splx/transformer.py) but uses PyTorch + autograd.
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

Esempio:
    from splx.torch_model import TorchGPT, TorchAdamOptimizer
    model = TorchGPT(vocab_size=8000, d_model=256, n_heads=4, n_layers=4,
                     d_ff=1024, max_seq_len=129, dropout_p=0.1,
                     active_vocab_size=501)   # 501 attivi, 7499 dormienti
    opt   = TorchAdamOptimizer(model.parameters(), lr=1e-3)
    loss  = model.train_step(ids_batch, opt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


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

        # TransformerEncoderLayer con norm_first=True (Pre-LN, come GPT-2)
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

        # Weight tying: LM head condivide i pesi con l'embedding
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

    def loss(self, logits: torch.Tensor, ids: torch.Tensor):
        """
        Cross-entropy loss autoregressive.
        logits: (T,V) o (B,T,V)  |  ids: (T,) o (B,T)
        Returns: scalar loss
        """
        squeeze = logits.dim() == 2
        if squeeze:
            logits = logits.unsqueeze(0)
            ids    = ids.unsqueeze(0)
        B, T, V = logits.shape
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            ids[:, 1:].reshape(-1))

    def train_step(self, ids: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   max_norm: float = 1.0) -> float:
        """Un gradient step completo. Ritorna la loss scalare."""
        optimizer.zero_grad()
        logits = self.forward(ids)
        loss   = self.loss(logits, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Slot activation — vocabolario con crescita biologica
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
    # Persistenza
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
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
        }, path)

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
# Optimizer wrapper con stessa interfaccia di AdamOptimizer
# ---------------------------------------------------------------------------

class TorchAdamOptimizer:
    """
    Wrapper leggero per torch.optim.Adam con interfaccia compatibile.

    weight_decay defaults to 0: with torch.optim.Adam the decay is COUPLED
    (added to the gradient) and progressively kills rarely-used embeddings
    and LayerNorm gains during online training. See PhysisML
    tests/test_1/splx/torch_model.py for details.
    """
    def __init__(self, parameters, lr=1e-3, weight_decay=0.0,
                 betas=(0.9, 0.999), eps=1e-8):
        self._opt = torch.optim.Adam(
            parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def zero_grad(self):  self._opt.zero_grad()
    def step(self):       self._opt.step()

    def state_dict(self): return self._opt.state_dict()
    def load_state_dict(self, sd): self._opt.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Helper: converte ids numpy → torch e viceversa
# ---------------------------------------------------------------------------

def to_torch(ids: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ids).long()

def ids_batch(all_ids: np.ndarray, starts, block_size: int) -> torch.Tensor:
    """Costruisce un mini-batch di sequenze da un array pre-codificato."""
    seqs = [all_ids[s:s + block_size + 1] for s in starts]
    return torch.from_numpy(np.stack(seqs)).long()
