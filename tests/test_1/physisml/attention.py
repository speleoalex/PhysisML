"""
Multi-head causal self-attention — pure NumPy.

Architecture (GPT-2 style):
  Q, K, V = x @ Wq, x @ Wk, x @ Wv      shapes: (T, d_model)
  split into H heads                      shapes: (H, T, d_head)
  scores = Q @ K.T / sqrt(d_head)        shape:  (H, T, T)
  + causal mask (-1e9 above diagonal)
  attn = softmax(scores)                  shape:  (H, T, T)
  ctx  = attn @ V                         shape:  (H, T, d_head)
  merge heads                             shape:  (T, d_model)
  out  = ctx @ Wo                         shape:  (T, d_model)
"""
import numpy as np
from .layers import Linear, Dropout
from .utils import softmax, DTYPE


class MultiHeadSelfAttention:

    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.1):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.scale    = np.sqrt(self.d_head)

        # Four projection matrices (no bias on qkv is fine; GPT-2 uses bias)
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

        self.dropout = Dropout(dropout_p)
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # Handles both (T, D) and (B, T, D) — adds/removes the batch dim as needed
        _squeeze = x.ndim == 2
        if _squeeze:
            x = x[np.newaxis]          # (1, T, D)

        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Step 1: project
        Q = self.Wq.forward(x, training)   # (B, T, D)
        K = self.Wk.forward(x, training)
        V = self.Wv.forward(x, training)

        # Step 2: split into heads  (B, H, T, dh)
        Q_h = Q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        K_h = K.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        V_h = V.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        # Step 3: scaled dot-product attention
        scores = Q_h @ K_h.transpose(0, 1, 3, 2) / self.scale   # (B, H, T, T)

        # Causal mask — broadcast su B e H
        mask = np.triu(np.ones((T, T), dtype=DTYPE), k=1) * -1e9
        scores = scores + mask[np.newaxis, np.newaxis, :, :]

        attn = softmax(scores)                      # (B, H, T, T)
        attn_drop = self.dropout.forward(attn, training)

        ctx = attn_drop @ V_h                       # (B, H, T, dh)

        # Step 4: merge heads
        ctx_merged = ctx.transpose(0, 2, 1, 3).reshape(B, T, D)   # (B, T, D)

        # Step 5: output projection
        out = self.Wo.forward(ctx_merged, training)                # (B, T, D)

        self._cache = {
            "x": x, "Q": Q, "K": K, "V": V,
            "Q_h": Q_h, "K_h": K_h, "V_h": V_h,
            "scores": scores, "attn": attn, "attn_drop": attn_drop,
            "ctx": ctx, "ctx_merged": ctx_merged,
            "B": B, "T": T, "H": H, "dh": dh, "_squeeze": _squeeze,
        }
        return out[0] if _squeeze else out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        B, T, H, dh = c["B"], c["T"], c["H"], c["dh"]

        if c["_squeeze"]:
            dout = dout[np.newaxis]    # (1, T, D)

        # Step 5 backward: output projection
        d_ctx_merged = self.Wo.backward(dout)                        # (B, T, D)

        # Step 4 backward: merge heads
        d_ctx = d_ctx_merged.reshape(B, T, H, dh).transpose(0, 2, 1, 3)  # (B, H, T, dh)

        # Step 3 backward: attention
        d_attn_drop = d_ctx @ c["V_h"].transpose(0, 1, 3, 2)        # (B, H, T, T)
        dV_h = c["attn_drop"].transpose(0, 1, 3, 2) @ d_ctx         # (B, H, T, dh)

        d_attn = self.dropout.backward(d_attn_drop)

        s = c["attn"]
        d_scores = s * (d_attn - (d_attn * s).sum(axis=-1, keepdims=True))
        d_scores = d_scores / self.scale

        dQ_h = d_scores @ c["K_h"]                                   # (B, H, T, dh)
        dK_h = d_scores.transpose(0, 1, 3, 2) @ c["Q_h"]

        # Step 2 backward: reshape heads back to (B, T, D)
        def _merge(dX_h):
            return dX_h.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

        dQ = _merge(dQ_h)
        dK = _merge(dK_h)
        dV = _merge(dV_h)

        # Step 1 backward: projections
        dx_q = self.Wq.backward(dQ)
        dx_k = self.Wk.backward(dK)
        dx_v = self.Wv.backward(dV)

        dx = dx_q + dx_k + dx_v
        return dx[0] if c["_squeeze"] else dx

    # ------------------------------------------------------------------
    # Parameter / gradient helpers
    # ------------------------------------------------------------------

    def get_params(self, prefix: str = "attn") -> dict:
        p = {}
        for name, layer in [("Wq", self.Wq), ("Wk", self.Wk),
                             ("Wv", self.Wv), ("Wo", self.Wo)]:
            p.update({f"{prefix}_{name}_{k}": v for k, v in layer.params.items()})
        return p

    def get_grads(self, prefix: str = "attn") -> dict:
        g = {}
        for name, layer in [("Wq", self.Wq), ("Wk", self.Wk),
                             ("Wv", self.Wv), ("Wo", self.Wo)]:
            g.update({f"{prefix}_{name}_{k}": v for k, v in layer.grads.items()})
        return g
