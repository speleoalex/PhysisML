"""
GPT-2 style decoder-only transformer — pure NumPy.

Architecture per block (Pre-LayerNorm):
  h   = x + Attention(LN1(x))
  out = h + FFN(LN2(h))

Full model:
  x      = TokenEmb[ids] + PosEmb[0..T]
  x      = Dropout(x)
  x      = Block_0(x) ... Block_N(x)
  x      = FinalLN(x)
  logits = x @ TokenEmb.W.T       ← weight-tied LM head
"""
import numpy as np
from .layers import LayerNorm, Dropout, Embedding, FFN
from .attention import MultiHeadSelfAttention
from .utils import log_softmax, softmax


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock:

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout_p: float = 0.1, idx: int = 0):
        self.idx     = idx
        self.ln1     = LayerNorm(d_model)
        self.attn    = MultiHeadSelfAttention(d_model, n_heads, dropout_p)
        self.ln2     = LayerNorm(d_model)
        self.ffn     = FFN(d_model, d_ff)
        self.drop1   = Dropout(dropout_p)
        self.drop2   = Dropout(dropout_p)
        self._cache: dict = {}

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # --- Attention sub-layer (pre-LN + residual) ---
        ln1_out  = self.ln1.forward(x, training)
        attn_out = self.attn.forward(ln1_out, training)
        attn_out = self.drop1.forward(attn_out, training)
        h = x + attn_out

        # --- FFN sub-layer (pre-LN + residual) ---
        ln2_out  = self.ln2.forward(h, training)
        ffn_out  = self.ffn.forward(ln2_out, training)
        ffn_out  = self.drop2.forward(ffn_out, training)
        out = h + ffn_out

        self._cache = {"x": x, "h": h, "ln1_out": ln1_out,
                       "attn_out": attn_out, "ln2_out": ln2_out, "ffn_out": ffn_out}
        return out

    # ------------------------------------------------------------------
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # --- FFN sub-layer backward ---
        # out = h + drop2(ffn(ln2(h)))   =>  dout flows into both h and ffn branch
        dffn_drop = self.drop2.backward(dout)
        dffn_ln2  = self.ffn.backward(dffn_drop)
        dh_from_ffn = self.ln2.backward(dffn_ln2)

        dh = dout + dh_from_ffn   # gradient through residual + FFN branch

        # --- Attention sub-layer backward ---
        # h = x + drop1(attn(ln1(x)))
        dattn_drop = self.drop1.backward(dh)
        dattn_ln1  = self.attn.backward(dattn_drop)
        dx_from_attn = self.ln1.backward(dattn_ln1)

        dx = dh + dx_from_attn   # gradient through residual + attention branch
        return dx

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        p = f"block{self.idx}"
        params = {}
        params.update({f"{p}_ln1_{k}": v  for k, v in self.ln1.params.items()})
        params.update({f"{p}_ln2_{k}": v  for k, v in self.ln2.params.items()})
        params.update(self.attn.get_params(prefix=f"{p}_attn"))
        params.update(self.ffn.get_params(prefix=f"{p}_ffn"))
        return params

    def get_grads(self) -> dict:
        p = f"block{self.idx}"
        grads = {}
        grads.update({f"{p}_ln1_{k}": v  for k, v in self.ln1.grads.items()})
        grads.update({f"{p}_ln2_{k}": v  for k, v in self.ln2.grads.items()})
        grads.update(self.attn.get_grads(prefix=f"{p}_attn"))
        grads.update(self.ffn.get_grads(prefix=f"{p}_ffn"))
        return grads


# ---------------------------------------------------------------------------
# GPT — full decoder-only model
# ---------------------------------------------------------------------------

class GPT:
    """
    GPT-2 style language model.

    Parameters
    ----------
    vocab_size  : size of token vocabulary (output of BPE tokenizer)
    d_model     : embedding / hidden dimension
    n_heads     : number of attention heads
    n_layers    : number of transformer blocks
    d_ff        : feed-forward hidden dimension (typically 4*d_model)
    max_seq_len : maximum sequence length (for positional embeddings)
    dropout_p   : dropout probability (0 = no dropout)
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int,
                 dropout_p: float = 0.1):
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_seq_len, d_model)
        self.emb_drop = Dropout(dropout_p)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout_p, idx=i)
            for i in range(n_layers)
        ]

        # Final layer norm
        self.final_ln = LayerNorm(d_model)

        # LM head is weight-tied to tok_emb.W  (no separate params)

        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, ids: np.ndarray, training: bool = True) -> np.ndarray:
        """
        ids : int array shape (T,) oppure (B, T)
        returns logits shape (T, V) oppure (B, T, V)
        """
        _squeeze = ids.ndim == 1
        if _squeeze:
            ids = ids[np.newaxis]      # (1, T)

        B, T = ids.shape
        pos = np.arange(T)

        tok   = self.tok_emb.forward(ids, training)    # (B, T, d_model)
        pos_v = self.pos_emb.forward(pos, training)    # (T, d_model) — broadcast su B
        x = self.emb_drop.forward(tok + pos_v, training)

        for block in self.blocks:
            x = block.forward(x, training)

        x = self.final_ln.forward(x, training)         # (B, T, d_model)
        logits = x @ self.tok_emb.params["W"].T        # (B, T, vocab_size)

        self._cache = {"ids": ids, "pos": pos, "x_final": x,
                       "B": B, "_squeeze": _squeeze}
        return logits[0] if _squeeze else logits

    # ------------------------------------------------------------------
    # Loss (cross-entropy, predicting next token)
    # ------------------------------------------------------------------

    def loss(self, logits: np.ndarray, ids: np.ndarray):
        """
        Autoregressive cross-entropy loss.
        Supporta sia (T, V)/(T,) che (B, T, V)/(B, T).
        Returns (scalar_loss, dlogits)
        """
        _squeeze = logits.ndim == 2
        if _squeeze:
            logits = logits[np.newaxis]
            ids    = ids[np.newaxis]

        B, T, V = logits.shape
        logits_pred = logits[:, :-1, :]      # (B, T-1, V)
        targets     = ids[:, 1:]             # (B, T-1)
        N = B * (T - 1)

        log_probs = log_softmax(logits_pred)
        # Gather log-prob at target positions
        b_idx = np.arange(B)[:, np.newaxis]
        t_idx = np.arange(T - 1)[np.newaxis, :]
        loss = -log_probs[b_idx, t_idx, targets].mean()

        probs = softmax(logits_pred)
        probs[b_idx, t_idx, targets] -= 1.0
        dlogits_pred = probs / N             # (B, T-1, V)

        dlogits = np.zeros_like(logits)
        dlogits[:, :-1, :] = dlogits_pred

        return loss, (dlogits[0] if _squeeze else dlogits)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dlogits: np.ndarray) -> None:
        """
        Backward pass. dlogits can be (T, V) or (B, T, V).
        """
        x_final = self._cache["x_final"]      # (B, T, d)
        _sq = self._cache["_squeeze"]

        if _sq:
            dlogits = dlogits[np.newaxis]      # (1, T, V)

        B, T, V = dlogits.shape
        d = self.d_model

        # LM head backward: logits = x_final @ W_emb.T
        dx_final  = dlogits @ self.tok_emb.params["W"]          # (B, T, d)
        # dW_emb da LM head: aggrega su B e T → (d, V)
        dW_emb_lm = x_final.reshape(-1, d).T @ dlogits.reshape(-1, V)

        dx = self.final_ln.backward(dx_final)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        dx = self.emb_drop.backward(dx)

        self.tok_emb.backward(dx)
        # pos_emb usa ids 1D (T,): accumula il gradiente sommando su B
        self.pos_emb.backward(dx.sum(axis=0) if dx.ndim == 3 else dx)

        self.tok_emb.grads["W"] = self.tok_emb.grads.get("W", 0) + dW_emb_lm.T

    # ------------------------------------------------------------------
    # Parameter / gradient collection (flat dicts for optimizer)
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        params = {}
        params.update({"tok_emb_W":  self.tok_emb.params["W"]})
        params.update({"pos_emb_W":  self.pos_emb.params["W"]})
        params.update({"final_ln_gamma": self.final_ln.params["gamma"],
                       "final_ln_beta":  self.final_ln.params["beta"]})
        for block in self.blocks:
            params.update(block.get_params())
        return params

    def get_grads(self) -> dict:
        grads = {}
        grads.update({"tok_emb_W":  self.tok_emb.grads.get("W", np.zeros_like(self.tok_emb.params["W"]))})
        grads.update({"pos_emb_W":  self.pos_emb.grads.get("W", np.zeros_like(self.pos_emb.params["W"]))})
        grads.update({"final_ln_gamma": self.final_ln.grads.get("gamma", np.zeros(self.d_model)),
                      "final_ln_beta":  self.final_ln.grads.get("beta",  np.zeros(self.d_model))})
        for block in self.blocks:
            grads.update(block.get_grads())
        return grads

    def apply_params(self, params: dict) -> None:
        """Write updated parameter values back into the model objects."""
        self.tok_emb.params["W"]       = params["tok_emb_W"]
        self.pos_emb.params["W"]       = params["pos_emb_W"]
        self.final_ln.params["gamma"]  = params["final_ln_gamma"]
        self.final_ln.params["beta"]   = params["final_ln_beta"]
        for block in self.blocks:
            p = f"block{block.idx}"
            block.ln1.params["gamma"] = params[f"{p}_ln1_gamma"]
            block.ln1.params["beta"]  = params[f"{p}_ln1_beta"]
            block.ln2.params["gamma"] = params[f"{p}_ln2_gamma"]
            block.ln2.params["beta"]  = params[f"{p}_ln2_beta"]
            for name, layer in [("Wq", block.attn.Wq), ("Wk", block.attn.Wk),
                                 ("Wv", block.attn.Wv), ("Wo", block.attn.Wo)]:
                layer.params["W"] = params[f"{p}_attn_{name}_W"]
                if "b" in layer.params:
                    layer.params["b"] = params[f"{p}_attn_{name}_b"]
            block.ffn.fc1.params["W"] = params[f"{p}_ffn_fc1_W"]
            block.ffn.fc1.params["b"] = params[f"{p}_ffn_fc1_b"]
            block.ffn.fc2.params["W"] = params[f"{p}_ffn_fc2_W"]
            block.ffn.fc2.params["b"] = params[f"{p}_ffn_fc2_b"]

    def save(self, path: str) -> None:
        np.savez(path, **self.get_params())

    def load(self, path: str) -> None:
        data = np.load(path)
        self.apply_params({k: data[k] for k in data.files})

    @property
    def num_params(self) -> int:
        return sum(v.size for v in self.get_params().values())
