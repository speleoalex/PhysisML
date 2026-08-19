"""
Atomic differentiable layers — all pure NumPy.

Each layer has:
  forward(x, training=True)  -> out
  backward(dout)             -> dx   (+ accumulates param grads internally)
  params  -> dict of {name: array}
  grads   -> dict of {name: array}   (populated after backward)
"""
import numpy as np
from .utils import DTYPE


class Linear:
    """Fully-connected layer: out = x @ W + b"""

    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        scale = np.sqrt(2.0 / d_in)
        self.params = {
            "W": np.random.randn(d_in, d_out).astype(DTYPE) * scale,
        }
        if bias:
            self.params["b"] = np.zeros(d_out, dtype=DTYPE)
        self.grads: dict = {}
        self._cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._cache = x
        out = x @ self.params["W"]
        if "b" in self.params:
            out = out + self.params["b"]
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._cache
        d_in, d_out = self.params["W"].shape
        # Reshape per gestire sia (T, d) che (B, T, d) — aggrega su tutte le
        # dimensioni tranne l'ultima (features)
        self.grads["W"] = x.reshape(-1, d_in).T @ dout.reshape(-1, d_out)
        if "b" in self.params:
            self.grads["b"] = dout.reshape(-1, d_out).sum(axis=0)
        return dout @ self.params["W"].T


class LayerNorm:
    """Layer normalisation along last axis: out = gamma*(x-mu)/sigma + beta"""

    def __init__(self, d: int, eps: float = 1e-5):
        self.eps = eps
        self.params = {
            "gamma": np.ones(d, dtype=DTYPE),
            "beta":  np.zeros(d, dtype=DTYPE),
        }
        self.grads: dict = {}
        self._cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        mu  = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out = self.params["gamma"] * x_hat + self.params["beta"]
        self._cache = (x, x_hat, mu, var)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, x_hat, mu, var = self._cache
        gamma = self.params["gamma"]
        N = x.shape[-1]
        sigma = np.sqrt(var + self.eps)

        # Gradients w.r.t. parameters
        self.grads["gamma"] = (dout * x_hat).sum(axis=tuple(range(dout.ndim - 1)))
        self.grads["beta"]  = dout.sum(axis=tuple(range(dout.ndim - 1)))

        # Gradient w.r.t. x (chain rule through variance and mean)
        dx_hat = dout * gamma
        dvar   = (dx_hat * (x - mu) * -0.5 * (var + self.eps) ** -1.5).sum(axis=-1, keepdims=True)
        dmu    = (dx_hat * (-1.0 / sigma)).sum(axis=-1, keepdims=True) \
                 + dvar * (-2.0 * (x - mu)).mean(axis=-1, keepdims=True)
        dx     = dx_hat / sigma + dvar * 2.0 * (x - mu) / N + dmu / N
        return dx


class GELU:
    """
    GELU activation using tanh approximation (identical to GPT-2):
      out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    _SQRT2OVERPI = np.sqrt(2.0 / np.pi)

    def __init__(self):
        self.params: dict = {}
        self.grads:  dict = {}
        self._cache = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        inner = self._SQRT2OVERPI * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = 0.5 * x * (1.0 + t)
        self._cache = (x, t, inner)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, t, inner = self._cache
        du_dx = self._SQRT2OVERPI * (1.0 + 3.0 * 0.044715 * x ** 2)
        dt_dx = (1.0 - t ** 2) * du_dx
        dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * dt_dx
        return dout * dgelu_dx


class Dropout:
    """
    Inverted dropout: during training, randomly zero entries and rescale.
    During inference, pass-through.
    """

    def __init__(self, p: float = 0.1):
        self.p = p
        self.params: dict = {}
        self.grads:  dict = {}
        self._mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self._mask = None
            return x
        self._mask = (np.random.rand(*x.shape) > self.p).astype(DTYPE) / (1.0 - self.p)
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return dout
        return dout * self._mask


class Embedding:
    """Token / positional embedding table. out = W[ids]."""

    def __init__(self, num_embeddings: int, d_model: int):
        self.params = {
            "W": np.random.randn(num_embeddings, d_model).astype(DTYPE) * 0.02
        }
        self.grads: dict = {}
        self._cache = None

    def forward(self, ids: np.ndarray, training: bool = True) -> np.ndarray:
        self._cache = ids
        return self.params["W"][ids]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        ids = self._cache
        dW = np.zeros_like(self.params["W"])
        np.add.at(dW, ids, dout)   # scatter-add (correct when ids repeat)
        self.grads["W"] = dW
        return None   # no upstream gradient (ids are discrete)


class FFN:
    """
    Position-wise Feed-Forward Network (2 layers, 4x expansion, GELU):
      h   = GELU(x @ W1 + b1)
      out = h @ W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int):
        self.fc1  = Linear(d_model, d_ff)
        self.act  = GELU()
        self.fc2  = Linear(d_ff, d_model)
        self.params: dict = {}   # populated via _collect_params
        self.grads:  dict = {}

    # ------------------------------------------------------------------
    # For convenient access from TransformerBlock
    # ------------------------------------------------------------------
    @property
    def _submodules(self):
        return [self.fc1, self.act, self.fc2]

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        h = self.fc1.forward(x, training)
        h = self.act.forward(h, training)
        return self.fc2.forward(h, training)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dh = self.fc2.backward(dout)
        dh = self.act.backward(dh)
        return self.fc1.backward(dh)

    def get_params(self, prefix: str = "ffn") -> dict:
        p = {}
        p.update({f"{prefix}_fc1_{k}": v for k, v in self.fc1.params.items()})
        p.update({f"{prefix}_fc2_{k}": v for k, v in self.fc2.params.items()})
        return p

    def get_grads(self, prefix: str = "ffn") -> dict:
        g = {}
        g.update({f"{prefix}_fc1_{k}": v for k, v in self.fc1.grads.items()})
        g.update({f"{prefix}_fc2_{k}": v for k, v in self.fc2.grads.items()})
        return g
