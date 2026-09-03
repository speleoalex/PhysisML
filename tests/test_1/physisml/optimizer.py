"""
Adam optimizer — pure NumPy (Kingma & Ba, 2015).

Algorithm per parameter θ:
  m_t = β₁·m_{t-1} + (1−β₁)·g_t
  v_t = β₂·v_{t-1} + (1−β₂)·g_t²
  m̂_t = m_t / (1−β₁^t)
  v̂_t = v_t / (1−β₂^t)
  θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)

Optionally applies weight decay (AdamW style) only to weight matrices,
not to biases or layer-norm parameters.
"""
import numpy as np
from .utils import DTYPE


class AdamOptimizer:

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-7 if DTYPE == np.float32 else 1e-8,  # float32 ha meno precisione
        weight_decay: float = 0.0,
    ):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0             # global step counter
        self._m: dict = {}               # first moments
        self._v: dict = {}               # second moments

    def step(self, params: dict, grads: dict) -> dict:
        """
        Update all parameters and return the new params dict.

        Parameters
        ----------
        params : {name: np.ndarray}  current parameter values
        grads  : {name: np.ndarray}  gradients (same keys as params)

        Returns
        -------
        updated params dict
        """
        self.t += 1
        t = self.t
        b1, b2, eps = self.beta1, self.beta2, self.eps

        # Bias-correction factors (precomputed once per step)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        new_params = {}
        for name, theta in params.items():
            g = grads.get(name)
            if g is None:
                new_params[name] = theta
                continue

            # Initialise moments to zero on first encounter
            if name not in self._m:
                self._m[name] = np.zeros_like(theta)
                self._v[name] = np.zeros_like(theta)

            # Weight decay (applied to gradient, AdamW style)
            if self.weight_decay > 0.0 and self._is_weight(name):
                g = g + self.weight_decay * theta

            # Update biased moment estimates
            self._m[name] = b1 * self._m[name] + (1.0 - b1) * g
            self._v[name] = b2 * self._v[name] + (1.0 - b2) * g ** 2

            # Bias-corrected estimates
            m_hat = self._m[name] / bc1
            v_hat = self._v[name] / bc2

            # Parameter update
            new_params[name] = theta - self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return new_params

    @staticmethod
    def _is_weight(name: str) -> bool:
        """Return True for weight matrices; False for biases and norm params."""
        name_lower = name.lower()
        return name_lower.endswith("_w") and "gamma" not in name_lower \
               and "beta" not in name_lower

    def state_dict(self) -> dict:
        return {"t": self.t, "m": self._m, "v": self._v}

    def load_state_dict(self, state: dict) -> None:
        self.t  = int(state["t"])
        self._m = dict(state["m"])
        self._v = dict(state["v"])
