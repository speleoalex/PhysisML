"""
DynamicEmbedding — an expandable embedding with sparse-ID support (pruning).

Against test_1's Embedding:
  - expand(): appends a row without touching the existing ones
  - prune():  zeroes a row and marks it "dead" (sparse ID)
  - dead_mask: boolean mask — True = row removed, gradient zeroed
  - pruning uses sparse IDs: the row stays in the array but takes no part
    in training. Future optimisation: periodic compaction of the IDs.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))
from physisml.utils import DTYPE
from typing import Set


class DynamicEmbedding:

    def __init__(self, vocab_size: int, d_model: int):
        self.d_model = d_model
        self.params: dict = {
            "W": np.random.randn(vocab_size, d_model).astype(DTYPE) * 0.02
        }
        self.grads: dict = {}
        # Sparse IDs: tokens removed by pruning — gradient zeroed, row zeroed
        self.dead_ids: Set[int] = set()
        self._cache_ids = None

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, ids: np.ndarray, training: bool = True) -> np.ndarray:
        self._cache_ids = ids
        return self.params["W"][ids]

    def backward(self, dout: np.ndarray) -> None:
        ids = self._cache_ids
        dW = np.zeros_like(self.params["W"])
        np.add.at(dW, ids, dout)
        # Zero the gradient of the dead rows (sparse IDs)
        if self.dead_ids:
            dead = list(self.dead_ids)
            dW[dead] = 0.0
        self.grads["W"] = dW

    # ------------------------------------------------------------------
    # Vocabulary expansion
    # ------------------------------------------------------------------

    def expand(self, new_id: int, init_vec: np.ndarray) -> None:
        """
        Append 1 row to the embedding for the new token.

        If new_id == len(W): append at the end (the normal case).
        If new_id > len(W):  pad with zero rows up to new_id (should not happen
                              with sequential IDs, but handled for robustness).
        """
        W = self.params["W"]
        current_size = W.shape[0]

        if new_id == current_size:
            self.params["W"] = np.vstack([W, init_vec.reshape(1, -1)])
        elif new_id > current_size:
            # Unexpected gap — fill it with zeros
            pad = np.zeros((new_id - current_size, self.d_model), dtype=DTYPE)
            self.params["W"] = np.vstack([W, pad, init_vec.reshape(1, -1)])
        else:
            # ID already exists — overwrite (e.g. after rollback)
            self.params["W"][new_id] = init_vec

    # ------------------------------------------------------------------
    # Pruning (sparse IDs)
    # ------------------------------------------------------------------

    def prune(self, token_id: int) -> None:
        """
        Remove the token logically: zero the row and mark it dead.
        The ID is never reassigned (sparse ID).
        Future optimisation: periodic compaction of the dead IDs.
        """
        if token_id < self.params["W"].shape[0]:
            self.params["W"][token_id] = 0.0
        self.dead_ids.add(token_id)

    @property
    def vocab_size(self) -> int:
        return self.params["W"].shape[0]

    @property
    def active_vocab_size(self) -> int:
        return self.vocab_size - len(self.dead_ids)
