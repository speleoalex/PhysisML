"""
DynamicEmbedding — embedding espandibile con supporto per ID sparsi (pruning).

Rispetto a Embedding di test_1:
  - expand(): aggiunge una riga senza toccare le esistenti
  - prune():  azzera una riga e la marca come "morta" (ID sparso)
  - dead_mask: maschera booleana — True = riga rimossa, grad azzerato
  - Il pruning usa ID sparsi: la riga resta nell'array ma non partecipa
    al training. Ottimizzazione futura: compattazione periodica degli ID.
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
        # ID sparsi: token rimossi dal pruning — grad azzerato, riga zeroed
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
        # Azzera gradient per righe morte (ID sparsi)
        if self.dead_ids:
            dead = list(self.dead_ids)
            dW[dead] = 0.0
        self.grads["W"] = dW

    # ------------------------------------------------------------------
    # Espansione vocabolario
    # ------------------------------------------------------------------

    def expand(self, new_id: int, init_vec: np.ndarray) -> None:
        """
        Aggiunge 1 riga all'embedding per il nuovo token.

        Se new_id == len(W): append in coda (caso normale).
        Se new_id > len(W):  padding con righe zero fino a new_id (non dovrebbe
                              accadere con ID sequenziali, ma gestito per robustezza).
        """
        W = self.params["W"]
        current_size = W.shape[0]

        if new_id == current_size:
            self.params["W"] = np.vstack([W, init_vec.reshape(1, -1)])
        elif new_id > current_size:
            # Gap inatteso — riempi con zero
            pad = np.zeros((new_id - current_size, self.d_model), dtype=DTYPE)
            self.params["W"] = np.vstack([W, pad, init_vec.reshape(1, -1)])
        else:
            # ID already exists — overwrite (e.g. after rollback)
            self.params["W"][new_id] = init_vec

    # ------------------------------------------------------------------
    # Pruning (ID sparsi)
    # ------------------------------------------------------------------

    def prune(self, token_id: int) -> None:
        """
        Rimuove logicamente il token: azzera la riga e la marca come morta.
        L'ID non viene riassegnato (ID sparso).
        Ottimizzazione futura: compattazione periodica degli ID morti.
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
