"""
DynamicAdam — estende AdamOptimizer con expand_moments() per vocabolario crescente.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

import numpy as np
from physisml.optimizer import AdamOptimizer


class DynamicAdam(AdamOptimizer):

    def expand_moments(self, param_name: str, n_new: int = 1) -> None:
        """
        Extend Adam moments for a parameter that has grown by n_new rows.

        Zero initialisation: mathematically correct for a token with no history.
        Adam will use an effective learning rate ~ lr in the first iterations
        (large bias correction for small t), then stabilises.

        Call AFTER model.vocab_expand() and BEFORE the next optimizer.step().
        """
        if param_name not in self._m:
            return  # momenti non ancora inizializzati — nessuna azione necessaria

        m = self._m[param_name]
        v = self._v[param_name]

        # Determina la shape di una nuova riga
        new_m = np.zeros((n_new,) + m.shape[1:], dtype=m.dtype)
        new_v = np.zeros((n_new,) + v.shape[1:], dtype=v.dtype)

        self._m[param_name] = np.concatenate([m, new_m], axis=0)
        self._v[param_name] = np.concatenate([v, new_v], axis=0)
