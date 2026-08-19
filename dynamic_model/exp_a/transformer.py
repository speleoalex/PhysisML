"""
TorchDynamicGPT — wrapper of TorchGPT for dynamic vocabulary.

TorchGPT already implements vocab_expand(init_vecs: Tensor), so
this class only adds Experiment A-specific utilities:
- Automatic np.ndarray → torch.Tensor conversion for init_vecs
- num_params property (already in TorchGPT, re-exposed for compatibility)

For everything else, use TorchGPT directly:
  forward(ids)          — ids can be np.ndarray or torch.Tensor
  loss(logits, ids)     — returns scalar
  train_step(ids, opt)  — full step, returns loss float
  vocab_expand(vecs)    — adds tokens, vecs: (N, d_model) Tensor or ndarray
  save(path) / load(path)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

import numpy as np
import torch

from splx.torch_model import TorchGPT


class TorchDynamicGPT(TorchGPT):
    """
    Estende TorchGPT con:
    - Accetta np.ndarray come input (li converte automaticamente)
    - vocab_expand accetta sia np.ndarray che torch.Tensor
    """

    def forward(self, ids, training: bool = True):
        """Accetta ids come np.ndarray o torch.Tensor."""
        if isinstance(ids, np.ndarray):
            ids = torch.from_numpy(ids).long()
        with torch.set_grad_enabled(training):
            return super().forward(ids)

    def loss(self, logits: torch.Tensor, ids):
        if isinstance(ids, np.ndarray):
            ids = torch.from_numpy(ids).long()
        return super().loss(logits, ids)

    def train_step(self, ids, optimizer, max_norm: float = 1.0) -> float:
        if isinstance(ids, np.ndarray):
            ids = torch.from_numpy(ids).long()
        return super().train_step(ids, optimizer._opt
                                   if hasattr(optimizer, '_opt') else optimizer,
                                   max_norm)

    def vocab_expand(self, init_vecs) -> None:
        """
        Aggiunge token al vocabolario.
        init_vecs: np.ndarray (N, d) o torch.Tensor (N, d)
        """
        if isinstance(init_vecs, np.ndarray):
            init_vecs = torch.from_numpy(init_vecs).float()
        if init_vecs.dim() == 1:
            init_vecs = init_vecs.unsqueeze(0)
        super().vocab_expand(init_vecs)
