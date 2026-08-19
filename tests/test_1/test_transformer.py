"""
Tests for the full GPT model + training convergence.
Run with: python -m pytest tests/test_transformer.py -v
"""
import numpy as np
import math
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splx.transformer import GPT
from splx.optimizer   import AdamOptimizer
from splx.utils       import clip_grad_norm, set_seed


TINY_CFG = dict(
    vocab_size  = 32,
    d_model     = 16,
    n_heads     = 2,
    n_layers    = 1,
    d_ff        = 64,
    max_seq_len = 17,   # block_size=16 + 1
    dropout_p   = 0.0,
)


class TestGPT:

    def test_forward_shape(self):
        set_seed(0)
        model = GPT(**TINY_CFG)
        ids = np.arange(8, dtype=np.int32)
        logits = model.forward(ids, training=False)
        assert logits.shape == (8, 32)

    def test_initial_loss_near_baseline(self):
        """Random model should produce loss ≈ log(vocab_size)."""
        set_seed(0)
        model = GPT(**TINY_CFG)
        ids = np.random.randint(0, 32, size=16).astype(np.int32)
        logits = model.forward(ids, training=False)
        loss, _ = model.loss(logits, ids)
        baseline = math.log(32)
        # Allow generous margin; random initialisation varies
        assert abs(float(loss) - baseline) < 2.0, \
            f"Initial loss {loss:.3f} far from baseline {baseline:.3f}"

    def test_loss_decreases_after_training(self):
        """Loss should drop significantly on a tiny repeated sequence."""
        set_seed(42)
        model = GPT(**TINY_CFG)
        opt   = AdamOptimizer(lr=1e-2)

        # Repeat a fixed token sequence many times
        ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
                       dtype=np.int32)

        initial_loss = None
        for step in range(200):
            logits = model.forward(ids, training=False)
            loss, dlogits = model.loss(logits, ids)
            if initial_loss is None:
                initial_loss = float(loss)
            model.backward(dlogits)
            grads = clip_grad_norm(model.get_grads(), max_norm=1.0)
            new_params = opt.step(model.get_params(), grads)
            model.apply_params(new_params)

        final_loss = float(loss)
        assert final_loss < initial_loss * 0.5, \
            f"Loss did not decrease enough: {initial_loss:.3f} -> {final_loss:.3f}"

    def test_save_load(self, tmp_path):
        set_seed(5)
        model = GPT(**TINY_CFG)
        ids = np.arange(8, dtype=np.int32)
        logits_before = model.forward(ids, training=False).copy()

        path = str(tmp_path / "ckpt.npz")
        model.save(path)

        model2 = GPT(**TINY_CFG)
        model2.load(path)
        logits_after = model2.forward(ids, training=False)

        assert np.allclose(logits_before, logits_after, atol=1e-10)

    def test_num_params_reasonable(self):
        model = GPT(**TINY_CFG)
        n = model.num_params
        # Should be in a sane range for tiny config
        assert 1_000 < n < 500_000, f"Unexpected param count: {n}"
