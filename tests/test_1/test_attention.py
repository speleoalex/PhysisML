"""
Tests for physisml/attention.py
Run with: python -m pytest tests/test_attention.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physisml.attention import MultiHeadSelfAttention
from physisml.utils     import numerical_gradient


def mse_loss(out):
    target = np.zeros_like(out)
    loss = 0.5 * np.sum((out - target) ** 2)
    return float(loss), out - target


class TestMultiHeadSelfAttention:

    def test_forward_shape(self):
        mhsa = MultiHeadSelfAttention(d_model=16, n_heads=2, dropout_p=0.0)
        x = np.random.randn(5, 16)
        out = mhsa.forward(x, training=False)
        assert out.shape == (5, 16)

    def test_causal_mask(self):
        """Position i must not attend to position j > i."""
        np.random.seed(10)
        mhsa = MultiHeadSelfAttention(d_model=8, n_heads=2, dropout_p=0.0)
        x = np.random.randn(4, 8)
        mhsa.forward(x, training=False)
        attn = mhsa._cache["attn"]   # (B, H, T, T) — B=1 when input is 2D
        T = x.shape[0]
        # Take the first batch element
        attn0 = attn[0] if attn.ndim == 4 else attn   # (H, T, T)
        for h in range(attn0.shape[0]):
            for i in range(T):
                for j in range(i + 1, T):
                    assert attn0[h, i, j] < 1e-6, \
                        f"Causal mask violated: attn[{h},{i},{j}]={attn0[h,i,j]}"

    def test_grad_check_dx(self):
        np.random.seed(11)
        mhsa = MultiHeadSelfAttention(d_model=8, n_heads=2, dropout_p=0.0)
        x = np.random.randn(3, 8) * 0.3

        # Analytical
        out = mhsa.forward(x.copy(), training=False)
        loss, dloss = mse_loss(out)
        dx_ana = mhsa.backward(dloss)

        # Numerical
        def f(x_):
            return float(mse_loss(mhsa.forward(x_.copy(), training=False))[0])

        dx_num = numerical_gradient(f, x.copy())
        diff = np.abs(dx_ana - dx_num)
        denom = np.maximum(np.abs(dx_ana) + np.abs(dx_num), 1e-8)
        err = float((diff / denom).max())
        assert err < 1e-3, f"MHSA dx grad check failed: max_err={err}"

    def test_different_inputs_different_outputs(self):
        mhsa = MultiHeadSelfAttention(d_model=8, n_heads=2, dropout_p=0.0)
        x1 = np.random.randn(4, 8)
        x2 = np.random.randn(4, 8)
        out1 = mhsa.forward(x1, training=False)
        out2 = mhsa.forward(x2, training=False)
        assert not np.allclose(out1, out2)
