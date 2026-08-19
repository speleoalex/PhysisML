"""
Tests for splx/layers.py — includes numerical gradient checks.
Run with: python -m pytest tests/test_layers.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splx.layers import Linear, LayerNorm, GELU, Dropout, Embedding, FFN
from splx.utils  import numerical_gradient


# ---- helpers ---------------------------------------------------------------

def grad_check(layer, x, loss_fn, eps=1e-4, tol=1e-4):
    """
    Compare analytical dx (from layer.backward) vs numerical gradient.
    Returns max relative error.
    """
    # Analytical
    out = layer.forward(x.copy(), training=False)
    loss, dloss = loss_fn(out)
    dx_analytical = layer.backward(dloss)

    # Numerical
    def f(x_):
        return float(loss_fn(layer.forward(x_.copy(), training=False))[0])

    dx_numerical = numerical_gradient(f, x.copy(), eps=eps)
    diff = np.abs(dx_analytical - dx_numerical)
    denom = np.maximum(np.abs(dx_analytical) + np.abs(dx_numerical), 1e-8)
    return float((diff / denom).max())


def mse_loss(out):
    """Simple scalar loss for gradient checking."""
    target = np.zeros_like(out)
    loss = 0.5 * np.sum((out - target) ** 2)
    dloss = out - target
    return loss, dloss


# ---- Linear ----------------------------------------------------------------

class TestLinear:

    def test_forward_shape(self):
        layer = Linear(8, 16)
        x = np.random.randn(5, 8)
        out = layer.forward(x)
        assert out.shape == (5, 16)

    def test_grad_check_dx(self):
        np.random.seed(0)
        layer = Linear(6, 4)
        x = np.random.randn(3, 6) * 0.5
        err = grad_check(layer, x, mse_loss)
        assert err < 1e-4, f"Linear dx grad check failed: max_err={err}"

    def test_grad_check_dW(self):
        np.random.seed(1)
        layer = Linear(4, 3)
        x = np.random.randn(5, 4) * 0.5
        out = layer.forward(x)
        _, dloss = mse_loss(out)
        layer.backward(dloss)

        # Numerical gradient for W
        def f(W):
            layer.params["W"] = W
            o = layer.forward(x)
            return float(mse_loss(o)[0])

        dW_num = numerical_gradient(f, layer.params["W"].copy())
        dW_ana = layer.grads["W"]
        diff = np.abs(dW_ana - dW_num)
        denom = np.maximum(np.abs(dW_ana) + np.abs(dW_num), 1e-8)
        err = float((diff / denom).max())
        assert err < 1e-4, f"Linear dW grad check failed: max_err={err}"


# ---- LayerNorm -------------------------------------------------------------

class TestLayerNorm:

    def test_forward_shape(self):
        ln = LayerNorm(8)
        x = np.random.randn(5, 8)
        out = ln.forward(x)
        assert out.shape == (5, 8)

    def test_output_normalized(self):
        ln = LayerNorm(16)
        x = np.random.randn(4, 16) * 10 + 5
        out = ln.forward(x)
        # With gamma=1, beta=0, output should be ~zero mean, ~unit std
        assert np.allclose(out.mean(axis=-1), 0, atol=1e-6)
        assert np.allclose(out.std(axis=-1),  1, atol=1e-4)

    def test_grad_check_dx(self):
        np.random.seed(2)
        ln = LayerNorm(8)
        x = np.random.randn(4, 8) * 0.5
        err = grad_check(ln, x, mse_loss)
        assert err < 1e-4, f"LayerNorm dx grad check failed: max_err={err}"


# ---- GELU ------------------------------------------------------------------

class TestGELU:

    def test_forward_shape(self):
        act = GELU()
        x = np.random.randn(5, 8)
        out = act.forward(x)
        assert out.shape == x.shape

    def test_positive_inputs_mostly_pass(self):
        act = GELU()
        x = np.array([[5.0, 10.0, 0.0, -5.0]])
        out = act.forward(x)
        assert out[0, 0] > 4.0     # large positive mostly preserved
        assert out[0, 3] < 0.0     # large negative mostly suppressed

    def test_grad_check(self):
        np.random.seed(3)
        act = GELU()
        x = np.random.randn(3, 6) * 0.5
        err = grad_check(act, x, mse_loss)
        assert err < 1e-4, f"GELU grad check failed: max_err={err}"


# ---- Dropout ---------------------------------------------------------------

class TestDropout:

    def test_eval_passthrough(self):
        drop = Dropout(p=0.5)
        x = np.ones((100, 10))
        out = drop.forward(x, training=False)
        assert np.allclose(out, x)

    def test_training_drops_roughly_half(self):
        np.random.seed(42)
        drop = Dropout(p=0.5)
        x = np.ones((1000, 10))
        out = drop.forward(x, training=True)
        frac_zero = (out == 0).mean()
        assert 0.4 < frac_zero < 0.6, f"Unexpected dropout fraction: {frac_zero}"

    def test_backward_respects_mask(self):
        np.random.seed(0)
        drop = Dropout(p=0.5)
        x = np.random.randn(10, 5)
        out = drop.forward(x, training=True)
        dout = np.ones_like(out)
        dx = drop.backward(dout)
        # Zero positions in forward should be zero in backward
        assert np.all((out == 0) == (dx == 0))


# ---- Embedding -------------------------------------------------------------

class TestEmbedding:

    def test_forward_shape(self):
        emb = Embedding(100, 16)
        ids = np.array([0, 5, 99, 3])
        out = emb.forward(ids)
        assert out.shape == (4, 16)

    def test_backward_scatter_add(self):
        emb = Embedding(10, 4)
        ids = np.array([1, 3, 1])   # id=1 appears twice
        out = emb.forward(ids)
        dout = np.ones_like(out)
        emb.backward(dout)
        # Gradient at id=1 should be sum of two rows (= 2.0 each)
        assert np.allclose(emb.grads["W"][1], 2.0)
        assert np.allclose(emb.grads["W"][3], 1.0)
        assert np.allclose(emb.grads["W"][0], 0.0)


# ---- FFN -------------------------------------------------------------------

class TestFFN:

    def test_forward_shape(self):
        ffn = FFN(d_model=16, d_ff=64)
        x = np.random.randn(5, 16)
        out = ffn.forward(x)
        assert out.shape == (5, 16)

    def test_grad_check(self):
        np.random.seed(4)
        ffn = FFN(d_model=8, d_ff=32)
        x = np.random.randn(3, 8) * 0.3
        err = grad_check(ffn, x, mse_loss)
        assert err < 1e-4, f"FFN dx grad check failed: max_err={err}"
