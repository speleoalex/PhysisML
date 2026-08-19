"""
Utility functions: softmax, sampling, gradient clipping, gradient check.
"""
import numpy as np

# Floating-point type used throughout the model.
# float32: ~1.5x faster, half the memory, sufficient precision for transformers.
# float64: more precise, useful for debugging and gradient checks.
DTYPE = np.float32


# ---------------------------------------------------------------------------
# Numerically stable softmax / log-softmax
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Works on any shape, operates on last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Sampling strategies for text generation
# ---------------------------------------------------------------------------

def sample_top_k(logits: np.ndarray, k: int = 40, temperature: float = 1.0) -> int:
    """Sample from top-k logits. logits shape: (vocab_size,)."""
    logits = logits / max(temperature, 1e-8)
    if k > 0:
        # Zero out everything except top-k
        top_k_indices = np.argpartition(logits, -k)[-k:]
        mask = np.full_like(logits, -1e9)
        mask[top_k_indices] = logits[top_k_indices]
        logits = mask
    probs = softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


def sample_top_p(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
    """Nucleus sampling (top-p). logits shape: (vocab_size,)."""
    logits = logits / max(temperature, 1e-8)
    probs = softmax(logits)
    # Sort descending
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    # Remove tokens once cumulative prob > p (keep at least 1)
    cutoff = np.searchsorted(cumulative, p) + 1
    top_p_idx = sorted_idx[:cutoff]
    top_p_probs = probs[top_p_idx]
    top_p_probs /= top_p_probs.sum()  # renormalize
    return int(np.random.choice(top_p_idx, p=top_p_probs))


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def clip_grad_norm(grads: dict, max_norm: float = 1.0) -> dict:
    """Global gradient norm clipping. Returns scaled grads dict."""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return {k: g * scale for k, g in grads.items()}
    return grads


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Compute numerical gradient of scalar-valued f w.r.t. x.
    Used for gradient checking in tests.
    """
    grad = np.zeros_like(x, dtype=np.float64)  # float64 per precisione nel gradient check
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        fp = float(f(x))
        x[idx] = orig - eps
        fm = float(f(x))
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def count_params(params: dict) -> int:
    """Count total number of scalar parameters."""
    return sum(v.size for v in params.values())
