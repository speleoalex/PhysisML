"""
PhysisML — a tiny GPT-2 style language model built from scratch in pure NumPy.

Public API
----------
BPETokenizer  : Byte-pair encoding tokenizer
GPT           : Full decoder-only transformer
AdamOptimizer : Adam parameter update rule
generate      : Autoregressive text generation helper
"""
from .tokenizer   import BPETokenizer
from .transformer import GPT
from .optimizer   import AdamOptimizer
from .utils       import set_seed, count_params

__all__ = ["BPETokenizer", "GPT", "AdamOptimizer", "set_seed", "count_params"]
