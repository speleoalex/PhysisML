"""
Configura il sys.path per importare da tests/test_1/splx/.
Da importare come primo statement in ogni modulo di dynamic_model:

    from dynamic_model._imports import *  # noqa
    from splx.layers import Linear, LayerNorm, ...
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")

if _TEST1 not in sys.path:
    sys.path.insert(0, _TEST1)
