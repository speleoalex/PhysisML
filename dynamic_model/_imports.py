"""
Configure sys.path so that tests/test_1/physisml/ can be imported.
Import it as the first statement of every dynamic_model module:

    from dynamic_model._imports import *  # noqa
    from physisml.layers import Linear, LayerNorm, ...
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")

if _TEST1 not in sys.path:
    sys.path.insert(0, _TEST1)
