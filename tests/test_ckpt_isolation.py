"""
Tests for the two ways an isolated checkpoint tree used to lie.

Both cost a four-hour experiment before they were found, and both were silent:

  * `--ckpt-base` did not redirect affect_memory.json, so a run against an
    isolated tree inherited the real build's curiosity memory and overwrote it
    on the way out;
  * the dream's memory bank (N3) reads every earlier level's session logs from
    the checkpoint tree. An isolated tree has none, so N3 had nothing to replay
    while N1 kept working normally — the same checkpoint and the same single
    dream recovered to 30.4% instead of 51.8%.

Run with:  python3 -m pytest tests/test_ckpt_isolation.py -v
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_model.train_curriculum import (_affect_memory_path,      # noqa: E402
                                            _load_memory_bank)


def test_the_curiosity_memory_follows_the_checkpoint_base():
    default = _affect_memory_path("it")
    assert default == os.path.join("models", "checkpoints", "it",
                                   "affect_memory.json"), \
        "a run without --ckpt-base must keep writing where it always did"
    isolated = _affect_memory_path("it", "models/checkpoints/exp/arm5")
    assert isolated == os.path.join("models", "checkpoints", "exp", "arm5",
                                    "affect_memory.json")
    assert isolated != default


def _write_session(d, name="session_20260101_000000.jsonl"):
    os.makedirs(d, exist_ok=True)
    rows = [
        {"turn": 1, "prompt": "di: il cane", "expected": "il cane!",
         "response": "il cane!", "feedback": None},
        # the grade of row 1 lives on row 2 — the offset the bank relies on
        {"turn": 2, "prompt": "di: la mamma", "expected": "la mamma!",
         "response": "la mamma!", "feedback": "+++"},
        {"turn": 3, "prompt": "di: il pane", "expected": "il pane!",
         "response": "pane", "feedback": "+++"},
    ]
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_a_bank_with_only_the_current_level_says_so(tmp_path):
    base = str(tmp_path / "ckpt")
    _write_session(os.path.join(base, "level_6"))
    out = io.StringIO()
    with redirect_stdout(out):
        bank = _load_memory_bank(base, 6)
    msg = out.getvalue()
    assert "ATTENZIONE" in msg and "1 livello/i su 7" in msg, msg
    assert "session_*.jsonl" in msg, "the message must say what to copy"
    assert bank, "the current level's own entries are still loaded"


def test_a_complete_tree_is_silent(tmp_path):
    base = str(tmp_path / "ckpt")
    for lvl in range(7):
        _write_session(os.path.join(base, f"level_{lvl}"))
    out = io.StringIO()
    with redirect_stdout(out):
        bank = _load_memory_bank(base, 6)
    assert "ATTENZIONE" not in out.getvalue()
    assert len({e["level"] for e in bank}) == 7, \
        "every level's memory must reach the bank"


def test_level_zero_never_warns(tmp_path):
    """There is no earlier level to be missing."""
    base = str(tmp_path / "ckpt")
    _write_session(os.path.join(base, "level_0"))
    out = io.StringIO()
    with redirect_stdout(out):
        _load_memory_bank(base, 0)
    assert "ATTENZIONE" not in out.getvalue()
