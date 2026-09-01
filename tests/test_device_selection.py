"""
Tests for choosing the compute device, and for the portability that makes the
choice free.

The CPU and the Arc are interchangeable mid-build only because a checkpoint
written on one reads back identically on the other. That is what these tests
pin down — together with the rule that a device request which cannot be
honoured says so, because a silent fallback to the CPU is a build running four
times slower with nothing in the log.

Run with:  python3 -m pytest tests/test_device_selection.py -v
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splx.torch_model import TorchGPT, get_device                    # noqa: E402
import splx.torch_model as tm                                        # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PHYSISML_DEVICE", raising=False)
    yield


def _pretend(monkeypatch, *available):
    monkeypatch.setattr(tm, "_device_available",
                        lambda n: n == "cpu" or n in available)


def test_auto_takes_the_best_device_available(monkeypatch):
    _pretend(monkeypatch, "xpu")
    assert get_device() == "xpu"
    _pretend(monkeypatch, "cuda", "xpu")
    assert get_device() == "cuda", "cuda outranks xpu"
    _pretend(monkeypatch)
    assert get_device() == "cpu"


def test_the_cpu_can_be_forced_while_a_gpu_is_present(monkeypatch):
    """The reason this exists: llama-server holds VRAM on the same card, and a
    result sometimes has to be comparable with an earlier CPU run."""
    _pretend(monkeypatch, "xpu")
    monkeypatch.setenv("PHYSISML_DEVICE", "cpu")
    assert get_device() == "cpu"


def test_an_impossible_request_falls_back_out_loud(monkeypatch, capsys):
    _pretend(monkeypatch)                      # no accelerator at all
    monkeypatch.setenv("PHYSISML_DEVICE", "xpu")
    assert get_device() == "cpu"
    err = capsys.readouterr().err
    assert "PHYSISML_DEVICE=xpu" in err and "non disponibile" in err


def test_a_request_that_can_be_honoured_is_silent(monkeypatch, capsys):
    _pretend(monkeypatch, "xpu")
    monkeypatch.setenv("PHYSISML_DEVICE", "xpu")
    assert get_device() == "xpu"
    assert capsys.readouterr().err == ""


def test_auto_and_empty_mean_the_same(monkeypatch):
    _pretend(monkeypatch, "xpu")
    for value in ("auto", "AUTO", "  auto  ", ""):
        monkeypatch.setenv("PHYSISML_DEVICE", value)
        assert get_device() == "xpu", value


def test_a_checkpoint_survives_the_round_trip(tmp_path):
    """load() maps to CPU whatever wrote the file, which is what lets a level
    be trained on the Arc and the next one on the CPU. Verified for real in
    both directions on this machine; here it is pinned for the CPU path."""
    cfg = dict(vocab_size=128, d_model=32, n_heads=4, n_layers=2, d_ff=64,
               max_seq_len=17, dropout_p=0.0, active_vocab_size=100)
    torch.manual_seed(3)
    m = TorchGPT(**cfg)
    path = str(tmp_path / "ckpt.pt")
    m.save(path)
    back = TorchGPT.load(path)
    for (na, a), (nb, b) in zip(m.state_dict().items(),
                                back.state_dict().items()):
        assert na == nb
        assert torch.equal(a, b), na
    assert back.active_vocab_size == 100
    assert back.forward(torch.randint(0, 100, (2, 17))).shape[-1] == 128
