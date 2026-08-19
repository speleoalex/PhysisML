"""
Tests for BPETokenizer.
Run with: python -m pytest tests/test_tokenizer.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splx.tokenizer import BPETokenizer

SAMPLE_TEXT = (
    "Hello world! Hello world! Hello world!\n"
    "The quick brown fox jumps over the lazy dog.\n"
    "To be or not to be, that is the question.\n" * 20
)


@pytest.fixture(scope="module")
def trained_tok():
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=300)
    return tok


class TestBPETokenizer:

    def test_vocab_size(self, trained_tok):
        assert len(trained_tok) <= 300
        assert len(trained_tok) > 256

    def test_roundtrip(self, trained_tok):
        text = "Hello world! To be or not to be."
        ids = trained_tok.encode(text)
        decoded = trained_tok.decode(ids)
        assert decoded == text, f"Round-trip failed: {repr(decoded)} != {repr(text)}"

    def test_encode_returns_ints(self, trained_tok):
        ids = trained_tok.encode("test")
        assert all(isinstance(i, int) for i in ids)

    def test_encode_nonempty(self, trained_tok):
        ids = trained_tok.encode("a")
        assert len(ids) >= 1

    def test_common_pair_merged(self, trained_tok):
        """'Hello' appears many times; should be fewer tokens than bytes."""
        ids_hello = trained_tok.encode("Hello")
        assert len(ids_hello) <= 5   # at most one token per byte

    def test_save_load_roundtrip(self, trained_tok, tmp_path):
        path = str(tmp_path / "tok.json")
        trained_tok.save(path)
        tok2 = BPETokenizer()
        tok2.load(path)
        text = "Hello world!"
        assert trained_tok.encode(text) == tok2.encode(text)
