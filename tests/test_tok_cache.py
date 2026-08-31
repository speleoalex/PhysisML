"""
Tests for the tokenised-corpus cache.

The cache exists for one measured reason: the dream re-encodes the same 4.3MB
of corpus on every cycle, 125s a time, six times per level — about 2.7 hours of
a 55-hour build. The danger it introduces is the interesting part: ids reused
across a vocabulary change would be silently WRONG, because one new merge
re-segments the text. So most of what is tested here is invalidation.

Run with:  python3 -m pytest tests/test_tok_cache.py -v
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_model.exp_b import trainer as tr                        # noqa: E402


class FakeTok:
    """A tokenizer whose encoding depends on its merges, like the real one."""

    def __init__(self, merges=("a b",)):
        self.vocab  = {i: c for i, c in enumerate("abcdefg ")}
        self.merges = list(merges)
        self.calls  = 0

    def __len__(self):
        return len(self.vocab)

    def encode(self, text):
        self.calls += 1
        # the merge count changes the ids, which is the whole point
        return [(ord(c) % 7) + len(self.merges) for c in text]


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "_TOK_CACHE_DIR", str(tmp_path / "tokcache"))
    yield


def test_the_second_call_reads_the_cache_and_returns_identical_ids():
    tok, text = FakeTok(), "abcabc defg " * 200
    a, hit_a = tr.encode_cached(tok, text)
    b, hit_b = tr.encode_cached(tok, text)
    assert (hit_a, hit_b) == (False, True)
    assert np.array_equal(a, b)
    assert tok.calls == 1, "the second call must not re-encode"
    assert a.dtype == np.int32


def test_a_new_merge_invalidates_the_cache():
    """The failure this guards: one merge re-segments everything, so ids from
    the old vocabulary would train the model on the wrong tokens."""
    text = "abcabc defg " * 200
    tok = FakeTok()
    ids_before, _ = tr.encode_cached(tok, text)
    grown = FakeTok(merges=("a b", "b c"))          # N2-B added a token
    ids_after, hit = tr.encode_cached(grown, text)
    assert hit is False
    assert not np.array_equal(ids_before, ids_after)


def test_a_different_text_is_a_different_entry():
    tok = FakeTok()
    tr.encode_cached(tok, "abc " * 100)
    _, hit = tr.encode_cached(tok, "abd " * 100)
    assert hit is False and tok.calls == 2


def test_ids_from_a_larger_vocabulary_are_refused():
    """Belt and braces on top of the key: an entry whose ids do not fit the
    current vocabulary is re-encoded rather than used."""
    tok, text = FakeTok(), "abc " * 50
    tr.encode_cached(tok, text)
    path = [os.path.join(tr._TOK_CACHE_DIR, f)
            for f in os.listdir(tr._TOK_CACHE_DIR)][0]
    np.save(path, np.array([len(tok) + 5], dtype=np.int32))   # out of range
    _, hit = tr.encode_cached(tok, text)
    assert hit is False


def test_a_corrupt_entry_does_not_break_the_build():
    tok, text = FakeTok(), "abc " * 50
    tr.encode_cached(tok, text)
    path = [os.path.join(tr._TOK_CACHE_DIR, f)
            for f in os.listdir(tr._TOK_CACHE_DIR)][0]
    with open(path, "wb") as f:
        f.write(b"not a numpy file")
    ids, hit = tr.encode_cached(tok, text)
    assert hit is False and len(ids) == len(text)


def test_an_unwritable_cache_directory_does_not_break_the_build(monkeypatch):
    monkeypatch.setattr(tr, "_TOK_CACHE_DIR", "/proc/definitely/not/writable")
    tok, text = FakeTok(), "abc " * 50
    ids, hit = tr.encode_cached(tok, text)
    assert hit is False and len(ids) == len(text)


def test_the_cache_is_pruned_to_its_cap(monkeypatch):
    monkeypatch.setattr(tr, "_TOK_CACHE_KEEP", 3)
    tok = FakeTok()
    for i in range(8):
        tr.encode_cached(tok, f"text number {i} " * 20)
    assert len(os.listdir(tr._TOK_CACHE_DIR)) <= 3


def test_no_temporary_file_is_left_behind():
    tok = FakeTok()
    tr.encode_cached(tok, "abc " * 50)
    assert not [f for f in os.listdir(tr._TOK_CACHE_DIR) if ".tmp" in f]
