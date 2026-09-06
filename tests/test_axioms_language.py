"""The axioms have to be words of the language being trained.

An axiom protects the embedding rows of the tokens its text encodes to. Both
halves of that sentence were broken for the English curriculum: the words were
hardcoded Italian, so `mamma` came back from tokenizer_en.json as m|am|ma and
the protection froze three arbitrary English subwords instead of a word, and
the space between two words of a phrase was protected like the words, which
scaled the gradient of the single most frequent row (30% of the English
corpus, 32% of the Italian) by 0.1 for the whole run.

These tests pin both: the words of a language are whole in that language's
vocabulary, and no separator ever ends up under protection.
"""
import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_model.train_curriculum import axioms_for, tokenizer_path  # noqa: E402
from dynamic_model import language as language_manifest         # noqa: E402
from physisml.tokenizer import BPETokenizer                     # noqa: E402
from physisml.torch_model import TorchGPT, TorchAdamOptimizer   # noqa: E402

LANGS = language_manifest.available()


def _tok(lang):
    path = os.path.join(_ROOT, tokenizer_path(lang))
    if not os.path.exists(path):
        pytest.skip(f"no vocabulary for '{lang}'")
    tok = BPETokenizer()
    tok.load(path)
    return tok


def _trainer(tok):
    from dynamic_model.exp_b.trainer import TrainerB
    n = max(tok.vocab.keys()) + 1
    torch.manual_seed(0)
    model = TorchGPT(n + 8, 32, 2, 1, 64, 65, 0.0, active_vocab_size=n)
    return TrainerB(model, tok, TorchAdamOptimizer(model.parameters(), lr=1e-3))


@pytest.mark.parametrize("lang", LANGS)
def test_axiom_words_are_whole_in_their_own_vocabulary(lang):
    """One token per word — a split axiom protects subwords, not the word."""
    tok = _tok(lang)
    texts = [t for t, _ in axioms_for("phonetic", lang)] + \
        [t for t, _ in axioms_for("grammar", lang)]
    for text in texts:
        words = text.split()
        ids = [i for i in tok.encode(text) if not tok.decode([i]).isspace()]
        assert len(ids) == len(words), (
            f"{lang}: '{text}' encodes to "
            f"{'|'.join(tok.decode([i]) for i in ids)}")


@pytest.mark.parametrize("lang", LANGS)
def test_no_separator_row_is_ever_protected(lang):
    """The space carries a third of the corpus; it is not part of any axiom."""
    tok = _tok(lang)
    tr = _trainer(tok)
    for kind in ("phonetic", "grammar"):
        for text, prot in axioms_for(kind, lang):
            tr.add_axiom(text, is_objective=True, protection=prot)
    for tid in tr.axioms._protected_ids:
        piece = tok.decode([tid])
        assert piece and not piece.isspace(), f"{lang}: row {tid} is a separator"


def test_whitespace_only_axiom_is_not_registered():
    tok = _tok("it")
    tr = _trainer(tok)
    tr.add_axiom("   ", is_objective=True, protection=1.0)
    assert not tr.axioms.list_axioms()
    assert not tr.axioms._protected_ids


def test_protection_reaches_the_gradient_of_the_content_rows_only():
    """What the axiom claims on screen is what the backward pass applies."""
    tok = _tok("it")
    tr = _trainer(tok)
    tr.add_axiom("io sono", is_objective=True, protection=0.9)
    space = tok.encode(" ")[0]
    content = [i for i in tok.encode("io sono") if i != space]

    grad = torch.ones(tr.model.tok_emb.weight.shape[0], 4)
    tr.axioms.apply_to_grad(grad)
    for tid in content:
        assert grad[tid][0] == pytest.approx(0.1)
    assert grad[space][0] == pytest.approx(1.0)


def test_an_undeclared_language_protects_nothing():
    """The fallback used to be Italian, which is how an English run spent
    hours protecting m|am|ma. No axioms is the honest answer: phase 0 and
    phase 1 print it, and no embedding row is frozen for the wrong reason."""
    assert axioms_for("phonetic", "de") == []
    assert axioms_for("grammar", "de") == []
