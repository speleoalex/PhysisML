"""The `<|NONE|>` row's two contracts: scorable, trainable, never emitted.

The frozen-row experiment (docs_internal/curiosita_meccanismo.md §13bis)
showed the whole cost of the eleventh class is the row winning the argmax
while the model writes: masking it at decoding time restored the checkpoint's
retention exactly, and no training was needed. These tests hold that wiring
in place, on toy models, without the reference checkpoint:

  * the row is live by default — otherwise `_sync_vocab_rows` would
    re-initialise the trained row with a random direction;
  * every decoding path masks it, so the token cannot reach the output;
  * the row still receives gradient, or the decay of §13 is dead;
  * a ten-class read is unchanged by the row's existence, and an
    eleven-class read switches the separation statistic to 1 - P(NONE).
"""
import json
import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                              # noqa: E402
from dynamic_model.exp_b import epistemic as ep                 # noqa: E402
from dynamic_model.exp_b.none_token import (                    # noqa: E402
    NONE_TOKEN, mask_scoring_rows, row_id, scoring_cut, scoring_only_ids)
from dynamic_model.train_curriculum import _sync_vocab_rows     # noqa: E402
from physisml.tokenizer import BPETokenizer                     # noqa: E402
from physisml.torch_model import TorchGPT, TorchAdamOptimizer   # noqa: E402

SEED_TOK = os.path.join(_ROOT, "dynamic_model", "data", "tokenizer_8k.json")
LEX = etp.Lex(etp.load_lexicon("it"), "it")
TEN = ep.classes_of(LEX)
CANE = next(n for n in LEX.nouns if n["w"] == "cane")


def _tok():
    tok = BPETokenizer()
    tok.load(SEED_TOK)
    return tok


def _toy(tok, seed=0):
    torch.manual_seed(seed)
    n = max(tok.vocab.keys()) + 1
    return TorchGPT(n + 8, 32, 2, 1, 64, 65, 0.0, active_vocab_size=n)


def _trainer(model, tok, lr=3e-3):
    from dynamic_model.exp_b.affect_state import AffectState
    from dynamic_model.exp_b.axioms import AxiomRegistry
    from dynamic_model.exp_b.modulator import AffectModulator
    from dynamic_model.exp_b.trainer import TrainerB
    af = AffectState()
    return TrainerB(model, tok, TorchAdamOptimizer(model.parameters(), lr=lr),
                    af, AffectModulator(af), AxiomRegistry())


@pytest.fixture()
def minted():
    """A toy with `<|NONE|>` minted the way scripts_internal/eleventh_class.py
    mints it: the next free id, then the row activated."""
    tok = _tok()
    model = _toy(tok)
    none_id = max(tok.vocab.keys()) + 1
    tok.register_special_token(NONE_TOKEN, none_id)
    _sync_vocab_rows(model, tok, label="test eleventh class")
    assert tok.encode(NONE_TOKEN) == [none_id]
    return model, tok, none_id


class TestTheRowStaysLive:

    def test_the_row_is_active_by_default(self, minted):
        """Dormant-by-default is the unsafe design: _sync_vocab_rows treats
        every row at or above active_vocab_size as fresh and overwrites it."""
        model, _tok_, none_id = minted
        assert model.active_vocab_size == none_id + 1

    def test_a_vocab_sync_does_not_destroy_a_trained_row(self, minted):
        model, tok, none_id = minted
        with torch.no_grad():
            model.tok_emb.weight[none_id].fill_(0.25)
        _sync_vocab_rows(model, tok, label="second sync")
        assert torch.allclose(model.tok_emb.weight[none_id],
                              torch.full((model.d_model,), 0.25))

    def test_the_row_still_gets_gradient(self, minted):
        """The decay of §13 is gold training pulling P(NONE) down. A mask
        inside forward would zero this gradient and kill it."""
        model, tok, none_id = minted
        tr = _trainer(model, tok)
        before = model.tok_emb.weight[none_id].detach().clone()
        for _ in range(3):
            tr.step(f"cos è {etp.indef(CANE)} cane?", f"{NONE_TOKEN}.", feedback=1.0)
        assert not torch.allclose(before, model.tok_emb.weight[none_id])


class TestTheRowIsNeverEmitted:

    def test_the_context_manager_masks_and_restores(self, minted):
        model, tok, none_id = minted
        assert scoring_only_ids(tok) == {none_id}
        with mask_scoring_rows(model, tok) as leftover:
            assert leftover == ()          # contiguous at the top: the integer suffices
            assert model.active_vocab_size == none_id
            logits = model.forward(torch.tensor(tok.encode("il cane"), dtype=torch.long))
            assert logits[-1, none_id] == float("-inf")
        assert model.active_vocab_size == none_id + 1

    def test_the_integer_is_restored_after_an_exception(self, minted):
        model, tok, none_id = minted
        with pytest.raises(RuntimeError):
            with mask_scoring_rows(model, tok):
                raise RuntimeError("boom")
        assert model.active_vocab_size == none_id + 1

    def test_a_trained_row_above_is_handed_back_instead(self, minted):
        """The integer cannot express 'mask this row but not the one above',
        so those ids come back to the decoding loop."""
        model, tok, none_id = minted
        model.active_vocab_size = none_id + 2      # a row above the special one
        with mask_scoring_rows(model, tok) as leftover:
            assert leftover == (none_id,)
            assert model.active_vocab_size == none_id + 2

    def test_generate_cannot_emit_the_token(self, minted):
        """The row is pushed to a norm that would dominate every argmax."""
        model, tok, none_id = minted
        with torch.no_grad():
            model.tok_emb.weight[none_id] = 40.0 * model.tok_emb.weight[none_id].sign()
        tr = _trainer(model, tok)
        out = tr.generate("cos è un cane?", max_tokens=12, base_temperature=0.0,
                          top_k=1, min_tokens=4)
        assert NONE_TOKEN not in out
        # And the model is scorable again the moment generation is over.
        assert model.active_vocab_size == none_id + 1

    def test_the_dominant_row_would_be_emitted_without_the_mask(self, minted):
        """The counter-test: without the mask the same row takes the argmax,
        which is exactly the retention cost §13bis measured."""
        model, tok, none_id = minted
        with torch.no_grad():
            model.tok_emb.weight[none_id] = 40.0 * model.tok_emb.weight[none_id].sign()
        ctx = torch.tensor(tok.encode("cos è un cane?"), dtype=torch.long)
        assert int(model.forward(ctx)[-1].argmax()) == none_id


class TestTheReadingSwitchesMode:

    def test_scoring_classes_follows_the_tokenizer(self, minted):
        _model, tok, _id = minted
        assert ep.scoring_classes(LEX, tok) == TEN + [NONE_TOKEN]
        assert ep.scoring_classes(LEX, _tok()) == TEN
        assert row_id(_tok()) is None

    def test_a_ten_class_read_ignores_the_row(self, minted):
        """The row enters the per-position log_softmax, so a ten-class read
        on a model that owns it must mask it — otherwise the margin degrades
        at frozen weights and the degradation is mistaken for forgetting."""
        model, tok, none_id = minted
        with torch.no_grad():
            model.tok_emb.weight[none_id] = 40.0 * model.tok_emb.weight[none_id].sign()
        masked = ep.class_posterior(model, tok, CANE, TEN)
        model.active_vocab_size = none_id       # already dormant: same read
        dormant = ep.class_posterior(model, tok, CANE, TEN)
        assert torch.allclose(masked, dormant, atol=1e-6)

    def test_eleven_classes_read_one_minus_p_none(self, minted):
        model, tok, none_id = minted
        eleven = TEN + [NONE_TOKEN]
        with torch.no_grad():
            model.tok_emb.weight[none_id] = 40.0 * model.tok_emb.weight[none_id].sign()
        v = ep.verdict(model, tok, CANE, eleven, threshold=0.5)
        assert v.p_none is not None and v.p_none > 0.9
        assert v.top_class == NONE_TOKEN
        assert abs(v.margin - (1.0 - v.p_none)) < 1e-6
        assert v.ignorant                      # the margin rule would say the opposite
        assert v.as_log()["p_none"] == round(v.p_none, 4)

    def test_ten_classes_still_read_the_margin(self):
        tok = _tok()
        v = ep.verdict(_toy(tok), tok, CANE, TEN, threshold=0.5)
        post = torch.tensor([v.posterior[c] for c in TEN])
        top2 = torch.topk(post, 2).values
        assert v.p_none is None
        assert abs(v.margin - float(top2[0] - top2[1])) < 1e-3
        assert "p_none" not in v.as_log()


class TestTheExportLeavesTheRowOut:
    """A GGUF has no active_vocab_size and llama.cpp samples every row it is
    given; the HF folder is loaded through config.json's integer. Either way
    the exported artefact must show a sampler what TrainerB.generate shows it:
    no `<|NONE|>` row to win the argmax."""

    def test_scoring_cut_is_the_integer_the_mask_uses(self):
        assert scoring_cut(11, {10}) == 10           # contiguous at the top
        assert scoring_cut(12, {10, 11}) == 10
        assert scoring_cut(10, {10}) == 10           # already dormant: unchanged
        assert scoring_cut(11, set()) == 11          # nothing to hide
        assert scoring_cut(12, {10}) is None         # a trained row above it

    @staticmethod
    def _saved(minted, tmp_path):
        model, tok, none_id = minted
        with torch.no_grad():
            model.tok_emb.weight[none_id] = 40.0 * model.tok_emb.weight[none_id].sign()
        ckpt = str(tmp_path / "model.pt")
        model.save(ckpt)
        tok.save(str(tmp_path / "tokenizer.json"))
        return ckpt, none_id

    def test_the_hf_config_ships_the_masked_integer(self, minted, tmp_path):
        import export_hf
        ckpt, none_id = self._saved(minted, tmp_path)
        out = str(tmp_path / "hf")
        written = {}

        def fake_save_file(tensors, path, metadata=None):
            written.update(tensors)
            open(path, "wb").close()

        export_hf.export_one(ckpt, out, fake_save_file, out)
        with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        # The weights ship in full; the row is dormant through the integer.
        assert written["tok_emb.weight"].shape[0] == cfg["vocab_size"]
        assert cfg["active_vocab_size"] == none_id
        assert cfg["scoring_only_tokens"] == {NONE_TOKEN: none_id}
        # And a checkpoint with no such row exports exactly as before.
        plain = str(tmp_path / "plain.pt")
        _toy(_tok()).save(plain)
        _tok().save(str(tmp_path / "tokenizer.json"))
        cfg2 = None
        export_hf.export_one(plain, str(tmp_path / "hf2"), fake_save_file,
                             str(tmp_path / "hf2"))
        with open(os.path.join(str(tmp_path / "hf2"), "config.json"),
                  encoding="utf-8") as f:
            cfg2 = json.load(f)
        assert cfg2["active_vocab_size"] == none_id      # the toy's own active
        assert cfg2["scoring_only_tokens"] == {}

    def test_the_gguf_has_no_such_row(self, minted, tmp_path, monkeypatch, capsys):
        llama = os.path.expanduser("~/git/llama.cpp/gguf-py")
        if os.path.isdir(llama) and llama not in sys.path:
            sys.path.insert(0, llama)
        gguf = pytest.importorskip("gguf")
        import export_gguf
        ckpt, none_id = self._saved(minted, tmp_path)
        out = str(tmp_path / "model.gguf")
        monkeypatch.setattr(sys, "argv", [
            "export_gguf.py", "--checkpoint", ckpt,
            "--tokenizer", str(tmp_path / "tokenizer.json"), "--output", out])
        export_gguf.main()
        assert f"scoring-only rows left out of the file: [{none_id}]" in capsys.readouterr().out

        reader = gguf.GGUFReader(out)
        field = reader.fields["tokenizer.ggml.tokens"]
        tokens = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
        assert len(tokens) == none_id
        assert NONE_TOKEN not in tokens
        embd = next(t for t in reader.tensors if t.name == "token_embd.weight")
        model = minted[0]
        assert embd.n_elements == none_id * model.d_model
