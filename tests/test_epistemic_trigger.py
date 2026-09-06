"""The epistemic trigger: a reading of the weights that names its referent.

Toy models throughout, except the last class, which measures the reference
checkpoint when it is on this machine (models/ is not in the repository).
Class heads being single tokens is already asserted by
test_ontology_curiosity.py::test_the_class_words_are_single_tokens.
"""
import copy
import json
import os
import random
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                       # noqa: E402
from dynamic_model.exp_b import epistemic as ep          # noqa: E402
from physisml.tokenizer import BPETokenizer              # noqa: E402
from physisml.torch_model import TorchGPT, TorchAdamOptimizer  # noqa: E402

SEED_TOK = os.path.join(_ROOT, "dynamic_model", "data", "tokenizer_8k.json")
# The published reference (L12), as in test_eos_and_logs.py. Level 13 is the
# autonomy level and its checkpoints move with every batch: measured on
# level_13/plateau_best on 2026-09-03 the guardrail below already FAILS
# (AUC 0.92 on the frozen probes) — see scripts/epistemic_report.py.
REFERENCE_CKPT = os.path.join(_ROOT, "models", "active.pt")
REFERENCE_TOK = os.path.join(_ROOT, "models", "active_tokenizer.json")


def _eos_of(path):
    """The EOS id a tokenizer file registers, or None."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("special_tokens", {}).get("<|EOS|>")
    except (OSError, ValueError):
        return None


def _active_is_italian():
    """Whether models/active.pt is the Italian model these tests measure.

    There is one active.pt for every language: the first English build moved it
    to its own level_1 checkpoint mid-run, and the assertions below — the L12
    lexicon, the six nouns L12 taught — then ran against a model that had never
    seen a word of Italian and failed for the wrong reason. The EOS id tells the
    two vocabularies apart: 2589 on the Italian one, 2516 on the English.
    """
    return (_eos_of(REFERENCE_TOK) is not None
            and _eos_of(REFERENCE_TOK) == _eos_of(SEED_TOK))

LEX = etp.Lex(etp.load_lexicon("it"), "it")
CLASSES = ep.classes_of(LEX)
CANE = next(n for n in LEX.nouns if n["w"] == "cane")


def _tok():
    tok = BPETokenizer()
    tok.load(SEED_TOK)
    return tok


def _toy(tok, seed=0):
    torch.manual_seed(seed)
    n = max(tok.vocab.keys()) + 1
    return TorchGPT(n, 32, 2, 1, 64, 65, 0.0, active_vocab_size=n)


def _trainer(model, tok, lr=3e-3):
    from dynamic_model.exp_b.affect_state import AffectState
    from dynamic_model.exp_b.axioms import AxiomRegistry
    from dynamic_model.exp_b.modulator import AffectModulator
    from dynamic_model.exp_b.trainer import TrainerB
    af = AffectState()
    return TrainerB(model, tok, TorchAdamOptimizer(model.parameters(), lr=lr),
                    af, AffectModulator(af), AxiomRegistry()), af


TAUGHT = ["cane", "gatto", "casa", "pane", "sole", "mamma"]


@pytest.fixture(scope="module")
def taught():
    """A toy that has seen six definitions forty times, and nothing else."""
    tok = _tok()
    model = _toy(tok)
    tr, _ = _trainer(model, tok)
    nouns = [n for n in LEX.nouns if n["w"] in TAUGHT]
    assert len(nouns) == len(TAUGHT)
    for _ in range(40):
        for n in nouns:
            tr.step(f"cos è {etp.indef(n)} {n['w']}?",
                    f"{etp.phrase(n)} è {LEX.cls_of(n)}.", feedback=1.0)
    return model, tok, nouns


class TestTheReading:

    def test_the_posterior_is_a_distribution_over_the_ten_classes(self):
        tok = _tok()
        post = ep.class_posterior(_toy(tok), tok, CANE, CLASSES)
        assert len(CLASSES) == 10
        assert post.shape == (10,)
        assert abs(float(post.sum()) - 1.0) < 1e-5
        assert float(post.min()) >= 0.0

    def test_the_prefix_is_the_loop_question_plus_the_gold_opening(self):
        assert ep.prefix_for(CANE) == "cos è un cane? il cane è "
        acqua = next(n for n in LEX.nouns if n["w"] == "acqua")
        # A mass noun takes no indefinite article, as in etp.ask_forms.
        assert ep.prefix_for(acqua).startswith("cos è l acqua?")
        assert ep.prefix_for(acqua).endswith(" è ")

    def test_the_verdict_names_the_referent_by_construction(self):
        tok = _tok()
        v = ep.verdict(_toy(tok), tok, CANE, CLASSES, threshold=0.5)
        assert v.referent == "cane"
        assert v.top_class in CLASSES
        assert 0.0 <= v.margin <= 1.0 and 0.0 <= v.entropy <= 1.0
        assert v.ignorant is (v.margin < 0.5)
        assert set(v.posterior) == set(CLASSES)
        assert set(v.as_log()) == {"margin", "p_top", "top_class", "tau"}

    def test_the_reading_leaves_the_affect_state_and_the_mode_alone(self):
        """Same contract as AffectState.peek_entropy: generate() would update
        the state at every token, so the trigger must not go through it."""
        tok = _tok()
        model = _toy(tok)
        tr, af = _trainer(model, tok)
        tr.step("ma", "ma", feedback=0.5)             # a non-trivial state
        before = copy.deepcopy(vars(af))
        model.train()
        ep.verdict(model, tok, CANE, CLASSES, threshold=0.3)
        assert vars(af) == before, "the reading moved the affect state"
        assert model.training, "the reading did not restore train mode"
        model.eval()
        ep.class_posterior(model, tok, CANE, CLASSES)
        assert not model.training

    def test_a_reading_is_deterministic(self):
        tok = _tok()
        model = _toy(tok)
        model.train()                                 # dropout_p is 0.0, but the flag is set
        a = ep.class_posterior(model, tok, CANE, CLASSES)
        b = ep.class_posterior(model, tok, CANE, CLASSES)
        assert torch.allclose(a, b)


class TestTheMarginFollowsTheWeights:

    def test_teaching_a_class_raises_its_margin_and_names_it(self, taught):
        model, tok, nouns = taught
        fresh = _toy(tok)
        for n in nouns:
            before = ep.verdict(fresh, tok, n, CLASSES, threshold=0.0)
            after = ep.verdict(model, tok, n, CLASSES, threshold=0.0)
            assert after.margin > before.margin, n["w"]
            assert after.top_class == LEX.cls_of(n), (n["w"], after.posterior)

    def test_the_margin_separates_taught_from_pseudo(self, taught):
        model, tok, nouns = taught
        pseudo = ep.pseudo_words(random.Random(1), 12)
        known = ep.margins(model, tok, nouns, CLASSES)
        unknown = ep.margins(model, tok, pseudo, CLASSES)
        assert ep.auc(known, unknown) >= 0.9, (known, unknown)

    def test_the_threshold_is_recomputed_from_the_weights(self, taught):
        model, tok, _ = taught
        fresh = _toy(tok)
        cal_fresh = ep.calibration(fresh, tok, LEX, n_pseudo=8, seed=3)
        cal_taught = ep.calibration(model, tok, LEX, n_pseudo=8, seed=3)
        assert cal_fresh.tau != cal_taught.tau
        for cal in (cal_fresh, cal_taught):
            lo, hi = sorted((cal.p95_pseudo, cal.p05_known))
            assert lo <= cal.tau <= hi
            assert cal.n_known == len(LEX.classified()) and cal.n_pseudo == 8
        assert ep.calibrate(model, tok, LEX, n_pseudo=8, seed=3) == cal_taught.tau


class TestPseudoWords:

    def test_same_seed_same_words_different_seed_fresh_ones(self):
        a = ep.pseudo_words(random.Random(7), 10)
        b = ep.pseudo_words(random.Random(7), 10)
        c = ep.pseudo_words(random.Random(8), 10)
        assert [w["w"] for w in a] == [w["w"] for w in b]
        assert {w["w"] for w in a} != {w["w"] for w in c}

    def test_none_of_them_is_a_word_the_curriculum_could_have_taught(self):
        words = {w["w"] for w in ep.pseudo_words(random.Random(11), 40)}
        taught = {n["w"] for n in LEX.nouns + LEX.unknown + LEX.bare_unknown
                  + LEX.bare_probe}
        assert len(words) == 40
        assert not words & taught
        assert not words & set(ep.l0_inventory("it"))

    def test_they_render_well_formed_prompts(self):
        for w in ep.pseudo_words(random.Random(2), 10):
            assert w["g"] in ("m", "f") and w["art"] in ("il", "lo", "la", "l")
            assert w["pseudo"] is True
            p = ep.prefix_for(w)
            assert p.startswith("cos è un") and p.endswith(f" {w['w']} è ")

    def test_the_inventory_is_syllables_only(self):
        inv = ep.l0_inventory("it")
        assert {"ma", "pa", "ba"} <= set(inv)
        assert all(ep._SYLLABLE.match(s) for s in inv)
        assert "cane" not in inv


class TestStatistics:

    def test_auc_is_rank_based(self):
        assert ep.auc([1, 2, 3], [0, 0, 0]) == 1.0
        assert ep.auc([0, 0], [1, 1]) == 0.0
        assert ep.auc([1], [1]) == 0.5
        assert ep.auc([0.9, 0.1], [0.5, 0.5]) == 0.5

    def test_percentile_interpolates(self):
        assert ep.percentile([1, 2, 3, 4, 5], 50) == 3
        assert ep.percentile([1, 2, 3, 4, 5], 5) == pytest.approx(1.2)
        assert ep.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


needs_reference = pytest.mark.skipif(
    not (os.path.exists(REFERENCE_CKPT) and os.path.exists(REFERENCE_TOK)),
    reason="reference checkpoint not on this machine")

is_italian = pytest.mark.skipif(
    os.path.exists(REFERENCE_TOK) and not _active_is_italian(),
    reason="models/active.pt is not the Italian model (another language was "
           "built last): ./set_model.sh models/checkpoints/it/level_12/"
           "final_dreamed.pt to measure it again")


@needs_reference
@is_italian
class TestOnTheReferenceCheckpoint:
    """The §9 guardrail of curiosita_meccanismo.md, and the §2 measurement
    replicated through this module: the margin separates the 53 classified
    nouns from the never-taught probes (AUC ≥ 0.95), and the mechanism agrees
    with the symbolic record on the nouns L12 taught."""

    @pytest.fixture(scope="class")
    def ref(self):
        from measure_repetition import load_pair
        tr, tok = load_pair(REFERENCE_CKPT, REFERENCE_TOK)
        return tr.model, tok

    def test_known_nouns_outrank_the_frozen_probes(self, ref):
        model, tok = ref
        # The 6 held-out unknown_nouns and the 7 bare names with role=probe:
        # taught nowhere, by construction of the lexicon.
        probes = LEX.unknown_of(probe=True) + LEX.bare_probe
        assert len(probes) == 13
        known = ep.margins(model, tok, LEX.classified(), CLASSES)
        never = ep.margins(model, tok, probes, CLASSES)
        assert ep.auc(known, never) >= 0.95, (sorted(known)[:5], sorted(never)[-5:])

    def test_the_calibrated_threshold_separates_the_bands(self, ref):
        model, tok = ref
        cal = ep.calibration(model, tok, LEX, n_pseudo=32, seed=0)
        assert cal.auc >= 0.95
        assert 0.0 < cal.tau < 1.0

    def test_mechanism_and_record_agree_on_what_l12_taught(self, ref):
        model, tok = ref
        cal = ep.calibration(model, tok, LEX, n_pseudo=32, seed=0)
        taught = LEX.unknown_of(probe=False)      # L12 taught these six
        for n in taught:
            v = ep.verdict(model, tok, n, CLASSES, cal.tau)
            assert not v.ignorant, (n["w"], v.margin, cal.tau)
            assert v.top_class == n["cls"], (n["w"], v.posterior)
