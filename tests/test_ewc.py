"""
Tests for the online-EWC baseline (exp_i).

What is pinned here, in order of how expensive the bug would be:
  - the sign-flip interaction: a negative-feedback turn must still be
    ATTRACTED to the anchor (the penalty joins the loss AFTER
    scaled_loss = -loss, or every '-' turn actively repels the weights);
  - the tied lm_head/tok_emb weight is counted ONCE in the Fisher;
  - LayerNorm and pos_emb are excluded (the ln_f 0.87 -> 0.008 precedent);
  - dormant vocabulary rows carry zero Fisher, so new vocabulary is free;
  - the online update rule and the sidecar roundtrip;
  - build.sh's EWC knob fallbacks match ewc.DEFAULTS (drift test, same
    contract as test_dream_plateau's);
  - fisher.pt travels with dream_until_plateau's snapshot/restore.

Run with:  python3 -m pytest tests/test_ewc.py -v
"""
import os
import re
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physisml.torch_model import TorchGPT, TorchAdamOptimizer           # noqa: E402
from physisml.tokenizer import BPETokenizer                             # noqa: E402
from dynamic_model.exp_b.ewc import EWC, DEFAULTS, SIDECAR          # noqa: E402

TOK_PATH = os.path.join(ROOT, "dynamic_model", "data", "tokenizer_8k.json")


def tiny_model(extra_dormant: int = 0):
    """A real-tokenizer-sized model small enough for CPU tests."""
    tok = BPETokenizer()
    tok.load(TOK_PATH)
    n = max(tok.vocab.keys()) + 1
    model = TorchGPT(n + extra_dormant, 32, 2, 1, 64, 65, 0.0,
                     active_vocab_size=n)
    return model, tok


def anchored(model, lam=10.0, gamma=0.9):
    """An EWC consolidated at the model's CURRENT weights with F = 1."""
    ewc = EWC(lam=lam, gamma=gamma)
    ones = {name: torch.ones_like(p)
            for name, p in EWC.protected_parameters(model)}
    ewc.consolidate(model, ones)
    return ewc


# ---------------------------------------------------------------------------
# Penalty math
# ---------------------------------------------------------------------------

def test_penalty_is_zero_at_the_anchor():
    model, _ = tiny_model()
    ewc = anchored(model)
    assert float(ewc.penalty(model).detach()) == 0.0


def test_an_empty_or_disabled_ewc_is_a_detached_zero():
    """Callers never branch: no Fisher, or lam<=0, must both be a plain 0."""
    model, _ = tiny_model()
    assert float(EWC(lam=10.0).penalty(model)) == 0.0
    ewc = anchored(model, lam=0.0)
    assert float(ewc.penalty(model)) == 0.0


def test_penalty_gradient_points_back_to_the_anchor():
    model, _ = tiny_model()
    ewc = anchored(model, lam=10.0)
    with torch.no_grad():
        model.tok_emb.weight += 0.1
    pen = ewc.penalty(model)
    pen.backward()
    g = model.tok_emb.weight.grad
    # Analytic gradient of 0.5*lam*F*(theta-theta*)^2 with F=1, delta=0.1:
    # lam * delta everywhere on the displaced tensor.
    assert torch.allclose(g, torch.full_like(g, 10.0 * 0.1), atol=1e-5)
    # And the analytic norm helper agrees with autograd's.
    assert ewc.penalty_grad_norm(model) == pytest.approx(float(g.norm()),
                                                         rel=1e-4)


def test_tied_weight_is_counted_once():
    """named_parameters() yields the tied lm_head/tok_emb Parameter once;
    state_dict() would yield it twice and double-count the model's largest
    tensor. The closed-form penalty is the single-count one."""
    model, _ = tiny_model()
    names = [n for n, _ in EWC.protected_parameters(model)]
    assert "tok_emb.weight" in names
    assert "lm_head.weight" not in names
    assert len(names) == len(set(names))

    ewc = anchored(model, lam=2.0)
    with torch.no_grad():
        model.tok_emb.weight += 0.1
    expected = 0.5 * 2.0 * model.tok_emb.weight.numel() * 0.1 ** 2
    assert float(ewc.penalty(model).detach()) == pytest.approx(expected,
                                                               rel=1e-3)


def test_layernorm_and_pos_emb_are_excluded():
    """Constant quadratic pressure on LayerNorm gains is the measured ln_f
    slow death (TorchAdamOptimizer docstring); pos_emb rides along."""
    model, _ = tiny_model()
    names = [n for n, _ in EWC.protected_parameters(model)]
    assert "pos_emb.weight" not in names
    ln_owners = {n for n, m in model.named_modules()
                 if isinstance(m, torch.nn.LayerNorm)}
    assert ln_owners, "the tiny model must contain LayerNorms for this test"
    for name in names:
        assert name.rsplit(".", 1)[0] not in ln_owners, name


# ---------------------------------------------------------------------------
# Online update + Fisher estimation
# ---------------------------------------------------------------------------

def test_online_update_rule():
    model, _ = tiny_model()
    ewc = EWC(lam=1.0, gamma=0.5)
    f1 = {name: torch.full_like(p, 2.0)
          for name, p in EWC.protected_parameters(model)}
    ewc.consolidate(model, f1)
    theta1 = {k: v.clone() for k, v in ewc.theta_star.items()}

    with torch.no_grad():
        model.tok_emb.weight += 1.0
    f2 = {name: torch.full_like(p, 3.0)
          for name, p in EWC.protected_parameters(model)}
    ewc.consolidate(model, f2)

    for name in f2:
        # F2 = gamma * F1 + F_new = 0.5*2 + 3 = 4
        assert torch.allclose(ewc.fisher[name],
                              torch.full_like(f2[name], 4.0))
    # The anchor is REFRESHED, not kept.
    assert not torch.allclose(ewc.theta_star["tok_emb.weight"],
                              theta1["tok_emb.weight"])
    assert torch.allclose(ewc.theta_star["tok_emb.weight"],
                          model.tok_emb.weight)


def test_dormant_rows_carry_zero_fisher():
    """Rows activated AFTER a snapshot must be unconstrained: the forward
    masks dormant logits, so their gradient — and hence their Fisher — is
    exactly zero, and new vocabulary learns freely under the penalty."""
    model, tok = tiny_model(extra_dormant=50)
    pairs = [{"prompt": "di: ma", "response": "ma!"},
             {"prompt": "di: la", "response": "la!"}]
    fisher, n_used, n_skipped = EWC.estimate_fisher(model, tok, pairs)
    assert n_used == len(pairs) and n_skipped == 0
    dormant = fisher["tok_emb.weight"][model.active_vocab_size:]
    assert float(dormant.abs().sum()) == 0.0
    # ...and the estimate found SOMETHING on the active side.
    assert float(fisher["tok_emb.weight"].abs().sum()) > 0.0


def test_sidecar_roundtrip(tmp_path):
    model, _ = tiny_model()
    ewc = anchored(model, lam=7.0, gamma=0.8)
    path = str(tmp_path / SIDECAR)
    ewc.save(path, level=3, n_pairs=42, source_sha256="abc123")

    back = EWC.load(path, lam=7.0)
    assert back.gamma == pytest.approx(0.8)   # gamma travels with the file
    assert back.meta["level"] == 3
    assert back.meta["n_pairs"] == 42
    assert back.meta["source_sha256"] == "abc123"
    for name, f in ewc.fisher.items():
        assert torch.allclose(back.fisher[name], f)
        assert torch.allclose(back.theta_star[name], ewc.theta_star[name])
    # Same weights, same anchor: the reloaded penalty is still zero.
    assert float(back.penalty(model).detach()) == 0.0


# ---------------------------------------------------------------------------
# The sign-flip regression test — the bug most worth pinning
# ---------------------------------------------------------------------------

def test_negative_feedback_still_attracts_the_anchor():
    """TrainerB.step does gradient ASCENT on negative feedback
    (scaled_loss = -loss). The penalty joins the loss AFTER that flip; if it
    were added before, every '-' turn would repel the weights from the
    anchor. With a dominant lambda, one negative step must REDUCE the
    distance to theta*."""
    from dynamic_model.exp_b.affect_state import AffectState
    from dynamic_model.exp_b.modulator import AffectModulator
    from dynamic_model.exp_b.axioms import AxiomRegistry
    from dynamic_model.exp_b.trainer import TrainerB

    model, tok = tiny_model()
    ewc = anchored(model, lam=1e6)
    with torch.no_grad():
        model.tok_emb.weight += 0.05

    af = AffectState()
    tr = TrainerB(model, tok,
                  TorchAdamOptimizer(model.parameters(), lr=1e-3),
                  af, AffectModulator(af), AxiomRegistry(), ewc=ewc)

    anchor = ewc.theta_star["tok_emb.weight"]
    with torch.no_grad():
        before = float((tr.model.tok_emb.weight - anchor).norm())
    out = tr.step("di: ma", "ma!", feedback=-1.0)
    assert out["loss"] is not None, "the training step must actually run"
    with torch.no_grad():
        after = float((tr.model.tok_emb.weight - anchor).norm())
    assert after < before, (
        f"negative feedback moved the weights AWAY from the anchor "
        f"({before:.5f} -> {after:.5f}): the penalty is on the wrong side "
        f"of the sign flip")


# ---------------------------------------------------------------------------
# Arm gating + plumbing
# ---------------------------------------------------------------------------

def test_n1_scope_helper():
    from dynamic_model.train_curriculum import _n1_replay_levels
    assert _n1_replay_levels(3, "dream") == [0, 1, 2, 3]
    assert _n1_replay_levels(3, "ewc") == [3]
    assert _n1_replay_levels(3, "none") == [3]
    assert _n1_replay_levels(0, "dream") == [0]


def test_n3_keeps_the_current_level_in_the_ewc_arms():
    """The lambda sweep measured what dropping N3 entirely costs: ~45 points
    of current-level diagonal against the same-budget dream arm. The ewc/none
    arms must lose only the CROSS-level replay — current-level consolidation
    stays equal across arms, or the comparison confounds the two."""
    from dynamic_model.train_curriculum import _select_n3_entries
    bank = [
        {"level": 0, "weight": 1.0},   # old, salient
        {"level": 2, "weight": 0.8},   # current, salient
        {"level": 2, "weight": -0.8},  # current, salient negative
        {"level": 2, "weight": 0.1},   # current, below threshold
    ]
    dream = _select_n3_entries(bank, 2, "dream", 0.5, positive_only=False)
    assert len(dream) == 3            # every salient entry, all levels
    for arm in ("ewc", "none"):
        got = _select_n3_entries(bank, 2, arm, 0.5, positive_only=False)
        assert [e["level"] for e in got] == [2, 2]   # current level only
    # light mode's positive-only filter still applies on top
    light = _select_n3_entries(bank, 2, "ewc", 0.8, positive_only=True)
    assert len(light) == 1 and light[0]["weight"] == 0.8


def test_prior_fisher_walkback_skips_the_level_itself(tmp_path):
    """A retried level can hold its own fisher.pt from a previous attempt;
    anchoring a level to its own past self is not the experiment."""
    from dynamic_model.train_curriculum import _find_prior_fisher
    base = tmp_path / "ckpt"
    for lvl in (0, 2):
        d = base / f"level_{lvl}"
        d.mkdir(parents=True)
        (d / SIDECAR).write_bytes(b"x")
    # level 2 has its own sidecar: the walk-back must NOT return it...
    assert _find_prior_fisher(str(base), 2) == str(base / "level_0" / SIDECAR)
    # ...and level 3 gets the newest one below it.
    assert _find_prior_fisher(str(base), 3) == str(base / "level_2" / SIDECAR)
    assert _find_prior_fisher(str(base), 0) is None


def test_build_sh_ewc_knobs_match_module_defaults():
    """build.sh cannot import Python, so its EWC knob fallbacks are the one
    copy that can drift from ewc.DEFAULTS. Same contract as
    test_dream_plateau.test_build_sh_fallbacks_match_the_module_defaults."""
    src = open(os.path.join(ROOT, "build.sh"), encoding="utf-8").read()

    def knob(name, pattern=r"([0-9.]+)"):
        m = re.search(rf'{name}=\$\{{{name}:-{pattern}\}}', src)
        assert m, f"{name} non trovato in build.sh"
        return m.group(1)

    assert float(knob("EWC_LAMBDA")) == DEFAULTS["lambda"]
    assert float(knob("EWC_GAMMA")) == DEFAULTS["gamma"]
    assert knob("ANTI_FORGETTING", r"([a-z]+)") == "dream"


def test_state_files_include_fisher():
    """A dream rollback that restored final_dreamed.pt without its sidecar
    would leave the anchor pointing at weights no longer on disk."""
    import dream_until_plateau as dp
    assert SIDECAR in dp._STATE_FILES


def test_topup_dreams_carry_the_arm_flags():
    """dream_until_plateau spawns its own phase-2 children: without an
    explicit pass-through every top-up dream silently reverts to the 'dream'
    arm regardless of what the session dreams did — the most likely silent
    bug in the whole comparison. Pinned at the source, the same way
    test_dream_plateau pins the best-state restore."""
    import inspect
    import dream_until_plateau as dp
    src = inspect.getsource(dp.run_one_dream)
    assert '"--anti-forgetting", args.anti_forgetting' in src
