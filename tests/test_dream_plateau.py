"""
Tests for the dynamic dream count.

MIN_DREAMS=6 was the knee of one curve, measured at level 10. The plateau rule
replaces the constant with the measurement itself, so what is tested here is
the decision function against the curves that MOTIVATED the constant — the
measured L10 gains reproduce ~6 without anyone writing 6 anywhere — and the
failure modes a fixed count cannot see: a dream that damages retention, a
level that saturates early, one that is still climbing at the old knee.

Run with:  python3 -m pytest tests/test_dream_plateau.py -v
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dream_until_plateau import (decide, DREAM, STOP_PLATEAU,      # noqa: E402
                                 STOP_REGRESSION, STOP_MAX,
                                 snapshot_state, restore_state)

# The defaults the CLI ships with, spelled out so a change there is a change
# HERE too — these numbers are calibrated against measured curves, not taste.
KW = dict(epsilon=0.02, patience=2, max_dreams=12, max_drop=0.05)


def _walk(curve_points, **kw):
    """Feed the rule a growing prefix of the curve; return dreams it takes."""
    args = {**KW, **kw}
    curve = [curve_points[0]]
    for nxt in curve_points[1:]:
        action, _ = decide(curve, **args)
        if action != DREAM:
            return action, len(curve) - 1
        curve.append(nxt)
    return decide(curve, **args)[0], len(curve) - 1


def test_the_measured_l10_curve_stops_near_the_old_knee():
    """+3.6/dream to six, +1.0 after: the constant this rule replaces came
    from this exact shape, so the rule must land where the constant did —
    without 6 appearing anywhere in it."""
    curve = [0.20, 0.236, 0.272, 0.308, 0.344, 0.380, 0.416,   # +3.6 x 6
             0.426, 0.436, 0.446, 0.456, 0.466, 0.476]         # +1.0 after
    action, taken = _walk(curve)
    assert action == STOP_PLATEAU
    assert 7 <= taken <= 8      # 6 paying dreams + the `patience` that proves it


def test_an_early_saturating_level_stops_early():
    action, taken = _walk([0.30, 0.50, 0.60, 0.605, 0.607, 0.608, 0.609])
    assert action == STOP_PLATEAU
    assert taken == 4           # 2 paying + 2 of patience — not six by decree


def test_a_still_climbing_level_outruns_the_old_constant():
    curve = [0.20] + [0.20 + 0.04 * i for i in range(1, 13)]
    action, taken = _walk(curve)
    assert action == STOP_MAX
    assert taken == 12


def test_regression_stops_everything_floor_included():
    """A dream that damaged retention must not be followed by forced dreams:
    the floor exists against complacency, not against evidence of harm."""
    action, why = decide([0.60, 0.65, 0.55], floor=6, **KW)
    assert action == STOP_REGRESSION
    assert "sotto il migliore" in why


def test_one_flat_delta_is_noise_not_a_plateau():
    """Inter-run noise is ±2.2 points: a single sub-epsilon gain must not
    stop the loop, or half the runs stop one dream early on luck."""
    assert decide([0.60, 0.61], **KW)[0] == DREAM          # one flat delta
    assert decide([0.60, 0.61, 0.615], **KW)[0] == STOP_PLATEAU


def test_the_floor_forces_dreams_through_an_early_plateau():
    action, why = decide([0.60, 0.605, 0.607], floor=6, **KW)
    assert action == DREAM and "pavimento" in why


def test_session_dreams_count_toward_floor_and_cap_but_not_the_window():
    # 5 session dreams + 1 here = 6: floor of 6 satisfied, plateau visible...
    assert decide([0.60, 0.605], floor=6, already_done=5, **KW)[0] == DREAM
    # ...but only after `patience` dreams measured HERE: prior gains happened
    # off-camera and cannot be part of the window.
    assert decide([0.60, 0.605, 0.607], floor=6, already_done=5,
                  **KW)[0] == STOP_PLATEAU
    # and the cap counts them fully
    assert decide([0.60, 0.65], already_done=11, **KW)[0] == STOP_MAX


def test_the_baseline_alone_never_stops_as_plateau():
    """With zero dreams taken there are no gains to judge: the rule must
    dream, not conclude."""
    assert decide([0.001], **KW)[0] == DREAM
    assert decide([0.95], **KW)[0] == DREAM


def test_snapshot_and_restore_move_the_pair_and_the_memory(tmp_path):
    """A dream can grow the vocabulary and rewrites tokenizer.json: restoring
    the weights alone would pair an old model with a new tokenizer, which the
    loader does not tolerate in that direction. The memory travels too."""
    base = tmp_path / "ckpt"
    lvl = base / "level_11"
    lvl.mkdir(parents=True)
    (lvl / "final_dreamed.pt").write_bytes(b"weights-v1")
    (lvl / "tokenizer.json").write_text('{"v": 1}')
    (base / "affect_memory.json").write_text('{"words": 1}')

    best = lvl / "plateau_best"
    snapshot_state(str(lvl), str(best))

    (lvl / "final_dreamed.pt").write_bytes(b"weights-v2-damaged")
    (lvl / "tokenizer.json").write_text('{"v": 2}')
    (base / "affect_memory.json").write_text('{"words": 2}')

    names = restore_state(str(lvl), str(best))
    assert sorted(names) == ["affect_memory.json", "final_dreamed.pt",
                             "tokenizer.json"]
    assert (lvl / "final_dreamed.pt").read_bytes() == b"weights-v1"
    assert json.loads((lvl / "tokenizer.json").read_text()) == {"v": 1}
    assert json.loads((base / "affect_memory.json").read_text()) == {"words": 1}


# ── the parameters that disable the loop while looking like a clean stop ─────
def test_an_empty_patience_window_is_refused_not_vacuously_true():
    """all() over [] is True: patience=0 would stop every run on its baseline
    with zero dreams measured, and 'stop-plateau' in the log would read as a
    healthy early stop. Found by adversarial review, verified by execution."""
    import pytest as _pt
    with _pt.raises(ValueError):
        decide([0.60], **{**KW, "patience": 0})
    with _pt.raises(ValueError):
        decide([0.60], **{**KW, "patience": -1})


def test_a_negative_max_drop_is_refused_not_read_as_regression():
    """curve[-1] < best - (-0.01) is true on the baseline itself: a sign-flip
    typo would report a regression that never happened, restore the identical
    snapshot, and pollute dream_curve.json with stopped='stop-regression'."""
    import pytest as _pt
    with _pt.raises(ValueError):
        decide([0.60], **{**KW, "max_drop": -0.01})


def test_the_test_constants_are_the_module_defaults():
    """KW was a hand-copy of the CLI defaults; this pins them to the single
    source so a retune in one place fails HERE instead of drifting."""
    from dream_until_plateau import DEFAULTS
    assert KW["epsilon"] == DEFAULTS["epsilon"]
    assert KW["patience"] == DEFAULTS["patience"]
    assert KW["max_drop"] == DEFAULTS["max_drop"]
    assert KW["max_dreams"] == DEFAULTS["cap"]


def test_build_sh_fallbacks_match_the_module_defaults():
    """build.sh cannot import Python, so its knob fallbacks are the one copy
    that CAN drift. Parse them out of the script text and compare."""
    import re
    from dream_until_plateau import DEFAULTS
    src = open(os.path.join(ROOT, "build.sh"), encoding="utf-8").read()

    def knob(name):
        m = re.search(rf'{name}=\$\{{{name}:-([0-9.]+)\}}', src)
        assert m, f"{name} non trovato in build.sh"
        return float(m.group(1))

    assert knob("MAX_DREAMS") == DEFAULTS["cap"]
    assert knob("DREAM_EPSILON") == DEFAULTS["epsilon"]
    assert knob("DREAM_PATIENCE") == DEFAULTS["patience"]
    assert knob("DREAM_MAX_DROP") == DEFAULTS["max_drop"]
