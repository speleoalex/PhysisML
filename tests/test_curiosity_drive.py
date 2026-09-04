"""The persistent drive of docs_internal/curiosita_meccanismo.md §6.

Charge and discharge, persistence across processes, a deterministic order
over the queue, and the two things it must never do: fire on a known noun,
or hand back a noun the queue does not hold.
"""
import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dynamic_model.exp_b.curiosity import CuriosityDrive, drive_path, DECLINED  # noqa: E402

Q = [{"w": w} for w in ("lontra", "ravanello", "spillo", "grembiule")]


def test_charge_accumulates_from_the_margin_and_saturates():
    d = CuriosityDrive()
    assert d.charge("spillo", 0.6) == 0.6
    assert d.charge("spillo", 0.6) == 1.0, "arousal is capped at 1"
    assert d.encounters("spillo") == 2
    assert d.unresolved() == [("spillo", 1.0)]


def test_a_known_noun_is_met_and_counted_but_leaves_no_debt():
    """tau - margin is negative when the weights hold a class: the drive
    must not fire on a noun the model knows, however often it is met."""
    d = CuriosityDrive()
    for _ in range(5):
        assert d.charge("cane", -0.3) == 0.0
    assert d.encounters("cane") == 5
    assert d.unresolved() == [] and d.pressure() == 0.0


def test_discharge_settles_the_debt_and_charging_again_does_not_reopen_it():
    d = CuriosityDrive()
    d.charge("lontra", 0.5)
    d.discharge("lontra")
    assert d.arousal("lontra") == 0.0 and d.pressure() == 0.0
    assert d.charge("lontra", 0.5) == 0.0, "an acquired noun does not nag again"
    assert d.debt["lontra"]["peak"] == 0.5
    d.charge("colore", 0.7)
    d.discharge("colore", how=DECLINED)
    assert d.debt["colore"]["resolved_how"] == DECLINED
    assert d.unresolved() == []


def test_pressure_is_open_debt_over_the_cap_clipped_to_one():
    d = CuriosityDrive(cap=2.0)
    d.charge("a", 0.5)
    d.charge("b", 0.5)
    assert d.pressure() == pytest.approx(0.5)
    d.charge("c", 1.0)
    d.charge("d", 1.0)
    assert d.pressure() == 1.0


def test_the_ledger_survives_a_process_boundary(tmp_path):
    path = drive_path(str(tmp_path / "affect_memory.json"))
    d = CuriosityDrive(path)
    d.charge("spillo", 0.63)
    d.charge("lontra", 0.2)
    d.discharge("lontra")
    d.save()
    assert os.path.basename(path) == "curiosity_drive.json"

    code = (
        "import sys, json; sys.path.insert(0, %r);"
        "from dynamic_model.exp_b.curiosity import CuriosityDrive;"
        "d = CuriosityDrive.load(%r);"
        "print(json.dumps([d.arousal('spillo'), d.arousal('lontra'),"
        " d.encounters('spillo'), d.last, d.debt['lontra']['resolved']]))"
    ) % (_ROOT, path)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout
    assert json.loads(out) == [0.63, 0.0, 1, "lontra", True]


def test_the_order_is_novelty_then_debt_then_fairness_and_deterministic():
    d = CuriosityDrive()
    # Round one: nothing met yet, the drive IS the round-robin.
    picks = []
    for delta in (0.3, 0.6, 0.1, 0.0):          # grembiule: known
        n = d.next_target(Q)
        picks.append(n["w"])
        d.charge(n["w"], delta)
    assert picks == ["lontra", "ravanello", "spillo", "grembiule"]
    # Round two: the highest open debt first, and never the same twice in a
    # row: ravanello (0.6) -> lontra (0.3) -> ravanello (1.0) -> ...
    assert d.next_target(Q)["w"] == "ravanello"
    d.charge("ravanello", 0.6)                   # saturates at 1.0
    assert d.next_target(Q)["w"] == "lontra", "not twice in a row"
    d.charge("lontra", 0.3)
    assert d.next_target(Q)["w"] == "ravanello"
    # A settled debt ranks with the known: last, by encounters.
    d.discharge("ravanello")
    d.last = None
    assert d.next_target(Q)["w"] == "lontra"
    # Same ledger, same answer: no randomness anywhere in the ordering.
    e = CuriosityDrive()
    e.debt = json.loads(json.dumps(d.debt))
    e.last = d.last
    assert [e.next_target(Q)["w"] for _ in range(3)] == \
        [d.next_target(Q)["w"] for _ in range(3)]


def test_the_drive_cannot_reintroduce_what_the_queue_filtered_out():
    """build_queue drops the reserve and the frozen probes; a debt carried in
    the ledger from another run must not bring one of them back."""
    d = CuriosityDrive()
    d.charge("tegola", 1.0)                      # a probe name, huge debt
    for _ in range(6):
        n = d.next_target(Q)
        assert n in Q
        d.charge(n["w"], 0.2)
    with pytest.raises(ValueError):
        d.next_target([])


def test_a_single_noun_queue_is_asked_again():
    d = CuriosityDrive()
    one = [{"w": "spillo"}]
    d.charge("spillo", 0.5)
    assert d.next_target(one)["w"] == "spillo"
