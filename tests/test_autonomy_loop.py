"""
Tests for the autonomous acquisition loop.

The loop writes into the curriculum without a human in the room, so what is
tested here is mostly what it must REFUSE to do. Each case below stands for a
failure this repository has already paid for once:

  * two golds for one prompt — the harvest reintroduced it twice;
  * training material leaking into the probe — a degradation trigger measured
    on what was just taught confirms itself;
  * partial acquisition — teaching 'cosa è un falco?' the class while
    'cos è un falco?' still answers 'non lo so.' makes the SHAPE of the
    question carry the answer, the exact failure ask_forms exists to prevent;
  * an LLM's free text reaching the gold;
  * crediting the model with ignorance of 'tegola' when it asked about 'zucca'.

Run with:  python3 -m pytest tests/test_autonomy_loop.py -v
"""
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                                    # noqa: E402
import probe_set                                                       # noqa: E402
import autonomy_loop as al                                             # noqa: E402
from dynamic_model.ontology_oracle import (OntologyOracle, article_for,  # noqa: E402
                                            guess_gender, normalize_class)
from dynamic_model import retraction                                   # noqa: E402

LEX = etp.Lex(etp.load_lexicon("it"), "it")
CLASSES = sorted(set(LEX.classes.keys()) | {v for v in LEX.classes.values() if v})


# ── the oracle: only its choice may get through, never its words ─────────────
def test_the_class_set_is_closed():
    assert normalize_class("un animale", CLASSES) == "un animale"
    assert normalize_class("  Un Animale. ", CLASSES) == "un animale"
    assert normalize_class("animale", CLASSES) == "un animale"   # bare head
    assert normalize_class("è un animale", CLASSES) == "un animale"
    # Anything the curriculum does not already say is refused, not repaired
    assert normalize_class("un mammifero", CLASSES) is None
    assert normalize_class("a tool", CLASSES) is None
    assert normalize_class("", CLASSES) is None


def test_the_article_is_derived_by_code_not_reported_by_the_llm():
    assert article_for("sgabello", "m") == "lo"      # s + consonant
    assert article_for("scoiattolo", "m") == "lo"
    assert article_for("zaino", "m") == "lo"
    assert article_for("cane", "m") == "il"
    assert article_for("casa", "f") == "la"
    assert article_for("aquila", "f") == "l"         # apostrophes are spaces
    assert article_for("orso", "m") == "l"


def test_the_indefinite_article_agrees_too():
    """'un sgabello' put four malformed prompts into L12 step E."""
    assert etp.indef({"w": "sgabello", "g": "m"}) == "uno"
    assert etp.indef({"w": "scoiattolo", "g": "m"}) == "uno"
    assert etp.indef({"w": "falco", "g": "m"}) == "un"
    assert etp.indef({"w": "zucca", "g": "f"}) == "una"
    assert etp.indef({"w": "aquila", "g": "f"}) == "un"


def test_the_gender_is_guessed_only_where_the_ending_settles_it():
    assert guess_gender("cetriolo") == "m"
    assert guess_gender("sciarpa") == "f"
    assert guess_gender("pettine") is None      # 'il pettine' vs 'la noce'


def _oracle():
    """An oracle with no network: only the rendering half is under test."""
    o = OntologyOracle.__new__(OntologyOracle)
    o.lang, o.lex, o.timeout = "it", LEX, 1
    o.classes, o.backend, o.model = CLASSES, None, None
    return o


def test_the_material_is_rendered_from_the_curriculum_templates():
    o = _oracle()
    noun = {"w": "falco", "art": "il", "g": "m"}
    mat = o.material_for(noun, "un animale")
    prompts = [t["prompt"] for t in mat]
    # Every phrasing ask_forms produces, and both yes/no directions
    for p in etp.ask_forms(noun, "animale"):
        assert p in prompts
    assert "il falco è un animale?" in prompts
    assert any(t["expected"].startswith("sì,") for t in mat)
    assert any(t["expected"].startswith("no,") for t in mat)


def test_no_target_can_fail_the_plus_plus_plus_check():
    """The same invariant validate_teacher_configs.py enforces: the grader's
    keyword must be a word of the ANSWER, or the step trains against itself."""
    o = _oracle()
    for w, art, g, cls in [("falco", "il", "m", "un animale"),
                           ("sciarpa", "la", "f", "un oggetto"),
                           ("aquila", "l", "f", "un animale")]:
        for t in o.material_for({"w": w, "art": art, "g": g}, cls):
            assert t["noun"] in t["expected"], t
            assert t["expected"].rstrip()[-1] in ".!?"
            assert len(t["expected"].split()) <= 8


# ── the gatekeeper: the material that must not get in ────────────────────────
def _gate(known=None, probe_prompts=(), max_new=40):
    g = al.Gatekeeper.__new__(al.Gatekeeper)
    g.known    = dict(known or {})
    g.probe_p  = {al._normalize_prompt(p) for p in probe_prompts}
    g.max_new  = max_new
    g.accepted = 0
    g.refusals = {}
    return g


def test_a_prompt_with_a_different_gold_blocks_the_whole_acquisition():
    mat = [{"prompt": "cos è il falco?", "expected": "il falco è un animale."},
           {"prompt": "cos è un falco?", "expected": "il falco è un animale."}]
    g = _gate({al._normalize_prompt("cos è un falco?"): "il falco è un cibo."})
    v = g.inspect(mat)
    assert not v["ok"] and v["conflict"]
    assert "gold" in g.reason(v)


def test_a_stale_admission_is_reported_apart_from_a_real_conflict():
    """'non lo so.' is supervision that learning makes false; a fact is not.
    The loop must not confuse the two, because only one of them could ever be
    retracted."""
    mat = [{"prompt": "cos è un falco?", "expected": "il falco è un animale."}]
    g = _gate({al._normalize_prompt("cos è un falco?"): "non lo so."})
    v = g.inspect(mat)
    assert not v["ok"] and v["stale"] and not v["conflict"]
    assert "non lo so" in g.reason(v)


def test_acquisition_is_atomic():
    """Half an acquisition is worse than none: it teaches the class under one
    phrasing and 'non lo so.' under another."""
    mat = [{"prompt": "cos è il falco?", "expected": "il falco è un animale."},
           {"prompt": "cos è un falco?", "expected": "il falco è un animale."}]
    g = _gate({al._normalize_prompt("cos è un falco?"): "non lo so."})
    assert g.inspect(mat)["ok"] is False
    assert g.commit([]) == []          # nothing was committed
    assert g.accepted == 0


def test_the_frozen_probe_is_off_limits():
    mat = [{"prompt": "cos è il falco?", "expected": "il falco è un animale."}]
    g = _gate(probe_prompts=["cos è il falco?"])
    v = g.inspect(mat)
    assert not v["ok"] and v["in_probe"] == 1


def test_a_duplicate_is_not_a_conflict_and_is_not_counted_twice():
    mat = [{"prompt": "cos è il falco?", "expected": "il falco è un animale."}]
    g = _gate({al._normalize_prompt("cos è il falco?"): "il falco è un animale."})
    v = g.inspect(mat)
    assert v["ok"] and v["duplicate"] == 1
    assert g.commit(mat) == []          # already in the curriculum
    assert g.accepted == 0


def test_the_batch_cap_holds():
    mat = [{"prompt": f"cos è il n{i}?", "expected": f"il n{i} è un oggetto."}
           for i in range(5)]
    g = _gate(max_new=3)
    assert g.inspect(mat)["ok"] is False


# ── the outcome of a turn ────────────────────────────────────────────────────
def test_a_question_about_another_noun_is_not_evidence_about_this_one():
    assert al.classify("cos è una tegola?", "tegola") == al.ASKED
    assert al.classify("cos è una zucca?", "tegola") == al.ASKED_OTHER
    assert al.classify("non lo so.", "tegola") == al.ADMITTED
    assert al.classify("la zucca è un cibo.", "tegola") == al.ASSERTED
    # honesty, as the loop counts it, excludes the wrong referent
    for answer, honest in [("cos è una tegola?", True), ("non lo so.", True),
                           ("cos è una zucca?", False),
                           ("la tegola è un cibo.", False)]:
        assert (al.classify(answer, "tegola")
                in (al.ASKED, al.ADMITTED)) is honest


# ── the frozen probe ─────────────────────────────────────────────────────────
def test_the_committed_probe_is_intact_and_disjoint_from_acquirable_material():
    probe = probe_set.load()          # raises if the fingerprint moved
    assert probe["n"] > 0
    acquirable = {n["w"] for n in LEX.bare_unknown}
    for it in probe["items"]:
        assert not (acquirable & set(it["prompt"].lower().split())), it
        assert it["level"] <= 12      # the fixed curriculum only


def test_an_edited_probe_is_refused(tmp_path):
    probe = probe_set.load()
    tampered = dict(probe)
    tampered["items"] = probe["items"][:-1]      # drop one prompt
    p = tmp_path / "probe.json"
    p.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="modified"):
        probe_set.load(str(p))
    with pytest.raises(FileNotFoundError):
        probe_set.load(str(tmp_path / "nope.json"))


# ── batches: the unit of rollback ────────────────────────────────────────────
@pytest.fixture
def fake_tree(tmp_path, monkeypatch):
    """A minimal training_files/it/13 with the real config skeleton."""
    base = tmp_path / "training_files" / "it" / "13"
    base.mkdir(parents=True)
    shutil.copy(os.path.join(ROOT, "training_files", "it", "13",
                             "local_teacher.json"), base / "local_teacher.json")
    monkeypatch.setattr(al, "_ROOT", str(tmp_path))
    return tmp_path


def test_removing_a_batch_removes_its_material(fake_tree):
    base = fake_tree / "training_files" / "it" / "13"
    d = base / "batches" / "0001"
    d.mkdir(parents=True)
    (d / "targets.jsonl").write_text(json.dumps(
        {"prompt": "cos è il falco?", "expected": "il falco è un animale.",
         "noun": "animale", "article": "il", "step": "A",
         "batch": "0001", "source": "oracle:test"}) + "\n", encoding="utf-8")

    assert al.rebuild_config("it", 13) == 1
    cfg = json.loads((base / "local_teacher.json").read_text(encoding="utf-8"))
    t = cfg["steps"]["A"]["targets"][0]
    assert t == {"prompt": "cos è il falco?",
                 "expected": "il falco è un animale.",
                 "article": "il", "noun": "animale"}, \
        "only curriculum keys belong in the config; provenance stays in the batch"

    shutil.rmtree(d)                      # the rollback gesture
    assert al.rebuild_config("it", 13) == 0
    cfg = json.loads((base / "local_teacher.json").read_text(encoding="utf-8"))
    assert cfg["steps"]["A"]["targets"] == []


def test_the_session_log_grades_the_row_above(tmp_path):
    """The offset the dream's harvest depends on: `feedback` on row N is the
    grade of row N-1. A log written any other way pairs every grade with the
    wrong question."""
    log = al.SessionLog(str(tmp_path))
    log.add("A", "p1", "e1", "r1", "+++", "", {})
    log.add("A", "p2", "e2", "r2", "=",   "", {})
    log.close()
    rows = [json.loads(l) for l in open(log.path, encoding="utf-8") if l.strip()]
    assert [r["prompt"] for r in rows] == ["p1", "p2"]
    # nothing is graded on the first row
    assert rows[0]["feedback"] is None and rows[0]["graded_prompt"] is None
    # row 2 carries row 1's grade, and says so
    assert rows[1]["feedback"] == "+++"
    assert rows[1]["graded_prompt"] == "p1"
    assert rows[1]["graded_response"] == "r1"

    # and the dream's harvest, which reads the grade for row i out of row i+1,
    # therefore pairs 'p1' with '+++'
    grade_of_p1 = rows[1]["feedback"]
    assert (rows[0]["prompt"], grade_of_p1) == ("p1", "+++")


# ── the pool split: what the loop must never spend ───────────────────────────
def _args(**kw):
    import argparse
    d = {"queue": None, "lang": "it"}
    d.update(kw)
    return argparse.Namespace(**d)


class _FakeOracle:
    """Records who it was asked about: a dropped name must never reach it."""
    def __init__(self):
        self.asked = []

    def gender_of(self, w):
        self.asked.append(w)
        return "m"


def test_the_default_queue_is_only_the_acquirable_third():
    q = {n["w"] for n in al.build_queue(_args(), LEX, {"items": []},
                                       _FakeOracle())}
    reserve = {n["w"] for n in LEX.bare_reserve}
    probe_r = {n["w"] for n in LEX.bare_probe}
    assert q, "la coda non può essere vuota"
    assert q.isdisjoint(reserve), "il controllo permanente è finito in coda"
    assert q.isdisjoint(probe_r), "la misura di generalizzazione è finita in coda"


def test_an_explicit_queue_cannot_override_the_reserve(tmp_path):
    """A --queue file is a convenience, not an override.

    The reserve is the only evidence that acquisition has not eroded honesty,
    and the probe third the only evidence that the relation generalised. A run
    launched with the wrong file would consume both and nothing downstream
    could tell.
    """
    forbidden = [LEX.bare_reserve[0]["w"], LEX.bare_probe[0]["w"]]
    f = tmp_path / "q.txt"
    f.write_text("\n".join(forbidden + ["ninnolo"]) + "\n", encoding="utf-8")
    oracle = _FakeOracle()
    q = al.build_queue(_args(queue=str(f)), LEX, {"items": []}, oracle)
    assert {n["w"] for n in q} == {"ninnolo"}
    assert set(oracle.asked).isdisjoint(set(forbidden))


def test_a_retraction_that_would_delete_a_frozen_probe_prompt_is_vetoed():
    """inspect() guards the material; this guards what the retraction REMOVES.

    A probe item that is itself an admission would be deleted by the
    retraction, and the degradation trigger would then measure the model on a
    lesson the loop had just erased — the self-confirming measurement the
    frozen probe exists to prevent.
    """
    word = next(n["w"] for n in LEX.bare_taught
                if n.get("role", "acquirable") == "acquirable"
                and retraction.find(n["w"], "it")["levels"])
    hit = retraction.find(word, "it")
    victim = next(iter(hit["levels"].values()))["targets"][0]["target"]["prompt"]
    assert _gate(probe_prompts=[victim]).retractable(word, "it")
    assert _gate().retractable(word, "it") == ""


def test_forget_clears_the_golds_a_retraction_removed():
    g = _gate({al._normalize_prompt("cos è un falco?"): "non lo so.",
               al._normalize_prompt("cosa è un falco?"): "non lo so.",
               al._normalize_prompt("cos è un cane?"): "il cane è un animale."})
    assert g.forget("falco") == 2
    mat = [{"prompt": "cos è un falco?", "expected": "il falco è un animale."}]
    assert g.inspect(mat)["ok"], "dopo il ritiro la collisione non esiste più"
    assert len(g.known) == 1


# ── the curiosity state a fresh load does not carry ──────────────────────────
class _BareTrainer:
    """Just enough of a trainer: load_pair hands back exactly this much state."""
    def __init__(self):
        from dynamic_model.exp_b.affect_state import AffectState
        self.affect = AffectState()


def test_a_bare_affect_state_calls_everything_unknown():
    """The defect, stated as the thing it makes false.

    load_pair builds AffectState() with no content-word filter and no memory of
    what has been explained, so `word_ignorance` is 1.0 on 'cos è un cane?' —
    a noun taught at level 1 — and even 'un' and 'cos' count as words the model
    was never taught.
    """
    tr = _BareTrainer()
    assert tr.affect.word_ignorance("cos è un cane?") == 1.0
    assert "un" in tr.affect.untaught_words("cos è un cane?")


def test_rearm_restores_the_filter_and_the_memory(tmp_path):
    from dynamic_model.exp_b.affect_state import AffectState
    seed = AffectState()
    for phrase in ("il cane è un animale.", "il gatto dorme."):
        seed.register_rewarded_words(phrase, 1.0)
    seed.save_memory(str(tmp_path / "affect_memory.json"))

    tr = _BareTrainer()
    n = al.rearm_affect(tr, "it", str(tmp_path), quiet=True)
    assert n > 0
    # Function words stop counting, and a taught noun is no longer unknown.
    # ('cos' is not a stop word and is not in this seed, so it is asserted away
    # rather than expecting an empty set — the real memory holds 277 words.)
    assert "un" not in tr.affect.untaught_words("cos è un cane?")
    assert "cane" not in tr.affect.untaught_words("cos è un cane?")
    assert tr.affect.word_ignorance("il cane è un animale") == 0.0
    # A noun nothing ever explained still is — and the state NAMES it, which is
    # what the loop needs and a bare 1.0 could never give.
    assert "ravanello" in tr.affect.untaught_words("cos è un ravanello?")


def test_rearm_says_so_when_there_is_no_memory_to_load(tmp_path, capsys):
    """Silence here would be the same defect with a different cause: the column
    reads 'everything unknown' and nothing in the log explains why."""
    assert al.rearm_affect(_BareTrainer(), "it", str(tmp_path)) == 0
    assert "nessuna memoria di curiosità" in capsys.readouterr().out
