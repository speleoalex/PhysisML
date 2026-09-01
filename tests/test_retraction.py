"""
Tests for retracting an admission of ignorance.

The autonomy loop can learn that a falcon is an animal while level 12 teaches
`cos è un falco?` -> `non lo so.` One of the two has to go, and removing the
admission is the semantics of a knowledge base that grows. What is tested here
is that the removal is reversible, complete, and refused where it would destroy
a measurement:

  * reversible LITERALLY — restore puts every item back at its index, so
    qa_corpus.txt (a derived, committed file that generate_qa_corpus.py --check
    compares byte for byte) comes back identical;
  * durable — the ledger keeps a regeneration of the pool from resurrecting
    what the loop just retracted;
  * all of a noun or none — the half-retraction is how contradictory
    supervision got in before;
  * never the reserve, never the probe, never a frozen-probe prompt.

Run with:  python3 -m pytest tests/test_retraction.py -v
"""
import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                                  # noqa: E402
from dynamic_model import retraction                                # noqa: E402


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A minimal curriculum: one level that teaches two admissions and a fact.

    Built rather than copied, so the test says what it depends on. The module
    resolves everything from _ROOT, and _regen chdirs to it, so monkeypatching
    that one name relocates the whole thing.
    """
    lvl = tmp_path / "training_files" / "it" / "12"
    lvl.mkdir(parents=True)
    cfg = {"lang": "it", "level": 12, "steps": {
        "A": {"targets": [
            {"prompt": "cos è un cane?", "expected": "il cane è un animale.",
             "noun": "cane"}]},
        "E": {"targets": [
            {"prompt": "cos è un falco?", "expected": "non lo so.",
             "noun": "so"},
            {"prompt": "cosa è un falco?", "expected": "non lo so.",
             "noun": "so"},
            {"prompt": "cos è una bussola?", "expected": "non lo so.",
             "noun": "so"}]}}}
    (lvl / "local_teacher.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs = [{"prompt": "cos è un cane?", "response": "il cane è un animale."},
             {"prompt": "cos è un falco?", "response": "non lo so."},
             {"prompt": "bene! cos è un falco?", "response": "non lo so."},
             {"prompt": "cos è una bussola?", "response": "non lo so."}]
    (lvl / "qa_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(retraction, "_ROOT", str(tmp_path))
    import generate_qa_corpus
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        generate_qa_corpus.generate(12, "it", quiet=True)
    finally:
        os.chdir(cwd)
    return tmp_path


# ── what an admission is ─────────────────────────────────────────────────────
def test_only_a_bare_admission_counts():
    assert retraction.is_admission("non lo so.")
    assert retraction.is_admission("  Non lo so ")
    assert retraction.is_admission("non so")
    # An assertion that happens to start the same way is not an admission:
    # retracting it would delete a fact.
    assert not retraction.is_admission("non lo so perché è nuovo.")
    assert not retraction.is_admission("il falco è un animale.")


# ── find ─────────────────────────────────────────────────────────────────────
def test_find_catches_every_phrasing_including_the_prefixed_harvest(tree):
    hit = retraction.find("falco", "it")
    assert list(hit["levels"]) == ["12"]
    assert len(hit["levels"]["12"]["targets"]) == 2
    # 'bene! cos è un falco?' too: leaving the teacher-prefixed variants behind
    # would keep N1 drilling 'non lo so' against the new class for ever.
    assert len(hit["levels"]["12"]["pairs"]) == 2
    assert hit["other"] == []


def test_find_reports_a_non_admission_gold_as_other(tree):
    assert retraction.find("cane", "it")["other"]


def test_a_word_is_matched_whole_not_as_a_substring(tree):
    # 'falc' must not reach 'falco', and 'falco' must not reach 'falcone'.
    assert retraction.find("falc", "it")["levels"] == {}


# ── the round trip ───────────────────────────────────────────────────────────
def test_retract_then_restore_is_byte_identical(tree):
    lvl = tree / "training_files" / "it" / "12"
    before = {p: _digest(lvl / p) for p in
              ("local_teacher.json", "qa_pairs.jsonl", "qa_corpus.txt")}
    save = str(tree / "batch")
    info = retraction.retract("falco", "it", save_dir=save, batch="0001")
    assert (info["targets"], info["pairs"]) == (2, 2)
    assert any(_digest(lvl / p) != before[p] for p in before)

    retraction.restore("falco", "it", save_dir=save)
    for p, d in before.items():
        assert _digest(lvl / p) == d, f"{p} non è tornato identico"


def test_the_corpus_no_longer_teaches_the_retracted_admission(tree):
    retraction.retract("falco", "it", save_dir=str(tree / "batch"))
    corpus = (tree / "training_files" / "it" / "12" / "qa_corpus.txt").read_text(
        encoding="utf-8")
    assert "falco" not in corpus
    assert "bussola" in corpus       # the other admission is untouched
    assert "cane" in corpus


def test_a_noun_with_a_real_fact_cannot_be_retracted(tree):
    with pytest.raises(ValueError):
        retraction.retract("cane", "it", save_dir=str(tree / "batch"))
    # and nothing was written
    assert not retraction.retracted("it")


def test_restore_without_a_save_file_refuses_instead_of_guessing(tree):
    with pytest.raises(FileNotFoundError):
        retraction.restore("falco", "it", save_dir=str(tree / "nowhere"))


# ── the ledger ───────────────────────────────────────────────────────────────
def test_the_ledger_records_and_a_restore_annuls_it(tree):
    save = str(tree / "batch")
    retraction.retract("falco", "it", save_dir=save, batch="0001",
                       source="oracle:test")
    led = retraction.retracted("it")
    assert set(led) == {"falco"}
    assert led["falco"]["batch"] == "0001"
    assert led["falco"]["source"] == "oracle:test"
    assert led["falco"]["levels"] == [12]
    retraction.restore("falco", "it", save_dir=save)
    assert retraction.retracted("it") == {}


def test_a_retracted_noun_leaves_the_generated_pool():
    """The ledger is read by the generator, not by its callers.

    expand_teacher_pools.py rebuilds step E from the lexicon on every rebuild.
    Without this, the first regeneration after an acquisition would put the
    retracted admission straight back, and nothing would say so.
    """
    lex = etp.Lex(etp.load_lexicon("it"), "it")
    taught = {n["w"] for n in lex.bare_taught}
    assert "falco" in taught                       # nothing retracted here
    real = etp._retracted_words("it")
    assert taught.isdisjoint(real)


# ── the three roles ─────────────────────────────────────────────────────────
def test_the_probe_role_is_taught_nowhere():
    """The only way left to ask whether the honesty relation generalised.

    Measured on the finished 0-12 model: 7% honest on names outside the pool
    against 6/6 inside it. Teaching these would answer the question by
    deleting it.
    """
    lex = etp.Lex(etp.load_lexicon("it"), "it")
    assert lex.bare_probe, "il terzo ruolo esiste per essere misurato"
    taught = {n["w"] for n in lex.bare_taught}
    for n in lex.bare_probe:
        assert n["w"] not in taught
        assert not retraction.find(n["w"], "it")["levels"], \
            f"{n['w']} ha role=probe ma il curriculum lo insegna"


def test_the_reserve_is_taught_and_never_retracted():
    lex = etp.Lex(etp.load_lexicon("it"), "it")
    assert lex.bare_reserve
    gone = etp._retracted_words("it")
    for n in lex.bare_reserve:
        assert n["w"] not in gone
        assert retraction.find(n["w"], "it")["levels"], \
            f"{n['w']} ha role=reserve ma nessun livello gli insegna 'non lo so'"


def test_the_acquirable_pool_is_large_enough_to_spend_from():
    """Every acquisition spends one example of the honesty lesson.

    With six nouns the lesson dies after six acquisitions, which is why the
    pool was widened before the loop was allowed to retract anything.
    """
    lex = etp.Lex(etp.load_lexicon("it"), "it")
    acquirable = [n for n in lex.bare_taught
                  if n.get("role", "acquirable") == "acquirable"]
    assert len(acquirable) >= 20


# ── the two-sided rollback ───────────────────────────────────────────────────
def test_discarding_a_batch_brings_back_what_it_retracted(tree, monkeypatch,
                                                          tmp_path):
    """The seam that matters most, and the only one whose failure is silent.

    A rolled-back acquisition that did not restore its retraction would leave
    the curriculum teaching NEITHER the class nor the admission — a noun the
    model has simply been made to forget, with nothing in any file to say so.
    """
    import autonomy_loop as al
    monkeypatch.setattr(al, "_ROOT", str(tree))

    lvl13 = tree / "training_files" / "it" / "13"
    lvl13.mkdir(parents=True)
    (lvl13 / "local_teacher.json").write_text(
        json.dumps({"lang": "it", "level": 13,
                    "steps": {"A": {"targets": []}, "B": {"targets": []},
                              "C": {"targets": []}}},
                   ensure_ascii=False), encoding="utf-8")
    lvl12 = tree / "training_files" / "it" / "12"
    before = {p: _digest(lvl12 / p) for p in
              ("local_teacher.json", "qa_pairs.jsonl", "qa_corpus.txt")}

    b = al.Batch("it", 13, str(tmp_path / "ckpt"), "")
    info = b.retract("falco", source="oracle:test")
    assert info["targets"] == 2
    b.add({"w": "falco"}, "un animale",
          [{"prompt": "cos è un falco?", "expected": "il falco è un animale.",
            "noun": "falco"}], source="oracle:test")
    b.write()
    assert (tree / "training_files" / "it" / "13" / "batches" / b.slug
            / "retracted.json").exists()
    assert retraction.retracted("it")

    b.discard(str(tmp_path / "nonexistent.pt"))
    for p, d in before.items():
        assert _digest(lvl12 / p) == d, f"{p} non è tornato dopo il rollback"
    assert retraction.retracted("it") == {}
