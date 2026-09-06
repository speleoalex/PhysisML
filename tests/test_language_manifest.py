"""
A language is a folder, not a branch in the code.

Two failures motivated this file, both silent:

  * the axiom words were a Python dict keyed by language, so the English build
    ran for hours protecting 'mamma' -- which tokenizer_en.json splits into
    m|am|ma -- and froze eight embedding rows that never occur in
    training_files/en at all;
  * every default in the publication chain said 'it'. There is one
    models/active.pt for every language and build.sh overwrites it at the end
    of every level, so exporting right after an English build would have
    stamped English weights with `"language": "it"` and the Italian card.

The tests below pin the two properties that make a third language a data
change: every language's words come from its own manifest, and no two
languages write to the same file.
"""
import ast
import glob
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_model import language as L                      # noqa: E402
from physisml.tokenizer import BPETokenizer                  # noqa: E402

LANGS = L.available()


def test_the_languages_are_discovered_from_disk():
    """No list of languages anywhere in the code."""
    assert "it" in LANGS
    for code in LANGS:
        assert os.path.isdir(os.path.join(_ROOT, "training_files", code))


@pytest.mark.parametrize("code", LANGS)
def test_every_language_declares_its_own_words(code):
    """Axioms and stop words cannot be derived from a language code."""
    lg = L.load(code)
    assert lg.has_manifest, (
        f"training_files/{code}/language.json is missing — the build would "
        f"fall back to another language's words")
    assert lg.axioms("phonetic"), f"{code}: no phonetic axioms declared"
    assert lg.axioms("grammar"),  f"{code}: no grammar axioms declared"
    assert len(lg.stop_words) > 20, f"{code}: stop word list looks unfinished"


@pytest.mark.parametrize("code", LANGS)
def test_axiom_words_are_whole_in_their_own_vocabulary(code):
    """An axiom that splits protects arbitrary subwords, not the word."""
    lg  = L.load(code)
    tok_path = os.path.join(_ROOT, lg.tokenizer)
    if not os.path.exists(tok_path):
        pytest.skip(f"no vocabulary for '{code}'")
    tok = BPETokenizer()
    tok.load(tok_path)
    for kind in ("phonetic", "grammar"):
        for text, _prot in lg.axioms(kind):
            for word in text.split():
                ids = tok.encode(word)
                assert len(ids) == 1, (
                    f"{code}/{kind}: '{word}' encodes to {len(ids)} tokens "
                    f"({'|'.join(tok.decode([i]) for i in ids)}) — the "
                    f"protection would land on subwords")


@pytest.mark.parametrize("code", LANGS)
def test_no_axiom_word_is_absent_from_the_corpus(code):
    """Protecting a row the corpus never trains is protection of nothing."""
    lg = L.load(code)
    base = os.path.join(_ROOT, "training_files", code)
    corpus = ""
    for lvl in sorted(int(d) for d in os.listdir(base)
                      if d.isdigit() and os.path.isdir(os.path.join(base, d))):
        p = os.path.join(base, str(lvl), "qa_corpus.txt")
        if os.path.exists(p):
            corpus += open(p, encoding="utf-8").read().lower()
    if not corpus:
        pytest.skip(f"no corpus for '{code}'")
    missing = [w for text, _ in lg.axioms("phonetic")
               for w in text.lower().split() if w not in corpus]
    assert not missing, (
        f"{code}: phonetic axioms absent from the corpus: {missing}")


def test_italian_paths_keep_their_historical_names():
    """Renaming them would orphan every published checkpoint and baseline."""
    it = L.load("it")
    assert it.tokenizer == "dynamic_model/data/tokenizer_8k.json"
    assert it.probe_set == "dynamic_model/data/probe_set.json"
    assert it.card      == "huggingface/README.md"
    assert it.out_dir   == "hf_upload"


def test_a_new_language_needs_no_code():
    """Everything a language does not declare follows the conventions."""
    fresh = L.Language("xx")
    assert fresh.tokenizer      == "dynamic_model/data/tokenizer_xx.json"
    assert fresh.probe_set      == "dynamic_model/data/probe_set_xx.json"
    assert fresh.card           == "huggingface/README.xx.md"
    assert fresh.out_dir        == "hf_upload_xx"
    assert fresh.curriculum_dir == os.path.join("training_files", "xx")
    assert fresh.checkpoint_dir == os.path.join("models", "checkpoints", "xx")
    assert fresh.hf_repo        == ""   # never guessed: a push cannot be undone


@pytest.mark.parametrize("slot", ["tokenizer", "probe_set", "card", "out_dir",
                                  "curriculum_dir", "checkpoint_dir"])
def test_no_two_languages_write_to_the_same_file(slot):
    """The question this file exists to answer: do the languages collide?"""
    seen = {}
    for code in LANGS:
        value = getattr(L.load(code), slot)
        assert value not in seen, (
            f"'{code}' and '{seen[value]}' both use {slot} = {value}")
        seen[value] = code


def test_declared_hf_repos_are_distinct():
    """Publishing one language over another's Hub repo replaces revisions the
    README and the Zenodo record point at."""
    repos = [L.load(c).hf_repo for c in LANGS if L.load(c).hf_repo]
    assert len(repos) == len(set(repos)), f"same Hub repo declared twice: {repos}"


@pytest.mark.parametrize("code", LANGS)
def test_a_vocabulary_identifies_its_language(code):
    """The guard the export chain relies on: a checkpoint has no language
    field, so the tokenizer beside it is the only evidence."""
    path = os.path.join(_ROOT, L.load(code).tokenizer)
    if not os.path.exists(path):
        pytest.skip(f"no vocabulary for '{code}'")
    assert L.detect(path) == code


@pytest.mark.parametrize("code", LANGS)
def test_a_trained_vocabulary_still_identifies_its_language(code):
    """The vocabulary a build ships is the base one plus what the levels
    added, and it has to be recognisable all the same.

    The first version of detect() accepted a difference of eight tokens, room
    for the special tokens and nothing more, so a language stopped being
    detectable the moment it learned words: level_13's Italian vocabulary is
    twenty tokens past its base and matched nothing, which sent every caller
    to the Italian default -- right for Italian by accident, wrong for
    everything else.
    """
    grown = sorted(glob.glob(os.path.join(
        _ROOT, "models", "checkpoints", code, "level_*", "tokenizer.json")))
    if not grown:
        pytest.skip(f"no build on disk for '{code}'")
    for path in grown:
        assert L.detect(path) == code, path


def test_no_module_keeps_a_table_of_words_per_language(_src_cache={}):
    """The regression this whole layout exists to prevent.

    A dict keyed by language code in a .py file is a list of languages the
    code knows about: it is complete on the day it is written and silently
    wrong on the day a language is added. Every such table is now a manifest.
    """
    offenders = []
    for mod in ("dynamic_model/train_curriculum.py",
                "dynamic_model/stop_words.py",
                "dynamic_model/run.py",
                "scripts/export_hf.py"):
        src = open(os.path.join(_ROOT, mod), encoding="utf-8").read()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if len(keys) > 1 and set(keys) <= set(LANGS):
                offenders.append(f"{mod}:{node.lineno} keyed by {keys}")
    assert not offenders, (
        "per-language tables left in the source:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("code", LANGS)
def test_the_training_loop_reads_the_manifest(code):
    """axioms_for() is the only path from the training loop to the words."""
    from dynamic_model.train_curriculum import axioms_for
    for kind in ("phonetic", "grammar"):
        assert axioms_for(kind, code) == L.load(code).axioms(kind)


def test_an_undeclared_language_gets_no_words():
    """Silence, not another language's grammar: the English build ran for
    hours protecting Italian rows because the fallback was Italian."""
    from dynamic_model.train_curriculum import axioms_for
    assert axioms_for("phonetic", "de") == []
    assert L.load("de").stop_words == set()
    assert L.load("de").polarity == {}


@pytest.mark.parametrize("code", LANGS)
def test_every_language_can_answer_yes_and_no(code):
    """Without these the grader cannot tell an answer from its opposite."""
    pol = L.load(code).polarity
    assert pol.get("yes") and pol.get("no"), (
        f"{code}: no \"polarity\" block — the closed questions go ungraded")


def test_the_italian_stop_words_match_the_module():
    """stop_words.py still exports the Italian set by name, for the callers
    that predate languages; it must be the manifest's, not a second copy."""
    from dynamic_model.stop_words import STOP_WORDS, for_language
    assert STOP_WORDS == L.load("it").stop_words
    assert for_language("en") == L.load("en").stop_words
