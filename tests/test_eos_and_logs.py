"""
Tests for the EOS registration and for the session-log pairing.

Both fixes exist because something was silently wrong for a long time:

  - the tokenizer retrained on 2026-08-25 dropped <|EOS|>, so every piece of
    EOS machinery downstream (generator stop, gold suffix, GGUF metadata) was
    inert and the model had no way to end an answer outside Python;
  - the session record carried a feedback symbol with no indication that it
    grades the PREVIOUS turn, so every per-prompt reading of the log paired
    grades with the wrong question.

Neither failure raised an error, which is exactly why they need tests.

Run with:  python3 -m pytest tests/test_eos_and_logs.py -v
"""
import glob
import importlib.util
import json
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(ROOT)

from physisml.tokenizer import BPETokenizer                            # noqa: E402
import analyze_log as AL                                           # noqa: E402

REFERENCE = "models/active_tokenizer.json"
# What a build started from scratch seeds itself with.
SEED = "dynamic_model/data/tokenizer_8k.json"

# models/ is a build artefact and is not in the repository, so a fresh clone —
# and CI — has no active tokenizer to inspect. The tests below assert facts
# about a tokenizer that a build PRODUCED; they cannot be rewritten against a
# synthetic one without asserting something else. Skip them explicitly, one by
# one, so the report says which checks did not run and why: the tests that read
# only committed material (the corpora, the from-scratch seed) still run in CI
# and are the ones that would catch the marker regressing at the source.
needs_reference = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, REFERENCE)),
    reason=f"{REFERENCE} is a build artefact: run ./build.sh to produce it")


def _load_tc():
    """train_curriculum.py is a script, not a module: load it by path."""
    spec = importlib.util.spec_from_file_location(
        "tc_for_tests", "dynamic_model/train_curriculum.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tc_for_tests"] = mod
    spec.loader.exec_module(mod)
    return mod


def _snapshot_groups():
    """Snapshots grouped by the build tree they belong to.

    The EOS id is a property of the vocabulary a build started from, and
    models/checkpoints/ now holds one tree per language: the Italian one carries
    EOS at 2589, the English one at 2516. A flat list compared everything with
    the active tokenizer, so the first English build made the Italian snapshots
    look wrong when nothing about them had changed. What has to hold is that one
    tree never disagrees with itself — the same weights reloaded with a
    different marker is the failure this guards.

    The active tokenizer and the standalone export are the last build's output
    and form their own group; the cross-check below is that their id belongs to
    one of the trees on disk.
    """
    groups = {}
    for path in sorted(glob.glob("models/checkpoints/*/level_*/tokenizer.json")):
        groups.setdefault(path.split(os.sep)[2], []).append(path)
    groups["<active>"] = [p for p in (REFERENCE, "standalone/tokenizer.json")
                          if os.path.exists(p)]
    return groups


def _eos_of(path):
    tok = BPETokenizer(); tok.load(path)
    return tok.get_special_id(tok.EOS_TOKEN), max(tok.vocab.keys()) + 1


# ---------------------------------------------------------------------------
# EOS registration
# ---------------------------------------------------------------------------

class TestEosRegistration:

    @needs_reference
    def test_the_active_tokenizer_has_an_eos(self):
        """The single fact whose absence made all the EOS code dead."""
        tok = BPETokenizer()
        tok.load(REFERENCE)
        assert tok.get_special_id(tok.EOS_TOKEN) is not None, \
            "no EOS in the active tokenizer: run scripts/register_eos.py"

    @needs_reference
    def test_no_activated_row_is_left_without_a_token(self):
        """Every id from 0 to the maximum must decode to something.

        _sync_vocab_rows() activates every embedding row up to max(vocab)+1, so
        a hole anywhere below the maximum is a row the sampler can emit and
        decode() then raises KeyError on. This is the constraint that makes
        'just use a reserved slot like 256' wrong for this tokenizer.

        It used to be written as `eos == last_real + 1`, which was true only at
        the moment register_eos.py ran. The dream's N2-B grows the vocabulary,
        and the L11->L12 rebuild added 17 tokens AFTER the EOS at 2589: the
        assertion failed on a tokenizer that has no hole at all. The property
        worth defending is contiguity, not the position of EOS within it.
        """
        tok = BPETokenizer()
        tok.load(REFERENCE)
        eos = tok.get_special_id(tok.EOS_TOKEN)
        assert eos is not None and eos in tok.vocab, "EOS not in the vocabulary"
        ids = set(tok.vocab)
        holes = [i for i in range(max(ids) + 1) if i not in ids]
        assert not holes, \
            f"{len(holes)} activated rows with no token: {holes[:10]}"

    @needs_reference
    def test_every_snapshot_that_can_hold_eos_agrees_on_the_id(self):
        """Same weights get loaded with different snapshots across a build.

        A snapshot with a different EOS id would silently reinterpret the row
        the model trained as 'end of answer'. Snapshots whose vocabulary is
        SHORTER cannot hold the id without leaving a hole and are skipped by
        register_eos.py — they must simply have no EOS at all, never a
        different one.
        """
        groups = _snapshot_groups()
        ids = {}
        for group, paths in groups.items():
            marked = {p: _eos_of(p) for p in paths}
            found = {eos for eos, _ in marked.values() if eos is not None}
            assert len(found) <= 1, \
                f"{group}: snapshots disagree on EOS — " + \
                ", ".join(f"{p}@{e}" for p, (e, _) in marked.items())
            if not found:
                continue
            eos_id = found.pop()
            ids[group] = eos_id
            for path, (have, size) in marked.items():
                if have is None:
                    assert size != eos_id, \
                        f"{path} could hold EOS at {eos_id} but has none"

        # The active tokenizer is a copy of the last build's: its id must be one
        # a tree on disk actually uses, or the weights and the marker come from
        # different builds.
        active = ids.pop("<active>", None)
        if active is not None and ids:
            assert active in ids.values(), \
                f"{REFERENCE} has EOS at {active}, no build tree does: {ids}"

    @needs_reference
    def test_eos_survives_save_and_load(self, tmp_path):
        tok = BPETokenizer(); tok.load(REFERENCE)
        out = str(tmp_path / "t.json")
        tok.save(out)
        again = BPETokenizer(); again.load(out)
        assert again.get_special_id(again.EOS_TOKEN) == \
            tok.get_special_id(tok.EOS_TOKEN)
        assert again.is_special(again.get_special_id(again.EOS_TOKEN))

    @needs_reference
    def test_eos_encodes_to_exactly_one_token(self):
        tok = BPETokenizer(); tok.load(REFERENCE)
        eos = tok.get_special_id(tok.EOS_TOKEN)
        assert tok.encode(tok.EOS_TOKEN) == [eos]
        assert tok.encode("il cane dorme." + tok.EOS_TOKEN) == \
            tok.encode("il cane dorme.") + [eos]
        assert tok.decode([eos]) == tok.EOS_TOKEN

    @needs_reference
    def test_registering_eos_does_not_change_ordinary_text(self):
        """Registering a special token switches encode() to a slower path that
        splits on the special literal. Ordinary text must come out identical,
        otherwise every corpus encoding — and every trained id — shifts.
        """
        with_eos = BPETokenizer(); with_eos.load(REFERENCE)
        without  = BPETokenizer(); without.load(REFERENCE)
        without.special_tokens = {}
        without.special_ids    = {}
        assert with_eos.special_tokens, "reference has no special token to test"

        # The corpus now ends every answer with the marker on purpose, and on
        # those lines the two tokenizers MUST differ — that is the fix, not a
        # regression. Strip it and compare what is left: the ordinary text.
        lines = []
        for f in sorted(glob.glob("training_files/it/*/qa_corpus.txt")):
            with open(f, encoding="utf-8", errors="replace") as fh:
                lines += [l.strip().replace(with_eos.EOS_TOKEN, "")
                          for l in fh if l.strip()]
        lines = [l for l in lines if l]
        assert lines, "no corpus to sample"
        random.seed(11)
        for s in random.sample(lines, min(400, len(lines))):
            assert with_eos.encode(s) == without.encode(s), \
                f"encoding changed for {s!r}"

    def test_the_corpus_carries_the_marker_on_every_answer(self):
        """The tokenizer knowing EOS is not the same as the model learning it.

        qa_corpus.txt is what phase 0 trains on and what every dream replays;
        the teaching gold's EOS is a rounding error next to it. With no marker
        here the 0-12 build put P(newline) = 0.97 and P(EOS) = 0.00004 right
        after a finished answer, so llama.cpp and ollama had nothing to stop
        on and kept generating the teaching dialogue.
        """
        eos = BPETokenizer().EOS_TOKEN if hasattr(BPETokenizer, "EOS_TOKEN") \
              else "<|EOS|>"
        checked = 0
        for f in sorted(glob.glob("training_files/it/*/qa_corpus.txt")):
            with open(f, encoding="utf-8") as fh:
                block = [l.rstrip("\n") for l in fh]
            # Layout is prompt / answer / blank, repeated: the answer is the
            # line before every blank one.
            for i, line in enumerate(block[:-1]):
                if block[i + 1].strip() == "" and line.strip():
                    assert line.endswith(eos), \
                        f"{f}: answer without the marker: {line!r}"
                    checked += 1
        assert checked > 100, f"only {checked} answers checked — corpus missing?"

    def test_both_corpus_writers_use_the_same_marker(self):
        """qa_corpus.txt has two writers and is committed output. When their
        demo-prefix regexes drifted the corpora disagreed for weeks; the
        marker is the same kind of shared constant.
        """
        import re as _re
        sys.path.insert(0, "scripts")
        import generate_qa_corpus as G
        src = open("dynamic_model/train_curriculum.py", encoding="utf-8").read()
        m = _re.search(r'_EOS_MARK = "(.+?)"', src)
        assert m, "_EOS_MARK not found in train_curriculum.py"
        assert G.EOS_MARK == m.group(1) == BPETokenizer().EOS_TOKEN

    def test_the_seed_a_from_zero_build_uses_has_an_eos(self):
        """A build with no checkpoints does not read models/active_tokenizer.json.

        train_curriculum picks TOKENIZER = dynamic_model/data/tokenizer_8k.json
        (falling back to tokenizer_base.json), so that file — not the active one
        — is what a from-scratch run seeds itself with. The first version of
        register_eos.py covered the checkpoint snapshots and missed exactly this
        one.
        """
        tok = BPETokenizer()
        tok.load(SEED)
        eos = tok.get_special_id(tok.EOS_TOKEN)
        assert eos is not None, f"no EOS in {SEED}"
        last_real = max(i for i in tok.vocab if not tok.is_special(i))
        assert eos == last_real + 1

    @needs_reference
    def test_the_exporter_would_declare_the_eos(self):
        """export_gguf derives the GGUF eos_token_id from the tokenizer, so
        registration is all that stands between ollama and a native stop.
        """
        tok = BPETokenizer(); tok.load(REFERENCE)
        eos_id = -1
        if tok.EOS_TOKEN in tok.special_tokens:
            eos_id = tok.special_tokens[tok.EOS_TOKEN]
        assert eos_id >= 0


class TestEosDoesNotLeakIntoTheCuriosityMemory:

    def test_the_eos_literal_is_not_remembered_as_a_word(self):
        """SIGNAL 1 trains on 'expected + <|EOS|>', and the curiosity memory is
        filled from the decoded target. The word regex sees letters and knows
        nothing about markup, so without stripping the literal the memory grows
        a word 'eos' — harmless in itself, but it means the memory records
        whatever markup happens to pass through it.
        """
        import numpy as np
        import torch
        from dynamic_model.exp_b.affect_state import AffectState
        from dynamic_model.exp_b.modulator import AffectModulator
        from dynamic_model.exp_b.axioms import AxiomRegistry
        from dynamic_model.exp_b.trainer import TrainerB
        from physisml.torch_model import TorchGPT, TorchAdamOptimizer

        tok = BPETokenizer(); tok.load(SEED)
        # Must cover the EOS id: a smaller vocabulary indexes out of range.
        n = max(tok.vocab.keys()) + 1
        model = TorchGPT(n, 32, 2, 1, 64, 65, 0.0, active_vocab_size=n)
        af = AffectState()
        tr = TrainerB(model, tok, TorchAdamOptimizer(model.parameters(), lr=1e-4),
                      af, AffectModulator(af), AxiomRegistry())
        tr.step("ma", "ma" + tok.EOS_TOKEN, feedback=0.8)
        assert "eos" not in af.words_rewarded, sorted(af.words_rewarded)


# ---------------------------------------------------------------------------
# Session-log pairing
# ---------------------------------------------------------------------------

class TestTurnRecord:

    def test_the_graded_triple_is_the_previous_exchange(self):
        tc = _load_tc()
        r = tc.turn_record(
            turn=9, step="A",
            prompt="il gelato è dolce?",
            expected="secondo me il gelato è dolce perché è giallo.",
            response="secondo me il gelato è dolce perché è caldo.",
            feedback="+++", comment="insegna con struttura completa",
            graded_prompt="il libro è utile?",
            graded_expected="secondo me il libro è utile perché insegna.",
            graded_response="secondo me il libro è utile perché insegna.",
            content="", affect={}, stats={})
        # the grade belongs to the previous exchange, and now says so
        assert r["graded_response"] == r["graded_expected"]
        assert r["feedback"] == "+++"
        # ...while the row's own answer is the wrong one, unlabelled by the grade
        assert r["response"] != r["expected"]

    def test_feedback_keeps_its_name_and_place_for_the_quality_gate(self):
        """build.sh reads r['feedback'] and nothing else. Renaming or moving it
        would silently zero every quality gate in the build.
        """
        tc = _load_tc()
        r = tc.turn_record(turn=2, step="A", prompt="p", expected="e",
                           response="r", feedback="++", comment="c",
                           graded_prompt="gp", graded_expected="ge",
                           graded_response="gr", content="", affect={},
                           stats={})
        assert r["feedback"] == "++"
        score = (1.0 if r.get("feedback") in ("+++", "++")
                 else 0.5 if r.get("feedback") == "+" else 0.0)
        assert score == 1.0

    def test_first_turn_has_nothing_graded(self):
        tc = _load_tc()
        r = tc.turn_record(turn=1, step="A", prompt="p", expected="e",
                           response="r", feedback=None, comment="",
                           graded_prompt=None, graded_expected=None,
                           graded_response=None, content="", affect={},
                           stats={})
        assert r["feedback"] is None
        assert r["graded_prompt"] is None


class TestAnalyzeLogPairing:

    def test_graded_fields_win_when_present(self):
        rec = {"prompt": "nuovo", "response": "nuova risposta",
               "graded_prompt": "vecchio", "graded_response": "vecchia"}
        assert AL.graded(rec) == ("vecchio", "vecchia")
        assert AL.is_aligned([rec])
        assert AL.alignment_note([rec]) == ""

    def test_legacy_logs_fall_back_and_are_flagged(self):
        """The 37 sessions already on disk have no graded_* fields. Reading them
        is still useful, but the off-by-one must be stated, not hidden.
        """
        rec = {"prompt": "nuovo", "response": "nuova risposta",
               "feedback": "+++"}
        assert AL.graded(rec) == ("nuovo", "nuova risposta")
        assert not AL.is_aligned([rec])
        assert "sfasati" in AL.alignment_note([rec])

    def test_a_real_session_file_still_parses(self):
        logs = sorted(glob.glob("models/checkpoints/it/level_*/session_*.jsonl"))
        if not logs:
            pytest.skip("no session logs on disk")
        recs = [json.loads(l) for l in open(logs[-1], encoding="utf-8")
                if l.strip()]
        assert recs
        for r in recs[:50]:
            p, resp = AL.graded(r)
            assert isinstance(p, str) and isinstance(resp, str)
