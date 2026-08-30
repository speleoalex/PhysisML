"""
Tests for the ontology (L11) and curiosity (L12) material.

These check the DATA and the pure functions, not the model: they must be able
to fail before an eight-hour build starts, because every defect they look for
is one that would otherwise show up only as a bad number at the end.

Run with:  python3 -m pytest tests/test_ontology_curiosity.py -v
"""
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

os.chdir(ROOT)          # the loaders use paths relative to the repo root

import expand_teacher_pools as E                                  # noqa: E402
from dynamic_model.local_teacher import LocalTeacher               # noqa: E402
from dynamic_model.stop_words import STOP_WORDS                    # noqa: E402

ONTO_LEVELS = (11, 12)


@pytest.fixture(scope="module")
def lex():
    return E.Lex(E.load_lexicon("it"))


def load_cfg(level):
    with open(f"training_files/it/{level}/local_teacher.json",
              encoding="utf-8") as f:
        return json.load(f)


def all_targets(level):
    cfg = load_cfg(level)
    for sname, step in cfg["steps"].items():
        tmpl = step.get("prompt_template", "{prompt}")
        for t in step["targets"]:
            if not isinstance(t, dict):
                continue          # L0 lists bare strings, not target dicts
            prompt = tmpl.format(prompt=t.get("prompt", ""),
                                 target=t.get("prompt", ""))
            yield sname, step, prompt, t


# ── the lexicon's ontology ─────────────────────────────────────────────────

class TestOntology:

    def test_every_noun_resolves_to_a_class(self, lex):
        missing = [n["w"] for n in lex.nouns if not lex.cls_of(n)]
        assert not missing, f"nomi senza classe: {missing}"

    def test_every_class_is_declared(self, lex):
        unknown = {lex.cls_of(n) for n in lex.nouns} - set(lex.classes)
        assert not unknown, f"classi usate ma non dichiarate: {unknown}"

    def test_hypernym_chains_terminate(self, lex):
        """No cycle: following `classes` must always reach a None root."""
        for start in lex.classes:
            seen, cur = [start], lex.classes[start]
            while cur is not None:
                assert cur not in seen, f"ciclo di iperonimi: {seen + [cur]}"
                assert cur in lex.classes, f"iperonimo non dichiarato: {cur!r}"
                seen.append(cur)
                cur = lex.classes[cur]

    def test_negative_pool_excludes_hypernyms(self, lex):
        """A wrong-class question needs a clean 'no'. 'il cane è una cosa?'
        does not have one, so the hypernyms must stay out of the pool."""
        hypernyms = {h for h in lex.classes.values() if h}
        overlap = set(lex.negative_pool) & hypernyms
        assert not overlap, f"iperonimi nel negative_pool: {overlap}"

    def test_probe_nouns_are_disjoint_from_taught(self, lex):
        taught = {n["w"] for n in lex.unknown_of(probe=False)}
        probe = {n["w"] for n in lex.unknown_of(probe=True)}
        assert taught and probe
        assert not (taught & probe)
        assert not ({n["w"] for n in lex.nouns} & (taught | probe)), \
            "un nome 'ignoto' è anche nel lessico noto"

    def test_probe_nouns_appear_in_no_training_material(self, lex):
        """The measurement is only valid on words the model never met."""
        import glob
        import re
        corpus = []
        for path in glob.glob("training_files/it/*/*.txt") + \
                    glob.glob("training_files/it/*/*.jsonl") + \
                    glob.glob("training_files/it/*/local_teacher.json"):
            with open(path, encoding="utf-8", errors="replace") as f:
                corpus.append((path, f.read().lower()))
        for n in lex.unknown_of(probe=True):
            pat = r"\b" + re.escape(n["w"]) + r"\b"
            hits = [p for p, text in corpus if re.search(pat, text)]
            assert not hits, f"il nome probe {n['w']!r} compare in {hits}"


# ── the generated pools ────────────────────────────────────────────────────

class TestPools:

    def test_every_classified_noun_is_asked_cos_e(self):
        """The is-a relation must cover the lexicon, not a sample of it.

        Each step used to draw its own random k nouns, so the three forms of
        the same relation were taught about three different subsets: 'cane' was
        in none of them and appeared only as the ANSWER to 'fai un esempio di
        animale'. Measured on the finished 0-12 model, that is exactly where it
        broke — 'cos è il cane?' -> 'il cane è una cosa', the right shape with a
        superordinate instead of the class, while 'cos è un cane?' and the
        yes/no form were both right.
        """
        import json as _json
        import re
        lex = _json.load(open("training_files/it/lexicon.json", encoding="utf-8"))
        onto = lex.get("ontology", {})
        kind_class = onto.get("kind_class", {})
        classified = [n["w"] for n in lex["nouns"]
                      if n.get("cls") or kind_class.get(n.get("kind"), "")]
        asked = {t["prompt"] for _s, _st, _p, t in all_targets(11)
                 if t["prompt"].startswith("cos è ")}
        missing = [w for w in classified
                   if not any(re.search(r"\b" + re.escape(w) + r"\b", p)
                              for p in asked)]
        assert not missing, f"nouns never asked 'cos è X?' at L11: {missing}"

    def test_the_bare_unknown_names_appear_nowhere_else(self):
        """L12 step E teaches 'non lo so.' about names the model cannot know.

        That is only true while the names appear in no trained file. A name
        that leaks into another level's material — or into unknown_nouns, which
        L12 itself teaches — turns the step into training the model to deny
        something it was taught.
        """
        import glob as _glob, json as _json, re
        lex = _json.load(open("training_files/it/lexicon.json", encoding="utf-8"))
        bare = [n["w"] for n in lex.get("bare_unknown_nouns", [])]
        assert bare, "no bare_unknown_nouns in the lexicon"
        taught = {n["w"] for n in lex.get("unknown_nouns", [])}
        assert not (set(bare) & taught), "a bare name is also in unknown_nouns"

        # Trained material only: *.txt at the level root, plus the pools and
        # pair files. _reference/ subdirectories are not globbed by phase 0.
        patterns = ["training_files/it/*/*.txt", "training_files/it/*/*.json",
                    "training_files/it/*/*.jsonl"]
        offenders = []
        for pat in patterns:
            for path in _glob.glob(pat):
                if "/12/" in path:          # step E is where they belong
                    continue
                text = open(path, encoding="utf-8", errors="replace").read()
                for w in bare:
                    if re.search(r"\b" + re.escape(w) + r"\b", text):
                        offenders.append(f"{w} in {path}")
        assert not offenders, f"bare names in trained material: {offenders[:5]}"

    @pytest.mark.parametrize("level", ONTO_LEVELS)
    def test_no_prompt_has_two_gold_answers(self, level):
        """Two answers for one prompt is contradictory supervision: whichever
        the model produces, the evaluator marks it wrong half the time."""
        gold = {}
        for sname, _step, prompt, t in all_targets(level):
            gold.setdefault(prompt, set()).add(t["expected"])
        clash = {p: v for p, v in gold.items() if len(v) > 1}
        assert not clash, f"prompt con gold contrastanti: {clash}"

    def test_no_prompt_has_two_golds_across_the_whole_curriculum(self):
        """The same check over ALL levels at once.

        A prompt is shared material: every level's qa_corpus is trained and
        replayed in the dream, and the retention matrix scores every checkpoint
        on every level. Two levels asking one question with different expected
        shapes made L3 read as regressed after L5 — 'cosa fa il cane?' was L3
        step D with one verb and L5 step B with two.
        """
        gold, where = {}, {}
        for level in range(13):
            if not os.path.exists(
                    f"training_files/it/{level}/local_teacher.json"):
                continue
            for sname, _step, prompt, t in all_targets(level):
                gold.setdefault(prompt, set()).add(t["expected"])
                where.setdefault(prompt, set()).add(f"L{level}{sname}")
        clash = {p: (sorted(v), sorted(where[p]))
                 for p, v in gold.items() if len(v) > 1}
        assert not clash, f"prompt con gold contrastanti: {clash}"

    def test_harvested_pairs_add_no_new_conflict(self):
        """The pairs harvested from sessions must not contradict the pools.

        _update_qa_pairs_from_sessions adds any prompt not already in
        qa_pairs.jsonl, without checking whether that prompt already has a
        different gold somewhere — which is how 'come ti chiami?' ended up
        answered both 'physisml!' (pool) and 'mi chiamo physisml.' (harvest).

        There is now a guard in _update_qa_pairs_from_sessions that refuses a
        candidate whose question already carries a different gold anywhere in
        the material, prefixes normalised away. This test is what catches the
        guard being weakened or bypassed.
        """
        known = set()          # all conflicts cleared: nothing may reappear
        gold = {}
        for level in range(13):
            d = f"training_files/it/{level}"
            if os.path.exists(f"{d}/local_teacher.json"):
                for _sn, _st, prompt, t in all_targets(level):
                    gold.setdefault(prompt, set()).add(t["expected"].strip())
            q = f"{d}/qa_pairs.jsonl"
            if os.path.exists(q):
                with open(q, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            o = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        gold.setdefault(o.get("prompt", "").strip(),
                                        set()).add(o.get("response", "").strip())
        clash = {p for p, v in gold.items() if len(v) > 1}
        assert clash <= known, f"conflitti nuovi: {sorted(clash - known)}"

    @pytest.mark.parametrize("level", ONTO_LEVELS)
    def test_every_gold_reaches_top_grade(self, level):
        """A target whose own gold answer cannot earn '+++' makes the teacher
        loop on it forever: '+' resets the failure counter without advancing."""
        teacher = LocalTeacher(lang="it", level=level)
        for sname, step, _prompt, t in all_targets(level):
            fb, _cmt = teacher._evaluate(t["expected"], t, step)
            assert fb == "+++", (level, sname, t["expected"], fb)

    @pytest.mark.parametrize("level", ONTO_LEVELS)
    def test_no_malformed_indefinite_articles(self, level):
        """'questa è un acqua' and 'questo è un sole' are not Italian: mass
        nouns take no indefinite article and unique referents take the
        definite one."""
        lexd = E.load_lexicon("it")
        bad_forms = [f"un {n['w']}" for n in lexd["nouns"]
                     if n.get("mass") or n.get("uniq")]
        bad_forms += [f"una {n['w']}" for n in lexd["nouns"]
                      if n.get("mass") or n.get("uniq")]
        for _sname, _step, prompt, t in all_targets(level):
            text = f"{prompt} {t['expected']}"
            for bad in bad_forms:
                assert bad not in text, f"{bad!r} in {text!r}"

    def test_l12_step_anchor_is_of_another_class(self, lex):
        """With a same-class anchor the model can produce the right class by
        copying it out of the prompt, and the turn measures nothing."""
        for sname, _step, prompt, t in all_targets(12):
            if sname not in ("A", "B"):
                continue
            head, _sep, _tail = prompt.partition(",")
            anchor_cls = next((c for c in sorted(lex.classes, key=len,
                                                 reverse=True)
                               if c in head), None)
            assert anchor_cls, prompt
            answer_cls = next((c for c in sorted(lex.classes, key=len,
                                                 reverse=True)
                               if c in t["expected"]), None)
            if answer_cls:          # step A's gold is a question, no class
                assert anchor_cls != answer_cls, prompt

    def test_generators_are_deterministic(self, lex):
        """The pools are committed, so regenerating must reproduce them."""
        for name in ("g_cos_e", "g_is_a_no", "g_chiedi_ignoto",
                     "g_non_chiedere"):
            gen = getattr(E, name)
            a = gen(random.Random("seed"), lex, 12)
            b = gen(random.Random("seed"), lex, 12)
            assert a == b, name


# ── the grade sanity check on the new shapes ───────────────────────────────

def _slice_from_train_curriculum() -> dict:
    """grade_by_coverage, without importing the whole module.

    train_curriculum imports torch and the anthropic client at module scope;
    this needs one pure function out of it, so the source slice holding it is
    compiled on its own, with STOP_WORDS supplied from its real home.
    """
    src = open("dynamic_model/train_curriculum.py", encoding="utf-8").read()
    i = src.index("def grade_by_coverage(")
    j = src.index("# " + "-" * 75, i)
    ns = {"STOP_WORDS": STOP_WORDS, "re": __import__("re")}
    exec(compile(src[i:j], "train_curriculum.py:slice", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def grade():
    """grade_by_coverage, without importing the whole training module.

    train_curriculum imports torch and the anthropic client at module scope;
    this test only needs one pure function out of it.
    """
    return _slice_from_train_curriculum()["grade_by_coverage"]


class TestGrading:

    @pytest.mark.parametrize("level", ONTO_LEVELS)
    def test_perfect_answer_is_not_downgraded(self, grade, level):
        """The anti-echo at L4+ scores only the content words ABSENT from the
        prompt. At L11 and L12 the is-a steps put every content word in the
        prompt, which is the case that used to grade a perfect answer '=' and
        apply feedback 0.0 — no weight update at all for a correct reply."""
        for sname, _step, prompt, t in all_targets(level):
            exp = t["expected"]
            sym, cov, cmt = grade("+++", exp, exp, prompt, level)
            assert sym == "+++", (level, sname, prompt, exp, sym, cmt)
            assert cov >= 0.99

    @pytest.mark.parametrize("level", ONTO_LEVELS)
    def test_prompt_echo_never_earns_echo_training(self, grade, level):
        """SIGNAL 4 reinforces the model's OWN output only when coverage
        reaches 0.99. A prompt echo must stay under it, or the level trains
        the model on its own parroting."""
        for sname, _step, prompt, t in all_targets(level):
            _sym, cov, _cmt = grade("+++", t["expected"], prompt, prompt, level)
            assert cov < 0.99, (level, sname, prompt, cov)

    @pytest.mark.parametrize("level", ONTO_LEVELS)
    def test_grammar_only_garbage_gets_nothing(self, grade, level):
        for _sname, _step, prompt, t in all_targets(level):
            sym, cov, _cmt = grade("+++", t["expected"],
                                   "il è un e la di che non", prompt, level)
            assert sym == "=" and cov == 0.0


# ── the measurement's own classifier ───────────────────────────────────────

class TestCuriosityMetric:

    def test_question_detection(self):
        from curiosity_rate import is_question
        for asked in ("cos è un ragno?", "cosa è un ragno", "<|ask|>il ragno",
                      "che cos è?"):
            assert is_question(asked), asked
        for stated in ("il ragno è un animale.", "physisml!", "il il il",
                       "", "un animale è un essere vivente."):
            assert not is_question(stated), stated

    def test_class_extraction(self, lex):
        from curiosity_rate import stated_class
        classes = list(lex.classes)
        assert stated_class("il ragno è un animale.", classes) == "un animale"
        assert stated_class("un animale è un essere vivente.",
                            classes) == "un essere vivente"
        assert stated_class("cos è un ragno?", classes) == ""

    def test_probe_prompts_cover_both_halves(self, lex):
        from curiosity_rate import build_probes
        probes = build_probes(lex, n_known=8)
        kinds = {p["kind"] for p in probes}
        assert {"ignoto", "noto"} <= kinds
        assert any(p["should_ask"] for p in probes)
        assert any(not p["should_ask"] for p in probes)


# ── the curiosity mechanism ────────────────────────────────────────────────

class TestRewardAsymmetry:
    """The teacher's reward, not the gate, is what makes asking a policy."""

    def fresh(self):
        from dynamic_model.local_teacher import LocalTeacher
        return LocalTeacher(lang="it", level=12)

    def ask(self, teacher, noun, article="un"):
        return teacher.turn(last_prompt="x",
                            last_response=f"cosa è {article} {noun}?", turn=2)

    def test_curiosity_is_off_unless_the_level_asks_for_it(self):
        from dynamic_model.local_teacher import LocalTeacher
        assert not LocalTeacher(lang="it", level=11).curiosity_enabled
        assert self.fresh().curiosity_enabled

    def test_first_question_on_an_unknown_name_is_rewarded_and_answered(self):
        t = self.fresh()
        r = self.ask(t, "ragno")
        assert r["feedback"] == 1.0
        assert r["mode"] == "curiosity_answer"
        # asking must BUY something: the answer becomes the next prompt
        assert "ragno" in r["next_prompt"] and "animale" in r["next_prompt"]
        assert r["expected"] == "il ragno è un animale."

    def test_asking_twice_is_penalised(self):
        t = self.fresh()
        self.ask(t, "ragno")
        r = self.ask(t, "ragno")
        assert r["feedback"] < 0
        assert r["mode"] == "curiosity_repeat"
        # the retry uses the definite form L11 step A trained on
        assert r["next_prompt"] == "cosa è il ragno?"

    def test_asking_about_a_known_name_is_penalised(self):
        """The discriminating case. A model that asks about everything scores a
        perfect ask-rate on the unknown set and has learned nothing."""
        r = self.ask(self.fresh(), "gatto")
        assert r["feedback"] < 0
        assert r["mode"] == "curiosity_repeat"

    def test_probe_names_are_never_rewarded(self):
        """They are the measurement. Teaching or rewarding them turns the
        curiosity metric into a recall test."""
        for noun in ("lumaca", "aquila", "spazzola", "pettine", "noce",
                     "collina"):
            r = self.ask(self.fresh(), noun, article="una")
            assert r.get("mode") is None, (noun, r)

    def test_a_statement_is_not_treated_as_a_question(self):
        t = self.fresh()
        r = t.turn(last_prompt="x", last_response="il ragno è un animale.",
                   turn=2)
        assert r.get("mode") is None

    def test_no_question_branch_on_the_first_turn(self):
        """Turn 1 has no response to grade; the branch must not fire on it."""
        t = self.fresh()
        r = t.turn(last_prompt="", last_response="cosa è un ragno?", turn=1)
        assert r.get("mode") is None


class TestAffectSignals:

    def state(self):
        from dynamic_model.exp_b.affect_state import AffectState
        return AffectState()

    # The demonstrative frame is taught from L2 on and appears in every L11/L12
    # prompt, so a state that has not seen it reads 'questo' as untaught and
    # every prompt comes out curious. Included in the baseline for that reason.
    FRAME = ("questo è un gatto", "il gatto è un animale.",
             "il cane è un animale.", "una cosa", "il fiore è una pianta.")

    def taught(self, *golds):
        """A state shown the frame plus `golds`, with the content filter on."""
        s = self.state()
        s.function_words = STOP_WORDS
        for g in self.FRAME + golds:
            s.register_rewarded_words(g, feedback=0.8)
        return s

    def test_word_ignorance_discriminates_known_from_untaught(self):
        s = self.taught("il gatto è un animale.", "il cane è un animale.")
        assert s.word_ignorance("questo è un gatto") == 0.0
        assert s.word_ignorance("questo è un ragno") > 0.0
        assert s.word_ignorance("") == 0.0
        assert s.word_ignorance("il un di e") == 0.0, \
            "solo parole funzione: niente di contenuto, niente ignoranza"

    def test_the_signal_does_not_depend_on_tokenization(self):
        """The token-id version scored the taught unknown 'ragno' (one token)
        at 0.20 and the held-out probe 'lumaca' (lu|ma|ca) at 0.43, so the
        ORDER was set by the vocabulary rather than by knowledge. Word forms
        must put every untaught noun on the same footing.
        """
        s = self.taught("il gatto è un animale.", "il cane è un animale.")
        base = "il cane è un animale, questo è un "
        vals = {n: s.word_ignorance(base + n)
                for n in ("ragno", "lumaca", "collina", "aquila")}
        assert len(set(vals.values())) == 1, vals
        assert s.word_ignorance(base + "gatto") == 0.0

    def test_ask_drive_is_presence_not_proportion(self):
        """word_ignorance for one untaught noun in a real L12 prompt is 0.25,
        under the gate's 0.5 threshold, and moves with prompt length. The drive
        the gate receives must answer 'is there anything new here?'.
        """
        s = self.taught("il gatto è un animale.", "il cane è un animale.")
        s.fear = 0.05
        unknown = "il cane è un animale, questo è un ragno"
        assert s.word_ignorance(unknown) < 0.5      # the fraction cannot pass
        assert s.ask_drive(unknown) > 0.5           # the drive does
        assert s.ask_drive("il cane è un animale, questo è un gatto") == 0.0

    def test_explaining_a_word_silences_the_drive(self):
        """The whole point of the asymmetry: after the teacher answers, asking
        again must stop being pushed by the gate as well as penalised.
        """
        s = self.taught("il gatto è un animale.")
        s.fear = 0.05
        p = "questo è un ragno"
        assert s.ask_drive(p) > 0.5
        s.register_rewarded_words("il ragno è un animale.", feedback=1.0)
        assert s.ask_drive(p) == 0.0

    def test_fear_keeps_a_punished_model_quiet(self):
        s = self.taught("il gatto è un animale.")
        p = "questo è un ragno"
        s.fear = 0.05
        calm = s.ask_drive(p)
        s.fear = 0.9
        assert s.ask_drive(p) < calm
        assert s.ask_drive(p) <= 0.5, "un modello spaventato non deve chiedere"

    def test_the_memory_survives_a_new_process(self, tmp_path):
        """Every build phase is a separate python process. Without persistence
        the set is empty at every session start, word_ignorance reads 1.0 for
        everything and the gate fires on every prompt — 'asks about
        everything', which is the failure this level must not have.
        """
        s = self.taught("il gatto è un animale.", "il cane è un animale.")
        f = str(tmp_path / "affect_memory.json")
        s.save_memory(f)
        fresh = self.state()
        fresh.function_words = STOP_WORDS
        n = fresh.load_memory(f)
        assert n >= 4
        assert fresh.word_ignorance("questo è un gatto") == 0.0
        assert fresh.word_ignorance("questo è un ragno") > 0.0

    def test_a_missing_memory_file_is_not_an_error(self):
        s = self.state()
        assert s.load_memory("nessun/file/qui.json") == 0
        assert s.load_memory("") == 0

    def test_dormant_slots_do_not_count_as_ignorance(self):
        """The bug this replaced: norms over the WHOLE embedding matrix
        measured allocated capacity, not knowledge — a constant 0.717 with
        2547 of 9000 slots in use, permanently above the 0.7 gate threshold."""
        import torch
        s = self.state()
        w = torch.zeros(9000, 8)
        w[:2547] = 1.0                              # active rows trained
        s.ignorance = 0.5
        s.update_from_weights(w, active_vocab_size=2547)
        assert s.ignorance < 0.5, "gli slot dormienti sono ancora contati"
        s2 = self.state()
        s2.ignorance = 0.5
        s2.update_from_weights(w)                   # old behaviour, no cap
        assert s2.ignorance > 0.5

    def test_peek_entropy_leaves_the_state_alone(self):
        import torch
        s = self.state()
        before = s.confidence
        flat = s.peek_entropy(torch.zeros(100), 100)
        peaked = s.peek_entropy(
            torch.tensor([20.0] + [0.0] * 99), 100)
        assert s.confidence == before, "peek_entropy ha toccato l'EWMA"
        assert flat > 0.99 and peaked < 0.1

    def test_fear_suppresses_curiosity(self):
        s = self.state()
        s.ignorance, s.fear = 0.8, 0.0
        calm = s.curiosity
        s.fear = 0.9
        assert s.curiosity < calm
        assert 0.0 <= s.curiosity <= 1.0


class TestAskGate:

    def build(self, **kw):
        import torch
        from dynamic_model.exp_b.affect_state import AffectState
        from dynamic_model.exp_b.modulator import AffectModulator
        st = AffectState()
        st.pleasure, st.pain = 0.0, 0.0     # keep the other gates quiet
        return st, AffectModulator(st, ask_token_id=7, **kw), torch.zeros(50)

    def test_gate_is_off_by_default(self):
        st, mod, logits = self.build()
        assert mod.ask_gate is False
        out = mod.modulate(logits, response_len=0, curiosity=1.0)
        assert out[7].item() == out[6].item()

    def test_gate_boosts_only_the_first_token(self):
        st, mod, logits = self.build(ask_gate=True)
        first = mod.modulate(logits, response_len=0, curiosity=1.0)
        mid = mod.modulate(logits, response_len=3, curiosity=1.0)
        assert first[7].item() > first[6].item(), "nessun boost al primo token"
        assert mid[7].item() == mid[6].item(), "boost a metà frase"

    def test_gate_stays_quiet_when_the_word_is_known(self):
        st, mod, logits = self.build(ask_gate=True)
        out = mod.modulate(logits, response_len=0, curiosity=0.0)
        assert out[7].item() == out[6].item()

    def test_gate_can_win_from_a_deep_deficit(self):
        """The reason the boost is relative and not a constant: on the L3
        checkpoint the ask token sat 8.3 logits behind the argmax, where a
        fixed +4 changed nothing."""
        import torch
        st, mod, _ = self.build(ask_gate=True)
        logits = torch.zeros(50)
        logits[3] = 9.0            # a confident, unrelated best token
        logits[7] = -0.4           # the ask token, far behind
        out = mod.modulate(logits, response_len=0, curiosity=1.0)
        assert out.argmax().item() == 7, "il gate non vince nemmeno a curiosità 1"

    def test_gate_is_only_plausible_near_the_threshold(self):
        """Just above the threshold the gate must make asking possible under
        sampling, not force it under greedy decoding."""
        import torch
        st, mod, _ = self.build(ask_gate=True)
        logits = torch.zeros(50)
        logits[3] = 9.0
        logits[7] = -0.4
        out = mod.modulate(logits, response_len=0,
                           curiosity=mod.ASK_THRESHOLD + 0.01)
        assert out.argmax().item() == 3
        assert out[7].item() > logits[7].item() * 10, "nessuna spinta"


def test_the_two_corpus_generators_agree():
    """qa_corpus.txt is written by two paths — scripts/generate_qa_corpus.py
    and _update_qa_pairs_from_sessions in the dream — and both are committed
    output. When their demo-prefix regexes drifted apart, every L0/L1 corpus
    disagreed between them: --check called them stale forever, and a clone
    trained on a different file than the build had written.
    """
    import re as _re
    import generate_qa_corpus as G
    src = open("dynamic_model/train_curriculum.py", encoding="utf-8").read()
    m = _re.search(r"_DEMO_RE = _re\.compile\(r'(.+)'\)", src)
    assert m, "_DEMO_RE non trovato in train_curriculum.py"
    assert G.DEMO_RE.pattern == m.group(1)

class TestVocabularyCoversTheNewLevels:
    """The vocabulary is what makes L11/L12 teachable at all.

    Everything here failed under the 2026-08-25 vocabulary and is the reason it
    had to be retrained before this build, not after.
    """

    SEED = "dynamic_model/data/tokenizer_8k.json"

    def seed(self):
        from splx.tokenizer import BPETokenizer
        tok = BPETokenizer()
        tok.load(self.SEED)
        return tok

    def test_the_gate_has_a_whole_word_anchor(self):
        """ask_token_id must find a real token, not a fragment.

        With 'cos' split as 'co'+'s' the gate would raise 'co', which opens
        'come' — the whole identity step of L1 — as well as 'con', 'corre',
        'colore'. That is not a nudge toward a question, it is noise injected
        at the first token of every answer, at the one level whose measurement
        is 'did the model ask?'.
        """
        from dynamic_model.exp_b.modulator import ASK_FORM, ask_token_id
        tok = self.seed()
        aid = ask_token_id(tok)
        assert aid is not None, \
            f"{ASK_FORM!r} segments as " \
            f"{[tok.decode([i]) for i in tok.encode(ASK_FORM)]}"
        assert tok.decode([aid]) == ASK_FORM.split()[0]

    def test_ask_token_id_refuses_a_fragment(self):
        """The guard, on the vocabulary that actually had the problem."""
        import os
        from splx.tokenizer import BPETokenizer
        from dynamic_model.exp_b.modulator import ask_token_id
        old = "dynamic_model/data/tokenizer_8k.pre_l11l12.json"
        if not os.path.exists(old):
            pytest.skip("no pre-L11/L12 vocabulary kept")
        tok = BPETokenizer(); tok.load(old)
        assert ask_token_id(tok) is None

    def test_the_class_words_are_single_tokens(self):
        """The class name is the informative word of every L11 answer: split
        into pieces it competes for probability with every other word sharing
        those pieces."""
        tok = self.seed()
        for w in ("animale", "persona", "cibo", "oggetto", "luogo", "cosa",
                  "pianta", "vivente", "essere"):
            assert len(tok.encode(w)) == 1, \
                f"{w!r} -> {[tok.decode([i]) for i in tok.encode(w)]}"

    def test_the_teachable_unknown_nouns_are_single_tokens(self):
        lex = E.load_lexicon("it")
        tok = self.seed()
        teachable = [e["w"] for e in lex.get("unknown_nouns", [])
                     if not e.get("probe")]
        assert teachable, "no teachable unknown nouns in the lexicon"
        for n in teachable:
            assert len(tok.encode(n)) == 1, \
                f"{n!r} -> {[tok.decode([i]) for i in tok.encode(n)]}"

    def test_the_probe_nouns_stay_out_of_the_vocabulary_pool(self):
        """The held-out half must not be in any target pool, or the vocabulary
        trainer sees it (it feeds on the pools) and the measurement set stops
        being unknown."""
        lex = E.load_lexicon("it")
        probes = [e["w"] for e in lex.get("unknown_nouns", [])
                  if e.get("probe")]
        assert probes, "no probe nouns in the lexicon"
        pooled = set()
        for level in range(13):
            path = f"training_files/it/{level}/local_teacher.json"
            if not os.path.exists(path):
                continue
            blob = json.dumps(json.load(open(path, encoding="utf-8")),
                              ensure_ascii=False).lower()
            for n in probes:
                if n in blob:
                    pooled.add((level, n))
        assert not pooled, f"probe nouns found in the pools: {sorted(pooled)}"
