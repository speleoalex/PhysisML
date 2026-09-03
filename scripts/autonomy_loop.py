#!/usr/bin/env python3
"""
autonomy_loop — the six-step cycle, closed, running unattended.

    answer -> judge own ignorance -> ask -> get an answer -> learn -> dream

Every stage but two already existed in this repository; this driver is the
wiring, plus the two missing pieces: the oracle that answers (in
dynamic_model/ontology_oracle.py) and the degradation trigger that decides when
to dream (here).

WHAT IT DOES, per interaction
  1. picks an unknown noun and asks about it in a shape the curriculum taught:
     the bare 'cos è un X?' or level 12's two-clause form
  2. generates greedily, with the same settings the evaluation harness uses
     (top_k=1 — 'temperature 0' alone samples in the modulated path)
  3. classifies the answer: asked / admitted ignorance / asserted a class
  4. on an honest answer, asks the oracle for the class and renders material
     from the curriculum's own templates
  5. accepts or refuses that material, then learns it with TrainerB plus
     interleaved gold rehearsal — never a bare SFT step
  6. every --probe-every interactions, scores the frozen probe; on degradation
     it closes the batch and dreams (phase_2_dream, unmodified)

THE UNIT OF WORK IS THE BATCH, not the level. Levels are integer arithmetic in
this codebase (the dream replays range(level+1), the parent checkpoint is
level-1), so acquisitions accumulate INSIDE level 13 as numbered batches:

    training_files/it/13/batches/0001/targets.jsonl   what was acquired
    models/checkpoints/it/level_13/batches/0001/      the state it produced
    training_files/it/13/local_teacher.json           rebuilt from all batches

Displayed as 13.1, 13.2 — that is the right mental model. The batch is what
gets rolled back when the probe says the acquisition cost more than it added.

DEFAULT IS CURIOSITY-DRIVEN: material is acquired only for turns where the
model was honest about not knowing. That is the experiment — the ask produces
the material. --teach-confabulations widens it to every turn and turns the loop
into ordinary supervision; the rate of each is logged either way.

Usage:
    python3 scripts/autonomy_loop.py --dry-run --interactions 20
    python3 scripts/autonomy_loop.py --interactions 50 --probe-every 10
    python3 scripts/autonomy_loop.py --queue my_nouns.json --interactions 200
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expand_teacher_pools as etp                                    # noqa: E402
import probe_set                                                       # noqa: E402
from curiosity_rate import is_question, says_dont_know                 # noqa: E402
from measure_repetition import load_pair, resolve                      # noqa: E402
from dynamic_model.ontology_oracle import (OntologyOracle, article_for,  # noqa: E402
                                            guess_gender as etp_guess_gender)
from dynamic_model.test_model import is_exact                          # noqa: E402
from dynamic_model.train_curriculum import (turn_record, _known_golds,  # noqa: E402
                                            _normalize_prompt, _EOS_MARK)
from dynamic_model import retraction                              # noqa: E402
from dynamic_model.stop_words import STOP_WORDS                   # noqa: E402
from dynamic_model.train_curriculum import _affect_memory_path    # noqa: E402

ASKED, ADMITTED, ASSERTED = "asked", "admitted", "asserted"
# A fourth outcome, and it exists because the dry run produced it: asked to
# introduce 'tegola', the model replied 'cos è una zucca?' — the SHAPE of the
# question generalised to a name it had never met, the referent did not. Left
# in the 'asked' bucket that would have credited the model with ignorance of
# 'tegola' and acquired material on the strength of it. A question about
# another noun is evidence of nothing about this one.
ASKED_OTHER = "asked_other"


# ── material acceptance ───────────────────────────────────────────────────────
class Gatekeeper:
    """
    Decides whether an acquisition may enter the curriculum — ALL of it or none.

    Every rule here is a lesson already paid for in this repo, which is why
    none of them is optional:

      * one gold per prompt, checked against EVERY level's material — the
        harvest is exactly the channel that reintroduced contradictory
        supervision twice before;
      * nothing that appears in the frozen probe, or the degradation trigger
        measures the model on what the loop just taught it;
      * a cap per batch, mirroring the dream's own max_new.

    ATOMIC, per noun, and that is not fussiness. Level 12 teaches
    'cos è un falco?' -> 'non lo so.' while the oracle answers
    'il falco è un animale.'; accepting the phrasings that do NOT collide
    ('cosa è un falco?') and dropping the one that does would teach the two
    answers under two shapes of the same question — which is precisely the
    failure ask_forms exists to prevent ('cosa è un cane?' answered 'non lo
    so.' because the article, not the noun, had become the feature).

    A collision with an ADMISSION of ignorance is reported apart from a
    collision with a fact: the first is stale supervision that learning has
    made false and could legitimately be retracted, the second is the
    curriculum overruling the oracle. The loop does not retract anything on
    its own — retracting reaches into 1480 lines of L12's corpus, and that is
    a decision, not a side effect.
    """

    ADMISSIONS = retraction.ADMISSIONS

    def __init__(self, lang: str, probe: dict, max_new: int):
        self.known    = _known_golds(lang)
        self.probe_p  = {_normalize_prompt(i["prompt"]) for i in probe["items"]}
        self.max_new  = max_new
        self.accepted = 0
        self.refusals = {}

    def _refuse(self, why: str, n: int = 1) -> None:
        self.refusals[why] = self.refusals.get(why, 0) + n

    def _is_admission(self, gold: str) -> bool:
        return retraction.is_admission(gold)

    def retractable(self, word: str, lang: str) -> str:
        """'' if the admissions about `word` may be removed, else why not.

        The frozen probe is the reason this check exists separately from
        `inspect`: inspect refuses material that lands ON a probe prompt, but a
        retraction removes DIFFERENT prompts — the admissions — and one of
        those could itself be a probe item. Removing it would make the
        degradation trigger measure a lesson the loop had just deleted, which
        is the self-confirming measurement the frozen probe exists to prevent.
        """
        hit = retraction.find(word, lang)
        if hit["other"]:
            return (f"{len(hit['other'])} golds on '{word}' are not admissions")
        for part in hit["levels"].values():
            for item in part["targets"]:
                if _normalize_prompt(item["target"]["prompt"]) in self.probe_p:
                    return f"'{item['target']['prompt']}' is in the frozen probe"
            for item in part["pairs"]:
                rec = json.loads(item["line"])
                if _normalize_prompt(rec["prompt"]) in self.probe_p:
                    return f"'{rec['prompt']}' is in the frozen probe"
        return ""

    def forget(self, word: str) -> int:
        """Drop every gold about `word` from the in-memory index.

        Called after a retraction has removed them from disk. Without this the
        next phrasing of the same noun still collides with a gold that no
        longer exists, and the acquisition is refused for a reason that is no
        longer true.
        """
        pat = re.compile(rf"\b{re.escape(word)}\b")
        dead = [p for p in self.known if pat.search(p)]
        for p in dead:
            del self.known[p]
        return len(dead)

    def inspect(self, material: list) -> dict:
        """What would happen, without changing anything."""
        stale, conflict, dup, in_probe = [], [], 0, 0
        for t in material:
            p    = _normalize_prompt(t["prompt"])
            gold = t["expected"].strip()
            if p in self.probe_p:
                in_probe += 1
                continue
            held = self.known.get(p)
            if held is None:
                continue
            if held == gold:
                dup += 1
            elif self._is_admission(held):
                stale.append((t["prompt"], held))
            else:
                conflict.append((t["prompt"], held))
        room = self.accepted + len(material) - dup <= self.max_new
        ok = not (stale or conflict or in_probe) and room
        return {"ok": ok, "stale": stale, "conflict": conflict,
                "in_probe": in_probe, "duplicate": dup, "room": room}

    def reason(self, verdict: dict) -> str:
        # The cap comes first: with the batch full, no amount of retracting
        # makes the acquisition fit, so naming the stale admissions here would
        # invite the exact retract-then-refuse sequence that orphaned ten
        # nouns on the first real run (2026-09-03).
        if not verdict["room"]:
            return "batch cap reached"
        if verdict["conflict"]:
            return (f"{len(verdict['conflict'])} prompts already have a "
                    f"different gold in the curriculum")
        if verdict["stale"]:
            return (f"{len(verdict['stale'])} 'non lo so' admissions have to "
                    f"be retracted first (--retract to do it)")
        if verdict["in_probe"]:
            return f"{verdict['in_probe']} prompts are in the frozen probe"
        return "batch cap reached"

    def commit(self, material: list) -> list:
        """Accept a whole acquisition. Only ever called after inspect().ok."""
        out = []
        for t in material:
            p = _normalize_prompt(t["prompt"])
            if self.known.get(p) == t["expected"].strip():
                continue                     # already there, nothing to add
            self.known[p] = t["expected"].strip()
            self.accepted += 1
            out.append(t)
        return out


# ── the batch ─────────────────────────────────────────────────────────────────
class Batch:
    """One acquisition: its material, its provenance, its rollback point."""

    def __init__(self, lang: str, level: int, ckpt_base: str, parent_ckpt: str):
        self.lang, self.level = lang, level
        self.data_dir = os.path.join(_ROOT, "training_files", lang, str(level),
                                     "batches")
        self.ckpt_dir = os.path.join(ckpt_base, f"level_{level}", "batches")
        self.id       = self._next_id()
        self.parent   = parent_ckpt
        self.targets  = []
        self.nouns    = []
        self.retracted = []

    def _next_id(self) -> int:
        ids = [int(os.path.basename(d)) for d in glob.glob(
               os.path.join(self.data_dir, "[0-9]" * 4))
               if os.path.basename(d).isdigit()]
        return (max(ids) + 1) if ids else 1

    @property
    def name(self) -> str:
        """'13.1' — how a batch is spoken about. The directory is zero-padded:
        alphabetical order puts 13.10 before 13.2, and that bug surfaces at the
        tenth acquisition."""
        return f"{self.level}.{self.id}"

    @property
    def slug(self) -> str:
        return f"{self.id:04d}"

    def add(self, noun: dict, cls: str, targets: list, source: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.nouns.append({"w": noun["w"], "cls": cls})
        for i, t in enumerate(targets):
            rec = dict(t)
            rec.update({"step": "A" if i < len(targets) - 2 else
                                ("B" if i == len(targets) - 2 else "C"),
                        "batch": self.slug, "source": source, "acquired": stamp,
                        "class": cls, "word": noun["w"]})
            self.targets.append(rec)

    def retract(self, word: str, source: str) -> dict:
        """Retract the admission of ignorance about `word`, into THIS batch.

        The save file goes in the batch directory, so undoing the batch undoes
        the retraction too: without that, a rolled-back acquisition would leave
        the curriculum permanently missing the lesson it was refused for.
        """
        d = os.path.join(self.data_dir, self.slug)
        os.makedirs(d, exist_ok=True)
        info = retraction.retract(word, self.lang, save_dir=d,
                                  batch=self.slug, source=source)
        if info["targets"] or info["pairs"]:
            self.retracted.append(info)
        return info

    def write(self) -> str:
        """Persist the batch and rebuild the level config from every batch."""
        d = os.path.join(self.data_dir, self.slug)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "targets.jsonl"), "w", encoding="utf-8") as f:
            for t in self.targets:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        if self.retracted:
            with open(os.path.join(d, "retracted.json"), "w",
                      encoding="utf-8") as f:
                json.dump(self.retracted, f, ensure_ascii=False, indent=1)
        rebuild_config(self.lang, self.level)
        return d

    def snapshot(self, dreamed: str) -> None:
        """Freeze the state this batch produced, for rollback.

        Hard-linked, not copied: the file is 94MB and identical to
        final_dreamed.pt until the next batch overwrites that name.
        """
        os.makedirs(os.path.join(self.ckpt_dir, self.slug), exist_ok=True)
        dst = os.path.join(self.ckpt_dir, self.slug, "dreamed.pt")
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(dreamed, dst)
        except OSError:
            shutil.copy2(dreamed, dst)

    def discard(self, dreamed: str) -> None:
        """Undo this batch: its material leaves the curriculum, whatever it
        retracted comes back, and its weights return to the state it started
        from.

        Restore BEFORE the directory is moved aside — the save files live in
        it, and a rollback that could not find them would leave the curriculum
        with neither the acquisition nor the admission it replaced.
        """
        d = os.path.join(self.data_dir, self.slug)
        for info in self.retracted:
            try:
                retraction.restore(info["word"], self.lang, save_dir=d)
            except (FileNotFoundError, KeyError) as e:
                print(f"  WARNING: the retraction of {info['word']} was not "
                      f"restored ({e}). The level that taught it now teaches "
                      f"neither the admission nor the class.")
        if os.path.isdir(d):
            shutil.move(d, d + f".rejected.{int(time.time())}")
        rebuild_config(self.lang, self.level)
        if self.parent and os.path.exists(self.parent):
            shutil.copy2(self.parent, dreamed)


def rebuild_config(lang: str, level: int) -> int:
    """
    Write local_teacher.json's targets from every batch on disk.

    The config is a projection of the batches, never an independent source:
    that is what makes removing a batch directory a complete rollback of its
    material. Only the curriculum keys are copied over — provenance stays in
    the batch file, where a schema check on the config cannot trip over it.
    """
    base = os.path.join(_ROOT, "training_files", lang, str(level))
    cfg_path = os.path.join(base, "local_teacher.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for step in cfg.get("steps", {}).values():
        step["targets"] = []
    n = 0
    for d in sorted(glob.glob(os.path.join(base, "batches", "[0-9]" * 4))):
        path = os.path.join(d, "targets.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                step = cfg["steps"].get(t.get("step", "A"))
                if step is None:
                    continue
                step["targets"].append({k: v for k, v in t.items()
                                        if k in ("prompt", "expected",
                                                 "article", "noun")})
                n += 1
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    return n


# ── the session log ───────────────────────────────────────────────────────────
class SessionLog:
    """
    Writes the level's session_*.jsonl in the format the dream already reads.

    One subtlety, and it is not cosmetic: `feedback` on row N is the grade of
    row N-1's response, and graded_prompt/graded_expected/graded_response say
    which exchange that was. The dream's harvest reads the grade for row i out
    of row i+1, so a log written the other way round pairs every grade with the
    wrong question — and the pairs it then writes into the curriculum are the
    wrong pairs. Row 1 has nothing graded yet and carries None, exactly as
    phase 1 does; the last exchange's grade is therefore absent from the file,
    also as in phase 1, and the harvest treats it as neutral.
    """

    def __init__(self, ckpt_dir: str, enabled: bool = True):
        self.path = os.path.join(
            ckpt_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.enabled = enabled
        self.turn    = 0
        self.prev    = None        # the exchange the next row's grade refers to
        if enabled:
            os.makedirs(ckpt_dir, exist_ok=True)

    def add(self, step: str, prompt: str, expected: str, response: str,
            symbol: str, comment: str, affect: dict) -> None:
        self.turn += 1
        prev = self.prev
        if self.enabled:
            rec = turn_record(
                turn=self.turn, step=step, prompt=prompt, expected=expected,
                response=response,
                feedback=prev["symbol"] if prev else None,
                comment=comment,
                graded_prompt=prev["prompt"] if prev else None,
                graded_expected=prev["expected"] if prev else None,
                graded_response=prev["response"] if prev else None,
                content="", affect=affect, stats={})
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.prev = {"prompt": prompt, "expected": expected,
                     "response": response, "symbol": symbol}

    def close(self) -> None:
        """Nothing buffered: rows are written as they happen."""
        self.prev = None


# ── the loop ──────────────────────────────────────────────────────────────────
def affect_dict(tr) -> dict:
    """The four affect values as JSON, the same shape phase 1 logs.

    Deliberately NOT AffectState.snapshot(): that returns a dataclass json
    cannot serialise, and it also appends to the history and advances the step
    counter — a logging call must not move the state it reports.
    """
    a = tr.affect
    return {"confidence": round(a.confidence, 3), "fear": round(a.fear, 3),
            "pleasure": round(a.pleasure, 3), "pain": round(a.pain, 3)}


def classify(answer: str, target: str = "") -> str:
    """The outcome of one turn, from the model's own words.

    `target` is the noun the turn is about: a question that does not name it
    is ASKED_OTHER, not ASKED. See the note on ASKED_OTHER.
    """
    if is_question(answer):
        if not target or target.lower() in answer.lower():
            return ASKED
        return ASKED_OTHER
    if says_dont_know(answer):
        return ADMITTED
    return ASSERTED


def ask_shapes(noun: dict, lex, rng) -> list:
    """
    The two shapes the curriculum taught for meeting a new name:

      * the bare open question, whose taught answer is 'non lo so.' (L12 E)
      * the two-clause introduction, whose taught answer IS a question (L12 A)

    Both are legitimate; asking only one would measure the shape instead of the
    knowledge, which is the mistake ask_forms exists to prevent.
    """
    out = [{"prompt": f"cos è {etp.indef(noun)} {noun['w']}?",
            "expected": "non lo so.", "step": "E"}]
    classified = lex.classified()
    anchor = rng.choice(classified) if classified else None
    if anchor:
        acls = lex.cls_of(anchor)
        out.append({"prompt": f"{etp.phrase(anchor)} è {acls}, {etp.intro(noun)}",
                    "expected": f"cos è {etp.indef(noun)} {noun['w']}?",
                    "step": "A"})
    return out


def build_queue(args, lex, probe, oracle) -> list:
    """
    The nouns to ask about: from --queue, else the lexicon's ACQUIRABLE bare
    unknowns.

    Excluded, always — and every exclusion is a measurement that would
    otherwise destroy itself:

      * anything the lexicon already classifies: the curriculum's answer has
        authority over the oracle's;
      * the probe half of `unknown_nouns`, which scripts/curiosity_rate.py
        measures the curiosity gap on;
      * `bare_unknown_nouns` with role 'reserve': the permanent control that
        acquisition has not eroded honesty. Acquiring one would spend the
        control to gain one noun;
      * role 'probe': taught nowhere, and the only way left to ask whether the
        honesty RELATION generalised. Teaching it answers the question by
        removing it.

    An explicit --queue is filtered the same way. A queue file is a
    convenience, not an override: the reserve exists so that no run, however
    it was launched, can quietly consume it.
    """
    reserved = {n["w"] for n in lex.unknown_of(probe=True)}
    reserved |= {n["w"] for n in lex.bare_reserve}
    reserved |= {n["w"] for n in lex.bare_probe}
    classified = {n["w"] for n in lex.nouns if lex.cls_of(n)}
    raw = []
    if args.queue:
        with open(args.queue, encoding="utf-8") as f:
            if args.queue.endswith(".json"):
                raw = json.load(f)
            else:
                raw = [ln.strip() for ln in f if ln.strip()
                       and not ln.startswith("#")]
    else:
        raw = [n for n in lex.bare_taught
               if n.get("role", "acquirable") == "acquirable"]
    out, seen = [], set()
    for item in raw:
        n = dict(item) if isinstance(item, dict) else {"w": str(item).strip()}
        w = n.get("w", "").lower()
        if not w or w in seen or w in reserved or w in classified:
            continue
        seen.add(w)
        # Gender and article are needed to BUILD the question, which happens
        # before the oracle is consulted — a queue of bare words would other-
        # wise crash on the first prompt. The ending decides it where it can;
        # '-e' settles nothing ('il pettine' against 'la noce'), and only then
        # is the oracle asked. The article is always derived by code.
        n["w"] = w
        if not n.get("g"):
            n["g"] = etp_guess_gender(w) or oracle.gender_of(w)
        if not n.get("art"):
            n["art"] = article_for(w, n["g"])
        out.append(n)
    return out


def rearm_affect(tr, lang: str, ckpt_base: str, quiet: bool = False) -> int:
    """Give a freshly loaded model back its curiosity state.

    `load_pair` builds a bare `AffectState()`, and the two things that make the
    curiosity signal mean anything are not in the checkpoint:

      * `function_words` — without the content-word filter every article and
        preposition counts as a word the model was never taught;
      * `words_rewarded`, from affect_memory.json — the only thing that carries
        'this has been explained to me' across processes, since every build
        phase is a separate one.

    Missing both, `AffectState.untaught_words` answers 'unknown' to everything,
    so `word_ignorance` is 1.0 on every prompt and `ask_drive` fires always.
    scripts/curiosity_rate.py has done this since the measurement was written;
    the loop never did, and its `affect` column in every session_*.jsonl is
    meaningless as a result. It has not produced a wrong acquisition — nothing
    branches on it today — but it is exactly the column the epistemic trigger
    is meant to calibrate against, and a ground truth that says 'unknown' to
    everything cannot check anything.

    Called after EVERY load_pair, not just the first: a dream and a rollback
    both rebuild the trainer, and a state re-armed only at startup goes bare
    again at the first dream — which is precisely where a long run spends its
    time.
    """
    tr.affect.function_words = STOP_WORDS
    path = _affect_memory_path(lang, ckpt_base)
    n = tr.affect.load_memory(path)
    if not quiet:
        print(f"  curiosity memory: {n} words from {path}" if n else
              f"  ⚠ no curiosity memory in {path}: every prompt reads as "
              f"unknown and the log's 'affect' column says nothing")
    return n


def degraded(now: dict, baseline: dict, args) -> str:
    """Why the model needs to sleep, or ''.

    Two signals, because at level 6 both moved together across one dream
    (exact 23.7% -> 84.3%, self-repetition 11.1% -> 3.4%): exact match falling
    away from the post-dream baseline, and repetition climbing.
    """
    drop = baseline["exact_rate"] - now["exact_rate"]
    if drop > args.max_drop:
        return f"exact -{drop:.1%} (threshold {args.max_drop:.0%})"
    if now["repetition_rate"] > args.max_repetition:
        return (f"repetition {now['repetition_rate']:.1%} "
                f"(threshold {args.max_repetition:.0%})")
    return ""


def run_dream(args, level: int) -> bool:
    """phase_2_dream, unmodified, in its own process.

    A subprocess and not an import: the dream builds its own model, optimizer,
    affect state and tokenizer from disk, and sharing this process's loaded
    model with it would mean two owners of the same weights.
    """
    cmd = [sys.executable, "-m", "dynamic_model.train_curriculum",
           "--phase", "2", "--level", str(level), "--lang", args.lang,
           "--dream-mode", args.dream_mode]
    if args.ckpt_base:
        cmd += ["--ckpt-base", args.ckpt_base]
    print(f"  → {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=_ROOT).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Autonomous acquisition loop")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--level", type=int, default=13,
                    help="the autonomy level (integer: the dream does "
                         "arithmetic on it)")
    ap.add_argument("--interactions", type=int, default=20)
    ap.add_argument("--probe-every", type=int, default=10,
                    help="score the frozen probe every N interactions")
    ap.add_argument("--max-drop", type=float, default=0.05,
                    help="exact-match fall from the post-dream baseline that "
                         "triggers a dream (0.05 = 5 points)")
    ap.add_argument("--max-repetition", type=float, default=0.08)
    ap.add_argument("--max-new", type=int, default=40,
                    help="cap on accepted targets per batch")
    ap.add_argument("--rehearsal-every", type=int, default=5)
    ap.add_argument("--rehearsal-k", type=int, default=4)
    ap.add_argument("--dream-mode", default="standard",
                    choices=["light", "standard", "deep"])
    ap.add_argument("--queue", default=None,
                    help="JSON list of {w,art,g} or a text file, one noun per line")
    ap.add_argument("--ckpt-base", default=None)
    ap.add_argument("--probe", default=probe_set.DEFAULT_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retract", action="store_true",
                    help="when an acquisition collides with a curriculum-"
                         "taught 'non lo so', RETRACT the admission (only for "
                         "nouns with role=acquirable). Without this flag the "
                         "acquisition is refused, with the reason printed")
    ap.add_argument("--teach-confabulations", action="store_true",
                    help="also acquire material on turns where the model "
                         "asserted a class instead of admitting ignorance "
                         "(off by default: the ask is the experiment)")
    ap.add_argument("--no-dream", action="store_true",
                    help="detect degradation but do not consolidate")
    ap.add_argument("--dry-run", action="store_true",
                    help="ask and judge, touch no weights and write nothing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ckpt_base = args.ckpt_base or os.path.join("models", "checkpoints", args.lang)
    level_dir = os.path.join(ckpt_base, f"level_{args.level}")
    lex = etp.Lex(etp.load_lexicon(args.lang))

    # ── the three things that must exist before anything is touched ─────────
    probe = probe_set.load(args.probe)          # raises if missing or edited
    oracle = OntologyOracle(lang=args.lang)
    if not oracle.is_available():
        print(f"No local LLM reachable: {oracle.status()}\n"
              f"The oracle is step 4 of the cycle — without it the loop has "
              f"nothing to learn. Start llama-server or ollama.")
        sys.exit(1)

    ckpt, tok_path = resolve(ckpt_base, args.level)
    if not ckpt:
        ckpt, tok_path = resolve(ckpt_base, args.level - 1)
    if not ckpt or not tok_path:
        print(f"No starting checkpoint under {ckpt_base}.")
        sys.exit(1)

    queue = build_queue(args, lex, probe, oracle)
    if not queue:
        print("Empty queue: no unknown noun left to ask about.")
        sys.exit(1)

    print(f"level      : {args.level}  (the batch is the unit of acquisition)")
    print(f"checkpoint : {ckpt}")
    print(f"oracle     : {oracle.status()}")
    print(f"probe      : {probe['n']} prompts, fingerprint {probe['fingerprint']}")
    print(f"queue      : {len(queue)} nouns — "
          f"{', '.join(n['w'] for n in queue[:8])}"
          f"{' …' if len(queue) > 8 else ''}")
    print(f"mode       : {'DRY RUN (no weights, no writes)' if args.dry_run else 'learning'}")

    tr, tok = load_pair(ckpt, tok_path)
    rearm_affect(tr, args.lang, ckpt_base)
    eos = tok.EOS_TOKEN if hasattr(tok, "EOS_TOKEN") else ""
    if not (eos and hasattr(tok, "get_special_id")
            and tok.get_special_id(eos) is not None):
        eos = ""

    baseline = None
    if not args.dry_run:
        print("\nprobe baseline…", end=" ", flush=True)
        baseline = probe_set.score(tr, tok, probe)
        print(f"exact {baseline['exact_rate']:.1%}  "
              f"repetition {baseline['repetition_rate']:.1%}")

    gate  = Gatekeeper(args.lang, probe, args.max_new)
    batch = Batch(args.lang, args.level, ckpt_base,
                  os.path.join(level_dir, "final_dreamed.pt")
                  if os.path.exists(os.path.join(level_dir, "final_dreamed.pt"))
                  else ckpt)
    log   = SessionLog(level_dir, enabled=not args.dry_run)
    gold_bank = load_gold_bank(args.lang, args.level)
    print(f"rehearsal  : {len(gold_bank)} gold pairs (levels 0-{args.level})")
    print(f"batch      : {batch.name}\n")

    tally = {ASKED: 0, ADMITTED: 0, ASSERTED: 0, ASKED_OTHER: 0}
    history, learned_targets = [], 0

    def generate(prompt: str, expected: str = "") -> str:
        n_exp = len(tok.encode(expected)) if expected else 12
        out = tr.generate(prompt,
                          max_tokens=max(24, min(2 * n_exp + 12, 120)),
                          base_temperature=0.0, top_k=1,
                          min_tokens=max(4, min(n_exp, 40)),
                          stop_after=max(0, min(n_exp - 1, 40)))
        return out[len(prompt):].strip()

    def rehearse() -> int:
        """Replay gold already taught, so acquiring does not erase.

        Drawn from EVERY level, not just this one: phase 1 defaults to the
        current level's own bank, which at the autonomy level starts empty —
        the scope that protects nothing is not the scope to inherit here.
        """
        if args.dry_run or not gold_bank:
            return 0
        picks = rng.sample(list(gold_bank.items()),
                           min(args.rehearsal_k, len(gold_bank)))
        for p, r in picks:
            tr.step(p, r + eos, feedback=0.6)
        return len(picks)

    for i in range(1, args.interactions + 1):
        noun  = queue[(i - 1) % len(queue)]
        shapes = ask_shapes(noun, lex, rng)
        shape  = shapes[((i - 1) // len(queue)) % len(shapes)]
        answer = generate(shape["prompt"], shape["expected"])
        outcome = classify(answer, noun["w"])
        tally[outcome] += 1
        honest = outcome in (ASKED, ADMITTED)
        mark = {ASKED: "?", ADMITTED: "~", ASSERTED: "!",
                ASKED_OTHER: "¿"}[outcome]
        print(f"{i:4d} {mark} {shape['prompt']!r:52} -> {answer!r:34}", end="")

        log.add(step=shape["step"], prompt=shape["prompt"],
                expected=shape["expected"], response=answer,
                symbol="+++" if honest else "=",
                comment=f"outcome: {outcome}", affect=affect_dict(tr))

        if not (honest or args.teach_confabulations):
            print("   (asks about another noun: nothing acquired)"
                  if outcome == ASKED_OTHER else
                  "   (confabulates: nothing acquired)")
            continue

        verdict = oracle.ask(noun)
        if not verdict.ok:
            print(f"   oracle: refused — {verdict.reason}")
            continue

        material = oracle.material_for(verdict.noun, verdict.cls, rng)
        check    = gate.inspect(material)
        # A collision with a stale admission is the ONE refusal that learning
        # can legitimately clear, and only for a noun the lexicon marks
        # acquirable: 'reserve' is the control that acquisition has not eroded
        # honesty, so spending it to gain one noun is the one trade never worth
        # making. The retraction happens BEFORE the material is learned, and
        # inside the batch, so one rollback undoes both halves. It also
        # requires room in the batch: the cap used to be checked only by the
        # re-inspect AFTER the retraction, so once the batch filled up every
        # further noun was retracted, then refused, and left with neither an
        # admission nor a class gold (ten silent holes on the first real run,
        # 2026-09-03).
        if (not check["ok"] and check["stale"] and not check["conflict"]
                and not check["in_probe"] and check["room"] and args.retract
                and noun.get("role", "acquirable") == "acquirable"):
            veto = gate.retractable(noun["w"], args.lang)
            if veto:
                gate._refuse(f"retraction impossible: {veto}")
                print(f"   retraction refused: {veto}")
                continue
            if args.dry_run:
                # Say what it would remove, and remove it from the in-memory
                # index only: a dry run that could not report the retraction
                # would report every acquisition as refused and hide the whole
                # point of --retract.
                hit = retraction.find(noun["w"], args.lang)
                n_t = sum(len(x["targets"]) for x in hit["levels"].values())
                n_p = sum(len(x["pairs"]) for x in hit["levels"].values())
                print(f"   would retract {n_t} targets and {n_p} pairs on "
                      f"{noun['w']} (L{','.join(hit['levels'])})")
                gate.forget(noun["w"])
            else:
                try:
                    info = batch.retract(noun["w"],
                                        source=f"oracle:{oracle.model}")
                except ValueError as e:
                    gate._refuse(f"retraction impossible: {e}")
                    print(f"   retraction refused: {e}")
                    continue
                gate.forget(noun["w"])
                print(f"   retracted: {info['targets']} targets and "
                      f"{info['pairs']} 'non lo so' pairs on {noun['w']} "
                      f"(L{','.join(str(l) for l in info['levels'])})")
            check = gate.inspect(material)
        if not check["ok"]:
            why = gate.reason(check)
            gate._refuse(why)
            print(f"   oracle: {verdict.cls} — acquisition refused: {why}")
            continue
        accepted = gate.commit(material)
        print(f"   oracle: {verdict.cls}  ({len(accepted)} targets accepted)")
        if not accepted:
            continue

        if not args.dry_run:
            for t in accepted:
                tr.step(t["prompt"], t["expected"] + eos, feedback=1.0)
                learned_targets += 1
            # The response is recorded for provenance, not as evidence that the
            # material stuck: asking right after teaching measures recency.
            # The probe is the evidence.
            for t in accepted:
                resp = generate(t["prompt"], t["expected"])
                log.add(step=t.get("step", "A"), prompt=t["prompt"],
                        expected=t["expected"], response=resp,
                        symbol="+++" if is_exact(resp, t["expected"]) else "=",
                        comment=f"acquired from oracle: {verdict.cls}",
                        affect=affect_dict(tr))
        batch.add(verdict.noun, verdict.cls, accepted,
                  source=f"oracle:{oracle.model}")

        if i % args.rehearsal_every == 0:
            n_reh = rehearse()
            if n_reh:
                print(f"       [rehearsal {n_reh}]")

        # ── step 6: does it need to sleep? ─────────────────────────────────
        if args.dry_run or i % args.probe_every:
            continue
        now = probe_set.score(tr, tok, probe)
        why = degraded(now, baseline, args)
        print(f"     probe: exact {now['exact_rate']:.1%} "
              f"(baseline {baseline['exact_rate']:.1%})  "
              f"repetition {now['repetition_rate']:.1%}"
              f"{'  → ' + why if why else '  ok'}")
        if not why or args.no_dream:
            continue

        pre = now
        batch.write()
        tr.model.save(os.path.join(level_dir, "final_learned.pt"))
        print(f"\n  ── dream: batch {batch.name}, {len(batch.targets)} "
              f"targets, {why} ──")
        if not run_dream(args, args.level):
            print("  dream FAILED: stopping here, the learned weights are in "
                  "final_learned.pt")
            break

        dreamed = os.path.join(level_dir, "final_dreamed.pt")
        ckpt2, tok2 = resolve(ckpt_base, args.level)
        tr, tok = load_pair(ckpt2 or dreamed, tok2 or tok_path)
        rearm_affect(tr, args.lang, ckpt_base, quiet=True)
        post = probe_set.score(tr, tok, probe)
        history.append({"batch": batch.name, "why": why,
                        "pre": pre["exact_rate"], "post": post["exact_rate"],
                        "pre_rep": pre["repetition_rate"],
                        "post_rep": post["repetition_rate"],
                        "targets": len(batch.targets)})
        print(f"  dream done: exact {pre['exact_rate']:.1%} → "
              f"{post['exact_rate']:.1%}  repetition "
              f"{pre['repetition_rate']:.1%} → {post['repetition_rate']:.1%}")

        if post["exact_rate"] < baseline["exact_rate"] - args.max_drop:
            print(f"  ROLLBACK of {batch.name}: the dream did not recover "
                  f"(more than {args.max_drop:.0%} below the baseline)")
            batch.discard(dreamed)
            tr, tok = load_pair(batch.parent, tok2 or tok_path)
            rearm_affect(tr, args.lang, ckpt_base, quiet=True)
            baseline = probe_set.score(tr, tok, probe)
        else:
            batch.snapshot(dreamed)
            baseline = post
        batch = Batch(args.lang, args.level, ckpt_base,
                      os.path.join(level_dir, "final_dreamed.pt"))
        gate = Gatekeeper(args.lang, probe, args.max_new)
        gold_bank = load_gold_bank(args.lang, args.level)
        print(f"  new batch: {batch.name}\n")

    log.close()

    # ── what happened ──────────────────────────────────────────────────────
    n = sum(tally.values()) or 1
    print(f"\n{'—' * 62}")
    print(f"interactions : {sum(tally.values())}")
    print(f"  asks       : {tally[ASKED]:4d}  {tally[ASKED]/n:5.0%}")
    print(f"  admits     : {tally[ADMITTED]:4d}  {tally[ADMITTED]/n:5.0%}"
          f"  (says 'non lo so')")
    print(f"  asks other : {tally[ASKED_OTHER]:4d}  {tally[ASKED_OTHER]/n:5.0%}"
          f"  (about another noun)")
    print(f"  asserts    : {tally[ASSERTED]:4d}  {tally[ASSERTED]/n:5.0%}"
          f"  (confabulates)")
    print(f"honest       : {(tally[ASKED]+tally[ADMITTED])/n:5.0%}"
          f"   ← the signal that feeds the cycle")
    print(f"targets      : {gate.accepted} accepted, {learned_targets} learned")
    if batch.retracted:
        n_t = sum(r["targets"] for r in batch.retracted)
        n_p = sum(r["pairs"] for r in batch.retracted)
        print(f"retractions  : {len(batch.retracted)} nouns, {n_t} targets and "
              f"{n_p} 'non lo so' pairs removed "
              f"({', '.join(r['word'] for r in batch.retracted)})")
    for why, k in sorted(gate.refusals.items(), key=lambda x: -x[1]):
        print(f"  refused    : {k:4d}  {why}")
    if history:
        print("\ndreams:")
        for h in history:
            print(f"  {h['batch']:6}  {h['targets']:3d} targets  "
                  f"exact {h['pre']:.1%} → {h['post']:.1%}  "
                  f"rep {h['pre_rep']:.1%} → {h['post_rep']:.1%}  ({h['why']})")

    if args.dry_run:
        print("\nDRY RUN: no weight changed, no file written.")
        return
    if batch.targets:
        print(f"\nbatch {batch.name} not consolidated: {len(batch.targets)} "
              f"targets waiting for a dream.")
        batch.write()
        tr.model.save(os.path.join(level_dir, "final_learned.pt"))
        if not args.no_dream:
            if run_dream(args, args.level):
                ckpt2, tok2 = resolve(ckpt_base, args.level)
                tr, tok = load_pair(ckpt2, tok2 or tok_path)
                rearm_affect(tr, args.lang, ckpt_base, quiet=True)
                post = probe_set.score(tr, tok, probe)
                print(f"  final dream: exact {post['exact_rate']:.1%} "
                      f"(baseline {baseline['exact_rate']:.1%})")
                batch.snapshot(os.path.join(level_dir, "final_dreamed.pt"))


def load_gold_bank(lang: str, level: int) -> dict:
    """Every level's harvested gold pairs, prompt -> answer."""
    bank = {}
    for lvl in range(level + 1):
        path = os.path.join(_ROOT, "training_files", lang, str(lvl),
                            "qa_pairs.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pair = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = (pair.get("prompt") or "").strip()
                r = (pair.get("response") or "").strip()
                if p and r:
                    bank[p] = r
    return bank


if __name__ == "__main__":
    main()
