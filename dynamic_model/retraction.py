"""
Retracting an admission of ignorance.

Level 12 teaches `cos è un falco?` -> `non lo so.` The autonomy loop can learn
that a falcon is an animal. Both cannot stay: one prompt carries one gold, and
two answers under two shapes of the same question is the exact failure
`ask_forms` exists to prevent — measured once already, when the indefinite
article itself came to mean 'unknown' and `cosa è un cane?` was answered
`non lo so.`

So an admission that learning has made false is REMOVED from the curriculum
that teaches it. That is the semantics of a knowledge base that grows: "I do
not know" is a state, not a fact.

Three properties this module has to have, and each one is a lesson already paid
for in this repo:

  * **Reversible literally, not by regeneration.** The removed items are saved
    verbatim WITH THEIR INDEX and put back where they were. Re-deriving them
    from the pool generators would work only until the pool changes, and
    appending them at the end of qa_pairs.jsonl would change qa_corpus.txt --
    a derived, committed file that `generate_qa_corpus.py --check` compares
    byte for byte.

  * **Durable against regeneration.** `expand_teacher_pools.py` rebuilds the
    step from the lexicon; without a ledger it would resurrect every retracted
    admission on the next run, silently. The ledger is the record the
    generators consult.

  * **All of a noun or none of it.** Removing the phrasings that collide and
    leaving the ones that do not is how contradictory supervision got in
    before.

The ledger lives in the curriculum (`training_files/<lang>/retracted.jsonl`),
not in the checkpoint tree: it is a fact about the material, and a fresh clone
has to see it.
"""
import json, os, re, time, glob, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Answers that count as an admission of ignorance. Compared after lowercasing
# and stripping final punctuation, never by substring: 'non lo so perché ...'
# would be an assertion.
ADMISSIONS = ("non lo so", "non so")

LEDGER_NAME = "retracted.jsonl"


def is_admission(gold: str) -> bool:
    return (gold or "").strip().lower().rstrip(".!?") in ADMISSIONS


def is_ignorance(gold: str, word: str) -> bool:
    """Does this gold treat `word` as something the model cannot know?

    Two shapes, not one, and the second was missed at first. Level 12 teaches
    both `cos è un falco?` -> `non lo so.` AND
    `il ragno è un animale, questo è un falco` -> `cos è un falco?`. Learning
    that a falcon is an animal falsifies both: once the class is known, being
    shown a falcon calls for the assertion (which is what step B already
    teaches about every known noun), not the question. Retracting only the
    admission would leave the level teaching the model to ask about something
    it had just been taught — a contradiction under two shapes of the same
    situation, which is the failure this module exists to prevent.

    The question has to be about THIS word, checked in the gold and not in the
    prompt: step G's prompts deliberately name a second noun as a distractor
    ('la zucca è un cibo, questo è un falco'), and that lesson is about falco,
    not about zucca.
    """
    if is_admission(gold):
        return True
    g = (gold or "").strip()
    return g.endswith("?") and bool(_word_re(word).search(g))


def stale_admission(prompt: str, gold: str, gone) -> bool:
    """Does (prompt, gold) teach ignorance about a word already retracted?

    `gone` is an iterable of retracted words (see `retracted()`). The word is
    looked for in the PROMPT, because a bare admission ("non lo so.") names no
    word at all — its subject is whatever the question asked about. The gold
    is then tested with `is_ignorance`, which also catches the ask-shaped gold
    ("cos è un tasso?") of the two-clause step.

    This is the predicate the dream's QA harvest needs: a session log records
    the gold OF THE MOMENT THE TURN RAN, and a turn asked before an acquisition
    carries "non lo so." even after the retraction has removed that admission
    from the curriculum. Harvesting it would re-teach the retracted admission
    right next to the new class gold — one prompt, two answers.
    """
    for w in gone:
        if _word_re(w).search(prompt or "") and is_ignorance(gold, w):
            return True
    return False


def ledger_path(lang: str = "it") -> str:
    return os.path.join(_ROOT, "training_files", lang, LEDGER_NAME)


def retracted(lang: str = "it") -> dict:
    """word -> the record of its retraction. Empty dict if nothing was."""
    path = ledger_path(lang)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Last record for a word wins, so a restore can annul an earlier
            # retraction by appending rather than rewriting the file.
            if rec.get("restored"):
                out.pop(rec.get("word"), None)
            else:
                out[rec.get("word")] = rec
    return out


def _word_re(word: str):
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def _levels(lang: str):
    base = os.path.join(_ROOT, "training_files", lang)
    for d in sorted(glob.glob(os.path.join(base, "[0-9]*")),
                    key=lambda p: int(os.path.basename(p))
                    if os.path.basename(p).isdigit() else 1 << 30):
        name = os.path.basename(d)
        if name.isdigit() and os.path.isdir(d):
            yield int(name), d


def find(word: str, lang: str = "it") -> dict:
    """Everything in the curriculum that answers about `word` with an admission.

    Also reports, under 'other', material about the same noun whose gold does
    NOT treat it as unknown. That list must be empty for a retraction to be
    safe: it would mean the noun is taught something else somewhere, and
    removing only part would leave the pool half-changed.
    """
    pat = _word_re(word)
    hit = {"word": word, "levels": {}, "other": []}
    for level, d in _levels(lang):
        cfg_path = os.path.join(d, "local_teacher.json")
        targets, pairs = [], []
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            for sid, step in (cfg.get("steps") or {}).items():
                for i, t in enumerate(step.get("targets", [])):
                    if not isinstance(t, dict):
                        continue
                    if not pat.search(t.get("prompt", "")):
                        continue
                    if is_ignorance(t.get("expected", ""), word):
                        targets.append({"step": sid, "index": i, "target": t})
                    else:
                        hit["other"].append(
                            {"level": level, "step": sid,
                             "prompt": t.get("prompt"),
                             "expected": t.get("expected")})
        qa_path = os.path.join(d, "qa_pairs.jsonl")
        if os.path.exists(qa_path):
            with open(qa_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    s = line.rstrip("\n")
                    if not s.strip():
                        continue
                    try:
                        rec = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if not pat.search(rec.get("prompt", "")):
                        continue
                    if is_ignorance(rec.get("response", ""), word):
                        pairs.append({"index": i, "line": s})
                    else:
                        hit["other"].append(
                            {"level": level, "step": "qa_pairs",
                             "prompt": rec.get("prompt"),
                             "expected": rec.get("response")})
        if targets or pairs:
            hit["levels"][str(level)] = {"targets": targets, "pairs": pairs}
    return hit


def _regen(levels, lang: str) -> None:
    """Rewrite qa_corpus.txt for each level, through the one generator.

    Imported and not reimplemented: the shuffle seed, the repetition count and
    the EOS marker already exist in two places that must agree
    (`generate_qa_corpus.py` and `_regen_corpus` in train_curriculum.py), and
    they drifted once — a clone then trained on a different file than the build
    wrote. A third copy would make that worse. The generator resolves paths
    from the cwd, hence the chdir.
    """
    import generate_qa_corpus
    cwd = os.getcwd()
    try:
        os.chdir(_ROOT)
        for level in sorted(int(l) for l in levels):
            generate_qa_corpus.generate(level, lang, quiet=True)
    finally:
        os.chdir(cwd)


def _save_path(save_dir: str, word: str) -> str:
    return os.path.join(save_dir, "retracted", f"{word}.json")


def retract(word: str, lang: str = "it", *, save_dir: str,
            batch: str = "", source: str = "", reason: str = "") -> dict:
    """Remove every admission about `word`, saving what it takes to put it back.

    Raises ValueError if the noun also carries a gold that does not treat it as
    unknown: that is not a retraction, it is a pool that needs fixing first.
    """
    hit = find(word, lang)
    if hit["other"]:
        raise ValueError(
            f"'{word}' porta anche {len(hit['other'])} gold che non lo "
            f"trattano come ignoto: non è un ritiro, è un pool da correggere")
    if not hit["levels"]:
        return {"word": word, "targets": 0, "pairs": 0, "levels": []}

    path = _save_path(save_dir, word)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hit, f, ensure_ascii=False, indent=1)

    n_t = n_p = 0
    for level, part in hit["levels"].items():
        d = os.path.join(_ROOT, "training_files", lang, level)
        if part["targets"]:
            cfg_path = os.path.join(d, "local_teacher.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # Descending, so an earlier removal does not shift a later index.
            for item in sorted(part["targets"],
                               key=lambda x: -x["index"]):
                cfg["steps"][item["step"]]["targets"].pop(item["index"])
                n_t += 1
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        if part["pairs"]:
            qa_path = os.path.join(d, "qa_pairs.jsonl")
            with open(qa_path, encoding="utf-8") as f:
                lines = f.read().split("\n")
            drop = {item["index"] for item in part["pairs"]}
            keep = [s for i, s in enumerate(lines) if i not in drop]
            n_p += len(drop)
            with open(qa_path, "w", encoding="utf-8") as f:
                f.write("\n".join(keep))
    _regen(hit["levels"].keys(), lang)

    with open(ledger_path(lang), "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "word": word, "levels": sorted(int(l) for l in hit["levels"]),
            "targets": n_t, "pairs": n_p, "batch": batch, "source": source,
            "reason": reason or "acquisita: l'ammissione è diventata falsa",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "saved": os.path.relpath(path, _ROOT),
        }, ensure_ascii=False) + "\n")

    return {"word": word, "targets": n_t, "pairs": n_p,
            "levels": sorted(int(l) for l in hit["levels"])}


def restore(word: str, lang: str = "it", *, save_dir: str) -> dict:
    """Put back what `retract` removed, at the indices it removed them from."""
    path = _save_path(save_dir, word)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"nessun salvataggio del ritiro di '{word}' in {save_dir}")
    with open(path, encoding="utf-8") as f:
        hit = json.load(f)

    n_t = n_p = 0
    for level, part in hit["levels"].items():
        d = os.path.join(_ROOT, "training_files", lang, level)
        if part["targets"]:
            cfg_path = os.path.join(d, "local_teacher.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # Ascending: each insert makes room for the next one's index.
            for item in sorted(part["targets"], key=lambda x: x["index"]):
                step = cfg["steps"].setdefault(item["step"], {"targets": []})
                step.setdefault("targets", []).insert(item["index"],
                                                      item["target"])
                n_t += 1
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        if part["pairs"]:
            qa_path = os.path.join(d, "qa_pairs.jsonl")
            with open(qa_path, encoding="utf-8") as f:
                lines = f.read().split("\n")
            for item in sorted(part["pairs"], key=lambda x: x["index"]):
                lines.insert(item["index"], item["line"])
                n_p += 1
            with open(qa_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
    _regen(hit["levels"].keys(), lang)

    with open(ledger_path(lang), "a", encoding="utf-8") as f:
        f.write(json.dumps({"word": word, "restored": True,
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                           ensure_ascii=False) + "\n")
    return {"word": word, "targets": n_t, "pairs": n_p}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("word")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--save-dir", default=os.path.join(_ROOT, "training_files",
                                                      "it", "13"))
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--show", action="store_true",
                    help="solo elenco, non tocca nulla")
    a = ap.parse_args()
    if a.show:
        hit = find(a.word, a.lang)
        for level, part in hit["levels"].items():
            print(f"  L{level}: {len(part['targets'])} target, "
                  f"{len(part['pairs'])} coppie")
        for o in hit["other"]:
            print(f"  ALTRO  L{o['level']} {o['step']}: "
                  f"{o['prompt']!r} -> {o['expected']!r}")
    elif a.restore:
        print(restore(a.word, a.lang, save_dir=a.save_dir))
    else:
        print(retract(a.word, a.lang, save_dir=a.save_dir, source="cli"))
