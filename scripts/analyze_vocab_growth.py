"""
Retrospective analysis of the dynamic vocabulary growth across a curriculum
build: one row per token created by the dream's N2-B phase.

Everything is reconstructed from artefacts already on disk — the per-level
tokenizer snapshots (`level_N/tokenizer.json`) and the teaching session logs
(`level_N/session_*.jsonl`) — so it can be run on any past build without
retraining anything.

The reconstruction of what N2-B actually saw reuses the training code itself
(`_load_memory_bank`, `_extract_patterns`, `build_growth_text` from
train_curriculum): a private copy of that logic would silently stop describing
the real build the first time the training changed.

CAVEAT: the memory bank is rebuilt from the logs, so the numbers derived from
it (growth-text size, threshold, source frequencies) are approximations. The
vocab-growing 'standard' dream runs after the FIRST teaching session of a
level, which is emulated with `current_level_sessions=1`, but a level whose
build needed a retry may have grown again later. Trends and orders of
magnitude are solid; absolute counts are not. Builds that carry
`growth_events.jsonl` (written by N2-B itself) need no reconstruction: those
values are exact and take precedence.

Usage:
    python3 scripts/analyze_vocab_growth.py                      # default it, L0-L10
    python3 scripts/analyze_vocab_growth.py --max-level 5
    python3 scripts/analyze_vocab_growth.py --ckpt-base models/exp_d/dyn_gold
    python3 scripts/analyze_vocab_growth.py --csv out.csv --no-corpus-freq
"""
import sys, os, argparse, csv, glob, json, re

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splx.tokenizer import BPETokenizer
from dynamic_model.train_curriculum import (
    _load_memory_bank, _extract_patterns, build_growth_text,
)

# Same constants as DynamicBPETokenizer.grow() — imported rather than copied
# so a change to the threshold shows up here too.
from dynamic_model.core.tokenizer import DynamicBPETokenizer

BASE_TOKENIZER = "dynamic_model/data/tokenizer_8k.json"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> BPETokenizer:
    tok = BPETokenizer()
    tok.load(path)
    return tok


def token_str(tok: BPETokenizer, tid: int) -> str:
    return tok.vocab[tid].decode("utf-8", errors="replace")


def level_cohorts(ckpt_base: str, max_level: int) -> list:
    """
    For each level, the token ids its snapshot added over the previous one.

    Returns a list of dicts: {level, path, prev_path, tok, prev_tok, new_ids}.
    Levels whose snapshot is missing are skipped with a warning — a build
    interrupted mid-curriculum still yields a partial analysis.
    """
    cohorts = []
    prev_path, prev_tok = None, None
    for lvl in range(max_level + 1):
        path = os.path.join(ckpt_base, f"level_{lvl}", "tokenizer.json")
        if not os.path.exists(path):
            print(f"  [warn] no snapshot for level {lvl} — skipped")
            continue
        tok = load_tokenizer(path)
        if prev_tok is None:
            # First snapshot found: measure it against the base tokenizer, so
            # a growth already present at L0 is not invisible.
            base = load_tokenizer(BASE_TOKENIZER)
            new_ids = sorted(set(tok.vocab) - set(base.vocab))
            cohorts.append({"level": lvl, "path": path, "prev_path": BASE_TOKENIZER,
                            "tok": tok, "prev_tok": base, "new_ids": new_ids})
        else:
            new_ids = sorted(set(tok.vocab) - set(prev_tok.vocab))
            cohorts.append({"level": lvl, "path": path, "prev_path": prev_path,
                            "tok": tok, "prev_tok": prev_tok, "new_ids": new_ids})
        prev_path, prev_tok = path, tok
    return cohorts


# ---------------------------------------------------------------------------
# What N2-B saw, reconstructed
# ---------------------------------------------------------------------------

def segment_counts(text: str) -> dict:
    """Occurrences of every whitespace-delimited segment — the unit grow()
    works on (merges never cross a word boundary)."""
    counts = {}
    for seg in re.findall(r"\S+", text):
        counts[seg] = counts.get(seg, 0) + 1
    return counts


def growth_context(ckpt_base: str, level: int, prev_tok: BPETokenizer) -> dict:
    """
    Rebuild the growth text of the level's vocab-growing dream and the
    adaptive threshold that gated it.
    """
    bank = _load_memory_bank(ckpt_base, level, current_level_sessions=1)
    patterns = _extract_patterns(bank, prev_tok) if bank else []
    texts = {src: build_growth_text(bank, patterns, sources=src)
             for src in ("both", "gold", "prompt")}

    # n_tokens_total exactly as grow() counts it: encode each \S+ segment with
    # the tokenizer as it was BEFORE the growth.
    n_tokens = sum(len(prev_tok.encode(seg))
                   for seg in re.findall(r"\S+", texts["both"]))
    threshold = max(5, int(n_tokens * 0.002))   # MIN_ABS, MIN_REL of grow()

    return {
        "bank_size":  len(bank),
        "n_patterns": len(patterns),
        "texts":      texts,
        "counts":     {src: segment_counts(t) for src, t in texts.items()},
        "n_tokens":   n_tokens,
        "threshold":  threshold,
    }


def first_seen(ckpt_base: str, level: int, needle: str) -> tuple:
    """Earliest session log (up to `level`) whose prompt or gold contains the
    string, as (session_file, iso_timestamp, field)."""
    best = (None, None, None)
    for lvl in range(level + 1):
        pattern = os.path.join(ckpt_base, f"level_{lvl}", "session_*.jsonl")
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or needle not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    field = None
                    if needle in (rec.get("expected") or ""):
                        field = "expected"
                    elif needle in (rec.get("prompt") or ""):
                        field = "prompt"
                    if field:
                        stamp = os.path.basename(path)[len("session_"):-len(".jsonl")]
                        return (os.path.basename(path), stamp, field)
    return best


# ---------------------------------------------------------------------------
# Corpus frequency
# ---------------------------------------------------------------------------

_corpus_cache = {}


def corpus_segment_counts(lang: str, level: int) -> dict:
    """Segment counts over the level's own training corpus (*.txt) and its
    gold Q&A pairs. Cached — the L3-L5 corpora are tens of MB."""
    if level in _corpus_cache:
        return _corpus_cache[level]
    counts = {}
    level_dir = os.path.join("training_files", lang, str(level))
    paths = sorted(glob.glob(os.path.join(level_dir, "*.txt")))
    qa = os.path.join(level_dir, "qa_pairs.jsonl")
    if os.path.exists(qa):
        paths.append(qa)
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    for seg in re.findall(r"\S+", line):
                        counts[seg] = counts.get(seg, 0) + 1
        except OSError:
            continue
    _corpus_cache[level] = counts
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--ckpt-base", default=None,
                    help="Checkpoint base (default: models/checkpoints/{lang})")
    ap.add_argument("--max-level", type=int, default=10)
    ap.add_argument("--csv", default="models/analysis/vocab_growth.csv")
    ap.add_argument("--no-corpus-freq", action="store_true",
                    help="Skip corpus frequency (avoids reading the large L3-L5 corpora)")
    args = ap.parse_args()

    ckpt_base = args.ckpt_base or os.path.join("models", "checkpoints", args.lang)
    cohorts = level_cohorts(ckpt_base, args.max_level)
    if not cohorts:
        print(f"No tokenizer snapshot found under {ckpt_base}")
        sys.exit(1)

    final_tok = cohorts[-1]["tok"]
    base_size = len(load_tokenizer(BASE_TOKENIZER).vocab)

    rows = []
    for c in cohorts:
        level, tok, prev_tok = c["level"], c["tok"], c["prev_tok"]
        if not c["new_ids"]:
            continue
        ctx = growth_context(ckpt_base, level, prev_tok)
        parents = {nid: (a, b) for (a, b, nid) in tok.merges}

        for tid in c["new_ids"]:
            s = token_str(tok, tid)
            pa, pb = parents.get(tid, (None, None))
            g = ctx["counts"]["gold"].get(s, 0)
            p = ctx["counts"]["prompt"].get(s, 0)
            origin = ("both" if g and p else
                      "gold" if g else
                      "prompt" if p else "unseen")
            session, stamp, field = first_seen(ckpt_base, level, s)
            corpus_freq = (None if args.no_corpus_freq
                           else corpus_segment_counts(args.lang, level).get(s, 0))
            rows.append({
                "level":         level,
                "token_id":      tid,
                "string":        s,
                "n_chars":       len(s),
                "parent_a":      pa,
                "parent_b":      pb,
                "parent_str":    (f"{token_str(tok, pa)}+{token_str(tok, pb)}"
                                  if pa is not None else ""),
                "origin":        origin,
                "freq_gold":     g,
                "freq_prompt":   p,
                "threshold":     ctx["threshold"],
                "growth_tokens": ctx["n_tokens"],
                "bank_size":     ctx["bank_size"],
                "first_session": session or "",
                "first_stamp":   stamp or "",
                "first_field":   field or "",
                "corpus_freq":   "" if corpus_freq is None else corpus_freq,
                # A token encode() can never produce is a dead slot that still
                # competes in the weight-tied softmax.
                "reachable":     final_tok.encode(s) == [tid],
            })

    # ── CSV ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n  Vocabolario: {base_size} (base) → {len(final_tok.vocab)} "
          f"= +{len(rows)} token in {len(cohorts)} livelli")
    print(f"  Matrice: {args.csv}\n")

    print("  ## Crescita per livello\n")
    print(f"  {'Liv':>4}  {'nuovi':>5}  {'soglia':>6}  {'growth_tok':>10}  "
          f"{'bank':>5}  {'gold':>4} {'prompt':>6} {'both':>4} {'??':>3}  irraggiungibili")
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r)
    for c in cohorts:
        lv = c["level"]
        rs = by_level.get(lv, [])
        if not rs:
            print(f"  {lv:>4}  {0:>5}       —           —      —"
                  f"     {'—':>4} {'—':>6} {'—':>4} {'—':>3}")
            continue
        n_dead = sum(1 for r in rs if not r["reachable"])
        cnt = lambda o: sum(1 for r in rs if r["origin"] == o)
        print(f"  {lv:>4}  {len(rs):>5}  {rs[0]['threshold']:>6}  "
              f"{rs[0]['growth_tokens']:>10,}  {rs[0]['bank_size']:>5}  "
              f"{cnt('gold'):>4} {cnt('prompt'):>6} {cnt('both'):>4} "
              f"{cnt('unseen'):>3}  {n_dead}")

    print("\n  ## Token per livello\n")
    for c in cohorts:
        rs = by_level.get(c["level"], [])
        if rs:
            print(f"  L{c['level']:<2} +{len(rs):<3} " +
                  " ".join(f"{r['string']!r}" for r in rs))

    n_dead_all = sum(1 for r in rows if not r["reachable"])
    n_unseen = sum(1 for r in rows if r["origin"] == "unseen")
    print(f"\n  Irraggiungibili da encode(): {n_dead_all}/{len(rows)}")
    if n_unseen:
        print(f"  Non ritrovati nel growth text ricostruito: {n_unseen} "
              f"(atteso > 0: il bank è ricostruito, vedi il caveat in testa)")


if __name__ == "__main__":
    main()
