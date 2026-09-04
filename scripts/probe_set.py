#!/usr/bin/env python3
"""
probe_set — the frozen sample of curriculum gold the autonomy loop measures on.

The loop needs to know when learning new material has started to cost it the
old. That question can only be answered on prompts the loop never teaches, so
the sample is drawn ONCE, written to disk, and committed. A probe rebuilt on
demand would drift towards whatever the loop happens to be good at, and the
degradation trigger would then confirm itself.

Two guards make the freeze real:

  * the file carries a `fingerprint` over its items — if it changes, the
    comparison with an earlier baseline is not a comparison any more;
  * `load()` refuses to build one silently. Creating it is an explicit act
    (`--write`), so a missing file stops the loop instead of inventing a
    baseline.

Usage:
    python3 scripts/probe_set.py --write            # freeze it (once)
    python3 scripts/probe_set.py --show             # what is in it
    python3 scripts/probe_set.py --score --checkpoint X --tokenizer Y
"""
import argparse
import hashlib
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dynamic_model.test_model import load_level_cases, is_exact       # noqa: E402
from measure_repetition import has_repetition                          # noqa: E402

DEFAULT_PATH = os.path.join(_ROOT, "dynamic_model", "data", "probe_set.json")
DEFAULT_SEED = 20260831
DEFAULT_PER_LEVEL = 8
DEFAULT_LEVELS = list(range(13))     # the fixed curriculum, 0-12


def fingerprint(items: list) -> str:
    """Stable hash of the prompts and golds, order included."""
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it['level']}\x1f{it['prompt']}\x1f{it['expected']}\x1e"
                 .encode("utf-8"))
    return h.hexdigest()[:16]


def build(lang: str = "it", per_level: int = DEFAULT_PER_LEVEL,
          seed: int = DEFAULT_SEED, levels: list = None) -> dict:
    """Sample `per_level` graded cases from each level's curriculum."""
    levels = levels or DEFAULT_LEVELS
    items = []
    for lvl in levels:
        cases = [(p, e) for p, e in load_level_cases(lang, lvl) if e]
        if not cases:
            continue
        # A dedicated RNG per level: adding a level later must not reshuffle
        # the sample of the levels before it.
        rng = random.Random(seed + lvl)
        for p, e in rng.sample(cases, min(per_level, len(cases))):
            items.append({"level": lvl, "prompt": p, "expected": e})
    return {"lang": lang, "seed": seed, "per_level": per_level,
            "levels": levels, "n": len(items),
            "fingerprint": fingerprint(items), "items": items}


def write(path: str = DEFAULT_PATH, **kw) -> dict:
    data = build(**kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def load(path: str = DEFAULT_PATH) -> dict:
    """Read the frozen probe. Raises if it is missing or has been edited."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no frozen probe set at {path} — create it once with "
            f"'python3 scripts/probe_set.py --write'. It is deliberately not "
            f"built on demand: a probe that follows the training material "
            f"cannot detect degradation.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    have = fingerprint(data.get("items", []))
    if data.get("fingerprint") != have:
        raise RuntimeError(
            f"probe set at {path} was modified (fingerprint {have}, file says "
            f"{data.get('fingerprint')}): baselines measured on the old one "
            f"are not comparable. Restore it from git, or re-freeze and start "
            f"a fresh baseline.")
    return data


def score(tr, tok, probe: dict) -> dict:
    """
    Score a loaded model on the probe.

    Generation parameters are copied from measure_repetition.evaluate_level so
    the numbers here and the published ones mean the same thing: greedy,
    length-aware stop, gold-derived budget.
    """
    per_level, exact, rep = {}, 0, 0
    misses = []
    for it in probe["items"]:
        expected = it["expected"]
        n_exp = len(tok.encode(expected))
        out = tr.generate(it["prompt"],
                          max_tokens=max(24, min(2 * n_exp + 12, 120)),
                          base_temperature=0.0, top_k=1,
                          min_tokens=max(4, min(n_exp, 40)),
                          stop_after=max(0, min(n_exp - 1, 40)))
        answer = out[len(it["prompt"]):].strip()
        ok = is_exact(answer, expected)
        d = per_level.setdefault(it["level"], {"n": 0, "exact": 0})
        d["n"] += 1
        d["exact"] += 1 if ok else 0
        exact += 1 if ok else 0
        if has_repetition(answer):
            rep += 1
        if not ok and len(misses) < 5:
            misses.append(f"{it['prompt']!r} -> {answer!r} (expected {expected!r})")
    n = probe["n"] or 1
    return {"n": probe["n"], "exact": exact, "exact_rate": exact / n,
            "repetition": rep, "repetition_rate": rep / n,
            "per_level": {k: v["exact"] / (v["n"] or 1)
                          for k, v in sorted(per_level.items())},
            "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen probe set")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--write", action="store_true", help="freeze a new one")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--checkpoint"), ap.add_argument("--tokenizer")
    ap.add_argument("--lang", default="it")
    ap.add_argument("--per-level", type=int, default=DEFAULT_PER_LEVEL)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()

    if a.write:
        if os.path.exists(a.path):
            print(f"{a.path} already exists: not overwriting it.\n"
                  f"The probe is frozen by definition — remove it by hand "
                  f"only if you want to start from a new baseline.")
            return
        d = write(a.path, lang=a.lang, per_level=a.per_level, seed=a.seed)
        print(f"→ {a.path}\n  {d['n']} prompts, "
              f"{len(d['levels'])} levels, fingerprint {d['fingerprint']}")
        return

    probe = load(a.path)
    print(f"probe: {probe['n']} prompts  fingerprint {probe['fingerprint']}  "
          f"(seed {probe['seed']}, {probe['per_level']}/level)")
    if a.show:
        for it in probe["items"]:
            print(f"  L{it['level']:<2} {it['prompt']!r} -> {it['expected']!r}")

    if a.score:
        if not a.checkpoint or not a.tokenizer:
            ap.error("--score requires --checkpoint and --tokenizer")
        from measure_repetition import load_pair
        tr, tok = load_pair(a.checkpoint, a.tokenizer)
        r = score(tr, tok, probe)
        print(f"\nexact {r['exact_rate']:.1%}  repetition "
              f"{r['repetition_rate']:.1%}")
        print("  per level: " + " ".join(f"L{k}:{v:.0%}"
                                           for k, v in r["per_level"].items()))
        for m in r["misses"]:
            print(f"  ✗ {m}")


if __name__ == "__main__":
    main()
