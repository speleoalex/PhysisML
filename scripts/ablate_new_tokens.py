"""
Vocabulary ablation: how much does the trained model actually rely on the
tokens the dream gave it?

For each condition the tokens under test are removed from the tokenizer (their
merges are dropped, so `encode` routes around them exactly as a model that had
never grown them would) AND suppressed in the output distribution. The level's
curriculum cases are then re-answered greedily and compared with the intact
baseline.

    baseline   → intact model and tokenizer, must reproduce the build's table
    all        → every token above the base vocabulary suppressed
    cohort     → one level's cohort at a time
    single     → one token at a time (expensive; use --levels to narrow)

WHAT THIS MEASURES: reliance at inference time. It does NOT measure the
benefit of having had the token during training — a token can be load-bearing
now (ablating it hurts) simply because training routed through it. The
training-time question needs the controlled arms
(scripts/experiment_vocab_growth.sh), not this script.

Usage:
    python3 scripts/ablate_new_tokens.py --mode all
    python3 scripts/ablate_new_tokens.py --mode cohort --levels 0-10
    python3 scripts/ablate_new_tokens.py --mode single --levels 1-2
"""
import sys, os, argparse, copy, json, glob

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from physisml.torch_model import TorchGPT, TorchAdamOptimizer
from physisml.tokenizer   import BPETokenizer
from dynamic_model.exp_b.affect_state import AffectState
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.trainer      import TrainerB
from dynamic_model.test_model import (
    load_level_cases, level_lexicon, score_output, is_exact,
)
from dynamic_model.compare_b import bits_per_char

torch.set_num_threads(12)

BASE_TOKENIZER  = "dynamic_model/data/tokenizer_8k.json"
VALIDATION_PATH = "dynamic_model/data/validation.txt"


# ---------------------------------------------------------------------------
# Tokenizer surgery
# ---------------------------------------------------------------------------

def prune_tokenizer(tok: BPETokenizer, banned: set) -> BPETokenizer:
    """
    Copy of `tok` without the banned tokens: their merges are dropped, so
    encode() falls back to the multi-token path the model learned before the
    growth. Merges whose parent was removed simply never fire.
    """
    if not banned:
        return tok
    out = copy.deepcopy(tok)
    out.merges = [(a, b, nid) for (a, b, nid) in out.merges if nid not in banned]
    for tid in banned:
        tok_bytes = out.vocab.pop(tid, None)
        if tok_bytes is not None:
            out.token_to_id.pop(tok_bytes, None)
    return out


def new_token_ids(ckpt_base: str, lang: str, max_level: int) -> dict:
    """{level: [token_id, ...]} for every token added over the base vocabulary,
    read from the per-level tokenizer snapshots."""
    base = BPETokenizer(); base.load(BASE_TOKENIZER)
    seen = set(base.vocab)
    cohorts = {}
    for lvl in range(max_level + 1):
        path = os.path.join(ckpt_base, f"level_{lvl}", "tokenizer.json")
        if not os.path.exists(path):
            continue
        t = BPETokenizer(); t.load(path)
        new = sorted(set(t.vocab) - seen)
        if new:
            cohorts[lvl] = new
        seen |= set(t.vocab)
    return cohorts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_trainer(model: TorchGPT, tok: BPETokenizer) -> TrainerB:
    affect = AffectState()
    return TrainerB(model, tok, TorchAdamOptimizer(model.parameters(), lr=1e-4),
                    affect, AffectModulator(affect))


def evaluate(model: TorchGPT, tok: BPETokenizer, cases: list, lex: list,
             banned: set = None, count_emissions: bool = False,
             prune: bool = True) -> dict:
    """
    Greedy exact-match over the level's cases. Mirrors dynamic_model/test_model
    (temperature 0, top_k 1, budget sized on the gold answer) so the baseline
    is comparable with the published build table.

    prune=True removes the banned tokens from the tokenizer as well, so the
    PROMPT is re-encoded the way a model that never grew them would see it.
    That is the faithful counterfactual, but it moves two things at once: the
    input encoding shifts away from what training saw, and the output path
    loses the token. prune=False keeps the training encoding and only closes
    the output path — the difference between the two isolates the input shift
    from genuine output reliance.
    """
    banned = banned or set()
    eval_tok = prune_tokenizer(tok, banned) if prune else tok
    trainer  = build_trainer(model, eval_tok)

    n_exact, scores, emissions = 0, [], {}
    for prompt, expected in cases:
        n_exp = len(eval_tok.encode(expected)) if expected else 0
        out = trainer.generate(
            prompt,
            max_tokens=int(max(25, min(2 * n_exp + 12, 120))),
            base_temperature=1.0,
            top_k=1,
            min_tokens=max(4, min(n_exp, 40)) if n_exp else 4,
            stop_after=max(0, min(n_exp - 1, 40)) if n_exp else 2,
            banned_ids=banned or None,
        )
        answer = out[len(prompt):].strip()
        if expected and is_exact(answer, expected):
            n_exact += 1
        scores.append(score_output(answer, lex))
        if count_emissions:
            for tid in eval_tok.encode(answer):
                emissions[tid] = emissions.get(tid, 0) + 1

    graded = sum(1 for _, e in cases if e)
    return {
        "exact":     n_exact / graded if graded else float("nan"),
        "n_exact":   n_exact,
        "n_graded":  graded,
        "score":     float(np.mean(scores)) if scores else float("nan"),
        "emissions": emissions,
    }


def level_bpc(model: TorchGPT, tok: BPETokenizer, text: str,
              banned: set = None) -> float:
    """bits/char of the validation text, with the banned tokens removed from
    the tokenizer (never from the targets, which would make the loss infinite)."""
    return bits_per_char(model, prune_tokenizer(tok, banned or set()), text)


def load_level(ckpt_base: str, level: int):
    """(model, tokenizer, checkpoint_path) for a level, preferring the
    post-dream checkpoint — the one the build table was measured on."""
    lvl_dir = os.path.join(ckpt_base, f"level_{level}")
    ckpt = next((p for p in (os.path.join(lvl_dir, "final_dreamed.pt"),
                             os.path.join(lvl_dir, "final_learned.pt"),
                             os.path.join(lvl_dir, "final.pt"))
                 if os.path.exists(p)), None)
    if not ckpt:
        return None, None, None
    model = TorchGPT.load(ckpt)
    tok   = BPETokenizer()
    for cand in [os.path.join(lvl_dir, "tokenizer.json")] + \
                [os.path.join(ckpt_base, f"level_{l}", "tokenizer.json")
                 for l in range(level - 1, -1, -1)] + [BASE_TOKENIZER]:
        if os.path.exists(cand):
            t = BPETokenizer(); t.load(cand)
            if len(t) >= model.active_vocab_size:
                tok = t
                break
    return model, tok, ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_levels(spec: str) -> list:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--ckpt-base", default=None)
    ap.add_argument("--levels", default="0-10", help="Levels to evaluate, e.g. '0-10' or '1,2,5'")
    ap.add_argument("--max-level", type=int, default=10,
                    help="Highest level whose tokenizer snapshot is read for the cohorts")
    ap.add_argument("--mode", default="all", choices=["baseline", "all", "cohort", "single"])
    ap.add_argument("--keep-encoding", action="store_true",
                    help="Ban the tokens in the output only, leaving the prompt "
                         "encoded as during training (isolates output reliance "
                         "from the input re-encoding shift)")
    ap.add_argument("--skip-bpc", action="store_true",
                    help="Skip bits-per-char (saves a pass over validation.txt per condition)")
    ap.add_argument("--out", default="models/analysis/ablation.json")
    args = ap.parse_args()

    ckpt_base = args.ckpt_base or os.path.join("models", "checkpoints", args.lang)
    levels    = parse_levels(args.levels)
    cohorts   = new_token_ids(ckpt_base, args.lang, args.max_level)
    all_new   = sorted(t for ids in cohorts.values() for t in ids)

    val_text = ""
    if not args.skip_bpc and os.path.exists(VALIDATION_PATH):
        with open(VALIDATION_PATH, encoding="utf-8") as f:
            val_text = f.read()

    print(f"\n  Coorti: " + "  ".join(f"L{l}:{len(ids)}" for l, ids in cohorts.items()))
    print(f"  Token nuovi totali: {len(all_new)}   modalità: {args.mode}"
          + ("   (solo output, encoding invariato)" if args.keep_encoding else "") + "\n")

    results = {"mode": args.mode, "ckpt_base": ckpt_base,
               "keep_encoding": args.keep_encoding, "levels": {}}

    for level in levels:
        model, tok, ckpt = load_level(ckpt_base, level)
        if model is None:
            print(f"  L{level}: nessun checkpoint — salto")
            continue
        cases = load_level_cases(args.lang, level)
        lex   = level_lexicon(args.lang, level, cases)
        # Only tokens the model at this level actually has can be ablated.
        active = {t for t in all_new if t < model.active_vocab_size}

        base = evaluate(model, tok, cases, lex, count_emissions=True)
        base_bpc = (level_bpc(model, tok, val_text)
                    if val_text and not args.keep_encoding else float("nan"))
        emitted  = {t: base["emissions"].get(t, 0) for t in sorted(active)}
        n_live   = sum(1 for v in emitted.values() if v)

        print(f"  ── L{level}  ({os.path.basename(ckpt)}, vocab attivo "
              f"{model.active_vocab_size}, {base['n_graded']} casi)")
        print(f"     baseline            exact {base['exact']:.0%} "
              f"({base['n_exact']}/{base['n_graded']})  score {base['score']:.2f}"
              + (f"  bpc {base_bpc:.3f}" if val_text else ""))
        print(f"     token nuovi emessi negli output: {n_live}/{len(active)}")

        level_res = {"checkpoint": ckpt, "baseline": {k: v for k, v in base.items()
                                                     if k != "emissions"},
                     "baseline_bpc": base_bpc,
                     "emissions": emitted, "conditions": {}}

        conditions = []
        if args.mode == "all":
            conditions = [("all", active)]
        elif args.mode == "cohort":
            conditions = [(f"cohort_L{l}", {t for t in ids if t in active})
                          for l, ids in cohorts.items()]
        elif args.mode == "single":
            conditions = [(f"tok_{t}", {t}) for t in sorted(active)]

        for name, banned in conditions:
            if not banned:
                continue
            res = evaluate(model, tok, cases, lex, banned=banned,
                           prune=not args.keep_encoding)
            bpc = (level_bpc(model, tok, val_text, banned)
                   if val_text and not args.keep_encoding else float("nan"))
            d_exact = res["exact"] - base["exact"]
            d_bpc   = bpc - base_bpc
            label   = name
            if args.mode == "single":
                s = tok.vocab[next(iter(banned))].decode("utf-8", errors="replace")
                label = f"{name} {s!r}"
            flag = "  ←" if abs(d_exact) >= 0.01 else ""
            print(f"     {label:<26} exact {res['exact']:.0%} "
                  f"({d_exact:+.0%})  score {res['score']:.2f} "
                  f"({res['score']-base['score']:+.2f})"
                  + (f"  bpc {bpc:.3f} ({d_bpc:+.3f})" if val_text else "") + flag)
            level_res["conditions"][name] = {
                "banned":   sorted(banned),
                "exact":    res["exact"],
                "d_exact":  d_exact,
                "score":    res["score"],
                "d_score":  res["score"] - base["score"],
                "bpc":      bpc,
                "d_bpc":    d_bpc,
            }

        results["levels"][level] = level_res
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Risultati: {args.out}")


if __name__ == "__main__":
    main()
