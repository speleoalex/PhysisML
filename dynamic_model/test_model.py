"""
Quick model statistics after a build.
Runs test prompts appropriate for the level and reports quality metrics.

Usage:
    python3 dynamic_model/test_model.py --level 0
    python3 dynamic_model/test_model.py --level 1 --checkpoint models/active.pt
"""
import sys, os, argparse, re, json
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
from physisml.torch_model import TorchGPT
from physisml.tokenizer   import BPETokenizer
from dynamic_model.exp_b.affect_state import AffectState
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.trainer      import TrainerB
from physisml.torch_model import TorchAdamOptimizer

torch.set_num_threads(12)

# Test prompts and expected vocabulary per level
LEVEL_CONFIG = {
    0: {
        "prompts": ["ma", "pa", "oh", "sì", "na na"],
        "vocab":   ["ma", "pa", "ta", "la", "na", "ba", "da", "oh", "ah", "sì", "no"],
        "desc":    "Fonemi e sillabe",
    },
    1: {
        "prompts": ["mamma", "il cane", "ciao", "sì no", "papà"],
        "vocab":   ["mamma", "papà", "cane", "gatto", "pane", "latte", "acqua",
                    "sole", "luna", "bello", "sì", "no", "ciao", "bravo"],
        "desc":    "Parole singole e famiglia",
    },
    2: {
        "prompts": ["il cane dorme", "la casa è", "buongiorno", "il gatto"],
        "vocab":   ["il", "la", "un", "una", "è", "sono", "dorme", "mangia",
                    "beve", "bello", "grande", "piccolo", "casa", "cane", "gatto"],
        "desc":    "Frasi semplici",
    },
    3: {
        "prompts": ["io sono", "dove vai", "hai fame", "il bambino"],
        "vocab":   ["io", "tu", "lui", "sono", "sei", "è", "ho", "hai",
                    "vado", "vai", "bene", "male", "fame", "sete"],
        "desc":    "Grammatica base",
    },
}


def level_description(lang: str, level: int) -> str:
    """The level's own description, from its local_teacher.json.

    LEVEL_CONFIG only ever had entries 0-3, and the fallback was
    LEVEL_CONFIG[1]: every level from 4 up printed 'Parole singole e famiglia'
    as its title. The teacher config is the source of truth for what a level
    teaches, and it is the same file load_level_cases() already reads.
    """
    path = os.path.join("training_files", lang, str(level), "local_teacher.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                desc = json.load(f).get("description", "")
            if desc:
                return desc
        except (json.JSONDecodeError, OSError):
            pass
    return LEVEL_CONFIG.get(level, {}).get("desc", f"Livello {level}")


def load_level_cases(lang: str, level: int) -> list:
    """
    Build the test cases from the level's ACTUAL curriculum
    (training_files/{lang}/{level}/local_teacher.json), falling back to the
    hardcoded LEVEL_CONFIG when no teacher config exists.

    The hardcoded prompts drifted away from what the levels teach — L3 was
    probed with 'io sono' / 'dove vai' / 'hai fame' while its curriculum is
    'il cane dorme!', 'la mamma mangia!', 'physisml!', numbers and colours —
    so a perfectly trained model scored 0%.

    Returns a list of (prompt, expected_or_None).
    """
    path = os.path.join("training_files", lang, str(level), "local_teacher.json")
    cases = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            # Mirror LocalTeacher._build_prompt: the runtime prompt is the
            # step's prompt_template applied to the target, NOT the raw
            # 'prompt' field. Most steps use 'di: {prompt}', so probing with
            # the bare target asks a question the model never saw — L2 scored
            # 0% that way while answering 'di: il cane' -> 'il cane!'
            # correctly. Retry/encouragement prefixes are left out on purpose:
            # the clean template is the canonical form.
            for step in cfg.get("steps", {}).values():
                tmpl     = step.get("prompt_template", "di {target}")
                tmpl_exp = step.get("expected_template", "{target}!")
                for t in step.get("targets", []):
                    if isinstance(t, dict):
                        pv  = t.get("prompt") or t.get("noun") or ""
                        exp = (t.get("expected") or (pv + "!")).strip()
                    elif isinstance(t, str):
                        pv  = t
                        exp = tmpl_exp.format(target=t)
                    else:
                        continue
                    if not pv:
                        continue
                    cases.append((tmpl.format(target=pv, prompt=pv), exp))
        except (json.JSONDecodeError, OSError):
            cases = []
    if not cases:
        cfg = LEVEL_CONFIG.get(level, LEVEL_CONFIG[1])
        cases = [(p, None) for p in cfg["prompts"]]
    return cases


def level_lexicon(lang: str, level: int, cases: list) -> list:
    """Words the level actually teaches (from its gold answers)."""
    lex = set()
    for _, exp in cases:
        for w in re.findall(r"[\w'àèéìòùÀÈÉÌÒÙ]+", (exp or "").lower()):
            if w:
                lex.add(w)
    if not lex:
        lex = set(LEVEL_CONFIG.get(level, LEVEL_CONFIG[1])["vocab"])
    return sorted(lex)


def score_output(text: str, vocab: list) -> float:
    """
    Fraction of the OUTPUT's words that belong to the level lexicon.

    The old version divided by len(vocab), i.e. it asked a single short answer
    to contain the WHOLE level vocabulary: with 14 words listed, a correct
    3-word reply could not exceed 21% and usually scored 0%. It also used a
    bare substring test ('è' matches inside any word), the same defect fixed
    in the teacher's keyword check.
    """
    words = re.findall(r"[\w'àèéìòùÀÈÉÌÒÙ]+", text.lower())
    if not words:
        return 0.0
    known = set(w.lower() for w in vocab)
    return sum(1 for w in words if w in known) / len(words)


def is_exact(text: str, expected: str) -> bool:
    """Normalised exact match against the gold answer."""
    if not expected:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s.replace("<|EOS|>", "")).strip().lower()
    a, b = norm(text), norm(expected)
    return a == b or a == b.rstrip("!.?")


def clean_output(text: str) -> str:
    """Remove excessive whitespace for display."""
    return re.sub(r"\s+", " ", text).strip()[:60]


def main():
    parser = argparse.ArgumentParser(description="Test model quality after build")
    parser.add_argument("--level",      type=int, default=0)
    parser.add_argument("--lang",       default="it")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer",  default=None,
                        help="Tokenizer JSON (auto-detected from checkpoint dir if omitted)")
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument("--top-k",      type=int, default=20)
    parser.add_argument("--samples",    type=int, default=12,
                        help="Max test cases sampled from the curriculum (0 = all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy/deterministic (default). >0 samples, "
                             "making results irreproducible run to run.")
    parser.add_argument("--repeats",    type=int, default=1,
                        help="Attempts per case when sampling; a case counts "
                             "as correct if any attempt matches.")
    args = parser.parse_args()

    # Resolve checkpoint
    ckpt = args.checkpoint
    if not ckpt:
        for candidate in [
            f"models/checkpoints/{args.lang}/level_{args.level}/final_learned.pt",
            f"models/checkpoints/{args.lang}/level_{args.level}/final.pt",
            "models/active.pt",
        ]:
            if os.path.exists(candidate):
                ckpt = candidate
                break
    if not ckpt or not os.path.exists(ckpt):
        print("Nessun checkpoint trovato.")
        sys.exit(1)

    # Load model
    model = TorchGPT.load(ckpt)

    # Auto-detect tokenizer: prefer level-specific, then 8K base, then 500-token base
    _tok_path = args.tokenizer
    if not _tok_path:
        _ckpt_dir = os.path.dirname(ckpt)
        _candidates = [
            os.path.join(_ckpt_dir, "tokenizer.json"),
            "models/active_tokenizer.json",
            "dynamic_model/data/tokenizer_8k.json",
            "dynamic_model/data/tokenizer_base.json",
        ]
        for _c in _candidates:
            if os.path.exists(_c):
                _t = BPETokenizer(); _t.load(_c)
                if len(_t) >= model.active_vocab_size:
                    _tok_path = _c
                    break
        if not _tok_path:
            _tok_path = "dynamic_model/data/tokenizer_8k.json"
    tok = BPETokenizer()
    tok.load(_tok_path)
    print(f"  Tokenizer: {_tok_path}  (vocab={len(tok)})")
    affect = AffectState()
    mod    = AffectModulator(affect)
    opt    = TorchAdamOptimizer(model.parameters(), lr=1e-4)
    trainer = TrainerB(model, tok, opt, affect, mod)

    cfg   = LEVEL_CONFIG.get(args.level, LEVEL_CONFIG[1])
    cases = load_level_cases(args.lang, args.level)
    lex   = level_lexicon(args.lang, args.level, cases)
    if args.samples and args.samples < len(cases):
        step = max(1, len(cases) // args.samples)
        cases = cases[::step][:args.samples]

    print(f"\n{'─'*68}")
    print(f"  TEST MODELLO — Livello {args.level}: {level_description(args.lang, args.level)}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Params: {model.num_params:,}   Vocab: {model.vocab_size}")
    _mode = "greedy (deterministico)" if args.temperature <= 0.0 else \
            f"campionamento T={args.temperature} x{args.repeats}"
    print(f"  Casi: {len(cases)} dal curriculum   Lessico: {len(lex)} parole   Decoding: {_mode}")
    print(f"{'─'*68}")

    scores, confidences, exacts = [], [], []

    for prompt, expected in cases:
        # Generate. Budget sized on the gold answer, mirroring the teaching
        # loop — a fixed max_tokens truncates the longer answers of high levels.
        n_exp = len(tok.encode(expected)) if expected else 0
        # Greedy by default: an evaluation metric must be reproducible. With
        # temperature 0.8 and 12 cases the same checkpoint scored 92% and 75%
        # on two consecutive runs — pure sampling noise.
        greedy = args.temperature <= 0.0
        attempts = 1 if greedy else max(1, args.repeats)
        generated, sc, ok = "", 0.0, False
        for _ in range(attempts):
            out = trainer.generate(
                prompt,
                max_tokens=int(max(args.max_tokens, min(2 * n_exp + 12, 120))),
                base_temperature=(1.0 if greedy else args.temperature),
                top_k=(1 if greedy else args.top_k),
                min_tokens=max(4, min(n_exp, 40)) if n_exp else 4,
                stop_after=max(0, min(n_exp - 1, 40)) if n_exp else 2,
            )
            cand = out[len(prompt):].strip()
            cand_ok = is_exact(cand, expected) if expected else False
            if not generated or cand_ok:
                generated = cand
                sc = score_output(cand, lex)
                ok = cand_ok
            if cand_ok:
                break
        display = clean_output(generated)
        scores.append(sc); exacts.append(ok)
        confidences.append(affect.confidence)

        mark = "✓" if ok else (" " if expected else "·")
        bar  = f"{'█' * int(sc * 20):<20}"
        exp_s = f"  atteso {expected!r}" if expected and not ok else ""
        print(f"  {mark} {repr(prompt):26s} → {repr(display):34s} [{bar}] {sc:.0%}{exp_s}")

    avg_score = float(np.mean(scores))
    avg_conf  = float(np.mean(confidences))
    n_exact   = sum(exacts)
    n_graded  = sum(1 for _, e in cases if e)
    exact_rate = n_exact / n_graded if n_graded else 0.0

    print(f"{'─'*68}")
    if n_graded:
        print(f"  Risposte esatte:           {n_exact}/{n_graded} = {exact_rate:.0%}  "
              f"({'ottimo' if exact_rate > 0.6 else 'buono' if exact_rate > 0.3 else 'da migliorare'})")
    print(f"  Parole del livello in output: {avg_score:.0%}")
    print(f"  Confidence media:          {avg_conf:.2f}")
    print(f"  Stato affettivo:           {affect}")
    print(f"{'─'*68}\n")

    # Exit code from exact match when a gold answer exists, else from lexicon use
    sys.exit(0 if (exact_rate if n_graded else avg_score) > 0.1 else 1)


if __name__ == "__main__":
    main()
