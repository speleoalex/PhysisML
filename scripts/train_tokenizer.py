"""
Train a new BPE tokenizer on the full corpus of one language.

Usage:
    python3 scripts/train_tokenizer.py                    # default: it, 8000 token
    python3 scripts/train_tokenizer.py --lang en          # the English curriculum
    python3 scripts/train_tokenizer.py --vocab-size 4000  # custom size
    python3 scripts/train_tokenizer.py --sample 50        # use 50MB of corpus
    python3 scripts/train_tokenizer.py --stats            # show corpus used
"""
import sys, os, glob, json, random, argparse, time
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, "tests", "test_1")
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physisml.tokenizer import BPETokenizer

CORPUS_ROOT = "training_files"
OUTPUT_BASE = "dynamic_model/data"


def levels_of(lang: str) -> list:
    """The levels this language has, read off the disk.

    A pinned count is what once excluded L11 (ontology) and L12 (curiosity)
    from both the tokenizer and the corpus statistics, and it would now stop
    the Italian scan one level short of L13. The disk is the only list that
    cannot go stale.
    """
    base = os.path.join(CORPUS_ROOT, lang)
    if not os.path.isdir(base):
        raise SystemExit(f"Error: no corpus for language '{lang}' ({base}/ not found)")
    return sorted(int(d) for d in os.listdir(base)
                  if d.isdigit() and os.path.isdir(os.path.join(base, d)))


def default_output(lang: str, vocab_size: int) -> str:
    """Where a language's vocabulary lives.

    Italian keeps the historical filename — every published checkpoint was
    trained against it. Any other language gets tokenizer_<lang>.json, which is
    the name dynamic_model/train_curriculum.tokenizer_path() looks for. The
    size stays out of that name on purpose: the '8k' in the Italian one has
    been wrong since the file came back holding 2590 tokens, and a name tied to
    --vocab-size would put the vocabulary where the build never looks.
    """
    if lang == "it":
        return os.path.join(OUTPUT_BASE, f"tokenizer_{vocab_size // 1000}k.json")
    return os.path.join(OUTPUT_BASE, f"tokenizer_{lang}.json")


def reference_tokenizer(lang: str) -> str:
    """The vocabulary the build uses today for this language: what the new one
    has to beat. For a language with none of its own that is the Italian file,
    which makes the comparison printed at the end the whole point of the run.
    """
    candidates = []
    if lang != "it":
        candidates.append(os.path.join(OUTPUT_BASE, f"tokenizer_{lang}.json"))
    candidates += [os.path.join(OUTPUT_BASE, "tokenizer_8k.json"),
                   os.path.join(OUTPUT_BASE, "tokenizer_base.json")]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def test_sentences(lang: str) -> list:
    """A handful of sentences to tokenise side by side, in the language being
    trained. Curriculum material, not messages — left untranslated on purpose,
    they are what the tokenizer is measured on. A language not listed here
    borrows its own gold answers from the last level it has.
    """
    fixed = {
        "it": ["il cane dorme sul tappeto",
               "la mamma cucina il pane",
               "buongiorno come stai oggi",
               "io voglio andare a casa"],
        "en": ["the dog sleeps on the carpet",
               "the mother cooks the bread",
               "the cat is small but strong",
               "i want to go home because i am tired"],
    }
    if lang in fixed:
        return fixed[lang]
    out = []
    for level in reversed(levels_of(lang)):
        path = os.path.join(CORPUS_ROOT, lang, str(level), "local_teacher.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        for step in cfg.get("steps", {}).values():
            for t in step.get("targets", []):
                if isinstance(t, dict) and t.get("expected"):
                    out.append(t["expected"])
        if out:
            break
    return sorted(set(out), key=len, reverse=True)[:4]


def collect_targets(lang: str, reps: int = 20) -> str:
    """The teacher's target pools, which ARE the curriculum.

    They live in local_teacher.json, not in .txt, so the corpus scan never saw
    them — and the vocabulary ended up not covering the words the model is
    actually taught: 49 of the 143 lexicon words came out split ('or'+'so',
    'ca'+'p'+'pe'+'llo'). Repeated a few times so they clear the merge
    threshold against a corpus hundreds of times their size.
    """
    parts = []
    for level in levels_of(lang):
        path = os.path.join(CORPUS_ROOT, lang, str(level), "local_teacher.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        for step in cfg.get("steps", {}).values():
            for t in step.get("targets", []):
                if isinstance(t, str):
                    parts.append(t)
                else:
                    parts.append(t.get("prompt", ""))
                    parts.append(t.get("expected", ""))
    text = "\n".join(p for p in parts if p)
    return "\n".join([text] * reps)


def collect_corpus(lang: str, sample_mb: int = 50) -> str:
    """
    Collect text from all levels, proportionally sampled up to sample_mb MB.
    Levels with more text contribute proportionally more to the sample.
    """
    sample_bytes = sample_mb * 1024 * 1024

    # Collect all files with their sizes
    all_files = []
    for level in levels_of(lang):
        d = os.path.join(CORPUS_ROOT, lang, str(level))
        for fpath in sorted(glob.glob(os.path.join(d, "*.txt"))):
            if "teacher_prompt" not in fpath:
                size = os.path.getsize(fpath)
                if size > 100:
                    all_files.append((fpath, size))

    total_size = sum(s for _, s in all_files)
    print(f"\n  Files found: {len(all_files)}  ({total_size/1e6:.1f} MB total)")

    if total_size <= sample_bytes:
        # Use everything
        parts = []
        for fpath, _ in all_files:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
            print(f"  + {os.path.relpath(fpath):<55} {os.path.getsize(fpath)/1e6:.1f}MB")
        return "\n\n".join(parts)

    # Sample proportionally by level weight
    collected = []
    collected_size = 0
    random.seed(42)

    for fpath, size in all_files:
        # Quota for this file: proportional to its share of total corpus
        quota = int(sample_bytes * size / total_size)
        if quota < 1000:
            quota = min(size, 1000)   # at least 1KB from every file

        with open(fpath, encoding="utf-8", errors="replace") as f:
            text = f.read()

        if len(text) <= quota:
            sample = text
        else:
            # Random window
            start = random.randint(0, max(0, len(text) - quota))
            sample = text[start:start + quota]

        collected.append(sample)
        collected_size += len(sample)
        print(f"  + {os.path.relpath(fpath):<55} {len(sample)/1e6:.2f}MB")

    print(f"\n  Total sample: {collected_size/1e6:.1f} MB")
    return "\n\n".join(collected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       default="it",
                        help="Language to train on: reads training_files/<lang>/ "
                             "(default: it)")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--no-targets", action="store_true",
                        help="Do not add the teacher target pools to the corpus")
    parser.add_argument("--sample",     type=int, default=50,
                        help="Max MB of corpus to use for training (default: 50)")
    parser.add_argument("--output",     default=None,
                        help="Output path (default: dynamic_model/data/tokenizer_{V}k.json)")
    parser.add_argument("--stats",      action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = default_output(args.lang, args.vocab_size)

    levels = levels_of(args.lang)

    if args.stats:
        print(f"  Language: {args.lang}  "
              f"({CORPUS_ROOT}/{args.lang}, levels {levels[0]}-{levels[-1]})")
        collect_corpus(args.lang, args.sample)
        return

    print(f"\n{'='*60}")
    print(f"  BPE tokenizer training")
    print(f"  Language   : {args.lang}  ({CORPUS_ROOT}/{args.lang}, "
          f"levels {levels[0]}-{levels[-1]})")
    print(f"  Vocab size : {args.vocab_size}")
    print(f"  Sample     : up to {args.sample} MB")
    print(f"  Output     : {args.output}")
    print(f"{'='*60}")

    print(f"\n  Collecting corpus...")
    text = collect_corpus(args.lang, args.sample)
    if not args.no_targets:
        tgt = collect_targets(args.lang)
        print(f"  + teacher target pools: {len(tgt):,} chars")
        text = text + "\n" + tgt
    print(f"  Corpus ready: {len(text):,} characters")

    print(f"\n  BPE training ({args.vocab_size} tokens)...")
    print(f"  (estimate: 3-10 minutes)")
    t0 = time.time()

    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size)

    # Register EOS immediately above the last BPE id, so the file always has
    # one. The vocabulary retrained on 2026-08-25 did NOT, and the whole EOS
    # machinery downstream is keyed on tok.get_special_id(EOS_TOKEN) being
    # non-None: TrainerB.generate stops on it, train_curriculum appends it to
    # every gold, export_gguf declares tokenizer.ggml.eos_token_id. With an
    # empty special_tokens all of that silently does nothing, and the model
    # has no way to say 'answer finished' outside the Python generator.
    #
    # The id is max+1 and not a reserved low slot on purpose: _sync_vocab_rows
    # activates every embedding row up to max(vocab)+1, so an EOS placed above
    # that would leave activated rows with no token behind them — emittable by
    # the sampler, unresolvable by decode(). This makes the file vocab_size+1
    # tokens long (8000 BPE + EOS@8000), which is the convention
    # train_curriculum.py already documents for tokenizer_8k.json.
    tok.register_special_token(BPETokenizer.EOS_TOKEN, max(tok.vocab.keys()) + 1)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")
    print(f"  Total tokens: {len(tok)}  "
          f"({len(tok) - 1} BPE + {BPETokenizer.EOS_TOKEN}@"
          f"{tok.get_special_id(BPETokenizer.EOS_TOKEN)})")

    # Show sample of new tokens
    multi_char = []
    for i in range(len(tok.vocab)):
        try:
            s = tok.decode([i])
            s_clean = s.replace('\ufffd', '')
            if len(s_clean) >= 3:
                multi_char.append((i, s_clean))
        except Exception:
            pass

    print(f"  Tokens with 3+ characters: {len(multi_char)}")
    print(f"  Examples (first 20 words):")
    shown = 0
    for i, s in multi_char:
        if s.strip() and s.replace(' ', '').isalpha():
            print(f"    [{i:5d}] {repr(s)}")
            shown += 1
            if shown >= 20:
                break

    # Compare with the vocabulary the build uses today for this language.
    ref_path = reference_tokenizer(args.lang)
    if ref_path:
        base_tok = BPETokenizer()
        base_tok.load(ref_path)
        print(f"\n  Tokenisation comparison "
              f"({os.path.basename(ref_path)} {len(base_tok)} vs new {len(tok)}):")
    else:
        base_tok = None
        print(f"\n  Tokenisation of the new {len(tok)}-token vocabulary:")
    for s in test_sentences(args.lang):
        new_ids = tok.encode(s)
        print(f"  '{s}'")
        if base_tok is not None:
            base_ids = base_tok.encode(s)
            print(f"    ref  ({len(base_ids)} tokens): "
                  f"{[base_tok.decode([i]) for i in base_ids]}")
        print(f"    new  ({len(new_ids)} tokens): {[tok.decode([i]) for i in new_ids]}")
        print()

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    tok.save(args.output)
    print(f"  Saved: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
