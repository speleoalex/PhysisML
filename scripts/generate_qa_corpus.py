"""
Generate a dialogue corpus file from QA pairs.

Converts qa_pairs.jsonl into dialogue text to add to the text training
corpus, so the model sees prompt→response pairs during phase 0 (text
training) as well as in the N2.5 dream.

Formato output (training_files/it/{level}/qa_corpus.txt):
    di: il cane dorme
    il cane dorme.

    cosa fa il cane?
    il cane dorme.

Usage:
    python3 scripts/generate_qa_corpus.py --levels 0 1 2 3 --reps 20
"""
import json, os, sys, argparse, random


CORPUS_SHUFFLE_SEED = 20260824   # keep in step with train_curriculum.py


def generate(level: int, lang: str = "it", reps: int = 20,
             quiet: bool = False) -> None:
    qa_path = os.path.join("training_files", lang, str(level), "qa_pairs.jsonl")
    out_path = os.path.join("training_files", lang, str(level), "qa_corpus.txt")

    if not os.path.exists(qa_path):
        print(f"  L{level}: no qa_pairs.jsonl — skip")
        return

    pairs = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not pairs:
        print(f"  L{level}: qa_pairs empty — skip")
        return

    # Generate corpus: reps repetitions of dialogues, shuffled order.
    # A dedicated RNG with a fixed seed, not the global one: qa_corpus.txt is
    # committed, so its contents must depend only on qa_pairs.jsonl and the
    # level. Shuffling the global stream would make the file differ between
    # machines and between runs for no reason. Must stay in step with
    # _CORPUS_SHUFFLE_SEED in dynamic_model/train_curriculum.py, which
    # regenerates the same file during the dream.
    rng = random.Random(CORPUS_SHUFFLE_SEED + level)
    lines = []
    for _ in range(reps):
        rng.shuffle(pairs)
        for pair in pairs:
            p = pair.get("prompt", "").strip()
            r = pair.get("response", "").strip()
            if p and r:
                lines.append(p)
                lines.append(r)
                lines.append("")  # blank line between dialogues

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if not quiet:
        chars = os.path.getsize(out_path)
        print(f"  L{level}: {len(pairs)} pairs × {reps} reps = {len(lines)//3}"
              f" dialogues  → {out_path}  ({chars/1024:.0f}KB)")


def check(level: int, lang: str = "it", reps: int = 20) -> bool:
    """Is the committed qa_corpus.txt the one qa_pairs.jsonl would produce?

    qa_corpus.txt is a derived file and is committed; qa_pairs.jsonl is its
    source. They drifted once already (2026-08-24: the L1 and L2 corpora had
    been built from an older pair set), which means a fresh clone trained on
    different data than the machine that produced the published numbers. Now
    that generation is deterministic, that question has a yes/no answer.
    """
    import tempfile, filecmp, shutil
    out = os.path.join("training_files", lang, str(level), "qa_corpus.txt")
    if not os.path.exists(out):
        print(f"  L{level}: qa_corpus.txt missing")
        return False
    with tempfile.TemporaryDirectory() as td:
        keep = os.path.join(td, "keep.txt")
        shutil.copy2(out, keep)
        generate(level, lang, reps, quiet=True)     # rewrites out
        same = filecmp.cmp(keep, out, shallow=False)
        if not same:
            shutil.copy2(keep, out)                 # leave the tree untouched
    print(f"  L{level}: {'in step' if same else 'STALE — regenerate'}")
    return same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--lang", default="it")
    parser.add_argument("--reps", type=int, default=20,
                        help="Dialogue repetitions in the corpus (default: 20)")
    parser.add_argument("--check", action="store_true",
                        help="Verify each qa_corpus.txt matches its "
                             "qa_pairs.jsonl instead of writing. Exits 1 if any "
                             "is stale.")
    args = parser.parse_args()

    if args.check:
        print(f"Checking qa_corpus.txt against qa_pairs.jsonl "
              f"({args.reps} reps)...")
        results = [check(l, args.lang, args.reps) for l in args.levels]
        if all(results):
            print("All in step.")
            return
        print("Stale corpora found. Regenerate with:")
        print(f"  python3 scripts/generate_qa_corpus.py --levels "
              f"{' '.join(str(l) for l in args.levels)} --lang {args.lang}")
        sys.exit(1)

    print(f"Generating qa_corpus.txt for levels {args.levels} ({args.reps} reps)...")
    for level in args.levels:
        generate(level, args.lang, args.reps)
    print("Done.")


if __name__ == "__main__":
    main()
