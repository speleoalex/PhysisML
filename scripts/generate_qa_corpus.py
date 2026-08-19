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
import json, os, argparse, random


def generate(level: int, lang: str = "it", reps: int = 20) -> None:
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

    # Generate corpus: reps repetitions of dialogues, shuffled order
    lines = []
    for _ in range(reps):
        random.shuffle(pairs)
        for pair in pairs:
            p = pair.get("prompt", "").strip()
            r = pair.get("response", "").strip()
            if p and r:
                lines.append(p)
                lines.append(r)
                lines.append("")  # blank line between dialogues

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    chars = os.path.getsize(out_path)
    print(f"  L{level}: {len(pairs)} pairs × {reps} reps = {len(lines)//3} dialogues"
          f"  → {out_path}  ({chars/1024:.0f}KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--lang", default="it")
    parser.add_argument("--reps", type=int, default=20,
                        help="Dialogue repetitions in the corpus (default: 20)")
    args = parser.parse_args()

    print(f"Generating qa_corpus.txt for levels {args.levels} ({args.reps} reps)...")
    for level in args.levels:
        generate(level, args.lang, args.reps)
    print("Done.")


if __name__ == "__main__":
    main()
