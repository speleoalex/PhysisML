"""
Download and distribute Italian text corpus across curriculum levels.

Sources:
  1. Wikipedia IT  — from HuggingFace datasets or local dump
  2. OpenSubtitles IT — from OPUS (conversational Italian)
  3. EuroParl IT   — from OPUS (formal Italian)

Levels assigned by Gulpease readability index:
  > 80  → L3-L4 (elementary)
  60-80 → L5-L7 (middle school)
  40-60 → L8-L9 (high school)
  < 40  → L10   (university)

Usage:
  # Show what would be downloaded without doing anything
  python3 scripts/build_corpus.py --dry-run

  # Download Wikipedia only (recommended first step)
  python3 scripts/build_corpus.py --source wikipedia

  # Download everything
  python3 scripts/build_corpus.py --all

  # Only classify and distribute already-downloaded data
  python3 scripts/build_corpus.py --distribute-only

  # Show corpus stats
  python3 scripts/build_corpus.py --stats
"""

import os, sys, re, gzip, json, math, argparse, subprocess, glob, shutil
from collections import defaultdict

CORPUS_BASE   = "training_files/it"
DOWNLOAD_DIR  = "corpus_raw"          # temporary download area
WIKI_DUMP_URL = "https://dumps.wikimedia.org/itwiki/latest/itwiki-latest-pages-articles.xml.bz2"
OPUS_SUBS_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/it.txt.gz"
OPUS_EURO_URL = "https://object.pouta.csc.fi/OPUS-Europarl/v8/mono/it.txt.gz"

# Target token counts per level (1 token ≈ 4-5 chars for Italian)
TARGET_TOKENS = {
    0:  50_000,
    1:  500_000,
    2:  5_000_000,
    3:  15_000_000,
    4:  20_000_000,
    5:  25_000_000,
    6:  30_000_000,
    7:  30_000_000,
    8:  35_000_000,
    9:  25_000_000,
    10: 15_000_000,
}

# ── Readability ──────────────────────────────────────────────────────────────

def gulpease(text: str) -> float:
    """
    Compute Gulpease readability index for Italian text.
    Higher = easier. Range 0-100.
    """
    words = text.split()
    if not words:
        return 50.0
    n_words = len(words)
    n_chars  = sum(len(w) for w in words)
    # Count sentences (end with . ! ?)
    n_sentences = max(1, len(re.findall(r'[.!?]+', text)))
    score = 89 + (300 * n_sentences - 10 * n_chars) / n_words
    return max(0.0, min(100.0, score))


def gulpease_to_level(score: float) -> int:
    """Map Gulpease score to curriculum level."""
    if score > 85:
        return 3
    elif score > 78:
        return 4
    elif score > 70:
        return 5
    elif score > 63:
        return 6
    elif score > 55:
        return 7
    elif score > 47:
        return 8
    elif score > 42:
        return 9
    else:
        return 10


# ── Stats ────────────────────────────────────────────────────────────────────

def count_chars(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def show_stats():
    """Print current corpus stats per level."""
    print(f"\n{'─'*65}")
    print(f"  ITALIAN CORPUS — current statistics")
    print(f"{'─'*65}")
    print(f"  {'Lev':>4}  {'File':>4}  {'MB':>8}  {'Tokens (est)':>14}  {'Target':>14}  {'%':>6}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*6}")

    total_chars = 0
    for level in range(11):
        d = os.path.join(CORPUS_BASE, str(level))
        files = [f for f in glob.glob(os.path.join(d, "*.txt"))
                 if "teacher_prompt" not in f]
        chars = sum(count_chars(f) for f in files)
        total_chars += chars
        tokens  = chars // 4                              # ~4 chars/token for Italian
        target  = TARGET_TOKENS.get(level, 0)
        pct     = int(100 * tokens / target) if target else 0
        bar     = "█" * min(20, int(pct / 5))
        print(f"  L{level:>2}:  {len(files):>4}  {chars/1e6:>8.2f}  "
              f"{tokens:>14,}  {target:>14,}  {pct:>5}%  {bar}")

    print(f"  {'─'*65}")
    print(f"  Total: {total_chars/1e6:.1f} MB  (~{total_chars//4:,} tokens)\n")


# ── Download helpers ─────────────────────────────────────────────────────────

def check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def download_file(url: str, dest: str) -> bool:
    """Download with wget or curl."""
    if os.path.exists(dest):
        print(f"  Already present: {dest}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Download: {url}")
    print(f"  → {dest}")
    if check_tool("wget"):
        ret = subprocess.call(["wget", "-q", "--show-progress", "-O", dest, url])
    elif check_tool("curl"):
        ret = subprocess.call(["curl", "-L", "-o", dest, url])
    else:
        print("  ERROR: wget or curl not found.")
        return False
    return ret == 0


# ── Wikipedia processing ─────────────────────────────────────────────────────

def process_wikipedia():
    """
    Download Wikipedia IT dump and distribute articles by Gulpease level.
    Uses wikiextractor if available, otherwise falls back to simple XML parsing.
    """
    dump_path = os.path.join(DOWNLOAD_DIR, "itwiki-latest-pages-articles.xml.bz2")

    print("\n=== Italian Wikipedia ===")

    # Check if wikiextractor is available
    has_extractor = check_tool("wikiextractor") or _has_python_module("wikiextractor")

    if not has_extractor:
        print("  wikiextractor not found.")
        print("  Install: pip3 install wikiextractor")
        print("  Or use --source huggingface to download via datasets.")
        print()

    # Try HuggingFace datasets (simplest, no dump needed)
    if _has_python_module("datasets"):
        print("  Using HuggingFace datasets (recommended method)...")
        _process_wikipedia_hf()
        return

    # Download dump
    if not download_file(WIKI_DUMP_URL, dump_path):
        print("  Download failed.")
        return

    if has_extractor:
        _process_wikipedia_dump(dump_path)
    else:
        print("  Dump downloaded but wikiextractor missing.")
        print(f"  To extract: pip3 install wikiextractor")
        print(f"  Then rerun: python3 scripts/build_corpus.py --source wikipedia")


def _has_python_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _process_wikipedia_hf():
    """Download and process Italian Wikipedia via HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip3 install datasets")
        return

    print("  Loading Italian Wikipedia from HuggingFace (may take a while)...")
    ds = load_dataset("wikipedia", "20220301.it", split="train", trust_remote_code=True)

    # Group articles by level
    level_buffers = defaultdict(list)
    level_chars   = defaultdict(int)

    print(f"  {len(ds):,} articles to classify...")
    for i, article in enumerate(ds):
        text  = article.get("text", "").strip()
        if len(text) < 100:
            continue
        score = gulpease(text[:2000])   # use first 2000 chars for speed
        level = gulpease_to_level(score)
        level_buffers[level].append(text)
        level_chars[level] += len(text)

        if (i + 1) % 50000 == 0:
            print(f"  {i+1:,}/{len(ds):,} articles processed...")

    # Write per-level files
    _flush_level_buffers(level_buffers, "wikipedia_hf")
    _print_distribution(level_chars, "Wikipedia HF")


def _process_wikipedia_dump(dump_path: str):
    """Process a downloaded Wikipedia XML dump with wikiextractor."""
    extract_dir = os.path.join(DOWNLOAD_DIR, "wiki_extracted")
    if not os.path.exists(extract_dir):
        print("  Extracting with wikiextractor...")
        subprocess.call([
            sys.executable, "-m", "wikiextractor.WikiExtractor",
            dump_path, "-o", extract_dir, "--json", "--quiet"
        ])

    print("  Classifying articles...")
    level_buffers = defaultdict(list)
    level_chars   = defaultdict(int)

    for jsonl_file in glob.glob(os.path.join(extract_dir, "**/wiki_*"), recursive=True):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                try:
                    obj  = json.loads(line)
                    text = obj.get("text", "").strip()
                    if len(text) < 100:
                        continue
                    score = gulpease(text[:2000])
                    level = gulpease_to_level(score)
                    level_buffers[level].append(text)
                    level_chars[level] += len(text)
                except (json.JSONDecodeError, KeyError):
                    continue

    _flush_level_buffers(level_buffers, "wikipedia_dump")
    _print_distribution(level_chars, "Wikipedia dump")


def _flush_level_buffers(buffers: dict, source_name: str):
    """Write buffered articles to training_files per level."""
    for level, articles in buffers.items():
        if level < 2 or level > 10:
            continue
        dest_dir  = os.path.join(CORPUS_BASE, str(level))
        dest_file = os.path.join(dest_dir, f"{source_name}_L{level}.txt")
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_file, "w", encoding="utf-8") as f:
            for art in articles:
                f.write(art.strip() + "\n\n")
        chars = os.path.getsize(dest_file)
        print(f"  L{level}: {dest_file}  ({chars/1e6:.1f} MB, {len(articles):,} articles)")


def _print_distribution(level_chars: dict, label: str):
    print(f"\n  Distribution {label}:")
    for level in sorted(level_chars):
        chars = level_chars[level]
        print(f"    L{level}: {chars/1e6:.1f} MB  (~{chars//4:,} tokens)")


# ── OpenSubtitles ────────────────────────────────────────────────────────────

def process_opensubtitles():
    """Download and distribute OpenSubtitles IT across L2-L5."""
    print("\n=== Italian OpenSubtitles ===")
    gz_path = os.path.join(DOWNLOAD_DIR, "opensubtitles_it.txt.gz")

    if not download_file(OPUS_SUBS_URL, gz_path):
        return

    print("  Extracting and distributing...")
    # OpenSubtitles lines are short — good for L2-L4 (conversational)
    # Distribute by line length as a simple difficulty proxy
    level_buffers = {2: [], 3: [], 4: [], 5: []}
    level_chars   = defaultdict(int)

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= 1000:
                text  = " ".join(batch)
                score = gulpease(text)
                level = min(5, max(2, gulpease_to_level(score)))
                level_buffers[level].extend(batch)
                level_chars[level] += sum(len(l) for l in batch)
                batch = []

    for level, lines in level_buffers.items():
        if not lines:
            continue
        dest_dir  = os.path.join(CORPUS_BASE, str(level))
        dest_file = os.path.join(dest_dir, f"opensubtitles_L{level}.txt")
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        chars = os.path.getsize(dest_file)
        print(f"  L{level}: {dest_file}  ({chars/1e6:.1f} MB)")

    _print_distribution(level_chars, "OpenSubtitles")


# ── EuroParl ────────────────────────────────────────────────────────────────

def process_europarl():
    """Download EuroParl IT — formal Italian for L8-L10."""
    print("\n=== Italian EuroParl ===")
    gz_path = os.path.join(DOWNLOAD_DIR, "europarl_it.txt.gz")

    if not download_file(OPUS_EURO_URL, gz_path):
        return

    print("  Distributing to L8-L10 (formal Italian)...")
    level_buffers = defaultdict(list)

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= 500:
                text  = " ".join(batch)
                score = gulpease(text)
                level = max(8, min(10, gulpease_to_level(score)))
                level_buffers[level].extend(batch)
                batch = []

    for level, lines in level_buffers.items():
        dest_dir  = os.path.join(CORPUS_BASE, str(level))
        dest_file = os.path.join(dest_dir, f"europarl_L{level}.txt")
        os.makedirs(dest_dir, exist_ok=True)
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        chars = os.path.getsize(dest_file)
        print(f"  L{level}: {dest_file}  ({chars/1e6:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Italian training corpus")
    parser.add_argument("--source",           choices=["wikipedia", "opensubtitles", "europarl"])
    parser.add_argument("--all",              action="store_true", help="Download all sources")
    parser.add_argument("--dry-run",          action="store_true", help="Show plan without downloading")
    parser.add_argument("--stats",            action="store_true", help="Show current corpus stats")
    parser.add_argument("--distribute-only",  action="store_true",
                        help="Classify already-downloaded raw data without re-downloading")
    args = parser.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if args.stats or (not args.source and not args.all and not args.distribute_only):
        show_stats()
        if not args.stats:
            print("  Use --source wikipedia|opensubtitles|europarl  or  --all")
            print("  Use --dry-run to preview the plan without downloading")
        return

    if args.dry_run:
        print("\n=== DOWNLOAD PLAN (dry-run) ===")
        print(f"  Wikipedia IT:     {WIKI_DUMP_URL}")
        print(f"    → ~450MB download, ~250M tokens, levels L3-L10")
        print(f"  OpenSubtitles IT: {OPUS_SUBS_URL}")
        print(f"    → ~700MB download, ~120M tokens, levels L2-L5")
        print(f"  EuroParl IT:      {OPUS_EURO_URL}")
        print(f"    → ~50MB download, ~10M tokens, levels L8-L10")
        print(f"\n  Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
        print(f"  Required free space: ~1.5GB")
        print(f"\n  Python requirements:")
        print(f"    pip3 install datasets wikiextractor")
        show_stats()
        return

    if args.source == "wikipedia" or args.all:
        process_wikipedia()

    if args.source == "opensubtitles" or args.all:
        process_opensubtitles()

    if args.source == "europarl" or args.all:
        process_europarl()

    show_stats()


if __name__ == "__main__":
    main()
