"""
Download and clean Italian Wikipedia articles for training.
Uses the Wikipedia REST API — no auth required.

Usage:
    python3 scripts/download_wikipedia.py --level 1
    python3 scripts/download_wikipedia.py --level 2 --output training_files/it/2/wikipedia.txt
    python3 scripts/download_wikipedia.py --topics cane gatto sole luna --output out.txt
"""
import sys, os, argparse, time, re, json
import urllib.request
import urllib.parse

# ---------------------------------------------------------------------------
# Predefined topic lists per level
# ---------------------------------------------------------------------------

TOPICS = {
    1: [
        # Animals
        "Cane", "Gatto", "Uccello", "Pesce", "Coniglio", "Cavallo",
        "Mucca", "Pecora", "Gallina", "Ape",
        # Family & body
        "Famiglia", "Bambino", "Mano", "Occhio", "Naso",
        # Food & nature
        "Pane", "Latte", "Acqua", "Frutta", "Sole", "Luna", "Fiore", "Albero",
        # Simple objects
        "Casa", "Porta", "Finestra", "Tavolo", "Sedia",
    ],
    2: [
        # Simple stories & themes
        "Fiaba", "Favola", "Pinocchio", "Cappuccetto Rosso", "Cenerentola",
        # Nature
        "Primavera", "Estate", "Autunno", "Inverno", "Pioggia", "Neve",
        # Animals (more)
        "Leone", "Elefante", "Farfalla", "Tartaruga", "Delfino",
        # Food
        "Pasta", "Pizza", "Gelato", "Frutto",
        # Places
        "Scuola", "Parco", "Mare", "Montagna", "Fiume",
    ],
    3: [
        # Science & culture (simple articles)
        "Fotosintesi", "Respiro", "Digestion", "Cuore",
        # Italian culture
        "Carnevale", "Natale", "Pasqua",
        # Geography
        "Italia", "Roma", "Milano", "Venezia", "Toscana",
        # Literature
        "Favole di Esopo", "Carlo Collodi", "Edmondo De Amicis",
    ],
    4: [
        # History & literature
        "Alessandro Manzoni", "Dante Alighieri", "Francesco Petrarca",
        "Rinascimento", "Risorgimento italiano",
        # Science
        "Leonardo da Vinci", "Galileo Galilei",
        # Geography
        "Appennini", "Po (fiume)", "Sicilia", "Sardegna",
    ],
}

# ---------------------------------------------------------------------------
# Wikipedia API
# ---------------------------------------------------------------------------

def fetch_summary(title: str, lang: str = "it") -> str:
    """Fetch article summary (intro only) in plain text."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhysisML/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            return data.get("extract", "").strip()
    except Exception as e:
        return ""


def fetch_full(title: str, lang: str = "it", max_chars: int = 8000) -> str:
    """Fetch full article text in plain text (via action API)."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "format": "json",
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhysisML/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            if text:
                return clean_text(text)[:max_chars]
    except Exception:
        pass
    return ""


def clean_text(text: str) -> str:
    """Remove Wikipedia markup artifacts."""
    # Remove section headers that are too long/metadata
    text = re.sub(r"={2,}[^=]+={2,}", "", text)
    # Remove citation markers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove lines that are only punctuation/numbers
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l and not re.match(r"^[\d\s\W]+$", l)]
    # Remove very short lines (likely navigation/metadata)
    lines = [l for l in lines if len(l) > 20]
    return "\n\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level",  type=int, default=None,
                        help="Curriculum level (determines topic list)")
    parser.add_argument("--topics", nargs="+", default=None,
                        help="Custom list of Wikipedia article titles")
    parser.add_argument("--output", default=None,
                        help="Output file (default: training_files/it/{level}/wikipedia.txt)")
    parser.add_argument("--lang",   default="it")
    parser.add_argument("--full",   action="store_true",
                        help="Fetch full article instead of summary only")
    parser.add_argument("--max-chars", type=int, default=5000,
                        help="Max chars per article in full mode (default: 5000)")
    args = parser.parse_args()

    # Resolve topic list
    if args.topics:
        topics = args.topics
    elif args.level is not None and args.level in TOPICS:
        topics = TOPICS[args.level]
    else:
        print("Specifica --level (1-4) o --topics TITOLO1 TITOLO2 ...")
        sys.exit(1)

    # Resolve output path
    out_path = args.output
    if not out_path and args.level is not None:
        os.makedirs(f"training_files/{args.lang}/{args.level}", exist_ok=True)
        out_path = f"training_files/{args.lang}/{args.level}/wikipedia.txt"

    print(f"Downloading {len(topics)} articles from Wikipedia ({args.lang})...")
    print(f"Mode: {'full text' if args.full else 'summary'}")
    print(f"Output: {out_path}\n")

    parts = []
    ok = 0
    for title in topics:
        if args.full:
            text = fetch_full(title, args.lang, args.max_chars)
        else:
            text = fetch_summary(title, args.lang)

        if text:
            parts.append(f"### {title}\n\n{text}")
            ok += 1
            print(f"  ✓ {title:30s}  ({len(text):,} chars)")
        else:
            print(f"  ✗ {title:30s}  (non trovato)")
        time.sleep(0.3)   # be polite

    combined = "\n\n\n".join(parts)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"\nSalvato: {out_path}  ({len(combined):,} chars, {ok}/{len(topics)} articoli)")


if __name__ == "__main__":
    main()
