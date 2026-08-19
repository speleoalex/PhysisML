"""
Analyze a teaching session log (.jsonl) and show progress statistics.

Usage:
    python3 scripts/analyze_log.py                          # latest session, summary
    python3 scripts/analyze_log.py --all                    # all sessions
    python3 scripts/analyze_log.py --prompts                # show all teacher prompts
    python3 scripts/analyze_log.py --prompts --filter +     # only turns with feedback +/++/+++
    python3 scripts/analyze_log.py path/to/session.jsonl    # specific file or folder
"""
import sys, os, json, glob, argparse
from collections import defaultdict

FEEDBACK_VALUE = {"+++": 1.0, "++": 0.8, "+": 0.5, "=": 0.0, "-": -1.0, None: 0.0}


def load_log(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_latest_log(directory: str) -> str:
    logs = sorted(glob.glob(os.path.join(directory, "session_*.jsonl")))
    if not logs:
        return None
    return logs[-1]


def analyze(records: list, path: str) -> None:
    if not records:
        print("  Empty log.")
        return

    print(f"\n{'─'*65}")
    print(f"  Log: {os.path.basename(path)}")
    print(f"  Total turns: {len(records)}")
    print(f"{'─'*65}")

    # Feedback distribution
    fb_counts = defaultdict(int)
    for r in records:
        fb = r.get("feedback")
        if fb:
            fb_counts[fb] += 1

    total_fb = sum(fb_counts.values())
    if total_fb:
        print(f"\n  Feedback distribution ({total_fb} ratings):")
        for sym in ["+++", "++", "+", "=", "-"]:
            n = fb_counts.get(sym, 0)
            bar = "█" * int(n / max(total_fb, 1) * 30)
            print(f"    {sym:>4}  {bar:<30}  {n:4d}  ({100*n//total_fb:2d}%)")

    # Rolling positive rate (window=10)
    pos_rate_history = []
    window = 10
    fb_values = [FEEDBACK_VALUE.get(r.get("feedback"), 0) for r in records if r.get("feedback")]
    for i in range(window, len(fb_values)+1):
        chunk = fb_values[i-window:i]
        pos_rate = sum(1 for v in chunk if v > 0) / window
        pos_rate_history.append(pos_rate)

    if pos_rate_history:
        print(f"\n  Positive rate (last {window} turns):")
        print(f"    Start:   {pos_rate_history[0]:.0%}")
        print(f"    End:     {pos_rate_history[-1]:.0%}")
        print(f"    Maximum: {max(pos_rate_history):.0%}  (at turn {pos_rate_history.index(max(pos_rate_history))+window})")

    # Affect progression
    affects = [r["affect"] for r in records if "affect" in r]
    if affects:
        print(f"\n  Affective state (start → end):")
        first, last = affects[0], affects[-1]
        for key in ["confidence", "fear", "pleasure", "pain"]:
            arrow = "↑" if last[key] > first[key] else "↓" if last[key] < first[key] else "→"
            print(f"    {key:12s}: {first[key]:.2f}  {arrow}  {last[key]:.2f}")

    # Step progression
    steps = [r.get("step") for r in records if r.get("step")]
    if steps:
        step_counts = defaultdict(int)
        for s in steps:
            step_counts[s] += 1
        print(f"\n  Step distribution: " +
              "  ".join(f"{s}={n}" for s, n in sorted(step_counts.items())))
        print(f"  Last step reached: {steps[-1]}")

    # Best and worst turns
    rated = [(r["turn"], r.get("feedback"), r.get("prompt",""), r.get("response",""))
             for r in records if r.get("feedback") in ("+++", "-")]
    best  = [(t,f,p,r) for t,f,p,r in rated if f == "+++"]
    worst = [(t,f,p,r) for t,f,p,r in rated if f == "-"]

    if best:
        print(f"\n  Best responses (+++):  {len(best)} turns")
        for t, _, p, resp in best[-3:]:
            print(f"    turn {t:4d}: {repr(p):20s} → {repr(resp[:30])}")
    if worst:
        print(f"\n  Negative responses (-):  {len(worst)} turns")
        for t, _, p, resp in worst[:3]:
            print(f"    turn {t:4d}: {repr(p):20s} → {repr(resp[:30])}")

    print(f"{'─'*65}")


def show_prompts(records: list, path: str, fb_filter: str = None) -> None:
    """Print a readable transcript of teacher prompts and model responses."""
    print(f"\n{'─'*70}")
    print(f"  DIALOGUE — {os.path.basename(path)}  ({len(records)} turns)")
    if fb_filter:
        print(f"  Filtro feedback: {fb_filter}")
    print(f"{'─'*70}")

    FB_ICON = {"+++": "✓✓✓", "++": "✓✓ ", "+": "✓  ", "=": "=  ", "-": "✗  ", None: "   "}

    for r in records:
        fb   = r.get("feedback")
        step = r.get("step", "?")
        turn = r.get("turn", 0)

        # Apply filter
        if fb_filter:
            if fb_filter == "+" and fb not in ("+", "++", "+++"):
                continue
            elif fb_filter == "-" and fb != "-":
                continue
            elif fb_filter == "=" and fb != "=":
                continue

        icon    = FB_ICON.get(fb, "   ")
        prompt  = r.get("prompt", "")
        resp    = r.get("response", "")[:35]
        comment = r.get("comment", "")[:35]
        affect  = r.get("affect", {})
        conf    = affect.get("confidence", 0)
        fear    = affect.get("fear", 0)

        print(f"\n  [{turn:3d}] [{step}] {icon}  conf={conf:.2f} fear={fear:.2f}")
        print(f"  Teacher: {prompt}")
        print(f"  Model:   {resp}")
        if comment:
            print(f"  Note:    {comment}")

    print(f"\n{'─'*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None,
                        help=".jsonl file, checkpoint folder, or omit for the latest")
    parser.add_argument("--all",     action="store_true",
                        help="Analyse all logs in models/checkpoints/")
    parser.add_argument("--prompts", action="store_true",
                        help="Show full dialogue (teacher prompts + model responses)")
    parser.add_argument("--filter",  default=None,
                        choices=["+", "=", "-"],
                        help="Show only turns with a given feedback: + = -")
    args = parser.parse_args()

    if args.all:
        logs = sorted(glob.glob("models/checkpoints/**/*.jsonl", recursive=True))
        if not logs:
            print("No logs found in models/checkpoints/")
            return
        for log_path in logs:
            records = load_log(log_path)
            if args.prompts:
                show_prompts(records, log_path, args.filter)
            else:
                analyze(records, log_path)
        return

    if not args.path:
        logs = sorted(glob.glob("models/checkpoints/**/*.jsonl", recursive=True))
        if not logs:
            print("No log found. Use: python3 scripts/analyze_log.py <path>")
            return
        args.path = logs[-1]
        print(f"Latest log found: {args.path}")

    if os.path.isdir(args.path):
        log_path = find_latest_log(args.path)
        if not log_path:
            print(f"No log found in {args.path}")
            return
        args.path = log_path

    records = load_log(args.path)
    if args.prompts:
        show_prompts(records, args.path, args.filter)
    else:
        analyze(records, args.path)


if __name__ == "__main__":
    main()
