#!/usr/bin/env python3
"""
Dream a level until dreaming stops paying: the empirical replacement for a
fixed MIN_DREAMS.

MIN_DREAMS=6 came from one curve, measured at level 10: +3.6 points of
cross-level exact match per dream from 1 to 6, +1.0 from 7 to 12, against 2.2
points of noise between identical runs. Six was the knee OF THAT CURVE — other
levels get the same constant only because nobody was measuring theirs. This
script measures it: score the frozen probe after every dream and stop when the
marginal gain has died.

The rule, per level:

    baseline = probe(final_dreamed.pt)
    repeat:
        dream once, ACCUMULATING (--checkpoint is what makes dreams add up;
        without it phase 2 restarts from final_learned.pt and every "dream N"
        is dream 1 again — the exact mistake measured on 2026-08-31, where a
        "six-dream" curve was six independent first dreams)
        score the probe
        regression beyond --max-drop  -> stop, RESTORE the best state
        gain < --epsilon for --patience consecutive dreams -> stop (plateau)
        --max reached                 -> stop
    (--min forces dreaming past an early plateau; --already-done are the
     session dreams that happened before this script ran, counted toward
     --min and --max but invisible to the plateau window)

With epsilon 0.02 and patience 2 the rule reproduces ~6 on the measured L10
curve by itself; a level that saturates earlier stops earlier, one still
climbing keeps going to the cap, and every run leaves its curve in
dream_curve.json — the number a fixed constant never records: what the last
dream was worth.

What it measures, said plainly: RETENTION. The probe is training material
(that is what dreams consolidate), so a plateau here says nothing about
generalization.

Restoring means restoring the PAIR plus the memory: a dream can grow the
vocabulary (N2-B) and rewrites tokenizer.json next to the checkpoint, and the
loader tolerates a newer tokenizer, not an older one — rolling back the
weights alone would pair an old model with a new tokenizer. affect_memory.json
travels with them so the curiosity ledger does not remember explanations the
restored weights never consolidated.

A dream that dies is MEASURED, not trusted: phase 2 saves the pair before it
can still fail (and the saves are plain in-place writes, so a kill can leave a
torn file), which means "the child exited nonzero" says nothing about what is
on disk. The salvage path loads and scores whatever is there — unloadable or
badly regressed brings back the best snapshot; loadable and sane is appended
to the curve as a real, measured state. Ctrl-C takes the same path, and the
curve is written on EVERY exit, salvage included.

Usage:
    python3 scripts/dream_until_plateau.py --level 11
    python3 scripts/dream_until_plateau.py --level 11 --min 6 --already-done 5
Exit codes: 0 = stopped by rule; 1 = refused at startup (no pair to
accumulate on, or bad parameters); 3 = a dream crashed (state salvaged);
130 = interrupted (state salvaged).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_set                                                   # noqa: E402

DREAM = "dream"
STOP_PLATEAU = "stop-plateau"
STOP_REGRESSION = "stop-regression"
STOP_MAX = "stop-max"

# The one home of the tuning constants. tests/test_dream_plateau.py asserts
# both against these AND that build.sh's fallback knobs carry the same values:
# three hand-copies of a calibrated number is how they drift apart silently.
DEFAULTS = {"epsilon": 0.02, "patience": 2, "max_drop": 0.05, "cap": 12}

# The pair a dream writes, plus the memory that must stay in step with them.
# fisher.pt (the exp_i EWC sidecar) rides along copy-if-exists: its anchor IS
# a snapshot of specific weights, so a rollback that restored the checkpoint
# without it would leave the anchor pointing at weights no longer on disk —
# the exact bug class this tuple exists to prevent for the tokenizer.
_STATE_FILES = ("final_dreamed.pt", "tokenizer.json", "fisher.pt")


def decide(curve, *, epsilon, patience, max_dreams, max_drop,
           floor=0, already_done=0):
    """(action, reason) for the NEXT step, given the exact-match curve.

    `curve` is [baseline, after dream 1, after dream 2, ...] for the dreams
    THIS run made; `already_done` are prior session dreams, which count toward
    the floor and the cap but contribute no points to the curve — their gains
    already happened and cannot be re-measured.

    Order matters and is part of the contract:
      regression first  — a dream that DAMAGED retention stops everything,
                          floor included: forcing more dreams onto a state
                          that just got worse is how damage compounds;
      cap second        — the measured curve saturates by ~10-12, and past it
                          the gains are under the noise (a floor above the cap
                          is therefore unreachable: the CLI refuses it);
      floor third       — below it, plateaus are ignored (the gate can pass
                          on one session dream, and one dream retains ~20%);
      plateau last      — the last `patience` marginal gains all under
                          `epsilon`. Patience exists because the dream is
                          stochastic: one flat delta is noise, two in a row is
                          a plateau (inter-run noise measured at ±2.2 points,
                          probe granularity ~1 point on 104 prompts).

    patience < 1 and max_drop < 0 are refused loudly: an empty patience window
    is vacuously "all under epsilon" (all() on []) and would stop every run on
    its baseline with zero dreams measured, and a negative max_drop reads the
    baseline itself as a regression. Both were found by adversarial review,
    both would disable the loop while looking like a clean stop.
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, not {patience}: with an "
                         f"empty window every run stops on its baseline")
    if max_drop < 0:
        raise ValueError(f"max_drop must be >= 0, not {max_drop}")
    done_here = len(curve) - 1
    done_total = already_done + done_here
    best = max(curve)
    if curve[-1] < best - max_drop:
        return STOP_REGRESSION, (f"exact {curve[-1]:.1%} is {best - curve[-1]:.1%} "
                                 f"below the best ({best:.1%})")
    if done_total >= max_dreams:
        return STOP_MAX, f"cap of {max_dreams} dreams reached"
    if done_total < floor:
        return DREAM, f"below the floor ({done_total}/{floor})"
    if done_here >= patience:
        recent = [curve[i] - curve[i - 1]
                  for i in range(done_here - patience + 1, done_here + 1)]
        if all(g < epsilon for g in recent):
            gains = ", ".join(f"{g:+.1%}" for g in recent)
            return STOP_PLATEAU, (f"last {patience} gains ({gains}) "
                                  f"under ε={epsilon:.1%}")
    return DREAM, "still gaining"


def snapshot_state(level_dir: str, dest_dir: str) -> None:
    """The best state so far, kept as a literal copy of the pair + memory."""
    os.makedirs(dest_dir, exist_ok=True)
    for name in _STATE_FILES:
        src = os.path.join(level_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest_dir, name))
    mem = os.path.join(os.path.dirname(level_dir), "affect_memory.json")
    if os.path.exists(mem):
        shutil.copy2(mem, os.path.join(dest_dir, "affect_memory.json"))


def restore_state(level_dir: str, src_dir: str) -> list:
    restored = []
    for name in _STATE_FILES:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(level_dir, name))
            restored.append(name)
    mem = os.path.join(src_dir, "affect_memory.json")
    if os.path.exists(mem):
        shutil.copy2(mem, os.path.join(os.path.dirname(level_dir),
                                       "affect_memory.json"))
        restored.append("affect_memory.json")
    return restored


def run_one_dream(args, dreamed: str) -> bool:
    """phase_2_dream in its own process, ACCUMULATING on final_dreamed.pt.

    Success is the exit code AND the checkpoint's mtime having moved:
    train_curriculum's main() exits 0 on some phase-2 failure paths (it prints
    the error and returns), and an exit code alone would then record a curve
    of no-op "dreams" that never touched the weights, each scoring exactly the
    same — a plateau by fabrication. All paths here are absolute, so the
    child's cwd=_ROOT cannot make parent and child disagree about which tree
    is being dreamed.
    """
    before = os.path.getmtime(dreamed)
    cmd = [sys.executable, "-u", "-m", "dynamic_model.train_curriculum",
           "--phase", "2", "--level", str(args.level), "--lang", args.lang,
           "--dream-mode", args.dream_mode,
           "--checkpoint", dreamed,
           "--ckpt-base", args.ckpt_base,
           # Threaded through explicitly: without this every top-up dream
           # would silently run as the 'dream' arm regardless of what the
           # session dreams did — the single most likely silent bug in the
           # exp_i comparison.
           "--anti-forgetting", args.anti_forgetting]
    if args.ewc_lambda is not None:
        cmd += ["--ewc-lambda", str(args.ewc_lambda)]
    if args.ewc_gamma is not None:
        cmd += ["--ewc-gamma", str(args.ewc_gamma)]
    print(f"  → {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=_ROOT).returncode != 0:
        return False
    if os.path.getmtime(dreamed) == before:
        print("  the dream process exited 0 but final_dreamed.pt was not "
              "rewritten: treating it as failed.")
        return False
    return True


def score_now(dreamed: str, tok_path: str, probe: dict) -> dict:
    """A fresh load every time: the dream just rewrote both files."""
    from measure_repetition import load_pair
    tr, tok = load_pair(dreamed, tok_path)
    return probe_set.score(tr, tok, probe)


def salvage(level_dir, dreamed, tok_path, probe, curve, reps, best_i,
            best_dir, max_drop, cause):
    """After a crashed or interrupted dream: measure the disk, do not trust it.

    phase 2 may have rewritten the pair before dying, and the writes are
    in-place — the state on disk is one of: the previous dream (child died
    early), a NEW unmeasured dream (child died after its saves), or a torn
    file (killed mid-save). Loading and scoring distinguishes all three:
    unloadable or badly regressed -> restore the best snapshot; loadable and
    sane -> append to the curve as a real measured point.

    Returns (note, restored).
    """
    print(f"  [salvage after {cause}] measuring the state on disk…", flush=True)
    try:
        r = score_now(dreamed, tok_path, probe)
    except KeyboardInterrupt:
        raise
    except Exception as e:                                 # noqa: BLE001
        names = restore_state(level_dir, best_dir)
        note = (f"pair on disk not loadable ({type(e).__name__}): "
                f"restored the best one ({', '.join(names)})")
        print(f"  {note}")
        return note, True
    curve.append(r["exact_rate"])
    reps.append(r["repetition_rate"])
    print(f"  state on disk: exact {curve[-1]:.1%} (best {curve[best_i]:.1%})")
    if curve[-1] < curve[best_i] - max_drop:
        names = restore_state(level_dir, best_dir)
        # The salvaged point stays in the curve — it was real — but the disk
        # goes back to the best state.
        note = (f"state measured at {curve[-1]:.1%}, below the best by more "
                f"than max_drop: restored ({', '.join(names)})")
        print(f"  {note}")
        return note, True
    return f"state on disk measured and kept ({curve[-1]:.1%})", False


def write_record(curve_path, record) -> None:
    history = []
    if os.path.exists(curve_path):
        try:
            with open(curve_path, encoding="utf-8") as f:
                history = json.load(f)
        except (ValueError, OSError):
            history = []
    history.append(record)
    with open(curve_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Dream a level until the frozen probe stops improving")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--ckpt-base", default=None)
    ap.add_argument("--epsilon", type=float, default=DEFAULTS["epsilon"],
                    help="marginal gain (fraction) under which a dream did "
                         "not pay (default %(default)s ≈ 2 prompts of 104)")
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                    help="consecutive sub-epsilon dreams before stopping "
                         "(>= 1)")
    ap.add_argument("--max-drop", type=float, default=DEFAULTS["max_drop"],
                    help="fall from the best score that stops and restores")
    ap.add_argument("--min", type=int, default=0, dest="floor",
                    help="dreams to run regardless (counting --already-done)")
    ap.add_argument("--max", type=int, default=DEFAULTS["cap"], dest="cap",
                    help="hard cap, counting --already-done (curve saturates "
                         "by ~10-12)")
    ap.add_argument("--already-done", type=int, default=0,
                    help="session dreams run before this script (build.sh "
                         "passes its DREAMS_DONE)")
    ap.add_argument("--dream-mode", default="standard",
                    choices=["light", "standard", "deep"])
    # exp_i arm flags, forwarded verbatim to every dream child (see
    # run_one_dream). Defaults reproduce the validated build.
    ap.add_argument("--anti-forgetting", default="dream",
                    choices=["dream", "ewc", "none"])
    ap.add_argument("--ewc-lambda", type=float, default=None)
    ap.add_argument("--ewc-gamma", type=float, default=None)
    ap.add_argument("--probe", default=probe_set.DEFAULT_PATH)
    a = ap.parse_args()

    # Refusals, not clamps: every one of these either disables the loop while
    # looking like a clean stop (patience<1, negative max_drop) or promises
    # something unreachable (floor past the cap, where decide() checks the cap
    # first — by contract — and would quietly truncate the floor).
    if a.patience < 1:
        ap.error(f"--patience {a.patience}: must be >= 1")
    if a.max_drop < 0:
        ap.error(f"--max-drop {a.max_drop}: must be >= 0")
    if a.epsilon < 0:
        ap.error(f"--epsilon {a.epsilon}: must be >= 0")
    if a.floor < 0 or a.already_done < 0 or a.cap < 1:
        ap.error("--min/--already-done must be >= 0, --max >= 1")
    if a.floor > a.cap:
        ap.error(f"--min {a.floor} > --max {a.cap}: the floor is "
                 f"unreachable (the cap wins, by contract)")

    # Absolute, before anything touches the filesystem: the dream child runs
    # with cwd=_ROOT, and a relative --ckpt-base would make parent and child
    # resolve DIFFERENT trees — the parent scoring one, the child dreaming
    # another (or nothing, exit 0 included).
    a.ckpt_base = os.path.abspath(a.ckpt_base or
                                  os.path.join("models", "checkpoints", a.lang))
    level_dir = os.path.join(a.ckpt_base, f"level_{a.level}")
    dreamed = os.path.join(level_dir, "final_dreamed.pt")
    tok_path = os.path.join(level_dir, "tokenizer.json")
    if not (os.path.exists(dreamed) and os.path.exists(tok_path)):
        # No pair to accumulate on: the level has not had its first in-session
        # dream. Refusing beats silently running phase 2 without --checkpoint,
        # which would look identical and consolidate nothing.
        print(f"Missing {dreamed} (or the tokenizer): the session makes the "
              f"first dream, this script ACCUMULATES on top of it.")
        return 1

    probe = probe_set.load(a.probe)          # raises if edited — by design
    best_dir = os.path.join(level_dir, "plateau_best")
    curve_path = os.path.join(level_dir, "dream_curve.json")

    r = score_now(dreamed, tok_path, probe)
    curve = [r["exact_rate"]]
    reps = [r["repetition_rate"]]
    print(f"\ndreams until plateau — L{a.level}  "
          f"(ε={a.epsilon:.1%}, patience={a.patience}, "
          f"already done {a.already_done}, floor {a.floor}, cap {a.cap})")
    print(f"  baseline: exact {curve[0]:.1%}  repetition {reps[0]:.1%}")
    snapshot_state(level_dir, best_dir)
    best_i = 0

    stopped, why, salvage_note = "", "", ""
    rc = 0
    try:
        while True:
            action, why = decide(curve, epsilon=a.epsilon, patience=a.patience,
                                 max_dreams=a.cap, max_drop=a.max_drop,
                                 floor=a.floor, already_done=a.already_done)
            stopped = action
            if action != DREAM:
                print(f"\n  stop: {action} — {why}")
                break
            k = len(curve)
            print(f"\n  ── dream {a.already_done + k} ({why}) ──", flush=True)
            t0 = time.time()
            if not run_one_dream(a, dreamed):
                salvage_note, _ = salvage(level_dir, dreamed, tok_path, probe,
                                          curve, reps, best_i, best_dir,
                                          a.max_drop, "a failed dream")
                stopped, why = "crash", "dream subprocess failed"
                rc = 3
                break
            r = score_now(dreamed, tok_path, probe)
            curve.append(r["exact_rate"])
            reps.append(r["repetition_rate"])
            d = curve[-1] - curve[-2]
            print(f"  dream {a.already_done + k}: exact {curve[-2]:.1%} → "
                  f"{curve[-1]:.1%} ({d:+.1%})  repetition {reps[-1]:.1%}  "
                  f"[{time.time() - t0:.0f}s]")
            if curve[-1] >= curve[best_i]:
                snapshot_state(level_dir, best_dir)
                best_i = len(curve) - 1
    except KeyboardInterrupt:
        # The interrupt reached the child too (same process group): its saves
        # may be torn. Same salvage as a crash; a second Ctrl-C during the
        # salvage exits raw, which is the user insisting.
        print("\n  interrupted.")
        salvage_note, _ = salvage(level_dir, dreamed, tok_path, probe,
                                  curve, reps, best_i, best_dir,
                                  a.max_drop, "an interruption")
        stopped, why = "interrupted", "Ctrl-C"
        rc = 130
    finally:
        # Whatever ended the loop, the disk gets the BEST measured state, not
        # the last one. Every point on the curve is the same deterministic
        # measurement, so past the best the extra dreams bought nothing by
        # definition — the first live run stopped on a plateau with dream 9 at
        # 81.7% on disk while dream 8 sat in the snapshot at 82.7% with a
        # third of the repetition, and the old rule (restore only beyond
        # max_drop) would have handed the WORSE state to the next level's
        # build. The regression stop is the same rule with a bigger gap.
        if curve[-1] < curve[best_i]:
            names = restore_state(level_dir, best_dir)
            print(f"  RESTORING the best state (dream "
                  f"{a.already_done + best_i}, exact {curve[best_i]:.1%}, "
                  f"disk had {curve[-1]:.1%}): {', '.join(names)}")
        write_record(curve_path, {
            "level": a.level, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "already_done": a.already_done,
            "epsilon": a.epsilon, "patience": a.patience,
            "max_drop": a.max_drop, "floor": a.floor, "cap": a.cap,
            "exact": [round(x, 4) for x in curve],
            "repetition": [round(x, 4) for x in reps],
            "stopped": stopped, "reason": why,
            "salvage": salvage_note or None,
            "best_index": best_i,
        })
        n_new = len(curve) - 1
        print(f"\n  curve: {' → '.join(f'{x:.1%}' for x in curve)}")
        print(f"  {n_new} points measured beyond the baseline "
              f"(session: {a.already_done}). Recorded in {curve_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
