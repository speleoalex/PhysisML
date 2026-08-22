#!/bin/bash
# Experiment exp_f: is retention a function of the number of dream cycles?
#
# THE QUESTION
# The retention matrix (scripts/retention_matrix.py) shows the diagonal at 96%
# but the final L10 checkpoint at 20% across all levels. The one row that
# retains earlier levels is L4 (100/83/100/71% on L0-L3, and 0% repetition),
# and L4 is the only level that ran ten teaching sessions — every session ends
# in a dream, and the dream's N1 replays EVERY level's qa_corpus. Every other
# level ran 1-3 sessions.
#
# So: does running extra dreams on the finished L10 checkpoint recover the
# earlier levels? No new teaching, no new gold — only consolidation of what is
# already in the session logs.
#
#   recovers  → retention is a function of consolidation cycles, and the fix is
#               structural: guarantee N dreams per level regardless of when the
#               quality gate happens to pass.
#   flat      → the damage is not repairable after the fact and has to be
#               prevented during the curriculum.
#
# COST: minutes per dream, no API cost (nothing is taught). Compare against
# hours per level for a rebuild.
#
# ISOLATION
# Writes to models/exp_f/ only. The validated tree (models/checkpoints/it) is
# copied for its session logs and tokenizers — 4.5MB — and the L10 checkpoint
# is copied once as the starting point. It is never written to.
#
# CAVEAT: phase 2 calls _update_qa_pairs_from_sessions(), which regenerates
# training_files/<lang>/<level>/qa_corpus.txt. That is shared state outside
# models/. The content is the same pair multiset in a different shuffle order,
# but it does show up as a git diff — this script reports it at the end.
#
# Usage:
#   ./scripts/experiment_extra_dreams.sh --confirm          # 6 dreams
#   N_DREAMS=10 ./scripts/experiment_extra_dreams.sh --confirm
#   LEVEL=5     ./scripts/experiment_extra_dreams.sh --confirm
#   RESUME=1    ./scripts/experiment_extra_dreams.sh --confirm   # continue
#
# Analysis (also printed as it goes):
#   python3 scripts/retention_matrix.py --compare \
#       models/analysis/retention_baseline.json \
#       models/exp_f/retention_after_6.json

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Deliberately not named LANG: that is the system locale variable,
# and overwriting it with "it" would set an invalid locale for the
# python subprocesses.
LNG="${LNG:-it}"
LEVEL="${LEVEL:-10}"
N_DREAMS="${N_DREAMS:-6}"
SEED="${SEED:-1}"
DREAM_MODE="${DREAM_MODE:-standard}"
SRC="models/checkpoints/$LNG"
EXP="models/exp_f"
BASELINE="models/analysis/retention_baseline.json"
RESUME="${RESUME:-0}"

if [ "$1" != "--confirm" ]; then
    echo "  exp_f — extra dream cycles on the finished L$LEVEL checkpoint"
    echo ""
    echo "  dreams      : $N_DREAMS   (mode $DREAM_MODE, seed $SEED)"
    echo "  source      : $SRC/level_$LEVEL/final_dreamed.pt  (read-only)"
    echo "  writes to   : $EXP/"
    echo "  measures    : the final row of the retention matrix after each dream"
    echo ""
    echo "  Re-run with --confirm to start."
    exit 0
fi

# ── Python: the GPU env if present, exactly as build.sh picks it ─────────────
CONDA_GPU_PYTHON="$HOME/miniforge3/envs/physisml_gpu/bin/python"
ONEAPI_VARS="/opt/intel/oneapi/2025.3/oneapi-vars.sh"
[ ! -f "$ONEAPI_VARS" ] && ONEAPI_VARS="/opt/intel/oneapi/setvars.sh"
if [ -f "$CONDA_GPU_PYTHON" ] && [ -f "$ONEAPI_VARS" ]; then
    source "$ONEAPI_VARS" > /dev/null 2>&1
    [ -d "$HOME/.local/lib" ] && export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
    PY="$CONDA_GPU_PYTHON"
else
    PY="python3"
fi
echo "  Python: $PY"

CKPT_BASE="$EXP/dreams_L${LEVEL}"

# ── Single-run lock ─────────────────────────────────────────────────────────
# Two concurrent runs silently corrupt each other: both write
# level_N/final_dreamed.pt and tokenizer.json in the same CKPT_BASE, and a
# fresh run rm -rf's the tree the other one is mid-dream on. mkdir is atomic,
# so it is the lock.
mkdir -p "$EXP"
LOCK="$EXP/.lock_L${LEVEL}"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo ""
    echo "  ERROR: another run holds the lock for L$LEVEL:"
    echo "    $LOCK"
    if [ -f "$LOCK/pid" ]; then
        LPID=$(cat "$LOCK/pid")
        if kill -0 "$LPID" 2>/dev/null; then
            echo "    still alive: PID $LPID  ($(ps -o etime= -p "$LPID" 2>/dev/null | tr -d ' ') elapsed)"
            echo "    Wait for it, or stop it first."
        else
            echo "    PID $LPID is gone — stale lock from a killed run."
            echo "    Remove it and re-run:  rm -rf $LOCK"
        fi
    fi
    exit 1
fi
echo "$$" > "$LOCK/pid"
# EXIT alone does not fire when bash is killed by a signal (a
# `timeout` firing leaves the lock behind), so trap the signals too.
trap 'rm -rf "$LOCK"' EXIT INT TERM

if [ "$RESUME" != "1" ]; then
    rm -rf "$CKPT_BASE"
    mkdir -p "$CKPT_BASE"
    # Session logs + tokenizers for every level: the memory bank is built from
    # range(level+1), and the tokenizer search walks back through lower levels.
    for d in "$SRC"/level_*; do
        lvl=$(basename "$d")
        mkdir -p "$CKPT_BASE/$lvl"
        cp "$d"/session_*.jsonl "$CKPT_BASE/$lvl/" 2>/dev/null || true
        [ -f "$d/tokenizer.json" ] && cp "$d/tokenizer.json" "$CKPT_BASE/$lvl/"
    done
    cp "$SRC/level_$LEVEL/final_dreamed.pt" "$CKPT_BASE/level_$LEVEL/final_dreamed.pt"
    echo "  Tree prepared: $CKPT_BASE  ($(du -sh "$CKPT_BASE" | cut -f1))"
fi

mkdir -p models/analysis
PREV="$CKPT_BASE/level_$LEVEL/final_dreamed.pt"
LOG="$EXP/dreams_L${LEVEL}.log"

# With RESUME=1, skip the dreams already measured instead of redoing them and
# overwriting their results. Each dream feeds on the previous one's output, so
# the checkpoint on disk is already at DONE dreams.
START=1
if [ "$RESUME" = "1" ]; then
    DONE=0
    for f in "$EXP"/retention_after_[0-9]*.json; do
        [ -f "$f" ] || continue
        b=$(basename "$f" .json); n=${b##*_}
        case "$n" in (*[!0-9]*) continue ;; esac
        [ "$n" -gt "$DONE" ] && DONE="$n"
    done
    START=$((DONE + 1))
    if [ "$START" -gt "$N_DREAMS" ]; then
        echo "  Already at $DONE dreams (>= N_DREAMS=$N_DREAMS). Nothing to do."
        echo "  Raise N_DREAMS to continue."
        exit 0
    fi
    echo "  Resuming: $DONE dreams done, running $START..$N_DREAMS"
fi

echo "" | tee -a "$LOG"
echo "=== exp_f  L$LEVEL  ${N_DREAMS} dreams  seed $SEED ===" | tee -a "$LOG"

for i in $(seq "$START" "$N_DREAMS"); do
    echo "" | tee -a "$LOG"
    echo "--- dream $i/$N_DREAMS ---" | tee -a "$LOG"
    # phase 2 overwrites level_N/final_dreamed.pt in CKPT_BASE, so each dream
    # feeds on the previous one's output. That is the point: cumulative
    # consolidation, exactly as ten sessions at L4 produced ten dreams.
    $PY -u -m dynamic_model.train_curriculum \
        --phase 2 --level "$LEVEL" --lang "$LNG" \
        --dream-mode "$DREAM_MODE" --seed "$SEED" \
        --checkpoint "$PREV" --ckpt-base "$CKPT_BASE" 2>&1 | tee -a "$LOG"

    OUT="$EXP/retention_after_${i}.json"
    echo "" | tee -a "$LOG"
    echo "--- retention after dream $i ---" | tee -a "$LOG"
    # Only the final row: it is the number the experiment is about, and it
    # costs one checkpoint load instead of eleven.
    $PY scripts/measure_repetition.py \
        --checkpoint "$CKPT_BASE/level_$LEVEL/final_dreamed.pt" \
        --tokenizer  "$CKPT_BASE/level_$LEVEL/tokenizer.json" \
        --levels "0-$LEVEL" --json "$OUT" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "  exp_f done — $N_DREAMS dreams on L$LEVEL" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "  per-dream results: $EXP/retention_after_*.json"
echo "  full matrix of the end state:"
echo "    python3 scripts/retention_matrix.py --ckpt-base $CKPT_BASE \\"
echo "        --levels 0-$LEVEL --json $EXP/retention_matrix_after.json"
echo "    python3 scripts/retention_matrix.py --compare \\"
echo "        $BASELINE $EXP/retention_matrix_after.json"

# Shared state touched by phase 2, reported rather than hidden.
DIRTY=$(git status --porcelain training_files/ | wc -l)
if [ "$DIRTY" -gt 0 ]; then
    echo ""
    echo "  NOTE: phase 2 regenerated $DIRTY file(s) under training_files/."
    echo "  Same pairs, different shuffle order. To discard:"
    echo "    git checkout -- training_files/"
fi
