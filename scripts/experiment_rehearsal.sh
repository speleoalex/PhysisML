#!/bin/bash
# Experiment exp_h: does rehearsing EARLIER levels close the retention gap?
#
# THE QUESTION
# The retention matrix puts the final L10 checkpoint at 20% across all levels
# against a 96% diagonal. Extra dream cycles recover half of that (to 48%, then
# saturating), and the reason they cannot do more is structural: the dream
# replays the corpus (N1) and the memory bank (N3) of every level, but the
# INTERLEAVED GOLD REHEARSAL — the channel that actually built the
# prompt->answer associations — has only ever drawn on the current level.
#
# THREE ARMS, one flag apart:
#   level     current level's gold pairs only            ← control, the build
#   balanced  union of all levels, half of each replay reserved for the current
#   all       plain union (the current level's share falls as levels accumulate)
#
# 'balanced' exists because dilution is a real risk, not a hypothetical: at L3
# the union is 535 pairs against 188 for the level itself, and the same
# dilution is what made N3 replay stop helping the level being taught.
#
# WHY L0->L3: L1, L2 and L3 each have earlier levels to retain, which is what
# the experiment is about, and these are the cheap levels. Arms run in the
# order above, so stopping early still leaves control-vs-balanced comparable.
#
# ISOLATION
# Each arm gets its own checkpoint tree under models/exp_h/. The validated tree
# is never written. training_files/<lang> IS shared state that phase 2 rewrites
# (qa_pairs.jsonl feeds the rehearsal bank itself, so contamination would
# change the treatment), and it is snapshotted once and restored before every
# arm and at the end.
#
# Usage:
#   ./scripts/experiment_rehearsal.sh --confirm
#   ARMS="level balanced" ./scripts/experiment_rehearsal.sh --confirm
#   SKIP_DONE=1 ./scripts/experiment_rehearsal.sh --confirm     # resume
#   TARGET_LEVEL=2 SESSIONS=1 DREAMS=3 ./scripts/experiment_rehearsal.sh --confirm

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LNG="${LNG:-it}"
TARGET_LEVEL="${TARGET_LEVEL:-3}"
SESSIONS="${SESSIONS:-2}"
DREAMS="${DREAMS:-6}"          # total dreams per level, matching build.sh MIN_DREAMS
EPOCHS_0="${EPOCHS_0:-3}"
TURNS="${TURNS:-100}"
SEED="${SEED:-1}"
ARMS="${ARMS:-level balanced all}"
SKIP_DONE="${SKIP_DONE:-0}"
EXP="${EXP:-models/exp_h}"
BASELINE="models/analysis/retention_baseline.json"

if [ "$1" != "--confirm" ]; then
    echo "  exp_h — rehearsal scope, L0->$TARGET_LEVEL"
    echo ""
    echo "  arms     : $ARMS"
    echo "  per level: phase 0 ($EPOCHS_0 ep) + $SESSIONS sessions x $TURNS turns + $DREAMS dreams"
    echo "  seed     : $SEED   teacher: local (deterministic, no API cost)"
    echo "  writes   : $EXP/   (models/checkpoints is never touched)"
    echo ""
    echo "  Re-run with --confirm."
    exit 0
fi

CONDA_GPU_PYTHON="$HOME/miniforge3/envs/physisml_gpu/bin/python"
if [ -f "$CONDA_GPU_PYTHON" ]; then PY="$CONDA_GPU_PYTHON"; else PY="python3"; fi
echo "  Python: $PY"

mkdir -p "$EXP"
LOCK="$EXP/.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "  ERROR: another run holds $LOCK"
    if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
        echo "  Stale (PID $(cat "$LOCK/pid") is gone). Remove it: rm -rf $LOCK"
    fi
    exit 1
fi
echo "$$" > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM

# Snapshot the shared input once; restore it before every arm.
SNAP="$EXP/training_files_snapshot"
if [ ! -d "$SNAP/$LNG" ]; then
    mkdir -p "$SNAP"
    cp -a "training_files/$LNG" "$SNAP/"
    echo "  Snapshot: $SNAP/$LNG"
fi
restore_inputs() {
    rm -rf "training_files/$LNG"
    cp -a "$SNAP/$LNG" "training_files/"
}

for ARM in $ARMS; do
    BASE="$EXP/${ARM}_s${SEED}"
    LOG="$EXP/${ARM}_s${SEED}.log"
    # Completion sentinel, written only after the arm's last dream. The
    # obvious test — level_<target>/final_dreamed.pt exists — is wrong: that
    # file appears as soon as the target level's FIRST dream runs, so an arm
    # interrupted midway through its last level looks finished and gets
    # skipped, and the retention matrix then measures a half-trained arm
    # without saying so.
    if [ "$SKIP_DONE" = "1" ] && [ -f "$BASE/.arm_complete" ]; then
        echo "  [skip] $ARM already complete"
        continue
    fi

    restore_inputs
    rm -rf "$BASE"; mkdir -p "$BASE"
    echo ""
    echo "================================================================"
    echo "  arm: $ARM   (--rehearsal-scope $ARM)   seed $SEED"
    echo "================================================================"

    {
      echo "=== exp_h arm=$ARM seed=$SEED  L0->$TARGET_LEVEL ==="
      for LEVEL in $(seq 0 "$TARGET_LEVEL"); do
        echo ""; echo "--- L$LEVEL phase 0 ---"
        $PY -u -m dynamic_model.train_curriculum \
            --phase 0 --level "$LEVEL" --lang "$LNG" \
            --epochs-0 "$EPOCHS_0" --seed "$SEED" \
            --rehearsal-scope "$ARM" --ckpt-base "$BASE"

        PREV="$BASE/level_${LEVEL}/final.pt"
        DONE_DREAMS=0

        for S in $(seq 1 "$SESSIONS"); do
          echo ""; echo "--- L$LEVEL session $S/$SESSIONS ---"
          $PY -u -m dynamic_model.train_curriculum \
              --phase 1 --level "$LEVEL" --lang "$LNG" \
              --tutor-model local --interactions "$TURNS" \
              --no-turn-ckpt --seed "$SEED" \
              --rehearsal-scope "$ARM" \
              --checkpoint "$PREV" --ckpt-base "$BASE"

          # Only the first session dreams in 'standard' mode, as build.sh does:
          # that is the dream that may grow the vocabulary.
          if [ "$S" -eq 1 ]; then MODE="standard"; else MODE="light"; fi
          echo ""; echo "--- L$LEVEL dream $MODE (after session $S) ---"
          $PY -u -m dynamic_model.train_curriculum \
              --phase 2 --level "$LEVEL" --lang "$LNG" \
              --dream-mode "$MODE" --seed "$SEED" \
              --rehearsal-scope "$ARM" --ckpt-base "$BASE"
          DONE_DREAMS=$((DONE_DREAMS + 1))
          PREV="$BASE/level_${LEVEL}/final_dreamed.pt"
        done

        # Top up to DREAMS, the same way build.sh now does. --checkpoint is
        # required: without it phase 2 restarts from final_learned.pt and
        # repeats the first dream instead of accumulating.
        while [ "$DONE_DREAMS" -lt "$DREAMS" ]; do
          echo ""; echo "--- L$LEVEL top-up dream $((DONE_DREAMS + 1))/$DREAMS ---"
          $PY -u -m dynamic_model.train_curriculum \
              --phase 2 --level "$LEVEL" --lang "$LNG" \
              --dream-mode standard --seed "$SEED" \
              --rehearsal-scope "$ARM" \
              --checkpoint "$PREV" --ckpt-base "$BASE"
          DONE_DREAMS=$((DONE_DREAMS + 1))
        done
      done
      echo ""; echo "=== arm $ARM complete ==="
    } 2>&1 | tee "$LOG"
    date -Is > "$BASE/.arm_complete"

    echo ""
    echo "  --- retention matrix, arm $ARM ---"
    $PY scripts/retention_matrix.py --ckpt-base "$BASE" \
        --levels "0-$TARGET_LEVEL" \
        --json "$EXP/retention_${ARM}.json" 2>&1 | tail -25
done

restore_inputs
echo ""
echo "================================================================"
echo "  exp_h done"
echo "================================================================"
for ARM in $ARMS; do
    J="$EXP/retention_${ARM}.json"
    [ -f "$J" ] && echo "  $ARM: $J"
done
echo ""
echo "  compare any two arms:"
echo "    python3 scripts/retention_matrix.py --compare \\"
echo "        $EXP/retention_level.json $EXP/retention_balanced.json"
