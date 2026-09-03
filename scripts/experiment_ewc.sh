#!/bin/bash
# Experiment exp_i: does the dream's cross-level replay beat standard EWC?
#
# THE QUESTION
# The README credits the 20% -> 88% retention jump to the dream — a replay
# pass over every earlier level's qa_corpus (N1) plus the memory bank (N3).
# Conceptually that is near-pure experience replay, and no benchmark in the
# repo compares it against a standard anti-forgetting method from the
# literature. Without one it stays plausible-but-unproven that the dream
# offers anything a classic replay buffer / parameter anchor would not.
#
# THREE ARMS, one flag apart (--anti-forgetting):
#   dream   N1 replays every level, N3 replays the whole bank   <- the build
#   ewc     N1 current level only, N3 off, online-EWC penalty in phases 1-2
#           (fisher.pt computed at every level boundary)
#   none    ewc's gating with no penalty                        <- the floor
# N2-A/N2-B/N2.5/REM/harvest and the phase-1 gold rehearsal (already
# current-level-only) stay identical across arms: the cross-level channel is
# the ONLY variable. REM's <=100 bank entries are a deliberate residual
# replay present in all arms.
#
# DREAMS is a FIXED count per level in every arm — not dream_until_plateau's
# rule, which would give arms different consolidation budgets and confound
# the treatment. 6 = MIN_DREAMS, the validated knee.
#
# MODES
#   MODE=sweep  L0->L2, seed 1, arms ewc_l100 ewc_l1000 ewc_l10000 — tune
#               lambda first: an untuned EWC arm would be unfair in reverse.
#   MODE=main   L0->L6, seeds 1 2, arms dream ewc none.
#
# GPU IS MANDATORY. Unlike exp_h this script replicates build.sh's full
# device block (oneAPI + LD_LIBRARY_PATH + PYTHONNOUSERSITE=1): without
# PYTHONNOUSERSITE a torch in ~/.local shadows the XPU build and everything
# runs on CPU silently at a third of the speed (measured: 911 vs 8283 tok/s).
# The run aborts unless torch.xpu is actually available; degrade to CPU only
# by exporting PHYSISML_DEVICE=cpu explicitly.
#
# ISOLATION
# Each arm x seed gets its own tree under models/exp_i/. Every arm runs the
# full curriculum from L0, so its session_*.jsonl accumulate naturally and
# the dream arm's N3 is fed (the 30.4%-vs-51.8% trap of isolated trees does
# not arise; there is deliberately no START_LEVEL knob — an incomplete arm
# restarts from scratch). training_files/<lang> is shared state phase 2
# rewrites (QA harvest), so it is snapshotted once and restored before every
# arm and at the end. Checkpoints are never shared between arms: already L0
# differs (the ewc/none dream skips N3 replay of L0's own memories).
#
# Usage:
#   MODE=sweep ./scripts/experiment_ewc.sh --confirm
#   MODE=main  ./scripts/experiment_ewc.sh --confirm
#   SKIP_DONE=1 MODE=main ./scripts/experiment_ewc.sh --confirm   # resume
#   EWC_LAMBDA=300 ARMS="ewc" SEEDS="1" MODE=main ./scripts/experiment_ewc.sh --confirm

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LNG="${LNG:-it}"
MODE="${MODE:-main}"
SESSIONS="${SESSIONS:-2}"
DREAMS="${DREAMS:-6}"          # fixed per level, ALL arms (see header)
EPOCHS_0="${EPOCHS_0:-3}"
TURNS="${TURNS:-100}"
SKIP_DONE="${SKIP_DONE:-0}"
EXP="${EXP:-models/exp_i}"
EWC_LAMBDA="${EWC_LAMBDA:-1000}"   # the 'ewc' arm's lambda (sweep arms carry
EWC_GAMMA="${EWC_GAMMA:-0.95}"     # their own in the arm name: ewc_l<value>)

if [ "$MODE" = "sweep" ]; then
    TARGET_LEVEL="${TARGET_LEVEL:-2}"
    SEEDS="${SEEDS:-1}"
    ARMS="${ARMS:-ewc_l100 ewc_l1000 ewc_l10000}"
else
    TARGET_LEVEL="${TARGET_LEVEL:-6}"
    SEEDS="${SEEDS:-1 2}"
    ARMS="${ARMS:-dream ewc none}"
fi

# arm name -> train_curriculum flags (echoed, so keep it one line)
arm_flags() {
    case "$1" in
        dream)  echo "--anti-forgetting dream" ;;
        none)   echo "--anti-forgetting none" ;;
        ewc)    echo "--anti-forgetting ewc --ewc-lambda $EWC_LAMBDA --ewc-gamma $EWC_GAMMA" ;;
        ewc_l*) echo "--anti-forgetting ewc --ewc-lambda ${1#ewc_l} --ewc-gamma $EWC_GAMMA" ;;
        *)      echo "" ;;
    esac
}

is_ewc_arm() { case "$1" in ewc|ewc_l*) return 0 ;; *) return 1 ;; esac; }

if [ "$1" != "--confirm" ]; then
    echo "  exp_i — dream replay vs online EWC, L0->$TARGET_LEVEL  (MODE=$MODE)"
    echo ""
    echo "  arms     : $ARMS"
    echo "  seeds    : $SEEDS"
    echo "  per level: phase 0 ($EPOCHS_0 ep) + $SESSIONS sessions x $TURNS turns + $DREAMS dreams (fixed)"
    echo "  ewc      : lambda=$EWC_LAMBDA gamma=$EWC_GAMMA (+ fisher.pt per level on ewc arms)"
    echo "  teacher  : local (deterministic, no API cost)"
    echo "  writes   : $EXP/   (models/checkpoints is never touched)"
    echo ""
    echo "  Re-run with --confirm."
    exit 0
fi

# ── Device: the Arc, or an explicit CPU opt-in ──────────────────────────────
# Replicates build.sh's block with two differences: the default is xpu (not
# auto), and an unavailable XPU aborts instead of falling back — the hour
# estimates assume the Arc, and a silent CPU fallback triples the run.
PHYSISML_DEVICE="${PHYSISML_DEVICE:-xpu}"
export PHYSISML_DEVICE
CONDA_GPU_PYTHON="$HOME/miniforge3/envs/physisml_gpu/bin/python"
ONEAPI_VARS="/opt/intel/oneapi/2025.3/oneapi-vars.sh"
[ ! -f "$ONEAPI_VARS" ] && ONEAPI_VARS="/opt/intel/oneapi/setvars.sh"

if [ "$PHYSISML_DEVICE" = "cpu" ]; then
    PY="python3"
    echo "  Device: cpu  (explicit PHYSISML_DEVICE=cpu — hour estimates x3)"
else
    if [ ! -f "$CONDA_GPU_PYTHON" ] || [ ! -f "$ONEAPI_VARS" ]; then
        echo "  ERROR: physisml_gpu env or oneAPI not found — the benchmark"
        echo "         must run on the Arc. PHYSISML_DEVICE=cpu to override."
        exit 1
    fi
    source "$ONEAPI_VARS" > /dev/null 2>&1
    if [ -d "$HOME/.local/lib" ]; then
        export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
    fi
    # Without this, a 'pip install --user torch' anywhere on the machine
    # silently replaces the env's XPU build (see build.sh for the measured
    # cost). exp_h did not carry this block; this experiment cannot afford
    # the ambiguity.
    export PYTHONNOUSERSITE=1
    PY="$CONDA_GPU_PYTHON"
    if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.xpu.is_available() else 1)" 2>/dev/null; then
        echo "  ERROR: torch.xpu is NOT available under $PY."
        echo "         (llama-server holding the card? ~/.local torch shadowing?)"
        echo "         Fix it, or run with PHYSISML_DEVICE=cpu explicitly."
        exit 1
    fi
    echo "  Device: xpu  (conda physisml_gpu, PYTHONNOUSERSITE=1)"
fi

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
trap 'rm -rf "$LOCK"' EXIT

# Pre-flight: corpora must match their qa_pairs.jsonl (see exp_h for why).
if ! $PY scripts/generate_qa_corpus.py --check --lang "$LNG" \
        --levels $(seq 0 "$TARGET_LEVEL") > /dev/null 2>&1; then
    echo ""
    echo "  ERROR: a qa_corpus.txt does not match its qa_pairs.jsonl."
    $PY scripts/generate_qa_corpus.py --check --lang "$LNG" \
        --levels $(seq 0 "$TARGET_LEVEL") 2>&1 | sed 's/^/    /'
    exit 1
fi
echo "  Corpora in step with their qa_pairs.jsonl"

# Snapshot the shared input once; restore it before every arm.
SNAP="$EXP/training_files_snapshot"
if [ ! -d "$SNAP/$LNG" ]; then
    mkdir -p "$SNAP"
    cp -a "training_files/$LNG" "$SNAP/"
    echo "  Snapshot: $SNAP/$LNG"
fi
restore_inputs() {
    [ -d "$SNAP/$LNG" ] || return 0   # never delete inputs without a snapshot
    rm -rf "training_files/$LNG"
    cp -a "$SNAP/$LNG" "training_files/"
}
# A Ctrl-C mid-arm leaves training_files/$LNG contaminated by the interrupted
# arm's harvest: put the snapshot back before the EXIT trap drops the lock.
trap 'restore_inputs; exit 130' INT TERM

# Per-level probe score (monitoring only — the frozen probe is read-only, so
# using it here costs nothing and shows an arm dying levels before the final
# retention matrix would).
probe_level() {
    local base="$1" level="$2" out="$3"
    local ckpt="$base/level_${level}/final_dreamed.pt" tokf="" l
    [ -f "$ckpt" ] || return 0
    for l in $(seq "$level" -1 0); do
        if [ -f "$base/level_${l}/tokenizer.json" ]; then
            tokf="$base/level_${l}/tokenizer.json"; break
        fi
    done
    [ -n "$tokf" ] || return 0
    {
        echo "=== level $level  $(date -Is) ==="
        $PY scripts/probe_set.py --score --checkpoint "$ckpt" \
            --tokenizer "$tokf" 2>&1
    } >> "$out"
}

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    FLAGS="$(arm_flags "$ARM")"
    if [ -z "$FLAGS" ]; then
        echo "  ERROR: unknown arm '$ARM'"; exit 1
    fi
    BASE="$EXP/${ARM}_s${SEED}"
    LOG="$EXP/${ARM}_s${SEED}.log"
    PROBELOG="$EXP/probe_${ARM}_s${SEED}.log"
    if [ "$SKIP_DONE" = "1" ] && [ -f "$BASE/.arm_complete" ]; then
        echo "  [skip] ${ARM}_s${SEED} already complete"
        continue
    fi

    restore_inputs
    rm -rf "$BASE"; mkdir -p "$BASE"
    rm -f "$PROBELOG"
    echo ""
    echo "================================================================"
    echo "  arm: $ARM   ($FLAGS)   seed $SEED"
    echo "================================================================"

    {
      echo "=== exp_i arm=$ARM seed=$SEED  L0->$TARGET_LEVEL  ($FLAGS) ==="
      for LEVEL in $(seq 0 "$TARGET_LEVEL"); do
        echo ""; echo "--- L$LEVEL phase 0 ---"
        $PY -u -m dynamic_model.train_curriculum \
            --phase 0 --level "$LEVEL" --lang "$LNG" \
            --epochs-0 "$EPOCHS_0" --seed "$SEED" \
            $FLAGS --ckpt-base "$BASE"

        PREV="$BASE/level_${LEVEL}/final.pt"
        DONE_DREAMS=0

        for S in $(seq 1 "$SESSIONS"); do
          echo ""; echo "--- L$LEVEL session $S/$SESSIONS ---"
          $PY -u -m dynamic_model.train_curriculum \
              --phase 1 --level "$LEVEL" --lang "$LNG" \
              --tutor-model local --interactions "$TURNS" \
              --no-turn-ckpt --seed "$SEED" \
              $FLAGS \
              --checkpoint "$PREV" --ckpt-base "$BASE"

          # Only the first session dreams in 'standard' mode, as build.sh
          # does: that is the dream that may grow the vocabulary.
          if [ "$S" -eq 1 ]; then MODE_D="standard"; else MODE_D="light"; fi
          echo ""; echo "--- L$LEVEL dream $MODE_D (after session $S) ---"
          $PY -u -m dynamic_model.train_curriculum \
              --phase 2 --level "$LEVEL" --lang "$LNG" \
              --dream-mode "$MODE_D" --seed "$SEED" \
              $FLAGS --ckpt-base "$BASE"
          DONE_DREAMS=$((DONE_DREAMS + 1))
          PREV="$BASE/level_${LEVEL}/final_dreamed.pt"
        done

        # Top up to DREAMS. --checkpoint is required: without it phase 2
        # restarts from final_learned.pt and repeats the first dream instead
        # of accumulating.
        while [ "$DONE_DREAMS" -lt "$DREAMS" ]; do
          echo ""; echo "--- L$LEVEL top-up dream $((DONE_DREAMS + 1))/$DREAMS ---"
          $PY -u -m dynamic_model.train_curriculum \
              --phase 2 --level "$LEVEL" --lang "$LNG" \
              --dream-mode standard --seed "$SEED" \
              $FLAGS \
              --checkpoint "$PREV" --ckpt-base "$BASE"
          DONE_DREAMS=$((DONE_DREAMS + 1))
        done

        # End-of-level boundary: the ewc arms anchor here, on the exact
        # checkpoint the next level inherits.
        if is_ewc_arm "$ARM"; then
          echo ""; echo "--- L$LEVEL fisher ---"
          $PY -u scripts/compute_fisher.py \
              --level "$LEVEL" --lang "$LNG" \
              --ckpt-base "$BASE" --gamma "$EWC_GAMMA"
        fi
      done
      echo ""; echo "=== arm $ARM seed $SEED complete ==="
    } 2>&1 | tee "$LOG"

    # Per-level probe curve (after the arm: probe_level loads checkpoints,
    # and interleaving it with training would double the device pressure).
    for LEVEL in $(seq 0 "$TARGET_LEVEL"); do
        probe_level "$BASE" "$LEVEL" "$PROBELOG" || true
    done

    # No-op guard: an ewc arm whose penalty never attached (fisher missing,
    # lambda mis-threaded) is 'none' wearing ewc's name. L0 legitimately has
    # no anchor, so the attach line must appear from L1 on.
    if is_ewc_arm "$ARM" && [ "$TARGET_LEVEL" -ge 1 ] \
            && ! grep -q "\[ewc\] penalty attached" "$LOG"; then
        echo ""
        echo "  ERROR: arm ${ARM}_s${SEED} never attached the EWC penalty"
        echo "         (grep '\[ewc\]' $LOG) — arm NOT marked complete."
        exit 1
    fi
    date -Is > "$BASE/.arm_complete"

    echo ""
    echo "  --- retention matrix, ${ARM}_s${SEED} ---"
    $PY scripts/retention_matrix.py --ckpt-base "$BASE" \
        --levels "0-$TARGET_LEVEL" \
        --json "$EXP/retention_${ARM}_s${SEED}.json" 2>&1 | tail -25
  done
done

restore_inputs
echo ""
echo "================================================================"
echo "  exp_i done  (MODE=$MODE)"
echo "================================================================"
for SEED in $SEEDS; do
  for ARM in $ARMS; do
    J="$EXP/retention_${ARM}_s${SEED}.json"
    [ -f "$J" ] && echo "  ${ARM}_s${SEED}: $J"
  done
done
echo ""
echo "  compare two arms (same seed):"
echo "    python3 scripts/retention_matrix.py --compare \\"
echo "        $EXP/retention_dream_s1.json $EXP/retention_ewc_s1.json"
echo ""
echo "  the three verdicts, against the 2.2-point run-to-run noise floor:"
echo "    ewc - none  > 2.2 on the final row  -> EWC does something"
echo "    dream - ewc < 2.2 on the final row  -> EWC matches the replay"
echo "    ewc diagonal within 2.2 of dream's  -> lambda is not taxing learning"
