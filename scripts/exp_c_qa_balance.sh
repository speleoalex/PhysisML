#!/bin/bash
# Experiment exp_c: QA balance — corpus vs dream epochs
#
# Tests 4 conditions on L1 (3 sessions) to measure impact of:
#   A) Including qa_corpus.txt in phase-0 text training
#   B) More N2.5 dream epochs (15 vs 5)
#
# Conditions:
#   baseline    : no qa_corpus in training, 5 dream N2.5 epochs (old default)
#   qa_corpus   : qa_corpus in training, 5 dream N2.5 epochs
#   qa_epochs   : no qa_corpus in training, 15 dream N2.5 epochs
#   qa_both     : qa_corpus in training, 15 dream N2.5 epochs (current default)
#
# Each condition:
#   - Starts from a SHARED L0/final_dreamed.pt (pre-built)
#   - Runs L1 phase 0 (text training, 3 epochs)
#   - Runs L1 3 teaching sessions (local teacher, auto mode)
#   - Runs L1 dream after each session
#
# Usage:
#   ./scripts/exp_c_qa_balance.sh [L0_DREAMED_PT]
#
#   L0_DREAMED_PT: path to L0 final_dreamed.pt to start from
#                  (default: models/checkpoints/it/level_0/final_dreamed.pt)
#
# Results saved in: models/exp_c/{condition}/
# Summary printed at the end.

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Config ---
L0_BASE="${1:-models/checkpoints/it/level_0/final_dreamed.pt}"
EXP_DIR="models/exp_c"
EPOCHS_1=3
SESSIONS=3
LANG="it"
LEVEL=1
GPU_PYTHON="${CONDA_GPU_PYTHON:-python3}"

# To resume after a partial run, set SKIP_CONDITIONS to conditions already complete.
# Example: SKIP_CONDITIONS="baseline qa_corpus" ./scripts/exp_c_qa_balance.sh
SKIP_CONDITIONS="${SKIP_CONDITIONS:-}"
ALL_CONDITIONS=("baseline" "qa_corpus" "qa_epochs" "qa_both")
CONDITIONS=()
for c in "${ALL_CONDITIONS[@]}"; do
    if echo "$SKIP_CONDITIONS" | grep -qw "$c"; then
        echo "  [skip] $c (already complete)"
    else
        CONDITIONS+=("$c")
    fi
done

# Args per condition:
#   baseline   : --no-qa-corpus --n-qa-epochs 5
#   qa_corpus  : (default qa_corpus included) --n-qa-epochs 5
#   qa_epochs  : --no-qa-corpus --n-qa-epochs 15
#   qa_both    : (default) --n-qa-epochs 15
declare -A EXTRA_ARGS=(
    [baseline]="--no-qa-corpus --n-qa-epochs 5"
    [qa_corpus]="--n-qa-epochs 5"
    [qa_epochs]="--no-qa-corpus --n-qa-epochs 15"
    [qa_both]="--n-qa-epochs 15"
)

# Select Python: use conda physisml_gpu env if available (GPU), else system python3 (CPU)
_CONDA_PY="$HOME/miniforge3/envs/physisml_gpu/bin/python"
if [ -n "$CONDA_GPU_PYTHON" ]; then
    GPU_PYTHON="$CONDA_GPU_PYTHON"
elif [ -f "$_CONDA_PY" ] && [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
    GPU_PYTHON="$_CONDA_PY"
    echo "  GPU: Intel Arc XPU  (conda physisml_gpu)"
else
    GPU_PYTHON="python3"
    echo "  GPU: unavailable — using CPU"
fi

echo "================================================================"
echo "  exp_c: QA balance experiment"
echo "  L0 base: $L0_BASE"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Epochs/phase-0: $EPOCHS_1  Sessions: $SESSIONS"
echo "================================================================"

if [ ! -f "$L0_BASE" ]; then
    echo "ERROR: L0 base checkpoint not found: $L0_BASE"
    echo "Run the build up to L0 first, then re-run this script."
    exit 1
fi

mkdir -p "$EXP_DIR"

# --- Run each condition ---
for COND in "${CONDITIONS[@]}"; do
    CKPT_BASE="$EXP_DIR/$COND"
    EXTRA="${EXTRA_ARGS[$COND]}"
    LOG_FILE="$EXP_DIR/${COND}.log"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Condition: $COND  ($EXTRA)"
    echo "  Checkpoint dir: $CKPT_BASE"
    echo "----------------------------------------------------------------"

    # Seed: copy L0 dreamed as starting point for L1
    mkdir -p "$CKPT_BASE/level_0"
    cp "$L0_BASE" "$CKPT_BASE/level_0/final_dreamed.pt"

    # Also copy tokenizer if present alongside L0
    L0_TOK="$(dirname "$L0_BASE")/tokenizer.json"
    if [ -f "$L0_TOK" ]; then
        cp "$L0_TOK" "$CKPT_BASE/level_0/tokenizer.json"
    fi

    {
        echo "=== exp_c condition: $COND  $(date) ==="
        echo "Extra args: $EXTRA"

        # Phase 0: text training
        echo ""
        echo "--- Phase 0 (text training) ---"
        $GPU_PYTHON -m dynamic_model.train_curriculum \
            --phase 0 \
            --level $LEVEL \
            --lang $LANG \
            --epochs-0 $EPOCHS_1 \
            --ckpt-base "$CKPT_BASE" \
            $EXTRA

        PREV_CKPT="$CKPT_BASE/level_${LEVEL}/final.pt"

        # Sessions + dream
        for SESSION in $(seq 1 $SESSIONS); do
            echo ""
            echo "--- Session $SESSION / $SESSIONS ---"
            $GPU_PYTHON -m dynamic_model.train_curriculum \
                --phase 1 \
                --level $LEVEL \
                --lang $LANG \
                --tutor-model local \
                --interactions auto \
                --no-turn-ckpt \
                --checkpoint "$PREV_CKPT" \
                --ckpt-base "$CKPT_BASE" \
                $EXTRA

            echo ""
            echo "--- Dream after session $SESSION ---"
            $GPU_PYTHON -m dynamic_model.train_curriculum \
                --phase 2 \
                --level $LEVEL \
                --lang $LANG \
                --dream-mode standard \
                --ckpt-base "$CKPT_BASE" \
                $EXTRA

            PREV_CKPT="$CKPT_BASE/level_${LEVEL}/final_dreamed.pt"
        done

        echo ""
        echo "=== Condition $COND completed ==="

    } 2>&1 | tee "$LOG_FILE"

    echo "  Log: $LOG_FILE"
done

echo ""
echo "================================================================"
echo "  exp_c COMPLETE"
echo "================================================================"
echo ""
echo "Summary per condition:"
echo ""

for COND in "${CONDITIONS[@]}"; do
    LOG="$EXP_DIR/${COND}.log"
    if [ ! -f "$LOG" ]; then
        echo "  $COND: log not found"
        continue
    fi

    # Extract positive rate from last session summary
    POS_LINE=$(grep "Positive (+/++/+++):" "$LOG" | tail -1)
    NEG_LINE=$(grep "Negative (-):" "$LOG" | tail -1)
    # Extract final N2.5 loss from last dream
    N25_LINE=$(grep "N2.5 loss:" "$LOG" | tail -1)
    # Extract final N1 loss from last dream
    N1_LINE=$(grep "N1 loss:" "$LOG" | tail -1)
    # Count new tokens added
    NEW_TOK=$(grep "N2-B: +" "$LOG" | tail -1 || echo "N2-B: +0 new tokens")

    echo "  [$COND]"
    echo "    ${POS_LINE:-positive: n/a}"
    echo "    ${NEG_LINE:-negative: n/a}"
    echo "    ${N25_LINE:-N2.5: n/a}"
    echo "    ${N1_LINE:-N1: n/a}"
    echo "    ${NEW_TOK}"
    echo ""
done

echo "For a detailed log comparison:"
for COND in "${CONDITIONS[@]}"; do
    echo "  grep -E 'Positive|N2.5|N1 loss' $EXP_DIR/${COND}.log"
done
echo ""
echo "To test a model:"
echo "  ./set_model.sh $EXP_DIR/<condition>/level_${LEVEL}/final_dreamed.pt"
