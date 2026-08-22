#!/bin/bash
# Experiment exp_d: what does dynamic vocabulary growth actually buy?
#
# Six arms over L0→L2, each in its own checkpoint tree, all sharing the level
# corpora, the deterministic local teacher and the seed:
#
#   static           tokenizer frozen at 8002 (--vocab-growth off)  ← control
#   dyn_both         current build behaviour                        ← reference
#   dyn_gold         growth from gold answers only     (acquisition)
#   dyn_prompt       growth from teacher prompts only  (exposure)
#   dyn_random_init  growth on, embeddings not inherited from the parents
#   dyn_noprotect    growth on, current level's drill targets not protected
#
# WHY L0→L2: 50 of the 81 tokens of the validated L0→L10 build are born at
# L1-L2, and those corpora are 88K/156K/212K — a full run is ~45-60 min. From
# L3 the corpora jump to 21/26/80MB (L4 alone took ~2h in the reference build),
# so higher levels are a separate campaign, not an afterthought here.
#
# COST: 6 arms × SEEDS(3) × ~50 min ≈ 15h of GPU. No API cost — the teacher is
# local and deterministic.
#
# Usage:
#   ./scripts/experiment_vocab_growth.sh --confirm            # all arms, seeds 1-3
#   SEEDS="1"     ./scripts/experiment_vocab_growth.sh --confirm
#   ARMS="static dyn_both" ./scripts/experiment_vocab_growth.sh --confirm
#   SKIP_DONE=1   ./scripts/experiment_vocab_growth.sh --confirm   # resume
#
# Analysis, once the runs are done:
#   python3 scripts/analyze_vocab_growth.py --ckpt-base models/exp_d/dyn_gold_s1
#   python3 scripts/ablate_new_tokens.py    --ckpt-base models/exp_d/dyn_both_s1 \
#           --levels 0-2 --mode cohort

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LANG="it"
TARGET_LEVEL=2
EPOCHS_0=3
SESSIONS=4            # build.sh runs at least MIN_SESSIONS_DYNAMIC=4 per level
EXP_DIR="models/exp_d"
SEEDS="${SEEDS:-1 2 3}"
SKIP_DONE="${SKIP_DONE:-0}"

ALL_ARMS="static dyn_both dyn_gold dyn_prompt dyn_random_init dyn_noprotect"
ARMS="${ARMS:-$ALL_ARMS}"

# Only the flags under test differ between arms — everything else is the build
# default, so a difference in the result belongs to the flag.
arm_args() {
    case "$1" in
        static)          echo "--vocab-growth off" ;;
        dyn_both)        echo "" ;;
        dyn_gold)        echo "--vocab-growth gold" ;;
        dyn_prompt)      echo "--vocab-growth prompt" ;;
        dyn_random_init) echo "--growth-init random" ;;
        dyn_noprotect)   echo "--protect-scope none" ;;
        *) echo "UNKNOWN" ;;
    esac
}

CONFIRMED=0
for arg in "$@"; do
    [ "$arg" = "--confirm" ] && CONFIRMED=1
done

N_ARMS=$(echo $ARMS | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
N_RUNS=$((N_ARMS * N_SEEDS))

echo "================================================================"
echo "  exp_d: vocabulary growth arms — L0→$TARGET_LEVEL"
echo "  Arms:  $ARMS"
echo "  Seeds: $SEEDS"
echo "  Runs:  $N_RUNS  (~50 min each → ~$((N_RUNS * 50 / 60))h of GPU, no API cost)"
echo "  Output: $EXP_DIR/{arm}_s{seed}/"
echo "================================================================"

if [ "$CONFIRMED" -ne 1 ]; then
    echo ""
    echo "  Dry run: re-run with --confirm to actually start."
    echo "  Per-arm flags:"
    for ARM in $ARMS; do
        printf "    %-16s %s\n" "$ARM" "$(arm_args "$ARM")"
    done
    exit 0
fi

# Select Python: conda physisml_gpu if available (XPU), else system python3.
_CONDA_PY="$HOME/miniforge3/envs/physisml_gpu/bin/python"
if [ -n "$CONDA_GPU_PYTHON" ]; then
    PY="$CONDA_GPU_PYTHON"
elif [ -f "$_CONDA_PY" ]; then
    for VARS in /opt/intel/oneapi/2025.3/oneapi-vars.sh /opt/intel/oneapi/setvars.sh; do
        [ -f "$VARS" ] && source "$VARS" --force > /dev/null 2>&1 && break
    done
    [ -d "$HOME/.local/lib" ] && export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
    PY="$_CONDA_PY"
else
    PY="python3"
fi
echo "  Python: $PY"

mkdir -p "$EXP_DIR"

for ARM in $ARMS; do
    EXTRA="$(arm_args "$ARM")"
    if [ "$EXTRA" = "UNKNOWN" ]; then
        echo "  Unknown arm: $ARM — skipped"
        continue
    fi

    for SEED in $SEEDS; do
        RUN="${ARM}_s${SEED}"
        CKPT_BASE="$EXP_DIR/$RUN"
        LOG="$EXP_DIR/${RUN}.log"

        if [ "$SKIP_DONE" -eq 1 ] && \
           [ -f "$CKPT_BASE/level_${TARGET_LEVEL}/final_dreamed.pt" ]; then
            echo "  [skip] $RUN already complete"
            continue
        fi

        echo ""
        echo "----------------------------------------------------------------"
        echo "  $RUN   flags: ${EXTRA:-<build default>}   seed: $SEED"
        echo "----------------------------------------------------------------"
        mkdir -p "$CKPT_BASE"

        {
            echo "=== exp_d $RUN  $(date) ==="
            echo "Arm flags: ${EXTRA:-<none>}   seed: $SEED"

            for LEVEL in $(seq 0 $TARGET_LEVEL); do
                echo ""
                echo "--- L$LEVEL phase 0 (text training) ---"
                $PY -u -m dynamic_model.train_curriculum \
                    --phase 0 --level "$LEVEL" --lang "$LANG" \
                    --epochs-0 "$EPOCHS_0" --seed "$SEED" \
                    --ckpt-base "$CKPT_BASE" $EXTRA

                PREV="$CKPT_BASE/level_${LEVEL}/final.pt"

                for SESSION in $(seq 1 $SESSIONS); do
                    echo ""
                    echo "--- L$LEVEL session $SESSION/$SESSIONS ---"
                    $PY -u -m dynamic_model.train_curriculum \
                        --phase 1 --level "$LEVEL" --lang "$LANG" \
                        --tutor-model local --interactions auto \
                        --no-turn-ckpt --seed "$SEED" \
                        --checkpoint "$PREV" \
                        --ckpt-base "$CKPT_BASE" $EXTRA

                    # Only session 1 dreams in 'standard' mode, exactly as
                    # build.sh does — that is the only dream that grows vocab.
                    if [ "$SESSION" -eq 1 ]; then DREAM="standard"; else DREAM="light"; fi
                    echo ""
                    echo "--- L$LEVEL dream ($DREAM) after session $SESSION ---"
                    $PY -u -m dynamic_model.train_curriculum \
                        --phase 2 --level "$LEVEL" --lang "$LANG" \
                        --dream-mode "$DREAM" --seed "$SEED" \
                        --ckpt-base "$CKPT_BASE" $EXTRA

                    PREV="$CKPT_BASE/level_${LEVEL}/final_dreamed.pt"
                done
            done
            echo ""
            echo "=== $RUN completed ==="
        } 2>&1 | tee "$LOG"
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  exp_d COMPLETE — quality per arm (greedy exact match)"
echo "================================================================"
for ARM in $ARMS; do
    for SEED in $SEEDS; do
        RUN="${ARM}_s${SEED}"
        CKPT_BASE="$EXP_DIR/$RUN"
        CKPT="$CKPT_BASE/level_${TARGET_LEVEL}/final_dreamed.pt"
        [ -f "$CKPT" ] || { echo "  $RUN: incomplete"; continue; }
        TOKENS=$(grep -c '"token_id"' "$CKPT_BASE"/level_*/growth_events.jsonl 2>/dev/null \
                 | awk -F: '{s+=$2} END {print s+0}')
        EXACT=$($PY dynamic_model/test_model.py --level "$TARGET_LEVEL" \
                    --checkpoint "$CKPT" --samples 0 2>/dev/null \
                | grep "Risposte esatte:" | tail -1)
        printf "  %-20s token nuovi: %-4s  %s\n" "$RUN" "$TOKENS" "${EXACT:-n/a}"
    done
done
echo ""
echo "  Poi: python3 scripts/analyze_vocab_growth.py --ckpt-base $EXP_DIR/<run> --max-level $TARGET_LEVEL"
