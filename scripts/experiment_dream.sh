#!/bin/bash
# Experiment: comparison L0-L2 with and without the dream phase.
#
# Trains the same curriculum (L0→L2) twice in separate directories:
#   Branch A — WITHOUT DREAM: models/experiment_dream/nodream/
#   Branch B — WITH DREAM:    models/experiment_dream/dream/
#
# Then compares with compare_checkpoints.py.
#
# Usage:
#   ./scripts/experiment_dream.sh              # both runs + comparison
#   ./scripts/experiment_dream.sh --nodream    # branch A only (no dream)
#   ./scripts/experiment_dream.sh --dream      # branch B only (with dream)
#   ./scripts/experiment_dream.sh --compare    # final comparison only

set -e
cd "$(dirname "$0")/.."   # ensure we work from the project root

LANG="it"
TARGET_LEVEL=2
EPOCHS_0=10
TUTOR="claude-haiku-4-5"
TEACH_TURNS=100

DIR_A="models/experiment_dream/nodream"
DIR_B="models/experiment_dream/dream"

# Load .env
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY not set."
  exit 1
fi

RUN_A=1; RUN_B=1
for arg in "$@"; do
  case "$arg" in
    --nodream) RUN_B=0 ;;
    --dream)   RUN_A=0 ;;
    --compare) RUN_A=0; RUN_B=0 ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║       EXPERIMENT: dream vs. no-dream (L0-L2)         ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Branch A (no dream): $DIR_A"
echo "║  Branch B (dream):    $DIR_B"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Branch A: WITHOUT DREAM ──────────────────────────────────────────────────
if [ "$RUN_A" -eq 1 ]; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Branch A — WITHOUT DREAM"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  mkdir -p "$DIR_A"

  for LEVEL in $(seq 0 $TARGET_LEVEL); do
    echo ""
    echo "  [A-L$LEVEL] Phase 0: text training..."
    python3 -u dynamic_model/train_curriculum.py \
      --phase    0 \
      --level    "$LEVEL" \
      --epochs-0 "$EPOCHS_0" \
      --lang     "$LANG" \
      --ckpt-base "$DIR_A"

    echo "  [A-L$LEVEL] Phase 1: Claude teaching ($TEACH_TURNS turns)..."
    python3 -u dynamic_model/train_curriculum.py \
      --phase        1 \
      --level        "$LEVEL" \
      --interactions "$TEACH_TURNS" \
      --tutor-model  "$TUTOR" \
      --lang         "$LANG" \
      --ckpt-base    "$DIR_A"

    echo "  [A-L$LEVEL] No dream — level complete."
  done

  echo ""
  echo "  ✓ Branch A complete: $DIR_A"
fi

# ── Branch B: WITH DREAM ─────────────────────────────────────────────────────
if [ "$RUN_B" -eq 1 ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Branch B — WITH DREAM"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  mkdir -p "$DIR_B"

  for LEVEL in $(seq 0 $TARGET_LEVEL); do
    echo ""
    echo "  [B-L$LEVEL] Phase 0: text training..."
    python3 -u dynamic_model/train_curriculum.py \
      --phase    0 \
      --level    "$LEVEL" \
      --epochs-0 "$EPOCHS_0" \
      --lang     "$LANG" \
      --ckpt-base "$DIR_B"

    echo "  [B-L$LEVEL] Phase 1: Claude teaching ($TEACH_TURNS turns)..."
    python3 -u dynamic_model/train_curriculum.py \
      --phase        1 \
      --level        "$LEVEL" \
      --interactions "$TEACH_TURNS" \
      --tutor-model  "$TUTOR" \
      --lang         "$LANG" \
      --ckpt-base    "$DIR_B"

    echo "  [B-L$LEVEL] Phase 2: dream (N1+N2+N3+REM)..."
    python3 -u dynamic_model/train_curriculum.py \
      --phase     2 \
      --level     "$LEVEL" \
      --lang      "$LANG" \
      --ckpt-base "$DIR_B"
  done

  echo ""
  echo "  ✓ Branch B complete: $DIR_B"
fi

# ── Comparison ───────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FINAL COMPARISON: no-dream vs dream"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for LEVEL in $(seq 0 $TARGET_LEVEL); do
  A="$DIR_A/level_$LEVEL/final_learned.pt"
  B="$DIR_B/level_$LEVEL/final_dreamed.pt"
  [ ! -f "$B" ] && B="$DIR_B/level_$LEVEL/final_learned.pt"

  if [ -f "$A" ] && [ -f "$B" ]; then
    echo ""
    echo "  Level $LEVEL:"
    echo "    A (no dream):   $A"
    echo "    B (with dream): $B"
    python3 scripts/compare_checkpoints.py "$A" "$B" 2>/dev/null
  else
    echo "  Level $LEVEL: missing files (A=$A, B=$B)"
  fi
done

# PPL timeline for both branches
echo ""
echo "  PPL timeline Branch A (no dream):"
python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tests/test_1')
from scripts.compare_checkpoints import ppl_timeline
ppl_timeline('$LANG', base='$DIR_A')
" 2>/dev/null || echo "  (error in timeline A)"

echo ""
echo "  PPL timeline Branch B (with dream):"
python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tests/test_1')
from scripts.compare_checkpoints import ppl_timeline
ppl_timeline('$LANG', base='$DIR_B')
" 2>/dev/null || echo "  (error in timeline B)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Experiment complete."
echo "  Logs saved in: $DIR_A  and  $DIR_B"
