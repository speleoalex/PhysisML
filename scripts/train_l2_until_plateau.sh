#!/bin/bash
# Continue training L2 only, session by session, until plateau detected.
# Resumes from models/checkpoints/it/level_2/final_learned.pt.
#
# Plateau: max improvement over the last PLATEAU_WINDOW sessions < MIN_DELTA.
# Stops also at MAX_SESSIONS as a safety cap.
#
# Usage:
#   ./scripts/train_l2_until_plateau.sh                   # default 12 max sessions
#   ./scripts/train_l2_until_plateau.sh 20                # custom max
set -e

LEVEL=2
LANG=it
INTERACTIONS=200
TUTOR_MODEL=hybrid

PLATEAU_WINDOW=3
MIN_DELTA=0.03
MIN_SESSIONS=4
MAX_SESSIONS=${1:-12}

CKPT_DIR="models/checkpoints/${LANG}/level_${LEVEL}"

# Python selection: prefer GPU env if it imports torch successfully,
# otherwise fall back to CPU env (created via miniforge envs/physisml_cpu).
CONDA_GPU_PYTHON="$HOME/miniforge3/envs/physisml_gpu/bin/python"
CONDA_CPU_PYTHON="$HOME/miniforge3/envs/physisml_cpu/bin/python"

PYTHON=""
ONEAPI_VARS="/opt/intel/oneapi/2025.3/oneapi-vars.sh"
[ ! -f "$ONEAPI_VARS" ] && ONEAPI_VARS="/opt/intel/oneapi/setvars.sh"

if [ -f "$CONDA_GPU_PYTHON" ] && [ -f "$ONEAPI_VARS" ]; then
  source "$ONEAPI_VARS" > /dev/null 2>&1
  [ -d "$HOME/.local/lib" ] && export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
  if "$CONDA_GPU_PYTHON" -c "import torch" 2>/dev/null; then
    PYTHON="$CONDA_GPU_PYTHON"
    echo "  Using GPU env (XPU)"
  fi
fi
if [ -z "$PYTHON" ] && [ -f "$CONDA_CPU_PYTHON" ]; then
  PYTHON="$CONDA_CPU_PYTHON"
  echo "  Using CPU env (GPU unavailable)"
fi
if [ -z "$PYTHON" ]; then
  PYTHON="python3"
  echo "  Using system python3"
fi

if [ ! -f "${CKPT_DIR}/final_learned.pt" ]; then
  echo "ERROR: ${CKPT_DIR}/final_learned.pt not found. Run the full build first."
  exit 1
fi

echo "════════════════════════════════════════════════════════"
echo "  L2 plateau training (max ${MAX_SESSIONS} sessions)"
echo "════════════════════════════════════════════════════════"

RATES=()
SESSION=0
BEST_RATE=0

while [ "$SESSION" -lt "$MAX_SESSIONS" ]; do
  SESSION=$((SESSION + 1))
  echo ""
  echo "── Session $SESSION/$MAX_SESSIONS ───────────────────────────"

  T0=$(date +%s)
  "$PYTHON" -u dynamic_model/train_curriculum.py \
    --phase 1 --level $LEVEL --interactions $INTERACTIONS \
    --tutor-model $TUTOR_MODEL --lang $LANG
  ELAPSED=$(( $(date +%s) - T0 ))

  # Compute rate from latest session JSONL
  LATEST_SESS=$(ls -t "${CKPT_DIR}"/session_*.jsonl 2>/dev/null | head -1)
  RATE=$("$PYTHON" -c "
import json
turns = [json.loads(l) for l in open('${LATEST_SESS}') if l.strip()]
fb = [t.get('feedback') for t in turns if 'feedback' in t]
strong = sum(1 for f in fb if f in ('++', '+++'))
print(f'{100*strong/max(len(fb),1):.1f}')
" 2>/dev/null || echo "0")

  RATES+=("$RATE")
  echo ""
  echo "  → Session $SESSION: ${RATE}%  duration=${ELAPSED}s"
  echo "  → History: ${RATES[*]}"

  # Save best if improved
  RATE_INT=$("$PYTHON" -c "print(int(float('${RATE}') * 10))")
  BEST_INT=$("$PYTHON" -c "print(int(float('${BEST_RATE}') * 10))")
  if [ "$RATE_INT" -gt "$BEST_INT" ]; then
    BEST_RATE="$RATE"
    cp "${CKPT_DIR}/final_dreamed.pt" "${CKPT_DIR}/best_session.pt" 2>/dev/null || true
    echo "  ↑ New best: ${RATE}% — best_session.pt updated"
  fi

  # Plateau check
  N=${#RATES[@]}
  if [ "$N" -ge "$MIN_SESSIONS" ] && [ "$N" -ge "$PLATEAU_WINDOW" ]; then
    PLATEAU=$("$PYTHON" -c "
rates = [float(r) for r in '${RATES[*]}'.split()]
recent = rates[-${PLATEAU_WINDOW}:]
improvements = [recent[i] - recent[i-1] for i in range(1, len(recent))]
print(1 if max(improvements, default=0) < ${MIN_DELTA}*100 else 0)
")
    if [ "$PLATEAU" = "1" ]; then
      echo ""
      echo "════════════════════════════════════════════════════════"
      echo "  ~ PLATEAU detected — peak ${BEST_RATE}%"
      echo "  Rates of the last ${PLATEAU_WINDOW} sessions: ${RATES[*]: -${PLATEAU_WINDOW}}"
      echo "════════════════════════════════════════════════════════"
      break
    fi
  fi
done

echo ""
echo "Done. Sessions: $SESSION  Best: ${BEST_RATE}%"
echo "Rates: ${RATES[*]}"
