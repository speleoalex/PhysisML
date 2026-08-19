#!/bin/bash
# Experiment: comparison of different "between sessions" strategies at L1.
#
# Tests how teaching quality varies over 3 consecutive sessions
# by changing what happens BETWEEN sessions:
#
#   dream-only  : dream only, no text retrain
#   retrain-only: text retrain only (2 epochs), no dream
#   standard    : dream + retrain (current behaviour)
#   none        : nothing, consecutive sessions without pause
#
# Each condition uses a separate checkpoint directory.
# Starts from L0/final_dreamed.pt as the common base.
#
# Usage:
#   ./scripts/experiment_between_sessions.sh              # all 4 conditions
#   ./scripts/experiment_between_sessions.sh dream-only   # one condition only
#   ./scripts/experiment_between_sessions.sh --compare    # final comparison only

set -e
cd "$(dirname "$0")/.."

LANG="it"
LEVEL=1
SESSIONS=3
TEACH_TURNS=200
TUTOR="local"
RETRAIN_EPOCHS=2
BASE_CKPT="models/checkpoints/it/level_0/final_dreamed.pt"

# Load .env
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Which conditions to run
if [ "$1" = "--compare" ]; then
  CONDITIONS=""
elif [ -n "$1" ]; then
  CONDITIONS="$1"
else
  CONDITIONS="dream-only retrain-only standard none"
fi

if [ ! -f "$BASE_CKPT" ]; then
  echo "ERROR: Base checkpoint not found: $BASE_CKPT"
  echo "Run first: ./build.sh 0 local auto"
  exit 1
fi

# ── Quality measurement helper ────────────────────────────────────────────────
measure_quality() {
  local ckpt_dir="$1"
  local level="$2"
  python3 -c "
import glob, json
logs = sorted(glob.glob('${ckpt_dir}/level_${level}/session_*.jsonl'))
if not logs:
    print('0.00')
    exit()
recs = [json.loads(l) for l in open(logs[-1]) if l.strip()]
last20 = recs[-20:]
pos = sum(1 for r in last20 if r.get('feedback') in ('+++','++','+'))
rate = pos / max(len(last20), 1)
print(f'{rate:.2f}')
" 2>/dev/null || echo "0.00"
}

count_sessions() {
  local ckpt_dir="$1"
  local level="$2"
  ls "$ckpt_dir/level_$level"/session_*.jsonl 2>/dev/null | wc -l
}

# ── Run one condition ─────────────────────────────────────────────────────────
run_condition() {
  local COND="$1"
  local CKPT_DIR="models/experiment_between_sessions/$COND"
  mkdir -p "$CKPT_DIR"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  CONDITION: $COND"
  echo "  Dir: $CKPT_DIR"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Setup: create a fake level_0 dir with the base checkpoint so phase_0 picks it up
  mkdir -p "$CKPT_DIR/level_0" "$CKPT_DIR/level_1"
  cp "$BASE_CKPT" "$CKPT_DIR/level_0/final_dreamed.pt"
  # Copy L0 tokenizer if present
  L0_TOK="models/checkpoints/it/level_0/tokenizer.json"
  [ -f "$L0_TOK" ] && cp "$L0_TOK" "$CKPT_DIR/level_0/tokenizer.json" 2>/dev/null || true

  # Phase 0: text training to create final.pt for level 1
  echo ""
  echo "  [Phase 0] Text training L1 (4 epochs)..."
  python3 -u dynamic_model/train_curriculum.py \
    --phase    0 \
    --level    "$LEVEL" \
    --epochs-0 4 \
    --lang     "$LANG" \
    --ckpt-base "$CKPT_DIR"

  QUALITIES=()

  for S in $(seq 1 $SESSIONS); do
    echo ""
    echo "  [Session $S/$SESSIONS]"

    # Phase 1: teaching
    python3 -u dynamic_model/train_curriculum.py \
      --phase        1 \
      --level        "$LEVEL" \
      --interactions "$TEACH_TURNS" \
      --tutor-model  "$TUTOR" \
      --lang         "$LANG" \
      --ckpt-base    "$CKPT_DIR"

    Q=$(measure_quality "$CKPT_DIR" "$LEVEL")
    QUALITIES+=("S${S}=${Q}")
    echo "  → Quality S${S}: ${Q} (last20)"

    if [ "$S" -lt "$SESSIONS" ]; then
      echo "  [Between sessions: $COND]"

      case "$COND" in
        dream-only)
          python3 -u dynamic_model/train_curriculum.py \
            --phase      2 \
            --level      "$LEVEL" \
            --lang       "$LANG" \
            --dream-mode light \
            --ckpt-base  "$CKPT_DIR"
          ;;

        retrain-only)
          python3 -u dynamic_model/train_curriculum.py \
            --phase    0 \
            --level    "$LEVEL" \
            --epochs-0 "$RETRAIN_EPOCHS" \
            --lang     "$LANG" \
            --ckpt-base "$CKPT_DIR"
          ;;

        standard)
          python3 -u dynamic_model/train_curriculum.py \
            --phase      2 \
            --level      "$LEVEL" \
            --lang       "$LANG" \
            --dream-mode light \
            --ckpt-base  "$CKPT_DIR"
          python3 -u dynamic_model/train_curriculum.py \
            --phase    0 \
            --level    "$LEVEL" \
            --epochs-0 "$RETRAIN_EPOCHS" \
            --lang     "$LANG" \
            --ckpt-base "$CKPT_DIR"
          ;;

        none)
          echo "  (no operation between sessions)"
          ;;
      esac
    fi
  done

  echo ""
  echo "  Summary $COND: ${QUALITIES[*]}"
  echo "${COND}: ${QUALITIES[*]}" >> /tmp/experiment_between_sessions_results.txt
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT: between-session strategies at L1        ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Sessions per condition  : $SESSIONS"
echo "║  Turns per session       : $TEACH_TURNS"
echo "║  Teacher                 : $TUTOR"
echo "║  Base checkpoint         : $BASE_CKPT"
echo "╚══════════════════════════════════════════════════════╝"

> /tmp/experiment_between_sessions_results.txt

for COND in $CONDITIONS; do
  run_condition "$COND"
done

# ── Final comparison ─────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FINAL COMPARISON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "
import glob, json, os

conditions = {}
for cond in ['dream-only','retrain-only','standard','none']:
    d = f'models/experiment_between_sessions/{cond}/level_${LEVEL}'
    logs = sorted(glob.glob(f'{d}/session_*.jsonl'))
    if not logs: continue
    qualities = []
    for log in logs:
        recs = [json.loads(l) for l in open(log) if l.strip()]
        last20 = recs[-20:]
        pos = sum(1 for r in last20 if r.get('feedback') in ('+++','++','+'))
        qualities.append(pos / max(len(last20),1))
    conditions[cond] = qualities

print(f'  {\"Condition\":<16}  {\"S1\":>6}  {\"S2\":>6}  {\"S3\":>6}  {\"Trend\":>8}')
print(f'  {\"-\"*16}  {\"-\"*6}  {\"-\"*6}  {\"-\"*6}  {\"-\"*8}')
for cond, qs in conditions.items():
    trend = '+' if len(qs) >= 2 and qs[-1] > qs[0] else ('=' if len(qs) < 2 or qs[-1] == qs[0] else '-')
    vals = [f'{q:.0%}' for q in qs]
    while len(vals) < 3: vals.append('-')
    print(f'  {cond:<16}  {vals[0]:>6}  {vals[1]:>6}  {vals[2]:>6}  {trend:>8}')
print()
print('  Winner: condition with trend + and highest S3 value.')
" 2>/dev/null || cat /tmp/experiment_between_sessions_results.txt

echo ""
echo "  PPL comparison:"
python3 scripts/compare_checkpoints.py \
  models/experiment_between_sessions/dream-only/level_${LEVEL}/final_learned.pt \
  models/experiment_between_sessions/standard/level_${LEVEL}/final_learned.pt \
  2>/dev/null || echo "  (run --compare after completing all conditions)"
