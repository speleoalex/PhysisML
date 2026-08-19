#!/bin/bash
# Automatic teaching session via Claude tutor.
#
# Usage:
#   ./teach.sh                    # 10 turns, level 0, it, haiku
#   ./teach.sh 100                # 100 turns
#   ./teach.sh 100 opus           # with claude-opus-4-6
#   ./teach.sh 100 haiku it 1     # level 1 (default: 0)
#
# Each session loads level_N/final_learned.pt if it exists (continues),
# else starts from level_N/final.pt (first session).
# Saves to level_N/final_learned.pt — never touches active.pt.

INTERACTIONS=${1:-10}   # number of turns, or 'auto'
MODEL_ARG=${2:-haiku}
LANG=${3:-it}
LEVEL=${4:-0}

case "$MODEL_ARG" in
  opus)   TUTOR="claude-opus-4-6"   ;;
  sonnet) TUTOR="claude-sonnet-4-6" ;;
  *)      TUTOR="claude-haiku-4-5"  ;;
esac

# Load .env if ANTHROPIC_API_KEY is not already set
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY not set."
  echo "  Add it to .env or: export ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

echo "Teaching: level=${LEVEL}  turns=${INTERACTIONS}  tutor=${TUTOR}  lang=${LANG}"
echo ""

python3 dynamic_model/train_curriculum.py \
  --phase        1 \
  --level        "$LEVEL" \
  --interactions "$INTERACTIONS" \
  --tutor-model  "$TUTOR" \
  --lang         "$LANG"
