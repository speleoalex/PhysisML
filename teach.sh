#!/bin/bash
# Automatic teaching session.
#
# Usage:
#   ./teach.sh                    # 10 turns, level 0, it, tutor=auto
#   ./teach.sh 100                # 100 turns
#   ./teach.sh 100 local          # rule-based tutor, offline
#   ./teach.sh 100 hybrid         # local prompts + ollama grading, offline
#   ./teach.sh 100 opus           # Claude tutor (needs ANTHROPIC_API_KEY)
#   ./teach.sh 100 auto it 1      # level 1 (default: 0)
#
# The Claude tutor is optional: 'auto' picks the local/hybrid teacher whenever
# the level ships training_files/<lang>/<level>/local_teacher.json.
#
# Each session loads level_N/final_learned.pt if it exists (continues),
# else starts from level_N/final.pt (first session).
# Saves to level_N/final_learned.pt — never touches active.pt.

INTERACTIONS=${1:-10}   # number of turns, or 'auto'
MODEL_ARG=${2:-auto}
LANG=${3:-it}
LEVEL=${4:-0}

case "$MODEL_ARG" in
  opus)             TUTOR="claude-opus-4-6"   ;;
  sonnet)           TUTOR="claude-sonnet-4-6" ;;
  haiku)            TUTOR="claude-haiku-4-5"  ;;
  local|hybrid|auto) TUTOR="$MODEL_ARG"       ;;
  *)                TUTOR="$MODEL_ARG"        ;;   # explicit model id
esac

# Load .env if ANTHROPIC_API_KEY is not already set
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

# The key is required only for the Claude tutor. 'auto' resolves to a local
# teacher whenever the level has a local_teacher.json, so check that too.
NEEDS_KEY=0
case "$TUTOR" in
  claude-*) NEEDS_KEY=1 ;;
  auto)     [ -f "training_files/$LANG/$LEVEL/local_teacher.json" ] || NEEDS_KEY=1 ;;
esac

if [ "$NEEDS_KEY" -eq 1 ] && [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Error: tutor '${TUTOR}' needs ANTHROPIC_API_KEY."
  echo "  Add it to .env, or: export ANTHROPIC_API_KEY=sk-ant-..."
  echo "  Or teach for free:  ./teach.sh ${INTERACTIONS} local ${LANG} ${LEVEL}"
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
