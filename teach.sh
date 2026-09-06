#!/bin/bash
# Automatic teaching session.
#
# Usage:
#   ./teach.sh                    # 10 turns, level 0, it, tutor=auto
#   ./teach.sh 100                # 100 turns
#   ./teach.sh 100 local          # rule-based tutor, offline
#   ./teach.sh 100 hybrid         # local prompts + local-LLM grading, offline
#   ./teach.sh 100 opus           # Claude tutor (needs ANTHROPIC_API_KEY)
#   ./teach.sh 100 auto it 1      # level 1 (default: 0)
#   ./teach.sh 100 auto --lang en --level 1
#
# The language picks the curriculum: training_files/<lang>/<level>/ for the
# material, models/checkpoints/<lang>/level_N/ for the weights. It can be given
# positionally (third argument) or as --lang, and PHYSISML_LANG sets it too.
#
# The Claude tutor is optional: 'auto' picks the local/hybrid teacher whenever
# the level ships training_files/<lang>/<level>/local_teacher.json.
#
# Each session loads level_N/final_learned.pt if it exists (continues),
# else starts from level_N/final.pt (first session).
# Saves to level_N/final_learned.pt — never touches active.pt.

# --lang/--level are pulled out first so they can appear anywhere; whatever is
# left keeps the historical positional meaning.
LANG="${PHYSISML_LANG:-it}"
LEVEL=""
_pos=()
_next=""
for arg in "$@"; do
  case "$_next" in
    lang)  LANG="$arg";  _next=""; continue ;;
    level) LEVEL="$arg"; _next=""; continue ;;
  esac
  case "$arg" in
    --lang)    _next=lang ;;
    --lang=*)  LANG="${arg#*=}" ;;
    --level)   _next=level ;;
    --level=*) LEVEL="${arg#*=}" ;;
    *)         _pos+=("$arg") ;;
  esac
done
set -- ${_pos[@]+"${_pos[@]}"}

INTERACTIONS=${1:-10}   # number of turns, or 'auto'
MODEL_ARG=${2:-auto}
[ -n "${3:-}" ] && LANG="$3"
[ -z "$LEVEL" ] && LEVEL=${4:-0}

if [ ! -d "training_files/$LANG/$LEVEL" ]; then
  echo "Error: no material for language '$LANG' level $LEVEL"
  echo "       (training_files/$LANG/$LEVEL/ is missing)."
  exit 1
fi
export PHYSISML_LANG="$LANG"

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
