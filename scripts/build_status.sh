#!/bin/bash
# Where is the build right now?
#
# build.sh prints everything it does but the log is thousands of lines, and
# the interesting state — which level, which session, what quality, what is
# running this second — is scattered across it. This reads it back.
#
# Usage:
#   ./scripts/build_status.sh [logfile]        # one shot
#   watch -n 60 ./scripts/build_status.sh      # keep an eye on it
#
# With no argument it takes the most recent *.log it can find in the usual
# places, so `./scripts/build_status.sh` alone normally does the right thing.

set -u
LOG="${1:-}"
if [ -z "$LOG" ]; then
  LOG=$(ls -t /tmp/claude-*/*/*/scratchpad/*.log ./*.log ./logs/*.log 2>/dev/null | head -1)
fi
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  echo "No build log found. Pass one: ./scripts/build_status.sh path/to.log"
  exit 1
fi

LANG_DIR="${LANG_DIR:-it}"
CKPT="models/checkpoints/$LANG_DIR"

echo "log      : $LOG"
STARTED=$(stat -c '%w' "$LOG" 2>/dev/null | cut -c1-16)
[ -n "$STARTED" ] && [ "$STARTED" != "-" ] && echo "started  : $STARTED"

# What is actually running now, and for how long. The build uses the conda
# python, whose process name is 'python' and not 'python3' — matching on the
# script name instead is what makes this work in both cases.
RUNNING=$(ps -eo etime=,args= | grep "[t]rain_curriculum.py" | head -1)
if [ -n "$RUNNING" ]; then
  ELAPSED=$(echo "$RUNNING" | awk '{print $1}')
  PHASE=$(echo "$RUNNING" | grep -o -- "--phase [0-9]*" | awk '{print $2}')
  LEVEL=$(echo "$RUNNING" | grep -o -- "--level [0-9]*" | awk '{print $2}')
  MODE=$(echo "$RUNNING"  | grep -o -- "--dream-mode [a-z]*" | awk '{print $2}')
  case "$PHASE" in
    0) WHAT="text training" ;;
    1) WHAT="teaching" ;;
    2) WHAT="dream${MODE:+ ($MODE)}" ;;
    *) WHAT="phase $PHASE" ;;
  esac
  echo "now      : level $LEVEL — $WHAT, running $ELAPSED"
else
  echo "now      : nothing running (finished, or aborted — check the tail)"
fi

echo
echo "levels done:"
for L in $(seq 0 12); do
  D="$CKPT/level_$L"
  [ -d "$D" ] || continue
  if [ -f "$D/final_dreamed.pt" ]; then
    WHEN=$(date -r "$D/final_dreamed.pt" '+%m-%d %H:%M')
    printf "  level %-2s dreamed   %s\n" "$L" "$WHEN"
  elif [ -f "$D/final_learned.pt" ]; then
    WHEN=$(date -r "$D/final_learned.pt" '+%m-%d %H:%M')
    printf "  level %-2s taught    %s  (dream pending)\n" "$L" "$WHEN"
  fi
done

echo
echo "sessions of the current level:"
grep -E "^  \[level [0-9]+\] Session" "$LOG" | tail -8 | sed 's/^  /  /'

# The gate decision is the line that says whether another session is coming.
echo
tail -400 "$LOG" | grep -E "✓ (Quality|High quality)|✗ (Regression|Best|MAX)|↻ Not converged|Plateau|consolidated with|Top-up dream" | tail -4

echo
echo "last log line:"
tail -1 "$LOG"
