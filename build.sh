#!/bin/bash
# Autobuild: trains the model from level 0 to level N automatically.
# Each level = phase 0 (text training) + teaching turns with Claude.
#
# Usage:
#   ./build.sh 1              # build levels 0→1, interactive start-level prompt
#   ./build.sh 2 opus         # use claude-opus-4-6 as tutor
#   ./build.sh 1 haiku 200    # 200 teaching turns per level
#   ./build.sh 1 --resume     # skip prompt, continue from next incomplete level
#   ./build.sh 2 haiku 100 --resume
#   ./build.sh 5 --lang en    # build the English curriculum (default: it)
#
# The language selects both the corpus (training_files/<lang>/) and where the
# weights land (models/checkpoints/<lang>/), so two languages never overwrite
# each other. PHYSISML_LANG=en works too.

# ── Language ────────────────────────────────────────────────────────────────
# The curriculum exists per language: training_files/<lang>/ holds the corpus
# and the teacher configs, models/checkpoints/<lang>/ the weights. Which one
# gets built is an argument, not a constant:
#
#   ./build.sh 5 --lang en          PHYSISML_LANG=en ./build.sh 5
#
# This deliberately does NOT read the shell's own $LANG (the locale). The two
# only share a name, and honouring the locale would send anyone running with
# LANG=en_US.UTF-8 into a different curriculum than the one they asked for.
LANG="${PHYSISML_LANG:-it}"
_args=()
_lang_next=0
for arg in "$@"; do
  if [ "$_lang_next" -eq 1 ]; then LANG="$arg"; _lang_next=0; continue; fi
  case "$arg" in
    --lang)   _lang_next=1 ;;
    --lang=*) LANG="${arg#*=}" ;;
    *)        _args+=("$arg") ;;
  esac
done
set -- ${_args[@]+"${_args[@]}"}
export PHYSISML_LANG="$LANG"

if [ ! -d "training_files/$LANG" ]; then
  echo "Error: no curriculum for language '$LANG' — training_files/$LANG/ is missing."
  printf "  Available:"
  for d in training_files/*/; do
    b=$(basename "$d"); [ ${#b} -eq 2 ] && printf " %s" "$b"
  done
  echo ""
  exit 1
fi
echo "  Language: $LANG  (training_files/$LANG → models/checkpoints/$LANG)"

TARGET_LEVEL=${1:-1}
# Highest level this language has, read off the disk. Used by the loops that
# scan or clean EVERY level rather than the requested range: those were
# hardcoded to 10 and silently ignored anything above it. Pinning the number
# instead is what made the Italian tree stop at 12 after level 13 was written,
# and it also left a stale level_13 checkpoint behind on a rebuild from an
# earlier level. Checkpoints count too, for exactly that reason: the clean-up
# loop must reach every directory that exists, not every level that is taught.
MAX_LEVEL=0
for _d in "training_files/$LANG"/[0-9]* "models/checkpoints/$LANG"/level_[0-9]*; do
  _n="${_d##*/}"; _n="${_n#level_}"
  case "$_n" in
    ''|*[!0-9]*) continue ;;
  esac
  [ "$_n" -gt "$MAX_LEVEL" ] && MAX_LEVEL="$_n"
done
if [ "$TARGET_LEVEL" -gt "$MAX_LEVEL" ] 2>/dev/null; then
  echo "Error: the '$LANG' curriculum stops at level $MAX_LEVEL, but level $TARGET_LEVEL was requested."
  exit 1
fi

# ── Device selection ────────────────────────────────────────────────────────
# Use the Intel Arc A370M if the conda physisml_gpu env is available.
# Native PyTorch+XPU (2.11+) needs the oneAPI 2025.3 runtime + pip-installed MKL.
# Falls back to system python3 (CPU) if the XPU import fails.
#
#   PHYSISML_DEVICE=auto   (default) GPU when usable, else CPU
#   PHYSISML_DEVICE=cpu              force the CPU: skips the GPU env entirely
#   PHYSISML_DEVICE=xpu              force the Arc: refuses to fall back quietly
#
# The two are interchangeable mid-build: a checkpoint written on the Arc reads
# back bit-identically on the CPU and the other way round (load() maps to CPU
# and the trainer moves the model), so a level can be trained on one and the
# next on the other. Verified both directions. Measured, at level 6: a dream
# takes 6 minutes on the Arc against 27 on the CPU.
PHYSISML_DEVICE="${PHYSISML_DEVICE:-auto}"
export PHYSISML_DEVICE
CONDA_GPU_PYTHON="$HOME/miniforge3/envs/physisml_gpu/bin/python"
ONEAPI_VARS="/opt/intel/oneapi/2025.3/oneapi-vars.sh"
[ ! -f "$ONEAPI_VARS" ] && ONEAPI_VARS="/opt/intel/oneapi/setvars.sh"

if [ "$PHYSISML_DEVICE" != "cpu" ] && [ -f "$CONDA_GPU_PYTHON" ] \
     && [ -f "$ONEAPI_VARS" ]; then
  source "$ONEAPI_VARS" > /dev/null 2>&1
  # MKL libs (libmkl_intel_lp64.so.2 etc.) come from pip into ~/.local/lib
  if [ -d "$HOME/.local/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
  fi
  # And the user site must be OFF, or ~/.local's torch shadows the env's.
  # A conda env puts ~/.local/lib/python3.X/site-packages AHEAD of its own
  # site-packages, so a plain 'pip install --user torch' anywhere on this
  # machine silently replaces the XPU build with whatever that one is — the
  # build then reports 'Device: cpu' and runs three times slower with no error.
  # Measured: 8,283 tok/s on the Arc against 911 on the CPU.
  export PYTHONNOUSERSITE=1
  PYTHON="$CONDA_GPU_PYTHON"
  GPU_STATUS=$("$PYTHON" -c "import torch; print('xpu' if torch.xpu.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
  echo "  Device: $GPU_STATUS  (conda physisml_gpu)"
else
  PYTHON="python3"
  GPU_STATUS="cpu"
  echo "  Device: cpu  (system python3)"
fi

# ── CPU power policy ────────────────────────────────────────────────────────
# Measured on this machine (i7-1360P, d=512/L=6 training step, batch 8):
#
#   EPP balance_performance (the desktop default)   870 tok/s   1.9 GHz  59°C
#   EPP performance                               1400 tok/s   4.0 GHz  85°C
#
# 1.6x, sustained, for one command — the CPU was sitting at 38% of its maximum
# clock while cool, because power-profiles-daemon keeps the 'balanced' profile.
# The build is ~94% dream, all of it this one training step, so the whole run
# scales with this number.
#
# NOT the 'performance' GOVERNOR, which measured SLOWER (746 tok/s): it pins a
# high P-state on the 8 E-cores too and the package power budget then splits
# across 16 threads instead of feeding the 4 P-cores. The knob is EPP.
#
# Restored on every exit path, Ctrl-C included. POWER_PROFILE=off skips it —
# the laptop runs at 85°C for the duration, which is within spec (crit 100°C)
# but audible.
POWER_PROFILE="${POWER_PROFILE:-performance}"
_POWER_SAVED=""

power_boost_on() {
  [ "$POWER_PROFILE" = "off" ] && return 0
  if command -v powerprofilesctl > /dev/null 2>&1; then
    _POWER_SAVED="$(powerprofilesctl get 2>/dev/null)"
    if [ -n "$_POWER_SAVED" ] && powerprofilesctl set "$POWER_PROFILE" 2>/dev/null; then
      echo "  Power: profile $_POWER_SAVED -> $POWER_PROFILE  (~1.6x, restored on exit)"
      return 0
    fi
    _POWER_SAVED=""
  fi
  # No power-profiles-daemon: write EPP directly, if sudo does not prompt.
  local epp=/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
  if [ -w "$epp" ] || sudo -n true 2>/dev/null; then
    _POWER_SAVED="epp:$(cat $epp 2>/dev/null)"
    for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
      echo performance | sudo -n tee "$f" > /dev/null 2>&1
    done
    echo "  Power: EPP ${_POWER_SAVED#epp:} -> performance  (restored on exit)"
  else
    echo "  Power: left as is (no powerprofilesctl, no passwordless sudo)."
    echo "         The CPU may be capped at ~40% of its clock: check with"
    echo "         'powerprofilesctl get' — 'performance' is worth ~1.6x here."
  fi
}

power_boost_off() {
  case "$_POWER_SAVED" in
    "") return 0 ;;
    epp:*) for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
             echo "${_POWER_SAVED#epp:}" | sudo -n tee "$f" > /dev/null 2>&1
           done ;;
    *) powerprofilesctl set "$_POWER_SAVED" 2>/dev/null ;;
  esac
  echo "  Power: restored to $_POWER_SAVED"
  _POWER_SAVED=""
}

# ── Yield the Arc to the trainer ────────────────────────────────────────────
# llama.cpp (Qwen3-VL-4B, -ngl 99) parks ~3.4 of the Arc's 4 GB. With the
# trainer on the same card that is not a squeeze, it is a coin flip: L11
# trained through it, L12's phase 0 died at batch 16 with
# UR_RESULT_ERROR_OUT_OF_HOST_MEMORY in loss.backward(). So when training on
# the Arc, move llama off it for the length of the build and put it back on
# every exit path, same discipline as the power profile.
#
# The switchers are machine-local (scripts_internal/, gitignored): they encode
# THIS machine's systemd unit and devices. On a machine without them this
# block is silent, and the coin flip is back — the warning below says so.
#
#   LLAMA_YIELD=cpu   (default) -dev none: server on the CPU, slower grading
#   LLAMA_YIELD=igpu  Iris Xe via Vulkan0: Arc equally free, faster server
#   LLAMA_YIELD=off   leave llama where it is
LLAMA_YIELD="${LLAMA_YIELD:-cpu}"
_LLAMA_YIELDED=""
gpu_yield_on() {
  [ "$LLAMA_YIELD" = "off" ] && return 0
  [ "$GPU_STATUS" = "xpu" ] || return 0
  local sw="scripts_internal/llama_${LLAMA_YIELD}.sh"
  if [ -x "$sw" ]; then
    if "$sw"; then
      _LLAMA_YIELDED=1
      echo "  llama.cpp: moved ($LLAMA_YIELD) for the duration of the build."
    else
      # A failed switch is NOT "nothing changed": the switcher writes its
      # drop-in before restarting, so llama may be half-moved. Roll back
      # explicitly rather than assume.
      echo "  ⚠ $sw failed: trying to put llama back on the Arc…"
      scripts_internal/llama_gpu.sh || \
        echo "  ⚠ the rollback failed too: check 'systemctl --user status llama'."
      echo "    If llama stayed on the Arc, phase 0/2 can die of"
      echo "    UR_RESULT_ERROR_OUT_OF_HOST_MEMORY."
    fi
  elif curl -sf --max-time 2 http://localhost:8080/health >/dev/null 2>&1; then
    echo "  ⚠ llama-server is live on the Arc and $sw does not exist here:"
    echo "    contended memory, phase 0/2 can die of OUT_OF_HOST_MEMORY."
  fi
}
gpu_yield_off() {
  [ -n "$_LLAMA_YIELDED" ] || return 0
  if [ -x scripts_internal/llama_gpu.sh ] && scripts_internal/llama_gpu.sh; then
    _LLAMA_YIELDED=""
    echo "  llama.cpp: back on the Arc."
  else
    # Flag left set on purpose: a second EXIT pass (nested trap) retries, and
    # the message names the manual command instead of pretending success.
    echo "  ⚠ llama did NOT come back on the Arc: run by hand"
    echo "      scripts_internal/llama_gpu.sh"
  fi
}

# Cleanup runs on EXIT only; INT/TERM convert into an exit instead of running
# the cleanup and letting the script fall through. Reviewed and confirmed: the
# old single trap undid the GPU yield and the power boost on the first Ctrl-C
# and then KEPT BUILDING wherever the interrupted command's failure was
# non-fatal — un-boosted, un-yielded, and looking normal.
trap 'gpu_yield_off; power_boost_off' EXIT
trap 'echo ""; echo "  interrotto."; exit 130' INT TERM
power_boost_on
gpu_yield_on

EPOCHS_0=3    # good balance: enough signal, not wasteful for small corpora (L0-L2)
              # for large corpora (L3+, 20MB+) consider --epochs-0 2 to save time
TUTOR_MODEL="haiku"
# Turns per session. This must cover at least one full pass through the level's
# target pool, or the teacher — which advances through targets in order — never
# reaches the tail of the pool and those targets are never taught. One pass
# costs sum(targets x advance_after_successes): after the 2026-08-25 pool
# expansion that is 274 turns at L2 and 243 at L3, against the 100 this used to
# be. The check below warns when the budget falls short again.
TEACH_TURNS=300   # number of turns per level, or 'auto'
RESUME=0
# Per-session hard cap. Has to clear the largest pool: L11 is 363 targets, so
# one pass is 726 turns. Below that the cap silently truncates the level.
MAX_TEACH_TURNS=900
MAX_SESSIONS=4        # max sessions for fixed-session levels (L0-L1)
MAX_SESSIONS_DYNAMIC=12  # safety cap for dynamic-session levels
RETRAIN_EPOCHS=2      # text retrain epochs between sessions (short re-anchor)

# Minimum dream cycles per level, topped up after the level clears its gate.
# Sessions are driven by the quality gate, which measures the CURRENT level, so
# an easy level runs one session and gets one dream. But retention of EARLIER
# levels tracks the number of dreams, and the two are unrelated: measured on
# the finished L10 checkpoint, exact match across all levels goes 20% (1 dream)
# -> 43% (6) -> 48% (10), saturating there. 6 is the knee of that curve; the
# cost is ~30 min per extra dream and no API calls, since nothing is taught.
# Set to 0 to restore the old behaviour (one dream per session, nothing more).
MIN_DREAMS=${MIN_DREAMS:-6}
# The dynamic side of the same knob: after MIN_DREAMS, keep dreaming while the
# frozen probe still improves, stop when the marginal gain dies (see
# scripts/dream_until_plateau.py — with these defaults the rule reproduces ~6
# on the L10 curve MIN_DREAMS came from, so 6-as-floor is a belt over
# suspenders until a few levels' own curves say the floor can drop to 2-3).
# The fallbacks here must equal DEFAULTS in scripts/dream_until_plateau.py —
# tests/test_dream_plateau.py parses this file and fails when they drift.
MAX_DREAMS=${MAX_DREAMS:-12}
DREAM_EPSILON=${DREAM_EPSILON:-0.02}
DREAM_PATIENCE=${DREAM_PATIENCE:-2}
DREAM_MAX_DROP=${DREAM_MAX_DROP:-0.05}
# Anti-forgetting arm (exp_i): dream (default, the validated build) | ewc
# (N1 restricted to the current level, N3 off, online-EWC penalty in phases
# 1-2, fisher.pt written at each level boundary) | none (ewc's gating with no
# penalty — the floor arm). The EWC_* fallbacks must equal DEFAULTS in
# dynamic_model/exp_b/ewc.py — tests/test_ewc.py parses this file and fails
# when they drift (same contract as the DREAM_* knobs above).
ANTI_FORGETTING=${ANTI_FORGETTING:-dream}
EWC_LAMBDA=${EWC_LAMBDA:-1000}
EWC_GAMMA=${EWC_GAMMA:-0.95}
BETWEEN_SESSIONS="dream-only"  # what happens between sessions:
                               #   dream-only  : only dream, no retrain (recommended by experiment)
                               #   standard    : dream + retrain (classic, but retrain hurts)
                               #   none        : nothing (fast but no consolidation)

# Dynamic session control (L2+):
#   plateau_window    : stop if improvement < min_delta for N consecutive sessions
#   min_delta         : minimum improvement per session to not count as plateau (absolute rate)
#   regression_thresh : revert to best checkpoint if rate drops by this much vs best
DYNAMIC_SESSIONS_MIN_LEVEL=0   # use dynamic sessions from this level up (all levels)
MIN_SESSIONS_DYNAMIC=4         # always run at least this many sessions before plateau/regression check
PLATEAU_WINDOW=3               # consecutive sessions without improvement before stopping
MIN_DELTA=0.03                 # 3% improvement required per session to not count as plateau
REGRESSION_THRESH=0.15         # drop vs best that triggers regression stop.
                               # NOTE: the 40-turn quality window has a standard
                               # error of ~0.08 — thresholds below that fire on
                               # pure noise (the old 0.07 did).
# Advance immediately only at very high quality — otherwise wait for plateau.
# This gives a "convergence signature" per level: we see the real peak rate.
# If a level plateaus at 5%, that's a diagnostic signal to intervene.
QUALITY_IMMEDIATE=0.70         # advance without waiting for plateau only above this
                               # rate (strict metric: 0.85 was unreachable dead code)

# QUALITY_THRESHOLD is a real gate: when a level plateaus (or hits the session
# cap) BELOW threshold, the build STOPS instead of advancing — training the
# next levels on unmastered foundations produced the L3+ cascade of the May
# build (L8 was "promoted" with a 80% gate and 0% real correctness).
# Set FORCE_ADVANCE=1 to restore the old advance-anyway behaviour.
FORCE_ADVANCE=${FORCE_ADVANCE:-0}

# Parse remaining args
for arg in "${@:2}"; do
  case "$arg" in
    --resume)            RESUME=1 ;;
    auto)                TEACH_TURNS="auto" ;;
    opus)                TUTOR_MODEL="opus" ;;
    sonnet)              TUTOR_MODEL="sonnet" ;;
    haiku)               TUTOR_MODEL="haiku" ;;
    --max-teach-turns=*) MAX_TEACH_TURNS="${arg#*=}" ;;
    [0-9]*)              TEACH_TURNS="$arg" ;;
  esac
done

case "$TUTOR_MODEL" in
  opus)   TUTOR="claude-opus-4-6"   ;;
  sonnet) TUTOR="claude-sonnet-4-6" ;;
  *)      TUTOR="claude-haiku-4-5"  ;;
esac

# Load .env
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
fi

# The Claude tutor is OPTIONAL: a level that ships local_teacher.json is taught
# by the local or hybrid teacher, no API involved. Do not fail here — only when
# a level actually has no local teacher to fall back on (checked in the loop).
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Note: ANTHROPIC_API_KEY not set — teaching with the local/hybrid tutor."
  echo "      Levels without training_files/$LANG/N/local_teacher.json will stop."
fi

# Local-LLM endpoints for the hybrid tutor (override for a remote GPU box).
# Either server will do: llama.cpp's llama-server speaks the OpenAI dialect on
# 8080, ollama its own on 11434. dynamic_model/llm_backend.py picks whichever
# answers, llama.cpp first.
LLAMA_SERVER_BASE="${LLAMA_SERVER_BASE:-http://localhost:8080}"
OLLAMA_BASE="${OLLAMA_BASE:-http://localhost:11434}"

# Is a grader reachable at all? Probed once, not per level.
local_llm_up() {
  curl -sf --max-time 2 "$LLAMA_SERVER_BASE/v1/models" > /dev/null 2>&1 && return 0
  curl -sf --max-time 2 "$OLLAMA_BASE/api/tags"        > /dev/null 2>&1 && return 0
  return 1
}

# ── Level completion, as the quality gate defines it ────────────────────────
# A level is complete when its LAST session cleared the threshold — not when a
# file exists. phase 1 always writes final_learned.pt, so the old --resume test
# ('final_learned.pt present and no GATE_FAILED') called a level complete after
# a phase-2 crash: the dream aborts with exit 1, the gate never runs, no
# GATE_FAILED is written, and the next --resume silently skips a level that
# scored 0.12. Same two numbers as the in-loop gate, in one place.

level_last_rate() {   # $1 = level → the strict rate of the last session, or empty
  $PYTHON -c "
import glob, json
logs = sorted(glob.glob('models/checkpoints/$LANG/level_$1/session_*.jsonl'))
if not logs: raise SystemExit(1)
with open(logs[-1]) as f:
    records = [json.loads(l) for l in f if l.strip()]
if not records: raise SystemExit(1)
score = sum(1.0 if r.get('feedback') in ('+++','++')
            else (0.5 if r.get('feedback') == '+' else 0.0) for r in records)
print(f'{score / len(records):.2f}')
" 2>/dev/null
}

level_threshold() {   # $1 = level → the level's own quality_threshold
  $PYTHON -c "
import json, os
p = f'training_files/$LANG/$1/local_teacher.json'
if os.path.exists(p):
    print(json.load(open(p)).get('quality_threshold', 0.40))
else:
    print({0:0.45, 1:0.40, 2:0.40, 3:0.35, 4:0.32, 5:0.30}.get($1, 0.30))
" 2>/dev/null || echo 0.35
}

level_complete() {    # $1 = level → 0 if complete, 1 otherwise
  local L="$1"
  [ -f "models/checkpoints/$LANG/level_${L}/final_learned.pt" ] || return 1
  [ -f "models/checkpoints/$LANG/level_${L}/GATE_FAILED" ] && return 1
  # The dream is part of finishing a level: build.sh aborts if it fails, so a
  # level without final_dreamed.pt never got past phase 2.
  [ -f "models/checkpoints/$LANG/level_${L}/final_dreamed.pt" ] || return 1
  local R T
  R=$(level_last_rate "$L"); T=$(level_threshold "$L")
  [ -z "$R" ] && return 1
  $PYTHON -c "import sys; sys.exit(0 if float('$R') >= float('$T') else 1)"
}

# ── Determine START_LEVEL ────────────────────────────────────────────────────

if [ "$RESUME" -eq 1 ]; then
  # --resume: auto-detect next incomplete level.
  # A level counts as complete only if final_learned.pt exists AND the quality
  # gate did not fail there (GATE_FAILED flag written on exit 2): phase 1
  # always saves final_learned.pt, so without the flag check a failed level
  # would be silently skipped on resume, bypassing the gate.
  START_LEVEL=0
  for L in $(seq 0 $TARGET_LEVEL); do
    if level_complete "$L"; then
      START_LEVEL=$((L + 1))
    else
      break
    fi
  done
  if [ "$START_LEVEL" -le "$TARGET_LEVEL" ]; then
    R=$(level_last_rate "$START_LEVEL"); T=$(level_threshold "$START_LEVEL")
    if [ -n "$R" ]; then
      echo "--resume: level $START_LEVEL is at quality ${R} (threshold ${T}) — continuing it"
    fi
  fi
  echo "--resume: levels 0→$((START_LEVEL - 1)) already complete, resuming from level $START_LEVEL"

else
  # Interactive: show available checkpoints, ask from which level to restart
  echo ""
  echo "  Available checkpoints:"
  LAST_COMPLETE=-1
  for L in $(seq 0 $MAX_LEVEL); do
    BASE="models/checkpoints/$LANG/level_${L}"
    if [ -f "$BASE/final_learned.pt" ]; then
      RQ=$(level_last_rate "$L"); TQ=$(level_threshold "$L")
      if level_complete "$L"; then
        echo "    level $L  ✓ complete        (quality ${RQ:-?} ≥ ${TQ})"
        LAST_COMPLETE=$L
      else
        echo "    level $L  … to complete     (quality ${RQ:-?} < ${TQ}$([ -f "$BASE/final_dreamed.pt" ] || echo ", dream missing"))"
      fi
    elif [ -f "$BASE/final.pt" ]; then
      echo "    level $L    final.pt (phase 0 complete)"
    fi
  done
  if [ "$LAST_COMPLETE" -eq -1 ]; then
    echo "    (no checkpoint — pristine model)"
  fi
  echo ""
  echo "  Which level do you want to restart from?"
  echo "    0        = start over from scratch (full reset)"
  if [ "$LAST_COMPLETE" -ge 0 ]; then
    printf "    1..%d     = keep the previous levels, restart from that level\n" "$((LAST_COMPLETE + 1))"
    printf "    Enter    = continue from the next incomplete level (level %d)\n" "$((LAST_COMPLETE + 1))"
  fi
  echo ""
  read -p "  Start level [Enter = continue]: " FROM_INPUT

  if [ -z "$FROM_INPUT" ]; then
    # Enter = continue from the first level that has not cleared its gate
    START_LEVEL=0
    for L in $(seq 0 $TARGET_LEVEL); do
      if level_complete "$L"; then START_LEVEL=$((L + 1)); else break; fi
    done
    echo "  → Continuing from level $START_LEVEL"
  elif [ "$FROM_INPUT" = "0" ]; then
    # Full reset
    echo "y" | ./reset.sh --lang "$LANG"
    echo ""
    START_LEVEL=0
  else
    START_LEVEL="$FROM_INPUT"
    # Delete checkpoints from START_LEVEL onward so they are rebuilt
    echo ""
    echo "  Deleting checkpoints from level $START_LEVEL onward..."
    for L in $(seq $START_LEVEL $MAX_LEVEL); do
      DIR="models/checkpoints/$LANG/level_${L}"
      if [ -d "$DIR" ]; then
        rm -rf "$DIR"
        echo "    removed: $DIR"
      fi
    done
    echo ""
  fi
fi

# Check there is something to do
if [ "$START_LEVEL" -gt "$TARGET_LEVEL" ]; then
  echo "All levels up to $TARGET_LEVEL are already complete."
  echo "To continue to a higher level: ./build.sh $((TARGET_LEVEL + 1)) --lang $LANG --resume"
  exit 0
fi

START_TIME=$(date +%s)

# Build loop
for LEVEL in $(seq $START_LEVEL $TARGET_LEVEL); do

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  LEVEL $LEVEL  [$LANG]"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Retrying this level: clear any previous quality-gate failure flag
  rm -f "models/checkpoints/$LANG/level_${LEVEL}/GATE_FAILED"

  # Phase 0: text training (skip if final.pt already exists in resume mode)
  FINAL_PT="models/checkpoints/$LANG/level_${LEVEL}/final.pt"
  if [ "$RESUME" -eq 1 ] && [ -f "$FINAL_PT" ]; then
    echo ""
    echo "  [phase 0] $FINAL_PT already present — skip"
  else
    echo ""
    echo "  [phase 0] Text training on training_files/$LANG/$LEVEL/ ..."
    $PYTHON -u dynamic_model/train_curriculum.py \
      --phase    0 \
      --level    "$LEVEL" \
      --epochs-0 "$EPOCHS_0" \
      --lang     "$LANG"

    if [ $? -ne 0 ]; then
      echo "Error in phase 0, level $LEVEL. Aborting."
      exit 1
    fi
  fi

  # Phase 1 + 2: Teaching loop — model stays on this level until quality is met.
  #
  # Two modes depending on DYNAMIC_SESSIONS_MIN_LEVEL:
  #
  # Fixed (L < DYNAMIC_SESSIONS_MIN_LEVEL):
  #   Run up to MAX_SESSIONS. Advance when quality threshold reached or cap hit.
  #
  # Dynamic (L >= DYNAMIC_SESSIONS_MIN_LEVEL):
  #   Keep running sessions as long as the model is improving.
  #   Stop conditions (in priority order):
  #     1. Quality threshold reached → advance ✓
  #     2. Regression: rate drops > REGRESSION_THRESH vs best → revert + stop
  #     3. Plateau: PLATEAU_WINDOW sessions with improvement < MIN_DELTA → advance
  #     4. Safety cap MAX_SESSIONS_DYNAMIC → advance anyway

  SESSION=0
  SESSION_RATES=()      # positive rates per session (for plateau/regression detection)
  BEST_SESSION_RATE=0   # best rate seen so far at this level
  BEST_SESSION=0        # session number with best rate
  DREAMS_DONE=0         # dreams run at this level (topped up to MIN_DREAMS below)
  while true; do
    SESSION=$((SESSION + 1))
    echo ""

    # Per-session hard cap on teaching turns
    if [ "$TEACH_TURNS" = "auto" ]; then
      EFFECTIVE_TURNS=$MAX_TEACH_TURNS
    else
      EFFECTIVE_TURNS=$TEACH_TURNS
    fi

    # Teacher selection per level:
    #   L0-L1: local (pure regex, instant — sufficient for phonemes/words)
    #   L2+:   hybrid (LocalTeacher prompts + local-LLM evaluation), or local
    #          again when no LLM server is reachable.
    # Both need local_teacher.json; the Claude tutor is used only for levels
    # that do not have one (currently: the English curriculum).
    HYBRID_MIN_LEVEL=2   # from this level up use hybrid instead of local
    EFFECTIVE_TUTOR="$TUTOR"
    if [ -f "training_files/$LANG/$LEVEL/local_teacher.json" ]; then
      if [ "$LEVEL" -lt "$HYBRID_MIN_LEVEL" ]; then
        EFFECTIVE_TUTOR="local"
      elif local_llm_up; then
        EFFECTIVE_TUTOR="hybrid"
      else
        EFFECTIVE_TUTOR="local"
      fi
    elif [ -z "$ANTHROPIC_API_KEY" ]; then
      # No local teacher for this level and no API key: the Claude tutor is the
      # only option left and it is unavailable.
      echo "Error: level $LEVEL has no training_files/$LANG/$LEVEL/local_teacher.json"
      echo "       and ANTHROPIC_API_KEY is not set, so no tutor can run."
      echo "       Add the key to .env, or generate a local teacher config for it."
      exit 1
    fi

    # Does the budget cover one pass through the pool? The teacher walks the
    # targets in order, so a short budget silently leaves the tail untaught.
    POOL_TURNS=$($PYTHON -c "
import json,sys
try:
    c=json.load(open('training_files/$LANG/$LEVEL/local_teacher.json'))
    print(sum(len(s['targets'])*s.get('advance_after_successes',2) for s in c['steps'].values()))
except Exception: print(0)" 2>/dev/null || echo 0)
    # Act on it, do not just say it. The script computes exactly how many turns
    # one pass needs, and a warning followed by running the short budget anyway
    # leaves the tail of the pool untaught while claiming the level was taught:
    # L11 is 363 targets and L12 is 285, both past the 300 this defaulted to.
    # Phase 1 is minutes (2 for 300 turns on the Arc, 17 on the CPU) against
    # hours of dreaming, so the budget is the cheapest thing in the build to be
    # generous with.
    if [ "${POOL_TURNS:-0}" -gt 0 ] && [ "$EFFECTIVE_TURNS" != "auto" ] \
       && [ "$EFFECTIVE_TURNS" -lt "$POOL_TURNS" ] 2>/dev/null; then
      if [ "$POOL_TURNS" -le "$MAX_TEACH_TURNS" ]; then
        echo "  [phase 1] turns ${EFFECTIVE_TURNS} → ${POOL_TURNS}: one full pass "
        echo "            over L$LEVEL's pool (the teacher advances in order)."
        EFFECTIVE_TURNS=$POOL_TURNS
      else
        echo "  ⚠ one pass over L$LEVEL's pool costs ${POOL_TURNS} turns, past"
        echo "    the MAX_TEACH_TURNS=${MAX_TEACH_TURNS} cap. The tail of the pool stays"
        echo "    untaught: raise --max-teach-turns."
        EFFECTIVE_TURNS=$MAX_TEACH_TURNS
      fi
    fi

    SESSION_START=$(date +%s)
    echo "  [phase 1] Session $SESSION — ${EFFECTIVE_TUTOR} (max ${EFFECTIVE_TURNS} turns) ..."

    # Ask gate: scaffolding for the curiosity level only. The reward asymmetry
    # in LocalTeacher (+1.0 for asking about something unknown, -0.3 for asking
    # about something already explained) can only fire once the model actually
    # asks, and a model that has never produced a question will not start on
    # its own from teacher-forced golds alone. The gate raises the logit of the
    # question opener at the first token when local ignorance is high, which is
    # what makes those turns happen at all. It is deliberately NOT passed to
    # phase 2 — the dream would then splice question forms into replay — and
    # NOT to any other level. The honest measurement is scripts/curiosity_rate.py
    # with the gate OFF: what the gate produces is turns, what the asymmetry
    # produces is a policy.
    ASK_GATE_FLAG=""
    if [ "$LEVEL" -ge 12 ]; then
      ASK_GATE_FLAG="--ask-gate"
    fi

    $PYTHON -u dynamic_model/train_curriculum.py \
      --phase        1 \
      --level        "$LEVEL" \
      --interactions "$EFFECTIVE_TURNS" \
      --tutor-model  "$EFFECTIVE_TUTOR" \
      --lang         "$LANG" \
      --anti-forgetting "$ANTI_FORGETTING" \
      --ewc-lambda   "$EWC_LAMBDA" \
      --ewc-gamma    "$EWC_GAMMA" \
      $ASK_GATE_FLAG

    if [ $? -ne 0 ]; then
      echo "Error in phase 1, level $LEVEL. Aborting."
      exit 1
    fi

    # Dream mode: standard on first session, light on retries
    if [ "$SESSION" -eq 1 ]; then
      DREAM_MODE="standard"
    else
      DREAM_MODE="light"
    fi
    echo ""
    echo "  [phase 2] Dream ($DREAM_MODE)..."
    $PYTHON -u dynamic_model/train_curriculum.py \
      --phase      2 \
      --level      "$LEVEL" \
      --lang       "$LANG" \
      --dream-mode "$DREAM_MODE" \
      --anti-forgetting "$ANTI_FORGETTING" \
      --ewc-lambda "$EWC_LAMBDA" \
      --ewc-gamma  "$EWC_GAMMA"

    if [ $? -ne 0 ]; then
      echo "Error in phase 2, level $LEVEL. Aborting."
      exit 1
    fi
    DREAMS_DONE=$((DREAMS_DONE + 1))

    # ── Quality check ────────────────────────────────────────────────
    LAST_POS=$(python3 -c "
import glob, json
logs = sorted(glob.glob('models/checkpoints/$LANG/level_${LEVEL}/session_*.jsonl'))
if not logs: print(0); exit()
records = []
with open(logs[-1]) as f:
    for line in f:
        line = line.strip()
        if line: records.append(json.loads(line))
# FULL-session window: a 40-turn tail had a standard error of ~0.08, so
# session-to-session comparisons (plateau/regression) fired on pure noise.
# Strong grades count 1.0; '+' is a soft positive (single content word)
# and counts 0.5 — it was inflating the gate.
window = records
score = sum(1.0 if r.get('feedback') in ('+++','++')
            else (0.5 if r.get('feedback') == '+' else 0.0)
            for r in window)
rate = score / max(len(window), 1)
print(f'{rate:.2f}')
" 2>/dev/null || echo "0")

    # Threshold: read from local_teacher.json if present, else use level-based default.
    # Calibrated for the STRICT gate metric (full session, strong grades = 1.0,
    # '+' = 0.5): the old 0.50-0.70 values were aspirational — never reached
    # even under the old any-positive/last-20 metric. Raise them as the
    # system's real convergence levels become known.
    QUALITY_THRESHOLD=$(python3 -c "
import json, os
cfg_path = f'training_files/$LANG/$LEVEL/local_teacher.json'
if os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    print(cfg.get('quality_threshold', 0.40))
else:
    # Claude levels: threshold decreases as complexity increases
    defaults = {0:0.45, 1:0.40, 2:0.40, 3:0.35, 4:0.32, 5:0.30}
    print(defaults.get(int('$LEVEL'), 0.30))
" 2>/dev/null || echo "0.35")

    QUALITY_OK=$(python3 -c "print(1 if float('${LAST_POS}') >= float('${QUALITY_THRESHOLD}') else 0)" 2>/dev/null || echo "0")

    echo ""
    SESSION_ELAPSED=$(( $(date +%s) - SESSION_START ))
    SESSION_RATES+=("$LAST_POS")
    echo "  [level $LEVEL] Session $SESSION: quality=$(python3 -c "print(f'{float(\"${LAST_POS}\"):.0%}')" 2>/dev/null)  duration=${SESSION_ELAPSED}s"

    # ── Fixed-session mode (L < DYNAMIC_SESSIONS_MIN_LEVEL) ──────────────────
    if [ "$LEVEL" -lt "$DYNAMIC_SESSIONS_MIN_LEVEL" ]; then
      # Fixed levels: advance immediately when quality threshold is reached
      if [ "$QUALITY_OK" = "1" ]; then
        echo "  ✓ Quality reached — advancing to level $((LEVEL + 1))."
        break
      fi
      if [ "$SESSION" -ge "$MAX_SESSIONS" ]; then
        if [ "$QUALITY_OK" != "1" ] && [ "$FORCE_ADVANCE" != "1" ]; then
          echo "  ✗ MAX_SESSIONS ($MAX_SESSIONS) reached BELOW THRESHOLD — build ABORTED at level $LEVEL."
          echo "    (FORCE_ADVANCE=1 to force advancing)"
          touch "models/checkpoints/$LANG/level_${LEVEL}/GATE_FAILED"
          exit 2
        fi
        echo "  ⚠ MAX_SESSIONS ($MAX_SESSIONS) reached — advancing anyway."
        break
      fi
      # Not converged, not at limit → between-session intervention (see below)

    # ── Dynamic-session mode (L >= DYNAMIC_SESSIONS_MIN_LEVEL) ───────────────
    else
      # Dynamic levels: advance immediately ONLY at very high quality (immediate convergence).
      # Otherwise wait for natural plateau — this records the real convergence signature.
      QUALITY_IMMEDIATE_OK=$(python3 -c "print(1 if float('${LAST_POS}') >= float('${QUALITY_IMMEDIATE}') else 0)" 2>/dev/null || echo "0")
      if [ "$QUALITY_IMMEDIATE_OK" = "1" ]; then
        echo "  ✓ High quality (${LAST_POS} ≥ ${QUALITY_IMMEDIATE}) — advancing immediately."
        break
      fi

      # Save best checkpoint when rate improves
      CURR_RATE_INT=$(python3 -c "print(int(float('${LAST_POS}') * 1000))" 2>/dev/null || echo "0")
      BEST_RATE_INT=$(python3 -c "print(int(float('${BEST_SESSION_RATE}') * 1000))" 2>/dev/null || echo "0")
      if [ "$CURR_RATE_INT" -gt "$BEST_RATE_INT" ]; then
        BEST_SESSION_RATE="$LAST_POS"
        BEST_SESSION="$SESSION"
        cp "models/checkpoints/$LANG/level_${LEVEL}/final_dreamed.pt" \
           "models/checkpoints/$LANG/level_${LEVEL}/best_session.pt" 2>/dev/null || true
        echo "  ↑ New best: ${LAST_POS} (session $SESSION) — best_session.pt saved."
      fi

      # Stop condition 2: regression (only after MIN_SESSIONS_DYNAMIC sessions)
      REGRESSED="0"
      if [ "${#SESSION_RATES[@]}" -ge "$MIN_SESSIONS_DYNAMIC" ]; then
        REGRESSED=$(python3 -c "
best = float('${BEST_SESSION_RATE}')
curr = float('${LAST_POS}')
thresh = float('${REGRESSION_THRESH}')
print(1 if best > 0.05 and curr < best - thresh else 0)
" 2>/dev/null || echo "0")
      fi
      if [ "$REGRESSED" = "1" ]; then
        echo "  ✗ Regression detected: ${LAST_POS} vs best ${BEST_SESSION_RATE} (threshold ${REGRESSION_THRESH})."
        BEST_PT="models/checkpoints/$LANG/level_${LEVEL}/best_session.pt"
        if [ -f "$BEST_PT" ]; then
          cp "$BEST_PT" "models/checkpoints/$LANG/level_${LEVEL}/final_dreamed.pt"
          echo "    Restoring best_session.pt (session $BEST_SESSION)."
        fi
        BELOW_THRESH=$(python3 -c "print(1 if float('${BEST_SESSION_RATE}') < float('${QUALITY_THRESHOLD}') else 0)" 2>/dev/null || echo "1")
        if [ "$BELOW_THRESH" = "1" ] && [ "$FORCE_ADVANCE" != "1" ]; then
          echo "  ✗ Best ${BEST_SESSION_RATE} below threshold ${QUALITY_THRESHOLD} — build ABORTED at level $LEVEL."
          echo "    (FORCE_ADVANCE=1 to force advancing)"
          touch "models/checkpoints/$LANG/level_${LEVEL}/GATE_FAILED"
          exit 2
        fi
        echo "    Advancing to the next level with the best checkpoint."
        break
      fi

      # Stop condition 3: plateau (only after MIN_SESSIONS_DYNAMIC sessions)
      N_SESSIONS="${#SESSION_RATES[@]}"
      if [ "$N_SESSIONS" -ge "$MIN_SESSIONS_DYNAMIC" ] && [ "$N_SESSIONS" -ge "$PLATEAU_WINDOW" ]; then
        PLATEAUED=$(python3 -c "
rates = [float(r) for r in '${SESSION_RATES[*]}'.split()]
window = ${PLATEAU_WINDOW}
min_delta = ${MIN_DELTA}
# Peak-based plateau: consecutive-diff comparisons on noisy rates fired on
# fluctuations while the model was still improving (e.g. L0 exact-match
# rising 8%→19% across sessions). Plateau = the PEAK of the last <window>
# sessions no longer improves on the peak of everything before them.
if len(rates) < window + 1: print(0); exit()
recent_peak = max(rates[-window:])
prior_peak  = max(rates[:-window])
print(1 if recent_peak <= prior_peak + min_delta else 0)
" 2>/dev/null || echo "0")
        if [ "$PLATEAUED" = "1" ]; then
          PEAK_RATE=$(python3 -c "rates=[float(r) for r in '${SESSION_RATES[*]}'.split()]; print(f'{max(rates):.0%}')" 2>/dev/null || echo "?")
          BELOW_THRESH=$(python3 -c "print(1 if float('${BEST_SESSION_RATE}') < float('${QUALITY_THRESHOLD}') else 0)" 2>/dev/null || echo "1")
          echo "  ~ Plateau detected — level $LEVEL peak: ${PEAK_RATE}"
          echo "    Session rates: ${SESSION_RATES[*]}"
          if [ "$BELOW_THRESH" = "1" ] && [ "$FORCE_ADVANCE" != "1" ]; then
            echo "  ✗ Plateau BELOW THRESHOLD (best ${BEST_SESSION_RATE} < ${QUALITY_THRESHOLD}) — build ABORTED at level $LEVEL."
            echo "    The level was not learned: going on would train later levels on missing foundations."
            echo "    (FORCE_ADVANCE=1 to force advancing)"
            # Leave the BEST weights on disk, not the last (possibly worse) session
            BEST_PT="models/checkpoints/$LANG/level_${LEVEL}/best_session.pt"
            [ -f "$BEST_PT" ] && cp "$BEST_PT" "models/checkpoints/$LANG/level_${LEVEL}/final_dreamed.pt"
            touch "models/checkpoints/$LANG/level_${LEVEL}/GATE_FAILED"
            exit 2
          fi
          echo "    Advancing to level $((LEVEL + 1))."
          break
        fi
      fi

      # Stop condition 4: safety cap
      if [ "$SESSION" -ge "$MAX_SESSIONS_DYNAMIC" ]; then
        BELOW_THRESH=$(python3 -c "print(1 if float('${BEST_SESSION_RATE}') < float('${QUALITY_THRESHOLD}') else 0)" 2>/dev/null || echo "1")
        if [ "$BELOW_THRESH" = "1" ] && [ "$FORCE_ADVANCE" != "1" ]; then
          echo "  ✗ MAX_SESSIONS_DYNAMIC ($MAX_SESSIONS_DYNAMIC) reached BELOW THRESHOLD (best ${BEST_SESSION_RATE} < ${QUALITY_THRESHOLD}) — build ABORTED at level $LEVEL."
          echo "    (FORCE_ADVANCE=1 to force advancing)"
          BEST_PT="models/checkpoints/$LANG/level_${LEVEL}/best_session.pt"
          [ -f "$BEST_PT" ] && cp "$BEST_PT" "models/checkpoints/$LANG/level_${LEVEL}/final_dreamed.pt"
          touch "models/checkpoints/$LANG/level_${LEVEL}/GATE_FAILED"
          exit 2
        fi
        echo "  ⚠ MAX_SESSIONS_DYNAMIC ($MAX_SESSIONS_DYNAMIC) reached — advancing anyway."
        break
      fi
    fi

    # Not stopped → between-session intervention
    echo "  ↻ Not converged — between-session strategy: $BETWEEN_SESSIONS"

    case "$BETWEEN_SESSIONS" in
      dream-only)
        # Experiment result: dream-only is stable (10-15%), retrain is destructive
        echo "  (dream already run, no retrain — dream-only strategy)"
        ;;
      standard)
        echo "  Text retrain ($RETRAIN_EPOCHS epochs)..."
        $PYTHON -u dynamic_model/train_curriculum.py \
          --phase    0 \
          --level    "$LEVEL" \
          --epochs-0 "$RETRAIN_EPOCHS" \
          --lang     "$LANG"
        if [ $? -ne 0 ]; then
          echo "Error in retrain, level $LEVEL. Aborting."
          exit 1
        fi
        ;;
      none)
        echo "  (no operation between sessions)"
        ;;
    esac
  done

  # ── Dream top-up ──────────────────────────────────────────────────────────
  # Retention is a function of consolidation cycles, not of when the quality
  # gate happens to pass. Measured on the finished L10 checkpoint, extra dreams
  # move exact match across ALL levels from 20% to 43% at six and 48% at ten,
  # saturating there: +3.6 points per dream from 1 to 6, +1.0 from 7 to 12, and
  # that second slope is under the 2.2-point noise between identical runs.
  #
  # Without this, a level that clears the gate on its first session gets one
  # dream and keeps ~20% of the earlier levels. In the reference build the only
  # level that retained them was L4 — the one that struggled and therefore ran
  # ten sessions, hence ten dreams. That was luck, not design.
  #
  # No new teaching happens here: the dream only reconsolidates what the
  # session logs already hold, and it costs the current level nothing (L10 went
  # 94% -> 100% while the earlier levels recovered).
  DREAMED="models/checkpoints/$LANG/level_${LEVEL}/final_dreamed.pt"
  if [ "$MIN_DREAMS" -gt 0 ] && [ ! -f GATE_FAILED ] && [ -f "$DREAMED" ]; then
    # The fixed top-up loop this replaces dreamed the level up to MIN_DREAMS
    # and stopped, measuring nothing: 6 was the knee of ONE curve (L10), and
    # whether dream 5 of THIS level was still worth anything nobody knew.
    # dream_until_plateau.py scores the frozen probe after every dream and
    # stops when the marginal gain dies (or restores the best state if a
    # dream damaged retention). It accumulates via --checkpoint internally —
    # the same requirement the old loop carried — and leaves each level's
    # curve in level_N/dream_curve.json.
    echo ""
    echo "  ── Dreams until plateau (session: $DREAMS_DONE, floor $MIN_DREAMS, cap $MAX_DREAMS) ──"
    $PYTHON -u scripts/dream_until_plateau.py \
      --level        "$LEVEL" \
      --lang         "$LANG" \
      --already-done "$DREAMS_DONE" \
      --min          "$MIN_DREAMS" \
      --max          "$MAX_DREAMS" \
      --epsilon      "$DREAM_EPSILON" \
      --patience     "$DREAM_PATIENCE" \
      --max-drop     "$DREAM_MAX_DROP" \
      --anti-forgetting "$ANTI_FORGETTING" \
      --ewc-lambda   "$EWC_LAMBDA" \
      --ewc-gamma    "$EWC_GAMMA"
    _PLATEAU_RC=$?
    if [ "$_PLATEAU_RC" -eq 3 ]; then
      # A crashed dream is survivable BECAUSE the script measured what the
      # crash left on disk and restored the best state if it was damaged —
      # the curve records the salvage.
      echo "  ⚠ a dream failed: state verified by the script (curve in level_${LEVEL}/dream_curve.json)."
      echo ""
      echo "  ✓ Level $LEVEL consolidated (curve in models/checkpoints/$LANG/level_${LEVEL}/dream_curve.json)."
    elif [ "$_PLATEAU_RC" -ne 0 ]; then
      # Anything else means ZERO or unknown consolidation: a missing/edited
      # probe (rc 1), bad DREAM_* values (rc 2), an interrupt (rc 130). The
      # loop this replaced guaranteed MIN_DREAMS top-ups unconditionally —
      # printing a checkmark here would ship a level with one dream (~20%
      # retention, measured) under a log line that says consolidated.
      echo ""
      echo "  ✗ dream_until_plateau exited with code $_PLATEAU_RC: the level is"
      echo "    NOT consolidated. Fix it (probe? DREAM_* values?) and rerun:"
      echo "    the build stops here instead of building on one dream's foundation."
      exit 1
    else
      echo ""
      echo "  ✓ Level $LEVEL consolidated (curve in models/checkpoints/$LANG/level_${LEVEL}/dream_curve.json)."
    fi
  fi

  # ── EWC Fisher (exp_i, ewc arm only) ────────────────────────────────────
  # The end-of-level boundary: the Fisher is computed on the exact
  # final_dreamed.pt the next level's phase 0 will inherit, right after the
  # top-up settled on the best measured state. Gated on the arm so the
  # validated build and the dream/none arms never write sidecars. A failure
  # aborts: the next level would otherwise train as the 'none' arm while the
  # log says ewc.
  if [ "$ANTI_FORGETTING" = "ewc" ] && [ -f "$DREAMED" ]; then
    echo ""
    echo "  ── Fisher EWC (level $LEVEL) ──"
    $PYTHON -u scripts/compute_fisher.py \
      --level "$LEVEL" --lang "$LANG" --gamma "$EWC_GAMMA"
    if [ $? -ne 0 ]; then
      echo "  ✗ compute_fisher.py failed: without the sidecar the next level"
      echo "    would run as the 'none' arm under a log that says ewc."
      exit 1
    fi
  fi

  # Update active.pt — prefer dreamed checkpoint if available
  LEARNED="models/checkpoints/$LANG/level_${LEVEL}/final_learned.pt"
  if [ -f "$DREAMED" ]; then
    ./set_model.sh "$DREAMED" > /dev/null 2>&1
    echo ""
    echo "  ✓ models/active.pt → level_${LEVEL}/final_dreamed.pt"
  elif [ -f "$LEARNED" ]; then
    ./set_model.sh "$LEARNED" > /dev/null 2>&1
    echo ""
    echo "  ✓ models/active.pt → level_${LEVEL}/final_learned.pt"
  fi

done

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║                BUILD COMPLETE                ║"
echo "╠══════════════════════════════════════════════╣"
printf "║  Levels completed   : %d → %d\n" "$START_LEVEL" "$TARGET_LEVEL"
printf "║  Total time         : %dm %ds\n" "$MINUTES" "$SECS"
echo "╠══════════════════════════════════════════════╣"
echo "║  Checkpoints:"
for LEVEL in $(seq 0 $TARGET_LEVEL); do
  BASE="models/checkpoints/$LANG/level_${LEVEL}"
  [ -f "$BASE/final.pt" ]         && printf "║    level_%d/final.pt         ✓\n" "$LEVEL"
  [ -f "$BASE/final_learned.pt" ] && printf "║    level_%d/final_learned.pt ✓\n" "$LEVEL"
  [ -f "$BASE/final_dreamed.pt" ] && printf "║    level_%d/final_dreamed.pt ✓\n" "$LEVEL"
done
echo "╠══════════════════════════════════════════════╣"
echo "║  To test:"
echo "║    $PYTHON dynamic_model/run.py"
echo "╚══════════════════════════════════════════════╝"

# Run the quality test on the final level with the SAME interpreter that did
# the training. Both this script and run.py import torch, which lives in the
# conda environment: the system python3 has none, so the build used to end on
# a ModuleNotFoundError after hours of work and printed no quality figure at
# all — the one number that says whether the build is worth keeping.
echo ""
$PYTHON dynamic_model/test_model.py --level "$TARGET_LEVEL" --lang "$LANG"
