#!/bin/bash
# Is a seeded dream bit-reproducible?
#
# WHY IT MATTERS
# Two supposedly identical runs of dream 1 (same seed, same input checkpoint)
# gave 26.0% and 23.8% retention. That 2.2-point spread is the noise floor for
# every arm-to-arm comparison: an effect smaller than it cannot be attributed
# to the flag under test from a single run per arm.
#
# ALREADY RULED OUT
#   - GPU nondeterminism: build.sh reports "Device: cpu" — the conda
#     physisml_gpu python imports torch from ~/.local/lib, a CUDA build whose
#     torch.xpu.is_available() is False. Nothing has been running on the Arc.
#   - CPU training: a seeded 20-step toy run hashes identically twice.
#   - The measurement: measure_repetition.py on one checkpoint gives
#     byte-identical results twice over all 11 levels.
#   - The input data: the levels' qa_pairs.jsonl / qa_corpus.txt were not
#     touched by the dreams (mtimes predate them).
#
# So this compares the dream itself: two runs, separate checkpoint trees so
# they cannot interfere, the shared training_files state restored in between,
# and the resulting weights hashed.
#
# Usage:
#   ./scripts/check_dream_determinism.sh --confirm
#   LEVEL=1 ./scripts/check_dream_determinism.sh --confirm   # cheaper level

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LNG="${LNG:-it}"
LEVEL="${LEVEL:-10}"
SEED="${SEED:-1}"
DREAM_MODE="${DREAM_MODE:-standard}"
SRC="models/checkpoints/$LNG"
EXP="models/exp_g"

if [ "$1" != "--confirm" ]; then
    echo "  Two identical dreams at L$LEVEL (seed $SEED, mode $DREAM_MODE),"
    echo "  in separate trees, weights hashed and compared."
    echo "  Re-run with --confirm."
    exit 0
fi

CONDA_GPU_PYTHON="$HOME/miniforge3/envs/physisml_gpu/bin/python"
if [ -f "$CONDA_GPU_PYTHON" ]; then PY="$CONDA_GPU_PYTHON"; else PY="python3"; fi
echo "  Python: $PY"

rm -rf "$EXP"; mkdir -p "$EXP"

# Snapshot the shared state phase 2 may rewrite, so run B sees run A's input
# and not run A's output.
SNAP="$EXP/training_files_snapshot"
mkdir -p "$SNAP"
cp -a "training_files/$LNG" "$SNAP/"

for RUN in a b; do
    BASE="$EXP/run_$RUN"
    mkdir -p "$BASE"
    for d in "$SRC"/level_*; do
        lvl=$(basename "$d")
        mkdir -p "$BASE/$lvl"
        cp "$d"/session_*.jsonl "$BASE/$lvl/" 2>/dev/null || true
        [ -f "$d/tokenizer.json" ] && cp "$d/tokenizer.json" "$BASE/$lvl/"
    done
    cp "$SRC/level_$LEVEL/final_dreamed.pt" "$BASE/level_$LEVEL/start.pt"

    # Restore the shared input before every run.
    rm -rf "training_files/$LNG"
    cp -a "$SNAP/$LNG" "training_files/"

    echo ""
    echo "  --- run $RUN ---"
    $PY -u -m dynamic_model.train_curriculum \
        --phase 2 --level "$LEVEL" --lang "$LNG" \
        --dream-mode "$DREAM_MODE" --seed "$SEED" \
        --checkpoint "$BASE/level_$LEVEL/start.pt" \
        --ckpt-base "$BASE" > "$EXP/run_$RUN.log" 2>&1
    echo "  done → $EXP/run_$RUN.log"
done

# Restore the shared input one last time, so the repo is left as it was.
rm -rf "training_files/$LNG"
cp -a "$SNAP/$LNG" "training_files/"

echo ""
python3 - "$EXP" "$LEVEL" <<'PYEOF'
import sys, hashlib, torch, os
exp, level = sys.argv[1], sys.argv[2]
def h(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    m = hashlib.sha256()
    for k in sorted(sd):
        m.update(sd[k].contiguous().float().numpy().tobytes())
    return m.hexdigest()[:24]
pa = f"{exp}/run_a/level_{level}/final_dreamed.pt"
pb = f"{exp}/run_b/level_{level}/final_dreamed.pt"
if not (os.path.exists(pa) and os.path.exists(pb)):
    print("  one of the runs produced no checkpoint — see the logs"); sys.exit(1)
ha, hb = h(pa), h(pb)
print(f"  run a: {ha}")
print(f"  run b: {hb}")
print(f"  => the dream is {'DETERMINISTIC' if ha == hb else 'NOT deterministic'}")
if ha != hb:
    # Where do they diverge, and by how much?
    sa = torch.load(pa, map_location="cpu", weights_only=False)["state_dict"]
    sb = torch.load(pb, map_location="cpu", weights_only=False)["state_dict"]
    diffs = []
    for k in sorted(sa):
        d = (sa[k].float() - sb[k].float()).abs()
        if d.numel() and d.max().item() > 0:
            diffs.append((d.max().item(), d.mean().item(), k))
    diffs.sort(reverse=True)
    print(f"  tensors differing: {len(diffs)}/{len(sa)}")
    for mx, mn, k in diffs[:6]:
        print(f"    {k:<44} max {mx:.3e}  mean {mn:.3e}")
PYEOF
