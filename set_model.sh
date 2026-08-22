#!/bin/bash
# Set the active model by copying it to models/active.pt.
# If models/active.pt already exists, moves it to models/backups/
# before overwriting it.
#
# Usage:
#   ./set_model.sh <model.pt>
#   ./set_model.sh                    ← list available models

if [ -z "$1" ]; then
  echo "Usage: $0 <model.pt>"
  echo ""
  echo "Available models:"
  find models/checkpoints models/backups -name "*.pt" 2>/dev/null \
    | sort | while read f; do
        size=$(du -sh "$f" 2>/dev/null | cut -f1)
        echo "  $size  $f"
    done
  exit 1
fi

if [ ! -f "$1" ]; then
  echo "Error: file not found: $1"
  exit 1
fi

mkdir -p models/backups

# Backup current active model (if it exists)
if [ -f models/active.pt ]; then
  TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
  BACKUP="models/backups/active_${TIMESTAMP}.pt"
  mv models/active.pt "$BACKUP"
  echo "Previous model backup → $BACKUP"
fi

cp "$1" models/active.pt
echo "Active model → models/active.pt  (from: $1)"

# Copy matching tokenizer if present in the same directory
SOURCE_DIR=$(dirname "$1")
if [ -f "$SOURCE_DIR/tokenizer.json" ]; then
  cp "$SOURCE_DIR/tokenizer.json" models/active_tokenizer.json
  # A level directory holds ONE tokenizer.json, saved at the end of the dream,
  # but TWO checkpoints: final_learned.pt (pre-dream) and final_dreamed.pt
  # (post-dream). Activating final_learned.pt therefore pairs it with a
  # tokenizer that knows tokens the model still has dormant (logit -inf,
  # zero embedding), which silently corrupts any prompt using them.
  VOCAB_CHECK=$(python3 - "$1" models/active_tokenizer.json <<'PYEOF'
import json, sys, torch
ckpt, tok = sys.argv[1], sys.argv[2]
try:
    active = torch.load(ckpt, map_location="cpu", weights_only=False)["config"].get("active_vocab_size")
    n_tok  = len(json.load(open(tok)).get("vocab", {}))
except Exception:
    sys.exit(0)
print(f"{active} {n_tok}")
PYEOF
)
  MODEL_ACTIVE=$(echo "$VOCAB_CHECK" | cut -d' ' -f1)
  TOK_N=$(echo "$VOCAB_CHECK" | cut -d' ' -f2)
  echo "Tokenizer → models/active_tokenizer.json  (vocab=${TOK_N:-?})"
  if [ -n "$MODEL_ACTIVE" ] && [ -n "$TOK_N" ] && [ "$MODEL_ACTIVE" != "$TOK_N" ]; then
    echo ""
    echo "  WARNING: tokenizer/model vocabulary mismatch"
    echo "    model active_vocab_size = $MODEL_ACTIVE"
    echo "    tokenizer tokens        = $TOK_N"
    if [ "$TOK_N" -gt "$MODEL_ACTIVE" ] 2>/dev/null; then
      echo "    The tokenizer knows tokens the model has dormant. If you"
      echo "    activated final_learned.pt, use final_dreamed.pt instead:"
      echo "    it is the checkpoint this tokenizer was saved with."
    fi
    echo ""
  fi
else
  rm -f models/active_tokenizer.json
fi

# Always update standalone/ with real file copies (never symlinks)
# so the standalone directory survives a build reset
if [ -d "standalone" ]; then
  cp models/active.pt standalone/model.pt
  if [ -f models/active_tokenizer.json ]; then
    cp models/active_tokenizer.json standalone/tokenizer.json
  fi
  echo "Standalone updated → standalone/model.pt + tokenizer.json  (real files)"
fi

echo "Start with: python3 dynamic_model/run.py"
echo "        or: cd standalone && python3 chat.py"
