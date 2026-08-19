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
  echo "Tokenizer → models/active_tokenizer.json  (vocab=$(python3 -c "import json; d=json.load(open('$SOURCE_DIR/tokenizer.json')); print(len(d.get('vocab',{})))" 2>/dev/null || echo '?'))"
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
