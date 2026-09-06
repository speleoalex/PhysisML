#!/bin/bash
# Reset the model to a clean slate.
#
# Moves the current active model and the checkpoints to a timestamped backup
# folder, then clears models/active.pt and the checkpoint tree.
# training_files/ is never touched.
#
# Usage:
#   ./reset.sh                  # reset the Italian curriculum (default)
#   ./reset.sh --lang en        # reset ONLY models/checkpoints/en/
#   ./reset.sh --lang all       # reset every language (the old behaviour)
#   ./reset.sh --dry-run        # show what would be moved, don't do it
#
# The language matters: checkpoints live in models/checkpoints/<lang>/, so a
# reset that ignored it would carry away the other language's whole build
# while you were only clearing this one.

DRY_RUN=0
LANG="${PHYSISML_LANG:-it}"
_lang_next=0
for arg in "$@"; do
    if [ "$_lang_next" -eq 1 ]; then LANG="$arg"; _lang_next=0; continue; fi
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --lang)    _lang_next=1 ;;
        --lang=*)  LANG="${arg#*=}" ;;
        *)         echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if [ "$LANG" = "all" ]; then
    CKPT_DIR="models/checkpoints"
    SCOPE="every language"
else
    CKPT_DIR="models/checkpoints/$LANG"
    SCOPE="language '$LANG'"
fi

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
BACKUP_DIR="models/backups/${TIMESTAMP}"

echo "=== Model reset — ${SCOPE} ==="
echo "Backup in: ${BACKUP_DIR}"
echo ""

# Count what exists
N_CKPT=$(find "$CKPT_DIR" -name "*.pt" 2>/dev/null | wc -l)
HAS_ACTIVE=$([ -f models/active.pt ] && echo "yes" || echo "no")

echo "  models/active.pt:     ${HAS_ACTIVE}"
echo "  ${CKPT_DIR}/:  ${N_CKPT} .pt files"
echo ""

if [ "$N_CKPT" -eq 0 ] && [ "$HAS_ACTIVE" = "no" ]; then
    echo "Nothing to move — model already clean."
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY RUN] No changes made."
    echo "  Would create: ${BACKUP_DIR}/"
    echo "  Would move: active.pt + ${CKPT_DIR}/"
    exit 0
fi

# Confirm
read -p "Confirm reset? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

# Move to backup
mkdir -p "${BACKUP_DIR}"

if [ -f models/active.pt ]; then
    mv models/active.pt "${BACKUP_DIR}/active.pt"
    echo "  Moved: models/active.pt → ${BACKUP_DIR}/"
fi

# Path relative to models/, so the backup mirrors the tree it came from:
# checkpoints/ for a full reset, checkpoints/<lang>/ for one language.
REL="${CKPT_DIR#models/}"

if [ -d "$CKPT_DIR" ] && [ "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
    mkdir -p "${BACKUP_DIR}/$(dirname "$REL")"
    mv "$CKPT_DIR" "${BACKUP_DIR}/${REL}"
    mkdir -p "$CKPT_DIR"   # recreate empty
    echo "  Moved: ${CKPT_DIR}/ → ${BACKUP_DIR}/${REL}/"
fi

echo ""
echo "Reset complete (${SCOPE}). Backup in: ${BACKUP_DIR}"
echo ""
echo "To start from scratch:"
echo "  python3 dynamic_model/train_curriculum.py --phase 0 --lang ${LANG}"
echo ""
echo "To restore the backup:"
echo "  cp ${BACKUP_DIR}/active.pt models/active.pt"
echo "  cp -r ${BACKUP_DIR}/${REL} models/$(dirname "$REL")/"
