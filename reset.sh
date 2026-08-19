#!/bin/bash
# Reset the model to a clean slate.
#
# Moves the current active model and all checkpoints to a timestamped
# backup folder, then clears models/active.pt and models/checkpoints/.
# training_files/ is never touched.
#
# Usage:
#   ./reset.sh              # backup and reset
#   ./reset.sh --dry-run    # show what would be moved, don't do it

DRY_RUN=0
[ "$1" = "--dry-run" ] && DRY_RUN=1

TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
BACKUP_DIR="models/backups/${TIMESTAMP}"

echo "=== Model reset ==="
echo "Backup in: ${BACKUP_DIR}"
echo ""

# Count what exists
N_CKPT=$(find models/checkpoints -name "*.pt" 2>/dev/null | wc -l)
HAS_ACTIVE=$([ -f models/active.pt ] && echo "yes" || echo "no")

echo "  models/active.pt:     ${HAS_ACTIVE}"
echo "  models/checkpoints/:  ${N_CKPT} .pt files"
echo ""

if [ "$N_CKPT" -eq 0 ] && [ "$HAS_ACTIVE" = "no" ]; then
    echo "Nothing to move — model already clean."
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY RUN] No changes made."
    echo "  Would create: ${BACKUP_DIR}/"
    echo "  Would move: active.pt + checkpoints/"
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

if [ -d models/checkpoints ] && [ "$(ls -A models/checkpoints 2>/dev/null)" ]; then
    mv models/checkpoints "${BACKUP_DIR}/checkpoints"
    mkdir -p models/checkpoints   # recreate empty
    echo "  Moved: models/checkpoints/ → ${BACKUP_DIR}/"
fi

echo ""
echo "Reset complete. Backup in: ${BACKUP_DIR}"
echo ""
echo "To start from scratch:"
echo "  python3 dynamic_model/train_curriculum.py --phase 0"
echo ""
echo "To restore the backup:"
echo "  cp ${BACKUP_DIR}/active.pt models/active.pt"
