#!/bin/bash
# PhysisML backup cleanup — keeps significant milestones
#
# MILESTONES TO KEEP:
#   2026-04-11_225505  — Baseline: first complete run L0-L5, 501 fixed tokens
#   2026-04-12_064448  — Dormant slots + LR=2e-5 (first stability fix)
#   2026-04-13_164633  — LocalTeacher + QA pairs (first optimised run)
#   2026-04-13_204325  — Best: L0-L2 with 8K tokens (state of the art)
#
# TO DELETE:
#   - All active_*.pt files (snapshots without context)
#   - Apr 11 dev directories (iterative runs before optimisations)
#   - Intermediate directories without distinctive characteristics

BACKUP_DIR="models/backups"
DRY_RUN=0
[ "$1" = "--dry-run" ] && DRY_RUN=1

KEEP=(
  "2026-04-11_225505"
  "2026-04-12_064448"
  "2026-04-13_164633"
  "2026-04-13_204325"
)

echo "=== PhysisML backup cleanup ==="
echo ""
echo "KEEP (milestone):"
for k in "${KEEP[@]}"; do
  size=$(du -sh "$BACKUP_DIR/$k" 2>/dev/null | cut -f1)
  echo "  ✓ $k  ($size)"
done
echo ""

# Compute the space to free
TOTAL_FREE=0

echo "TO DELETE:"

# 1. active_*.pt files
count_active=$(ls "$BACKUP_DIR"/active_*.pt 2>/dev/null | wc -l)
size_active=$(du -sh "$BACKUP_DIR"/active_*.pt 2>/dev/null | tail -1 | cut -f1 || echo "0")
echo "  - $count_active active_*.pt files  (~$size_active)"

# 2. Directories not in KEEP
for dir in "$BACKUP_DIR"/2026-*/; do
  dirname=$(basename "$dir")
  keep=0
  for k in "${KEEP[@]}"; do
    [ "$dirname" = "$k" ] && keep=1 && break
  done
  if [ "$keep" -eq 0 ]; then
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  - $dirname  ($size)"
  fi
done

echo ""

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[DRY RUN] No changes made."
  echo "Re-run without --dry-run to proceed."
  exit 0
fi

read -p "Confirm deletion? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
  echo "Cancelled."
  exit 0
fi

echo ""
echo "Deleting..."

# Remove active_*.pt
rm -f "$BACKUP_DIR"/active_*.pt
echo "  ✓ active_*.pt files deleted"

# Remove directories not in KEEP
for dir in "$BACKUP_DIR"/2026-*/; do
  dirname=$(basename "$dir")
  keep=0
  for k in "${KEEP[@]}"; do
    [ "$dirname" = "$k" ] && keep=1 && break
  done
  if [ "$keep" -eq 0 ]; then
    rm -rf "$dir"
    echo "  ✓ $dirname deleted"
  fi
done

echo ""
echo "Done. Backup directory size now:"
du -sh "$BACKUP_DIR"
echo ""
echo "Backups kept:"
for k in "${KEEP[@]}"; do
  size=$(du -sh "$BACKUP_DIR/$k" 2>/dev/null | cut -f1)
  echo "  ✓ $k  ($size)"
done
