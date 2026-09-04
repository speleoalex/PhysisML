#!/bin/bash
# PhysisML backup cleanup — keeps significant milestones
#
# MILESTONES TO KEEP (2026-09-01):
#   2026-08-27_174537  — the most recent full snapshot of levels 0-9. Not a
#                        copy of the live tree: L0 and L5 differ by md5 and it
#                        has no level_10, so it is a superseded build kept as
#                        the one fallback if the live foundation is lost.
#
# TO DELETE:
#   - All active_*.pt files (snapshots of models/active.pt without context;
#     set_model.sh writes one on every switch and never rotates them)
#   - The 2026-08-19 / 08-20 series: eleven snapshots of one afternoon, all
#     superseded by the build that produced the published checkpoint
#   - 2026-08-26_100401: superseded by 08-27
#
# NOT here, and must not be: models/checkpoints/it/_pre_rebuild_20260901 holds
# the published L11/L12 (88.1%) while the rebuild regenerates them. It lives
# under checkpoints/ on purpose — build.sh only deletes level_N directories.
#
# A KEEP entry that does not exist on disk is an ERROR, not a no-op: this list
# was still naming four April directories that had long since been removed, so
# running the script would have deleted everything while reporting that it had
# preserved the milestones.

BACKUP_DIR="models/backups"
DRY_RUN=0
[ "$1" = "--dry-run" ] && DRY_RUN=1

KEEP=(
  "2026-08-27_174537"
)

# Refuse to run on a stale list. Without this the script is most dangerous
# exactly when it is most out of date.
MISSING=0
for k in "${KEEP[@]}"; do
  if [ ! -d "$BACKUP_DIR/$k" ]; then
    echo "ERROR: milestone '$k' does not exist in $BACKUP_DIR."
    echo "       The KEEP list is stale: update it before deleting,"
    echo "       or the script deletes everything believing it preserves something."
    MISSING=1
  fi
done
[ "$MISSING" -eq 1 ] && exit 1

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
