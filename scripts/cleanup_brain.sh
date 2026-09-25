#!/usr/bin/env bash
# ==============================================================================
# Limmireu Pipeline Global Rule 4: VM Shutdown & Brain Cleanup Script
# Compresses old session directories in ~/.gemini/antigravity-cli/brain into a zip archive
# and removes original directories.
# ==============================================================================

set -euo pipefail

BRAIN_DIR="/home/limmireu1214/.gemini/antigravity-cli/brain"
CURRENT_CONV="ca1d3285-c549-4c28-b1de-640a795d3c23"
ARCHIVE_NAME="archive_$(date +%Y%m%d_%H%M%S).zip"

if [[ ! -d "$BRAIN_DIR" ]]; then
    echo "[Brain Cleanup] Brain directory does not exist: $BRAIN_DIR"
    exit 0
fi

cd "$BRAIN_DIR"

OLD_DIRS=()
for d in */; do
    d_clean="${d%/}"
    if [[ "$d_clean" != "$CURRENT_CONV" && -d "$d_clean" ]]; then
        OLD_DIRS+=("$d_clean")
    fi
done

if [[ ${#OLD_DIRS[@]} -eq 0 ]]; then
    echo "[Brain Cleanup] No old brain directories found to clean up."
    exit 0
fi

echo "[Brain Cleanup] Found ${#OLD_DIRS[@]} old directories to archive: ${OLD_DIRS[*]}"
zip -q -r "$ARCHIVE_NAME" "${OLD_DIRS[@]}"

for d in "${OLD_DIRS[@]}"; do
    rm -rf "$d"
done

echo "[Brain Cleanup] Successfully archived old directories into $ARCHIVE_NAME and removed original directories."
ls -lh "$ARCHIVE_NAME"
