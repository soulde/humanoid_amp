#!/usr/bin/env bash
# Sync skrl train/play scripts from Isaac Lab releases and reapply local tweaks.
set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/scripts/reinforcement_learning/skrl"
FILES=(train.py play.py)
DEST_DIR="scripts/skrl"

fetch() {
  local url="$1"
  local dest="$2"
  curl -fsSL "${url}" -o "${dest}"
}

for name in "${FILES[@]}"; do
  url="${BASE_URL}/${name}"
  dest="${DEST_DIR}/${name}"
  echo "Fetching ${url}"
  fetch "${url}" "${dest}"
  sed -i "/# PLACEHOLDER: Extension template/a import humanoid_amp.tasks  # noqa: F401" "${dest}"
done

echo "Done."
