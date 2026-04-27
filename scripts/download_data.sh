#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# download_data.sh — fetch NASA C-MAPSS turbofan engine degradation dataset
#
# The C-MAPSS dataset is hosted on the NASA Prognostics Data Repository.
# It contains 4 sub-datasets (FD001–FD004) plus an RUL ground-truth file.
#
# Files:
#   train_FDxxx.txt — run-to-failure trajectories (training)
#   test_FDxxx.txt  — truncated trajectories (testing)
#   RUL_FDxxx.txt   — ground-truth RUL for the last cycle of each test trajectory
# ------------------------------------------------------------------------------
set -euo pipefail

DEST="data/raw/cmapss"
mkdir -p "$DEST"

# Mirror that has been historically stable for the C-MAPSS dataset.
# If it goes down, the canonical source is:
#   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
URL="https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
ARCHIVE="$DEST/cmapss.zip"

if [[ -f "$DEST/train_FD001.txt" ]]; then
  echo "✅ Dataset already present at $DEST — skipping download."
  exit 0
fi

echo "→ Downloading NASA C-MAPSS (~70 MB) ..."
if command -v curl >/dev/null 2>&1; then
  curl -L -o "$ARCHIVE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE" "$URL"
else
  echo "❌ Need curl or wget to download." >&2
  exit 1
fi

echo "→ Extracting ..."
if command -v unzip >/dev/null 2>&1; then
  unzip -o -q "$ARCHIVE" -d "$DEST"
else
  python -c "import zipfile; zipfile.ZipFile('$ARCHIVE').extractall('$DEST')"
fi

rm -f "$ARCHIVE"
echo "✅ Done. Files in $DEST:"
ls -lh "$DEST" | head -20
