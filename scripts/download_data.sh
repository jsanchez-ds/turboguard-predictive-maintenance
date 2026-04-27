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

# PHM Datasets S3 mirror — stable, official-source archive.
# Canonical source (when it works):
#   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
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

echo "→ Extracting outer archive ..."
if command -v unzip >/dev/null 2>&1; then
  unzip -o -q "$ARCHIVE" -d "$DEST"
else
  python -c "import zipfile; zipfile.ZipFile('$ARCHIVE').extractall('$DEST')"
fi
rm -f "$ARCHIVE"

# The official archive is a zip-of-zips: outer wraps a folder containing CMAPSSData.zip.
# Find the inner zip wherever it lands and flatten the layout so the loader expects.
INNER=$(find "$DEST" -maxdepth 3 -name "CMAPSSData.zip" | head -n 1 || true)
if [[ -n "$INNER" ]]; then
  echo "→ Extracting inner CMAPSSData.zip ..."
  if command -v unzip >/dev/null 2>&1; then
    unzip -o -q "$INNER" -d "$DEST"
  else
    python -c "import zipfile; zipfile.ZipFile('$INNER').extractall('$DEST')"
  fi
  rm -f "$INNER"
  # Remove any now-empty nested wrapper folder.
  find "$DEST" -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null || true
  find "$DEST" -mindepth 1 -maxdepth 1 -type d -exec rmdir --ignore-fail-on-non-empty {} + 2>/dev/null || true
fi

echo "✅ Done. Files in $DEST:"
ls -lh "$DEST" | head -20
