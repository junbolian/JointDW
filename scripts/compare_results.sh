#!/usr/bin/env bash
# Compare results/ (scripts/reproduce.sh) with the released results data/reference_results/, file by file.
# Prints the files that differ or are missing on either side and exits 1 if there is one.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d results ] || { echo "results/ is missing: run scripts/reproduce.sh first"; exit 1; }
[ -d data/reference_results ] || { echo "data/reference_results/ is missing: run scripts/fetch_data.sh first"; exit 1; }
if diff -rq data/reference_results results; then
    echo "results/ is identical to data/reference_results/ ($(find results -type f | wc -l) files)"
else
    exit 1
fi
