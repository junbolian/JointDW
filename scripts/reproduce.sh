#!/usr/bin/env bash
# Every table and figure file of the paper from the released data in data/ (scripts/fetch_data.sh), without any key.
# Writes results/ (the result tables) and results/paper/ (the table bodies and the figure data).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m llmdw.scoring.results --work data --census-corrections data/census_corrections.json \
    --large-models --rich-models --out results
python -m llmdw.tables.figure2 --out results/figure2.csv
python -m llmdw.tables.paper --results results --work data --out results/paper
