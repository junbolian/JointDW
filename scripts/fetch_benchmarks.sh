#!/usr/bin/env bash
# Download the public benchmark files of the benchmark instances (Trigeiro, Thomas and McClain at suerie.de with the
# test list of the GCG repository; the Canad R instances of CommaLAB; the capacitated warehouse location instances of
# the OR-Library; four LOTSIZELIB AMPL copies used only as a check) from their original locations into
# data/benchmark_raw/, and verify each data file against the SHA-256 recorded when the instances were built
# (config/benchmark_sources.json). The files are not redistributed with this repository; DATA.md names their sources.
# Then `python -m llmdw.benchmark.standard build --raw data/benchmark_raw --out work/benchmark_std` rebuilds the model
# files, whose SHA-256 is recorded in data/work/benchmark_std/instances/*/meta.json.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m llmdw.benchmark.sources --out data/benchmark_raw --recorded config/benchmark_sources.json
echo "data/benchmark_raw/ is ready"
