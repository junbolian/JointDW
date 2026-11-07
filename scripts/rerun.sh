#!/usr/bin/env bash
# The solver runs behind the released data (optional and slow: several days on one workstation with 24 cores and 48 GB).
# Nothing here is needed to reproduce the tables, which scripts/reproduce.sh builds from the released records.
# Needs the solver extras (pip install -e ".[solvers,learned]") and, for MIPLIB-NL, the MPS files exported from
# MIPLIB-NL into work/mps/ (README, "Building instances"); the benchmark instances need scripts/fetch_benchmarks.sh.
# Each part writes under work/ and is resumable (finished evaluations are reused).
#
# usage: scripts/rerun.sh census|five_duals|large_models|benchmark|reference_optima [PARALLEL]
#
# Settings of record: GCG root node with the given decomposition, no presolve, cuts or heuristics, SCIP propagation off,
# 120 s per evaluation, one thread, seed 0; failed evaluations retried twice, then once with seed 1; per-process
# address-space cap 5 GB (10 GB for the models in config/high_memory_models.txt) under a shared 40 GB budget.
set -euo pipefail
cd "$(dirname "$0")/.."
PART=${1:?census, five_duals, large_models, benchmark or reference_optima}
P=${2:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LLMDW_CHILD_MEMORY_GB=${LLMDW_CHILD_MEMORY_GB:-5} LLMDW_MEMORY_BUDGET_GB=${LLMDW_MEMORY_BUDGET_GB:-40}
export LLMDW_MIN_AVAILABLE_GB=${LLMDW_MIN_AVAILABLE_GB:-10}
case "$PART" in
census)
    # gain tables of the 108 MIPLIB-NL models with 2 to 6 families; graph40-40-1rand and graph40-80-1rand take their LP
    # bound from the HiGHS interior-point method with crossover
    grep -vx -e graph40-40-1rand -e graph40-80-1rand <(ls work/mps | sed -n 's/\.mps\.gz$//p') > work/census_names.txt
    python -m llmdw.gcg.batch oracle --mps-dir work/mps --out work/oracle --max-families 6 --time 120 --parallel "$P" \
        --jobs-per-instance 2 --seed 0 --names work/census_names.txt \
        --high-memory config/high_memory_models.txt --high-memory-gb 10
    printf 'graph40-40-1rand\ngraph40-80-1rand\n' > work/census_names_last.txt
    LLMDW_LP_SOLVER=highs-ipm python -m llmdw.gcg.batch oracle --mps-dir work/mps --out work/oracle --max-families 6 \
        --time 120 --parallel 2 --jobs-per-instance 2 --seed 0 --names work/census_names_last.txt \
        --high-memory config/high_memory_models.txt --high-memory-gb 10
    python -m llmdw.gcg.batch lookahead --mps-dir work/mps --oracle work/oracle --out work/lookahead --block-time 60 \
        --parallel "$P"
    python -m llmdw.scoring.structure --oracle-root work/oracle --lookahead-root work/lookahead --out work/structure
    ;;
five_duals)
    # the lookahead at four further optimal LP duals (seeds 1 to 4) and the best of five
    python -m llmdw.lookahead.five_duals batch --work work --out work/five_duals --parallel "$P" \
        --high-memory config/high_memory_models.txt --high-memory-gb 10
    ;;
large_models)
    # the 30 MIPLIB-NL models with 7 to 26 families: LP bound, lookahead, and the orders with ten evaluations each
    python -m llmdw.scoring.large_models --work . prepare --dataset "${MIPLIBNL_DATASET:?path of the MIPLIB-NL dataset/}" \
        --parallel 2
    LLMDW_CHILD_MEMORY_GB=10 python -m llmdw.scoring.large_models --work . run --parallel 2
    ;;
benchmark)
    # the 59 benchmark instances: build, selection, ground truth (sets of at most two families), lookahead at one and at
    # five duals, the sets of the methods, and the branch-and-price runs (GCG defaults, one thread, seed 0, 1800 s)
    python -m llmdw.benchmark.standard build --raw data/benchmark_raw --out work/benchmark_std
    python -m llmdw.benchmark.standard select --out work/benchmark_std --parallel "$P"
    S=work/benchmark_std
    NAMES=$(paste -sd, data/work/benchmark_std/full_set/all_instances.txt)
    python -m llmdw.benchmark.truth evaluate --root $S --names "$NAMES" --jobs "$P" --max-set 2 \
        --baselines gcg_spfwh,gcg_spfwh_names
    python -m llmdw.benchmark.truth classify --root $S --names "$NAMES"
    python -m llmdw.benchmark.truth lookahead --root $S --names "$NAMES" --parallel 2 --max-set 2
    # the answers of GPT-6.1 Sol and Claude Sonnet 5.5 (original presentation, three samples) are the released records
    # in data/work/benchmark_std/answers; README, "Running the language models", sends them again
    A=data/work/benchmark_std/answers
    python -m llmdw.benchmark.bnp plan --root $S --names "$NAMES" --answers "GPT-6.1 Sol=$A/gpt-6.1-sol" \
        "Claude Sonnet 5.5=$A/claude-sonnet-5-5"
    python -m llmdw.benchmark.bnp run --root $S --names "$NAMES" --parallel "$P"
    ;;
reference_optima)
    # the reference optima replaced under llmdw.scoring.bounds (data/work/reference_optima)
    mkdir -p work/reference_optima
    for n in b-ball radiationm18-12-05; do
        python -m llmdw.instances.reference_optimum --mps work/mps/$n.mps.gz --out work/reference_optima/$n.json --time 3600
    done
    ;;
*)
    echo "unknown part: $PART"; exit 2 ;;
esac
