#!/usr/bin/env bash
# Every table and figure file of the paper from the released data in data/ (scripts/fetch_data.sh), without a solver
# and without a language-model call. Writes results/ in the layout of data/reference_results/:
#   results/                  the result tables (census, templates, MIPLIB-NL, rich models, large models)
#   results/paper/            the table bodies and the figure data of the paper (llmdw.tables.paper)
#   results/experiment_A/     the control presentations with every coefficient
#   results/experiment_B/     the five-dual lookahead
#   results/experiment_C/     the benchmark instances (original presentation and data variant)
#   results/learned/          the learned predictor (from the released features, data/work/features)
#   results/combined/         the combined orders (union-first, stacked)
#   results/paper/experiments the table bodies of these experiments (llmdw.tables.experiments)
# scripts/compare_results.sh compares results/ with data/reference_results/.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data; W=data/work; R=results
step() { echo "== $1 ($(date '+%H:%M:%S'))"; }
step "result tables"
python -m llmdw.scoring.results --work $D --census-corrections $D/census_corrections.json \
    --large-models --rich-models --out $R
step "control presentations"
python -m llmdw.scoring.experiments control --work $D --out $R/experiment_A \
    --data "GPT-6.1 Sol=$W/rich_models_with_data/gpt-6.1-sol" "Claude Sonnet 5.5=$W/rich_models_with_data/claude-sonnet-5-5" \
    --controls "GPT-6.1 Sol=$W/controls/answers/gpt-6.1-sol" "Claude Sonnet 5.5=$W/controls/answers/claude-sonnet-5-5"
step "five-dual lookahead"
ANSWERS=()
for c in "claude-sonnet-5-5:Claude Sonnet 5.5" "gpt-6.1-sol:GPT-6.1 Sol" "grok-4-20-reasoning:Grok 4.20 (Reasoning)" \
         "grok-4-20-non-reasoning:Grok 4.20 (Non-Reasoning)" "deepseek-v4-pro:DeepSeek-V4-Pro (Non-Thinking)"; do
    dir=${c%%:*}; name=${c#*:}
    ANSWERS+=("templates:$name=$W/support_templates/$dir" "miplib:$name=$W/support_miplib/$dir"
              "rich_models:$name=$W/support_rich_models/$dir")
done
python -m llmdw.scoring.experiments five_duals --work $D --five-duals $W/five_duals --out $R/experiment_B \
    --answers "${ANSWERS[@]}"
step "rich models with every coefficient against the five-dual lookahead"
python -m llmdw.scoring.experiments data_five_duals --work $D --five-duals $W/five_duals --out $R \
    --data "GPT-6.1 Sol=$W/rich_models_with_data/gpt-6.1-sol" "Claude Sonnet 5.5=$W/rich_models_with_data/claude-sonnet-5-5" \
           "Grok 4.20 (Reasoning)=$W/rich_models_with_data/grok-4-20-reasoning" \
           "Grok 4.20 (Non-Reasoning)=$W/rich_models_with_data/grok-4-20-non-reasoning" \
           "DeepSeek-V4-Pro (Non-Thinking)=$W/rich_models_with_data/deepseek-v4-pro" \
    --controls "GPT-6.1 Sol=$W/controls/answers/gpt-6.1-sol" "Claude Sonnet 5.5=$W/controls/answers/claude-sonnet-5-5"
step "benchmark instances"
B=$W/benchmark_std
python -m llmdw.benchmark.tables --root $B --names-file $B/full_set/all_instances.txt \
    --answers "GPT-6.1 Sol=$B/answers/gpt-6.1-sol" "Claude Sonnet 5.5=$B/answers/claude-sonnet-5-5" \
    --first-batch "$(paste -sd, $B/first_batch/instances.txt)" --out $R/experiment_C
python -m llmdw.benchmark.data_tables --root $B --names-file $B/full_set/all_instances.txt \
    --answers "GPT-6.1 Sol=$B/answers/gpt-6.1-sol" "Claude Sonnet 5.5=$B/answers/claude-sonnet-5-5" \
    --data-answers "GPT-6.1 Sol=$B/answers_data/gpt-6.1-sol" "Claude Sonnet 5.5=$B/answers_data/claude-sonnet-5-5" \
    --out $R/experiment_C
JOINTDW=("templates:GPT-6.1 Sol, original=$W/support_templates/gpt-6.1-sol"
         "miplib:GPT-6.1 Sol, original=$W/support_miplib/gpt-6.1-sol"
         "rich_models:GPT-6.1 Sol, text=$W/support_rich_models/gpt-6.1-sol"
         "rich_models:GPT-6.1 Sol, data=$W/rich_models_with_data/gpt-6.1-sol")
step "learned predictor"
python -m llmdw.scoring.learned --work $D --five-duals $W/five_duals --features $W/features --out $R/learned \
    --sets templates,rich_models,miplib --threads 2 --jointdw "${JOINTDW[@]}"
step "combined orders (union-first, stacked)"
python -m llmdw.scoring.combined --work $D --five-duals $W/five_duals --features $W/features --out $R/combined \
    --sets templates,rich_models,miplib --threads 2 --jointdw "${JOINTDW[@]}"
step "table bodies and figure data of the paper"
python -m llmdw.tables.figure2 --out $R/figure2.csv
python -m llmdw.tables.paper --results $R --work $D --out $R/paper
python -m llmdw.tables.experiments --results $R --out $R/paper/experiments
step "done"
