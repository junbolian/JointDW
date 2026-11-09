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
#   results/paper/experiments the table bodies of these experiments (llmdw.tables.experiments) and of the experiments
#                             below
#   results/experiment_D/     the time-optimal decompositions on the benchmark instances (every run of every set,
#                             seeds 0 to 2, under the validity rule of llmdw.benchmark.validity)
#   results/experiment_E/     the rich models with 12 and 16 families
#   results/experiment_F/     the classes without a joint group (generalized assignment, bin packing)
#   results/eval_time/, results/eval_time_large/   the time of one evaluation per collection
#   results/eval_growth/      the evaluation time against the number of convexified families
#   results/experiment_G/     the cost accounting (requests, prices, net benefit)
#   results/paper/figure_data the figure data of these experiments
# Named outputs of the tables added in the revision:
#   Table 18  results/paper/experiments/rich_models_data_vs_five_dual.tex (results/rich_models_data_paired_five_dual.csv)
#   Table 23  results/paper/experiments/experiment_C_paired_hit1.tex (results/experiment_C/benchmark_paired_hit1.csv,
#             benchmark_paired_gain.csv)
#   Table 24  results/paper/experiments/experiment_C_data_methods.tex, experiment_C_data_paired.tex
#             (results/experiment_C/benchmark_data_methods.csv, benchmark_data_paired.csv, benchmark_data_predictions.csv)
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
for t in "Table 18:rich_models_data_vs_five_dual.tex" "Table 23:experiment_C_paired_hit1.tex" \
         "Table 24:experiment_C_data_methods.tex" "Table 24:experiment_C_data_paired.tex"; do
    f=$R/paper/experiments/${t#*:}; [ -s "$f" ] || { echo "missing: ${t%%:*} $f"; exit 1; }; echo "${t%%:*}: $f"
done
X=$R/paper/experiments; FIG=$R/paper/figure_data; NJ=$W/no_joint_group
mkdir -p "$FIG"
step "time-optimal decompositions on the benchmark instances"
python -m llmdw.benchmark.time_truth tables --root $B --names-file $B/full_set/all_instances.txt --out $R/experiment_D \
    --tex-dir $X --answers "GPT-6.1 Sol=$B/answers/gpt-6.1-sol" "Claude Sonnet 5.5=$B/answers/claude-sonnet-5-5" \
    --data-configs "GPT-6.1 Sol, data" "Claude Sonnet 5.5, data"
cp $R/experiment_D/figure_data/experiment_D_tradeoff.csv $FIG/
step "classes without a joint group"
python -m llmdw.benchmark.no_joint_group score --root $NJ --names-file $NJ/instances_used_selected.txt \
    --answers "GPT-6.1 Sol=$NJ/answers/gpt-6.1-sol" "Claude Sonnet 5.5=$NJ/answers/claude-sonnet-5-5" \
    --prompt-groups $NJ/prompt_groups.csv --out $R/experiment_F --tex-dir $X
step "rich models with 12 and 16 families"
python -m llmdw.scoring.rich_models_large --work $W --published $D --out $R/experiment_E --tex-dir $X --figure-dir $FIG
python -m llmdw.tables.rich_figure_data --cost-vs-k $FIG/experiment_E_cost_vs_k.csv --results $R \
    --large-predictions $R/experiment_E/predictions.csv --out $FIG
step "time of one evaluation and its growth with the number of convexified families"
python -m llmdw.benchmark.eval_time tables --paths $W/eval_time/paths.json --runs $W/eval_time/runs --out $R/eval_time
python -m llmdw.benchmark.eval_time tables --paths $W/eval_time_large/paths.json --runs $W/eval_time_large/runs \
    --out $R/eval_time_large
python -m llmdw.scoring.eval_growth --oracle-root $W/oracle --paths $W/eval_time/paths.json --runs $W/eval_time/runs \
    --d-runs $R/experiment_D/bnp_runs.csv --out $R/eval_growth --tex $X/eval_growth.tex --figure $FIG/eval_growth.csv
step "cost accounting"
mkdir -p $R/experiment_G && cp $W/cost_accounting/prices.csv $R/experiment_G/prices.csv
python -m llmdw.scoring.cost_accounting requests --data $D --work $W --prices $W/cost_accounting/prices.csv \
    --out $R/experiment_G
python -m llmdw.scoring.cost_accounting tables --data $D --work $W --requests $R/experiment_G/requests.csv \
    --prices $W/cost_accounting/prices.csv --eval-time $R/eval_time/eval_time.csv \
    --eval-time-large-paths $W/eval_time_large/paths.json --eval-time-large-runs $W/eval_time_large/runs \
    --bnp-runs $R/experiment_D/bnp_runs.csv --bench-root $B --bench-answers $B/answers \
    --bench-names $B/full_set/all_instances.txt --out $R/experiment_G --tex-dir $X
for f in experiment_D_methods experiment_D_paired experiment_D_tradeoff experiment_E_instances experiment_E_predictions \
         experiment_E_search experiment_E_paired experiment_F_methods experiment_F_paired experiment_G_net_benefit \
         eval_growth; do
    [ -s "$X/$f.tex" ] || { echo "missing: $X/$f.tex"; exit 1; }
done
for f in experiment_D_tradeoff experiment_E_cost_vs_k search_cost_vs_misses eval_growth rich_cost_by_class_k rich_hit1; do
    [ -s "$FIG/$f.csv" ] || { echo "missing: $FIG/$f.csv"; exit 1; }
done
step "done"
