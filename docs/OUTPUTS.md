# Outputs of `scripts/reproduce.sh`

Every output below is written by `scripts/reproduce.sh`, which reads only `data/`. Paths are relative to the repository root.

## Figures

| Output | What it contains | Result file |
|---|---|---|
| Figure 1 | gains of all family subsets of the lot-sizing template with two items and two periods | `results/paper/figure_1.csv` |
| Figure 2 | (a) the queries of the staged search for *k* = 6, *s* = 3, *w* = 2; (b) queries of the staged search and the lower bounds for *w* = 3, *s* = 6, *k* = 8 to 24 | `results/paper/figure_2a.csv`, `figure_2b.csv`; `results/figure2.csv` |
| Figure 3 | (a) MIPLIB-NL census; (b, c) recall under the four presentations on the templates and on MIPLIB-NL, with the lookahead recall | `results/paper/figure_3a_certain.csv`, `figure_3a_possible.csv`, `figure_3a_none.csv`, `figure_3b_templates.csv`, `figure_3c_miplib.csv` |
| Figure 4 | rich models with 7 to 16 families: hit@1 given the text and given every coefficient; evaluations at *s* = 2 by class and number of families for the covering order, the two lookahead orders, JointDW and its interleaving with the five-dual lookahead | `results/paper/figure_data/rich_hit1.csv`, `rich_cost_by_class_k.csv` |
| Figure 5 | rich models at *s* = 2: evaluations of the staged order alone and of JointDW by the number of missed members, with the stage counts and the covering order | `results/paper/figure_data/search_cost_vs_misses_after_cut.csv` |

## Tables of the main text

| Output | What it contains | Result file |
|---|---|---|
| Table 1 | instance collections: number of instances, number of families, source of the gains | `results/paper/experiments/tab_instances.tex` |
| Table 2 | hit@1 and recall in the original presentation on the templates and MIPLIB-NL, false alarms on the assignment templates | `results/paper/table_01.tex` |
| Table 3 | mean evaluations until a best decomposition is evaluated, by order and *s* | `results/paper/table_09.tex` |
| Table 4 | benchmark instances: share of instances on which the first set of each method is the best pair, by class; share on which it is the time-optimal set and its time ratio; the same two quantities on the instances without a joint group | `results/paper/experiments/tab_bench.tex`; `results/experiment_C/benchmark_methods.csv`, `results/experiment_D/methods_time_budget.csv`, `results/experiment_F/methods_time.csv` |

## Tables of the appendix

| Output | What it contains | Result file |
|---|---|---|
| Table 5 | exact gains of the templates | `results/templates_table.csv`, `results/covering_numbers.csv` |
| Table 6 | rich models with 7 to 9 families: families, size, LP gap, minimal best sets, zero-marginal groups | `results/rich_models_instances.csv`, `results/rich_models_groups.csv` |
| Table 7 | rich models with 12 to 16 families: families, size, LP gap, zero-marginal groups, instances with a positive gain at *s* = 2 | `results/experiment_E/instances.csv`, `results/paper/experiments/experiment_E_instances.tex` |
| Table 8 | benchmark instances: counts, sizes, LP gaps, best pairs | `results/experiment_C/benchmark_instances.csv`, `results/paper/experiments/experiment_C_instances.tex` |
| Table 9 | recall by presentation, mean ± standard error | `results/paper/table_06.tex` |
| Table 10 | paired differences in recall between presentations | `results/paper/table_07.tex` |
| Table 11 | Grok 4.20 (Reasoning) minus Grok 4.20 (Non-Reasoning), paired over instances | `results/paper/table_08.tex` |
| Table 12 | five-dual lookahead: first set and search cost, and JointDW interleaved with it | `results/paper/experiments/experiment_B_first_set.tex`, `experiment_B_search.tex`; `results/experiment_B/` |
| Table 13 | rich models with 7 to 9 families: hit@1 and recall given the text and given every coefficient | `results/paper/table_11.tex` |
| Table 14 | rich models with 7 to 9 families: evaluations until a best set is evaluated | `results/paper/table_12.tex` |
| Table 15 | control presentations with every coefficient: hit@1, recall and search cost with the names permuted or replaced by labels | `results/paper/experiments/experiment_A_prediction.tex`, `experiment_A_search.tex`; `results/experiment_A/` |
| Table 16 | rich models with every coefficient: JointDW and its interleaving with the five-dual lookahead minus the five-dual lookahead, *s* = 2 and *s* = 3 | `results/paper/experiments/rich_models_data_vs_five_dual.tex`; `results/rich_models_data_paired_five_dual.csv` |
| Table 17 | learned predictor: hit@1, search cost, labeling cost, by split and variant | `results/paper/experiments/learned_predictor.tex`; `results/learned/` |
| Table 18 | learned predictor against the number of training instances of the same class | `results/paper/experiments/learned_predictor_training_size.tex`; `results/learned/` |
| Table 19 | benchmark instances, per class and method: hit@1, gain of the first set, share on which the first set is the time-optimal set, strictly and within the tolerance, and time ratio | `results/paper/experiments/tab_app_bench.tex`; `results/experiment_C/benchmark_methods.csv`, `results/experiment_D/methods_time_budget.csv` |
| Table 20 | benchmark instances: paired differences in the time-optimal hit and in the logarithm of the time ratio | `results/paper/experiments/tab_app_bench_time.tex`; `results/experiment_D/paired_time_truth_budget.csv` |
| Table 21 | benchmark instances: paired differences in hit@1 and in the gain of the first set | `results/paper/experiments/experiment_C_paired_hit1.tex`; `results/experiment_C/benchmark_paired_hit1.csv`, `benchmark_paired_gain.csv` |
| Table 22 | benchmark instances: branch-and-price on the best pair, the other pairs, the single families and all three families | `results/paper/experiments/tab_app_tradeoff.tex`; `results/experiment_D/tradeoff_budget.csv` |
| Table 23 | benchmark instances with every coefficient shown: hit@1, gain, time-optimal hit and time ratio, and paired differences against the baselines and the original presentation | `results/paper/experiments/tab_app_bench_data.tex`; `results/experiment_C/benchmark_data_methods.csv`, `benchmark_data_paired.csv`, `results/experiment_D/methods_time_budget.csv`, `paired_time_truth_budget.csv`, `paired_time_truth.csv` |
| Table 24 | instances without a joint group, per class and method: hit@1, time ratios, false alarms, sets outside the budget | `results/paper/experiments/experiment_F_methods.tex`, `experiment_F_paired.tex`; `results/experiment_F/methods.csv`, `methods_time.csv` |
| Table 25 | rich models with 12 to 16 families: hit@1, recall and evaluations until a best set is evaluated | `results/paper/experiments/experiment_E_predictions.tex`, `experiment_E_search.tex`, `experiment_E_paired.tex`; `results/experiment_E/predictions.csv`, `search_costs.csv`, `paired_five_dual.csv`, `paired_covering.csv` |
| Table 26 | rich models at *s* = 2: evaluations of the staged order alone and of JointDW by the number of missed members | `results/paper/experiments/tab_app_misses.tex`; `results/paper/figure_data/search_cost_vs_misses_after_cut.csv`, `results/rich_models/misses_per_prediction.csv` |
| Table 27 | rich models at *s* = 3: predictions whose anchor has two families while the best set has three | `results/paper/experiments/tab_app_larger.tex`; `results/paper/figure_data/search_cost_vs_misses_after_cut.csv` |
| Table 28 | cost of a prediction: latency, cost at list prices, evaluations saved, net time and break-even time | `results/paper/experiments/experiment_G_net_benefit.tex`; `results/experiment_G/` |
| Table 29 | time of one evaluation by the number of convexified families, and its growth exponent | `results/paper/experiments/eval_growth.tex`; `results/eval_growth/` |
| Table 30 | paired differences in evaluations against the covering and the lookahead orders | `results/paper/table_15.tex` |

## Further outputs

These outputs have no table number in the paper.

| Output | What it contains | Result file |
|---|---|---|
| Search cost without standard errors | mean evaluations until a best decomposition is evaluated, by order and *s*, on the templates and MIPLIB-NL | `results/paper/table_02.tex` |
| First set against the lookahead | instances on which a configuration and the lookahead order name a true joint support first | `results/paper/table_10.tex` |
| Ranked variant | coverage@1, coverage@3, and the ranked order interleaved with the lookahead minus the lookahead | `results/paper/table_13.tex`; `results/ranked_prompt.csv`, `ranked_prompt_paired.csv` |
| Solve time on MIPLIB-NL | paired differences in solve time against the covering and the lookahead orders | `results/paper/table_14.tex` |
| Rich models with 7 to 9 families, control presentations | hit@1 given the text, every coefficient and the two control presentations; evaluations at *s* = 2 by class | `results/paper/figure_4a.csv`, `figure_4b.csv` |
| Combined orders | union-first and stacked orders: search cost against the five-dual lookahead order, JointDW interleaved and the learned order | `results/paper/experiments/combined_union_first.tex`, `combined_stacked.tex`; `results/combined/` |
| Benchmark instances against the best pair | per class and method: hit@1, gain of the first set, branch-and-price time, final gap, instances solved; paired time differences and time ratios against the best pair; with every coefficient shown, the same and its paired differences | `results/paper/experiments/experiment_C_methods.tex`, `experiment_C_paired_time.tex`, `experiment_C_against_true_best.tex`, `experiment_C_data_methods.tex`, `experiment_C_data_paired.tex`; `results/experiment_C/` |
| Time-optimal set among all sets | the methods against the time-optimal set among every set, including all three families: hits, time ratios, paired differences, and bound strength against pricing effort | `results/paper/experiments/experiment_D_methods.tex`, `experiment_D_paired.tex`, `experiment_D_tradeoff.tex`; `results/experiment_D/time_truth.csv`, `methods_time.csv`, `paired_time_truth.csv`, `tradeoff_by_class.csv`, `bnp_runs.csv`; `results/paper/figure_data/experiment_D_tradeoff.csv` |
| Misses before the cut | search cost of the rich models by the number of missed members counted on the anchor before its cut to *s* families, and the same body by the misses after the cut | `results/paper/figure_data/search_cost_vs_misses.csv`; `results/paper/experiments/search_cost_vs_misses_after_cut.tex` |
| Rich models across the number of families | evaluations of every order by class, *k* and *s* | `results/paper/figure_data/experiment_E_cost_vs_k.csv` |
| Time of one evaluation | the census time of every set that the search orders visit, per collection | `results/eval_time/`, `results/eval_time_large/`, `results/experiment_G/eval_time.csv` |
| Answer-level scores | one row per answer of the rich models with 12 to 16 families and of the instances without a joint group | `results/experiment_E/answers_scored.csv`, `results/experiment_F/answers_scored.csv` |

## Modules and constants

`scripts/reproduce.sh` reads only `data/`, sends no request to a language model and needs no solver. It runs `python -m llmdw.scoring.results` (the result tables, written to `results/`); the scoring of the control presentations, of the five-dual lookahead and of the rich models with every coefficient against it (`python -m llmdw.scoring.experiments`); the benchmark instances (`python -m llmdw.benchmark.tables`, `python -m llmdw.benchmark.data_tables`); the learned predictor and the combined orders on the released features (`python -m llmdw.scoring.learned`, `python -m llmdw.scoring.combined`); the table bodies and figure data of the paper (`python -m llmdw.tables.figure2`, `python -m llmdw.tables.paper`, `python -m llmdw.tables.experiments`); the time-optimal sets on the benchmark instances (`python -m llmdw.benchmark.time_truth tables`); the classes without a joint group (`python -m llmdw.benchmark.no_joint_group score`); the rich models with 12 to 16 families and the misses of every prediction (`python -m llmdw.scoring.rich_models_large`, `python -m llmdw.tables.rich_figure_data`); the time of one evaluation and its growth with the number of convexified families (`python -m llmdw.benchmark.eval_time tables`, `python -m llmdw.scoring.eval_growth`); the cost accounting (`python -m llmdw.scoring.cost_accounting requests` and `tables`); and the tables in the layout of the paper (`python -m llmdw.tables.paper_v1`). The table bodies and figure data are written to `results/paper/`. `scripts/compare_results.sh` compares every file with the released results in `data/reference_results/`. Every search cost is the mean over 1,000 random orders of the families with the recorded seed, every standard error comes from a two-stage bootstrap with 10,000 resamples and seed 0 that resamples the groups of instances with identical prompts and then the instances within each drawn group, and lookahead values within 1e-9 are tied; these constants are in [`llmdw/constants.py`](../llmdw/constants.py). The instance files behind Table 8 are rebuilt by `scripts/fetch_benchmarks.sh` and `python -m llmdw.benchmark.standard build`, whose model files match the SHA-256 recorded in `data/work/benchmark_std/instances/*/meta.json`.

## Tests and timings

`tests/test_paper_tables.py` rebuilds every table body from the released results and from `results/` and compares it with the reference bodies in `tests/reference/`, and recomputes the cells of Tables 6 to 8 that no result CSV holds (the sizes, the number of instances of each minimal best set, the number of zero-marginal groups per instance, the instances with a positive gain and the instance counts) from the instance files and the gain tables in `data/work/`. With `JOINTDW_PAPER_TEX` set to the LaTeX source of the paper, it also compares every table body of the paper with these results. On a fresh clone of this repository (one workstation, Python 3.13), the installation took 16 s, `scripts/fetch_data.sh` 3 s, `scripts/fetch_benchmarks.sh` 28 s, rebuilding and checking the 219 instance files of the standard formulations 74 s and the 80 instance files of generalized assignment and bin packing 2 s, the tests 27 s and `scripts/reproduce.sh` 66 minutes (the learned predictor 21 minutes and the combined orders 30 minutes of it); `scripts/compare_results.sh` then found `results/` identical to `data/reference_results/` (328 files).

## Repository layout in detail

```
llmdw/instances/   templates, rich-model generators (7 to 9 families; 12 to 16 families in rich_models_large), MIPLIB-NL
                   export, family detection, model summaries, reference optima
llmdw/exact/       exact gain tables (HiGHS with rational verification in python-flint)
llmdw/lookahead/   one-pass lookahead at an optimal LP dual, and at five optimal duals
llmdw/search/      staged search, covering orders, interleaving, relabeling-invariant search cost
llmdw/prompts/     presentations, per-instance labels, prompt rendering, answer parsing, name rule, re-reading of answers
llmdw/providers/   OpenAI-compatible client, cache, request records, configurations, replay check
llmdw/scoring/     hit@1, recall, coverage@m, cluster bootstrap, census, bound validity rule, all result tables,
                   control presentations, five-dual lookahead, learned predictor, combined orders, large models,
                   rich models with 12 to 16 families and the misses of each prediction (rich_models_large),
                   lattice structure of the gain tables (gain_structure), evaluation time against the number of
                   convexified families (eval_growth), latency and cost of the predictions (cost_accounting)
llmdw/benchmark/   benchmark sources, data checks, standard formulations, ground truth, branch-and-price, tables,
                   time-optimal sets (time_truth), reference optima and validity of branch-and-price runs (validity),
                   generalized assignment and bin packing (extra_classes, no_joint_group), time of one evaluation
                   (eval_time)
llmdw/tables/      table bodies and figure data of the paper (official names), cell records of the table bodies
                   (cells), figure data of the rich models (rich_figure_data), tables in the layout of the paper,
                   their comparison with the paper source and the derived cells of the instance tables (paper_v1)
llmdw/gcg/         GCG decompositions and bound evaluations, memory caps, the shared memory budget and the cap on
                   active solver processes
scripts/           fetch_data.sh, fetch_benchmarks.sh, reproduce.sh, compare_results.sh, rerun.sh
config/            public_settings.json (request settings), benchmark_sources.json (URLs and SHA-256 of the
                   benchmark files), high_memory_models.txt (models solved with a 10 GB cap)
prompts/           system message, user template, ranked and data variants
tests/             offline tests; tests/reference holds the table bodies of the sample of record and the derived
                   cells of the instance tables
docs/EXPERIMENTS.md   the settings of every experiment
docs/figures/      the figures of this README (lattice.png, cost.png, results.png, rich.png, misses.png)
```
