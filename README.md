<div align="center">

<h1>Do Language Models Know Which Constraints Belong Together?</h1>

<p>
<b>Junbo Jacob Lian</b><sup>1,2,3</sup> &nbsp;·&nbsp;
<b>Kaixuan Chen</b><sup>3</sup> &nbsp;·&nbsp;
<b>Hanzhang Qin</b><sup>1</sup> &nbsp;·&nbsp;
<b>Huiling Chen</b><sup>4</sup> &nbsp;·&nbsp;
<b>Chaoyu Zhang</b><sup>5</sup> &nbsp;·&nbsp;
<b>Chung-Piaw Teo</b><sup>1</sup>
</p>

<p>
<sup>1</sup> Institute of Operations Research and Analytics, National University of Singapore<br>
<sup>2</sup> Wenzhou Buyi Pharmacy<br>
<sup>3</sup> McCormick School of Engineering, Northwestern University<br>
<sup>4</sup> College of Computer Science and Artificial Intelligence, Wenzhou University<br>
<sup>5</sup> Department of Decision Analytics and Operations, City University of Hong Kong
</p>

<p>
<a href="https://arxiv.org/abs/2611.05726"><img src="https://img.shields.io/badge/arXiv-2611.05726-b31b1b.svg" alt="arXiv 2611.05726"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="MIT license"></a>
<a href="DATA.md"><img src="https://img.shields.io/badge/data-CC%20BY%204.0%20%2F%20CC%20BY--SA%204.0-lightgrey.svg" alt="Data license"></a>
<a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.13-3776AB.svg" alt="Python 3.13"></a>
</p>

<img src="docs/figures/lattice.png" width="440" alt="Gains of every family-level Dantzig-Wolfe relaxation of a small lot-sizing instance">

<p><i>Share of the integrality gap closed by every family-level Dantzig-Wolfe relaxation of a lot-sizing instance with two items and two periods (B: balance rows, S: setup rows, C: capacity rows). The shaded sets have gain zero, so a search that adds one family at a time never leaves the empty set, while the pairs BS and CS each close 63% of the gap.</i></p>

</div>

## Overview

Dantzig-Wolfe reformulation strengthens the linear relaxation of a mixed-integer program by convexifying a chosen set of constraint families, and every candidate costs a column generation run. The choice is hard when families strengthen the bound only together, because a search that adds one family at a time then stops at the linear relaxation. We call such a set of families a zero-marginal group. This repository holds the code and data of the [paper](https://arxiv.org/abs/2611.05726): exact gain tables for small templates and for 84 rich models with 7 to 16 constraint families; the census of zero-marginal groups in MIPLIB-NL; 59 standard lot-sizing, network-design and facility-location instances and 76 generalized assignment and bin-packing instances without a joint group, with their gain tables and branch-and-price runs; the prompts and the recorded answers of five language-model configurations under the presentations of each model; the one-pass lookahead at an optimal LP dual and its five-dual version; a learned predictor trained on the gain tables; the search orders (JointDW, its interleaving with the lookahead, covering orders); the latency and cost of every request; and the scripts that build every table and figure of the paper from the released data without calling any API.

**Main results**

- Zero-marginal groups are common: three of ten MIPLIB-NL models whose bound can be improved have a certain one and another third a possible one, and on most of them no single family strengthens the bound at all. Every one of the 59 benchmark instances has one, and GCG's detector, which follows the block structure of the matrix, convexifies a set with no gain on 38 of them.
- When each evaluation may convexify at most *s* of *k* families, every method that observes only bounds needs a covering number of evaluations, of order *k*<sup>*w*</sup>, to find a hidden group of *w* families. A prediction that misses *e* members reduces the cost to Θ(*k*<sup>*e*</sup>) through one staged search that does not need to know *e*, and the search costs only a lower-order term more when the prediction is entirely wrong. With several groups the *r*-th best prediction sets the cost, and when an evaluation costs the *d*-th power of the number of convexified families the optimal cost is Θ(*k*<sup>min(*e*,*d*)</sup>).
- The one-pass lookahead at an optimal LP dual vanishes on groups whose families share no variable. JointDW takes the prediction of a language model, verifies it with the staged search, and alternates it with the lookahead order, which delays each of the two orders by at most a factor of two.
- On MIPLIB-NL the strongest language models identify zero-marginal groups about as often as the lookahead and on partly different models. On textbook templates their predictions rest on the names: permuting or anonymizing the family names removes them. Reasoning roughly doubles the accuracy of the same model.
- On the rich models, which place the classical group among further modeling features, GPT-6.1 Sol given every coefficient names a true group first in 249 of its 252 answers on models with 7 to 16 families, and on the models with 7 to 9 families also when the family names are permuted or replaced by labels. Interleaved with the five-dual lookahead, it needs 1.8 evaluations for pairs on the models with 12 to 16 families, where a covering needs 17 to 60. A predictor trained on the gain tables of the same class matches this only after hundreds of labeled evaluations and does not carry over to another class.
- The search cost follows the number of missed members. A prediction without a miss costs one evaluation. With one member of a pair missed, JointDW needs about four fifths of the evaluations of the covering order with seven families and about half with 16, and with both members missed about 1.3 times as many.
- On the benchmark instances the best pair is also the fastest set of at most two families for branch-and-price on 51 of the 59 instances. The sets chosen by the two language models take 1.3 and 1.4 times the time of the fastest set in geometric mean, against 1.7 times for the five-dual lookahead and 2.3 times for GCG's detector. On the 76 instances without a joint group, GPT-6.1 Sol names none.
- One answer takes from half a minute to more than twenty minutes, so a prediction pays for its latency only where evaluations are slow. On MIPLIB-NL, where an evaluation takes 43 seconds on average, the prediction of GPT-6.1 Sol pays once an evaluation takes more than 28 seconds.

<p align="center"><img src="docs/figures/cost.png" width="640" alt="Queries of the staged search and evaluations to find a hidden group"></p>

*Figure 2. (a) Queries of the staged search for k = 6, s = 3, w = 2 and the prediction P = {1, 2}: the queries of stage e contain every hidden group with e members outside P, so a correct prediction is confirmed by the first query. (b) Evaluations to find a hidden group of w = 3 families with s = 6, for predictions that miss e members and without a prediction, with the lower bounds as dashed lines.*

<p align="center"><img src="docs/figures/results.png" alt="Census of MIPLIB-NL and recall by presentation"></p>

*Figure 3. (a) Census of MIPLIB-NL: for each model with a positive gain, the largest gain of a single family against the largest gain of a set of families, both as shares of the LP gap; marks on the vertical axis have no family that strengthens the bound alone, and blue marks have a zero-marginal group at the empty set. (b, c) Recall of true joint supports under the four presentations on the templates and on MIPLIB-NL; the dashed line is the recall of the lookahead order.*

<p align="center"><img src="docs/figures/rich.png" width="640" alt="Rich models: hit@1 and evaluations"></p>

*Figure 4. Rich models with 7 to 16 families. (a) hit@1 of GPT-6.1 Sol (circles) and Claude Sonnet 5.5 (squares) given the text (open) and every coefficient (filled), on the models with 7 to 9 and with 12 to 16 families. (b) Mean evaluations until a best set is evaluated at s = 2 against the number of families k, for lot sizing, network design and facility location: covering order, lookahead order at the HiGHS dual and with five duals, JointDW with GPT-6.1 Sol given every coefficient, and JointDW interleaved with the five-dual lookahead. Bars: one standard error.*

<p align="center"><img src="docs/figures/misses.png" width="640" alt="Search cost by the number of missed members"></p>

*Figure 5. Search cost on the rich models at s = 2, by the number e of members of a best pair that the prediction misses, with one mark per class and number of families k. (a) The staged order alone, with the number of queries of stages 0 to e as dashed lines. (b) JointDW and the covering order. Bars: one standard error.*

## Contents

- [Installation](#installation)
- [Data](#data)
- [Reproducing the paper](#reproducing-the-paper)
- [Running the language models](#running-the-language-models)
- [Building instances](#building-instances)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## Installation

The code needs Python 3.13.

```bash
git clone https://github.com/junbolian/JointDW.git
cd JointDW
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[test]"
```

`pip install -e .` installs the package with NumPy 2.5.4. The extra `learned` adds XGBoost (xgboost-cpu 3.4.1) with scikit-learn 1.9.1 and SciPy 1.18.1, which only the learned predictor uses; `pip install -e ".[learned]"` is all that reproducing the tables and figures from the released data needs. The extra `solvers` adds what the computations need: HiGHS (highspy 1.15.1) and python-flint 0.9.0, with SymPy 1.14.0, for the exact gain tables and the lookahead; PySCIPOpt 6.0.0 with PyGCGOpt 1.0.0b0 on SCIP 10 for the column generation evaluations, the benchmark instances and branch-and-price; gurobipy 13.0.3 and pandas 2.2.3 for exporting the MIPLIB-NL models. The extra `test` adds pytest 9.2.0 and both other extras. Bound evaluations solve the root node only, with presolving, separation, primal heuristics and SCIP propagation off, one thread and seed 0.

## Data

The data are the asset `jointdw_data_v1.2.tar.gz` of the release [v1.2](https://github.com/junbolian/JointDW/releases/tag/v1.2), which supersedes v1.1.1, v1.1 and v1.0.

| Asset | Contents | Size |
|---|---|---|
| `jointdw_data_v1.2.tar.gz` | everything of v1.1.1: templates and rich models with 7 to 9 families with their exact gain tables; MIPLIB-NL gain tables with the invalid-bound column and the corrected reference optima; lookahead values at one and at five duals; every recorded request and response of the five configurations under the four presentations, the ranked variant, the data variant (on the rich models and on the benchmark instances, with the answers set aside for text added by an endpoint), the two control presentations and the benchmark instances; the rendered prompts; the learned-predictor features; the benchmark gain tables, ground truth and branch-and-price records and logs; the large-model evaluations. In addition: the rich models with 12 to 16 families with their gain tables and records; the generalized assignment and bin-packing gain tables and records; the branch-and-price records and logs of the time-optimal sets; the latency and cost records with the price list and its dates; the answer-level scores; the search cost by the number of missed members; and the results CSVs behind every table and figure | 117.6 MB |

```bash
scripts/fetch_data.sh
```

downloads the data into `data/` and checks the SHA-256 recorded in the script. [DATA.md](DATA.md) describes every part of the data, how it was produced, the bound validity rule, the validity rule of branch-and-price runs, the reference optima, the prices and its license. The MPS files of the MIPLIB-NL models are not part of the release; [Building instances](#building-instances) shows how to export them from MIPLIB-NL.

The archive does not contain the instance files built from public benchmark files, nor the user messages of the 405 recorded prompts of the data variant on the benchmark instances, which embed a complete benchmark instance; the prompt builder (`llmdw.prompts.propose.build_messages` with the `with_data` template) recreates these messages from the rebuilt instance files, identical to the recorded ones for all 405. `scripts/fetch_benchmarks.sh` downloads the public benchmark files from their original locations (the OR-Library, CommaLAB and suerie.de, with the test list of the GCG repository, and the OR-Library files gap1 to gap12 and binpack1 of the generalized assignment and bin-packing instances) into `data/benchmark_raw/` and verifies every data file against the SHA-256 recorded when the instances were built. `python -m llmdw.benchmark.standard build --raw data/benchmark_raw --out work/benchmark_std` recreates the 219 instance files of the standard formulations, and `python -m llmdw.benchmark.standard check --out work/benchmark_std --recorded data/work/benchmark_std` verifies each against the SHA-256 recorded in its `meta.json` (219 of 219 identical on a fresh clone); `python -m llmdw.benchmark.extra_classes build --raw data/benchmark_raw --out work/no_joint_group` and `python -m llmdw.benchmark.extra_classes check --out work/no_joint_group --recorded data/work/no_joint_group` do the same for the 80 instance files of generalized assignment and bin packing (80 of 80 identical). [DATA.md](DATA.md) names the sources and their terms.

## Reproducing the paper

```bash
scripts/reproduce.sh
scripts/compare_results.sh
```

`scripts/reproduce.sh` reads only `data/`, sends no request to a language model and needs no solver. It runs `python -m llmdw.scoring.results` (the result tables, written to `results/`); the scoring of the control presentations, of the five-dual lookahead and of the rich models with every coefficient against it (`python -m llmdw.scoring.experiments`); the benchmark instances (`python -m llmdw.benchmark.tables`, `python -m llmdw.benchmark.data_tables`); the learned predictor and the combined orders on the released features (`python -m llmdw.scoring.learned`, `python -m llmdw.scoring.combined`); the table bodies and figure data of the paper (`python -m llmdw.tables.figure2`, `python -m llmdw.tables.paper`, `python -m llmdw.tables.experiments`); the time-optimal sets on the benchmark instances (`python -m llmdw.benchmark.time_truth tables`); the classes without a joint group (`python -m llmdw.benchmark.no_joint_group score`); the rich models with 12 to 16 families and the misses of every prediction (`python -m llmdw.scoring.rich_models_large`, `python -m llmdw.tables.rich_figure_data`); the time of one evaluation and its growth with the number of convexified families (`python -m llmdw.benchmark.eval_time tables`, `python -m llmdw.scoring.eval_growth`); the cost accounting (`python -m llmdw.scoring.cost_accounting requests` and `tables`); and the tables in the layout of the paper (`python -m llmdw.tables.paper_v1`). The table bodies and figure data are written to `results/paper/`. `scripts/compare_results.sh` compares every file with the released results in `data/reference_results/`. Every search cost is the mean over 1,000 random orders of the families with the recorded seed, every standard error comes from a two-stage bootstrap with 10,000 resamples and seed 0 that resamples the groups of instances with identical prompts and then the instances within each drawn group, and lookahead values within 1e-9 are tied; these constants are in [`llmdw/constants.py`](llmdw/constants.py). The instance files behind Table 8 are rebuilt by `scripts/fetch_benchmarks.sh` and `python -m llmdw.benchmark.standard build`, whose model files match the SHA-256 recorded in `data/work/benchmark_std/instances/*/meta.json`.

| Output | What it contains | Command | Result file |
|---|---|---|---|
| Figure 1 | gains of all family subsets of the lot-sizing template with two items and two periods | `scripts/reproduce.sh` | `results/paper/figure_1.csv` |
| Figure 2 | (a) the queries of the staged search for *k* = 6, *s* = 3, *w* = 2; (b) queries of the staged search and the lower bounds for *w* = 3, *s* = 6, *k* = 8 to 24 | `scripts/reproduce.sh` | `results/paper/figure_2a.csv`, `figure_2b.csv`; `results/figure2.csv` |
| Figure 3 | (a) MIPLIB-NL census; (b, c) recall under the four presentations on the templates and on MIPLIB-NL, with the lookahead recall | `scripts/reproduce.sh` | `results/paper/figure_3a_certain.csv`, `figure_3a_possible.csv`, `figure_3a_none.csv`, `figure_3b_templates.csv`, `figure_3c_miplib.csv` |
| Figure 4 | rich models with 7 to 16 families: hit@1 given the text and given every coefficient; evaluations at *s* = 2 by class and number of families for the covering order, the two lookahead orders, JointDW and its interleaving with the five-dual lookahead | `scripts/reproduce.sh` | `results/paper/figure_data/rich_hit1.csv`, `rich_cost_by_class_k.csv` |
| Figure 5 | rich models at *s* = 2: evaluations of the staged order alone and of JointDW by the number of missed members, with the stage counts and the covering order | `scripts/reproduce.sh` | `results/paper/figure_data/search_cost_vs_misses_after_cut.csv` |
| Table 1 | instance collections: number of instances, number of families, source of the gains | `scripts/reproduce.sh` | `results/paper/experiments/tab_instances.tex` |
| Table 2 | hit@1 and recall in the original presentation on the templates and MIPLIB-NL, false alarms on the assignment templates | `scripts/reproduce.sh` | `results/paper/table_01.tex` |
| Table 3 | mean evaluations until a best decomposition is evaluated, by order and *s* | `scripts/reproduce.sh` | `results/paper/table_09.tex` |
| Table 4 | benchmark instances: share of instances on which the first set of each method is the best pair, by class; share on which it is the time-optimal set and its time ratio; the same two quantities on the instances without a joint group | `scripts/reproduce.sh` | `results/paper/experiments/tab_bench.tex`; `results/experiment_C/benchmark_methods.csv`, `results/experiment_D/methods_time_budget.csv`, `results/experiment_F/methods_time.csv` |
| Table 5 | exact gains of the templates | `scripts/reproduce.sh` | `results/templates_table.csv`, `results/covering_numbers.csv` |
| Table 6 | rich models with 7 to 9 families: families, size, LP gap, minimal best sets, zero-marginal groups | `scripts/reproduce.sh` | `results/rich_models_instances.csv`, `results/rich_models_groups.csv` |
| Table 7 | rich models with 12 to 16 families: families, size, LP gap, zero-marginal groups, instances with a positive gain at *s* = 2 | `scripts/reproduce.sh` | `results/experiment_E/instances.csv`, `results/paper/experiments/experiment_E_instances.tex` |
| Table 8 | benchmark instances: counts, sizes, LP gaps, best pairs | `scripts/reproduce.sh` | `results/experiment_C/benchmark_instances.csv`, `results/paper/experiments/experiment_C_instances.tex` |
| Table 9 | recall by presentation, mean ± standard error | `scripts/reproduce.sh` | `results/paper/table_06.tex` |
| Table 10 | paired differences in recall between presentations | `scripts/reproduce.sh` | `results/paper/table_07.tex` |
| Table 11 | Grok 4.20 (Reasoning) minus Grok 4.20 (Non-Reasoning), paired over instances | `scripts/reproduce.sh` | `results/paper/table_08.tex` |
| Table 12 | five-dual lookahead: first set and search cost, and JointDW interleaved with it | `scripts/reproduce.sh` | `results/paper/experiments/experiment_B_first_set.tex`, `experiment_B_search.tex`; `results/experiment_B/` |
| Table 13 | rich models with 7 to 9 families: hit@1 and recall given the text and given every coefficient | `scripts/reproduce.sh` | `results/paper/table_11.tex` |
| Table 14 | rich models with 7 to 9 families: evaluations until a best set is evaluated | `scripts/reproduce.sh` | `results/paper/table_12.tex` |
| Table 15 | control presentations with every coefficient: hit@1, recall and search cost with the names permuted or replaced by labels | `scripts/reproduce.sh` | `results/paper/experiments/experiment_A_prediction.tex`, `experiment_A_search.tex`; `results/experiment_A/` |
| Table 16 | rich models with every coefficient: JointDW and its interleaving with the five-dual lookahead minus the five-dual lookahead, *s* = 2 and *s* = 3 | `scripts/reproduce.sh` | `results/paper/experiments/rich_models_data_vs_five_dual.tex`; `results/rich_models_data_paired_five_dual.csv` |
| Table 17 | learned predictor: hit@1, search cost, labeling cost, by split and variant | `scripts/reproduce.sh` | `results/paper/experiments/learned_predictor.tex`; `results/learned/` |
| Table 18 | learned predictor against the number of training instances of the same class | `scripts/reproduce.sh` | `results/paper/experiments/learned_predictor_training_size.tex`; `results/learned/` |
| Table 19 | benchmark instances, per class and method: hit@1, gain of the first set, share on which the first set is the time-optimal set, strictly and within the tolerance, and time ratio | `scripts/reproduce.sh` | `results/paper/experiments/tab_app_bench.tex`; `results/experiment_C/benchmark_methods.csv`, `results/experiment_D/methods_time_budget.csv` |
| Table 20 | benchmark instances: paired differences in the time-optimal hit and in the logarithm of the time ratio | `scripts/reproduce.sh` | `results/paper/experiments/tab_app_bench_time.tex`; `results/experiment_D/paired_time_truth_budget.csv` |
| Table 21 | benchmark instances: paired differences in hit@1 and in the gain of the first set | `scripts/reproduce.sh` | `results/paper/experiments/experiment_C_paired_hit1.tex`; `results/experiment_C/benchmark_paired_hit1.csv`, `benchmark_paired_gain.csv` |
| Table 22 | benchmark instances: branch-and-price on the best pair, the other pairs, the single families and all three families | `scripts/reproduce.sh` | `results/paper/experiments/tab_app_tradeoff.tex`; `results/experiment_D/tradeoff_budget.csv` |
| Table 23 | benchmark instances with every coefficient shown: hit@1, gain, time-optimal hit and time ratio, and paired differences against the baselines and the original presentation | `scripts/reproduce.sh` | `results/paper/experiments/tab_app_bench_data.tex`; `results/experiment_C/benchmark_data_methods.csv`, `benchmark_data_paired.csv`, `results/experiment_D/methods_time_budget.csv`, `paired_time_truth_budget.csv`, `paired_time_truth.csv` |
| Table 24 | instances without a joint group, per class and method: hit@1, time ratios, false alarms, sets outside the budget | `scripts/reproduce.sh` | `results/paper/experiments/experiment_F_methods.tex`, `experiment_F_paired.tex`; `results/experiment_F/methods.csv`, `methods_time.csv` |
| Table 25 | rich models with 12 to 16 families: hit@1, recall and evaluations until a best set is evaluated | `scripts/reproduce.sh` | `results/paper/experiments/experiment_E_predictions.tex`, `experiment_E_search.tex`, `experiment_E_paired.tex`; `results/experiment_E/predictions.csv`, `search_costs.csv`, `paired_five_dual.csv`, `paired_covering.csv` |
| Table 26 | rich models at *s* = 2: evaluations of the staged order alone and of JointDW by the number of missed members | `scripts/reproduce.sh` | `results/paper/experiments/tab_app_misses.tex`; `results/paper/figure_data/search_cost_vs_misses_after_cut.csv`, `results/rich_models/misses_per_prediction.csv` |
| Table 27 | rich models at *s* = 3: predictions whose anchor has two families while the best set has three | `scripts/reproduce.sh` | `results/paper/experiments/tab_app_larger.tex`; `results/paper/figure_data/search_cost_vs_misses_after_cut.csv` |
| Table 28 | cost of a prediction: latency, cost at list prices, evaluations saved, net time and break-even time | `scripts/reproduce.sh` | `results/paper/experiments/experiment_G_net_benefit.tex`; `results/experiment_G/` |
| Table 29 | time of one evaluation by the number of convexified families, and its growth exponent | `scripts/reproduce.sh` | `results/paper/experiments/eval_growth.tex`; `results/eval_growth/` |
| Table 30 | paired differences in evaluations against the covering and the lookahead orders | `scripts/reproduce.sh` | `results/paper/table_15.tex` |
| Search cost without standard errors | mean evaluations until a best decomposition is evaluated, by order and *s*, on the templates and MIPLIB-NL | `scripts/reproduce.sh` | `results/paper/table_02.tex` |
| First set against the lookahead | instances on which a configuration and the lookahead order name a true joint support first | `scripts/reproduce.sh` | `results/paper/table_10.tex` |
| Ranked variant | coverage@1, coverage@3, and the ranked order interleaved with the lookahead minus the lookahead | `scripts/reproduce.sh` | `results/paper/table_13.tex`; `results/ranked_prompt.csv`, `ranked_prompt_paired.csv` |
| Solve time on MIPLIB-NL | paired differences in solve time against the covering and the lookahead orders | `scripts/reproduce.sh` | `results/paper/table_14.tex` |
| Rich models with 7 to 9 families, control presentations | hit@1 given the text, every coefficient and the two control presentations; evaluations at *s* = 2 by class | `scripts/reproduce.sh` | `results/paper/figure_4a.csv`, `figure_4b.csv` |
| Combined orders | union-first and stacked orders: search cost against the five-dual lookahead order, JointDW interleaved and the learned order | `scripts/reproduce.sh` | `results/paper/experiments/combined_union_first.tex`, `combined_stacked.tex`; `results/combined/` |
| Benchmark instances against the best pair | per class and method: hit@1, gain of the first set, branch-and-price time, final gap, instances solved; paired time differences and time ratios against the best pair; with every coefficient shown, the same and its paired differences | `scripts/reproduce.sh` | `results/paper/experiments/experiment_C_methods.tex`, `experiment_C_paired_time.tex`, `experiment_C_against_true_best.tex`, `experiment_C_data_methods.tex`, `experiment_C_data_paired.tex`; `results/experiment_C/` |
| Time-optimal set among all sets | the methods against the time-optimal set among every set, including all three families: hits, time ratios, paired differences, and bound strength against pricing effort | `scripts/reproduce.sh` | `results/paper/experiments/experiment_D_methods.tex`, `experiment_D_paired.tex`, `experiment_D_tradeoff.tex`; `results/experiment_D/time_truth.csv`, `methods_time.csv`, `paired_time_truth.csv`, `tradeoff_by_class.csv`, `bnp_runs.csv`; `results/paper/figure_data/experiment_D_tradeoff.csv` |
| Misses before the cut | search cost of the rich models by the number of missed members counted on the anchor before its cut to *s* families, and the same body by the misses after the cut | `scripts/reproduce.sh` | `results/paper/figure_data/search_cost_vs_misses.csv`; `results/paper/experiments/search_cost_vs_misses_after_cut.tex` |
| Rich models across the number of families | evaluations of every order by class, *k* and *s* | `scripts/reproduce.sh` | `results/paper/figure_data/experiment_E_cost_vs_k.csv` |
| Time of one evaluation | the census time of every set that the search orders visit, per collection | `scripts/reproduce.sh` | `results/eval_time/`, `results/eval_time_large/`, `results/experiment_G/eval_time.csv` |
| Answer-level scores | one row per answer of the rich models with 12 to 16 families and of the instances without a joint group | `scripts/reproduce.sh` | `results/experiment_E/answers_scored.csv`, `results/experiment_F/answers_scored.csv` |

`tests/test_paper_tables.py` rebuilds every table body from the released results and from `results/` and compares it with the reference bodies in `tests/reference/`, and recomputes the cells of Tables 6 to 8 that no result CSV holds (the sizes, the number of instances of each minimal best set, the number of zero-marginal groups per instance, the instances with a positive gain and the instance counts) from the instance files and the gain tables in `data/work/`. With `JOINTDW_PAPER_TEX` set to the LaTeX source of the paper, it also compares every table body of the paper with these results. On a fresh clone of this repository (one workstation, Python 3.13), the installation took 16 s, `scripts/fetch_data.sh` 3 s, `scripts/fetch_benchmarks.sh` 28 s, rebuilding and checking the 219 instance files of the standard formulations 74 s and the 80 instance files of generalized assignment and bin packing 2 s, the tests 27 s and `scripts/reproduce.sh` 66 minutes (the learned predictor 21 minutes and the combined orders 30 minutes of it); `scripts/compare_results.sh` then found `results/` identical to `data/reference_results/` (328 files).

## Running the language models

Every answer that the paper scores is in the released answer files, and every request has an entry in the released response cache, so reproducing the paper sends no request. A new run needs a key and a base URL for each provider, read from the environment variables listed in [`.env.example`](.env.example); the client reads the cache before it sends anything.

| Configuration | Provider | Endpoint identifier | Additional request fields | Notes |
|---|---|---|---|---|
| Claude Sonnet 5.5 | Anthropic | `claude-sonnet-5-5` | none | no request option set; the default thinking of the endpoint applies |
| GPT-6.1 Sol | OpenAI | `gpt-6.1-sol` | `{"reasoning_effort": "medium"}` | medium reasoning effort; the API returned the snapshot gpt-6.1-sol-2026-09-29 for the rich models and the prompt variants and gpt-6.1-sol otherwise |
| Grok 4.20 (Reasoning) | xAI | `grok-4-20-reasoning` | none | |
| Grok 4.20 (Non-Reasoning) | xAI | `grok-4-20-non-reasoning` | none | |
| DeepSeek-V4-Pro (Non-Thinking) | DeepSeek | `deepseek-v4-pro` | `{"thinking": {"type": "disabled"}}` | non-thinking mode |

[`config/public_settings.json`](config/public_settings.json) holds these settings and the other client settings of each configuration (streaming, JSON mode, timeout, no temperature and no completion limit). `--config` takes the endpoint identifier and applies them.

Every request consists of one system message ([`prompts/decomp_v2.md`](prompts/decomp_v2.md), section SYSTEM) and one user message ([`prompts/decomp_v2.md`](prompts/decomp_v2.md), section USER) that ask for one JSON object with the per-family answers, the joint supports and up to three candidate reformulations. The four presentations are original (names, description and formulation), names only, permuted (the family names permuted among the families) and anonymized (F1, ..., Fk and V1, ..., Vm). The ranked variant ([`prompts/user_ranked.txt`](prompts/user_ranked.txt)) also asks for up to five ranked groups with probabilities, and the data variant ([`prompts/user_with_data.txt`](prompts/user_with_data.txt)) adds the complete instance in CPLEX LP format and is used on the rich models and on the benchmark instances. Each instance and presentation has three samples. The records contain:

| Records | Presentations | Configurations | Samples | Instances |
|---|---|---|---|---|
| templates and MIPLIB-NL | original, names only, permuted, anonymized | all five | three | 15 templates, 102 scored MIPLIB-NL models |
| rich models | original (the text) | all five | three | 24 rich models with 7 to 9 families |
| data variant | the complete instance in CPLEX LP format | all five | three | 24 rich models with 7 to 9 families |
| ranked variant | original, with ranked groups | all five | three | 13 templates, 16 MIPLIB-NL models, 24 rich models |
| control presentations with coefficients | data permuted, data anonymized | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 24 rich models with 7 to 9 families |
| benchmark instances | original | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 59 instances, giving eight distinct prompts: one for lot sizing, four for network design and three for facility location |
| benchmark instances with every coefficient | data variant (the complete instance in CPLEX LP format) | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 59 instances, one prompt per instance; the prompt-token guard of this run and the set-aside GPT-6.1 Sol responses: [DATA.md, run setting of the benchmark data variant](DATA.md#run-setting-of-the-benchmark-data-variant) |
| rich models with 12 to 16 families | original (the text) | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 60 rich models with 12 to 16 families (180 answers per configuration) |
| rich models with 12 to 16 families, every coefficient | data variant (the complete instance in CPLEX LP format) | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 60 rich models with 12 to 16 families (180 answers per configuration) |
| generalized assignment and bin packing | original | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 76 instances without a joint group (60 generalized assignment, 16 bin packing): 228 answers of Claude Sonnet 5.5 and 218 of GPT-6.1 Sol; the other 10 samples of GPT-6.1 Sol (gap_c1040_1 to gap_c1040_3, gap_c848_1, gap_c848_2) have no answer, every attempt having been rejected because it reported no reasoning, and are scored as missing |
| latency and cost | every request of the records above | all five | every sample | 11,636 requests, rejected attempts included (`results/experiment_G/requests.csv`), each priced at the list price of its configuration on the day of the request (26 configurations and days, `data/work/cost_accounting/prices.csv`) |

The control presentations permute the family names among the families (data permuted) or replace them by labels F1, ..., Fk and V1, ..., Vm (data anonymized), with every coefficient shown. The permutation and the label map of each instance are drawn from a seed derived from the instance name (`--labels per-instance`, `llmdw.prompts.counterfactual`); after the permutation the pair carrying the classical names is a true group in no instance.

With `MODEL` set to an endpoint identifier of the table:

```bash
python -m llmdw.providers.smoke --config "$MODEL"
python -m llmdw.providers.support_run --config "$MODEL" --templates data/work/templates \
    --conditions original,names_only,permuted,anonymized --samples 3 --parallel 4 \
    --out "data/work/support_templates/$MODEL"
python -m llmdw.providers.support_run --config "$MODEL" --rich-models data/work/rich_models --prompt with_data \
    --conditions original --samples 3 --parallel 4 --out "data/work/rich_models_with_data/$MODEL"
```

The rich models with 12 to 16 families, under the original presentation and the data variant, with `MODEL` set to `gpt-6.1-sol` or `claude-sonnet-5-5`:

```bash
python -m llmdw.providers.support_run --config "$MODEL" --rich-models data/work/rich_models_large \
    --conditions original --samples 3 --parallel 4 --out "data/work/support_rich_models_large/$MODEL"
python -m llmdw.providers.support_run --config "$MODEL" --rich-models data/work/rich_models_large --prompt with_data \
    --conditions original --samples 3 --parallel 4 --out "data/work/rich_models_large_with_data/$MODEL"
```

The instances without a joint group need their instance files (`scripts/fetch_benchmarks.sh` and `python -m llmdw.benchmark.extra_classes build --raw data/benchmark_raw --out work/no_joint_group`); one request per configuration was in flight:

```bash
cp data/work/no_joint_group/all_instances.txt work/no_joint_group/instances.txt
python -m llmdw.providers.support_run --config "$MODEL" --rich-models work/no_joint_group \
    --conditions original --samples 3 --parallel 1 --out "work/no_joint_group_answers/$MODEL"
```

The latency and cost records are read from the call records by `python -m llmdw.scoring.cost_accounting requests` and priced by `python -m llmdw.scoring.cost_accounting tables`, both run by `scripts/reproduce.sh`; [DATA.md](DATA.md#prices) describes the price list and the rule that chose each price.

The data variant on the benchmark instances needs the built instances (`scripts/rerun.sh benchmark`, which builds `work/benchmark_std`) and the run settings recorded in [`DATA.md`](DATA.md) (prompt-token guard factor 0.50 for GPT-6.1 Sol and 0.85 for Claude Sonnet 5.5, client timeout 600 s for Claude Sonnet 5.5; DATA.md also describes the exact-count check, which rejects a GPT-6.1 Sol response whose prompt tokens differ from the tiktoken `o200k_base` count of the request plus 42):

```bash
mkdir -p work/benchmark_data && cp data/work/benchmark_std/full_set/all_instances.txt work/benchmark_data/instances.txt
ln -sfn ../benchmark_std/instances work/benchmark_data/instances
python -m llmdw.providers.support_run --config gpt-6.1-sol --rich-models work/benchmark_data --prompt with_data \
    --conditions original --samples 3 --parallel 2 --prompt-tokens-per-char 0.50 --out work/answers_data/gpt-6.1-sol
python -m llmdw.providers.support_run --config claude-sonnet-5-5 --rich-models work/benchmark_data --prompt with_data \
    --conditions original --samples 3 --parallel 2 --prompt-tokens-per-char 0.85 --timeout 600 \
    --out work/answers_data/claude-sonnet-5-5
```

The replay check, `python -m llmdw.providers.replay --config "$MODEL"`, sends recorded requests again byte for byte, with the response cache bypassed, and compares the prompt tokens reported now with the recorded ones. It needs the key of the configuration, and the tests skip it without one.

## Building instances

The released data make these steps unnecessary for reproducing the tables.

<details>
<summary>Templates, hidden-group programs, rich models, MIPLIB-NL models, benchmark instances and instances without a joint group</summary>

The templates (lot sizing, network design, facility location, generalized assignment) and the hidden-group programs are fixed instances with exact gains for every set of families:

```bash
python -m llmdw.instances.export --out data/work/templates
```

The rich models are random draws of three classes (lot sizing, network design, facility location). A draw is accepted when it is feasible, its LP gap is at least two percent, and no single family closes more than half of the best gain among the sets of at most three families; the first eight accepted draws of each class are the instances, with exact gains for every set of at most three families:

```bash
python -m llmdw.instances.rich_models_build --out data/work/rich_models --parallel 8
```

The rich models with 12 to 16 families add standard modeling features on the same variables to the three classes: lot sizing and facility location with 12 and 16 families, network design with 12 and 14. The menu with *k* families is the families of the smaller model followed by the first new families in a fixed order (`llmdw.instances.rich_models_large.NEW_FAMILIES`). The data of draw *i* of a class and *k* come from `random.Random(7000 + 100 k + 10 c + i)`, with *c* the index of the class; a draw is accepted under the rule of the smaller rich models, an infeasible network-design draw is repaired by redrawing the new families of a minimal conflicting family set up to ten times, and the first ten accepted draws of each class and *k* are the 60 instances, with exact gains for every set of at most three families (at most two for network design with 14 families). The part `rich_models_large` of `scripts/rerun.sh` builds them (`python -m llmdw.instances.rich_models_large build` per class and *k*, then `finalize`) and computes their five-dual lookahead:

```bash
scripts/rerun.sh rich_models_large
```

The MIPLIB-NL models are exported by running each MIPLIB-NL script with the optimize call replaced by an MPS export that keeps row and column names; no model is solved, so the size-limited gurobipy license suffices. Families are the row names with digit runs removed, and the paper keeps the 108 models with two to six families:

```bash
git clone https://github.com/optsuite/MIPLIB-NL.git
git -C MIPLIB-NL checkout 9f14ad1782eaa547f7dad982d1edb186321ffd88
python -m llmdw.instances.miplibnl_export --dataset MIPLIB-NL/dataset --out data/work/mps --jobs 4 --timeout 1800
python -m llmdw.gcg.batch oracle    --mps-dir data/work/mps --out data/work/oracle --max-families 6 --time 120 --parallel 8
python -m llmdw.gcg.batch lookahead --mps-dir data/work/mps --oracle data/work/oracle --out data/work/lookahead --block-time 60 --parallel 8
python -m llmdw.scoring.structure --oracle-root data/work/oracle --lookahead-root data/work/lookahead --out data/work/structure
```

The benchmark instances use standard formulations of three classes on public data, with families named by their role:

- lot sizing: the 20 smallest usable instances of Trigeiro, Thomas and McClain (1989) by items times periods, with the families balance, setup linking and capacity with setup times;
- network design: the R instances of Crainic, Frangioni and Gendron (2001) with the families flow balance, arc linking and arc capacity, every instance with a feasible LP and a positive LP gap, up to the 20 smallest by number of variables;
- facility location: the OR-Library capacitated instances cap41 to cap134 (Beasley, 1990) with split demand and the families assignment, capacity and linking, the 19 instances with a positive LP gap.

The proved optima come from SCIP with one thread, seed 0 and a one-hour limit; the ground truth evaluates every set of at most two families with the GCG root-node settings of the census (no presolve, cuts or heuristics, SCIP propagation off, 120 seconds, one thread, seed 0); branch-and-price runs default GCG with one thread, seed 0 and 1800 seconds, once more with the next seed after a failure, and identical reformulations are run once and shared between methods:

```bash
scripts/fetch_benchmarks.sh
scripts/rerun.sh benchmark
scripts/rerun.sh time_optimal
```

`scripts/rerun.sh time_optimal` runs branch-and-price on every nonempty family set of the 59 instances, with GCG as shipped and SCIP alone, seeds 0, 1 and 2, one thread and 1800 seconds; a run that ends abnormally is repeated once with the same seed.

The instances without a joint group use standard formulations on OR-Library files, with families named by their role:

- generalized assignment: every instance of gap1 to gap12 (60) with the families Assign and Capacity; the files are maximization problems, written with negated objective coefficients and minimized;
- bin packing: the Kantorovich formulation of binpack1 with the families Assign (each item in exactly one bin), Capacity (the sizes in a bin at most the capacity if the bin is used) and SymmetryBreaking (y<sub>j</sub> ≥ y<sub>j+1</sub>), with as many bins as the first-fit-decreasing packing uses; the 16 of the 20 instances whose LP relaxation is feasible and whose LP gap against the SCIP reference of one hour is positive.

`scripts/rerun.sh no_joint_group` builds them, computes the SCIP reference, the gain of every set of at most *k* - 1 families and the lookahead, and runs branch-and-price on the sets of the methods and on GCG as shipped (seeds 0, 1 and 2, 1800 seconds), with the feasibility check of a solution better than the reference optimum:

```bash
scripts/fetch_benchmarks.sh
scripts/rerun.sh no_joint_group
```

`scripts/rerun.sh` also reruns the census, the five-dual lookahead, the large models, the two reference optima and the time of one evaluation on the templates and the rich models (`scripts/rerun.sh census|five_duals|large_models|reference_optima|eval_time`); these runs are optional and take days.

</details>

## Repository layout

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

## Citation

```bibtex
@misc{lian2026jointdw,
  title         = {Do Language Models Know Which Constraints Belong Together?},
  author        = {Lian, Junbo Jacob and Chen, Kaixuan and Qin, Hanzhang and Chen, Huiling and Zhang, Chaoyu and Teo, Chung-Piaw},
  year          = {2026},
  note          = {Preprint},
  eprint        = {2611.05726},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.05726}
}
```

## License

The code is released under the MIT License ([LICENSE](LICENSE)). The data are released under CC BY 4.0, except the parts derived from MIPLIB-NL (the model summaries, gain tables, lookahead values and census of the MIPLIB-NL models, and the MIPLIB-NL descriptions inside the prompts and records), which are released under CC BY-SA 4.0 as MIPLIB-NL is. When you use those parts, cite MIPLIB-NL (Li, Lu, Wei, Chen, Liu, Lan, Zhang and Wen, Constructing industrial-scale optimization modeling benchmark, arXiv:2602.10450, 2026) and MIPLIB 2017 (Gleixner et al., MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library, 2021, doi:10.1007/s12532-020-00194-3). [DATA.md](DATA.md) gives the license of each part.
