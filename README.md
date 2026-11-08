<div align="center">

<h1>Do Language Models Know Which Constraints Belong Together?</h1>

<p>
<b>Junbo Jacob Lian</b><sup>1,2,3</sup> &nbsp;·&nbsp;
<b>Kaixuan Chen</b><sup>3</sup> &nbsp;·&nbsp;
<b>Hanzhang Qin</b><sup>1</sup> &nbsp;·&nbsp;
<b>Chaoyu Zhang</b><sup>4</sup> &nbsp;·&nbsp;
<b>Chung-Piaw Teo</b><sup>1</sup>
</p>

<p>
<sup>1</sup> Institute of Operations Research and Analytics, National University of Singapore<br>
<sup>2</sup> Wenzhou Buyi Pharmacy<br>
<sup>3</sup> McCormick School of Engineering, Northwestern University<br>
<sup>4</sup> Department of Decision Analytics and Operations, City University of Hong Kong
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

Dantzig-Wolfe reformulation strengthens the linear relaxation of a mixed-integer program by convexifying a chosen set of constraint families, and every candidate costs a column generation run. The choice is hard when families strengthen the bound only together, because a search that adds one family at a time then stops at the linear relaxation. We call such a set of families a zero-marginal group. This repository holds the code and data of the [paper](https://arxiv.org/abs/2611.05726): exact gain tables for small templates and for 24 rich models, the census of zero-marginal groups in MIPLIB-NL, 59 standard lot-sizing, network-design and facility-location instances with their gain tables and branch-and-price runs, the prompts and the recorded answers of five language-model configurations under the presentations of each model, the one-pass lookahead at an optimal LP dual and its five-dual version, a learned predictor trained on the gain tables, the search orders (JointDW, its interleaving with the lookahead, covering orders), and the scripts that build every table and figure of the paper from the released data without calling any API.

**Main results**

- Zero-marginal groups are common: three of ten MIPLIB-NL models whose bound can be improved have a certain one and another third a possible one, and on most of them no single family strengthens the bound at all. Every one of the 59 benchmark instances has one, and GCG's detector, which follows the block structure of the matrix, convexifies a set with no gain on 38 of them.
- When each evaluation may convexify at most *s* of *k* families, every method that observes only bounds needs a covering number of evaluations, of order *k*<sup>*w*</sup>, to find a hidden group of *w* families. A prediction that misses *e* members reduces the cost to Θ(*k*<sup>*e*</sup>) through one staged search that does not need to know *e*, and the search costs only a lower-order term more when the prediction is entirely wrong.
- The one-pass lookahead at an optimal LP dual is blind to groups whose families share no variable. JointDW takes the prediction of a language model, verifies it with the staged search, and interleaves it with the lookahead at no more than twice the cost of the better of the two.
- On MIPLIB-NL the strongest language models identify zero-marginal groups about as often as the lookahead and on partly different models. On textbook templates their predictions rest on the names: permuting or anonymizing the family names removes them. Reasoning roughly doubles the accuracy of the same model.
- On the rich models, which place the classical group among further modeling features, GPT-6.1 Sol given every coefficient names the classical group in all but one answer, also when the family names are permuted or replaced by labels, and then needs fewer evaluations than the lookahead for pairs of families. A predictor trained on the gain tables of the same class matches this only after hundreds of labeled evaluations and does not carry over to another class.
- On the benchmark instances the language models name the best pair more often than the lookahead and the detector, and the sets chosen by GPT-6.1 Sol come closest to the branch-and-price time of the best set.

<p align="center"><img src="docs/figures/cost.png" width="640" alt="Queries of the staged search and evaluations to find a hidden group"></p>

*Figure 2. (a) Queries of the staged search for k = 6, s = 3, w = 2 and the prediction P = {1, 2}: the queries of stage e contain every hidden group with e members outside P, so a correct prediction is confirmed by the first query. (b) Evaluations to find a hidden group of w = 3 families with s = 6, for predictions that miss e members and without a prediction, with the lower bounds as dashed lines.*

<p align="center"><img src="docs/figures/results.png" alt="Census of MIPLIB-NL and recall by presentation"></p>

*Figure 3. (a) Census of MIPLIB-NL: for each model with a positive gain, the largest gain of a single family against the largest gain of a set of families, both as shares of the LP gap; marks on the vertical axis have no family that strengthens the bound alone, and blue marks have a zero-marginal group at the empty set. (b, c) Recall of true joint supports under the four presentations on the templates and on MIPLIB-NL; the dashed line is the recall of the lookahead order.*

<p align="center"><img src="docs/figures/rich.png" width="640" alt="Rich models: hit@1 and evaluations"></p>

*Figure 4. Rich models. (a) hit@1 given the text (open circles) and every coefficient (filled circles), and, for two configurations, every coefficient with the names permuted (crosses) or replaced by labels (plus signs). (b) Mean evaluations until a best set is evaluated at s = 2 for lot sizing, network design and facility location: covering order (gray squares), lookahead order at the HiGHS dual (filled diamonds) and with five duals (open diamonds), JointDW with GPT-6.1 Sol given the text (open circles) and every coefficient (filled circles).*

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

The data are an asset of the release [v1.1.1](https://github.com/junbolian/JointDW/releases/tag/v1.1.1), which supersedes v1.1 and v1.0.

| Asset | Contents | Size | SHA-256 |
|---|---|---|---|
| `jointdw_data_v1.1.1.tar.gz` | templates and rich models with their exact gain tables; MIPLIB-NL gain tables with the invalid-bound column and the corrected reference optima; lookahead values at one and at five duals; every recorded request and response of the five configurations under the four presentations, the ranked variant, the data variant (on the rich models and on the benchmark instances, with the answers set aside for text added by an endpoint), the two control presentations and the benchmark instances; the rendered prompts; the learned-predictor features; the benchmark gain tables, ground truth and branch-and-price records and logs; the large-model evaluations; the results CSVs behind every table and figure | 87.1 MB | `50d407c37cb39e7e9cc92899084b405b36ead99ebd7b516c7473a7ca40319f0b` |
| `SHA256SUMS` | the checksum of the archive | | |

```bash
scripts/fetch_data.sh
```

downloads the data into `data/` and checks the SHA-256 recorded in the script. [DATA.md](DATA.md) describes every part of the data, how it was produced, the bound validity rule and its license. The MPS files of the MIPLIB-NL models are not part of the release; [Building instances](#building-instances) shows how to export them from MIPLIB-NL.

The benchmark instances are built from public benchmark files that are not redistributed here: `scripts/fetch_benchmarks.sh` downloads them from their original locations (the OR-Library, CommaLAB and suerie.de, with the test list of the GCG repository) into `data/benchmark_raw/` and verifies every data file against the SHA-256 recorded when the instances were built. The archive does not contain the 219 benchmark instance files built from them; `python -m llmdw.benchmark.standard build --raw data/benchmark_raw --out work/benchmark_std` recreates them, and `python -m llmdw.benchmark.standard check --out work/benchmark_std --recorded data/work/benchmark_std` verifies each against the SHA-256 recorded in its `meta.json` (219 of 219 identical on a fresh clone). [DATA.md](DATA.md) names the sources and their terms.

## Reproducing the paper

```bash
scripts/reproduce.sh
scripts/compare_results.sh
```

`scripts/reproduce.sh` reads only `data/`, sends no request to a language model and needs no solver. It runs `python -m llmdw.scoring.results` (the result tables, written to `results/`), the scoring of the control presentations, the five-dual lookahead and the benchmark instances (`python -m llmdw.scoring.experiments`, `python -m llmdw.benchmark.tables`), the learned predictor and the combined orders on the released features (`python -m llmdw.scoring.learned`, `python -m llmdw.scoring.combined`), and the table bodies and figure data of the paper (`python -m llmdw.tables.figure2`, `python -m llmdw.tables.paper`, `python -m llmdw.tables.experiments`, written to `results/paper/`). `scripts/compare_results.sh` compares every file with the released results in `data/reference_results/`. Every search cost is the mean over 1,000 random orders of the families with the recorded seed, every standard error comes from a bootstrap with 10,000 resamples and seed 0 that resamples prompt groups, and lookahead values within 1e-9 are tied; these constants are in [`llmdw/constants.py`](llmdw/constants.py). Table 4 needs no data (it follows from the definition of the staged search), and the instance files behind Table 7 are rebuilt by `scripts/fetch_benchmarks.sh` and `python -m llmdw.benchmark.standard build`, whose model files match the SHA-256 recorded in `data/work/benchmark_std/instances/*/meta.json`.

| Output | What it contains | Command | Result file |
|---|---|---|---|
| Figure 1 | gains of all family subsets of the lot-sizing template with two items and two periods | `scripts/reproduce.sh` | `results/paper/figure_1.csv` |
| Figure 2, Table 4 | (a) the queries of the staged search for *k* = 6, *s* = 3, *w* = 2; (b) queries of the staged search and the lower bounds for *w* = 3, *s* = 6, *k* = 8 to 24 | `scripts/reproduce.sh` | `results/paper/figure_2a.csv`, `results/figure2.csv` |
| Figure 3 | (a) MIPLIB-NL census; (b, c) recall under the four presentations on the templates and on MIPLIB-NL, with the lookahead recall | `scripts/reproduce.sh` | `results/paper/figure_3a_certain.csv`, `figure_3a_possible.csv`, `figure_3a_none.csv`, `figure_3b_templates.csv`, `figure_3c_miplib.csv` |
| Figure 4 | rich models: hit@1 given the text, given every coefficient, and under the two control presentations; evaluations at *s* = 2 by class for the covering order, the lookahead order at one and at five duals, and JointDW | `scripts/reproduce.sh` | `results/paper/figure_4a.csv`, `figure_4b.csv` |
| Table 1 | hit@1 and recall in the original presentation on the templates and MIPLIB-NL, false alarms on the assignment templates | `scripts/reproduce.sh` | `results/paper/table_01.tex` |
| Table 2 | mean evaluations until a best decomposition is evaluated, by order and *s* | `scripts/reproduce.sh` | `results/paper/table_02.tex` |
| Table 3 | benchmark instances: share of instances on which the first set of each method is the true best pair, by class, and the branch-and-price time ratio against the true best pair | `scripts/reproduce.sh` | `results/experiment_C/benchmark_methods.csv`, `benchmark_paired_true_best.csv` |
| Table 5 | exact gains of the templates | `scripts/reproduce.sh` | `results/templates_table.csv` |
| Table 6 | rich models: families, size, LP gap, minimal best sets, zero-marginal groups | `scripts/reproduce.sh` | `results/rich_models_instances.csv`, `results/rich_models_groups.csv` |
| Table 7 | benchmark instances: counts, sizes, LP gaps, best pairs | `scripts/reproduce.sh` | `results/experiment_C/benchmark_instances.csv`, `results/paper/experiments/experiment_C_instances.tex` |
| Table 8 | recall by presentation, mean ± standard error | `scripts/reproduce.sh` | `results/paper/table_06.tex` |
| Table 9 | paired differences in recall between presentations | `scripts/reproduce.sh` | `results/paper/table_07.tex` |
| Table 10 | Grok 4.20 (Reasoning) minus Grok 4.20 (Non-Reasoning), paired over instances | `scripts/reproduce.sh` | `results/paper/table_08.tex` |
| Table 11 | evaluations by size limit, mean ± standard error | `scripts/reproduce.sh` | `results/paper/table_09.tex` |
| Table 12 | instances on which a configuration and the lookahead order name a true joint support first | `scripts/reproduce.sh` | `results/paper/table_10.tex` |
| Table 13 | five-dual lookahead: first set and search cost, and JointDW interleaved with it | `scripts/reproduce.sh` | `results/paper/experiments/experiment_B_first_set.tex`, `experiment_B_search.tex`; `results/experiment_B/` |
| Table 14 | rich models: hit@1 and recall given the text and given every coefficient | `scripts/reproduce.sh` | `results/paper/table_11.tex` |
| Table 15 | rich models: evaluations until a best set is evaluated | `scripts/reproduce.sh` | `results/paper/table_12.tex` |
| Table 16 | ranked variant: coverage@1, coverage@3, and the ranked order interleaved with the lookahead minus the lookahead | `scripts/reproduce.sh` | `results/paper/table_13.tex` |
| Table 17 | control presentations with every coefficient: hit@1, recall and search cost with the names permuted or replaced by labels | `scripts/reproduce.sh` | `results/paper/experiments/experiment_A_prediction.tex`, `experiment_A_search.tex`; `results/experiment_A/` |
| Table 18 | rich models with every coefficient: JointDW and its interleaving with the five-dual lookahead minus the five-dual lookahead, *s* = 2 and *s* = 3, pooled and by class, data presentation and the two control presentations | `scripts/reproduce.sh` | `results/paper/experiments/rich_models_data_vs_five_dual.tex`; `results/rich_models_data_paired_five_dual.csv` |
| Table 19 | learned predictor: hit@1, search cost, labeling cost, by split and variant | `scripts/reproduce.sh` | `results/paper/experiments/learned_predictor.tex`; `results/learned/` |
| Table 20 | learned predictor against the number of training instances of the same class | `scripts/reproduce.sh` | `results/paper/experiments/learned_predictor_training_size.tex` |
| Table 21 | benchmark instances, per class and method: hit@1, gain of the first set, branch-and-price time, final gap, instances solved | `scripts/reproduce.sh` | `results/paper/experiments/experiment_C_methods.tex`; `results/experiment_C/benchmark_methods.csv` |
| Table 22 | benchmark instances: paired time differences and time ratios against the true best pair | `scripts/reproduce.sh` | `results/paper/experiments/experiment_C_paired_time.tex`, `experiment_C_against_true_best.tex` |
| Table 23 | benchmark instances: paired differences in hit@1 and in the gain of the first set, JointDW under the original presentation minus each baseline, and the five-dual lookahead minus GCG's detector | `scripts/reproduce.sh` | `results/paper/experiments/experiment_C_paired_hit1.tex`; `results/experiment_C/benchmark_paired_hit1.csv`, `benchmark_paired_gain.csv` |
| Table 24 | benchmark instances with every coefficient shown: hit@1, gain, time, final gap, solved, time ratio and samples without a joint support of the two configurations, and their paired differences against the baselines and against the original presentation | `scripts/reproduce.sh` | `results/paper/experiments/experiment_C_data_methods.tex`, `experiment_C_data_paired.tex`; `results/experiment_C/benchmark_data_methods.csv`, `benchmark_data_paired.csv`, `benchmark_data_predictions.csv` |
| Table 25 | paired differences in solve time on MIPLIB-NL against the covering and the lookahead orders | `scripts/reproduce.sh` | `results/paper/table_14.tex` |
| Table 26 | paired differences in evaluations against the covering and the lookahead orders | `scripts/reproduce.sh` | `results/paper/table_15.tex` |
| Combined orders (Appendix F) | union-first and stacked orders: search cost against the five-dual lookahead order, JointDW interleaved and the learned order | `scripts/reproduce.sh` | `results/paper/experiments/combined_union_first.tex`, `combined_stacked.tex`; `results/combined/` |

`tests/test_paper_tables.py` rebuilds every table body from the released results and from `results/` and compares it with the reference bodies in `tests/reference/`. On a fresh clone of this repository (one workstation, Python 3.13), the installation took 15 s, `scripts/fetch_data.sh` 4 s, `scripts/fetch_benchmarks.sh` 24 s, rebuilding and checking the 219 benchmark instance files 63 s, the tests 59 s and `scripts/reproduce.sh` 54 minutes (the learned predictor 18 minutes and the combined orders 29 minutes of it); `scripts/compare_results.sh` then found `results/` identical to `data/reference_results/` (167 files).

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
| rich models | original (the text) | all five | three | 24 rich models |
| data variant | the complete instance in CPLEX LP format | all five | three | 24 rich models |
| ranked variant | original, with ranked groups | all five | three | 13 templates, 16 MIPLIB-NL models, 24 rich models |
| control presentations with coefficients | data permuted, data anonymized | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 24 rich models |
| benchmark instances | original | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 59 instances, giving eight distinct prompts: one for lot sizing, four for network design and three for facility location |
| benchmark instances with every coefficient | data variant (the complete instance in CPLEX LP format) | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 59 instances, one prompt per instance; the prompt-token guard of this run and the set-aside GPT-6.1 Sol responses: [DATA.md, run setting of the benchmark data variant](DATA.md#run-setting-of-the-benchmark-data-variant) |

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
<summary>Templates, hidden-group programs, rich models, MIPLIB-NL models and benchmark instances</summary>

The templates (lot sizing, network design, facility location, generalized assignment) and the hidden-group programs are fixed instances with exact gains for every set of families:

```bash
python -m llmdw.instances.export --out data/work/templates
```

The rich models are random draws of three classes (lot sizing, network design, facility location). A draw is accepted when it is feasible, its LP gap is at least two percent, and no single family closes more than half of the best gain among the sets of at most three families; the first eight accepted draws of each class are the instances, with exact gains for every set of at most three families:

```bash
python -m llmdw.instances.rich_models_build --out data/work/rich_models --parallel 8
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
```

`scripts/rerun.sh` also reruns the census, the five-dual lookahead, the large models and the two reference optima (`scripts/rerun.sh census|five_duals|large_models|reference_optima`); these runs are optional and take days.

</details>

## Repository layout

```
llmdw/instances/   templates, rich-model generators, MIPLIB-NL export, family detection, model summaries, reference optima
llmdw/exact/       exact gain tables (HiGHS with rational verification in python-flint)
llmdw/lookahead/   one-pass lookahead at an optimal LP dual, and at five optimal duals
llmdw/search/      staged search, covering orders, interleaving, relabeling-invariant search cost
llmdw/prompts/     presentations, per-instance labels, prompt rendering, answer parsing, name rule, re-reading of answers
llmdw/providers/   OpenAI-compatible client, cache, request records, configurations, replay check
llmdw/scoring/     hit@1, recall, coverage@m, cluster bootstrap, census, bound validity rule, all result tables,
                   control presentations, five-dual lookahead, learned predictor, combined orders, large models
llmdw/benchmark/   benchmark sources, data checks, standard formulations, ground truth, branch-and-price, tables
llmdw/tables/      table bodies and figure data of the paper (official names)
llmdw/gcg/         GCG decompositions and bound evaluations, memory caps and the shared memory budget
scripts/           fetch_data.sh, fetch_benchmarks.sh, reproduce.sh, compare_results.sh, rerun.sh
config/            public_settings.json (request settings), benchmark_sources.json (URLs and SHA-256 of the
                   benchmark files), high_memory_models.txt (models solved with a 10 GB cap)
prompts/           system message, user template, ranked and data variants
tests/             offline tests; tests/reference holds the table bodies of the sample of record
docs/figures/      the figures of this README
```

## Citation

```bibtex
@misc{lian2026jointdw,
  title         = {Do Language Models Know Which Constraints Belong Together?},
  author        = {Lian, Junbo Jacob and Chen, Kaixuan and Qin, Hanzhang and Zhang, Chaoyu and Teo, Chung-Piaw},
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
