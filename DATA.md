# Data

The release asset `jointdw_data_v1.1.tar.gz` unpacks to `data/`. It is the sample of record of the revised paper, dated 2026-11-07 (the bound validity rule applied): the census and every gain evaluation without SCIP propagation, the answer name rule applied to every recorded answer, the five-dual lookahead, the control presentations (Experiment A), the five-dual lookahead (Experiment B), the standard formulations on benchmark data (Experiment C), the learned structural predictor and the combined orders. Every number of the revised paper comes from these files. `scripts/fetch_data.sh` downloads the archive and checks its SHA-256 (recorded in the script and in the release asset `SHA256SUMS`); `scripts/reproduce.sh` builds every table and figure from it without a solver or a language-model call, and `scripts/compare_results.sh` compares the result with `data/reference_results/`.

## Contents

| Path in `data/` | What it is | How it was produced | License |
|---|---|---|---|
| `work/templates/<instance>/` | the 15 templates: `model.mps`, the description (`instance.json`), one sentence per family (`model.md`) and `ground_truth.json`, the exact gain of every family set with its zero-marginal groups | `python -m llmdw.instances.export`, gains in rational arithmetic (`llmdw.exact`) | CC BY 4.0 |
| `work/lookahead_templates/` | the one-pass lookahead of every family set of each template | `llmdw.lookahead` at the HiGHS dual | CC BY 4.0 |
| `work/rich_models/` | the 24 rich models: `instances.txt`, `draws.json` (every drawn candidate and the reason of a rejection) and per instance the model, its text, `ground_truth.json` (exact gains of every set of at most three families), `lookahead.json` and `gcg_anchors.json` | `python -m llmdw.instances.rich_models_build --parallel 8`, seed 0 | CC BY 4.0 |
| `work/oracle/`, `work/miplib_set/` | the gain tables of the MIPLIB-NL models: per evaluated set the dual bound, the status (convergence flag), the solve time, the settings and every attempt of a failed evaluation; `miplib_set` holds the 102 scored models | `python -m llmdw.gcg.batch oracle --max-families 6 --time 120` (GCG on SCIP 10, root node, no presolve, cuts or heuristics, SCIP propagation off, one thread, seed 0; a failed evaluation is retried twice, then once with seed 1; the LP bound of graph40-40-1rand and graph40-80-1rand from the HiGHS interior-point method with crossover) | CC BY-SA 4.0 |
| `work/reference_optima/` | the SCIP runs that replaced a reference optimum contradicted by a recorded primal solution (b-ball, radiationm18-12-05): status, bounds and time; the values are in `census_corrections.json` | SCIP, one thread, seed 0, 3600 s | CC BY-SA 4.0 |
| `work/lookahead/` | the lookahead values of the MIPLIB-NL models at the HiGHS dual (one dual) | `python -m llmdw.gcg.batch lookahead --block-time 60` | CC BY-SA 4.0 |
| `work/five_duals/<set>/<instance>/` | the five-dual lookahead of the templates, the scored MIPLIB-NL models and the rich models: `five_duals.json` (per family set the value at each of the five duals and their maximum) and `dual1.json` to `dual4.json` (the lookahead at each further optimal dual, with the dual vector in `.y.gz`) | `python -m llmdw.lookahead.five_duals batch` (block time 60 s, instance limit 3600 s, dual seeds 1 to 4) | CC BY-SA 4.0 for MIPLIB-NL, CC BY 4.0 otherwise |
| `work/structure/structure_rows.json` | the census of zero-marginal groups | `python -m llmdw.scoring.structure` | CC BY-SA 4.0 |
| `work/large_models/` | the 30 MIPLIB-NL models with 7 to 26 families: LP bound, lookahead, prompt, the order of every method and every evaluated decomposition | `python -m llmdw.scoring.large_models prepare` and `run` | CC BY-SA 4.0 |
| `work/model_summaries/<phase>/` | per model the facts that the tables read: objective sense, integrality of the objective, the family summary of the prompt, the order of the families in the model file and the columns of each family | `python -m llmdw.instances.summaries` | CC BY 4.0 for templates and rich models, CC BY-SA 4.0 for MIPLIB-NL |
| `work/mps/export_summary.csv` | the export status of every MIPLIB-NL model | `python -m llmdw.instances.miplibnl_export` | CC BY-SA 4.0 |
| `work/support_templates/`, `work/support_miplib/` | answers to the original prompt under the four presentations | `python -m llmdw.providers.support_run`, three samples per instance and presentation | CC BY 4.0; MIPLIB-NL text inside them CC BY-SA 4.0 |
| `work/support_rich_models/` | answers to the original prompt on the rich models (original presentation) | as above | CC BY 4.0 |
| `work/rich_models_with_data/` | answers to the data variant (the complete instance in CPLEX LP format) | `--prompt with_data` | CC BY 4.0 |
| `work/controls/` | Experiment A: answers to the data variant under the permuted and the anonymized presentation with per-instance labels (`answers/<endpoint identifier>/<instance>/`, with `labels.json`), their cache entries and call records | `--prompt with_data --conditions permuted,anonymized --labels per-instance` | CC BY 4.0 |
| `work/benchmark_std/` | Experiment C: `selection.json` and the instance lists (`full_set/`, `first_batch/`); per instance (`instances/<name>/`) `meta.json` (source file, size and the SHA-256 of the model file), `reference.json` (reference optimum), `lp.json`, `oracle.json` and `oracle/` (the ground truth: every evaluated set with its record, decomposition and solver log), `classification.json`, `lookahead.json`, `five_duals.json` and `dual1.json` to `dual4.json`, `bnp_plan.json` (the set of each method) and `bnp/` (the branch-and-price run of each set: record, decomposition and solver log); `answers/`, `llm_cache/` and `runs/` of the language models | `python -m llmdw.benchmark.standard`, `llmdw.benchmark.truth`, `llmdw.benchmark.bnp`, `llmdw.providers.support_run` | CC BY 4.0 |
| `work/features/s_max_<s>/<set>/<instance>.json` | the features of the learned structural predictor of every family set of at most s families (s = 3 for the predictor, 99 for every set in the combined orders) | `python -m llmdw.scoring.learned --features DIR --write-features` | CC BY-SA 4.0 for MIPLIB-NL, CC BY 4.0 otherwise |
| `work/support_ranked/`, `work/ranked_set/` | answers to the ranked variant, and the instances it was sent for | `--prompt ranked` | CC BY 4.0; MIPLIB-NL text inside them CC BY-SA 4.0 |
| `prompts/<prompt>/<instance>/<presentation>.json` | the rendered messages of every request | taken from the answer files | as the answers |
| `llm_cache/<SHA-256>.json` | the response to every request, with the answer text as returned | `llmdw.providers.llm_client` | as the answers |
| `runs/llm_calls.jsonl`, `runs/llm_retries.jsonl` | one line per call (time, tag, request SHA-256, requested and returned model, usage, latency, finish reason, cache flag) and one line per retried attempt | `llmdw.providers.llm_client` | CC BY 4.0 |
| `prompt_groups.csv` | per instance its prompt group: instances whose requests are byte-identical, which the bootstrap draws together | `llmdw.scoring.cluster` | CC BY 4.0 |
| `census_corrections.json` | corrections of the census figure data (reference optima and invalid bounds), each with its reason | read by `python -m llmdw.scoring.results --census-corrections` | CC BY-SA 4.0 |
| `reference_results/census_evaluations.csv`, `reference_results/large_models_invalid_bounds.csv` | every recorded census evaluation with its outcome and the rule that invalidates its bound (empty when valid), and the invalid bounds of the large models | `llmdw.scoring.bounds`, through `llmdw.scoring.results` | CC BY-SA 4.0 |
| `reference_results/` | the result tables of the sample of record, in `paper/` the table bodies and figure data of the paper, and in `experiment_A/`, `experiment_B/`, `experiment_C/`, `learned/` and `combined/` the tables of the experiments | `scripts/reproduce.sh` | CC BY 4.0 |

Answer files are `<folder>/<endpoint identifier>/<instance>/<presentation>.json` for the first sample and `<presentation>.s<i>.json` for sample i. Each holds the messages, the parsed answer, its validation, the usage and `content_sha256`, the SHA-256 of the answer text. The rich-model instances are named `e6c_<class>_<index>`.

## Configurations

| Endpoint identifier | Official name | Provider |
|---|---|---|
| `claude-sonnet-5-5` | Claude Sonnet 5.5 | Anthropic |
| `gpt-6.1-sol` | GPT-6.1 Sol | OpenAI |
| `grok-4-20-reasoning` | Grok 4.20 (Reasoning) | xAI |
| `grok-4-20-non-reasoning` | Grok 4.20 (Non-Reasoning) | xAI |
| `deepseek-v4-pro` | DeepSeek-V4-Pro (Non-Thinking) | DeepSeek |

The request settings are in `config/public_settings.json` of the repository. Every recorded answer of Claude Sonnet 5.5, Grok 4.20 (Reasoning) and Grok 4.20 (Non-Reasoning) was requested without additional fields, every answer of GPT-6.1 Sol with `{"reasoning_effort": "medium"}`, and every answer of DeepSeek-V4-Pro with `{"thinking": {"type": "disabled"}}`.

## Records

Request and response records are kept as they were sent and received, with one change: each usage object keeps only the standard token fields (prompt, completion and total tokens, cached prompt tokens and reasoning tokens), and the other fields that the endpoints added, among them billing fields, are removed. The records contain no keys and no endpoint addresses. The responses of GPT-6.1 Sol on the rich models, in the data variant and in the ranked variant carry the snapshot identifier returned by the API, `gpt-6.1-sol-2026-09-29`; its responses on the templates and MIPLIB-NL name `gpt-6.1-sol`.

Every request has a cache entry. For 463 answers the request is byte-identical to the request of another instance, the cache holds one response for both, and each answer file holds the parsed answer of its own instance.

## Bound validity

The recorded evaluations are kept as they were returned. Every table reads them through one rule (`llmdw.scoring.bounds`): a recorded dual bound counts as no bound when it is not finite or at least 1e19 in absolute value, when it is beyond the reference optimum, or when it is beyond the best primal solution value recorded by SCIP in any evaluation of the same model, each by more than the tolerance max(1e-4 x |optimum - LP bound|, 1e-4 x max(1, |optimum|)) (without a reference optimum, 1e-4 x max(1, |primal value|)). The invalid bounds are flagged in `reference_results/census_evaluations.csv` and `reference_results/large_models_invalid_bounds.csv`. A reference optimum that a recorded primal solution contradicts by more than the same tolerance was replaced by the optimum that SCIP proves (one thread, seed 0, 3600 s; `census_corrections.json`, `work/reference_optima/`): b-ball, a maximization, from 1.5 to 3.0, and radiationm18-12-05, a minimization, from 17574 to 17566. No recorded bound lies beyond either new reference.

Under the rule the greedy forward selection of two large models takes another path. On sct31 it stops after the nine single families; on sct32 it needs one set that had not been evaluated, which was evaluated once in the census setting (GCG root node, SCIP propagation off, 120 s, one thread, seed 0, 10 GB): `work/large_models/sct32/evals/`, the convexified families cap[#] and pbudget, status time limit, bound -62.99083349240084. This is the only evaluation added to the records of the earlier release.

## Benchmark data of Experiment C

The instances of Experiment C are built from public benchmark files, which this archive does not contain: the pages of the sources state no license that permits redistribution, so the files are fetched from their original locations, and the model files built from them are not in the archive either (the gain tables, ground truth, branch-and-price records and answers are). `scripts/fetch_benchmarks.sh` downloads them from their original locations into `data/benchmark_raw/` and checks every file against the SHA-256 recorded when the instances were built (`config/benchmark_sources.json`); `python -m llmdw.benchmark.standard build` then rebuilds the model files, whose SHA-256 is recorded in each `meta.json`.

| Source | Files | Location |
|---|---|---|
| OR-Library, capacitated warehouse location (J. E. Beasley) | `cap41` to `cap134` and `capinfo.html` | https://people.brunel.ac.uk/~mastjjb/jeb/orlib/ |
| CommaLAB, multicommodity network design, Canad R instances (T. G. Crainic, A. Frangioni, B. Gendron) | `R.tgz` and the data page | https://commalab.di.unipi.it/datasets/mmcf/ |
| Capacitated lot sizing, test set of Trigeiro, Thomas and McClain, as distributed at suerie.de | `data.tar.bz2`, `conv_ttm_new.txt` | https://suerie.de/testsets/clspl/ |
| GCG repository, test list and proved optima of the Trigeiro instances (one fixed commit) | `ls-trigeiro.test`, `ls-trigeiro.solu` | https://github.com/scipopt/gcg |
| LOTSIZELIB, AMPL copies of four Trigeiro instances, used only to check the suerie.de data value by value | `tr6_15`, `tr6_30`, `tr12_30`, `tr24_30` and `trigeiro.mod` | https://www.dcc.fc.up.pt/~jpp/lsize/ |

Every data file is checked against its recorded SHA-256; the two web pages (the OR-Library information page and the CommaLAB data page) are kept for provenance and not checked, because their content may change. When you use the instances, cite their sources: Beasley, J. E. OR-Library: distributing test problems by electronic mail. Journal of the Operational Research Society 41, 1069-1072, 1990; Crainic, T. G., Frangioni, A., and Gendron, B. Bundle-based relaxation methods for multicommodity capacitated fixed charge network design. Discrete Applied Mathematics 112, 73-99, 2001; Trigeiro, W. W., Thomas, L. J., and McClain, J. O. Capacitated lot sizing with setup times. Management Science 35, 353-366, 1989.

## MIPLIB-NL

The MIPLIB-NL models are not released as MPS files, because MIPLIB 2017 states no license for redistributing its instances. They can be exported from MIPLIB-NL (https://github.com/optsuite/MIPLIB-NL, commit 9f14ad1782eaa547f7dad982d1edb186321ffd88) with `python -m llmdw.instances.miplibnl_export`, as the README describes; the tables need only the released summaries and features.

## Licenses

The data are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), except the parts marked CC BY-SA 4.0 above, which are derived from MIPLIB-NL and are released under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), as MIPLIB-NL is. The archive contains the same statement in `NOTICE`. When you use those parts, cite MIPLIB-NL and MIPLIB 2017:

- Li, Z., Lu, H., Wei, T., Chen, Y., Liu, W., Lan, Y., Zhang, F., and Wen, Z. Constructing industrial-scale optimization modeling benchmark. arXiv:2602.10450, 2026.
- Gleixner, A., Hendel, G., Gamrath, G., Achterberg, T., Bastubbe, M., Berthold, T., Christophel, P., Jarck, K., Koch, T., Linderoth, J., Luebbecke, M., Mittelmann, H. D., Ozyurt, D., Ralphs, T. K., Salvagnin, D., and Shinano, Y. MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library. Mathematical Programming Computation 13, 443-490, 2021. doi:10.1007/s12532-020-00194-3.
