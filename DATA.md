# Data

The release asset `jointdw_data.tar.gz` unpacks to `data/`. It is the sample of record of the paper, dated 2026-11-05: every answer was produced between 30 October and 2 November 2026, and every number of the paper comes from these files. `scripts/fetch_data.sh` downloads and checks it; `scripts/reproduce.sh` builds every table and figure from it.

## Contents

| Path in `data/` | What it is | How it was produced | License |
|---|---|---|---|
| `work/templates/<instance>/` | the 15 templates: `model.mps`, the description (`instance.json`), one sentence per family (`model.md`) and `ground_truth.json`, the exact gain of every family set with its zero-marginal groups | `python -m llmdw.instances.export`, gains in rational arithmetic (`llmdw.exact`) | CC BY 4.0 |
| `work/lookahead_templates/` | the one-pass lookahead of every family set of each template | `llmdw.lookahead` at the HiGHS dual | CC BY 4.0 |
| `work/rich_models/` | the 24 rich models: `instances.txt`, `draws.json` (every drawn candidate and the reason of a rejection) and per instance the model, its text, `ground_truth.json` (exact gains of every set of at most three families), `lookahead.json` and `gcg_anchors.json` | `python -m llmdw.instances.rich_models_build --parallel 8`, seed 0 | CC BY 4.0 |
| `work/oracle/`, `work/miplib_set/` | the gain tables of the MIPLIB-NL models: per evaluated set the dual bound, the status (convergence) and the solve time; `miplib_set` holds the 102 scored models | `python -m llmdw.gcg.batch oracle --max-families 6 --time 120` (GCG on SCIP 10, root node, no presolve, cuts or heuristics, one thread, seed 0) | CC BY-SA 4.0 |
| `work/lookahead/` | the lookahead values of the MIPLIB-NL models | `python -m llmdw.gcg.batch lookahead --block-time 60` | CC BY-SA 4.0 |
| `work/structure/structure_rows.json` | the census of zero-marginal groups | `python -m llmdw.scoring.structure` | CC BY-SA 4.0 |
| `work/large_models/` | the 30 MIPLIB-NL models with 7 to 26 families: LP bound, lookahead, prompt, the order of every method and every evaluated decomposition | `python -m llmdw.scoring.large_models prepare` and `run` | CC BY-SA 4.0 |
| `work/model_summaries/<phase>/` | per model the facts that the tables read: objective sense, integrality of the objective, the family summary of the prompt, the order of the families in the model file and the columns of each family | `python -m llmdw.instances.summaries` | CC BY 4.0 for templates and rich models, CC BY-SA 4.0 for MIPLIB-NL |
| `work/mps/export_summary.csv` | the export status of every MIPLIB-NL model | `python -m llmdw.instances.miplibnl_export` | CC BY-SA 4.0 |
| `work/support_templates/`, `work/support_miplib/` | answers to the original prompt under the four presentations | `python -m llmdw.providers.support_run`, three samples per instance and presentation | CC BY 4.0; MIPLIB-NL text inside them CC BY-SA 4.0 |
| `work/support_rich_models/` | answers to the original prompt on the rich models (original presentation) | as above | CC BY 4.0 |
| `work/rich_models_with_data/` | answers to the data variant (the complete instance in CPLEX LP format) | `--prompt with_data` | CC BY 4.0 |
| `work/support_ranked/`, `work/ranked_set/` | answers to the ranked variant, and the instances it was sent for | `--prompt ranked` | CC BY 4.0; MIPLIB-NL text inside them CC BY-SA 4.0 |
| `prompts/<prompt>/<instance>/<presentation>.json` | the rendered messages of every request | taken from the answer files | as the answers |
| `llm_cache/<SHA-256>.json` | the response to every request, with the answer text as returned | `llmdw.providers.llm_client` | as the answers |
| `runs/llm_calls.jsonl`, `runs/llm_retries.jsonl` | one line per call (time, tag, request SHA-256, requested and returned model, usage, latency, finish reason, cache flag) and one line per retried attempt | `llmdw.providers.llm_client` | CC BY 4.0 |
| `prompt_groups.csv` | per instance its prompt group: instances whose requests are byte-identical, which the bootstrap draws together | `llmdw.scoring.cluster` | CC BY 4.0 |
| `census_corrections.json` | corrections of the census figure data (reference optima and invalid bounds), each with its reason | read by `python -m llmdw.scoring.results --census-corrections` | CC BY-SA 4.0 |
| `reference_results/` | the result tables of the sample of record, and in `paper/` the table bodies and figure data of the paper | `scripts/reproduce.sh` | CC BY 4.0 |

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

## MIPLIB-NL

The MIPLIB-NL models are not released as MPS files, because MIPLIB 2017 states no license for redistributing its instances. They can be exported from MIPLIB-NL (https://github.com/optsuite/MIPLIB-NL, commit 9f14ad1782eaa547f7dad982d1edb186321ffd88) with `python -m llmdw.instances.miplibnl_export`, as the README describes; the tables need only the released summaries.

## Licenses

The data are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), except the parts marked CC BY-SA 4.0 above, which are derived from MIPLIB-NL and are released under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), as MIPLIB-NL is. The archive contains the same statement in `NOTICE`. When you use those parts, cite MIPLIB-NL and MIPLIB 2017:

- Li, Z., Lu, H., Wei, T., Chen, Y., Liu, W., Lan, Y., Zhang, F., and Wen, Z. Constructing industrial-scale optimization modeling benchmark. arXiv:2602.10450, 2026.
- Gleixner, A., Hendel, G., Gamrath, G., Achterberg, T., Bastubbe, M., Berthold, T., Christophel, P., Jarck, K., Koch, T., Linderoth, J., Luebbecke, M., Mittelmann, H. D., Ozyurt, D., Ralphs, T. K., Salvagnin, D., and Shinano, Y. MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library. Mathematical Programming Computation 13, 443-490, 2021. doi:10.1007/s12532-020-00194-3.
