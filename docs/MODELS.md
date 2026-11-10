# Running the language models

Every answer that the paper scores is in the released answer files, and every request has an entry in the released response cache, so reproducing the paper sends no request. A new run needs a key and a base URL for each provider, read from the environment variables listed in [`.env.example`](../.env.example); the client reads the cache before it sends anything.

| Configuration | Provider | Endpoint identifier | Additional request fields | Notes |
|---|---|---|---|---|
| Claude Sonnet 5.5 | Anthropic | `claude-sonnet-5-5` | none | no request option set; the default thinking of the endpoint applies |
| GPT-6.1 Sol | OpenAI | `gpt-6.1-sol` | `{"reasoning_effort": "medium"}` | medium reasoning effort; the API returned the snapshot gpt-6.1-sol-2026-09-29 for the rich models and the prompt variants and gpt-6.1-sol otherwise |
| Grok 4.20 (Reasoning) | xAI | `grok-4-20-reasoning` | none | |
| Grok 4.20 (Non-Reasoning) | xAI | `grok-4-20-non-reasoning` | none | |
| DeepSeek-V4-Pro (Non-Thinking) | DeepSeek | `deepseek-v4-pro` | `{"thinking": {"type": "disabled"}}` | non-thinking mode |

[`config/public_settings.json`](../config/public_settings.json) holds these settings and the other client settings of each configuration (streaming, JSON mode, timeout, no temperature and no completion limit). `--config` takes the endpoint identifier and applies them.

Every request consists of one system message ([`prompts/decomp_v2.md`](../prompts/decomp_v2.md), section SYSTEM) and one user message ([`prompts/decomp_v2.md`](../prompts/decomp_v2.md), section USER) that ask for one JSON object with the per-family answers, the joint supports and up to three candidate reformulations. The four presentations are original (names, description and formulation), names only, permuted (the family names permuted among the families) and anonymized (F1, ..., Fk and V1, ..., Vm). The ranked variant ([`prompts/user_ranked.txt`](../prompts/user_ranked.txt)) also asks for up to five ranked groups with probabilities, and the data variant ([`prompts/user_with_data.txt`](../prompts/user_with_data.txt)) adds the complete instance in CPLEX LP format and is used on the rich models and on the benchmark instances. Each instance and presentation has three samples. The records contain:

| Records | Presentations | Configurations | Samples | Instances |
|---|---|---|---|---|
| templates and MIPLIB-NL | original, names only, permuted, anonymized | all five | three | 15 templates, 102 scored MIPLIB-NL models |
| rich models | original (the text) | all five | three | 24 rich models with 7 to 9 families |
| data variant | the complete instance in CPLEX LP format | all five | three | 24 rich models with 7 to 9 families |
| ranked variant | original, with ranked groups | all five | three | 13 templates, 16 MIPLIB-NL models, 24 rich models |
| control presentations with coefficients | data permuted, data anonymized | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 24 rich models with 7 to 9 families |
| benchmark instances | original | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 59 instances, giving eight distinct prompts: one for lot sizing, four for network design and three for facility location |
| benchmark instances with every coefficient | data variant (the complete instance in CPLEX LP format) | GPT-6.1 Sol, Claude Sonnet 5.5 | three | 59 instances, one prompt per instance; the prompt-token guard of this run and the set-aside GPT-6.1 Sol responses: [DATA.md, run setting of the benchmark data variant](../DATA.md#run-setting-of-the-benchmark-data-variant) |
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

The latency and cost records are read from the call records by `python -m llmdw.scoring.cost_accounting requests` and priced by `python -m llmdw.scoring.cost_accounting tables`, both run by `scripts/reproduce.sh`; [DATA.md](../DATA.md#prices) describes the price list and the rule that chose each price.

The data variant on the benchmark instances needs the built instances (`scripts/rerun.sh benchmark`, which builds `work/benchmark_std`) and the run settings recorded in [`DATA.md`](../DATA.md) (prompt-token guard factor 0.50 for GPT-6.1 Sol and 0.85 for Claude Sonnet 5.5, client timeout 600 s for Claude Sonnet 5.5; DATA.md also describes the exact-count check, which rejects a GPT-6.1 Sol response whose prompt tokens differ from the tiktoken `o200k_base` count of the request plus 42):

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
