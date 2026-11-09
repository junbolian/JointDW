"""Cost of the language-model predictions against the solver time they save.

No language model and no solver is called; everything is read from the call logs, the price sources, the answer files,
the gain tables and the solver records.

Requests (requests)
    One row per request sent to a language model (a call-log line with cached = false) in the call logs of the sample
    of record (llmdw.providers.llm_client writes one line per call: time, tag, request_sha256, model_requested,
    model_returned, usage, latency, finish_reason, cached, and error for an attempt that was rejected and sent again):
    the released logs of the templates, MIPLIB-NL and the rich models with 7 to 9 families (runs/llm_calls.jsonl), of
    the controls (work/controls/runs/llm_calls.jsonl) and of the benchmark instances (work/benchmark_std/runs/
    llm_calls.jsonl and llm_calls_data.jsonl), and the logs of the rich models with 12 to 16 families
    (work/rich_models_large_requests/runs/<config>/llm_calls.jsonl) and of the classes without a joint group
    (work/no_joint_group/runs/<config>/llm_calls.jsonl). A line that appears in two logs (same time, tag, request_sha256,
    latency, cached and model) is counted once. Every sent request is a paid request and is costed, including the
    attempts rejected by the client (error "incomplete: ..." or "prompt_inflation") and requests sent twice. Lines with
    cached = true are cache hits (no request; counted separately) and lines with cached = null record a cache entry
    set aside (no request).
    Times: the log time is the end of the request in the time zone of the machine (America/Chicago); the start of the
    request is the log time minus the latency, and its America/Chicago calendar date is the day of the request.
    Tokens: prompt tokens (usage.prompt_tokens), cached prompt tokens (usage.prompt_tokens_details.cached_tokens),
    cache-write tokens (the cache-creation tokens of Claude Sonnet 5.5, 5-minute and 1-hour writes, where recorded:
    usage.cache_write_5m_tokens and usage.cache_write_1h_tokens of the released records), completion tokens (usage.completion_tokens; they include the reasoning tokens, which are
    also listed: usage.completion_tokens_details.reasoning_tokens) and the latency in seconds.
    Cost (USD) = [(prompt - cached - cache writes) x input price + cached x cached-input price + 5-minute writes x
    5-minute write price + 1-hour writes x 1-hour write price + completion x output price] / 10^6, prices per million
    tokens; the cached-input price applies only to the cached prompt tokens recorded in the usage. DeepSeek prices by
    the time of day (peak and off-peak rates, from the page); the rate is chosen by the start of the request in UTC.
    Outputs: requests.csv, requests_summary.csv (per experiment and model, and per model: requests, rejected attempts,
    repeated sends of a request already answered, cache hits, tokens, latency median and mean, cost).

Prices (prices)
    sources.json lists every price source: the pricing page of each provider as archived by the Internet Archive
    (Wayback Machine snapshots; Claude Sonnet 5.5 platform.claude.com/docs/en/about-claude/pricing, OpenAI
    developers.openai.com/api/docs/pricing, DeepSeek api-docs.deepseek.com/quick_start/pricing, xAI
    docs.x.ai/developers/models) and the pages captured on the days of the requests (2026-11-08, 2026-11-09), each
    with its date (America/Chicago), the file and its SHA-256, and the prices read from the page by hand. For every
    model and day with a request the price of the source of that day is used (the last one of the day); else the
    nearest source before the day; else the nearest after; else the capture of 2026-11-08 (price_rule). Writes
    prices.csv (model, day, rule, source, URL, snapshot, prices).

Net benefit (tables)
    Collections: templates, MIPLIB-NL, rich models with 7 to 9 families, rich models with 12 to 16 families, and the
    benchmark instances of the time-optimal decompositions (values_bench, from the stored gain tables, best-of-five lookahead values and
    anchors of the plans of Experiment C; with --bench-root, else rows with the time of one evaluation and the
    latency and cost of one prediction only). On the benchmark instances each row also has the net time saved with
    the measured mean time of one evaluation of each instance (evaluations saved x that time - latency, per instance)
    and the share of instances where it is positive.
    Search costs per instance (evaluations until a best set is evaluated, the mean over the 1,000 family orders of
    llmdw.search.relabel) by the functions of the published tables: templates and MIPLIB-NL as
    llmdw.scoring.experiments.five_duals_tables (the instances with a true joint support and a five-dual result; s
    capped at k - 1; per instance, JointDW is the mean over its three samples, a failure when one sample fails), with
    the covering order (results.search_runs, spec ("covering",)) added; the rich models with 7 to 9 and with 12 to 16
    families as llmdw.scoring.rich_models_large.score_set (per instance, JointDW is the mean over the samples with a value;
    s = 3 is not computed for network design with k = 14). Orders: JointDW (spec ("model", P), P the first predicted
    joint support of the answer, else the families convexified by its top candidate) and JointDW interleaved with the
    five-dual order (spec ("interleaved", P) on the lookahead values at the best of five optimal duals); baselines: the
    covering order and the five-dual lookahead order. Evaluations saved = baseline minus order (positive when the order
    needs fewer evaluations), per instance.
    Time of one evaluation (eval_time.csv; one definition for every collection, that of the census,
    llmdw.scoring.results.oracle_seconds: wall_seconds of the solver child, else GCG's solving_time, a missing value at
    the limit, capped at 120 s): templates and rich models with 7 to 9 families from llmdw.benchmark.eval_time (GCG root
    evaluation in the census setting of the sets that the search orders visit); MIPLIB-NL from the census records
    (llmdw.scoring.results.oracle_times) of the sets visited by the orders of this table (recorded with
    llmdw.benchmark.eval_time.Recorder while they are computed; sets of the gain table without a census record are left
    out and counted); rich models with 12 to 16 families from llmdw.benchmark.eval_time --set large when its records
    exist (otherwise not available); per instance the mean over its visited sets, then the mean over instances (the
    collection's mean time, used for every row of the collection), with sd / sqrt(n). Benchmark instances (no root
    evaluation is timed there): the mean time of one branch-and-price run of a family set (GCG, the time-optimal decompositions,
    bnp_runs.csv; the sets of families, not GCG's default or SCIP), over its seeds, sets and instances, an unsolved run
    (time limit or failed twice) at 1800 s; per instance the mean over its runs, then the mean over instances.
    Prediction (one prediction = one answer = one request): the request that produced the answer of a cell (instance,
    presentation, sample): the last log line of the cell's tag without error; an answer read from the cache takes the
    latency and cost of the sent request with the same request_sha256 (or, when none is in the logs, the latency
    recorded with the cache hit and the cost of its usage at the price of its day). Rejected attempts are costed in
    requests.csv but are not predictions. Per instance, the latency and the cost are the means over its answered
    samples. Tags: support:<instance>:original:s<j> is the original presentation and with_data:<instance>:original:s<j>
    the coefficient presentation (rich models, benchmark instances).
    Per row (collection, class or all, configuration, presentation, s, order, baseline), over the instances with a
    value of the order, of the baseline and a latency: evaluations saved (mean); solver time represented = evaluations
    saved x the collection's mean time of one evaluation (columns *_collection_time); latency of one prediction (s)
    and its cost (USD); net time saved = solver time represented - latency (per instance, then the mean); break-even
    evaluation time = mean latency / mean evaluations saved (defined when the evaluations saved are positive). On the
    benchmark instances the measure of record uses each instance's own measured time of one evaluation (columns
    *_own_time, with the share of instances where the net time saved is positive), and the orders that use the
    five-dual values (the five-dual lookahead and JointDW interleaved with the five-dual order) are charged the
    stored time of the five-dual lookahead of the instance (bench_lookahead_seconds; side_seconds_charged = the
    order's minus the baseline's), in both net times and in the break-even time; lookahead_time.csv holds that time.
    Every row, per class and pooled, uses the same bootstrap; a class with a single prompt group is resampled in two
    stages too (the group, then its instances), so a standard error of 0 there means equal per-instance values (for
    example one request answering every instance of the group). Standard errors: two-stage bootstrap
    (llmdw.scoring.cluster: prompt groups, then instances within each drawn group; 10,000 resamples, seed 0) on the
    per-instance values, the same draws for every quantity of a row; the standard error of the break-even time is that
    of the ratio of the replicate means (given only when every replicate has positive evaluations saved). Asterisk:
    the 95 percent percentile interval excludes 0 (evaluations saved, solver time, net time). Prompt groups: those of
    the published tables (prompt_groups.csv of the released data; for the rich models with 12 to 16 families
    rich_models_large.prompt_groups); every instance its own group in the coefficient presentation. Every row records
    n_instances and n_groups.
    Outputs: eval_time.csv, predictions.csv (per collection, configuration and presentation: answered cells,
    instances, latency and cost of one prediction), net_benefit.csv, and the table body experiment_G_net_benefit.tex.

Usage
-----
python -m llmdw.scoring.cost_accounting prices --data DATA --work WORK --sources PRICING/sources.json --out OUT
python -m llmdw.scoring.cost_accounting requests --data DATA --work WORK --prices OUT/prices.csv --out OUT
python -m llmdw.scoring.cost_accounting tables --data DATA --work WORK --requests OUT/requests.csv \\
    --eval-time EVAL/eval_time.csv [--eval-time-large-paths LARGE/paths.json --eval-time-large-runs LARGE/runs] \\
    --bnp-runs RESULTS/experiment_D/bnp_runs.csv --out OUT --tex-dir TEX [--family-orders 1000]
DATA is the released data (runs/, work/, prompt_groups.csv); WORK is its work/ folder (rich_models_large/,
rich_models_large_requests/, no_joint_group/). The prices step reads the captured price pages, which are not part of the
released data; the released prices.csv (data/work/cost_accounting/prices.csv) is its output.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from llmdw.constants import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, FAMILY_ORDERS

LOCAL_TZ = ZoneInfo("America/Chicago")
CONFIGS_E = ("gpt-6.1-sol", "claude-sonnet-5-5")
PROVIDER = {"claude-sonnet-5-5": "anthropic", "gpt-6.1-sol": "openai", "deepseek-v4-pro": "deepseek",
            "grok-4-20-reasoning": "xai", "grok-4-20-non-reasoning": "xai"}
FALLBACK_DAY = "2026-11-08"
RULES = ("on the day", "nearest source before the day", "nearest source after the day",
         f"the capture of {FALLBACK_DAY}")
BNP_LIMIT = 1800.0
MTOK = 1e6

COLL_TEMPLATES, COLL_MIPLIB, COLL_RICH, COLL_LARGE, COLL_BENCH, COLL_F = (
    "templates", "MIPLIB-NL", "rich models, 7 to 9 families", "rich models, 12 to 16 families",
    "benchmark instances", "benchmark classes of Experiment F")
COLLECTIONS = (COLL_TEMPLATES, COLL_MIPLIB, COLL_RICH, COLL_LARGE, COLL_BENCH, COLL_F)
PRESENTATION_LABELS = {"original": "original", "data": "coefficients"}

COVERING = "covering"
FIVE = "lookahead, five duals"
JOINTDW = "JointDW"
INTER_FIVE = "JointDW interleaved with the five-dual order"
ORDERS = (JOINTDW, INTER_FIVE)
BASELINES = (COVERING, FIVE)


# ---------------------------------------------------------------- call logs

def log_files(data: Path, work: Path) -> list:
    """(experiment, path) of every call log of the sample of record, in reading order."""
    out = [("templates, MIPLIB-NL and rich models", data / "runs/llm_calls.jsonl"),
           ("controls", data / "work/controls/runs/llm_calls.jsonl"),
           ("Experiment C", data / "work/benchmark_std/runs/llm_calls.jsonl"),
           ("Experiment C", data / "work/benchmark_std/runs/llm_calls_data.jsonl")]
    for cfg in CONFIGS_E:
        out.append(("Experiment E", work / f"rich_models_large_requests/runs/{cfg}/llm_calls.jsonl"))
    for cfg in CONFIGS_E:
        out.append(("Experiment F", work / f"no_joint_group/runs/{cfg}/llm_calls.jsonl"))
    return out


def read_logs(files: list) -> tuple[list, int]:
    """(lines with experiment, source and line number; duplicates dropped). A missing file is an error."""
    seen, out, dup = set(), [], 0
    for exp, path in files:
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("time"), r.get("tag"), r.get("request_sha256"), r.get("latency"), r.get("cached"),
                       r.get("model_requested"))
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)
                out.append({**r, "_experiment": exp, "_source": str(path), "_line": i})
    return out, dup


def parse_tag(tag: str) -> dict:
    """support:<instance>:<presentation>:s<j> (also with_data:, ranked:)."""
    p = str(tag or "").split(":")
    kind = p[0] if p else ""
    inst = p[1] if len(p) > 1 else ""
    cond = p[2] if len(p) > 2 else ""
    m = re.match(r"s(\d+)$", p[3]) if len(p) > 3 else None
    return {"prompt": kind, "instance": inst, "condition": cond, "sample": int(m.group(1)) if m else 0}


def presentation_of(prompt: str, condition: str) -> str:
    """original, data (the coefficient presentation) or another presentation (the condition, or prompt:condition)."""
    if prompt == "support":
        return condition
    if prompt == "with_data":
        return "data" if condition == "original" else f"data, {condition}"
    return f"{prompt}, {condition}"


def start_time(r: dict) -> dt.datetime:
    """The start of the request (log time minus latency), timezone-aware (America/Chicago)."""
    t = dt.datetime.strptime(r["time"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=LOCAL_TZ)
    return t - dt.timedelta(seconds=float(r.get("latency") or 0.0))


def usage_tokens(u: dict | None) -> dict:
    """Prompt, cached, cache-write (5 minutes, 1 hour), completion and reasoning tokens of one usage record."""
    u = u or {}
    ptd = u.get("prompt_tokens_details") or {}
    ctd = u.get("completion_tokens_details") or {}
    if "cache_write_5m_tokens" in u or "cache_write_1h_tokens" in u:     # the released request records
        w5, w1 = int(u.get("cache_write_5m_tokens") or 0), int(u.get("cache_write_1h_tokens") or 0)
    else:                                                                # the provider's cache-creation counts
        cl = (u.get("billing_usage") or {}).get("claude_usage") or {}
        cc = cl.get("cache_creation") or {}
        w_all = int(cl.get("cache_creation_input_tokens") or 0)
        w1 = int(cc.get("ephemeral_1h_input_tokens") or 0)
        w5 = int(cc["ephemeral_5m_input_tokens"]) if "ephemeral_5m_input_tokens" in cc else max(0, w_all - w1)
    return {"prompt_tokens": int(u.get("prompt_tokens") or 0), "cached_prompt_tokens": int(ptd.get("cached_tokens") or 0),
            "cache_write_5m_tokens": w5, "cache_write_1h_tokens": w1,
            "completion_tokens": int(u.get("completion_tokens") or 0),
            "reasoning_tokens": int(ctd.get("reasoning_tokens") or 0)}


def error_category(err: str | None) -> str:
    if not err:
        return ""
    e = str(err)
    for pat, lab in ((r"empty content", "incomplete: empty content"), (r"no reasoning", "incomplete: no reasoning"),
                     (r"prompt tokens \d+ differ", "incomplete: prompt tokens differ from the accepted answer"),
                     (r"no finish_reason", "incomplete: cut off"), (r"prompt_inflation", "prompt inflation"),
                     (r"model_mismatch", "model mismatch")):
        if re.search(pat, e):
            return lab
    return "other"


# ---------------------------------------------------------------- prices

def load_sources(path: Path) -> list:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return [s for s in d["sources"] if s.get("usable", True)]


def choose_source(model: str, day: str, sources: list) -> tuple[dict | None, str]:
    """(source, rule): the source of the day (the last one), else the nearest before, else the nearest after, else
    the capture of FALLBACK_DAY (sources with a price for the model only)."""
    cands = sorted((s for s in sources if model in (s.get("prices") or {})),
                   key=lambda s: (s["date"], s.get("time_utc") or ""))
    on = [s for s in cands if s["date"] == day]
    if on:
        return on[-1], RULES[0]
    before = [s for s in cands if s["date"] < day]
    if before:
        return before[-1], RULES[1]
    after = [s for s in cands if s["date"] > day]
    if after:
        return after[0], RULES[2]
    fb = [s for s in sources if s["date"] == FALLBACK_DAY and s.get("kind") == "own capture"]
    return (fb[0] if fb else None), RULES[3]


def is_peak(t_utc: dt.datetime, price: dict) -> bool:
    """Whether a request that starts at t_utc is in the peak hours of a time-of-day price."""
    hours = price.get("peak_hours_utc")
    if not hours:
        return False
    if t_utc.weekday() not in price.get("peak_weekdays", range(7)):
        return False
    h = t_utc.hour + t_utc.minute / 60 + t_utc.second / 3600
    return any(a <= h < b for a, b in hours)


def rates(price: dict, t_utc: dt.datetime) -> tuple[str, dict]:
    """(rate label, {input, cached_input, output, cache_write_5m, cache_write_1h}) per million tokens."""
    if price.get("peak_hours_utc"):
        if is_peak(t_utc, price):
            lab, pre = "peak", ""
        else:
            lab, pre = "off-peak", "offpeak_"
    else:
        lab, pre = "standard", ""
    out = {k: price.get(pre + k, price.get(k)) for k in ("input", "cached_input", "output")}
    out["cache_write_5m"] = price.get("cache_write_5m", out["input"])
    out["cache_write_1h"] = price.get("cache_write_1h", out["input"])
    return lab, out


def cost_usd(tok: dict, r: dict) -> float:
    """The cost formula of the module docstring (prices r per million tokens)."""
    cached = tok["cached_prompt_tokens"]
    w5, w1 = tok.get("cache_write_5m_tokens", 0), tok.get("cache_write_1h_tokens", 0)
    plain = tok["prompt_tokens"] - cached - w5 - w1
    if plain < 0:
        raise ValueError(f"cached and cache-write tokens exceed the prompt tokens: {tok}")
    c = plain * r["input"] + tok["completion_tokens"] * r["output"]
    if cached:
        c += cached * r["cached_input"]
    if w5:
        c += w5 * r["cache_write_5m"]
    if w1:
        c += w1 * r["cache_write_1h"]
    return c / MTOK


def price_rows(lines: list, sources: list, names: dict) -> list:
    days = sorted({(r["model_requested"], start_time(r).date().isoformat()) for r in lines
                   if r.get("cached") is not None})
    rows = []
    for model, day in days:
        s, rule = choose_source(model, day, sources)
        p = (s or {}).get("prices", {}).get(model, {})
        rows.append({"model": names.get(model, model), "config": model, "provider": PROVIDER.get(model, ""),
                     "day": day, "price_rule": rule, "source_date": (s or {}).get("date"),
                     "source_kind": (s or {}).get("kind"), "url": (s or {}).get("url"),
                     "snapshot": (s or {}).get("snapshot"), "snapshot_url": (s or {}).get("snapshot_url"),
                     "source_folder": (s or {}).get("folder"), "sha256": (s or {}).get("sha256"),
                     "input_usd_per_mtok": p.get("input"), "cached_input_usd_per_mtok": p.get("cached_input"),
                     "output_usd_per_mtok": p.get("output"), "cache_write_5m_usd_per_mtok": p.get("cache_write_5m"),
                     "cache_write_1h_usd_per_mtok": p.get("cache_write_1h"),
                     "offpeak_input_usd_per_mtok": p.get("offpeak_input"),
                     "offpeak_cached_input_usd_per_mtok": p.get("offpeak_cached_input"),
                     "offpeak_output_usd_per_mtok": p.get("offpeak_output"),
                     "peak_hours_utc": json.dumps(p["peak_hours_utc"]) if p.get("peak_hours_utc") else "",
                     "where_on_the_page": p.get("where", "")})
    return rows


def prices_from_csv(path: Path) -> dict:
    """{(config, day): price dict as in sources.json, plus the rule and source} from prices.csv."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
            p = {"input": f("input_usd_per_mtok"), "cached_input": f("cached_input_usd_per_mtok"),
                 "output": f("output_usd_per_mtok"), "rule": r["price_rule"], "source": r["source_folder"],
                 "source_date": r["source_date"], "url": r["url"], "snapshot": r["snapshot"]}
            for k in ("cache_write_5m", "cache_write_1h", "offpeak_input", "offpeak_cached_input", "offpeak_output"):
                if f(f"{k}_usd_per_mtok") is not None:
                    p[k] = f(f"{k}_usd_per_mtok")
            if r.get("peak_hours_utc"):
                p["peak_hours_utc"] = json.loads(r["peak_hours_utc"])
                p["peak_weekdays"] = [0, 1, 2, 3, 4]
            out[(r["config"], r["day"])] = p
    return out


# ---------------------------------------------------------------- collections of the instances

def collection_lists(data: Path, work: Path) -> dict:
    """{instance: collection} from the instance lists of each collection."""
    out = {}

    def add(names, coll):
        for n in names:
            if n in out and out[n] != coll:
                raise ValueError(f"{n} in two collections")
            out[n] = coll

    W = data / "work"
    add([p.name for p in (W / "templates").iterdir() if p.is_dir()], COLL_TEMPLATES)
    add([p.name for p in (W / "miplib_set").iterdir() if p.is_dir()], COLL_MIPLIB)
    add([p.name for p in (W / "oracle").iterdir() if p.is_dir()], COLL_MIPLIB)
    add((W / "rich_models/instances.txt").read_text().split(), COLL_RICH)
    add((work / "rich_models_large/instances.txt").read_text().split(), COLL_LARGE)
    add((W / "benchmark_std/full_set/all_instances.txt").read_text().split(), COLL_BENCH)
    fl = work / "no_joint_group/all_instances.txt"
    if fl.exists():
        add(fl.read_text().split(), COLL_F)
    return out


# ---------------------------------------------------------------- requests

def request_rows(lines: list, prices: dict, names: dict, coll_of: dict) -> list:
    rows = []
    for r in lines:
        if r.get("cached") is not False:
            continue
        cfg = r["model_requested"]
        t0 = start_time(r)
        day = t0.date().isoformat()
        p = prices.get((cfg, day))
        if p is None:
            raise KeyError(f"no price for {cfg} on {day}")
        tok = usage_tokens(r.get("usage"))
        lab, rt = rates(p, t0.astimezone(dt.timezone.utc))
        tg = parse_tag(r.get("tag"))
        rows.append({"experiment": r["_experiment"], "source": r["_source"], "line": r["_line"],
                     "time": r["time"], "start_utc": t0.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "day": day, "model": names.get(cfg, cfg), "config": cfg,
                     "model_returned": r.get("model_returned") or "", "tag": r.get("tag"), "prompt": tg["prompt"],
                     "collection": coll_of.get(tg["instance"], ""), "instance": tg["instance"],
                     "presentation": presentation_of(tg["prompt"], tg["condition"]), "sample": tg["sample"],
                     "request_sha256": r.get("request_sha256"), **tok, "latency_s": r.get("latency"),
                     "finish_reason": r.get("finish_reason") or "", "rejected": error_category(r.get("error")),
                     "price_day": p["source_date"], "price_rule": p["rule"], "price_source": p["source"],
                     "rate": lab, "input_usd_per_mtok": rt["input"], "cached_input_usd_per_mtok": rt["cached_input"],
                     "cache_write_5m_usd_per_mtok": rt["cache_write_5m"] if tok["cache_write_5m_tokens"] else "",
                     "cache_write_1h_usd_per_mtok": rt["cache_write_1h"] if tok["cache_write_1h_tokens"] else "",
                     "output_usd_per_mtok": rt["output"], "cost_usd": cost_usd(tok, rt)})
    return rows


def _stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.median(xs), sum(xs) / len(xs)


def summary_rows(req: list, lines: list) -> list:
    """Per experiment and model, and per model over the experiments."""
    hits = Counter((r["_experiment"], r["model_requested"]) for r in lines if r.get("cached") is True)
    aside = Counter((r["_experiment"], r["model_requested"]) for r in lines if r.get("cached") is None)
    groups = defaultdict(list)
    for r in req:
        groups[(r["experiment"], r["config"])].append(r)
        groups[("all", r["config"])].append(r)
        groups[("all", "all")].append(r)
    out = []
    order = {e: i for i, e in enumerate(dict.fromkeys(r["experiment"] for r in req))}
    for (exp, cfg), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] == "all", order.get(kv[0][0], 99),
                                                                  kv[0][1] == "all", kv[0][1])):
        answered = defaultdict(int)
        repeated = 0
        for r in sorted(rs, key=lambda x: x["start_utc"]):
            if r["rejected"]:
                continue
            key = (r["config"], r["request_sha256"])
            if answered[key]:
                repeated += 1
            answered[key] += 1
        med, mean = _stats(r["latency_s"] for r in rs)
        ch = sum(v for (e, c), v in hits.items() if (exp == "all" or e == exp) and (cfg == "all" or c == cfg))
        sa = sum(v for (e, c), v in aside.items() if (exp == "all" or e == exp) and (cfg == "all" or c == cfg))
        out.append({"experiment": exp, "model": rs[0]["model"] if cfg != "all" else "all", "config": cfg,
                    "requests": len(rs), "rejected_attempts": sum(1 for r in rs if r["rejected"]),
                    "repeated_sends_of_an_answered_request": repeated,
                    "distinct_requests": len({r["request_sha256"] for r in rs}), "cache_hits": ch,
                    "cache_entries_set_aside": sa,
                    "first_day": min(r["day"] for r in rs), "last_day": max(r["day"] for r in rs),
                    "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
                    "cached_prompt_tokens": sum(r["cached_prompt_tokens"] for r in rs),
                    "cache_write_tokens": sum(r["cache_write_5m_tokens"] + r["cache_write_1h_tokens"] for r in rs),
                    "completion_tokens": sum(r["completion_tokens"] for r in rs),
                    "reasoning_tokens": sum(r["reasoning_tokens"] for r in rs),
                    "latency_median_s": med, "latency_mean_s": mean,
                    "latency_sum_s": sum(float(r["latency_s"] or 0) for r in rs),
                    "cost_usd": sum(r["cost_usd"] for r in rs),
                    "cost_rejected_attempts_usd": sum(r["cost_usd"] for r in rs if r["rejected"])})
    return out


# ---------------------------------------------------------------- predictions (answers)

def answer_cells(lines: list, req: list, prices: dict, coll_of: dict) -> dict:
    """{(collection, config, presentation, instance, sample): {"latency", "cost", "from_cache", "found"}}: the request
    that produced the answer of each cell (module docstring)."""
    sent = {}
    for q in req:
        if not q["rejected"]:
            sent[(q["config"], q["request_sha256"])] = q          # the last answered send of a request
    last = {}
    for r in lines:
        if r.get("cached") is None or r.get("error"):
            continue
        tg = parse_tag(r.get("tag"))
        if tg["prompt"] not in ("support", "with_data"):
            continue
        key = (coll_of.get(tg["instance"], ""), r["model_requested"], presentation_of(tg["prompt"], tg["condition"]),
               tg["instance"], tg["sample"])
        last[key] = r
    out = {}
    for key, r in last.items():
        q = sent.get((r["model_requested"], r.get("request_sha256")))
        if q is not None:
            out[key] = {"latency": float(q["latency_s"]), "cost": q["cost_usd"], "from_cache": r["cached"] is True,
                        "found": True}
        else:
            t0 = start_time(r)
            p = prices.get((r["model_requested"], t0.date().isoformat()))
            c = None
            if p is not None:
                _, rt = rates(p, t0.astimezone(dt.timezone.utc))
                c = cost_usd(usage_tokens(r.get("usage")), rt)
            out[key] = {"latency": float(r.get("latency") or 0.0), "cost": c, "from_cache": True, "found": False}
    return out


def per_instance_prediction(cells: dict) -> dict:
    """{(collection, config, presentation): {instance: (mean latency, mean cost, samples)}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for (coll, cfg, pres, inst, _), v in cells.items():
        acc[(coll, cfg, pres)][inst].append(v)
    out = {}
    for key, d in acc.items():
        out[key] = {}
        for n, vs in d.items():
            costs = [v["cost"] for v in vs if v["cost"] is not None]
            out[key][n] = (sum(v["latency"] for v in vs) / len(vs), sum(costs) / len(costs) if costs else None,
                           len(vs))
    return out


def prediction_rows(cells: dict, names: dict) -> list:
    rows = []
    groups = defaultdict(list)
    for (coll, cfg, pres, inst, s), v in cells.items():
        groups[(coll, cfg, pres)].append((inst, v))
    for (coll, cfg, pres), xs in sorted(groups.items(), key=lambda kv: (COLLECTIONS.index(kv[0][0])
                                                                         if kv[0][0] in COLLECTIONS else 99,
                                                                         kv[0][1], kv[0][2])):
        lat = [v["latency"] for _, v in xs]
        cost = [v["cost"] for _, v in xs if v["cost"] is not None]
        rows.append({"collection": coll, "model": names.get(cfg, cfg), "config": cfg,
                     "presentation": PRESENTATION_LABELS.get(pres, pres), "answered_cells": len(xs),
                     "instances": len({i for i, _ in xs}), "answers_from_cache": sum(v["from_cache"] for _, v in xs),
                     "answers_from_cache_without_logged_request": sum(not v["found"] for _, v in xs),
                     "latency_median_s": statistics.median(lat), "latency_mean_s": sum(lat) / len(lat),
                     "cost_mean_usd": sum(cost) / len(cost) if cost else None,
                     "cost_median_usd": statistics.median(cost) if cost else None})
    return rows


# ---------------------------------------------------------------- search values per instance

def _run(name, truth, s, spec):
    from llmdw.scoring import results
    return results.run_value(results.search_runs(name, truth, s, spec)[0])


def values_b(data: Path, phase: str, configs: list, record=None) -> dict:
    """Templates or MIPLIB-NL: per-instance values of the covering order, the five-dual lookahead, JointDW and JointDW
    interleaved with the five-dual order (original presentation), as llmdw.scoring.experiments.five_duals_tables, s =
    2 and 3 (capped at k - 1). Returns {"vals": {(order, config, presentation, s): {instance: value}}, "class",
    "groups", "visited"}. record: an llmdw.benchmark.eval_time.Recorder installed while the values are computed."""
    from contextlib import nullcontext
    from llmdw.scoring import cluster, experiments, results
    from llmdw.search import relabel

    insts = experiments._phase_instances(data, phase)
    cluster.set_groups({})
    results.use_prompt_groups(data, phase, insts)
    groups = dict(cluster.GROUPS)
    five_root = data / "work/five_duals"
    five = {}
    for n, i in insts.items():
        p = five_root / phase / n / "five_duals.json"
        if p.exists():
            five[n] = {**i["truth"], "lookahead": experiments._five_list(p, len(i["truth"]["families"]),
                                                                          phase == "templates")}
    names = [n for n in insts if insts[n]["truth"]["joint"] and n in five]
    names = [n for n in names if insts[n]["truth"]["table"] and max(insts[n]["truth"]["table"].values()) > 0]
    recs = {}
    for cfg in configs:
        d = data / f"work/support_{phase}/{cfg}"
        if d.exists():
            recs[cfg] = [r for r in results.load_answers(d, insts) if r["condition"] == "original"]
    vals = defaultdict(dict)
    with (record.installed() if record is not None else nullcontext()):
        for label, truth_of in (("highs", lambda n: insts[n]["truth"]), ("five", lambda n: five[n])):
            relabel.clear()
            for n in names:
                t = truth_of(n)
                k = len(t["families"])
                for size in (2, 3):
                    s = min(size, k - 1)
                    if label == "highs":
                        vals[(COVERING, "", "", size)][n] = _run(n, t, s, ("covering",))
                    else:
                        vals[(FIVE, "", "", size)][n] = _run(n, t, s, ("lookahead",))
            for cfg, rs in recs.items():
                acc = defaultdict(lambda: defaultdict(list))
                for r in rs:
                    n = r["instance"]
                    if n not in names:
                        continue
                    t = truth_of(n)
                    k = len(t["families"])
                    P = results.support_anchor(r, t["families"])
                    for size in (2, 3):
                        s = min(size, k - 1)
                        if label == "highs":
                            acc[(JOINTDW, size)][n].append(_run(n, t, s, ("model", P)))
                        else:
                            acc[(INTER_FIVE, size)][n].append(_run(n, t, s, ("interleaved", P)))
                for (o, size), d in acc.items():
                    vals[(o, cfg, "original", size)] = {n: (None if any(x is None for x in v) else sum(v) / len(v))
                                                       for n, v in d.items()}
        relabel.clear()
    cls = {n: (insts[n]["class"] if phase == "templates" else "all") for n in names}
    return {"vals": dict(vals), "class": cls, "groups": groups, "names": names}


def values_e(root: Path, five_root: Path, answer_roots: dict, summary_work: Path, phase: str, configs: list,
             groups: dict | None) -> dict:
    """Rich models: per-instance values as llmdw.scoring.rich_models_large.score_set (both presentations)."""
    from llmdw.scoring import rich_models_large as xe

    S = xe.score_set(root, five_root, answer_roots, summary_work, phase, configs, False)
    insts = S["insts"]
    if groups is None:
        groups = xe.prompt_groups(insts)
    vals = {}
    for (o, cfg, pres, s), d in S["vals"].items():
        o2 = {xe.COVERING: COVERING, xe.FIVE: FIVE, xe.JOINTDW: JOINTDW, xe.INTER_FIVE: INTER_FIVE}.get(o)
        if o2 is not None:
            vals[(o2, cfg, pres, s)] = d
    return {"vals": vals, "class": {i["name"]: i["class"] for i in insts}, "groups": groups,
            "names": [i["name"] for i in insts], "sizes": {i["name"]: i["sizes"] for i in insts}}


def values_bench(root: Path, answers: Path, names_file: Path, configs: dict) -> dict:
    """Benchmark instances of Experiment C (the 59 of the time-optimal decompositions): per-instance values of the covering order, the
    five-dual lookahead, JointDW and JointDW interleaved with the five-dual order, s = 2 and 3 (capped at k - 1), from
    the stored records only: the gain table of oracle.json (llmdw.benchmark.bnp), the best-of-five lookahead values
    (five_duals.json) and the anchor of every sample (bnp_plan.json for the original presentation,
    bnp_plan_data.json for the coefficient presentation; llmdw.scoring.results.support_anchor). Per instance, JointDW
    is the mean over its samples (a failure when one sample fails). Prompt groups from the original prompts, as in
    the time-optimal decompositions. configs: {configuration: official name}. side_seconds: per instance the stored time of the
    five-dual lookahead of Experiment C (bench_lookahead_seconds)."""
    from llmdw.benchmark import bnp
    from llmdw.scoring import cluster, experiments
    from llmdw.search import relabel

    insts = Path(names_file).read_text().split()
    official = {v: k for k, v in configs.items()}
    prompts = {}
    for n in insts:
        prompts[n] = {}
        for cfg in configs:
            f = Path(answers) / cfg / n / "original.json"
            if f.exists():
                prompts[n][cfg] = json.loads(f.read_text())["messages"]
    groups = cluster.prompt_groups(prompts)
    truths, plans, cls = {}, {}, {}
    for n in insts:
        d = Path(root) / "instances" / n
        t = bnp._truth(d)
        k = len(t["families"])
        truths[n] = {"families": t["families"], "table": t["table"],
                     "lookahead": experiments._five_list(d / "five_duals.json", k, True)}
        plans[n] = {"original": json.loads((d / "bnp_plan.json").read_text())["methods"],
                    "data": json.loads((d / "bnp_plan_data.json").read_text())["methods"]}
        cls[n] = json.loads((d / "meta.json").read_text())["class"]
    vals = defaultdict(dict)
    relabel.clear()
    for n in insts:
        t = truths[n]
        k = len(t["families"])
        for size in (2, 3):
            s = min(size, k - 1)
            vals[(COVERING, "", "", size)][n] = _run(n, t, s, ("covering",))
            vals[(FIVE, "", "", size)][n] = _run(n, t, s, ("lookahead",))
            for pres, key in (("original", "{}"), ("data", "{}, data")):
                for name, cfg in official.items():
                    samples = plans[n][pres].get(key.format(name)) or []
                    for o, kind in ((JOINTDW, "model"), (INTER_FIVE, "interleaved")):
                        v = [_run(n, t, s, (kind, frozenset(x["set"]) & frozenset(t["families"]))) for x in samples]
                        if v:
                            vals[(o, cfg, pres, size)][n] = None if any(x is None for x in v) else sum(v) / len(v)
    relabel.clear()
    return {"vals": dict(vals), "class": cls, "groups": groups, "names": insts,
            "side_seconds": bench_lookahead_seconds(root, insts)}


def bench_lookahead_seconds(root: Path, names: list) -> dict:
    """Per benchmark instance, the stored time of the five-dual lookahead (Experiment C, one thread, this machine):
    the seconds of the recorded lookahead at the HiGHS dual (lookahead.json: the LP solve and one pricing round per
    candidate set) plus those of the four further duals (dual1.json to dual4.json next to it: model read, LP, the
    optimal-face dual and one pricing round per candidate set; recorded to 0.1 s). The files are those that
    five_duals.json was combined from (its "recorded" path)."""
    out = {}
    for n in names:
        rec = Path(json.loads((Path(root) / "instances" / n / "five_duals.json").read_text())["recorded"])
        # a relative path of the released data is relative to data/ (root is data/work/benchmark_std)
        at = rec if rec.is_absolute() or rec.exists() else Path(root).resolve().parents[1] / rec
        secs = json.loads(at.read_text())["seconds"]
        for k in (1, 2, 3, 4):
            d = json.loads((at.parent / f"dual{k}.json").read_text())
            if Path(d["recorded"]) != rec:
                raise ValueError(f"{n}: dual{k}.json was not computed from {rec}")
            secs += d["seconds"]
        out[n] = secs
    return out


def lookahead_time_rows(V: dict) -> list:
    """Mean and SE (sd / sqrt(n)) over instances of the five-dual lookahead time, per class and pooled."""
    rows = []
    for c in sorted(set(V["class"].values())) + ["all"]:
        xs = [V["side_seconds"][n] for n in V["names"] if c == "all" or V["class"][n] == c]
        m, se = _mean_se(xs)
        rows.append({"collection": COLL_BENCH, "class": c, "n_instances": len(xs), "mean_seconds": m, "se_seconds": se,
                     "min_seconds": min(xs), "max_seconds": max(xs),
                     "definition": "stored time of the five-dual lookahead of Experiment C: lookahead.json seconds "
                                   "(HiGHS dual) plus dual1.json to dual4.json seconds"})
    return rows


# ---------------------------------------------------------------- time of one evaluation

def _mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    return m, (statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None)


EVAL_TIME_DEFINITION = ("GCG root evaluation in the census setting of the sets the search orders visit "
                        "(llmdw.benchmark.eval_time); census time of a set (results.oracle_seconds)")


def eval_time_rows(rows: list, collection_key: str, label: str, source: str) -> list | None:
    """The rows of one collection of llmdw.benchmark.eval_time.time_tables (or of its eval_time.csv)."""
    out = []
    for r in rows:
        if r["collection"] != collection_key:
            continue
        f = lambda k: float(r[k]) if r.get(k) not in (None, "") else None
        n = lambda k: int(float(r[k])) if r.get(k) not in (None, "") else 0
        out.append({"collection": label, "class": r["class"], "definition": EVAL_TIME_DEFINITION, "source": source,
                    "n_instances": n("n_instances_with_time"), "n_sets": n("n_sets_with_time"),
                    "n_sets_left_out": n("n_sets_missing") + n("n_sets_not_run"), "n_at_limit": n("n_sets_at_limit"),
                    "mean_seconds": f("mean_of_instance_means"), "se_seconds": f("se_over_instances"),
                    "pooled_mean_seconds": f("pooled_mean"), "pooled_se_seconds": f("pooled_se"),
                    "gcg_mean_seconds": f("gcg_mean_of_instance_means"), "gcg_se_seconds": f("gcg_se_over_instances")})
    return out or None


def eval_time_from_csv(path: Path, collection_key: str, label: str, source: str) -> list | None:
    if not Path(path).exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return eval_time_rows(list(csv.DictReader(fh)), collection_key, label, source)


def eval_time_large(paths: Path, runs: Path, label: str) -> list | None:
    """The rich models with 12 to 16 families: llmdw.benchmark.eval_time.time_tables when every visited set has a
    record; None otherwise (not available)."""
    from llmdw.benchmark import eval_time

    if not (paths and runs and Path(paths).exists() and Path(runs).exists()):
        return None
    P = json.loads(Path(paths).read_text(encoding="utf-8"))
    rows, _ = eval_time.time_tables(P, Path(runs))
    allr = [r for r in rows if r["class"] == "all"]
    if not allr or allr[0]["n_sets_missing"]:
        return None
    return eval_time_rows(rows, "rich_models_large", label, str(paths))


def miplib_eval_time(data: Path, names: list, visited: dict) -> list:
    from llmdw.scoring import results

    per, n_sets, left, at = {}, 0, 0, 0
    for n in names:
        t = results.oracle_times(data, n)
        xs = []
        for T in visited.get(n, {}):
            if T in t:
                xs.append(t[T])
                at += t[T] >= results.EVAL_LIMIT_S
            else:
                left += 1
        n_sets += len(xs)
        if xs:
            per[n] = xs
    means = [sum(v) / len(v) for v in per.values()]
    pooled = [x for v in per.values() for x in v]
    m, se = _mean_se(means)
    pm, pse = _mean_se(pooled)
    return [{"collection": COLL_MIPLIB, "class": "all",
             "definition": "census evaluation (results.oracle_times) of the sets the search orders of this table visit",
             "source": str(data / "work/oracle"), "n_instances": len(per), "n_sets": n_sets, "n_sets_left_out": left,
             "n_at_limit": at, "mean_seconds": m, "se_seconds": se, "pooled_mean_seconds": pm, "pooled_se_seconds": pse}]


def bnp_eval_time_per_instance(path: Path) -> dict:
    """Benchmark instances: per instance, the mean time of one branch-and-price run of a family set (bnp_eval_time)."""
    per = defaultdict(list)
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["set"] in ("gcg_default", "scip"):
                continue
            per[r["instance"]].append(min(BNP_LIMIT, float(r["time"])) if r["solved"] == "True" else BNP_LIMIT)
    return {n: sum(v) / len(v) for n, v in per.items()}


def bnp_eval_time(path: Path) -> list:
    """Benchmark instances: the mean time of one branch-and-price run of a family set (module docstring)."""
    if not Path(path).exists():
        return []
    per = defaultdict(list)
    cls = {}
    unsolved = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["set"] in ("gcg_default", "scip"):
                continue
            solved = r["solved"] == "True"
            t = min(BNP_LIMIT, float(r["time"])) if solved else BNP_LIMIT
            unsolved += not solved
            per[r["instance"]].append(t)
            cls[r["instance"]] = r["class"]
    rows = []
    for c in sorted(set(cls.values())) + ["all"]:
        ns = [n for n in per if c == "all" or cls[n] == c]
        means = [sum(per[n]) / len(per[n]) for n in ns]
        pooled = [x for n in ns for x in per[n]]
        m, se = _mean_se(means)
        pm, pse = _mean_se(pooled)
        rows.append({"collection": COLL_BENCH, "class": c,
                     "definition": "GCG branch-and-price of a family set (1 thread, 1800 s, seeds 0 to 2), an unsolved "
                                   "run at 1800 s", "source": str(path), "n_instances": len(ns), "n_sets": len(pooled),
                     "n_sets_left_out": 0,
                     "n_at_limit": sum(1 for n in ns for x in per[n] if x >= BNP_LIMIT),
                     "mean_seconds": m, "se_seconds": se, "pooled_mean_seconds": pm, "pooled_se_seconds": pse})
    return rows


# ---------------------------------------------------------------- net benefit rows

def _sel(d: dict, names) -> dict:
    return {n: d[n] for n in names if n in d}


def _boot(values: dict):
    from llmdw.scoring import cluster
    return cluster.boot(values, np.mean, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)


def _star(lo, hi) -> str:
    return "*" if lo is not None and (lo > 0 or hi < 0) else ""


def net_row(saved: dict, lat: dict, cost: dict, t_eval: float | None, t_inst: dict | None = None,
            charge: dict | None = None) -> dict:
    """The quantities of one row from per-instance values over the same instances (module docstring). t_eval: the
    collection's mean time of one evaluation (columns *_collection_time). t_inst: the measured mean time of one
    evaluation of each instance (benchmark instances; columns *_own_time: evaluations saved x the instance's time -
    latency - charge, and the share of instances where it is positive). charge: per instance, the seconds of side
    computation the order needs minus those the baseline needs (the five-dual lookahead), subtracted from every net
    time saved and added to the latency in the break-even time."""
    from llmdw.scoring import cluster

    names = sorted(n for n in saved if saved[n] is not None and lat.get(n) is not None)
    S = {n: saved[n] for n in names}
    L = {n: lat[n] for n in names}
    X = {n: (charge or {}).get(n, 0.0) for n in names}
    C = {n: cost[n] for n in names if cost.get(n) is not None}
    n_inst, n_grp = cluster.counts(S)
    out = {"n_instances": n_inst, "n_groups": n_grp}
    if not names:
        return out
    m_s = sum(S.values()) / len(S)
    lo, hi, se = _boot(S)
    out.update(evaluations_saved_mean=m_s, evaluations_saved_se=se, evaluations_saved_asterisk=_star(lo, hi))
    m_l = sum(L.values()) / len(L)
    _, _, se_l = _boot(L)
    out.update(latency_mean_s=m_l, latency_se_s=se_l)
    m_x = sum(X.values()) / len(X)
    if charge is not None:
        _, _, se_x = _boot(X)
        out.update(side_seconds_charged_mean=m_x, side_seconds_charged_se=se_x)
    if C:
        _, _, se_c = _boot(C)
        out.update(cost_mean_usd=sum(C.values()) / len(C), cost_se_usd=se_c)
    if t_eval is not None:
        T = {n: S[n] * t_eval for n in names}
        lo, hi, se = _boot(T)
        out.update(solver_seconds_saved_collection_time_mean=m_s * t_eval, solver_seconds_saved_collection_time_se=se,
                   solver_seconds_saved_collection_time_asterisk=_star(lo, hi))
        N = {n: S[n] * t_eval - L[n] - X[n] for n in names}
        lo, hi, se = _boot(N)
        out.update(net_seconds_saved_collection_time_mean=sum(N.values()) / len(N),
                   net_seconds_saved_collection_time_se=se, net_seconds_saved_collection_time_asterisk=_star(lo, hi))
    if t_inst:
        own = [n for n in names if t_inst.get(n) is not None]
        if own:
            E = {n: t_inst[n] for n in own}
            T = {n: S[n] * t_inst[n] for n in own}
            N = {n: T[n] - L[n] - X[n] for n in own}
            P = {n: float(v > 0) for n, v in N.items()}
            _, _, se_e = _boot(E)
            lo_t, hi_t, se_t = _boot(T)
            lo, hi, se = _boot(N)
            _, _, se_p = _boot(P)
            out.update(eval_seconds_own_mean=sum(E.values()) / len(E), eval_seconds_own_se=se_e,
                       solver_seconds_saved_own_time_mean=sum(T.values()) / len(T),
                       solver_seconds_saved_own_time_se=se_t, solver_seconds_saved_own_time_asterisk=_star(lo_t, hi_t),
                       net_seconds_saved_own_time_mean=sum(N.values()) / len(N), net_seconds_saved_own_time_se=se,
                       net_seconds_saved_own_time_asterisk=_star(lo, hi),
                       share_positive_net_own_time=sum(P.values()) / len(P), share_positive_net_own_time_se=se_p,
                       n_instances_own_time=len(N))
    if m_s > 0:
        LX = {n: L[n] + X[n] for n in names}
        rs = cluster.replicates(S, np.mean, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        rl = cluster.replicates(LX, np.mean, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        ok = bool(np.all(rs > 0))
        out.update(break_even_eval_seconds=(m_l + m_x) / m_s,
                   break_even_eval_seconds_se=float(np.std(rl / rs, ddof=1)) if ok else None,
                   break_even_replicates_without_savings=int(np.sum(rs <= 0)))
    return out


def break_even(latency: float, saved: float) -> float | None:
    """Latency of one prediction divided by the evaluations saved (None when nothing is saved)."""
    return latency / saved if saved > 0 else None


def saved(baseline: float | None, order: float | None) -> float | None:
    """Evaluations saved by the order against the baseline: baseline minus order."""
    if baseline is None or order is None:
        return None
    return baseline - order


USES_FIVE_DUALS = (FIVE, INTER_FIVE)


def net_rows(coll: str, V: dict, pred: dict, t_eval: float | None, t_note: str, names: dict,
             t_inst: dict | None = None) -> list:
    """Rows of one collection: per class and pooled, configuration, presentation, s, order and baseline. With
    V["side_seconds"] ({instance: seconds of the five-dual lookahead}), the orders that use the five-dual values (the
    five-dual lookahead order and JointDW interleaved with the five-dual order; the interleaved order of the main
    experiments uses the lookahead at the HiGHS dual instead) are charged that time (net_row, charge)."""
    from llmdw.scoring import cluster

    vals, cls, groups = V["vals"], V["class"], V["groups"]
    classes = ["all"] + sorted(set(cls.values()) - {"all"})
    rows = []
    keys = sorted({(cfg, pres) for (o, cfg, pres, s) in vals if o in ORDERS},
                  key=lambda x: (list(names).index(x[0]) if x[0] in names else 99, x[1] != "original", x[1]))
    for c in classes:
        insts = [n for n in V["names"] if c == "all" or cls[n] == c]
        for cfg, pres in keys:
            pi = pred.get((coll, cfg, pres), {})
            lat = {n: v[0] for n, v in pi.items()}
            cst = {n: v[1] for n, v in pi.items()}
            cluster.set_groups({} if pres == "data" else groups)
            for s in (2, 3):
                nc = [n for n in insts if "sizes" in V and s not in V["sizes"][n]]
                for o in ORDERS:
                    A = _sel(vals.get((o, cfg, pres, s), {}), insts)
                    for b in BASELINES:
                        B = _sel(vals.get((b, "", "", s), {}), insts)
                        sv = {n: saved(B.get(n), A.get(n)) for n in A}
                        sv = {n: v for n, v in sv.items() if v is not None}
                        r = {"collection": coll, "class": c, "model": names.get(cfg, cfg), "config": cfg,
                             "presentation": PRESENTATION_LABELS.get(pres, pres), "s": s, "order": o, "baseline": b,
                             "status": "not computed" if insts and len(nc) == len(insts) else "computed",
                             "n_not_computed": len(nc), "n_failed": sum(1 for n in insts if n in A and A[n] is None),
                             "n_without_prediction": sum(1 for n in sv if lat.get(n) is None),
                             "eval_seconds": t_eval, "eval_seconds_source": t_note}
                        side = V.get("side_seconds")
                        ch = ({n: side[n] * ((o in USES_FIVE_DUALS) - (b in USES_FIVE_DUALS)) for n in sv}
                              if side is not None else None)
                        r["side_computation_charged"] = (
                            "five-dual lookahead (stored times of Experiment C)" if side is not None else "none")
                        r.update(net_row(sv, lat, cst, t_eval, t_inst, ch))
                        rows.append(r)
    cluster.set_groups({})
    return rows


def bench_rows(pred: dict, t_rows: list, names: dict) -> list:
    """Benchmark instances: no search-cost table; the time of one evaluation and one prediction."""
    from llmdw.scoring import cluster

    cluster.set_groups({})
    t = next((r["mean_seconds"] for r in t_rows if r["class"] == "all"), None)
    rows = []
    for (coll, cfg, pres), d in sorted(pred.items()):
        if coll != COLL_BENCH:
            continue
        lat = {n: v[0] for n, v in d.items()}
        cst = {n: v[1] for n, v in d.items() if v[1] is not None}
        _, _, se_l = _boot(lat)
        _, _, se_c = _boot(cst)
        rows.append({"collection": coll, "class": "all", "model": names.get(cfg, cfg), "config": cfg,
                     "presentation": PRESENTATION_LABELS.get(pres, pres), "s": "", "order": JOINTDW, "baseline": "",
                     "status": "no search-cost table (evaluations saved not available)",
                     "eval_seconds": t, "eval_seconds_source": "branch-and-price, Experiment D",
                     "n_instances": len(lat), "n_groups": cluster.counts(lat)[1],
                     "latency_mean_s": sum(lat.values()) / len(lat), "latency_se_s": se_l,
                     "cost_mean_usd": sum(cst.values()) / len(cst) if cst else None, "cost_se_usd": se_c})
    return rows


# ---------------------------------------------------------------- table body

def _f(x, d=2):
    from llmdw.tables import experiments as tx
    return tx.f2(x, d)


def _pm(r, m, se, star="", d=2):
    from llmdw.tables import experiments as tx
    if r is None or r.get(m) in (None, ""):
        return "--"
    if r.get(se) in (None, ""):
        return _f(r.get(m), d)
    return tx.pm(r.get(m), r.get(se), d, r.get(star, "") if star else "")


def tex_body(rows: list) -> str:
    head = ("% Experiment G: evaluations saved and solver time against the latency of one prediction (all instances "
            "of each collection). Columns: configuration, presentation, order; s; evaluations saved against the "
            "covering order and against the five-dual lookahead order; latency of one prediction (s); cost of one "
            "prediction (USD); net time saved (s) against the covering order and against the five-dual order "
            "(evaluations saved times the collection's mean time of one evaluation, minus the latency; on the "
            "benchmark instances, evaluations saved times each instance's own measured time of one evaluation, minus "
            "the latency and minus the time of the five-dual lookahead for the orders that use it, per instance); "
            "break-even time of one evaluation (s) against both (latency, plus the lookahead time charged on the "
            "benchmark instances, divided by evaluations saved). Mean $\\pm$ standard "
            "error, two-stage bootstrap over prompt groups and instances; asterisk: 95 percent interval excludes 0; "
            "--: not available. Source: results/experiment_G/net_benefit.csv")
    lines = [head]
    pooled = [r for r in rows if r["class"] == "all"]
    for coll in COLLECTIONS:
        rs = [r for r in pooled if r["collection"] == coll]
        if not rs:
            continue
        t = rs[0].get("eval_seconds")
        lab = coll[0].upper() + coll[1:]
        tt = f"{_f(t, 3) if t is not None else '--'} s"
        if coll == COLL_BENCH:
            tt = (f"each instance's own (mean {tt}); five-dual lookahead charged")
        lines.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{lab}; time of one evaluation "
                     f"{tt}}}}} \\\\")
        keys = list(dict.fromkeys((r["model"], r["presentation"], r["order"], r["s"]) for r in rs))
        for m, p, o, s in keys:
            cov = next((r for r in rs if (r["model"], r["presentation"], r["order"], r["s"], r["baseline"])
                        == (m, p, o, s, COVERING)), None)
            fiv = next((r for r in rs if (r["model"], r["presentation"], r["order"], r["s"], r["baseline"])
                        == (m, p, o, s, FIVE)), None)
            base = cov or fiv or next((r for r in rs if (r["model"], r["presentation"], r["order"], r["s"])
                                       == (m, p, o, s)), None)
            if base is None:
                continue
            if base.get("status", "").startswith("no search-cost"):
                cells = ["--", "--", _pm(base, "latency_mean_s", "latency_se_s"),
                         _pm(base, "cost_mean_usd", "cost_se_usd", d=4), "--", "--", "--", "--"]
                lines.append(f"{m}, {p} & -- & " + " & ".join(cells) + " \\\\")
                continue
            if base.get("status") == "not computed":
                lines.append(f"{m}, {p}, {o} & {s} & " + " & ".join(["n.c."] * 8) + " \\\\")
                continue
            nk = "own_time" if coll == COLL_BENCH else "collection_time"
            cells = [_pm(cov, "evaluations_saved_mean", "evaluations_saved_se", "evaluations_saved_asterisk"),
                     _pm(fiv, "evaluations_saved_mean", "evaluations_saved_se", "evaluations_saved_asterisk"),
                     _pm(base, "latency_mean_s", "latency_se_s"), _pm(base, "cost_mean_usd", "cost_se_usd", d=4),
                     _pm(cov, f"net_seconds_saved_{nk}_mean", f"net_seconds_saved_{nk}_se",
                         f"net_seconds_saved_{nk}_asterisk"),
                     _pm(fiv, f"net_seconds_saved_{nk}_mean", f"net_seconds_saved_{nk}_se",
                         f"net_seconds_saved_{nk}_asterisk"),
                     _pm(cov, "break_even_eval_seconds", "break_even_eval_seconds_se"),
                     _pm(fiv, "break_even_eval_seconds", "break_even_eval_seconds_se")]
            lines.append(f"{m}, {p}, {o} & {s} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- csv

def write_csv(path: Path, rows: list):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.10g}" if isinstance(r.get(k), float) else r[k]))
                        for k in keys})


def read_requests(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("latency_s", "cost_usd"):
            r[k] = float(r[k]) if r[k] != "" else None
    return rows


# ---------------------------------------------------------------- driver

def _names():
    from llmdw.providers import settings
    return settings.names()


def cmd_prices(args) -> int:
    data, work, out = Path(args.data).expanduser(), Path(args.work).expanduser(), Path(args.out).expanduser()
    lines, _ = read_logs(log_files(data, work))
    rows = price_rows(lines, load_sources(Path(args.sources).expanduser()), _names())
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "prices.csv", rows)
    print(f"{len(rows)} (model, day) prices -> {out / 'prices.csv'}")
    return 0


def cmd_requests(args) -> int:
    data, work, out = Path(args.data).expanduser(), Path(args.work).expanduser(), Path(args.out).expanduser()
    names = _names()
    lines, dup = read_logs(log_files(data, work))
    prices = prices_from_csv(Path(args.prices).expanduser())
    coll = collection_lists(data, work)
    req = request_rows(lines, prices, names, coll)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "requests.csv", req)
    summ = summary_rows(req, lines)
    for r in summ:
        r["duplicate_log_lines_dropped"] = dup if r["experiment"] == "all" and r["config"] == "all" else ""
    write_csv(out / "requests_summary.csv", summ)
    print(f"{len(req)} requests ({dup} duplicate log lines dropped) -> {out}")
    return 0


def cmd_tables(args) -> int:
    from llmdw.benchmark import eval_time as et
    from llmdw.scoring import rich_models_large as xe, results
    from llmdw.search import relabel

    data, work, out = Path(args.data).expanduser(), Path(args.work).expanduser(), Path(args.out).expanduser()
    names = _names()
    results.NAMES.clear()
    results.NAMES.update(names)
    relabel.configure(int(args.family_orders))
    configs = list(names)
    lines, _ = read_logs(log_files(data, work))
    coll_of = collection_lists(data, work)
    req = read_requests(Path(args.requests).expanduser())
    prices = prices_from_csv(Path(args.prices).expanduser()) if args.prices else {}
    cells = answer_cells(lines, req, prices, coll_of)
    pred = per_instance_prediction(cells)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "predictions.csv", prediction_rows(cells, names))

    t_rows = []
    T = eval_time_from_csv(Path(args.eval_time).expanduser(), "templates", COLL_TEMPLATES, args.eval_time) or []
    R = eval_time_from_csv(Path(args.eval_time).expanduser(), "rich_models", COLL_RICH, args.eval_time) or []
    L = eval_time_large(Path(args.eval_time_large_paths).expanduser() if args.eval_time_large_paths else None,
                        Path(args.eval_time_large_runs).expanduser() if args.eval_time_large_runs else None, COLL_LARGE)
    t_rows += T + R
    if L:
        t_rows += L
    else:
        t_rows.append({"collection": COLL_LARGE, "class": "all", "definition": "not available: the GCG root "
                       "evaluations of the visited sets (llmdw.benchmark.eval_time --set large) were not run",
                       "source": args.eval_time_large_paths or ""})

    def t_of(rows):
        return next((r.get("mean_seconds") for r in rows if r["class"] == "all"), None) if rows else None

    net = []
    rec = et.Recorder()
    for phase, coll, trows in (("templates", COLL_TEMPLATES, T), ("miplib", COLL_MIPLIB, None)):
        V = values_b(data, phase, configs, rec if phase == "miplib" else None)
        if phase == "miplib":
            visited = {n: set(rec.sets.get(n, {})) for n in V["names"]}
            trows = miplib_eval_time(data, V["names"], visited)
            t_rows += trows
        net += net_rows(coll, V, pred, t_of(trows), trows[0]["definition"] if trows else "", names)
    pub_groups = xe.published_groups(data)
    V = values_e(data / xe.P_ROOT, data / xe.P_FIVE, {p: data / d for p, d in xe.P_ANSWERS.items()}, data,
                 "rich_models", configs, pub_groups)
    net += net_rows(COLL_RICH, V, pred, t_of(R), R[0]["definition"] if R else "", names)
    V = values_e(work / xe.E_ROOT, work / xe.E_FIVE, {p: work / d for p, d in xe.E_ANSWERS.items()},
                 work / xe.E_ROOT, "rich_models_large", list(CONFIGS_E), None)
    net += net_rows(COLL_LARGE, V, pred, t_of(L), L[0]["definition"] if L else "not available", names)
    B = bnp_eval_time(Path(args.bnp_runs).expanduser()) if args.bnp_runs else []
    t_rows += B
    if args.bench_root and B:
        V = values_bench(Path(args.bench_root).expanduser(), Path(args.bench_answers).expanduser(),
                         Path(args.bench_names).expanduser(), {c: names[c] for c in CONFIGS_E})
        net += net_rows(COLL_BENCH, V, pred, t_of(B), B[0]["definition"], names,
                        bnp_eval_time_per_instance(Path(args.bnp_runs).expanduser()))
        write_csv(out / "lookahead_time.csv", lookahead_time_rows(V))
    else:
        net += bench_rows(pred, B, names)
    write_csv(out / "eval_time.csv", t_rows)
    write_csv(out / "net_benefit.csv", net)
    tex = Path(args.tex_dir).expanduser() if args.tex_dir else out / "tables/new"
    tex.mkdir(parents=True, exist_ok=True)
    (tex / "experiment_G_net_benefit.tex").write_text(tex_body(net), encoding="utf-8")
    print(f"wrote predictions.csv, eval_time.csv, net_benefit.csv to {out} and the table body to {tex}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prices", "requests", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True, help="the released data (runs/, work/, prompt_groups.csv)")
        p.add_argument("--work", required=True, help="data/work of the released data (holds rich_models_large/, "
                                                     "rich_models_large_requests/ and no_joint_group/)")
        p.add_argument("--out", required=True)
        if name == "prices":
            p.add_argument("--sources", required=True, help="sources.json of the price sources")
        if name == "requests":
            p.add_argument("--prices", required=True, help="prices.csv written by the prices step")
        if name == "tables":
            p.add_argument("--requests", required=True)
            p.add_argument("--prices", required=True, help="prices.csv (cost of answers read from the cache)")
            p.add_argument("--eval-time", required=True, help="eval_time.csv of llmdw.benchmark.eval_time tables")
            p.add_argument("--eval-time-large-paths", help="paths.json of eval_time --set large")
            p.add_argument("--eval-time-large-runs", help="its runs/ folder")
            p.add_argument("--bnp-runs", help="bnp_runs.csv of the time-optimal decompositions")
            p.add_argument("--bench-root", help="root of the benchmark instances of the time-optimal decompositions (instances/<name>/)")
            p.add_argument("--bench-answers", help="folder of the original answers per configuration (prompt groups)")
            p.add_argument("--bench-names", help="list of the benchmark instances of the time-optimal decompositions")
            p.add_argument("--tex-dir")
            p.add_argument("--family-orders", default=str(FAMILY_ORDERS))
    args = ap.parse_args(argv)
    return {"prices": cmd_prices, "requests": cmd_requests, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())


def tex_from_csv(out: Path, tex: Path) -> list:
    """experiment_G_net_benefit.tex rebuilt from net_benefit.csv."""
    from llmdw.tables.experiments import typed_rows

    (Path(tex) / "experiment_G_net_benefit.tex").write_text(tex_body(typed_rows(Path(out) / "net_benefit.csv")),
                                                            encoding="utf-8")
    return ["experiment_G_net_benefit.tex"]
