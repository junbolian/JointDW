"""Tokens used per configuration so far, and a projection for the calls that are still to be made.

Paper: Appendix D: identical samples and retried attempts.

The answer files of a finished phase (for example the template phase of the support predictions) store the usage of every call and
the messages that were sent. From them this module measures, per configuration, prompt tokens per prompt
character and completion tokens per call. The prompts of the planned calls are built offline exactly as
support_run builds them, so their length is known; the projection multiplies their characters by the measured
ratio and adds the mean (and the 90th percentile) of the completion tokens per call. The ratio of projected to
used tokens converts the amount the endpoint charged for the finished phase into an estimate for the planned one.

The same records check the endpoint: a call whose prompt tokens exceed the configuration's median tokens per
character by more than 300 tokens and 30 percent is counted as a call with added text. The client rejects gross
cases at run time (llm_client.PromptInflation); this count also finds smaller ones.

When llm_retries.jsonl exists (next to LLMDW_LOG, or --retry-log), the report also counts the retried attempts per
configuration and reason (gateway, rate_limit, network, incomplete, inflation, other).

The report also counts, per configuration, the cells (instance and condition) in which two or more samples have
byte-identical answer text (content_sha256 of the answer file), a sign that the endpoint served a stored answer
instead of an independent sample. For answer files without content_sha256, two samples count
as identical when their parsed answers and completion tokens are equal. A configuration in which more than 2 percent
(IDENTICAL_STOP_SHARE) of its cells with two or more samples have byte-identical samples is marked STOP (`python -m llmdw.providers.usage --identical-check DIR` exits with
code 3).

--configs limits the report to the named configurations (default: every subdirectory of --runs).

Usage
-----
python -m llmdw.providers.usage --runs data/work/support_templates --plan-mps-dir data/work/mps --plan-dataset MIPLIB-NL/dataset \
    --plan-instances-from data/work/oracle --plan-out data/work/support_miplib --samples 3
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from llmdw.prompts import counterfactual
from llmdw.instances.families import read_model, summarize
from llmdw.prompts.propose import PROMPTS, build_messages, load_template
from llmdw.providers.support_run import result_path

CONDITIONS = ["original", "names_only", "permuted", "anonymized"]


def planned_messages(summary: dict, instance_dir: str | None, condition: str, k: int = 3, time_limit: float = 120.0,
                     seed: int = 0) -> list[dict]:
    """The messages that propose(..., prompt_version="v2", condition=condition) sends."""
    maps = counterfactual.maps_for(summary, condition, seed)
    template = load_template(PROMPTS / "decomp_v2.md")
    return build_messages(summary, instance_dir, k, time_limit, None, None, template=template, maps=maps,
                          withhold=counterfactual.withhold_text(condition))


def used(run_dir: Path) -> dict:
    """Calls, tokens and prompt characters of every answer file under run_dir (one configuration)."""
    calls, no_usage, prompt_tok, prompt_chars, completion, per_call = 0, 0, 0, 0, [], []
    for path in sorted(run_dir.glob("*/*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        usage = rec.get("usage") or {}
        calls += 1
        if not usage.get("prompt_tokens"):
            no_usage += 1
            continue
        chars = sum(len(m["content"]) for m in rec.get("messages", []))
        prompt_tok += int(usage["prompt_tokens"])
        prompt_chars += chars
        completion.append(int(usage.get("completion_tokens") or 0))
        if chars:
            per_call.append((int(usage["prompt_tokens"]), chars))
    completion.sort()
    n = len(completion)
    ratios = sorted(pt / ch for pt, ch in per_call)
    median = ratios[len(ratios) // 2] if ratios else None
    added = [pt - median * ch for pt, ch in per_call
             if median and pt - median * ch > 300 and pt > 1.3 * median * ch]
    return {"calls": calls, "calls_without_usage": no_usage, "prompt_tokens": prompt_tok,
            "completion_tokens": sum(completion), "tokens_per_char": prompt_tok / prompt_chars if prompt_chars else None,
            "completion_mean": sum(completion) / n if n else None,
            "completion_p90": completion[min(n - 1, int(0.9 * n))] if n else None,
            "calls_with_added_text": len(added), "largest_added_tokens": round(max(added)) if added else 0}


IDENTICAL_STOP_SHARE = 0.02


def identical_samples(run_dir: Path) -> dict:
    """Cells (instance, condition) of one configuration with two or more samples, and those among them in which at
    least two samples have byte-identical answer text; by_hash counts the cells whose every file has content_sha256."""
    cells = defaultdict(dict)
    for path in sorted(run_dir.glob("*/*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if "content_sha256" in rec:
            key = ("sha256", rec["content_sha256"])
        else:
            key = ("answer", json.dumps(rec.get("answer"), sort_keys=True),
                   (rec.get("usage") or {}).get("completion_tokens"))
        cells[(rec["instance"], rec["condition"])][rec.get("sample", 0)] = key
    multi = {c: v for c, v in cells.items() if len(v) >= 2}
    same = []
    for (inst, cond), v in sorted(multi.items()):
        groups = defaultdict(list)
        for sample, key in sorted(v.items()):
            groups[key].append(sample)
        alike = [g for g in groups.values() if len(g) >= 2]
        if alike:
            same.append(f"{inst}:{cond} samples {', '.join(map(str, alike[0]))}")
    by_hash = sum(1 for v in multi.values() if all(k[0] == "sha256" for k in v.values()))
    share = len(same) / len(multi) if multi else 0.0
    return {"cells": len(multi), "identical": len(same), "by_hash": by_hash, "examples": same[:3], "share": share,
            "stop": share > IDENTICAL_STOP_SHARE}


RETRY_REASONS = ["gateway", "rate_limit", "network", "incomplete", "inflation", "other"]


def retries(path: Path) -> dict:
    """Retried attempts per configuration and reason, from llm_retries.jsonl (configuration = the config field,
    or the model when a line has none)."""
    counts = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cfg = rec.get("config") or f"model {rec.get('model')}"
        by_reason = counts.setdefault(cfg, dict.fromkeys(RETRY_REASONS, 0))
        reason = rec.get("reason") if rec.get("reason") in by_reason else "other"
        by_reason[reason] += 1
    return counts


def plan(mps_dir: Path, dataset: Path, instances_from: Path, out: Path | None, samples: int,
         conditions: list[str]) -> dict:
    """Calls of a support_run over the instances of an oracle root that have no answer file under out yet."""
    dirs = sorted(p for p in instances_from.iterdir() if p.is_dir() and (p / "oracle.json").exists())
    calls, chars, instances = 0, 0, 0
    for d in dirs:
        mps = mps_dir / f"{d.name}.mps.gz"
        if not mps.exists():
            continue
        instances += 1
        summary = summarize(read_model(str(mps)))
        for cond in conditions:
            if not counterfactual.applicable(summary, cond):
                continue
            missing = [s for s in range(samples) if out is None or not result_path(out, d.name, cond, s).exists()]
            if not missing:
                continue
            n_chars = sum(len(m["content"]) for m in planned_messages(summary, str(dataset / d.name), cond))
            calls += len(missing)
            chars += len(missing) * n_chars
    return {"instances": instances, "calls": calls, "prompt_chars": chars}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="directory with one subdirectory per configuration (its model identifier)")
    ap.add_argument("--identical-check", help="answer directory of one configuration and phase: print its identical "
                    "samples and exit 3 when more than 2 percent of its cells have them")
    ap.add_argument("--plan-mps-dir")
    ap.add_argument("--plan-dataset")
    ap.add_argument("--plan-instances-from")
    ap.add_argument("--plan-out", help="answer root of the planned phase; existing answers are not counted")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--retry-log", help="retried attempts (default: llm_retries.jsonl next to LLMDW_LOG)")
    ap.add_argument("--configs", help="comma-separated configurations to report (default: every subdirectory)")
    args = ap.parse_args(argv)
    if args.identical_check:
        d = identical_samples(Path(args.identical_check))
        print(json.dumps(d))
        return 3 if d["stop"] else 0
    if not args.runs:
        ap.error("--runs is required")
    conds = [c for c in args.conditions.split(",") if c]
    configs = sorted(p for p in Path(args.runs).iterdir() if p.is_dir())
    if args.configs:
        wanted = [c for c in args.configs.split(",") if c]
        configs = [p for p in configs if p.name in wanted]
    print("| configuration | calls | prompt tokens | completion tokens | completion per call, mean and p90 | "
          "calls with added text, largest addition |")
    print("|---|---:|---:|---:|---:|---:|")
    stats = {}
    for c in configs:
        s = used(c)
        stats[c.name] = s
        mean = f"{s['completion_mean']:.0f}" if s["completion_mean"] is not None else "n/a"
        p90 = s["completion_p90"] if s["completion_p90"] is not None else "n/a"
        print(f"| {c.name} | {s['calls']} | {s['prompt_tokens']} | {s['completion_tokens']} | {mean}, {p90} | "
              f"{s['calls_with_added_text']}, {s['largest_added_tokens']} |")
        if s["calls_without_usage"]:
            print(f"|   {c.name}: {s['calls_without_usage']} calls returned no usage and are not counted | | | | | |")
    print()
    print("| configuration | cells with two or more samples | cells with two or more byte-identical samples | "
          "share | cells checked by content hash | examples |")
    print("|---|---:|---:|---:|---:|---|")
    for c in configs:
        d = identical_samples(c)
        flag = " STOP (over 2%)" if d["stop"] else ""
        print(f"| {c.name} | {d['cells']} | {d['identical']} | {100 * d['share']:.1f}%{flag} | {d['by_hash']} | "
              f"{'; '.join(d['examples']) or '-'} |")
    retry_log = Path(args.retry_log) if args.retry_log else \
        Path(os.environ.get("LLMDW_LOG", "runs/llm_calls.jsonl")).with_name("llm_retries.jsonl")
    if retry_log.exists():
        print()
        print("| configuration | retried attempts | " + " | ".join(RETRY_REASONS) + " |")
        print("|---|---:|" + "---:|" * len(RETRY_REASONS))
        names = {c.name for c in configs}
        for cfg, by_reason in sorted(retries(retry_log).items()):
            if args.configs and cfg not in names:
                continue
            print(f"| {cfg} | {sum(by_reason.values())} | " + " | ".join(str(by_reason[r]) for r in RETRY_REASONS) + " |")
    if not args.plan_instances_from:
        return 0
    print()
    print("| configuration | planned calls | projected prompt tokens | projected completion tokens, mean and p90 | "
          "projected total over used total, mean and p90 |")
    print("|---|---:|---:|---:|---:|")
    for c in configs:
        out = Path(args.plan_out) / c.name if args.plan_out else None
        p = plan(Path(args.plan_mps_dir), Path(args.plan_dataset), Path(args.plan_instances_from), out, args.samples,
                 conds)
        s = stats[c.name]
        if s["tokens_per_char"] is None:
            print(f"| {c.name} | {p['calls']} | n/a | n/a | n/a |")
            continue
        prompt = p["prompt_chars"] * s["tokens_per_char"]
        comp_mean, comp_p90 = p["calls"] * s["completion_mean"], p["calls"] * s["completion_p90"]
        used_total = s["prompt_tokens"] + s["completion_tokens"]
        print(f"| {c.name} | {p['calls']} | {prompt:.0f} | {comp_mean:.0f}, {comp_p90:.0f} | "
              f"{(prompt + comp_mean) / used_total:.1f}, {(prompt + comp_p90) / used_total:.1f} |")
    print(f"\nPlanned instances: {p['instances']} (oracle instances with an MPS file). The last column multiplies "
          f"what the endpoint charged for the finished phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
