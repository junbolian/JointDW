"""Experiment C, data variant: the answers of GPT-6.1 Sol and Claude Sonnet 5.5 to the data variant of the prompt (the
complete instance in CPLEX LP format) on the 59 benchmark instances, scored as in Experiment C (llmdw.benchmark.tables):
per sample the anchor of JointDW (first joint support, else the convexified families of the top-ranked reformulation) as
the set, hit@1 against the sets of maximal gain among the evaluated sets of at most two families, the gain of the set
(share of the LP gap) and the GCG branch-and-price run of the set (bnp_plan_data.json; runs on identical sets are
shared with the recorded runs). Read only, like llmdw.benchmark.tables.

Every instance has its own data prompt, so the bootstrap draws instances (10,000 resamples, seed 0; asterisk when the
95% interval excludes 0), also for the comparisons with the original presentation.

Outputs:
  benchmark_data_predictions.csv  per instance, configuration and sample: the set, whether a joint support was
                                  returned, hit, gain, branch-and-price time, final gap and status, and the prompt
                                  tokens reported for the request (the exact count sent, from the answer's usage)
  benchmark_data_methods.csv      per class and pooled, each method (the four methods without samples, the two
                                  configurations under the original presentation and under the data variant): hit@1, gain,
                                  time, final gap, solved, samples without a set and without a joint support, and the
                                  geometric mean time ratio (t + 1 s) / (t_best + 1 s) against the true best set with its
                                  interval and asterisk
  benchmark_data_paired.csv       per class and pooled, each configuration under the data variant minus the single-dual
                                  lookahead, the five-dual lookahead, GCG's detector and the same configuration under the
                                  original presentation: paired differences of hit@1, gain and time

Usage
-----
python -m llmdw.benchmark.data_tables --root data/work/benchmark_std --names-file .../full_set/all_instances.txt \
    --answers "GPT-6.1 Sol=.../answers/gpt-6.1-sol" "Claude Sonnet 5.5=.../answers/claude-sonnet-5-5" \
    --data-answers "GPT-6.1 Sol=.../answers_data/gpt-6.1-sol" "Claude Sonnet 5.5=.../answers_data/claude-sonnet-5-5" \
    --out results/experiment_C
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmdw.benchmark import tables as T
from llmdw.scoring import cluster

DATA = ", data"
BASELINES = ("lookahead (single dual)", "lookahead (five duals)", "GCG default")


def _usage(adir: Path, name: str, sample: int) -> dict:
    f = adir / name / ("original.json" if sample == 0 else f"original.s{sample}.json")
    a = json.loads(f.read_text(encoding="utf-8"))
    u = a.get("usage") or {}
    return {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "finish_reason": a.get("finish_reason"), "content_sha256": a.get("content_sha256")}


def build(root: Path, names: list, answers: dict, data_answers: dict) -> dict:
    cluster.set_groups({})                      # every instance its own cluster
    configs = list(answers) + [c + DATA for c in data_answers]
    vals = {n: T.instance_values(root, n, configs, ("bnp_plan.json", "bnp_plan_data.json")) for n in names}
    tabs = T.tables(vals, configs)
    methods = list(T.SINGLE) + configs
    pi = {n: {m: T.per_instance(v, m) for m in methods} for n, v in vals.items()}
    preds = []
    for n, v in vals.items():
        plan = json.loads((root / "instances" / n / "bnp_plan_data.json").read_text())
        for c, adir in data_answers.items():
            fs = {x["sample"]: x["first_support"] for x in plan["methods"][c + DATA]}
            for r in v["methods"][c + DATA]:
                preds.append({"instance": n, "class": v["class"], "configuration": c, "presentation": "data",
                              **r, "joint_support": bool(fs.get(r["sample"])),
                              **_usage(Path(adir), n, r["sample"])})
    ratio = {(r["class"], r["method"]): r for r in tabs["paired_true_best"]}
    meth = []
    for r in tabs["methods"]:
        x = ratio.get((r["class"], r["method"]), {})
        row = {**r, **{k: x.get(k) for k in ("geometric_mean_ratio", "ratio_lo", "ratio_hi", "se_log_ratio",
                                             "asterisk")}}
        row["ratio_asterisk"] = row.pop("asterisk")
        if r["method"].endswith(DATA):
            names_c = [n for n in vals if r["class"] in ("all", vals[n]["class"])]
            row["n_samples_without_joint_support"] = sum(
                not p["joint_support"] for p in preds
                if p["configuration"] + DATA == r["method"] and p["instance"] in names_c)
            row["n_samples"] = sum(1 for p in preds if p["configuration"] + DATA == r["method"]
                                   and p["instance"] in names_c)
        meth.append(row)
    paired = []
    for cls in T.CLASSES + ("all",):
        names_c = sorted(n for n, v in vals.items() if cls == "all" or v["class"] == cls)
        for c in data_answers:
            for b in BASELINES + (c,):
                for key, metric in (("hit", "hit@1"), ("gain", "gain"), ("time", "time")):
                    r = T._paired({n: pi[n][c + DATA][key] for n in names_c}, {n: pi[n][b][key] for n in names_c})
                    paired.append({"class": cls, "configuration": c, "presentation": "data",
                                   "against": b if b != c else f"{c}, original", "metric": metric, **r})
    return {"predictions": preds, "methods": meth, "paired": paired}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--answers", nargs="*", default=[], help="NAME=DIR, original presentation")
    ap.add_argument("--data-answers", nargs="*", default=[], help="NAME=DIR, data variant")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = Path(args.names_file).read_text().split()
    res = build(root, names, dict(a.split("=", 1) for a in args.answers),
                dict(a.split("=", 1) for a in args.data_answers))
    T._write(out / "benchmark_data_predictions.csv", res["predictions"])
    T._write(out / "benchmark_data_methods.csv", res["methods"])
    T._write(out / "benchmark_data_paired.csv", res["paired"])
    print(f"Experiment C, data variant: {len(names)} instances -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
