"""Experiment C tables: standard formulations on benchmark data, the set of each method and its branch-and-price run.

Read only: oracle.json, classification.json, bnp_plan.json and the branch-and-price rows of every instance
(llmdw.benchmark.truth, llmdw.benchmark.bnp); nothing is evaluated or solved here. A missing branch-and-price row is an
error.

Methods and their sets (bnp_plan.json): the true best set, the lookahead at the HiGHS dual (single dual), the lookahead
at the best of five optimal duals (five duals), GCG's default decomposition, and GPT-6.1 Sol and Claude Sonnet 5.5
(original presentation, three samples, the anchor of JointDW in each sample).

Per instance and method:
  hit@1        the method's set is a set of maximal gain among the evaluated sets (ties included); GCG default: its
               detected decomposition is family-aligned and its family set is such a set
  gain         the GCG root bound of the method's set (census setting, propagation off) as a share of the LP gap,
               (z(T) - z_LP) / (z* - z_LP); GCG default: the root bound of GCG's detected decomposition in the same
               setting (baseline gcg_spfwh of oracle.json); a set that was not evaluated has no gain
  time, gap    GCG branch-and-price of the method's set (1 thread, seed 0, 1800 s; GCG default: GCG as shipped):
               solving time and final gap; a run that failed twice counts as unsolved at 1800 s with the root bound as
               its bound (llmdw.benchmark.bnp.run_one) and is listed separately in every table (n_failed_twice);
               a final gap below 1e-9 (floating-point residue of a solved run) is 0
  solved       status optimal
A sample of a language model without a set (no joint support and no reformulation) has gain 0, is no hit, and has no
branch-and-price run (counted in n_samples_without_set). For the language models, every per-instance value is the mean
over its samples before the bootstrap; the solved count of a language model is the sum over instances of the share of
its samples that were solved, and its count of instances with gain 0 is the sum over instances of the share of its
samples whose set has gain 0.

Comparisons (paired over instances, cluster bootstrap over prompt groups, llmdw.scoring.cluster, 10,000 resamples, seed
0; asterisk when the 95% interval excludes 0):
  JointDW (each model) minus the single-dual lookahead, the five-dual lookahead and GCG default: time difference (s);
  JointDW (each model) minus the single-dual lookahead, the five-dual lookahead and GCG default, and the five-dual
  lookahead minus GCG default: hit@1 and the gain of the first set (benchmark_paired_hit1.csv, benchmark_paired_gain.csv);
  each method against the true best set: the paired time ratio (t + 1 s) / (t_best + 1 s), reported as the geometric
  mean over instances (exp of the mean log ratio; interval and asterisk on the mean log ratio, standard error of the
  mean log ratio), and the paired gap difference (gap minus gap of the true best set).
Rows per class and for all classes pooled ("all").

Usage
-----
python -m llmdw.benchmark.tables --root work/benchmark_std --names-file work/benchmark_std/full_set/all_instances.txt \
    --answers "GPT-6.1 Sol=work/benchmark_std/answers/gpt-6.1-sol" "Claude Sonnet 5.5=..." --first-batch ls_F1,... \
    --out work/benchmark_std/results
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from llmdw.benchmark import bnp
from llmdw.scoring import cluster, results

CLASSES = ("lot_sizing", "network_design", "facility_location")
SINGLE = ("true best set", "lookahead (single dual)", "lookahead (five duals)", "GCG default")
PLAN_KEY = {"true best set": "true_best", "lookahead (single dual)": "lookahead",
            "lookahead (five duals)": "lookahead_five_duals", "GCG default": "gcg_detected"}
TIME_SHIFT = 1.0
ZERO = 1e-9
GAP_ZERO = 1e-9


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _row_or_error(root: Path, name: str, T) -> dict:
    d = root / "instances" / name / "bnp"
    key = "gcg_default" if T == "gcg_default" else bnp.set_key(T)
    from llmdw.gcg.oracle import key_tag
    tag = "gcg_default" if T == "gcg_default" else key_tag(key)
    first = d / f"{tag}.row.json"
    if not first.exists():
        raise FileNotFoundError(f"{name}: no branch-and-price row for {key}")
    if not json.loads(first.read_text())["ok"] and not (d / f"{tag}.s1.row.json").exists():
        raise FileNotFoundError(f"{name}: failed run of {key} without its seed-1 repeat")
    return bnp.run_one((str(root), name, T))      # every row exists: run_one only reads


def instance_values(root: Path, name: str, configs: list, plan_files=("bnp_plan.json",)) -> dict:
    """plan_files: the first gives the methods without samples and its configurations; each further file adds the
    configurations of configs that the earlier files do not hold (bnp_plan_data.json: the data variant)."""
    d = root / "instances" / name
    o = json.loads((d / "oracle.json").read_text())
    p = json.loads((d / plan_files[0]).read_text())
    for extra in plan_files[1:]:
        q = json.loads((d / extra).read_text())
        p["methods"].update({c: q["methods"][c] for c in configs if c not in p["methods"] and c in q["methods"]})
    cl = json.loads((d / "classification.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    from llmdw.scoring import bounds

    bounds.annotate(o)                     # an invalid recorded bound gives no gain (llmdw.scoring.bounds)
    fams, lp, ref = o["families"], o["lp_bound"], o["reference_optimal_value"]
    scale = abs(ref - lp)
    gains = {}
    for r in o["subsets"]:
        if r.get("linking_var_families"):
            continue
        T = bnp.set_key(set(fams) - set(r["master_families"]))
        gains[T] = ((r["dual_bound"] - lp) / scale) if (scale > 0 and r.get("dual_bound") is not None
                                                         and r.get("status") != "failed"
                                                         and not r.get("bound_invalid")) else None
    ties = {bnp.set_key(T) for T in p["methods"]["true_best_ties"]}
    jobs = {(bnp.set_key(T) if T != "gcg_default" else T): T for pf in plan_files for _, _, T in bnp.jobs_of(root, name, pf)}
    runs = {r["set"]: r for r in (_row_or_error(root, name, T) for T in jobs.values())}

    def bnp_of(key):
        r = runs.get(key)
        if r is None:
            return None
        gap = r["gap"] if r["gap"] is None or abs(r["gap"]) >= GAP_ZERO else 0.0
        return {"time": r["solving_time"], "gap": gap, "solved": bool(r.get("solved")),
                "failed_twice": bool(r.get("failed_twice"))}

    methods = {}
    for m in SINGLE:
        T = p["methods"].get(PLAN_KEY[m])
        key = bnp.set_key(T) if T else None
        if m == "GCG default":
            b = o["baselines"]["gcg_spfwh"]
            g = (b["dual_bound"] - lp) / scale if (b.get("ok") and b.get("dual_bound") is not None and scale > 0) else None
            methods[m] = [{"sample": None, "set": key, "hit": key in ties, "gain": g, **(bnp_of("gcg_default") or {})}]
        else:
            methods[m] = [{"sample": None, "set": key, "hit": key in ties, "gain": gains.get(key) if key else None,
                           **(bnp_of(key) or {})}]
    for c in configs:
        rows = []
        for x in p["methods"].get(c) or []:
            key = bnp.set_key(x["set"]) if x["set"] else None
            if key is None:
                rows.append({"sample": x["sample"], "set": None, "hit": False, "gain": 0.0})
            else:
                rows.append({"sample": x["sample"], "set": key, "hit": key in ties, "gain": gains.get(key),
                             **(bnp_of(key) or {})})
        methods[c] = rows
    return {"name": name, "class": meta["class"], "rows": meta["rows"], "columns": meta["columns"],
            "binaries": meta["binaries"], "families": fams, "lp_bound": lp, "optimum": ref,
            "proved_optimal": o["reference_proved_optimal"], "lp_gap": (ref - lp) / abs(ref),
            "classification": cl["classification"],
            "zero_marginal_group": any(g["size"] >= 2 and g["status"] in ("certain", "possible") for g in cl["groups"]),
            "groups": cl["groups"], "gains": gains, "true_best": bnp.set_key(p["methods"]["true_best"]),
            "true_best_ties": sorted(ties), "methods": methods, "runs": runs,
            "baselines_run": o.get("baselines_run", list(o["baselines"]))}


def per_instance(v: dict, m: str) -> dict:
    """Mean over the samples of one method on one instance (a single row for the other methods)."""
    rows = v["methods"][m]
    with_run = [r for r in rows if r.get("time") is not None]
    return {"hit": _mean([float(r["hit"]) for r in rows]), "gain": _mean([r["gain"] for r in rows]),
            "gain_zero": _mean([float(abs(r["gain"]) <= ZERO) for r in rows if r["gain"] is not None]),
            "time": _mean([r["time"] for r in with_run]), "gap": _mean([r["gap"] for r in with_run]),
            "solved": _mean([float(r["solved"]) for r in with_run]),
            "failed_twice": any(r.get("failed_twice") for r in rows),
            "no_set": sum(1 for r in rows if r["set"] is None)}


def _se(values: dict) -> dict:
    vals = {n: x for n, x in values.items() if x is not None}
    lo, hi, se = results.boot_ci_se(vals) if vals else (None, None, None)
    return {"mean": _mean(vals.values()), "se": se, "lo": lo, "hi": hi, "n": len(vals)}


def _paired(a: dict, b: dict) -> dict:
    both = sorted(n for n in a if a[n] is not None and b.get(n) is not None)
    diffs = {n: a[n] - b[n] for n in both}
    lo, hi, se = results.boot_ci_se(diffs) if diffs else (None, None, None)
    return {"n_instances": len(both), "mean_a": _mean(a[n] for n in both), "mean_b": _mean(b[n] for n in both),
            "mean_difference": _mean(diffs.values()), "se_difference": se, "lo": lo, "hi": hi,
            "asterisk": "*" if lo is not None and (lo > 0 or hi < 0) else ""}


def _ratio(a: dict, b: dict) -> dict:
    both = sorted(n for n in a if a[n] is not None and b.get(n) is not None)
    logs = {n: math.log((a[n] + TIME_SHIFT) / (b[n] + TIME_SHIFT)) for n in both}
    lo, hi, se = results.boot_ci_se(logs) if logs else (None, None, None)
    m = _mean(logs.values())
    return {"n_instances": len(both), "geometric_mean_ratio": math.exp(m) if m is not None else None,
            "ratio_lo": math.exp(lo) if lo is not None else None, "ratio_hi": math.exp(hi) if hi is not None else None,
            "mean_log_ratio": m, "se_log_ratio": se,
            "asterisk": "*" if lo is not None and (lo > 0 or hi < 0) else ""}


def tables(vals: dict, configs: list) -> dict:
    methods = list(SINGLE) + list(configs)
    pi = {n: {m: per_instance(v, m) for m in methods} for n, v in vals.items()}
    out = {"zero_marginal": [], "methods": [], "paired_jointdw": [], "paired_true_best": [], "paired_hit1": [],
           "paired_gain": []}
    for cls in CLASSES + ("all",):
        names = sorted(n for n, v in vals.items() if cls == "all" or v["class"] == cls)
        if not names:
            continue
        z = [vals[n]["zero_marginal_group"] for n in names]
        out["zero_marginal"].append({"class": cls, "n_instances": len(names), "with_zero_marginal_group": sum(z),
                                     "share": sum(z) / len(z),
                                     "certain": sum(vals[n]["classification"] == "certain" for n in names),
                                     "possible": sum(vals[n]["classification"] == "possible" for n in names)})
        for m in methods:
            row = {"class": cls, "method": m, "n_instances": len(names)}
            for k in ("hit", "gain", "time", "gap"):
                s = _se({n: pi[n][m][k] for n in names})
                row.update({f"{k}_mean": s["mean"], f"{k}_se": s["se"], f"{k}_n": s["n"]})
            row["gain_zero_instances"] = sum(pi[n][m]["gain_zero"] or 0.0 for n in names
                                             if pi[n][m]["gain_zero"] is not None)
            row["solved"] = sum(pi[n][m]["solved"] for n in names if pi[n][m]["solved"] is not None)
            row["n_failed_twice"] = sum(pi[n][m]["failed_twice"] for n in names)
            row["failed_twice_instances"] = " ".join(n for n in names if pi[n][m]["failed_twice"])
            row["n_samples_without_set"] = sum(pi[n][m]["no_set"] for n in names)
            out["methods"].append(row)
        for c in configs:
            for b in ("lookahead (single dual)", "lookahead (five duals)", "GCG default"):
                r = _paired({n: pi[n][c]["time"] for n in names}, {n: pi[n][b]["time"] for n in names})
                out["paired_jointdw"].append({"class": cls, "configuration": c, "against": b, **r,
                                              "failed_twice_instances": " ".join(
                                                  n for n in names if pi[n][c]["failed_twice"]
                                                  or pi[n][b]["failed_twice"])})
        # paired differences of hit@1 and of the gain of the first set: JointDW against each baseline, and the
        # five-dual lookahead against GCG's detector
        pairs = [(c, b) for c in configs for b in ("lookahead (single dual)", "lookahead (five duals)", "GCG default")]
        pairs.append(("lookahead (five duals)", "GCG default"))
        for a, b in pairs:
            for key, table in (("hit", "paired_hit1"), ("gain", "paired_gain")):
                r = _paired({n: pi[n][a][key] for n in names}, {n: pi[n][b][key] for n in names})
                out[table].append({"class": cls, "configuration": a, "against": b, **r})
        for m in methods[1:]:
            t = _ratio({n: pi[n][m]["time"] for n in names}, {n: pi[n]["true best set"]["time"] for n in names})
            g = _paired({n: pi[n][m]["gap"] for n in names}, {n: pi[n]["true best set"]["gap"] for n in names})
            out["paired_true_best"].append({"class": cls, "method": m, **t,
                                            "gap_mean_difference": g["mean_difference"],
                                            "gap_se_difference": g["se_difference"], "gap_lo": g["lo"],
                                            "gap_hi": g["hi"], "gap_asterisk": g["asterisk"],
                                            "failed_twice_instances": " ".join(
                                                n for n in names if pi[n][m]["failed_twice"]
                                                or pi[n]["true best set"]["failed_twice"])})
    return out


def _write(path: Path, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def report(vals: dict, tabs: dict, configs: list, first_batch: list, selection: dict | None) -> str:
    L = ["# Experiment C: standard formulations on benchmark data", ""]
    L.append(__doc__.split("Usage")[0].strip())
    L += ["", "## Instances", ""]
    for cls in CLASSES:
        names = [n for n, v in vals.items() if v["class"] == cls]
        L.append(f"- {cls}: {len(names)} instances, {sum(vals[n]['proved_optimal'] for n in names)} with a proved "
                 f"optimum; without a proved optimum: "
                 f"{' '.join(n for n in names if not vals[n]['proved_optimal']) or 'none'}")
    if selection:
        L += ["", "Exclusions at selection (selection.json):", ""]
        for cls, v in selection.items():
            ex = [x for x in v["screened"] if x.get("excluded")]
            L.append(f"- {cls}: {v['screened'] if isinstance(v['screened'], int) else len(v['screened'])} screened, "
                     f"{len(ex)} excluded: " + "; ".join(f"{x['name']} ({x['excluded']})" for x in ex))
    L += ["", f"Instances of the first batch ({', '.join(first_batch)}): computed first, with the same settings "
          "(propagation off in every GCG root evaluation, SCIP defaults in branch-and-price, 1800 s, the failure rule); "
          "nothing in their evaluation changed, so they were not recomputed. Their oracle.json also holds the strong-"
          "score detection baseline, which no table uses; the other instances do not evaluate it. Their answers "
          "were parsed before the answer-name rule; re-read with the rule, every joint support and every "
          "anchor is unchanged (only candidate lists of ten answers of Claude Sonnet 5.5 change), so no number here "
          "depends on it.", ""]
    L += ["## Zero-marginal groups at the empty set", "", "| class | instances | with group | share | certain | possible |",
          "|---|---|---|---|---|---|"]
    for r in tabs["zero_marginal"]:
        L.append(f"| {r['class']} | {r['n_instances']} | {r['with_zero_marginal_group']} | {_f(r['share'])} | "
                 f"{r['certain']} | {r['possible']} |")
    L += ["", "## Methods: hit@1, gain of the first set, branch-and-price", "",
          "| class | method | hit@1 | gain (share of LP gap) | gain 0 (instances) | time (s) | final gap | solved | "
          "failed twice |", "|---|---|---|---|---|---|---|---|---|"]
    for r in tabs["methods"]:
        L.append(f"| {r['class']} | {r['method']} | {_f(r['hit_mean'])} ± {_f(r['hit_se'])} | "
                 f"{_f(r['gain_mean'])} ± {_f(r['gain_se'])} | {_f(r['gain_zero_instances'], 2)} | "
                 f"{_f(r['time_mean'], 1)} ± {_f(r['time_se'], 1)} | {_f(r['gap_mean'], 5)} ± {_f(r['gap_se'], 5)} | "
                 f"{_f(r['solved'], 2)}/{r['n_instances']} | {r['n_failed_twice']} {r['failed_twice_instances']} |")
    L += ["", "## JointDW against the baselines: paired time difference (s)", "",
          "| class | configuration | against | n | mean JointDW | mean baseline | difference ± SE | 95% interval | "
          "failed twice |", "|---|---|---|---|---|---|---|---|---|"]
    for r in tabs["paired_jointdw"]:
        L.append(f"| {r['class']} | {r['configuration']} | {r['against']} | {r['n_instances']} | {_f(r['mean_a'], 1)} | "
                 f"{_f(r['mean_b'], 1)} | {_f(r['mean_difference'], 1)} ± {_f(r['se_difference'], 1)}{r['asterisk']} | "
                 f"[{_f(r['lo'], 1)}, {_f(r['hi'], 1)}] | {r['failed_twice_instances']} |")
    L += ["", "## Each method against the true best set", "",
          "| class | method | n | time ratio (geometric mean) | 95% interval | SE of mean log ratio | "
          "gap difference ± SE | 95% interval | failed twice |", "|---|---|---|---|---|---|---|---|---|"]
    for r in tabs["paired_true_best"]:
        L.append(f"| {r['class']} | {r['method']} | {r['n_instances']} | {_f(r['geometric_mean_ratio'])}{r['asterisk']} | "
                 f"[{_f(r['ratio_lo'])}, {_f(r['ratio_hi'])}] | {_f(r['se_log_ratio'])} | "
                 f"{_f(r['gap_mean_difference'], 5)} ± {_f(r['gap_se_difference'], 5)}{r['gap_asterisk']} | "
                 f"[{_f(r['gap_lo'], 5)}, {_f(r['gap_hi'], 5)}] | {r['failed_twice_instances']} |")
    fails = [(n, k, r) for n, v in sorted(vals.items()) for k, r in v["runs"].items() if r.get("failed_twice")]
    retried = [(n, k, r) for n, v in sorted(vals.items()) for k, r in v["runs"].items() if r.get("retried_after_failure")]
    L += ["", "## Branch-and-price runs that failed", ""]
    L += [f"- {n} {{{k}}}: failed twice ({r.get('failure_seed0')}; {r.get('failure_seed1')}); counted unsolved at "
          f"1800 s, bound {r.get('root_bound')} ({r.get('root_bound_source')}), final gap {_f(r.get('gap'), 6)}"
          for n, k, r in fails] or ["- none failed twice"]
    L += [f"- {n} {{{k}}}: failed with seed 0 ({r.get('failure_seed0')}), solved with seed 1" for n, k, r in retried]
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--answers", nargs="*", default=[], help="NAME=DIR, the official name and the answer folder")
    ap.add_argument("--first-batch", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = Path(args.names_file).read_text().split()
    answers = dict(a.split("=", 1) for a in args.answers)
    configs = list(answers)
    prompts = {}
    for n in names:
        prompts[n] = {}
        for c, adir in answers.items():
            f = Path(adir) / n / "original.json"
            if f.exists():
                prompts[n][c] = json.loads(f.read_text())["messages"]
    groups = cluster.prompt_groups(prompts)
    cluster.set_groups(groups)
    vals = {n: instance_values(root, n, configs) for n in names}
    tabs = tables(vals, configs)
    _write(out / "benchmark_instances.csv", [
        {k: v[k] for k in ("name", "class", "rows", "columns", "binaries", "lp_bound", "optimum", "proved_optimal",
                           "lp_gap", "classification", "zero_marginal_group", "true_best")}
        | {"true_best_ties": " ".join(v["true_best_ties"]), "first_batch": v["name"] in args.first_batch.split(","),
           "baselines_run": " ".join(v["baselines_run"]), "prompt_group": groups.get(v["name"], v["name"]),
           "groups": "; ".join(f"{'|'.join(g['group'])} ({g['status']}, {g['gain']:.6g})" for g in v["groups"])}
        for v in vals.values()])
    _write(out / "benchmark_gains.csv", [{"instance": n, "set": T, "gain_share": g}
                                         for n, v in vals.items() for T, g in sorted(v["gains"].items())])
    _write(out / "benchmark_predictions.csv", [{"instance": n, "class": v["class"], "method": m, **r}
                                               for n, v in vals.items() for m, rows in v["methods"].items()
                                               for r in rows])
    _write(out / "benchmark_branch_and_price.csv", [
        {"instance": n, **{k: r.get(k) for k in ("set", "status", "solving_time", "wall_seconds", "gap", "nodes",
                                                  "root_bound", "root_bound_source", "primal_bound", "dual_bound",
                                                  "solved", "error", "failed_twice", "retried_after_failure",
                                                  "failure_seed0", "failure_seed1", "propagation")}}
        for n, v in vals.items() for r in v["runs"].values()])
    _write(out / "benchmark_zero_marginal.csv", tabs["zero_marginal"])
    _write(out / "benchmark_methods.csv", tabs["methods"])
    _write(out / "benchmark_paired_jointdw.csv", tabs["paired_jointdw"])
    _write(out / "benchmark_paired_true_best.csv", tabs["paired_true_best"])
    _write(out / "benchmark_paired_hit1.csv", tabs["paired_hit1"])
    _write(out / "benchmark_paired_gain.csv", tabs["paired_gain"])
    sel = root / "selection.json"
    (out / "REPORT.md").write_text(report(vals, tabs, configs, [p for p in args.first_batch.split(",") if p],
                                          json.loads(sel.read_text()) if sel.exists() else None), encoding="utf-8")
    print(f"Experiment C tables: {len(vals)} instances -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
