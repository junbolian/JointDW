"""The sets of the methods and GCG branch-and-price on the instances of llmdw.benchmark.extra_classes
(generalized assignment and bin packing; the instances of instances_used_selected.txt).

Scoring settings of the experiment (applied in memory; oracle.json is not changed):
- reference value z* (llmdw.benchmark.validity): the best of the SCIP reference (oracle.json reference_optimal_value),
  for bin packing the best-known number of bins stated in the file (meta.json size.best_known_bins_in_file), and a
  branch-and-price solution accepted under rule 3 of the validity rule (reference_accepted.json); it is the primal
  bound of the bound validity rule (llmdw.scoring.bounds), the reference optimum of the LP gap and of the gain shares,
  and the reference of the validity rule of branch-and-price runs (every run is scored through
  llmdw.benchmark.validity.apply).
- sets outside the budget: on bin packing {Assign, Capacity} (only the symmetry-breaking rows stay in the master) and
  on generalized assignment the pair (the whole program, already excluded as the full set). Their true bound is the
  integer optimum; they are never a set of gain 0 and never the best set.

plan  per instance the set of each method (llmdw.benchmark.bnp.plan with the truth above): the best set, the lookahead
      at the HiGHS dual and at five duals, GCG's detected set and the anchor of JointDW of every answer of the original
      presentation; written to <instance>/bnp_plan.json.
bnp   GCG branch-and-price on every distinct set of the plan and GCG as shipped (llmdw.benchmark.bnp.jobs_of), seeds 0,
      1 and 2, the settings and the failure rule of the time-optimal decompositions (llmdw.benchmark.time_truth.run_one: one repeat with
      the same seed, then unsolved at the limit with its root bound). Existing runs are reused.
score the measures of Experiments C and D, per class and pooled, from the stored files only (no solver run: a run
      without its row file has not been run yet; an instance enters the time measures once every run of its plan
      exists, and <out>/PARTIAL.txt records the counts while runs are missing):
        hit@1, bound-optimal set  the method's set is a set of maximal gain of the truth above (ties included; the
                     bound-optimal set is recomputed under z*, llmdw.benchmark.validity.bound_optimal); not
                     scored on an instance where every valid set within the budget has gain at most 1e-9 (no set
                     closes any of the LP gap); a set outside the budget is never the best set
        hit@1, time-optimal set   strict and within 10 percent (llmdw.benchmark.time_truth); the time-optimal set is
                     the first of the sets that were run, the distinct sets of the plan (GCG as shipped is a reference,
                     not a set), by the rule of the time-optimal decompositions: it is time-optimal among the planned sets within the
                     budget only, as the bound-optimal set is; a method whose first set is outside the budget keeps
                     its own measured time in the time ratios (counted in set_outside_budget of methods_time.csv)
                     and is a miss for hit@1 against the time-optimal set
        time ratio   the time statistic of the method's set (geometric mean over the seeds of t + 1 s, unsolved at
                     1801 s) over that of the time-optimal set, and over that of GCG as shipped: geometric mean over
                     instances, SE = SE of the mean log ratio times the geometric mean, asterisk when the 95 percent
                     interval of the mean log ratio excludes 0; GCG's detector is timed by GCG as shipped
        gain         the share of the LP gap of the method's set (Experiment C); a first set outside the budget has
                     no gain within the budget: 0 in every pooled gain (so every method is averaged over the same
                     instances), counted in set_outside_budget and not among the sets of gain 0
        false alarms samples of a configuration that predict a joint support on an instance without a joint support
                     within the budget (llmdw.scoring.analyze_support, joint_false, as on the assignment templates;
                     no F instance has a joint support within the budget: Capacity alone closes the whole gap on 18
                     GAP instances, and on the other instances the rest of the gap is closed only by {Assign,
                     Capacity}, which lies outside the budget); a predicted
                     support outside the budget is a false alarm only where Capacity alone closes the whole gap (gain
                     share of {Capacity} at least 1 - 1e-6) and otherwise names a set outside the budget (counted
                     apart); for the methods without samples, the count of sets outside the budget
      For a configuration every per-instance value is the mean over its samples (a sample without a set: no hit, gain
      0, no time). Paired differences: JointDW (each configuration) minus each baseline and the five-dual lookahead
      minus GCG's detector. Every standard error and interval: two-stage bootstrap over prompt groups and the
      instances within them (llmdw.scoring.cluster, 10,000 resamples, seed 0); every row records its instances and
      prompt groups. Table bodies experiment_F_methods.tex and experiment_F_paired.tex go to --tex-dir, their cell
      records to <out>/cells/.

Usage
-----
python -m llmdw.benchmark.no_joint_group plan --root work/no_joint_group \
    --names-file work/no_joint_group/instances_used_selected.txt \
    --answers "GPT-6.1 Sol=work/no_joint_group/answers/gpt-6.1-sol" "Claude Sonnet 5.5=work/no_joint_group/answers/claude-sonnet-5-5"
python -m llmdw.benchmark.no_joint_group bnp --root work/no_joint_group \
    --names-file work/no_joint_group/instances_used_selected.txt --parallel 8
python -m llmdw.benchmark.no_joint_group score --root data/work/no_joint_group \
    --names-file data/work/no_joint_group/instances_used_selected.txt \
    --answers "GPT-6.1 Sol=data/work/no_joint_group/answers/gpt-6.1-sol" \
              "Claude Sonnet 5.5=data/work/no_joint_group/answers/claude-sonnet-5-5" \
    --prompt-groups data/work/no_joint_group/prompt_groups.csv --out results/experiment_F \
    --tex-dir results/paper/experiments
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llmdw.benchmark import bnp

SEEDS = (0, 1, 2)
OUTSIDE_BUDGET = {"bin_packing": [frozenset({"Assign_#", "Capacity_#"})],
                  "generalized_assignment": [frozenset({"Assign_#", "Capacity_#"})]}


def instance_class(d: Path) -> str:
    return json.loads((d / "meta.json").read_text())["class"]


SOURCE_LABELS = {"SCIP, proved optimal": "SCIP reference", "SCIP, best solution found": "SCIP reference",
                 "best-known count of the OR-Library file": "best-known bins"}


def reference(d: Path, oracle: dict) -> tuple:
    """(z*, source): the reference optimum of llmdw.benchmark.validity (the best of SCIP's value, the best-known count
    of the file for bin packing and an accepted branch-and-price solution), with the source labels of this
    experiment."""
    from llmdw.benchmark import validity

    z, src = validity.reference_optimum(d, oracle)
    return z, SOURCE_LABELS.get(src, "feasible branch-and-price solution" if src else None)


def oracle_for_scoring(d: Path) -> dict:
    """oracle.json with the reference value of the classes without a joint group (in memory)."""
    o = json.loads((d / "oracle.json").read_text())
    o["reference_optimal_value"], o["reference_source"] = reference(d, o)
    return o


def outside_budget(d: Path) -> list:
    return list(OUTSIDE_BUDGET.get(instance_class(d), []))


def truth(d: Path) -> dict:
    """The truth of llmdw.benchmark.bnp._truth under the settings of the classes without a joint group: the gain table of the valid
    evaluations against the reference value, without the sets outside the budget."""
    from llmdw.scoring import structure

    o = oracle_for_scoring(d)
    tab = structure.instance_tables(o, None)["tables"].get((), {})
    out = set(outside_budget(d))
    return {"families": o["families"], "table": {T: v["gain"] for T, v in tab.items() if T and T not in out},
            "oracle": o}


def plan(root: Path, names: list, answers: dict) -> list:
    return [bnp.plan(root, n, answers, truth=truth(root / "instances" / n)) for n in names]


def jobs(root: Path, names: list) -> list:
    return [(r, n, T, s) for name in names for (r, n, T) in bnp.jobs_of(root, name) for s in SEEDS]


# ---------------------------------------------------------------- scoring (read only)

CLASSES = ("generalized_assignment", "bin_packing")
CLASS_NAMES = {"generalized_assignment": "Generalized assignment", "bin_packing": "Bin packing", "all": "All classes"}
CAPACITY = "Capacity_#"
GAIN_ZERO = 1e-9
CLOSES_GAP = 1 - 1e-6
# the instances of the selection whose every valid set within the budget has gain 0; derived in score() and checked
# against this list
NOT_SCORED_EXPECTED = ("bp_u120_02", "bp_u120_06", "bp_u120_09", "bp_u120_13")
BOUND_OPTIMAL = "bound-optimal set"
BASELINES = ("lookahead (single dual)", "lookahead (five duals)", "GCG default")
PLAN_KEY = {BOUND_OPTIMAL: "true_best", "lookahead (single dual)": "lookahead",
            "lookahead (five duals)": "lookahead_five_duals", "GCG default": "gcg_detected"}
GCG_SHIPPED = "gcg_default"
PAIRED_BOUND = ("hit_bound", "gain")
PAIRED_TIME = ("hit_time_strict", "hit_time_tolerance", "log_ratio_time_optimal")


def _no_solver() -> None:
    """Scoring reads stored files only: every solver entry point raises."""
    from llmdw.gcg import gcg_run

    def _refuse(*a, **k):
        raise RuntimeError("the score command reads stored rows only: a solver run was requested")

    gcg_run.run = _refuse


def scoring_info(d: Path) -> dict:
    """The truth of one instance for scoring: the gains of the valid sets within the budget, the joint supports within
    the budget (zero-marginal groups of at least two families with every evaluation below them converged, as in
    llmdw.scoring.analyze_support), the gain share of {Capacity}, the not-scored flag of hit@1 against the bound-
    optimal set, and the gain of GCG's detected decomposition (Experiment C)."""
    from llmdw.scoring import structure

    o = oracle_for_scoring(d)
    info = structure.instance_tables(o, None)
    out = {bnp.set_key(T) for T in outside_budget(d)}
    tab = {T: v for T, v in info["tables"].get((), {}).items() if bnp.set_key(T) not in out}
    lp, sense, scale = o["lp_bound"], o["sense"], info["scale"]
    tol = 1e-4 if scale else 1e-6 * max(1.0, abs(lp or 0.0))
    failed = {T for T in info["failed"].get((), set()) if bnp.set_key(T) not in out}
    groups = structure.zero_marginal_groups(tab, tol=tol, failed=failed)
    gains = {bnp.set_key(T): v["gain"] for T, v in tab.items() if T}
    cap = gains.get(CAPACITY)
    b = o["baselines"].get("gcg_spfwh") or {}
    gcg_gain = ((structure._minform(b["dual_bound"], sense) - structure._minform(lp, sense)) / scale
                if b.get("ok") and b.get("dual_bound") is not None and not b.get("bound_invalid") and scale else None)
    from llmdw.benchmark import validity

    bo, bo_ties = validity.bound_optimal(o, outside_budget(d))
    return {"class": instance_class(d), "families": o["families"], "gains": gains, "outside": sorted(out),
            "bound_optimal": bo, "bound_optimal_ties": bo_ties,
            "joint": [bnp.set_key(g["group"]) for g in groups if g["size"] >= 2 and g["status"] == "certain"],
            "capacity_gain": cap, "capacity_closes_gap": cap is not None and cap >= CLOSES_GAP,
            "not_scored": all(v <= GAIN_ZERO for v in gains.values()), "gcg_detected_gain": gcg_gain,
            "lp_bound": lp, "reference": o["reference_optimal_value"], "reference_source": o["reference_source"],
            "proved_optimal": o.get("reference_proved_optimal")}


def support_outcome(support, info: dict):
    """The predicted joint support of one sample (the first one, the anchor of JointDW): None without one; "false
    alarm" on an instance without a joint support within the budget, and for a support outside the budget where Capacity
    alone closes the whole gap; "outside budget" for a support outside the budget elsewhere; "instance with a joint
    support" otherwise."""
    if support is None:
        return None
    if bnp.set_key(support) in info["outside"]:
        return "false alarm" if info["capacity_closes_gap"] else "outside budget"
    return "instance with a joint support" if info["joint"] else "false alarm"


def stored_run(root: Path, name: str, T, seed: int):
    """One run under the failure rule of the time-optimal decompositions, read from its row files (llmdw.benchmark.time_truth.run_one);
    None while a row is missing: not yet run, or a failed first attempt whose repeat has not ended."""
    from llmdw.benchmark import time_truth

    d = root / "instances" / name / "bnp"

    def read(attempt):
        try:
            return json.loads((d / f"{time_truth._tag(T, seed, attempt)}.row.json").read_text())
        except (FileNotFoundError, ValueError):
            return None

    first = read(0)
    if first is None or (not first.get("ok") and read(1) is None):
        return None
    from llmdw.benchmark import validity

    # every row exists: run_one only reads; the run as scored under the validity rule (llmdw.benchmark.validity)
    return validity.apply(time_truth.run_one((str(root), name, T, seed)), oracle_for_scoring(root / "instances" / name),
                          validity.verdicts(root, name))


def run_rows(root: Path, names: list) -> tuple:
    """(rows, missing jobs): one row per stored run with the quantities of its log (llmdw.benchmark.time_truth.
    stats_rows, with the reference value of the classes without a joint group), and the planned runs without their rows."""
    from llmdw.benchmark import time_truth

    rows, missing, oracles = [], [], {}
    for job in jobs(root, names):
        _, n, T, seed = job
        r = stored_run(root, n, T, seed)
        if r is None:
            missing.append(job)
            continue
        d = root / "instances" / n
        o = oracles.setdefault(n, oracle_for_scoring(d))
        attempt = 1 if r.get("attempts") == 2 else 0
        lpath = time_truth.log_path(root, n, T, seed, attempt)
        st = time_truth.parse_stats(lpath.read_text(encoding="utf-8", errors="replace")) if lpath.exists() else {}
        lpb, ref = o["lp_bound"], o["reference_optimal_value"]
        rb, rb_src = r.get("root_bound"), r.get("root_bound_source")
        if (rb is None and r.get("nodes") == 1 and r.get("status") == "optimal" and r.get("dual_bound") is not None
                and r.get("validity_rule") is None):
            rb, rb_src = r["dual_bound"], "solved at the root"
        rows.append({"instance": n, "class": instance_class(d), "set": r["set"], "seed": seed,
                     "status": r.get("status"), "solved": bool(r.get("solved")),
                     "status_recorded": r.get("status_recorded"), "validity_rule": r.get("validity_rule"),
                     "reported_value": r.get("primal_bound") if r.get("status_recorded") == "optimal" else None,
                     "z_star": ref, "time": r.get("solving_time"),
                     "gap": r.get("gap"), "root_bound": rb, "root_bound_source": rb_src,
                     "root_gap_closed": ((rb - lpb) / (ref - lpb) if rb is not None and ref not in (None, lpb) else None),
                     "attempts": r.get("attempts"), "failed_twice": bool(r.get("failed_twice")),
                     **{k: st.get(k) for k in ("root_time", "root_time_source", "master_lp_iterations",
                                               "master_root_lp_iterations", "columns", "pricing_calls",
                                               "pricing_problems", "blocks", "aggregated_blocks",
                                               "identical_blocks_aggregated", "pricing_time", "pricing_share",
                                               "pricing_solvers", "nodes")},
                     **{k: r.get(k) for k in time_truth.CONTENTION}})
    return rows, missing


def sample_rows(name: str, info: dict, plan: dict, configs: list, tt: dict | None) -> list:
    """One row per method and sample of one instance; tt: its time truth (llmdw.benchmark.time_truth.time_truth),
    None while runs of its plan are missing."""
    m = dict(plan["methods"])
    ties = set(info["bound_optimal_ties"])          # recomputed under z* (scoring_info)
    m["true_best"] = info["bound_optimal"].split("|")
    outside = set(info["outside"])

    def row(method, sample, T, status=None, support=None, gain=None):
        key = bnp.set_key(T) if T else None
        if key is None:
            g = 0.0                                   # a sample without a set (Experiment C)
        elif key in outside:
            g = 0.0                                   # no gain within the budget: 0 in every pooled gain
        else:
            g = gain if gain is not None else info["gains"].get(key)
        r = {"instance": name, "class": info["class"], "method": method, "sample": sample, "status": status,
             "set": key, "first_support": bnp.set_key(support) if support is not None else None,
             "set_outside_budget": key in outside,
             "hit_bound": None if info["not_scored"] else key in ties,
             "gain": g, "gain_zero": (False if key in outside else None if g is None else g <= GAIN_ZERO)}
        if method in configs:
            r["support"] = support_outcome(support, info)
            r["false_alarm"] = r["support"] == "false alarm"
            r["support_outside_budget"] = r["support"] == "outside budget"
        if tt is not None:
            sm, opt, tol = tt["summary"], tt["time_optimal"], set(tt["within_tolerance"])
            s = sm.get(GCG_SHIPPED if method == "GCG default" else key)
            r.update({"time_optimal": opt, "hit_time_strict": key == opt, "hit_time_tolerance": key in tol,
                      "time_stat": s["time_stat"] if s else None, "solved_seeds": s["solved_seeds"] if s else None,
                      "mean_gap": s["mean_gap"] if s else None, "failed_twice": s["failed_twice"] if s else None,
                      "log_ratio_time_optimal": math.log(s["time_stat"] / sm[opt]["time_stat"]) if s else None,
                      "log_ratio_gcg_shipped": (math.log(s["time_stat"] / sm[GCG_SHIPPED]["time_stat"])
                                                if s and GCG_SHIPPED in sm else None)})
        return r

    out = [row(BOUND_OPTIMAL, None, m["true_best"])]
    out += [row(b, None, m.get(PLAN_KEY[b]), gain=info["gcg_detected_gain"] if b == "GCG default" else None)
            for b in BASELINES]
    for c in configs:
        out += [row(c, x["sample"], x["set"], x.get("status"), x.get("first_support")) for x in m.get(c) or []]
    return out


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def per_instance(rows: list) -> dict:
    """The per-instance values of one method: means over its samples (one row for the methods without samples)."""
    timed = [r for r in rows if r.get("time_stat") is not None]
    return {"hit_bound": _mean(r["hit_bound"] for r in rows), "gain": _mean(r["gain"] for r in rows),
            "gain_zero": _mean(r["gain_zero"] for r in rows),
            "hit_time_strict": _mean(r.get("hit_time_strict") for r in rows),
            "hit_time_tolerance": _mean(r.get("hit_time_tolerance") for r in rows),
            "time_stat": _mean(r["time_stat"] for r in timed),
            "solved": _mean(r["solved_seeds"] / len(SEEDS) for r in timed),
            "mean_gap": _mean(r["mean_gap"] for r in timed),
            "log_ratio_time_optimal": _mean(r["log_ratio_time_optimal"] for r in timed),
            "log_ratio_gcg_shipped": _mean(r["log_ratio_gcg_shipped"] for r in timed),
            "failed_twice": any(r.get("failed_twice") for r in rows),
            "n_samples": len(rows), "no_set": sum(r["set"] is None for r in rows),
            "false_alarms": sum(bool(r.get("false_alarm")) for r in rows),
            "support_outside_budget": sum(bool(r.get("support_outside_budget")) for r in rows),
            "set_outside_budget": sum(r["set_outside_budget"] for r in rows)}


def _boot(values: dict) -> dict:
    from llmdw.scoring import cluster, results

    v = {n: x for n, x in values.items() if x is not None}
    if not v:
        return {"mean": None, "se": None, "lo": None, "hi": None, "n": 0, "groups": 0}
    lo, hi, se = results.boot_ci_se(v)
    return {"mean": sum(v.values()) / len(v), "se": se, "lo": lo, "hi": hi, "n": len(v),
            "groups": cluster.n_groups(v)}


def _star(b: dict) -> str:
    return "*" if b["lo"] is not None and (b["lo"] > 0 or b["hi"] < 0) else ""


def _ratio(prefix: str, b: dict) -> dict:
    """A geometric-mean ratio from the bootstrap of the mean log ratio (llmdw.benchmark.time_truth.measure_tables)."""
    g = math.exp(b["mean"]) if b["mean"] is not None else None
    return {f"{prefix}_geo": g, f"{prefix}_se": g * b["se"] if g is not None and b["se"] is not None else None,
            f"{prefix}_asterisk": _star(b), f"{prefix}_lo": math.exp(b["lo"]) if b["lo"] is not None else None,
            f"{prefix}_hi": math.exp(b["hi"]) if b["hi"] is not None else None, f"{prefix}_se_log": b["se"],
            f"{prefix}_n": b["n"], f"{prefix}_groups": b["groups"]}


def tables(pi: dict, cls: dict, methods: list, configs: list) -> dict:
    """pi: {instance: {method: per_instance(...)}}. Rows per class and pooled: methods (Experiment C), methods_time
    (the time-optimal decompositions), paired (hit@1 against the bound-optimal set, gain) and paired_time_truth (hit@1 against the
    time-optimal set, strict and within tolerance, and the mean log time ratio)."""
    from llmdw.scoring import cluster

    out = {"methods": [], "methods_time": [], "paired": [], "paired_time_truth": []}
    pairs = [(c, b) for c in configs for b in BASELINES] + [("lookahead (five duals)", "GCG default")]
    for c in CLASSES + ("all",):
        ns = sorted(n for n in pi if c == "all" or cls[n] == c)
        if not ns:
            continue
        timed = [n for n in ns if pi[n][methods[0]]["hit_time_strict"] is not None]
        for m in methods:
            base = {"class": c, "method": m, "n_instances": len(ns), "n_groups": cluster.n_groups(ns)}
            row = dict(base)
            for k in ("hit_bound", "gain"):
                b = _boot({n: pi[n][m][k] for n in ns})
                row.update({f"{k}_mean": b["mean"], f"{k}_se": b["se"], f"{k}_n": b["n"], f"{k}_groups": b["groups"]})
            row["n_not_scored"] = sum(pi[n][m]["hit_bound"] is None for n in ns)
            row["gain_zero_instances"] = sum(pi[n][m]["gain_zero"] or 0.0 for n in ns)
            b = _boot({n: pi[n][m]["time_stat"] for n in ns})
            row.update({"time_stat_mean": b["mean"], "time_stat_se": b["se"], "time_stat_n": b["n"],
                        "time_stat_groups": b["groups"],
                        "solved": sum(pi[n][m]["solved"] for n in ns if pi[n][m]["solved"] is not None),
                        "n_failed_twice": sum(pi[n][m]["failed_twice"] for n in ns),
                        "failed_twice_instances": " ".join(n for n in ns if pi[n][m]["failed_twice"]),
                        "n_samples": sum(pi[n][m]["n_samples"] for n in ns),
                        "n_samples_without_set": sum(pi[n][m]["no_set"] for n in ns),
                        "false_alarms": sum(pi[n][m]["false_alarms"] for n in ns) if m in configs else None,
                        "support_outside_budget": (sum(pi[n][m]["support_outside_budget"] for n in ns)
                                                   if m in configs else None),
                        "set_outside_budget": sum(pi[n][m]["set_outside_budget"] for n in ns)})
            out["methods"].append(row)
            row = {**base, "n_time": len(timed), "n_groups_time": cluster.n_groups(timed),
                   "set_outside_budget": sum(pi[n][m]["set_outside_budget"] for n in timed)}
            for k in ("hit_time_strict", "hit_time_tolerance"):
                b = _boot({n: pi[n][m][k] for n in timed})
                row.update({f"{k}_mean": b["mean"], f"{k}_se": b["se"]})
            row.update(_ratio("ratio_time_optimal", _boot({n: pi[n][m]["log_ratio_time_optimal"] for n in timed})))
            row.update(_ratio("ratio_gcg_shipped", _boot({n: pi[n][m]["log_ratio_gcg_shipped"] for n in timed})))
            out["methods_time"].append(row)
        for a, b in pairs:
            for table, keys, names in (("paired", PAIRED_BOUND, ns), ("paired_time_truth", PAIRED_TIME, timed)):
                for k in keys:
                    d = {n: pi[n][a][k] - pi[n][b][k] for n in names
                         if pi[n][a][k] is not None and pi[n][b][k] is not None}
                    bb = _boot(d)
                    r = {"class": c, "method": a, "against": b, "quantity": k, "n_instances": bb["n"],
                         "n_groups": bb["groups"], "mean_a": _mean(pi[n][a][k] for n in d),
                         "mean_b": _mean(pi[n][b][k] for n in d), "mean_difference": bb["mean"],
                         "se_difference": bb["se"], "lo": bb["lo"], "hi": bb["hi"], "asterisk": _star(bb)}
                    if k.startswith("log_ratio"):
                        g = _ratio("ratio", bb)
                        r.update({"geometric_mean_ratio": g["ratio_geo"], "ratio_se": g["ratio_se"]})
                    out[table].append(r)
    return out


def read_prompt_groups(path: Path, names: list) -> dict:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        g = {r["instance"]: r["group"] for r in csv.DictReader(f)}
    return {n: g.get(n, n) for n in names}


def answer_rows(names: list, cls: dict, plans: dict, answers: dict) -> list:
    """Accounting of the answer cells per configuration and class: planned cells (instances x samples), answer files,
    unparseable answers, cells without a file, samples without a set, samples with a predicted joint support, and the
    cells whose status in the plan differs from the files now present."""
    from llmdw.scoring import results

    out = []
    for c, adir in answers.items():
        for k in CLASSES + ("all",):
            ns = [n for n in names if k == "all" or cls[n] == k]
            row = {"configuration": c, "class": k, "n_instances": len(ns), "planned_cells": 0, "answer_files": 0,
                   "unparseable": 0, "missing_files": 0, "samples_without_set": 0,
                   "samples_with_predicted_support": 0, "plan_status_mismatch": 0}
            for n in ns:
                plan = {x["sample"]: x for x in plans[n]["methods"].get(c) or []}
                for s in range(results.SAMPLES):
                    row["planned_cells"] += 1
                    f = results.result_path(Path(adir), n, "original", s)
                    if f.exists():
                        row["answer_files"] += 1
                        status = "answered" if json.loads(f.read_text(encoding="utf-8")).get("answer") is not None \
                            else "unparseable"
                        row["unparseable"] += status == "unparseable"
                    else:
                        row["missing_files"] += 1
                        status = "missing"
                    x = plan.get(s) or {}
                    row["plan_status_mismatch"] += x.get("status") != status
                    row["samples_without_set"] += not x.get("set")
                    row["samples_with_predicted_support"] += x.get("first_support") is not None
            out.append(row)
    return out


def _tex_header(text: str, partial: str) -> str:
    return "% " + text + (f" Partial: {partial}." if partial else "")


def tex_methods(meth: list, mtime: list, configs: list, partial: str):
    from llmdw.tables.cells import Cells, count
    from llmdw.tables.experiments import pm

    rec = Cells("experiment_F_methods")
    lines = [_tex_header(
        "Experiment F: per class and method. Columns: method; hit@1 against the bound-optimal set (not scored on the "
        "instances where every set within the budget has gain 0); hit@1 against the time-optimal set (the first of "
        "the planned sets within the budget by the rule of Experiment D), strict and within 10 percent; time ratio of"
        " the method's set to the time-optimal set and to GCG as shipped (geometric mean over instances $\\pm$ the SE "
        "of the mean log ratio times the geometric mean; asterisk: the 95 percent interval of the mean log ratio "
        "excludes 0); false alarms (samples predicting a joint support where none lies within the budget); samples "
        "naming a set outside the budget (for the methods without samples: instances whose set is outside the "
        "budget). Mean $\\pm$ standard error (two-stage bootstrap: prompt groups, then instances within "
        "them). Sources: results/experiment_F/methods.csv, methods_time.csv.", partial)]
    tm = {(r["class"], r["method"]): r for r in mtime}
    for c in CLASSES + ("all",):
        rows = [r for r in meth if r["class"] == c]
        if not rows:
            continue
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{CLASS_NAMES[c]}}}}} \\\\")
        rec.heading(CLASS_NAMES[c])
        for r in rows:
            t = tm[(c, r["method"])]
            label = _label(r["method"], configs)
            fa = count(r["false_alarms"]) if r["false_alarms"] is not None else "--"
            ob = r["support_outside_budget"] if r["support_outside_budget"] is not None else r["set_outside_budget"]
            lines.append(f"{label} & {pm(r['hit_bound_mean'], r['hit_bound_se'])} & "
                         f"{pm(t['hit_time_strict_mean'], t['hit_time_strict_se'])} & "
                         f"{pm(t['hit_time_tolerance_mean'], t['hit_time_tolerance_se'])} & "
                         f"{pm(t['ratio_time_optimal_geo'], t['ratio_time_optimal_se'], 2, t['ratio_time_optimal_asterisk'])} & "
                         f"{pm(t['ratio_gcg_shipped_geo'], t['ratio_gcg_shipped_se'], 2, t['ratio_gcg_shipped_asterisk'])} & "
                         f"{fa} & {count(ob)} \\\\")
            rec.add(label, "hit@1, bound-optimal set", r, r["hit_bound_mean"], r["hit_bound_se"],
                    n=r["hit_bound_n"], groups=r["hit_bound_groups"])
            for k, col in (("hit_time_strict", "hit@1, time-optimal set"),
                           ("hit_time_tolerance", "hit@1, within 10 percent of the time-optimal set")):
                rec.add(label, col, t, t[f"{k}_mean"], t[f"{k}_se"], n_key="n_time", groups_key="n_groups_time")
            for k, col in (("ratio_time_optimal", "time ratio to the time-optimal set"),
                           ("ratio_gcg_shipped", "time ratio to GCG as shipped")):
                rec.add(label, col, t, t[f"{k}_geo"], t[f"{k}_se"], t[f"{k}_asterisk"], n=t[f"{k}_n"],
                        groups=t[f"{k}_groups"])
            if r["false_alarms"] is not None:
                rec.add(label, "false alarms", r, r["false_alarms"])
            rec.add(label, "outside the budget", r, ob)
    return "\n".join(lines) + "\n", rec


def tex_paired(paired: list, ptime: list, configs: list, partial: str):
    from llmdw.tables.cells import Cells
    from llmdw.tables.experiments import pm

    rec = Cells("experiment_F_paired")
    lines = [_tex_header(
        "Experiment F: paired differences, JointDW (original presentation) minus each baseline and the five-dual "
        "lookahead minus GCG's detector. Columns: method; baseline; class; instances scored against the bound-optimal "
        "set; difference in hit@1 against the bound-optimal set; instances with every run; difference in hit@1 against "
        "the time-optimal set, strict and within 10 percent; time ratio of the two sets (geometric mean $\\pm$ the SE of "
        "the mean log ratio times the geometric mean). Mean $\\pm$ standard error (two-stage bootstrap: prompt groups, "
        "then instances within them; asterisk: 95 percent interval excludes 0). Sources: "
        "results/experiment_F/paired.csv, paired_time_truth.csv.", partial)]
    idx = {(r["class"], r["method"], r["against"], r["quantity"]): r for r in paired + ptime}
    keys = list(dict.fromkeys((r["class"], r["method"], r["against"]) for r in paired))
    for c, a, b in keys:
        h, s, t, lr = (idx[(c, a, b, q)] for q in ("hit_bound", "hit_time_strict", "hit_time_tolerance",
                                                   "log_ratio_time_optimal"))
        la, lb = _label(a, configs), _label(b, configs)
        lines.append(f"{la} & {lb} & {CLASS_NAMES[c]} & {h['n_instances']} & "
                     f"{pm(h['mean_difference'], h['se_difference'], 2, h['asterisk'])} & {s['n_instances']} & "
                     f"{pm(s['mean_difference'], s['se_difference'], 2, s['asterisk'])} & "
                     f"{pm(t['mean_difference'], t['se_difference'], 2, t['asterisk'])} & "
                     f"{pm(lr.get('geometric_mean_ratio'), lr.get('ratio_se'), 2, lr['asterisk'])} \\\\")
        row = f"{la} minus {lb}, {CLASS_NAMES[c]}"
        rec.add(row, "hit@1, bound-optimal set", h, h["mean_difference"], h["se_difference"], h["asterisk"])
        rec.add(row, "hit@1, time-optimal set", s, s["mean_difference"], s["se_difference"], s["asterisk"])
        rec.add(row, "hit@1, within 10 percent", t, t["mean_difference"], t["se_difference"], t["asterisk"])
        rec.add(row, "time ratio", lr, lr.get("geometric_mean_ratio"), lr.get("ratio_se"), lr["asterisk"])
    return "\n".join(lines) + "\n", rec


def _label(m: str, configs: list) -> str:
    from llmdw.tables.experiments import METHOD_NAMES

    if m == BOUND_OPTIMAL:
        return "Bound-optimal set"
    return f"JointDW, {m}" if m in configs else METHOD_NAMES.get(m, m)


def score(root: Path, names: list, answers: dict, groups_csv: Path, out: Path, tex_dir: Path | None,
          expected_not_scored=NOT_SCORED_EXPECTED) -> dict:
    """The scoring of the module docstring; returns the tables and the counts of the runs. expected_not_scored: the
    instances where every set has gain 0, checked against the ones derived here."""
    _no_solver()
    from llmdw.benchmark import time_truth
    from llmdw.scoring import cluster

    configs = list(answers)
    methods = [BOUND_OPTIMAL, *BASELINES, *configs]
    groups = read_prompt_groups(groups_csv, names)
    cluster.set_groups(groups)
    dirs = {n: root / "instances" / n for n in names}
    cls = {n: instance_class(dirs[n]) for n in names}
    info = {n: scoring_info(dirs[n]) for n in names}
    plans = {n: json.loads((dirs[n] / "bnp_plan.json").read_text()) for n in names}
    not_scored = tuple(sorted(n for n in names if info[n]["not_scored"]))
    expected = tuple(sorted(x for x in expected_not_scored if x in names))
    if not_scored != expected:
        raise ValueError(f"instances where every set has gain 0: {not_scored}, expected {expected}")
    rows, missing = run_rows(root, names)
    incomplete = {j[1] for j in missing}
    complete = [n for n in names if n not in incomplete]
    tt = (time_truth.time_truth([r for r in rows if r["instance"] in complete], root,
                                exclude={n: info[n]["outside"] for n in complete}) if complete else {})
    preds = {n: sample_rows(n, info[n], plans[n], configs, tt.get(n)) for n in names}
    pi = {n: {m: per_instance([r for r in preds[n] if r["method"] == m]) for m in methods} for n in names}
    # the same per-instance time values as the time-optimal decompositions (llmdw.benchmark.time_truth.method_values)
    for n, v in time_truth.method_values(root, tt, configs, []).items():
        for m in (*BASELINES, *configs):
            for q, k in (("hit_strict", "hit_time_strict"), ("hit_tolerance", "hit_time_tolerance"),
                         ("log_ratio", "log_ratio_time_optimal")):
                a, b = v[m][q], pi[n][m][k]
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                    raise AssertionError(f"{n} {m} {q}: {a} != {b}")
    tabs = tables(pi, cls, methods, configs)
    planned = jobs(root, names)
    status = []
    for k in CLASSES + ("all",):
        ns = [n for n in names if k == "all" or cls[n] == k]
        p = [j for j in planned if j[1] in ns]
        mi = [j for j in missing if j[1] in ns]
        status.append({"class": k, "n_instances": len(ns), "n_groups": cluster.n_groups(ns), "runs_planned": len(p),
                       "runs_present": len(p) - len(mi), "runs_missing": len(mi),
                       "instances_with_every_run": sum(n in complete for n in ns),
                       "instances_with_missing_runs": sum(n not in complete for n in ns)})
    out.mkdir(parents=True, exist_ok=True)
    W = time_truth._write
    W(out / "instances.csv", [instance_row(n, info[n], plans[n], groups[n], dirs[n], tt.get(n),
                                           sum(j[1] == n for j in planned), sum(j[1] == n for j in missing))
                              for n in names])
    W(out / "gains.csv", [{"instance": n, "class": cls[n], "set": T, "gain_share": g, "outside_budget": False}
                          for n in names for T, g in sorted(info[n]["gains"].items())]
      + [{"instance": n, "class": cls[n], "set": T, "gain_share": None, "outside_budget": True}
         for n in names for T in info[n]["outside"]])
    W(out / "predictions.csv", [r for n in names for r in preds[n]])
    W(out / "bnp_runs.csv", rows)
    W(out / "time_truth.csv", [
        {"instance": n, "class": cls[n], "time_optimal": t["time_optimal"],
         "time_stat": t["summary"][t["time_optimal"]]["time_stat"],
         "solved_seeds": t["summary"][t["time_optimal"]]["solved_seeds"],
         "time_optimal_outside_budget": t["time_optimal"] in info[n]["outside"],
         "ranked_sets": "planned sets within the budget", "n_ranked_sets": len(t["order"]),
         "within_tolerance": " ".join(t["within_tolerance"]), "bound_optimal": t["bound_optimal"],
         "bound_optimal_ties": " ".join(t["bound_optimal_ties"]), "coincide": t["coincide"],
         "not_scored_bound_optimal": info[n]["not_scored"], "order": " ".join(t["order"]),
         "gcg_shipped_time_stat": t["summary"].get(GCG_SHIPPED, {}).get("time_stat")} for n, t in tt.items()])
    W(out / "methods.csv", tabs["methods"])
    W(out / "methods_time.csv", tabs["methods_time"])
    W(out / "paired.csv", tabs["paired"])
    W(out / "paired_time_truth.csv", tabs["paired_time_truth"])
    W(out / "answers.csv", answer_rows(names, cls, plans, answers))
    W(out / "runs_status.csv", status)
    allr = status[-1]
    partial = (f"{allr['runs_present']} of {allr['runs_planned']} branch-and-price runs present, "
               f"{allr['instances_with_every_run']} of {allr['n_instances']} instances with every run"
               if missing else "")
    marker = out / "PARTIAL.txt"
    if missing:
        marker.write_text(f"Branch-and-price incomplete: {partial}. The time measures (hit@1 against the time-optimal "
                          "set, time ratios, time statistics, paired time measures) cover only the instances with "
                          "every run; rerun the scoring when the runs have ended.\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()
    tex = Path(tex_dir) if tex_dir else out / "tables"
    tex.mkdir(parents=True, exist_ok=True)
    body, rec = tex_methods(tabs["methods"], tabs["methods_time"], configs, partial)
    (tex / "experiment_F_methods.tex").write_text(body, encoding="utf-8")
    rec.write(out / "cells" / "experiment_F_methods.csv")
    body, rec = tex_paired(tabs["paired"], tabs["paired_time_truth"], configs, partial)
    (tex / "experiment_F_paired.tex").write_text(body, encoding="utf-8")
    rec.write(out / "cells" / "experiment_F_paired.csv")
    return {"tables": tabs, "status": status, "not_scored": not_scored, "complete": complete}


def instance_row(n: str, info: dict, plan: dict, group: str, d: Path, tt: dict | None, planned: int,
                 missing: int) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    scip = json.loads((d / "oracle.json").read_text()).get("reference_optimal_value")
    ref, lp = info["reference"], info["lp_bound"]
    return {"name": n, "class": info["class"], "prompt_group": group, "rows": meta.get("rows"),
            "columns": meta.get("columns"), "binaries": meta.get("binaries"), "families": " ".join(info["families"]),
            "lp_bound": lp, "reference_value": ref, "reference_source": info["reference_source"],
            "scip_reference": scip, "best_known_bins_in_file": (meta.get("size") or {}).get("best_known_bins_in_file"),
            "scip_proved_optimal": info["proved_optimal"],
            "lp_gap": (ref - lp) / abs(ref) if ref not in (None, 0) and lp is not None else None,
            "capacity_gain_share": info["capacity_gain"], "capacity_closes_gap": info["capacity_closes_gap"],
            "joint_supports_within_budget": " ".join(info["joint"]),
            "rest_of_gap_closed_only_outside_budget": not info["capacity_closes_gap"],
            "not_scored_bound_optimal": info["not_scored"],
            "bound_optimal": info["bound_optimal"], "bound_optimal_ties": " ".join(info["bound_optimal_ties"]),
            "outside_budget_sets": " ".join(info["outside"]), "runs_planned": planned,
            "runs_present": planned - missing, "time_optimal": tt["time_optimal"] if tt else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "bnp", "count", "score"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--answers", nargs="*", default=[],
                    help="plan, score: NAME=DIR, the official name and the answer folder")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--prompt-groups", help="score: CSV of instance, class, prompt group")
    ap.add_argument("--out", help="score: the folder of the result files")
    ap.add_argument("--tex-dir", help="score: the folder of the table bodies (default <out>/tables)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    names = Path(args.names_file).read_text().split()
    if args.cmd == "plan":
        ps = plan(root, names, dict(a.split("=", 1) for a in args.answers))
        print(json.dumps({"instances": len(ps), "runs": len(jobs(root, names))}))
        return 0
    if args.cmd == "score":
        if not (args.out and args.prompt_groups):
            ap.error("score needs --out and --prompt-groups")
        res = score(root, names, dict(a.split("=", 1) for a in args.answers), Path(args.prompt_groups),
                    Path(args.out), Path(args.tex_dir) if args.tex_dir else None)
        for r in res["status"]:
            print(json.dumps(r))
        print(json.dumps({"not_scored_bound_optimal": list(res["not_scored"])}))
        return 0
    pending = jobs(root, names)
    if args.cmd == "count":
        new = [j for j in pending if not (root / "instances" / j[1] / "bnp" /
                                       f"{_tag(j[2], j[3])}.row.json").exists()]
        print(json.dumps({"runs": len(pending), "existing": len(pending) - len(new), "new": len(new)}))
        return 0
    from llmdw.benchmark import time_truth

    # bin packing first (the long runs), so that the end is not one long run alone
    cls = {n: instance_class(root / "instances" / n) for n in names}
    pending.sort(key=lambda j: (cls[j[1]] != "bin_packing", j[3]))
    print(f"{len(pending)} runs", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for r in pool.map(time_truth.run_one, pending):
            print(json.dumps({k: r.get(k) for k in ("instance", "set", "seed", "status", "solving_time", "gap",
                                                    "attempts")}), flush=True)
    return 0


def _tag(T, seed: int) -> str:
    from llmdw.benchmark import time_truth

    return time_truth._tag(T, seed)


if __name__ == "__main__":
    raise SystemExit(main())


def tex_from_csv(results_dir: Path, tex_dir: Path, configs=("GPT-6.1 Sol", "Claude Sonnet 5.5")) -> list:
    """The table bodies experiment_F_methods.tex and experiment_F_paired.tex rebuilt from the result CSVs of score
    (methods.csv, methods_time.csv, paired.csv, paired_time_truth.csv)."""
    from llmdw.tables.experiments import typed_rows

    r = Path(results_dir)
    out = []
    for name, fn, a, b in (("experiment_F_methods", tex_methods, "methods.csv", "methods_time.csv"),
                           ("experiment_F_paired", tex_paired, "paired.csv", "paired_time_truth.csv")):
        body = fn(typed_rows(r / a), typed_rows(r / b), list(configs), "")
        body = body[0] if isinstance(body, tuple) else body
        (Path(tex_dir) / f"{name}.tex").write_text(body, encoding="utf-8")
        out.append(f"{name}.tex")
    return out
