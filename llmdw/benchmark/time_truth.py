"""Time-optimal decompositions on the benchmark instances.

run    GCG branch-and-price on every nonempty set of the families of each instance, GCG as shipped ("gcg_default") and
       SCIP alone on the original formulation ("scip"), each with seeds 0, 1 and 2, with the settings of Experiment C
       (llmdw.benchmark.bnp: default settings, one thread, 1800 s, the decomposition given as a file). A run with the
       same set and seed that exists already (Experiment C) is reused. A run that ends abnormally (a solver error) is
       repeated once with the same seed (files with the suffix .r1); if that fails too, it counts as unsolved at the
       limit with its root bound (from the log of the repeat, else the census root evaluation of the set).
stats  per run, from GCG's (or SCIP's) statistics in the log: total time, solved, final gap, root bound and root gap
       closed, root time, columns generated, pricing calls, pricing problems and aggregated blocks, pricing time and its
       share of the total time, the pricing solvers used. The number of column generation iterations at the root is
       not in the statistics output and is recorded as missing.

Usage
-----
python -m llmdw.benchmark.time_truth run --root work/benchmark_std --names-file .../all_instances.txt --parallel 8
python -m llmdw.benchmark.time_truth stats --root work/benchmark_std --names-file ... --out results/experiment_D
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llmdw.benchmark import bnp

SEEDS = (0, 1, 2)
REFERENCES = ("gcg_default", "scip")
UNSOLVED_TIME = 1801.0          # the time of an unsolved run in the time statistic


def family_sets(fams) -> list:
    fams = sorted(fams)
    return [tuple(c) for r in range(1, len(fams) + 1) for c in itertools.combinations(fams, r)]


def jobs(root: Path, names: list) -> list:
    out = []
    for n in names:
        fams = json.loads((root / "instances" / n / "oracle.json").read_text())["families"]
        for T in family_sets(fams) + list(REFERENCES):
            for s in SEEDS:
                out.append((str(root), n, T, s))
    return out


def _tag(T, seed: int, attempt: int = 0) -> str:
    from llmdw.gcg.oracle import key_tag

    base = T if T in REFERENCES else key_tag(bnp.set_key(T))
    return base + (f".s{seed}" if seed else "") + (f".r{attempt}" if attempt else "")


def run_one(job) -> dict:
    """One run under the failure rule of the time-optimal decompositions (one repeat with the same seed)."""
    root, name, T, seed = job
    d = Path(root) / "instances" / name
    row = bnp._bnp(d, name, T, seed)
    if row.get("ok"):
        return {**row, "attempts": 1}
    rep = bnp._bnp(d, name, T, seed, attempt=1)
    if rep.get("ok"):
        return {**rep, "attempts": 2, "failure_first": row.get("error")}
    bound, src = rep.get("root_bound"), "root node statistics of the repeat"
    if bound is None and T not in REFERENCES:
        o = json.loads((d / "oracle.json").read_text())
        for r in o["subsets"]:
            if (bnp.set_key(set(o["families"]) - set(r["master_families"])) == bnp.set_key(T)
                    and not r.get("linking_var_families")):
                bound, src = r.get("dual_bound"), "census root evaluation (oracle.json)"
    return {**rep, "attempts": 2, "failed_twice": True, "failure_first": row.get("error"),
            "failure_repeat": rep.get("error"), "solved": False, "solving_time": bnp.BNP_TIME,
            "status": "failed twice (counted unsolved at the time limit)", "root_bound": bound, "root_bound_source": src}


def log_path(root: Path, name: str, T, seed: int, attempt: int) -> Path:
    return root / "instances" / name / "bnp" / f"{_tag(T, seed, attempt)}.log"


_NUM = r"([-+0-9.eE]+|-|infinite)"


def _display_seconds(tok: str):
    m = re.fullmatch(r"([0-9.]+)([smhd])", tok.strip())
    if not m:
        return None
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def scip_root_display_time(text: str):
    """SCIP alone: the time column of the last display line at node 1 before the first line at a later node (the end of
    the root node, at the display resolution); None when the log has no display line."""
    last = None
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 3:
            continue
        t = _display_seconds(cols[0][1:] if cols[0][:1].isalpha() else cols[0])
        node = cols[1].strip()
        if t is None or not node.isdigit():
            continue
        if int(node) > 1:
            break
        last = t
    return last


def parse_stats(text: str) -> dict:
    """The quantities of the time-optimal decompositions from a log with GCG's (or SCIP's) statistics; None where the log has none."""
    def num(pat, flags=re.S):
        m = re.search(pat, text, flags)
        if not m or m.group(1) in ("-", "infinite"):
            return None
        return float(m.group(1))

    out = {"total_time": num(r"Solving Time \(sec\)\s*:\s*" + _NUM),
           "nodes": num(r"Solving Nodes\s*:\s*" + _NUM),
           "root_time": num(r"time in root node\s*:\s*" + _NUM)}
    out["root_time_source"] = "GCG statistics: time in root node" if out["root_time"] is not None else None
    # root column-generation iterations are not in the statistics output; recorded instead, labeled as such: the
    # pricing calls (Pricers, gcg, Calls; whole solve) and the master LP iterations (Master Program statistics)
    master = text.split("Master Program statistics:", 1)[1].split("Original Program statistics:", 1)[0] \
        if "Master Program statistics:" in text else None
    out["master_lp_iterations"] = out["master_root_lp_iterations"] = None
    if master is not None:
        its = [int(m.group(1)) for m in re.finditer(
            r"^\s*(?:primal LP|dual LP|lex dual LP|barrier LP)\s*:\s*[0-9.]+\s+[0-9]+\s+([0-9]+)", master, re.M)]
        out["master_lp_iterations"] = sum(its) if its else None
        m = re.search(r"Root Node\s*:.*?Final Root Iters\s*:\s*([0-9]+)", master, re.S)
        out["master_root_lp_iterations"] = int(m.group(1)) if m else None
    else:
        t = scip_root_display_time(text)
        if t is not None:
            out["root_time"], out["root_time_source"] = t, "SCIP display: time of the last line at node 1"
    m = re.search(r"Pricers\s*:.*?\n\s*problem variables:.*?\n\s*gcg\s*:\s*([0-9.]+)\s+[0-9.]+\s+([0-9]+)\s+([0-9]+)", text)
    out["pricing_time"], out["pricing_calls"], out["columns"] = ((float(m.group(1)), int(m.group(2)), int(m.group(3)))
                                                                 if m else (None, None, None))
    m = re.search(r"Matrix has ([0-9]+) blocks?, using ([0-9]+) pricing problems?", text)
    out["blocks"], out["pricing_problems"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    m = re.search(r"Decomp statistics\s*:.*?blocks\s*:\s*([0-9]+)\s*\n\s*aggr\. blocks\s*:\s*([0-9]+)", text, re.S)
    if m:
        out["decomp_blocks"], out["aggregated_blocks"] = int(m.group(1)), int(m.group(2))
    else:
        out["decomp_blocks"] = out["aggregated_blocks"] = None
    out["identical_blocks_aggregated"] = (None if out["pricing_problems"] is None or out["blocks"] is None
                                          else out["pricing_problems"] < out["blocks"])
    m = re.search(r"Pricing Solver\s*:[^\n]*\n((?:\s+\S+\s*:[^\n]*\n)+)", text)
    used = []
    if m:
        for line in m.group(1).splitlines():
            parts = line.split(":")
            if len(parts) < 2:
                continue
            nums = parts[1].split()
            counts = [int(x) for x in nums[:4] if x.isdigit()]
            if sum(counts) > 0:
                used.append(parts[0].strip())
    out["pricing_solvers"] = " ".join(used) if m else None
    out["pricing_share"] = (out["pricing_time"] / out["total_time"]
                            if out["pricing_time"] is not None and out["total_time"] else None)
    return out


def _write(path: Path, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def scored_run(root: Path, job, oracles: dict | None = None) -> dict:
    """One stored run (run_one only reads) as scored under the validity rule of llmdw.benchmark.validity: z* of the
    instance, rules 1 to 3; the recorded status stays in status_recorded."""
    from llmdw.benchmark import validity

    _, n, T, seed = job
    oracles = {} if oracles is None else oracles
    d = root / "instances" / n
    o = oracles.setdefault(n, validity.oracle_with_reference(d))
    return validity.apply(run_one(job), o, validity.verdicts(root, n))


def stats_rows(root: Path, names: list) -> list:
    """One row per instance, set (or reference) and seed: the run record under the validity rule
    (llmdw.benchmark.validity) and the quantities of its log."""
    from llmdw.benchmark import validity

    out, oracles = [], {}
    for job in jobs(root, names):
        _, n, T, seed = job
        r = scored_run(root, job, oracles)                    # every run exists: run_one only reads
        attempt = 1 if r.get("attempts") == 2 else 0
        lp = log_path(root, n, T, seed, attempt)
        st = parse_stats(lp.read_text(encoding="utf-8", errors="replace")) if lp.exists() else {}
        o = oracles.setdefault(n, validity.oracle_with_reference(root / "instances" / n))
        lpb, ref = o["lp_bound"], o["reference_optimal_value"]
        rb, rb_src = r.get("root_bound"), r.get("root_bound_source")
        if (rb is None and r.get("nodes") == 1 and r.get("status") == "optimal" and r.get("dual_bound") is not None
                and r.get("validity_rule") is None):
            rb, rb_src = r["dual_bound"], "solved at the root"      # the rule of llmdw.benchmark.bnp
        rec = root / "instances" / n / "bnp" / f"{_tag(T, seed, attempt)}.json"     # run record (earlier rows)
        rec = json.loads(rec.read_text()) if rec.exists() else {}
        cont = {k: r.get(k, rec.get(k)) for k in CONTENTION}
        out.append({"instance": n, "class": json.loads((root / "instances" / n / "meta.json").read_text())["class"],
                    "set": r["set"], "seed": seed, "status": r.get("status"), "solved": bool(r.get("solved")),
                    "status_recorded": r.get("status_recorded"), "validity_rule": r.get("validity_rule"),
                    "reported_value": r.get("primal_bound") if r.get("status_recorded") == "optimal" else None,
                    "z_star": ref,
                    "time": r.get("solving_time"), "gap": r.get("gap"), "root_bound": rb, "root_bound_source": rb_src,
                    "root_gap_closed": ((rb - lpb) / (ref - lpb) if rb is not None and ref not in (None, lpb) else None),
                    "attempts": r.get("attempts"), "failed_twice": bool(r.get("failed_twice")),
                    **{k: st.get(k) for k in ("root_time", "root_time_source", "master_lp_iterations",
                                              "master_root_lp_iterations", "columns", "pricing_calls",
                                              "pricing_problems", "blocks", "aggregated_blocks",
                                              "identical_blocks_aggregated", "pricing_time", "pricing_share",
                                              "pricing_solvers", "nodes")}, **cont})
    return out


# contention at the start and end of each run (llmdw.gcg.gcg_run.run); the load average and the number of solver
# processes are recorded for the runs after the first restart of the batch, the available memory for every run
CONTENTION = ("started_at", "ended_at", "load1_start", "load1_end", "mem_available_gb_start", "mem_available_gb_end",
              "solver_processes_start", "slot_wait_seconds")


TOLERANCE = 0.10


def _geo(xs):
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def set_summary(runs: list) -> dict:
    """The seeds of one set (or reference) on one instance: solved count, time statistic (geometric mean of t + 1,
    unsolved at 1801), mean final gap, and the means of the quantities of the logs."""
    def tval(r):
        return (r["time"] if r["solved"] and r["time"] is not None else UNSOLVED_TIME - 1) + 1
    def mean(k):
        v = [r[k] for r in runs if r.get(k) not in (None, "")]
        return sum(v) / len(v) if v else None
    return {"solved_seeds": sum(r["solved"] for r in runs), "time_stat": _geo([tval(r) for r in runs]),
            "mean_gap": mean("gap"), "root_gap_closed": mean("root_gap_closed"), "pricing_share": mean("pricing_share"),
            "pricing_calls": mean("pricing_calls"), "master_lp_iterations": mean("master_lp_iterations"),
            "master_root_lp_iterations": mean("master_root_lp_iterations"), "nodes": mean("nodes"),
            "columns": mean("columns"),
            "root_time": mean("root_time"), "failed_twice": any(r["failed_twice"] for r in runs)}


def rank_key(sm: dict):
    """More seeds solved first; then the time statistic; sets unsolved on every seed by mean final gap."""
    if sm["solved_seeds"] == 0:
        return (0, 0.0, sm["mean_gap"] if sm["mean_gap"] is not None else float("inf"))
    return (-sm["solved_seeds"], sm["time_stat"], 0.0)


def time_truth(rows: list, root: Path, exclude: dict | None = None) -> dict:
    """Per instance: the summary of every set and reference, the time-optimal set, the sets within tolerance, and the
    bound-optimal set (recomputed from the gain table under z*, llmdw.benchmark.validity.bound_optimal; it equals the
    true best set of bnp_plan.json unless z* or the validity of a bound changed, bound_optimal_changed). exclude:
    {instance: set keys} left out of the ranking and of the bound-optimal set (their summaries stay, so a method naming
    one keeps its own measured time)."""
    from llmdw.benchmark import validity

    by = {}
    for r in rows:
        by.setdefault(r["instance"], {}).setdefault(r["set"], []).append(r)
    out = {}
    for n, sets in by.items():
        sm = {k: set_summary(v) for k, v in sets.items()}
        skip = set((exclude or {}).get(n, ()))
        fam = {k: v for k, v in sm.items() if k not in REFERENCES and k not in skip}
        order = sorted(fam, key=lambda k: (rank_key(fam[k]), len(k.split("|")), k))
        best = order[0]
        tol = [k for k in order if fam[k]["solved_seeds"] == fam[best]["solved_seeds"] and fam[best]["solved_seeds"] > 0
               and fam[k]["time_stat"] <= (1 + TOLERANCE) * fam[best]["time_stat"]] or [best]
        plan = json.loads((root / "instances" / n / "bnp_plan.json").read_text())
        planned = bnp.set_key(plan["methods"]["true_best"])
        bound_best, ties = validity.bound_optimal(validity.oracle_with_reference(root / "instances" / n),
                                                  [frozenset(k.split("|")) for k in skip])
        out[n] = {"summary": sm, "order": order, "time_optimal": best, "within_tolerance": tol,
                  "bound_optimal": bound_best, "bound_optimal_ties": ties,
                  "bound_optimal_changed": bound_best != planned, "coincide": bound_best == best}
    return out


BASELINES = ("lookahead (single dual)", "lookahead (five duals)", "GCG default")
PLAN_KEY = {"lookahead (single dual)": "lookahead", "lookahead (five duals)": "lookahead_five_duals",
            "GCG default": "gcg_detected"}
CLASSES = ("lot_sizing", "network_design", "facility_location")


def method_values(root: Path, truth: dict, configs: list, data_configs: list) -> dict:
    """Per instance and method: hit@1 against the time-optimal set (strict, within tolerance) and the log of the time
    ratio of the first set to the time-optimal set (time statistics); for a configuration the means over its samples
    (a sample without a set: no hit, no ratio)."""
    import math

    out = {}
    for n, t in truth.items():
        plan = json.loads((root / "instances" / n / "bnp_plan.json").read_text())
        pdat = root / "instances" / n / "bnp_plan_data.json"
        plan_data = json.loads(pdat.read_text()) if pdat.exists() else {"methods": {}}
        opt, tol, sm = t["time_optimal"], set(t["within_tolerance"]), t["summary"]
        base = sm[opt]["time_stat"]
        vals = {}
        for m in BASELINES:
            T = plan["methods"].get(PLAN_KEY[m])
            key = bnp.set_key(T) if T else None
            tkey = "gcg_default" if m == "GCG default" else key       # GCG's detector: GCG as shipped is its run
            vals[m] = {"hit_strict": float(key == opt), "hit_tolerance": float(key in tol),
                       "log_ratio": math.log(sm[tkey]["time_stat"] / base) if tkey in sm else None}
        for c, src in [(c, plan["methods"]) for c in configs] + [(c, plan_data["methods"]) for c in data_configs]:
            rows = src.get(c) or []
            hs = [float(bool(x["set"]) and bnp.set_key(x["set"]) == opt) for x in rows]
            ht = [float(bool(x["set"]) and bnp.set_key(x["set"]) in tol) for x in rows]
            lr = [math.log(sm[bnp.set_key(x["set"])]["time_stat"] / base) for x in rows if x["set"]]
            vals[c] = {"hit_strict": sum(hs) / len(hs) if hs else None, "hit_tolerance": sum(ht) / len(ht) if ht else None,
                       "log_ratio": sum(lr) / len(lr) if lr else None}
        out[n] = vals
    return out


def measure_tables(root: Path, names: list, truth: dict, configs: list, data_configs: list, groups: dict) -> dict:
    import math

    from llmdw.scoring import cluster, results

    cls = {n: json.loads((root / "instances" / n / "meta.json").read_text())["class"] for n in names}
    mv = method_values(root, truth, configs, data_configs)
    methods = list(BASELINES) + configs + data_configs

    def boot(values, data_variant):
        cluster.set_groups({} if data_variant else groups)
        v = {n: x for n, x in values.items() if x is not None}
        if not v:
            return {"mean": None, "se": None, "lo": None, "hi": None, "n": 0, "groups": 0}
        lo, hi, se = results.boot_ci_se(v)
        return {"mean": sum(v.values()) / len(v), "se": se, "lo": lo, "hi": hi, "n": len(v),
                "groups": cluster.n_groups(v)}

    def excludes_zero(b):
        return "*" if b["lo"] is not None and (b["lo"] > 0 or b["hi"] < 0) else ""

    meth, paired = [], []
    for c in CLASSES + ("all",):
        ns = [n for n in names if c == "all" or cls[n] == c]
        for m in methods:
            dv = m in data_configs
            cluster.set_groups({} if dv else groups)
            row = {"class": c, "method": m, "n_instances": len(ns), "n_groups": cluster.n_groups(ns)}
            for q in ("hit_strict", "hit_tolerance"):
                b = boot({n: mv[n][m][q] for n in ns}, dv)
                row.update({f"{q}_mean": b["mean"], f"{q}_se": b["se"]})
            b = boot({n: mv[n][m]["log_ratio"] for n in ns}, dv)
            geo = math.exp(b["mean"]) if b["mean"] is not None else None
            # the geometric mean, the SE of the mean log ratio times the geometric mean, and the asterisk of the
            # interval of the mean log ratio
            row.update({"time_ratio_geo": geo,
                        "time_ratio_se": geo * b["se"] if geo is not None and b["se"] is not None else None,
                        "time_ratio_asterisk": excludes_zero(b),
                        "time_ratio_lo": math.exp(b["lo"]) if b["lo"] is not None else None,
                        "time_ratio_hi": math.exp(b["hi"]) if b["hi"] is not None else None,
                        "se_log_ratio": b["se"], "n_ratio": b["n"], "n_groups_ratio": b["groups"]})
            meth.append(row)
        pairs = [(a, b) for a in configs + data_configs for b in BASELINES] + [("lookahead (five duals)", "GCG default")]
        for a, b in pairs:
            dv = a in data_configs
            for q in ("hit_strict", "hit_tolerance", "log_ratio"):
                d = {n: mv[n][a][q] - mv[n][b][q] for n in ns
                     if mv[n][a][q] is not None and mv[n][b][q] is not None}
                bb = boot(d, dv)
                paired.append({"class": c, "method": a, "against": b, "quantity": q, "n_instances": bb["n"],
                               "n_groups": bb["groups"], "mean_difference": bb["mean"], "se_difference": bb["se"],
                               "lo": bb["lo"], "hi": bb["hi"], "asterisk": excludes_zero(bb),
                               "bootstrap": "instances" if dv else "prompt groups, then instances within them"})
    cluster.set_groups(groups)
    return {"methods": meth, "paired": paired, "values": mv}


def tradeoff(truth: dict, cls: dict) -> tuple:
    """Figure data (per instance and set) and the per-class means for the bound-optimal set, the time-optimal set, the
    singletons and the full triple."""
    fig, agg = [], []
    keys = ("root_gap_closed", "time_stat", "pricing_share", "pricing_calls", "master_root_lp_iterations", "nodes")
    for n, t in truth.items():
        for k, v in t["summary"].items():
            if k in REFERENCES:
                continue
            fig.append({"class": cls[n], "instance": n, "set": k, **{q: v[q] for q in keys}})
    for c in CLASSES:
        ts = {n: t for n, t in truth.items() if cls[n] == c}
        groups = {"bound-optimal set": lambda t: [t["bound_optimal"]], "time-optimal set": lambda t: [t["time_optimal"]],
                  "singletons": lambda t: [k for k in t["summary"] if k not in REFERENCES and "|" not in k],
                  "full triple": lambda t: [k for k in t["summary"] if k not in REFERENCES and k.count("|") == 2]}
        for g, pick in groups.items():
            vals = [t["summary"][k] for t in ts.values() for k in pick(t) if k in t["summary"]]
            row = {"class": c, "sets": g, "n": len(vals)}
            for q in keys:
                v = [x[q] for x in vals if x[q] is not None]
                row[f"{q}_mean"] = sum(v) / len(v) if v else None
            agg.append(row)
    return fig, agg


CLASS_LABELS = {"lot_sizing": "Lot sizing", "network_design": "Network design", "facility_location": "Facility location",
                "all": "All classes"}


def _label(m: str) -> str:
    """Display label of a method: the data-variant configurations are shown as "coefficients"."""
    return m[:-len(", data")] + ", coefficients" if m.endswith(", data") else m


def tex_bodies(out: Path, tex_dir: Path) -> dict:
    """The table bodies of the time-optimal decompositions from the CSVs in out (methods_time.csv, paired_time_truth.csv,
    tradeoff_by_class.csv): mean $\\pm$ standard error, asterisk when the 95 percent interval excludes 0."""
    from llmdw.tables.experiments import dict_rows, f2, pm

    meth, pair, trade = (dict_rows(out / f) for f in ("methods_time.csv", "paired_time_truth.csv",
                                                       "tradeoff_by_class.csv"))
    classes = [c for c in list(CLASSES) + ["all"] if any(r["class"] == c for r in meth)]
    L = ["% Experiment D: each method's first set against the time-optimal set. Columns: method; instances; hit@1 "
         "strict; hit@1 within 10 percent; time of the first set relative to the time-optimal set (geometric mean of "
         "the ratios of time statistics; standard error: SE of the mean log ratio times the ratio). Mean $\\pm$ "
         "standard error, two-stage bootstrap over prompt groups and instances (instances for the coefficient "
         "presentation), asterisk: 95 percent interval excludes 0 (ratio: excludes 1). Source: "
         "results/experiment_D/methods_time.csv"]
    for c in classes:
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{CLASS_LABELS[c]}}}}} \\\\")
        for r in (x for x in meth if x["class"] == c):
            L.append(f"{_label(r['method'])} & {r['n_instances']} & {pm(r['hit_strict_mean'], r['hit_strict_se'])} & "
                     f"{pm(r['hit_tolerance_mean'], r['hit_tolerance_se'])} & "
                     f"{pm(r['time_ratio_geo'], r['time_ratio_se'], 3, r['time_ratio_asterisk'])} \\\\")
    (tex_dir / "experiment_D_methods.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    P = ["% Experiment D: paired differences. Columns: method minus baseline; instances; hit@1 strict; hit@1 within 10 "
         "percent; mean log ratio of the time statistics. Mean $\\pm$ standard error, two-stage bootstrap over prompt "
         "groups and instances (instances for the coefficient presentation), asterisk: 95 percent interval excludes "
         "0. Source: results/experiment_D/paired_time_truth.csv"]
    for c in classes:
        P.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{CLASS_LABELS[c]}}}}} \\\\")
        rows = [x for x in pair if x["class"] == c]
        for a, b in dict.fromkeys((x["method"], x["against"]) for x in rows):
            q = {x["quantity"]: x for x in rows if x["method"] == a and x["against"] == b}
            cells = [pm(q[k]["mean_difference"], q[k]["se_difference"], 3 if k == "log_ratio" else 2, q[k]["asterisk"])
                     if k in q else "--" for k in ("hit_strict", "hit_tolerance", "log_ratio")]
            n = next(iter(q.values()))["n_instances"]
            P.append(f"{_label(a)} minus {_label(b)} & {n} & " + " & ".join(cells) + " \\\\")
    (tex_dir / "experiment_D_paired.tex").write_text("\n".join(P) + "\n", encoding="utf-8")
    T = ["% Experiment D: bound strength and pricing effort per class. Columns: sets; instances; root gap closed; time "
         "statistic (s); pricing share of the total time; pricing calls; master LP iterations at the root; nodes "
         "(means over instances; --: not in the statistics, the full triple has an empty master). Source: "
         "results/experiment_D/tradeoff_by_class.csv"]
    for c in [c for c in CLASSES if any(r["class"] == c for r in trade)]:
        T.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{CLASS_LABELS[c]}}}}} \\\\")
        for r in (x for x in trade if x["class"] == c):
            T.append(f"{r['sets']} & {r['n']} & {f2(r['root_gap_closed_mean'], 3)} & {f2(r['time_stat_mean'], 1)} & "
                     f"{f2(r['pricing_share_mean'], 3)} & {f2(r['pricing_calls_mean'], 0)} & "
                     f"{f2(r['master_root_lp_iterations_mean'], 0)} & {f2(r['nodes_mean'], 0)} \\\\")
    (tex_dir / "experiment_D_tradeoff.tex").write_text("\n".join(T) + "\n", encoding="utf-8")
    return {"methods": len(L) - 1, "paired": len(P) - 1, "tradeoff": len(T) - 1}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "stats", "plan", "tables"))
    ap.add_argument("--answers", nargs="*", default=[], help="tables: NAME=DIR of the original presentation (prompt groups)")
    ap.add_argument("--data-configs", nargs="*", default=[], help="tables: the data-variant configurations of bnp_plan_data.json")
    ap.add_argument("--root", required=True)
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--tex-dir", help="tables: where the table bodies are written (default <out>/tables/new)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    names = Path(args.names_file).read_text().split()
    pending = jobs(root, names)
    if args.cmd == "plan":
        new = [j for j in pending if not (root / "instances" / j[1] / "bnp" / f"{_tag(j[2], j[3])}.row.json").exists()]
        print(json.dumps({"runs": len(pending), "existing": len(pending) - len(new), "new": len(new)}))
    elif args.cmd == "run":
        # the long runs first (lot sizing, then the sets and references without a recorded time), so that the end is
        # not one long run alone
        order = {"lot_sizing": 0, "network_design": 1, "facility_location": 2}
        cls = {n: json.loads((root / "instances" / n / "meta.json").read_text())["class"] for n in names}
        pending.sort(key=lambda j: (order[cls[j[1]]], j[3]))
        print(f"{len(pending)} runs", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for r in pool.map(run_one, pending):
                print(json.dumps({k: r.get(k) for k in ("instance", "set", "seed", "status", "solving_time", "gap",
                                                        "attempts")}), flush=True)
    elif args.cmd == "stats":
        _write(Path(args.out) / "bnp_runs.csv", stats_rows(root, names))
    else:
        from llmdw.scoring import cluster

        out = Path(args.out)
        rows = stats_rows(root, names)
        _write(out / "bnp_runs.csv", rows)
        answers = dict(a.split("=", 1) for a in args.answers)
        prompts = {n: {c: json.loads((Path(d) / n / "original.json").read_text())["messages"]
                       for c, d in answers.items() if (Path(d) / n / "original.json").exists()} for n in names}
        groups = cluster.prompt_groups(prompts)
        truth = time_truth(rows, root)
        cls = {n: json.loads((root / "instances" / n / "meta.json").read_text())["class"] for n in names}
        _write(out / "time_truth.csv", [{"instance": n, "class": cls[n], "time_optimal": t["time_optimal"],
                                         "time_stat": t["summary"][t["time_optimal"]]["time_stat"],
                                         "solved_seeds": t["summary"][t["time_optimal"]]["solved_seeds"],
                                         "within_tolerance": " ".join(t["within_tolerance"]),
                                         "bound_optimal": t["bound_optimal"], "coincide": t["coincide"],
                                         "order": " ".join(t["order"])} for n, t in truth.items()])
        res = measure_tables(root, names, truth, list(answers), args.data_configs, groups)
        _write(out / "methods_time.csv", res["methods"])
        _write(out / "paired_time_truth.csv", res["paired"])
        fig, agg = tradeoff(truth, cls)
        _write(out / "tradeoff_by_class.csv", agg)
        _write(out / "figure_data" / "experiment_D_tradeoff.csv", fig)
        tex = Path(args.tex_dir) if args.tex_dir else out / "tables" / "new"
        tex.mkdir(parents=True, exist_ok=True)
        tex_bodies(out, tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
