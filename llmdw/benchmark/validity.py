"""Reference optimum and validity of branch-and-price runs (the benchmark instances and the classes without a joint
group).

Reference optimum z* of an instance: the best of
  1. the value proved or found by SCIP (reference.json, oracle.json reference_optimal_value);
  2. for bin packing, the best-known count of the OR-Library file (meta.json size.best_known_bins_in_file);
  3. a solution accepted under rule 3 below (reference_accepted.json of the instance).
The same z* is used for the gains, the gain shares, the bound validity rule (llmdw.scoring.bounds) and the rules below.

Validity rule for every branch-and-price run (every set and seed, GCG with its own detector, SCIP alone). tau is the
tolerance of the bound validity rule (llmdw.scoring.bounds.tolerance: 1e-4 times the larger of |z* - z_LP| and
max(1, |z*|)); better and worse follow the objective sense.
  1. A run that reports optimality counts as solved only if its reported optimal value is within tau of z*.
  2. A run has an invalid dual bound if it reports optimality at a value worse than z* by more than tau, or ends with a
     dual bound beyond z* by more than tau. It counts as failed under the failure rule of the time-optimal experiment:
     unsolved at the 1800-second limit (1801 in the time statistic), its root bound kept only if the bound validity
     rule accepts it, its final gap computed from that root bound, else from the LP bound.
  3. A run with a solution value better than z* by more than tau: its solution (rerun of the same set and seed with
     solution output when the run wrote none, bnp/solution_check/) is checked against every row of the model as built,
     the variable bounds and integrality, with absolute tolerance 1e-6. A feasible solution becomes z* (its source is
     recorded in reference_accepted.json) and rules 1 and 2 are applied again to every run of the instance; an
     infeasible one makes the run failed as in rule 2.
The recorded runs are never changed: apply() returns the run as scored.

Usage
-----
python -m llmdw.benchmark.validity scan --root R --names-file F          # every run against the rules (JSON lines)
python -m llmdw.benchmark.validity solutions --root R --names-file F     # rule 3: rerun with solution output and check
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llmdw.scoring import bounds

FEAS_TOL = 1e-6
UNSOLVED_TIME = 1800.0
FAILED_STATUS = "invalid dual bound (counted unsolved at the time limit)"
INFEASIBLE_STATUS = "infeasible solution (counted unsolved at the time limit)"


# ---------------------------------------------------------------- reference optimum

def accepted(d: Path) -> dict | None:
    p = d / "reference_accepted.json"
    return json.loads(p.read_text()) if p.exists() else None


def reference_optimum(d: Path, oracle: dict) -> tuple:
    """(z*, source) of one instance (module docstring); (None, None) without a reference."""
    sense = oracle["sense"]
    cands = []
    scip = oracle.get("reference_optimal_value")
    if bounds.finite(scip):
        ref = json.loads((d / "reference.json").read_text()) if (d / "reference.json").exists() else {}
        cands.append((float(scip), "SCIP, proved optimal" if ref.get("proved_optimal") else "SCIP, best solution found"))
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("class") == "bin_packing":
        bk = (meta.get("size") or {}).get("best_known_bins_in_file")
        if bk is not None:
            cands.append((float(bk), "best-known count of the OR-Library file"))
    acc = accepted(d)
    if acc:
        cands.append((float(acc["value"]), acc["source"]))
    if not cands:
        return None, None
    pick = min if sense == "minimize" else max
    best = pick(v for v, _ in cands)
    return best, next(s for v, s in cands if v == best)


def oracle_with_reference(d: Path) -> dict:
    """oracle.json with z* as reference_optimal_value (in memory) and its source."""
    o = json.loads((d / "oracle.json").read_text())
    o["scip_reference_value"] = o.get("reference_optimal_value")
    o["reference_optimal_value"], o["reference_source"] = reference_optimum(d, o)
    return o


def tolerance(oracle: dict) -> float:
    return bounds.tolerance(oracle.get("lp_bound"), oracle.get("reference_optimal_value"), oracle["sense"])


def bound_optimal(oracle: dict, outside=()) -> tuple:
    """(bound-optimal set, its ties) from the gain table of the oracle under its reference optimum, by the rule of
    llmdw.benchmark.bnp.plan (maximal gain within 1e-9; ties by fewer families, then name); outside: the family sets
    outside the budget, never the best set."""
    from llmdw.benchmark.bnp import set_key
    from llmdw.scoring import structure

    tab = structure.instance_tables(json.loads(json.dumps(oracle)), None)["tables"].get((), {})
    out = {frozenset(T) for T in outside}
    gains = {T: v["gain"] for T, v in tab.items() if T and T not in out}
    best = max(gains.values())
    tops = sorted((T for T, v in gains.items() if v >= best - 1e-9), key=lambda T: (len(T), sorted(T)))
    return set_key(tops[0]), [set_key(T) for T in tops]


# ---------------------------------------------------------------- the rules

def _worse(a, b, sense, tol):
    """a worse than b by more than tol (a primal value)."""
    return a > b + tol if sense == "minimize" else a < b - tol


def _beyond(bound, z, sense, tol):
    """a dual bound beyond z by more than tol."""
    return bound > z + tol if sense == "minimize" else bound < z - tol


def classify(run: dict, z, sense: str, tol: float) -> str | None:
    """None (valid), "rule 2" (invalid dual bound) or "rule 3" (a solution better than z*) for one run record."""
    if z is None or not run.get("ok", True) or run.get("failed_twice"):
        return None
    st, p, q = run.get("status"), run.get("primal_bound"), run.get("dual_bound")
    if bounds.finite(p) and _worse(z, p, sense, tol):
        return "rule 3"
    if st == "optimal" and bounds.finite(p) and _worse(p, z, sense, tol):
        return "rule 2"
    if bounds.finite(q) and _beyond(q, z, sense, tol):
        return "rule 2"
    return None


def apply(run: dict, oracle: dict, verdicts: dict | None = None) -> dict:
    """The run as scored under the rules (module docstring); adds validity_rule, status_recorded and, for a failed
    run, the root bound and final gap of the failure rule. verdicts: {run tag: "feasible" | "infeasible"} of rule 3."""
    sense, lp, z = oracle["sense"], oracle.get("lp_bound"), oracle.get("reference_optimal_value")
    tol = tolerance(oracle)
    rule = classify(run, z, sense, tol)
    out = {**run, "validity_rule": rule, "status_recorded": run.get("status")}
    if rule is None:
        if run.get("status") == "optimal" and z is not None:
            out["solved"] = bounds.finite(run.get("primal_bound")) and abs(run["primal_bound"] - z) <= tol
        return out
    if rule == "rule 3" and (verdicts or {}).get(run.get("tag")) == "feasible":
        return {**out, "validity_rule": "rule 3, feasible (z* replaced)"}
    rb = run.get("root_bound")
    keep = rb is not None and bounds.invalid_reason(rb, sense, lp, z, None, tol) is None
    b = rb if keep else lp
    return {**out, "status": FAILED_STATUS if rule == "rule 2" else INFEASIBLE_STATUS, "solved": False,
            "solving_time": UNSOLVED_TIME, "root_bound": rb if keep else None,
            "root_bound_source": run.get("root_bound_source") if keep else "rejected by the bound validity rule",
            "dual_bound": b, "gap": abs(z - b) / max(abs(z), 1e-9) if b is not None else None,
            "gap_reference": "z* and the root bound" if keep else "z* and the LP bound",
            "validity_rule": rule if rule == "rule 2" else "rule 3, infeasible solution"}


# ---------------------------------------------------------------- rule 3: solution output and feasibility

def solution_paths(d: Path, tag: str) -> dict:
    s = d / "bnp" / "solution_check"
    return {"json": s / f"{tag}.json", "log": s / f"{tag}.log", "sol": s / f"{tag}.sol.json"}


def rerun_with_solution(d: Path, T, seed: int, tag: str) -> dict:
    """The run of set T (a family-set key, or "gcg_default") and seed with the settings of llmdw.benchmark.bnp and
    solution output, into bnp/solution_check/ (the recorded run is not touched)."""
    from llmdw.benchmark import bnp
    from llmdw.gcg import gcg_run
    from llmdw.gcg.oracle import key_tag

    p = solution_paths(d, tag)
    p["json"].parent.mkdir(parents=True, exist_ok=True)
    if p["json"].exists() and p["sol"].exists():
        return json.loads(p["json"].read_text())
    mps = str(d / "model.mps")
    if T == "gcg_default":
        return gcg_run.run("gcg-default", mps, str(p["json"]), time_limit=bnp.BNP_TIME, seed=seed,
                           propagation="default", log_path=str(p["log"]), write_sol=str(p["sol"]))
    dec = d / "bnp" / f"{key_tag(T)}.dec"
    return gcg_run.run("dw-given", mps, str(p["json"]), dec=str(dec), time_limit=bnp.BNP_TIME, seed=seed,
                       propagation="default", log_path=str(p["log"]), write_sol=str(p["sol"]))


def check_solution(mps: Path, values: dict) -> dict:
    """Every row of the model as built, the variable bounds and integrality, absolute tolerance FEAS_TOL."""
    from llmdw.instances.families import read_model

    m = read_model(str(mps))
    unknown = [v for v in values if v not in m.var_type]
    x = {v: values.get(v, 0.0) for v in m.var_type}
    rows_bad, worst = [], 0.0
    for r in m.rows:
        a = sum(c * x[v] for v, c in r.coefs.items())
        viol = max(0.0, (r.lhs - a) if r.lhs > -1e19 else 0.0, (a - r.rhs) if r.rhs < 1e19 else 0.0)
        worst = max(worst, viol)
        if viol > FEAS_TOL:
            rows_bad.append({"row": r.name, "activity": a, "lhs": r.lhs, "rhs": r.rhs, "violation": viol})
    bnd = [v for v in m.var_type if x[v] < m.var_lb[v] - FEAS_TOL or x[v] > m.var_ub[v] + FEAS_TOL]
    integ = [v for v, t in m.var_type.items() if t in ("B", "I") and abs(x[v] - round(x[v])) > FEAS_TOL]
    obj = sum(c * x[v] for v, c in m.obj.items()) + m.obj_offset
    return {"feasible": not rows_bad and not bnd and not integ and not unknown, "rows": len(m.rows),
            "rows_violated": rows_bad, "max_row_violation": worst, "bounds_violated": bnd,
            "integrality_violated": integ, "unknown_variables": unknown, "objective": obj}


def bin_loads(raw_file: Path, problem: str, values: dict) -> dict:
    """The packing of a bin-packing solution recomputed from the OR-Library file: every item in exactly one bin, every
    bin load at most the capacity; the best-known count of the file and ceil(sum of sizes / capacity)."""
    from llmdw.benchmark.extra_classes import read_binpack

    d = next(p for p in read_binpack(raw_file.read_text(encoding="latin-1")) if p["id"] == problem)
    C, sizes = d["capacity"], d["sizes"]
    bins = {}
    for v, val in values.items():
        if v.startswith("Pack_") and round(val) == 1:
            i, j = (int(t) for t in v.split("_")[1:3])
            bins.setdefault(j, []).append(i)
    assigned = sorted(i for items in bins.values() for i in items)
    loads = {j: sum(sizes[i - 1] for i in items) for j, items in sorted(bins.items())}
    return {"problem": problem, "capacity": C, "items": d["items"], "best_known_bins_in_file": d["best_known_bins"],
            "lower_bound_ceil_sum_over_capacity": math.ceil(sum(sizes) / C), "sum_of_sizes": sum(sizes),
            "bins_used": len(bins), "every_item_once": assigned == list(range(1, d["items"] + 1)),
            "max_load": max(loads.values()), "loads_within_capacity": all(v <= C for v in loads.values()),
            "loads": loads}


# ---------------------------------------------------------------- scan and the rule-3 check over a collection

def run_records(d: Path) -> list:
    """Every stored run record of an instance (bnp/*.row.json, the solution-check reruns excluded)."""
    return [json.loads(p.read_text()) for p in sorted((d / "bnp").glob("*.row.json"))]


def scan(root: Path, names: list) -> list:
    out = []
    for n in names:
        d = root / "instances" / n
        o = oracle_with_reference(d)
        tol = tolerance(o)
        for r in run_records(d):
            rule = classify(r, o["reference_optimal_value"], o["sense"], tol)
            if rule:
                out.append({"instance": n, "set": r.get("set"), "seed": r.get("seed"), "tag": r.get("tag"),
                            "status": r.get("status"), "primal_bound": r.get("primal_bound"),
                            "dual_bound": r.get("dual_bound"), "z_star": o["reference_optimal_value"],
                            "z_star_source": o["reference_source"], "tau": tol, "rule": rule})
    return out


def solutions(root: Path, names: list, raw: Path | None = None) -> list:
    """Rule 3 over a collection: rerun each case with solution output, check it, and record an accepted z*."""
    out = []
    for case in scan(root, names):
        if case["rule"] != "rule 3":
            continue
        d = root / "instances" / case["instance"]
        res = rerun_with_solution(d, case["set"], case["seed"], case["tag"])
        p = solution_paths(d, case["tag"])
        rec = {**case, "rerun_status": res.get("status"), "rerun_primal_bound": res.get("primal_bound"),
               "rerun_seconds": res.get("solving_time"), "solution_file": str(p["sol"].relative_to(root))}
        if not p["sol"].exists():
            rec["feasibility"] = {"feasible": False, "reason": "the rerun wrote no solution"}
        else:
            sol = json.loads(p["sol"].read_text())
            rec["feasibility"] = check_solution(d / "model.mps", sol["values"])
            meta = json.loads((d / "meta.json").read_text())
            if meta.get("class") == "bin_packing" and raw is not None:
                rec["bin_loads"] = bin_loads(raw / "orlib_binpack" / meta["source"]["file"], meta["source"]["problem"],
                                             sol["values"])
        (p["json"].parent / f"{case['tag']}.check.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        out.append(rec)
    sense_of = {}
    for rec in out:
        d = root / "instances" / rec["instance"]
        sense = sense_of.setdefault(rec["instance"], json.loads((d / "oracle.json").read_text())["sense"])
        f = rec["feasibility"]
        if not f.get("feasible"):
            continue
        cur = accepted(d)
        better = cur is None or (f["objective"] < cur["value"] if sense == "minimize" else f["objective"] > cur["value"])
        if better:
            (d / "reference_accepted.json").write_text(json.dumps(
                {"value": f["objective"], "source": f"feasible solution of the branch-and-price run {rec['set']}, "
                                                    f"seed {rec['seed']} (checked against every row of the model)",
                 "run_tag": rec["tag"], "check": str((Path(rec["solution_file"]).parent / f"{rec['tag']}.check.json"))},
                indent=1), encoding="utf-8")
    return out


def verdicts(root: Path, name: str) -> dict:
    """{run tag: "feasible" | "infeasible"} of the rule-3 checks of an instance."""
    s = root / "instances" / name / "bnp" / "solution_check"
    out = {}
    for p in sorted(s.glob("*.check.json")) if s.exists() else []:
        r = json.loads(p.read_text())
        out[r["tag"]] = "feasible" if r["feasibility"].get("feasible") else "infeasible"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("scan", "solutions"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--raw", help="the raw benchmark folder (bin loads from the OR-Library file)")
    args = ap.parse_args(argv)
    root, names = Path(args.root), Path(args.names_file).read_text().split()
    rows = scan(root, names) if args.cmd == "scan" else solutions(root, names, Path(args.raw) if args.raw else None)
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k not in ("bin_loads",)} | (
            {"bin_loads": {k: v for k, v in r["bin_loads"].items() if k != "loads"}} if "bin_loads" in r else {}),
            default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
