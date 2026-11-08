"""Experiment C: the set of each method and full GCG branch-and-price on every distinct set.

plan  (before any branch-and-price run) per instance the set of each method, from files that exist at that point:
        true best set        the set of maximal GCG bound among the evaluated sets (oracle.json, census setting;
                             ties: fewer families, then name)
        lookahead            the first set of the lookahead order at the HiGHS dual (lookahead.json;
                             llmdw.scoring.results.lookahead_order: by value, values within 1e-9 tied, then fewer
                             families, then name)
        lookahead_five_duals the first set of the same order on the best-of-five values (five_duals.json)
        gcg_detected         the family set of GCG's detected decomposition in the bound-evaluation setting
                             (baseline gcg_spfwh of oracle.json) when it is family-aligned; GCG's own default run
                             (gcg_default below) is the reference in every case
        <configuration>      per sample, the anchor of JointDW (llmdw.scoring.results.support_anchor: the first joint
                             support, else the convexified families of the top-ranked reformulation)
      written to <instance>/bnp_plan.json with the time of writing.
run   GCG branch-and-price, one thread, seed 0, 1800 s, default settings otherwise (SCIP's default propagation
      included: this is a solve-time experiment that measures solver performance, not the bound, so it does not use the
      propagation-off setting of the bound evaluations; every row records propagation "default"): llmdw.gcg.gcg_run dw-given for
      every distinct set (master = the other families, blocks by connected components; identical sets are run once),
      and gcg-default (GCG as shipped: presolve, detection, default score) for GCG's default decomposition. Recorded:
      status, time, primal and dual bound, gap, nodes, and the root bound (Final Dual Bound of the root node of the
      original program, parsed from GCG's statistics; for a run solved at the root, where GCG prints no root bound,
      the final dual bound). A run that ends with a solver error is repeated once with seed 1; if that fails too, it
      counts as unsolved at the time limit with the root bound as its bound and is flagged failed_twice.

Usage
-----
python -m llmdw.benchmark.bnp plan --root work/benchmark_std --names ... --answers GPT-6.1\\ Sol=work/benchmark_std/answers/gpt-6.1-sol ...
python -m llmdw.benchmark.bnp run --root work/benchmark_std --names ... --parallel 18
python -m llmdw.benchmark.bnp plan --plan-file bnp_plan_data.json --root ... --names ... --answers "GPT-6.1 Sol, data=..."
python -m llmdw.benchmark.bnp run --plan-file bnp_plan_data.json --root ... --names ... --parallel 8
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BNP_TIME = 1800.0
_ROOT = re.compile(r"Original Program statistics:.*?Root Node\s*:.*?Final Dual Bound\s*:\s*([-+0-9.eE]+|-|infinite)", re.S)


def set_key(T) -> str:
    return "|".join(sorted(T))


def _truth(d: Path) -> dict:
    from llmdw.scoring import structure

    o = json.loads((d / "oracle.json").read_text())
    tab = structure.instance_tables(o, None)["tables"].get((), {})
    return {"families": o["families"], "table": {T: v["gain"] for T, v in tab.items() if T}, "oracle": o}


def _first_of_order(fams, lookahead_table: dict):
    from llmdw.scoring import results

    k = len(fams)
    truth = {"families": fams, "lookahead": [(frozenset(r["families"]), r["lookahead"] or 0.0)
                                             for r in lookahead_table.values() if r["lookahead"] is not None]}
    order = results.lookahead_order(truth)
    return sorted(order[0]) if order else None


def plan(root: Path, name: str, answers: dict, plan_file: str = "bnp_plan.json") -> dict:
    from llmdw.instances.families import read_model, summarize
    from llmdw.scoring import results

    d = root / "instances" / name
    t = _truth(d)
    fams, tab = t["families"], t["table"]
    best = max(tab.values())
    tops = sorted((T for T, v in tab.items() if v >= best - 1e-9), key=lambda T: (len(T), sorted(T)))
    out = {"name": name, "families": fams, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "gains": {set_key(T): v for T, v in sorted(tab.items(), key=lambda kv: (len(kv[0]), sorted(kv[0])))},
           "methods": {"true_best": sorted(tops[0]), "true_best_ties": [sorted(T) for T in tops]}}
    la = json.loads((d / "lookahead.json").read_text())
    five = json.loads((d / "five_duals.json").read_text())
    out["methods"]["lookahead"] = _first_of_order(fams, la["table"])
    out["methods"]["lookahead_five_duals"] = _first_of_order(fams, five["table"])
    bl = t["oracle"]["baselines"].get("gcg_spfwh") or {}
    out["methods"]["gcg_detected"] = (sorted(set(fams) - set(bl["aligned_master_families"]))
                                      if bl.get("representable") and bl.get("aligned_master_families") is not None
                                      else None)
    out["gcg_detected_detail"] = {k: bl.get(k) for k in ("status", "dual_bound", "chosen_n_blocks",
                                                          "chosen_n_master_rows", "representable", "degenerate")}
    summ = summarize(read_model(str(d / "model.mps")))
    insts = {name: {"summary": summ, "truth": {"families": fams}}}
    for cfg, adir in answers.items():
        recs = [r for r in results.load_answers(Path(adir), insts) if r["condition"] == "original"]
        out["methods"][cfg] = [{"sample": r.get("sample", 0), "status": r.get("status"),
                                "set": sorted(results.support_anchor(r, fams)),
                                "first_support": sorted((r.get("supports") or [{}])[0].get("families") or [])
                                if r.get("supports") else None} for r in recs]
    (d / plan_file).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _root_bound(log: str):
    m = _ROOT.search(log)
    if not m or m.group(1) in ("-", "infinite"):
        return None
    return float(m.group(1))


def _bnp(d: Path, name: str, T, seed: int) -> dict:
    """One branch-and-price run; the result row (written next to its log)."""
    from llmdw.gcg import gcg_run
    from llmdw.gcg.dec_io import from_master_families, write_dec
    from llmdw.gcg.oracle import key_tag
    from llmdw.instances.families import family_of, read_model

    out = d / "bnp"
    out.mkdir(exist_ok=True)
    mps = str(d / "model.mps")
    key = "gcg_default" if T == "gcg_default" else set_key(T)
    tag = ("gcg_default" if T == "gcg_default" else key_tag(key)) + (f".s{seed}" if seed else "")
    res_path = out / f"{tag}.row.json"
    if res_path.exists():
        return json.loads(res_path.read_text())
    if T == "gcg_default":
        # solve-time experiment: SCIP's default propagation (bound evaluations use propagation off, llmdw.gcg.gcg_run)
        res = gcg_run.run("gcg-default", mps, str(out / f"{tag}.json"), time_limit=BNP_TIME, seed=seed,
                          propagation="default")
    else:
        model = read_model(mps)
        fams = sorted({family_of(r.name) for r in model.rows})
        blocks, master = from_master_families(model, set(fams) - set(T), set())
        dec = str(out / f"{key_tag(key)}.dec")
        write_dec(dec, blocks, master, header=f"convexified families: {key}")
        # solve-time experiment: SCIP's default propagation (bound evaluations use propagation off, llmdw.gcg.gcg_run)
        res = gcg_run.run("dw-given", mps, str(out / f"{tag}.json"), dec=dec, time_limit=BNP_TIME, seed=seed,
                          propagation="default")
    log = out / f"{tag}.log"
    row = {"instance": name, "set": key, "tag": tag, "time_limit": BNP_TIME, "seed": seed,
           **{k: res.get(k) for k in ("ok", "status", "dual_bound", "primal_bound", "nodes", "solving_time",
                                      "wall_seconds", "blocks", "linking_vars", "error", "exit_code", "propagation")}}
    row["root_bound"] = _root_bound(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else None
    row["root_bound_source"] = "root node statistics" if row["root_bound"] is not None else None
    if row["root_bound"] is None and row.get("nodes") == 1 and row.get("status") == "optimal":
        row["root_bound"], row["root_bound_source"] = row.get("dual_bound"), "solved at the root"
    p, q = row.get("primal_bound"), row.get("dual_bound")
    row["gap"] = (abs(p - q) / max(abs(p), 1e-9)) if p is not None and q is not None else None
    row["solved"] = row.get("status") == "optimal"
    res_path.write_text(json.dumps(row, indent=1), encoding="utf-8")
    return row


def run_one(job) -> dict:
    """A run; a failed run (solver error) is repeated once with seed 1, and a run that fails again counts as unsolved
    at the time limit with the root bound as its bound (from the log of the repeated run when GCG printed one,
    otherwise the census root evaluation of the set, oracle.json) and is flagged failed_twice."""
    root, name, T = job
    d = Path(root) / "instances" / name
    row = _bnp(d, name, T, 0)
    if row.get("ok"):
        return row
    retry = _bnp(d, name, T, 1)
    if retry.get("ok"):
        return {**retry, "retried_after_failure": True, "failure_seed0": row.get("error")}
    bound, src = retry.get("root_bound"), "root node statistics of the seed-1 run"
    if bound is None and T != "gcg_default":
        o = json.loads((d / "oracle.json").read_text())
        for r in o["subsets"]:
            if set_key(set(o["families"]) - set(r["master_families"])) == set_key(T) and not r.get("linking_var_families"):
                bound, src = r.get("dual_bound"), "census root evaluation (oracle.json)"
    if bound is None and T == "gcg_default":
        b = json.loads((d / "oracle.json").read_text())["baselines"].get("gcg_spfwh") or {}
        bound, src = b.get("dual_bound"), "GCG detected decomposition, census root evaluation (oracle.json)"
    ref = json.loads((d / "reference.json").read_text())["best_value"]
    return {**row, "failed_twice": True, "failure_seed0": row.get("error"), "failure_seed1": retry.get("error"),
            "status": "failed twice (counted unsolved at the time limit)", "solved": False, "solving_time": BNP_TIME,
            "dual_bound": bound, "root_bound": bound, "root_bound_source": src,
            "gap": (abs(ref - bound) / max(abs(ref), 1e-9)) if bound is not None and ref is not None else None,
            "gap_reference": "SCIP reference optimum (no primal bound from the failed runs)"}


def jobs_of(root: Path, name: str, plan_file: str = "bnp_plan.json") -> list:
    p = json.loads((root / "instances" / name / plan_file).read_text())
    sets = {set_key(p["methods"]["true_best"]): p["methods"]["true_best"]}
    for m in ("lookahead", "lookahead_five_duals", "gcg_detected"):
        if p["methods"].get(m):
            sets[set_key(p["methods"][m])] = p["methods"][m]
    for m, v in p["methods"].items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            for r in v:
                if r["set"]:
                    sets[set_key(r["set"])] = r["set"]
    return [(str(root), name, T) for T in sets.values()] + [(str(root), name, "gcg_default")]


def report(root: Path, names: list) -> list:
    """Per instance of the first batch: size, LP bound, optimum, LP gap, the gains of the evaluated sets (shares of the LP gap, with
    the convergence flag and the evaluation time), the census classification, the set of each method and whether it
    is the true best set, GCG's detected decomposition, every branch-and-price run and the answers of the language
    models; and the measured times that the projection of the full run uses."""
    from llmdw.gcg.oracle import key_tag

    out = []
    for n in names:
        d = root / "instances" / n
        meta = json.loads((d / "meta.json").read_text())
        o = json.loads((d / "oracle.json").read_text())
        p = json.loads((d / "bnp_plan.json").read_text())
        cl = json.loads((d / "classification.json").read_text())
        five = json.loads((d / "five_duals.json").read_text())
        lp, ref = o["lp_bound"], o["reference_optimal_value"]
        scale = abs(ref - lp) if ref is not None else None
        fams = o["families"]
        gains = []
        for r in o["subsets"]:
            T = sorted(set(fams) - set(r["master_families"]))
            g = (r["dual_bound"] - lp) / scale if (scale and r.get("dual_bound") is not None) else None
            gains.append({"set": set_key(T), "gain_share": g, "status": r.get("status"),
                          "seconds": r.get("solving_time"), "wall_seconds": r.get("wall_seconds")})
        best = set_key(p["methods"]["true_best"])
        meth = {}
        for m, v in p["methods"].items():
            if m == "true_best_ties":
                continue
            if isinstance(v, list) and v and isinstance(v[0], dict):
                meth[m] = [{"sample": x["sample"], "set": set_key(x["set"]), "first_support": x["first_support"],
                            "is_true_best": set_key(x["set"]) == best} for x in v]
            else:
                meth[m] = {"set": set_key(v) if v else None, "is_true_best": bool(v) and set_key(v) == best}
        runs = {}
        for job in jobs_of(root, n):
            r = run_one(job)
            if r.get("root_bound") is None and r.get("nodes") == 1 and r.get("status") == "optimal":
                r["root_bound"], r["root_bound_source"] = r.get("dual_bound"), "solved at the root"
            runs[r["set"]] = {k: r.get(k) for k in ("status", "solving_time", "wall_seconds", "gap", "nodes",
                                                     "root_bound", "root_bound_source", "primal_bound", "dual_bound",
                                                     "solved", "error", "failed_twice", "retried_after_failure",
                                                     "failure_seed0", "failure_seed1")}
        la_secs = json.loads((d / "lookahead.json").read_text())["seconds"]
        dual_secs = sum(json.loads((d / f"dual{s}.json").read_text())["seconds"] for s in (1, 2, 3, 4))
        out.append({"name": n, "class": meta["class"], "rows": meta["rows"], "columns": meta["columns"],
                    "binaries": meta["binaries"], "lp_bound": lp, "optimum": ref,
                    "lp_gap": (ref - lp) / abs(ref) if ref else None, "classification": cl["classification"],
                    "groups": cl["groups"], "gains": gains, "true_best": best, "methods": meth,
                    "gcg_detected_detail": p["gcg_detected_detail"], "branch_and_price": runs,
                    "duals_differing_from_highs": sum(1 for x in five["duals"][1:] if x.get("differs_from_highs")),
                    "times": {"evaluation_cpu_seconds": sum((r.get("wall_seconds") or 0) for r in o["subsets"])
                              + sum((b.get("wall_seconds") or 0) for b in o["baselines"].values()),
                              "lookahead_seconds": la_secs + dual_secs,
                              "bnp_cpu_seconds": sum((r.get("wall_seconds") or 0) for r in runs.values()),
                              "bnp_runs": len(runs)}})
    (root / "first_batch_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run", "report"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--answers", nargs="*", default=[], help="NAME=DIR, the official name and the answer folder")
    ap.add_argument("--parallel", type=int, default=18)
    ap.add_argument("--plan-file", default="bnp_plan.json",
                    help="the plan file of each instance (bnp_plan_data.json: the data variant on the benchmark instances; "
                         "its runs share the rows of identical sets)")
    args = ap.parse_args(argv)
    root, names = Path(args.root), args.names.split(",")
    if args.cmd == "plan":
        answers = dict(a.split("=", 1) for a in args.answers)
        for n in names:
            p = plan(root, n, answers, args.plan_file)
            print(json.dumps({"name": n, **{k: v for k, v in p["methods"].items() if not isinstance(v, list)
                                            or not v or not isinstance(v[0], dict)}}), flush=True)
    elif args.cmd == "report":
        report(root, names)
    else:
        todo = [j for n in names for j in jobs_of(root, n, args.plan_file)]
        print(f"{len(todo)} branch-and-price runs", flush=True)
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for r in pool.map(run_one, todo):
                print(json.dumps({k: r.get(k) for k in ("instance", "set", "status", "solving_time", "gap", "nodes")}),
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
