"""Ground truth and lookahead orders of the benchmark instances.

evaluate   GCG evaluation of every set T of at most max_set (default 3) constraint families (master = the other families, no linking
           variable families), in the census setting of the MIPLIB-NL oracle (llmdw.gcg.oracle.evaluate_subset: root
           node, no presolve, no cuts, no heuristics, one thread, seed 0, 120 s), the LP bound and GCG's own detected
           decompositions in the same setting (the baselines of llmdw.gcg.oracle). Written as oracle.json in the format
           of llmdw.gcg.oracle, so llmdw.scoring.structure and llmdw.scoring.analyze_support read it unchanged; the
           reference optimum is SCIP's best value (reference.json). --baselines names the detection baselines that
           run (default: all of llmdw.gcg.oracle.GCG_BASELINES); the list is recorded as baselines_run.
           Every GCG root evaluation, the baselines included, runs with SCIP propagation off (llmdw.gcg.gcg_run); the
           rows are in oracle/, one record per evaluation, failed evaluations retried (llmdw.gcg.oracle).
classify   the census classification (llmdw.scoring.structure) from the GCG evaluations alone, written before any
           lookahead value exists: certain when a zero-marginal group of two or more families is certain, possible
           when the instance has a possible group but no certain one, incomplete when its only groups depend on a failed
           evaluation, none otherwise.
lookahead  the one-pass lookahead of every set of at most max_set families at the dual returned by HiGHS
           (llmdw.lookahead.lookahead.lookahead_table, block time 60 s) and at four further optimal duals, with the
           best-of-five table (llmdw.lookahead.five_duals).

Usage
-----
python -m llmdw.benchmark.truth evaluate --root work/benchmark_std --names ls_F1,nd_r01_4 --jobs 20 --max-set 2
python -m llmdw.benchmark.truth evaluate ... --baselines gcg_spfwh,gcg_spfwh_names
python -m llmdw.benchmark.truth classify --root work/benchmark --names ...
python -m llmdw.benchmark.truth lookahead --root work/benchmark_std --names ... --parallel 6 --max-set 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

EVAL_TIME = 120.0
BLOCK_TIME = 60.0
MAX_SET = 3


def instance_dir(root: Path, name: str) -> Path:
    return root / "instances" / name


def evaluate(root: Path, name: str, jobs: int, max_set: int = 3, baselines=None) -> dict:
    """oracle.json of one instance (see the module docstring)."""
    from llmdw.gcg.oracle import GCG_BASELINES, best_row, evaluate_subset, lp_bound, run_baseline
    from llmdw.gcg.gcg_run import write_json_atomic
    from llmdw.instances.families import family_of, objective_integral, read_model, var_family_map

    chosen = {b: GCG_BASELINES[b] for b in (baselines or GCG_BASELINES)}
    d = instance_dir(root, name)
    out = d / "oracle"
    out.mkdir(exist_ok=True)
    mps = str(d / "model.mps")
    model = read_model(mps)
    integral = objective_integral(model)
    fams = sorted({family_of(r.name) for r in model.rows})
    vfams = sorted(set(var_family_map(model.var_type).values()))
    ref = json.loads((d / "reference.json").read_text())
    summary = {"mps": mps, "families": fams, "variable_families": vfams, "k": len(fams), "k_var": len(vfams),
               "time_limit": EVAL_TIME, "sense": model.sense, "reference_optimal_value": ref["best_value"],
               "reference_proved_optimal": ref["proved_optimal"], "with_linking": False, "use_linking": False,
               "objective_integral": integral, "max_set": max_set, "propagation": "off",
               "baselines_run": list(chosen)}
    lp = lp_bound(mps, out)
    summary["lp_bound"] = lp.get("dual_bound") if lp.get("ok") else None
    sets = [frozenset(c) for r in range(1, max_set + 1) for c in itertools.combinations(fams, r)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {b: pool.submit(run_baseline, model, mps, out, b, opts, EVAL_TIME, 0, integral)
                for b, opts in chosen.items()}
        rows = list(pool.map(lambda T: evaluate_subset(model, mps, out, set(fams) - T, EVAL_TIME, 0,
                                                       integral=integral), sets))
        summary["baselines"] = {b: f.result() for b, f in futs.items()}
    summary["n_candidates"] = len(sets)
    summary["subsets"] = rows
    summary["failed"] = [r["key"] for r in rows if r.get("status") == "failed"]
    summary["oracle"] = best_row(rows, model.sense)
    write_json_atomic(d / "oracle.json", summary)
    return summary


def classify(root: Path, name: str) -> dict:
    """Census classification from the GCG evaluations alone (no lookahead), tolerance as in
    llmdw.scoring.analyze_support.miplib_truth."""
    from llmdw.scoring import structure

    d = instance_dir(root, name)
    o = json.loads((d / "oracle.json").read_text())
    info = structure.instance_tables(o, None)
    tab = info["tables"].get((), {})
    tol = 1e-4 if info["scale"] else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
    failed = info["failed"].get((), set())
    groups = structure.zero_marginal_groups(tab, tol=tol, failed=failed)
    joint = [g for g in groups if g["size"] >= 2]
    status = ("certain" if any(g["status"] == "certain" for g in joint)
              else "possible" if any(g["status"] == "possible" for g in joint)
              else "incomplete" if joint else "none")
    censored = sorted("|".join(sorted(T)) for T, v in tab.items() if v["censored"])
    res = {"name": name, "classification": status, "tolerance": tol,
           "groups": [{"group": sorted(g["group"]), "size": g["size"], "status": g["status"], "gain": g["gain"]}
                      for g in groups],
           "censored_sets": censored, "failed_sets": sorted("|".join(sorted(T)) for T in failed),
           "evaluated_sets": len(tab) - 1, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "classification.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- lookahead, single dual and best of five

def lookahead_one(args) -> dict:
    """lookahead.json (HiGHS dual, llmdw.lookahead.lookahead.lookahead_table), dual<seed>.json for the four further
    optimal duals and five_duals.json (llmdw.lookahead.five_duals), sets of at most max_set families, block time 60 s,
    no instance limit."""
    root, name, max_set = args
    from llmdw.lookahead import five_duals
    from llmdw.lookahead.lookahead import lookahead_table

    d = instance_dir(Path(root), name)
    t0 = time.time()
    la = lookahead_table(str(d / "model.mps"), block_time=BLOCK_TIME, max_size=max_set)
    (d / "lookahead.json").write_text(json.dumps(la, indent=1), encoding="utf-8")
    outs = []
    for seed in five_duals.DUAL_SEEDS:
        out = d / f"dual{seed}.json"
        five_duals.one(str(d / "model.mps"), str(d / "lookahead.json"), seed, str(out), BLOCK_TIME, None)
        outs.append(str(out))
    res = five_duals.combine(str(d / "lookahead.json"), outs, str(d / "five_duals.json"))
    return {"name": name, "seconds": round(time.time() - t0, 1),
            "duals_differ": sum(1 for x in res["duals"][1:] if x.get("differs_from_highs"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("evaluate", "classify", "lookahead"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--jobs", type=int, default=20, help="GCG evaluations at the same time (evaluate)")
    ap.add_argument("--parallel", type=int, default=6, help="instances at the same time (lookahead)")
    ap.add_argument("--max-set", type=int, default=MAX_SET, help="largest family set evaluated")
    ap.add_argument("--baselines", default=None,
                    help="comma-separated detection baselines to run (evaluate; default: all)")
    args = ap.parse_args(argv)
    root, names = Path(args.root), args.names.split(",")
    if args.cmd == "evaluate":
        for n in names:
            t0 = time.time()
            s = evaluate(root, n, args.jobs, args.max_set,
                         args.baselines.split(",") if args.baselines else None)
            print(f"{n:16s} {s['n_candidates']} sets in {time.time() - t0:.0f} s", flush=True)
    elif args.cmd == "classify":
        for n in names:
            print(json.dumps(classify(root, n)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            for r in pool.map(lookahead_one, [(str(root), n, args.max_set) for n in names]):
                print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
