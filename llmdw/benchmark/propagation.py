"""Check: how much of the census root bounds comes from SCIP's domain propagation.

The census (llmdw.gcg.oracle, Appendix F) evaluates each decomposition at the root with SCIP's default propagation
(propagating/maxrounds 100, propagating/maxroundsroot 1000) on the original problem, whose binaries are integral, and
compares with z(empty) from llmdw.gcg.gcg_run lp (every variable continuous), where propagation cannot change the value.
This module re-evaluates, with propagation off (gcg_run --no-prop) and otherwise the census setting, every candidate the
census evaluated, without touching the census files:

miplib     the scored MIPLIB-NL models with a certain zero-marginal group (work/miplib_set): every (master families,
           linking variable families) candidate of their oracle.json, same time limit and seed
templates  the 15 templates, which have exact gains and no census run: every set of constraint families T with
           1 <= |T| <= k - 1 (master = the other families, no linking variables), evaluated with SCIP propagation enabled and disabled,
           each compared with the exact gain of ground_truth.json

Results: <out>/<set>/<name>/ (rows, logs) and <out>/<set>/<name>.json (gains with and without propagation per set, best
sets, zero-marginal groups at the empty set; the census tolerance of llmdw.scoring.analyze_support).

Usage
-----
python -m llmdw.benchmark.propagation miplib --work DATA --out work/propagation_check --jobs 18
python -m llmdw.benchmark.propagation templates --work DATA --out work/propagation_check --jobs 18
python -m llmdw.benchmark.propagation count --work DATA
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path


def _tables(o: dict, rows: list) -> dict:
    """GCG-only gain table {(T, L): gain} in the census scaling (share of the LP gap when the reference is known)."""
    from llmdw.scoring import structure

    info = structure.instance_tables({**o, "subsets": rows}, None)
    return info


def _groups(tab: dict, tol: float) -> list:
    from llmdw.scoring import structure

    return sorted(("|".join(sorted(g["group"])), g["status"]) for g in structure.zero_marginal_groups(tab, tol=tol)
                  if g["size"] >= 2)


def _best(tab: dict, tol: float) -> list:
    vals = {T: v["gain"] for T, v in tab.items() if T}
    if not vals:
        return []
    b = max(vals.values())
    return sorted("|".join(sorted(T)) for T, v in vals.items() if v >= b - tol)


def compare(o: dict, rows_on: list, rows_off: list) -> dict:
    on, off = _tables(o, rows_on), _tables(o, rows_off)
    tol = 1e-4 if on["scale"] else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
    out = {"tolerance": tol, "scaled": bool(on["scale"]), "linking_sets": {}}
    for L in sorted(set(on["tables"]) | set(off["tables"])):
        a, b = on["tables"].get(L, {}), off["tables"].get(L, {})
        sets = sorted(set(a) | set(b), key=lambda T: (len(T), sorted(T)))
        out["linking_sets"]["|".join(L) or "(none)"] = {
            "gains": [{"set": "|".join(sorted(T)), "with_propagation": a.get(T, {}).get("gain"),
                       "without_propagation": b.get(T, {}).get("gain"),
                       "censored_with": a.get(T, {}).get("censored"), "censored_without": b.get(T, {}).get("censored")}
                      for T in sets if T],
            "best_with": _best(a, tol), "best_without": _best(b, tol),
            "groups_with": _groups(a, tol) if a else [], "groups_without": _groups(b, tol) if b else []}
    main = out["linking_sets"].get("(none)", {})
    out["best_set_changes"] = main.get("best_with") != main.get("best_without")
    out["groups_appear"] = sorted(set(map(tuple, main.get("groups_without", []))) - set(map(tuple, main.get("groups_with", []))))
    out["groups_disappear"] = sorted(set(map(tuple, main.get("groups_with", []))) - set(map(tuple, main.get("groups_without", []))))
    out["single_family_gain_with"] = any(g["with_propagation"] and g["with_propagation"] > tol and "|" not in g["set"]
                                         for g in main.get("gains", []))
    out["single_family_gain_without"] = any(g["without_propagation"] and g["without_propagation"] > tol
                                            and "|" not in g["set"] for g in main.get("gains", []))
    return out


def run_miplib(work: Path, out: Path, jobs: int) -> list:
    from llmdw.gcg.oracle import evaluate_subset
    from llmdw.instances.families import objective_integral, read_model
    from llmdw.scoring import analyze_support

    res = []
    root = work / "work"
    names = sorted(p.name for p in (root / "miplib_set").iterdir() if (p / "oracle.json").exists())
    targets = []
    for n in names:
        t = analyze_support.miplib_truth(root / "miplib_set" / n / "oracle.json", root / "lookahead" / f"{n}.json")
        if t is not None and t["joint"]:
            targets.append(n)
    for n in targets:
        o = json.loads((root / "miplib_set" / n / "oracle.json").read_text())
        mps = str(root / "mps" / f"{n}.mps.gz")
        model = read_model(mps)
        integral = objective_integral(model)
        d = out / "miplib" / n
        d.mkdir(parents=True, exist_ok=True)
        cands = [(set(r.get("master_families") or []), set(r.get("linking_var_families") or [])) for r in o["subsets"]]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            off = list(pool.map(lambda c: evaluate_subset(model, mps, d, c[0], o["time_limit"], 0, linking_var_families=c[1],
                                                          integral=integral, no_prop=True), cands))
        cmp_ = compare(o, o["subsets"], off)
        rec = {"name": n, "candidates": len(cands), **cmp_}
        (out / "miplib" / f"{n}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        res.append(rec)
        print(f"{n}: best set changes {cmp_['best_set_changes']}, groups appear {cmp_['groups_appear']}, "
              f"disappear {cmp_['groups_disappear']}", flush=True)
    return res


def run_templates(work: Path, out: Path, jobs: int, time_limit: float = 120.0) -> list:
    from llmdw.gcg import gcg_run
    from llmdw.gcg.oracle import evaluate_subset
    from llmdw.instances.families import family_of, objective_integral, read_model

    res = []
    for td in sorted(p for p in (work / "work" / "templates").iterdir() if (p / "ground_truth.json").exists()):
        gt = json.loads((td / "ground_truth.json").read_text())
        mps = str(td / "model.mps")
        model = read_model(mps)
        integral = objective_integral(model)
        fams = sorted({family_of(r.name) for r in model.rows})
        sets = [frozenset(c) for r in range(1, len(fams)) for c in itertools.combinations(fams, r)]
        rows = {}
        for prop in (True, False):
            d = out / "templates" / td.name / ("with_propagation" if prop else "without_propagation")
            d.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                rows[prop] = list(pool.map(lambda T: evaluate_subset(model, mps, d, set(fams) - T, time_limit, 0,
                                                                     integral=integral, no_prop=not prop), sets))
        lp = gcg_run.run("lp", mps, str(out / "templates" / td.name / "lp.json"))
        lpb = float(Fraction(gt["lp_bound"]))
        exact = {frozenset(k.split("|")): float(Fraction(v)) for k, v in gt["gain"].items() if k}
        sign = 1.0 if model.sense == "minimize" else -1.0
        per = []
        for T, a, b in zip(sets, rows[True], rows[False]):
            g = lambda r: (sign * (r["dual_bound"] - lpb)) if r.get("ok") and r.get("dual_bound") is not None else None  # noqa: E731
            per.append({"set": "|".join(sorted(T)), "exact_gain": exact.get(T), "gcg_with_propagation": g(a),
                        "gcg_without_propagation": g(b), "status_with": a.get("status"), "status_without": b.get("status")})
        tol = 1e-6 * max(1.0, abs(lpb))
        dev = lambda key: [p["set"] for p in per if p[key] is not None and p["exact_gain"] is not None  # noqa: E731
                           and abs(p[key] - p["exact_gain"]) > tol]
        rec = {"name": td.name, "families": fams, "lp_bound_exact": lpb, "lp_bound_gcg_lp": lp.get("dual_bound"),
               "sets": per, "deviates_from_exact_with_propagation": dev("gcg_with_propagation"),
               "deviates_from_exact_without_propagation": dev("gcg_without_propagation"), "tolerance": tol}
        (out / "templates" / f"{td.name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        res.append(rec)
        print(f"{td.name}: sets off the exact gain with propagation {rec['deviates_from_exact_with_propagation']}, "
              f"without {rec['deviates_from_exact_without_propagation']}", flush=True)
    return res


def count(work: Path) -> dict:
    """The scored MIPLIB-NL models with a positive single-family gain in the census table (could be affected)."""
    from llmdw.scoring import analyze_support, structure

    root = work / "work"
    hit = []
    names = sorted(p.name for p in (root / "miplib_set").iterdir() if (p / "oracle.json").exists())
    scored = 0
    for n in names:
        t = analyze_support.miplib_truth(root / "miplib_set" / n / "oracle.json", root / "lookahead" / f"{n}.json")
        if t is None:
            continue
        scored += 1
        o = json.loads((root / "miplib_set" / n / "oracle.json").read_text())
        info = structure.instance_tables(o, None)
        tab = info["tables"].get((), {})
        tol = 1e-4 if info["scale"] else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
        if any(len(T) == 1 and v["gain"] > tol for T, v in tab.items()):
            hit.append(n)
    return {"scored": scored, "with_single_family_gain": len(hit), "names": hit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("miplib", "templates", "count"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=18)
    args = ap.parse_args(argv)
    work = Path(args.work).expanduser()
    if args.cmd == "count":
        print(json.dumps(count(work), indent=1))
    elif args.cmd == "miplib":
        run_miplib(work, Path(args.out), args.jobs)
    else:
        run_templates(work, Path(args.out), args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
