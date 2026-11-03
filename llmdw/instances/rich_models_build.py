"""Generation of the rich models and their exact gains.

Paper: Section 6 and Appendix C: the rich models (Table 5).

Classes (llmdw.instances.templates): rich_lot_sizing (2 items, 3 periods; 9 families, 6 binaries), rich_network_design
(4 nodes and 7 arcs or 5 nodes and 8 arcs, 2 commodities; 7 families) and rich_facility_location (3 facilities, 3 or
4 customers; 7 families). Candidate data are drawn per class from random.Random(seed * 100 + class index). A
candidate is rejected when it is infeasible, when its LP gap (IP - LP) / |IP| is below 2 percent, or when a single
family closes more than half of the best gain among the sets of at most 3 families. The instances are the first 8
accepted candidates of each class in draw order.

Each instance directory holds the MPS file, the description and formulation text, ground_truth.json (the exact gain
of every set of at most 3 families from hull.dw_bound with a rational certificate, the zero-marginal groups and their
source from hull.group_source), lookahead.json (the one-pass lookahead at the HiGHS dual for the same sets) and
gcg_anchors.json (the decompositions GCG detects in the oracle setting: dw-auto, root node, no cuts, no heuristics,
120 s). draws.json records every candidate with the reason of a rejection.

Usage
-----
python -m llmdw.instances.rich_models_build --out data/work/rich_models --parallel 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

PER_CLASS = 8
MAX_SET = 3
MIN_GAP = 0.02
MAX_SINGLE_SHARE = 0.5
CLASSES = ("lot_sizing", "network_design", "facility_location")
NAME_PREFIX = "e6c"
GCG_BASELINES = {"gcg_spfwh": dict(score="spfwh", names=False), "gcg_strong": dict(score="strong", names=False),
                 "gcg_spfwh_names": dict(score="spfwh", names=True)}
TOPOLOGIES = {4: ((0, 1), (1, 0), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)),
              5: ((0, 1), (1, 0), (0, 2), (1, 2), (2, 1), (1, 3), (2, 4), (3, 4))}


def instance_name(cls: str, idx: int) -> str:
    return f"{NAME_PREFIX}_{cls}_{idx:02d}"


def draw_params(cls: str, rng: random.Random) -> dict:
    r = rng.randint
    if cls == "lot_sizing":
        # one setup per period and no initial stock: item B has no demand and no safety stock in period 1
        return {"demand": [[r(1, 4) for _ in range(3)], [0, r(1, 4), r(1, 4)]], "capacity": [r(5, 8) for _ in range(3)],
                "setup_time": [r(1, 2) for _ in range(2)], "min_lot": [r(1, 3) for _ in range(2)],
                "storage": [r(4, 7) for _ in range(3)], "safety": [r(0, 2), 0],
                "overtime": [r(1, 3) for _ in range(3)], "setup_budget": r(3, 5), "holding": [r(1, 2) for _ in range(2)],
                "setup": [r(4, 9) for _ in range(2)], "overtime_cost": r(2, 4)}
    if cls == "network_design":
        n = rng.choice((4, 5))
        arcs = TOPOLOGIES[n]
        dests = rng.sample(range(2, n), 2) if n > 3 else [n - 1, n - 2]
        fixed = [r(4, 10) for _ in arcs]
        return {"n_nodes": n, "arcs": [list(a) for a in arcs],
                "commodities": [[0, dests[0], r(1, 3)], [0, dests[1], r(1, 3)]],
                "capacity": [r(2, 5) for _ in arcs], "fixed": fixed, "unit": [1 for _ in arcs],
                "budget": int(sum(fixed) * rng.uniform(0.6, 0.9)), "max_out": [r(2, 3)] + [r(1, 2) for _ in range(n - 1)]}
    J = rng.choice((3, 4))
    open_cost = [r(8, 16) for _ in range(3)]
    return {"demand": [r(1, 4) for _ in range(J)], "capacity": [r(5, 10) for _ in range(3)], "open_cost": open_cost,
            "assign_cost": [[r(1, 3) for _ in range(J)] for _ in range(3)],
            "budget": int(sum(open_cost) * rng.uniform(0.65, 0.95)), "min_open": r(1, 2),
            "min_use": [r(0, 2) for _ in range(3)], "conflict": sorted(rng.sample(range(3), 2))}


def build_milp(cls: str, params: dict, name: str):
    from llmdw.instances import templates

    p = {k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
         for k, v in params.items()}
    fn = {"lot_sizing": templates.rich_lot_sizing, "network_design": templates.rich_network_design,
          "facility_location": templates.rich_facility_location}[cls]
    return fn(**p, name=name)


def evaluate(cls: str, idx: int, params: dict) -> dict:
    """The acceptance checks of one candidate and, for an accepted one, the exact gains of the sets of at most 3
    families."""
    from llmdw.exact import hull

    name = instance_name(cls, idx)
    m = build_milp(cls, params, name)
    out = {"class": cls, "candidate": idx, "name": name, "params": params, "k": len(m.families),
           "binaries": sum(v.integer for v in m.vars)}
    t0 = time.time()
    sc = m.to_scip()
    sc.hideOutput()
    sc.setParam("limits/time", 600)
    sc.optimize()
    if sc.getStatus() != "optimal":
        return {**out, "accepted": False, "reason": f"MILP status {sc.getStatus()}"}
    ip = Fraction(sc.getObjVal()).limit_denominator(10 ** 6)
    try:
        lp = hull.lp_bound(m)
    except RuntimeError as e:
        return {**out, "accepted": False, "reason": f"LP failed: {str(e)[:120]}"}
    gap = float((ip - lp) / abs(ip)) if ip else float("inf")
    out.update({"lp_bound": str(lp), "ip_scip": float(ip), "lp_gap": gap})
    if gap < MIN_GAP:
        return {**out, "accepted": False, "reason": f"LP gap {gap:.4f} below {MIN_GAP}"}
    fams = m.families
    tab = {frozenset(): Fraction(0)}
    for r in range(1, MAX_SET + 1):
        for T in itertools.combinations(fams, r):
            tab[frozenset(T)] = hull.dw_bound(m, T) - lp
    best = max(tab.values())
    single = max(tab[frozenset([f])] for f in fams)
    out.update({"best_gain": float(best), "best_single_gain": float(single), "seconds": round(time.time() - t0, 1)})
    if best <= 0:
        return {**out, "accepted": False, "reason": "no positive gain among sets of at most 3 families"}
    if single > MAX_SINGLE_SHARE * best:
        return {**out, "accepted": False, "reason": f"a single family closes {float(single / best):.2f} of the best gain"}
    groups = [W for W in hull.zero_marginal_groups(tab) if len(W) >= 2]
    sources = {}
    for W in groups:
        try:
            sources["|".join(sorted(W))] = hull.group_source(m, W)
        except RuntimeError as e:
            sources["|".join(sorted(W))] = f"not determined ({str(e)[:60]})"
    out.update({"accepted": True, "table": {"|".join(sorted(T)): str(v) for T, v in tab.items()},
                "groups": [sorted(W) for W in groups], "group_sources": sources,
                "group_gains": {"|".join(sorted(W)): float(tab[frozenset(W)]) for W in groups},
                "seconds": round(time.time() - t0, 1)})
    return out


def write_instance(res: dict, root: Path) -> Path:
    from llmdw.lookahead.lookahead import lookahead_table

    m = build_milp(res["class"], res["params"], res["name"])
    d = root / "instances" / res["name"]
    d.mkdir(parents=True, exist_ok=True)
    m.write_mps(str(d / "model.mps"))
    (d / "instance.json").write_text(json.dumps({"abstract_problem": m.description}, indent=1), encoding="utf-8")
    lines = ["# Formulation", "", "Constraint families:"] + [f"- {f}: {m.family_text.get(f, '')}" for f in m.families]
    (d / "model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tab = {k: Fraction(v) for k, v in res["table"].items()}
    best = max(tab.values())
    gt = {"families": m.families, "lp_bound": res["lp_bound"], "gain": res["table"], "joint_supports": res["groups"],
          "group_sources": res["group_sources"],
          "strengthens_alone": sorted(f for f in m.families if tab[f] > 0),
          "best_sets": sorted((k.split("|") for k, v in tab.items() if v == best and k), key=len),
          "note": f"exact gains of the sets of at most {MAX_SET} families; IP optimum (SCIP) {res['ip_scip']}"}
    (d / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    la = lookahead_table(str(d / "model.mps"), block_time=60.0, max_size=MAX_SET)
    (d / "lookahead.json").write_text(json.dumps(la, indent=1), encoding="utf-8")
    return d


def gcg_anchors(name: str, root: Path, time_limit: float = 120.0) -> dict:
    """GCG's detected decompositions in the oracle setting, as in oracle.run_oracle: {baseline: convexified families
    or None (not representable or degenerate)} with the status of the run."""
    from llmdw.gcg import gcg_run
    from llmdw.gcg.dec_io import aligned_master_families, family_profile, is_component_partition, read_dec_full
    from llmdw.instances.families import family_of, read_model

    d = root / "instances" / name
    mps = str(d / "model.mps")
    model = read_model(mps)
    fams = sorted({family_of(r.name) for r in model.rows})
    out = {}
    for b, opts in GCG_BASELINES.items():
        dec = str(d / f"{b}.dec")
        res = gcg_run.run("dw-auto", mps, str(d / f"{b}.json"), root=True, no_cuts=True, no_round=True, no_heur=True,
                          time_limit=time_limit, seed=0, write_chosen=dec, **opts)
        row = {"status": res.get("status"), "dual_bound": res.get("dual_bound"), "convexified": None}
        if os.path.exists(dec):
            dd = read_dec_full(dec)
            prof = family_profile(dd["master"], [r.name for r in model.rows])
            aligned = aligned_master_families(prof)
            degenerate = (not dd["master"]) and len(dd["blocks"]) == 1
            if not degenerate and aligned is not None and is_component_partition(model, dd["blocks"], dd["master"]):
                row["convexified"] = sorted(set(fams) - set(aligned))
            row["degenerate"] = degenerate
        out[b] = row
    (d / "gcg_anchors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def safe_evaluate(cls: str, idx: int, params: dict) -> dict:
    try:
        return evaluate(cls, idx, params)
    except Exception as e:  # noqa: BLE001 - a failed exact computation rejects the candidate, with the reason
        return {"class": cls, "candidate": idx, "name": instance_name(cls, idx), "params": params, "accepted": False,
                "reason": f"{type(e).__name__}: {str(e)[:160]}"}


def build(root: Path, parallel: int = 8, seed: int = 0, progress=None) -> dict:
    """Candidates per class from their own random stream, evaluated in parallel; new candidates of a class are drawn
    while its accepted and running candidates are fewer than PER_CLASS. Every finished candidate is stored in
    root/candidates/, and a restart reads it instead of evaluating it again (the streams are drawn in the same order,
    and the stored parameters are checked against the draw). progress, when given, receives each finished
    candidate."""
    root.mkdir(parents=True, exist_ok=True)
    rngs = {c: random.Random(seed * 100 + i) for i, c in enumerate(CLASSES)}
    idx = {c: 0 for c in CLASSES}
    results = {c: {} for c in CLASSES}
    futures = {}
    cdir = root / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)

    def store(res):
        f = cdir / f"{res['class']}_{res['candidate']:03d}.json"
        tmp = f.with_suffix(".json.part")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        os.replace(tmp, f)

    with multiprocessing.get_context("fork").Pool(parallel) as pool:
        def draw_next(c):
            params = draw_params(c, rngs[c])         # the stream is drawn in order, also for stored candidates
            f = cdir / f"{c}_{idx[c]:03d}.json"
            if f.exists():
                res = json.loads(f.read_text(encoding="utf-8"))
                if res["params"] != json.loads(json.dumps(params)):
                    raise RuntimeError(f"{f.name}: stored parameters differ from the seeded draw")
                results[c][idx[c]] = res
            else:
                futures[pool.apply_async(safe_evaluate, (c, idx[c], params))] = c
            idx[c] += 1

        def running(c):
            return sum(1 for fc in futures.values() if fc == c)

        def short(c):
            return sum(1 for r in results[c].values() if r["accepted"]) + running(c) < PER_CLASS

        quota = max(1, parallel // len(CLASSES))       # each class gets its share of the slots, the rest is shared
        while True:
            for c in CLASSES:
                while short(c) and running(c) < quota and len(futures) < parallel:
                    draw_next(c)
            for c in CLASSES:
                while short(c) and len(futures) < parallel:
                    draw_next(c)
            if not futures and not any(short(c) for c in CLASSES):
                break
            if not futures:
                continue
            done = next((f for f in futures if f.ready()), None)
            if done is None:
                time.sleep(2)
                continue
            c = futures.pop(done)
            res = done.get()
            store(res)
            results[c][res["candidate"]] = res
            if progress is not None:
                progress(res)
    accepted = {c: [results[c][j] for j in sorted(results[c]) if results[c][j]["accepted"]][:PER_CLASS]
                for c in CLASSES}
    draws = [{k: v for k, v in r.items() if k != "table"} for c in CLASSES for _, r in sorted(results[c].items())]
    (root / "draws.json").write_text(json.dumps({"seed": seed, "streams": "random.Random(seed * 100 + class index)",
                                                 "draws": draws}, indent=1), encoding="utf-8")
    names = []
    for c in CLASSES:
        for res in accepted[c]:
            write_instance(res, root)
            names.append(res["name"])
    (root / "instances.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return {"accepted": {c: len(v) for c, v in accepted.items()}, "draws": len(draws)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (instances/, candidates/, draws.json, instances.txt)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.out)
    keys = ("name", "accepted", "reason", "k", "lp_gap", "best_gain", "best_single_gain", "seconds")
    res = build(root, args.parallel, args.seed,
                progress=lambda r: print(json.dumps({k: r.get(k) for k in keys}), flush=True))
    print(json.dumps(res), flush=True)
    names = [n for n in (root / "instances.txt").read_text().split() if n]
    pending = [n for n in names if not (root / "instances" / n / "gcg_anchors.json").exists()]
    with ThreadPoolExecutor(args.parallel) as pool:
        for n, r in zip(pending, pool.map(lambda x: gcg_anchors(x, root), pending)):
            print(n, json.dumps({b: (x["status"], x["convexified"]) for b, x in r.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
