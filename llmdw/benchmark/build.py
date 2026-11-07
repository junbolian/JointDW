"""Rich models on public benchmark data: selection, side-constraint parameters, model files, optimum reference, screening.

The formulations are those of the rich models (llmdw.instances.templates: rich_lot_sizing, rich_network_design,
rich_facility_location; the same families, family names and row definitions). Every number the benchmark contains is
taken from it; every side-constraint parameter it does not contain is set by the rules below, fixed before any
instance was built and applied to every instance alike (no per-instance adjustment).

Rescaling rule. The rich-model generator (llmdw.instances.rich_models_build.draw_params) draws these parameters at
random on the scale of its own data. Each is fixed at the midpoint of the generator's range and multiplied by the ratio
of the benchmark mean to the generator mean of the quantity it is drawn against. Generator means: demand per item and
period 25/12 (item A draws 1..4 in all 3 periods, item B 1..4 in periods 2 and 3 and 0 in period 1), total demand per
period 25/6, capacity per period 6.5 (5..8), facility capacity 7.5 (5..10). Benchmark means are taken over the instance.

Lot sizing (TTM data: demand, capacity, setup times, setup costs, holding costs, overtime cost per unit):
  storage cap per period     5.5 (4..7) x mean total demand per period / (25/6)
  minimum lot per item       2 (1..3) x mean demand per item and period / (25/12)
  safety stock per item      0.5 (item A 0..2, midpoint 1; item B fixed 0; mean 0.5) x mean demand / (25/12)
  overtime limit per period  2 (1..3) x mean capacity / 6.5
  setup budget               4 of I x T = 6 (3..5), the same fraction: (2/3) x I x T
  setups per period          I in period 1 (no limit), ceil(I/2) in every later period (the
                             generator's limit 1 fits its two items only)
  overtime cost per unit     from the data (UEBER-KS.PRN)
Network design (Canad R data: arcs with variable cost, capacity, fixed cost; commodities with origin, destination,
demand):
  build budget               0.75 (0.6..0.9) x sum of the fixed costs
  out-degree limit of node v phi x out-degree of v, phi = 93/112: the generator's midpoints (2.5 at node 1, 1.5 at the
                             others) over its out-degrees, sum over the nodes with outgoing arcs, are 5.5/7 (4 nodes)
                             and 7/8 (5 nodes); the two topologies are drawn with equal probability. Not rounded.
Facility location (OR-Library cap data: capacity, fixed cost, demand, cost of allocating all demand):
  opening budget             0.8 (0.65..0.95) x sum of the opening costs
  minimum open count         1.5 (1..2) of 3 facilities, the same fraction: I / 2
  minimum utilization        1 (0..2) x mean capacity / 7.5, for every facility
  conflicting pair           facilities 1 and 2
  per-unit serving cost      allocation cost of all demand / demand (the template multiplies by the demand)

Selection (fixed before building): lot sizing from the 76 instances of GCG's ls-trigeiro.test that are in the suerie.de
data, without production coefficients other than 1, the 10 smallest by items x periods, ties by TTM name (class letter,
number, suffix); network design the 10 smallest R instances by arcs x commodities, ties by name; facility location the
10 smallest of cap41..cap134 by facilities x customers, ties by name.

Screening (llmdw.benchmark.build screen): an instance is excluded when its LP relaxation is infeasible, when SCIP
(one thread, seed 0, one hour) proves it infeasible or finds no solution, or when its LP gap (reference - LP) / |reference|
is not positive (at most 1e-9); the reference is SCIP's best solution value.

Usage
-----
python -m llmdw.benchmark.build build --raw work/benchmark/raw --out work/benchmark
python -m llmdw.benchmark.build reference --out work/benchmark --parallel 20
python -m llmdw.benchmark.build screen --out work/benchmark
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

CLASSES = ("lot_sizing", "network_design", "facility_location")
PER_CLASS = 10
FIRST_BATCH_PER_CLASS = 2
GEN_DEMAND = Fraction(25, 12)
GEN_TOTAL_DEMAND = Fraction(25, 6)
GEN_CAPACITY = Fraction(13, 2)
GEN_FACILITY_CAPACITY = Fraction(15, 2)
OUT_DEGREE_FRACTION = Fraction(93, 112)
REFERENCE_TIME = 3600.0
GAP_TOL = 1e-9


def F(x) -> Fraction:
    return Fraction(str(x)) if isinstance(x, float) else Fraction(x)


def name_key(name: str):
    m = re.match(r"([A-Za-z]+)(\d+)(.*)", name)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (name, 0, "")


# ---------------------------------------------------------------- readers

def read_canad(text: str) -> dict:
    nums = [int(x) for x in re.findall(r"-?\d+", text.split(":", 1)[1] if ":" in text.split("\n", 1)[0] else text)]
    n, A, K = nums[:3]
    p, arcs, comms = 3, [], []
    for _ in range(A):
        i, j, var, cap, fix, _a, _b = nums[p:p + 7]
        arcs.append({"from": i - 1, "to": j - 1, "unit": var, "capacity": cap, "fixed": fix})
        p += 7
    for _ in range(K):
        o, d, dem = nums[p:p + 3]
        comms.append({"origin": o - 1, "destination": d - 1, "demand": dem})
        p += 3
    return {"nodes": n, "arcs": arcs, "commodities": comms}


def read_cap(text: str) -> dict:
    tok = text.split()
    pos = 0

    def nxt():
        nonlocal pos
        pos += 1
        return tok[pos - 1]

    I, J = int(nxt()), int(nxt())
    cap, fixed = [], []
    for _ in range(I):
        cap.append(F(nxt().rstrip(".")))
        fixed.append(F(nxt().rstrip(".") or "0"))
    demand, alloc = [], [[None] * J for _ in range(I)]
    for j in range(J):
        demand.append(F(nxt().rstrip(".")))
        for i in range(I):
            alloc[i][j] = F(nxt())
    return {"facilities": I, "customers": J, "capacity": cap, "fixed": fixed, "demand": demand, "alloc": alloc}


# ---------------------------------------------------------------- parameters (the rules of the module docstring)

def lot_sizing_params(d: dict) -> dict:
    I, T = d["items"], d["periods"]
    dem = [[F(v) for v in row] for row in d["demand"]]
    cap = [F(v) for v in d["capacity"]]
    mean_dem = sum(sum(r) for r in dem) / (I * T)
    mean_tot = sum(sum(dem[i][t] for i in range(I)) for t in range(T)) / T
    mean_cap = sum(cap) / T
    return {"demand": dem, "capacity": cap, "setup_time": [F(v) for v in d["setup_time"]],
            "min_lot": [Fraction(2) * mean_dem / GEN_DEMAND] * I,
            "storage": [Fraction(11, 2) * mean_tot / GEN_TOTAL_DEMAND] * T,
            "safety": [Fraction(1, 2) * mean_dem / GEN_DEMAND] * I,
            "overtime": [Fraction(2) * mean_cap / GEN_CAPACITY] * T,
            "setup_budget": Fraction(2, 3) * I * T, "holding": [F(v) for v in d["holding"]],
            "setup": [F(v) for v in d["setup_cost"]], "overtime_cost": F(d["overtime_cost"]),
            "max_setups": [I] + [math.ceil(I / 2)] * (T - 1)}


def network_design_params(d: dict) -> dict:
    n = d["nodes"]
    arcs = [(a["from"], a["to"]) for a in d["arcs"]]
    outdeg = [sum(1 for i, _ in arcs if i == v) for v in range(n)]
    fixed = [F(a["fixed"]) for a in d["arcs"]]
    return {"n_nodes": n, "arcs": arcs,
            "commodities": [(c["origin"], c["destination"], F(c["demand"])) for c in d["commodities"]],
            "capacity": [F(a["capacity"]) for a in d["arcs"]], "fixed": fixed, "unit": [F(a["unit"]) for a in d["arcs"]],
            "budget": Fraction(3, 4) * sum(fixed), "max_out": [OUT_DEGREE_FRACTION * k for k in outdeg]}


def facility_location_params(d: dict) -> dict:
    I, J = d["facilities"], d["customers"]
    mean_cap = sum(d["capacity"]) / I
    return {"demand": d["demand"], "capacity": d["capacity"], "open_cost": d["fixed"],
            "assign_cost": [[d["alloc"][i][j] / d["demand"][j] for j in range(J)] for i in range(I)],
            "budget": Fraction(4, 5) * sum(d["fixed"]), "min_open": Fraction(I, 2),
            "min_use": [mean_cap / GEN_FACILITY_CAPACITY] * I, "conflict": (0, 1)}


def build_milp(cls: str, params: dict, name: str):
    from llmdw.instances import templates

    def tup(v):
        return tuple(tup(x) for x in v) if isinstance(v, (list, tuple)) else v

    p = {k: tup(v) for k, v in params.items()}
    fn = {"lot_sizing": templates.rich_lot_sizing, "network_design": templates.rich_network_design,
          "facility_location": templates.rich_facility_location}[cls]
    return fn(**p, name=name)


# ---------------------------------------------------------------- selection

def candidates(raw: Path) -> dict:
    """{class: (selected [(name, data, source)], excluded before selection [(name, reason)], sizes)}."""
    from llmdw.benchmark import trigeiro

    out = {}
    manifest = {(f["source"], f["file"]): f for f in json.loads((raw / "sources.json").read_text())["files"]}
    # lot sizing
    data = trigeiro.read_suerie(raw)
    names, _ = trigeiro.gcg_list(raw)
    pre, pool = [], []
    for g in names:
        t = trigeiro.ttm_name(g)
        if t not in data:
            pre.append((g, "not in the suerie.de data"))
            continue
        d = data[t]
        if any(abs(c - 1) > 1e-12 for c in d["prod_coef"]):
            pre.append((g, "production coefficients other than 1"))
            continue
        pool.append((d["items"] * d["periods"], name_key(t), f"ls_{t}", d,
                     {"source": "trigeiro_suerie", "file": "data.tar.bz2", "member": d["folder"], "ttm": t,
                      "gcg_name": g, "sha256": manifest[("trigeiro_suerie", "data.tar.bz2")]["sha256"]}))
    pool.sort(key=lambda x: (x[0], x[1]))
    out["lot_sizing"] = ([(n, d, s) for _, _, n, d, s in pool[:PER_CLASS]], pre,
                         [(n, sz) for sz, _, n, _, _ in pool])
    # network design
    pool = []
    with tarfile.open(raw / "canad_r" / "R.tgz", "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".dow"):
                d = read_canad(tf.extractfile(m).read().decode("latin-1"))
                base = Path(m.name).name.removesuffix(".dow")
                pool.append((len(d["arcs"]) * len(d["commodities"]), base, "nd_" + base.replace(".", "_"), d,
                             {"source": "canad_r", "file": "R.tgz", "member": m.name,
                              "sha256": manifest[("canad_r", "R.tgz")]["sha256"]}))
    pool.sort(key=lambda x: (x[0], [int(v) for v in re.findall(r"\d+", x[1])]))
    out["network_design"] = ([(n, d, s) for _, _, n, d, s in pool[:PER_CLASS]], [], [(n, sz) for sz, _, n, _, _ in pool])
    # facility location
    pool = []
    for f in sorted((raw / "orlib_cap").glob("cap*.txt")):
        d = read_cap(f.read_text(encoding="latin-1"))
        pool.append((d["facilities"] * d["customers"], int(f.stem[3:]), "fl_" + f.stem, d,
                     {"source": "orlib_cap", "file": f.name, "sha256": manifest[("orlib_cap", f.name)]["sha256"]}))
    pool.sort(key=lambda x: (x[0], x[1]))
    out["facility_location"] = ([(n, d, s) for _, _, n, d, s in pool[:PER_CLASS]], [],
                                [(n, sz) for sz, _, n, _, _ in pool])
    return out


def _jsonable(v):
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def cmd_build(raw: Path, root: Path) -> dict:
    sel = candidates(raw)
    inst_root = root / "instances"
    listing, info = [], {}
    for cls in CLASSES:
        chosen, pre, sizes = sel[cls]
        info[cls] = {"selected": [n for n, _, _ in chosen], "excluded_before_selection": pre,
                     "pool_size": len(sizes)}
        for rank, (name, d, src) in enumerate(chosen):
            params = {"lot_sizing": lot_sizing_params, "network_design": network_design_params,
                      "facility_location": facility_location_params}[cls](d)
            m = build_milp(cls, params, name)
            dd = inst_root / name
            dd.mkdir(parents=True, exist_ok=True)
            m.write_mps(str(dd / "model.mps"))
            (dd / "instance.json").write_text(json.dumps({"abstract_problem": m.description}, indent=1), encoding="utf-8")
            lines = ["# Formulation", "", "Constraint families:"] + [f"- {f}: {m.family_text.get(f, '')}"
                                                                    for f in m.families]
            (dd / "model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            size = {"lot_sizing": lambda: {"items": d["items"], "periods": d["periods"]},
                    "network_design": lambda: {"nodes": d["nodes"], "arcs": len(d["arcs"]),
                                               "commodities": len(d["commodities"])},
                    "facility_location": lambda: {"facilities": d["facilities"], "customers": d["customers"]}}[cls]()
            meta = {"name": name, "class": cls, "rank": rank, "first_batch": rank < FIRST_BATCH_PER_CLASS, "source": src,
                    "size": size, "families": m.families, "k": len(m.families), "rows": len(m.rows),
                    "columns": len(m.vars), "binaries": sum(v.integer for v in m.vars),
                    "mps_sha256": hashlib.sha256((dd / "model.mps").read_bytes()).hexdigest(),
                    "params": _jsonable(params)}
            (dd / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            listing.append(name)
    (root / "selection.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    (root / "built.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
    return info


# ---------------------------------------------------------------- reference and screening

def _reference(args) -> dict:
    import time

    import pyscipopt

    d, time_limit = args
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(str(Path(d) / "model.mps"))
    m.setParam("limits/time", time_limit)
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("lp/threads", 1)
    m.setParam("randomization/randomseedshift", 0)
    t0 = time.time()
    m.optimize()
    res = {"status": m.getStatus(), "best_value": m.getPrimalbound() if m.getNSols() else None,
           "dual_bound": m.getDualbound(), "gap": m.getGap() if m.getNSols() else None,
           "proved_optimal": m.getStatus() == "optimal", "seconds": round(time.time() - t0, 2),
           "time_limit": time_limit, "threads": 1, "seed": 0}
    (Path(d) / "reference.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_reference(root: Path, parallel: int, names=None, time_limit: float = REFERENCE_TIME) -> None:
    todo = [root / "instances" / n for n in (names or (root / "built.txt").read_text().split())
            if not (root / "instances" / n / "reference.json").exists()]
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        for d, r in zip(todo, pool.map(_reference, [(str(d), time_limit) for d in todo])):
            print(f"{d.name:24s} {r['status']:12s} best {r['best_value']} gap {r['gap']} {r['seconds']} s", flush=True)


def lp_bound(d: Path) -> dict:
    """LP bound of the instance in d (llmdw.gcg.oracle.lp_bound: gcg_run lp with the retries of failed runs)."""
    from llmdw.gcg.oracle import lp_bound as oracle_lp_bound

    return oracle_lp_bound(str(d / "model.mps"), d)


def cmd_screen(root: Path, names=None) -> dict:
    rows = []
    for n in names or (root / "built.txt").read_text().split():
        d = root / "instances" / n
        meta = json.loads((d / "meta.json").read_text())
        lp = lp_bound(d)
        ref = json.loads((d / "reference.json").read_text()) if (d / "reference.json").exists() else None
        row = {"name": n, "class": meta["class"], "rank": meta["rank"], "lp_status": lp.get("status"),
               "lp_bound": lp.get("dual_bound") if lp.get("ok") else None,
               "reference_status": ref and ref["status"], "reference_value": ref and ref["best_value"],
               "proved_optimal": ref and ref["proved_optimal"], "reference_gap": ref and ref["gap"]}
        if not lp.get("ok") or lp.get("status") != "optimal":
            row["excluded"] = f"LP relaxation {lp.get('status') or lp.get('error')}"
        elif ref is None:
            row["excluded"] = None
            row["pending"] = "no reference yet"
        elif ref["status"] == "infeasible":
            row["excluded"] = "MILP infeasible (SCIP)"
        elif ref["best_value"] is None:
            row["excluded"] = f"no solution found by SCIP within {int(ref['time_limit'])} s"
        else:
            v, lpb = ref["best_value"], row["lp_bound"]
            gap = (v - lpb) / abs(v) if v else float("inf")
            row["lp_gap"] = gap
            row["excluded"] = None if gap > GAP_TOL else f"no positive LP gap ({gap:.3g})"
        rows.append(row)
    (root / "screening.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "reference", "screen"))
    ap.add_argument("--raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=20)
    ap.add_argument("--names", help="comma-separated instance names (default: every built instance)")
    ap.add_argument("--time", type=float, default=REFERENCE_TIME)
    args = ap.parse_args(argv)
    root = Path(args.out)
    names = args.names.split(",") if args.names else None
    if args.cmd == "build":
        print(json.dumps(cmd_build(Path(args.raw), root), indent=1))
    elif args.cmd == "reference":
        cmd_reference(root, args.parallel, names, args.time)
    else:
        for r in cmd_screen(root, names):
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
