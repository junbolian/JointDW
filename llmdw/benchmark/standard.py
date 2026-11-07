"""Experiment C: standard published formulations on their own benchmark data (three constraint families each, no added
families, no generated parameters).

lot sizing         TTM (Trigeiro, Thomas and McClain) data, the cross-check model of llmdw.benchmark.trigeiro: Balance
                   (s_i,t-1 + x_it - s_it = d_it, no initial stock), SetupLink (x_it <= M_it y_it, M_it the remaining
                   demand of item i from period t on), Capacity (sum_i x_it + st_i y_it <= C_t); no overtime
network design     Canad R data: FlowBalance, ArcLink (x_qa <= min(d_q, u_a) y_a), ArcCapacity (sum_q x_qa <= u_a y_a);
                   flow cost plus fixed cost
facility location  OR-Library cap data, llmdw.instances.templates.facility_location: Assign (split demand,
                   sum_i x_ij = 1), Capacity (sum_j d_j x_ij <= s_i y_i), Link (x_ij <= y_i); the per-unit serving cost
                   is the cost of allocating all demand divided by the demand

Selection (before any evaluation): lot sizing, the 20 smallest of the 74 usable cross-checked instances by items x
periods, ties by TTM name (class letter, number, suffix); network design and facility location, every instance whose
LP relaxation is feasible and whose LP gap against the SCIP optimum (one thread, seed 0, one hour) is positive, then the
20 smallest by number of variables, ties by name. Instances are screened in increasing size until 20 pass, which
selects the same instances as screening all of them. Instances without a proved optimum are marked.

Usage
-----
python -m llmdw.benchmark.standard build --raw work/benchmark/raw --out work/benchmark_std
python -m llmdw.benchmark.standard select --out work/benchmark_std --parallel 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import string
import tarfile
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

from llmdw.benchmark.build import F, GAP_TOL, REFERENCE_TIME, _reference, name_key, read_canad, read_cap
from llmdw.instances.milp import MILP

PER_CLASS = 20
FIRST_BATCH_PER_CLASS = 2
CLASSES = ("lot_sizing", "network_design", "facility_location")


def letters(k: int) -> str:
    """A, B, ..., Z, AA, AB, ... (spreadsheet column names)."""
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = string.ascii_uppercase[r] + s
    return s


def lot_sizing(d: dict, name: str) -> MILP:
    I, T = d["items"], d["periods"]
    dem = [[F(v) for v in row] for row in d["demand"]]
    m = MILP(name)
    x, s, y = {}, {}, {}
    for i in range(I):
        rem = [sum(dem[i][t:]) for t in range(T)]
        for t in range(T):
            x[i, t] = m.add_var(f"Produce_{letters(i)}_{t + 1}", 0, rem[t])
            s[i, t] = m.add_var(f"Stock_{letters(i)}_{t + 1}", 0, sum(dem[i]))
            y[i, t] = m.add_var(f"Setup_{letters(i)}_{t + 1}", 0, 1, integer=True)
    for i in range(I):
        for t in range(T):
            coefs = {x[i, t]: 1, s[i, t]: -1}
            if t > 0:
                coefs[s[i, t - 1]] = 1
            m.add_row(f"Balance_{i + 1}_{t + 1}", "Balance_#_#", coefs, lo=dem[i][t], up=dem[i][t])
    for i in range(I):
        for t in range(T):
            m.add_row(f"SetupLink_{i + 1}_{t + 1}", "SetupLink_#_#", {x[i, t]: 1, y[i, t]: -m.vars[x[i, t]].ub}, up=0)
    for t in range(T):
        m.add_row(f"Capacity_{t + 1}", "Capacity_#",
                  {**{x[i, t]: 1 for i in range(I)}, **{y[i, t]: F(d["setup_time"][i]) for i in range(I)}},
                  up=F(d["capacity"][t]))
    obj = {}
    for i in range(I):
        for t in range(T):
            obj[s[i, t]] = F(d["holding"][i])
            obj[y[i, t]] = F(d["setup_cost"][i])
    m.set_objective(obj)
    m.description = (f"{I} products are made on one machine over {T} periods to meet known demands. Each period the "
                     "machine has a limited production capacity shared by all products. Producing a product in a "
                     "period requires a setup for that product in that period, which uses part of the capacity. "
                     "Minimize setup and holding cost.")
    m.family_text = {"Balance_#_#": "inventory balance of each product in each period",
                     "SetupLink_#_#": "a product can be produced in a period only if its setup is paid",
                     "Capacity_#": "production plus setup times of all products in a period are at most the capacity"}
    return m


def network_design(d: dict, name: str) -> MILP:
    arcs, comms, n = d["arcs"], d["commodities"], d["nodes"]
    m = MILP(name)
    x = {(k, a): m.add_var(f"Flow_{letters(k)}_{a + 1}", 0, F(c["demand"]))
         for k, c in enumerate(comms) for a in range(len(arcs))}
    y = [m.add_var(f"Open_{a + 1}", 0, 1, integer=True) for a in range(len(arcs))]
    for k, c in enumerate(comms):
        for v in range(n):
            coefs = {}
            for a, arc in enumerate(arcs):
                if arc["from"] == v:
                    coefs[x[k, a]] = coefs.get(x[k, a], 0) + 1
                if arc["to"] == v:
                    coefs[x[k, a]] = coefs.get(x[k, a], 0) - 1
            if not coefs:
                continue
            rhs = F(c["demand"]) if v == c["origin"] else (-F(c["demand"]) if v == c["destination"] else 0)
            m.add_row(f"FlowBalance_{k + 1}_{v + 1}", "FlowBalance_#_#", coefs, lo=rhs, up=rhs)
    for k, c in enumerate(comms):
        for a, arc in enumerate(arcs):
            m.add_row(f"ArcLink_{k + 1}_{a + 1}", "ArcLink_#_#",
                      {x[k, a]: 1, y[a]: -min(F(c["demand"]), F(arc["capacity"]))}, up=0)
    for a, arc in enumerate(arcs):
        m.add_row(f"ArcCapacity_{a + 1}", "ArcCapacity_#",
                  {**{x[k, a]: 1 for k in range(len(comms))}, y[a]: -F(arc["capacity"])}, up=0)
    obj = {y[a]: F(arc["fixed"]) for a, arc in enumerate(arcs)}
    for (k, a), j in x.items():
        obj[j] = F(arcs[a]["unit"])
    m.set_objective(obj)
    m.description = (f"{len(comms)} commodities must be shipped from their origins to their destinations through a "
                     f"network of {n} nodes and {len(arcs)} directed arcs. An arc can carry flow only if it is opened, "
                     "which has a fixed cost; an opened arc has a capacity shared by all commodities, and every unit "
                     "of flow has a per-arc cost. Minimize fixed plus flow cost.")
    m.family_text = {"FlowBalance_#_#": "flow conservation of each commodity at each node",
                     "ArcLink_#_#": "a commodity can use an arc only if the arc is opened, up to the smaller of its "
                                    "demand and the arc capacity",
                     "ArcCapacity_#": "the total flow on an arc is at most its capacity if the arc is opened"}
    return m


def facility_location(d: dict, name: str) -> MILP:
    """llmdw.instances.templates.facility_location with facility names A, B, ..., Z, AA, ... (the template's own names
    for up to 26 facilities, so its model files are reproduced exactly there)."""
    I, J = d["facilities"], d["customers"]
    demand, cap, open_cost = d["demand"], d["capacity"], d["fixed"]
    m = MILP(name)
    x = {(i, j): m.add_var(f"Serve_{letters(i)}_{j + 1}", 0, 1) for i in range(I) for j in range(J)}
    y = [m.add_var(f"Open_{letters(i)}", 0, 1, integer=True) for i in range(I)]
    for j in range(J):
        m.add_row(f"Assign_{j + 1}", "Assign_#", {x[i, j]: 1 for i in range(I)}, lo=1, up=1)
    for i in range(I):
        m.add_row(f"Capacity_{i + 1}", "Capacity_#", {**{x[i, j]: demand[j] for j in range(J)}, y[i]: -cap[i]}, up=0)
    for i in range(I):
        for j in range(J):
            m.add_row(f"Link_{i + 1}_{j + 1}", "Link_#_#", {x[i, j]: 1, y[i]: -1}, up=0)
    obj = {y[i]: open_cost[i] for i in range(I)}
    for (i, j), v in x.items():
        obj[v] = d["alloc"][i][j] / demand[j] * demand[j]
    m.set_objective(obj)
    m.description = (f"{J} customers with known demands are served from {I} candidate facilities. A facility "
                     "can serve customers only if it is opened, which has a fixed cost, and the demand it serves "
                     "is at most its capacity; a customer's demand may be split among facilities, and serving "
                     "costs depend on the facility and the customer. Minimize opening plus serving cost.")
    m.family_text = {"Assign_#": "the demand of each customer is fully served",
                     "Capacity_#": "the demand served by a facility is at most its capacity if it is opened",
                     "Link_#_#": "a customer can be served by a facility only if the facility is opened"}
    return m


def pools(raw: Path) -> dict:
    """{class: [(sort key, name, builder, data, source)]} in selection order."""
    from llmdw.benchmark import trigeiro

    manifest = {(f["source"], f["file"]): f for f in json.loads((raw / "sources.json").read_text())["files"]}
    out = {}
    data = trigeiro.read_suerie(raw)
    names, _ = trigeiro.gcg_list(raw)
    ls = []
    for g in names:
        t = trigeiro.ttm_name(g)
        if t in data and all(abs(c - 1) < 1e-12 for c in data[t]["prod_coef"]):
            d = data[t]
            ls.append(((d["items"] * d["periods"], name_key(t)), f"ls_{t}", d,
                       {"source": "trigeiro_suerie", "file": "data.tar.bz2", "member": d["folder"], "ttm": t,
                        "gcg_name": g, "sha256": manifest[("trigeiro_suerie", "data.tar.bz2")]["sha256"]}))
    out["lot_sizing"] = sorted(ls, key=lambda r: r[0])
    nd = []
    with tarfile.open(raw / "canad_r" / "R.tgz", "r:gz") as tf:
        for mem in tf.getmembers():
            if mem.isfile() and mem.name.endswith(".dow"):
                d = read_canad(tf.extractfile(mem).read().decode("latin-1"))
                base = Path(mem.name).name.removesuffix(".dow")
                nvar = len(d["arcs"]) * (len(d["commodities"]) + 1)
                nd.append(((nvar, [int(v) for v in base[1:].split(".")]), "nd_" + base.replace(".", "_"), d,
                           {"source": "canad_r", "file": "R.tgz", "member": mem.name,
                            "sha256": manifest[("canad_r", "R.tgz")]["sha256"]}))
    out["network_design"] = sorted(nd, key=lambda r: r[0])
    fl = []
    for f in sorted((raw / "orlib_cap").glob("cap*.txt")):
        d = read_cap(f.read_text(encoding="latin-1"))
        fl.append(((d["facilities"] * (d["customers"] + 1), int(f.stem[3:])), "fl_" + f.stem, d,
                   {"source": "orlib_cap", "file": f.name, "sha256": manifest[("orlib_cap", f.name)]["sha256"]}))
    out["facility_location"] = sorted(fl, key=lambda r: r[0])
    return out


BUILDERS = {"lot_sizing": lot_sizing, "network_design": network_design, "facility_location": facility_location}


def write(cls: str, name: str, d: dict, src: dict, root: Path, order: int) -> dict:
    m = BUILDERS[cls](d, name)
    dd = root / "instances" / name
    dd.mkdir(parents=True, exist_ok=True)
    m.write_mps(str(dd / "model.mps"))
    (dd / "instance.json").write_text(json.dumps({"abstract_problem": m.description}, indent=1), encoding="utf-8")
    lines = ["# Formulation", "", "Constraint families:"] + [f"- {f}: {m.family_text.get(f, '')}" for f in m.families]
    (dd / "model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"name": name, "class": cls, "size_order": order, "source": src, "families": m.families,
            "k": len(m.families), "rows": len(m.rows), "columns": len(m.vars),
            "binaries": sum(v.integer for v in m.vars),
            "mps_sha256": hashlib.sha256((dd / "model.mps").read_bytes()).hexdigest()}
    (dd / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def cmd_build(raw: Path, root: Path) -> dict:
    P = pools(raw)
    plan = {}
    for cls in CLASSES:
        cand = P[cls][:PER_CLASS] if cls == "lot_sizing" else P[cls]
        plan[cls] = []
        for order, (_, name, d, src) in enumerate(cand):
            try:
                meta = write(cls, name, d, src, root, order)
                plan[cls].append({"name": name, "rows": meta["rows"], "columns": meta["columns"]})
            except ValueError as e:
                plan[cls].append({"name": name, "not_built": str(e)})
    (root / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return {c: len(v) for c, v in plan.items()}


def _screen_one(root: Path, name: str) -> dict:
    from llmdw.benchmark.build import lp_bound

    d = root / "instances" / name
    lp = lp_bound(d)
    row = {"name": name, "lp_status": lp.get("status"), "lp_bound": lp.get("dual_bound") if lp.get("ok") else None}
    if not lp.get("ok") or lp.get("status") != "optimal":
        row["excluded"] = f"LP relaxation {lp.get('status') or lp.get('error')}"
        return row
    ref = json.loads((d / "reference.json").read_text()) if (d / "reference.json").exists() else None
    if ref is None:
        row["pending"] = True
        return row
    row.update({"reference_value": ref["best_value"], "proved_optimal": ref["proved_optimal"], "reference_gap": ref["gap"],
                "reference_status": ref["status"], "reference_seconds": ref["seconds"]})
    if ref["status"] == "infeasible":
        row["excluded"] = "MILP infeasible (SCIP)"
    elif ref["best_value"] is None:
        row["excluded"] = f"no solution found by SCIP within {int(ref['time_limit'])} s"
    else:
        gap = (ref["best_value"] - row["lp_bound"]) / abs(ref["best_value"]) if ref["best_value"] else float("inf")
        row["lp_gap"] = gap
        row["excluded"] = None if gap > GAP_TOL else f"no positive LP gap ({gap:.3g})"
    return row


def cmd_select(root: Path, parallel: int) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    result = {}
    for cls in CLASSES:
        built = [p["name"] for p in plan[cls] if "not_built" not in p]
        rows, passed = [], []
        i = 0
        while i < len(built) and (cls == "lot_sizing" or len(passed) < PER_CLASS):
            chunk = built[i:i + parallel]
            lp_rows = {n: _screen_one(root, n) for n in chunk}
            need = [n for n in chunk if lp_rows[n].get("pending")]
            with ProcessPoolExecutor(max_workers=parallel) as pool:
                list(pool.map(_reference, [(str(root / "instances" / n), REFERENCE_TIME) for n in need]))
            for n in chunk:
                r = _screen_one(root, n) if n in need else lp_rows[n]
                rows.append(r)
                if cls == "lot_sizing" or not r.get("excluded"):
                    passed.append(n)
            i += parallel
        sel = passed[:PER_CLASS]
        result[cls] = {"screened": rows, "selected": sel, "first_batch": sel[:FIRST_BATCH_PER_CLASS],
                       "not_screened": built[i:], "not_built": [p for p in plan[cls] if "not_built" in p]}
        print(cls, "screened", len(rows), "selected", len(sel), flush=True)
    (root / "selection.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "select"))
    ap.add_argument("--raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=20)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(cmd_build(Path(args.raw), Path(args.out))))
    else:
        cmd_select(Path(args.out), args.parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
