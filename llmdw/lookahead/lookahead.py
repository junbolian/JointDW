"""One-pricing-call lookahead bound for family-level decompositions.

Paper: Section 4 (side information from LP duals) and Section 6: the lookahead order of every table.

Let y be optimal LP duals of all rows and cbar = c - A^T y the LP reduced costs (minimization
form). For a set T of constraint families, split the rows of T into connected components C
(rows linked by a shared variable). The lookahead is

    l(T) = sum_C h(C),
    h(C) = min_{x in X_C} (c - sum_{i not in C} y_i a_i) x  -  sum_{i in C} y_i rhs_i(y_i)  -  cbar_C . x*_C

where X_C is the integer set of block C (its rows, bounds, integrality), rhs_i(y_i) is the row side
selected by the sign of y_i and x* is the LP optimum (cbar_C . x*_C is the box minimum of cbar on C by
complementary slackness). Then 0 <= l(T) <= z(T) - z_LP: l(T) is the Lagrangian bound of the
decomposition T evaluated at the LP duals (one optimization per block), so it is a valid lower bound
on the bound gain of T, and it is additive over connected components. When a block is not solved to
optimality within the time limit, its dual bound is used, which keeps l(T) a valid lower bound. Each block is solved by
SCIP with its default settings except one thread and SCIP propagation off (BLOCK_PROPAGATION).

Memory: when LLMDW_CHILD_MEMORY_GB is set, every block solve gets the SCIP memory limit 0.8 * cap (a block
stopped by it keeps its dual bound, so l(T) stays valid), and llmdw.gcg.batch runs each instance in its own
child process under the address-space cap (llmdw.gcg.address_limit).

Time: total_time bounds the whole instance. The budget is checked before every new block solve, so an
instance ends at most one block time after the budget runs out. The family set in progress then keeps the
sum over its evaluated blocks, which is still a valid lower bound because every block term is nonnegative,
and is marked complete = false; later family sets are not evaluated, and the result has truncated = true.

Usage
-----
python -m llmdw.lookahead.lookahead --mps work/mps/X.mps.gz --out work/lookahead/X.json --block-time 60 --max-families 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict

from llmdw.instances.families import family_of, read_model

INF = 1e19
# SCIP parameters set to 0 in every block solve: propagation off, as in every bound evaluation that feeds a gain
# (llmdw.gcg.gcg_run). A block is a MIP solved on its own (its rows, bounds and integrality), so its optimal value does
# not depend on propagation; only the dual bound of a block stopped at the time limit can.
BLOCK_PROPAGATION = ("propagating/maxrounds", "propagating/maxroundsroot")


def clean_row_duals(rows, y, tol: float) -> tuple[list, int]:
    """Row duals with every dual whose sign selects an infinite side (a one-sided row, or a free row) set to 0, and
    the number of duals set to 0. Such a dual is solver noise only up to the dual feasibility tolerance tol; a larger
    one is a wrong sign, and the instance fails with a message naming the row and the dual."""
    out, zeroed = list(y), 0
    for i, r in enumerate(rows):
        yi = out[i]
        if (yi > 0 and r.lhs <= -INF) or (yi < 0 and r.rhs >= INF):
            if abs(yi) > tol:
                raise RuntimeError(f"row {r.name}: dual {yi!r} selects an infinite side and exceeds the dual "
                                   f"feasibility tolerance {tol!r}")
            out[i] = 0.0
            zeroed += 1
    return out, zeroed


def lp_duals(model):
    """LP relaxation (min form) with HiGHS: returns z_LP, x*, row duals y (sign-checked), cbar."""
    import highspy
    import numpy as np

    sign = 1.0 if model.sense == "minimize" else -1.0
    names = list(model.var_type)
    col = {v: k for k, v in enumerate(names)}
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "off")
    solver = os.environ.get("LLMDW_LP_SOLVER", "highs")
    if solver == "highs-ipm":      # interior point with crossover (basic solution, so row duals as for simplex)
        highspy.Highs.resetGlobalScheduler(True)    # "threads" takes effect only on a fresh scheduler
        for opt, val in (("solver", "ipm"), ("run_crossover", "on"), ("threads", 1)):
            h.setOptionValue(opt, val)
    lo = np.array([-inf if model.var_lb[v] <= -INF else model.var_lb[v] for v in names])
    up = np.array([inf if model.var_ub[v] >= INF else model.var_ub[v] for v in names])
    c = np.array([sign * model.obj.get(v, 0.0) for v in names])
    h.addCols(len(names), c, lo, up, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([]))
    starts, idx, vals = [], [], []
    for r in model.rows:
        starts.append(len(idx))
        for v, a in r.coefs.items():
            idx.append(col[v])
            vals.append(a)
    rlo = np.array([-inf if r.lhs <= -INF else r.lhs for r in model.rows])
    rup = np.array([inf if r.rhs >= INF else r.rhs for r in model.rows])
    h.addRows(len(model.rows), rlo, rup, len(idx), np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32),
              np.array(vals))
    t_lp = time.time()
    h.run()
    lp_seconds = time.time() - t_lp
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"LP relaxation not optimal: {h.modelStatusToString(h.getModelStatus())} "
                           f"({solver}, {lp_seconds:.1f} s)")
    sol = h.getSolution()
    z = h.getInfo().objective_function_value
    x = list(sol.col_value)
    opt = h.getOptionValue("dual_feasibility_tolerance")          # highspy returns (status, value)
    dual_tol = float(opt[1] if isinstance(opt, tuple) else opt)
    y, duals_zeroed = clean_row_duals(model.rows, list(sol.row_dual), dual_tol)
    aty = [0.0] * len(names)
    for i, r in enumerate(model.rows):
        for v, a in r.coefs.items():
            aty[col[v]] += a * y[i]
    cbar = [c[k] - aty[k] for k in range(len(names))]

    def lagrangian(ys, cb):
        tot = sum(cb[k] * x[k] for k in range(len(names)))
        for i, r in enumerate(model.rows):
            if ys[i] > 0:
                tot += ys[i] * r.lhs
            elif ys[i] < 0:
                tot += ys[i] * r.rhs
        return tot

    if abs(lagrangian(y, cbar) - z) > 1e-6 * max(1.0, abs(z)):
        raise RuntimeError("dual sign convention check failed (Lagrangian at LP duals != LP value)")
    return {"z": z, "x": x, "y": y, "cbar": cbar, "names": names, "col": col, "sign": sign,
            "lp_solver": solver, "lp_seconds": lp_seconds, "duals_zeroed": duals_zeroed, "dual_tolerance": dual_tol}


def _components(model, rows):
    parent = {i: i for i in rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for i in rows:
        for v in model.rows[i].coefs:
            if v in owner:
                ra, rb = find(owner[v]), find(i)
                if ra != rb:
                    parent[ra] = rb
            else:
                owner[v] = i
    comp = defaultdict(list)
    for i in rows:
        comp[find(i)].append(i)
    return [sorted(c) for c in comp.values()]


def block_value(model, lp, comp_rows, time_limit: float) -> dict:
    """h(C) for one connected block (see module docstring)."""
    import pyscipopt

    rows = set(comp_rows)
    vars_ = sorted({v for i in comp_rows for v in model.rows[i].coefs})
    col, y = lp["col"], lp["y"]
    inside = defaultdict(float)  # sum_{i in C} y_i a_i per variable
    const = 0.0
    for i in comp_rows:
        r = model.rows[i]
        for v, a in r.coefs.items():
            inside[v] += y[i] * a
        if y[i] > 0:
            const -= y[i] * r.lhs
        elif y[i] < 0:
            const -= y[i] * r.rhs
    for v in vars_:
        const -= lp["cbar"][col[v]] * lp["x"][col[v]]
    m = pyscipopt.Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("parallel/maxnthreads", 1)
    for p in BLOCK_PROPAGATION:     # propagation off, as in every bound evaluation (llmdw.gcg.gcg_run)
        m.setParam(p, 0)
    cap_gb = os.environ.get("LLMDW_CHILD_MEMORY_GB")
    if cap_gb:  # graceful stop (status memlimit, dual bound kept) below the hard cap that llmdw.gcg.batch sets
        m.setParam("limits/memory", 0.8 * float(cap_gb) * 1024.0)
    xs = {}
    for v in vars_:
        t = model.var_type[v]
        lb = model.var_lb[v] if model.var_lb[v] > -INF else None
        ub = model.var_ub[v] if model.var_ub[v] < INF else None
        xs[v] = m.addVar(name=v, lb=lb, ub=ub, vtype={"B": "B", "I": "I", "C": "C"}[t])
    for i in comp_rows:
        r = model.rows[i]
        expr = pyscipopt.quicksum(a * xs[v] for v, a in r.coefs.items())
        lhs = r.lhs if r.lhs > -INF else None
        rhs = r.rhs if r.rhs < INF else None
        if lhs is not None and rhs is not None and lhs == rhs:
            m.addCons(expr == lhs)
        else:
            if lhs is not None:
                m.addCons(expr >= lhs)
            if rhs is not None:
                m.addCons(expr <= rhs)
    # block objective: c - sum_{i not in C} y_i a_i = cbar + sum_{i in C} y_i a_i
    m.setObjective(pyscipopt.quicksum((lp["cbar"][col[v]] + inside[v]) * xs[v] for v in vars_), "minimize")
    t0 = time.time()
    m.optimize()
    status = m.getStatus()
    if status == "optimal":
        val = m.getObjVal()
    elif status in ("infeasible",):
        return {"h": None, "status": status, "seconds": time.time() - t0, "rows": len(comp_rows), "vars": len(vars_)}
    else:
        val = m.getDualbound()
    return {"h": val + const, "status": status, "seconds": time.time() - t0, "rows": len(comp_rows),
            "vars": len(vars_)}


def lookahead_table(mps: str, max_families: int = 6, block_time: float = 60.0, total_time: float | None = None,
                    families=None, max_size: int | None = None) -> dict:
    """The lookahead of every family set (every size up to k, or up to max_size families when it is given; with
    max_size the limit max_families on k does not apply). Sets are evaluated by size, then in lexicographic order."""
    model = read_model(mps)
    lp = lp_duals(model)
    fam_rows = defaultdict(list)
    for i, r in enumerate(model.rows):
        fam_rows[family_of(r.name)].append(i)
    fams = families or sorted(fam_rows)
    if max_size is None and len(fams) > max_families:
        raise ValueError(f"{len(fams)} families > max_families={max_families}")
    cache, table = {}, {}
    t_start = time.time()
    time_limit_end = t_start + total_time if total_time else None
    truncated = False
    for size in range(1, min(len(fams), max_size or len(fams)) + 1):
        if truncated:
            break
        for T in itertools.combinations(fams, size):
            if time_limit_end is not None and time.time() > time_limit_end:
                truncated = True
                break
            rows = [i for f in T for i in fam_rows[f]]
            comps = _components(model, rows)
            total, statuses, secs, done = 0.0, [], 0.0, 0
            for comp in comps:
                key = tuple(comp)
                if key not in cache:
                    if time_limit_end is not None and time.time() > time_limit_end:
                        truncated = True
                        break
                    cache[key] = block_value(model, lp, comp, block_time)
                b = cache[key]
                statuses.append(b["status"])
                secs += b["seconds"]
                done += 1
                if b["h"] is None:
                    total = None
                    break
                total += max(0.0, b["h"])
            complete = total is None or done == len(comps)
            table["|".join(T)] = {"families": list(T), "lookahead": total, "n_blocks": len(comps),
                                  "blocks_evaluated": done, "complete": complete,
                                  "all_optimal": complete and all(s == "optimal" for s in statuses),
                                  "block_seconds": secs}
            if truncated:
                break
    # gains are in minimization units (a positive value tightens the bound); lp_bound includes the offset
    return {"mps": mps, "sense": model.sense, "lp_bound": lp["sign"] * lp["z"] + model.obj_offset, "families": fams,
            "table": table, "truncated": truncated, "seconds": time.time() - t_start,
            "lp_solver": lp["lp_solver"], "lp_seconds": lp["lp_seconds"], "duals_zeroed": lp["duals_zeroed"],
            "dual_tolerance": lp["dual_tolerance"], "block_propagation": {p: 0 for p in BLOCK_PROPAGATION}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-families", type=int, default=6)
    ap.add_argument("--block-time", type=float, default=60.0)
    ap.add_argument("--total-time", type=float, default=None)
    ap.add_argument("--max-size", type=int, default=None, help="only sets of at most this many families (for models with many families)")
    args = ap.parse_args(argv)
    res = lookahead_table(args.mps, args.max_families, args.block_time, args.total_time, max_size=args.max_size)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v["lookahead"] for k, v in res["table"].items()}))


if __name__ == "__main__":
    main()
