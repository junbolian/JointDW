"""Lookahead at further optimal LP duals, and the best-of-five-duals lookahead.

The one-pass lookahead (llmdw.lookahead.lookahead) evaluates every family set at the dual HiGHS returns. Four further
optimal duals are taken from the optimal face of the dual of the LP relaxation: dual seed s (1 to 4) maximizes r_s . y
for r_s standard normal (numpy default_rng(s)) subject to dual feasibility and dual objective at least
z_LP - 1e-9 max(1, |z_LP|). The face can contain lines (for example the duals of flow-balance rows that sum to zero), so
every row dual is boxed to |y_i| <= 10 (1 + max |y_HiGHS|); the number of components at the box is recorded. The
lookahead at each further dual uses the same primal optimum x* (complementary slackness holds between any optimal
primal and any optimal dual, so the term cbar_C . x*_C stays the box minimum), the same family sets, the same block time
and the same instance time as the recorded single-dual lookahead. The best-of-five value of a set is the maximum over the
five duals (the recorded HiGHS dual and the four further ones) of the values that exist.

Paper: Section 4 (side information from LP duals); the five-dual variant answers the objection that one dual is
arbitrary when the dual optimum is not unique.

Usage
-----
python -m llmdw.lookahead.five_duals one --mps M --lookahead L.json --seed 1 --out D1.json [--block-time 60 --total-time 3600]
python -m llmdw.lookahead.five_duals combine --lookahead L.json --duals D1.json D2.json D3.json D4.json --out L5.json
python -m llmdw.lookahead.five_duals batch --work work --out work/five_duals --parallel 20
    every instance of the three sets of the paper with its recorded lookahead: templates (work/templates,
    work/lookahead_templates; block time 60 s, no instance limit), the 102 scored MIPLIB-NL models (work/miplib_set,
    work/mps, work/lookahead; block time 60 s, instance limit 3600 s, as llmdw.gcg.batch lookahead) and the rich models
    (work/rich_models; block time 60 s, no instance limit, as llmdw.instances.rich_models_build); one child process per
    instance and dual seed, one thread, address space capped at LLMDW_CHILD_MEMORY_GB when it is set (--high-memory-gb
    for the instances listed in --high-memory); a failed child is retried twice. --lookahead-root reads the recorded
    tables from another directory with the same layout. Block solves use SCIP propagation off
    (llmdw.lookahead.lookahead.BLOCK_PROPAGATION).
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import time
from collections import defaultdict
from pathlib import Path

DUAL_SEEDS = (1, 2, 3, 4)
FACE_TOL = 1e-9
BOX_FACTOR = 10.0
DIFFER_TOL = 1e-6


def optimal_face_dual(model, lp: dict, seed: int) -> dict:
    """Row duals y (min form) maximizing r_seed . y over the boxed optimal face of the dual of the LP relaxation.

    Dual of min c x, lhs <= A x <= rhs, lb <= x <= ub: variables yL_i >= 0 (finite lhs), yU_i <= 0 (finite rhs),
    dL_j >= 0 (finite lb), dU_j <= 0 (finite ub); A^T (yL + yU) + dL + dU = c; objective
    lhs.yL + rhs.yU + lb.dL + ub.dU >= z_LP - tol. The row dual is y = yL + yU."""
    import highspy
    import numpy as np

    from llmdw.lookahead.lookahead import INF

    names, col, sign = lp["names"], lp["col"], lp["sign"]
    m, n = len(model.rows), len(names)
    box = BOX_FACTOR * (1.0 + max((abs(v) for v in lp["y"]), default=0.0))
    c = np.array([sign * model.obj.get(v, 0.0) for v in names])
    kinds, idxs, objs, los, ups = [], [], [], [], []

    def add(kind, i, objc, lo, up):
        kinds.append(kind), idxs.append(i), objs.append(objc), los.append(lo), ups.append(up)

    for i, r in enumerate(model.rows):
        if r.lhs > -INF:
            add(0, i, r.lhs, 0.0, box)
        if r.rhs < INF:
            add(1, i, r.rhs, -box, 0.0)
    for j, v in enumerate(names):
        if model.var_lb[v] > -INF:
            add(2, j, model.var_lb[v], 0.0, highspy.kHighsInf)
        if model.var_ub[v] < INF:
            add(3, j, model.var_ub[v], -highspy.kHighsInf, 0.0)
    starts, rix, vals = [], [], []
    for k in range(len(kinds)):
        starts.append(len(rix))
        if kinds[k] < 2:
            for v, a in model.rows[idxs[k]].coefs.items():
                rix.append(col[v])
                vals.append(a)
        else:
            rix.append(idxs[k])
            vals.append(1.0)
        if objs[k]:
            rix.append(n)
            vals.append(objs[k])
    z = lp["z"]
    r = np.random.default_rng(seed).standard_normal(m)
    cost = np.array([-r[idxs[k]] if kinds[k] < 2 else 0.0 for k in range(len(kinds))])
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "off")
    rlo = np.concatenate([c, [z - FACE_TOL * max(1.0, abs(z))]])
    rup = np.concatenate([c, [highspy.kHighsInf]])
    t0 = time.time()
    st = [h.addRows(n + 1, rlo, rup, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([])),
          h.addCols(len(kinds), cost, np.array(los), np.array(ups), len(rix), np.array(starts, dtype=np.int32),
                    np.array(rix, dtype=np.int32), np.array(vals, dtype=float)),
          h.run()]
    status = h.modelStatusToString(h.getModelStatus())
    out = {"seed": seed, "status": status, "box": box, "face_seconds": round(time.time() - t0, 3)}
    if any(x != highspy.HighsStatus.kOk for x in st) or h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return {**out, "y": None}
    x = list(h.getSolution().col_value)
    y = [0.0] * m
    for k in range(len(kinds)):
        if kinds[k] < 2:
            y[idxs[k]] += x[k]
    out.update({"y": y, "dual_objective": float(sum(x[k] * objs[k] for k in range(len(kinds)))), "z_lp": z,
                "components_at_box": sum(1 for v in y if abs(v) >= box * (1 - 1e-9))})
    return out


def lp_with_duals(model, lp: dict, y: list) -> dict:
    """The lp dict of llmdw.lookahead.lookahead with other row duals and the same primal optimum x*."""
    from llmdw.lookahead.lookahead import clean_row_duals

    y, zeroed = clean_row_duals(model.rows, list(y), lp["dual_tolerance"] * 10)
    names, col = lp["names"], lp["col"]
    c = [lp["sign"] * model.obj.get(v, 0.0) for v in names]
    aty = [0.0] * len(names)
    for i, r in enumerate(model.rows):
        for v, a in r.coefs.items():
            aty[col[v]] += a * y[i]
    cbar = [c[k] - aty[k] for k in range(len(names))]
    lag = sum(cbar[k] * lp["x"][k] for k in range(len(names)))
    for i, r in enumerate(model.rows):
        if y[i] > 0:
            lag += y[i] * r.lhs
        elif y[i] < 0:
            lag += y[i] * r.rhs
    return {**lp, "y": y, "cbar": cbar, "duals_zeroed": zeroed, "lagrangian_at_x": lag}


def lookahead_at(model, lp: dict, sets: list, block_time: float, total_time: float | None) -> dict:
    """The lookahead of the given family sets (lists of families, in the recorded order) at the duals of `lp`, with
    the block and instance time rules of llmdw.lookahead.lookahead.lookahead_table."""
    from llmdw.instances.families import family_of
    from llmdw.lookahead.lookahead import _components, block_value

    fam_rows = defaultdict(list)
    for i, r in enumerate(model.rows):
        fam_rows[family_of(r.name)].append(i)
    cache, table = {}, {}
    t_start = time.time()
    end = t_start + total_time if total_time else None
    truncated = False
    for T in sets:
        if end is not None and time.time() > end:
            truncated = True
            break
        rows = [i for f in T for i in fam_rows[f]]
        comps = _components(model, rows)
        total, statuses, done = 0.0, [], 0
        for comp in comps:
            key = tuple(comp)
            if key not in cache:
                if end is not None and time.time() > end:
                    truncated = True
                    break
                cache[key] = block_value(model, lp, comp, block_time)
            b = cache[key]
            statuses.append(b["status"])
            done += 1
            if b["h"] is None:
                total = None
                break
            total += max(0.0, b["h"])
        complete = total is None or done == len(comps)
        table["|".join(T)] = {"families": list(T), "lookahead": total, "complete": complete,
                              "all_optimal": complete and all(s == "optimal" for s in statuses)}
        if truncated:
            break
    return {"table": table, "truncated": truncated, "seconds": time.time() - t_start}


def one(mps: str, recorded: str, seed: int, out: str, block_time: float, total_time: float | None) -> dict:
    from llmdw.instances.families import read_model
    from llmdw.lookahead.lookahead import lp_duals

    rec = json.loads(Path(recorded).read_text(encoding="utf-8"))
    if rec.get("lp_solver") and rec["lp_solver"] != "highs":
        os.environ["LLMDW_LP_SOLVER"] = rec["lp_solver"]
    t0 = time.time()
    model = read_model(mps)
    lp = lp_duals(model)
    lp_bound = lp["sign"] * lp["z"] + model.obj_offset
    res = {"mps": mps, "recorded": recorded, "seed": seed, "block_time": block_time, "total_time": total_time,
           "lp_bound": lp_bound, "recorded_lp_bound": rec.get("lp_bound"),
           "lp_bound_agrees": abs(lp_bound - rec["lp_bound"]) <= 1e-7 * max(1.0, abs(rec["lp_bound"]))}
    d = optimal_face_dual(model, lp, seed)
    y = d.pop("y")
    res["dual"] = d
    if y is None:
        res.update({"table": None, "truncated": None, "seconds": round(time.time() - t0, 1)})
    else:
        lp2 = lp_with_duals(model, lp, y)
        y0 = lp["y"]
        res["dual"].update({"max_abs_difference_from_highs": max((abs(a - b) for a, b in zip(y, y0)), default=0.0),
                            "differs_from_highs": max((abs(a - b) for a, b in zip(y, y0)), default=0.0)
                            > DIFFER_TOL * (1.0 + max((abs(v) for v in y0), default=0.0)),
                            "lagrangian_at_x_minus_z": lp2["lagrangian_at_x"] - lp["z"],
                            "duals_zeroed": lp2["duals_zeroed"]})
        sets = [v["families"] for v in rec["table"].values()]
        la = lookahead_at(model, lp2, sets, block_time, total_time)
        res.update({"table": la["table"], "truncated": la["truncated"], "lookahead_seconds": round(la["seconds"], 1),
                    "seconds": round(time.time() - t0, 1)})
        with gzip.open(str(Path(out)) + ".y.gz", "wt", encoding="utf-8") as fh:
            json.dump(y, fh)
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def combine(recorded: str, duals: list, out: str) -> dict:
    """The best-of-five table: per set the maximum of the values that exist, with every value."""
    rec = json.loads(Path(recorded).read_text(encoding="utf-8"))
    ds = [json.loads(Path(p).read_text(encoding="utf-8")) for p in duals]
    table = {}
    for key, row in rec["table"].items():
        per = [row.get("lookahead")] + [(d["table"] or {}).get(key, {}).get("lookahead") for d in ds]
        vals = [v for v in per if v is not None]
        table[key] = {"families": row["families"], "lookahead": max(vals) if vals else None, "per_dual": per,
                      "argmax": per.index(max(vals)) if vals else None}
    limits = [{"seed": d["seed"], "truncated": d["truncated"], "status": d["dual"]["status"],
               "incomplete_sets": sum(1 for r in (d["table"] or {}).values() if not r["complete"]),
               "sets_without_optimal_blocks": sum(1 for r in (d["table"] or {}).values() if not r["all_optimal"]),
               "missing_sets": len(rec["table"]) - len(d["table"] or {})} for d in ds]
    res = {"recorded": recorded, "families": rec["families"], "sense": rec["sense"], "lp_bound": rec["lp_bound"],
           "table": table, "duals": [{"seed": 0, "source": "HiGHS (recorded lookahead)"}] + [d["dual"] for d in ds],
           "lp_bound_agrees": all(d["lp_bound_agrees"] for d in ds), "limits": limits,
           "recorded_truncated": rec.get("truncated")}
    Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def jobs(work: Path, lookahead_root: Path | None = None) -> list:
    """(set, name, mps, recorded lookahead, block time, instance time) of every instance. The recorded single-dual
    tables are read from lookahead_root (default: work) with the layout lookahead_templates/<name>.json,
    lookahead/<name>.json and rich_models/instances/<name>/lookahead.json."""
    la = Path(lookahead_root or work)
    out = []
    for d in sorted(p for p in (work / "templates").iterdir() if (p / "model.mps").exists()):
        out.append(("templates", d.name, str(d / "model.mps"), str(la / "lookahead_templates" / f"{d.name}.json"),
                    60.0, None))
    for d in sorted(p for p in (work / "miplib_set").iterdir() if p.is_dir()):
        out.append(("miplib", d.name, str(work / "mps" / f"{d.name}.mps.gz"), str(la / "lookahead" / f"{d.name}.json"),
                    60.0, 3600.0))
    for n in (work / "rich_models" / "instances.txt").read_text().split():
        d = work / "rich_models" / "instances" / n
        out.append(("rich_models", n, str(d / "model.mps"), str(la / "rich_models" / "instances" / n / "lookahead.json"),
                    60.0, None))
    return out


def batch(work: Path, root: Path, parallel: int, lookahead_root: Path | None = None, high_memory: set | None = None,
          high_memory_gb: float = 10.0, defer_high_memory: bool = False) -> None:
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from llmdw.gcg import memory_budget

    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cap = os.environ.get("LLMDW_CHILD_MEMORY_GB")
    todo = []
    for phase, name, mps, rec, bt, tt in jobs(work, lookahead_root):
        missing = [x for x in (mps, rec) if not Path(x).exists()]
        if missing:
            print(f"{phase}/{name}: missing {missing}", flush=True)
            continue
        if defer_high_memory and name in (high_memory or ()):
            continue        # left for a later run of the batch (resumable)
        for seed in DUAL_SEEDS:
            todo.append((phase, name, mps, rec, bt, tt, seed))
    # longest recorded lookahead first
    rec_secs = {r: json.loads(Path(r).read_text()).get("seconds") or 0 for _, _, _, r, _, _, _ in todo}
    todo.sort(key=lambda j: -rec_secs[j[3]])

    def work_one(j):
        phase, name, mps, rec, bt, tt, seed = j
        d = root / phase / name
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"dual{seed}.json"
        if out.exists():
            return j, "skip"
        cmd = [sys.executable, "-m", "llmdw.lookahead.five_duals", "one", "--mps", mps, "--lookahead", rec,
               "--seed", str(seed), "--out", str(out) + ".part", "--block-time", str(bt)]
        if tt:
            cmd += ["--total-time", str(tt)]
        c = str(high_memory_gb) if name in (high_memory or ()) else cap
        env1 = dict(env, **({"LLMDW_CHILD_MEMORY_GB": c} if c else {}))
        if c:
            from llmdw.gcg import address_limit
            cmd = address_limit.prefix(c) + cmd
        part = Path(str(out) + ".part")
        codes, clocks = [], []
        for attempt in range(3):            # a failed child is retried twice (llmdw.gcg.oracle.SAME_SETTING_RETRIES)
            log = d / (f"dual{seed}.log" if attempt == 0 else f"dual{seed}.a{attempt}.log")
            # shared memory budget (llmdw.gcg.memory_budget), waited for before the child starts
            with memory_budget.reserve(float(c) if c else None, key=mps) as wait:
                clock = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "mem_available_gb_start": memory_budget.mem_available_gb(), "budget_wait_seconds": round(wait, 3)}
                with open(log, "w", encoding="utf-8") as fh:
                    p = subprocess.run(cmd, cwd=str(repo), env=env1, stdout=fh, stderr=subprocess.STDOUT)
                clock.update({"ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                              "mem_available_gb_end": memory_budget.mem_available_gb()})
            codes.append(p.returncode)
            clocks.append(clock)
            if p.returncode == 0 and part.exists():
                r = json.loads(part.read_text(encoding="utf-8"))
                r.update({"attempts": codes, "attempt_clocks": clocks, "memory_gb": float(c) if c else None})
                part.write_text(json.dumps(r, indent=1), encoding="utf-8")
                if Path(str(part) + ".y.gz").exists():
                    os.replace(str(part) + ".y.gz", str(out) + ".y.gz")
                os.replace(part, out)
                return j, "done" + (f" (attempt {attempt})" if attempt else "")
        (d / f"dual{seed}.failed.json").write_text(json.dumps({"status": "failed", "exit_codes": codes,
                                                               "attempt_clocks": clocks,
                                                               "memory_gb": float(c) if c else None}),
                                                   encoding="utf-8")
        return j, f"failed (exits {codes})"

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for j, st in pool.map(work_one, todo):
            print(f"{j[0]}/{j[1]} seed {j[6]}: {st}", flush=True)
    for phase, name, mps, rec, bt, tt in jobs(work, lookahead_root):
        d = root / phase / name
        ds = [d / f"dual{s}.json" for s in DUAL_SEEDS]
        if all(x.exists() for x in ds):
            combine(rec, [str(x) for x in ds], str(d / "five_duals.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("one")
    o.add_argument("--mps", required=True)
    o.add_argument("--lookahead", required=True)
    o.add_argument("--seed", type=int, required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--block-time", type=float, default=60.0)
    o.add_argument("--total-time", type=float, default=None)
    c = sub.add_parser("combine")
    c.add_argument("--lookahead", required=True)
    c.add_argument("--duals", nargs="+", required=True)
    c.add_argument("--out", required=True)
    b = sub.add_parser("batch")
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--parallel", type=int, default=20)
    b.add_argument("--lookahead-root", help="where the recorded single-dual tables are (default: --work)")
    b.add_argument("--high-memory", help="file with the instance names that get the cap --high-memory-gb")
    b.add_argument("--high-memory-gb", type=float, default=10.0)
    b.add_argument("--defer-high-memory", action="store_true",
                   help="skip the --high-memory instances in this run (scheduling only; a later run computes them)")
    args = ap.parse_args(argv)
    if args.cmd == "batch":
        high = set(Path(args.high_memory).read_text(encoding="utf-8").split()) if args.high_memory else set()
        batch(Path(args.work), Path(args.out), args.parallel, Path(args.lookahead_root) if args.lookahead_root else None,
              high, args.high_memory_gb, args.defer_high_memory)
        return 0
    if args.cmd == "one":
        r = one(args.mps, args.lookahead, args.seed, args.out, args.block_time, args.total_time)
        print(json.dumps({k: r[k] for k in ("seed", "lp_bound_agrees", "truncated", "seconds")}))
    else:
        combine(args.lookahead, args.duals, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
