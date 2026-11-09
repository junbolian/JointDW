"""Trigeiro, Thomas and McClain (TTM) lot-sizing data: the suerie.de files, the plain TTM model, and the two checks of
the data before any instance is built.

suerie.de format (one folder test0NNN per instance): index.prn "items, periods, machines"; kapaz.prn capacity per period;
l0.prn initial stock per item; lagkost.prn holding cost per item; p-bedarf.prn demand, one line per item over the periods;
prodkoef.prn "machine, item, coefficient"; ruestk.prn setup cost per item; ruestz.prn "machine, item, setup time";
UEBER-KS.PRN overtime cost per unit. conv_ttm_new.txt maps each folder number to a TTM name.

Check (a): the plain TTM model (inventory balance, setup link, capacity with setup times; optionally overtime at the
cost of UEBER-KS.PRN) built from the suerie.de data for every instance of GCG's ls-trigeiro.test (name F10N is TTM F10,
G30 is TTM G30), solved with SCIP (one thread, seed 0, one hour), compared with ls-trigeiro.solu at 1e-6 relative.
Check (b): the four AMPL copies of LOTSIZELIB (tr6_15, tr6_30, tr12_30, tr24_30) matched to TTM names by their demand
matrices, then compared value by value (demand, capacity, production coefficient RHO, setup time ST, setup cost F, holding
cost H).

Usage
-----
python -m llmdw.benchmark.trigeiro check-a --raw work/benchmark/raw --out work/benchmark/trigeiro_check --parallel 20
python -m llmdw.benchmark.trigeiro check-b --raw work/benchmark/raw --out work/benchmark/trigeiro_check
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _nums(text: str) -> list[float]:
    return [float(x) for x in NUM.findall(text)]


def ttm_map(raw: Path) -> dict:
    """folder number -> TTM name (without .dat)."""
    out = {}
    for line in (raw / "trigeiro_suerie" / "conv_ttm_new.txt").read_text(encoding="latin-1").splitlines():
        m = re.match(r"\s*TTM:(\S+)\s+New format:\s*(\d+)", line)
        if m:
            out[int(m.group(2))] = m.group(1).removesuffix(".dat")
    return out


def read_suerie(raw: Path) -> dict:
    """{TTM name: instance dict} from data.tar.bz2, read in memory (the archive is kept byte-identical)."""
    names = ttm_map(raw)
    files = {}
    with tarfile.open(raw / "trigeiro_suerie" / "data.tar.bz2", "r:bz2") as tf:
        for m in tf.getmembers():
            if m.isfile():
                folder, fname = m.name.split("/")[-2:]
                files.setdefault(folder, {})[fname.lower()] = tf.extractfile(m).read().decode("latin-1")
    out = {}
    for folder, f in files.items():
        no = int(folder[4:])
        I, T, M = (int(x) for x in _nums(f["index.prn"])[:3])
        dem = _nums(f["p-bedarf.prn"])
        pk = _nums(f["prodkoef.prn"])
        st = _nums(f["ruestz.prn"])
        inst = {"ttm": names[no], "folder": folder, "items": I, "periods": T, "machines": M,
                "capacity": _nums(f["kapaz.prn"])[:T], "initial_stock": _nums(f["l0.prn"])[:I],
                "holding": _nums(f["lagkost.prn"])[:I], "demand": [dem[i * T:(i + 1) * T] for i in range(I)],
                "prod_coef": pk[2::3][:I], "setup_cost": _nums(f["ruestk.prn"])[:I], "setup_time": st[2::3][:I],
                "overtime_cost": _nums(f["ueber-ks.prn"])[0]}
        assert len(dem) == I * T and len(inst["capacity"]) == T, folder
        out[inst["ttm"]] = inst
    return out


def gcg_list(raw: Path) -> tuple[list, dict]:
    """(GCG instance names of ls-trigeiro.test, {GCG name: proved optimum of ls-trigeiro.solu})."""
    names = [Path(x).name.removesuffix(".gz").removesuffix(".lp")
             for x in (raw / "trigeiro_gcg" / "ls-trigeiro.test").read_text().split() if x.endswith((".lp", ".lp.gz"))]
    solu = {}
    for line in (raw / "trigeiro_gcg" / "ls-trigeiro.solu").read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[0] == "=opt=":
            solu[p[1]] = float(p[2])
    return names, solu


def ttm_name(gcg_name: str) -> str:
    """F10N -> F10 (the N of the F set), other names unchanged."""
    return gcg_name[:-1] if re.fullmatch(r"F\d+N", gcg_name) else gcg_name


def plain_model(inst: dict, overtime: bool):
    """The plain TTM model with pyscipopt: min setup + holding (+ overtime) cost; s_i,t-1 + x_it - s_it = d_it with
    s_i,0 = initial stock; x_it <= M_it y_it with M_it the remaining demand of i from t on; sum_i (p_i x_it + st_i y_it)
    <= C_t (+ o_t)."""
    import pyscipopt

    I, T = inst["items"], inst["periods"]
    m = pyscipopt.Model()
    m.hideOutput()
    x = {(i, t): m.addVar(f"x_{i}_{t}", lb=0) for i in range(I) for t in range(T)}
    s = {(i, t): m.addVar(f"s_{i}_{t}", lb=0) for i in range(I) for t in range(T)}
    y = {(i, t): m.addVar(f"y_{i}_{t}", vtype="B") for i in range(I) for t in range(T)}
    o = {t: m.addVar(f"o_{t}", lb=0) for t in range(T)} if overtime else {}
    d = inst["demand"]
    for i in range(I):
        for t in range(T):
            prev = s[i, t - 1] if t else inst["initial_stock"][i]
            m.addCons(prev + x[i, t] - s[i, t] == d[i][t])
            m.addCons(x[i, t] <= max(0.0, sum(d[i][t:]) - (inst["initial_stock"][i] if t == 0 else 0)) * y[i, t])
    for t in range(T):
        lhs = pyscipopt.quicksum(inst["prod_coef"][i] * x[i, t] + inst["setup_time"][i] * y[i, t] for i in range(I))
        m.addCons(lhs <= inst["capacity"][t] + (o[t] if overtime else 0))
    obj = pyscipopt.quicksum(inst["setup_cost"][i] * y[i, t] + inst["holding"][i] * s[i, t]
                             for i in range(I) for t in range(T))
    if overtime:
        obj += pyscipopt.quicksum(inst["overtime_cost"] * o[t] for t in range(T))
    m.setObjective(obj, "minimize")
    return m


def solve_plain(args) -> dict:
    inst, overtime, time_limit = args
    import time

    m = plain_model(inst, overtime)
    m.setParam("limits/time", time_limit)
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("lp/threads", 1)
    m.setParam("randomization/randomseedshift", 0)
    t0 = time.time()
    m.optimize()
    st = m.getStatus()
    return {"ttm": inst["ttm"], "overtime": overtime, "status": st,
            "primal": m.getPrimalbound() if m.getNSols() else None, "dual": m.getDualbound(),
            "gap": m.getGap() if m.getNSols() else None, "seconds": round(time.time() - t0, 2)}


def check_a(raw: Path, out: Path, parallel: int, time_limit: float) -> dict:
    data = read_suerie(raw)
    names, solu = gcg_list(raw)
    absent = [n for n in names if ttm_name(n) not in data]
    names = [n for n in names if ttm_name(n) in data]
    jobs = [(data[ttm_name(n)], ot, time_limit) for n in names for ot in (False, True)]
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        res = list(pool.map(solve_plain, jobs))
    rows = []
    for n in names:
        row = {"gcg_name": n, "ttm": ttm_name(n), "folder": data[ttm_name(n)]["folder"], "solu": solu.get(n)}
        for r in res:
            if r["ttm"] == ttm_name(n):
                tag = "with_overtime" if r["overtime"] else "without_overtime"
                row[f"{tag}_status"] = r["status"]
                row[f"{tag}_value"] = r["primal"]
                row[f"{tag}_seconds"] = r["seconds"]
                ok = (r["status"] == "optimal" and row["solu"] is not None and r["primal"] is not None
                      and abs(r["primal"] - row["solu"]) <= 1e-6 * max(1.0, abs(row["solu"])))
                row[f"{tag}_agrees"] = ok if r["status"] == "optimal" else None
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_a.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summary = {}
    for tag in ("without_overtime", "with_overtime"):
        fin = [r for r in rows if r[f"{tag}_status"] == "optimal"]
        summary[tag] = {"instances": len(rows), "finished": len(fin),
                        "agree": sum(1 for r in fin if r[f"{tag}_agrees"]),
                        "mismatch": [(r["gcg_name"], r["solu"], r[f"{tag}_value"]) for r in fin if not r[f"{tag}_agrees"]],
                        "not_finished": [(r["gcg_name"], r[f"{tag}_status"]) for r in rows
                                         if r[f"{tag}_status"] != "optimal"]}
    summary["not_in_suerie_data"] = absent
    (out / "check_a_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def read_ampl(path: Path) -> dict:
    """The parameters of one AMPL .dat copy: NProd, T, C (default or table), RHO, H, ST, F, DEM (periods x products)."""
    text = path.read_text(encoding="latin-1")
    I = int(re.search(r"param NProd\s*:=\s*(\d+)", text).group(1))
    T = int(re.search(r"param T\s*:=\s*(\d+)", text).group(1))
    out = {"items": I, "periods": T}
    m = re.search(r"param C default\s+([\d.]+)\s*;", text)
    out["capacity"] = [float(m.group(1))] * T if m else None

    def block(name):
        mm = re.search(rf"param {name}\b[^;]*?:=(.*?);", text, re.S)
        return mm.group(1) if mm else None

    def by_item(name):
        dflt = re.search(rf"param {name} default\s+([\d.]+)\s*;", text)
        if dflt:
            return [float(dflt.group(1))] * I
        b = block(name)
        vals = {}
        for line in b.strip().splitlines():
            p = _nums(line)
            if len(p) >= 2:
                vals[int(p[0])] = p[-1]
        return [vals[i + 1] for i in range(I)]

    out["prod_coef"] = by_item("RHO")
    out["holding"] = by_item("H")
    out["setup_time"] = by_item("ST")
    out["setup_cost"] = by_item("F")
    rows = [p for p in (_nums(l) for l in block("DEM").strip().splitlines()) if p]
    out["demand"] = [[r[1 + i] for r in rows] for i in range(I)]      # rows are periods, columns products
    assert len(rows) == T, path
    return out


def check_b(raw: Path, out: Path) -> list:
    data = read_suerie(raw)
    res = []
    for f in sorted((raw / "lotsizelib_ampl").glob("tr*.dat")):
        a = read_ampl(f)
        same_size = [d for d in data.values() if d["items"] == a["items"] and d["periods"] == a["periods"]]
        by_demand = [d["ttm"] for d in same_size if d["demand"] == a["demand"]]
        row = {"file": f.name, "items": a["items"], "periods": a["periods"], "candidates_same_size": len(same_size),
               "ttm_by_demand": by_demand}
        if len(by_demand) == 1:
            d = data[by_demand[0]]
            row["folder"] = d["folder"]
            diffs = []
            for key in ("capacity", "prod_coef", "holding", "setup_time", "setup_cost", "demand"):
                av, dv = a[key], d[key]
                flat = (lambda v: [x for r in v for x in r]) if key == "demand" else (lambda v: v)
                for k, (p, q) in enumerate(zip(flat(av), flat(dv))):
                    if abs(p - q) > 1e-9 * max(1.0, abs(q)):
                        diffs.append({"field": key, "index": k, "ampl": p, "suerie": q})
                if len(flat(av)) != len(flat(dv)):
                    diffs.append({"field": key, "length_ampl": len(flat(av)), "length_suerie": len(flat(dv))})
            row["values_compared"] = sum(len(a[k]) for k in ("capacity", "prod_coef", "holding", "setup_time",
                                                             "setup_cost")) + a["items"] * a["periods"]
            row["mismatches"] = diffs
        res.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_b.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check-a", "check-b"))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=20)
    ap.add_argument("--time", type=float, default=3600.0)
    args = ap.parse_args(argv)
    if args.cmd == "check-a":
        print(json.dumps(check_a(Path(args.raw), Path(args.out), args.parallel, args.time), indent=1))
    else:
        print(json.dumps(check_b(Path(args.raw), Path(args.out)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
