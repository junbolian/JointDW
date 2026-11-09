"""Two further benchmark classes with standard formulations on their own published data (no added families, no
generated parameters): the classes without a joint group.

generalized assignment  OR-Library gap1..gap12 (Osman 1995; Cattrysse, Salomon and Van Wassenhove 1994): Assign (each
                        job to exactly one agent), Capacity (resource used at an agent at most its capacity); the files
                        that gapinfo.html marks as maximization (gap1..gap12) have their objective coefficients negated
                        and minimized, so the optimal value is the negative of the published one; files marked as
                        minimization would be kept as they are
bin packing             OR-Library binpack1 (Falkenauer 1994), Kantorovich formulation with as many bins as the
                        first-fit-decreasing packing uses (recorded per instance): Assign (each item in exactly one bin),
                        Capacity (sizes in a bin at most the bin capacity if the bin is used), SymmetryBreaking
                        (y_j >= y_j+1)

The raw files are the OR-Library files that scripts/fetch_benchmarks.sh downloads into data/benchmark_raw/orlib_gap/ and
data/benchmark_raw/orlib_binpack/ (listed with their SHA-256 in data/benchmark_raw/sources.json); the SHA-256 of each
source file and of each model file built from it is recorded in meta.json.

Selection (before any evaluation): every instance of gap1..gap12 and binpack1 whose LP relaxation is feasible and whose
LP gap against the SCIP reference (one thread, seed 0, one hour) is positive (the selection rule of the standard
formulations on the benchmark instances).

Usage
-----
python -m llmdw.benchmark.extra_classes build --raw data/benchmark_raw --out work/no_joint_group
python -m llmdw.benchmark.extra_classes select --out work/no_joint_group --parallel 4
python -m llmdw.benchmark.extra_classes names --out work/no_joint_group
python -m llmdw.benchmark.extra_classes verify --raw data/benchmark_raw --out work/no_joint_group
python -m llmdw.benchmark.extra_classes check --out work/no_joint_group --recorded data/work/no_joint_group

The released data hold no file built from the OR-Library files (model.mps, model.md, instance.json); build recreates them
and check compares the SHA-256 of every rebuilt model file with the one recorded in the released meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from llmdw.benchmark.build import REFERENCE_TIME, _reference
from llmdw.benchmark.standard import _screen_one, letters
from llmdw.instances.milp import MILP

CLASSES = ("generalized_assignment", "bin_packing")
USED_CLASSES = CLASSES


# ---------------------------------------------------------------- readers

def read_gap(text: str) -> list:
    t = [int(v) for v in text.split()]
    P, k, out = t[0], 1, []
    for _ in range(P):
        m, n = t[k], t[k + 1]
        k += 2
        cost = [t[k + i * n:k + (i + 1) * n] for i in range(m)]
        k += m * n
        use = [t[k + i * n:k + (i + 1) * n] for i in range(m)]
        k += m * n
        cap = t[k:k + m]
        k += m
        out.append({"agents": m, "jobs": n, "cost": cost, "use": use, "capacity": cap})
    if k != len(t):
        raise ValueError(f"gap file: {len(t) - k} numbers left over")
    return out


def read_binpack(text: str) -> list:
    t = text.split()
    P, k, out = int(t[0]), 1, []
    for _ in range(P):
        ident = t[k]
        C, n, best = int(t[k + 1]), int(t[k + 2]), int(t[k + 3])
        sizes = [int(v) for v in t[k + 4:k + 4 + n]]
        k += 4 + n
        out.append({"id": ident, "capacity": C, "items": n, "best_known_bins": best, "sizes": sizes,
                    "ffd_bins": first_fit_decreasing(sizes, C)})
    if k != len(t):
        raise ValueError(f"binpack file: {len(t) - k} tokens left over")
    return out


def first_fit_decreasing(sizes: list, capacity: int) -> int:
    """Number of bins of the first-fit-decreasing packing (sizes sorted by decreasing size, ties in file order; each
    item into the first open bin with room, else a new bin)."""
    loads = []
    for s in sorted(sizes, key=lambda v: -v):
        for b, load in enumerate(loads):
            if load + s <= capacity:
                loads[b] += s
                break
        else:
            loads.append(s)
    return len(loads)


# OR-Library gapinfo.html: the optimal values of gap1..gap12 are given "when solved as maximisation problems"
GAP_MAXIMIZATION = {f"gap{g}.txt" for g in range(1, 13)}


def published_gap_optima(html: str) -> dict:
    """{problem id (e.g. c515-1): published optimal value} from the table in gapinfo.html."""
    import re

    return {a: int(b) for a, b in re.findall(r"\b(c\d{3,4}-\d)\s+(\d+)", html)}


# ---------------------------------------------------------------- formulations

def generalized_assignment(d: dict, name: str) -> MILP:
    m_, n = d["agents"], d["jobs"]
    m = MILP(name)
    x = {(i, j): m.add_var(f"Assign_{letters(i)}_{j + 1}", 0, 1, integer=True) for i in range(m_) for j in range(n)}
    for j in range(n):
        m.add_row(f"Assign_{j + 1}", "Assign_#", {x[i, j]: 1 for i in range(m_)}, lo=1, up=1)
    for i in range(m_):
        m.add_row(f"Capacity_{i + 1}", "Capacity_#", {x[i, j]: d["use"][i][j] for j in range(n)}, up=d["capacity"][i])
    sign = -1 if d["maximization"] else 1
    m.set_objective({x[i, j]: sign * d["cost"][i][j] for i in range(m_) for j in range(n)})
    if d["maximization"]:
        m.description = (f"{n} jobs are assigned to {m_} agents. Each job is assigned to exactly one agent; assigning a "
                         "job to an agent has a value and uses some of the agent's resource, both depending on the agent "
                         "and the job, and the resource used at an agent is at most its capacity. The total value is "
                         "maximized; the objective is its negative, which is minimized.")
    else:
        m.description = (f"{n} jobs are assigned to {m_} agents. Each job is assigned to exactly one agent; assigning a "
                         "job to an agent has a cost and uses some of the agent's resource, both depending on the agent "
                         "and the job, and the resource used at an agent is at most its capacity. Minimize the total "
                         "assignment cost.")
    m.family_text = {"Assign_#": "each job is assigned to exactly one agent",
                     "Capacity_#": "the resource used by the jobs assigned to an agent is at most its capacity"}
    return m


def bin_packing(d: dict, name: str) -> MILP:
    n, U, C = d["items"], d["ffd_bins"], d["capacity"]
    m = MILP(name)
    x = {(i, j): m.add_var(f"Pack_{i + 1}_{j + 1}", 0, 1, integer=True) for i in range(n) for j in range(U)}
    y = [m.add_var(f"Use_{j + 1}", 0, 1, integer=True) for j in range(U)]
    for i in range(n):
        m.add_row(f"Assign_{i + 1}", "Assign_#", {x[i, j]: 1 for j in range(U)}, lo=1, up=1)
    for j in range(U):
        m.add_row(f"Capacity_{j + 1}", "Capacity_#", {**{x[i, j]: d["sizes"][i] for i in range(n)}, y[j]: -C}, up=0)
    for j in range(U - 1):
        m.add_row(f"SymmetryBreaking_{j + 1}", "SymmetryBreaking_#", {y[j]: 1, y[j + 1]: -1}, lo=0)
    m.set_objective({v: 1 for v in y})
    m.description = (f"{n} items with known sizes are packed into at most {U} identical bins of capacity {C}. Each item "
                     "is packed into exactly one bin, and the sizes of the items in a bin add up to at most the capacity "
                     "if the bin is used. Minimize the number of bins used.")
    m.family_text = {"Assign_#": "each item is packed into exactly one bin",
                     "Capacity_#": "the sizes of the items in a bin are at most its capacity if the bin is used",
                     "SymmetryBreaking_#": "a bin is used only if the bin before it is used"}
    return m


BUILDERS = {"generalized_assignment": generalized_assignment, "bin_packing": bin_packing}


# ---------------------------------------------------------------- pools, build, selection

def pools(raw: Path) -> dict:
    """{class: [(sort key, name, data, source)]} in selection order."""
    manifest = {f["file"]: f for f in json.loads((raw / "sources.json").read_text())["files"]
                if f["source"] in ("orlib_gap", "orlib_binpack")}
    out = {"generalized_assignment": [], "bin_packing": []}
    for g in range(1, 13):
        f = f"gap{g}.txt"
        for p, d in enumerate(read_gap((raw / "orlib_gap" / f).read_text(encoding="latin-1"))):
            ident = f"c{d['agents']}{d['jobs']}-{p + 1}"
            d["maximization"] = f in GAP_MAXIMIZATION
            out["generalized_assignment"].append(((g, p), f"gap_c{d['agents']}{d['jobs']}_{p + 1}", d,
                                                  {"source": "orlib_gap", "file": f, "problem": ident,
                                                   "sha256": manifest[f]["sha256"]}))
    for p, d in enumerate(read_binpack((raw / "orlib_binpack" / "binpack1.txt").read_text(encoding="latin-1"))):
        out["bin_packing"].append(((p,), f"bp_{d['id']}", d, {"source": "orlib_binpack", "file": "binpack1.txt",
                                                              "problem": d["id"],
                                                              "sha256": manifest["binpack1.txt"]["sha256"]}))
    return out


def write(cls: str, name: str, d: dict, src: dict, root: Path, order: int) -> dict:
    m = BUILDERS[cls](d, name)
    dd = root / "instances" / name
    dd.mkdir(parents=True, exist_ok=True)
    m.write_mps(str(dd / "model.mps"))
    (dd / "instance.json").write_text(json.dumps({"abstract_problem": m.description}, indent=1), encoding="utf-8")
    lines = ["# Formulation", "", "Constraint families:"] + [f"- {f}: {m.family_text.get(f, '')}" for f in m.families]
    (dd / "model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    size = {"generalized_assignment": lambda: {"agents": d["agents"], "jobs": d["jobs"]},
            "bin_packing": lambda: {"items": d["items"], "bins": d["ffd_bins"], "capacity": d["capacity"],
                                    "best_known_bins_in_file": d["best_known_bins"]},
            }[cls]()
    meta = {"name": name, "class": cls, "size_order": order, "source": src, "size": size, "families": m.families,
            "k": len(m.families), "rows": len(m.rows), "columns": len(m.vars),
            "binaries": sum(v.integer for v in m.vars),
            "mps_sha256": hashlib.sha256((dd / "model.mps").read_bytes()).hexdigest()}
    if cls == "generalized_assignment":
        meta["objective_negated"] = d["maximization"]
    (dd / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def cmd_build(raw: Path, root: Path) -> dict:
    P = pools(raw)
    plan = {"excluded_before_selection": {}}
    for cls in CLASSES:
        plan[cls] = []
        for order, (_, name, d, src) in enumerate(P[cls]):
            meta = write(cls, name, d, src, root, order)
            plan[cls].append({"name": name, "rows": meta["rows"], "columns": meta["columns"]})
    (root / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return {c: len(v) for c, v in plan.items()}


def cmd_select(root: Path, parallel: int) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    result = {}
    for cls in CLASSES:
        built = [p["name"] for p in plan[cls]]
        limit = len(built)
        rows, passed, i = [], [], 0
        while i < len(built) and len(passed) < limit:
            chunk = built[i:i + parallel]
            lp_rows = {n: _screen_one(root, n) for n in chunk}
            need = [n for n in chunk if lp_rows[n].get("pending")]
            with ProcessPoolExecutor(max_workers=parallel) as pool:
                list(pool.map(_reference, [(str(root / "instances" / n), REFERENCE_TIME) for n in need]))
            for n in chunk:
                r = _screen_one(root, n) if n in need else lp_rows[n]
                rows.append(r)
                if not r.get("excluded"):
                    passed.append(n)
            i += parallel
        result[cls] = {"screened": rows, "selected": passed[:limit], "not_screened": built[i:]}
        print(cls, "screened", len(rows), "selected", len(passed[:limit]), flush=True)
    (root / "selection.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    sel = [n for c in CLASSES for n in result[c]["selected"]]
    (root / "all_instances.txt").write_text("\n".join(sel) + "\n", encoding="utf-8")
    return result


def cmd_verify(raw: Path, root: Path) -> dict:
    """GAP: the SCIP reference of every built instance against the optimal value published in gapinfo.html (negated
    for the files marked as maximization). Written as gap_published_check.json."""
    pub = published_gap_optima((raw / "orlib_gap" / "gapinfo.html").read_text(encoding="latin-1"))
    rows = []
    for meta_p in sorted((root / "instances").glob("gap_*/meta.json")):
        meta = json.loads(meta_p.read_text())
        ref_p = meta_p.parent / "reference.json"
        ref = json.loads(ref_p.read_text()) if ref_p.exists() else None
        ident = meta["source"]["problem"]
        want = pub.get(ident)
        if want is not None and meta["objective_negated"]:
            want = -want
        got = ref and ref["best_value"]
        rows.append({"name": meta["name"], "problem": ident, "published_in_model_sense": want,
                     "scip_best_value": got, "scip_proved_optimal": ref and ref["proved_optimal"],
                     "match": None if want is None or got is None else abs(got - want) < 1e-6})
    res = {"instances": len(rows), "with_published_value": sum(r["published_in_model_sense"] is not None for r in rows),
           "matches": sum(r["match"] is True for r in rows), "mismatches": [r for r in rows if r["match"] is False],
           "rows": rows}
    (root / "gap_published_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def cmd_names(root: Path) -> dict:
    """The instance lists of the experiment, the classes of USED_CLASSES only, in the order of plan.json:
    instances_used.txt, every built instance of these classes (gap1..gap12 and binpack1), and instances_used_selected.txt,
    those that pass the screening of selection.json (the selection rule of the standard formulations)."""
    plan = json.loads((root / "plan.json").read_text())
    sel = json.loads((root / "selection.json").read_text())
    built = [p["name"] for c in USED_CLASSES for p in plan[c]]
    passed = [n for c in USED_CLASSES for n in sel[c]["selected"]]
    (root / "instances_used.txt").write_text("\n".join(built) + "\n", encoding="utf-8")
    (root / "instances_used_selected.txt").write_text("\n".join(passed) + "\n", encoding="utf-8")
    return {"built": len(built), "selected": len(passed),
            "not_selected": {r["name"]: r["excluded"] for c in USED_CLASSES for r in sel[c]["screened"]
                             if r.get("excluded")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "select", "verify", "names", "check"))
    ap.add_argument("--raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--recorded", help="check: the released instances (data/work/no_joint_group)")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        from llmdw.benchmark.standard import cmd_check

        res = cmd_check(Path(args.out), Path(args.recorded))
        print(json.dumps(res))
        return 0 if res["identical"] == res["instances"] else 1
    if args.cmd == "build":
        print(json.dumps(cmd_build(Path(args.raw), Path(args.out))))
    elif args.cmd == "names":
        print(json.dumps(cmd_names(Path(args.out))))
    elif args.cmd == "verify":
        res = cmd_verify(Path(args.raw), Path(args.out))
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    else:
        cmd_select(Path(args.out), args.parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
