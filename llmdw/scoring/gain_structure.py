"""Lattice structure of gain tables: monotonicity, minimal positive sets, the gain beyond them, and whether the best set
within the size budget is a union of the minimal positive sets it contains.

A gain table of one instance holds g(T), the gain of convexifying the family set T as a share of the LP gap
(IP optimum - LP bound), for every set T of at most s families, with g(empty) = 0. With tol = TOL (shares, that is
TOL x LP gap in objective units):

- monotonicity violation: a pair S < T (S a proper subset of T, both in the table, S possibly empty) with
  g(S) > g(T) + tol;
- minimal positive set: T with g(T) > tol and g(S) <= tol for every proper subset S;
- for every T with g(T) > tol: d1(T) = g(T) - max g(M) over the minimal positive sets M within T, and
  d2(T) = g(T) - the largest sum of g(M) over collections of pairwise disjoint minimal positive sets within T;
- best set: the sets of at most s families with g(T) >= max g - tol. The best set is a union when some best set B
  equals the union of the minimal positive sets within B; when max g <= tol the instance has no positive set.

Tables (read only):

- MIPLIB-NL census (llmdw.gcg.oracle): the table without linking variables (llmdw.scoring.structure.instance_tables
  without the lookahead), after the census corrections of llmdw.scoring.results.census_figure (reference optima and
  dropped bounds) and the bound validity rule of llmdw.scoring.bounds. Its lattice is every nonempty set of at most
  k - 1 families (the set of all k families is no decomposition), so s = k - 1. An instance enters only when the
  LP gap exists (a reference optimum different from the LP bound) and the table is complete: every set of the lattice
  has a valid dual bound (converged, or stopped at the limit with a bound). A bound stopped at the limit gives a lower
  bound on the gain; the number of such sets is reported per instance.
- rich models (llmdw.instances.rich_models_build): the accepted candidates; exact gains of every set of at most 3
  families, normalized by ip - lp with ip = Fraction(ip_scip).limit_denominator(10**6) as in the build.
- larger rich models (llmdw.instances.rich_models_large): the first PER_CLASS accepted candidates of each class and
  k in draw order; the table is used up to table_complete_to families (3 when absent).

Outputs in --out: per_instance.csv, per_set.csv (one row per positive set), summary.csv (by collection) and
excluded.csv (instances left out, with the reason).

Usage
-----
python -m llmdw.scoring.gain_structure --work WORK --out OUT [--oracle-root DIR] [--census-corrections FILE]
    [--rich-root DIR] [--large-root DIR] [--tol 1e-6]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path

TOL = 1e-6
PER_CLASS = 10
CLASS_LABEL = {"lot_sizing": "lot sizing", "network_design": "network design", "facility_location": "facility location"}
CENSUS = "MIPLIB-NL"


# ---------------------------------------------------------------- one table

def key(T) -> str:
    return "|".join(sorted(T))


def _proper_subsets(T: frozenset):
    items = sorted(T)
    for r in range(len(items)):
        for S in itertools.combinations(items, r):
            yield frozenset(S)


def best_packing(T: frozenset, minimal: dict) -> float:
    """Largest sum of g(M) over collections of pairwise disjoint minimal positive sets M within T (0 when none)."""
    inside = sorted((M for M in minimal if M <= T), key=key)

    def rec(i: int, used: frozenset) -> float:
        if i == len(inside):
            return 0.0
        best = rec(i + 1, used)
        M = inside[i]
        if not (M & used):
            best = max(best, minimal[M] + rec(i + 1, used | M))
        return best

    return rec(0, frozenset())


def analyze_table(g: dict, s: int, tol: float = TOL) -> tuple[dict, list]:
    """g: {frozenset: gain share} for every set of at most s families (the empty set may be absent: g = 0). Returns the
    per-instance measures and one row per positive set (set, size, g, d1, d2)."""
    g = {T: float(v) for T, v in g.items() if len(T) <= s}
    g.setdefault(frozenset(), 0.0)
    viol, max_viol = 0, 0.0
    for T, v in g.items():
        for S in _proper_subsets(T):
            if S in g and g[S] > v + tol:
                viol += 1
                max_viol = max(max_viol, g[S] - v)
    positive = {T: v for T, v in g.items() if T and v > tol}
    minimal = {T: v for T, v in positive.items()
               if all(g.get(S, 0.0) <= tol for S in _proper_subsets(T))}
    sets = []
    for T in sorted(positive, key=lambda x: (len(x), key(x))):
        v = positive[T]
        d1 = v - max(minimal[M] for M in minimal if M <= T)   # T contains a minimal positive set (table complete)
        d2 = v - best_packing(T, minimal)
        sets.append({"set": key(T), "size": len(T), "g": v, "d1": d1, "d2": d2})
    nonempty = {T: v for T, v in g.items() if T}
    best = max(nonempty.values()) if nonempty else 0.0
    best_sets = sorted((T for T, v in nonempty.items() if v >= best - tol), key=lambda x: (len(x), key(x)))
    if best <= tol:
        union = "no positive set"
    else:
        union = any(frozenset().union(*[M for M in minimal if M <= B]) == B for B in best_sets)
    sizes = Counter(len(M) for M in minimal)
    mins = sorted(minimal, key=lambda x: (len(x), key(x)))
    overlap = sum(1 for i, A in enumerate(mins) for B in mins[i + 1:] if A & B)   # pairs sharing a family
    d1s, d2s = [r["d1"] for r in sets], [r["d2"] for r in sets]
    out = {"s": s, "best_gain": best, "n_sets_in_table": len(nonempty), "n_violations": viol, "max_violation": max_viol,
           "n_minimal_positive": len(minimal),
           "minimal_pairwise_disjoint": (overlap == 0) if minimal else "no positive set",
           "n_minimal_pairs_sharing_a_family": overlap,
           "minimal_sizes": ";".join(f"{r}:{sizes.get(r, 0)}" for r in range(1, s + 1)),
           "minimal_sets": "; ".join(key(M) for M in sorted(minimal, key=lambda x: (len(x), key(x)))),
           "n_positive_sets": len(positive),
           "d1_min": min(d1s) if d1s else None, "d1_median": statistics.median(d1s) if d1s else None,
           "d1_max": max(d1s) if d1s else None,
           "d2_min": min(d2s) if d2s else None, "d2_median": statistics.median(d2s) if d2s else None,
           "d2_max": max(d2s) if d2s else None,
           "best_sets": "; ".join(key(B) for B in best_sets) if best > tol else "",
           "best_is_union": union, "_minimal_size_counts": sizes}
    return out, sets


# ---------------------------------------------------------------- census tables

def apply_corrections(name: str, o: dict, corrections: dict) -> list:
    """The corrections of llmdw.scoring.results.census_figure, in memory: replaced reference optima and dropped
    bounds. Returns the descriptions of the changes."""
    opt_fix, drop = corrections.get("optimum", {}), corrections.get("drop_bounds", {})
    changes = []
    if name in opt_fix:
        changes.append(f"optimum {o.get('reference_optimal_value')} -> {opt_fix[name]}")
        o["reference_optimal_value"] = opt_fix[name]
    if name in drop:
        before = len(o.get("subsets") or [])
        o["subsets"] = [x for x in o.get("subsets") or [] if x.get("key") not in set(drop[name])]
        changes.append(f"{before - len(o['subsets'])} bound(s) dropped")
    return changes


def census_table(name: str, o: dict, corrections: dict) -> tuple[dict | None, str | None]:
    """(table record, None) for an instance with an LP gap and a complete table without linking variables, else
    (None, reason). The record holds the families, s, the LP gap, the gains and the number of sets stopped at the
    limit."""
    from llmdw.scoring import structure

    notes = corrections.get("notes", {})
    opt_fix = corrections.get("optimum", {})
    changes = apply_corrections(name, o, corrections)
    info = structure.instance_tables(o, None)           # also marks the invalid bounds (llmdw.scoring.bounds)
    fams = info["families"]
    k = len(fams)
    if info["scale"] is None:
        if name in opt_fix and opt_fix[name] is None and name in notes:
            return None, "reference optimum removed: " + notes[name]
        if o.get("reference_optimal_value") is None:
            return None, "no reference optimum"
        return None, "LP bound equals the optimum"
    if k < 2:
        return None, f"k = {k}: no set of at most k - 1 families"
    rows = {}
    for r in o.get("subsets") or []:
        if r.get("linking_var_families"):
            continue
        T = frozenset(set(fams) - set(r.get("master_families") or []))
        if T and len(T) < k:
            rows[T] = r
    lattice = [frozenset(c) for r in range(1, k) for c in itertools.combinations(fams, r)]
    tab = info["tables"].get((), {})
    dropped = set(corrections.get("drop_bounds", {}).get(name, []))
    why = Counter()
    for T in lattice:
        r = rows.get(T)
        if r is None and key(set(fams) - T) in dropped:      # the key of a set without linking variables (master)
            why["bound dropped by the census corrections"] += 1
        elif r is None:
            why["not evaluated"] += 1
        elif not r.get("ok"):
            why["failed"] += 1
        elif r.get("bound_invalid"):
            why["invalid bound"] += 1
        elif r.get("dual_bound") is None:
            why["no bound"] += 1
        elif T not in tab:
            why["not in the table"] += 1
    if why:
        return None, (f"incomplete table: {sum(why.values())} of {len(lattice)} sets without a valid bound ("
                      + ", ".join(f"{n} {what}" for what, n in sorted(why.items())) + ")")
    gains = {T: tab[T]["gain"] for T in lattice}
    stopped = sum(1 for T in lattice if tab[T]["censored"])
    return {"families": fams, "s": k - 1, "lp_gap": info["scale"], "gains": gains, "n_stopped_at_limit": stopped,
            "corrections": "; ".join(changes)}, None


def census_instances(oracle_root: Path, corrections: dict):
    """Yields (name, record or None, reason or None) for every census instance."""
    for p in sorted(oracle_root.glob("*/oracle.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        rec, why = census_table(p.parent.name, o, corrections)
        yield p.parent.name, rec, why


# ---------------------------------------------------------------- exact tables

def exact_table(res: dict, s: int) -> tuple[dict | None, str | None]:
    """The gain shares of an accepted rich-model candidate up to s families: (record, None) or (None, reason)."""
    ip = Fraction(res["ip_scip"]).limit_denominator(10 ** 6)
    lp = Fraction(res["lp_bound"])
    width = ip - lp
    if width <= 0:
        return None, "no positive LP gap"
    tab = {frozenset(k.split("|")) if k else frozenset(): Fraction(v) for k, v in res["table"].items()}
    fams = sorted(set().union(*tab.keys()))
    missing = [c for r in range(1, s + 1) for c in itertools.combinations(fams, r) if frozenset(c) not in tab]
    if missing:
        return None, f"incomplete table: {len(missing)} sets of at most {s} families missing"
    gains = {T: float(v / width) for T, v in tab.items() if len(T) <= s}
    return {"families": fams, "s": s, "lp_gap": float(width), "gains": gains, "n_stopped_at_limit": None,
            "corrections": ""}, None


def rich_instances(root: Path):
    """Yields (collection, name, record or None, reason or None) for the accepted rich-model candidates; the instance
    list (instances.txt) is checked against them when present."""
    acc = []
    for p in sorted(root.glob("candidates/*.json")):
        res = json.loads(p.read_text(encoding="utf-8"))
        if res.get("accepted"):
            acc.append(res)
    listed = root / "instances.txt"
    if listed.exists():
        names = set(listed.read_text(encoding="utf-8").split())
        got = {r["name"] for r in acc}
        if names != got:
            print(f"warning: accepted candidates and instances.txt differ: only accepted {sorted(got - names)}, "
                  f"only listed {sorted(names - got)}")
    for res in sorted(acc, key=lambda r: (r["class"], r["candidate"])):
        coll = f"rich models: {CLASS_LABEL.get(res['class'], res['class'])} (k = {res['k']})"
        rec, why = exact_table(res, 3)
        yield coll, res["name"], rec, why


def large_rich_instances(root: Path, per_class: int = PER_CLASS):
    """Yields (collection, name, record or None, reason or None) for the first per_class accepted candidates of each
    class and k in draw order (file name <class>_k<K>_<index>.json)."""
    by = {}
    for p in sorted(root.glob("candidates/*.json")):
        m = re.fullmatch(r"(.+)_k(\d+)_(\d+)\.json", p.name)
        if not m:
            continue
        by.setdefault((m.group(1), int(m.group(2))), {})[int(m.group(3))] = p
    for (cls, k), files in sorted(by.items()):
        acc = []
        for i in sorted(files):
            res = json.loads(files[i].read_text(encoding="utf-8"))
            if res.get("accepted"):
                acc.append((i, res))
            if len(acc) == per_class:
                break
        if acc and len(acc) == per_class:
            gaps = [i for i in range(acc[-1][0]) if i not in files]
            if gaps:
                print(f"note: {cls} k = {k}: draws {gaps} undecided before the last selected one")
        coll = f"{CLASS_LABEL.get(cls, cls)} k = {k}"
        for _, res in acc:
            rec, why = exact_table(res, int(res.get("table_complete_to") or 3))
            yield coll, res["name"], rec, why


# ---------------------------------------------------------------- summary

def quantile(xs: list, q: float):
    """Linear interpolation between order statistics (numpy's default)."""
    if not xs:
        return None
    ys = sorted(xs)
    h = (len(ys) - 1) * q
    lo = int(h)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (h - lo) * (ys[hi] - ys[lo])


QUANTILES = (("min", 0.0), ("q10", 0.1), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("q90", 0.9), ("max", 1.0))


def summary_row(name: str, rows: list, sets: list) -> dict:
    n = len(rows)
    out = {"collection": name, "n_instances": n,
           "share_no_violation": sum(r["n_violations"] == 0 for r in rows) / n if n else None}
    nm = Counter(min(r["n_minimal_positive"], 5) for r in rows)
    for j in range(5):
        out[f"n_minimal_{j}"] = nm.get(j, 0)
    out["n_minimal_5plus"] = nm.get(5, 0)
    sizes = Counter()
    for r in rows:
        sizes.update(r["_minimal_size_counts"])
    for j in (1, 2, 3):
        out[f"minimal_size_{j}"] = sizes.get(j, 0)
    out["minimal_size_4plus"] = sum(v for j, v in sizes.items() if j >= 4)
    for d in ("d1", "d2"):
        vals = [x[d] for x in sets]
        out[f"{d}_n"] = len(vals)
        for lab, q in QUANTILES:
            out[f"{d}_{lab}"] = quantile(vals, q)
    withm = [r for r in rows if r["minimal_pairwise_disjoint"] != "no positive set"]
    out["share_minimal_pairwise_disjoint"] = (sum(r["minimal_pairwise_disjoint"] is True for r in withm) / len(withm)
                                              if withm else None)
    out["n_minimal_pairs_sharing_a_family"] = sum(r["n_minimal_pairs_sharing_a_family"] for r in rows)
    pos = [r for r in rows if r["best_is_union"] != "no positive set"]
    out["n_with_positive_best"] = len(pos)
    out["share_best_is_union"] = sum(r["best_is_union"] is True for r in pos) / len(pos) if pos else None
    out["n_no_positive_set"] = n - len(pos)
    return out


def groupings(collections: list) -> list:
    """(name, member collections) of the summary: every collection, then the groups of several."""
    out = [(c, [c]) for c in collections]
    rich = [c for c in collections if c.startswith("rich models: ")]
    large = [c for c in collections if c != CENSUS and c not in rich]
    if len(rich) > 1:
        out.append(("rich models (k = 7/9)", rich))
    if len(large) > 1:
        out.append(("larger rich models (k = 12 to 16)", large))
    if rich and large:
        out.append(("all rich models", rich + large))
    return out


# ---------------------------------------------------------------- output

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.10g}"
    return "" if v is None else v


def write_csv(path: Path, rows: list, header_comment: str | None = None, fields: list | None = None):
    keys = list(fields or [])
    for r in rows:
        keys += [k for k in r if k not in keys and not k.startswith("_")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if header_comment:
            for line in header_comment.splitlines():
                fh.write(f"# {line}\n")
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})


SUMMARY_NOTE = """gains are shares of the LP gap (IP optimum - LP bound); tol = {tol} (shares)
MIPLIB-NL: the census table without linking variables (oracle evaluations only, census corrections and bound validity
rule applied); lattice = every nonempty set of at most k - 1 families, budget s = k - 1; complete table = every set of
the lattice has a valid dual bound (converged, or stopped at the limit with a bound: a lower bound on its gain); an
instance without an LP gap or with an incomplete table is in excluded.csv
rich models: exact gains of every set of at most s families (s = 3; s = table_complete_to when recorded)
d1, d2 quantiles pooled over all (instance, positive set) pairs; share_best_is_union over the instances with a
positive best set"""


def run(args) -> dict:
    work = Path(args.work)
    out = Path(args.out)
    tol = args.tol
    oracle_root = Path(args.oracle_root) if args.oracle_root else work / "work" / "oracle"
    rich_root = Path(args.rich_root) if args.rich_root else work / "work" / "rich_models"
    large_root = Path(args.large_root) if args.large_root else work / "work" / "rich_models_large"
    corr_path = Path(args.census_corrections) if args.census_corrections else work / "census_corrections.json"
    corrections = json.loads(corr_path.read_text(encoding="utf-8")) if corr_path.exists() else {}
    if not corr_path.exists():
        print(f"note: no census corrections at {corr_path}")

    per_inst, per_set, excluded, order = [], [], [], []

    def add(coll, name, rec, why):
        if coll not in order:
            order.append(coll)
        if rec is None:
            excluded.append({"collection": coll, "instance": name, "reason": why})
            return
        m, sets = analyze_table(rec["gains"], rec["s"], tol)
        row = {"collection": coll, "instance": name, "k": len(rec["families"]), "s": rec["s"], "lp_gap": rec["lp_gap"],
               "best_gain": m["best_gain"], "n_sets_in_table": m["n_sets_in_table"],
               "n_stopped_at_limit": rec["n_stopped_at_limit"]}
        row.update({k: v for k, v in m.items() if k not in row})
        row["census_corrections"] = rec["corrections"]
        per_inst.append(row)
        per_set.extend({"instance": name, "collection": coll, **x} for x in sets)

    if oracle_root.exists():
        for name, rec, why in census_instances(oracle_root, corrections):
            add(CENSUS, name, rec, why)
    if rich_root.exists():
        for coll, name, rec, why in rich_instances(rich_root):
            add(coll, name, rec, why)
    if large_root.exists():
        for coll, name, rec, why in large_rich_instances(large_root):
            add(coll, name, rec, why)

    summary = []
    for gname, members in groupings(order):
        rows = [r for r in per_inst if r["collection"] in members]
        sets = [x for x in per_set if x["collection"] in members]
        summary.append(summary_row(gname, rows, sets))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_instance.csv", per_inst)
    write_csv(out / "per_set.csv", per_set)
    write_csv(out / "summary.csv", summary, SUMMARY_NOTE.format(tol=tol))
    write_csv(out / "excluded.csv", excluded, fields=["collection", "instance", "reason"])
    return {"instances": Counter(r["collection"] for r in per_inst),
            "excluded": Counter((e["collection"], e["reason"].split(":")[0]) for e in excluded)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="root holding work/oracle, work/rich_models and the larger rich models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-root", help="census oracle directory (default WORK/work/oracle)")
    ap.add_argument("--census-corrections", help="census corrections JSON (default WORK/census_corrections.json)")
    ap.add_argument("--rich-root", help="rich models directory (default WORK/work/rich_models)")
    ap.add_argument("--large-root", help="larger rich models directory (default WORK/work/rich_models_large)")
    ap.add_argument("--tol", type=float, default=TOL, help="tolerance as a share of the LP gap")
    args = ap.parse_args(argv)
    res = run(args)
    for c, n in res["instances"].items():
        print(f"{c}: {n} instances")
    for (c, why), n in sorted(res["excluded"].items()):
        print(f"excluded {c}: {why}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
