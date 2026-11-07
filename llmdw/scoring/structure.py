"""Structure census of family-level decompositions (zero-marginal groups) from oracle evaluations.

Paper: Section 6: the census of zero-marginal groups (Figure 3(a)).

For each instance and each fixed set L of linking variable families, the gains
g(T) = bound(T, L) - LP (minimization units; reported as closed-gap fractions when the reference
optimum is known) form a table over the sets T of convexified families. An evaluation that hit
the time limit gives only a lower bound on g(T) ("censored"). Optional lookahead values
(llmdw.lookahead.lookahead) are also lower bounds. From these:

- certain zero:   g(S) = 0 from a converged evaluation
- certain gain:   a converged or censored evaluation, or a lookahead value, above tol
- zero-marginal group W at base B (default B = empty set, i.e. the LP): g(B|W) > g(B) and
  g(B|S) = g(B) for every proper subset S of W. "certain" when every such zero is certain,
  "possible" when some of them are censored.
- greedy check (converged values only): does adding one family at a time reach the best value?

Failed evaluations (llmdw.gcg.oracle: every attempt exited abnormally, status "failed"; older rows without a result
count the same) give no value: they are not in the table, never count as a gain of 0, never enter a tie, and are listed
per table in ``failed``. A group decision that depends on a failed set is "incomplete": a proper subset whose gain is
unknown because its evaluation failed (and no lookahead value is positive), or a candidate group W whose own evaluation
failed while every proper subset is a zero. The best-set decision of a table is incomplete when any set failed.

Without linking variables, the set of all k families is not in the table: convexifying every family leaves
an empty master, which is the MILP itself and not a decomposition, and the evaluation is limited to s = k - 1 families.
The oracle already gives that set no bound (an empty master with a single block is degenerate); the lookahead has a
value for it, which is left out. With linking variables an empty master is a decomposition and stays in its table.

Usage
-----
python -m llmdw.scoring.structure --oracle-root work/oracle --lookahead-root work/lookahead --out work/structure
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

CONVERGED = {"optimal", "nodelimit", "gaplimit"}


def _minform(v, sense):
    return v if sense == "minimize" else -v


def instance_tables(oracle: dict, lookahead: dict | None = None, tol: float = 1e-6) -> dict:
    """The gain tables of one oracle record. Every recorded bound passes llmdw.scoring.bounds first: an invalid bound
    counts as no bound (its evaluation is listed in ``invalid``), as every table that reads evaluations does."""
    from llmdw.scoring import bounds

    invalid = bounds.annotate(oracle)
    sense = oracle["sense"]
    fams = list(oracle["families"])
    lp = oracle.get("lp_bound")
    ref = oracle.get("reference_optimal_value")
    scale = None
    if lp is not None and ref is not None:
        scale = abs(_minform(ref, sense) - _minform(lp, sense)) or None
    tables, failed = {}, {}
    for s in oracle.get("subsets") or []:
        T = frozenset(set(fams) - set(s.get("master_families") or []))
        L = tuple(sorted(s.get("linking_var_families") or []))
        if not L and len(T) == len(fams):
            continue
        if not s.get("ok"):
            failed.setdefault(L, set()).add(T)
            continue
        if s.get("dual_bound") is None or s.get("bound_invalid"):
            continue
        gain = _minform(s["dual_bound"], sense) - _minform(lp, sense)
        if scale:
            gain /= scale
        tab = tables.setdefault(L, {frozenset(): {"gain": 0.0, "censored": False, "source": "lp"}})
        tab[T] = {"gain": max(0.0, gain), "censored": s.get("status") not in CONVERGED, "source": "oracle"}
    if lookahead:
        tab = tables.setdefault((), {frozenset(): {"gain": 0.0, "censored": False, "source": "lp"}})
        for row in lookahead["table"].values():
            if row["lookahead"] is None:
                continue
            T = frozenset(row["families"])
            if len(T) == len(fams):
                continue
            v = row["lookahead"] / scale if scale else row["lookahead"]
            cur = tab.get(T)
            if cur is None:
                tab[T] = {"gain": v, "censored": True, "source": "lookahead"}   # lower bound only
            elif v > cur["gain"] + tol:
                cur.update({"gain": v, "source": cur["source"] + "+lookahead"})
    return {"families": fams, "scale": scale, "tables": tables, "failed": failed, "invalid": invalid}


def zero_marginal_groups(tab: dict, base=frozenset(), tol: float = 1e-6, failed=None) -> list:
    """Zero-marginal groups at base (module docstring). failed: the sets of the table whose evaluation failed; a
    decision that depends on one of them gets the status "incomplete" (and gain None when W itself failed)."""
    base = frozenset(base)
    failed = set(failed or ())
    if base not in tab:
        return []
    g0 = tab[base]["gain"]
    out = []
    unknown = lambda key: key in failed and (key not in tab or tab[key]["gain"] <= g0 + tol)  # noqa: E731
    for T in sorted(list(tab) + [x for x in failed if x not in tab], key=len):   # table order, then failed sets
        if not T > base:
            continue
        W = T - base
        if T in tab and tab[T]["gain"] > g0 + tol:
            status = "certain"
        elif unknown(T):
            status = "incomplete"          # the candidate's own evaluation failed
        else:
            continue
        ok = True
        for r in range(len(W)):
            for S in itertools.combinations(sorted(W), r):
                key = base | frozenset(S)
                if unknown(key):
                    status = "incomplete"
                    continue
                if key not in tab:
                    ok = False
                    break
                if tab[key]["gain"] > g0 + tol:
                    ok = False
                    break
                if tab[key]["censored"] and status == "certain":
                    status = "possible"
            if not ok:
                break
        if ok and not any(W > prev["group"] for prev in out):
            out.append({"group": W, "size": len(W), "status": status,
                        "gain": tab[T]["gain"] if T in tab else None})
    return out


def greedy_check(tab: dict, tol: float = 1e-6) -> dict:
    conv = {T: v["gain"] for T, v in tab.items() if not v["censored"]}
    if not conv:
        return {"reaches_best": None}
    best = max(conv.values())
    cur, evals, path = frozenset(), 0, []
    fams = sorted(set().union(*conv.keys()))
    while True:
        nbrs = [cur | {f} for f in fams if f not in cur and (cur | {f}) in conv]
        evals += len(nbrs)
        if not nbrs:
            break
        nxt = max(nbrs, key=lambda T: conv[T])
        if conv[nxt] <= conv[cur] + tol:
            break
        cur = nxt
        path.append(sorted(cur))
    return {"reaches_best": conv[cur] >= best - tol, "greedy_value": conv[cur], "best_converged": best,
            "evaluations": evals, "path": path}


def summarize(root: str, lookahead_root: str | None, out: str) -> dict:
    rows = []
    for p in sorted(Path(root).glob("*/oracle.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        la = None
        if lookahead_root:
            q = Path(lookahead_root) / f"{p.parent.name}.json"
            if q.exists():
                la = json.loads(q.read_text(encoding="utf-8"))
        info = instance_tables(o, la)
        # closed-gap units: a gain below 1e-4 of the LP-to-optimum gap counts as zero
        tol = 1e-4 if info["scale"] else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
        for L, tab in info["tables"].items():
            failed = info["failed"].get(L, set())
            groups = zero_marginal_groups(tab, tol=tol, failed=failed)
            rows.append({"instance": p.parent.name, "linking": list(L), "k": len(info["families"]),
                         "n_sets": len(tab), "closed_gap_units": info["scale"] is not None,
                         "groups": [{**g, "group": sorted(g["group"])} for g in groups],
                         "joint_certain": sum(1 for g in groups if g["size"] >= 2 and g["status"] == "certain"),
                         "joint_possible": sum(1 for g in groups if g["size"] >= 2 and g["status"] == "possible"),
                         "joint_incomplete": sum(1 for g in groups if g["size"] >= 2 and g["status"] == "incomplete"),
                         "failed_sets": sorted("|".join(sorted(T)) for T in failed),
                         "best_set_incomplete": bool(failed),
                         "greedy": greedy_check(tab, tol=tol)})
    Path(out).mkdir(parents=True, exist_ok=True)
    insts = sorted({r["instance"] for r in rows})
    summ = {"n_instances": len(insts),
            "with_certain_joint_group": sorted({r["instance"] for r in rows if r["joint_certain"]}),
            "with_possible_joint_group": sorted({r["instance"] for r in rows if r["joint_possible"]}),
            "with_incomplete_joint_group": sorted({r["instance"] for r in rows if r["joint_incomplete"]}),
            "with_failed_evaluations": sorted({r["instance"] for r in rows if r["failed_sets"]}),
            "greedy_misses_best": sorted({r["instance"] for r in rows if r["greedy"].get("reaches_best") is False})}
    (Path(out) / "structure_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (Path(out) / "structure_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-root", required=True)
    ap.add_argument("--lookahead-root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(summarize(args.oracle_root, args.lookahead_root, args.out), indent=1))


if __name__ == "__main__":
    main()
