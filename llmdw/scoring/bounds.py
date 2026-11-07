"""Validity of a recorded GCG bound and of a reference optimum (one rule for every table that reads recorded
evaluations).

A recorded dual bound is invalid, and its evaluation counts as giving no bound, when

1. it is not finite or |bound| >= INFINITE (SCIP's infinity; GCG reports it, among others, when the time limit is
   reached while the master problem is still infeasible);
2. a reference optimum exists and the bound is beyond it by more than the tolerance: bound > optimum + tol for a
   minimization, bound < optimum - tol for a maximization;
3. the bound is beyond a primal solution value recorded by SCIP for the model (the best over every evaluation of the
   same model): above it for a minimization, below it for a maximization, by more than the tolerance.

The tolerance (rules 2 and 3) is the larger of TOL x |optimum - LP bound| (TOL closed-gap units) and
TOL x max(1, |optimum|), with TOL = 1e-4. Without a reference optimum the second term uses the recorded primal solution
value instead of the optimum and the first term does not exist (the large models: rules 1 and 3 only). A reference
optimum on the wrong side of the LP bound (below it for a minimization, above it for a maximization) is no reference.
A bound beyond the optimum by at most the tolerance stays valid; its gain share is clipped to 1 in the figure data
only. The recorded evaluations are never changed: annotate() adds the rule in memory.

A reference optimum is contradicted when a recorded primal solution value is better than it by more than the same
tolerance (reference_contradicted); such a model is solved again (SCIP, one thread, seed 0, one hour) and a proved
optimum replaces the reference, otherwise the model has no reference optimum (census_corrections.json).
"""
from __future__ import annotations

import math

INFINITE = 1e19
TOL = 1e-4
RULES = {"infinite": "1: the bound is not finite or |bound| >= 1e19",
         "beyond the reference optimum": "2: the bound is beyond the reference optimum by more than the tolerance",
         "beyond a primal solution": "3: the bound is beyond a primal solution value recorded by SCIP for the model "
                                     "by more than the tolerance"}


def finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v) and abs(v) < INFINITE


def reference_valid(lp, optimum, sense: str) -> bool:
    """A reference optimum is a reference when it is finite and not on the wrong side of the LP bound."""
    if not finite(optimum):
        return False
    if not finite(lp):
        return True
    eps = 1e-9 * max(1.0, abs(lp))
    return optimum >= lp - eps if sense == "minimize" else optimum <= lp + eps


def tolerance(lp, optimum, sense: str, primal=None) -> float:
    """max(TOL closed-gap units, TOL x max(1, |optimum|)); without a reference optimum TOL x max(1, |primal|)."""
    parts = []
    if reference_valid(lp, optimum, sense):
        if finite(lp) and optimum != lp:
            parts.append(TOL * abs(optimum - lp))
        parts.append(TOL * max(1.0, abs(optimum)))
    elif finite(primal):
        parts.append(TOL * max(1.0, abs(primal)))
    return max(parts) if parts else TOL * max(1.0, abs(lp) if finite(lp) else 1.0)


def incumbent(rows, sense: str):
    """The best finite primal solution value of the rows (the smallest for a minimization), or None."""
    vals = [r.get("primal_bound") for r in rows if isinstance(r, dict) and finite(r.get("primal_bound"))]
    if not vals:
        return None
    return min(vals) if sense == "minimize" else max(vals)


def _beyond(value, ref, sense: str, tol: float) -> bool:
    return value > ref + tol if sense == "minimize" else value < ref - tol


def invalid_reason(bound, sense: str, lp=None, optimum=None, primal=None, tol: float | None = None) -> str | None:
    """The rule that makes a recorded bound invalid (a key of RULES), or None when it is valid or absent."""
    if bound is None:
        return None
    if not finite(bound):
        return "infinite"
    if tol is None:
        tol = tolerance(lp, optimum, sense, primal)
    if reference_valid(lp, optimum, sense) and _beyond(bound, optimum, sense, tol):
        return "beyond the reference optimum"
    if finite(primal) and _beyond(bound, primal, sense, tol):
        return "beyond a primal solution"
    return None


def reference_contradicted(lp, optimum, sense: str, primal) -> bool:
    """A recorded primal solution value better than the reference optimum by more than the tolerance."""
    if not (reference_valid(lp, optimum, sense) and finite(primal)):
        return False
    return _beyond(optimum, primal, sense, tolerance(lp, optimum, sense, primal))


def oracle_rows(oracle: dict) -> tuple[list, list]:
    return (oracle.get("subsets") or [],
            [b for b in (oracle.get("baselines") or {}).values() if isinstance(b, dict)])


def annotate(oracle: dict) -> list:
    """Mark every evaluation of an oracle record (subsets and baselines) whose bound is invalid: row["bound_invalid"] =
    the rule, in memory. Returns one entry per invalid evaluation."""
    sense, lp, opt = oracle["sense"], oracle.get("lp_bound"), oracle.get("reference_optimal_value")
    subsets, baselines = oracle_rows(oracle)
    primal = incumbent(subsets + baselines, sense)
    tol = tolerance(lp, opt, sense, primal)
    out = []
    for kind, rows in (("subset", subsets), ("baseline", baselines)):
        for r in rows:
            if not r.get("ok", True):
                continue
            why = invalid_reason(r.get("dual_bound"), sense, lp, opt, primal, tol)
            r["bound_invalid"] = why
            if why:
                out.append({"kind": kind, "key": r.get("key"), "master_families": r.get("master_families"),
                            "linking_var_families": r.get("linking_var_families"), "dual_bound": r.get("dual_bound"),
                            "status": r.get("status"), "solving_time": r.get("solving_time"), "rule": why,
                            "reference_optimum": opt if reference_valid(lp, opt, sense) else None,
                            "primal_solution": primal, "tolerance": tol})
    return out
