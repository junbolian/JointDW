"""Exact family-level Dantzig-Wolfe relaxations of small MILPs.

Paper: Appendix B: exact gains of the templates and the rich models (Tables 4 and 5, Figure 1).

For a set T of constraint families, X_T is the set of points that satisfy the rows of T, the
variable bounds and integrality, and the DW relaxation is P_T = LP cap conv(X_T); its bound is
z(T) = min{c.x : x in P_T} (Geoffrion). Here conv(X_T) is written explicitly over the variables
that the rows of T touch (other variables only meet their bounds, whose hull is the box):
- pure integer touched variables: conv of the enumerated feasible integer points;
- mixed: Balas' extended formulation of the convex hull of the union, over all assignments of
  the touched integer variables, of the bounded polyhedra of the touched continuous variables.
Every LP is solved with an exact certificate (exactlp.solve). Instances must be tiny: the
number of integer assignments is capped by max_assignments.

Functions
- dw_bound(milp, T)             exact z(T)
- min_over(milp, T, alpha)      exact min alpha.x over P_T
- is_valid(milp, T, alpha, b)   alpha.x >= b valid for P_T
- supports(milp, alpha, b)      minimal family sets T for which the inequality is valid
- gain_table(milp)              z(T) - z(empty) for every T (empty = the LP bound)
- mobius(table), zero_marginal_groups(table, base)
- intersection_bound(milp, sets) exact min over the intersection of several P_T
- group_source(milp, W, base)   'polyhedral' or 'lp_geometric' (definition in the docstring)
"""
from __future__ import annotations

import itertools
from fractions import Fraction

from llmdw.exact import exactlp


class TooLarge(RuntimeError):
    pass


def _lp_rows(milp):
    return [(dict(r.coefs), r.lo, r.up) for r in milp.rows]


def _base(milp, objective=None):
    """Columns and rows of the LP relaxation (minimization)."""
    n = len(milp.vars)
    obj = objective if objective is not None else milp.obj
    c = [Fraction(obj.get(j, 0)) for j in range(n)]
    return c, _lp_rows(milp), [v.lb for v in milp.vars], [v.ub for v in milp.vars]


def _dw_model(milp, T, objective=None, max_assignments: int = 1 << 16):
    """Columns and rows of the exact LP over P_T (minimization)."""
    c, rows, col_lo, col_up = _base(milp, objective)
    _add_hull(milp, T, c, rows, col_lo, col_up, max_assignments)
    return c, rows, col_lo, col_up


def _add_hull(milp, T, c, rows, col_lo, col_up, max_assignments: int = 1 << 16):
    """Append to (c, rows, col_lo, col_up) an explicit formulation of x in conv(X_T) (in place)."""
    T = [f for f in milp.families if f in set(T)]
    if not T:
        return
    trows = milp.rows_of(T)
    touched = milp.vars_of(T)
    I = [j for j in touched if milp.vars[j].integer]
    C = [j for j in touched if not milp.vars[j].integer]
    for j in I:
        v = milp.vars[j]
        if v.lb is None or v.ub is None:
            raise TooLarge(f"integer variable {v.name} needs finite bounds")
    for j in C:
        v = milp.vars[j]
        if v.lb is None or v.ub is None:
            raise TooLarge(f"continuous variable {v.name} touched by T needs finite bounds")
    count = 1
    for j in I:
        count *= int(milp.vars[j].ub - milp.vars[j].lb + 1)
        if count > max_assignments:
            raise TooLarge(f"{count}+ integer assignments")
    ranges = [range(int(milp.vars[j].lb), int(milp.vars[j].ub) + 1) for j in I]
    pos = {}

    def new_col(lo, up):
        c.append(Fraction(0))
        col_lo.append(lo)
        col_up.append(up)
        return len(c) - 1

    assignments = []
    int_only_rows = [r for r in trows if all(j in set(I) for j in r.coefs)]
    for ys in itertools.product(*ranges) if I else [()]:
        y = dict(zip(I, (Fraction(v) for v in ys)))
        ok = True
        for r in int_only_rows:
            act = sum(a * y[j] for j, a in r.coefs.items())
            if (r.lo is not None and act < r.lo) or (r.up is not None and act > r.up):
                ok = False
                break
        if ok:
            assignments.append(y)
    if not assignments:
        raise ValueError("no integer assignment satisfies the rows of T: P_T is empty")
    lam = [new_col(Fraction(0), Fraction(1)) for _ in assignments]
    rows.append(({k: Fraction(1) for k in lam}, Fraction(1), Fraction(1)))
    # integer variables: x_j = sum_y y_j lambda_y
    for j in I:
        coefs = {j: Fraction(1)}
        for k, y in zip(lam, assignments):
            if y[j] != 0:
                coefs[k] = coefs.get(k, Fraction(0)) - y[j]
        rows.append((coefs, Fraction(0), Fraction(0)))
    if C:
        w = []
        for k, y in zip(lam, assignments):
            wy = {j: new_col(None, None) for j in C}
            w.append(wy)
            for j in C:
                v = milp.vars[j]
                rows.append(({wy[j]: Fraction(1), k: -v.lb}, Fraction(0), None))   # w >= lb * lambda
                rows.append(({wy[j]: Fraction(1), k: -v.ub}, None, Fraction(0)))   # w <= ub * lambda
            for r in trows:
                coefs = {}
                const = Fraction(0)
                for j, a in r.coefs.items():
                    if j in wy:
                        coefs[wy[j]] = coefs.get(wy[j], Fraction(0)) + a
                    else:
                        const += a * y[j]
                if r.lo is not None and r.up is not None and r.lo == r.up:
                    cc = dict(coefs)
                    cc[k] = cc.get(k, Fraction(0)) + const - r.lo
                    rows.append((cc, Fraction(0), Fraction(0)))
                    continue
                if r.lo is not None:
                    cc = dict(coefs)
                    cc[k] = cc.get(k, Fraction(0)) + const - r.lo
                    rows.append((cc, Fraction(0), None))
                if r.up is not None:
                    cc = dict(coefs)
                    cc[k] = cc.get(k, Fraction(0)) + const - r.up
                    rows.append((cc, None, Fraction(0)))
        for j in C:
            coefs = {j: Fraction(1)}
            for wy in w:
                coefs[wy[j]] = Fraction(-1)
            rows.append((coefs, Fraction(0), Fraction(0)))


def _certified(c, rows, lo, up, what):
    res = exactlp.solve(c, rows, lo, up)
    if not res["verified"]:
        raise RuntimeError(f"exact certificate failed for {what} ({res['status']}, {res['method']})")
    return res


def _solve(milp, T, objective=None):
    return _certified(*_dw_model(milp, T, objective), what=f"T={sorted(T)}")


def dw_bound(milp, T) -> Fraction:
    return _solve(milp, T)["obj"]


def lp_bound(milp) -> Fraction:
    return dw_bound(milp, [])


def intersection_bound(milp, sets, objective=None) -> Fraction:
    """Exact min c.x over the intersection of P_T for T in sets (one hull formulation per T on the same x)."""
    c, rows, lo, up = _base(milp, objective)
    for T in sets:
        _add_hull(milp, T, c, rows, lo, up)
    return _certified(c, rows, lo, up, what=f"intersection of {[sorted(T) for T in sets]}")["obj"]


def group_source(milp, W, base=()) -> str:
    """Source of a zero-marginal group W at base (|W| >= 2), by definition:
    Q = intersection of P_{base | S} over the proper subsets S of W (the maximal ones suffice);
    'lp_geometric' if min over Q exceeds z(base): inequalities whose supports miss a member of W cut off
    the LP optimum together although no single proper subset does; 'polyhedral' otherwise: the gain needs
    an inequality every support of which meets all of W."""
    W, B = frozenset(W), frozenset(base)
    if len(W) < 2:
        return "single"
    zB = dw_bound(milp, B)
    zQ = intersection_bound(milp, [B | (W - {f}) for f in sorted(W)])
    return "lp_geometric" if zQ > zB else "polyhedral"


def min_over(milp, T, alpha: dict) -> Fraction:
    return _solve(milp, T, objective={j: Fraction(a) for j, a in alpha.items()})["obj"]


def is_valid(milp, T, alpha: dict, beta) -> bool:
    return min_over(milp, T, alpha) >= Fraction(beta)


def supports(milp, alpha: dict, beta, families=None) -> list:
    """Minimal family sets T for which alpha.x >= beta is valid for P_T (validity is monotone in T)."""
    fams = list(families or milp.families)
    valid = {}
    for r in range(len(fams) + 1):
        for T in itertools.combinations(fams, r):
            T = frozenset(T)
            if any(S < T and valid.get(S) for S in valid):
                valid[T] = True
                continue
            valid[T] = is_valid(milp, T, alpha, beta)
    return [T for T, ok in valid.items() if ok and not any(S < T and valid[S] for S in valid)]


def gain_table(milp, families=None, skip=None) -> dict:
    """{frozenset T: z(T) - z(empty)} for every subset T of the families (exact)."""
    fams = list(families or milp.families)
    z0 = lp_bound(milp)
    table = {}
    for r in range(len(fams) + 1):
        for T in itertools.combinations(fams, r):
            T = frozenset(T)
            if skip and skip(T):
                continue
            table[T] = (dw_bound(milp, T) - z0) if T else Fraction(0)
    return table


def mobius(table: dict) -> dict:
    """Moebius transform m(S) = sum_{T subset S} (-1)^{|S|-|T|} g(T) over sets whose subsets are all known."""
    out = {}
    for S in table:
        subs = [frozenset(c) for r in range(len(S) + 1) for c in itertools.combinations(sorted(S), r)]
        if all(T in table for T in subs):
            out[S] = sum(((-1) ** (len(S) - len(T))) * table[T] for T in subs)
    return out


def zero_marginal_groups(table: dict, base=frozenset(), tol=Fraction(0)) -> list:
    """Minimal W (disjoint from base) with g(base|W) > g(base) + tol while every proper subset S of W
    has g(base|S) <= g(base) + tol. Groups of size >= 2 are joint supports of the gain at base."""
    base = frozenset(base)
    g0 = table[base]
    out = []
    cand = sorted((T - base for T in table if T >= base and T != base), key=len)
    for W in cand:
        if table[base | W] <= g0 + tol:
            continue
        subs = [frozenset(c) for r in range(len(W)) for c in itertools.combinations(sorted(W), r)]
        if all((base | S) in table and table[base | S] <= g0 + tol for S in subs):
            out.append(W)
    return out
