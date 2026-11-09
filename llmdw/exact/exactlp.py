"""LP optimum certified in exact rational arithmetic.

Paper: Appendix B: the rational certificate behind every exact gain (Tables 4 and 5, Figure 1).

solve(c, rows, col_lo, col_up) minimizes c.x subject to lo_i <= a_i.x <= up_i and column
bounds, with rational data (Fractions; None = infinite). Procedure:
1. HiGHS dual/primal simplex in floating point, presolve off, returns an optimal basis.
2. The basic solution of that basis is recomputed in exact rational arithmetic (python-flint):
   basic values from B z_B = -N z_N, duals from B^T y = c_B, reduced costs d = c - A^T y.
3. Exact primal feasibility (bounds of basic columns and row activities) and exact dual
   feasibility (sign of each nonbasic reduced cost against the bound it sits at) are checked.
   If both hold, the basis is optimal and the objective value is exact ("verified").
4. Otherwise the problem is re-solved with SymPy's exact simplex when it is small, else the
   result is returned with verified = False (callers treat that as an error).
The certificate is checked with exact arithmetic only; HiGHS is used to find it, not to trust it.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np


def _fq(v):
    from flint import fmpq

    v = Fraction(v)
    return fmpq(v.numerator, v.denominator)


def _to_fraction(q) -> Fraction:
    return Fraction(int(q.p), int(q.q))


def _highs_solve(c, rows, col_lo, col_up):
    import highspy

    inf = highspy.kHighsInf
    n, m = len(c), len(rows)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "off")
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)
    h.setOptionValue("dual_feasibility_tolerance", 1e-9)
    lo = np.array([-inf if v is None else float(v) for v in col_lo])
    up = np.array([inf if v is None else float(v) for v in col_up])
    h.addCols(n, np.array([float(v) for v in c]), lo, up, 0, np.array([], dtype=np.int32),
              np.array([], dtype=np.int32), np.array([]))
    starts, idx, vals = [], [], []
    for coefs, _, _ in rows:
        starts.append(len(idx))
        for j, a in coefs.items():
            idx.append(j)
            vals.append(float(a))
    rlo = np.array([-inf if r[1] is None else float(r[1]) for r in rows])
    rup = np.array([inf if r[2] is None else float(r[2]) for r in rows])
    if m:
        h.addRows(m, rlo, rup, len(idx), np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32),
                  np.array(vals))
    h.run()
    status = h.getModelStatus()
    basis = h.getBasis()
    return h, status, basis


# The basis system: the full basis matrix of [A | -I] (m x m), or the reduced system of the rows whose slack is
# nonbasic and the basic structural columns (a row with a basic slack only fixes that slack, and its dual value is 0),
# which has the same solution x and the same duals y. REDUCED_BASIS selects the default of _verify.
REDUCED_BASIS = True
CERTIFICATES = {False: "exact basis check, full basis matrix", True: "exact basis check, reduced basis system"}


def certificate(reduced=None) -> str:
    return CERTIFICATES[REDUCED_BASIS if reduced is None else bool(reduced)]


def _verify(c, rows, col_lo, col_up, col_status, row_status, reduced=None):
    """Exact check of a simplex basis. Returns (ok, obj, x) with exact Fractions. reduced: solve the reduced basis
    system instead of the full basis matrix (REDUCED_BASIS by default); the checks are the same."""
    reduced = REDUCED_BASIS if reduced is None else reduced
    import highspy
    from flint import fmpq, fmpq_mat

    BS = highspy.HighsBasisStatus
    n, m = len(c), len(rows)
    status = list(col_status) + list(row_status)
    basic = [k for k in range(n + m) if status[k] == BS.kBasic]
    if len(basic) != m:
        return False, None, None
    lo = list(col_lo) + [r[1] for r in rows]
    up = list(col_up) + [r[2] for r in rows]
    # column k of [A | -I]
    cols = [dict() for _ in range(n)]
    for i, (coefs, _, _) in enumerate(rows):
        for j, a in coefs.items():
            cols[j][i] = Fraction(a)

    def column(k):
        return cols[k] if k < n else {k - n: Fraction(-1)}

    val = {}
    for k in range(n + m):
        if status[k] == BS.kBasic:
            continue
        st = status[k]
        if st == BS.kLower:
            v = lo[k]
        elif st == BS.kUpper:
            v = up[k]
        elif st == BS.kZero:
            v = Fraction(0)
        else:
            v = lo[k] if lo[k] is not None else (up[k] if up[k] is not None else Fraction(0))
        if v is None:
            return False, None, None
        val[k] = Fraction(v)
    rhs = [Fraction(0)] * m
    for k, v in val.items():
        if v != 0:
            for i, a in column(k).items():
                rhs[i] -= a * v
    if reduced:
        bcols = [k for k in basic if k < n]
        tight = [i for i in range(m) if status[n + i] != BS.kBasic]
        if len(bcols) != len(tight):
            return False, None, None
        tpos = {i: q for q, i in enumerate(tight)}
        if bcols:
            Br = fmpq_mat(len(tight), len(bcols))
            for p, k in enumerate(bcols):
                for i, a in cols[k].items():
                    q = tpos.get(i)
                    if q is not None:
                        Br[q, p] = _fq(a)
            try:
                zB = Br.solve(fmpq_mat(len(tight), 1, [_fq(rhs[i]) for i in tight]))
            except ZeroDivisionError:
                return False, None, None
            for p, k in enumerate(bcols):
                val[k] = _to_fraction(zB[p, 0])
        for k in basic:                                   # a basic slack: the activity of its row
            if k >= n:
                val[k] = sum((Fraction(a) * val[j] for j, a in rows[k - n][0].items()), Fraction(0))
    elif m:
        B = fmpq_mat(m, m)
        for p, k in enumerate(basic):
            for i, a in column(k).items():
                B[i, p] = _fq(a)
        try:
            zB = B.solve(fmpq_mat(m, 1, [_fq(v) for v in rhs]))
        except ZeroDivisionError:
            return False, None, None
        for p, k in enumerate(basic):
            val[k] = _to_fraction(zB[p, 0])
    # primal feasibility
    for k in basic:
        v = val[k]
        if lo[k] is not None and v < lo[k]:
            return False, None, None
        if up[k] is not None and v > up[k]:
            return False, None, None
    # duals: B^T y = c_B
    cost = [Fraction(v) for v in c] + [Fraction(0)] * m
    if reduced:
        yv = [Fraction(0)] * m                            # rows with a basic slack: dual value 0
        if bcols:
            try:
                yr = Br.transpose().solve(fmpq_mat(len(bcols), 1, [_fq(cost[k]) for k in bcols]))
            except ZeroDivisionError:
                return False, None, None
            for q, i in enumerate(tight):
                yv[i] = _to_fraction(yr[q, 0])
    elif m:
        cB = fmpq_mat(m, 1, [_fq(cost[k]) for k in basic])
        try:
            y = B.transpose().solve(cB)
        except ZeroDivisionError:
            return False, None, None
        yv = [_to_fraction(y[i, 0]) for i in range(m)]
    else:
        yv = []
    basic_set = set(basic)
    for k in range(n + m):
        if k in basic_set:
            continue
        d = cost[k] - sum(a * yv[i] for i, a in column(k).items())
        fixed = lo[k] is not None and up[k] is not None and lo[k] == up[k]
        if fixed:
            continue
        st = status[k]
        at_lo = lo[k] is not None and val[k] == lo[k]
        at_up = up[k] is not None and val[k] == up[k]
        if at_lo and not at_up and d < 0:
            return False, None, None
        if at_up and not at_lo and d > 0:
            return False, None, None
        if not at_lo and not at_up and d != 0:
            return False, None, None
    x = [val[k] for k in range(n)]
    obj = sum(Fraction(c[j]) * x[j] for j in range(n))
    return True, obj, x


def _sympy_solve(c, rows, col_lo, col_up):
    import sympy
    from sympy.solvers.simplex import lpmin

    xs = sympy.symbols(f"x0:{len(c)}")
    cons = []
    for coefs, lo, up in rows:
        e = sum(sympy.Rational(Fraction(a).numerator, Fraction(a).denominator) * xs[j] for j, a in coefs.items())
        if lo is not None and up is not None and lo == up:
            cons.append(sympy.Eq(e, sympy.Rational(lo.numerator, lo.denominator)))
            continue
        if lo is not None:
            cons.append(e >= sympy.Rational(lo.numerator, lo.denominator))
        if up is not None:
            cons.append(e <= sympy.Rational(up.numerator, up.denominator))
    for j in range(len(c)):
        if col_lo[j] is not None:
            cons.append(xs[j] >= sympy.Rational(col_lo[j].numerator, col_lo[j].denominator))
        if col_up[j] is not None:
            cons.append(xs[j] <= sympy.Rational(col_up[j].numerator, col_up[j].denominator))
    obj = sum(sympy.Rational(Fraction(v).numerator, Fraction(v).denominator) * xs[j] for j, v in enumerate(c))
    val, sol = lpmin(obj, cons)
    x = [Fraction(int(sympy.fraction(sol.get(s, 0))[0]), int(sympy.fraction(sol.get(s, 0))[1])) for s in xs]
    return Fraction(int(sympy.fraction(val)[0]), int(sympy.fraction(val)[1])), x


def solve(c, rows, col_lo, col_up, sympy_limit: int = 4000) -> dict:
    """Minimize c.x; rows are (coefs dict, lo, up). Returns status, obj (Fraction), x, verified, method."""
    import highspy

    h, status, basis = _highs_solve(c, rows, col_lo, col_up)
    MS = highspy.HighsModelStatus
    if status == MS.kInfeasible:
        return {"status": "infeasible", "obj": None, "x": None, "verified": False, "method": "highs"}
    if status == MS.kUnbounded:
        return {"status": "unbounded", "obj": None, "x": None, "verified": False, "method": "highs"}
    if status == MS.kOptimal:
        ok, obj, x = _verify(c, rows, col_lo, col_up, basis.col_status, basis.row_status)
        if ok:
            return {"status": "optimal", "obj": obj, "x": x, "verified": True, "method": "highs+exact_basis",
                    "certificate": certificate()}
    if len(c) * max(1, len(rows)) <= sympy_limit:
        obj, x = _sympy_solve(c, rows, col_lo, col_up)
        return {"status": "optimal", "obj": obj, "x": x, "verified": True, "method": "sympy_exact_simplex"}
    float_obj = h.getInfo().objective_function_value
    return {"status": "optimal_unverified", "obj": Fraction(float_obj), "x": None, "verified": False,
            "method": "highs"}
