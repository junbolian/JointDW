"""Checks of the bounds of the staged search (Theorem 2, thm:pred): exact arithmetic for v_e, the least weight of an
e-cover, and MILP/LP (HiGHS through SciPy) for covering numbers.

1. v_e: v_e >= L_e; v_0 = 1; v_w = 1/C(s, w) = L_w; v_e = 1/C(s-w, e) = L_e when e(s+1) <= w(s-w+1); a cover with
   members of one size weighs at least v_e; the counts of the appendix paragraph (95 triples, 16 with v_e above L_e,
   largest factor 4/3, 8 where one size is not optimal); the cases (3, 6, 2) and (4, 7, 3); one size attains v_e for
   s <= 3.
2. Exact covering numbers of the hidden groups: tau >= ceil(v_e C(k-w, e)), and the LP value equals L_e C(k-w, e) for
   k >= s + w.
3. Finite k, (s, w, e) = (3, 6, 2) with greedy coverings: the stage of the staged search is at least the lower bound.

The checks need SciPy (the extra "learned") and are skipped without it. They are skipped by default; set
JOINTDW_SLOW_TESTS=1 to run them.
"""
import os
from fractions import Fraction as Fr
from functools import lru_cache
from itertools import combinations
from math import ceil, comb

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("JOINTDW_SLOW_TESTS") != "1",
                                reason="set JOINTDW_SLOW_TESTS=1 to check the bounds of the staged search")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
from scipy.optimize import Bounds, LinearConstraint, linprog, milp  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402


def v_exact(s, w, e):
    """least weight of an e-cover, exact DP over bitmasks of covered (w-e)-subsets"""
    t = w - e
    targets = list(combinations(range(w), t)); n = len(targets)
    cols = {}
    for a in range(0, min(w, s - e) + 1):
        for A in combinations(range(w), a):
            m = 0
            for i, T in enumerate(targets):
                if set(T) <= set(A): m |= 1 << i
            if m:
                wt = Fr(1, comb(s - a, e))
                if m not in cols or wt < cols[m]: cols[m] = wt
    # drop dominated columns
    items = list(cols.items())
    items = [(m, c) for (m, c) in items if not any((m | m2) == m2 and m != m2 and c2 <= c for (m2, c2) in items)]
    full = (1 << n) - 1

    @lru_cache(None)
    def f(mask):
        if mask == full: return Fr(0)
        i = 0
        while mask >> i & 1: i += 1
        r = None
        for m, c in items:
            if m >> i & 1:
                val = c + f(mask | m)
                if r is None or val < r: r = val
        return r
    return f(0)


def cov(n, a, t, tl=60):
    if t == 0 or n <= a: return 1
    a = min(a, n)
    targets = list(combinations(range(n), t)); cols = list(combinations(range(n), a))
    idx = {T: i for i, T in enumerate(targets)}
    M = lil_matrix((len(targets), len(cols)))
    for j, A in enumerate(cols):
        for T in combinations(A, t): M[idx[T], j] = 1
    r = milp(np.ones(len(cols)), constraints=LinearConstraint(M.tocsr(), lb=1), integrality=np.ones(len(cols)),
             bounds=Bounds(0, 1), options={'time_limit': tl})
    assert r.status == 0
    return int(round(r.fun))


def uniform_weight(s, w, e):
    return min(Fr(cov(w, a, w - e), comb(s - a, e)) for a in range(w - e, min(w, s - e) + 1))


def Lfrac(s, w, e):
    A = range(w - e, min(w, s - e) + 1)
    return Fr(comb(w, e), max(comb(a, w - e) * comb(s - a, e) for a in A))


def system(k, s, w, e):
    P = list(range(w)); O = list(range(w, k))
    cands = [frozenset(A) | frozenset(B) for A in combinations(P, w - e) for B in combinations(O, e)]
    idx = {c: i for i, c in enumerate(cands)}
    cols = []
    for a in range(w - e, min(w, s - e) + 1):
        b = min(s - a, k - w)
        for A in combinations(P, a):
            for B in combinations(O, b): cols.append((A, B))
    M = lil_matrix((len(cands), len(cols)))
    for j, (A, B) in enumerate(cols):
        for A2 in combinations(A, w - e):
            for B2 in combinations(B, e): M[idx[frozenset(A2) | frozenset(B2)], j] = 1
    return M.tocsr(), len(cands), len(cols)


def greedy_cov(n, a, t):
    if t == 0 or n <= a: return 1
    targets = set(combinations(range(n), t)); cnt = 0
    cols = list(combinations(range(n), a))
    while targets:
        best = max(cols, key=lambda A: sum(1 for T in combinations(A, t) if T in targets))
        for T in combinations(best, t): targets.discard(T)
        cnt += 1
    return cnt


def run_checks():
    """The checks of the module docstring; returns (failed checks, summary lines)."""
    fails, notes = [], []

    def check(name, cond, info=''):
        if not cond: fails.append(f"FAIL {name} {info}")

    # 1. v_e: bounds, closed forms, counts of the appendix paragraph
    n = gap = improved = 0; mx = Fr(1)
    for w in range(2, 7):
        for s in range(w, 11):
            for e in range(0, w + 1):
                v = v_exact(s, w, e); fr = Lfrac(s, w, e)
                check('v>=frac', v >= fr, (s, w, e))
                if e == 0: check('v0', v == 1, (s, w))
                if e == w: check('vw', v == Fr(1, comb(s, w)) and fr == v, (s, w))
                if e * (s + 1) <= w * (s - w + 1): check('cond', v == Fr(1, comb(s - w, e)) == fr, (s, w, e))
                if 1 <= e < w:
                    n += 1; u = uniform_weight(s, w, e)
                    check('uniform>=v', u >= v, (s, w, e))
                    if v > fr: gap += 1; mx = max(mx, v / fr)
                    if u > v: improved += 1
    notes.append(f"triples {n} | v>L-constant: {gap} | max factor {mx} | one size not optimal: {improved}")
    check('counts', (n, gap, improved, mx) == (95, 16, 8, Fr(4, 3)))
    check('(3,6,2)', v_exact(6, 3, 2) == Fr(4, 15) and uniform_weight(6, 3, 2) == Fr(3, 10) and Lfrac(6, 3, 2) == Fr(1, 4))
    check('(4,7,3)', v_exact(7, 4, 3) == Fr(1, 5) == Lfrac(7, 4, 3)
          and [Fr(cov(4, a, 1), comb(7 - a, 3)) for a in (1, 2, 3, 4)] == [Fr(4, 20), Fr(2, 10), Fr(2, 4), Fr(1, 1)])
    # one size attains v for s<=3
    for s in (2, 3):
        for w in range(2, s + 1):
            for e in range(1, w): check('s<=3 one size', uniform_weight(s, w, e) == v_exact(s, w, e), (s, w, e))

    # 2. exact covering numbers of the hidden groups: tau >= ceil(v*C(k-w,e)); LP value == L_e for k>=s+w
    cases = [(3, 2, 1, range(5, 10)), (4, 3, 1, range(6, 10)), (4, 2, 1, range(6, 10)), (5, 3, 2, range(7, 10)),
             (6, 3, 2, range(8, 11)), (6, 4, 2, range(8, 10)), (5, 2, 1, range(7, 10)), (4, 3, 2, range(6, 9))]
    m = 0
    for (s, w, e, ks) in cases:
        v = v_exact(s, w, e)
        for k in ks:
            M, nc, ncol = system(k, s, w, e)
            r = milp(np.ones(ncol), constraints=LinearConstraint(M, lb=1), integrality=np.ones(ncol), bounds=Bounds(0, 1),
                     options={'time_limit': 20})
            lb = ceil(v * comb(k - w, e)); L = Lfrac(s, w, e) * comb(k - w, e)
            if r.status == 0:
                m += 1; tau = int(round(r.fun))
                check('tau>=v*C', tau >= lb, (k, s, w, e, tau, lb))
            if k >= s + w:
                lp = linprog(np.ones(ncol), A_ub=-M, b_ub=-np.ones(nc), bounds=(0, None), method='highs')
                check('LP==L_e', abs(lp.fun - float(L)) < 1e-7, (k, s, w, e, lp.fun, float(L)))
    notes.append(f"exact covering numbers checked: {m}")

    # 3. finite-k upper bound of (a) >= lower bound of (b), and ratio tends to 1 (greedy coverings as upper bounds on cov)
    for (s, w, e) in [(6, 3, 2)]:
        v = v_exact(s, w, e)
        for k in (9, 12, 15, 18):
            mixed = greedy_cov(k - w, 4, 2) + greedy_cov(k - w, 5, 2)       # cover {{p1,p2},{p3}}
            one = 3 * greedy_cov(k - w, 5, 2)                                # three singletons
            lb = ceil(v * comb(k - w, e))
            check('stage>=lb', mixed >= lb and one >= lb, (k, mixed, one, lb))
            notes.append(f"(3,6,2) k={k}: lower bound {lb}, stage 2 mixed (greedy) {mixed}, one size (greedy) {one}")
    return fails, notes


def test_bounds_of_the_staged_search():
    fails, notes = run_checks()
    print("\n".join(notes))
    assert fails == [], "\n".join(fails)
