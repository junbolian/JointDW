"""The reduced basis system of the exact basis check gives the same result as the full basis matrix."""
from __future__ import annotations

import itertools
import random
from fractions import Fraction

from llmdw.exact import exactlp, hull
from llmdw.instances import templates


def _random_lp(rng):
    n, m = rng.randint(1, 7), rng.randint(0, 7)
    c = [Fraction(rng.randint(-5, 5), rng.randint(1, 3)) for _ in range(n)]
    lo = [Fraction(rng.randint(-3, 0)) if rng.random() < 0.8 else None for _ in range(n)]
    up = [Fraction(rng.randint(1, 4)) if rng.random() < 0.8 else None for _ in range(n)]
    rows = []
    for _ in range(m):
        coefs = {j: Fraction(rng.randint(-4, 4), rng.randint(1, 2)) for j in rng.sample(range(n), rng.randint(1, n))}
        coefs = {j: a for j, a in coefs.items() if a != 0} or {0: Fraction(1)}
        kind, b = rng.random(), Fraction(rng.randint(-3, 6))
        if kind < 0.25:
            rows.append((coefs, b, b))                                   # equality
        elif kind < 0.5:
            rows.append((coefs, b, b + rng.randint(0, 3)))               # ranged (degenerate when width 0)
        elif kind < 0.75:
            rows.append((coefs, b, None))
        else:
            rows.append((coefs, None, b))
    return c, rows, lo, up


def test_reduced_equals_full_on_random_lps():
    rng = random.Random(20261108)
    checked = 0
    for _ in range(300):
        c, rows, lo, up = _random_lp(rng)
        h, status, basis = exactlp._highs_solve(c, rows, lo, up)
        if str(status).split(".")[-1] != "kOptimal":
            continue
        full = exactlp._verify(c, rows, lo, up, basis.col_status, basis.row_status, reduced=False)
        red = exactlp._verify(c, rows, lo, up, basis.col_status, basis.row_status, reduced=True)
        assert full == red
        checked += full[0]
    assert checked >= 100


def test_reduced_rejects_what_full_rejects():
    # a wrong basis (all structural columns at their lower bounds, every slack basic) is checked the same way
    rng = random.Random(7)
    for _ in range(100):
        c, rows, lo, up = _random_lp(rng)
        import highspy
        BS = highspy.HighsBasisStatus
        cs = [BS.kLower if v is not None else BS.kZero for v in lo]
        rs = [BS.kBasic] * len(rows)
        assert (exactlp._verify(c, rows, lo, up, cs, rs, reduced=False)
                == exactlp._verify(c, rows, lo, up, cs, rs, reduced=True))


def test_template_gains_unchanged(monkeypatch):
    for make in (templates.uls, templates.clsp, templates.gap):
        m = make()
        sets = [list(T) for r in range(len(m.families) + 1) for T in itertools.combinations(m.families, r)]
        monkeypatch.setattr(exactlp, "REDUCED_BASIS", False)
        full = [hull.dw_bound(m, T) for T in sets]
        monkeypatch.setattr(exactlp, "REDUCED_BASIS", True)
        red = [hull.dw_bound(m, T) for T in sets]
        assert full == red
