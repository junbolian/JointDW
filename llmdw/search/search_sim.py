"""Offline simulation of decomposition search on a gain table (no solver calls).

Paper: Sections 4 and 5: the covering order, the staged search and JointDW (Tables 2, 9, 12, 14 and 15).

A gain table maps frozenset(T) -> gain for the family sets T that may be evaluated (|T| <= s).
For each policy we count evaluations until a set with the maximal gain is evaluated.

random              uniform random order over all evaluable sets; exact expectation (N + 1)/(M + 1)
greedy              add one family at a time (all single additions evaluated); None if it stalls
covering            prediction-free: evaluate size-s sets that cover all w-subsets for w = 1, 2, ...
anchored(P)         staged anchoring of Theorem 2: stage e = 0, 1, ..., |P| covers the groups that keep
                    |P| - e members of P and add e other families; then every size-s set (for |W| != |P|)
robust(P)           alternate the staged order with the prediction-free covering order; never more than
                    twice the better of the two
ranked(order)       evaluate sets in a given order (e.g. by lookahead value or an LLM ranking)

Family order: every order is built from a list of the families, and only the positions in that list matter (ties go
to the earlier position). The list is `fams` where a function takes it, else the families sorted by name. The result
tables average over random relabelings of the families (llmdw.search.relabel), so they do not depend on the names.

Usage: functions only (called by the analysis, theory_sim and the tests).
"""
from __future__ import annotations

import functools
import itertools


def best_value(table: dict) -> float:
    return max(table.values())


def maximizers(table: dict, tol: float = 1e-9) -> set:
    b = best_value(table)
    return {T for T, v in table.items() if v >= b - tol}


def random_expected(table: dict, tol: float = 1e-9) -> float:
    cand = [T for T in table if T]
    m = len([T for T in cand if T in maximizers(table, tol)])
    return (len(cand) + 1) / (m + 1)


def greedy(table: dict, tol: float = 1e-9, fams=None):
    fams = list(fams) if fams is not None else sorted(set().union(*table.keys()))
    best = best_value(table)
    cur, seen = frozenset(), set()
    while True:
        nbrs = [cur | {f} for f in fams if f not in cur and (cur | {f}) in table]
        seen.update(nbrs)
        if any(table[T] >= best - tol for T in nbrs):
            return len(seen)
        if not nbrs:
            return None
        nxt = max(nbrs, key=lambda T: table[T])
        if table[nxt] <= table[cur] + tol:
            return None
        cur = nxt


def greedy_covering(ground: list, s: int, w: int) -> list:
    """Greedy covering of all w-subsets of `ground` by s-subsets (ln-factor approximation). The covering depends only
    on the positions in `ground` (ties go to the earlier block in itertools.combinations order)."""
    ground = list(ground)
    return [frozenset(ground[i] for i in B) for B in greedy_covering_positions(len(ground), s, w)]


@functools.lru_cache(maxsize=None)
def greedy_covering_positions(n: int, s: int, w: int) -> tuple:
    """greedy_covering of range(n), as tuples of positions."""
    return tuple(tuple(sorted(B)) for B in _greedy_covering(tuple(range(n)), s, w))


def _greedy_covering(ground: tuple, s: int, w: int) -> tuple:
    ground = list(ground)
    if w == 0:
        return (frozenset(ground[:s]),)
    s = min(s, len(ground))
    if w == s:                  # every block covers only itself: the greedy takes them in lexicographic order
        return tuple(frozenset(c) for c in itertools.combinations(ground, s))
    uncovered = {frozenset(c) for c in itertools.combinations(ground, w)}
    blocks = []
    while uncovered:
        best, best_cov = None, -1
        for B in itertools.combinations(ground, s):
            B = frozenset(B)
            cov = sum(1 for c in uncovered if c <= B)
            if cov > best_cov:
                best, best_cov = B, cov
        blocks.append(best)
        uncovered = {c for c in uncovered if not c <= best}
    return tuple(blocks)


def _first_hit(order: list, table: dict, tol: float = 1e-9):
    best = best_value(table)
    seen = set()
    for T in order:
        if T in seen or T not in table:
            continue
        seen.add(T)
        if table[T] >= best - tol:
            return len(seen)
    return None


def covering_order(fams: list, s: int) -> list:
    order = []
    for w in range(1, s + 1):
        order += greedy_covering(fams, s, w)
    return order


def covering(table: dict, s: int, tol: float = 1e-9, fams=None):
    fams = list(fams) if fams is not None else sorted(set().union(*table.keys()))
    return _first_hit(covering_order(fams, s), table, tol)


def staged_order(fams: list, s: int, predicted) -> list:
    """Theorem 2's staged anchoring for a predicted group P (truncated to its first s families in the order of fams).

    Stage e (e = 0, 1, ..., |P|) queries A | R for A in a covering of the (|P| - e)-subsets of P by a-sets
    and R in a covering of the e-subsets of the other families by (s - a)-sets. The split a is the one with
    the fewest queries (ties: the larger a). a = |P| keeps all of P in every query; it is optimal whenever
    e(s + 1) <= |P|(s - |P| + 1), and stage |P| uses a = 0.
    """
    P = [f for f in fams if f in frozenset(predicted)]
    if len(P) > s:
        P = P[:s]
    w = len(P)
    if w == 0:
        return []
    rest = [f for f in fams if f not in P]
    order = []
    for e in range(w + 1):
        best = None
        for a in range(min(w, s - e), w - e - 1, -1):          # larger a first, so ties keep more of P
            A = greedy_covering(P, a, w - e)
            R = [frozenset()] if e == 0 else greedy_covering(rest, s - a, e)
            if best is None or len(A) * len(R) < len(best[0]) * len(best[1]):
                best = (A, R)
        if best is not None:
            order += [x | y for x in best[0] for y in best[1]]
    return order


def anchored(table: dict, s: int, predicted, tol: float = 1e-9, fams=None):
    fams = list(fams) if fams is not None else sorted(set().union(*table.keys()))
    order = staged_order(fams, s, predicted)
    order += [frozenset(c) for c in itertools.combinations(fams, min(s, len(fams)))]  # |W| != |P|: the rest
    return _first_hit(order, table, tol)


def ranked(table: dict, order: list, tol: float = 1e-9):
    return _first_hit([frozenset(T) for T in order], table, tol)


def robust(table: dict, s: int, predicted, tol: float = 1e-9, fams=None):
    fams = list(fams) if fams is not None else sorted(set().union(*table.keys()))
    a = staged_order(fams, s, predicted)
    b = covering_order(fams, s)
    order = [x for pair in itertools.zip_longest(a, b) for x in pair if x is not None]
    return _first_hit(order, table, tol)
