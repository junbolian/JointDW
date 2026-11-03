"""Search costs that do not depend on the names of the families.

Paper: Section 6 (measures): the search cost of every table, invariant to the names of the families.

Every search order is built from a list of the families and depends only on the positions in that list (covering
designs, the staged search and its truncation of the anchor to s families, the shortlist orders, greedy and
lookahead tie-breaking). To make a result invariant to relabeling the families, each order is run under PERMUTATIONS
uniformly random orders of the families of an instance, and the per-instance value is the mean over these runs.

Orders: for each instance, numpy default_rng on SeedSequence(0, spawn_key=(key,)) with key the first 8 bytes of
sha256(instance name) (one stream per instance, keyed by its name) draws PERMUTATIONS permutations; permutation p
orders the families as [base[p[0]], base[p[1]], ...], where base lists the families in the order of their first row
in the model file (file_order), which does not depend on their names. Renaming the families therefore leaves every
result unchanged, and the mean over the permutations removes the dependence on the file order up to Monte Carlo
error. Ties in the lookahead order (equal value and size) go to the set whose positions, sorted, come first.

A run that never reaches the target (greedy stalls, or an order without a target set) has no value (nan in `runs`); a
failed run counts as a failure of the instance, not as a missing value: llmdw.scoring.results.run_value returns None when any
run fails (`failed_share` gives the share of runs without a value). The random order keeps its closed form, which does
not depend on an order.

Usage: functions only (llmdw.scoring.results, llmdw.scoring.rich_models and llmdw.scoring.large_models).
"""
from __future__ import annotations

from llmdw import constants
from llmdw.constants import FAMILY_ORDER_SEED, FAMILY_ORDERS

import hashlib
import itertools

import numpy as np

from llmdw.search import search_sim

PERMUTATIONS = FAMILY_ORDERS
IDENTITY = False            # validation: the families sorted by name only (no relabeling)
_orders: dict = {}
_memo: dict = {}


def configure(permutations: int = 1000, identity: bool = False):
    """Number of family orders per instance (the first n of the stream), or the identity order only."""
    global PERMUTATIONS, IDENTITY
    PERMUTATIONS, IDENTITY = int(permutations), bool(identity)
    clear()


def file_order(model, fams) -> list:
    """The families of fams in the order of their first row in the model (families without a row last, by name).
    model is a model of llmdw.instances.families or a summary of llmdw.instances.summaries."""
    from llmdw.instances.families import family_of
    want, out = set(fams), []
    seq = model.family_sequence if hasattr(model, "family_sequence") else (family_of(r.name) for r in model.rows)
    for f in seq:
        if f in want and f not in out:
            out.append(f)
    return out + sorted(want - set(out))


def family_orders(name: str, base, n: int | None = None) -> list:
    """The PERMUTATIONS family orders of an instance (cached): the permutations applied to base (file_order); the
    first n of the stream when n is given. With IDENTITY, the families sorted by name only."""
    n = PERMUTATIONS if n is None else n
    base = list(base)
    if IDENTITY:
        return [sorted(base)]
    key = (name, tuple(base), n)
    if key not in _orders:
        seed = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(np.random.SeedSequence(FAMILY_ORDER_SEED, spawn_key=(seed,)))
        _orders[key] = [[base[i] for i in rng.permutation(len(base))] for _ in range(n)]
    return _orders[key]


def clear():
    _orders.clear()
    _memo.clear()


# ---------------------------------------------------------------- lazy orders (as search_sim and llmdw.scoring.results)

def covering_iter(fams, s):
    """search_sim.covering_order(fams, s), lazily."""
    fams = list(fams)
    for w in range(1, s + 1):
        for B in search_sim.greedy_covering_positions(len(fams), s, w):
            yield frozenset(fams[i] for i in B)


def staged_iter(fams, s, predicted):
    """search_sim.staged_order(fams, s, predicted), lazily."""
    fams = list(fams)
    pred = frozenset(predicted)
    P = [f for f in fams if f in pred][:s]
    w = len(P)
    if w == 0:
        return
    rest = [f for f in fams if f not in P]
    for e in range(w + 1):
        best = None
        for a in range(min(w, s - e), w - e - 1, -1):
            A = search_sim.greedy_covering_positions(len(P), a, w - e)
            R = ((),) if e == 0 else search_sim.greedy_covering_positions(len(rest), s - a, e)
            if best is None or len(A) * len(R) < len(best[0]) * len(best[1]):
                best = (A, R)
        if best is not None:
            for x in best[0]:
                X = frozenset(P[i] for i in x)
                for y in best[1]:
                    yield X | frozenset(rest[i] for i in y)


def alternate(a, b):
    """a and b alternately (results.model_order and results.interleave; repeats are skipped by first_hit)."""
    for x, y in itertools.zip_longest(a, b):
        if x is not None:
            yield x
        if y is not None:
            yield y


def model_iter(fams, s, P):
    """results.model_order: the staged order around P alternated with the covering order."""
    return alternate(staged_iter(fams, s, P), covering_iter(fams, s))


def shortlist_iter(C, fams, s):
    """results.shortlist_order: the covering order inside C (C in the order of fams), then that of all families,
    without repeats (the repeats are removed before an interleaving, as in shortlist_order)."""
    fams = list(fams)
    inner = [f for f in fams if f in frozenset(C)]
    seen = set()
    for T in itertools.chain(covering_iter(inner, min(s, len(inner))) if inner else (), covering_iter(fams, s)):
        if T not in seen:
            seen.add(T)
            yield T


TIE_TOL = constants.TIE_TOL


def tie_classes(values, tol: float = TIE_TOL) -> dict:
    """{value: class}: the distinct values in decreasing order, with a new class whenever a value lies more than tol
    below the previous one (single linkage), so two values within tol always share a class; class 0 holds the
    largest values. Rankings of lookahead values sort by class, so tied values never order by their float noise."""
    out, cls, prev = {}, -1, None
    for v in sorted(set(values), reverse=True):
        if prev is None or prev - v > tol:
            cls += 1
        out[v] = cls
        prev = v
    return out


def lookahead_list(lookahead, fams, max_size: int) -> list:
    """results.lookahead_order for the order fams: the sets with at most max_size families by decreasing lookahead
    value (values within TIE_TOL tied, tie_classes); ties to fewer families, then to the earlier sorted positions in
    fams."""
    rank = {f: i for i, f in enumerate(fams)}
    la = [(T, v) for T, v in (lookahead or []) if T and len(T) <= max_size]
    cls = tie_classes(v for _, v in la)
    return [T for T, _ in sorted(la, key=lambda tv: (cls[tv[1]], len(tv[0]), sorted(rank[f] for f in tv[0])))]


def set_key(T):
    """Sort key of a set of families: size, then the sorted names (independent of string hashing)."""
    return len(T), sorted(T)


def seconds(times, sets, limit=120.0):
    """Sum of times (default limit) over the distinct sets, in sorted order (set_key), so that the float sum does not
    depend on string hashing; None without times."""
    if times is None:
        return None
    return sum(times.get(x, limit) for x in sorted(set(sets), key=set_key))


def first_hit(order, tab, goal, times=None, limit=120.0):
    """(evaluations, seconds) until the first set of goal; repeats and sets outside tab are skipped. seconds is the
    sum of times (default limit) over the evaluated sets, or None without times; (None, None) when no set of goal is
    reached."""
    seen = set()
    for T in order:
        if T in seen or T not in tab:
            continue
        seen.add(T)
        if T in goal:
            return len(seen), seconds(times, seen, limit)
    return None, None


def greedy_hit(tab, goal, fams, times=None, limit=120.0):
    """results.greedy_hit in the order fams: (evaluations, seconds)."""
    cur, seen = frozenset(), []
    seen_set = set()
    while True:
        nbrs = [cur | {f} for f in fams if f not in cur and (cur | {f}) in tab]
        for T in nbrs:
            if T not in seen_set:
                seen_set.add(T)
                seen.append(T)
        if any(T in goal for T in nbrs):
            return len(seen), seconds(times, seen, limit)
        if not nbrs:
            return None, None
        nxt = max(nbrs, key=lambda T: tab[T])
        if tab[nxt] <= tab[cur] + 1e-9:
            return None, None
        cur = nxt


# ---------------------------------------------------------------- averaging over the orders

def runs(key, name, fams, fn):
    """fn(order) -> (evaluations or None, seconds or None) for one family order; returns two arrays over the
    family orders of the instance (nan where there is no value), memoized under key."""
    if key not in _memo:
        ev, sec = [], []
        for order in family_orders(name, fams):
            c, t = fn(order)
            ev.append(np.nan if c is None else float(c))
            sec.append(np.nan if t is None else float(t))
        _memo[key] = (np.asarray(ev), np.asarray(sec))
    return _memo[key]


def mean(arr):
    """Mean over the runs with a value; None when no run has one."""
    arr = np.asarray(arr, dtype=float)
    ok = arr[~np.isnan(arr)]
    return float(ok.mean()) if len(ok) else None


def failed_share(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.isnan(arr).mean()) if len(arr) else 0.0


# ---------------------------------------------------------------- ranked lists and narrowed lookahead orders

def dedupe(it):
    """The sets of it in order, each once (repeats removed before an interleaving, as in shortlist_iter)."""
    seen = set()
    for T in it:
        if T not in seen:
            seen.add(T)
            yield T


def ranked_iter(fams, s, ranked):
    """Ranked-list order for the order fams: the sets of the ranked list with at most s families, in list order; then
    the staged search of JointDW around the first set of the list (staged_iter); then the covering order; without
    repeats. An empty list gives the covering order."""
    ranked = [frozenset(T) for T in ranked]
    head = (T for T in ranked if T and len(T) <= s)
    staged = staged_iter(fams, s, ranked[0]) if ranked else iter(())
    return dedupe(itertools.chain(head, staged, covering_iter(fams, s)))


def narrow_iter(lookahead, fams, s, C):
    """"Narrow then lookahead" for the order fams: the sets of at most s families inside C in lookahead order (ties
    as in lookahead_list), then all other sets of at most s families in lookahead order; without repeats. An empty C
    gives the plain lookahead order."""
    la = lookahead_list(lookahead, fams, s)
    C = frozenset(C)
    return dedupe(itertools.chain((T for T in la if T <= C), (T for T in la if not T <= C)))
