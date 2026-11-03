"""Numbers for the search-cost figure: covering bounds of Theorem 1 and the prediction formula of Theorem 2.

Paper: Section 4: Theorems 1 and 2.

For k families, block budget s and a hidden group of size w (gain = c * [W subset of T]):
- schonheim(k, s, w)   lower bound on the covering number C(k, s, w), hence on worst-case evaluations
- ratio(k, s, w)       C(k, w) / C(s, w), the asymptotic value (Rodl)
- greedy_cover(k,s,w)  size of a greedy covering design (constructive upper bound; small k only)
- predicted(k,s,w,e)   Theorem 2(b) lower bound with a predicted group missing e members
                       (C(k - w, e) / C(s - w, e) when e(s + 1) <= w(s - w + 1); C(k - w, w) / C(s, w) for e = w)
- simulate(...)        evaluations of search_sim policies on the indicator gain table (small k)

Usage
-----
python -m llmdw.search.theory_sim --out work/theory_sim.csv --kmax 24 --s 6 --w 3
"""
from __future__ import annotations

import argparse
import csv
import itertools
from math import ceil, comb

from llmdw.search import search_sim


def schonheim(k: int, s: int, w: int) -> int:
    if w == 0:
        return 1
    return ceil(k / s * schonheim(k - 1, s - 1, w - 1))


def ratio(k: int, s: int, w: int) -> float:
    return comb(k, w) / comb(s, w)


def predicted(k: int, s: int, w: int, e: int) -> float:
    """Lower bound of Theorem 2(b) for a predicted group of size w missing e members."""
    if e == 0:
        return 1.0
    if e == w:
        return comb(k - w, w) / comb(s, w)
    best = max(comb(a, w - e) * comb(s - a, e) for a in range(w - e, min(w, s) + 1))
    return comb(w, e) * comb(k - w, e) / best


def greedy_cover(k: int, s: int, w: int) -> int:
    return len(search_sim.greedy_covering(list(range(k)), s, w))


def indicator_table(k: int, s: int, hidden) -> dict:
    W = frozenset(hidden)
    return {frozenset(T): (1.0 if W <= set(T) else 0.0)
            for r in range(s + 1) for T in itertools.combinations(range(k), r)}


def simulate(k: int, s: int, w: int, e: int, reps: int = 30, seed: int = 0) -> dict:
    """Mean evaluations over random placements of the hidden group (labels permuted), with a predicted
    group that keeps w - e true members and replaces e by random outsiders."""
    import random

    rng = random.Random(seed)
    acc = {"greedy_failures": 0, "covering": 0.0, "anchored": 0.0, "random_expected": 0.0}
    for _ in range(reps):
        perm = list(range(k))
        rng.shuffle(perm)
        hidden = perm[:w]
        outsiders = rng.sample(perm[w:], e)
        pred = hidden[: w - e] + outsiders
        tab = indicator_table(k, s, hidden)
        acc["greedy_failures"] += search_sim.greedy(tab) is None
        acc["covering"] += search_sim.covering(tab, s) / reps
        acc["anchored"] += search_sim.anchored(tab, s, pred) / reps
        acc["random_expected"] += search_sim.random_expected(tab) / reps
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmax", type=int, default=24)
    ap.add_argument("--s", type=int, default=6)
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--simulate-kmax", type=int, default=11)
    args = ap.parse_args(argv)
    rows = []
    for k in range(args.s + 1, args.kmax + 1):
        row = {"k": k, "s": args.s, "w": args.w, "schonheim": schonheim(k, args.s, args.w),
               "ratio": ratio(k, args.s, args.w)}
        for e in range(args.w + 1):
            row[f"predicted_e{e}"] = predicted(k, args.s, args.w, e)
        if k <= args.simulate_kmax:
            row["greedy_cover"] = greedy_cover(k, args.s, args.w)
            for e in (0, 1):
                sim = simulate(k, args.s, args.w, e)
                row[f"sim_anchored_e{e}"] = round(sim["anchored"], 2)
            row["sim_covering"] = round(sim["covering"], 2)
            row["sim_random"] = round(sim["random_expected"], 2)
            row["sim_greedy_failures"] = sim["greedy_failures"]
        rows.append(row)
    keys = sorted({k2 for r in rows for k2 in r}, key=lambda x: (x not in ("k", "s", "w"), x))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
