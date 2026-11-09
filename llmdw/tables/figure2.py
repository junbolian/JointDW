"""Worst-case number of bound evaluations to find a hidden group of w = 3 families when every
evaluation may convexify at most s = 6 families (Figure 2 and Table 3 of the paper).

Columns of the output file (space separated):
  k            number of constraint families
  free_lb      C(k, w) / C(s, w): lower bound for search without a prediction (Theorem 1)
  free_greedy  size of a greedy covering of the w-subsets by s-sets, an achievable worst case
  lb_e{1,2,3}  L_e of Theorem 2(b): lower bound when a predicted group of size w misses e members
  st_e{1,2,3}  queries in stages 0..e of the staged search of Theorem 2(a) built with greedy coverings,
               the worst case over hidden groups that miss e members
"""
from __future__ import annotations

import argparse
import csv
import itertools
from math import comb
from pathlib import Path

import numpy as np

S, W = 6, 3
K_MIN, K_MAX = 8, 24
FREE_GREEDY_KMAX = 24


def greedy_cover_size(n: int, b: int, t: int) -> int:
    """Size of a greedy covering of all t-subsets of an n-set by b-subsets (b <= n): each step takes the block that
    covers the most uncovered t-subsets, the first such block in itertools order on ties (np.argmax). Every block
    stores the indices of its t-subsets (int32, shape (blocks, C(b, t))), so memory grows with C(n, b) C(b, t)."""
    if t == 0:
        return 1
    b = min(b, n)
    if t > b:
        raise ValueError("t > b")
    index = {c: i for i, c in enumerate(itertools.combinations(range(n), t))}
    nblocks, per = comb(n, b), comb(b, t)
    idx = np.empty((nblocks, per), dtype=np.int32)
    for r, block in enumerate(itertools.combinations(range(n), b)):
        idx[r] = [index[c] for c in itertools.combinations(block, t)]
    uncovered = np.ones(len(index), dtype=np.int32)
    count = 0
    while uncovered.any():
        gains = uncovered[idx].sum(axis=1)
        r = int(np.argmax(gains))
        uncovered[idx[r]] = 0
        count += 1
    return count


def stage_size(k: int, e: int) -> int:
    """Smallest stage e over the splits a, as in Theorem 2(a), with greedy coverings."""
    best = None
    for a in range(W - e, min(W, S - e) + 1):
        size = greedy_cover_size(W, a, W - e) * greedy_cover_size(k - W, S - a, e)
        best = size if best is None else min(best, size)
    return best


def lower_bound(k: int, e: int) -> float:
    if e == 0:
        return 1.0
    fmax = max(comb(a, W - e) * comb(S - a, e) for a in range(W - e, min(W, S - e) + 1))
    return comb(W, e) * comb(k - W, e) / fmax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        row = {"k": k, "free_lb": comb(k, W) / comb(S, W)}
        row["free_greedy"] = greedy_cover_size(k, S, W) if k <= FREE_GREEDY_KMAX else "nan"
        cum = 0
        for e in range(W + 1):
            cum += stage_size(k, e)
            if e >= 1:
                row[f"lb_e{e}"] = lower_bound(k, e)
                row[f"st_e{e}"] = cum
        rows.append(row)
    keys = ["k", "free_lb", "free_greedy", "lb_e1", "st_e1", "lb_e2", "st_e2", "lb_e3", "st_e3"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, delimiter=" ", lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({key: (f"{r[key]:.4f}" if isinstance(r[key], float) else r[key]) for key in keys})


if __name__ == "__main__":
    main()
