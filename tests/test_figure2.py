"""Figure 2 data (llmdw.tables.figure2): the greedy covering sizes."""
import itertools

import numpy as np

from llmdw.tables import figure2


def _dense_greedy(n, b, t):
    """The dense coverage-matrix greedy covering (the earlier implementation), for small n."""
    tsets = list(itertools.combinations(range(n), t))
    index = {c: i for i, c in enumerate(tsets)}
    blocks = list(itertools.combinations(range(n), b))
    cover = np.zeros((len(blocks), len(tsets)), dtype=np.uint8)
    for r, block in enumerate(blocks):
        for c in itertools.combinations(block, t):
            cover[r, index[c]] = 1
    uncovered = np.ones(len(tsets), dtype=np.int64)
    count = 0
    while uncovered.any():
        r = int(np.argmax(cover @ uncovered))
        uncovered[cover[r].astype(bool)] = 0
        count += 1
    return count


FREE_GREEDY = {8: 4, 9: 9, 10: 10, 11: 15, 12: 15, 13: 24, 14: 26, 15: 34, 16: 43, 17: 55, 18: 52, 19: 71, 20: 85,
               21: 103, 22: 77, 23: 104, 24: 122}


def test_index_greedy_equals_the_dense_greedy_on_small_sets():
    for n, b, t in [(6, 3, 2), (7, 4, 2), (8, 6, 3), (9, 5, 3), (10, 6, 3), (11, 6, 3), (9, 4, 1), (7, 7, 3)]:
        assert figure2.greedy_cover_size(n, b, t) == _dense_greedy(n, b, t), (n, b, t)


def test_free_greedy_values_k_8_to_24():
    assert figure2.FREE_GREEDY_KMAX == 24
    for k, v in FREE_GREEDY.items():
        assert figure2.greedy_cover_size(k, figure2.S, figure2.W) == v, k
