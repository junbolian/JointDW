"""Lattice measures of a gain table (llmdw.scoring.gain_structure) on hand-made tables."""
import pytest

from llmdw.scoring import gain_structure as gs

F = frozenset


def _table():
    # A, B zero; C positive; AB a joint gain; BC below C (one violation); ABC above the packing C + AB
    return {F("A"): 0.0, F("B"): 0.0, F("C"): 0.2, F("AB"): 0.5, F("AC"): 0.2, F("BC"): 0.1, F("ABC"): 0.9}


def test_measures():
    m, sets = gs.analyze_table(_table(), 3)
    assert m["n_sets_in_table"] == 7
    assert m["n_violations"] == 1 and m["max_violation"] == pytest.approx(0.1)
    assert m["n_minimal_positive"] == 2 and m["minimal_sizes"] == "1:1;2:1;3:0"
    assert m["minimal_sets"] == "C; A|B"
    d = {r["set"]: (r["d1"], r["d2"]) for r in sets}
    assert set(d) == {"C", "A|B", "A|C", "B|C", "A|B|C"}
    assert d["A|B|C"] == pytest.approx((0.4, 0.2))
    assert d["B|C"] == pytest.approx((-0.1, -0.1))
    assert d["A|C"] == pytest.approx((0.0, 0.0))
    assert m["best_gain"] == pytest.approx(0.9) and m["best_sets"] == "A|B|C" and m["best_is_union"] is True


def test_best_not_a_union_and_ties():
    # budget 2: the best set AB contains only the minimal positive set A
    m, _ = gs.analyze_table({F("A"): 0.3, F("B"): 0.0, F("AB"): 0.4}, 2)
    assert m["best_is_union"] is False and m["d1_max"] == pytest.approx(0.1)
    # a tie within the tolerance with the minimal set B makes it hold
    m, _ = gs.analyze_table({F("A"): 0.0, F("B"): 0.4, F("AB"): 0.4 + 1e-8}, 2)
    assert m["best_sets"] == "B; A|B" and m["best_is_union"] is True


def test_no_positive_set_and_budget():
    m, sets = gs.analyze_table({F("A"): 0.0, F("B"): 1e-9, F("AB"): 0.0}, 2)
    assert m["best_is_union"] == "no positive set" and m["n_minimal_positive"] == 0 and not sets
    m, _ = gs.analyze_table(_table(), 2)               # sets beyond the budget are left out
    assert m["n_sets_in_table"] == 6 and m["best_sets"] == "A|B"


def _oracle(rows, opt=10.0):
    return {"sense": "minimize", "lp_bound": 0.0, "reference_optimal_value": opt, "families": ["A", "B", "C"],
            "subsets": [{"key": "|".join(m), "master_families": m, "linking_var_families": [], "ok": st != "failed",
                         "status": st, "dual_bound": b, "primal_bound": None} for m, st, b in rows],
            "baselines": {}}


def test_census_lattice():
    # the lattice of k = 3 families: every nonempty set of at most 2 families (master = the complement)
    rows = [(["B", "C"], "optimal", 0.0), (["A", "C"], "optimal", 0.0), (["A", "B"], "timelimit", 2.0),
            (["C"], "optimal", 5.0), (["B"], "optimal", 2.0), (["A"], "optimal", 2.0)]
    rec, why = gs.census_table("x", _oracle(rows), {})
    assert why is None and rec["s"] == 2 and rec["lp_gap"] == 10.0 and rec["n_stopped_at_limit"] == 1
    assert rec["gains"][F("AB")] == pytest.approx(0.5) and rec["gains"][F("C")] == pytest.approx(0.2)
    rows[0] = (["B", "C"], "failed", None)
    rec, why = gs.census_table("x", _oracle(rows), {})
    assert rec is None and why.startswith("incomplete table: 1 of 6") and why.endswith("(1 failed)")
    rec, why = gs.census_table("x", _oracle(rows, opt=12.0), {"optimum": {"x": None}, "notes": {"x": "wrong side"}})
    assert rec is None and why == "reference optimum removed: wrong side"


def test_minimal_sets_sharing_a_family():
    from llmdw.scoring.gain_structure import analyze_table
    F = frozenset
    g = {F(): 0, F("a"): 0, F("b"): 0, F("c"): 0, F("ab"): 0.5, F("bc"): 0.4, F("ac"): 0, F("abc"): 0.6}
    out, _ = analyze_table(g, 3)
    assert out["n_minimal_positive"] == 2
    assert out["n_minimal_pairs_sharing_a_family"] == 1 and out["minimal_pairwise_disjoint"] is False
    g2 = {F(): 0, F("a"): 0.2, F("b"): 0.1, F("c"): 0, F("ab"): 0.3, F("ac"): 0.2, F("bc"): 0.1, F("abc"): 0.3}
    out2, _ = analyze_table(g2, 3)
    assert out2["minimal_pairwise_disjoint"] is True and out2["n_minimal_pairs_sharing_a_family"] == 0
