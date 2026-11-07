"""Validity of recorded GCG bounds and of reference optima (llmdw.scoring.bounds), on the recorded census cases and on
one synthetic case per rule."""
import math

from llmdw.scoring import bounds, structure


def _oracle(sense, lp, opt, rows):
    return {"sense": sense, "lp_bound": lp, "reference_optimal_value": opt, "families": ["A", "B"],
            "subsets": [{"key": k, "master_families": m, "linking_var_families": [], "ok": True, "status": st,
                         "dual_bound": b, "primal_bound": pb} for k, m, st, b, pb in rows], "baselines": {}}


# the recorded census cases: (sense, LP bound, reference optimum, recorded bound, best recorded primal, expected rule)
CASES = {
    "supportcase26": ("minimize", 1288.1021611356973, 1745.123813, 1e20, 2698.1843496183847, "infinite"),
    "radiationm18-12-05": ("minimize", 17554.499999999996, 17574.0, 1e20, 17572.0, "infinite"),
    "noswot": ("maximize", 42.1301309620843, 41.0, 40.58620689655172, 41.0, "beyond the reference optimum"),
    "cvs08r139-94": ("maximize", 139.0, 116.0, 79.0, 79.0, "beyond the reference optimum"),
    "neos-631517": ("minimize", 6494999.999936, 11490666.665, 11491266.666634023, 11804980.666634, None),
}


def test_recorded_cases():
    for name, (sense, lp, opt, b, primal, rule) in CASES.items():
        assert bounds.invalid_reason(b, sense, lp, opt, primal) == rule, name


def test_combined_tolerance_of_neos_631517():
    sense, lp, opt, b, primal, _ = CASES["neos-631517"]
    tol = bounds.tolerance(lp, opt, sense, primal)
    assert math.isclose(tol, 1e-4 * opt)                              # the relative term is the larger one here
    assert 1e-4 * abs(opt - lp) < b - opt <= tol                      # beyond 1e-4 closed-gap units, within the tolerance
    sense, lp, opt, b, primal, _ = CASES["noswot"]
    assert opt - b > bounds.tolerance(lp, opt, sense, primal)


def test_rule_1_synthetic():
    for b in (1e19, -1e19, math.inf, float("nan")):
        assert bounds.invalid_reason(b, "minimize", 0.0, 10.0, None) == "infinite"
    assert bounds.invalid_reason(9.0, "minimize", 0.0, 10.0, None) is None


def test_rule_2_synthetic():
    assert bounds.invalid_reason(10.5, "minimize", 0.0, 10.0, None) == "beyond the reference optimum"
    assert bounds.invalid_reason(10.0005, "minimize", 0.0, 10.0, None) is None        # within 1e-4 x 10
    assert bounds.invalid_reason(-10.5, "maximize", 0.0, -10.0, None) == "beyond the reference optimum"
    # a reference on the wrong side of the LP bound is no reference
    assert bounds.invalid_reason(5.0, "minimize", 1.0, 0.0, None) is None


def test_rule_3_synthetic_and_without_an_optimum():
    assert bounds.invalid_reason(8.0, "minimize", 0.0, 10.0, 7.0) == "beyond a primal solution"
    assert bounds.invalid_reason(8.0, "minimize", 0.0, None, 7.0) == "beyond a primal solution"
    assert bounds.invalid_reason(7.0005, "minimize", 0.0, None, 7.0) is None           # within 1e-4 x 7
    assert bounds.invalid_reason(2.0, "maximize", 5.0, None, 3.0) == "beyond a primal solution"


def test_reference_contradicted_by_a_primal_solution():
    assert bounds.reference_contradicted(3.6363636363636362, 1.5, "maximize", 3.0)          # b-ball
    assert bounds.reference_contradicted(17554.499999999996, 17574.0, "minimize", 17572.0)  # radiationm18-12-05
    assert not bounds.reference_contradicted(155306.7, 155393.0, "minimize", 155392.0)       # radiationm40-10-02
    assert not bounds.reference_contradicted(0.0, 10.0, "minimize", 10.0)


def test_instance_tables_drops_invalid_bounds_and_lists_them():
    o = _oracle("minimize", 0.0, 10.0, [("A", ["A"], "optimal", 1e20, None), ("B", ["B"], "optimal", 4.0, 12.0),
                                         ("C", [], "optimal", 11.0, None)])
    o["families"] = ["A", "B"]
    info = structure.instance_tables(o, None)
    tab = info["tables"][()]
    # T = the families outside the master: master A (bound 1e20, invalid) is T = {B}, master B is T = {A}
    assert frozenset({"A"}) in tab and frozenset({"B"}) not in tab
    # the full set (no master) is not in the table, but its recorded bound is checked as every other one
    assert [i["rule"] for i in info["invalid"]] == ["infinite", "beyond the reference optimum"]
    assert math.isclose(tab[frozenset({"A"})]["gain"], 0.4)
