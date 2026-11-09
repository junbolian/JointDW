"""Reference optimum and validity rule of branch-and-price runs (llmdw.benchmark.validity): the known cases of the
benchmark instances and of bin packing, and one synthetic case per rule."""
import json
import math

import pytest

from llmdw.benchmark import validity as V
from llmdw.instances.families import Model, Row


def _inst(tmp_path, cls="bin_packing", scip=50.0, proved=False, best_known=None, accepted=None):
    d = tmp_path / "i"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"class": cls, "size": {"best_known_bins_in_file": best_known}}))
    (d / "reference.json").write_text(json.dumps({"best_value": scip, "proved_optimal": proved}))
    (d / "oracle.json").write_text(json.dumps({"sense": "minimize", "lp_bound": 48.81333333333333,
                                               "reference_optimal_value": scip, "families": [], "subsets": [],
                                               "baselines": {}}))
    if accepted is not None:
        (d / "reference_accepted.json").write_text(json.dumps({"value": accepted, "source": "feasible solution"}))
    return d


def _oracle(z, lp, sense="minimize"):
    return {"sense": sense, "lp_bound": lp, "reference_optimal_value": z}


def _run(status, primal, dual, root=None, t=10.0, tag="t"):
    return {"ok": True, "status": status, "primal_bound": primal, "dual_bound": dual, "root_bound": root,
            "root_bound_source": "root node statistics" if root is not None else None, "solving_time": t,
            "solved": status == "optimal", "gap": 0.0, "tag": tag}


# ---------------------------------------------------------------- reference optimum

def test_reference_is_the_best_of_scip_best_known_and_accepted(tmp_path):
    d = _inst(tmp_path, scip=49.0, best_known=48)
    o = json.loads((d / "oracle.json").read_text())
    assert V.reference_optimum(d, o) == (48.0, "best-known count of the OR-Library file")
    d2 = _inst(tmp_path / "b", scip=50.0, best_known=50, accepted=49.0)
    o2 = json.loads((d2 / "oracle.json").read_text())
    assert V.reference_optimum(d2, o2) == (49.0, "feasible solution")
    d3 = _inst(tmp_path / "c", cls="facility_location", scip=10.0, proved=True, best_known=5)
    o3 = json.loads((d3 / "oracle.json").read_text())
    assert V.reference_optimum(d3, o3) == (10.0, "SCIP, proved optimal")      # best-known counts only for bin packing


def test_tolerance_is_that_of_the_bound_validity_rule():
    # bp_u120_02: LP 45.2933, z* 46: 1e-4 x max(|z* - z_LP|, max(1, |z*|)) = 0.0046
    assert math.isclose(V.tolerance(_oracle(46.0, 45.29333333333334)), 0.0046)
    # fl_cap43: z* 1,153,000.45
    assert math.isclose(V.tolerance(_oracle(1153000.45, 1152152.2788352298)), 115.300045)


# ---------------------------------------------------------------- rule 1

def test_rule1_optimal_within_tau_is_solved_and_outside_is_not():
    o = _oracle(46.0, 45.29333333333334)
    assert V.apply(_run("optimal", 46.004, 46.004), o)["solved"] is True        # within tau = 0.0046
    assert V.apply(_run("optimal", 46.0, 46.0), o)["validity_rule"] is None


# ---------------------------------------------------------------- rule 2 (the known cases and synthetic ones)

def test_rule2_bp_u120_02_assign_capacity_optimal_at_47_against_proved_46():
    o = _oracle(46.0, 45.29333333333334)
    r = V.apply(_run("optimal", 47.0, 47.0, root=47.0, t=1.6), o)
    assert r["validity_rule"] == "rule 2" and r["solved"] is False and r["solving_time"] == 1800.0
    assert r["status"] == V.FAILED_STATUS and r["status_recorded"] == "optimal"
    # the root bound 47 is beyond z*: rejected, the final gap comes from the LP bound
    assert r["root_bound"] is None and math.isclose(r["gap"], (46.0 - 45.29333333333334) / 46.0)


def test_rule2_bp_u120_06_against_the_best_known_count():
    o = _oracle(48.0, 47.58)
    assert V.classify(_run("optimal", 49.0, 49.0), 48.0, "minimize", V.tolerance(o)) == "rule 2"


def test_rule2_fl_cap43_link_reported_value_against_proved_optimum():
    o = _oracle(1153000.45, 1152152.2788352298)
    rec = _run("optimal", 1158169.045903503, 1158169.045903503, root=1158169.045903503)
    diff = rec["primal_bound"] - o["reference_optimal_value"]
    assert diff > V.tolerance(o) and V.apply(rec, o)["validity_rule"] == "rule 2"


def test_rule2_dual_bound_beyond_z_star_at_the_time_limit_keeps_a_valid_root_bound():
    o = _oracle(100.0, 90.0)
    r = V.apply(_run("timelimit", 104.0, 100.5, root=95.0, t=1800.0), o)           # dual 100.5 > 100 + 0.01
    assert r["validity_rule"] == "rule 2" and r["root_bound"] == 95.0
    assert math.isclose(r["gap"], 5.0 / 100.0) and r["gap_reference"] == "z* and the root bound"


def test_rule2_maximization_sense():
    o = _oracle(-100.0, -90.0, sense="maximize")
    assert V.classify(_run("optimal", -101.0, -101.0), -100.0, "maximize", V.tolerance(o)) == "rule 2"
    assert V.classify(_run("timelimit", -100.0, -99.0), -100.0, "maximize", V.tolerance(o)) is None


# ---------------------------------------------------------------- rule 3

def test_rule3_bp_u120_19_better_solution_feasible_replaces_z_star():
    o = _oracle(50.0, 48.81333333333333)
    rec = _run("optimal", 49.0, 49.0, tag="c3a22b2ec4e")
    assert V.classify(rec, 50.0, "minimize", V.tolerance(o)) == "rule 3"
    assert V.apply(rec, o, {"c3a22b2ec4e": "feasible"})["validity_rule"] == "rule 3, feasible (z* replaced)"
    # once z* is the accepted value, rule 1 holds
    r = V.apply(rec, _oracle(49.0, 48.81333333333333))
    assert r["validity_rule"] is None and r["solved"] is True


def test_rule3_infeasible_solution_is_failed():
    o = _oracle(50.0, 48.0)
    r = V.apply(_run("optimal", 49.0, 49.0, root=48.5, tag="x"), o, {"x": "infeasible"})
    assert r["validity_rule"] == "rule 3, infeasible solution" and r["status"] == V.INFEASIBLE_STATUS
    assert r["solving_time"] == 1800.0 and r["root_bound"] == 48.5


def test_valid_and_failed_twice_runs_are_unchanged():
    o = _oracle(46.0, 45.0)
    assert V.classify({**_run("optimal", 47.0, 47.0), "failed_twice": True}, 46.0, "minimize", 0.0046) is None
    assert V.classify(_run("timelimit", 47.0, 45.9), 46.0, "minimize", 0.0046) is None


# ---------------------------------------------------------------- feasibility check and bin loads

def test_check_solution_rows_bounds_and_integrality(monkeypatch):
    m = Model(rows=[Row("Assign_1", 1.0, 1.0, {"Pack_1_1": 1.0, "Pack_1_2": 1.0}),
                    Row("Capacity_1", -1e20, 0.0, {"Pack_1_1": 6.0, "Use_1": -10.0})],
              var_type={"Pack_1_1": "B", "Pack_1_2": "B", "Use_1": "B"}, var_lb={"Pack_1_1": 0, "Pack_1_2": 0, "Use_1": 0},
              var_ub={"Pack_1_1": 1, "Pack_1_2": 1, "Use_1": 1}, obj={"Use_1": 1.0}, sense="minimize")
    import llmdw.instances.families as fam
    monkeypatch.setattr(fam, "read_model", lambda path: m)
    ok = V.check_solution("x.mps", {"Pack_1_1": 1.0, "Use_1": 1.0})
    assert ok["feasible"] and ok["objective"] == 1.0
    bad = V.check_solution("x.mps", {"Pack_1_1": 1.0})                         # bin not opened: capacity row violated
    assert not bad["feasible"] and bad["rows_violated"][0]["row"] == "Capacity_1"
    frac = V.check_solution("x.mps", {"Pack_1_1": 0.5, "Pack_1_2": 0.5, "Use_1": 1.0})
    assert not frac["feasible"] and frac["integrality_violated"]


def test_bin_loads_and_lower_bound(tmp_path):
    f = tmp_path / "binpack.txt"
    f.write_text("1\n u_test\n 10 3 2\n 6\n 5\n 4\n")
    r = V.bin_loads(f, "u_test", {"Pack_1_1": 1.0, "Pack_2_2": 1.0, "Pack_3_2": 1.0})
    assert r["bins_used"] == 2 and r["every_item_once"] and r["max_load"] == 9 and r["loads_within_capacity"]
    assert r["lower_bound_ceil_sum_over_capacity"] == 2 and r["best_known_bins_in_file"] == 2
