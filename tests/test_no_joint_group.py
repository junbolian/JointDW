"""The reference value, the sets outside the budget, and the scoring on fake records (no solver run;
the solver entry point raises while scoring)."""
import csv
import json
import math

import pytest

from llmdw.benchmark import no_joint_group as ef
from llmdw.benchmark import time_truth


def _inst(tmp_path, cls, ref, best):
    d = tmp_path / "x"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"class": cls, "size": {"best_known_bins_in_file": best}}))
    sense = "maximize" if cls == "generalized_assignment" else "minimize"
    return d, {"reference_optimal_value": ref, "sense": sense}


def test_bin_packing_reference_is_the_smaller_value(tmp_path):
    d, o = _inst(tmp_path, "bin_packing", 49.0, 48)
    assert ef.reference(d, o) == (48.0, "best-known bins")


def test_bin_packing_keeps_scip_when_not_larger(tmp_path):
    d, o = _inst(tmp_path, "bin_packing", 46.0, 46)
    assert ef.reference(d, o) == (46.0, "SCIP reference")


def test_generalized_assignment_keeps_scip(tmp_path):
    d, o = _inst(tmp_path, "generalized_assignment", -100.0, None)
    assert ef.reference(d, o) == (-100.0, "SCIP reference")


def test_outside_budget_sets():
    assert frozenset({"Assign_#", "Capacity_#"}) in ef.OUTSIDE_BUDGET["bin_packing"]


# ---------------------------------------------------------------- scoring on fake records (no solver run)

A, C, S = "Assign_#", "Capacity_#", "SymmetryBreaking_#"
AC = frozenset({A, C})


def _info(**kw):
    base = {"outside": ["Assign_#|Capacity_#"], "joint": [], "capacity_closes_gap": False}
    return {**base, **kw}


def test_support_outside_budget_is_a_false_alarm_only_where_capacity_closes_the_gap():
    assert ef.support_outcome([A, C], _info(capacity_closes_gap=True)) == "false alarm"
    assert ef.support_outcome([A, C], _info(capacity_closes_gap=False)) == "outside budget"
    # a support within the budget on an instance without a joint support: a false alarm in every case
    assert ef.support_outcome([A, S], _info(capacity_closes_gap=False)) == "false alarm"
    assert ef.support_outcome([A, S], _info(joint=["Assign_#|SymmetryBreaking_#"])) == "instance with a joint support"
    assert ef.support_outcome(None, _info()) is None


def _write_instance(root, name, gains, plan_sets, runs, closes=False):
    """A bin-packing instance: oracle.json with the given gain shares (LP 10, reference 12), bnp_plan.json with the
    given sets and one configuration "M", and the branch-and-price rows {set key: (time, solved)} (seeds 0, 1, 2)."""
    d = root / "instances" / name
    (d / "bnp").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"class": "bin_packing", "size": {"best_known_bins_in_file": 12},
                                             "rows": 1, "columns": 1, "binaries": 1}))
    fams = [A, C, S]
    subsets = [{"master_families": sorted(set(fams) - set(T.split("|"))), "dual_bound": 10 + 2 * g,
                "status": "optimal", "ok": True, "linking_var_families": []} for T, g in gains.items()]
    (d / "oracle.json").write_text(json.dumps({
        "sense": "minimize", "families": fams, "lp_bound": 10.0, "reference_optimal_value": 12.0,
        "reference_proved_optimal": True, "subsets": subsets,
        "baselines": {"gcg_spfwh": {"ok": True, "dual_bound": 10.0, "representable": True,
                                    "aligned_master_families": [C]}}}))
    table = {frozenset(T.split("|")): g for T, g in gains.items() if T != "Assign_#|Capacity_#"}
    best = max(table.values())
    ties = sorted((sorted(T) for T, v in table.items() if v >= best - 1e-9), key=lambda T: (len(T), T))
    plan = {"methods": {"true_best": ties[0], "true_best_ties": ties, "lookahead": plan_sets["lookahead"],
                        "lookahead_five_duals": plan_sets["five"], "gcg_detected": [A, S],
                        "M": [{"sample": i, "status": st, "set": s, "first_support": sup}
                              for i, (s, sup, st) in enumerate(plan_sets["M"])]}}
    (d / "bnp_plan.json").write_text(json.dumps(plan))
    for key, (t, solved) in runs.items():
        T = key if key == "gcg_default" else key.split("|")
        for seed in (0, 1, 2):
            (d / "bnp" / f"{time_truth._tag(T, seed)}.row.json").write_text(json.dumps(
                {"instance": name, "set": key, "seed": seed, "ok": True, "status": "optimal" if solved else "timelimit",
                 "solving_time": t, "solved": solved, "gap": 0.0 if solved else 0.1, "nodes": 3,
                 "primal_bound": 12.0 if solved else None, "dual_bound": 12.0 if solved else 11.0}))
    return d


def _fake_root(tmp_path, complete_p2=False):
    root = tmp_path / "bench"
    gains1 = {A: 0.0, C: 0.5, S: 0.0, "Assign_#|SymmetryBreaking_#": 0.0, "Capacity_#|SymmetryBreaking_#": 0.5}
    sets = {"lookahead": [A], "five": [A, C],
            "M": [([C], None, "answered"), ([A, C], [A, C], "answered"), ([], None, "missing")]}
    runs1 = {C: (100.0, True), A: (1800.0, False), "Assign_#|Capacity_#": (5.0, True),
             "Assign_#|SymmetryBreaking_#": (50.0, True), "gcg_default": (1.0, True),
             S: (0.5, True)}                                       # S is not planned: its rows are not read
    _write_instance(root, "p1", gains1, sets, runs1)
    gains2 = {A: 0.0, C: 1e-12, S: 0.0, "Assign_#|SymmetryBreaking_#": 0.0, "Capacity_#|SymmetryBreaking_#": 0.0}
    runs2 = {C: (10.0, True), A: (20.0, True), "Assign_#|Capacity_#": (30.0, True)}
    if complete_p2:
        runs2.update({"Assign_#|SymmetryBreaking_#": (40.0, True), "gcg_default": (2.0, True)})
    _write_instance(root, "p2", gains2, sets, runs2)                # missing runs unless complete_p2
    ans = tmp_path / "answers"
    for n in ("p1", "p2"):
        (ans / n).mkdir(parents=True)
        for s, f in ((0, "original.json"), (1, "original.s1.json")):
            (ans / n / f).write_text(json.dumps({"answer": {}, "supports": []}))
    g = tmp_path / "groups.csv"
    with open(g, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "class", "group"])
        w.writerows([["p1", "bin_packing", "p1"], ["p2", "bin_packing", "p1"]])
    return root, ans, g


def _score(tmp_path, monkeypatch, root, ans, g):
    from llmdw.gcg import gcg_run

    monkeypatch.setattr(gcg_run, "run", gcg_run.run)               # restored after the test (score replaces it)
    return ef.score(root, ["p1", "p2"], {"M": str(ans)}, g, tmp_path / "out", tmp_path / "tex",
                    expected_not_scored=("p2",))


def test_score_not_scored_rule_and_partial_runs(tmp_path, monkeypatch):
    root, ans, g = _fake_root(tmp_path)
    before = sorted(p.name for p in (root / "instances" / "p2" / "bnp").iterdir())
    res = _score(tmp_path, monkeypatch, root, ans, g)
    # reading never writes into a folder of runs, and no solver can start
    assert sorted(p.name for p in (root / "instances" / "p2" / "bnp").iterdir()) == before
    from llmdw.gcg import gcg_run
    with pytest.raises(RuntimeError):
        gcg_run.run("dw-given", "x.mps", "x.json")
    assert res["not_scored"] == ("p2",) and res["complete"] == ["p1"]
    meth = {r["method"]: r for r in res["tables"]["methods"] if r["class"] == "bin_packing"}
    # p2: every set has gain 0, so no method is scored on naming the best set there; p1 is scored
    assert meth["M"]["hit_bound_n"] == 1 and meth["M"]["n_not_scored"] == 1 and meth["M"]["n_instances"] == 2
    assert math.isclose(meth["M"]["hit_bound_mean"], 1 / 3)
    # a set outside the budget is never gain 0 and has no gain; a sample without a set has gain 0
    # a first set outside the budget has gain 0 in the pooled gain and is not counted among the sets of gain 0
    five = meth["lookahead (five duals)"]
    assert five["gain_n"] == 2 and five["gain_mean"] == 0.0 and five["gain_zero_instances"] == 0
    # Capacity does not close the gap: the predicted {Assign, Capacity} names a set outside the budget
    assert meth["M"]["false_alarms"] == 0 and meth["M"]["support_outside_budget"] == 2
    assert meth["lookahead (five duals)"]["set_outside_budget"] == 2
    st = {r["class"]: r for r in res["status"]}["all"]
    assert st["runs_missing"] == 6 and st["instances_with_every_run"] == 1
    assert (tmp_path / "out" / "PARTIAL.txt").exists()
    assert (tmp_path / "tex" / "experiment_F_methods.tex").read_text().count("Partial") == 1
    for f in ("methods.csv", "methods_time.csv", "paired.csv", "paired_time_truth.csv"):
        for r in csv.DictReader(open(tmp_path / "out" / f)):
            assert r["n_instances"] != "" and r["n_groups"] != ""
            assert not any("/" in v for v in r.values())


def test_time_optimal_set_is_among_the_planned_sets_within_the_budget(tmp_path, monkeypatch):
    root, ans, g = _fake_root(tmp_path, complete_p2=True)
    res = _score(tmp_path, monkeypatch, root, ans, g)
    tt = {r["instance"]: r for r in csv.DictReader(open(tmp_path / "out" / "time_truth.csv"))}
    # GCG as shipped (1 s) and the unplanned {SymmetryBreaking} (0.5 s) are faster but not candidates, and
    # {Assign, Capacity} (5 s) is planned but outside the budget: {Assign, SymmetryBreaking} (50 s) is time-optimal
    assert tt["p1"]["time_optimal"] == "Assign_#|SymmetryBreaking_#"
    assert tt["p1"]["time_optimal_outside_budget"] == "False"
    assert tt["p1"]["ranked_sets"] == "planned sets within the budget" and tt["p1"]["n_ranked_sets"] == "3"
    assert tt["p2"]["time_optimal"] == C
    assert not (tmp_path / "out" / "PARTIAL.txt").exists()
    mt = {(r["class"], r["method"]): r for r in res["tables"]["methods_time"]}
    # p1: samples Capacity, {Assign, Capacity}, none (no hit); p2: Capacity (hit), {Assign, Capacity}, none
    assert math.isclose(mt[("bin_packing", "M")]["hit_time_strict_mean"], 1 / 6)
    five = mt[("bin_packing", "lookahead (five duals)")]
    # its set {Assign, Capacity} is outside the budget on both: a miss, with its own measured time in the ratio
    assert five["hit_time_strict_mean"] == 0.0 and five["set_outside_budget"] == 2
    assert math.isclose(five["ratio_time_optimal_geo"], math.exp((math.log(6 / 51) + math.log(31 / 11)) / 2))
    bo = mt[("bin_packing", ef.BOUND_OPTIMAL)]
    # bound-optimal set p1: Capacity (101 s against 51 s); p2: Assign, the first of its ties (21 s against 11 s)
    assert math.isclose(bo["ratio_time_optimal_geo"], math.exp((math.log(101 / 51) + math.log(21 / 11)) / 2))
    assert bo["set_outside_budget"] == 0
    gcg = mt[("bin_packing", "GCG default")]
    assert gcg["ratio_gcg_shipped_geo"] == 1.0


def test_time_truth_exclude_keeps_the_summary():
    rows = [{"instance": "p", "set": k, "seed": s, "time": t, "solved": True, "gap": 0.0, "failed_twice": False}
            for k, t in (("a", 5.0), ("b", 9.0)) for s in (0, 1, 2)]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "instances" / "p").mkdir(parents=True)
        (Path(d) / "instances" / "p" / "bnp_plan.json").write_text(json.dumps(
            {"methods": {"true_best": ["b"], "true_best_ties": [["b"]]}}))
        (Path(d) / "instances" / "p" / "meta.json").write_text(json.dumps({"class": "x"}))
        (Path(d) / "instances" / "p" / "oracle.json").write_text(json.dumps(
            {"sense": "minimize", "families": ["a", "b"], "lp_bound": 0.0, "reference_optimal_value": 1.0,
             "subsets": [{"master_families": ["b"], "dual_bound": 0.2, "ok": True, "status": "optimal"},
                         {"master_families": ["a"], "dual_bound": 0.8, "ok": True, "status": "optimal"}],
             "baselines": {}}))
        tt = time_truth.time_truth(rows, Path(d), exclude={"p": ["a"]})["p"]
    assert tt["time_optimal"] == "b" and tt["order"] == ["b"] and "a" in tt["summary"]


def test_false_alarm_where_capacity_closes_the_gap(tmp_path, monkeypatch):
    root, ans, g = _fake_root(tmp_path, complete_p2=True)
    o = json.loads((root / "instances" / "p1" / "oracle.json").read_text())
    for s in o["subsets"]:
        if s["master_families"] == [A, S]:
            s["dual_bound"] = 12.0                                  # {Capacity} closes the whole gap
    (root / "instances" / "p1" / "oracle.json").write_text(json.dumps(o))
    res = _score(tmp_path, monkeypatch, root, ans, g)
    meth = {r["method"]: r for r in res["tables"]["methods"] if r["class"] == "bin_packing"}
    assert meth["M"]["false_alarms"] == 1 and meth["M"]["support_outside_budget"] == 1
