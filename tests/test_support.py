"""Exact ground truth: every structural fact stated in templates.py is checked here.

Run from the repository root:  python -m pytest -q tests/test_support.py
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from llmdw.exact import exactlp, hull
from llmdw.instances import templates


def _g(table, *fams):
    return table[frozenset(fams)]


def test_exact_lp_certificate():
    # min x + y  s.t. 2x + y >= 1, x + 3y >= 1, x, y >= 0  -> 3/5 at (2/5, 1/5)
    res = exactlp.solve([1, 1], [({0: 2, 1: 1}, Fraction(1), None), ({0: 1, 1: 3}, Fraction(1), None)],
                        [Fraction(0), Fraction(0)], [None, None])
    assert res["verified"] and res["obj"] == Fraction(3, 5) and res["x"] == [Fraction(2, 5), Fraction(1, 5)]


def test_uls_joint_support():
    m = templates.uls()
    t = hull.gain_table(m)
    assert _g(t, "Balance_#") == 0 and _g(t, "SetupLink_#") == 0
    assert _g(t, "Balance_#", "SetupLink_#") > 0
    a, b = templates.uls_ls_inequality(m, 2, (2,))          # nontrivial (l,S): needs both families
    assert hull.supports(m, a, b) == [frozenset({"Balance_#", "SetupLink_#"})]
    a, b = templates.uls_ls_inequality(m, 3, (2, 3))        # l = last period: implied by the LP
    assert hull.supports(m, a, b) == [frozenset()]


def test_clsp_zero_marginal_pairs():
    m = templates.clsp()
    t = hull.gain_table(m)
    assert all(_g(t, f) == 0 for f in m.families)
    groups = {frozenset(W) for W in hull.zero_marginal_groups(t)}
    assert groups == {frozenset({"Balance_#_#", "SetupLink_#_#"}), frozenset({"Capacity_#", "SetupLink_#_#"})}


def test_needle_gain_is_indicator():
    m = templates.needle(k=4, hidden=(1, 2))
    W = set(m.facts["hidden"])
    for T, g in hull.gain_table(m).items():
        assert g == (Fraction(1, 2) if W <= set(T) else 0)


def test_geometric_complementarity_with_singleton_supports():
    m = templates.geometric()
    t = hull.gain_table(m)
    assert _g(t, "LimitA_#") == 0 and _g(t, "LimitB_#") == 0 and _g(t, "LimitA_#", "LimitB_#") == Fraction(1, 2)
    x1 = 0
    assert hull.supports(m, {x1: -1}, 0) == [frozenset({"LimitA_#"})]   # x1 <= 0 has a singleton support


def test_group_sources_follow_the_definition():
    g = templates.geometric()
    tab = hull.gain_table(g)
    groups = [W for W in hull.zero_marginal_groups(tab) if len(W) >= 2]
    assert sorted(sorted(W) for W in groups) == [["Cap_#", "Cover_#", "LimitA_#"], ["Cap_#", "Cover_#", "LimitB_#"],
                                                  ["LimitA_#", "LimitB_#"]]
    assert tab[frozenset(["LimitA_#", "LimitB_#"])] == Fraction(1, 2)
    assert tab[frozenset(["Cap_#", "Cover_#", "LimitA_#"])] == Fraction(1, 4)
    assert all(hull.group_source(g, W) == "lp_geometric" for W in groups)
    c = templates.clsp()
    assert hull.group_source(c, ["Balance_#_#", "SetupLink_#_#"]) == "polyhedral"
    assert hull.group_source(c, ["Capacity_#", "SetupLink_#_#"]) == "polyhedral"
    n = templates.needle(k=4, hidden=(1, 2))
    assert hull.group_source(n, n.facts["hidden"]) == "polyhedral"


def test_gap_singleton_support():
    m = templates.gap()
    t = hull.gain_table(m)
    assert _g(t, "Job_#") == 0 and _g(t, "Capacity_#") > 0 and _g(t, "Capacity_#") == _g(t, "Capacity_#", "Job_#")


@pytest.mark.parametrize("make", [templates.uls, templates.clsp, templates.gap,
                                  templates.geometric, lambda: templates.needle(k=4, hidden=(1, 2))])
def test_full_convexification_equals_mip_optimum(make):
    m = make()
    s = m.to_scip()
    s.hideOutput()
    s.optimize()
    assert abs(float(m.report_bound(hull.dw_bound(m, m.families))) - s.getObjVal()) < 1e-6


def _write(m, tmp_path):
    path = str(tmp_path / f"{m.name}.lp")
    m.to_scip().writeProblem(path, trans=False, verbose=False)
    return path


def test_lookahead_is_a_lower_bound_and_can_miss_a_joint_group(tmp_path):
    from llmdw.lookahead import lookahead

    for m in (templates.uls(), templates.clsp()):
        exact = hull.gain_table(m)
        la = lookahead.lookahead_table(_write(m, tmp_path), block_time=30)
        for row in la["table"].values():
            assert row["lookahead"] <= float(exact[frozenset(row["families"])]) + 1e-6
    la = lookahead.lookahead_table(_write(templates.uls(), tmp_path), block_time=30)
    assert abs(la["table"]["Balance_#|SetupLink_#"]["lookahead"] - 14 / 3) < 1e-6
    la = lookahead.lookahead_table(_write(templates.clsp(), tmp_path), block_time=30)
    assert la["table"]["Capacity_#|SetupLink_#_#"]["lookahead"] < 1e-6          # misses a group of gain 51/10


def test_search_simulation_matches_needle_structure():
    from llmdw.search import search_sim

    m = templates.needle(k=8, hidden=(2, 5), cycle_len=5)
    tab = {T: float(v) for T, v in hull.gain_table(m, skip=lambda T: len(T) > 3).items()}
    W = m.facts["hidden"]
    assert search_sim.greedy(tab) is None                       # zero marginal gains everywhere
    assert search_sim.anchored(tab, 3, W) == 1
    assert search_sim.anchored(tab, 3, [W[0], "ConflictA_#"]) <= 6     # e = 1: at most C(6,1)/C(1,1)
    assert search_sim.covering(tab, 3) <= len(search_sim.greedy_covering(sorted(set().union(*tab)), 3, 2)) + 8


def test_structure_census_flags_censored_zeros():
    from llmdw.scoring import structure

    o = {"sense": "minimize", "families": ["A", "B", "C"], "lp_bound": 0.0, "reference_optimal_value": 1.0,
         "subsets": [{"ok": True, "dual_bound": 0.0, "status": "nodelimit", "master_families": ["B", "C"]},
                     {"ok": True, "dual_bound": 0.0, "status": "timelimit", "master_families": ["A", "C"]},
                     {"ok": True, "dual_bound": 0.5, "status": "nodelimit", "master_families": ["C"]}]}
    tab = structure.instance_tables(o)["tables"][()]
    groups = structure.zero_marginal_groups(tab, tol=1e-4)
    assert len(groups) == 1 and groups[0]["group"] == frozenset({"A", "B"}) and groups[0]["status"] == "possible"


def test_full_family_set_without_linking_variables_is_not_a_decomposition(tmp_path):
    """Without linking variables the set of all families (an empty master: the MILP itself) is left out of the
    table, whether a value comes from the oracle or from the lookahead; with linking variables it stays."""
    import json

    from llmdw.scoring import analyze_support, structure

    o = {"sense": "minimize", "families": ["A", "B"], "lp_bound": 0.0, "reference_optimal_value": 1.0,
         "subsets": [{"ok": True, "dual_bound": 0.0, "status": "nodelimit", "master_families": ["B"]},
                     {"ok": True, "dual_bound": 0.0, "status": "nodelimit", "master_families": ["A"]},
                     {"ok": True, "dual_bound": 1.0, "status": "nodelimit", "master_families": []},
                     {"ok": True, "dual_bound": 0.4, "status": "nodelimit", "master_families": [],
                      "linking_var_families": ["x"]}]}
    la = {"families": ["A", "B"], "table": {
        "A": {"families": ["A"], "lookahead": 0.0}, "B": {"families": ["B"], "lookahead": 0.0},
        "A|B": {"families": ["A", "B"], "lookahead": 0.9}}}
    tables = structure.instance_tables(o, la)["tables"]
    assert set(tables[()]) == {frozenset(), frozenset({"A"}), frozenset({"B"})}
    assert frozenset({"A", "B"}) in tables[("x",)]
    assert structure.zero_marginal_groups(tables[()], tol=1e-4) == []
    (tmp_path / "oracle.json").write_text(json.dumps(o))
    (tmp_path / "la.json").write_text(json.dumps(la))
    truth = analyze_support.miplib_truth(tmp_path / "oracle.json", tmp_path / "la.json")
    assert truth["joint"] == [] and frozenset({"A", "B"}) not in truth["table"]
    assert all(len(T) < 2 for T, _ in truth["lookahead"])


def test_network_design_and_facility_location_have_only_joint_gains():
    """Default data of the network-design and facility-location classes: no family strengthens alone, and the joint supports are the ones
    recorded in the generator facts, all polyhedral."""
    from llmdw.instances import export

    for m in (templates.network_design(), templates.facility_location()):
        gt = export.ground_truth(m)
        assert gt["strengthens_alone"] == []
        assert gt["joint_supports"] == m.facts["joint_support"]
        assert set(gt["group_sources"].values()) == {"polyhedral"}


def test_miplib_instance_is_scored_only_with_an_oracle_bound(tmp_path):
    """An instance is scored when at least one family set was evaluated by the oracle with a bound; failed
    evaluations with lookahead values only (graph40-20-1rand in the oracle run) or no values at all are not scored."""
    import json

    from llmdw.scoring.analyze_support import miplib_truth, scorable

    base = {"sense": "minimize", "families": ["A_#", "B_#"], "lp_bound": 10.0, "reference_optimal_value": 12.0}
    good = dict(base, subsets=[{"ok": True, "dual_bound": 11.0, "status": "optimal", "master_families": ["B_#"],
                                "linking_var_families": []}])
    killed = dict(base, subsets=[{"ok": False, "dual_bound": None, "error": "child exit killed",
                                  "master_families": ["B_#"], "linking_var_families": []}])
    look = {"table": {"A_#": {"families": ["A_#"], "lookahead": 0.5}}}
    paths = {}
    for name, o in (("good", good), ("killed", killed), ("empty", dict(base, subsets=[]))):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(o))
    la = tmp_path / "look.json"
    la.write_text(json.dumps(look))
    assert scorable(paths["good"]) and miplib_truth(paths["good"], None) is not None
    assert not scorable(paths["killed"], la) and miplib_truth(paths["killed"], la) is None
    assert not scorable(paths["empty"]) and miplib_truth(paths["empty"], None) is None


def test_results_statistics_and_needle_levels():
    from llmdw.scoring import results

    lo, hi = results.wilson(5, 20)
    assert abs(lo - 0.1119) < 1e-3 and abs(hi - 0.4687) < 1e-3
    lab = lambda xs: [(f"i{k}", x) for k, x in enumerate(xs)]  # noqa: E731   values with instance names
    assert results.boot_ci(lab([1.0, 1.0, 1.0])) == (1.0, 1.0)
    lo, hi = results.boot_ci(lab([0.0, 1.0] * 10))
    assert lo < 0.5 < hi and results.boot_ci(lab([0.0, 1.0] * 10)) == (lo, hi)   # seed 0: reproducible
    summary = {"constraint_families": [{"family": f, "rows": r} for f, r in (("A", 2), ("B", 2), ("C", 3), ("D", 2))]}
    lv = results.needle_levels(summary, frozenset({"A", "B"}))
    assert (lv["chance"], lv["matching_sets"], lv["strict_chance"]) == (1 / 6, 3, 1 / 3)


def test_results_paired_configurations():
    from llmdw.scoring import results

    instances = {n: {"truth": {"joint": [frozenset({"A", "B"})]}, "class": "clsp"} for n in ("x", "y", "z")}
    rows_a = [{"instance": n, "condition": "original", "sample": s, "joint_hit1": True, "joint_recall": 1.0,
               "evals_jointdw": 2} for n in instances for s in range(3)]
    rows_b = [{**r, "joint_hit1": False, "joint_recall": 0.5, "evals_jointdw": 5} for r in rows_a]
    out = {r["measure"]: r for r in results.paired_configs("templates", "cfg_a", "cfg_b", rows_a, rows_b, instances)
           if r["condition"] == "original"}
    assert out["joint_hit1"]["mean_difference"] == 1.0 and out["joint_recall"]["mean_difference"] == 0.5
    assert out["evals_jointdw"]["mean_difference"] == -3 and out["evals_jointdw"]["n_instances"] == 3
    assert (out["joint_recall"]["lo"], out["joint_recall"]["hi"]) == (0.5, 0.5)


def test_results_offline_helpers():
    from llmdw.scoring import results

    A, B, C = frozenset({"A"}), frozenset({"B"}), frozenset({"C"})
    truth = {"families": ["A", "B", "C"],
             "lookahead": [(A | B, 0.5), (A, 0.0), (A | B | C, 9.0), (B | C, 0.5), (frozenset(), 0.0)]}
    assert results.lookahead_order(truth) == [A | B, B | C, A]       # no full set; ties: fewer families, then name
    assert (results.s_value("2", 3), results.s_value("3", 3), results.s_value("k-1", 5), results.s_value("2", 2)) \
        == (2, 2, 4, 1)
    table = {frozenset(): 0.0, A: 0.0, A | B: 1.0, A | B | C: 2.0}
    assert results.restricted(table, 2) == {frozenset(): 0.0, A: 0.0, A | B: 1.0}
    assert results.interleave([A, B, C], [B, A | B]) == [A, B, A | B, C]
    lo, hi = results.boot_stat([("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)], results.np.median)
    assert lo <= 2.5 <= hi


def test_large_model_orders():
    import itertools

    from llmdw.scoring import large_models as lm
    from llmdw.search import search_sim

    fams = [f"F{i}" for i in range(7)]
    design = lm.covering_design(fams, 100)
    pairs = {frozenset(p) for T in design for p in itertools.combinations(sorted(T), 2)}
    assert len(pairs) == 21 and all(len(T) == 3 for T in design)          # every pair covered by sets of 3
    assert lm.covering_design(fams, 4) == design[:4]                      # lazy prefix, fixed by the family order
    assert lm.random_order(fams, 1, 10) == lm.random_order(fams, 1, 10) != lm.random_order(fams, 2, 10)
    P = frozenset({"F1", "F4"})
    assert lm.staged_lazy(fams, 3, P, 5) == search_sim.staged_order(fams, 3, P)[:len(lm.staged_lazy(fams, 3, P, 5))]
    assert len(lm.staged_lazy(fams, 3, P, 5)) >= 5
    assert lm.covering_lazy(fams, 3, 4) == search_sim.covering_order(fams, 3)[:len(lm.covering_lazy(fams, 3, 4))]
    table = {frozenset(c): float(len(c) == 2 and set(c) == {"F1", "F4"}) for r in range(4)
             for c in itertools.combinations(fams, r)}
    order = lm.jointdw_order(fams, P, 10)
    assert search_sim._first_hit(order, table) == search_sim.robust(table, 3, P)   # the same search as robust
    la = {"table": {"a": {"families": ["F1", "F4"], "lookahead": 2.0}, "b": {"families": ["F0"], "lookahead": 2.0},
                    "c": {"families": ["F2"], "lookahead": None}}}
    lo = lm.lookahead_order(fams, la)
    assert lo[:2] == [frozenset({"F0"}), frozenset({"F1", "F4"})] and len(lo) == len(lm.all_sets(fams))
    assert lm.method_family("jointdw|cfg_a|s1") == ("jointdw", "cfg_a") and lm.method_family("random_2") == ("random", "")


def test_large_model_greedy_respects_the_budget_and_stops_without_improvement():
    from llmdw.scoring import large_models as lm

    class Stub:
        fams = order = ["A", "B", "C", "D"]                 # the family order of every method (Instance.order)

        def gain(self, T):
            return {frozenset("A"): 1.0, frozenset("AB"): 2.0}.get(frozenset(T), 0.0)

    order = lm.run_greedy(Stub(), budget=10)
    assert order[:4] == [frozenset(x) for x in "ABCD"]                   # step 1: every single family
    assert order[4:7] == [frozenset("AB"), frozenset("AC"), frozenset("AD")]
    assert len(order) == 9                                               # step 3 (AB + one) improves nothing: stop
    assert len(lm.run_greedy(Stub(), budget=3)) == 3


def test_results_sample_of_record_and_large_model_outcomes():
    from llmdw.scoring import large_models as lm, results

    T = {"reliability": [{"phase": "templates", "config": "cfg_x", "answers": 179, "unparseable": 2,
                          "cells_without_answer": 1, "cells_planned": 180}]}
    times = {("templates", "cfg_x"): {"first_answer": "a", "last_answer": "b"}}
    rows = results.sample_of_record(T, ["cfg_x"], times)
    assert rows[0] == {"config": "cfg_x", "phase": "templates", "answer_files": 179, "first_answer": "a",
                       "last_answer": "b", "unparseable": 2, "no_answer": 1, "cells_planned": 180}
    assert rows[1]["phase"] == "miplib" and rows[1]["answer_files"] is None
    assert lm.outcome({"cmp_bound": None, "status": None}) == "no bound"
    assert lm.outcome({"cmp_bound": 1.0, "status": "nodelimit"}) == "converged"
    assert lm.outcome({"cmp_bound": 1.0, "status": "timelimit"}) == "stopped at the limit with a bound"


def test_results_theorem2_targets_and_anchors():
    from llmdw.scoring import results
    from llmdw.search import search_sim

    A, B, C = frozenset("A"), frozenset("B"), frozenset("C")
    tab = {frozenset(): 0.0, A: 0.0, B: 0.0, C: 0.0, A | B: 0.96, A | C: 0.0, B | C: 0.0, A | B | C: 1.0}
    assert results.targets(tab, "max") == {A | B | C}
    assert results.targets(tab, "min95") == {A | B}                      # A|B|C reaches 0.95 but is not minimal
    order = [A | C, A | B | C, A | B]
    assert results.first_hit_set(order, tab, {A | B})[0] == 3
    assert results.greedy_hit(tab, {A | B | C})[0] == search_sim.greedy(tab)
    rec = {"supports": [{"families": ["A", "B"]}], "candidates": [{"master_families": ["A"]}]}
    assert results.support_anchor(rec, ["A", "B", "C"]) == A | B and results.anchor(rec, ["A", "B", "C"]) == B | C
    assert results.support_anchor({"supports": [], "candidates": []}, ["A"]) == frozenset()


def test_results_paired_search():
    from pathlib import Path

    from llmdw.scoring import results

    fams = ["A", "B", "C", "D"]
    AB = frozenset("AB")
    table = {frozenset(T): (1.0 if AB <= frozenset(T) else 0.0)
             for T in ["", "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD"]}
    truth = {"families": fams, "table": table, "joint": [AB],
             "lookahead": [(T, v) for T, v in table.items()]}
    instances = {"x": {"truth": truth, "class": "clsp"}}
    recs = {"cfg_x": [{"instance": "x", "condition": "original", "sample": 0,
                    "supports": [{"families": ["A", "B"]}], "candidates": []}]}
    rows = results.paired_search("templates", recs, instances, Path("."))
    r = {(x["limit"], x["contrast"]): x for x in rows}
    assert len(rows) == 3 * 4 and all(x["unit"] == "evaluations" for x in rows)
    assert r[("2", "jointdw minus covering")]["mean_a"] == 1.0            # the anchor itself is the first query
    assert r[("2", "jointdw minus covering")]["mean_difference"] == 1.0 - r[("2", "jointdw minus covering")]["mean_b"]


def test_results_shortlist_anchor():
    """Shortlists from the listed joint supports, the fallback to the current anchor, the shortlist order, and the
    section 10 rows: size, coverage of the best set, evaluations and paired differences."""
    from llmdw.scoring import results
    from llmdw.search import search_sim

    fams = ["A", "B", "C", "D"]
    AB = frozenset("AB")
    table = {frozenset(T): (1.0 if AB <= frozenset(T) else 0.0)
             for T in ["", "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD"]}
    truth = {"families": fams, "table": table, "joint": [AB], "lookahead": [(T, v) for T, v in table.items()]}
    instances = {"x": {"truth": truth, "class": "clsp"}}
    listed = {"instance": "x", "condition": "original", "sample": 0, "candidates": [],
              "supports": [{"families": ["A", "C"]}, {"families": ["B", "D"]}, {"families": ["C", "D"]}]}
    unlisted = {"instance": "x", "condition": "original", "sample": 1, "supports": [],
                "candidates": [{"master_families": ["C", "D"]}]}
    sl = results.shortlists(listed, fams)
    assert sl == {"S1": frozenset("AC"), "S2": frozenset("ABCD"), "S3": frozenset("ABCD"), "Sall": frozenset("ABCD")}
    assert set(results.shortlists(unlisted, fams).values()) == {AB}       # no support listed: the current anchor
    order = results.shortlist_order(frozenset("AC"), fams, 2)
    cover = search_sim.covering_order(fams, 2)
    assert order[0] == frozenset("AC") and len(order) == len(set(order)) == len(set(cover) | {frozenset("AC")})
    assert results.shortlist_order(frozenset(), fams, 2) == list(dict.fromkeys(cover))   # empty shortlist: covering
    from llmdw.search import relabel
    relabel.configure(identity=True)                      # the orders by name, which the checks below spell out
    try:
        rows, paired = results.shortlist_search("templates", {"cfg_x": [listed, unlisted]}, instances)
    finally:
        relabel.configure(1000)
    r = {(x["limit"], x["method"]): x for x in rows if x["condition"] == "original"}
    s1 = r[("2", "S1")]
    first_listed = order.index(AB) + 1                    # AB is the only set of maximal gain with at most 2 families
    assert s1["mean_size"] == 2 and s1["best_inside_share"] == 0.5 and s1["samples"] == 2
    assert s1["mean"] == (first_listed + 1) / 2           # the fallback anchor AB is evaluated first
    assert r[("2", "S2")]["best_inside_share"] == 1.0 and r[("2", "Sall")]["mean_size"] == 3.0
    assert r[("2", "S1+lookahead")]["mean"] <= s1["mean"] and r[("2", "lookahead")]["mean"] == 1.0
    p = {(x["limit"], x["contrast"]): x for x in paired if x["condition"] == "original"}
    d = p[("2", "S1 minus jointdw")]
    assert d["n_instances"] == 1 and d["mean_difference"] == s1["mean"] - r[("2", "jointdw")]["mean"]
    assert p[("2", "S1 minus lookahead")]["mean_difference"] == s1["mean"] - 1.0


def test_greedy_covering_shortcut_for_w_equal_s_matches_the_greedy():
    import itertools

    from llmdw.search import search_sim

    for k, s in ((5, 2), (6, 3), (4, 4)):
        ground = [f"F{i}" for i in range(k)]
        uncovered = {frozenset(c) for c in itertools.combinations(ground, s)}
        greedy = []                                  # the plain greedy of _greedy_covering, for w = s
        while uncovered:
            best = max((frozenset(B) for B in itertools.combinations(ground, s)),
                       key=lambda B: (sum(1 for c in uncovered if c <= B), -list(itertools.combinations(ground, s)).index(tuple(sorted(B, key=ground.index)))))
            greedy.append(best)
            uncovered = {c for c in uncovered if not c <= best}
        assert search_sim.greedy_covering(ground, s, s) == greedy


def test_results_census_figure_corrections(tmp_path):
    import json

    from llmdw.scoring import results

    o = {"sense": "minimize", "families": ["A", "B", "C"], "lp_bound": 0.0, "reference_optimal_value": 2.0,
         "subsets": [{"ok": True, "key": "B|C", "dual_bound": 1.0, "status": "nodelimit", "master_families": ["B", "C"]},
                     {"ok": True, "key": "C", "dual_bound": 3.0, "status": "optimal", "master_families": ["C"]}]}
    for n, opt in (("x", 2.0), ("y", None)):
        d = tmp_path / "work/oracle" / n
        d.mkdir(parents=True)
        (d / "oracle.json").write_text(json.dumps({**o, "reference_optimal_value": opt}))
    (tmp_path / "work/structure").mkdir(parents=True)
    (tmp_path / "work/structure/structure_rows.json").write_text("[]")
    rows, left = results.census_figure(tmp_path, {})
    assert rows[0]["max_set_gain"] == 1.5 and left == [{"instance": "y", "reason": "no reference optimum"}]
    rows, _ = results.census_figure(tmp_path, {"drop_bounds": {"x": ["C"]}, "optimum": {"x": 4.0}})
    assert rows[0]["max_set_gain"] == 0.25 and "1 invalid bound(s) dropped" in rows[0]["corrections"]


def test_rich_generators_and_draws():
    import random

    from llmdw.instances import rich_models_build as rb
    from llmdw.instances.families import family_of
    from llmdw.instances import templates

    expect = {templates.rich_lot_sizing: (9, 6), templates.rich_network_design: (7, 7),
              templates.rich_facility_location: (7, 3)}
    for fn, (k, nb) in expect.items():
        m = fn()
        assert len(m.families) == k and sum(v.integer for v in m.vars) == nb
        assert all(family_of(r.name) == r.family for r in m.rows) and set(m.family_text) == set(m.families)
        assert not m.maximize and all(v.lb is not None and v.ub is not None for v in m.vars)   # hull needs bounds
    for c in rb.CLASSES:                       # reproducible streams; data within the class's binary limit
        a, b = random.Random(7), random.Random(7)
        pa, pb = rb.draw_params(c, a), rb.draw_params(c, b)
        assert pa == pb
        m = rb.build_milp(c, pa, "t")
        assert sum(v.integer for v in m.vars) <= 14


def _rich_stub(cls, idx, params):                      # module-level, so that the process pool can pickle it
    import os
    from pathlib import Path

    marks = Path(os.environ["RICH_TEST_MARKS"])
    (marks / f"{cls}_{idx}").write_text("x")
    return {"class": cls, "candidate": idx, "name": f"x_{cls}_{idx:02d}", "params": params,
            "accepted": idx % 2 == 0, "reason": None if idx % 2 == 0 else "stub"}


def test_rich_build_resumes_from_stored_candidates(monkeypatch, tmp_path):
    from llmdw.instances import rich_models_build as rb

    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv("RICH_TEST_MARKS", str(marks))
    monkeypatch.setattr(rb, "PER_CLASS", 2)
    monkeypatch.setattr(rb, "safe_evaluate", _rich_stub)
    monkeypatch.setattr(rb, "write_instance", lambda res, root: None)
    first = rb.build(tmp_path / "rich", parallel=3)
    assert first["accepted"] == {c: 2 for c in rb.CLASSES}
    n_calls = len(list(marks.iterdir()))
    assert len(list((tmp_path / "rich" / "candidates").glob("*.json"))) == n_calls
    for f in marks.iterdir():
        f.unlink()
    assert rb.build(tmp_path / "rich", parallel=3)["accepted"] == first["accepted"] and not list(marks.iterdir())   # nothing again
    assert (tmp_path / "rich" / "instances.txt").read_text().split() == \
        [f"x_{c}_{i:02d}" for c in rb.CLASSES for i in (0, 2)]


def test_bootstrap_standard_error_of_a_known_sample():
    import numpy as np

    from llmdw.scoring import large_models as lm, results

    x = [0.0, 1.0] * 50                                    # mean 0.5; se of the mean = sd(x) / sqrt(n) = 0.0503
    xl = [(f"i{k}", v) for k, v in enumerate(x)]           # one instance per value (singleton prompt groups)
    lo, hi, se = results.boot_ci_se(xl)
    assert abs(se - np.std(x, ddof=1) / np.sqrt(len(x))) < 0.002 and lo < 0.5 < hi
    assert (lo, hi) == results.boot_ci(xl)                 # the same replicates give the interval and the se
    reps = np.asarray(x)[np.random.default_rng(0).integers(0, len(x), size=(10_000, len(x)))].mean(axis=1)
    assert se == float(np.std(reps, ddof=1))
    assert results.boot_ci_se([("a", 2.0), ("b", 2.0), ("c", 2.0)]) == (2.0, 2.0, 0.0)
    assert lm.boot(xl) == (lo, hi, se) and results.boot_stat_se(xl, np.median)[:2] == results.boot_stat(xl, np.median)


def test_results_templates_table_rows(tmp_path):
    """templates_table: one row per joint support with the LP gap and the share of it the group closes (exact,
    rounded half up), the objective sense taken from the model, and group "none" for a template without one."""
    from llmdw.scoring import results
    from llmdw.instances import export

    root = tmp_path / "work" / "templates"
    for name, make in export.DEFAULT:
        if name in ("clsp_2x2", "needle_k4", "gap_2x3"):
            export.write_instance(make(), root)
    rows = [{k: str(v) for k, v in r.items()} for r in results.templates_table(tmp_path)]
    assert [(r["instance"], r["k"], r["lp_gap_percent"], r["group"], r["gap_closed_percent"]) for r in rows] == [
        ("clsp_2x2", "3", "32.4", "BS", "63"), ("clsp_2x2", "3", "32.4", "CS", "63"),
        ("gap_2x3", "2", "60.0", "none", ""), ("needle_k4", "4", "8.3", "BC", "100")]
    assert [r["class"] for r in rows] == ["Lot sizing", "Lot sizing", "Generalized assignment", "Hidden group"]


def test_results_search_invariant_to_family_names(tmp_path):
    """Renaming the families of templates (so that their order by name is reversed) leaves every search table
    unchanged: the orders are permutations of the families' order in the model file (llmdw.search.relabel)."""
    import json

    from llmdw.search import relabel
    from llmdw.scoring import results
    from llmdw.instances import export

    root = tmp_path / "work" / "templates"
    for name, make in export.DEFAULT:
        if name in ("clsp_2x2", "clsp_3x2", "needle_k4", "cfl"):
            export.write_instance(make(), root)
    relabel.configure(50)
    try:
        def answers(instances, rename):
            recs = []
            for n, i in instances.items():
                fams = sorted(i["truth"]["families"])
                guesses = [sorted(i["truth"]["joint"][0]) if i["truth"]["joint"] else fams[:2], fams[-2:], []]
                for cond in results.CONDITIONS:
                    for s, g in enumerate(guesses):
                        recs.append({"instance": n, "condition": cond, "sample": s, "status": "answered",
                                     "supports": [{"families": [rename(f) for f in g]}] if g else [],
                                     "candidates": [{"master_families": [rename(f) for f in fams if f not in g]}],
                                     "family_info": {}})
            return recs

        def tables(instances, recs):
            relabel.clear()                      # the cache is keyed by instance name: no reuse across labelings
            recs_by = {"mx": recs}
            rows_by = {"mx": results.score_all(recs, instances)}
            out = {"base": results.baseline_search("templates", instances)[0],
                   "points": results.jointdw_search("templates", "mx", rows_by["mx"], instances)[2],
                   "size": results.search_by_size("templates", recs_by, instances),
                   "paired": results.paired_search("templates", recs_by, instances, tmp_path),
                   "theorem2": results.theorem2("templates", recs_by, instances, tmp_path)[1:],
                   "shortlist": results.shortlist_search("templates", recs_by, instances)}
            return json.dumps(out, sort_keys=True, default=str)

        inst = results.load_instances(tmp_path, "templates")
        plain = tables(inst, answers(inst, lambda f: f))
        renamed = {}
        for n, i in inst.items():
            order = sorted(i["truth"]["families"])
            new = {f: f"F{len(order) - j:02d}_#" for j, f in enumerate(order)}      # reverses the order by name
            rn = lambda T: frozenset(new[f] for f in T)  # noqa: E731
            t = i["truth"]
            truth = {**t, "families": [new[f] for f in t["families"]], "family_order": [new[f] for f in t["family_order"]],
                     "joint": [rn(W) for W in t["joint"]], "alone": {new[f] for f in t["alone"]},
                     "table": {rn(T): v for T, v in t["table"].items()}}
            renamed[n] = {**i, "truth": truth, "_map": new}
        recs = []
        for r in answers(inst, lambda f: f):
            new = renamed[r["instance"]]["_map"]
            recs.append({**r, "supports": [{"families": [new[f] for f in x["families"]]} for x in r["supports"]],
                         "candidates": [{"master_families": [new[f] for f in x["master_families"]]}
                                        for x in r["candidates"]]})
        assert tables(renamed, recs) == plain
    finally:
        relabel.configure(1000)


def test_lookahead_values_within_tolerance_are_tied():
    from llmdw.search import relabel
    cls = relabel.tie_classes([1.0, 1.0 - 5e-10, 0.5, 3e-15, 0.0, -2e-15])
    assert cls[1.0] == cls[1.0 - 5e-10] == 0 and cls[0.5] == 1 and cls[3e-15] == cls[0.0] == cls[-2e-15] == 2
    assert relabel.tie_classes([0.0, 2e-9]) == {2e-9: 0, 0.0: 1}
    # values equal up to rounding noise order by size, then by position, not by the noise
    la = [(frozenset({"a", "b"}), 3e-15), (frozenset({"c", "d"}), 0.0), (frozenset({"b"}), -1e-15)]
    assert relabel.lookahead_list(la, ["d", "c", "b", "a"], 2) == [frozenset({"b"}), frozenset({"c", "d"}),
                                                                   frozenset({"a", "b"})]


def test_ranked_list_and_narrow_lookahead_orders():
    from llmdw.search import relabel
    from llmdw.scoring import results
    fams = ["a", "b", "c", "d"]
    fs = frozenset
    # ranked-list order: list sets with at most s families in list order, then staged around the first, then covering
    order = list(relabel.ranked_iter(fams, 2, [fs("cd"), fs("abc"), fs("a"), fs("cd")]))
    assert order[:2] == [fs("cd"), fs("a")] and len(order) == len(set(order))
    assert set(order) >= {fs(x) for x in ("ab", "ac", "ad", "bc", "bd", "cd")}
    assert list(relabel.ranked_iter(fams, 2, [])) == list(relabel.dedupe(relabel.covering_iter(fams, 2)))
    # narrow then lookahead: sets inside C in lookahead order, then the others in lookahead order
    la = [(fs("a"), 3.0), (fs("ab"), 5.0), (fs("cd"), 4.0), (fs("c"), 1.0), (fs("d"), 1.0 + 1e-12)]
    assert list(relabel.narrow_iter(la, fams, 2, fs("cd"))) == [fs("cd"), fs("c"), fs("d"), fs("ab"), fs("a")]
    assert list(relabel.narrow_iter(la, fams, 2, fs())) == relabel.lookahead_list(la, fams, 2)
    # L_cand and L_vote
    rec = {"candidates": [{"master_families": ["a", "b"]}, {"master_families": ["a"]}, {"master_families": ["b", "a"]}]}
    assert results.cand_list(rec, fams) == [fs("cd"), fs("bcd")]
    cell = [{"supports": [{"families": ["a", "b"]}, {"families": ["c"]}]}, {"supports": [{"families": ["c"]}]},
            {"supports": [{"families": ["b", "a"]}, {"families": ["d"]}]}]
    assert results.vote_list(cell, fams) == [fs("ab"), fs("c"), fs("d")]    # 2 votes each; mean position 0 before 0.5
