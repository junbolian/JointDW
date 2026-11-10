"""The time-optimal set within the budget and the one tolerance function (llmdw.benchmark.time_truth), and the misses
counted on the anchor after the cut to s families (MODULE.misses_per_prediction), on synthetic records."""
import itertools
import json

from llmdw.benchmark import time_truth as tt


def _instance(root, name="p"):
    d = root / "instances" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"class": "lot_sizing"}))
    (d / "bnp_plan.json").write_text(json.dumps({"methods": {"true_best": ["a", "b"], "true_best_ties": [["a", "b"]]}}))
    subsets = [{"master_families": sorted({"a", "b", "c"} - set(T)), "dual_bound": g, "ok": True, "status": "optimal"}
               for T, g in ((("a",), 0.2), (("b",), 0.1), (("c",), 0.0), (("a", "b"), 0.9), (("a", "c"), 0.3),
                            (("b", "c"), 0.2))]
    (d / "oracle.json").write_text(json.dumps({"sense": "minimize", "families": ["a", "b", "c"], "lp_bound": 0.0,
                                               "reference_optimal_value": 1.0, "subsets": subsets, "baselines": {}}))


def _runs(key, times_gaps):
    return [{"instance": "p", "set": key, "seed": s, "time": t if t is not None else 1800.0, "solved": t is not None,
             "gap": g, "failed_twice": False} for s, (t, g) in enumerate(times_gaps)]


def test_within_the_budget_every_set_unsolved_the_gap_ranks_and_every_set_is_within_tolerance(tmp_path):
    _instance(tmp_path)
    rows = (_runs("a|b|c", [(1.0, 0.0)] * 3)                         # the fastest, but all three families
            + _runs("a", [(None, 0.05)] * 3) + _runs("b", [(None, 0.02)] * 3) + _runs("c", [(None, 0.30)] * 3)
            + _runs("a|b", [(None, 0.04)] * 3) + _runs("a|c", [(None, 0.10)] * 3) + _runs("b|c", [(None, 0.20)] * 3)
            + _runs("gcg_default", [(5.0, 0.0)] * 3))
    t = tt.time_truth_budget(rows, tmp_path)["p"]
    assert "a|b|c" not in t["order"] and "a|b|c" in t["summary"]       # its own time stays for a method naming it
    assert t["order"] == ["b", "a|b", "a", "a|c", "b|c", "c"]           # every set unsolved: by mean final gap
    assert t["time_optimal"] == "b"
    assert sorted(t["within_tolerance"]) == sorted(t["order"])          # every unsolved set: 1801 <= 1.1 x 1801
    assert t["bound_optimal"] == "a|b" and not t["coincide"]


def test_one_tolerance_function_for_both_rankings(tmp_path):
    _instance(tmp_path)
    rows = (_runs("a|b|c", [(None, 0.01)] * 3) + _runs("a", [(None, 0.05)] * 3) + _runs("b", [(None, 0.02)] * 3)
            + _runs("c", [(None, 0.30)] * 3) + _runs("a|b", [(None, 0.04)] * 3) + _runs("a|c", [(None, 0.10)] * 3)
            + _runs("b|c", [(None, 0.20)] * 3))
    every = tt.time_truth(rows, tmp_path)["p"]                       # every set: the time-optimal decompositions
    assert every["time_optimal"] == "a|b|c" and len(every["within_tolerance"]) == 7
    fam = {k: tt.set_summary(v) for k, v in {k: [r for r in rows if r["set"] == k] for k in every["order"]}.items()}
    assert tt.within_tolerance(fam, every["order"], every["time_optimal"]) == every["within_tolerance"]


def test_ranking_within_the_budget_by_solved_seeds_then_time(tmp_path):
    _instance(tmp_path)
    rows = (_runs("a|b|c", [(1.0, 0.0)] * 3) + _runs("a", [(10.0, 0.0), (12.0, 0.0), (None, 0.1)])
            + _runs("b", [(30.0, 0.0)] * 3) + _runs("a|b", [(31.0, 0.0)] * 3) + _runs("c", [(None, 0.2)] * 3)
            + _runs("a|c", [(None, 0.3)] * 3) + _runs("b|c", [(None, 0.4)] * 3))
    t = tt.time_truth_budget(rows, tmp_path)["p"]
    assert t["order"][:3] == ["b", "a|b", "a"]                         # three solved seeds before two
    assert t["within_tolerance"] == ["b", "a|b"]                        # 32 <= 1.1 x 31


def test_anchor_cut_to_s_families_and_its_misses():
    from llmdw.scoring import rich_models_large as ee
    from llmdw.search import relabel

    fams = ["a", "b", "c", "d"]
    table = {frozenset(c): (1.0 if set(c) >= {"a", "b"} else 0.0) for r in range(1, 4)
             for c in itertools.combinations(fams, r)}
    table[frozenset()] = 0.0
    inst = {"name": "synthetic_cut", "class": "lot_sizing", "k": 4, "sizes": [2],
            "truth": {"families": fams, "table": table, "joint": [frozenset({"a", "b"})]}}
    rec = {"instance": "synthetic_cut", "sample": 0, "supports": [{"families": ["a", "c", "d"]}], "candidates": []}
    relabel.configure(permutations=24)
    try:
        (r,) = ee.misses_per_prediction("synthetic", [inst], {("gpt-6.1-sol", "original"): [rec]})
    finally:
        relabel.configure(permutations=1000)
    assert r["case"] == "anchor cut to s" and r["anchor_size"] == 3 and r["minimal_best_sets"] == "a|b"
    assert r["e_before_cut"] == 1                                       # {a, b} minus {a, c, d}
    dist = dict(x.split(":") for x in r["e_after_cut_distribution"].split())
    assert sum(int(v) for v in dist.values()) == 24 and set(dist) <= {"1", "2"}   # e = 1 when a is kept, else 2
    assert r["check2_cost_at_most_stage_queries"] and r["check3_jointdw_at_most_twice_covering_entries"]


def test_tables_of_the_misses_in_the_layout_of_the_paper():
    from llmdw.tables import paper_v1

    def row(cls, k, s, kind, e="", n="3", **kw):
        r = {"class": cls, "k": k, "s": s, "kind": kind, "e": e, "n_predictions": n, "cost_o1_mean": "", "cost_o1_se": "",
             "cost_jointdw_mean": "2.5", "cost_jointdw_se": "0.1", "stage_queries_0_to_e_min": "",
             "stage_queries_0_to_e_max": "", "mean_evaluations_covering": "", "e_after_cut_mean": "",
             "o1_share_orders_reaching_best_mean": ""}
        r.update(kw)
        return r

    within = "anchor within s, |W| <= |P|"
    rows = [row("network_design", "7", "2", "anchor cut to s", cost_o1_mean="4", cost_o1_se="0.2", e_after_cut_mean="1.5"),
            row("network_design", "7", "2", within, "1", cost_o1_mean="3", cost_o1_se="0", stage_queries_0_to_e_min="11",
                stage_queries_0_to_e_max="11", mean_evaluations_covering="11.04"),
            row("lot_sizing", "12", "2", within, "0", cost_o1_mean="1", cost_o1_se="0", stage_queries_0_to_e_min="1",
                stage_queries_0_to_e_max="1", mean_evaluations_covering="33.66"),
            row("lot_sizing", "12", "3", "anchor within s, |W| > |P|", o1_share_orders_reaching_best_mean="0.934",
                mean_evaluations_covering="106.07"),
            row("lot_sizing", "9", "3", within, "0")]
    assert paper_v1.tex_app_misses(rows).splitlines() == [
        "Lot sizing & 12 & 0 & 3 & 1.00 $\\pm$ 0.00 & 2.50 $\\pm$ 0.10 & 1 & 33.7 \\\\",
        "Network design & 7 & 1 & 3 & 3.00 $\\pm$ 0.00 & 2.50 $\\pm$ 0.10 & 11 & 11.0 \\\\",
        " &  & cut (1.50) & 3 & 4.00 $\\pm$ 0.20 & 2.50 $\\pm$ 0.10 & -- & -- \\\\"]
    assert paper_v1.tex_app_larger(rows).splitlines() == ["Lot sizing & 12 & 3 & 0.93 & 2.50 $\\pm$ 0.10 & 106.1 \\\\"]
