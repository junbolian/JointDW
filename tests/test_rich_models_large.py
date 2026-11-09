import csv
import itertools
import json

import pytest

from llmdw.scoring import rich_models_large as ee
from llmdw.search import relabel, search_sim

FAMS = ["A", "B", "C", "D"]


@pytest.mark.parametrize("k,s,w", [(7, 2, 2), (9, 3, 2), (9, 3, 3), (12, 3, 2), (12, 3, 3), (12, 2, 1), (16, 3, 3)])
def test_stage_counts_follow_the_staged_order(k, s, w):
    counts = ee.stage_counts(k, s, w)
    order = search_sim.staged_order(list(range(k)), s, list(range(w)))
    assert len(counts) == w + 1 and sum(counts) == len(order)
    assert counts[0] == 1                                          # stage 0 keeps the whole predicted set
    lazy = list(relabel.staged_iter(list(range(k)), s, list(range(w))))
    assert lazy == order
    assert ee.staged_queries(k, s, w, w) == len(set(order))
    assert [ee.staged_queries(k, s, w, e) for e in range(w + 1)] == sorted(ee.staged_queries(k, s, w, e)
                                                                           for e in range(w + 1))


def test_misses_take_the_closest_best_set_and_its_smallest_size():
    goal = {frozenset("AB"), frozenset("ABC"), frozenset("CD")}
    assert ee.misses(frozenset("AB"), goal) == (0, 2)
    assert ee.misses(frozenset("A"), goal) == (1, 2)
    assert ee.misses(frozenset("AC"), goal) == (1, 2)               # AB, ABC and CD each miss one: smallest size 2
    assert ee.misses(frozenset("E"), goal) == (2, 2)


def _gains(complete_to):
    out = {}
    for r in range(complete_to + 1):
        for T in itertools.combinations(FAMS, r):
            out["|".join(T)] = "1" if {"A", "B"} <= set(T) else ("1/4" if "C" in T else "0")
    return out


def _instance(root, name, complete_to, gcg=None):
    d = root / "instances" / name
    d.mkdir(parents=True)
    gains = _gains(complete_to)
    (d / "ground_truth.json").write_text(json.dumps({
        "families": FAMS, "k": 4, "lp_bound": "1", "ip_scip": 3.0, "lp_gap": 0.5, "table_complete_to": complete_to,
        "gain": gains, "joint_supports": [["A", "B"]], "strengthens_alone": ["C"], "best_sets": []}))
    la = {k: {"families": [x for x in k.split("|") if x], "lookahead": (0.5 if k == "C" else 0.0)}
          for k in gains if k}
    (d / "lookahead.json").write_text(json.dumps({"table": la}))
    if gcg:
        (d / "gcg_anchors.json").write_text(json.dumps({"gcg_spfwh": {"convexified": gcg},
                                                        "gcg_strong": {"convexified": None}}))
    five = root.parent / "five" / name
    five.mkdir(parents=True)
    f5 = {k: {**v, "lookahead": (0.9 if k == "A|B" else v["lookahead"])} for k, v in la.items()}
    (five / "five_duals.json").write_text(json.dumps({"table": f5}))
    s = root / "work/model_summaries/rich_models_large"
    s.mkdir(parents=True, exist_ok=True)
    (s / f"{name}.json").write_text(json.dumps({
        "sense": "minimize", "objective_integral": True, "summary": {"constraint_families": FAMS},
        "family_sequence": FAMS, "family_columns": {}, "family_columns_nonzero": {}}))


def _answer(root, cfg, name, sample, families):
    d = root / cfg / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("original.json" if sample == 0 else f"original.s{sample}.json")
    f.write_text(json.dumps({"instance": name, "condition": "original", "sample": sample, "answer": {},
                             "supports": [{"families": families}], "candidates": [], "family_info": {}}))


def _rows(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def work(tmp_path, monkeypatch):
    w = tmp_path / "w"
    root = w / ee.E_ROOT
    a, b = "rich4_lot_sizing_00", "rich4_network_design_01"
    _instance(root, a, 3, gcg=["A", "C", "D"])
    _instance(root, b, 2)
    (root / "instances.txt").write_text(f"{a}\n{b}\n")
    (root / "draws.json").write_text(json.dumps({"draws": [{"name": a, "selected": True, "rows": 5},
                                                           {"name": b, "selected": True, "rows": 6}]}))
    (w / ee.E_FIVE).parent.mkdir(parents=True, exist_ok=True)
    (root.parent / "five").rename(w / ee.E_FIVE)
    ans = w / ee.E_ANSWERS["original"]
    _answer(ans, "cfg-x", a, 0, ["A", "B"])
    _answer(ans, "cfg-x", a, 1, ["A", "C"])
    _answer(ans, "cfg-x", b, 0, ["D"])
    monkeypatch.setattr(ee, "prompt_groups", lambda insts: {})
    return w


def test_end_to_end_marks_sizes_above_the_table_as_not_computed(work, tmp_path):
    out = tmp_path / "out"
    assert ee.main(["--work", str(work), "--out", str(out), "--configs", "cfg-x", "--family-orders", "5"]) == 0
    srch = _rows(out / "search_costs.csv")
    nd3 = [r for r in srch if r["class"] == "network_design" and r["s"] == "3"]
    assert nd3 and all(r["status"] == "not computed" and r["n_instances"] == "0" for r in nd3)
    pooled3 = next(r for r in srch
                   if r["class"] == "all" and r["k"] == "4" and r["s"] == "3" and r["order"] == "covering")
    assert pooled3["status"] == "computed" and pooled3["n_not_computed"] == "1" and pooled3["n_instances"] == "1"
    assert any(r["order"] == "GCG anchor (SPFWH score)" for r in srch)
    assert not any(r["order"] == "GCG anchor (strong score)" for r in srch)      # no representable decomposition
    pred = _rows(out / "predictions.csv")
    ls = next(r for r in pred if r["class"] == "lot_sizing" and r["k"] == "4")
    # the planned cell without a file (sample 2) is a failed answer: one hit of three samples
    assert float(ls["hit1_mean"]) == pytest.approx(1 / 3, abs=1e-6)
    assert float(ls["recall_mean"]) == pytest.approx(1 / 3, abs=1e-6)
    st = {r["presentation"]: r for r in _rows(out / "answers.csv")}
    assert st["original"]["answer_files"] == "3" and st["original"]["missing"] == "3"
    assert st["data"]["answer_files"] == "0"
    gain = _rows(out / "gain_tables" / "rich4_network_design_01.csv")
    assert {r["best_s3"] for r in gain} == {"not computed"} and any(r["best_s2"] == "True" for r in gain)
    fig = _rows(out / "figures/data/experiment_E_cost_vs_k.csv")
    assert {r["status"] for r in fig if r["class"] == "network_design" and r["s"] == "3"} == {"not computed"}
    mis = _rows(out / "figures/data/search_cost_vs_misses.csv")
    row = next(r for r in mis if r["class"] == "lot_sizing" and r["s"] == "2" and r["e"] == "0")
    assert row["n_predictions"] == "1" and float(row["mean_evaluations"]) == 1.0
    assert any(r["class"] == "network_design" and r["s"] == "3" and r["e"] == "not computed" for r in mis)
    assert not (out / "PARTIAL.txt").exists()
    for t in ("instances", "predictions", "search", "paired"):
        assert (out / "tables/new" / f"experiment_E_{t}.tex").read_text().startswith("% Experiment E")


def test_answered_only_leaves_missing_cells_out(work, tmp_path):
    out = tmp_path / "out"
    ee.main(["--work", str(work), "--out", str(out), "--configs", "cfg-x", "--family-orders", "5", "--answered-only"])
    assert (out / "PARTIAL.txt").exists()
    st = {r["presentation"]: r for r in _rows(out / "answers.csv")}
    assert st["original"]["scored"] == "3"
    pred = _rows(out / "predictions.csv")
    nd = next(r for r in pred if r["class"] == "network_design")
    assert nd["n_answers"] == "1"
    ls = next(r for r in pred if r["class"] == "lot_sizing")
    assert float(ls["hit1_mean"]) == 0.5


def test_groups_follow_the_presentation():
    from llmdw.scoring import cluster, rich_models_large as ee

    ee.set_groups_of_set({"a": "a", "b": "a", "c": "c"})
    vals = {"a": 1.0, "b": 3.0, "c": 2.0}
    with ee.groups_for("original"):
        assert cluster.n_groups(vals) == 2
    with ee.groups_for("data"):
        assert cluster.n_groups(vals) == 3
    assert cluster.n_groups(vals) == 2
    ee.set_groups_of_set({})
