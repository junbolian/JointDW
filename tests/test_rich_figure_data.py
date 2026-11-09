"""Figure data of the rich models across k (llmdw.tables.rich_figure_data) and the typed reader of the result CSVs
(llmdw.tables.experiments.typed_rows), on fake files."""
import csv

from llmdw.tables import experiments, rich_figure_data as rf


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_cost_rows_one_per_class_k_and_s_with_every_method():
    base = {"set": "rich models, 12 to 16 families", "class": "lot_sizing", "k": "12", "status": "computed",
            "n_instances": "10", "n_groups": "10", "n_failed": "0"}
    rows = [{**base, "order": o, "config": c, "presentation": p, "s": s, "mean_evaluations": "3.5", "se": "0.1"}
            for s in ("2", "3") for o, c, p in [("covering", "", ""), ("JointDW", "GPT-6.1 Sol", "data")]]
    out = rf.cost_by_class_k(rows)
    assert [(r["class"], r["k"], r["s"]) for r in out] == [("lot_sizing", 12, 2), ("lot_sizing", 12, 3)]
    assert out[0]["covering_mean"] == "3.5" and out[0]["jointdw_gpt_6_1_sol_data_n"] == "10"
    assert out[0]["lookahead_five_duals_mean"] == ""                     # not in the input: empty, not invented


def test_hit1_rows_of_both_collections_and_the_control_presentations(tmp_path):
    pred = {"config": "GPT-6.1 Sol", "presentation": "original", "class": "all", "n_instances": "24", "n_groups": "24",
            "hit1": "0.5", "hit1_se": "0.1"}
    _write(tmp_path / "rich_models_prediction.csv", [pred])
    _write(tmp_path / "rich_models_with_data_prediction.csv", [{**pred, "presentation": "data"}])
    _write(tmp_path / "experiment_A" / "control_prediction.csv",
           [{"configuration": "GPT-6.1 Sol", "presentation": p, "class": "pooled", "hit1_mean": "0.4", "hit1_se": "0.1",
             "n_instances": "24", "n_groups": "24"} for p in ("data", "data, permuted", "data, anonymized")])
    large = tmp_path / "large.csv"
    _write(large, [{"config": "GPT-6.1 Sol", "presentation": "data", "class": "all", "k": "all", "n_instances": "60",
                    "n_groups": "60", "hit1_mean": "0.6", "hit1_se": "0.05"}])
    rows = rf.hit1_rows(tmp_path, large)
    pres = sorted((r["set"], r["presentation"]) for r in rows)
    assert ("rich models, 7 to 9 families", "data variant, labels") in pres
    assert ("rich models, 12 to 16 families", "data variant") in pres
    assert sum(r["presentation"] == "data variant" and r["set"].endswith("7 to 9 families") for r in rows) == 1
    assert {r["k"] for r in rows} == {"7 to 9", "12 to 16"}


def test_typed_rows(tmp_path):
    p = tmp_path / "x.csv"
    _write(p, [{"a": "", "b": "True", "c": "3", "d": "0.25", "e": "1e-05", "f": "text", "g": "12"}])
    r = experiments.typed_rows(p)[0]
    assert r == {"a": None, "b": True, "c": 3, "d": 0.25, "e": 1e-05, "f": "text", "g": 12}
