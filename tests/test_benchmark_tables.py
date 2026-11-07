"""Experiment C tables (llmdw.benchmark.tables)."""
import math

from llmdw.benchmark import tables
from llmdw.scoring import cluster


def _v(cls, methods):
    return {"class": cls, "classification": "certain", "zero_marginal_group": True, "methods": methods}


def _one(set_, hit, gain, time, gap, solved=True, failed=False):
    return [{"sample": None, "set": set_, "hit": hit, "gain": gain, "time": time, "gap": gap, "solved": solved,
             "failed_twice": failed}]


def test_language_model_values_are_sample_means():
    rows = [{"sample": 0, "set": "A|B", "hit": True, "gain": 1.0, "time": 2.0, "gap": 0.0, "solved": True},
            {"sample": 1, "set": None, "hit": False, "gain": 0.0},
            {"sample": 2, "set": "A", "hit": False, "gain": 0.0, "time": 4.0, "gap": 0.1, "solved": False,
             "failed_twice": True}]
    p = tables.per_instance({"methods": {"m": rows}}, "m")
    assert p["hit"] == 1 / 3 and p["gain"] == 1 / 3 and p["gain_zero"] == 2 / 3
    assert p["time"] == 3.0 and p["gap"] == 0.05 and p["solved"] == 0.5
    assert p["failed_twice"] and p["no_set"] == 1


def test_time_ratio_is_a_shifted_geometric_mean():
    cluster.set_groups({})
    r = tables._ratio({"a": 3.0, "b": 0.0}, {"a": 1.0, "b": 1.0})
    assert math.isclose(r["geometric_mean_ratio"], math.sqrt((4 / 2) * (1 / 2)))
    assert r["n_instances"] == 2


def test_tables_rows_and_failed_twice_listing():
    cluster.set_groups({})
    vals = {}
    for i, cls in enumerate(("lot_sizing", "lot_sizing", "facility_location")):
        ms = {"true best set": _one("A|B", True, 1.0, 1.0, 0.0),
              "lookahead (single dual)": _one("A", False, 0.0, 1800.0, 0.01, False, i == 2),
              "lookahead (five duals)": _one("A|B", True, 1.0, 1.0, 0.0),
              "GCG default": _one("B|C", False, 0.0, 5.0, 0.0),
              "X": _one("A|B", True, 1.0, 2.0, 0.0)}
        vals[f"i{i}"] = _v(cls, ms)
    t = tables.tables(vals, ["X"])
    assert [r["class"] for r in t["zero_marginal"]] == ["lot_sizing", "facility_location", "all"]
    row = next(r for r in t["methods"] if r["class"] == "all" and r["method"] == "lookahead (single dual)")
    assert row["n_failed_twice"] == 1 and row["failed_twice_instances"] == "i2" and row["gain_zero_instances"] == 3
    assert row["solved"] == 0
    pj = [r for r in t["paired_jointdw"] if r["class"] == "all"]
    assert len(pj) == 3 and next(r for r in pj if r["against"] == "GCG default")["mean_difference"] == -3.0
    tb = next(r for r in t["paired_true_best"] if r["class"] == "all" and r["method"] == "X")
    assert math.isclose(tb["geometric_mean_ratio"], 1.5) and tb["gap_mean_difference"] == 0.0
