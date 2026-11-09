"""Tables of the time-optimal decompositions on fake records (llmdw.benchmark.time_truth.measure_tables): time ratios with their standard
error and asterisk, and the prompt groups of every row. No solver run and no branch-and-price record is read."""
import json
import math

from llmdw.benchmark import time_truth
from llmdw.scoring import cluster


def _instance(root, name, cls, sets):
    d = root / "instances" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"class": cls}))
    plan = {"methods": {"lookahead": sets["lookahead"], "lookahead_five_duals": sets["five"],
                        "gcg_detected": None, "M": [{"sample": s, "set": sets["model"]} for s in range(3)]}}
    (d / "bnp_plan.json").write_text(json.dumps(plan))


def test_measure_tables_on_fake_rows(tmp_path):
    names, truth = [], {}
    for i, (cls, t) in enumerate((("lot_sizing", 10.0), ("lot_sizing", 50.0), ("lot_sizing", 400.0),
                                  ("network_design", 3.0))):
        n = f"x{i}"
        names.append(n)
        _instance(tmp_path, n, cls, {"lookahead": ["A"], "five": ["A", "B"], "model": ["A", "B"]})
        truth[n] = {"time_optimal": "A|B", "within_tolerance": ["A|B"],
                    "summary": {"A|B": {"time_stat": t}, "A": {"time_stat": t * (2 + i)},
                                "gcg_default": {"time_stat": t * 3}}}
    groups = {"x0": "x0", "x1": "x0", "x2": "x0", "x3": "x3"}      # lot sizing: one prompt for three instances
    res = time_truth.measure_tables(tmp_path, names, truth, ["M"], [], groups)
    row = next(r for r in res["methods"] if r["class"] == "lot_sizing" and r["method"] == "lookahead (single dual)")
    assert row["n_instances"] == 3 and row["n_groups"] == 1 and row["n_groups_ratio"] == 1
    assert row["se_log_ratio"] > 0 and math.isclose(row["time_ratio_se"], row["time_ratio_geo"] * row["se_log_ratio"])
    assert row["time_ratio_asterisk"] == "*"                        # every ratio is at least 2
    same = next(r for r in res["methods"] if r["class"] == "lot_sizing" and r["method"] == "M")
    assert same["time_ratio_geo"] == 1.0 and same["time_ratio_se"] == 0.0 and same["time_ratio_asterisk"] == ""
    pooled = next(r for r in res["methods"] if r["class"] == "all" and r["method"] == "lookahead (single dual)")
    assert pooled["n_instances"] == 4 and pooled["n_groups"] == 2
    pr = next(r for r in res["paired"] if r["class"] == "lot_sizing" and r["method"] == "M"
              and r["against"] == "lookahead (single dual)" and r["quantity"] == "log_ratio")
    assert pr["n_groups"] == 1 and pr["se_difference"] > 0 and pr["asterisk"] == "*"
    cluster.set_groups({})


def test_one_pricing_problem_is_parsed():
    from llmdw.benchmark import time_truth as tt

    one = tt.parse_stats("Matrix has 1 blocks, using 1 pricing problem.\n")
    six = tt.parse_stats("Matrix has 6 blocks, using 6 pricing problems.\n")
    assert (one["blocks"], one["pricing_problems"]) == (1, 1)
    assert (six["blocks"], six["pricing_problems"]) == (6, 6)


def test_coefficient_label():
    from llmdw.benchmark import time_truth as tt

    assert tt._label("GPT-6.1 Sol, data") == "GPT-6.1 Sol, coefficients"
    assert tt._label("lookahead (five duals)") == "lookahead (five duals)"
