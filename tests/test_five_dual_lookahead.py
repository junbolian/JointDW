"""Five-dual lookahead (llmdw.lookahead.five_duals) and its scoring helpers (llmdw.scoring.experiments)."""
import json

from llmdw.lookahead import five_duals
from llmdw.scoring import experiments


def _recorded(tmp_path):
    rec = {"families": ["A", "B", "C"], "sense": "minimize", "lp_bound": 1.0, "truncated": False,
           "table": {"A": {"families": ["A"], "lookahead": 0.0},
                     "A|B": {"families": ["A", "B"], "lookahead": 2.0},
                     "B|C": {"families": ["B", "C"], "lookahead": None},
                     "A|B|C": {"families": ["A", "B", "C"], "lookahead": 3.0}}}
    p = tmp_path / "lookahead.json"
    p.write_text(json.dumps(rec))
    return p


def _dual(tmp_path, seed, table, lp_ok=True):
    d = {"seed": seed, "truncated": False, "lp_bound_agrees": lp_ok,
         "dual": {"seed": seed, "status": "optimal", "differs_from_highs": True},
         "table": {k: {"lookahead": v, "complete": True, "all_optimal": True} for k, v in table.items()}}
    p = tmp_path / f"dual{seed}.json"
    p.write_text(json.dumps(d))
    return p


def test_best_of_five_is_the_maximum_of_the_values_that_exist(tmp_path):
    rec = _recorded(tmp_path)
    duals = [_dual(tmp_path, 1, {"A": 0.5, "A|B": 1.0, "B|C": None}),
             _dual(tmp_path, 2, {"A": 0.0, "A|B": 4.0, "B|C": 1.5})]
    res = five_duals.combine(str(rec), [str(p) for p in duals], str(tmp_path / "five_duals.json"))
    t = res["table"]
    assert t["A"]["lookahead"] == 0.5 and t["A"]["argmax"] == 1
    assert t["A|B"]["lookahead"] == 4.0 and t["A|B"]["per_dual"] == [2.0, 1.0, 4.0] and t["A|B"]["argmax"] == 2
    assert t["B|C"]["lookahead"] == 1.5 and t["B|C"]["per_dual"] == [None, None, 1.5]
    # a set the further duals did not evaluate keeps the recorded value and counts as missing
    assert t["A|B|C"]["lookahead"] == 3.0 and [x["missing_sets"] for x in res["limits"]] == [1, 1]
    assert res["lp_bound_agrees"] and res["duals"][0]["seed"] == 0 and len(res["duals"]) == 3
    assert json.loads((tmp_path / "five_duals.json").read_text())["table"] == t


def test_lp_bound_disagreement_is_reported(tmp_path):
    rec = _recorded(tmp_path)
    res = five_duals.combine(str(rec), [str(_dual(tmp_path, 1, {"A": 0.0}, lp_ok=False))], str(tmp_path / "o.json"))
    assert not res["lp_bound_agrees"]


def test_five_list_orders_by_value_and_leaves_out_the_full_set(tmp_path):
    rec = _recorded(tmp_path)
    five_duals.combine(str(rec), [str(_dual(tmp_path, 1, {"A": 0.5, "A|B": 1.0, "B|C": 2.5}))], str(tmp_path / "f.json"))
    lst = experiments._five_list(tmp_path / "f.json", 3)
    assert [sorted(T) for T, _ in lst] == [["B", "C"], ["A", "B"], ["A"]]
    assert [v for _, v in lst] == [2.5, 2.0, 0.5]
    full = experiments._five_list(tmp_path / "f.json", 3, full_k=True)
    assert sorted(full[0][0]) == ["A", "B", "C"]


def test_paired_difference_uses_instances_with_both_values():
    a = {"i1": 1.0, "i2": 3.0, "i3": None, "i4": 2.0}
    b = {"i1": 0.0, "i2": 1.0, "i3": 5.0}
    r = experiments._paired(a, b)
    assert r["n_instances"] == 2 and r["mean_a"] == 2.0 and r["mean_b"] == 0.5 and r["mean_difference"] == 1.5
    assert r["asterisk"] in ("", "*") and r["lo"] <= r["mean_difference"] <= r["hi"]


def test_mean_ignores_missing_values():
    assert experiments._mean([1.0, None, 3.0]) == 2.0 and experiments._mean([None]) is None
