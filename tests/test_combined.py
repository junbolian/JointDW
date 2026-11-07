"""Combined orders (llmdw.scoring.combined)."""
from collections import Counter

import pytest

from llmdw.scoring import combined, learned
from llmdw.search import relabel


def test_size_spec():
    assert combined.size_spec(5, "2") == (2, False)
    assert combined.size_spec(3, "3") == (2, False)
    assert combined.size_spec(6, "k-1") == (5, True)


def _items():
    items = {}
    for c in learned.RICH_CLASSES:
        for i in range(8):
            items[f"rich_{c}_{i:02d}"] = {"set": "rich_models", "class": c}
    for t, c in (("clsp_a", "clsp"), ("ndp_a", "ndp"), ("cfl_a", "cfl"), ("gap_a", "gap"), ("needle_a", "needle")):
        items[t] = {"set": "templates", "class": c}
    for i in range(3):
        items[f"mip{i}"] = {"set": "miplib", "class": "miplib"}
    return items


def test_third_source_holds_every_instance_out_once():
    items = _items()
    seen = Counter()
    for test, train in combined.third_source_folds(items):
        assert not set(test) & set(train)
        seen.update(test)
    assert set(seen) == set(items) and set(seen.values()) == {1}
    gap = next(tr for te, tr in combined.third_source_folds(items) if te == ["gap_a"])
    assert set(gap) == {n for n, i in items.items() if i["set"] in ("templates", "rich_models")} - {"gap_a"}
    mip = next(tr for te, tr in combined.third_source_folds(items) if te == ["mip0"])
    assert set(mip) == {"mip1", "mip2"}


def test_stacked_features():
    A, B, C = "A", "B", "C"
    it = {"name": "x", "truth": {"families": [A, B, C]},
          "features": {frozenset({A}): [], frozenset({A, B}): [], frozenset({A, C}): []}}
    recs = [{"instance": "x", "condition": "original", "supports": [{"families": [A, B]}],
             "candidates": [{"master_families": [C]}]},                     # reformulation {A, B}
            {"instance": "x", "condition": "original", "supports": [], "candidates": []},
            {"instance": "x", "condition": "original", "supports": [{"families": [A, C]}],
             "candidates": [{"master_families": [B]}]}]                     # reformulation {A, C}
    f = combined.stacked_features(it, recs)
    assert f[frozenset({A, B})] == [1 / 3, 1 / 3]
    assert f[frozenset({A})] == [0.0, 2 / 3]
    assert f[frozenset({A, C})] == [1 / 3, 1 / 3]


def test_best_single_source_is_the_lowest_cell_mean():
    best, means = combined._best({"a": {"x": 2.0, "y": 4.0}, "b": {"x": 3.0, "y": 2.0}}, ["x", "y"])
    assert best == "b" and means == {"a": 3.0, "b": 2.5}


def _instance(five):
    A, B, C, D = "A", "B", "C", "D"
    fams = [A, B, C, D]
    table = {frozenset(x): 0.0 for x in ({A}, {B}, {C}, {D}, {A, C}, {A, D}, {B, C}, {B, D}, {C, D})}
    table[frozenset({A, B})] = 1.0
    t = {"families": fams, "table": table, "joint": [frozenset({A, B})], "family_order": fams}
    return {"name": "union_test", "truth": t, "five": five}


def test_union_first_evaluates_U_first():
    relabel.clear()
    five = [(frozenset({"A"}), 3.0), (frozenset({"C", "D"}), 2.0), (frozenset({"B"}), 1.0)]
    it = _instance(five)
    rec = {"instance": "union_test", "condition": "original", "supports": [{"families": ["B"]}], "candidates": []}
    # anchor {B} (a support with one family), F5 = {A}: U = {A, B}, |U| = 2 <= s = 2, and U is the target
    v = combined.answer_values(it, [rec], "2")
    assert v["union-first"] == 1.0 and v["share |U|<=s"] == 1.0
    assert v["jointdw interleaved"] >= v["union-first"]


def test_union_first_falls_back_when_U_is_too_large():
    relabel.clear()
    five = [(frozenset({"C", "D"}), 3.0), (frozenset({"A", "B"}), 2.0)]
    it = _instance(five)
    rec = {"instance": "union_test", "condition": "original", "supports": [{"families": ["A", "B"]}],
           "candidates": []}
    v = combined.answer_values(it, [rec], "2")   # U = {A, B, C, D}: 4 > 2
    assert v["share |U|<=s"] == 0.0 and v["union-first"] == v["jointdw interleaved"]


def test_no_k_minus_1_for_rich_models():
    rich = {"set": "rich_models"}
    assert not combined.defined(rich, "k-1") and combined.defined(rich, "3")
    assert combined.defined({"set": "templates"}, "k-1") and combined.defined({"set": "miplib"}, "k-1")


def test_rich_models_left_out_of_training_at_k_minus_1():
    import numpy as np
    rng = np.random.default_rng(2)

    def item(name, st):
        feats = {frozenset({f}): list(rng.random(len(learned.FEATURE_NAMES))) for f in "ABC"}
        feats.update({frozenset(p): list(rng.random(len(learned.FEATURE_NAMES))) for p in ("AB", "AC", "BC")})
        return {"name": name, "set": st, "features": feats,
                "truth": {"families": list("ABC"), "table": {T: float(v[0]) for T, v in feats.items()}}}
    m = combined.fit_size([item("t", "templates"), item("r", "rich_models")], "k-1", "structural", 1)
    assert m.get_booster().num_boosted_rounds() == 100
    with pytest.raises(ValueError):          # nothing to train on: rich models only
        combined.fit_size([item("r", "rich_models")], "k-1", "structural", 1)


def test_instance_without_positive_gain_has_no_target():
    t = {"families": ["A", "B", "C"], "table": {frozenset({"A"}): 0.0, frozenset({"A", "B"}): 1.0}}
    assert combined.has_target({"truth": t, "set": "templates"}, "2")
    z = {"families": ["A", "B", "C"], "table": {frozenset({"A"}): 0.0, frozenset({"B"}): 0.0}}
    assert not combined.has_target({"truth": z, "set": "templates"}, "2")
