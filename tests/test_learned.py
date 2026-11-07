"""Learned structural predictor (llmdw.scoring.learned)."""
import itertools

import numpy as np
import pytest

from llmdw.instances.families import Model, Row
from llmdw.scoring import learned
from llmdw.search import relabel


def _model():
    # families A (two rows, x1..x4, two blocks), B (one row linking x1, x3), C (one row on y)
    rows = [Row("A_1", -np.inf, 4.0, {"x1": 1.0, "x2": 1.0}),
            Row("A_2", -np.inf, 5.0, {"x3": 1.0, "x4": 1.0}),
            Row("B_1", 1.0, np.inf, {"x1": 1.0, "x3": -1.0}),
            Row("C", -np.inf, 2.0, {"y": 2.0, "x4": 1.0})]
    m = Model(rows=rows, var_type={"x1": "B", "x2": "B", "x3": "C", "x4": "C", "y": "I"},
              var_lb={}, var_ub={}, obj={"x1": 1.0, "y": 3.0}, sense="minimize")
    m.build_var_rows()
    return m


FAM_ROWS = {"A": [0, 1], "B": [2], "C": [3]}


def test_set_features_blocks_border_and_families():
    m = _model()
    f = learned.set_features(m, FAM_ROWS, frozenset({"A"}), 0.5)
    assert f["size"] == 1 and f["n_blocks"] == 2
    assert f["block_rows_mean"] == 1.0 and f["block_cols_max"] == 2
    assert f["border_rows"] == 2 and f["border_rows_share"] == 0.5
    assert f["rhs_block_mean"] == 4.5 and f["rhs_block_max"] == 5.0
    assert f["family_rows_sum"] == 2 and f["family_vars_sum"] == 4 and f["shared_vars"] == 0
    assert f["int_share"] == 0.5 and f["lookahead_rel"] == 0.5
    assert f["tu_score"] == 1.0
    g = learned.set_features(m, FAM_ROWS, frozenset({"A", "B"}), 0.0)
    assert g["n_blocks"] == 1 and g["shared_vars"] == 2 and g["family_rows_min"] == 1
    assert set(f) == set(learned.FEATURE_NAMES)


def test_heller_tompkins():
    ok = [Row("r1", 0, 1, {"a": 1, "b": 1}), Row("r2", 0, 1, {"a": -1, "c": 1})]
    assert learned.heller_tompkins(ok)
    assert not learned.heller_tompkins([Row("r", 0, 1, {"a": 2})])
    three = [Row(f"r{i}", 0, 1, {"a": 1}) for i in range(3)]
    assert not learned.heller_tompkins(three)
    # odd cycle of equal-sign pairs: no valid split
    odd = [Row("r1", 0, 1, {"a": 1, "b": 1}), Row("r2", 0, 1, {"b": 1, "c": 1}), Row("r3", 0, 1, {"c": 1, "a": 1})]
    assert not learned.heller_tompkins(odd)


def test_labels_are_gain_shares():
    t = {"table": {frozenset({"A"}): 1.0, frozenset({"A", "B"}): 4.0, frozenset({"A", "B", "C"}): 8.0}}
    assert learned.labels(t, 2) == {frozenset({"A"}): 0.25, frozenset({"A", "B"}): 1.0}
    z = {"table": {frozenset({"A"}): 0.0, frozenset({"B"}): -1.0}}
    assert set(learned.labels(z, 2).values()) == {0.0}


def test_candidate_sets_exclude_the_full_set():
    sets = learned.candidate_sets(["A", "B", "C"], 3)
    assert frozenset({"A", "B", "C"}) not in sets and len(sets) == 6


def _items():
    items = {}
    for c in learned.RICH_CLASSES:
        for i in range(8):
            items[f"rich_{c}_{i:02d}"] = {"set": "rich_models", "class": c}
    for t, c in (("clsp_a", "clsp"), ("clsp_b", "clsp"), ("ndp_a", "ndp"), ("cfl_a", "cfl"), ("gap_a", "gap"),
                 ("gap_b", "gap")):
        items[t] = {"set": "templates", "class": c}
    for i in range(3):
        items[f"mip{i}"] = {"set": "miplib", "class": "miplib"}
    return items


def test_within_class_leaves_one_out():
    items = _items()
    folds = list(learned.split_folds(items, "within_class"))
    for _, test, train in folds:
        assert len(test) == 1 and test[0] not in train
        assert all(items[n]["class"] == items[test[0]]["class"] and items[n]["set"] == items[test[0]]["set"]
                   for n in train)
    assert sum(1 for _, t, _ in folds if items[t[0]]["set"] == "rich_models") == 24
    assert {t[0] for _, t, _ in folds if items[t[0]]["set"] == "templates"} == {"clsp_a", "clsp_b", "gap_a", "gap_b"}


def test_leave_one_class_out():
    items = _items()
    folds = {lab: (test, train) for lab, test, train in learned.split_folds(items, "leave_one_class_out")}
    test, train = folds["held-out class lot_sizing"]
    assert set(test) == {n for n, i in items.items() if i["class"] in ("lot_sizing", "clsp")}
    assert not set(test) & set(train)
    assert {"mip0", "mip1", "mip2", "ndp_a", "cfl_a"} <= set(train) and "gap_a" not in train


def test_miplibnl_leave_one_out():
    folds = list(learned.split_folds(_items(), "miplibnl_leave_one_out"))
    assert len(folds) == 3 and all(len(tr) == 2 and t[0] not in tr for _, t, tr in folds)


def test_training_size_draws():
    folds = list(learned.split_folds(_items(), "training_size"))
    assert len(folds) == 3 * len(learned.CURVE_SIZES) * learned.CURVE_DRAWS
    again = list(learned.split_folds(_items(), "training_size"))
    assert folds == again
    for (c, size, seed), test, train in folds:
        assert len(train) == size and len(test) == 8 - size and not set(test) & set(train)


def test_unknown_split():
    with pytest.raises(ValueError):
        list(learned.split_folds(_items(), "unknown"))


def test_learned_order_follows_the_lookahead_rule_and_first_joint_set():
    vals = [(frozenset({"A"}), 0.9), (frozenset({"A", "B"}), 0.8), (frozenset({"B"}), 0.8)]
    order = relabel.lookahead_list(vals, ["A", "B"], 2)
    assert order[0] == frozenset({"A"}) and order[1] == frozenset({"B"})      # tie at 0.8: fewer families first
    assert learned._first_joint(order) == frozenset({"A", "B"})


def test_fit_is_deterministic_and_importance_normalized():
    rng = np.random.default_rng(0)
    feats = {frozenset({f}): list(rng.random(len(learned.FEATURE_NAMES))) for f in "ABCDE"}
    for a, b in itertools.combinations("ABCDE", 2):
        feats[frozenset({a, b})] = list(rng.random(len(learned.FEATURE_NAMES)))
    table = {T: float(sum(v[:3])) for T, v in feats.items()}
    it = {"features": feats, "truth": {"table": table}}
    fit_a, n1 = learned.fit([it], 2, 1)
    fit_b, _ = learned.fit([it], 2, 1)
    p1, p2 = learned.predict(fit_a, it, 2), learned.predict(fit_b, it, 2)
    assert n1 == len(table) and p1 == p2
    imp = learned.importance(fit_a)
    assert set(imp) == set(learned.FEATURE_NAMES) and abs(sum(imp.values()) - 1) < 1e-9


def test_report_names_splits_only():
    txt = learned.report({}, list(learned.SPLITS), {}, [])
    heads = [line[3:] for line in txt.splitlines() if line.startswith("## ")]
    assert heads == ["Learner", "Variants", "Features"] + list(learned.SPLITS)
    assert "max_depth = 13" in txt and "learning_rate = 0.1" in txt and "n_estimators = 100" in txt
    assert "10.1007/s10479-018-3067-9" in txt


def test_variants_and_hyperparameters():
    assert set(learned.VARIANTS) == {"structural", "structural+lookahead"}
    assert "lookahead_rel" not in learned.VARIANTS["structural"]
    assert learned.VARIANTS["structural+lookahead"] == learned.FEATURE_NAMES
    assert len(learned.VARIANTS["structural"]) == len(learned.FEATURE_NAMES) - 1
    rng = np.random.default_rng(1)
    feats = {frozenset({f}): list(rng.random(len(learned.FEATURE_NAMES))) for f in "ABCD"}
    it = {"features": feats, "truth": {"table": {T: float(v[-1]) for T, v in feats.items()}}}
    m, _ = learned.fit([it], 2, 1, "structural")
    p = m.get_params()
    assert (p["max_depth"], p["learning_rate"], p["n_estimators"], p["random_state"]) == (13, 0.1, 100, 0)
    assert m.n_features_in_ == len(learned.VARIANTS["structural"])
    assert set(learned.importance(m, "structural")) == set(learned.VARIANTS["structural"])
    assert len(learned.predict(m, it, 2, "structural")) == 4


def test_feature_cache_round_trip_without_the_model(tmp_path, monkeypatch):
    from llmdw.instances import families

    truth = {"families": ["A", "B", "C"], "lookahead": [(frozenset({"A", "B"}), 2.0)]}
    monkeypatch.setattr(families, "read_model", lambda path: _model())
    it = {"name": "inst", "set": "templates", "truth": truth, "model_path": tmp_path / "missing.mps"}
    learned.add_features({"inst": it}, 2, tmp_path / "features", write=True)
    assert learned.feature_file(tmp_path / "features", it, 2).exists()

    def no_model(path):
        raise AssertionError("the model is read although the feature file exists")

    monkeypatch.setattr(families, "read_model", no_model)
    again = {"name": "inst", "set": "templates", "truth": truth, "model_path": tmp_path / "missing.mps"}
    learned.add_features({"inst": again}, 2, tmp_path / "features")
    assert list(again["features"]) == list(it["features"])
    assert all(again["features"][T] == [float(x) for x in it["features"][T]] for T in it["features"])
