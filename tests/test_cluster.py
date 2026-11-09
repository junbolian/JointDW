import numpy as np
import pytest

from llmdw.scoring import cluster, results


def _old_instance_bootstrap(vals, stat=np.mean, resamples=10_000, seed=0):
    v = np.asarray(vals, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(resamples, len(v)))
    reps = stat(v[idx], axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(reps, ddof=1))


def test_singleton_groups_give_exactly_the_instance_bootstrap():
    cluster.set_groups({})
    vals = {f"i{k}": (None if x is None else float(x)) for k, x in enumerate([3, 1, 4, 1, 5, 9, 2, 6, None, 5])}
    plain = [v for v in vals.values() if v is not None]
    assert results.boot_stat_se(vals, np.mean) == _old_instance_bootstrap(plain)
    assert results.boot_stat_se(vals, np.median) == _old_instance_bootstrap(plain, np.median)
    assert results.boot_ci(vals) == _old_instance_bootstrap(plain)[:2]


def test_identical_prompts_are_resampled_together():
    cluster.set_groups({"a": "g", "b": "g"})              # a and b share a prompt group, c is alone
    reps = cluster.replicates({"a": 1.0, "b": 1.0, "c": 0.0}, np.mean, 2000, 0)
    assert set(np.round(reps, 12)) <= {1.0, round(2 / 3, 12), 0.0}       # whole groups only: never 1/3
    meds = cluster.replicates({"a": 1.0, "b": 1.0, "c": 0.0}, np.median, 500, 0)
    assert set(meds) <= {0.0, 1.0}
    cluster.set_groups({})
    assert round(1 / 3, 12) in set(np.round(cluster.replicates({"a": 1.0, "b": 1.0, "c": 0.0}, np.mean, 2000, 0), 12))


def test_prompt_groups_close_transitively_across_presentations():
    m = lambda t: [{"role": "user", "content": t}]  # noqa: E731
    prompts = {"a": {"original": m("x"), "anonymized": m("p")}, "b": {"original": m("x"), "anonymized": m("q")},
               "c": {"original": m("y"), "anonymized": m("q")}, "d": {"original": m("z"), "anonymized": m("r")}}
    assert cluster.prompt_groups(prompts) == {"a": "a", "b": "a", "c": "a", "d": "d"}


def test_the_cluster_bootstrap_needs_instance_names():
    with pytest.raises(TypeError):
        results.boot_stat_se([1.0, 2.0, 3.0])


# ---------------------------------------------------------------- the two-stage bootstrap

def _one_stage_reference(values, groups, stat=np.mean, resamples=10_000, seed=0):
    """The one-stage bootstrap over prompt groups as it was before the two-stage bootstrap (verbatim logic)."""
    items = [(n, v) for n, v in (values.items() if isinstance(values, dict) else values) if v is not None]
    vals = np.asarray([v for _, v in items], dtype=float)
    labels = [groups.get(n, n) for n, _ in items]
    order = list(dict.fromkeys(labels))
    rng = np.random.default_rng(seed)
    if len(order) == len(vals):
        reps = stat(vals[rng.integers(0, len(vals), size=(resamples, len(vals)))], axis=1)
    else:
        pos = {g: i for i, g in enumerate(order)}
        members = [[] for _ in order]
        for i, g in enumerate(labels):
            members[pos[g]].append(i)
        gi = rng.integers(0, len(order), size=(resamples, len(order)))
        reps = np.array([stat(np.concatenate([vals[members[g]] for g in row])) for row in gi])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(reps, ddof=1))


def _close(a, b):
    return all(np.isclose(x, y, rtol=0, atol=1e-12) for x, y in zip(a, b))


def test_singleton_groups_give_the_numbers_of_before():
    vals = {f"i{k}": float(x) for k, x in enumerate([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])}
    renamed = {f"i{k}": f"g{k}" for k in range(10)}            # every group one instance, under other labels
    for groups in ({}, renamed):
        cluster.set_groups(groups)
        assert cluster.boot(vals) == _one_stage_reference(vals, groups)
        assert cluster.boot(vals, np.median) == _one_stage_reference(vals, groups, np.median)
    # several values per instance (the samples of one instance stay together), every group one instance
    pairs = [("a", 1.0), ("a", 0.0), ("b", 2.0), ("c", 0.5), ("c", 0.5), ("c", 1.5)]
    cluster.set_groups({})
    assert _close(cluster.boot(pairs), _one_stage_reference(pairs, {}))
    assert cluster.boot(pairs, np.median) == _one_stage_reference(pairs, {}, np.median)


def test_one_group_with_varying_values_has_a_positive_standard_error():
    vals = {f"i{k}": float(k) for k in range(20)}
    groups = {n: "one prompt" for n in vals}
    cluster.set_groups(groups)
    lo, hi, se = cluster.boot(vals)
    assert se > 0 and lo < np.mean(list(vals.values())) < hi
    assert _one_stage_reference(vals, groups)[2] == 0.0          # the one-stage bootstrap had no spread
    assert cluster.counts(vals) == (20, 1) and cluster.n_groups(vals) == 1
    # a single group draws its instances as the instance bootstrap does
    cluster.set_groups({})
    assert cluster.boot(vals) == (lo, hi, se)


def test_two_stages_draw_instances_within_drawn_groups():
    cluster.set_groups({"a": "g", "b": "g", "c": "h"})
    reps = cluster.replicates({"a": 1.0, "b": 0.0, "c": 4.0}, np.mean, 4000, 0)
    # stage 1 draws g or h twice; stage 2 draws two instances within each drawn g (a or b) and c within h
    possible = {round(x, 12) for x in [k / 4 for k in range(5)] + [(k + 4) / 3 for k in range(3)] + [4.0]}
    got = set(np.round(reps, 12))
    assert got <= possible and {0.25, 0.75} <= got                 # instances within g are resampled
    meds = cluster.replicates({"a": 1.0, "b": 0.0, "c": 4.0}, np.median, 500, 0)
    assert set(meds) <= {0.0, 0.5, 1.0, 4.0}
    cluster.set_groups({})


def test_several_values_per_instance_stay_together_in_stage_two():
    cluster.set_groups({"a": "g", "b": "g"})
    reps = cluster.replicates([("a", 1.0), ("a", 1.0), ("b", 0.0)], np.mean, 2000, 0)
    assert set(np.round(reps, 12)) <= {0.0, round(2 / 3, 12), 1.0}       # a contributes both of its values
    cluster.set_groups({})
