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
