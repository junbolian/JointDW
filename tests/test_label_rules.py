"""The per-instance label rule of the permuted and anonymized presentations (llmdw.prompts.counterfactual)."""
import hashlib

from llmdw.prompts import counterfactual

SUMMARY = {"constraint_families": [{"family": f} for f in ("A", "B", "C", "D", "E", "F")],
           "variable_families": [{"family": v} for v in ("x", "y", "z")]}


def test_seed_is_first_8_hex_digits_of_sha256():
    assert counterfactual.instance_seed("abc") == int(hashlib.sha256(b"abc").hexdigest()[:8], 16)


def test_permuted_keeps_true_groups_apart_and_is_a_derangement():
    groups = [["A", "B"], ["B", "C"]]
    for name in ("p1", "p2", "p3", "p4"):
        maps, seed = counterfactual.instance_maps(SUMMARY, "permuted", name, groups)
        assert seed >= counterfactual.instance_seed(name)
        assert all(a != b for a, b in maps["cons"].items()) and all(a != b for a, b in maps["vars"].items())
        assert all(maps["cons"][f] not in set(g) for g in groups for f in g)
        assert sorted(maps["cons"].values()) == sorted(maps["cons"])
        assert counterfactual.instance_maps(SUMMARY, "permuted", name, groups) == (maps, seed)


def test_anonymized_labels_are_a_permutation_per_instance():
    labels = set()
    for name in ("p1", "p2", "p3", "p4", "p5"):
        maps, _ = counterfactual.instance_maps(SUMMARY, "anonymized", name)
        assert sorted(maps["cons"].values()) == sorted(f"F{i}" for i in range(1, 7))
        assert sorted(maps["vars"].values()) == ["V1", "V2", "V3"]
        labels.add(tuple(maps["cons"][f] for f in "ABCDEF"))
    assert len(labels) > 1


def test_fixed_rule_unchanged():
    assert counterfactual.maps_for(SUMMARY, "anonymized")["cons"] == {f: f"F{i + 1}" for i, f in enumerate("ABCDEF")}
