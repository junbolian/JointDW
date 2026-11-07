"""Counterfactual presentations of a model to the LLM: which information drives its answers.

Paper: Section 6 and Appendix D: the four presentations (Figure 3(b, c), Tables 6 to 8).

condition    family names shown                     description and formulation
original     true names                             shown
names_only   true names                             withheld
permuted     constraint names permuted among the    withheld
             constraint families (a derangement),
             variable names among variable families
anonymized   F1..Fk and V1..Vm                       withheld

Comparing original with names_only measures what the text adds; names_only with permuted measures
whether the answers follow the names or the statistics (a model that reads the statistics is not
hurt by permuted names, a model that reads the names is); anonymized is the statistics-only floor.
Answers are mapped back to the true names before validation and scoring.

Label rules (maps_for: the rule of the paper's presentations; instance_maps: per-instance random labels)
fixed          anonymized: F1..Fk and V1..Vm in the order of the family tables (decreasing number of rows or columns);
               permuted: a derangement drawn from random.Random(seed), the same seed for every instance
per-instance   seed = the first 8 hex digits of SHA-256 of the instance name (as an integer); anonymized: the labels
               F1..Fk and V1..Vm assigned by a random permutation (random.Random(seed).shuffle); permuted: a derangement
               of the constraint family names and one of the variable family names from random.Random(seed), redrawn
               with seed + 1, seed + 2, ... until no family of a forbidden group (the true joint supports) receives the
               name of a family of the same group. The seed used is returned with the maps.
"""
from __future__ import annotations

import hashlib
import random

CONDITIONS = ("original", "names_only", "permuted", "anonymized")


def withhold_text(condition: str) -> bool:
    return condition != "original"


def _derangement(items: list, rng: random.Random) -> dict:
    if len(items) < 2:
        return {x: x for x in items}
    while True:
        perm = items[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))


def maps_for(summary: dict, condition: str, seed: int = 0) -> dict | None:
    """Name maps in the format of propose.anonymize_maps, or None for the true names."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    if condition in ("original", "names_only"):
        return None
    cons = [f["family"] for f in summary["constraint_families"]]
    vars_ = [f["family"] for f in summary["variable_families"]]
    if condition == "anonymized":
        c = {f: f"F{i + 1}" for i, f in enumerate(cons)}
        v = {f: f"V{i + 1}" for i, f in enumerate(vars_)}
    else:
        rng = random.Random(seed)
        c = _derangement(cons, rng)
        v = _derangement(vars_, rng)
    return {"cons": c, "vars": v, "cons_inv": {b: a for a, b in c.items()}, "vars_inv": {b: a for a, b in v.items()}}


def applicable(summary: dict, condition: str) -> bool:
    """permuted needs at least two constraint families."""
    return condition != "permuted" or len(summary["constraint_families"]) >= 2


def instance_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def instance_maps(summary: dict, condition: str, name: str, groups=()) -> tuple[dict | None, int | None]:
    """(maps, seed used) under the per-instance rule (module docstring); groups: the families of each true joint
    support, which the permuted presentation keeps apart from each other's names."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    if condition in ("original", "names_only"):
        return None, None
    cons = [f["family"] for f in summary["constraint_families"]]
    vars_ = [f["family"] for f in summary["variable_families"]]
    seed = instance_seed(name)
    if condition == "anonymized":
        rng = random.Random(seed)
        cl, vl = [f"F{i + 1}" for i in range(len(cons))], [f"V{i + 1}" for i in range(len(vars_))]
        rng.shuffle(cl)
        rng.shuffle(vl)
        c, v = dict(zip(cons, cl)), dict(zip(vars_, vl))
    else:
        groups = [set(g) for g in groups]
        while True:
            rng = random.Random(seed)
            c = _derangement(cons, rng)
            v = _derangement(vars_, rng)
            if all(c[f] not in g for g in groups for f in g):
                break
            seed += 1
    return ({"cons": c, "vars": v, "cons_inv": {b: a for a, b in c.items()},
             "vars_inv": {b: a for a, b in v.items()}}, seed)
