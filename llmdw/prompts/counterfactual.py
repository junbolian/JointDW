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
"""
from __future__ import annotations

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
