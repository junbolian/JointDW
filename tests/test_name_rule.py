"""Name rule of the answer parser (llmdw.prompts.propose.normalize_name / normalize_answer / interpret).

The strings are the 20 distinct dropped family names listed in the parser diagnosis of the rich-model answers, with the
family names those prompts showed (one list per class; identical across the instances of a class)."""
import json

import pytest

from llmdw.prompts import propose
from llmdw.prompts.reparse import reparse_record

LS_CONS = ["Balance_#_#", "SetupLink_#_#", "MinLot_#_#", "SafetyStock_#_#", "Capacity_#", "StorageCap_#",
           "MaxSetupsPerPeriod_#", "OvertimeLimit_#", "SetupBudget"]
LS_VARS = ["Setup_B_#", "Setup_A_#", "Overtime_#", "Produce_A_#", "Stock_A_#", "Produce_B_#", "Stock_B_#"]
ND_CONS = ["ArcLink_#_#", "FlowBalance_#_#", "ArcCapacity_#", "NodeOutDegree_#", "OppositeArcs_#", "InArcCover_#",
           "BuildBudget"]
ND_VARS = ["Open_#", "Flow_A_#", "Flow_B_#"]
FL_CONS = ["Link_#_#", "Assign_#", "Capacity_#", "MinUtilization_#", "OpenBudget", "OpenCount", "FacilityConflict"]
FL_VARS = ["Serve_A_#", "Serve_B_#", "Serve_C_#", "Open_B", "Open_A", "Open_C"]

# (string as written, names of its kind shown in the prompt, expected reading; None = stays dropped)
CASES = [
    ("Open_#:B", ND_VARS, "Open_#"),                      # 1, as a linking variable family
    ("Open_C:B", FL_VARS, "Open_C"),                      # 2
    ("Open_A:B", FL_VARS, "Open_A"),                      # 3
    ("Open_B:B", FL_VARS, "Open_B"),                      # 4
    ("Setup_B_#:B", LS_VARS, "Setup_B_#"),                # 5
    ("Setup_A_#:B", LS_VARS, "Setup_A_#"),                # 6
    ("Produce_B_#:C", LS_VARS, "Produce_B_#"),            # 7
    ("Overtime_#:C", LS_VARS, "Overtime_#"),              # 8
    ("NodeOutDegree_#_#", ND_CONS, "NodeOutDegree_#"),    # 9
    ("Overtime:C", LS_VARS, "Overtime_#"),                # 10
    ("Open:B", ND_VARS, "Open_#"),                        # 11
    ("SafetyStock_#", LS_CONS, "SafetyStock_#_#"),        # 12
    ("Open_:B", ND_VARS, "Open_#"),                       # 13
    ("Stock_A:C", LS_VARS, "Stock_A_#"),                  # 14
    ("OppositeArcs_#_#", ND_CONS, "OppositeArcs_#"),      # 15
    ("InArcCover_#_#", ND_CONS, "InArcCover_#"),          # 16
    ("Setup_A:B", LS_VARS, "Setup_A_#"),                  # 17
    ("Setup_B:B", LS_VARS, "Setup_B_#"),                  # 18
    ("Stock_B:C", LS_VARS, "Stock_B_#"),                  # 19
    ("Setup_B:#:B", LS_VARS, None),                       # 20, stays dropped
    # the other strings that stay dropped
    ("Open_#:B", ND_CONS, None),                          # a variable family written as a constraint family
    ("Setup_A:#:B", LS_VARS, None),
    ("Open:#", ND_VARS, None),
]


@pytest.mark.parametrize("s,names,expected", CASES)
def test_listed_strings(s, names, expected):
    assert propose.normalize_name(s, names) == (s if expected is None else expected)


def test_shown_name_is_kept_as_is():
    for names in (LS_CONS, LS_VARS, ND_CONS, ND_VARS, FL_CONS, FL_VARS):
        for n in names:
            assert propose.normalize_name(n, names) == n
    # an exact match wins even where the key is shared by two shown names
    assert propose.normalize_name("X_#", ["X_#", "X_#_#"]) == "X_#"


def test_ambiguous_or_unknown_stays_dropped():
    assert propose.normalize_name("X:B", ["X_#", "X_#_#"]) == "X:B"           # two shown names with the key "x"
    assert propose.normalize_name("Unrelated_#", ND_CONS) == "Unrelated_#"
    assert propose.normalize_name("balance_#_#", LS_CONS) == "Balance_#_#"     # case ignored


def _summary(cons, vars_):
    return {"num_rows": 1, "num_columns": 1, "objective_sense": "minimize",
            "constraint_families": [{"family": f, "variable_families": {}} for f in cons],
            "variable_families": [{"family": v} for v in vars_]}


def test_interpret_reads_linking_family_with_type_letter():
    summary = _summary(ND_CONS, ND_VARS)
    ans = {"families": {}, "joint_supports": [{"families": ["FlowBalance_#_#", "ArcCapacity_#"]}],
           "candidates": [{"master_constraint_families": ["ArcCapacity_#", "BuildBudget"],
                           "linking_variable_families": ["Open_#:B"]}]}
    out = propose.interpret(json.dumps(ans), summary, 3, None, "v2")
    assert out["candidates"] and out["candidates"][0]["linking_var_families"] == ["Open_#"]
    assert [s["families"] for s in out["supports"]] == [["ArcCapacity_#", "FlowBalance_#_#"]]
    bad = dict(ans, joint_supports=[{"families": ["FlowBalance_#_#", "Open_#:B"]}])
    out = propose.interpret(json.dumps(bad), summary, 3, None, "v2")
    assert out["supports"] == [] and any("unknown families" in p for p in out["problems"])


def test_rule_uses_the_presentation_names():
    # permuted presentation: the answer is read against the names shown, then mapped back to the true names
    summary = _summary(["A_#", "B_#"], ["x_#", "y_#"])
    maps = {"cons": {"A_#": "B_#", "B_#": "A_#"}, "vars": {"x_#": "y_#", "y_#": "x_#"}}
    maps.update({"cons_inv": {b: a for a, b in maps["cons"].items()}, "vars_inv": {b: a for a, b in maps["vars"].items()}})
    ans = {"families": {}, "joint_supports": [],
           "candidates": [{"master_constraint_families": ["A"], "linking_variable_families": ["x:C"]}]}
    out = propose.interpret(json.dumps(ans), summary, 3, maps, "v2")
    assert out["candidates"][0]["master_families"] == ["B_#"]
    assert out["candidates"][0]["linking_var_families"] == ["y_#"]


def test_reparse_of_a_recorded_renamed_answer():
    # a record stores the answer mapped back to the true names; reparse renames it again before reading it
    summary = _summary(["A_#", "B_#"], ["x_#", "y_#"])
    maps = {"cons": {"A_#": "B_#", "B_#": "A_#"}, "vars": {"x_#": "y_#", "y_#": "x_#"}}
    maps.update({"cons_inv": {b: a for a, b in maps["cons"].items()}, "vars_inv": {b: a for a, b in maps["vars"].items()}})
    shown = {"families": {}, "joint_supports": [],
             "candidates": [{"master_constraint_families": ["A_#"], "linking_variable_families": ["x_#:C"]}]}
    stored = propose._deanon_answer(shown, maps)     # what the run recorded (unknown "x_#:C" left as it is)
    rec = {"instance": "t", "condition": "permuted", "answer": stored, "supports": [], "candidates": [],
           "problems": [], "family_info": {}}
    new, changed = reparse_record(rec, summary, maps)
    assert changed and new["candidates"][0]["master_families"] == ["B_#"]
    assert new["candidates"][0]["linking_var_families"] == ["y_#"]
