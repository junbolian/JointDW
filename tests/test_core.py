"""Offline tests: no LLM endpoint and no Gurobi license needed.

Run from the repository root:  python -m pytest -q tests
Instance: set LLMDW_TEST_MPS to an exported MIPLIB-NL model (default: tests/data/tiny_gap.lp,
written by the test itself).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- tiny model

TINY_GAP = """\\ generalized assignment, 6 jobs x 3 machines
Minimize
 obj: 9 x_0_0 + 2 x_0_1 + 7 x_0_2 + 6 x_1_0 + 4 x_1_1 + 3 x_1_2 + 5 x_2_0 + 8 x_2_1 + 1 x_2_2
    + 7 x_3_0 + 6 x_3_1 + 9 x_3_2 + 4 x_4_0 + 3 x_4_1 + 8 x_4_2 + 3 x_5_0 + 9 x_5_1 + 2 x_5_2
Subject To
 assign_0: x_0_0 + x_0_1 + x_0_2 = 1
 assign_1: x_1_0 + x_1_1 + x_1_2 = 1
 assign_2: x_2_0 + x_2_1 + x_2_2 = 1
 assign_3: x_3_0 + x_3_1 + x_3_2 = 1
 assign_4: x_4_0 + x_4_1 + x_4_2 = 1
 assign_5: x_5_0 + x_5_1 + x_5_2 = 1
 cap_0: 5 x_0_0 + 4 x_1_0 + 6 x_2_0 + 3 x_3_0 + 7 x_4_0 + 2 x_5_0 <= 11
 cap_1: 6 x_0_1 + 5 x_1_1 + 3 x_2_1 + 4 x_3_1 + 2 x_4_1 + 6 x_5_1 <= 10
 cap_2: 4 x_0_2 + 3 x_1_2 + 5 x_2_2 + 6 x_3_2 + 4 x_4_2 + 5 x_5_2 <= 9
Binary
 x_0_0 x_0_1 x_0_2 x_1_0 x_1_1 x_1_2 x_2_0 x_2_1 x_2_2 x_3_0 x_3_1 x_3_2 x_4_0 x_4_1 x_4_2 x_5_0 x_5_1 x_5_2
End
"""


@pytest.fixture(scope="module")
def tiny_mps(tmp_path_factory):
    path = os.environ.get("LLMDW_TEST_MPS")
    if path:
        return path
    p = tmp_path_factory.mktemp("data") / "tiny_gap.lp"
    p.write_text(TINY_GAP)
    return str(p)


def test_families_and_types(tiny_mps):
    from llmdw.instances.families import read_model, summarize

    s = summarize(read_model(tiny_mps))
    fams = {f["family"]: f for f in s["constraint_families"]}
    if "assign_#" in fams:
        assert fams["assign_#"]["miplib_types"] == {"set_partitioning": 6}
        assert fams["cap_#"]["miplib_types"] == {"knapsack": 3}


def test_dec_round_trip_and_components(tiny_mps, tmp_path):
    from llmdw.gcg.dec_io import from_master_families, is_component_partition, read_dec, write_dec
    from llmdw.instances.families import read_model

    m = read_model(tiny_mps)
    if not any(r.name.startswith("assign_") for r in m.rows):
        pytest.skip("custom model")
    blocks, master = from_master_families(m, {"assign_#"})
    assert len(blocks) == 3 and all(len(b) == 1 for b in blocks)  # one knapsack block per machine
    dec = tmp_path / "c.dec"
    write_dec(str(dec), blocks, master)
    b2, master2, pres = read_dec(str(dec))
    assert b2 == blocks and master2 == master and not pres
    assert is_component_partition(m, b2, master2)


def test_gcg_given_beats_lp(tiny_mps, tmp_path):
    """GAP: convexifying the knapsack rows must close part of the LP gap (Geoffrion)."""
    from llmdw.gcg import gcg_run
    from llmdw.gcg.dec_io import from_master_families, write_dec
    from llmdw.instances.families import read_model

    m = read_model(tiny_mps)
    if not any(r.name.startswith("assign_") for r in m.rows):
        pytest.skip("custom model")
    lp = gcg_run.run("lp", tiny_mps, str(tmp_path / "lp.json"))
    blocks, master = from_master_families(m, {"assign_#"})
    write_dec(str(tmp_path / "k.dec"), blocks, master)
    dw = gcg_run.run("dw-given", tiny_mps, str(tmp_path / "dw.json"), dec=str(tmp_path / "k.dec"),
                     root=True, no_cuts=True, no_round=True, no_heur=True, time_limit=60)
    blocks2, master2 = from_master_families(m, {"cap_#"})
    write_dec(str(tmp_path / "a.dec"), blocks2, master2)
    dw2 = gcg_run.run("dw-given", tiny_mps, str(tmp_path / "dw2.json"), dec=str(tmp_path / "a.dec"),
                      root=True, no_cuts=True, no_round=True, no_heur=True, time_limit=60)
    assert lp["ok"] and dw["ok"] and dw2["ok"]
    assert dw["dual_bound"] > lp["dual_bound"] + 1e-6  # knapsack blocks lack the integrality property
    # assignment rows have the integrality property: convexifying them alone gives the LP bound
    assert abs(dw2["dual_bound"] - lp["dual_bound"]) <= 1e-6 * max(1, abs(lp["dual_bound"]))


def test_lp_with_highs_interior_point_matches_the_default_lp(tiny_mps, tmp_path, monkeypatch):
    """LLMDW_LP_SOLVER=highs-ipm: the oracle LP and the lookahead LP give the same optimum as the default solvers."""
    from llmdw.gcg import gcg_run
    from llmdw.lookahead.lookahead import lookahead_table

    monkeypatch.delenv("LLMDW_LP_SOLVER", raising=False)
    ref = gcg_run.run("lp", tiny_mps, str(tmp_path / "lp.json"))
    la_ref = lookahead_table(tiny_mps, 6, 10.0, None)
    monkeypatch.setenv("LLMDW_LP_SOLVER", "highs-ipm")
    ipm = gcg_run.run("lp", tiny_mps, str(tmp_path / "lp_ipm.json"))
    la = lookahead_table(tiny_mps, 6, 10.0, None)
    tol = 1e-6 * max(1.0, abs(ref["dual_bound"]))
    assert ref["ok"] and ipm["ok"] and ipm["lp_solver"] == "highs-ipm" and ipm["highs_model_status"] == "Optimal"
    assert ipm["solving_time"] >= 0 and abs(ipm["dual_bound"] - ref["dual_bound"]) <= tol
    assert la_ref["lp_solver"] == "highs" and la["lp_solver"] == "highs-ipm" and la["lp_seconds"] >= 0
    assert abs(la["lp_bound"] - la_ref["lp_bound"]) <= tol
    for key, row in la_ref["table"].items():
        assert abs(la["table"][key]["lookahead"] - row["lookahead"]) <= 1e-6 * max(1.0, abs(row["lookahead"]))


def test_extract_json_and_validate(tiny_mps):
    from llmdw.instances.families import read_model, summarize
    from llmdw.providers.llm_client import extract_json
    from llmdw.prompts.propose import validate

    a = extract_json('Sure.\n```json\n{"decompose": true, "candidates": [{"master_constraint_families": ["assign_#"]}]}\n```')
    assert a["decompose"] is True
    s = summarize(read_model(tiny_mps))
    fams = [f["family"] for f in s["constraint_families"]]
    ans = {"candidates": [{"master_constraint_families": [fams[0]]},
                          {"master_constraint_families": [fams[0]]},
                          {"master_constraint_families": fams},
                          {"master_constraint_families": ["nope"]},
                          {"master_constraint_families": []}]}
    valid, problems = validate(ans, s, k=3)
    assert len(valid) == 1 and len(problems) == 4


def test_anonymized_prompt_hides_names(tiny_mps, monkeypatch):
    from llmdw.instances.families import read_model, summarize
    from llmdw.prompts.propose import ScriptedLLM, propose

    s = summarize(read_model(tiny_mps))
    first = s["constraint_families"][0]["family"]
    monkeypatch.setenv("LLMDW_ANONYMIZE", "1")
    llm = ScriptedLLM([{"families": {}, "joint_supports": [], "candidates": [{"master_constraint_families": ["F1"]}]}])
    out = propose(llm, s, None, k=3, time_limit=60)
    prompt = out["messages"][1]["content"]
    assert first not in prompt and "F1" in prompt
    assert out["candidates"][0]["master_families"] == [first]


def test_validate_rejects_malformed_answers(tiny_mps):
    from llmdw.instances.families import read_model, summarize
    from llmdw.prompts.propose import validate

    s = summarize(read_model(tiny_mps))
    assert validate(["not", "an", "object"], s, 3)[0] == []
    assert validate("ok", s, 3)[0] == []
    v, probs = validate({"candidates": [{"master_constraint_families": [["nested"]]}]}, s, 3)
    assert v == [] and "malformed" in probs[0]


def test_model_guard_is_exact():
    from llmdw.providers.llm_client import LLMClient

    c = LLMClient(base_url="http://localhost:1", key="x", model="model-5.4")
    assert c.model_matches("model-5.4") and c.model_matches("model-5.4-2026-08-01")
    assert not c.model_matches("model-5.4-mini") and not c.model_matches("model-5.40")


def test_answer_sections_are_stripped():
    from llmdw.prompts.propose import strip_answer_sections

    md = "# Model\n\n## Objective\nmin sum x\n\n## Optimal Value\n**13 blocks**\n\n## Notes\nok\n"
    out = strip_answer_sections(md)
    assert "13 blocks" not in out and "Optimal" not in out and "min sum x" in out and "ok" in out




def test_lookahead_batch_runs_each_instance_in_a_capped_child(tiny_mps, tmp_path, monkeypatch):
    """The lookahead batch runs every instance in its own child process (llmdw.lookahead.lookahead) and gives the same
    table as an in-process run; under a cap too small to start the child, it records the failure and leaves no
    result file behind, so a resumed batch does not mistake the instance for a finished one."""
    import gzip
    import shutil

    import pyscipopt

    from llmdw.gcg import batch
    from llmdw.lookahead.lookahead import lookahead_table

    mps_dir, oracle = tmp_path / "mps", tmp_path / "oracle"
    mps_dir.mkdir()
    (oracle / "tiny").mkdir(parents=True)
    (oracle / "tiny" / "oracle.json").write_text("{}", encoding="utf-8")
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(tiny_mps)
    plain = tmp_path / "tiny.mps"
    m.writeProblem(str(plain), trans=False, verbose=False)
    with open(plain, "rb") as src, gzip.open(mps_dir / "tiny.mps.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)

    def run(out, cap):
        if cap is None:
            monkeypatch.delenv("LLMDW_CHILD_MEMORY_GB", raising=False)
        else:
            monkeypatch.setenv("LLMDW_CHILD_MEMORY_GB", cap)
        batch.main(["lookahead", "--mps-dir", str(mps_dir), "--oracle", str(oracle), "--out", str(out),
                    "--block-time", "10", "--total-time", "120", "--parallel", "1"])

    ref = lookahead_table(str(mps_dir / "tiny.mps.gz"), 6, 10.0, 120.0)
    for out, cap in ((tmp_path / "plain", None), (tmp_path / "capped", "2")):
        run(out, cap)
        res = json.loads((out / "tiny.json").read_text(encoding="utf-8"))
        assert set(res["table"]) == set(ref["table"])
        for key, row in ref["table"].items():
            assert res["table"][key]["lookahead"] == pytest.approx(row["lookahead"], abs=1e-6)
        assert not (out / "tiny.json.part").exists()
    if os.name == "posix":
        out = tmp_path / "too_small"
        run(out, "0.05")
        assert not (out / "tiny.json").exists() and not (out / "tiny.json.part").exists()
        assert (out / "tiny.error.txt").read_text(encoding="utf-8").startswith("child exit")


def test_lookahead_time_budget_is_checked_between_blocks(tiny_mps):
    """When the budget runs out, the family set in progress keeps the sum over its evaluated blocks (a lower bound
    on its full lookahead value), is marked incomplete, and no later family set is evaluated."""
    from llmdw.lookahead.lookahead import lookahead_table

    full = lookahead_table(tiny_mps, 6, 10.0, None)
    assert not full["truncated"] and all(r["complete"] for r in full["table"].values())
    cut = lookahead_table(tiny_mps, 6, 10.0, 1e-9)
    assert cut["truncated"] and len(cut["table"]) <= 1
    for key, row in cut["table"].items():
        assert not row["complete"] and row["blocks_evaluated"] < row["n_blocks"]
        assert row["lookahead"] <= full["table"][key]["lookahead"] + 1e-9


def test_noise_dual_on_an_infinite_side_is_zeroed_and_a_large_one_fails():
    """A one-sided row (x >= 1) gets a dual on its infinite side: -1e-12 is solver noise below the dual feasibility
    tolerance and is set to 0; -1e-3 is a wrong sign and the check fails, naming the row and the dual. A free row
    follows the same rule; a dual on a finite side is kept."""
    from types import SimpleNamespace

    from llmdw.lookahead.lookahead import clean_row_duals

    ge = SimpleNamespace(name="ge_row", lhs=1.0, rhs=1e20)          # x >= 1: only the lower side is finite
    free = SimpleNamespace(name="free_row", lhs=-1e20, rhs=1e20)
    y, zeroed = clean_row_duals([ge, free], [-1e-12, 5e-13], 1e-7)
    assert y == [0.0, 0.0] and zeroed == 2
    y, zeroed = clean_row_duals([ge], [0.5], 1e-7)                   # positive dual selects the finite lhs: kept
    assert y == [0.5] and zeroed == 0
    with pytest.raises(RuntimeError, match=r"row ge_row: dual -0\.001 selects an infinite side"):
        clean_row_duals([ge], [-1e-3], 1e-7)
    with pytest.raises(RuntimeError, match=r"row free_row: dual 0\.002"):
        clean_row_duals([free], [2e-3], 1e-7)
