"""Prompt v2, counterfactual conditions, support runs and their scoring (scripted LLM, no endpoint).

Run from the repository root:  python -m pytest -q tests/test_support_llm.py
"""
from __future__ import annotations

import json

import pytest

from llmdw.scoring import analyze_support
from llmdw.prompts import counterfactual
from llmdw.providers import support_run
from llmdw.instances.families import read_model, summarize
from llmdw.prompts.propose import ScriptedLLM, propose
from llmdw.instances import export, templates


@pytest.fixture(scope="module")
def tmpl(tmp_path_factory):
    out = tmp_path_factory.mktemp("templates")
    for make in (templates.uls, templates.clsp, templates.gap):
        export.write_instance(make(), out)
    return out


def test_counterfactual_maps_are_derangements(tmpl):
    s = summarize(read_model(str(tmpl / "uls" / "model.mps")))
    m = counterfactual.maps_for(s, "permuted", seed=3)
    assert all(a != b for a, b in m["cons"].items())
    assert all(m["cons_inv"][m["cons"][f]] == f for f in m["cons"])
    assert counterfactual.maps_for(s, "original") is None


def test_v2_answer_is_mapped_back_and_text_withheld(tmpl):
    s = summarize(read_model(str(tmpl / "uls" / "model.mps")))
    shown = counterfactual.maps_for(s, "permuted", seed=0)["cons"]
    answer = {"families": {shown["Balance_#"]: {"strengthens_alone": "no"}},
              "joint_supports": [{"families": [shown["Balance_#"], shown["SetupLink_#"]], "inequality_class": "(l,S)"}],
              "decompose": True,
              "candidates": [{"master_constraint_families": [], "linking_variable_families": []}]}
    res = propose(ScriptedLLM([answer]), s, str(tmpl / "uls"), 3, 120, prompt_version="v2", condition="permuted")
    assert res["supports"][0]["families"] == ["Balance_#", "SetupLink_#"]
    assert "Balance_#" in res["family_info"]
    user = res["messages"][1]["content"]
    assert "(withheld)" in user and "setup cost" not in user
    res2 = propose(ScriptedLLM([answer]), s, str(tmpl / "uls"), 3, 120, prompt_version="v2", condition="original")
    assert "setup cost" in res2["messages"][1]["content"]


def test_support_run_and_scoring(tmpl, tmp_path):
    right = {"families": {}, "joint_supports": [{"families": ["Balance_#_#", "SetupLink_#_#"]}], "decompose": True,
             "candidates": [{"master_constraint_families": ["Capacity_#"], "linking_variable_families": []}]}
    wrong = {"families": {}, "joint_supports": [], "decompose": False,
             "candidates": [{"master_constraint_families": ["Balance_#_#", "Capacity_#"], "linking_variable_families": []}]}
    recs = support_run.run_instance(ScriptedLLM([right, wrong]), "clsp", str(tmpl / "clsp" / "model.mps"),
                                    str(tmpl / "clsp"), ["original", "anonymized"], tmp_path)
    truth = analyze_support.template_truth(tmpl / "clsp")
    by = {r["condition"]: analyze_support.score(r, truth) for r in recs}
    assert by["original"]["joint_hit1"] is True and by["anonymized"]["joint_hit1"] is False
    assert by["original"]["e"] == 0 and by["original"]["evals_anchored"] == 1
    assert by["anonymized"]["e"] == 1 and by["anonymized"]["evals_greedy"] is None   # every single family: zero gain
    assert json.loads((tmp_path / "clsp" / "original.json").read_text())["supports"][0]["families"] == \
        ["Balance_#_#", "SetupLink_#_#"]


def test_json_mode_switch(monkeypatch):
    from llmdw.providers.llm_client import LLMClient

    monkeypatch.setenv("LLMDW_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLMDW_MODEL", "m")
    monkeypatch.setenv("LLMDW_JSON_MODE", "0")
    assert "response_format" not in LLMClient()._body([{"role": "user", "content": "x"}], json_mode=True)
    monkeypatch.setenv("LLMDW_JSON_MODE", "1")
    assert "response_format" in LLMClient()._body([{"role": "user", "content": "x"}], json_mode=True)


def test_extra_body_pins_reasoning_and_keys_the_cache(monkeypatch, tmp_path):
    from llmdw.providers.llm_client import LLMClient

    sent = []

    def fake_post(self, body):
        sent.append(body)
        return {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.setenv("LLMDW_EXTRA_BODY", '{"reasoning_effort": "high", "model": "ignored"}')
    kw = dict(base_url="http://localhost:1/v1", key="k", model="m", cache_dir=str(tmp_path / "cache"),
              log_path=str(tmp_path / "calls.jsonl"))
    msg = [{"role": "user", "content": "x"}]
    LLMClient(**kw).chat(msg)
    assert sent[0]["reasoning_effort"] == "high" and sent[0]["model"] == "m"
    LLMClient(**kw).chat(msg)                                   # same request: answered from the cache
    LLMClient(**kw, extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"}).chat(msg)
    assert len(sent) == 2 and sent[1]["thinking"] == {"type": "enabled"}   # other setting, other cache entry


def test_samples_are_separate_cache_entries_and_never_reach_the_request(monkeypatch, tmp_path):
    from llmdw.providers.llm_client import LLMClient

    sent = []

    def fake_post(self, body):
        sent.append(body)
        return {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"n\": %d}" % len(sent)},
                                                        "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    kw = dict(base_url="http://localhost:1/v1", key="k", model="m", cache_dir=str(tmp_path / "cache"),
              log_path=str(tmp_path / "calls.jsonl"))
    msg = [{"role": "user", "content": "x"}]
    first, second = LLMClient(**kw).chat(msg), LLMClient(**kw).chat(msg, sample=1)
    assert len(sent) == 2 and first["content"] != second["content"]
    assert sent[0] == sent[1]                                   # identical requests; only the cache key differs
    assert LLMClient(**kw).chat(msg, sample=1)["cached"] and LLMClient(**kw).chat(msg)["cached"]
    assert len(sent) == 2


def test_support_run_writes_one_file_per_sample_and_resumes(tmpl, tmp_path):
    answer = {"joint_supports": [], "decompose": False, "candidates": []}
    llm = ScriptedLLM([answer] * 4)
    support_run.run_instance(llm, "gap", str(tmpl / "gap" / "model.mps"), str(tmpl / "gap"),
                             ["original", "anonymized"], tmp_path, samples=2)
    assert sorted(p.name for p in (tmp_path / "gap").iterdir()) == \
        ["anonymized.json", "anonymized.s1.json", "original.json", "original.s1.json"]
    assert [c["sample"] for c in llm.calls] == [0, 1, 0, 1]
    again = ScriptedLLM([])
    recs = support_run.run_instance(again, "gap", str(tmpl / "gap" / "model.mps"), str(tmpl / "gap"),
                                    ["original", "anonymized"], tmp_path, samples=2)
    assert again.calls == [] and sorted(r["sample"] for r in recs) == [0, 0, 1, 1]


def test_paired_differences_average_samples_within_instances():
    rows = [{"instance": "a", "condition": "names_only", "sample": 0, "joint_recall": 1.0},
            {"instance": "a", "condition": "names_only", "sample": 1, "joint_recall": 0.0},
            {"instance": "a", "condition": "permuted", "sample": 0, "joint_recall": 0.0},
            {"instance": "b", "condition": "names_only", "sample": 0, "joint_recall": 1.0},
            {"instance": "b", "condition": "permuted", "sample": 0, "joint_recall": 1.0}]
    res = analyze_support.paired(rows, "names_only", "permuted")
    assert res["n_instances"] == 2 and abs(res["mean_difference"] - 0.25) < 1e-12


def test_smoke_report_flags_an_endpoint_cache_hit(monkeypatch, capsys):
    from llmdw.providers import smoke
    from llmdw.providers.llm_client import LLMClient

    def fake_post(self, body):
        return {"model": "m-2026-08-01", "usage": {"completion_tokens": 9},
                "choices": [{"message": {"content": "{\"prime\": false, \"smallest_factor\": 17}"},
                             "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    for k, v in {"LLMDW_BASE_URL": "http://localhost:1/v1", "LLMDW_MODEL": "m", "LLMDW_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    assert smoke.main([]) == 1                       # identical answers returned at once: an endpoint cache is suspected
    report = json.loads(capsys.readouterr().out)
    assert report["model_matches"] and report["first"]["answer_correct"] and report["endpoint_cache_suspected"]


def test_answer_is_the_last_json_object_with_the_required_keys():
    """Prose with braces before the answer, a fenced answer after the prose, a schema echo before the answer,
    and a malformed object are handled by one rule."""
    from llmdw.providers.llm_client import extract_json
    from llmdw.prompts.propose import REQUIRED_KEYS

    req = REQUIRED_KEYS["v2"]
    ans = {"families": {"A_#": {}}, "joint_supports": [], "candidates": []}
    prose = "The sets {T} and {S, T} matter.\n\n```json\n" + json.dumps(ans) + "\n```"
    assert extract_json(prose, req) == ans
    echo = json.dumps({"families": {}, "joint_supports": [], "candidates": [], "x": 1}) + "\n" + json.dumps(ans)
    assert extract_json(echo, req) == ans                                      # several: the last one
    partial = json.dumps({"candidates": []}) + " then " + json.dumps(ans) + " and " + json.dumps({"note": 1})
    assert extract_json(partial, req) == ans                                   # only objects with every key count
    with pytest.raises(ValueError):
        extract_json('{"families": {"A": {"x": 1}, "joint_supports": [], "candidates": []', req)   # malformed
    with pytest.raises(ValueError):
        extract_json(json.dumps({"joint_supports": [], "candidates": []}), req)                     # a key missing
    assert extract_json('{"prime": false, "smallest_factor": 17}') == {"prime": False, "smallest_factor": 17}
