import hashlib
import json

import pytest

from llmdw.prompts import propose
from llmdw.providers import support_run
from llmdw.providers.llm_client import LLMClient
from llmdw.prompts.propose import ScriptedLLM
from llmdw.instances import export, templates


@pytest.fixture(scope="module")
def tmpl(tmp_path_factory):
    out = tmp_path_factory.mktemp("templates")
    export.write_instance(templates.clsp(), out)
    return out


def test_ranked_prompt_is_v2_with_two_additions_only():
    v2, rk = propose.prompt_template("v2"), propose.prompt_template("ranked")
    assert rk["SYSTEM"] == v2["SYSTEM"] and rk["FEEDBACK"] == v2["FEEDBACK"]
    fields = (' "ranked_groups": [\n  {{"families": ["...", "..."], "probability": 0.0}}\n ],\n'
              ' "relevant_families": ["..."],\n')
    s = (' Under "ranked_groups", list up to 5 different sets of constraint families, most likely first: each set is '
         'your guess for the smallest set of families whose convexification gives the best bound of any family-level '
         'DW reformulation of this model, and "probability" is your probability that it is that set. Under '
         '"relevant_families", list the constraint families that you expect the smallest best set to contain; list '
         'more families rather than fewer when unsure.')
    assert rk["USER"].count(fields) == 1 and rk["USER"].count(s) == 1
    assert rk["USER"].replace(fields, "").replace(s, "") == v2["USER"]
    assert rk["USER"].index(fields) > rk["USER"].index('"joint_supports"')


def test_ranked_groups_and_relevant_families_are_validated(tmpl):
    from llmdw.instances.families import read_model, summarize
    summ = summarize(read_model(str(next(tmpl.iterdir()) / "model.mps")))
    fams = sorted(f["family"] for f in summ["constraint_families"])
    a = {"families": {}, "joint_supports": [], "candidates": [],
         "ranked_groups": [{"families": fams[:2], "probability": 0.6}, {"families": [fams[0]], "probability": 0.2},
                           {"families": [fams[1], fams[0]], "probability": 0.1}, {"families": ["nope", fams[0]]},
                           {"families": fams[1:3] if len(fams) > 2 else fams[:2], "probability": 1.5}],
         "relevant_families": [fams[0], "nope"]}
    out = propose.interpret(json.dumps(a), summ, 3, None, "ranked")
    assert out["ranked_groups"][0] == {"families": sorted(fams[:2]), "rank": 1, "probability": 0.6}
    assert out["ranked_groups"][1] == {"families": [fams[0]], "rank": 2, "probability": 0.2}   # one family is valid
    assert len(out["ranked_groups"]) == (3 if len(fams) > 2 else 2)          # duplicate and unknown dropped
    assert out["ranked_groups"][-1]["probability"] is None or len(fams) <= 2  # 1.5 is not a probability
    assert out["relevant_families"] == [fams[0]] and any("relevant_families" in p for p in out["problems"])
    none = propose.interpret(json.dumps({"families": {}, "joint_supports": [], "candidates": []}), summ, 3, None,
                             "ranked")
    assert none["ranked_groups"] == [] and none["relevant_families"] is None


def test_prompt_tag_changes_the_cache_key_only_when_set(monkeypatch, tmp_path):
    client = LLMClient(base_url="http://localhost:1/v1", key="k", model="m", cache_dir=str(tmp_path / "c"),
                       log_path=str(tmp_path / "calls.jsonl"))
    msgs = [{"role": "user", "content": "x"}]
    seen = {}

    def fake_post(body):
        return {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"a\": 1}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.delenv("LLMDW_PROMPT_TAG", raising=False)
    client.chat(msgs, sample=1)
    raw = json.dumps(client._body(msgs, True), sort_keys=True, ensure_ascii=False) + "\n#sample=1"
    plain = hashlib.sha256(raw.encode()).hexdigest()
    assert (tmp_path / "c" / f"{plain}.json").exists()                      # the key of the original prompt
    monkeypatch.setenv("LLMDW_PROMPT_TAG", "user_ranked.txt")
    out = client.chat(msgs, sample=1)
    assert not out["cached"]                                                # the other arm's answer is not reused
    tagged = hashlib.sha256((raw.replace("\n#sample=1", "") + "\n#prompt=user_ranked.txt\n#sample=1").encode()).hexdigest()
    assert (tmp_path / "c" / f"{tagged}.json").exists()


def test_support_run_records_ranked_fields_under_their_own_tags(tmpl, tmp_path):
    from llmdw.instances.families import read_model, summarize
    d = next(tmpl.iterdir())
    fams = sorted(f["family"] for f in summarize(read_model(str(d / "model.mps")))["constraint_families"])
    ans = {"families": {}, "joint_supports": [], "candidates": [], "ranked_groups": [{"families": fams[:2]}],
           "relevant_families": fams[:1]}
    llm = ScriptedLLM([ans])
    recs = support_run.run_instance(llm, d.name, str(d / "model.mps"), str(d), ["original"], tmp_path, samples=1,
                                    prompt_version="ranked", tag_prefix="ranked")
    assert llm.calls[0]["tag"] == f"ranked:{d.name}:original:s0"
    assert "ranked_groups" in llm.calls[0]["messages"][1]["content"]
    assert recs[0]["ranked_groups"][0]["families"] == sorted(fams[:2]) and recs[0]["relevant_families"] == fams[:1]
    assert recs[0]["prompt_version"] == "ranked"


def test_prompt_token_tolerance_override(monkeypatch, tmp_path):
    from llmdw.providers import llm_client
    c = LLMClient(base_url="http://localhost:1/v1", key="k", model="m", cache_dir=str(tmp_path / "c"),
                  log_path=str(tmp_path / "calls.jsonl"))
    ref = c._prompt_ref_path("request", "b")
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text('{"prompt_tokens": 2024}')
    resp = {"usage": {"prompt_tokens": 2097}}
    monkeypatch.delenv("LLMDW_PROMPT_TOKENS_REFERENCE_TOLERANCE", raising=False)
    assert "tolerance 5" in c._guard_reason(resp, "b", "m")
    monkeypatch.setenv("LLMDW_PROMPT_TOKENS_REFERENCE_TOLERANCE", "80")
    assert c._guard_reason(resp, "b", "m") is None
    assert llm_client.PROMPT_TOKENS_REFERENCE_TOLERANCE == 5


def test_with_data_prompt_adds_only_the_lp_section(tmpl):
    v2, wd = propose.prompt_template("v2"), propose.prompt_template("with_data")
    assert wd["SYSTEM"] == v2["SYSTEM"] and wd["FEEDBACK"] == v2["FEEDBACK"]
    assert wd["USER"].replace("## Model with data\n{model_lp}\n\n", "") == v2["USER"]
    d = next(tmpl.iterdir())
    lp = propose.model_lp_text(str(d))
    from llmdw.instances.families import read_model, summarize
    model = read_model(str(d / "model.mps"))
    assert all(f" {r.name}:" in lp for r in model.rows) and not any(x.startswith("\\") for x in lp.splitlines())
    msgs = propose.build_messages(summarize(model), str(d), 3, 120.0, template=wd)
    assert "## Model with data\n" + lp + "\n\n## Model statistics" in msgs[1]["content"]
