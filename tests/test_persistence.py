"""Persistence: atomic cache, rejected keys and quotas, stopped threads, cut-off files, cost projection.

Run from the repository root:  python -m pytest -q tests/test_persistence.py
"""
from __future__ import annotations

import gzip
import io
import json
import shutil
import urllib.error

import pytest

from llmdw.providers import support_run, usage
from llmdw.instances.families import read_model, summarize
from llmdw.providers.llm_client import LLMClient, EndpointRejected
from llmdw.prompts.propose import ScriptedLLM
from llmdw.instances import export, templates


@pytest.fixture(scope="module")
def tmpl(tmp_path_factory):
    out = tmp_path_factory.mktemp("templates")
    for make in (templates.clsp, templates.gap):
        export.write_instance(make(), out)
    return out


def _client(tmp_path, **kw):
    return LLMClient(base_url="http://localhost:1/v1", key="k", model="m", cache_dir=str(tmp_path / "cache"),
                     log_path=str(tmp_path / "calls.jsonl"), **kw)


def _http_error(code, body):
    return urllib.error.HTTPError("http://localhost:1/v1/chat/completions", code, "x", {}, io.BytesIO(body.encode()))


def test_cut_off_cache_entry_is_moved_and_the_request_sent_again(monkeypatch, tmp_path):
    sent = []

    def fake_post(self, body):
        sent.append(body)
        return {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    msg = [{"role": "user", "content": "x"}]
    _client(tmp_path).chat(msg)
    entry = next((tmp_path / "cache").glob("*.json"))
    entry.write_text('{"response": {"model": "m", "cho')            # a write cut off by a crash
    out = _client(tmp_path).chat(msg)
    assert len(sent) == 2 and not out["cached"]
    assert json.loads(entry.read_text())["response"]["model"] == "m"
    assert (entry.parent / (entry.name + ".corrupt")).exists()
    assert not list((tmp_path / "cache").glob("*.part"))
    assert _client(tmp_path).chat(msg)["cached"] and len(sent) == 2


@pytest.mark.parametrize("code,body", [(401, "invalid token"), (403, "\u7528\u6237\u989d\u5ea6\u4e0d\u8db3"),
                                       (429, '{"error": {"code": "insufficient_quota"}}')])
def test_rejected_key_or_used_up_quota_stops_at_once(monkeypatch, tmp_path, code, body):
    attempts = []

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        raise _http_error(code, body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(EndpointRejected):
        _client(tmp_path).chat([{"role": "user", "content": "x"}])
    assert len(attempts) == 1
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())


def test_rate_limit_is_retried(monkeypatch, tmp_path):
    attempts = []

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        raise _http_error(429, "rate limit reached, try again in 20s")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(RuntimeError) as err:
        _client(tmp_path, max_retries=3).chat([{"role": "user", "content": "x"}])
    assert not isinstance(err.value, EndpointRejected) and len(attempts) == 3


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("code", [524, 520, 522, 523])
def test_gateway_error_is_retried_and_only_the_answer_is_cached(monkeypatch, tmp_path, code):
    attempts = []
    answer = {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(code, "status code 524")
        return _FakeResponse(json.dumps(answer).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("LLMDW_STREAM", "0")
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    out = _client(tmp_path).chat([{"role": "user", "content": "x"}])
    assert len(attempts) == 2 and out["content"] == "{\"ok\": true}" and not out["cached"]
    entries = list((tmp_path / "cache").glob("*.json"))
    assert len(entries) == 1
    assert json.loads(entries[0].read_text())["response"]["choices"][0]["message"]["content"] == "{\"ok\": true}"
    assert not list((tmp_path / "cache").glob("*.part"))


def test_exhausted_gateway_retries_count_as_incomplete_and_the_run_goes_on(tmpl, monkeypatch, tmp_path, capsys):
    answer = {"model": "m", "usage": {}, "choices": [{"message": {"content": json.dumps(
        {"families": {}, "joint_supports": [], "assessment": "", "decompose": False, "candidates": []})},
        "finish_reason": "stop"}]}

    def fake_urlopen(req, timeout=None):
        if b"SetupLink" not in req.data:           # the request of the gap template fails on every attempt
            raise _http_error(524, "status code 524")
        return _FakeResponse(json.dumps(answer).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    for k, v in {"LLMDW_BASE_URL": "http://localhost:1/v1", "LLMDW_API_KEY": "k", "LLMDW_MODEL": "m",
                 "LLMDW_CACHE_DIR": str(tmp_path / "cache"), "LLMDW_LOG": str(tmp_path / "calls.jsonl"),
                 "LLMDW_STREAM": "0", "LLMDW_TEMPERATURE": "none"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    out = tmp_path / "out"
    support_run.main(["--templates", str(tmpl), "--conditions", "original", "--samples", "1", "--parallel", "2",
                      "--out", str(out)])
    lines = capsys.readouterr().out.splitlines()
    assert (out / "clsp" / "original.json").exists() and not (out / "gap" / "original.json").exists()
    incomplete = [line for line in lines if line.startswith("INCOMPLETE")]
    assert len(incomplete) == 1 and incomplete[0].startswith("INCOMPLETE gap original s0") and "524" in incomplete[0]
    assert not [line for line in lines if line.startswith("STOPPED")]
    assert lines[-1] == "all 2 instances finished"


def test_each_retried_attempt_is_logged_apart_from_the_call_log(monkeypatch, tmp_path):
    attempts = []
    answer = {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        if len(attempts) <= 2:
            raise _http_error(524, "status code 524")
        return _FakeResponse(json.dumps(answer).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("LLMDW_STREAM", "0")
    monkeypatch.setenv("LLMDW_CONFIG", "cfg_x")
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    out = _client(tmp_path).chat([{"role": "user", "content": "x"}], tag="support:a:original:s0")
    assert out["content"] == "{\"ok\": true}" and len(attempts) == 3
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    retried = [json.loads(line) for line in (tmp_path / "llm_retries.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and "error" not in calls[0]
    assert len(retried) == 2
    for r in retried:
        assert (r["reason"], r["status"], r["config"]) == ("gateway", 524, "cfg_x")
        assert r["tag"] == "support:a:original:s0" and r["request_sha256"] == calls[0]["request_sha256"]
        assert r["model"] == "m" and r["seconds"] >= 0
    assert usage.retries(tmp_path / "llm_retries.jsonl") == {"cfg_x": {"gateway": 2, "rate_limit": 0, "network": 0,
                                                                  "incomplete": 0, "inflation": 0, "other": 0}}


def test_a_stop_ends_an_instance_before_its_next_call(tmpl, tmp_path):
    answer = {"joint_supports": [], "decompose": False, "candidates": []}
    llm = ScriptedLLM([answer] * 8)
    with pytest.raises(support_run.RunStopped):
        support_run.run_instance(llm, "gap", str(tmpl / "gap" / "model.mps"), str(tmpl / "gap"),
                                 ["original", "anonymized"], tmp_path, samples=2,
                                 should_stop=lambda: len(llm.calls) >= 1)
    assert len(llm.calls) == 1 and [p.name for p in (tmp_path / "gap").iterdir()] == ["original.json"]
    again = ScriptedLLM([answer] * 8)                                # the resume makes only the missing calls
    support_run.run_instance(again, "gap", str(tmpl / "gap" / "model.mps"), str(tmpl / "gap"),
                             ["original", "anonymized"], tmp_path, samples=2)
    assert len(again.calls) == 3


def test_usage_counts_cells_with_byte_identical_samples(tmpl, tmp_path):
    a = {"joint_supports": [], "decompose": False, "candidates": []}
    b = {"joint_supports": [], "decompose": True, "candidates": []}
    runs = tmp_path / "cfg_x"                                   # original: samples 0 and 2 identical; anonymized: all differ
    support_run.run_instance(ScriptedLLM([a, b, a, a, b, {**b, "x": 1}]), "gap", str(tmpl / "gap" / "model.mps"),
                             str(tmpl / "gap"), ["original", "anonymized"], runs, samples=3)
    recs = [json.loads(p.read_text()) for p in sorted((runs / "gap").glob("*.json"))]
    assert all(len(r["content_sha256"]) == 64 for r in recs)
    d = usage.identical_samples(runs)
    assert (d["cells"], d["identical"], d["by_hash"]) == (2, 1, 2)
    assert d["examples"] == ["gap:original samples 0, 2"]
    for p in (runs / "gap").glob("anonymized*.json"):           # files without content_sha256: parsed answer and completion tokens
        r = json.loads(p.read_text())
        del r["content_sha256"]
        r["answer"] = a
        p.write_text(json.dumps(r))
    d = usage.identical_samples(runs)
    assert (d["cells"], d["identical"], d["by_hash"]) == (2, 2, 1)
    assert d["share"] == 1.0 and d["stop"] and usage.main(["--identical-check", str(runs)]) == 3
    for p in (runs / "gap").glob("*.json"):                           # one cell left, without identical samples
        if p.name.startswith("anonymized"):
            p.unlink()
    (runs / "gap" / "original.s2.json").unlink()
    assert not usage.identical_samples(runs)["stop"] and usage.main(["--identical-check", str(runs)]) == 0


def test_cost_projection_rebuilds_the_prompts_that_were_sent(tmpl, tmp_path):
    class CountingLLM(ScriptedLLM):
        def chat(self, messages, tag="", json_mode=True, sample=0):
            out = super().chat(messages, tag, json_mode, sample)
            out["usage"] = {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                            "completion_tokens": 100 + sample}
            return out

    answer = {"joint_supports": [], "decompose": False, "candidates": []}
    runs = tmp_path / "support_templates" / "cfg_y"
    recs = support_run.run_instance(CountingLLM([answer] * 16), "clsp", str(tmpl / "clsp" / "model.mps"),
                                    str(tmpl / "clsp"), usage.CONDITIONS, runs, samples=2)
    s = summarize(read_model(str(tmpl / "clsp" / "model.mps")))
    for r in recs:
        assert usage.planned_messages(s, str(tmpl / "clsp"), r["condition"]) == r["messages"]
    stats = usage.used(runs)
    assert stats["calls"] == len(recs) == 8 and abs(stats["tokens_per_char"] - 0.25) < 0.01
    (tmp_path / "oracle" / "clsp").mkdir(parents=True)
    (tmp_path / "oracle" / "clsp" / "oracle.json").write_text("{}")
    (tmp_path / "mps").mkdir()
    with open(tmpl / "clsp" / "model.mps", "rb") as src, gzip.open(tmp_path / "mps" / "clsp.mps.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    done = support_run.result_path(tmp_path / "miplib" / "cfg_y", "clsp", "original", 0)
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    p = usage.plan(tmp_path / "mps", tmpl, tmp_path / "oracle", tmp_path / "miplib" / "cfg_y", 2, usage.CONDITIONS)
    assert p["instances"] == 1 and p["calls"] == 7


def test_4000_added_tokens_on_a_13000_character_prompt_are_rejected(monkeypatch, tmp_path):
    from llmdw.providers.llm_client import PROMPT_TOKENS_PER_CHAR_MAX, PromptInflation

    chars = 13000
    base = round(0.4051 * chars)           # a ratio below the default factor
    usage = {"prompt_tokens": base + 4000}

    def fake_post(self, body):
        return {"model": "m", "usage": dict(usage),
                "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.delenv("LLMDW_ALLOW_PROMPT_INFLATION", raising=False)
    monkeypatch.delenv("LLMDW_PROMPT_TOKENS_PER_CHAR", raising=False)   # the default factor
    assert PROMPT_TOKENS_PER_CHAR_MAX == 0.65
    msg = [{"role": "user", "content": "x" * chars}]
    with pytest.raises(PromptInflation):
        _client(tmp_path).chat(msg)
    usage["prompt_tokens"] = base                                       # the same prompt without added text
    assert not _client(tmp_path).chat(msg)["cached"]


def test_4000_added_tokens_are_rejected_at_a_configuration_factor_of_045(monkeypatch, tmp_path):
    from llmdw.providers.llm_client import PromptInflation, prompt_tokens_per_char

    chars = 13000
    base = round(0.25 * chars)             # a typical ratio of prompt tokens to characters
    usage = {"prompt_tokens": base + 4000}  # 7,250 tokens against a limit of 0.45 * 13,000 + 300 = 6,150

    def fake_post(self, body):
        return {"model": "m", "usage": dict(usage),
                "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.delenv("LLMDW_ALLOW_PROMPT_INFLATION", raising=False)
    monkeypatch.setenv("LLMDW_PROMPT_TOKENS_PER_CHAR", "0.45")
    assert prompt_tokens_per_char() == 0.45
    msg = [{"role": "user", "content": "x" * chars}]
    with pytest.raises(PromptInflation):
        _client(tmp_path).chat(msg)
    usage["prompt_tokens"] = base
    assert not _client(tmp_path).chat(msg)["cached"]


def test_endpoint_added_text_is_rejected_and_never_cached(monkeypatch, tmp_path):
    from llmdw.providers.llm_client import PromptInflation

    usage = {"prompt_tokens": 22519}

    def fake_post(self, body):
        return {"model": "m", "usage": dict(usage),
                "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.delenv("LLMDW_ALLOW_PROMPT_INFLATION", raising=False)
    msg = [{"role": "user", "content": "Is 391 a prime number? Answer with one JSON object."}]
    with pytest.raises(PromptInflation):
        _client(tmp_path).chat(msg)
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").glob("*.json"))
    assert "prompt_inflation" in (tmp_path / "calls.jsonl").read_text()
    usage["prompt_tokens"] = 40                                         # a clean channel: answered and cached
    assert not _client(tmp_path).chat(msg)["cached"] and _client(tmp_path).chat(msg)["cached"]
    entry = next((tmp_path / "cache").glob("*.json"))                   # an inflated entry in the cache
    data = json.loads(entry.read_text())
    data["response"]["usage"]["prompt_tokens"] = 22519
    entry.write_text(json.dumps(data))
    out = _client(tmp_path).chat(msg)                                   # set aside and requested again
    assert not out["cached"] and (entry.parent / (entry.name + ".inflated")).exists()


def test_usage_audit_counts_calls_with_added_text(tmp_path):
    for i, pt in enumerate([250, 260, 255, 1500]):                      # the last call carries about 1250 extra tokens
        path = tmp_path / "cfg_y" / f"inst{i}" / "original.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"usage": {"prompt_tokens": pt, "completion_tokens": 10},
                                    "messages": [{"role": "user", "content": "x" * 1000}]}))
    s = usage.used(tmp_path / "cfg_y")
    assert s["calls_with_added_text"] == 1 and 1200 < s["largest_added_tokens"] < 1300


def test_smoke_report_flags_endpoint_added_text(monkeypatch, capsys):
    from llmdw.providers import smoke

    def fake_post(self, body):
        return {"model": "m", "usage": {"prompt_tokens": 22519, "completion_tokens": 9},
                "choices": [{"message": {"content": "{\"prime\": false, \"smallest_factor\": 17}"},
                             "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    for k, v in {"LLMDW_BASE_URL": "http://localhost:1/v1", "LLMDW_MODEL": "m", "LLMDW_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    assert smoke.main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["first"]["answer_correct"] and report["first"]["prompt_tokens_within_limit"] is False


class _FakeStream:
    def __init__(self, lines):
        self.lines = [ln.encode() for ln in lines]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_streamed_answer_is_put_back_together_and_cached_under_the_same_key(monkeypatch, tmp_path):
    bodies = []
    chunks = ['data: {"model": "m", "choices": [{"index": 0, "delta": {"reasoning_content": "391 = 17*23"}}]}',
              ': keep-alive',
              'data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "{\\"ok\\": "}}]}',
              'data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "true}"}, "finish_reason": "stop"}]}',
              'data: {"model": "m", "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 9}}',
              'data: [DONE]']

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _FakeStream(chunks)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("LLMDW_STREAM", raising=False)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    msg = [{"role": "user", "content": "Is 391 prime? Answer in JSON."}]
    out = _client(tmp_path).chat(msg)
    assert bodies[0]["stream"] is True and bodies[0]["stream_options"] == {"include_usage": True}
    assert out["content"] == '{"ok": true}' and out["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 12 and out["model"] == "m"
    entry = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert entry["response"]["choices"][0]["message"]["reasoning_content"] == "391 = 17*23"
    monkeypatch.setenv("LLMDW_STREAM", "0")                              # the same request without streaming
    assert _client(tmp_path).chat(msg)["cached"] and len(bodies) == 1


def test_stream_error_with_quota_message_stops_at_once(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        return _FakeStream(['data: {"error": {"message": "insufficient_quota"}}'])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.delenv("LLMDW_STREAM", raising=False)
    with pytest.raises(EndpointRejected):
        _client(tmp_path).chat([{"role": "user", "content": "x"}])


def test_stream_error_without_quota_is_retried_like_a_gateway_error(monkeypatch, tmp_path):
    """An error event inside the stream that names no quota is sent again (logged as gateway, stream_error); after
    the last attempt GatewayExhausted, which support_run counts as an INCOMPLETE call."""
    from llmdw.providers.llm_client import GatewayExhausted

    broken = ['data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "{\\"ok"}}]}',
              'data: {"error": {"message": "Upstream response stream was interrupted", "type": "upstream_error"}}']
    good = ['data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "{\\"ok\\": true}"}, "finish_reason": "stop"}]}',
            'data: {"model": "m", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}', 'data: [DONE]']
    replies = [broken, good]

    def fake_urlopen(req, timeout=None):
        return _FakeStream(replies.pop(0) if replies else broken)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.delenv("LLMDW_STREAM", raising=False)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.setenv("LLMDW_CONFIG", "cfg_x")
    out = _client(tmp_path).chat([{"role": "user", "content": "x"}])
    assert out["content"] == '{"ok": true}'
    retried = [json.loads(line) for line in (tmp_path / "llm_retries.jsonl").read_text().splitlines()]
    assert [(r["reason"], r["status"]) for r in retried] == [("gateway", "stream_error")]
    with pytest.raises(GatewayExhausted):                               # every attempt broken: no answer, run goes on
        _client(tmp_path, max_retries=3).chat([{"role": "user", "content": "y"}])


def test_empty_answer_counts_as_missing_not_as_a_crash():
    from llmdw.providers.llm_client import extract_json

    with pytest.raises(ValueError):
        extract_json("")
    with pytest.raises(ValueError):
        extract_json("no object here")


def test_incomplete_response_is_retried_never_cached_and_does_not_stop_the_run(monkeypatch, tmp_path, tmpl):
    from llmdw.providers.llm_client import IncompleteResponse

    replies = []

    def fake_post(self, body):
        replies.append(1)
        return {"model": "m", "usage": {"prompt_tokens": 10},
                "choices": [{"message": {"content": "", "reasoning_content": "thinking"}, "finish_reason": "length"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.setenv("LLMDW_INCOMPLETE_RETRIES", "1")
    with pytest.raises(IncompleteResponse):
        _client(tmp_path).chat([{"role": "user", "content": "x"}])
    assert len(replies) == 2 and not list((tmp_path / "cache").glob("*.json")) if (tmp_path / "cache").exists() else True
    # in a run, the call writes no file and the instance goes on with its other calls
    monkeypatch.setenv("LLMDW_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLMDW_MODEL", "m")
    llm = _client(tmp_path / "run")
    out = tmp_path / "answers"
    support_run.run_instance(llm, "gap", str(tmpl / "gap" / "model.mps"), str(tmpl / "gap"),
                             ["original", "anonymized"], out, samples=1)
    assert not (out / "gap").exists() or not list((out / "gap").iterdir())


def test_cut_off_cache_entry_is_set_aside(monkeypatch, tmp_path):
    def fake_post(self, body):
        return {"model": "m", "usage": {}, "choices": [{"message": {"content": "{\"ok\": 1}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    msg = [{"role": "user", "content": "x"}]
    _client(tmp_path).chat(msg)
    entry = next((tmp_path / "cache").glob("*.json"))
    data = json.loads(entry.read_text())
    data["response"]["choices"][0]["finish_reason"] = None             # a cut-off stream in the cache
    entry.write_text(json.dumps(data))
    assert not _client(tmp_path).chat(msg)["cached"]
    assert (entry.parent / (entry.name + ".incomplete")).exists()


def _answer(pt, reasoning=None, reasoning_tokens=None):
    usage = {"prompt_tokens": pt}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    msg = {"content": "{\"ok\": true}"}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {"model": "m", "usage": usage, "choices": [{"message": msg, "finish_reason": "stop"}]}


def test_required_reasoning_rejects_an_answer_without_reasoning(monkeypatch, tmp_path):
    from llmdw.providers.llm_client import IncompleteResponse

    replies = [_answer(40), _answer(40, reasoning="thinking ...")]

    def fake_post(self, body):
        return replies.pop(0)

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.setenv("LLMDW_REQUIRE_REASONING", "1")
    msg = [{"role": "user", "content": "x"}]
    out = _client(tmp_path).chat(msg)                                   # first reply had no reasoning: sent again
    assert not out["cached"] and not replies
    assert "no reasoning" in (tmp_path / "calls.jsonl").read_text()
    entry = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert entry["response"]["choices"][0]["message"]["reasoning_content"] == "thinking ..."
    replies[:] = [_answer(40), _answer(40), _answer(40)]                # never any reasoning: no answer, nothing cached
    with pytest.raises(IncompleteResponse, match="no reasoning"):
        _client(tmp_path).chat([{"role": "user", "content": "y"}])
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    replies[:] = [_answer(40, reasoning_tokens=12)]                     # reasoning tokens alone also count
    assert not _client(tmp_path).chat([{"role": "user", "content": "z"}])["cached"]


def test_samples_of_one_request_must_report_the_same_prompt_tokens(monkeypatch, tmp_path):
    replies = [_answer(100), _answer(179), _answer(103)]

    def fake_post(self, body):
        return replies.pop(0)

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.delenv("LLMDW_REQUIRE_REASONING", raising=False)
    monkeypatch.delenv("LLMDW_REFERENCE_PROMPT_TOKENS_FROM", raising=False)
    msg = [{"role": "user", "content": "x"}]
    _client(tmp_path).chat(msg, sample=0)                               # 100 accepted and kept as the reference
    out = _client(tmp_path).chat(msg, sample=1)                         # 179 rejected (differs by 79), 103 accepted
    assert (out["usage"] or {})["prompt_tokens"] == 103 and not replies
    assert "prompt tokens 179 differ from 100" in (tmp_path / "calls.jsonl").read_text()


def test_reference_prompt_tokens_from_another_configuration(monkeypatch, tmp_path):
    replies = [_answer(100), _answer(179), _answer(102)]

    def fake_post(self, body):
        return replies.pop(0)

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_REQUIRE_REASONING", raising=False)
    msg = [{"role": "user", "content": "x"}]
    monkeypatch.setenv("LLMDW_CONFIG", "cfg_a")                         # one configuration answers first
    _client(tmp_path, extra_body={"thinking": {"type": "disabled"}}).chat(msg)
    monkeypatch.setenv("LLMDW_CONFIG", "cfg_b")
    monkeypatch.setenv("LLMDW_REFERENCE_PROMPT_TOKENS_FROM", "cfg_a")
    out = _client(tmp_path, extra_body={"thinking": {"type": "enabled"}}).chat(msg)   # another request, same messages
    assert (out["usage"] or {})["prompt_tokens"] == 102 and not replies
    assert "differ from 100 of the accepted answer to the same configuration cfg_a" in (tmp_path / "calls.jsonl").read_text()


def test_inflation_retry_mode_retries_and_stops_on_a_window(monkeypatch, tmp_path):
    """With LLMDW_INFLATION_MODE=retry an inflated answer is sent again like an incomplete one (reason inflation
    in llm_retries.jsonl, nothing cached), and the run stops when more than 5 of the last 20 paid calls were inflated."""
    from llmdw.providers.llm_client import IncompleteResponse, PromptInflation

    seq = []

    def fake_post(self, body):
        pt = seq.pop(0) if seq else 10
        return {"model": "m", "usage": {"prompt_tokens": pt},
                "choices": [{"message": {"content": "{\"ok\": true}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.setenv("LLMDW_PROMPT_TOKENS_PER_CHAR", "0.45")
    monkeypatch.setenv("LLMDW_INFLATION_MODE", "retry")
    monkeypatch.setenv("LLMDW_INCOMPLETE_RETRIES", "2")
    monkeypatch.setenv("LLMDW_CONFIG", "cfg_x")
    c = _client(tmp_path)
    seq[:] = [5000, 10]                                              # inflated, then clean
    assert c.chat([{"role": "user", "content": "a"}])["content"] == "{\"ok\": true}"
    retried = [json.loads(line) for line in (tmp_path / "llm_retries.jsonl").read_text().splitlines()]
    assert [r["reason"] for r in retried] == ["inflation"]
    seq[:] = [5000, 5000, 5000]                                      # every attempt inflated: no answer, run goes on
    with pytest.raises(IncompleteResponse):
        c.chat([{"role": "user", "content": "b"}])
    cached = [json.loads(p.read_text())["response"]["usage"]["prompt_tokens"] for p in (tmp_path / "cache").glob("*.json")]
    assert cached == [10]                                            # only the clean answer of "a" is cached
    seq[:] = [5000, 5000, 5000]                                      # 4 + 2 = 6 inflated of the last 8 paid calls
    with pytest.raises(PromptInflation, match="6 of the last"):
        c.chat([{"role": "user", "content": "c"}])
    monkeypatch.delenv("LLMDW_INFLATION_MODE")                      # default unchanged: the first inflated answer stops
    seq[:] = [5000]
    with pytest.raises(PromptInflation):
        _client(tmp_path).chat([{"role": "user", "content": "d"}])


def test_a_stop_file_starts_no_new_call(tmpl, monkeypatch, tmp_path, capsys):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        raise AssertionError("no call may start once the stop file exists")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    for k, v in {"LLMDW_BASE_URL": "http://localhost:1/v1", "LLMDW_API_KEY": "k", "LLMDW_MODEL": "m",
                 "LLMDW_CACHE_DIR": str(tmp_path / "cache"), "LLMDW_LOG": str(tmp_path / "calls.jsonl"),
                 "LLMDW_STREAM": "0", "LLMDW_TEMPERATURE": "none"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    stop = tmp_path / "STOP"
    stop.write_text("")
    out = tmp_path / "out"
    support_run.main(["--templates", str(tmpl), "--conditions", "original", "--samples", "1", "--parallel", "2",
                      "--out", str(out), "--stop-file", str(tmp_path / "absent"), "--stop-file", str(stop)])
    lines = capsys.readouterr().out.splitlines()
    assert not calls and not out.exists() or not any(out.rglob("*.json"))
    assert any(line.startswith("STOP FILE") for line in lines) and lines[-1] == "all 2 instances finished"


def test_malformed_stream_chunk_is_retried_like_a_gateway_error(monkeypatch, tmp_path):
    """A data line cut off inside the stream (invalid JSON) is a stream error: sent again, logged as gateway."""
    broken = ['data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "{\\"ok"}}]}',
              'data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "unterminated']
    good = ['data: {"model": "m", "choices": [{"index": 0, "delta": {"content": "{\\"ok\\": true}"}, "finish_reason": "stop"}]}',
            'data: {"model": "m", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}', 'data: [DONE]']
    replies = [broken, good]

    def fake_urlopen(req, timeout=None):
        return _FakeStream(replies.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.delenv("LLMDW_STREAM", raising=False)
    monkeypatch.delenv("LLMDW_EXTRA_BODY", raising=False)
    monkeypatch.setenv("LLMDW_CONFIG", "cfg_x")
    out = _client(tmp_path).chat([{"role": "user", "content": "x"}])
    assert out["content"] == '{"ok": true}'
    retried = [json.loads(line) for line in (tmp_path / "llm_retries.jsonl").read_text().splitlines()]
    assert [(r["reason"], r["status"]) for r in retried] == [("gateway", "stream_error")]
