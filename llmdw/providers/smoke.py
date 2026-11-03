"""Smoke test of one model configuration (the LLMDW_* variables of the current shell), without the cache.

Paper: Appendix D: the short check of a configuration before its runs.

It sends the same short question twice and prints one JSON report:
- returned model id and whether it matches LLMDW_MODEL (the guard of llm_client)
- the request settings actually sent (temperature, response_format, extra body)
- finish reason, latency, usage, reasoning token count when the endpoint reports one, and the length of
  reasoning_content when the endpoint returns it (thinking modes)
- whether the answer is correct (391 = 17 * 23)
- whether the two answers look like an endpoint-side cache hit (identical content and usage, second call < 1 s)
- whether the reported prompt tokens stay within the limit of llm_client (no text added by the endpoint)

Exit code 0 when the model id matches, both answers are correct and complete, no cache hit is suspected and the
prompt tokens of both calls stay within the limit.

Usage
-----
python -m llmdw.providers.smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time

from llmdw.providers.llm_client import LLMClient, extract_json, inflated_prompt_tokens, prompt_chars

PROMPT = ("Is 391 a prime number? Answer with one JSON object of the form "
          '{"prime": true or false, "smallest_factor": integer or null} and nothing else.')


def _one(client: LLMClient, body: dict) -> dict:
    t0 = time.time()
    resp = client._post(body)
    latency = time.time() - t0
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = resp.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    content = msg.get("content") or ""
    try:
        ans = extract_json(content)
        correct = ans.get("prime") is False and ans.get("smallest_factor") == 17
    except (ValueError, json.JSONDecodeError, AttributeError):
        correct = False
    return {"returned_model": resp.get("model"), "finish_reason": choice.get("finish_reason"),
            "latency_s": round(latency, 2), "usage": usage, "reasoning_tokens": details.get("reasoning_tokens"),
            "reasoning_content_chars": len(msg.get("reasoning_content") or ""), "message_keys": sorted(msg),
            "content": content[:200], "answer_correct": correct,
            "prompt_tokens_within_limit": inflated_prompt_tokens(resp, prompt_chars(body["messages"])) is None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="model identifier of a configuration of config/public_settings.json")
    args = ap.parse_args(argv)
    if args.config:
        import os

        from llmdw.providers import settings
        os.environ.update(settings.environment(args.config))
    client = LLMClient(cache_dir=tempfile.mkdtemp(prefix="llmdw_smoke_"))
    body = client._body([{"role": "user", "content": PROMPT}], json_mode=True)
    sent = {k: v for k, v in body.items() if k != "messages"}
    first, second = _one(client, body), _one(client, body)
    cache_suspected = (first["content"] == second["content"] and first["usage"] == second["usage"]
                       and second["latency_s"] < 1.0)
    report = {"requested_model": client.model, "request_settings": sent,
              "model_matches": all(client.model_matches(r["returned_model"] or "") for r in (first, second)),
              "endpoint_cache_suspected": cache_suspected, "first": first, "second": second}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    ok = (report["model_matches"] and not cache_suspected
          and all(r["answer_correct"] and r["finish_reason"] == "stop" and r["prompt_tokens_within_limit"]
                  for r in (first, second)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
