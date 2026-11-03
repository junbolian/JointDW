"""Replay check of a configuration: resend recorded requests byte for byte and compare the prompt tokens.

Paper: Appendix D: the recorded requests sent again to compare the prompt tokens.

The request body of each selected answer file is rebuilt from its messages and the configuration's settings
(config/public_settings.json); its SHA-256 must equal the cache key of the recorded request in the released cache.
The body is then sent through llmdw.providers.llm_client without the response cache, into a temporary directory, so
nothing released is written. The check passes when every request is answered and the prompt tokens reported now
equal those of the recorded answer. It needs the key and base URL of the configuration (.env.example); the tests
skip it when they are not set.

Usage
-----
python -m llmdw.providers.replay --work data --config $MODEL --requests 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def selected(work: Path, model: str, n: int) -> list[dict]:
    """The n answer files of the original presentation and sample 0 with the shortest prompts (templates first)."""
    recs = []
    for phase in ("support_templates", "support_miplib"):
        for f in sorted((work / "work" / phase / model).glob("*/original.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if (r.get("usage") or {}).get("prompt_tokens"):
                recs.append(r)
    return sorted(recs, key=lambda r: sum(len(m["content"]) for m in r["messages"]))[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data")
    ap.add_argument("--config", required=True, help="model identifier of a configuration")
    ap.add_argument("--requests", type=int, default=3)
    args = ap.parse_args(argv)
    from llmdw.providers import settings
    os.environ.update(settings.environment(args.config))
    if not os.environ.get("LLMDW_API_KEY") or not os.environ.get("LLMDW_BASE_URL"):
        print("REPLAY SKIPPED: the key or the base URL of the configuration is not set")
        return 2
    from llmdw.providers.llm_client import LLMClient
    work = Path(args.work)
    tmp = Path(tempfile.mkdtemp(prefix="llmdw_replay_"))
    client = LLMClient(cache_dir=str(tmp / "cache"), log_path=str(tmp / "calls.jsonl"))
    ok = True
    for rec in selected(work, args.config, args.requests):
        body = client._body(rec["messages"], True)
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        recorded = (work / "llm_cache" / f"{key}.json").exists()
        resp = client._post(body)
        now = (resp.get("usage") or {}).get("prompt_tokens")
        then = rec["usage"]["prompt_tokens"]
        same = recorded and now == then
        ok &= same
        print(json.dumps({"instance": rec["instance"], "request_recorded": recorded, "prompt_tokens_recorded": then,
                          "prompt_tokens_now": now, "equal": same}))
    print("REPLAY PASS" if ok else "REPLAY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
