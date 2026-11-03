"""OpenAI-compatible chat client with a request cache, a call log and a model guard.

Paper: Appendix D: the requests, the cache and the checks of every response.

Configuration (environment variables)
-------------------------------------
LLMDW_BASE_URL     base URL of an OpenAI-compatible endpoint, e.g. https://<endpoint>/v1
LLMDW_API_KEY      API key of the endpoint
LLMDW_MODEL        model string sent with every request (pin it; every answer is tied to it)
LLMDW_TEMPERATURE  sampling temperature (default 0); "none" omits the field for models that reject it
LLMDW_MAX_TOKENS   optional completion token cap (leave unset for reasoning models)
LLMDW_JSON_MODE    1 (default) sends response_format=json_object; 0 omits it (answers are parsed from text)
LLMDW_EXTRA_BODY   optional JSON object merged into every request body, used to pin the reasoning setting, e.g.
                   '{"reasoning_effort": "medium"}' or '{"thinking": {"type": "disabled"}}'; it is part of the
                   request, so it is part of the cache key
LLMDW_CACHE_DIR    request cache directory (default .llm_cache); identical requests are answered from it.
                   chat(..., sample=i) with i >= 1 draws an independent answer to the same request: the sample
                   index is part of the cache key (not of the request), and sample 0 has the key of the request
                   alone. LLMDW_PROMPT_TAG, when set, is also part of the cache key (support_run sets it to the
                   user prompt file of the ranked and with-data prompts)
LLMDW_LOG          call log, one JSON line per call (default runs/llm_calls.jsonl)
LLMDW_ALLOW_MODEL_MISMATCH=1  accept a response whose "model" field is not LLMDW_MODEL (optionally followed
                              by a date suffix such as -YYYY-MM-DD); by default such a response raises
                              ModelMismatch: an endpoint that substitutes another model silently would make
                              the answers incomparable

Persistence
-----------
A response is written to the cache before chat() returns, through a temporary file that is flushed to disk and
then renamed, so a crash leaves either no entry or a complete one. An entry that cannot be parsed is moved to
<name>.corrupt and the request is sent again. HTTP 401, 402 and 403,
and HTTP 429 with a quota message, raise EndpointRejected at once: retrying does not fix a wrong key, a key outside
the model's group or a used-up quota. The run stops, and a resume after the cause is fixed repeats no call that
was answered.

Prompt check
------------
An endpoint can add text of its own in front of a request, and the model then reads it. Every response is therefore
checked against the request: when its usage reports more than LLMDW_PROMPT_TOKENS_PER_CHAR (default
PROMPT_TOKENS_PER_CHAR_MAX) prompt tokens per character of our messages plus PROMPT_TOKENS_SLACK, the answer is
neither used nor cached, and PromptInflation is
raised. A cached entry that fails the check is moved to <name>.inflated and requested again. The limit lies far
above the ratio of English text, formulas and numbers in the tokenizers used, so only added text exceeds it.
LLMDW_ALLOW_PROMPT_INFLATION=1 switches the check off, for diagnosis only.
LLMDW_INFLATION_MODE=retry (default: stop) treats such an answer like an incomplete one instead: not cached,
sent again up to LLMDW_INCOMPLETE_RETRIES times, then IncompleteResponse (an INCOMPLETE line in support_run, and the run
goes on); each retried attempt is logged with reason "inflation". The run still stops (PromptInflation, a STOPPED line)
when more than INFLATION_WINDOW_LIMIT of the client's last INFLATION_WINDOW paid calls were rejected for inflation.

Streaming
---------
Requests are sent with stream=true and stream_options.include_usage=true, and the streamed chunks are put back
together into the response shape of a non-streamed call (content, reasoning_content, finish_reason, model, usage).
A streamed answer keeps the connection busy, so the endpoint or an upstream proxy does not cut off a call that thinks
for minutes (an upstream proxy can answer HTTP 524 after about 100 s otherwise). The stream fields are added only to the HTTP
request, never to the cache key, so streamed and non-streamed answers share the cache. LLMDW_STREAM=0 sends
non-streamed requests. LLMDW_TIMEOUT sets the socket timeout in seconds (default 600); with streaming it bounds
the silence between two chunks, not the length of a call. An error event inside the stream that names a quota raises
EndpointRejected; any other (for example "Upstream response stream was interrupted") is retried like a gateway error,
logged with reason gateway and status stream_error, and ends in GatewayExhausted after the last attempt.

Incomplete responses
--------------------
A response without a finish_reason (a stream cut off by the endpoint) or with empty content (the thinking used up the
completion limit) holds no answer. It is not cached; the request is sent again up to LLMDW_INCOMPLETE_RETRIES times
(default 2), and then IncompleteResponse is raised. A cached entry of that kind is moved to <name>.incomplete and
requested again.

Run-time guards
---------------
An answer is rejected like an incomplete one (not cached, sent again, IncompleteResponse after the retries) when
LLMDW_REQUIRE_REASONING=1 and it shows neither reasoning tokens nor reasoning text, or when its prompt tokens differ
by more than PROMPT_TOKENS_REFERENCE_TOLERANCE (or LLMDW_PROMPT_TOKENS_REFERENCE_TOLERANCE) from those of the first accepted answer to the same request (the
request hash without the sample index) or, with LLMDW_REFERENCE_PROMPT_TOKENS_FROM=<configuration>, to the same
messages in that configuration (LLMDW_CONFIG). The reference counts are kept in <cache>/prompt_tokens/. A cached entry
that fails a guard is moved to <name>.rejected and requested again.

Retry log
---------
Every retried attempt (an HTTP code that is retried, a network error or timeout, or an incomplete answer that is
sent again) adds one line to llm_retries.jsonl next to the call log: time, tag, request_sha256, model, config
(LLMDW_CONFIG), status (HTTP code or exception class), reason (gateway, rate_limit, network,
incomplete, inflation or other) and the seconds since that attempt started. The call log keeps one record per answer or
incomplete answer.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class ModelMismatch(RuntimeError):
    pass


class EndpointRejected(RuntimeError):
    """The endpoint refused the request for a reason that retrying does not fix."""


class IncompleteResponse(RuntimeError):
    """The response holds no answer: it has no finish_reason or its content is empty."""


class GatewayExhausted(RuntimeError):
    """Every attempt failed and the last one with a gateway error (HTTP 520, 522, 523, 524, or an error event
    inside a stream that does not name a quota) or a network error or timeout: the call has no answer, like an
    incomplete response, and a later run sends it again."""


class StreamError(RuntimeError):
    """An error event inside a stream that does not name a quota (for example "Upstream response stream was
    interrupted"); _post retries it like a gateway error."""


_GATEWAY_CODES = (520, 522, 523, 524)


def incomplete_reason(resp) -> str | None:
    choice = ((resp.get("choices") or [{}])[0] if isinstance(resp, dict) else {}) or {}
    content = (choice.get("message") or {}).get("content") or ""
    if not choice.get("finish_reason"):
        return "no finish_reason (the response was cut off)"
    if not content.strip():
        return f"empty content (finish_reason {choice.get('finish_reason')})"
    return None


def reasoning_present(resp) -> bool:
    """The answer shows reasoning: reasoning tokens in the usage, or reasoning text in the message."""
    if not isinstance(resp, dict):
        return False
    usage = resp.get("usage") or {}
    if ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) > 0:
        return True
    if (usage.get("completion_thinking_tokens") or 0) > 0:
        return True
    msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
    return isinstance(msg.get("reasoning_content"), str) and bool(msg["reasoning_content"].strip())


PROMPT_TOKENS_REFERENCE_TOLERANCE = 5
# LLMDW_INFLATION_MODE=retry: the run stops when more than 5 of the last 20 paid calls were rejected for inflation
INFLATION_WINDOW, INFLATION_WINDOW_LIMIT = 20, 5


class PromptInflation(RuntimeError):
    """The response reports far more prompt tokens than the request holds: the endpoint added text."""


# Prompt-token limit per call, used by chat() and llmdw.providers.smoke: LLMDW_PROMPT_TOKENS_PER_CHAR prompt tokens per prompt
# character plus PROMPT_TOKENS_SLACK. The factor is set per configuration; the default is PROMPT_TOKENS_PER_CHAR_MAX.
AUTH_SCHEME = "Bearer"
PROMPT_TOKENS_PER_CHAR_MAX = 0.65
PROMPT_TOKENS_SLACK = 300


def prompt_tokens_per_char() -> float:
    """The factor of the prompt-token limit: LLMDW_PROMPT_TOKENS_PER_CHAR, or PROMPT_TOKENS_PER_CHAR_MAX when unset."""
    raw = os.environ.get("LLMDW_PROMPT_TOKENS_PER_CHAR", "").strip()
    return float(raw) if raw else PROMPT_TOKENS_PER_CHAR_MAX


def prompt_chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))


def inflated_prompt_tokens(resp: dict, chars: int) -> int | None:
    """The reported prompt tokens when they exceed the limit for a request of `chars` characters, else None."""
    pt = (resp.get("usage") or {}).get("prompt_tokens") if isinstance(resp, dict) else None
    if not isinstance(pt, (int, float)):
        return None
    return int(pt) if pt > prompt_tokens_per_char() * chars + PROMPT_TOKENS_SLACK else None


def _quota_message(text: str) -> bool:
    low = text.lower()
    return "quota" in low or "insufficient" in low or "\u989d\u5ea6" in text or "\u4f59\u989d" in text


_LOG_LOCK = threading.Lock()
# reason recorded in llm_retries.jsonl for each retried HTTP code (409 and 425 are recorded as "other")
_RETRY_REASON = {429: "rate_limit", 408: "network",
                 **{c: "gateway" for c in (500, 502, 503, 504, 520, 522, 523, 524)}}


class LLMClient:
    def __init__(self, base_url: str | None = None, key: str | None = None, model: str | None = None,
                 temperature: float | str | None = None, max_tokens: int | None = None,
                 cache_dir: str | None = None, log_path: str | None = None, timeout: float = 600.0,
                 max_retries: int = 6, extra_body: dict | None = None):
        self.base_url = (base_url or os.environ.get("LLMDW_BASE_URL", "")).rstrip("/")
        self.api_key = key or os.environ.get("LLMDW_API_KEY", "")
        self.model = model or os.environ.get("LLMDW_MODEL", "")
        t = temperature if temperature is not None else os.environ.get("LLMDW_TEMPERATURE", "0")
        self.temperature = None if str(t).lower() == "none" else float(t)
        mt = max_tokens if max_tokens is not None else os.environ.get("LLMDW_MAX_TOKENS")
        self.max_tokens = int(mt) if mt else None
        self.cache_dir = Path(cache_dir or os.environ.get("LLMDW_CACHE_DIR", ".llm_cache"))
        self.log_path = Path(log_path or os.environ.get("LLMDW_LOG", "runs/llm_calls.jsonl"))
        self.retry_log_path = self.log_path.with_name("llm_retries.jsonl")   # retried attempts, not in the call log
        self._ctx = threading.local()      # tag and request key of the call in progress, per thread, for _log_retry
        self.timeout = float(os.environ.get("LLMDW_TIMEOUT", timeout))
        self.max_retries = max_retries
        self.allow_mismatch = os.environ.get("LLMDW_ALLOW_MODEL_MISMATCH") == "1"
        self.allow_inflation = os.environ.get("LLMDW_ALLOW_PROMPT_INFLATION") == "1"
        self.inflation_retry = os.environ.get("LLMDW_INFLATION_MODE", "").strip().lower() == "retry"
        self._paid = collections.deque(maxlen=INFLATION_WINDOW)   # per paid call: rejected for inflation or not
        self._paid_lock = threading.Lock()
        if extra_body is None:
            raw_extra = os.environ.get("LLMDW_EXTRA_BODY", "").strip()
            extra_body = json.loads(raw_extra) if raw_extra else {}
        if not isinstance(extra_body, dict):
            raise ValueError("LLMDW_EXTRA_BODY must be a JSON object")
        self.extra_body = {k: v for k, v in extra_body.items() if k not in ("model", "messages")}
        if not self.base_url or not self.model:
            raise ValueError("set LLMDW_BASE_URL and LLMDW_MODEL (and LLMDW_API_KEY)")

    def _body(self, messages, json_mode: bool) -> dict:
        body = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if json_mode and os.environ.get("LLMDW_JSON_MODE", "1") != "0":   # LLMDW_JSON_MODE=0 for endpoints that reject it
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        return body

    def chat(self, messages: list[dict], tag: str = "", json_mode: bool = True, sample: int = 0) -> dict:
        """Send one chat request. Returns {"content", "model", "usage", "cached", "latency"}.

        sample >= 1 asks for another independent answer to the same request (a separate cache entry)."""
        body = self._body(messages, json_mode)
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
        prompt_tag = os.environ.get("LLMDW_PROMPT_TAG", "").strip()
        if prompt_tag:                          # the ranked and with-data prompts: their own cache keys
            raw += f"\n#prompt={prompt_tag}"
        if sample:
            raw += f"\n#sample={int(sample)}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        base_key = key if not sample else hashlib.sha256(raw.rsplit("\n#sample=", 1)[0].encode("utf-8")).hexdigest()
        msg_key = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        chars = prompt_chars(messages)
        t0 = time.time()
        entry = self._read_cache(cache_file, tag, key) if cache_file.exists() else None
        if entry is not None:
            resp = entry.get("response", entry) if isinstance(entry, dict) else entry
            if not self.allow_inflation and inflated_prompt_tokens(resp, chars) is not None:
                self._set_aside(cache_file, ".inflated", tag, key, "inflated_cache_entry_moved")
            elif incomplete_reason(resp) is not None:
                self._set_aside(cache_file, ".incomplete", tag, key, "incomplete_cache_entry_moved")
            elif self._guard_reason(resp, base_key, msg_key) is not None:
                self._set_aside(cache_file, ".rejected", tag, key,
                                f"rejected_cache_entry_moved: {self._guard_reason(resp, base_key, msg_key)}")
            else:
                # a cache hit is charged the latency of the original call, so that time budgets
                # (equal-budget comparisons) do not depend on whether a run was repeated
                out = self._unpack(resp, cached=True,
                                   latency=float(entry.get("latency", 0.0)) if isinstance(entry, dict) else 0.0)
                self._record_prompt_tokens(resp, base_key, msg_key)
                self._log(tag, key, out)
                return out
        retries = int(os.environ.get("LLMDW_INCOMPLETE_RETRIES", "2"))
        self._ctx.tag, self._ctx.key = tag, key
        for attempt in range(retries + 1):
            t0 = time.time()
            resp = self._post(body)
            latency = time.time() - t0
            out = self._unpack(resp, cached=False, latency=latency)
            pt = None if self.allow_inflation else inflated_prompt_tokens(resp, chars)
            if self.inflation_retry:
                self._count_paid(tag, key, out, inflated=pt is not None, chars=chars)
            why = incomplete_reason(resp) or self._guard_reason(resp, base_key, msg_key)
            if why is None and pt is not None and self.inflation_retry:
                self._log(tag, key, out, error="prompt_inflation")
                if attempt < retries:
                    self._log_retry("inflation", None, latency, detail=f"{pt} prompt tokens for {chars} characters")
                why = f"prompt inflation: {pt} prompt tokens for a request of {chars} characters"
                continue
            if why is not None:                 # checked first: a cut-off stream may lack the model field
                self._log(tag, key, out, error=f"incomplete: {why}")
                if attempt < retries:
                    self._log_retry("incomplete", None, latency, detail=why)
                continue
            returned = out["model"] or ""
            if not self.model_matches(returned) and not self.allow_mismatch:
                self._log(tag, key, out, error="model_mismatch")
                raise ModelMismatch(f"requested {self.model!r}, response model {returned!r}")
            if pt is not None:
                self._log(tag, key, out, error="prompt_inflation")
                raise PromptInflation(f"the response reports {pt} prompt tokens for a request of {chars} characters "
                                      f"(limit {prompt_tokens_per_char()} per character plus {PROMPT_TOKENS_SLACK}): "
                                      f"the endpoint added text that the model read, so the answer is not used")
            break
        else:
            raise IncompleteResponse(f"no answer after {retries + 1} attempts: {why}")
        self._write_cache(cache_file, {"response": resp, "latency": latency})
        self._record_prompt_tokens(resp, base_key, msg_key)
        self._log(tag, key, out)
        return out

    def _count_paid(self, tag: str, key: str, out: dict, inflated: bool, chars: int) -> None:
        """LLMDW_INFLATION_MODE=retry: record one paid call; stop the run when too many recent ones were inflated."""
        with self._paid_lock:
            self._paid.append(inflated)
            n = sum(self._paid)
        if inflated and n > INFLATION_WINDOW_LIMIT:
            self._log(tag, key, out, error="prompt_inflation")
            raise PromptInflation(f"{n} of the last {len(self._paid)} paid calls were rejected for prompt inflation "
                                  f"(limit {INFLATION_WINDOW_LIMIT} of {INFLATION_WINDOW}; this request: {chars} "
                                  f"characters, limit {prompt_tokens_per_char()} per character plus "
                                  f"{PROMPT_TOKENS_SLACK}): the endpoint keeps adding text")

    # ------------------------------------------------------------------ run-time guards (reasoning, prompt tokens)

    def _guard_reason(self, resp: dict, base_key: str, msg_key: str) -> str | None:
        """Why an answer is rejected like an incomplete one, or None: no reasoning although the configuration
        requires it, or prompt tokens that differ from those of an earlier accepted answer to the same request
        (or, with LLMDW_REFERENCE_PROMPT_TOKENS_FROM, to the same messages of that configuration)."""
        if os.environ.get("LLMDW_REQUIRE_REASONING") == "1" and not reasoning_present(resp):
            return "no reasoning (neither reasoning tokens nor reasoning text; LLMDW_REQUIRE_REASONING=1)"
        pt = (resp.get("usage") or {}).get("prompt_tokens") if isinstance(resp, dict) else None
        if not isinstance(pt, (int, float)):
            return None
        refs = [("request", self._prompt_ref_path("request", base_key))]
        other = os.environ.get("LLMDW_REFERENCE_PROMPT_TOKENS_FROM", "").strip()
        if other:
            refs.append((f"configuration {other}", self._prompt_ref_path(f"messages/{other}", msg_key)))
        for label, path in refs:
            ref = self._read_prompt_ref(path)
            tol = int(os.environ.get("LLMDW_PROMPT_TOKENS_REFERENCE_TOLERANCE", PROMPT_TOKENS_REFERENCE_TOLERANCE))
            if ref is not None and abs(pt - ref) > tol:
                return (f"prompt tokens {int(pt)} differ from {ref} of the accepted answer to the same {label} "
                        f"(tolerance {tol})")
        return None

    def _prompt_ref_path(self, kind: str, key: str) -> Path:
        return self.cache_dir / "prompt_tokens" / kind / f"{key}.json"

    @staticmethod
    def _read_prompt_ref(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))["prompt_tokens"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _record_prompt_tokens(self, resp: dict, base_key: str, msg_key: str) -> None:
        """Keep the prompt tokens of the first accepted answer per request (and per messages and configuration)."""
        pt = (resp.get("usage") or {}).get("prompt_tokens") if isinstance(resp, dict) else None
        if not isinstance(pt, (int, float)):
            return
        paths = [self._prompt_ref_path("request", base_key)]
        config = os.environ.get("LLMDW_CONFIG", "").strip()
        if config:
            paths.append(self._prompt_ref_path(f"messages/{config}", msg_key))
        for path in paths:
            if not path.exists():
                self._write_cache(path, {"prompt_tokens": int(pt)})

    def _read_cache(self, path: Path, tag: str, key: str):
        """The cached entry, or None after moving an entry that cannot be parsed to <name>.corrupt."""
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            self._set_aside(path, ".corrupt", tag, key, "corrupt_cache_entry_moved")
            return None

    def _set_aside(self, path: Path, suffix: str, tag: str, key: str, error: str) -> None:
        try:
            os.replace(path, path.with_name(path.name + suffix))
        except OSError:
            pass
        self._log(tag, key, {}, error=error)

    @staticmethod
    def _write_cache(path: Path, entry: dict) -> None:
        """Write through a temporary file in the same directory, flushed to disk, then renamed over the entry."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def model_matches(self, returned: str) -> bool:
        """Exact model id, optionally followed by a date suffix (-YYYY-MM-DD or -YYYYMMDD)."""
        return re.fullmatch(re.escape(self.model) + r"(-\d{4}-?\d{2}-?\d{2})?", returned or "") is not None

    def _post(self, body: dict) -> dict:
        stream = os.environ.get("LLMDW_STREAM", "1") != "0"
        payload = dict(body, stream=True, stream_options={"include_usage": True}) if stream else body
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last, last_gateway = None, False
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"{AUTH_SCHEME} {self.api_key}"})
            t_attempt = time.time()
            retried = attempt < self.max_retries - 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    if stream:
                        return _read_stream(r)
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                text = e.read()[:500].decode("utf-8", "replace")
                last, last_gateway = f"HTTP {e.code}: {text}", e.code in _GATEWAY_CODES
                if e.code in (401, 402, 403) or (e.code == 429 and _quota_message(text)):
                    raise EndpointRejected(f"{last} | the endpoint rejected the key, its group or its quota; fix the cause "
                                        f"and resume, answered calls are in the cache") from e
                # 520-524: gateway errors of the endpoint's proxy (524 = upstream timeout); they carry no answer
                if e.code not in (408, 409, 425, 429, 500, 502, 503, 504) + _GATEWAY_CODES:
                    raise RuntimeError(last) from e
                if retried:
                    self._log_retry(_RETRY_REASON.get(e.code, "other"), e.code, time.time() - t_attempt)
            except StreamError as e:                 # an error event inside the stream: like a gateway error
                last, last_gateway = str(e), True
                if retried:
                    self._log_retry("gateway", "stream_error", time.time() - t_attempt, detail=str(e)[:300])
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last, last_gateway = repr(e), True
                if retried:
                    self._log_retry("network", type(e).__name__, time.time() - t_attempt)
            time.sleep(min(120, 2 ** attempt * 5))
        if last_gateway:
            raise GatewayExhausted(f"no answer after {self.max_retries} attempts: {last}")
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last}")

    @staticmethod
    def _unpack(resp: dict, cached: bool, latency: float) -> dict:
        choice = (resp.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        return {"content": content, "model": resp.get("model"), "usage": resp.get("usage"),
                "finish_reason": choice.get("finish_reason"), "cached": cached, "latency": round(latency, 3)}

    def _log(self, tag: str, key: str, out: dict, error: str | None = None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "request_sha256": key,
               "model_requested": self.model, "model_returned": out.get("model"), "usage": out.get("usage"),
               "finish_reason": out.get("finish_reason"), "cached": out.get("cached"), "latency": out.get("latency")}
        if error:
            rec["error"] = error
        with _LOG_LOCK, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _log_retry(self, reason: str, status, seconds: float, detail: str | None = None) -> None:
        """One line per retried attempt in llm_retries.jsonl (status: HTTP code or exception class)."""
        self.retry_log_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": getattr(self._ctx, "tag", ""),
               "request_sha256": getattr(self._ctx, "key", ""), "model": self.model,
               "config": os.environ.get("LLMDW_CONFIG") or None, "status": status, "reason": reason,
               "seconds": round(seconds, 3)}
        if detail:
            rec["detail"] = detail
        with _LOG_LOCK, open(self.retry_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_stream(lines) -> dict:
    """Put the chunks of a streamed chat completion back together into one non-streamed response."""
    content, reasoning, model, finish, usage, role = [], [], None, None, None, None
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:            # a chunk cut off inside the stream: the stream was interrupted
            raise StreamError(f"malformed stream chunk: {data[:200]}") from None
        if chunk.get("error"):
            text = json.dumps(chunk["error"], ensure_ascii=False)[:500]
            if _quota_message(text):
                raise EndpointRejected(f"stream error: {text} | the endpoint rejected the key, its group or its quota")
            raise StreamError(f"stream error: {text}")
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            role = delta.get("role") or role
            if delta.get("content"):
                content.append(delta["content"])
            part = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(part, str):
                reasoning.append(part)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    message = ({"role": role} if role else {}) | {"content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage,
            "streamed": True}


def extract_json(text: str, required=()) -> dict:
    """The answer object of a model response (one rule for every configuration): among the top-level JSON
    objects in the text (prose and code fences around them are skipped), the last one that has every key in
    `required`; with no required keys, the last object. A brace that does not start a valid JSON object (prose such
    as "{T}", or a malformed object) is skipped. ValueError when no object qualifies."""
    dec = json.JSONDecoder()
    found = []
    i = text.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found.append(obj)
        i = text.find("{", end)
    good = [o for o in found if all(k in o for k in required)]
    if not good:
        raise ValueError(f"no JSON object with the keys {', '.join(required)} in the answer" if required and found
                         else "no JSON object in the answer")
    return good[-1]
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object in the answer")
