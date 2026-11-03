"""The five language-model configurations of the paper (Section 6 and Appendix D), read from config/public_settings.json.

Each entry gives the official name, the provider, the model identifier sent to the endpoint, the additional request
fields (extra_body), the number of samples and the client settings, and the names of the environment variables that
hold the key and the base URL. The model identifier is the key of a configuration in the data (one answer directory
per configuration); tables print the official name.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "public_settings.json"
PAIR_NAMES = ("Grok 4.20 (Reasoning)", "Grok 4.20 (Non-Reasoning)")      # the same model with and without reasoning


def load(path: Path | str | None = None) -> list[dict]:
    return json.loads(Path(path or DEFAULT).read_text(encoding="utf-8"))["configurations"]


def pair(path: Path | str | None = None) -> tuple[str, str]:
    """The model identifiers of the configurations of PAIR_NAMES."""
    by_name = {c["name"]: c["model"] for c in load(path)}
    return by_name[PAIR_NAMES[0]], by_name[PAIR_NAMES[1]]


def names(path: Path | str | None = None) -> dict:
    """{model identifier: official name}, in the order of the paper."""
    return {c["model"]: c["name"] for c in load(path)}


def environment(model: str, path: Path | str | None = None) -> dict:
    """The LLMDW_* variables that make llmdw.providers.llm_client send the requests of one configuration, with the
    key and the base URL taken from the variables the entry names (empty when they are not set)."""
    import os

    c = next(x for x in load(path) if x["model"] == model)
    return {"LLMDW_MODEL": c["model"], "LLMDW_EXTRA_BODY": json.dumps(c["extra_body"]) if c["extra_body"] else "",
            "LLMDW_STREAM": "1" if c["streaming"] else "0", "LLMDW_JSON_MODE": "1" if c["json_mode"] else "0",
            "LLMDW_TEMPERATURE": "none" if c["temperature"] is None else str(c["temperature"]),
            "LLMDW_TIMEOUT": str(c["timeout_s"]), "LLMDW_MAX_TOKENS": "" if c["completion_limit"] is None
            else str(c["completion_limit"]), "LLMDW_PROMPT_TOKENS_PER_CHAR": str(c["prompt_tokens_per_char"]),
            "LLMDW_REQUIRE_REASONING": "1" if c["require_reasoning"] else "0",
            "LLMDW_INFLATION_MODE": c["inflation_mode"], "LLMDW_CONFIG": c["model"],
            "LLMDW_API_KEY": os.environ.get(c["api_key_env"], ""),
            "LLMDW_BASE_URL": os.environ.get(c["base_url_env"], "")}


def table_rows(path: Path | str | None = None) -> list[dict]:
    """The request settings per configuration, as rows of settings.csv."""
    rows = []
    for c in load(path):
        rows.append({"config": c["model"], "provider": c["provider"], "model": c["model"],
                     "extra_body": json.dumps(c["extra_body"], sort_keys=True) if c["extra_body"] else "none",
                     "samples": c["samples"], "streaming": c["streaming"], "json_mode": c["json_mode"],
                     "temperature": "none" if c["temperature"] is None else c["temperature"],
                     "timeout_s": c["timeout_s"],
                     "completion_limit": "none" if c["completion_limit"] is None else c["completion_limit"],
                     "prompt_tokens_per_char": c["prompt_tokens_per_char"],
                     "require_reasoning": c["require_reasoning"], "inflation_mode": c["inflation_mode"]})
    return rows
