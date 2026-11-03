"""The replay check (llmdw.providers.replay) against an endpoint. It needs a key and a base URL in the environment
(.env.example) and the released data; without them the test is skipped."""
import os
from pathlib import Path

import pytest

from llmdw.providers import replay, settings

ROOT = Path(__file__).resolve().parents[1]
READY = [c for c in settings.load() if os.environ.get(c["api_key_env"]) and os.environ.get(c["base_url_env"])]


@pytest.mark.skipif(not READY or not (ROOT / "data" / "llm_cache").exists(),
                    reason="no key and base URL of any configuration in the environment, or no released data")
def test_replay_of_one_recorded_request():
    assert replay.main(["--work", str(ROOT / "data"), "--config", READY[0]["model"], "--requests", "1"]) == 0
