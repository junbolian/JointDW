"""The LP solve of the census runs under the model's per-process cap (llmdw.gcg.oracle.lp_bound)."""
from llmdw.gcg import gcg_run, oracle


def test_lp_bound_passes_the_memory_cap(tmp_path, monkeypatch):
    seen = []

    def fake_run(mode, mps, out_json, **kw):
        seen.append((mode, kw.get("memory_gb")))
        return {"ok": True, "status": "optimal", "dual_bound": 1.0, "wall_seconds": 0.1}

    monkeypatch.setattr(gcg_run, "run", fake_run)
    res = oracle.lp_bound("m.mps", tmp_path, memory_gb=10)
    assert seen == [("lp", 10)] and res["ok"] and res["attempts"][0]["ok"]
    seen.clear()
    (tmp_path / "other").mkdir()
    oracle.lp_bound("m.mps", tmp_path / "other")
    assert seen == [("lp", None)]          # default: LLMDW_CHILD_MEMORY_GB, read by gcg_run.run
