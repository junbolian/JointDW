"""Shared memory budget (llmdw.gcg.memory_budget)."""
import json
import os
import threading
import time

from llmdw.gcg import memory_budget as mb


def _e(cap, key):
    return {"pid": os.getpid(), "cap": cap, "key": key, "since": 0}


def test_budget_sum():
    st = {"running": {"a": _e(5, "x1"), "b": _e(5, "x2")}, "waiting": [["c", _e(5, "x3")]]}
    assert mb.eligible(st, "c", 15)
    assert not mb.eligible(st, "c", 14)


def test_arrival_order_holds_back_later_children():
    # a 10 GB child that does not fit yet keeps a later 5 GB child from taking its place
    st = {"running": {"a": _e(5, "x1"), "b": _e(5, "x2")}, "waiting": [["c", _e(10, "h1")], ["d", _e(5, "x3")]]}
    assert not mb.eligible(st, "c", 15) and not mb.eligible(st, "d", 15)
    assert mb.eligible(st, "c", 20) and not mb.eligible(st, "d", 20)


def test_high_memory_models_only_limited_by_the_sum():
    st = {"running": {"a": _e(10, "h1"), "b": _e(10, "h2")}, "waiting": [["c", _e(10, "h3")], ["d", _e(10, "h4")]]}
    assert mb.eligible(st, "c", 40) and mb.eligible(st, "d", 40)
    assert mb.eligible(st, "c", 30) and not mb.eligible(st, "d", 30)


def test_reserve_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMDW_MEMORY_BUDGET_GB", "10")
    monkeypatch.setenv("LLMDW_MEMORY_BUDGET_DIR", str(tmp_path))
    lock = threading.Lock()
    held, peak = [0.0], [0.0]

    def job(cap, key):
        with mb.reserve(cap, key, poll=0.05):
            with lock:
                held[0] += cap
                peak[0] = max(peak[0], held[0])
            time.sleep(0.2)
            with lock:
                held[0] -= cap

    ts = [threading.Thread(target=job, args=(5.0, f"m{i}")) for i in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] <= 10.0 and held[0] == 0.0


def test_dead_process_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMDW_MEMORY_BUDGET_GB", "10")
    monkeypatch.setenv("LLMDW_MEMORY_BUDGET_DIR", str(tmp_path))
    dead = {"pid": 2 ** 22 + 12345, "cap": 10.0, "key": "x", "since": 0}
    (tmp_path / "state.json").write_text(json.dumps({"running": {"t": dead}, "waiting": []}))
    with mb.reserve(10.0, "y", poll=0.05) as wait:
        assert wait < 1.0


def test_no_budget_no_wait(monkeypatch):
    monkeypatch.delenv("LLMDW_MEMORY_BUDGET_GB", raising=False)
    with mb.reserve(100.0, "z") as wait:
        assert wait == 0.0
