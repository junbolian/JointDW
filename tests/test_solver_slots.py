import os
import subprocess
import sys

from llmdw.gcg import solver_slots


def test_slot_without_cap(monkeypatch):
    monkeypatch.delenv("LLMDW_MAX_SOLVERS", raising=False)
    with solver_slots.slot() as w:
        assert w == 0.0


def test_slot_counts_gcg_children(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMDW_SOLVER_SLOTS_DIR", str(tmp_path))
    base = solver_slots.active()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "llmdw.gcg.gcg_run", "--child"])
    try:
        assert solver_slots.active() == base + 1
        monkeypatch.setenv("LLMDW_MAX_SOLVERS", str(base + 2))
        with solver_slots.slot(poll=0.01) as w:
            assert w < 5
    finally:
        p.kill()
        p.wait()
    assert solver_slots.load1() is None or solver_slots.load1() >= 0
    assert os.path.exists(tmp_path / "lock")
