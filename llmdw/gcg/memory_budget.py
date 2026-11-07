"""Memory budget shared by every solver child on the machine (several batch drivers run side by side).

With LLMDW_MEMORY_BUDGET_GB set, a child starts only when the per-process memory caps (address-space limits) of all
running children, its own included, sum to at most that budget; a high-memory model's child counts at its own cap.
Waiting children are served in arrival order: a waiting child that does not fit yet holds back the children behind it.

The state is a JSON file in LLMDW_MEMORY_BUDGET_DIR (default llmdw-memory-budget-<uid> in the system temporary directory), read and written under
an exclusive file lock. Entries of processes that no longer exist are dropped, so a killed driver frees its share. The
wait happens before a child's clock starts (llmdw.gcg.gcg_run.run, llmdw.lookahead.five_duals.batch), so it never uses
a time limit. Without LLMDW_MEMORY_BUDGET_GB, or for a child without a cap, reserve() does nothing.
"""
from __future__ import annotations

import fcntl
import itertools
import json
import os
import random
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_counter = itertools.count()


def _settings():
    budget = os.environ.get("LLMDW_MEMORY_BUDGET_GB")
    if not budget:
        return None
    d = Path(os.environ.get("LLMDW_MEMORY_BUDGET_DIR") or Path(tempfile.gettempdir()) / f"llmdw-memory-budget-{os.getuid()}")
    return float(budget), d


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _locked(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "lock", "a+", encoding="utf-8") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            p = d / "state.json"
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                st = {"running": {}, "waiting": []}
            st["running"] = {t: e for t, e in st["running"].items() if _alive(e["pid"])}
            st["waiting"] = [[t, e] for t, e in st["waiting"] if _alive(e["pid"])]
            yield st
            tmp = d / f"state.json.tmp{os.getpid()}"
            tmp.write_text(json.dumps(st), encoding="utf-8")
            os.replace(tmp, p)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def eligible(st: dict, token: str, budget: float) -> bool:
    """Whether the waiting entry token may start now (module docstring)."""
    free = budget - sum(min(e["cap"], budget) for e in st["running"].values())
    for t, e in st["waiting"]:
        cap = min(e["cap"], budget)
        if t == token:
            return cap <= free + 1e-9
        free -= cap
    return False


@contextmanager
def reserve(cap_gb: float | None, key: str, poll: float = 2.0):
    """Hold a share cap_gb of the budget while the block runs; key names the model (recorded in the state file).
    Yields the seconds spent waiting (0.0 without a budget)."""
    s = _settings()
    if s is None or not cap_gb:
        yield 0.0
        return
    budget, d = s
    token = f"{os.getpid()}-{threading.get_ident()}-{next(_counter)}"
    entry = {"pid": os.getpid(), "cap": float(cap_gb), "key": str(key), "since": time.time()}
    t0 = time.time()
    while True:
        with _locked(d) as st:
            if not any(t == token for t, _ in st["waiting"]):
                st["waiting"].append([token, entry])
            if eligible(st, token, budget):
                st["waiting"] = [[t, e] for t, e in st["waiting"] if t != token]
                st["running"][token] = entry
                break
        time.sleep(poll * (0.5 + random.random()))
    try:
        yield time.time() - t0
    finally:
        with _locked(d) as st:
            st["running"].pop(token, None)


def mem_available_gb() -> float | None:
    """MemAvailable of /proc/meminfo in GB (None where it does not exist)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 2 ** 20, 2)
    except OSError:
        pass
    return None
