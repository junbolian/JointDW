"""Cap on the number of active solver processes on the machine (several batch drivers run side by side).

With LLMDW_MAX_SOLVERS set, a solver child of llmdw.gcg.gcg_run starts only when fewer than that many solver processes
are active. Counted from /proc: the children of gcg_run (command line with llmdw.gcg.gcg_run and --child), and the
worker processes of the batch drivers that solve in their own process (forked pool workers, whose command line equals
that of their parent: the instance builds and the finalize step of llmdw.instances.rich_models_large and the screening
of llmdw.benchmark.extra_classes). The check and the start of the child happen under an exclusive file lock in
LLMDW_SOLVER_SLOTS_DIR (default llmdw-solver-slots-<uid> in the system temporary directory), so two drivers never pass
the check for the same free slot. The wait comes before the clock of the run starts. Without LLMDW_MAX_SOLVERS,
nothing waits.
"""
from __future__ import annotations

import fcntl
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

DRIVERS = ("llmdw.instances import rich_models_large", "llmdw.instances.rich_models_large",
           "llmdw.benchmark.extra_classes")


def _cmdline(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _ppid(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            return fh.read().rsplit(")", 1)[1].split()[1]
    except (OSError, IndexError):
        return ""


def active() -> int:
    """Number of active solver processes (module docstring)."""
    if not os.path.isdir("/proc"):
        return 0
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        cmd = _cmdline(pid)
        if not cmd:
            continue
        if "llmdw.gcg.gcg_run" in cmd and " --child" in cmd:
            n += 1
        elif any(d in cmd for d in DRIVERS) and _cmdline(_ppid(pid)) == cmd:
            n += 1
    return n


def load1() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


@contextmanager
def slot(poll: float = 5.0):
    """Wait, under the lock, until a solver slot is free; the block starts the child and leaves the lock once the child
    exists. Yields the seconds waited."""
    cap = os.environ.get("LLMDW_MAX_SOLVERS")
    if not cap:
        yield 0.0
        return
    d = Path(os.environ.get("LLMDW_SOLVER_SLOTS_DIR") or Path(tempfile.gettempdir()) / f"llmdw-solver-slots-{os.getuid()}")
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(d / "lock", "a+", encoding="utf-8") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            while active() >= int(cap):
                time.sleep(poll + poll * random.random())
            yield round(time.time() - t0, 3)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
