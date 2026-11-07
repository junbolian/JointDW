"""Run a command under an address-space limit: the per-process memory cap of every solver child.

prefix(cap_gb) returns the arguments that start a command through this module, which sets RLIMIT_AS (soft and hard) to
cap_gb GiB and then replaces itself by the command, so the limit applies to the command's own process. A child that
exceeds the cap fails its allocation and exits abnormally (a failed evaluation, llmdw.gcg.oracle). On a system without
the resource module (not POSIX) no limit is set.

Usage: python -m llmdw.gcg.address_limit BYTES -- COMMAND [ARGS ...]
"""
from __future__ import annotations

import os
import sys


def prefix(cap_gb) -> list:
    """[python, -m, llmdw.gcg.address_limit, <bytes>, --] for a cap in GiB, or [] without a cap or off POSIX."""
    if not cap_gb or os.name != "posix":
        return []
    return [sys.executable, "-m", "llmdw.gcg.address_limit", str(int(float(cap_gb) * 2 ** 30)), "--"]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[1] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    limit, cmd = int(argv[0]), argv[2:]
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    os.execvp(cmd[0], cmd)
    return 127                      # not reached


if __name__ == "__main__":
    raise SystemExit(main())
