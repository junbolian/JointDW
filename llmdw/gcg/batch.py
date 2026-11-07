"""Batch drivers (resumable: finished instances are skipped).

Paper: Section 6 and Appendix E: the MIPLIB-NL gain tables and lookahead values.

python -m llmdw.gcg.batch oracle    --mps-dir work/mps --out work/oracle --max-families 6 --time 120 --parallel $P \
       [--names NAMES_FILE] [--high-memory NAMES_FILE --high-memory-gb 10]
python -m llmdw.gcg.batch lookahead --mps-dir work/mps --oracle work/oracle --out work/lookahead --block-time 60 --parallel $P

oracle     every family-level decomposition of each linear instance with 2..max-families constraint families
           (GCG root bound, bound-evaluation setting with SCIP propagation off) and GCG's own choice under three
           settings. Resumable at the level of single evaluations (llmdw.gcg.oracle.evaluate_subset); the instances
           listed in --high-memory run with the address-space cap --high-memory-gb instead of LLMDW_CHILD_MEMORY_GB
lookahead  one-pricing-call lookahead bound (llmdw.lookahead.lookahead) for every family set of the same instances;
           each instance runs in its own child process (python -m llmdw.lookahead.lookahead), under
           an address-space cap of LLMDW_CHILD_MEMORY_GB (llmdw.gcg.address_limit) when that variable is set, with one BLAS/OpenMP thread.
           The child writes <name>.json.part, renamed to <name>.json only after a clean exit, so an
           instance killed at the cap is not mistaken for a finished one on resume; its log is <name>.log;
           a failed child is retried twice, and a failure after the last attempt is recorded in <name>.error.txt.
           Block solves use SCIP propagation off (llmdw.lookahead.lookahead.BLOCK_PROPAGATION).

Instance order: increasing number of constraint families, then increasing model size, so a partial run
already covers the cheap part of the set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _capped(cmd: list) -> list:
    """Run cmd under the address-space cap LLMDW_CHILD_MEMORY_GB when it is set (llmdw.gcg.address_limit, the same cap
    as llmdw.gcg.gcg_run)."""
    from llmdw.gcg import address_limit
    return address_limit.prefix(os.environ.get("LLMDW_CHILD_MEMORY_GB")) + cmd


def _instances(mps_dir: Path) -> list[dict]:
    out = []
    for meta in sorted(mps_dir.glob("*.meta.json")):
        name = meta.name[: -len(".meta.json")]
        m = json.loads(meta.read_text(encoding="utf-8"))
        mps = mps_dir / f"{name}.mps.gz"
        if mps.exists():
            out.append({"name": name, "mps": str(mps), "meta": m})
    return out


def _families_count(mps: str) -> int:
    from llmdw.instances.families import family_of, read_model

    return len({family_of(r.name) for r in read_model(mps).rows})


def _skip_reason(meta: dict) -> str | None:
    """Models outside the linear MILP class handled here (indicator/general, SOS, quadratic)."""
    for key in ("num_genconstrs", "num_sos", "num_qconstrs"):
        if (meta.get(key) or 0) > 0:
            return f"{key}={meta.get(key)}"
    return None


def cmd_oracle(args):
    from llmdw.gcg.oracle import run_oracle

    insts = _instances(Path(args.mps_dir))
    if args.names:
        keep = set(Path(args.names).read_text(encoding="utf-8").split())
        insts = [it for it in insts if it["name"] in keep]
    high = set(Path(args.high_memory).read_text(encoding="utf-8").split()) if args.high_memory else set()
    plan, planlog = [], []
    for it in insts:
        reason = _skip_reason(it["meta"])
        k = None
        if reason is None:
            try:
                k = _families_count(it["mps"])
            except Exception as e:  # noqa: BLE001 - e.g. a row type the reader does not accept
                reason = f"read_error: {type(e).__name__}: {e}"
        if reason is None and not (2 <= k <= args.max_families):
            reason = f"constraint_families={k}"
        planlog.append({"instance": it["name"], "families": k, "planned": reason is None, "reason": reason or ""})
        if reason is None:
            plan.append((k, it["meta"].get("num_nonzeros") or 0, it))
    plan.sort(key=lambda x: (x[0], x[1]))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "oracle_plan.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["instance", "families", "planned", "reason"])
        w.writeheader()
        w.writerows(planlog)
    print(f"{len(plan)} instances planned (2..{args.max_families} constraint families); "
          f"plan with skip reasons in {Path(args.out) / 'oracle_plan.csv'}", flush=True)

    def work(item):
        k, _, it = item
        out = Path(args.out) / it["name"]
        if (out / "oracle.json").exists():
            return it["name"], "skip"
        try:
            ref = it["meta"].get("reference_optimal_value")
            try:
                ref = float(ref)
            except (TypeError, ValueError):
                ref = None
            s = run_oracle(it["mps"], str(out), args.time, args.jobs_per_instance, args.max_families, args.seed, ref,
                           memory_gb=args.high_memory_gb if it["name"] in high else None)
            failed = len(s.get("failed") or [])
            return it["name"], "done" + (f" ({failed} failed evaluations)" if failed else "")
        except Exception:  # noqa: BLE001
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return it["name"], "error"

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, st in pool.map(work, plan):
            print(f"{name:40s} {st}", flush=True)


def cmd_lookahead(args):
    names = sorted(p.parent.name for p in Path(args.oracle).glob("*/oracle.json"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def work(name):
        path = out_dir / f"{name}.json"
        if path.exists():
            return name, "skip"
        part, log = out_dir / f"{name}.json.part", out_dir / f"{name}.log"
        cmd = [sys.executable, "-m", "llmdw.lookahead.lookahead", "--mps", str(Path(args.mps_dir).resolve() / f"{name}.mps.gz"),
               "--out", str(part.resolve()), "--max-families", str(args.max_families),
               "--block-time", str(args.block_time), "--total-time", str(args.total_time)]
        codes = []
        for attempt in range(3):        # a failed child is retried twice (llmdw.gcg.oracle.SAME_SETTING_RETRIES)
            try:
                with open(log, "w", encoding="utf-8") as fh:
                    proc = subprocess.run(_capped(cmd), cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
            except Exception:  # noqa: BLE001 - the child could not be started
                (out_dir / f"{name}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                return name, "error"
            codes.append(proc.returncode)
            if proc.returncode == 0 and part.exists():
                os.replace(part, path)
                return name, "done" + (f" (attempt {attempt})" if attempt else "")
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        (out_dir / f"{name}.error.txt").write_text(f"child exits {codes}\n{tail}", encoding="utf-8")
        part.unlink(missing_ok=True)
        return name, f"error (child exit {proc.returncode})"

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, st in pool.map(work, names):
            print(f"{name:40s} {st}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--mps-dir", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--max-families", type=int, default=6)
    o.add_argument("--time", type=float, default=120)
    o.add_argument("--parallel", type=int, default=4)
    o.add_argument("--jobs-per-instance", type=int, default=1)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--names", help="file with the instance names to run (default: every instance)")
    o.add_argument("--high-memory", help="file with the instance names that get the cap --high-memory-gb")
    o.add_argument("--high-memory-gb", type=float, default=10.0)
    la = sub.add_parser("lookahead")
    la.add_argument("--mps-dir", required=True)
    la.add_argument("--oracle", required=True)
    la.add_argument("--out", required=True)
    la.add_argument("--max-families", type=int, default=6)
    la.add_argument("--block-time", type=float, default=60)
    la.add_argument("--total-time", type=float, default=3600)
    la.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args(argv)
    {"oracle": cmd_oracle, "lookahead": cmd_lookahead}[args.cmd](args)


if __name__ == "__main__":
    main()
