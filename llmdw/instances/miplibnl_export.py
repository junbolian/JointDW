"""Export MIPLIB-NL reference models to MPS files that keep constraint and variable names.

Paper: Section 6 and Appendix E: the MIPLIB-NL models.

Each MIPLIB-NL instance directory contains ``instance.json``, ``data/`` and ``solve.py``,
a gurobipy script that builds the model and calls ``Model.optimize()``. This module runs
every script as ``__main__`` in a child process whose working directory is the instance
directory, and replaces ``gurobipy.Model.optimize`` by a function that writes the model
to MPS and stops the script. No optimization is performed, so the size-limited license
that ships with ``pip install gurobipy`` is sufficient.

Some scripts hard-code placeholder paths (for example
``path_to_your_file/<name>/<name>.json``). A path that does not exist is redirected to
the instance's ``instance.json`` (for ``*.json``) or to ``data/<basename>`` /
``<basename>`` inside the instance directory when such a file exists.

Usage
-----
python -m llmdw.instances.miplibnl_export --dataset MIPLIB-NL/dataset --out work/mps --jobs 4

Output per instance ``<name>``: ``<name>.mps.gz`` and ``<name>.meta.json`` (sizes,
objective sense, reference optimal value, export time). A summary table
``export_summary.csv`` lists the status of every instance.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXIT_OK = 0
EXIT_NO_OPTIMIZE = 3
EXIT_SCRIPT_ERROR = 4


def _child(instance_dir: str, out_prefix: str) -> int:
    """Run one solve.py with Model.optimize replaced by an MPS export. Returns exit code."""
    import builtins
    import io
    import runpy
    import traceback

    inst = Path(instance_dir).resolve()
    out_prefix = str(Path(out_prefix).resolve())
    os.chdir(inst)
    os.makedirs("logs", exist_ok=True)  # many scripts write ./logs/log.txt; the directory is not in the repository
    sys.path.insert(0, str(inst))
    sys.argv = ["solve.py"] + json.loads(os.environ.get("LLMDW_ARGV", "[]"))

    real_open = builtins.open
    real_exists = os.path.exists
    real_isfile = os.path.isfile

    def redirect(path):
        try:
            sp = os.fspath(path)
        except TypeError:
            return path
        if not isinstance(sp, str) or real_exists(sp):
            return path
        base = os.path.basename(sp)
        if base.endswith(".json") and real_exists(inst / "instance.json"):
            return str(inst / "instance.json")
        for cand in (inst / "data" / base, inst / base):
            if real_exists(cand):
                return str(cand)
        return path

    def patched_open(file, *args, **kwargs):
        return real_open(redirect(file), *args, **kwargs)

    builtins.open = patched_open
    io.open = patched_open
    os.path.exists = lambda p: real_exists(redirect(p))
    os.path.isfile = lambda p: real_isfile(redirect(p))

    try:
        import pandas as pd

        real_read_csv = pd.read_csv

        def patched_read_csv(f, *args, **kwargs):
            if isinstance(f, (str, os.PathLike)):
                f = redirect(f)
            return real_read_csv(f, *args, **kwargs)

        pd.read_csv = patched_read_csv
    except ImportError:
        pass

    import gurobipy

    class ExportDone(BaseException):
        pass

    t_start = time.time()

    def export_optimize(self, *args, **kwargs):
        self.update()
        self.write(out_prefix + ".mps.gz")
        meta = {
            "model_name": self.ModelName,
            "num_vars": self.NumVars,
            "num_constrs": self.NumConstrs,
            "num_int_vars": self.NumIntVars,
            "num_bin_vars": self.NumBinVars,
            "num_nonzeros": self.NumNZs,
            "num_sos": self.NumSOS,
            "num_qconstrs": self.NumQConstrs,
            "num_genconstrs": self.NumGenConstrs,
            "model_sense": "min" if self.ModelSense == 1 else "max",
            "build_seconds": round(time.time() - t_start, 3),
        }
        with real_open(out_prefix + ".gurobi_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        raise ExportDone()

    gurobipy.Model.optimize = export_optimize

    try:
        runpy.run_path(str(inst / "solve.py"), run_name="__main__")
    except ExportDone:
        return EXIT_OK
    except SystemExit:
        return EXIT_NO_OPTIMIZE
    except BaseException:  # noqa: BLE001 - report any script failure
        traceback.print_exc()
        return EXIT_SCRIPT_ERROR
    return EXIT_NO_OPTIMIZE


# Command-line arguments tried in order when a script exits before calling optimize()
# (several scripts expect the data folder or the JSON path as argv[1]).
ARGV_ATTEMPTS = ([], ["data"], ["instance.json"], ["."])


def _export_one(instance_dir: Path, out_dir: Path, timeout: int) -> dict:
    name = instance_dir.name
    out_prefix = out_dir / name
    log_path = out_dir / f"{name}.export.log"
    t0 = time.time()
    status, used_argv = "not_run", None
    with open(log_path, "w", encoding="utf-8") as log:
        for argv in ARGV_ATTEMPTS:
            log.write(f"=== attempt argv={argv}\n")
            log.flush()
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "llmdw.instances.miplibnl_export", "--child", str(instance_dir), str(out_prefix)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "LLMDW_ARGV": json.dumps(argv),
                         "PYTHONPATH": str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get("PYTHONPATH", "")},
                )
                code = proc.returncode
                status = {EXIT_OK: "ok", EXIT_NO_OPTIMIZE: "no_optimize_call",
                          EXIT_SCRIPT_ERROR: "script_error"}.get(code, f"exit_{code}")
            except subprocess.TimeoutExpired:
                status = "timeout"
            used_argv = argv
            if status != "no_optimize_call":
                break
    row = {"instance": name, "status": status, "seconds": round(time.time() - t0, 1), "argv": json.dumps(used_argv)}
    meta_path = Path(str(out_prefix) + ".gurobi_meta.json")
    inst_json = instance_dir / "instance.json"
    ref = None
    if inst_json.exists():
        with open(inst_json, encoding="utf-8") as fh:
            ref = json.load(fh).get("optimal_value")
    if status == "ok" and meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["reference_optimal_value"] = ref
        meta["export_status"] = status
        meta["export_argv"] = used_argv
        with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        meta_path.unlink()
        row.update({k: meta[k] for k in ("num_vars", "num_constrs", "num_int_vars", "num_nonzeros", "model_sense")})
    row["reference_optimal_value"] = ref
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=2, metavar=("INSTANCE_DIR", "OUT_PREFIX"), help=argparse.SUPPRESS)
    ap.add_argument("--dataset", type=Path, help="MIPLIB-NL dataset/ directory")
    ap.add_argument("--out", type=Path, help="output directory for MPS files")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per instance")
    ap.add_argument("--only", nargs="*", default=None, help="instance names to export")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args(argv)

    if args.child:
        return _child(*args.child)

    if args.dataset is None or args.out is None:
        ap.error("--dataset and --out are required")
    args.out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in args.dataset.iterdir() if (p / "solve.py").exists())
    if args.only:
        wanted = set(args.only)
        dirs = [p for p in dirs if p.name in wanted]
    if args.skip_existing:
        dirs = [p for p in dirs if not (args.out / f"{p.name}.meta.json").exists()]

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(lambda d: _export_one(d, args.out, args.timeout), dirs):
            rows.append(row)
            print(f"{row['instance']:40s} {row['status']:18s} {row['seconds']:8.1f}s", flush=True)

    summary = args.out / "export_summary.csv"
    old = {}
    if summary.exists():
        with open(summary, encoding="utf-8") as fh:
            old = {r["instance"]: r for r in csv.DictReader(fh)}
    for r in rows:
        old[r["instance"]] = r
    fields = ["instance", "status", "seconds", "argv", "num_vars", "num_constrs", "num_int_vars",
              "num_nonzeros", "model_sense", "reference_optimal_value"]
    with open(summary, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for name in sorted(old):
            w.writerow(old[name])
    n_ok = sum(1 for r in old.values() if r["status"] == "ok")
    print(f"exported {n_ok}/{len(old)}; summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
