"""Run GCG, SCIP or Gurobi on one model in a child process and write a JSON result.

Paper: Appendix E: the bound evaluations of the MIPLIB-NL models.

Modes
-----
dw-given     GCG with a given decomposition (.dec, original problem), all detectors off,
             no presolve. Used for every LLM candidate and for the oracle enumeration.
dw-auto      GCG detection on the original problem (no presolve) with the selected score
             (``--score``) and optionally the name classifiers (``--names``); the chosen
             decomposition is written to ``--write-chosen``. Same setting as dw-given, so
             bounds are comparable decomposition by decomposition.
gcg-default  GCG as shipped (presolve, detection, default score). End-to-end baseline only.
scip         SCIP on the compact model.
gurobi       Gurobi on the compact model (needs a full Gurobi license).
lp           LP relaxation of the compact model (SCIP, every variable made continuous).

Bound-evaluation setting (``--root --no-cuts --no-heur --no-round``): node limit 1, SCIP
separators and GCG master separators off, primal heuristics off (a primal bound would
trigger reduced-cost fixing and move the root bound away from the DW bound),
``pricing/masterpricer/abortpricingint`` off, and no rounding
of the dual bound for integral objectives, so the reported dual bound is the Dantzig-Wolfe
bound of the decomposition (or the best Lagrangian bound reached before the time limit).
This follows Bastubbe, Luebbecke and Witt (SEA 2018), who also switched off the handling of
integral optima. SCIP rounds dual bounds up when it detects an integral objective and has no
parameter to stop it; ``--no-round`` adds one continuous variable fixed to 0 with objective
coefficient 1, which makes the objective non-integral without changing any value.

Examples
--------
python -m llmdw.gcg.gcg_run dw-auto  --mps X.mps.gz --root --no-cuts --time 300 --out r.json --write-chosen chosen.dec
python -m llmdw.gcg.gcg_run dw-given --mps X.mps.gz --dec cand.dec --root --no-cuts --time 300 --out r.json
python -m llmdw.gcg.gcg_run lp --mps X.mps.gz --out lp.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODES = ("dw-given", "dw-auto", "gcg-default", "scip", "gurobi", "lp")


# ----------------------------------------------------------------------------- child side

def _common_scip_params(m, args):
    if args.time is not None:
        m.setRealParam("limits/time", float(args.time))
    if args.root:
        m.setLongintParam("limits/nodes", 1)
    m.setIntParam("randomization/randomseedshift", int(args.seed))
    m.setIntParam("lp/threads", 1)
    m.setIntParam("parallel/maxnthreads", 1)
    cap_gb = os.environ.get("LLMDW_CHILD_MEMORY_GB")
    if cap_gb:  # graceful stop (status memlimit) below the hard cap that run() puts on the child process
        m.setRealParam("limits/memory", 0.8 * float(cap_gb) * 1024.0)


def _no_cuts_gcg(m):
    from pygcgopt import SCIP_PARAMSETTING

    m.setBoolParam("sepa/original/enable", False)
    m.setBoolParam("sepa/basis/enable", False)
    m.setSeparating(SCIP_PARAMSETTING.OFF)


def _dec_has_linking(m, dec_path: str) -> bool:
    """True if some variable occurs in rows of two different blocks of the .dec."""
    from llmdw.gcg.dec_io import read_dec

    blocks, _, _ = read_dec(dec_path)
    by_name = {c.name: c for c in m.getConss()}
    owner = {}
    for bi, b in enumerate(blocks):
        for n in b:
            c = by_name.get(n)
            if c is None:
                continue
            for v in m.getValsLinear(c):
                vname = v if isinstance(v, str) else v.name
                if owner.setdefault(vname, bi) != bi:
                    return True
    return False


def _child_gcg(args) -> dict:
    import pygcgopt

    from llmdw.gcg.dec_io import read_dec

    m = pygcgopt.Model()
    m.readProblem(args.mps)
    add_breaker = args.no_round
    if args.mode == "dw-given" and args.no_round and _dec_has_linking(m, args.dec):
        # PyGCGOpt 1.0.0b0 segfaults when a user-given decomposition with linking variables is
        # combined with the extra objective column, so such
        # decompositions run without it; oracle.cmp_bound rounds all bounds of integral-objective
        # instances, which keeps comparisons consistent.
        add_breaker = False
    if add_breaker:
        m.addVar(name="llmdw_objint_breaker", vtype="C", lb=0.0, ub=0.0, obj=1.0)
    _common_scip_params(m, args)
    m.setIntParam("pricing/masterpricer/nthreads", 1)
    if args.no_cuts:
        _no_cuts_gcg(m)
    if args.no_heur:
        from pygcgopt import SCIP_PARAMSETTING

        m.setHeuristics(SCIP_PARAMSETTING.OFF)
    if args.root:
        m.setBoolParam("pricing/masterpricer/abortpricingint", False)
    out = {"no_round_applied": bool(add_breaker)}
    if args.mode == "dw-given":
        for det in m.listDetectors():
            m.setDetectorEnabled(det, False)
            m.setDetectorFinishingEnabled(det, False)
            m.setDetectorPostprocessingEnabled(det, False)
        blocks, master, presolved = read_dec(args.dec)
        if presolved:
            raise ValueError("only decompositions of the original problem are supported")
        by_name = {c.name: c for c in m.getConss()}
        missing = [n for b in blocks for n in b if n not in by_name] + [n for n in master if n not in by_name]
        if missing:
            raise ValueError(f"{len(missing)} row names of the .dec are not in the model, e.g. {missing[:3]}")
        listed = {n for b in blocks for n in b} | set(master)
        unlisted = [n for n in by_name if n not in listed]
        master_conss = [by_name[n] for n in master] + [by_name[n] for n in unlisted]
        out["unlisted_rows_to_master"] = len(unlisted)
        if not blocks:
            raise ValueError("decomposition without blocks; evaluate the LP bound instead")
        # Build the decomposition explicitly. Variables that occur in rows of more than one block
        # must be declared linking: PyGCGOpt 1.0.0b0 segfaults during the solve when blocks share
        # undeclared variables.
        block_conss = [[by_name[n] for n in b] for b in blocks]
        var_blocks = {}
        for bi, conss in enumerate(block_conss):
            for c in conss:
                for v in m.getValsLinear(c):
                    vname = v if isinstance(v, str) else v.name
                    var_blocks.setdefault(vname, set()).add(bi)
        vars_by_name = {v.name: v for v in m.getVars()}
        linking = [vars_by_name[n] for n, bs in var_blocks.items() if len(bs) > 1]
        pd = m.createDecomposition()
        pd.fixConssToMaster(master_conss)
        for bi, conss in enumerate(block_conss):
            pd.fixConssToBlockId(conss, bi)
        for v in linking:
            pd.fixVarToLinking(v)
        m.addDecomposition(pd)
        out["linking_vars"] = len(linking)
        m.detect()
        out["given_blocks"] = len(blocks)
        out["given_master_rows"] = len(master_conss)
    elif args.mode in ("dw-auto", "gcg-default"):
        if args.time is not None:
            # detection (e.g. the strong score, which solves pricing problems) gets the same cap
            m.setIntParam("detection/maxtime", max(1, int(args.time)))
        if args.score:
            m.setStringParam("detection/scores/selected", args.score)
        if args.names:
            m.setBoolParam("detection/classification/consclassifier/consnamenonumbers/enabled", True)
            m.setBoolParam("detection/classification/consclassifier/consnamelevenshtein/enabled", True)
        if args.mode == "dw-auto":
            m.detect()
            if args.write_chosen:
                m.writeProblem(args.write_chosen, trans=False)
            out["n_finished_decompositions"] = len(m.listDecompositions())
    m.optimize()
    m.printStatistics()
    out.update(_collect(m))
    return out


def _child_scip(args) -> dict:
    import pyscipopt
    from pyscipopt import SCIP_PARAMSETTING

    m = pyscipopt.Model()
    m.readProblem(args.mps)
    if args.no_round and args.mode != "lp":
        m.addVar(name="llmdw_objint_breaker", vtype="C", lb=0.0, ub=0.0, obj=1.0)
    _common_scip_params(m, args)
    if args.mode == "lp":
        for v in m.getVars():
            m.chgVarType(v, "C")
        m.setPresolve(SCIP_PARAMSETTING.OFF)
    else:
        if args.no_cuts:
            m.setSeparating(SCIP_PARAMSETTING.OFF)
        if args.no_heur:
            m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.optimize()
    return _collect(m)


def _child_lp_highs(args) -> dict:
    """LP relaxation with HiGHS interior point and crossover (LLMDW_LP_SOLVER=highs-ipm): the same model with every
    variable continuous, presolve off, one thread. Any status other than Optimal is a failure of the LP."""
    import highspy
    import numpy as np

    highspy.Highs.resetGlobalScheduler(True)    # "threads" below takes effect only on a fresh scheduler
    h = highspy.Highs()
    if h.readModel(args.mps) != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS could not read {args.mps}")
    for opt, val in (("solver", "ipm"), ("run_crossover", "on"), ("presolve", "off"), ("threads", 1)):
        h.setOptionValue(opt, val)
    lp_time = os.environ.get("LLMDW_LP_TIME")
    if lp_time:
        h.setOptionValue("time_limit", float(lp_time))
    n = h.getLp().num_col_
    if n:
        h.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8))
    t0 = time.time()
    h.run()
    secs = time.time() - t0
    status = h.modelStatusToString(h.getModelStatus())
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS LP status {status} after {secs:.1f} s (not Optimal, not used)")
    obj = h.getInfo().objective_function_value
    return {"status": "optimal", "dual_bound": obj, "primal_bound": obj, "solving_time": secs,
            "lp_solver": "highs-ipm", "highs_model_status": status}


def _child_gurobi(args) -> dict:
    import gurobipy as gp

    m = gp.read(args.mps)
    m.Params.Threads = 1
    m.Params.Seed = int(args.seed)
    if args.time is not None:
        m.Params.TimeLimit = float(args.time)
    if args.root:
        m.Params.NodeLimit = 1
    if args.no_cuts:
        m.Params.Cuts = 0
    t0 = time.time()
    m.optimize()
    status = {2: "optimal", 3: "infeasible", 9: "timelimit", 8: "nodelimit", 5: "unbounded"}.get(m.Status, str(m.Status))
    res = {"status": status, "solving_time": m.Runtime, "nodes": int(m.NodeCount),
           "sense": "minimize" if m.ModelSense == 1 else "maximize", "wall_seconds": time.time() - t0}
    try:
        res["dual_bound"] = m.ObjBound
    except Exception:  # noqa: BLE001
        res["dual_bound"] = None
    res["primal_bound"] = m.ObjVal if m.SolCount > 0 else None
    return res


def _collect(m) -> dict:
    sense = m.getObjectiveSense()
    try:
        primal = m.getPrimalbound() if m.getNSols() > 0 else None
    except Exception:  # noqa: BLE001
        primal = None
    return {
        "status": m.getStatus(),
        "dual_bound": m.getDualbound(),
        "primal_bound": primal,
        "nodes": m.getNNodes(),
        "solving_time": m.getSolvingTime(),
        "sense": sense,
    }


def _child_main(args) -> int:
    t0 = time.time()
    try:
        if args.mode in ("dw-given", "dw-auto", "gcg-default"):
            res = _child_gcg(args)
        elif args.mode == "lp" and os.environ.get("LLMDW_LP_SOLVER") == "highs-ipm":
            res = _child_lp_highs(args)
        elif args.mode in ("scip", "lp"):
            res = _child_scip(args)
        else:
            res = _child_gurobi(args)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    res["wall_seconds"] = round(time.time() - t0, 3)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    sys.stdout.flush()
    return 0 if res["ok"] else 1


# ----------------------------------------------------------------------------- parent side

_RE_CHOSEN = re.compile(r"Chosen structure has (\d+) blocks?, (?:(\d+) linking variables?, )?(\d+) master-only \(static\) variables?, and (\d+) linking constraints?")
_RE_MATRIX = re.compile(r"Matrix has (\d+) blocks?, using (\d+) pricing problems?")
_RE_PRICING_ROW = re.compile(r"^\s{2}(\S[^:]*?)\s*:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_gcg_log(text: str) -> dict:
    """Pull decomposition size and pricing time out of a GCG log with statistics."""
    out = {}
    mm = _RE_CHOSEN.search(text)
    if mm:
        out["chosen_blocks"] = int(mm.group(1))
        out["chosen_linking_vars"] = int(mm.group(2)) if mm.group(2) else 0
        out["master_only_vars"] = int(mm.group(3))
        out["linking_conss"] = int(mm.group(4))
    mm = _RE_MATRIX.search(text)
    if mm:
        out["blocks"], out["pricing_problems"] = map(int, mm.groups())
    # Pricing Solver table: name : #HeurFarkas #OptFarkas #HeurRedcost #OptRedcost then four times
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Pricing Solver"):
            total = 0.0
            calls = 0
            for row in lines[i + 1:i + 30]:
                r = _RE_PRICING_ROW.match(row)
                if not r:
                    if row.strip() and not row.startswith("  "):
                        break
                    continue
                calls += sum(int(r.group(k)) for k in (2, 3, 4, 5))
                total += sum(float(r.group(k)) for k in (6, 7, 8, 9))
            out["pricing_calls"] = calls
            out["pricing_seconds"] = round(total, 3)
            break
    return out


def run(mode: str, mps: str, out_json: str, *, dec: str | None = None, root: bool = False,
        no_cuts: bool = False, no_round: bool = False, no_heur: bool = False, score: str | None = None,
        names: bool = False,
        time_limit: float | None = None, seed: int = 0, write_chosen: str | None = None,
        log_path: str | None = None, margin: float = 120.0) -> dict:
    """Run one solve in a child process (isolates crashes) and return the parsed result."""
    if mode not in MODES:
        raise ValueError(mode)
    cmd = [sys.executable, "-m", "llmdw.gcg.gcg_run", mode, "--child", "--mps", mps, "--out", out_json, "--seed", str(seed)]
    if dec:
        cmd += ["--dec", dec]
    if root:
        cmd.append("--root")
    if no_cuts:
        cmd.append("--no-cuts")
    if no_round:
        cmd.append("--no-round")
    if no_heur:
        cmd.append("--no-heur")
    if score:
        cmd += ["--score", score]
    if names:
        cmd.append("--names")
    if time_limit is not None:
        cmd += ["--time", str(time_limit)]
    if write_chosen:
        cmd += ["--write-chosen", write_chosen]
    cap_gb = os.environ.get("LLMDW_CHILD_MEMORY_GB")
    if cap_gb and os.name == "posix" and shutil.which("prlimit"):
        # hard address-space cap per solve, so one large decomposition cannot exhaust the machine
        cmd = ["prlimit", f"--as={int(float(cap_gb) * 2 ** 30)}", "--"] + cmd
    log_path = log_path or (out_json[:-5] if out_json.endswith(".json") else out_json) + ".log"
    # dw-auto and gcg-default may spend up to time_limit in detection before the solve starts
    factor = 2.0 if mode in ("dw-auto", "gcg-default") else 1.0
    wall_limit = None if time_limit is None else factor * float(time_limit) + margin
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t0 = time.time()
    Path(out_json).unlink(missing_ok=True)  # a result file of another run is never read
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=wall_limit, env=env)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = "killed"
    res = {}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            res = json.load(fh)
    if not res:
        res = {"ok": False, "error": f"child exit {code}", "wall_seconds": round(time.time() - t0, 3)}
    res["exit_code"] = code
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            res.update(parse_gcg_log(fh.read()))
    except OSError:
        pass
    res.update({"mode": mode, "mps": mps, "dec": dec, "root": root, "no_cuts": no_cuts, "no_round": no_round, "no_heur": no_heur, "score": score,
                "names": names, "time_limit": time_limit, "seed": seed})
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--mps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dec")
    ap.add_argument("--root", action="store_true")
    ap.add_argument("--no-cuts", action="store_true")
    ap.add_argument("--no-round", action="store_true")
    ap.add_argument("--no-heur", action="store_true")
    ap.add_argument("--score")
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--time", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-chosen")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return _child_main(args)
    res = run(args.mode, args.mps, args.out, dec=args.dec, root=args.root, no_cuts=args.no_cuts, no_round=args.no_round, no_heur=args.no_heur,
              score=args.score, names=args.names, time_limit=args.time, seed=args.seed,
              write_chosen=args.write_chosen)
    print(json.dumps({k: res.get(k) for k in ("ok", "status", "dual_bound", "primal_bound", "solving_time",
                                              "blocks", "pricing_seconds", "error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
