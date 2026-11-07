"""Reference optimum of an exported model: SCIP with its defaults, one thread, seed 0, a time limit.

A model whose best recorded primal solution value is better than its reference optimum by more than the tolerance of
llmdw.scoring.bounds is solved again with this command; an optimum that SCIP proves within the limit becomes the
reference (data/census_corrections.json), otherwise the model has no reference optimum. The two runs of the sample of
record are in data/work/reference_optima/.

Usage
-----
python -m llmdw.instances.reference_optimum --mps work/mps/b-ball.mps.gz --out b-ball.json [--time 3600]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def solve(mps: str, time_limit: float = 3600.0, log: str | None = None) -> dict:
    from pyscipopt import Model

    m = Model()
    if log:
        m.setLogfile(log)
    m.readProblem(mps)
    m.setParam("limits/time", time_limit)
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("lp/threads", 1)
    m.setParam("randomization/randomseedshift", 0)
    t0 = time.time()
    m.optimize()
    return {"mps": Path(mps).name, "status": m.getStatus(), "primal": m.getPrimalbound(), "dual": m.getDualbound(),
            "gap": m.getGap(), "seconds": round(time.time() - t0, 3), "solving_time": m.getSolvingTime(),
            "sense": m.getObjectiveSense(), "time_limit": time_limit, "threads": 1, "seed": 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=3600.0)
    args = ap.parse_args(argv)
    res = solve(args.mps, args.time, str(Path(args.out).with_suffix(".log")))
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
