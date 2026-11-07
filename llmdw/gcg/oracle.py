"""Family-level oracle: evaluate every decomposition of the family space of an instance.

Paper: Section 6 and Appendix E: the MIPLIB-NL gain tables (Figure 3(a), Tables 1, 2, 9, 14 and 15).

A candidate is a pair (M, L): M a set of constraint families kept in the master, L a set of
variable families allowed to link blocks; the other rows are split into blocks by connected
components over non-linking variables. M ranges over proper subsets of the constraint
families; L over all variable-family subsets when k_c + k_v <= max_families, otherwise only
the empty set (``use_linking`` in oracle.json says which). Excluded: (empty, empty), and any
candidate whose materialization has an empty master and a single block, because its pricing
problem is the original MIP (the same rule is applied to GCG's own choices, which are then
flagged ``degenerate``). Every candidate is evaluated in the bound-evaluation setting (root
node, no cuts, no heuristics, no presolve, SCIP propagation off; see llmdw.gcg.gcg_run) with the same time limit, which gives the best
decomposition of the space (the oracle) and the bound of every alternative.

Comparison bound: for instances whose objective is integral, every bound is rounded to the
next integer in its own direction (``cmp_bound``) before any comparison, selection or
display to the LLM, because user-given decompositions with linking variables run without the
non-rounding device (see gcg_run) and would otherwise be favoured by SCIP's rounding.

The run also records, in the same setting:
  lp                    LP relaxation of the compact model
  gcg_spfwh             GCG's own detection and default score (spfwh)
  gcg_strong            GCG detection with the strong decomposition score
  gcg_spfwh_names       GCG detection with the two name classifiers enabled
GCG's chosen decompositions are written as .dec files and mapped to a master-family set
when they are family-aligned.

Usage
-----
python -m llmdw.gcg.oracle --mps work/mps/X.mps.gz --out work/oracle/X --time 300 --jobs 4 --max-families 6
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llmdw.gcg import gcg_run
from llmdw.gcg.dec_io import (aligned_master_families, family_profile, from_master_families, is_component_partition,
                     read_dec_full, write_dec)
from llmdw.instances.families import family_of, objective_integral, read_model, var_family_map

GCG_BASELINES = {
    "gcg_spfwh": dict(score="spfwh", names=False),
    "gcg_strong": dict(score="strong", names=False),
    "gcg_spfwh_names": dict(score="spfwh", names=True),
}


def round_bound(z, sense: str, integral: bool):
    """Round a dual bound to the next integer in its own direction when the objective is integral."""
    import math

    if z is None or not integral:
        return z
    tol = 1e-6 * max(1.0, abs(z))
    return float(math.ceil(z - tol)) if sense == "minimize" else float(math.floor(z + tol))


def cmp_value(row: dict):
    """The bound used for comparisons: ``cmp_bound`` when present, else the dual bound."""
    v = row.get("cmp_bound", row.get("dual_bound"))
    return v if (row.get("ok") and v is not None) else None


def subset_key(master_families, linking_var_families=None) -> str:
    key = "|".join(sorted(master_families)) if master_families else "<none>"
    if linking_var_families:
        key += " +L:" + "|".join(sorted(linking_var_families))
    return key


def key_tag(key: str) -> str:
    import hashlib

    return "c" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


# Failed evaluations (Task: crash handling). An evaluation fails when its child exits abnormally or returns no result
# (a crash such as exit -11, an exception such as a SCIP memory error, a kill at the wall-clock limit). A failed
# evaluation is retried twice with the same settings, then once with the same settings and the seed shifted by
# RETRY_SEED_SHIFT (cuts and heuristics stay off). If every attempt fails the row has status "failed", ok false, no
# bound, and the exit code and log of every attempt; it is never a gain of 0 and never enters a tie or a group decision
# (llmdw.scoring.structure records it as failed and marks the decisions that depend on it incomplete).
SAME_SETTING_RETRIES = 2
RETRY_SEED_SHIFT = 1


def attempt_seeds(seed: int) -> list:
    return [seed] * (1 + SAME_SETTING_RETRIES) + [seed + RETRY_SEED_SHIFT]


def run_with_retries(run_one, out_json: str, seed: int) -> dict:
    """run_one(out_json_of_attempt, seed) -> gcg_run result. Attempt 0 writes out_json, attempt n >= 1 writes
    <out>.a<n>.json (and its log next to it). Returns the result of the first successful attempt (or of the last one)
    with ``attempts`` (one entry per attempt), ``succeeded_attempt`` (None when every attempt failed) and
    ``seed_used``."""
    attempts, res = [], {}
    base = out_json[:-5] if out_json.endswith(".json") else out_json
    for n, sd in enumerate(attempt_seeds(seed)):
        path = out_json if n == 0 else f"{base}.a{n}.json"
        res = run_one(path, sd)
        attempts.append({"attempt": n, "seed": sd, "ok": bool(res.get("ok")), "status": res.get("status"),
                         "exit_code": res.get("exit_code"), "error": res.get("error"), "log_path": res.get("log_path"),
                         "wall_seconds": res.get("wall_seconds"), "started_at": res.get("started_at"),
                         "ended_at": res.get("ended_at"), "mem_available_gb_start": res.get("mem_available_gb_start"),
                         "mem_available_gb_end": res.get("mem_available_gb_end"),
                         "budget_wait_seconds": res.get("budget_wait_seconds")})
        if res.get("ok"):
            break
    res = dict(res)
    res["attempts"] = attempts
    res["succeeded_attempt"] = attempts[-1]["attempt"] if res.get("ok") else None
    res["seed_used"] = attempts[-1]["seed"]
    res["wall_seconds_all_attempts"] = round(sum(a["wall_seconds"] or 0.0 for a in attempts), 3)
    if not res.get("ok"):
        res["status"] = "failed"
    return res


def _setting_of(row: dict) -> str:
    """Propagation setting of a stored result; results written before the setting was recorded used SCIP defaults."""
    return (row.get("propagation") or {}).get("setting", "default")


def _load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def evaluate_subset(model, mps: str, out_dir: Path, master_families: set, time_limit: float, seed: int = 0,
                    tag: str | None = None, linking_var_families: set | None = None,
                    integral: bool | None = None, propagation: str = "off",
                    memory_gb: float | None = None) -> dict:
    """Materialize one decomposition (master families, linking variable families), run GCG on
    it in the bound-evaluation setting (propagation off by default, see llmdw.gcg.gcg_run), and return the result row.

    Storage: one record per evaluation, ``<tag>.row.json``, written atomically after the last attempt, so a record on
    disk is complete. Evaluations are deterministic (single thread, fixed seed), so a record already present in
    ``out_dir`` for the same key, time limit, seed and propagation setting is reused with its original solving time,
    which is what the evaluation costs in any time accounting. A killed run loses at most the evaluations in flight."""
    if integral is None:
        integral = objective_integral(model)
    key = subset_key(master_families, linking_var_families)
    tag = tag or key_tag(key)
    out_dir = Path(out_dir).resolve()
    row = _load_json(out_dir / f"{tag}.row.json")
    if (row and row.get("time_limit") == time_limit and row.get("seed") == seed and row.get("key") == key
            and _setting_of(row) == propagation):
        if "cmp_bound" not in row:
            row["cmp_bound"] = round_bound(row.get("dual_bound"), model.sense, integral)
        return row
    blocks, master = from_master_families(model, set(master_families), set(linking_var_families or ()))
    row = {"key": key, "master_families": sorted(master_families),
           "linking_var_families": sorted(linking_var_families or ()), "tag": tag,
           "n_blocks": len(blocks), "n_master_rows": len(master),
           "largest_block_rows": max((len(b) for b in blocks), default=0),
           "time_limit": time_limit, "seed": seed, "propagation": {"setting": propagation}}
    if not blocks:
        row.update({"ok": True, "status": "no_blocks", "dual_bound": None, "cmp_bound": None})
        return row
    if not master and len(blocks) == 1:
        # one block holding every row: the pricing problem is the original MIP
        row.update({"ok": True, "status": "degenerate_single_block", "dual_bound": None, "cmp_bound": None})
        return row
    dec = str(out_dir / f"{tag}.dec")
    write_dec(dec, blocks, master, header=f"master families: {key}")
    res = run_with_retries(
        lambda path, sd: gcg_run.run("dw-given", mps, path, dec=dec, root=True, no_cuts=True, no_round=True,
                                     no_heur=True, time_limit=time_limit, seed=sd, propagation=propagation,
                                     memory_gb=memory_gb),
        str(out_dir / f"{tag}.json"), seed)
    row.update({k: res.get(k) for k in ("ok", "status", "dual_bound", "primal_bound", "solving_time",
                                        "wall_seconds", "pricing_seconds", "pricing_problems", "linking_vars",
                                        "no_round_applied", "error", "exit_code", "log_path", "propagation",
                                        "memory_gb", "attempts", "succeeded_attempt", "seed_used",
                                        "wall_seconds_all_attempts")})
    row["dec_path"] = dec
    row["cmp_bound"] = round_bound(row.get("dual_bound"), model.sense, integral) if row.get("ok") else None
    gcg_run.write_json_atomic(out_dir / f"{tag}.row.json", row)
    return row


def lp_bound(mps: str, out: Path, memory_gb: float | None = None) -> dict:
    """LP relaxation of the compact model (gcg_run lp, unchanged SCIP defaults), with the retries of
    run_with_retries; a complete lp.json already in out is reused. memory_gb: the per-process cap of the model's
    evaluations (llmdw.gcg.address_limit, as for the GCG children; default LLMDW_CHILD_MEMORY_GB), so the LP solve of a
    high-memory model runs under the same cap as its evaluations; a solve that exceeds it fails like any attempt."""
    path = Path(out) / "lp.json"
    prev = _load_json(path)
    if prev and prev.get("ok") and "attempts" in prev:
        return prev
    res = run_with_retries(lambda p, sd: gcg_run.run("lp", mps, p, seed=sd, memory_gb=memory_gb), str(path), 0)
    gcg_run.write_json_atomic(path, res)
    return res


def run_baseline(model, mps: str, out: Path, name: str, opts: dict, time_limit: float, seed: int, integral: bool,
                 propagation: str = "off", memory_gb: float | None = None) -> dict:
    """GCG's own detection (dw-auto) in the bound-evaluation setting, propagation off by default, with the retries of
    run_with_retries; returns the baseline row, written atomically to <name>.row.json and reused from there."""
    out = Path(out)
    prev = _load_json(out / f"{name}.row.json")
    if prev and prev.get("time_limit") == time_limit and _setting_of(prev) == propagation:
        return prev
    dec_path = str(out / f"{name}.dec")
    res = run_with_retries(
        lambda path, sd: gcg_run.run("dw-auto", mps, path, root=True, no_cuts=True, no_round=True, no_heur=True,
                                     time_limit=time_limit, seed=sd, write_chosen=dec_path, propagation=propagation,
                                     memory_gb=memory_gb, **opts),
        str(out / f"{name}.json"), seed)
    row = {k: res.get(k) for k in ("ok", "status", "dual_bound", "solving_time", "wall_seconds", "blocks",
                                    "pricing_seconds", "error", "exit_code", "log_path", "propagation", "memory_gb",
                                    "attempts", "succeeded_attempt", "seed_used")}
    row["time_limit"] = time_limit
    row["cmp_bound"] = round_bound(row.get("dual_bound"), model.sense, integral) if row.get("ok") else None
    row["degenerate"] = False
    if res.get("ok") and os.path.exists(dec_path):
        d = read_dec_full(dec_path)
        prof = family_profile(d["master"], [r.name for r in model.rows])
        aligned = aligned_master_families(prof)
        # same exclusion as for the family space: an empty master with a single block
        # makes the pricing problem the original MIP, which is not a reformulation
        degenerate = (not d["master"]) and len(d["blocks"]) == 1
        representable = (not degenerate and aligned is not None
                         and is_component_partition(model, d["blocks"], d["master"]))
        row["degenerate"] = degenerate
        row["chosen_n_blocks"] = len(d["blocks"])
        row["chosen_n_master_rows"] = len(d["master"])
        row["master_family_share"] = prof
        row["n_linking_vars"] = len(d["var_sections"].get("LINKINGVARS", []))
        row["representable"] = representable
        row["aligned_master_families"] = sorted(aligned) if representable else None
    gcg_run.write_json_atomic(out / f"{name}.row.json", row)
    return row


def run_oracle(mps: str, out_dir: str, time_limit: float, jobs: int, max_families: int, seed: int = 0,
               reference: float | None = None, with_linking: bool = True, memory_gb: float | None = None) -> dict:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = read_model(mps)
    integral = objective_integral(model)
    fams = sorted({family_of(r.name) for r in model.rows})
    vfams = sorted(set(var_family_map(model.var_type).values()))
    summary = {"mps": mps, "families": fams, "variable_families": vfams, "k": len(fams), "k_var": len(vfams),
               "time_limit": time_limit, "sense": model.sense, "reference_optimal_value": reference,
               "with_linking": with_linking, "objective_integral": integral, "propagation": "off",
               "memory_gb": memory_gb or (float(os.environ["LLMDW_CHILD_MEMORY_GB"])
                                          if os.environ.get("LLMDW_CHILD_MEMORY_GB") else None)}
    lp = lp_bound(mps, out, memory_gb=memory_gb)
    summary["lp_bound"] = lp.get("dual_bound") if lp.get("ok") else None
    summary["lp_status"] = lp.get("status")

    # GCG baselines in the same bound-evaluation setting
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        summary["baselines"] = dict(zip(GCG_BASELINES, pool.map(
            lambda it: run_baseline(model, mps, out, it[0], it[1], time_limit, seed, integral, memory_gb=memory_gb),
            GCG_BASELINES.items())))

    use_linking = with_linking and len(fams) + len(vfams) <= max_families
    summary["use_linking"] = use_linking
    if len(fams) > max_families:
        summary["oracle"] = None
        summary["skipped_reason"] = f"{len(fams)} families > max_families={max_families}"
    else:
        # Candidate space: (master constraint families M, linking variable families L) with M a
        # proper subset of the constraint families. (M, L) = (empty, empty) puts every row into
        # blocks with nothing linking them, i.e. the original MIP split into independent parts;
        # it is not a reformulation and is excluded. L ranges over all variable-family subsets
        # only when |constraint families| + |variable families| <= max_families.
        cons_sets = [frozenset(c) for r in range(0, len(fams)) for c in itertools.combinations(fams, r)]
        var_sets = ([frozenset(c) for r in range(0, len(vfams) + 1) for c in itertools.combinations(vfams, r)]
                    if use_linking else [frozenset()])
        cands = [(M, L) for M in cons_sets for L in var_sets if M or L]
        summary["n_candidates"] = len(cands)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(lambda c: evaluate_subset(model, mps, out, set(c[0]), time_limit, seed,
                                                           linking_var_families=set(c[1]), integral=integral,
                                                           memory_gb=memory_gb), cands))
        summary["subsets"] = rows
        summary["failed"] = [r["key"] for r in rows if r.get("status") == "failed"]
        summary["oracle"] = best_row(rows, model.sense)
    gcg_run.write_json_atomic(out / "oracle.json", summary)
    return summary


def better(a: float, b: float, sense: str, tol: float = 1e-6) -> bool:
    """True if dual bound a is strictly better than b (higher for min, lower for max)."""
    scale = max(1.0, abs(a), abs(b))
    return (a - b) > tol * scale if sense == "minimize" else (b - a) > tol * scale


def best_row(rows: list[dict], sense: str) -> dict | None:
    """Best comparison bound (``cmp_bound``, see the module docstring); ties within tolerance
    broken by solving time."""
    cand = [r for r in rows if cmp_value(r) is not None]
    if not cand:
        return None
    best = cand[0]
    for r in cand[1:]:
        if better(cmp_value(r), cmp_value(best), sense):
            best = r
        elif not better(cmp_value(best), cmp_value(r), sense):
            if (r.get("solving_time") or 1e18) < (best.get("solving_time") or 1e18):
                best = r
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-families", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", type=float, default=None)
    ap.add_argument("--no-linking", action="store_true", help="enumerate master constraint families only")
    args = ap.parse_args(argv)
    s = run_oracle(args.mps, args.out, args.time, args.jobs, args.max_families, args.seed, args.reference,
                   with_linking=not args.no_linking)
    o = s.get("oracle")
    print(json.dumps({"k": s["k"], "lp": s["lp_bound"],
                      "oracle": None if o is None else {k: o.get(k) for k in ("key", "dual_bound", "solving_time")},
                      "baselines": {k: (v.get("dual_bound"), v.get("aligned_master_families")) for k, v in s["baselines"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
