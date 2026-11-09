"""Rich models with 12 and 16 families (network design: 12 and 14).

The existing rich-model classes (llmdw.instances.rich_models_build, llmdw.instances.templates) with further families,
each a standard modeling feature on the existing variables (templates._rich_*_extra). The menu with k families is the
existing families followed by the first k - (existing) new families in the order of NEW_FAMILIES, so that the k = 16
menu holds all of them; network design uses k = 12 and k = 14 (its first 7 new families, without SplitLimit and
DegreeBalance). Candidate data: the base data are drawn as for the existing rich models
(rich_models_build.draw_params) and the data of the new families by draw_extra, both from
random.Random(SEED_BASE + 100 * k + 10 * class index + seed); a candidate is accepted under the acceptance rule of the
existing rich models (feasible, LP gap at least 2 percent, no single family closing more than half of the best gain
among the sets of at most 3 families), and the instances are the first PER_CLASS accepted candidates of each class
and k in draw order. Each accepted instance gets the exact gain of every set of at most 3 families
(rich_models_build.evaluate, the method of Appendix B) with the evaluation time of every set.

After the builds, finalize writes the instance directories in the layout of the rich models (model files,
ground_truth.json, lookahead.json at the HiGHS dual, gcg_anchors.json), instances.txt and draws.json.

Usage
-----
python -m llmdw.instances.rich_models_large build --root work/rich_models_large --class lot_sizing --k 12 --parallel 4
python -m llmdw.instances.rich_models_large finalize --root work/rich_models_large --parallel 4
"""
from __future__ import annotations

import random
from fractions import Fraction

from llmdw.instances import rich_models_build as rb

PER_CLASS = 10
KS = (12, 16)
KS_BY_CLASS = {"lot_sizing": (12, 16), "network_design": (12, 14), "facility_location": (12, 16)}
SEED_BASE = 7000
EXISTING = {"lot_sizing": 9, "network_design": 7, "facility_location": 7}
NEW_FAMILIES = {
    "lot_sizing": ["ProductStorageCap", "ProductionCap", "TotalOvertime", "ProductionBudget", "WorkforceLimit",
                   "HoldingBudget", "NoConsecutiveSetups"],
    "network_design": ["NodeInDegree", "NodeThroughput", "CommodityArcCap", "MinBuiltArcs", "FlowCostBudget",
                       "ArcConflict", "OutArcCover", "SplitLimit", "DegreeBalance"],
    "facility_location": ["MaxCustomersPerFacility", "DistanceLimit", "RegionCover", "TotalCapacity",
                          "TransportBudget", "SplitLimit", "MinAssignment", "FacilityPairCover",
                          "TransportPreference"],
}


def menu(cls: str, k: int) -> list:
    n = k - EXISTING[cls]
    if not 0 <= n <= len(NEW_FAMILIES[cls]):
        raise ValueError(f"{cls}: no menu with {k} families")
    return NEW_FAMILIES[cls][:n]


def instance_name(cls: str, k: int, idx: int) -> str:
    return f"rich{k}_{cls}_{idx:02d}"


def draw_extra(cls: str, base: dict, families: list, rng: random.Random, only=None, rule=None) -> dict:
    """Data of the new families (families: the new families of the menu), drawn after the base data from the same
    generator; only: draw these families alone (the feasibility repair of network design); rule: the ArcConflict
    admissibility rule (nd_admissible_pairs), ARC_CONFLICT_RULE by default."""
    r = rng.randint
    out = {}
    if cls == "lot_sizing":
        I, T = len(base["demand"]), len(base["demand"][0])
        tot = [sum(base["demand"][i]) for i in range(I)]

        def workforce():                 # labor hours per unit, and hours per period relative to the machine time
            w = [r(1, 2) for _ in range(I)]
            return [w, [int(max(w) * (base["capacity"][t] + base["overtime"][t]) * rng.uniform(0.6, 1.0))
                        for t in range(T)]]

        def holding():                   # weights, and a budget relative to the weighted total demand
            h = [r(1, 2) for _ in range(I)]
            return [h, int(sum(h[i] * (tot[i] + base["safety"][i]) for i in range(I)) * rng.uniform(0.6, 1.2)) + 1]
        draws = {"ProductStorageCap": lambda: [r(3, 7) for _ in range(I)],
                 "ProductionCap": lambda: [[r(4, 10) for _ in range(T)] for _ in range(I)],
                 "TotalOvertime": lambda: r(2, 6),
                 "ProductionBudget": lambda: [tot[i] + base["safety"][i] + r(0, 3) for i in range(I)],
                 "WorkforceLimit": workforce,
                 "HoldingBudget": holding,
                 "NoConsecutiveSetups": lambda: True}
    elif cls == "network_design":
        n, A, K = base["n_nodes"], len(base["arcs"]), len(base["commodities"])

        def flow_cost_budget():          # weights, and a budget relative to their value at the base LP optimum
            c = [[r(1, 3) for _ in range(A)] for _ in range(K)]
            xs = base_lp_flows(base)          # None: the base model is infeasible and the candidate is rejected
            v = 0 if xs is None else sum(c[k][a] * xs[k][a] for k in range(K) for a in range(A))
            return [c, int(v * rng.uniform(1.2, 2.0)) + 1]

        def arc_conflict():              # one or two admissible pairs (nd_admissible_pairs): a single pair when no
            adm = nd_admissible_pairs(base, families, rule)   # second keeps every commodity routable, none if none
            want = r(1, 2)
            out_ = []
            for _ in range(want):
                ok = [q for q in adm if q not in out_ and nd_routable(base, out_ + [q])]
                if not ok:
                    break
                out_.append(ok[r(0, len(ok) - 1)])
            return [list(q) for q in out_]
        draws = {"NodeInDegree": lambda: [r(1, 2) for _ in range(n)],
                 "NodeThroughput": lambda: [r(2, 6) for _ in range(n)],
                 "CommodityArcCap": lambda: [[r(1, 3) for _ in range(A)] for _ in range(K)],
                 "MinBuiltArcs": lambda: r(2, 4),
                 "FlowCostBudget": flow_cost_budget,
                 "ArcConflict": arc_conflict,
                 "OutArcCover": lambda: True,
                 "SplitLimit": lambda: str(Fraction(r(60, 90), 100)),
                 "DegreeBalance": lambda: True}
    else:
        I, J = len(base["capacity"]), len(base["demand"])
        def partition():
            perm = rng.sample(range(I), I)
            cut = r(1, I - 1)
            return [sorted(perm[:cut]), sorted(perm[cut:])]
        draws = {"MaxCustomersPerFacility": lambda: [r(1, J) for _ in range(I)],
                 "DistanceLimit": lambda: [[r(0, I - 1), r(0, J - 1)] for _ in range(r(1, 2))],
                 "RegionCover": partition,
                 "TotalCapacity": lambda: True,
                 "TransportBudget": lambda: int(sum(min(base["assign_cost"][i][j] for i in range(I)) * base["demand"][j]
                                                    for j in range(J)) * rng.uniform(1.2, 1.8)) + 1,
                 "SplitLimit": lambda: str(Fraction(r(2, 4), 4)),
                 "MinAssignment": lambda: [[r(0, I - 1), r(0, J - 1), str(Fraction(1, r(3, 5)))] for _ in range(r(1, 2))],
                 "FacilityPairCover": lambda: [sorted(rng.sample(range(I), 2)) for _ in range(r(1, 2))],
                 "TransportPreference": lambda: [[r(0, J - 1), *rng.sample(range(I), 2)] for _ in range(r(1, 2))]}
    for f in (families if only is None else only):
        out[f] = draws[f]()
    return out


def base_lp_flows(base: dict) -> list:
    """Flows x[k][a] of an optimal solution of the LP relaxation of the base network-design model (the existing 7
    families, without the new ones); SCIP, one thread, presolve off."""
    from llmdw.instances import templates

    q = {kk: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
         for kk, v in base.items()}
    m = templates.rich_network_design(**q, name="base_lp")
    sc = m.to_scip()
    sc.hideOutput()
    for v in sc.getVars():
        sc.chgVarType(v, "C")
    sc.setParam("parallel/maxnthreads", 1)
    sc.setPresolve(pyscipopt_off())
    sc.optimize()
    if sc.getStatus() == "infeasible":
        return None
    if sc.getStatus() != "optimal":
        raise RuntimeError(f"base LP relaxation {sc.getStatus()}")
    val = {v.name: sc.getVal(v) for v in sc.getVars()}
    A, K = len(base["arcs"]), len(base["commodities"])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [[max(0.0, val[f"Flow_{letters[k]}_{a + 1}"]) for a in range(A)] for k in range(K)]


def pyscipopt_off():
    from pyscipopt import SCIP_PARAMSETTING

    return SCIP_PARAMSETTING.OFF


def nd_routable(base: dict, pairs: list) -> bool:
    """Whether some choice of one closed arc per conflicting pair leaves an origin-destination path for every
    commodity (graph connectivity only)."""
    import itertools

    arcs = [tuple(a) for a in base["arcs"]]
    for closed in itertools.product(*[list(q) for q in pairs]) if pairs else [()]:
        open_ = [a for i, a in enumerate(arcs) if i not in set(closed)]
        ok = True
        for org, dest, _ in base["commodities"]:
            seen, stack = {org}, [org]
            while stack:
                v = stack.pop()
                for i, j in open_:
                    if i == v and j not in seen:
                        seen.add(j)
                        stack.append(j)
            if dest not in seen:
                ok = False
                break
        if ok:
            return True
    return False


def nd_admissible_pairs(base: dict, families: list, rule=None) -> list:
    """Arc pairs allowed in ArcConflict: the pair leaves every commodity routable, and neither arc enters a node covered
    by InArcCover with fewer than d incoming arcs or (when OutArcCover is in the menu) leaves a node with fewer than d
    outgoing arcs. Rule 1: d = 3. Rule 2: d = 2, so that after removing either arc alone every node still satisfies
    InArcCover and OutArcCover."""
    import itertools

    d = {1: 3, 2: 2}[ARC_CONFLICT_RULE if rule is None else rule]

    arcs = [tuple(a) for a in base["arcs"]]
    n = base["n_nodes"]
    ins = {v: [a for a, (_, j) in enumerate(arcs) if j == v] for v in range(n)}
    outs = {v: [a for a, (i, _) in enumerate(arcs) if i == v] for v in range(n)}
    sources = {o for o, _, _ in base["commodities"]}
    covered_in = {d for _, d, _ in base["commodities"]} - sources        # the InArcCover nodes of the template
    bad = set()
    for v in covered_in:
        if len(ins[v]) < d:
            bad.update(ins[v])
    if "OutArcCover" in families:
        for v in range(n):
            if outs[v] and len(outs[v]) < d:
                bad.update(outs[v])
    return [q for q in itertools.combinations(range(len(arcs)), 2)
            if not (set(q) & bad) and nd_routable(base, [q])]


def _as_model_data(extra: dict) -> dict:
    """Fractions written as strings in the parameter files become Fractions for the model."""
    def conv(v):
        if isinstance(v, str):
            return Fraction(v)
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v
    return {k: conv(v) for k, v in extra.items()}


def draw(cls: str, k: int, seed: int, rule=None) -> dict:
    rng = random.Random(SEED_BASE + 100 * k + 10 * rb.CLASSES.index(cls) + seed)
    base = rb.draw_params(cls, rng)
    return {**base, "extra": draw_extra(cls, base, menu(cls, k), rng, rule=rule)}


def build_milp(cls: str, params: dict, name: str):
    from llmdw.instances import templates

    p = dict(params)
    extra = _as_model_data(p.pop("extra"))
    fn = {"lot_sizing": templates.rich_lot_sizing, "network_design": templates.rich_network_design,
          "facility_location": templates.rich_facility_location}[cls]
    q = {kk: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
         for kk, v in p.items()}
    return fn(**q, extra=extra, name=name)


CLASSICAL = {"lot_sizing": ("Balance_#_#", "SetupLink_#_#"), "network_design": ("ArcCapacity_#", "FlowBalance_#_#"),
             "facility_location": ("Assign_#", "Capacity_#")}


REPAIR_TRIES = 10
ARC_CONFLICT_RULE = 2                # nd_admissible_pairs; candidates decided under rule 1 record no rule
NO_PAIR = "no admissible ArcConflict pair"
RATE_MIN = 0.05                      # network design: stop a build whose acceptance rate stays below this
RATE_MIN_DRAWS = 40                  # ... over the first RATE_MIN_DRAWS or more draws (a prefix of decided draws;
                                     # with ArcConflict in the menu, of the draws of ARC_CONFLICT_RULE)
PAIR_TABLE_LIMIT = 24 * 3600.0       # network design, k = 14: CPU seconds of singletons and pairs per instance
TRIPLES_ON_REQUEST = {("network_design", 14)}


def _status(m, fams=None, time_limit=60.0) -> str:
    from llmdw.instances.milp import MILP

    mm = m
    if fams is not None:
        mm = MILP(m.name)
        mm.vars, mm.obj, mm.maximize = m.vars, m.obj, m.maximize
        mm.rows = [r for r in m.rows if r.family in set(fams)]
    sc = mm.to_scip()
    sc.hideOutput()
    sc.setParam("limits/time", time_limit)
    sc.setParam("parallel/maxnthreads", 1)
    sc.optimize()
    return sc.getStatus()


def minimal_conflict(m) -> list:
    """A minimal infeasible set of families (deletion filter in family order; SCIP, 60 s per check)."""
    keep = list(m.families)
    for f in list(keep):
        trial = [g for g in keep if g != f]
        if _status(m, trial) == "infeasible":
            keep = trial
    return keep


def repair(cls: str, k: int, idx: int, params: dict) -> tuple:
    """Network design: while the candidate is infeasible, the new families of a minimal conflicting set get new data
    (base data unchanged), up to REPAIR_TRIES times, from random.Random(f"repair-{cls}-{k}-{idx}"); an ArcConflict
    without an admissible pair is redrawn within the same tries. Returns the final parameters, the history and the
    last status."""
    import json

    history = []
    rng = random.Random(f"repair-{cls}-{k}-{idx}")
    base = {kk: v for kk, v in params.items() if kk != "extra"}
    new = menu(cls, k)
    cur = json.loads(json.dumps(params))
    for t in range(REPAIR_TRIES + 1):
        if "ArcConflict" in new and not cur["extra"].get("ArcConflict"):     # no admissible pair: redraw it
            history.append({"try": t, "conflict": [NO_PAIR], "redrawn": ["ArcConflict"]})
            if t == REPAIR_TRIES:
                return cur, history, NO_PAIR
            cur["extra"] = {**cur["extra"],
                            **json.loads(json.dumps(draw_extra(cls, base, new, rng, only=["ArcConflict"])))}
            continue
        m = build_milp(cls, cur, instance_name(cls, k, idx))
        st = _status(m, time_limit=600.0)
        if st != "infeasible":
            return cur, history, st
        conflict = minimal_conflict(m)
        redraw = [f for f in new if any(fam.split("_")[0] == f for fam in conflict)]
        history.append({"try": t, "conflict": conflict, "redrawn": redraw})
        if t == REPAIR_TRIES or not redraw:
            return cur, history, st
        cur["extra"] = {**cur["extra"], **json.loads(json.dumps(draw_extra(cls, base, new, rng, only=redraw)))}
    return cur, history, st


def evaluate(cls: str, k: int, idx: int, params: dict) -> dict:
    """The acceptance checks of one candidate and the exact gains its decision needs, with the evaluation time of every
    set. Gains are normalized by the LP gap (ip - lp), so the best gain is at most 1. Order: singletons, pairs, triples.
    Rejected as soon as the largest singleton gain exceeds 0.5; accepted as soon as it is at most half of the best pair
    gain; otherwise triples decide. Accepted instances get the full table of the sets of at most 3 families, except
    network design at k = 14, whose triples are computed only as far as the decision needs (the rest on request)."""
    import itertools
    import os
    import time

    from llmdw.exact import exactlp, hull

    name = instance_name(cls, k, idx)
    out = {"class": cls, "k_menu": k, "candidate": idx, "name": name}
    t0 = time.time()
    if cls == "network_design":
        out["initial_params"] = params
        out["arc_conflict_rule"] = ARC_CONFLICT_RULE
        if base_lp_flows({kk: v for kk, v in params.items() if kk != "extra"}) is None:
            return {**out, "params": params, "accepted": False,
                    "reason": "MILP status infeasible (base model LP relaxation infeasible)",
                    "seconds": round(time.time() - t0, 2)}
        params, hist, st = repair(cls, k, idx, params)
        out["repair"] = hist
        if st == NO_PAIR:
            return {**out, "params": params, "accepted": False, "reason": f"{NO_PAIR} after {REPAIR_TRIES} redraws",
                    "seconds": round(time.time() - t0, 2)}
    out["params"] = params
    m = build_milp(cls, params, name)
    out.update({"k": len(m.families), "rows": len(m.rows), "columns": len(m.vars),
                "binaries": sum(v.integer for v in m.vars)})
    sc = m.to_scip()
    sc.hideOutput()
    sc.setParam("limits/time", 600)
    sc.optimize()
    if sc.getStatus() != "optimal":
        return {**out, "accepted": False, "reason": f"MILP status {sc.getStatus()}", "seconds": round(time.time() - t0, 2)}
    ip = Fraction(sc.getObjVal()).limit_denominator(10 ** 6)
    try:
        lp = hull.lp_bound(m)
    except RuntimeError as e:
        return {**out, "accepted": False, "reason": f"LP failed: {str(e)[:120]}"}
    gap = float((ip - lp) / abs(ip)) if ip else float("inf")
    out.update({"lp_bound": str(lp), "ip_scip": float(ip), "lp_gap": gap})
    if gap < rb.MIN_GAP:
        return {**out, "accepted": False, "reason": f"LP gap {gap:.4f} below {rb.MIN_GAP}", "seconds": round(time.time() - t0, 2)}
    fams = m.families
    width = ip - lp
    tab, secs, certs = {frozenset(): Fraction(0)}, {}, {}
    on_request = (cls, k) in TRIPLES_ON_REQUEST
    out["memory_wait_seconds"] = _wait_for_memory()
    # pauses of a worker under memory shortage are logged by process id and time, so they can be matched to set_seconds
    out["worker_pid"], out["table_started_at"] = os.getpid(), round(time.time(), 1)

    def ev(T):
        t1 = time.time()
        tab[frozenset(T)] = hull.dw_bound(m, T) - lp
        secs["|".join(sorted(T))] = round(time.time() - t1, 4)
        # the certificate that produced the time of the set (gains are the same; llmdw.exact.exactlp.certificate);
        # candidates without this record were computed with the full basis matrix
        certs["|".join(sorted(T))] = exactlp.certificate()

    def done(accepted, reason, stage, **kw):
        best = max(tab.values())
        single = max(tab[frozenset([f])] for f in fams)
        res = {**out, "best_gain_evaluated": float(best), "best_single_gain": float(single),
               "best_single_gain_normalized": float(single / width), "set_seconds": secs,
               "set_certificate": certs,
               "evaluation_seconds": round(sum(secs.values()), 2), "decided_after": stage,
               "accepted": accepted, "reason": reason, "seconds": round(time.time() - t0, 1), **kw}
        return res

    for f in fams:
        ev([f])
    single = max(tab[frozenset([f])] for f in fams)
    if single > width / 2:
        return done(False, "single-family rule: the largest singleton gain exceeds half of the LP gap", "singletons")
    for T in itertools.combinations(fams, 2):
        ev(T)
        if on_request and sum(secs.values()) > PAIR_TABLE_LIMIT:
            return done(False, "pair table exceeds 24 CPU-hours", "pairs (stopped)", stop_build=True)
    best_pair = max(tab[frozenset(T)] for T in itertools.combinations(fams, 2))
    decided = single <= best_pair / 2
    if not decided or not on_request:
        for T in itertools.combinations(fams, 3):
            ev(T)
            if on_request and single <= tab[frozenset(T)] / 2:
                decided = True
                break
    best = max(tab.values())
    if best <= 0:
        return done(False, "no positive gain among sets of at most 3 families", "triples")
    if single > rb.MAX_SINGLE_SHARE * best:
        return done(False, f"single-family rule: a single family closes {float(single / best):.2f} of the best gain",
                    "triples")
    stage = "pairs" if (on_request and single <= best_pair / 2) else "triples"
    complete = not on_request
    view = {T: v for T, v in tab.items() if complete or len(T) <= 2}
    best_v = max(view.values())
    groups = [W for W in hull.zero_marginal_groups(view) if len(W) >= 2]
    best_sets = sorted("|".join(sorted(T)) for T, v in view.items() if T and v == best_v)
    cp = "|".join(sorted(CLASSICAL[cls]))
    return done(True, None, stage, table={"|".join(sorted(T)): str(v) for T, v in tab.items()},
                table_complete_to=3 if complete else 2, best_gain=float(best_v), best_sets=best_sets,
                groups=[sorted(W) for W in groups], classical_pair=cp,
                classical_pair_gain=float(tab.get(frozenset(CLASSICAL[cls]), Fraction(-1))),
                classical_pair_is_best=cp in best_sets,
                classical_pair_is_group=sorted(CLASSICAL[cls]) in [sorted(W) for W in groups])


def _wait_for_memory() -> float:
    """With LLMDW_GAIN_MIN_AVAILABLE_GB set, wait until the machine has that much available memory; a gain table starts
    only then (the exact gains of large sets need several GB each: the dense basis check of llmdw.exact.exactlp).
    Returns the seconds waited."""
    import os
    import time

    need = os.environ.get("LLMDW_GAIN_MIN_AVAILABLE_GB")
    t0 = time.time()
    while need and os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", encoding="ascii") as fh:
            avail = next((int(x.split()[1]) for x in fh if x.startswith("MemAvailable:")), None)
        if avail is None or avail / 2 ** 20 >= float(need):
            break
        time.sleep(10)
    return round(time.time() - t0, 1)


def safe_evaluate(cls, k, idx, params):
    try:
        return evaluate(cls, k, idx, params)
    except Exception as e:  # noqa: BLE001
        return {"class": cls, "k_menu": k, "candidate": idx, "name": instance_name(cls, k, idx), "params": params,
                "arc_conflict_rule": ARC_CONFLICT_RULE, "accepted": False,
                "reason": f"error: {type(e).__name__}: {str(e)[:200]}"}


def build(root, cls: str, k: int, parallel: int = 4, per_class: int = PER_CLASS) -> dict:
    """The first per_class accepted candidates of one class and k, in draw order; candidates are evaluated in parallel
    and stored in root/candidates/ (a restart reads them); stops drawing once enough are accepted or running."""
    import itertools
    import json
    import multiprocessing
    import os
    import time
    from pathlib import Path

    root = Path(root)
    cdir = root / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    results, futures, idx = {}, {}, 0
    with multiprocessing.get_context("fork").Pool(parallel) as pool:
        def accepted_before(j):
            return sum(1 for i, r in results.items() if r["accepted"] and i < j)

        while True:
            n_acc = sum(1 for r in results.values() if r["accepted"])
            enough = n_acc >= per_class and all(i in results for i in range(min(idx, max(
                [i for i, r in results.items() if r["accepted"]][:per_class] or [0]) + 1)))
            while not enough and len(futures) < parallel and n_acc + len(futures) < per_class + parallel:
                f = cdir / f"{cls}_k{k}_{idx:03d}.json"
                params = json.loads(json.dumps(draw(cls, k, idx)))
                if f.exists():
                    res = json.loads(f.read_text())
                    if res.get("initial_params", res["params"]) != json.loads(json.dumps(
                            draw(cls, k, idx, rule=res.get("arc_conflict_rule", 1)))):
                        raise RuntimeError(f"{f.name}: stored parameters differ from the seeded draw")
                    results[idx] = res
                else:
                    futures[pool.apply_async(safe_evaluate, (cls, k, idx, params))] = idx
                idx += 1
                n_acc = sum(1 for r in results.values() if r["accepted"])
            acc = sorted(i for i, r in results.items() if r["accepted"])
            if len(acc) >= per_class and all(i in results for i in range(acc[per_class - 1] + 1)):
                break
            if not futures:
                continue
            done = next((f for f in futures if f.ready()), None)
            if done is None:
                time.sleep(2)
                continue
            j = futures.pop(done)
            res = done.get()
            tmp = cdir / f"{cls}_k{k}_{j:03d}.json.part"
            tmp.write_text(json.dumps(res, indent=1))
            os.replace(tmp, cdir / f"{cls}_k{k}_{j:03d}.json")
            results[j] = res
            print(json.dumps({"class": cls, "k": k, "candidate": j, "accepted": res["accepted"],
                              "reason": res.get("reason"), "seconds": res.get("seconds"),
                              "decided_after": res.get("decided_after"), "repairs": len(res.get("repair", []))}),
                  flush=True)
            stop = None
            if res.get("stop_build"):
                stop = f"candidate {j}: {res['reason']}"
            # the guard counts the draws of the current ArcConflict rule only (all draws when the menu has no
            # ArcConflict), as far as they are decided without a gap in draw order
            ruled = "ArcConflict" in menu(cls, k)
            elig = [i for i in range(idx + 1) if not (ruled and i in results
                                                      and results[i].get("arc_conflict_rule", 1) != ARC_CONFLICT_RULE)]
            pre = list(itertools.takewhile(lambda i: i in results, elig))
            n_pre, n_acc_pre = len(pre), sum(results[i]["accepted"] for i in pre)
            if cls == "network_design" and n_pre >= RATE_MIN_DRAWS and n_acc_pre < RATE_MIN * n_pre:
                stop = (f"acceptance rate {n_acc_pre}/{n_pre} below {RATE_MIN:.0%} over the first {n_pre} draws"
                        + (f" of ArcConflict rule {ARC_CONFLICT_RULE}" if ruled else ""))
            if stop:
                print(json.dumps({"class": cls, "k": k, "STOP": stop}), flush=True)
                pool.terminate()
                return {"class": cls, "k": k, "stopped": stop, "draws_decided": n_pre,
                        "accepted": [results[i]["name"] for i in sorted(results) if results[i]["accepted"]]}
        pool.terminate()
    acc = sorted(i for i, r in results.items() if r["accepted"])[:per_class]
    return {"class": cls, "k": k, "draws": acc[-1] + 1 if acc else idx, "accepted": [results[i]["name"] for i in acc]}


# ---------------------------------------------------------------- instance directories (after the builds)

def candidates(root, cls: str, k: int) -> dict:
    """{draw index: stored candidate} of one class and k (root/candidates/<cls>_k<k>_<idx>.json)."""
    import json
    from pathlib import Path

    out = {}
    for f in sorted((Path(root) / "candidates").glob(f"{cls}_k{k}_*.json")):
        r = json.loads(f.read_text())
        out[r["candidate"]] = r
    return out


def select(root, cls: str, k: int, per_class: int = PER_CLASS) -> list:
    """The instances of one class and k: the first per_class accepted candidates in draw order (as build returns
    them); every earlier draw must be decided."""
    res = candidates(root, cls, k)
    acc = sorted(i for i, r in res.items() if r["accepted"])[:per_class]
    if len(acc) < per_class:
        raise RuntimeError(f"{cls} k = {k}: {len(acc)} accepted candidates, {per_class} needed")
    missing = [i for i in range(acc[-1] + 1) if i not in res]
    if missing:
        raise RuntimeError(f"{cls} k = {k}: draws {missing} before the last selected one are not decided")
    return [res[i] for i in acc]


def draw_records(root, cls: str, k: int, per_class: int = PER_CLASS) -> list:
    """Every decided candidate of one class and k with the reason of a rejection; in_selection_prefix marks the draws
    up to the last selected instance (the draws that decide the selection); network design records its ArcConflict
    admissibility rule where ArcConflict is in the menu (no record: rule 1)."""
    res = candidates(root, cls, k)
    last = select(root, cls, k, per_class)[-1]["candidate"]
    sel = {r["candidate"] for r in select(root, cls, k, per_class)}
    keys = ("candidate", "name", "accepted", "reason", "decided_after", "k", "rows", "columns", "binaries", "lp_gap",
            "seconds", "evaluation_seconds")
    out = []
    for i in sorted(res):
        r = res[i]
        row = {key: r.get(key) for key in keys}
        row.update({"class": cls, "k_menu": k, "in_selection_prefix": i <= last, "selected": i in sel})
        if cls == "network_design":
            # the rule matters only where ArcConflict is in the menu (k = 14)
            row["arc_conflict_rule"] = r.get("arc_conflict_rule", 1) if "ArcConflict" in menu(cls, k) else None
            row["repairs"] = len(r.get("repair") or [])
        out.append(row)
    return out


def complete_to(res: dict) -> int:
    """The largest s such that the table of an accepted candidate holds every set of at most s families (recorded as
    table_complete_to by the later builds; the earlier ones hold the full table of the sets of at most 3 families)."""
    import itertools
    import math

    fams = sorted({x for key in res["table"] for x in key.split("|") if x})
    if len(fams) != res["k"]:
        raise RuntimeError(f"{res['name']}: {len(fams)} families in the table, {res['k']} recorded")
    keys = {frozenset(x for x in key.split("|") if x) for key in res["table"]}
    s = 0
    while s < 3 and all(frozenset(T) in keys for T in itertools.combinations(fams, s + 1)):
        s += 1
    if "table_complete_to" in res and res["table_complete_to"] != s:
        raise RuntimeError(f"{res['name']}: table complete to {s}, recorded {res['table_complete_to']}")
    assert sum(math.comb(len(fams), r) for r in range(1, s + 1)) <= len(keys)
    return s


def write_instance(res: dict, root) -> "Path":
    """The instance directory in the layout of the rich models (llmdw.instances.rich_models_build.write_instance):
    model.mps, instance.json, model.md, ground_truth.json (the exact gains of the sets of at most table_complete_to
    families, the best sets among them, the zero-marginal groups, the classical pair) and lookahead.json (the one-pass
    lookahead at the HiGHS dual for the same sets). The sources of the zero-marginal groups are not computed."""
    import json
    from pathlib import Path

    from llmdw.lookahead.lookahead import lookahead_table

    m = build_milp(res["class"], res["params"], res["name"])
    if len(m.families) != res["k"]:
        raise RuntimeError(f"{res['name']}: {len(m.families)} families, {res['k']} recorded")
    d = Path(root) / "instances" / res["name"]
    d.mkdir(parents=True, exist_ok=True)
    m.write_mps(str(d / "model.mps"))
    (d / "instance.json").write_text(json.dumps({"abstract_problem": m.description}, indent=1), encoding="utf-8")
    lines = ["# Formulation", "", "Constraint families:"] + [f"- {f}: {m.family_text.get(f, '')}" for f in m.families]
    (d / "model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    s = complete_to(res)
    gain = {key: v for key, v in res["table"].items() if len([x for x in key.split("|") if x]) <= s}
    tab = {key: Fraction(v) for key, v in gain.items()}
    best = max(tab.values())
    gt = {"families": m.families, "k": len(m.families), "lp_bound": res["lp_bound"], "ip_scip": res["ip_scip"],
          "lp_gap": res["lp_gap"], "table_complete_to": s, "gain": gain, "joint_supports": res["groups"],
          "group_sources": None,
          "strengthens_alone": sorted(f for f in m.families if tab[f] > 0),
          "best_sets": sorted((key.split("|") for key, v in tab.items() if v == best and key), key=lambda x: (len(x), x)),
          "classical_pair": res["classical_pair"].split("|"), "classical_pair_is_best": res["classical_pair_is_best"],
          "classical_pair_is_group": res["classical_pair_is_group"],
          "note": (f"exact gains of the sets of at most {s} families"
                   + ("" if s == 3 else "; sets of 3 families not computed (on request)")
                   + f"; IP optimum (SCIP) {res['ip_scip']}")}
    (d / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    if not (d / "lookahead.json").exists():
        la = lookahead_table(str(d / "model.mps"), block_time=60.0, max_size=s)
        (d / "lookahead.json").write_text(json.dumps(la, indent=1), encoding="utf-8")
    return d


def _finalize_one(args):
    res, root = args
    import time

    t0 = time.time()
    write_instance(res, root)
    return res["name"], round(time.time() - t0, 1)


def finalize(root, parallel: int = 4, per_class: int = PER_CLASS) -> dict:
    """Instance directories, instances.txt and draws.json for every class and k of KS_BY_CLASS, then GCG's
    decompositions of each instance in the oracle setting (rich_models_build.gcg_anchors, gcg_anchors.json).
    Resumable: an existing lookahead.json or gcg_anchors.json is kept."""
    import json
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    root = Path(root)
    sel, draws = [], {"seed_base": SEED_BASE, "streams": "random.Random(SEED_BASE + 100 * k + 10 * class index + seed)",
                      "draws": []}
    for cls, ks in KS_BY_CLASS.items():
        for k in ks:
            sel += select(root, cls, k, per_class)
            draws["draws"] += draw_records(root, cls, k, per_class)
    (root / "draws.json").write_text(json.dumps(draws, indent=1), encoding="utf-8")
    (root / "instances.txt").write_text("\n".join(r["name"] for r in sel) + "\n", encoding="utf-8")
    with multiprocessing.get_context("fork").Pool(parallel) as pool:
        for name, sec in pool.imap_unordered(_finalize_one, [(r, root) for r in sel]):
            print(json.dumps({"instance": name, "seconds": sec}), flush=True)
    pending = [r["name"] for r in sel if not (root / "instances" / r["name"] / "gcg_anchors.json").exists()]
    with ThreadPoolExecutor(parallel) as pool:
        for n, a in zip(pending, pool.map(lambda x: rb.gcg_anchors(x, root), pending)):
            print(n, json.dumps({b: (x["status"], x["convexified"]) for b, x in a.items()}), flush=True)
    return {"instances": len(sel)}


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="draw candidates of one class and k until the instances are accepted, with their "
                                      "exact gain tables (candidates/ under --root; resumable)")
    b.add_argument("--root", required=True)
    b.add_argument("--class", dest="cls", required=True, choices=list(rb.CLASSES))
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--parallel", type=int, default=4)
    f = sub.add_parser("finalize", help="instance directories, draws.json, instances.txt and GCG's decompositions")
    f.add_argument("--root", required=True, help="directory with candidates/ of the builds")
    f.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        print(json.dumps(build(args.root, args.cls, args.k, parallel=args.parallel)), flush=True)
    elif args.cmd == "finalize":
        print(json.dumps(finalize(args.root, args.parallel)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
