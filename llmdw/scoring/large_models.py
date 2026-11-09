"""Budgeted search on MIPLIB-NL models with more than 6 constraint families.

Paper: Appendix F: models with more than six families.

Setting: no linking variables, size limit s = 3 (sets of at most 3 convexified families), the oracle's evaluation setting
(oracle.evaluate_subset: GCG through PyGCGOpt with a given decomposition, root node, no presolve, cuts or primal
heuristics, SCIP propagation off, 120 s, one thread, seed 0; failed evaluations retried as in llmdw.gcg.oracle). Every evaluated set is stored per instance (`<tag>.row.json`) and shared by all
methods, so a set is evaluated once; each method is charged one evaluation for every set in its own order.

Family order: every method uses one order of the families, the first of the 1,000 family orders of llmdw.search.relabel
(the first permutation of the instance's stream, applied to the families' order in the model file), so no result
depends on the names of the families.

Methods (budget B evaluations per instance):
  covering      a covering design of all pairs by sets of 3 families: greedy (the first set in lexicographic order of
                the positions in the family order that covers the most uncovered pairs), in that order
  greedy        forward selection: at each step every single addition (sets of at most 3 families) in the family
                order, then the best one (ties to the earlier); it stops when no addition improves or the budget is used
  lookahead     sets of at most 3 families by decreasing one-pass lookahead (llmdw.lookahead.lookahead, oracle settings), ties to
                fewer families, then to the earlier positions in the family order; sets without a lookahead value (time
                limit) last, in the same order
  random_<seed> a uniformly random order of the sets of at most 3 families (random.Random(seed).sample of the sets
                listed in the family order), seeds 0, 1, 2
  jointdw       the staged search around a model's predicted set (top candidate) interleaved with the covering order of
                search_sim (as `search_sim.robust`), per answer (original presentation, three samples)
  interleaved   a model's JointDW order and the lookahead order alternately, repeats skipped

Metric: after b = 1..B evaluations, the best bound so far as a share of the gap between the LP bound and the best bound
found on that instance by any method (instances where no evaluated set has a positive gain are reported, not scored).

Files: <work>/work/large_models/instances.txt and one directory per instance (instance.json, lookahead.json,
messages.json, orders.json, evals/); the MPS files in <work>/work/mps.

Usage
-----
python -m llmdw.scoring.large_models prepare --work data --dataset MIPLIB-NL/dataset --parallel 8   # LP bound, lookahead, prompts
python -m llmdw.scoring.large_models run --work data --parallel 8                                   # methods without answers
python -m llmdw.scoring.large_models analyze --work data --out results
"""
from __future__ import annotations

from llmdw.constants import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from llmdw.search import search_sim
from llmdw.providers import usage
from llmdw.instances.families import family_of, read_model, summarize
from llmdw.gcg.oracle import evaluate_subset, lp_bound

S = 3
BUDGET = 10
TIME_LIMIT = 120.0
SEEDS = (0, 1, 2)
LARGE_DIR = "work/large_models"


def _minform(v, sense):
    return v if sense == "minimize" else -v


# ---------------------------------------------------------------- orders

def all_sets(fams, s=S):
    """The sets of at most s families, by size, then in lexicographic order of the positions in fams."""
    return [frozenset(c) for r in range(1, s + 1) for c in itertools.combinations(list(fams), r)]


def covering_design(order, n, size=S):
    """The first n sets of the greedy covering of all pairs by sets of `size` families, the families in the given
    order, computed lazily: a set covering `size choose 2` new pairs is taken at once."""
    order = list(order)
    covered, out = set(), []
    full = size * (size - 1) // 2
    total = len(order) * (len(order) - 1) // 2
    while len(out) < n and len(covered) < total:
        best, best_new = None, -1
        for c in itertools.combinations(range(len(order)), size):
            new = sum(1 for p in itertools.combinations(c, 2) if p not in covered)
            if new > best_new:
                best, best_new = c, new
                if new == full:
                    break
        covered.update(itertools.combinations(best, 2))
        out.append(frozenset(order[i] for i in best))
    return out


def lookahead_order(order, la: dict | None, s=S) -> list:
    if not la:
        return []
    rank = {f: i for i, f in enumerate(order)}
    pos = lambda T: sorted(rank[f] for f in T)  # noqa: E731
    vals = {frozenset(r["families"]): r["lookahead"] for r in la["table"].values() if len(r["families"]) <= s}
    from llmdw.search.relabel import tie_classes          # values within 1e-9 tied, as in llmdw.scoring.results
    cls = tie_classes(v for v in vals.values() if v is not None)
    have = sorted((T for T in vals if vals[T] is not None), key=lambda T: (cls[vals[T]], len(T), pos(T)))
    rest = sorted((T for T in all_sets(order, s) if vals.get(T) is None), key=lambda T: (len(T), pos(T)))
    return have + rest


def random_order(order, seed, n, s=S):
    sets = all_sets(order, s)
    return random.Random(seed).sample(sets, min(n, len(sets)))


def staged_lazy(fams, s, predicted, need):
    """search_sim.staged_order, stopping once it holds `need` sets (the later stages are expensive for large k)."""
    P = [f for f in fams if f in frozenset(predicted)]
    if len(P) > s:
        P = P[:s]
    w = len(P)
    if w == 0:
        return []
    rest = [f for f in fams if f not in P]
    order = []
    for e in range(w + 1):
        best = None
        for a in range(min(w, s - e), w - e - 1, -1):
            A = search_sim.greedy_covering(P, a, w - e)
            R = [frozenset()] if e == 0 else search_sim.greedy_covering(rest, s - a, e)
            if best is None or len(A) * len(R) < len(best[0]) * len(best[1]):
                best = (A, R)
        if best is not None:
            order += [x | y for x in best[0] for y in best[1]]
        if len(order) >= need:
            break
    return order


def covering_lazy(fams, s, need):
    """search_sim.covering_order (coverings of the w-subsets for w = 1, 2, ...), stopping once it holds `need` sets."""
    order = []
    for w in range(1, s + 1):
        order += search_sim.greedy_covering(list(fams), s, w)
        if len(order) >= need:
            break
    return order


def interleave(a, b):
    out, seen = [], set()
    for pair in itertools.zip_longest(a, b):
        for x in pair:
            if x is not None and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def jointdw_order(fams, predicted, n, s=S):
    return interleave(staged_lazy(fams, s, predicted, n), covering_lazy(fams, s, n))[:n]


# ---------------------------------------------------------------- instances and evaluations

class Instance:
    def __init__(self, name: str, work: Path):
        self.name, self.dir = name, work / LARGE_DIR / name
        self.mps = str(work / "work/mps" / f"{name}.mps.gz")
        self.meta = json.loads((self.dir / "instance.json").read_text(encoding="utf-8"))
        self.fams = self.meta["families"]
        self.work = work
        self._model = None
        self._view = None
        self._order = None

    @property
    def order(self) -> list:
        """The family order of every method: the first of the 1,000 orders of llmdw.search.relabel."""
        if self._order is None:
            from llmdw.search import relabel
            base = relabel.file_order(self.view, self.fams)
            self._order = relabel.family_orders(self.name, base, 1)[0] if not relabel.IDENTITY else sorted(self.fams)
        return self._order

    @property
    def model(self):
        if self._model is None:
            self._model = read_model(self.mps)
        return self._model

    @property
    def view(self):
        """The stored summary of the model (llmdw.instances.summaries)."""
        if self._view is None:
            from llmdw.instances import summaries
            self._view = summaries.load(self.work, "large_models", self.name, self.mps)
        return self._view

    def lookahead(self):
        p = self.dir / "lookahead.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @property
    def primal(self):
        """The best primal solution value recorded by SCIP in any stored evaluation of the model (rule 3 of
        llmdw.scoring.bounds)."""
        if not hasattr(self, "_primal"):
            from llmdw.scoring import bounds
            rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.dir / "evals").glob("*.row.json"))]
            self._primal = bounds.incumbent(rows, self.meta["sense"])
        return self._primal

    def bound_invalid(self, row: dict):
        """The rule of llmdw.scoring.bounds that makes the recorded bound of row invalid, or None (no reference
        optimum for these models: rules 1 and 3)."""
        from llmdw.scoring import bounds
        return bounds.invalid_reason(row.get("dual_bound"), self.meta["sense"], self.meta["lp_bound"], None, self.primal)

    def gain(self, T) -> float | None:
        """Bound gain of convexifying T (the other families in the master), in minimization units; evaluated once.
        A stored evaluation is used as recorded (its row holds its settings, propagation included); a missing one is
        computed from the MPS file (llmdw.gcg.oracle.evaluate_subset, propagation off)."""
        from llmdw.gcg.oracle import key_tag, round_bound, subset_key
        master = set(self.fams) - set(T)
        path = self.dir / "evals" / f"{key_tag(subset_key(master, None))}.row.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if "cmp_bound" not in row:
                row["cmp_bound"] = round_bound(row.get("dual_bound"), self.meta["sense"], self.view.objective_integral)
        else:
            row = evaluate_subset(self.model, self.mps, self.dir / "evals", master, TIME_LIMIT, seed=0,
                                  integral=self.view.objective_integral)
        b = row.get("cmp_bound")
        lp = self.meta["lp_bound"]
        if b is None or lp is None or self.bound_invalid(row):
            return None
        return _minform(b, self.meta["sense"]) - _minform(lp, self.meta["sense"])


def prepare_one(name: str, dataset: Path, work: Path) -> dict:
    d = work / LARGE_DIR / name
    (d / "evals").mkdir(parents=True, exist_ok=True)
    mps = str(work / "work/mps" / f"{name}.mps.gz")
    model = read_model(mps)
    fams = sorted({family_of(r.name) for r in model.rows})
    meta_path = d / "instance.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if "lp_bound" not in meta:
        lp = lp_bound(mps, d)
        meta = {"name": name, "families": fams, "k": len(fams), "sense": model.sense,
                "lp_status": lp.get("status"), "lp_bound": lp.get("dual_bound") if lp.get("ok") else None}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    la = d / "lookahead.json"
    if not la.exists() and meta["lp_bound"] is not None and meta.get("lp_status") != "infeasible":
        cmd = [sys.executable, "-m", "llmdw.lookahead.lookahead", "--mps", mps, "--out", str(la) + ".part", "--max-size", str(S),
               "--block-time", "60", "--total-time", "3600"]
        cap = os.environ.get("LLMDW_CHILD_MEMORY_GB")
        if cap:
            from llmdw.gcg import address_limit
            cmd = address_limit.prefix(cap) + cmd
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(d / "lookahead.log", "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), env=env, stdout=fh,
                                  stderr=subprocess.STDOUT)
        if proc.returncode == 0 and Path(str(la) + ".part").exists():
            os.replace(str(la) + ".part", la)
    msgs = usage.planned_messages(summarize(model), str(dataset / name) if (dataset / name).is_dir() else None,
                                  "original")
    (d / "messages.json").write_text(json.dumps(msgs, indent=1), encoding="utf-8")
    lav = json.loads(la.read_text(encoding="utf-8")) if la.exists() else None
    return {"instance": name, "k": len(fams), "lp_status": meta.get("lp_status"), "lp_bound": meta["lp_bound"],
            "lookahead": "done" if lav else "missing", "lookahead_truncated": lav.get("truncated") if lav else None,
            "prompt_chars": sum(len(m["content"]) for m in msgs)}


# ---------------------------------------------------------------- methods

def run_greedy(inst: Instance, budget=BUDGET) -> list:
    """Forward selection; returns the evaluated sets in order (at most `budget`)."""
    order, cur, cur_gain = [], frozenset(), 0.0
    while len(order) < budget:
        cands = [cur | {f} for f in inst.order if f not in cur and len(cur) < S]
        if not cands:
            break
        gains = {}
        for T in cands:
            if len(order) >= budget:
                break
            order.append(T)
            gains[T] = inst.gain(T)
        best = max((T for T in gains if gains[T] is not None), key=lambda T: gains[T], default=None)
        if best is None or gains[best] <= cur_gain + 1e-9 or len(gains) < len(cands):
            break
        cur, cur_gain = best, gains[best]
    return order


def first_top_candidate(rec: dict, fams) -> frozenset:
    top = (rec.get("candidates") or [None])[0]
    return frozenset(fams) - frozenset(top["master_families"]) if top else frozenset()


def method_orders(inst: Instance, answers: dict | None = None, budget=BUDGET) -> dict:
    """{method: [sets]} of the non-adaptive methods (greedy runs separately); answers: {config: [records]}."""
    la = lookahead_order(inst.order, inst.lookahead())
    out = {"covering": covering_design(inst.order, budget), "lookahead": la[:budget]}
    for sd in SEEDS:
        out[f"random_{sd}"] = random_order(inst.order, sd, budget)
    for cfg, recs in (answers or {}).items():
        for r in recs:
            jo = jointdw_order(inst.order, first_top_candidate(r, inst.fams), budget)
            out[f"jointdw|{cfg}|s{r.get('sample', 0)}"] = jo
            out[f"interleaved|{cfg}|s{r.get('sample', 0)}"] = interleave(jo, la[:budget])[:budget]
    return out


def load_answers(root: Path, cfg: str, name: str) -> list:
    d = root / cfg / name
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("original*.json"))] if d.exists() else []


def run_instance(name: str, answers_root: Path | None, configs: list, work: Path) -> dict:
    inst = Instance(name, work)
    if inst.meta.get("lp_bound") is None:
        return {"instance": name, "skipped": "no LP bound"}
    answers = {c: load_answers(answers_root, c, name) for c in configs} if answers_root else None
    orders = method_orders(inst, answers)
    orders["greedy"] = run_greedy(inst)
    for T in sorted({T for o in orders.values() for T in o}, key=lambda T: (len(T), sorted(T))):
        inst.gain(T)
    (inst.dir / "orders.json").write_text(json.dumps({m: [sorted(T) for T in o] for m, o in orders.items()}, indent=1),
                                          encoding="utf-8")
    return {"instance": name, "methods": len(orders), "sets": len({T for o in orders.values() for T in o})}


# ---------------------------------------------------------------- analysis

def curves(inst: Instance, budget=BUDGET) -> dict:
    orders = {m: [frozenset(T) for T in o]
              for m, o in json.loads((inst.dir / "orders.json").read_text(encoding="utf-8")).items()}
    gains = {}
    for o in orders.values():
        for T in o:
            if T not in gains:
                gains[T] = inst.gain(T)
    best = max((g for g in gains.values() if g is not None), default=None)
    tol = 1e-6 * max(1.0, abs(inst.meta["lp_bound"]))                      # the census tolerance without an optimum
    out = {"best_gain": best, "positive": best is not None and best > tol,
           "curves": {}}
    for m, o in orders.items():
        cur, c = 0.0, []
        for b in range(budget):
            if b < len(o) and gains.get(o[b]) is not None:
                cur = max(cur, gains[o[b]])
            c.append(cur / best if out["positive"] else None)
        out["curves"][m] = c
    return out


def n_groups(values) -> int:
    """Prompt groups among the instances of values (llmdw.scoring.cluster)."""
    from llmdw.scoring import cluster
    return cluster.n_groups(values)


def boot(values, stat=np.mean, n=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """(lo, hi, se): 95% percentile bootstrap interval and the standard deviation (ddof = 1) of the same replicates;
    values with instance names, prompt groups and their instances resampled (llmdw.scoring.cluster)."""
    from llmdw.scoring import cluster
    return cluster.boot(values, stat, n, seed)


def method_family(m: str) -> tuple[str, str]:
    """('jointdw', c) for 'jointdw|c|s0' with c a configuration; ('random', '') for 'random_1'; (m, '') otherwise."""
    if "|" in m:
        kind, cfg, _ = m.split("|")
        return kind, cfg
    return ("random", "") if m.startswith("random_") else (m, "")


def analyze(names: list, out: Path, work: Path, budget=BUDGET) -> dict:
    per_inst, by_method = [], defaultdict(dict)
    for n in names:
        inst = Instance(n, work)
        if not (inst.dir / "orders.json").exists():
            continue
        c = curves(inst, budget)
        per_inst.append({"instance": n, "k": inst.meta["k"], "best_gain": c["best_gain"], "positive_gain": c["positive"]})
        if not c["positive"]:
            continue
        grouped = defaultdict(list)
        for m, cur in c["curves"].items():
            grouped[method_family(m)].append(cur)
        for key, cs in grouped.items():                      # samples and seeds averaged per instance
            by_method[key][n] = [float(np.mean([x[b] for x in cs])) for b in range(budget)]
    curve_rows, summary = [], []
    for (kind, cfg), d in sorted(by_method.items()):
        for b in range(budget):
            vals = [d[n][b] for n in d]
            lo, hi, se = boot([(n, d[n][b]) for n in d])
            curve_rows.append({"method": kind, "config": cfg, "b": b + 1, "n_instances": len(d),
                               "n_groups": n_groups(d), "mean_share": float(np.mean(vals)), "lo": lo, "hi": hi, "se": se})
        area = [float(np.mean(d[n])) for n in d]
        reach = [next((b + 1 for b in range(budget) if d[n][b] >= 0.95 - 1e-12), None) for n in d]
        rv = [r if r is not None else budget for r in reach]
        alo, ahi, ase = boot(list(zip(list(d), area)))
        summary.append({"method": kind, "config": cfg, "n_instances": len(d), "n_groups": n_groups(d),
                        "area": float(np.mean(area)),
                        "area_lo": alo, "area_hi": ahi, "area_se": ase, "evals_to_95_mean": float(np.mean(rv)),
                        "evals_to_95_median": float(np.median(rv)), "censored_at_budget": sum(r is None for r in reach)})
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("large_models_instances.csv", per_inst), ("large_models_curves.csv", curve_rows),
                        ("large_models_summary.csv", summary)):
        with open(out / fname, "w", newline="", encoding="utf-8") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return {"instances": per_inst, "curves": curve_rows, "summary": summary}


# ---------------------------------------------------------------- report (section of RESULTS.md)

EXCLUSION_RULE = ("Instances with more than 26 constraint families are left out, because the lookahead over all "
                  "sets of at most three families does not fit its time limit.")
CONVERGED = {"optimal", "nodelimit", "gaplimit"}
CURVE_POINTS = (1, 2, 3, 5, 10)


def evaluation_rows(inst: Instance) -> list:
    """The stored evaluations of the sets in the orders of the instance (orders.json); other evaluations in the store
    are not counted."""
    from llmdw.gcg.oracle import key_tag, subset_key
    orders = json.loads((inst.dir / "orders.json").read_text(encoding="utf-8"))
    sets = {frozenset(T) for o in orders.values() for T in o}
    rows = []
    for T in sorted(sets, key=lambda T: (len(T), sorted(T))):
        p = inst.dir / "evals" / f"{key_tag(subset_key(set(inst.fams) - set(T), None))}.row.json"
        if p.exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            r["bound_invalid"] = inst.bound_invalid(r) if r.get("ok", True) else None
            rows.append(r)
    return rows


def outcome(row: dict) -> str:
    """converged, stopped at the limit with a bound, or no bound (no comparable bound, or an invalid one)."""
    if row.get("cmp_bound") is None or row.get("bound_invalid"):
        return "no bound"
    return "converged" if row.get("status") in CONVERGED else "stopped at the limit with a bound"


def report(names: list, out: Path, work: Path, budget=BUDGET) -> tuple[str, dict]:
    """The section on instances with more than 6 families (markdown) and its tables, written as CSVs to out."""
    res = analyze(names, out, work, budget)
    pos = {r["instance"]: r for r in res["instances"]}
    per_inst, evals, errors, invalid = [], Counter(), Counter(), []
    for n in names:
        inst = Instance(n, work)
        rows = evaluation_rows(inst)
        oc = Counter(outcome(r) for r in rows)
        evals.update(oc)
        errors.update(("invalid bound (rule " + r["bound_invalid"] + ")") if r.get("bound_invalid") else
                      (r.get("error") or r.get("status") or "unknown")[:70] for r in rows if outcome(r) == "no bound")
        bad = [r for r in rows if r.get("bound_invalid")]
        evals["invalid bound"] += len(bad)
        invalid += [{"instance": n, "key": r.get("key"), "dual_bound": r.get("dual_bound"), "status": r.get("status"),
                     "solving_time": r.get("solving_time"), "rule": r["bound_invalid"], "primal_solution": inst.primal}
                    for r in bad]
        la = inst.lookahead()
        with_bound = [r for r in rows if r.get("cmp_bound") is not None and not r.get("bound_invalid")]
        kind = ("no bound" if not with_bound else "positive gain" if pos.get(n, {}).get("positive_gain")
                else "bounds, no positive gain")
        per_inst.append({"instance": n, "k": inst.meta["k"], "lp_bound": inst.meta["lp_bound"],
                         "evaluations": len(rows), "converged": oc["converged"],
                         "at_limit_with_bound": oc["stopped at the limit with a bound"], "no_bound": oc["no bound"],
                         "invalid_bound": len(bad),
                         "best_gain": pos.get(n, {}).get("best_gain"), "outcome": kind,
                         "lookahead_sets": len(la["table"]) if la else 0,
                         "lookahead_sets_planned": len(all_sets(inst.fams)),
                         "lookahead_cut_off": bool(la and la.get("truncated"))})
    outcome_rows = [{"outcome": k, "evaluations": evals[k]} for k in
                    ("converged", "stopped at the limit with a bound", "no bound", "invalid bound")]
    error_rows = [{"error": e, "evaluations": c} for e, c in errors.most_common()]
    kinds = Counter(r["outcome"] for r in per_inst)
    tables = {"large_models_per_instance": per_inst, "large_models_outcomes": outcome_rows,
              "large_models_errors": error_rows}
    if invalid:
        tables["large_models_invalid_bounds"] = invalid
    for fname, rows in tables.items():
        with open(out / f"{fname}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def tab(rows, keys):
        f = lambda v: "" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))  # noqa: E731
        return "\n".join(["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] +
                          ["| " + " | ".join(f(r.get(k)) for k in keys) + " |" for r in rows])
    cut = [r for r in per_inst if r["lookahead_cut_off"]]
    curve_pts = [r for r in res["curves"] if r["b"] in CURVE_POINTS]
    md = ["## 8. Budgeted search on instances with more than 6 families", "",
          EXCLUSION_RULE, "",
          f"Instances: the {len(names)} exported MIPLIB-NL instances skipped by the oracle for more than 6 constraint families "
          "that have at most 26 (none has general constraints). Setting: no linking variables, sets of at most 3 "
          "convexified families, the oracle's evaluation setting (GCG with the given decomposition through PyGCGOpt, root "
          "node, no presolve, cuts or primal heuristics, 120 s, one thread, 8 instances in parallel, 5 GB per solve); "
          "every evaluated set is stored per instance and shared by the methods. Methods without LLM, budget "
          f"{budget} evaluations, all in one order of the families (the first of the 1,000 family orders of "
          "llmdw.search.relabel, which does not depend on their names): covering (greedy covering of all pairs by sets of 3 "
          "families), greedy forward selection, lookahead order (sets of at most 3 families by decreasing lookahead, "
          "ties to fewer families, then to the earlier positions), random orders (seeds 0, 1, 2; averaged per "
          "instance). JointDW and "
          "the interleaved order were not run (no language-model calls on these instances).", "",
          f"Outcome per instance: {kinds['no bound']} without any bound, {kinds['bounds, no positive gain']} with bounds "
          f"but no positive gain, {kinds['positive gain']} with a positive gain (tolerance 1e-6 max(1, |LP|)). "
          "`large_models_per_instance.csv`.", "",
          tab(per_inst, ["instance", "k", "evaluations", "converged", "at_limit_with_bound", "no_bound", "invalid_bound",
                         "best_gain", "outcome", "lookahead_sets", "lookahead_sets_planned", "lookahead_cut_off"]), "",
          "Evaluation outcomes over all instances (`large_models_outcomes.csv`) and the errors of the evaluations "
          "without a bound (`large_models_errors.csv`; the same types occurred in the oracle runs). A recorded bound "
          "that llmdw.scoring.bounds marks invalid counts as no bound; the line \"invalid bound\" counts these among "
          "the evaluations without a bound, and `large_models_invalid_bounds.csv` lists them:", "",
          tab(outcome_rows, ["outcome", "evaluations"]), "", tab(error_rows, ["error", "evaluations"]), "",
          *([tab(invalid, ["instance", "key", "dual_bound", "status", "solving_time", "rule", "primal_solution"]), ""]
            if invalid else []),
          "Lookahead cut off at the 3,600 s limit (every set of 1 and 2 families has a value; the triples without one "
          "come last in the lookahead order): " + (", ".join(f"{r['instance']} ({r['lookahead_sets']} of "
                                                              f"{r['lookahead_sets_planned']} sets)" for r in cut)
                                                    or "none") + ".", "",
          f"Curves on the {kinds['positive gain']} instances with a positive gain: after b evaluations, the best bound "
          "so far as a share of the gap between the LP bound and the best bound found by any method on the instance; "
          "means over instances with 95% bootstrap intervals (10,000 resamples, seed 0). Area = mean over b = 1 to "
          f"{budget}; evaluations to 95 percent of the best found, censored at {budget}. `large_models_curves.csv` (every b), "
          "`large_models_summary.csv`.", "",
          tab(curve_pts, ["method", "b", "n_instances", "mean_share", "lo", "hi", "se"]), "",
          tab(res["summary"], ["method", "n_instances", "area", "area_lo", "area_hi", "area_se", "evals_to_95_mean",
                               "evals_to_95_median", "censored_at_budget"]), ""]
    return "\n".join(md), tables


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "run", "analyze"])
    ap.add_argument("--work", required=True, help="the work root (holds work/large_models and work/mps)")
    ap.add_argument("--dataset", help="prepare: the MIPLIB-NL dataset/ directory (descriptions for the prompts)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--answers", help="answer root of the predictions on these instances (<root>/<config>/<instance>/original*.json)")
    ap.add_argument("--configs", default="", help="model identifiers of the configurations with answers")
    ap.add_argument("--out", help="analyze: output directory of the tables")
    args = ap.parse_args(argv)
    work = Path(args.work).resolve()
    names = [ln.strip() for ln in (work / LARGE_DIR / "instances.txt").read_text().splitlines() if ln.strip()]
    configs = [c for c in args.configs.split(",") if c]
    if args.step == "prepare":
        with ThreadPoolExecutor(args.parallel) as pool:
            for r in pool.map(lambda n: prepare_one(n, Path(args.dataset), work), names):
                print(json.dumps(r), flush=True)
    elif args.step == "run":
        root = Path(args.answers) if args.answers else None
        with ThreadPoolExecutor(args.parallel) as pool:
            for r in pool.map(lambda n: run_instance(n, root, configs if root else [], work), names):
                print(json.dumps(r), flush=True)
    else:
        res = analyze(names, Path(args.out or work / LARGE_DIR / "results"), work)
        print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
