"""Time of one evaluation on the templates and the rich models (7 to 9 families, and 12 to 16 families): GCG's root
column generation on the family sets that the search orders of the result tables visit.

The search tables of the templates and the rich models count evaluations of family sets; only MIPLIB-NL has a time per
set (llmdw.scoring.results.oracle_times, from the census). This module measures, for the templates and the rich models,
the time of one evaluation as GCG's root evaluation of the set in the census setting, on the sets that some search order
of those tables actually evaluates (not on the whole tables).

paths   The template and rich-model phases exactly as llmdw.scoring.results builds them (load_instances, load_answers,
        score_all, baseline_search, search_by_size, paired_search, theorem2, shortlist_search, ranked_and_narrow,
        ranked_prompt_tables, and llmdw.scoring.rich_models.llm_context and rich_tables, the tables of the prompt with
        data included), run with a recorder installed around llmdw.search.relabel: relabel.runs is wrapped to know the
        instance and the search order of each run, and relabel.first_hit and relabel.greedy_hit receive a recording
        mapping as `times` whose get(key, default) notes key. These two functions consult `times` exactly on the sets a
        successful run evaluates (relabel.seconds over the evaluated sets), so the record is the union, over every
        family order of llmdw.search.relabel, every size limit and target, every configuration, presentation and
        sample, of the sets evaluated by covering, greedy, lookahead, the model anchors (first support and top
        candidate), the GCG anchors, the interleavings, the shortlists, the ranked lists and "narrow then lookahead"
        with their lookahead interleavings. A run that reaches no target gives its instance no value in any table and
        is counted, not recorded. The returned evaluations are those of an unpatched run, and the seconds stay None
        where the tables pass no times (relabel's memo is cleared first, so every run is computed under the recorder).
        Bootstrap intervals (llmdw.scoring.cluster.boot) and prompt groups are replaced by no-ops while the tables are
        built, since they do not change which sets are evaluated, and the CSV files of rich_tables go to a temporary
        folder. The random order is excluded: its expected cost runs over every set of the table.
        --set large records instead the search orders of the rich models with 12 to 16 families
        (llmdw.scoring.rich_models_large: load_set, load_answers and search_values on the instances of
        work/rich_models_large, the answers of both presentations; collection "rich_models_large"), under the
        same recorder: the covering order, the lookahead order at the HiGHS dual and at the best of five duals, the GCG
        anchors, JointDW and JointDW interleaved with the five-dual order, for s = 2 and 3 (s = 2 only where the gain
        table stops at two families, network design with k = 14). The published rich models scored alongside them in
        experiment_e are the collection "rich_models" of the default set and are not recorded again.
        Writes paths.json (per instance: collection, class, families, k, the size limits used, table size, the model
        file, the visited sets "Fam1|Fam2" sorted by size then names, and per search order the sets visited and the
        failed runs) and paths_summary.csv (per instance, per collection and class, and per collection).
run     For each visited set one GCG root evaluation in the census setting, by llmdw.gcg.oracle.evaluate_subset (master =
        the other families, no linking variable families, blocks by connected components, the decomposition written as
        <tag>.dec by llmdw.gcg.dec_io.write_dec, then llmdw.gcg.gcg_run.run "dw-given" with root, no cuts, no rounding
        device, no heuristics, SCIP propagation off (the default of dw-given), one thread, seed 0, time limit 120 s, the
        retries of llmdw.gcg.oracle.run_with_retries), so the machine-wide caps of gcg_run (LLMDW_MAX_SOLVERS,
        llmdw.gcg.memory_budget, LLMDW_CHILD_MEMORY_GB) apply. One record per instance and set,
        <out>/runs/<instance>/<tag>.row.json (tag = llmdw.gcg.oracle.key_tag of the master families); an existing record
        is skipped, so the step resumes. A set whose decomposition has no block or a single block and an empty master
        is not run (as in the census) and gets a record without a time.
tables  Per collection (templates, rich models, rich models with 12 to 16 families) and class, and over each
        collection: the mean time of one evaluation,
        where the time of a set is that of the published census, by the same function
        (llmdw.scoring.results.oracle_seconds, used by oracle_times: wall_seconds of the solver child, else GCG's
        solving_time, a missing value counted at the limit, capped at 120 s); a set is at the limit when that time is
        120 s. Sets the census setting does not run (no block, or a single block with an empty master) are left out
        and counted. Per instance the mean over its visited sets, then the mean over instances with the standard error
        sd / sqrt(n) (ddof 1); and the mean over all visited sets pooled, with sd / sqrt(number of sets). GCG's
        reported time (solving_time, getSolvingTime of the original program, "Solving Time (sec)" in the statistics;
        root only, so the root time; capped at 120 s, a missing value at the limit) is reported in separate columns
        with the same statistics. Sets without a record are left out and counted. Writes eval_time.csv and eval_time_instances.csv.

Usage
-----
python -m llmdw.benchmark.eval_time paths --work data --out results/eval_time
python -m llmdw.benchmark.eval_time paths --set large --work data/work --out results/eval_time_large \
    [--configs gpt-6.1-sol,claude-sonnet-5-5]           (--work holds rich_models_large/)
python -m llmdw.benchmark.eval_time run --paths results/eval_time/paths.json --out results/eval_time --parallel 4
python -m llmdw.benchmark.eval_time tables --paths results/eval_time/paths.json --runs results/eval_time/runs \
    --out results/eval_time
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from llmdw.constants import FAMILY_ORDERS
from llmdw.search import relabel

COLLECTIONS = ("templates", "rich_models", "rich_models_large")
SETS = ("published", "large")      # the collections recorded by paths: templates and rich models, or the larger ones
EVAL_TIME = 120.0            # the census time limit (llmdw.benchmark.truth.EVAL_TIME, results.EVAL_LIMIT_S)
SEED = 0


def set_name(T) -> str:
    return "|".join(sorted(T))


def set_sort(keys):
    """Set names sorted by size, then by the sorted names (relabel.set_key)."""
    return sorted(keys, key=lambda k: relabel.set_key(frozenset(x for x in k.split("|") if x)))


# ---------------------------------------------------------------- recording the evaluated sets

class _Probe:
    """A `times` mapping for relabel.first_hit / greedy_hit: get(key, default) notes key and answers as the mapping it
    wraps (default when there is none)."""

    def __init__(self, note, times):
        self.note, self.times = note, times

    def get(self, key, default=None):
        self.note(key)
        return default if self.times is None else self.times.get(key, default)


class Recorder:
    """Per instance the sets evaluated by successful search runs, with the search order (spec kind) of each run."""

    def __init__(self):
        self.sets = defaultdict(lambda: defaultdict(set))        # instance -> set -> {kinds}
        self.failed = defaultdict(lambda: defaultdict(int))      # instance -> kind -> failed runs
        self.runs = defaultdict(lambda: defaultdict(int))        # instance -> kind -> runs
        self.sizes = defaultdict(set)                            # instance -> {(s, full)}
        self._ctx = threading.local()

    def _note(self, T):
        name, kind = self._ctx.current
        self.sets[name][frozenset(T)].add(kind)

    @contextmanager
    def installed(self):
        """Wrap relabel.runs, relabel.first_hit and relabel.greedy_hit (restored on exit); clears relabel's memo before
        and after."""
        orig_runs, orig_first, orig_greedy = relabel.runs, relabel.first_hit, relabel.greedy_hit
        rec = self

        def runs(key, name, fams, fn):
            if isinstance(key, tuple) and len(key) == 6 and key[0] == name:      # results.search_runs
                s, full, spec = key[1], key[2], key[4]
                kind = spec[0]
                rec.sizes[name].add((s, bool(full)))
            else:
                kind = str(key[0])

            def wrapped(order):
                prev = getattr(rec._ctx, "current", None)
                rec._ctx.current = (name, kind)
                try:
                    c, t = fn(order)
                finally:
                    rec._ctx.current = prev
                rec.runs[name][kind] += 1
                if c is None:
                    rec.failed[name][kind] += 1
                return c, t
            return orig_runs(key, name, fams, wrapped)

        def first_hit(order, tab, goal, times=None, limit=120.0):
            c, sec = orig_first(order, tab, goal, _Probe(rec._note, times), limit)
            return c, (sec if times is not None else None)

        def greedy_hit(tab, goal, fams, times=None, limit=120.0):
            c, sec = orig_greedy(tab, goal, fams, _Probe(rec._note, times), limit)
            return c, (sec if times is not None else None)

        relabel.clear()
        relabel.runs, relabel.first_hit, relabel.greedy_hit = runs, first_hit, greedy_hit
        try:
            yield self
        finally:
            relabel.runs, relabel.first_hit, relabel.greedy_hit = orig_runs, orig_first, orig_greedy
            relabel.clear()


@contextmanager
def _quiet_tables():
    """No bootstrap and no prompt groups while the tables are built (neither changes an evaluated set)."""
    from llmdw.scoring import cluster, results

    orig_boot, orig_groups = cluster.boot, results.use_prompt_groups
    cluster.boot = lambda *a, **k: (None, None, None)
    results.use_prompt_groups = lambda *a, **k: []
    try:
        yield
    finally:
        cluster.boot, results.use_prompt_groups = orig_boot, orig_groups
        cluster.set_groups({})


def build_tables(work: Path, configs: list) -> dict:
    """Run the template and rich-model search tables of llmdw.scoring.results (as its main does for these phases);
    returns {collection: {instance: {"class", "truth", "model"}}} and the answer counts per configuration."""
    from llmdw.scoring import results, rich_models

    out, answers = {}, {}
    phase = "templates"
    instances = results.load_instances(work, phase)
    results.baseline_search(phase, instances)
    recs_by = {}
    for cfg in configs:
        recs = results.load_answers(work / f"work/support_{phase}/{cfg}", instances)
        answers[f"{phase}/{cfg}"] = sum(1 for r in recs if r["status"] != "missing")
        results.score_all(recs, instances)
        recs_by[cfg] = recs
    results.search_by_size(phase, recs_by, instances)
    results.paired_search(phase, recs_by, instances, work)
    results.theorem2(phase, recs_by, instances, work)
    results.shortlist_search(phase, recs_by, instances)
    results.ranked_and_narrow(phase, recs_by, instances)
    orig_recs = {phase: {c: [r for r in recs if r["condition"] == "original"] for c, recs in recs_by.items()}}
    out[phase] = {n: {"class": i["class"], "truth": i["truth"], "model": work / "work/templates" / n / "model.mps"}
                  for n, i in instances.items()}
    phase_inst = {phase: instances}
    r_inst, r_recs = rich_models.llm_context(work, configs)
    for cfg, recs in r_recs.items():
        answers[f"{results.RICH_PHASE}/{cfg}"] = sum(1 for r in recs if r["status"] != "missing")
    phase_inst[results.RICH_PHASE], orig_recs[results.RICH_PHASE] = r_inst, r_recs
    if r_recs:
        results.shortlist_search(results.RICH_PHASE, r_recs, r_inst, limits=("2", "3"), conditions=("original",))
        results.ranked_and_narrow(results.RICH_PHASE, r_recs, r_inst, limits=("2", "3"), conditions=("original",))
    if (work / "work/support_ranked").exists():
        results.ranked_prompt_tables(work, configs, phase_inst, orig_recs)
    with tempfile.TemporaryDirectory() as tmp:
        rich_models.rich_tables(work, Path(tmp), configs)
    out["rich_models"] = {n: {"class": i["class"], "truth": i["truth"],
                              "model": work / rich_models.RICH_DIR / "instances" / n / "model.mps"}
                          for n, i in r_inst.items()}
    return {"collections": out, "answers": answers}


def build_tables_large(work: Path, configs: list) -> dict:
    """Run the search orders of llmdw.scoring.rich_models_large on the rich models with 12 to 16 families (work holds
    rich_models_large/); returns {collections: {"rich_models_large": {instance: {"class", "truth", "model"}}}, answers}."""
    from llmdw.scoring import rich_models_large as xe

    insts = xe.load_set(work / xe.E_ROOT, work / xe.E_FIVE, work / xe.E_ROOT, "rich_models_large")
    recs, status = xe.load_answers({p: work / d for p, d in xe.E_ANSWERS.items()}, insts, configs, False)
    xe.search_values(insts, recs)
    answers = {f"rich_models_large/{r['configuration']}/{r['presentation']}": r["answer_files"] for r in status}
    out = {i["name"]: {"class": i["class"], "truth": i["truth"], "model": i["dir"] / "model.mps"} for i in insts}
    return {"collections": {"rich_models_large": out}, "answers": answers}


def collect(work: Path, configs: list, family_orders: int = FAMILY_ORDERS, which: str = "published") -> dict:
    """paths.json content (see the module docstring); which: "published" (templates and rich models with 7 to 9
    families) or "large" (rich models with 12 to 16 families)."""
    if which not in SETS:
        raise ValueError(which)
    relabel.configure(family_orders)
    rec = Recorder()
    with rec.installed(), _quiet_tables():
        built = build_tables(work, configs) if which == "published" else build_tables_large(work, configs)
    inst_out = {}
    for coll, insts in built["collections"].items():
        for n, i in sorted(insts.items()):
            t = i["truth"]
            fams = sorted(t["families"])
            tab = [T for T in t["table"] if T]
            visited = rec.sets.get(n, {})
            outside = [T for T in visited if T not in t["table"]]
            if outside:
                raise ValueError(f"{n}: visited sets outside the table: {outside[:3]}")
            by_kind = defaultdict(set)
            for T, kinds in visited.items():
                for kd in kinds:
                    by_kind[kd].add(T)
            inst_out[n] = {"collection": coll, "class": i["class"], "families": fams, "k": len(fams),
                           "s": sorted({s for s, _ in rec.sizes.get(n, set())}),
                           "full_table_used": any(f for _, f in rec.sizes.get(n, set())),
                           "table_size": len(tab), "max_set_size": max((len(T) for T in tab), default=0),
                           "model": str(i["model"]),
                           "visited": set_sort(set_name(T) for T in visited),
                           "visited_by_order": {kd: len(v) for kd, v in sorted(by_kind.items())},
                           "runs_by_order": dict(sorted(rec.runs.get(n, {}).items())),
                           "failed_runs_by_order": {kd: c for kd, c in sorted(rec.failed.get(n, {}).items()) if c}}
    return {"settings": {"set": which, "family_orders": family_orders, "configs": configs, "work": str(work),
                         "answer_files": built["answers"], "time_limit": EVAL_TIME, "seed": SEED,
                         "excluded": "random order (its expected cost runs over every set of the table)"},
            "instances": inst_out}


def _quantiles(xs):
    return (min(xs), statistics.median(xs), max(xs)) if xs else (None, None, None)


def summary_rows(paths: dict) -> list:
    rows = []
    insts = paths["instances"]
    for n, i in sorted(insts.items(), key=lambda kv: (kv[1]["collection"], kv[1]["class"], kv[0])):
        v = len(i["visited"])
        rows.append({"level": "instance", "collection": i["collection"], "class": i["class"], "instance": n,
                     "k": i["k"], "n_instances": 1, "visited": v, "table_size": i["table_size"],
                     "share": v / i["table_size"] if i["table_size"] else None,
                     "visited_min": v, "visited_median": v, "visited_max": v})
    groups = defaultdict(list)
    for n, i in insts.items():
        groups[("class", i["collection"], i["class"])].append(i)
        groups[("collection", i["collection"], "all")].append(i)
    for (level, coll, cls), xs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0] == "collection", kv[0][2])):
        v, tb = sum(len(i["visited"]) for i in xs), sum(i["table_size"] for i in xs)
        lo, med, hi = _quantiles([len(i["visited"]) for i in xs])
        rows.append({"level": level, "collection": coll, "class": cls, "instance": "", "k": "",
                     "n_instances": len(xs), "visited": v, "table_size": tb, "share": v / tb if tb else None,
                     "visited_min": lo, "visited_median": med, "visited_max": hi})
    return rows


def write_csv(path: Path, rows: list):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


# ---------------------------------------------------------------- GCG root evaluations

def record_path(runs_dir: Path, name: str, fams, T) -> Path:
    """The record of the evaluation of T (convexified families) on instance name."""
    from llmdw.gcg.oracle import key_tag, subset_key
    return Path(runs_dir) / name / f"{key_tag(subset_key(sorted(set(fams) - set(T))))}.row.json"


def jobs(paths: dict, runs_dir: Path) -> list:
    """(instance, convexified set) of every visited set without a record."""
    out = []
    for n, i in sorted(paths["instances"].items()):
        for key in i["visited"]:
            T = frozenset(x for x in key.split("|") if x)
            if not record_path(runs_dir, n, i["families"], T).exists():
                out.append((n, T))
    return out


def evaluate(paths: dict, out: Path, parallel: int, time_limit: float = EVAL_TIME, memory_gb: float | None = None,
             names: set | None = None) -> int:
    """GCG root evaluation of every visited set without a record (see the module docstring); returns the number run."""
    from llmdw.gcg import gcg_run
    from llmdw.gcg.oracle import evaluate_subset
    from llmdw.instances.families import family_of, objective_integral, read_model

    runs_dir = Path(out).resolve() / "runs"
    pending = [j for j in jobs(paths, runs_dir) if names is None or j[0] in names]
    models = {}
    for n in sorted({n for n, _ in pending}):
        i = paths["instances"][n]
        model = read_model(i["model"])
        fams = sorted({family_of(r.name) for r in model.rows})
        if fams != sorted(i["families"]):
            raise ValueError(f"{n}: the families of {i['model']} differ from those of the table")
        models[n] = (model, objective_integral(model))
        (runs_dir / n).mkdir(parents=True, exist_ok=True)
    print(f"{len(pending)} evaluations to run ({parallel} at a time)", flush=True)

    def one(job):
        n, T = job
        i = paths["instances"][n]
        model, integral = models[n]
        master = set(i["families"]) - set(T)
        row = evaluate_subset(model, i["model"], runs_dir / n, master, time_limit, SEED, integral=integral,
                              memory_gb=memory_gb)
        path = record_path(runs_dir, n, i["families"], T)
        row = {**row, "instance": n, "convexified": sorted(T)}
        gcg_run.write_json_atomic(path, row)
        print(f"{n} {set_name(T)}: {row.get('status')} {row.get('wall_seconds')}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        done = list(pool.map(one, pending))
    return len(done)


# ---------------------------------------------------------------- tables

NOT_RUN = ("no_blocks", "degenerate_single_block")


def eval_seconds(row: dict, limit: float = EVAL_TIME) -> tuple[float | None, bool]:
    """(time of one evaluation as in the census, at the limit); (None, False) for a set that is not run."""
    from llmdw.scoring import results

    if row.get("status") in NOT_RUN:
        return None, False
    t = results.oracle_seconds(row, limit)
    return t, t >= limit


def gcg_seconds(row: dict, limit: float = EVAL_TIME) -> float | None:
    """GCG's reported time, capped at the limit (a missing value at the limit); None if not run."""
    if row.get("status") in NOT_RUN:
        return None
    t = row.get("solving_time")
    return min(limit, t) if t is not None else limit


def _mean_se(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    return m, se


def time_tables(paths: dict, runs_dir: Path, limit: float = EVAL_TIME) -> tuple[list, list]:
    """(rows per collection and class, rows per instance); see the module docstring."""
    per_inst, per_wall, lim_inst, inst_rows = {}, {}, {}, []
    for n, i in sorted(paths["instances"].items()):
        ts, ws, missing, capped, failed, not_run, no_time = [], [], 0, 0, 0, 0, 0
        for key in i["visited"]:
            T = frozenset(x for x in key.split("|") if x)
            p = record_path(runs_dir, n, i["families"], T)
            if not p.exists():
                missing += 1
                continue
            row = json.loads(p.read_text(encoding="utf-8"))
            t, at = eval_seconds(row, limit)
            if t is None:
                not_run += 1
                continue
            ts.append(t)
            ws.append(gcg_seconds(row, limit))
            capped += at
            failed += row.get("status") == "failed" or row.get("ok") is False
            no_time += row.get("solving_time") is None
        m, se = _mean_se(ts)
        wm, wse = _mean_se(ws)
        per_inst[n], per_wall[n], lim_inst[n] = ts, ws, capped
        inst_rows.append({"collection": i["collection"], "class": i["class"], "instance": n, "k": i["k"],
                          "visited": len(i["visited"]), "with_record": len(ts) + not_run, "missing": missing,
                          "not_run": not_run, "with_time": len(ts), "at_limit": capped, "failed": failed,
                          "without_gcg_time": no_time,
                          "mean_seconds": m, "se_seconds": se,
                          "max_seconds": max(ts) if ts else None, "sum_seconds": sum(ts) if ts else None,
                          "mean_gcg_seconds": wm, "se_gcg_seconds": wse,
                          "sum_gcg_seconds": sum(ws) if ws else None})
    groups = defaultdict(list)
    for n, i in paths["instances"].items():
        groups[(i["collection"], i["class"])].append(n)
        groups[(i["collection"], "all")].append(n)
    rows = []
    order = {c: j for j, c in enumerate(COLLECTIONS)}
    by = {r["instance"]: r for r in inst_rows}
    for (coll, cls), ns in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1] == "all", kv[0][1])):
        means = [sum(per_inst[n]) / len(per_inst[n]) for n in ns if per_inst[n]]
        wmeans = [sum(per_wall[n]) / len(per_wall[n]) for n in ns if per_wall[n]]
        pooled = [t for n in ns for t in per_inst[n]]
        wpooled = [t for n in ns for t in per_wall[n]]
        mi, sei = _mean_se(means)
        mp, sep = _mean_se(pooled)
        wmi, wsei = _mean_se(wmeans)
        wmp, wsep = _mean_se(wpooled)
        rows.append({"collection": coll, "class": cls, "n_instances": len(ns), "n_instances_with_time": len(means),
                     "n_sets": sum(len(paths["instances"][n]["visited"]) for n in ns),
                     "n_sets_missing": sum(by[n]["missing"] for n in ns),
                     "n_sets_not_run": sum(by[n]["not_run"] for n in ns), "n_sets_with_time": len(pooled),
                     "n_sets_at_limit": sum(lim_inst[n] for n in ns),
                     "n_sets_without_gcg_time": sum(by[n]["without_gcg_time"] for n in ns),
                     "mean_of_instance_means": mi, "se_over_instances": sei,
                     "pooled_mean": mp, "pooled_se": sep,
                     "gcg_mean_of_instance_means": wmi, "gcg_se_over_instances": wsei,
                     "gcg_pooled_mean": wmp, "gcg_pooled_se": wsep})
    return rows, inst_rows


# ---------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paths", help="the family sets that the search orders visit")
    p.add_argument("--work", required=True, help="the work root (holds work/), as for llmdw.scoring.results; with "
                   "--set large the folder that holds rich_models_large/ (data/work)")
    p.add_argument("--set", choices=SETS, default="published",
                   help="published: templates and rich models (7 to 9 families); large: rich models, 12 to 16 families")
    p.add_argument("--out", required=True)
    p.add_argument("--configs", help="model identifiers, comma-separated (default: every configuration of --settings)")
    p.add_argument("--settings", default=None, help="the configurations (default: config/public_settings.json)")
    p.add_argument("--family-orders", type=int, default=FAMILY_ORDERS)
    r = sub.add_parser("run", help="GCG root evaluation of every visited set (resumable)")
    r.add_argument("--paths", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--time", type=float, default=EVAL_TIME)
    r.add_argument("--memory-gb", type=float, default=None, help="per-process cap (default LLMDW_CHILD_MEMORY_GB)")
    r.add_argument("--names", help="only these instances, comma-separated")
    t = sub.add_parser("tables", help="mean time of one evaluation per collection and class")
    t.add_argument("--paths", required=True)
    t.add_argument("--runs", required=True, help="the runs/ folder written by run")
    t.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "paths":
        from llmdw.providers import settings
        if args.configs:
            configs = [c for c in args.configs.split(",") if c]
        elif args.set == "large":
            from llmdw.scoring import rich_models_large
            configs = list(rich_models_large.CONFIGS)
        else:
            configs = list(settings.names(args.settings))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        paths = collect(Path(args.work).resolve(), configs, args.family_orders, args.set)
        (out / "paths.json").write_text(json.dumps(paths, indent=1), encoding="utf-8")
        write_csv(out / "paths_summary.csv", summary_rows(paths))
        tot = sum(len(i["visited"]) for i in paths["instances"].values())
        print(f"{len(paths['instances'])} instances, {tot} visited sets -> {out}", flush=True)
    elif args.cmd == "run":
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        names = {x for x in args.names.split(",") if x} if args.names else None
        evaluate(paths, Path(args.out), args.parallel, args.time, args.memory_gb, names)
    else:
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        rows, inst_rows = time_tables(paths, Path(args.runs))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "eval_time.csv", rows)
        write_csv(out / "eval_time_instances.csv", inst_rows)
        print(f"wrote eval_time.csv and eval_time_instances.csv to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
