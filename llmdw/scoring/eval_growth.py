"""Time of one evaluation against the number of convexified families of the set.

The size of a family set is the number of its convexified families (the families of the model minus the master
families). Three sources, each optional on the command line:

census   (--oracle-root) The published MIPLIB-NL census: every <oracle-root>/<model>/oracle.json. The convexified
         families of a record are the families of the model minus master_families; records with linking variable
         families are skipped, as in llmdw.scoring.results.oracle_times. The time of a set is the census time
         (llmdw.benchmark.eval_time.eval_seconds, that is llmdw.scoring.results.oracle_seconds: wall_seconds of the
         solver child, else GCG's solving_time, a missing value counted at the limit, capped at 120 s); a set is at
         the limit when that time is 120 s. A record with status no_blocks or degenerate_single_block (a set the census
         setting does not run) has no time: it is left out and counted. Records with status failed keep their census
         time and are counted. The census corrections file (census_corrections.json: replaced reference optima, dropped
         bounds, notes) concerns optima and bounds only; it changes no time, and the record whose bound it drops keeps
         its time, as in oracle_times, which reads the records without the corrections. The file is not read.
eval     (--paths, --runs) The GCG root evaluations of llmdw.benchmark.eval_time on the templates and the rich models:
         per instance of paths.json (collection and class from it) the visited sets only, each read from
         eval_time.record_path. Two times per set, reported separately: the census time (eval_time.eval_seconds, at
         the limit when 120 s) and GCG's time (eval_time.gcg_seconds: solving_time capped at 120 s, a missing value at
         the limit; at the limit when 120 s). Sets with a status in eval_time.NOT_RUN are left out and counted; sets
         without a record are left out and counted.
bnp      (--d-runs) Branch-and-price runs of llmdw.benchmark.time_truth, read from the bnp_runs.csv file given (this
         module only reads that file; it starts no run). The reference rows (set gcg_default and scip) are excluded.
         Three times per run: root_time, pricing_time and time (the total time). A run value is censored when it is
         missing or at least the limit (llmdw.benchmark.bnp.BNP_TIME = 1800 s), and for the total time also when the
         run is not solved. Per instance and set the seed median of each time is the median over its seeds with the
         censored values ordered above every other value; the set is at the limit for that time when this median
         involves a censored value (at least half of the seeds censored), so for the total time when the median-seed
         run is unsolved or at the limit. The sets at the limit with at least half of their seeds without a value are
         also counted on their own.
         Sets with a run that failed twice are counted.

Measures (sets at the limit are counted and left out of every median, quartile, slope and ratio):

by_size.csv   per collection (class "all"), per class for the rich models and the branch-and-price instances, per time
              and per set size: the number of models with a set of that size, the number of sets with a time, the
              number not run, the number at the limit, the number below the limit, the number of those with time 0,
              and the median and quartiles of the time among the sets below the limit.
slopes.csv    level "model": per model (instance) and time, the median time per size over its sets below the limit;
              a size whose median is 0 cannot be logged and is dropped and counted; with at least three sizes left, the
              least-squares slope of log(median) on log(size). Level "group": per collection (and class as in
              by_size) and time, the number of models with a set below the limit, the number with a slope, and the
              median and quartiles of the slopes.
d_ratios.csv  (branch-and-price only) per instance and for root and pricing time, the median over the sets of size 2
              of the seed medians divided by the same median over the sets of size 1, and size 3 over size 2 (sets
              below the limit only); a ratio with a zero denominator is undefined and counted. Level "class": per class
              (and "all") the median and quartiles over instances, with the number of instances having sets of both
              sizes below the limit.

Medians: statistics.median. Quartiles: linear interpolation between order statistics (numpy's default,
llmdw.scoring.gain_structure.quantile), q25 and q75.

Outputs: --out receives by_size.csv, slopes.csv and (with --d-runs) d_ratios.csv; --tex the table body; --figure the
figure data (per collection, class, time and size: counts, median and quartiles).

Usage
-----
python -m llmdw.scoring.eval_growth --oracle-root data/work/oracle \
    --paths results/eval_time/paths.json --runs results/eval_time/runs [--d-runs results/experiment_D/bnp_runs.csv] \
    --out results/eval_growth --tex tables/new/eval_growth.tex --figure figures/data/eval_growth.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from llmdw.benchmark import bnp, eval_time, time_truth
from llmdw.scoring.gain_structure import quantile, write_csv

CENSUS = "MIPLIB-NL"
BENCHMARK = "benchmark"
COLLECTIONS = (CENSUS, "templates", "rich_models", BENCHMARK)
BY_CLASS = ("rich_models", BENCHMARK)                 # collections reported per class as well
VARIANTS = {CENSUS: ("census",), "templates": ("census", "gcg"), "rich_models": ("census", "gcg"),
            BENCHMARK: ("root", "pricing", "total")}
EVAL_LIMIT = eval_time.EVAL_TIME                      # 120 s
BNP_LIMIT = bnp.BNP_TIME                              # 1800 s
REFERENCES = time_truth.REFERENCES                    # gcg_default, scip
D_COLUMNS = {"root": "root_time", "pricing": "pricing_time", "total": "time"}
RATIO_PAIRS = ((2, 1), (3, 2))

COLLECTION_LABEL = {CENSUS: "MIPLIB-NL", "templates": "Templates", "rich_models": "Rich models",
                    BENCHMARK: "Branch-and-price instances"}
CLASS_LABEL = {"lot_sizing": "lot sizing", "network_design": "network design", "facility_location": "facility location"}
VARIANT_LABEL = {"census": "census time", "gcg": "GCG time", "root": "root time", "pricing": "pricing time",
                 "total": "total time"}


def _item(collection, cls, model, size, variant, t=None, at_limit=False, not_run=False, failed=False,
          missing=False) -> dict:
    """One set and one time: t is None when the set is not run or at the limit."""
    return {"collection": collection, "class": cls, "model": model, "size": size, "variant": variant,
            "t": None if (at_limit or not_run) else t, "at_limit": bool(at_limit), "not_run": bool(not_run),
            "failed": bool(failed), "missing": bool(missing)}


# ---------------------------------------------------------------- sources

def census_items(oracle_root: Path, limit: float = EVAL_LIMIT) -> tuple[list, list]:
    """(items, problems) of the census records without linking variables (see the module docstring)."""
    items, problems = [], []
    for p in sorted(Path(oracle_root).glob("*/oracle.json")):
        name = p.parent.name
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{CENSUS} {name}: unreadable ({type(e).__name__})")
            continue
        fams = set(o.get("families") or [])
        for r in o.get("subsets") or []:
            if r.get("linking_var_families"):
                continue
            T = fams - set(r.get("master_families") or [])
            if not T:
                problems.append(f"{CENSUS} {name}: record {r.get('key')} with no convexified family")
                continue
            t, at = eval_time.eval_seconds(r, limit)
            items.append(_item(CENSUS, "", name, len(T), "census", t, at, not_run=t is None,
                               failed=r.get("status") == "failed"))
    return items, problems


def eval_items(paths: dict, runs_dir: Path, limit: float = EVAL_LIMIT) -> tuple[list, list]:
    """(items, problems) of the visited sets of paths.json, census time and GCG's time (see the module docstring)."""
    items, problems = [], []
    for n, i in sorted(paths["instances"].items()):
        coll, cls = i["collection"], i["class"]
        for key in i["visited"]:
            T = frozenset(x for x in key.split("|") if x)
            p = eval_time.record_path(runs_dir, n, i["families"], T)
            if not p.exists():
                problems.append(f"{coll} {n} {key}: no record")
                continue
            try:
                row = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                problems.append(f"{coll} {n} {key}: unreadable ({type(e).__name__})")
                continue
            failed = row.get("status") == "failed" or row.get("ok") is False
            t, at = eval_time.eval_seconds(row, limit)
            items.append(_item(coll, cls, n, len(T), "census", t, at, not_run=t is None, failed=failed))
            g = eval_time.gcg_seconds(row, limit)
            items.append(_item(coll, cls, n, len(T), "gcg", g, g is not None and g >= limit, not_run=g is None,
                               failed=failed))
    return items, problems


def _num(v):
    if v is None or str(v).strip() in ("", "None", "nan"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def _true(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def seed_median(values: list):
    """Median over seeds with the censored values (None) ordered above every other value; None when it involves a
    censored value (at least half of the values censored) or there is no value."""
    n, c = len(values), sum(v is None for v in values)
    if n == 0 or 2 * c >= n:
        return None
    return statistics.median(sorted(v for v in values if v is not None) + [math.inf] * c)


def bnp_items(rows: list, limit: float = BNP_LIMIT) -> tuple[list, list]:
    """(items, problems) of the branch-and-price runs (rows of bnp_runs.csv); see the module docstring."""
    by, cls_of, problems = defaultdict(list), {}, []
    for r in rows:
        if r.get("set") in REFERENCES:
            continue
        if not r.get("instance") or not r.get("set"):
            problems.append(f"{BENCHMARK}: row without instance or set")
            continue
        by[(r["instance"], r["set"])].append(r)
        cls_of.setdefault(r["instance"], r.get("class") or "")
    items = []
    for (n, key), runs in sorted(by.items()):
        size = len([x for x in key.split("|") if x])
        failed = any(_true(r.get("failed_twice")) for r in runs)
        for variant, col in D_COLUMNS.items():
            vals, raw_missing = [], []
            for r in runs:
                x = _num(r.get(col))
                raw_missing.append(x is None)
                if x is None or x >= limit or (variant == "total" and not _true(r.get("solved"))):
                    x = None
                vals.append(x)
            m = seed_median(vals)
            missing = m is None and 2 * sum(raw_missing) >= len(runs)
            items.append(_item(BENCHMARK, cls_of[n], n, size, variant, m, m is None, failed=failed, missing=missing))
    return items, problems


def read_d_runs(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- measures

def _groups(items: list) -> list:
    """(collection, class, variant) of the summaries, in table order: each collection over all its classes, then per
    class for the collections of BY_CLASS."""
    present = {(x["collection"], x["class"]) for x in items}
    out = []
    for coll in COLLECTIONS:
        classes = sorted({c for k, c in present if k == coll})
        if not classes:
            continue
        for v in VARIANTS[coll]:
            out.append((coll, "all", v))
        if coll in BY_CLASS:
            for c in classes:
                for v in VARIANTS[coll]:
                    out.append((coll, c, v))
    return out


def _members(items, coll, cls, variant):
    return [x for x in items if x["collection"] == coll and x["variant"] == variant and (cls == "all" or x["class"] == cls)]


def _stats(xs: list) -> dict:
    return {"median": statistics.median(xs) if xs else None, "q25": quantile(xs, 0.25), "q75": quantile(xs, 0.75)}


def by_size_rows(items: list) -> list:
    rows = []
    for coll, cls, v in _groups(items):
        mem = _members(items, coll, cls, v)
        for s in sorted({x["size"] for x in mem}):
            xs = [x for x in mem if x["size"] == s]
            run = [x for x in xs if not x["not_run"]]
            below = [x["t"] for x in run if not x["at_limit"]]
            rows.append({"collection": coll, "class": cls, "time": v, "size": s,
                         "n_models": len({x["model"] for x in xs}), "n_sets": len(run),
                         "n_not_run": len(xs) - len(run), "n_at_limit": sum(x["at_limit"] for x in run),
                         "n_at_limit_missing_value": sum(x["missing"] for x in run),
                         "n_below_limit": len(below), "n_below_limit_zero": sum(t == 0 for t in below),
                         "n_failed": sum(x["failed"] for x in run), **_stats(below)})
    return rows


def ols_slope(xs: list, ys: list) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def model_slope(item_list: list) -> dict:
    """Median time per size over the sets below the limit, sizes with median 0 dropped, and the slope of log(median)
    on log(size) with at least three sizes left (else None)."""
    per = defaultdict(list)
    for x in item_list:
        if x["t"] is not None:
            per[x["size"]].append(x["t"])
    med = {s: statistics.median(ts) for s, ts in sorted(per.items())}
    pos = {s: m for s, m in med.items() if m > 0}
    slope = (ols_slope([math.log(s) for s in pos], [math.log(m) for m in pos.values()]) if len(pos) >= 3 else None)
    return {"n_sizes": len(med), "n_sizes_zero_median": len(med) - len(pos), "sizes": ";".join(map(str, pos)),
            "medians": ";".join(f"{m:.6g}" for m in pos.values()), "slope": slope}


def slope_rows(items: list) -> list:
    model_rows, by_model = [], {}
    for coll in COLLECTIONS:
        for v in VARIANTS[coll]:
            mem = [x for x in items if x["collection"] == coll and x["variant"] == v]
            for m in sorted({x["model"] for x in mem}):
                xs = [x for x in mem if x["model"] == m]
                res = model_slope(xs)
                if res["n_sizes"] == 0:
                    continue
                row = {"level": "model", "collection": coll, "class": xs[0]["class"], "time": v, "model": m, **res}
                model_rows.append(row)
                by_model[(coll, v, m)] = row
    group_rows = []
    for coll, cls, v in _groups(items):
        rs = [r for r in model_rows if r["collection"] == coll and r["time"] == v and (cls == "all" or r["class"] == cls)]
        sl = [r["slope"] for r in rs if r["slope"] is not None]
        group_rows.append({"level": "group", "collection": coll, "class": cls, "time": v, "model": "",
                           "n_models": len(rs), "n_models_used": len(sl),
                           "n_models_fewer_than_three_sizes": len(rs) - len(sl),
                           "n_sizes_zero_median": sum(r["n_sizes_zero_median"] for r in rs), **_stats(sl)})
    return group_rows + model_rows


def ratio_rows(items: list) -> list:
    mem = [x for x in items if x["collection"] == BENCHMARK and x["variant"] in ("root", "pricing")]
    inst_rows = []
    for v in ("root", "pricing"):
        for n in sorted({x["model"] for x in mem}):
            xs = [x for x in mem if x["variant"] == v and x["model"] == n and x["t"] is not None]
            med = {s: statistics.median([x["t"] for x in xs if x["size"] == s]) for s in {x["size"] for x in xs}}
            cls = next(x["class"] for x in mem if x["model"] == n)
            for hi, lo in RATIO_PAIRS:
                if hi not in med or lo not in med:
                    continue
                ratio = med[hi] / med[lo] if med[lo] > 0 else None
                inst_rows.append({"level": "instance", "class": cls, "time": v, "sizes": f"{hi}/{lo}", "instance": n,
                                  "median_numerator": med[hi], "median_denominator": med[lo], "ratio": ratio})
    classes = ["all"] + sorted({x["class"] for x in mem})
    rows = []
    for cls in classes:
        for v in ("root", "pricing"):
            for hi, lo in RATIO_PAIRS:
                rs = [r for r in inst_rows if r["time"] == v and r["sizes"] == f"{hi}/{lo}"
                      and (cls == "all" or r["class"] == cls)]
                rat = [r["ratio"] for r in rs if r["ratio"] is not None]
                rows.append({"level": "class", "class": cls, "time": v, "sizes": f"{hi}/{lo}", "instance": "",
                             "n_instances_both_sizes": len(rs), "n_zero_denominator": len(rs) - len(rat),
                             "n_ratios": len(rat), **_stats(rat)})
    return rows + inst_rows


# ---------------------------------------------------------------- outputs

def fmt_seconds(x) -> str:
    if x is None:
        return "--"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    if x >= 0.001:
        return f"{x:#.3g}"
    return f"{x:.4f}"


def fmt2(x) -> str:
    return "--" if x is None else f"{x:.2f}"


def fmt3(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def _group_label(coll, cls, v) -> str:
    c = COLLECTION_LABEL[coll]
    if coll in BY_CLASS:
        c += ", all classes" if cls == "all" else f", {CLASS_LABEL.get(cls, cls)}"
    return f"{c}; {VARIANT_LABEL[v]}"


TEX_COLUMNS = 5


def tex_body(size_rows: list, slopes: list, ratios: list | None) -> str:
    head = ("% Time of one evaluation against the number of convexified families. Columns: families convexified; "
            "sets with a time; sets at the limit; median (s); quartiles (s), over the sets below the limit. "
            "Row slope: models with a slope; models with a set below the limit but fewer than three sizes; median; "
            "quartiles of the least-squares slopes of log(median time) on log(size). Sources: "
            "results/eval_growth/by_size.csv, "
            "slopes.csv" + (", d_ratios.csv" if ratios else ""))
    lines = [head]
    groups = [r for r in slopes if r["level"] == "group"]
    for g in groups:
        coll, cls, v = g["collection"], g["class"], g["time"]
        lines.append(f"\\multicolumn{{{TEX_COLUMNS}}}{{@{{}}l}}{{\\emph{{{_group_label(coll, cls, v)}}}}} \\\\")
        for r in size_rows:
            if (r["collection"], r["class"], r["time"]) != (coll, cls, v):
                continue
            q = "--" if r["q25"] is None else f"[{fmt_seconds(r['q25'])}, {fmt_seconds(r['q75'])}]"
            lines.append(f"${r['size']}$ & {r['n_sets']} & {r['n_at_limit']} & {fmt_seconds(r['median'])} & {q} \\\\")
        q = "--" if g["q25"] is None else f"[{fmt3(g['q25'])}, {fmt3(g['q75'])}]"
        lines.append(f"slope & {g['n_models_used']} & {g['n_models_fewer_than_three_sizes']} & {fmt3(g['median'])} & "
                     f"{q} \\\\")
    if ratios:
        lines.append("% Branch-and-price: ratio of the median seed-median time of the sets of the two sizes, per "
                     "instance. Columns: sizes; instances with both sizes; ratios with a zero denominator; median; "
                     "quartiles over instances")
        for cls in dict.fromkeys(r["class"] for r in ratios if r["level"] == "class"):
            for v in ("root", "pricing"):
                lab = (f"{COLLECTION_LABEL[BENCHMARK]}, " + ("all classes" if cls == "all" else CLASS_LABEL.get(cls, cls))
                       + f"; {VARIANT_LABEL[v]} ratio")
                lines.append(f"\\multicolumn{{{TEX_COLUMNS}}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
                for r in ratios:
                    if r["level"] == "class" and r["class"] == cls and r["time"] == v:
                        q = "--" if r["q25"] is None else f"[{fmt2(r['q25'])}, {fmt2(r['q75'])}]"
                        lines.append(f"${r['sizes']}$ & {r['n_instances_both_sizes']} & {r['n_zero_denominator']} & "
                                     f"{fmt2(r['median'])} & {q} \\\\")
    return "\n".join(lines) + "\n"


def figure_rows(size_rows: list) -> list:
    return [{"collection": r["collection"], "class": r["class"], "time": r["time"], "families convexified": r["size"],
             "sets below the limit": r["n_below_limit"], "sets at the limit": r["n_at_limit"],
             "lower quartile (s)": r["q25"], "median (s)": r["median"], "upper quartile (s)": r["q75"]}
            for r in size_rows]


def run(args) -> dict:
    items, problems = [], []
    if args.oracle_root:
        it, pr = census_items(Path(args.oracle_root))
        items += it
        problems += pr
    if args.paths:
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        runs_dir = Path(args.runs) if args.runs else Path(args.paths).parent / "runs"
        it, pr = eval_items(paths, runs_dir)
        items += it
        problems += pr
    ratios = None
    if args.d_runs:
        it, pr = bnp_items(read_d_runs(Path(args.d_runs)))
        items += it
        problems += pr
        ratios = ratio_rows(items)
    size_rows, slopes = by_size_rows(items), slope_rows(items)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "by_size.csv", size_rows)
    write_csv(out / "slopes.csv", slopes)
    if ratios is not None:
        write_csv(out / "d_ratios.csv", ratios)
    if args.tex:
        Path(args.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.tex).write_text(tex_body(size_rows, slopes, ratios), encoding="utf-8")
    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        write_csv(Path(args.figure), figure_rows(size_rows))
    return {"items": len(items), "problems": problems, "by_size": size_rows, "slopes": slopes, "ratios": ratios}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oracle-root", help="the published census oracle directory (<model>/oracle.json)")
    ap.add_argument("--paths", help="paths.json of llmdw.benchmark.eval_time")
    ap.add_argument("--runs", help="the runs/ folder of llmdw.benchmark.eval_time (default: next to --paths)")
    ap.add_argument("--d-runs", help="bnp_runs.csv of llmdw.benchmark.time_truth (read only)")
    ap.add_argument("--out", required=True, help="folder for by_size.csv, slopes.csv and d_ratios.csv")
    ap.add_argument("--tex", help="the table body (.tex)")
    ap.add_argument("--figure", help="the figure data (.csv)")
    args = ap.parse_args(argv)
    if not (args.oracle_root or args.paths or args.d_runs):
        ap.error("give at least one of --oracle-root, --paths, --d-runs")
    res = run(args)
    for p in res["problems"]:
        print(f"left out: {p}")
    print(f"{res['items']} (set, time) items; {len(res['problems'])} left out -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def tex_from_csv(out: Path, tex: Path) -> list:
    """eval_growth.tex rebuilt from by_size.csv, slopes.csv and d_ratios.csv."""
    from llmdw.tables.experiments import typed_rows

    out = Path(out)
    ratios = typed_rows(out / "d_ratios.csv") if (out / "d_ratios.csv").exists() else None
    (Path(tex) / "eval_growth.tex").write_text(tex_body(typed_rows(out / "by_size.csv"), typed_rows(out / "slopes.csv"),
                                                        ratios), encoding="utf-8")
    return ["eval_growth.tex"]
