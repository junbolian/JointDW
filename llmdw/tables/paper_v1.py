"""Table bodies of the benchmark tables of the paper in its layout and rounding, rebuilt from the result CSVs (no
solver run): tab_bench.tex (left: the first set is the best pair, from the standard formulations; middle: the time-
optimal set within the budget; right: the classes without a joint group), tab_app_bench.tex, tab_app_bench_time.tex,
tab_app_tradeoff.tex and tab_app_bench_data.tex; the rich-model tables tab_app_misses.tex ($s=2$: the predictions
whose best pair has no more families than the anchor, by the misses e, and the anchors cut to two families) and
tab_app_larger.tex ($s=3$: the anchors with fewer families than the best set), from
figure_data/search_cost_vs_misses_after_cut.csv; and tab_instances.tex, the number of instances and the range of k of
each instance collection, counted from the instance lists of the results. The log time ratio of a set against another does not depend on the
reference set, so the differences of the coefficient presentation against the baselines are read from
paired_time_truth.csv and those against the original presentation from paired_time_truth_budget.csv.

Inputs under RESULTS: experiment_C/benchmark_methods.csv, benchmark_data_methods.csv, benchmark_data_paired.csv;
experiment_D/methods_time_budget.csv, paired_time_truth_budget.csv, tradeoff_budget.csv, paired_time_truth.csv;
experiment_F/methods_time.csv; paper/figure_data/search_cost_vs_misses_after_cut.csv; templates_table.csv,
census_instances.csv, rich_models_instances.csv, experiment_E/instances.csv, experiment_C/benchmark_instances.csv,
experiment_D/time_truth.csv, experiment_F/instances.csv.

Usage
-----
python -m llmdw.tables.paper_v1 --results results --out results/paper/experiments
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

CLASSES = ("lot_sizing", "network_design", "facility_location")
CLASS_KEY = {"Lot sizing": "lot_sizing", "Network design": "network_design", "Facility location": "facility_location",
             "Lot Sizing": "lot_sizing", "all": "all", "pooled": "all"}
CONFIGS = ["GPT-6.1 Sol", "Claude Sonnet 5.5"]
BASELINES = ("lookahead (single dual)", "lookahead (five duals)", "GCG default")
LABEL = {"best pair": "Best pair", "true best set": "Best pair", "lookahead (single dual)": "Lookahead (HiGHS dual)",
         "lookahead (five duals)": "Lookahead (five duals)", "GCG default": "GCG's detector",
         "GPT-6.1 Sol": "GPT-6.1 Sol", "Claude Sonnet 5.5": "Claude Sonnet 5.5"}
CLASS_HEAD = {"lot_sizing": "Lot sizing, 20 instances", "network_design": "Network design, 20 instances",
              "facility_location": "Facility location, 19 instances", "all": "All instances, 59 instances"}


def _rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if "class" in r:
            r["class"] = CLASS_KEY.get(r["class"], r["class"])
        for k in ("method", "against"):
            if k in r:
                r[k] = r[k].replace(", coefficients", ", data")
    return rows


def _first(*paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(paths[0])


def num(x, d):
    if x in (None, ""):
        return "--"
    v = float(x)
    s = f"{abs(v):.{d}f}"
    return ("$-$" if v < 0 and float(s) != 0 else "") + s


def pm(m, se, d, star=""):
    if m in (None, ""):
        return "--"
    return f"{num(m, d)} $\\pm$ {num(se, d)}" + ("$^{*}$" if star == "*" else "")


def tex_bench(mc: list, cm: list, fm: list) -> str:
    C = {(r["class"], r["method"]): r for r in cm}
    M = {(r["class"], r["method"]): r for r in mc}
    F = {(r["class"], r["method"]): r for r in fm}
    L = []
    for m in ("lookahead (single dual)", "lookahead (five duals)", "GCG default", None, "GPT-6.1 Sol", "Claude Sonnet 5.5"):
        if m is None:
            L.append("\\multicolumn{9}{@{}l}{\\emph{JointDW with}} \\\\")
            continue
        left = [num(C[(c, m)]["hit_mean"], 2) for c in CLASSES + ("all",)]
        mid, f = M[("all", m)], F[("all", m)]
        L.append(f"{LABEL[m]} & " + " & ".join(left) + f" & {num(mid['hit_strict_mean'], 2)} & "
                 f"{pm(mid['time_ratio_geo'], mid['time_ratio_se'], 2, mid['time_ratio_asterisk'])} & "
                 f"{num(f['hit_time_strict_mean'], 2)} & "
                 f"{pm(f['ratio_time_optimal_geo'], f['ratio_time_optimal_se'], 2, f['ratio_time_optimal_asterisk'])} \\\\")
    return "\n".join(L) + "\n"


def tex_app_bench(mc: list, cm: list) -> str:
    C = {(r["class"], r["method"]): r for r in cm}
    M = {(r["class"], r["method"]): r for r in mc}
    L = []
    for c in CLASSES + ("all",):
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{CLASS_HEAD[c]}}}}} \\\\")
        for m, ck in (("best pair", "true best set"), ("lookahead (single dual)",) * 2, ("lookahead (five duals)",) * 2,
                      ("GCG default",) * 2, ("GPT-6.1 Sol",) * 2, ("Claude Sonnet 5.5",) * 2):
            a, b = C[(c, ck)], M[(c, m)]
            lab = LABEL[m] if m not in CONFIGS else f"JointDW with {m}"
            zero = int(round(float(a["gain_zero_instances"]))) if a["gain_zero_instances"] else 0
            L.append(f"{lab} & {pm(a['hit_mean'], a['hit_se'], 2)} & {pm(a['gain_mean'], a['gain_se'], 3)} & {zero} & "
                     f"{pm(b['hit_strict_mean'], b['hit_strict_se'], 2)} & "
                     f"{pm(b['hit_tolerance_mean'], b['hit_tolerance_se'], 2)} & "
                     f"{pm(b['time_ratio_geo'], b['time_ratio_se'], 3, b['time_ratio_asterisk'])} \\\\")
    return "\n".join(L) + "\n"


def tex_app_bench_time(pc: list) -> str:
    P = {(r["class"], r["method"], r["against"], r["quantity"]): r for r in pc}
    L, prev = [], None
    for a, b in [(x, y) for x in CONFIGS for y in BASELINES] + [("lookahead (five duals)", "GCG default")]:
        lab = (f"JointDW with {a}" if a in CONFIGS else LABEL[a]) if a != prev else ""
        prev = a
        cells = [pm(P[(c, a, b, "log_ratio")]["mean_difference"], P[(c, a, b, "log_ratio")]["se_difference"], 2,
                    P[(c, a, b, "log_ratio")]["asterisk"]) for c in CLASSES + ("all",)]
        h = P[("all", a, b, "hit_strict")]
        L.append(f"{lab} & {LABEL[b]} & " + " & ".join(cells) + f" & {pm(h['mean_difference'], h['se_difference'], 2, h['asterisk'])} \\\\")
    return "\n".join(L) + "\n"


def tex_app_tradeoff(tr: list) -> str:
    L = []
    for c in CLASSES:
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{CLASS_HEAD[c]}}}}} \\\\")
        for r in (x for x in tr if x["class"] == c):
            pr = "--" if r["pricing_share_mean"] in (None, "") else num(r["pricing_share_mean"], 2)
            nd = "--" if r["nodes_mean"] in (None, "") else str(int(round(float(r["nodes_mean"]))))
            L.append(f"{r['sets'][0].upper() + r['sets'][1:]} & {r['n_sets']} & {num(r['share_solved_runs'], 2)} & "
                     f"{num(r['time_stat_mean'], 1)} & {pr} & {nd} \\\\")
    return "\n".join(L) + "\n"


def tex_app_bench_data(mc: list, pc: list, d_paired: list, dm: list, dp: list) -> str:
    M = {(r["class"], r["method"]): r for r in mc}
    DM = {(r["class"], r["method"]): r for r in dm}
    DP = {(r["class"], r["configuration"], r["against"].replace(", original", ""), r["metric"]): r for r in dp}
    PC = {(r["class"], r["method"], r["against"], r["quantity"]): r for r in pc}
    PD = {(r["class"], r["method"], r["against"], r["quantity"]): r for r in d_paired}
    L = []
    for c in CLASSES + ("all",):
        L.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{CLASS_HEAD[c]}}}}} \\\\")
        for cfg in CONFIGS:
            a, b = DM[(c, f"{cfg}, data")], M[(c, f"{cfg}, data")]
            L.append(f"{cfg} & {pm(a['hit_mean'], a['hit_se'], 2)} & {pm(a['gain_mean'], a['gain_se'], 3)} & "
                     f"{pm(b['hit_strict_mean'], b['hit_strict_se'], 2)} & "
                     f"{pm(b['time_ratio_geo'], b['time_ratio_se'], 3, b['time_ratio_asterisk'])} & "
                     f"{int(float(a['n_samples_without_joint_support'] or 0))} \\\\")
    L.append("% bottom")
    for c in ("network_design", "facility_location", "all"):
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{CLASS_HEAD[c]}}}}} \\\\")
        for cfg in CONFIGS:
            for i, b in enumerate(list(BASELINES) + ["original"]):
                key = cfg if b == "original" else b
                h, g = DP[(c, cfg, key, "hit@1")], DP[(c, cfg, key, "gain")]
                lr = PC[(c, f"{cfg}, data", cfg, "log_ratio")] if b == "original" else PD[(c, f"{cfg}, data", b, "log_ratio")]
                L.append(f"{cfg if i == 0 else ''} & {'Original presentation' if b == 'original' else LABEL[b]} & "
                         f"{pm(h['mean_difference'], h['se_difference'], 2, h['asterisk'])} & "
                         f"{pm(g['mean_difference'], g['se_difference'], 3, g['asterisk'])} & "
                         f"{pm(lr['mean_difference'], lr['se_difference'], 2, lr['asterisk'])} \\\\")
    return "\n".join(L) + "\n"


CLASS_NAME = {"lot_sizing": "Lot sizing", "network_design": "Network design", "facility_location": "Facility location"}
MISSES = "paper/figure_data/search_cost_vs_misses_after_cut.csv"


def _by_class_k(rows: list) -> list:
    return sorted(rows, key=lambda r: (CLASSES.index(r["class"]), int(r["k"])))


def _class_k_heads(rows: list) -> list:
    """The class and k cells of each row: printed at the first row of a class and of a k only."""
    out, prev = [], (None, None)
    for r in rows:
        c, k = r["class"], r["k"]
        out.append((CLASS_NAME[c] if c != prev[0] else "", k if (c, k) != prev else ""))
        prev = (c, k)
    return out


def tex_app_misses(mis: list) -> str:
    """$s=2$: per class and k, the rows e = 0, 1, 2 of the anchors within s whose best pair has no more families than
    the anchor, then the row of the anchors cut to two families (mean misses after the cut in parentheses)."""
    rows = [r for r in mis if r["s"] == "2" and r["kind"] in ("anchor within s, |W| <= |P|", "anchor cut to s")]
    rows = sorted(_by_class_k(rows), key=lambda r: (CLASSES.index(r["class"]), int(r["k"]),
                                                     r["kind"] == "anchor cut to s", int(r["e"] or 0)))
    L = []
    for r, (c, k) in zip(rows, _class_k_heads(rows)):
        o1, jd = pm(r["cost_o1_mean"], r["cost_o1_se"], 2), pm(r["cost_jointdw_mean"], r["cost_jointdw_se"], 2)
        if r["kind"] == "anchor cut to s":
            L.append(f"{c} & {k} & cut ({num(r['e_after_cut_mean'], 2)}) & {r['n_predictions']} & {o1} & {jd} & -- & -- \\\\")
        else:
            a, b = r["stage_queries_0_to_e_min"], r["stage_queries_0_to_e_max"]
            q = a if a == b else f"{a}--{b}"
            L.append(f"{c} & {k} & {r['e']} & {r['n_predictions']} & {o1} & {jd} & {q} & "
                     f"{num(r['mean_evaluations_covering'], 1)} \\\\")
    return "\n".join(L) + "\n"


def tex_app_larger(mis: list) -> str:
    """$s=3$: per class and k, the anchors within s with fewer families than the best set: share of the orders in which
    O1 alone reaches a best set, JointDW, covering order."""
    rows = _by_class_k([r for r in mis if r["s"] == "3" and r["kind"] == "anchor within s, |W| > |P|"])
    L = []
    for r, (c, _) in zip(rows, _class_k_heads(rows)):
        L.append(f"{c} & {r['k']} & {r['n_predictions']} & {num(r['o1_share_orders_reaching_best_mean'], 2)} & "
                 f"{pm(r['cost_jointdw_mean'], r['cost_jointdw_se'], 2)} & {num(r['mean_evaluations_covering'], 1)} \\\\")
    return "\n".join(L) + "\n"


def _count_and_k(rows: list, name: str, ks) -> str:
    ks = sorted({int(k) for k in ks})
    return f"{len({r[name] for r in rows})} & " + (str(ks[0]) if len(ks) == 1 else f"{ks[0]}--{ks[-1]}")


def tex_instances(R: Path) -> str:
    """Number of instances and range of k of each instance collection, counted from the instance lists."""
    tm, ce = _rows(R / "templates_table.csv"), _rows(R / "census_instances.csv")
    ri, rl = _rows(R / "rich_models_instances.csv"), _rows(R / "experiment_E/instances.csv")
    bi, tt = _rows(R / "experiment_C/benchmark_instances.csv"), _rows(R / "experiment_D/time_truth.csv")
    nj = _rows(R / "experiment_F/instances.csv")
    bk = {max(len(x.split("|")) for x in r["order"].split()) for r in tt}
    L = [f"templates & {_count_and_k(tm, 'instance', (r['k'] for r in tm))}",
         f"census & {_count_and_k(ce, 'instance', (r['k'] for r in ce))}",
         f"rich models & {_count_and_k(ri, 'instance', (r['k'] for r in ri))}",
         f" & {_count_and_k(rl, 'instance', (r['k'] for r in rl))}",
         f"benchmark & {_count_and_k(bi, 'name', bk)}",
         f"without a joint support & {_count_and_k(nj, 'name', (len(r['families'].split()) for r in nj))}"]
    return "\n".join(x + " \\\\" for x in L) + "\n"


def write_tables(results: Path, out: Path) -> list:
    R, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C, D, F = R / "experiment_C", R / "experiment_D", R / "experiment_F"
    cm = _rows(_first(C / "benchmark_methods.csv", C / "methods.csv"))
    dm = _rows(_first(C / "benchmark_data_methods.csv", C / "data_methods.csv"))
    dp = _rows(_first(C / "benchmark_data_paired.csv", C / "data_paired.csv"))
    mc, pc = _rows(D / "methods_time_budget.csv"), _rows(D / "paired_time_truth_budget.csv")
    bodies = {"tab_bench.tex": tex_bench(mc, cm, _rows(F / "methods_time.csv")),
              "tab_app_bench.tex": tex_app_bench(mc, cm),
              "tab_app_bench_time.tex": tex_app_bench_time(pc),
              "tab_app_tradeoff.tex": tex_app_tradeoff(_rows(D / "tradeoff_budget.csv")),
              "tab_app_bench_data.tex": tex_app_bench_data(mc, pc, _rows(D / "paired_time_truth.csv"), dm, dp),
              "tab_app_misses.tex": tex_app_misses(_rows(R / MISSES)),
              "tab_app_larger.tex": tex_app_larger(_rows(R / MISSES)),
              "tab_instances.tex": tex_instances(R)}
    for name, body in bodies.items():
        (out / name).write_text(body, encoding="utf-8")
    return list(bodies)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(" ".join(write_tables(Path(a.results), Path(a.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- comparison with the table bodies of the paper source

import json as _json  # noqa: E402
import math as _math  # noqa: E402
import re as _re  # noqa: E402

_NUMBER = _re.compile(r"(\$-\$|-)?(\d+\.\d+|\d+)")
_NAMES = ("GPT-6.1 Sol", "Claude Sonnet 5.5", "Grok 4.20", "DeepSeek-V4-Pro", "MIPLIB-NL", "OR-Library")
# tables whose layout the code writes, and the reference body each is compared with, number by number
POSITIONAL = {"tab:llm": "paper/table_01.tex", "tab:search": "paper/table_09.tex", "tab:app-recall": "paper/table_06.tex",
              "tab:app-paired": "paper/table_07.tex", "tab:app-reason": "paper/table_08.tex",
              "tab:rich-pred": "paper/table_11.tex", "tab:rich-search": "paper/table_12.tex",
              "tab:app-paired-search": "paper/table_15.tex", "tab:app-learned": "experiments/learned_predictor.tex",
              "tab:bench": "experiments/tab_bench.tex", "tab:app-bench": "experiments/tab_app_bench.tex",
              "tab:app-bench-time": "experiments/tab_app_bench_time.tex",
              "tab:app-tradeoff": "experiments/tab_app_tradeoff.tex",
              "tab:app-bench-data": "experiments/tab_app_bench_data.tex",
              "tab:app-misses": "experiments/tab_app_misses.tex", "tab:app-larger": "experiments/tab_app_larger.tex",
              "tab:instances": "experiments/tab_instances.tex",
              "tab:app-first": "experiments/tab_app_first.tex"}       # written by llmdw.scoring.first_sets
# paper v1.0 printed earlier standard errors in these tables; from paper v1.1 on they carry the values of this release
STANDARD_ERRORS_BEFORE_V1_1 = ("tab:bench", "tab:app-bench", "tab:app-bench-time", "tab:app-bench-data")
# the other tables: every printed number is looked for among the values of these result CSVs (relative to RESULTS)
SOURCES = {"tab:exact": ["templates_table.csv", "covering_numbers.csv", "figure2.csv"],
           "tab:rich-inst": ["rich_models_instances.csv"], "tab:rich-large-inst": ["experiment_E/instances.csv"],
           "tab:bench-inst": ["experiment_C/benchmark_instances.csv"], "tab:app-five": ["experiment_B/*.csv"],
           "tab:app-control": ["experiment_A/*.csv"],
           "tab:app-rich-five": ["rich_models_data_paired_five_dual.csv", "experiment_B/*.csv"],
           "tab:app-learned-size": ["learned/*.csv"],
           "tab:app-bench-paired": ["experiment_C/benchmark_paired_hit1.csv", "experiment_C/benchmark_paired_gain.csv"],
           "tab:app-nojoint": ["experiment_F/methods.csv", "experiment_F/methods_time.csv"],
           "tab:app-rich-large": ["experiment_E/search_costs.csv", "experiment_E/paired_five_dual.csv",
                                  "experiment_E/paired_covering.csv"],
           "tab:app-cost": ["experiment_G/*.csv"], "tab:app-growth": ["eval_growth/*.csv"]}
# cells derived from the instance lists, not stored as a value in a result CSV: (row, number)
DERIVED = {"tab:rich-inst": [(0, "35"), (0, "2"), (0, "2"), (1, "8"), (2, "8"), (2, "2")],
           "tab:rich-large-inst": [(0, "3.4"), (1, "5.6"), (2, "3.5"), (4, "2.9"), (5, "9.7")],
           "tab:bench-inst": [(0, "20"), (1, "20"), (2, "19")]}


def table_block(tex: str, label: str) -> str:
    i = tex.index(f"\\label{{{label}}}")
    return tex[tex.rfind("\\begin{table", 0, i):tex.index("\\end{table", i)]


def body_rows(block: str) -> list:
    out = []
    for m in _re.finditer(r"\\midrule(.*?)\\bottomrule", block, _re.S):
        for line in m.group(1).split("\\\\"):
            line = " ".join(line.replace("\\midrule", " ").split())
            if line and not line.startswith("\\cmidrule"):
                out.append(line)
    return out


def built_rows(text: str) -> list:
    out = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line or line.startswith("%") or line.strip("\\ ") == "midrule":
            continue
        out.append(line.rstrip("\\").strip())
    return [r for r in out if r]


def numbers(line: str) -> list:
    """(number, is a standard error) of a row: the numbers after model names, labels and citations are taken out; a
    range a--b gives two numbers; a number right after $\\pm$ is a standard error."""
    for m in _NAMES:
        line = line.replace(m, "")
    line = line.replace("\\midrule", "")
    line = _re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", line)
    line = _re.sub(r"\\cite[pt]?\{[^}]*\}", "", line)
    line = _re.sub(r"[A-Za-z]+-?\d+(\.\d+)?", "", line)
    line = _re.sub(r"\$k=\d+\$|\d+ instances|\d+\\%", "", line)
    line = _re.sub(r"(\d)--(\d)", r"\1 \2", line)
    out = []
    for m in _NUMBER.finditer(line):
        se = line[max(0, m.start() - 8):m.start()].rstrip().endswith("$\\pm$")
        out.append((("-" if m.group(1) else "") + m.group(2), se))
    return out


def _values(results: Path, globs: list) -> set:
    vals = set()
    for g in globs:
        for f in sorted(Path(results).glob(g)):
            with open(f, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    for c in row:
                        try:
                            v = float(c)
                        except ValueError:
                            continue
                        if _math.isfinite(v):
                            vals.add(v)
    return vals


def _found(x: str, vals: set) -> bool:
    d = len(x.split(".")[1]) if "." in x else 0
    t = float(x)
    for v in vals:
        for w in (v, 100 * v, _math.exp(v) if abs(v) < 50 else None):
            if w is not None and abs(round(w, d) - t) < 0.5 * 10 ** -d * 1.0001:
                return True
    return False


def compare_with_paper(tex: str, results: Path, reference: Path) -> dict:
    """Per table label of the paper source: the positional differences (row, label, column, paper, rebuilt, is a
    standard error) against the reference body, or the printed numbers not found among the values of its result CSVs
    (row, label, number, declared as derived)."""
    out = {}
    for lab in _re.findall(r"\\label\{(tab:[^}]+)\}", tex):
        P = body_rows(table_block(tex, lab))
        if lab in POSITIONAL:
            B = built_rows((Path(reference) / POSITIONAL[lab]).read_text(encoding="utf-8"))
            diffs = [("rows", len(P), len(B))] if len(P) != len(B) else []
            for i, (p, b) in enumerate(zip(P, B)):
                a, c = numbers(p), numbers(b)
                if len(a) != len(c):
                    diffs.append((i, p.split("&")[0].strip()[:40], "count", len(a), len(c)))
                    continue
                diffs += [(i, p.split("&")[0].strip()[:40], j, x[0], y[0], x[1])
                          for j, (x, y) in enumerate(zip(a, c)) if x[0] != y[0]]
            out[lab] = {"kind": "position", "reference": POSITIONAL[lab], "differences": diffs}
        else:
            vals = _values(results, SOURCES[lab])
            derived = list(DERIVED.get(lab, []))
            miss = []
            for i, r in enumerate(P):
                for x, _ in numbers(r):
                    if not _found(x, vals):
                        miss.append((i, r.split("&")[0].strip()[:40], x, (i, x) in derived))
            out[lab] = {"kind": "presence", "sources": SOURCES[lab], "not_found": miss}
    return out


def unexpected(report: dict, standard_errors_allowed=()) -> list:
    """The differences that are neither a declared derived cell nor a standard error of a table named in
    standard_errors_allowed (STANDARD_ERRORS_BEFORE_V1_1 for paper v1.0; nothing for later versions)."""
    bad = []
    for lab, r in report.items():
        if r["kind"] == "position":
            for d in r["differences"]:
                if not (lab in standard_errors_allowed and len(d) == 6 and d[5]):
                    bad.append((lab, d))
        else:
            bad += [(lab, d) for d in r["not_found"] if not d[3]]
    return bad


def write_report(report: dict, path: Path) -> None:
    Path(path).write_text(_json.dumps(report, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------- derived cells of the instance tables
# The columns of the instance tables whose cells no result CSV holds (column index in the body of the paper, name, and
# whether the order of the numbers in a cell is fixed). instance_cells recomputes them from the instance files and the
# gain tables of the released data; compare_instance_cells compares them with the paper, cell by cell, and confirms
# that every cell declared in DERIVED is among the cells it checks.
INSTANCE_COLUMNS = {"tab:rich-inst": [(2, "rows x columns", True), (4, "minimal best sets, instances each", False),
                                      (5, "zero-marginal groups per instance, range", True)],
                    "tab:rich-large-inst": [(5, "zero-marginal groups per instance, mean", True),
                                            (6, "instances with a positive gain at s = 2", True)],
                    "tab:bench-inst": [(2, "instances", True)]}


def _mps_size(path: Path) -> tuple:
    """(rows without the objective, columns) of a model file in MPS format."""
    rows, cols, sec = 0, set(), None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                sec = line.split()[0]
                continue
            f = line.split()
            if sec == "ROWS" and f[0] != "N":
                rows += 1
            elif sec == "COLUMNS" and "'MARKER'" not in line:
                cols.add(f[0])
    return rows, len(cols)


def _class_of(name: str) -> str:
    return next(c for c in CLASSES if f"_{c}_" in name)


def _range(xs: list) -> list:
    return [str(min(xs))] if min(xs) == max(xs) else [str(min(xs)), str(max(xs))]


def _gain_tables(root: Path) -> dict:
    """{class: [(instance folder, ground_truth.json)]} of the instances listed in root/instances.txt."""
    out = {}
    for n in (root / "instances.txt").read_text(encoding="utf-8").split():
        d = root / "instances" / n
        out.setdefault(_class_of(n), []).append((d, _json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))))
    return out


def instance_cells(work: Path) -> dict:
    """{label: [row, ...]}, each row {column name: [numbers as printed]}, rows in the order of the paper, recomputed from
    the instance files and the gain tables under WORK (data/work):
    tab:rich-inst, one row per class (rich_models/): the distinct rows and columns of the model files (model.mps, rows
    without the objective); the number of instances of each minimal best set, a set of ground_truth.json best_sets with
    no best proper subset (in increasing order, since the cell orders them by set); the range of the number of
    zero-marginal groups (joint_supports) per instance.
    tab:rich-large-inst, one row per class and k (rich_models_large/): the mean number of zero-marginal groups per
    instance, one decimal; the number of instances on which a set of one or two families has a positive exact gain.
    tab:bench-inst, one row per class: the number of instances of benchmark_std/full_set/all_instances.txt, with the
    class of each instance from its meta.json."""
    from collections import Counter
    from fractions import Fraction

    work = Path(work)
    rich = []
    for c in CLASSES:
        L = _gain_tables(work / "rich_models")[c]
        best = Counter()
        for _, g in L:
            B = [frozenset(s) for s in g["best_sets"]]
            best.update(s for s in B if not any(t < s for t in B))
        rich.append({"rows x columns": [str(x) for s in sorted({_mps_size(d / "model.mps") for d, _ in L}) for x in s],
                     "minimal best sets, instances each": [str(v) for v in sorted(best.values())],
                     "zero-marginal groups per instance, range": _range([len(g["joint_supports"]) for _, g in L])})
    large, tables = [], _gain_tables(work / "rich_models_large")
    for c in CLASSES:
        for k in sorted({len(g["families"]) for _, g in tables[c]}):
            L = [g for _, g in tables[c] if len(g["families"]) == k]
            pos = sum(any(0 < len(s.split("|")) <= 2 and s and Fraction(v) > 0 for s, v in g["gain"].items()) for g in L)
            large.append({"zero-marginal groups per instance, mean": [f"{sum(len(g['joint_supports']) for g in L) / len(L):.1f}"],
                          "instances with a positive gain at s = 2": [str(pos)]})
    B = work / "benchmark_std"
    n = Counter(_json.loads((B / "instances" / x / "meta.json").read_text(encoding="utf-8"))["class"]
                for x in (B / "full_set" / "all_instances.txt").read_text(encoding="utf-8").split())
    return {"tab:rich-inst": rich, "tab:rich-large-inst": large,
            "tab:bench-inst": [{"instances": [str(n[c])]} for c in CLASSES]}


def _cell_numbers(cell: str) -> list:
    cell = _re.sub(r"\\cite[pt]?\{[^}]*\}", "", cell).replace("\\times", " ")
    return _re.findall(r"\d+\.\d+|\d+", cell)


def compare_instance_cells(tex: str, work: Path) -> dict:
    """Per instance table: the cells of INSTANCE_COLUMNS that differ from instance_cells (row, label, column, paper,
    recomputed), and the cells of DERIVED that are not among the checked cells of their row."""
    cells, out = instance_cells(work), {}
    for lab, cols in INSTANCE_COLUMNS.items():
        P = body_rows(table_block(tex, lab))
        diffs = [("rows", len(P), len(cells[lab]))] if len(P) != len(cells[lab]) else []
        unchecked = []
        for i, (p, row) in enumerate(zip(P, cells[lab])):
            parts = [x.strip() for x in p.split("&")]
            checked = []
            for j, name, ordered in cols:
                a = _cell_numbers(parts[j]) if j < len(parts) else []
                a = a if ordered else sorted(a, key=float)
                if a != row[name]:
                    diffs.append((i, parts[0][:40], name, a, row[name]))
                checked += a
            for r, x in DERIVED.get(lab, []):
                if r == i:
                    if x in checked:
                        checked.remove(x)
                    else:
                        unchecked.append((r, x))
        out[lab] = {"columns": [name for _, name, _ in cols], "differences": diffs, "derived_not_checked": unchecked}
    return out
