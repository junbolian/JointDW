"""LaTeX table bodies of the experiments added in the revision, from the CSV files of scripts/reproduce.sh: the control
presentations with every coefficient (Experiment A), the five-dual lookahead (Experiment B), the benchmark instances
(Experiment C), the learned predictor and the combined orders. Rows only; the column heads are in the paper.

Mean +- standard error as in llmdw.tables.paper; an asterisk marks a paired difference whose 95 percent interval excludes
zero.

Usage
-----
python -m llmdw.tables.experiments --results results --out results/paper/experiments
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

CLASSES = {"clsp": "Lot sizing", "ndp": "Network design", "cfl": "Facility location", "lot_sizing": "Lot sizing",
           "network_design": "Network design", "facility_location": "Facility location"}
EXACT = {**CLASSES, "templates": "Templates", "miplib": "MIPLIB-NL", "rich_models": "Rich models",
         "within_class": "within class", "leave_one_class_out": "leave one class out",
         "miplibnl_leave_one_out": "MIPLIB-NL leave one out", "training_size": "training size",
         "structural+lookahead": "structural + lookahead"}
METHOD_NAMES = {"lookahead (single dual)": "Lookahead, HiGHS dual", "lookahead (five duals)": "Lookahead, five duals",
                "GCG default": "GCG's detector"}
CFG_ORDER = ["Claude Sonnet 5.5", "GPT-6.1 Sol", "Grok 4.20 (Reasoning)", "Grok 4.20 (Non-Reasoning)",
             "DeepSeek-V4-Pro (Non-Thinking)"]
# paired time differences (s) below this in absolute value are floating-point residue of identical times: printed as
# 0.0 without an asterisk in the paired-time table
TIME_RESIDUE = 1e-9


def clean_text(s: str) -> str:
    """Labels of the rows: official set, class, split and variant names; JointDW."""
    if s in EXACT:
        return EXACT[s]
    return re.sub(r"(?i)(?<![A-Za-z])jointdw(?![a-z])", "JointDW", s)


def f2(x, d=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    s = f"{v:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") and float(s) != 0 else s.lstrip("-") if float(s) == 0 else s


def pm(m, se, d=2, star=""):
    if m in ("", None):
        return "--"
    return f"{f2(m, d)} $\\pm$ {f2(se, d)}" + ("$^{*}$" if star == "*" else "")


def dict_rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tables(results: Path, out: Path) -> list:
    """Write the table bodies of the experiments to out; returns their names."""
    R = Path(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def tex_table(out_dir, name, src, comment, lines):
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(name)

    CFG_ORDER = ["Claude Sonnet 5.5", "GPT-6.1 Sol", "Grok 4.20 (Reasoning)", "Grok 4.20 (Non-Reasoning)",
                 "DeepSeek-V4-Pro (Non-Thinking)"]

    # Experiment A
    A = dict_rows(R / "experiment_A/control_prediction.csv")
    lines = []
    for cls in ["pooled", "lot_sizing", "network_design", "facility_location"]:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{'Pooled' if cls == 'pooled' else CLASSES[cls]}}}}} \\\\")
        for cfg in ["GPT-6.1 Sol", "Claude Sonnet 5.5"]:
            cells = [cfg]
            for pr in ["data", "data, permuted", "data, anonymized"]:
                r = next(r for r in A if r["configuration"] == cfg and r["presentation"] == pr and r["class"] == cls)
                cells += [pm(r["hit1_mean"], r["hit1_se"]), pm(r["recall_mean"], r["recall_se"])]
            lines.append(" & ".join(cells) + " \\\\")
    tex_table(out, "experiment_A_prediction.tex", "experiment_A_label_controls/control_prediction.csv",
              "% Experiment A (label controls, rich models with every coefficient). Columns: configuration; hit@1 and recall "
              "for data, data permuted, data anonymized (mean $\\pm$ standard error over instances). Source: "
              "data/experiment_A_label_controls/control_prediction.csv", lines)
    AS = dict_rows(R / "experiment_A/control_search.csv")
    lines = []
    for s in ["2", "3"]:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{$s={s}$}}}} \\\\")
        for cfg in ["GPT-6.1 Sol", "Claude Sonnet 5.5"]:
            for meth in ["jointdw", "interleaved"]:
                cells = [f"{cfg}, {'JointDW' if meth == 'jointdw' else 'interleaved'}"]
                for pr in ["data", "data, permuted", "data, anonymized"]:
                    r = next((r for r in AS if r["configuration"] == cfg and r["presentation"] == pr and r["class"] == "pooled"
                              and r["s"] == s and r["method"] == meth), None)
                    cells.append(pm(r["evaluations_mean"], r["evaluations_se"]) if r else "--")
                lines.append(" & ".join(cells) + " \\\\")
    tex_table(out, "experiment_A_search.tex", "experiment_A_label_controls/control_search.csv",
              "% Experiment A: evaluations until a best set, pooled over the 24 rich models. Columns: configuration and "
              "order; data, data permuted, data anonymized (mean $\\pm$ standard error). Source: "
              "data/experiment_A_label_controls/control_search.csv", lines)

    # Experiment B
    B1 = dict_rows(R / "experiment_B/five_duals_first_set.csv")
    lines = []
    for r in B1:
        lines.append(f"{EXACT.get(r['set'], r['set'])} & {r['order']} & {pm(r['hit1_mean'], r['hit1_se'])} & "
                     f"{pm(r['recall_mean'], r['recall_se'])} & {r['n_instances']} \\\\")
    tex_table(out, "experiment_B_first_set.tex", "experiment_B_five_dual_lookahead/five_dual_lookahead_first_set.csv",
              "% Experiment B (five-dual lookahead): first set of the order. Columns: set; order; hit@1; recall (mean $\\pm$ "
              "standard error); instances. Source: data/experiment_B_five_dual_lookahead/five_dual_lookahead_first_set.csv", lines)
    B2 = dict_rows(R / "experiment_B/five_duals_search.csv")
    lines = []
    for st in ["templates", "miplib", "rich_models"]:
        rs = [r for r in B2 if r["set"] == st]
        if not rs:
            continue
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{EXACT[st]}}}}} \\\\")
        keys = []
        for r in rs:
            k = (r["order"], r["configuration"])
            if k not in keys:
                keys.append(k)
        for k in keys:
            cells = [clean_text(k[0] + (f", {k[1]}" if k[1] else ""))]
            for s in ["2", "3", "k-1"]:
                r = next((r for r in rs if (r["order"], r["configuration"]) == k and r["size_limit"] == s), None)
                cells.append(pm(r["evaluations_mean"], r["evaluations_se"]) if r else "--")
            lines.append(" & ".join(cells) + " \\\\")
    tex_table(out, "experiment_B_search.tex", "experiment_B_five_dual_lookahead/five_dual_lookahead_search.csv",
              "% Experiment B: evaluations until a best set by order. Columns: order (and configuration); $s=2$; $s=3$; "
              "$s=k-1$ (mean $\\pm$ standard error). Source: data/experiment_B_five_dual_lookahead/five_dual_lookahead_search.csv",
              lines)

    # Experiment C
    CM = dict_rows(R / "experiment_C/benchmark_methods.csv")
    lines = []
    for cls in ["lot_sizing", "network_design", "facility_location", "all"]:
        lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{CLASSES.get(cls, 'All classes')}}}}} \\\\")
        for r in [r for r in CM if r["class"] == cls]:
            lines.append(f"{r['method']} & {pm(r['hit_mean'], r['hit_se'])} & {pm(r['gain_mean'], r['gain_se'], 3)} & "
                         f"{f2(r['gain_zero_instances'])} & {pm(r['time_mean'], r['time_se'], 1)} & "
                         f"{pm(r['gap_mean'], r['gap_se'], 5)} & {f2(r['solved'])}/{r['n_instances']} & {r['n_failed_twice']} \\\\")
    tex_table(out, "experiment_C_methods.tex", "experiment_C_benchmark/methods.csv",
              "% Experiment C: per class and method. Columns: method; hit@1; gain of the first set (share of the LP gap); "
              "instances with gain 0; branch-and-price time (s); final gap; solved/instances; runs failed twice. Mean $\\pm$ "
              "standard error (cluster bootstrap over prompt groups). Source: data/experiment_C_benchmark/methods.csv", lines)
    CP = dict_rows(R / "experiment_C/benchmark_paired_jointdw.csv")
    lines = []
    for cls in ["lot_sizing", "network_design", "facility_location", "all"]:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{CLASSES.get(cls, 'All classes')}}}}} \\\\")
        for r in [r for r in CP if r["class"] == cls]:
            m, se, star = r["mean_difference"], r["se_difference"], r["asterisk"]
            if m not in ("", None) and abs(float(m)) < TIME_RESIDUE:   # floating-point residue of identical times
                m, se, star = 0.0, (se if se in ("", None) or abs(float(se)) >= TIME_RESIDUE else 0.0), ""
            lines.append(f"JointDW, {r['configuration']} & {r['against']} & {r['n_instances']} & "
                         f"{pm(m, se, 1, star)} & {r['failed_twice_instances'] or '--'} \\\\")
    tex_table(out, "experiment_C_paired_time.tex", "experiment_C_benchmark/paired_jointdw.csv",
              "% Experiment C: JointDW minus each baseline, paired time difference (s). Columns: configuration; baseline; "
              "instances; difference $\\pm$ standard error (asterisk: 95 percent interval excludes 0); runs failed twice. "
              "Source: data/experiment_C_benchmark/paired_jointdw.csv", lines)
    CT = dict_rows(R / "experiment_C/benchmark_paired_true_best.csv")
    lines = []
    for cls in ["lot_sizing", "network_design", "facility_location", "all"]:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{CLASSES.get(cls, 'All classes')}}}}} \\\\")
        for r in [r for r in CT if r["class"] == cls]:
            lines.append(f"{r['method']} & {r['n_instances']} & {f2(r['geometric_mean_ratio'], 3)}{'$^{*}$' if r['asterisk'] == '*' else ''} "
                         f"& [{f2(r['ratio_lo'], 3)}, {f2(r['ratio_hi'], 3)}] & "
                         f"{pm(r['gap_mean_difference'], r['gap_se_difference'], 5, r['gap_asterisk'])} \\\\")
    tex_table(out, "experiment_C_against_true_best.tex", "experiment_C_benchmark/paired_true_best.csv",
              "% Experiment C: each method against the true best set. Columns: method; instances; time ratio (t+1)/(t_best+1), "
              "geometric mean (asterisk: interval of the mean log ratio excludes 0); 95 percent interval; gap difference "
              "$\\pm$ standard error. Source: data/experiment_C_benchmark/paired_true_best.csv", lines)
    CI = dict_rows(R / "experiment_C/benchmark_instances.csv")
    CZ = dict_rows(R / "experiment_C/benchmark_zero_marginal.csv")
    lines = []
    for cls in ["lot_sizing", "network_design", "facility_location"]:
        rs = [r for r in CI if r["class"] == cls]
        z = next(r for r in CZ if r["class"] == cls)
        lines.append(f"{CLASSES[cls]} & {len(rs)} & {sum(r['proved_optimal'] == 'True' for r in rs)} & "
                     f"{len({r['prompt_group'] for r in rs})} & {z['with_zero_marginal_group']} & {z['certain']} & {z['possible']} \\\\")
    z = next(r for r in CZ if r["class"] == "all")
    lines.append(f"All classes & {len(CI)} & {sum(r['proved_optimal'] == 'True' for r in CI)} & "
                 f"{len({(r['class'], r['prompt_group']) for r in CI})} & {z['with_zero_marginal_group']} & {z['certain']} & {z['possible']} \\\\")
    tex_table(out, "experiment_C_instances.tex", "experiment_C_benchmark/instances.csv, zero_marginal.csv",
              "% Experiment C instances. Columns: class; instances; proved optimum; distinct prompts (prompt groups); with a "
              "zero-marginal group at the empty set; certain; possible. Sources: data/experiment_C_benchmark/instances.csv, "
              "data/experiment_C_benchmark/zero_marginal.csv", lines)

    # learned predictor
    SPLITS = ["within_class", "leave_one_class_out", "miplibnl_leave_one_out"]
    lines = []
    for sp in SPLITS:
        for var in ["structural", "structural+lookahead"]:
            P = dict_rows(R / f"learned/learned_prediction_{sp}_{var}.csv")
            S = dict_rows(R / f"learned/learned_search_{sp}_{var}.csv")
            L = dict_rows(R / f"learned/learned_labeling_{sp}_{var}.csv")
            lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{EXACT[sp]}, {EXACT.get(var, var)}}}}} \\\\")
            for st in ["templates", "rich_models", "miplib"]:
                for s in ["2", "3"]:
                    p = next((r for r in P if r["set"] == st and r["class"] == "all" and r["s"] == s and r["order"] == "learned"), None)
                    q = next((r for r in P if r["set"] == st and r["class"] == "all" and r["s"] == s and r["order"] != "learned"), None)
                    so = next((r for r in S if r["set"] == st and r["class"] == "all" and r["s"] == s and r["method"] == "learned order"), None)
                    ss = next((r for r in S if r["set"] == st and r["class"] == "all" and r["s"] == s and r["method"].startswith("staged")), None)
                    lb = next((r for r in L if r["set"] == st and r["class"] == "all" and r["s"] == s), None)
                    if not (p or so):
                        continue
                    lines.append(f"{EXACT[st]} & ${s}$ & {pm(p['hit1_mean'], p['hit1_se']) if p else '--'} & "
                                 f"{pm(q['hit1_mean'], q['hit1_se']) if q else '--'} & "
                                 f"{pm(so['evaluations_mean'], so['evaluations_se']) if so else '--'} & "
                                 f"{pm(ss['evaluations_mean'], ss['evaluations_se']) if ss else '--'} & "
                                 f"{f2(lb['labeling_cost_mean'], 1) if lb else '--'} & {so['n_instances'] if so else '--'} \\\\")
    tex_table(out, "learned_predictor.tex", "learned_predictor/*.csv",
              "% Learned predictor (XGBoost, both variants, three splits; the training-size split is in "
              "learned_predictor_training_size.tex). Columns: set; s; hit@1 learned; hit@1 five-dual lookahead; evaluations of "
              "the learned order; evaluations of the staged search anchored at the learned first set (mean $\\pm$ standard "
              "error); mean labeling cost (evaluations to label the training instances); instances. Sources: "
              "data/learned_predictor/prediction_*.csv, search_*.csv, labeling_*.csv", lines)
    lines = []
    for var in ["structural", "structural+lookahead"]:
        T = dict_rows(R / f"learned/learned_training_size_training_size_{var}.csv")
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{EXACT.get(var, var)}}}}} \\\\")
        for r in T:
            lines.append(f"{CLASSES[r['class']]} & {r['training_size']} & ${r['s']}$ & {r['draws_with_value']}/{r['draws']} & "
                         f"{pm(r['cost_mean'], r['cost_se'])} & {f2(r['labeling_cost_mean'], 1)} \\\\")
    tex_table(out, "learned_predictor_training_size.tex", "learned_predictor/training_size_*.csv",
              "% Learned predictor against training size (rich models). Columns: class; training instances; s; draws with "
              "a value / draws; evaluations of the learned order (mean $\\pm$ standard error over draws); mean labeling cost. "
              "Sources: data/learned_predictor/training_size_structural.csv, training_size_structural_lookahead.csv", lines)

    # combined orders
    U = dict_rows(R / "combined/union_search.csv")
    lines = []
    for st in ["templates", "rich_models", "miplib"]:
        rs = [r for r in U if r["set"] == st and r["class"] == "all"]
        if not rs:
            continue
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{EXACT[st]}}}}} \\\\")
        keys = []
        for r in rs:
            if (r["configuration"], r["method"]) not in keys:
                keys.append((r["configuration"], r["method"]))
        for k in keys:
            cells = [clean_text(f"{k[1]}, {k[0]}")]
            for s in ["2", "3", "k-1"]:
                r = next((r for r in rs if (r["configuration"], r["method"]) == k and r["size_limit"] == s), None)
                if r and r["evaluations_mean"] == "" and r.get("share_mean"):
                    cells.append(pm(r["share_mean"], r["share_se"]))
                else:
                    cells.append(pm(r["evaluations_mean"], r["evaluations_se"]) if r else "--")
            cells.append(next((r["n_instances"] for r in rs if (r["configuration"], r["method"]) == k), "--"))
            lines.append(" & ".join(cells) + " \\\\")
    tex_table(out, "combined_union_first.tex", "combined_orders/union_search.csv",
              "% Combined orders, union-first. Columns: order and answers; $s=2$; $s=3$; $s=k-1$ (evaluations, mean $\\pm$ "
              "standard error); instances. Source: data/combined_orders/union_search.csv", lines)
    lines = []
    for sp in SPLITS:
        S = dict_rows(R / f"combined/stacked_search_{sp}.csv")
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{EXACT[sp]}}}}} \\\\")
        for st in ["templates", "rich_models", "miplib"]:
            rs = [r for r in S if r["set"] == st and r["class"] == "all"]
            keys = []
            for r in rs:
                k = (r["configuration"], r["rich_model_answers"], r["method"])
                if k not in keys:
                    keys.append(k)
            for k in keys:
                cells = [EXACT[st], clean_text(f"{k[2]}; {k[0]}" + (f"; rich models: {k[1]}" if k[1] else ""))]
                for s in ["2", "3"]:
                    r = next((r for r in rs if (r["configuration"], r["rich_model_answers"], r["method"]) == k and r["size_limit"] == s), None)
                    cells.append(pm(r["evaluations_mean"], r["evaluations_se"]) if r else "--")
                lines.append(" & ".join(cells) + " \\\\")
    tex_table(out, "combined_stacked.tex", "combined_orders/stacked_search_*.csv",
              "% Combined orders, stacked learned predictor. Columns: set; order and answers; $s=2$; $s=3$ (evaluations, mean "
              "$\\pm$ standard error). Sources: data/combined_orders/stacked_search_*.csv", lines)

    # benchmark instances: paired differences of hit@1 and of the gain of the first set
    hp, gp = R / "experiment_C/benchmark_paired_hit1.csv", R / "experiment_C/benchmark_paired_gain.csv"
    if hp.exists() and gp.exists():
        H, G = dict_rows(hp), dict_rows(gp)
        gain = {(r["configuration"], r["against"], r["class"]): r for r in G}
        lines = []
        for r in H:
            g = gain[(r["configuration"], r["against"], r["class"])]
            lines.append(f"{METHOD_NAMES.get(r['configuration'], 'JointDW, ' + r['configuration'])} & "
                         f"{METHOD_NAMES[r['against']]} & {CLASSES.get(r['class'], 'All classes')} & {r['n_instances']} & "
                         f"{pm(r['mean_difference'], r['se_difference'], 2, r['asterisk'])} & "
                         f"{pm(g['mean_difference'], g['se_difference'], 2, g['asterisk'])} \\\\")
        tex_table(out, "experiment_C_paired_hit1.tex", "", "", lines)
    # rich models with every coefficient: JointDW and its interleaving minus the five-dual lookahead order
    dp = R / "rich_models_data_paired_five_dual.csv"
    if dp.exists():
        D = dict_rows(dp)
        idx = {(r["configuration"], r["presentation"], r["class"], r["method"], r["s"]): r for r in D}
        keys = []
        for r in D:
            if (r["configuration"], r["presentation"]) not in keys:
                keys.append((r["configuration"], r["presentation"]))
        keys.sort(key=lambda k: (CFG_ORDER.index(k[0]), ["data", "data, permuted", "data, anonymized"].index(k[1])))
        lines = []
        for cls in ["pooled", "lot_sizing", "network_design", "facility_location"]:
            lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{'Pooled' if cls == 'pooled' else CLASSES[cls]}}}}} \\\\")
            for cfg, pres in keys:
                cells = [f"{cfg}, {pres}"]
                for meth in ("JointDW", "JointDW interleaved with the five-dual order"):
                    for sz in ("2", "3"):
                        r = idx.get((cfg, pres, cls, meth, sz))
                        cells.append(pm(r["mean_difference"], r["se_difference"], 2, r["asterisk"]) if r else "--")
                lines.append(" & ".join(cells) + " \\\\")
        tex_table(out, "rich_models_data_vs_five_dual.tex", "", "", lines)
    # benchmark instances, data variant (every coefficient shown): methods and paired differences
    mp, pp = R / "experiment_C/benchmark_data_methods.csv", R / "experiment_C/benchmark_data_paired.csv"
    if mp.exists() and pp.exists():
        def label(m):
            if m == "true best set":
                return "True best set"
            if m in METHOD_NAMES:
                return METHOD_NAMES[m]
            return f"JointDW, {m[:-len(', data')]}, data" if m.endswith(", data") else f"JointDW, {m}, original"
        M = dict_rows(mp)
        lines = []
        for cls in ["lot_sizing", "network_design", "facility_location", "all"]:
            lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{CLASSES.get(cls, 'All classes')}}}}} \\\\")
            for r in [r for r in M if r["class"] == cls]:
                ratio = ("--" if r["method"] == "true best set" else
                         f"{f2(r['geometric_mean_ratio'], 3)}{'$^{*}$' if r['ratio_asterisk'] == '*' else ''} "
                         f"[{f2(r['ratio_lo'], 3)}, {f2(r['ratio_hi'], 3)}]")
                nojs = (f"{r['n_samples_without_joint_support']}/{r['n_samples']}"
                        if r.get("n_samples_without_joint_support") not in (None, "") else "--")
                lines.append(f"{label(r['method'])} & {pm(r['hit_mean'], r['hit_se'])} & {pm(r['gain_mean'], r['gain_se'], 3)} & "
                             f"{pm(r['time_mean'], r['time_se'], 1)} & {pm(r['gap_mean'], r['gap_se'], 5)} & "
                             f"{f2(r['solved'])}/{r['n_instances']} & {ratio} & {nojs} \\\\")
        tex_table(out, "experiment_C_data_methods.tex", "", "", lines)
        P = dict_rows(pp)
        idx = {(r["class"], r["configuration"], r["against"], r["metric"]): r for r in P}
        keys = []
        for r in P:
            if (r["class"], r["configuration"], r["against"]) not in keys:
                keys.append((r["class"], r["configuration"], r["against"]))
        lines = []
        for cls, cfg, against in keys:
            cells = [f"JointDW, {cfg}, data", METHOD_NAMES.get(against, against.replace(", original", ", original presentation")
                                                             if against.endswith(", original") else against),
                     CLASSES.get(cls, "All classes"), idx[(cls, cfg, against, "hit@1")]["n_instances"]]
            for metric, d in (("hit@1", 2), ("gain", 3), ("time", 1)):
                r = idx[(cls, cfg, against, metric)]
                cells.append(pm(r["mean_difference"], r["se_difference"], d, r["asterisk"]))
            lines.append(" & ".join(str(c) for c in cells) + " \\\\")
        tex_table(out, "experiment_C_data_paired.tex", "", "", lines)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the output directory of scripts/reproduce.sh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    names = write_tables(Path(args.results), Path(args.out))
    print(f"wrote {len(names)} table bodies to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
