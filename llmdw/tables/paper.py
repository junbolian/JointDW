"""LaTeX bodies of Tables 1, 2 and 6 to 15 of the paper and the data behind Figures 2(b), 3 and 4, from the CSV files
that llmdw.scoring.results writes.

Numbers are rounded half up (decimal.ROUND_HALF_UP) to two decimals; a minus sign is written $-$; an asterisk marks a
paired difference whose 95 percent bootstrap interval excludes zero. Rows name the configurations by their official
names, in the order of the paper.

Files written to --out:
  table_01.tex ... table_15.tex   one body per table (Tables 1, 2, 6 to 15)
  figure_1.csv                    share of the LP gap closed by every family set of the template clsp_2x2 (with --work)
  figure_2a.csv                   the queries of the staged search for k = 6, s = 3, w = 2 and the prediction {1, 2}
  figure_2b.csv                   staged-search queries and lower bounds for w = 3, s = 6 (the CSV of llmdw.tables.figure2)
  figure_3a_<kind>.csv            census: largest single-family gain and largest set gain (shares of the LP gap, at most
                                  1) of each model with a positive gain, per kind of group
  figure_3b_templates.csv, figure_3c_miplib.csv   recall by presentation per configuration, and the lookahead recall
  figure_4a.csv, figure_4b.csv    rich models: hit@1 given the text, every coefficient and the two control presentations
                                  (results/experiment_A); evaluations at s = 2 by class, with the five-dual lookahead
                                  (results/learned)

Usage
-----
python -m llmdw.tables.paper --results results --work data --out results/paper
"""
from __future__ import annotations

import argparse
import csv
import shutil
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from llmdw.providers import settings

PHASES = [("templates", "Templates"), ("miplib", "MIPLIB-NL")]
PRESENTATIONS = ["original", "names_only", "permuted", "anonymized"]
PRESENTATION_NAMES = {"original": "Original", "names_only": "Names only", "permuted": "Permuted",
                      "anonymized": "Anonymized"}
LIMITS = ["2", "3", "k-1"]
LIMIT_NAMES = {"2": "$s=2$", "3": "$s=3$", "k-1": "$s=k-1$"}
PAIRED = ["jointdw minus covering", "jointdw minus lookahead", "interleaved minus covering",
          "interleaved minus lookahead"]
CLASSES = [("lot_sizing", "LS"), ("network_design", "ND"), ("facility_location", "FL")]
RICH_PHASE = "rich_models"
CENSUS_TOL = 1e-4                                # a gain counts as positive above this share of the LP gap
FIGURE_4B_CONFIGURATION = "GPT-6.1 Sol"         # the configuration whose JointDW evaluations Figure 4(b) shows


def configurations() -> list[str]:
    return list(settings.names().values())


def fmt(x, d: int = 2) -> str:
    if x in ("", None):
        return "--"
    q = Decimal(str(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return f"{q:.{d}f}".replace("-", "$-$")


def star(lo, hi) -> str:
    return "$^{*}$" if (float(lo) > 0 or float(hi) < 0) else ""


def pm(m, se, d: int = 2) -> str:
    return f"{fmt(m, d)} $\\pm$ {fmt(se, d)}"


def rounded(x) -> Decimal:
    return Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def read(results: Path, name: str) -> list[dict]:
    with open(results / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def heading(n: int, text: str) -> str:
    return f"\\multicolumn{{{n}}}{{@{{}}l}}{{\\emph{{{text}}}}} \\\\"


def tables(results: Path) -> dict[int, str]:
    """{table number: LaTeX body}."""
    cfg = configurations()
    pred = {(r["phase"], r["config"], r["presentation"]): r for r in read(results, "prediction.csv") if r["class"] == "all"}
    alarm = {r["config"]: r for r in read(results, "false_alarm.csv") if r["presentation"] == "original"}
    look = {r["phase"]: r for r in read(results, "lookahead_like_models.csv") if r["m"] == "1"}
    out = {}

    rows = []
    for c in cfg:
        a, b = pred[("templates", c, "original")], pred[("miplib", c, "original")]
        rows.append(f"{c} & {fmt(a['hit1'])} & {fmt(a['recall'])} & {fmt(b['hit1'])} & {fmt(b['recall'])} & "
                    f"{fmt(alarm[c]['false_alarm'])} \\\\")
    rows.append("\\midrule")
    t, m = look["templates"], look["miplib"]
    rows.append(f"Lookahead order & {fmt(t['lookahead_hit1'])} & {fmt(t['lookahead_recall_at_m'])} & "
                f"{fmt(m['lookahead_hit1'])} & {fmt(m['lookahead_recall_at_m'])} & -- \\\\")
    out[1] = "\n".join(rows) + "\n"

    sbs = {(r["phase"], r["limit"], r["method"], r["config"]): r for r in read(results, "search_by_size.csv")
           if r["presentation"] in ("", "original")}
    methods = [("Random, expected", "random", ""), ("Covering", "covering", ""), ("Lookahead", "lookahead", "")]
    methods += [(f"JointDW, {c}", "jointdw", c) for c in cfg]
    cols = [(p, lim) for p, _ in PHASES for lim in LIMITS]
    best = {col: min(rounded(sbs[(col[0], col[1], m, c)]["mean"]) for _, m, c in methods if m != "random")
            for col in cols}
    rows = []
    for i, (label, m, c) in enumerate(methods):
        cells = []
        for col in cols:
            v = sbs[(col[0], col[1], m, c)]["mean"]
            s = fmt(v)
            if m != "random" and rounded(v) == best[col]:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if i == 2:
            rows.append("\\midrule")
    out[2] = "\n".join(rows) + "\n"

    rows = []
    for p, title in PHASES:
        rows.append(heading(5, title))
        for c in cfg:
            rows.append(f"{c} & " + " & ".join(pm(pred[(p, c, k)]["recall"], pred[(p, c, k)]["recall_se"])
                                               for k in PRESENTATIONS) + " \\\\")
    out[6] = "\n".join(rows) + "\n"

    paired = {(r["phase"], r["config"], r["contrast"]): r for r in read(results, "paired.csv")}
    contrasts = ["names_only minus permuted", "names_only minus anonymized", "original minus anonymized"]
    rows = []
    for p, title in PHASES:
        rows.append(heading(4, title))
        for c in cfg:
            cells = [pm(paired[(p, c, k)]["mean_difference"], paired[(p, c, k)]["se_difference"])
                     + star(paired[(p, c, k)]["lo"], paired[(p, c, k)]["hi"]) for k in contrasts]
            rows.append(f"{c} & " + " & ".join(cells) + " \\\\")
    out[7] = "\n".join(rows) + "\n"

    pc = {(r["phase"], r["presentation"], r["measure"]): r for r in read(results, "paired_configs.csv")}
    rows = []
    for p, title in PHASES:
        rows.append(heading(3, title))
        for k in PRESENTATIONS:
            cells = [pm(pc[(p, k, meas)]["mean_difference"], pc[(p, k, meas)]["se_difference"])
                     + star(pc[(p, k, meas)]["lo"], pc[(p, k, meas)]["hi"]) for meas in ("joint_hit1", "joint_recall")]
            rows.append(f"{PRESENTATION_NAMES[k]} & " + " & ".join(cells) + " \\\\")
    out[8] = "\n".join(rows) + "\n"

    rows = []
    for p, title in PHASES:
        rows.append(heading(4, title))
        for label, m, c in ([("Random order", "random", ""), ("Covering order", "covering", ""),
                             ("Lookahead order", "lookahead", "")] + [(f"JointDW, {c}", "jointdw", c) for c in cfg]):
            rows.append(f"{label} & " + " & ".join(pm(sbs[(p, lim, m, c)]["mean"], sbs[(p, lim, m, c)]["mean_se"])
                                                   for lim in LIMITS) + " \\\\")
    out[9] = "\n".join(rows) + "\n"

    def count(x):
        f = float(x)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    comp = {(r["phase"], r["config"]): r for r in read(results, "complementarity.csv")}
    rows = []
    for p, title in PHASES:
        for c in cfg:
            r = comp[(p, c)]
            rows.append(f"{title} & {c} & {count(r['both'])} & {count(r['model_only'])} & {count(r['lookahead_only'])} & "
                        f"{count(r['neither'])} \\\\")
    out[10] = "\n".join(rows) + "\n"

    text = {r["config"]: r for r in read(results, "rich_models_prediction.csv") if r["class"] == "all"}
    data = {r["config"]: r for r in read(results, "rich_models_with_data_prediction.csv") if r["class"] == "all"}
    out[11] = "\n".join(f"{c} & {pm(text[c]['hit1'], text[c]['hit1_se'])} & {pm(data[c]['hit1'], data[c]['hit1_se'])} & "
                        f"{pm(text[c]['recall'], text[c]['recall_se'])} & {pm(data[c]['recall'], data[c]['recall_se'])} \\\\"
                        for c in cfg) + "\n"

    s0 = {(r["s"], r["method"], r["config"]): r for r in read(results, "rich_models_search.csv") if r["class"] == "all"}
    s1 = {(r["s"], r["method"], r["config"]): r for r in read(results, "rich_models_with_data_search.csv")
          if r["class"] == "all"}
    rows = []
    for label, m in [("Random order", "random"), ("Covering order", "covering"), ("Lookahead order", "lookahead")]:
        rows.append(f"{label} & \\multicolumn{{2}}{{c}}{{{pm(s0[('2', m, '')]['mean'], s0[('2', m, '')]['mean_se'])}}} & "
                    f"\\multicolumn{{2}}{{c}}{{{pm(s0[('3', m, '')]['mean'], s0[('3', m, '')]['mean_se'])}}} \\\\")
    r2, r3 = s0[("2", "gcg_anchor_gcg_spfwh_names", "")], s0[("3", "gcg_anchor_gcg_spfwh_names", "")]
    rows.append(f"GCG anchor, {r2['n_instances']} instances & \\multicolumn{{2}}{{c}}{{{pm(r2['mean'], r2['mean_se'])}}} & "
                f"\\multicolumn{{2}}{{c}}{{{pm(r3['mean'], r3['mean_se'])}}} \\\\")
    for m, title in [("jointdw", "JointDW"), ("interleaved", "Interleaved with lookahead")]:
        rows.append(heading(5, title))
        for c in cfg:
            cells = []
            for s in ("2", "3"):
                cells += [pm(s0[(s, m, c)]["mean"], s0[(s, m, c)]["mean_se"]),
                          pm(s1[(s, m, c)]["mean"], s1[(s, m, c)]["mean_se"])]
            rows.append(f"{c} & " + " & ".join(cells) + " \\\\")
    out[12] = "\n".join(rows) + "\n"

    rp = {(r["phase"], r["limit"], r["config"]): r for r in read(results, "ranked_prompt.csv")}
    rpp = {(r["phase"], r["limit"], r["config"], r["contrast"]): r for r in read(results, "ranked_prompt_paired.csv")}
    rows = []
    for p, title in [("miplib", "MIPLIB-NL"), (RICH_PHASE, "Rich models")]:
        rows.append(heading(5, title))
        for c in cfg:
            r = rp[(p, "2", c)]
            cells = [pm(r["coverage_at_1"], r["coverage_at_1_se"]), pm(r["coverage_at_3"], r["coverage_at_3_se"])]
            for s in ("2", "3"):
                q = rpp[(p, s, c, "ranked+lookahead minus lookahead")]
                cells.append(pm(q["mean_difference"], q["se_difference"]) + star(q["lo"], q["hi"]))
            rows.append(f"{c} & " + " & ".join(cells) + " \\\\")
    out[13] = "\n".join(rows) + "\n"

    ps = {(r["phase"], r["limit"], r["config"], r["unit"], r["contrast"]): r for r in read(results, "paired_search.csv")}

    def paired_blocks(blocks):
        rows = []
        for p, unit, title in blocks:
            for lim in LIMITS:
                rows.append(heading(5, f"{title}, {LIMIT_NAMES[lim]}"))
                for c in cfg:
                    cells = []
                    for k in PAIRED:
                        r = ps[(p, lim, c, unit, k)]
                        cells.append(pm(r["mean_difference"], r["se_difference"]) + star(r["lo"], r["hi"]))
                    rows.append(f"{c} & " + " & ".join(cells) + " \\\\")
        return "\n".join(rows) + "\n"
    out[14] = paired_blocks([("miplib", "seconds", "MIPLIB-NL, seconds")])
    out[15] = paired_blocks([("templates", "evaluations", "Templates, evaluations"),
                             ("miplib", "evaluations", "MIPLIB-NL, evaluations")])
    return out


def figure_1(work: Path, out: Path, template: str = "clsp_2x2") -> str:
    """Figure 1: every family set of the template, with the share of the LP gap its relaxation closes (exact,
    rounded half up to whole percent; the families as the letters of Table 4)."""
    import json
    from fractions import Fraction

    from llmdw.scoring.results import _round_half_up, family_letter
    gt = json.loads((work / "work/templates" / template / "ground_truth.json").read_text(encoding="utf-8"))
    gain = {k: Fraction(v) for k, v in gt["gain"].items()}
    full = gain["|".join(sorted(gt["families"]))]
    def letters(k):
        return "".join(sorted(family_letter(x) for x in k.split("|") if x))
    keys = sorted(gain, key=lambda k: (len(letters(k)), letters(k)))
    with open(out / "figure_1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["families", "gap_closed_percent"])
        for k in keys:
            w.writerow([letters(k) or "LP", _round_half_up(100 * gain[k] / full, 0)])
    return "figure_1.csv"


def figure_2a(out: Path) -> str:
    """Figure 2(a): the queries of the staged search (search_sim.staged_order) for k = 6, s = 3 and the predicted
    group {1, 2}, one row per query, from the definition."""
    from llmdw.search.search_sim import staged_order
    with open(out / "figure_2a.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query", "families"])
        for i, T in enumerate(staged_order([1, 2, 3, 4, 5, 6], 3, {1, 2}), 1):
            w.writerow([i, " ".join(str(f) for f in sorted(T))])
    return "figure_2a.csv"


def figures(results: Path, out: Path) -> list[str]:
    """Write the figure data; returns the file names."""
    names = [figure_2a(out)]
    if (results / "figure2.csv").exists():
        shutil.copyfile(results / "figure2.csv", out / "figure_2b.csv")
        names.append("figure_2b.csv")
    rows = read(results, "census_figure.csv")
    for kind in ("certain", "possible", "none"):
        with open(out / f"figure_3a_{kind}.csv", "w", encoding="utf-8") as fh:
            fh.write("single set\n")
            for r in rows:
                if r["group_kind"] == kind and float(r["max_set_gain"]) > CENSUS_TOL:   # a positive gain
                    single, best = (min(1.0, float(r[c] or 0)) for c in ("max_single_family_gain", "max_set_gain"))
                    fh.write(f"{single:.6f} {best:.6f}\n")
        names.append(f"figure_3a_{kind}.csv")
    pred = {(r["phase"], r["config"], r["presentation"]): r for r in read(results, "prediction.csv") if r["class"] == "all"}
    look = {r["phase"]: r for r in read(results, "lookahead_like_models.csv") if r["m"] == "1"}
    cfg = configurations()
    for p, fname in (("templates", "figure_3b_templates.csv"), ("miplib", "figure_3c_miplib.csv")):
        with open(out / fname, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["presentation"] + cfg + ["lookahead"])
            for k in PRESENTATIONS:
                w.writerow([k] + [f"{float(pred[(p, c, k)]['recall']):.3f}" for c in cfg]
                           + [f"{float(look[p]['lookahead_recall_at_m']):.3f}"])
        names.append(fname)
    text = {r["config"]: r for r in read(results, "rich_models_prediction.csv") if r["class"] == "all"}
    data = {r["config"]: r for r in read(results, "rich_models_with_data_prediction.csv") if r["class"] == "all"}
    # the control presentations with every coefficient (results/experiment_A), for the configurations run with them
    ctrl_path = results / "experiment_A" / "control_prediction.csv"
    ctrl = {}
    if ctrl_path.exists():
        with open(ctrl_path, newline="", encoding="utf-8") as fh:
            ctrl = {(r["configuration"], r["presentation"]): r for r in csv.DictReader(fh) if r["class"] == "pooled"}

    def control(c, pres):
        r = ctrl.get((c, pres))
        return f"{float(r['hit1_mean']):.3f}" if r else ""
    with open(out / "figure_4a.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration", "hit1_text", "hit1_every_coefficient", "hit1_data_permuted", "hit1_data_anonymized"])
        for c in cfg:
            w.writerow([c, f"{float(text[c]['hit1']):.3f}", f"{float(data[c]['hit1']):.3f}",
                        control(c, "data, permuted"), control(c, "data, anonymized")])
    s0 = {(r["class"], r["s"], r["method"], r["config"]): r for r in read(results, "rich_models_search.csv")}
    s1 = {(r["class"], r["s"], r["method"], r["config"]): r for r in read(results, "rich_models_with_data_search.csv")}
    gpt = FIGURE_4B_CONFIGURATION
    # the five-dual lookahead order by class (its cost does not depend on the split: results/learned)
    five_path = results / "learned" / "learned_search_within_class_structural.csv"
    five = {}
    if five_path.exists():
        with open(five_path, newline="", encoding="utf-8") as fh:
            five = {r["class"]: r for r in csv.DictReader(fh) if r["set"] == "rich_models" and r["s"] == "2"
                    and r["method"] == "five-dual lookahead order"}
    with open(out / "figure_4b.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "covering", "lookahead", "lookahead_five_duals", f"jointdw_text ({gpt})",
                    f"jointdw_every_coefficient ({gpt})"])
        for cls, short in CLASSES:
            w.writerow([short] + [f"{float(s0[(cls, '2', m, '')]['mean']):.2f}" for m in ("covering", "lookahead")]
                       + [f"{float(five[cls]['evaluations_mean']):.2f}" if cls in five else ""]
                       + [f"{float(s0[(cls, '2', 'jointdw', gpt)]['mean']):.2f}",
                          f"{float(s1[(cls, '2', 'jointdw', gpt)]['mean']):.2f}"])
    names += ["figure_4a.csv", "figure_4b.csv"]
    return names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the output directory of llmdw.scoring.results")
    ap.add_argument("--work", help="the data root (data): Figure 1 from its templates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, body in tables(res).items():
        (out / f"table_{n:02d}.tex").write_text(body, encoding="utf-8")
    figures(res, out)
    if args.work:
        figure_1(Path(args.work), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
