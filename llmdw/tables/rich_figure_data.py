"""Figure data of the rich models across the number of families, from the stored result CSVs only (no solver run, no
request).

rich_cost_by_class_k.csv  one row per rich-model class, number of families k (7 to 9, 12, 14, 16 as built) and budget
                          s in {2, 3}: for each method the mean number of evaluations until a best set is evaluated,
                          its standard error and the number of instances (from the cost-against-k figure data, which
                          holds both collections with the bootstraps of their tables)
rich_hit1.csv             hit@1 with its standard error per configuration and presentation (original; data variant;
                          data variant with the family names permuted; data variant with labels), per class and
                          pooled, and per k: the rich models with 7 to 9 families (rich_models_prediction.csv,
                          rich_models_with_data_prediction.csv, the control presentations) and with 12 and 16 families
                          (predictions.csv of that collection)

Usage
-----
python -m llmdw.tables.rich_figure_data --cost-vs-k FIG/experiment_E_cost_vs_k.csv --results RESULTS \
    --large-predictions results/experiment_E/predictions.csv --out FIG
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

METHODS = [("covering", "covering", "", ""),
           ("lookahead_highs_dual", "lookahead, HiGHS dual", "", ""),
           ("lookahead_five_duals", "lookahead, five duals", "", "")]
for _cfg, _slug in (("GPT-6.1 Sol", "gpt_6_1_sol"), ("Claude Sonnet 5.5", "claude_sonnet_5_5")):
    for _pres in ("original", "data"):
        METHODS.append((f"jointdw_{_slug}_{_pres}", "JointDW", _cfg, _pres))
for _cfg, _slug in (("GPT-6.1 Sol", "gpt_6_1_sol"), ("Claude Sonnet 5.5", "claude_sonnet_5_5")):
    for _pres in ("original", "data"):
        METHODS.append((f"interleaved_five_dual_{_slug}_{_pres}", "JointDW interleaved with the five-dual order", _cfg, _pres))
CLASSES = ("lot_sizing", "network_design", "facility_location")
PRESENTATIONS = {"original": "original", "data": "data variant", "data, permuted": "data variant, names permuted",
                 "data, anonymized": "data variant, labels"}


def _rows(p: Path) -> list:
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(p: Path, rows: list, keys: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def cost_by_class_k(cost_vs_k: list) -> list:
    idx = {(r["class"], r["k"], r["s"], r["order"], r["config"], r["presentation"]): r for r in cost_vs_k
           if r.get("status", "computed") == "computed"}
    cells = sorted({(r["set"], r["class"], int(r["k"]), int(r["s"])) for r in cost_vs_k if r["class"] in CLASSES
                    and r["s"] in ("2", "3")}, key=lambda c: (CLASSES.index(c[1]), c[2], c[3]))
    out = []
    for st, c, k, s in cells:
        row = {"set": st, "class": c, "k": k, "s": s}
        for slug, order, cfg, pres in METHODS:
            r = idx.get((c, str(k), str(s), order, cfg, pres))
            row.update({f"{slug}_mean": r["mean_evaluations"] if r else "", f"{slug}_se": r["se"] if r else "",
                        f"{slug}_n": r["n_instances"] if r else ""})
        out.append(row)
    return out


K_SMALL = {"lot_sizing": "9", "network_design": "7", "facility_location": "7", "all": "7 to 9"}


def hit1_rows(results: Path, large_predictions: Path) -> list:
    out = []
    small = [(r, "original") for r in _rows(results / "rich_models_prediction.csv")]
    small += [(r, "data") for r in _rows(results / "rich_models_with_data_prediction.csv")]
    for r, pres in small:
        cls = "all" if r["class"] in ("all", "pooled") else r["class"]
        out.append({"set": "rich models, 7 to 9 families", "class": cls, "k": K_SMALL[cls],
                    "configuration": r["config"], "presentation": PRESENTATIONS[pres], "n_instances": r["n_instances"],
                    "n_groups": r["n_groups"], "hit1_mean": r["hit1"], "hit1_se": r["hit1_se"]})
    for r in _rows(results / "experiment_A" / "control_prediction.csv"):
        if r["presentation"] == "data":
            continue                                   # the data variant comes from rich_models_with_data_prediction
        cls = "all" if r["class"] in ("all", "pooled") else r["class"]
        out.append({"set": "rich models, 7 to 9 families", "class": cls, "k": K_SMALL[cls],
                    "configuration": r["configuration"], "presentation": PRESENTATIONS[r["presentation"]],
                    "n_instances": r["n_instances"], "n_groups": r["n_groups"], "hit1_mean": r["hit1_mean"],
                    "hit1_se": r["hit1_se"]})
    for r in _rows(large_predictions):
        out.append({"set": "rich models, 12 to 16 families", "class": r["class"],
                    "k": "12 to 16" if r["k"] == "all" else r["k"], "configuration": r["config"],
                    "presentation": PRESENTATIONS[r["presentation"]], "n_instances": r["n_instances"],
                    "n_groups": r["n_groups"], "hit1_mean": r["hit1_mean"], "hit1_se": r["hit1_se"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-vs-k", required=True)
    ap.add_argument("--results", required=True, help="the results folder of the main tables (rich models, controls)")
    ap.add_argument("--large-predictions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    cost = cost_by_class_k(_rows(Path(a.cost_vs_k)))
    _write(out / "rich_cost_by_class_k.csv", cost, list(cost[0]))
    hit = hit1_rows(Path(a.results), Path(a.large_predictions))
    _write(out / "rich_hit1.csv", hit, list(hit[0]))
    print(f"rich_cost_by_class_k.csv {len(cost)} rows; rich_hit1.csv {len(hit)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
