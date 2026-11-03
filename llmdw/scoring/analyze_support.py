"""Scoring of support predictions: does the model predict which constraint families must be convexified together,

Paper: Section 6 (measures): hit@1, recall, false alarms and the prediction error.
and how much search does its prediction save?

Ground truth
- template instances: ground_truth.json from llmdw.instances.export (exact rational computation)
- MIPLIB-NL instances: oracle.json (+ lookahead.json) through llmdw.scoring.structure; a joint support counts
  as true when it is "certain" (every zero below it comes from a converged evaluation). An instance is scored
  only when at least one family set (without linking variables) was evaluated by the oracle with a bound
  (scorable); an instance with lookahead values only is not scored

Per (instance, condition, sample); summaries average over samples and instances, and the paired differences between
conditions average over instances the per-instance sample means
- joint_hit1      the first predicted joint support is a true joint support
- joint_recall    fraction of true joint supports that are predicted
- joint_false     a joint support is predicted on an instance that has none (false alarm)
- alone_acc       accuracy of "strengthens alone" over families answered yes/no
- e               |B* minus P| for the best convexified set B* (closest best set) and P the families
                  convexified by the top-ranked candidate
- evals_anchored  evaluations to reach the best gain with the staged anchoring of Theorem 2 around P
                  (search_sim.anchored); evals_robust interleaves it with the prediction-free covering order.
                  Baselines on the same table: random (expected), greedy, covering and, when available, the
                  lookahead ranking and GCG's own choice as an anchor (budget s = k - 1 families)

Usage
-----
python -m llmdw.scoring.analyze_support --runs data/work/support_templates/$MODEL --templates data/work/templates --out results/scores_templates
python -m llmdw.scoring.analyze_support --runs data/work/support_miplib/$MODEL --oracle-root data/work/oracle \
    --lookahead-root data/work/lookahead --out results/scores_miplib
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from llmdw.search import search_sim
from llmdw.scoring import structure


def template_truth(d: Path) -> dict:
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    fams = gt["families"]
    table = {}
    for key, v in gt["gain"].items():
        T = frozenset(x for x in key.split("|") if x)
        if len(T) < len(fams):          # the full set is the MILP itself (excluded, as in DW practice)
            table[T] = float(Fraction(v))
    return {"families": fams, "joint": [frozenset(x) for x in gt["joint_supports"]],
            "alone": set(gt["strengthens_alone"]), "table": table, "lookahead": None}


def scorable_table(tab: dict | None) -> bool:
    """A MIPLIB-NL instance is scored only when its table without linking variables holds at least one family set
    evaluated by the oracle with a bound: a true joint support needs converged evaluations, so lookahead values alone
    cannot define one."""
    return bool(tab) and any(T and "oracle" in v["source"] for T, v in tab.items())


def scorable(oracle_path: Path, lookahead_path: Path | None = None) -> bool:
    o = json.loads(oracle_path.read_text(encoding="utf-8"))
    la = json.loads(lookahead_path.read_text(encoding="utf-8")) if lookahead_path and lookahead_path.exists() else None
    return scorable_table(structure.instance_tables(o, la)["tables"].get(()))


def miplib_truth(oracle_path: Path, lookahead_path: Path | None) -> dict | None:
    o = json.loads(oracle_path.read_text(encoding="utf-8"))
    la = json.loads(lookahead_path.read_text(encoding="utf-8")) if lookahead_path and lookahead_path.exists() else None
    info = structure.instance_tables(o, la)
    tab = info["tables"].get(())
    if not scorable_table(tab):
        return None
    tol = 1e-4 if info["scale"] else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
    groups = structure.zero_marginal_groups(tab, tol=tol)
    joint = [g["group"] for g in groups if g["size"] >= 2 and g["status"] == "certain"]
    alone = {next(iter(g["group"])) for g in groups if g["size"] == 1}
    table = {T: v["gain"] for T, v in tab.items()}
    lookahead = None
    if la:
        k = len(info["families"])                   # the full family set is not a decomposition (see structure)
        lookahead = sorted(((frozenset(r["families"]), r["lookahead"] or 0.0) for r in la["table"].values()
                            if len(r["families"]) < k), key=lambda kv: -kv[1])
    gcg = {}
    for name, bl in (o.get("baselines") or {}).items():
        if bl.get("representable") and bl.get("aligned_master_families") is not None:
            gcg[name] = frozenset(info["families"]) - frozenset(bl["aligned_master_families"])
    return {"families": info["families"], "joint": joint, "alone": alone, "table": table, "lookahead": lookahead,
            "gcg": gcg}


def score(rec: dict, truth: dict) -> dict:
    fams = truth["families"]
    pred = [frozenset(s["families"]) for s in rec.get("supports") or []]
    out = {"instance": rec["instance"], "condition": rec["condition"], "sample": rec.get("sample", 0),
           "n_true_joint": len(truth["joint"])}
    if truth["joint"]:
        out["joint_hit1"] = bool(pred) and pred[0] in truth["joint"]
        out["joint_recall"] = sum(1 for W in truth["joint"] if W in pred) / len(truth["joint"])
    else:
        out["joint_false"] = bool(pred)
    answered = [(f, v.get("strengthens_alone")) for f, v in (rec.get("family_info") or {}).items()
                if v.get("strengthens_alone") in ("yes", "no")]
    if answered:
        out["alone_acc"] = sum(1 for f, a in answered if (a == "yes") == (f in truth["alone"])) / len(answered)
    table = truth["table"]
    s = len(fams) - 1
    if table and max(table.values()) > 0:
        best = max(table.values())
        best_sets = [T for T, v in table.items() if v >= best - 1e-9 and T]
        top = (rec.get("candidates") or [None])[0]
        P = frozenset(fams) - frozenset(top["master_families"]) if top else frozenset()
        out["e"] = min(len(B - P) for B in best_sets)
        out["evals_anchored"] = search_sim.anchored(table, s, P) if P else None
        out["evals_robust"] = search_sim.robust(table, s, P) if P else None
        out["evals_random"] = search_sim.random_expected(table)
        out["evals_greedy"] = search_sim.greedy(table)
        out["evals_covering"] = search_sim.covering(table, s)
        if truth.get("lookahead"):
            from llmdw.search.relabel import lookahead_list       # the lookahead order by name, values within 1e-9 tied
            out["evals_lookahead"] = search_sim.ranked(table, lookahead_list(truth["lookahead"], sorted(fams),
                                                                             len(fams)))
        for name, Pg in (truth.get("gcg") or {}).items():          # GCG's own choice as an anchor
            out[f"e_{name}"] = min(len(B - Pg) for B in best_sets)
            out[f"evals_robust_{name}"] = search_sim.robust(table, s, Pg)
    return out


def by_instance(rows: list, key: str) -> dict:
    """{(instance, condition): mean of key over the samples of that instance and condition}."""
    acc = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            acc[(r["instance"], r["condition"])].append(float(r[key]))
    return {ic: sum(v) / len(v) for ic, v in acc.items()}


def paired(rows: list, a: str, b: str, key: str = "joint_recall") -> dict:
    """Mean over instances of (sample mean under condition a) - (sample mean under condition b), on the
    instances that have key under both conditions (for example a = names_only, b = permuted or anonymized)."""
    m = by_instance(rows, key)
    diffs = [m[(i, a)] - m[(i, b)] for (i, c) in m if c == a and (i, b) in m]
    if not diffs:
        return {"n_instances": 0, "mean_difference": None}
    mean = sum(diffs) / len(diffs)
    sd = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5 if len(diffs) > 1 else None
    return {"n_instances": len(diffs), "mean_difference": mean,
            "std_error": sd / len(diffs) ** 0.5 if sd is not None else None}


def aggregate(rows: list) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    agg = {}
    for cond, rs in by.items():
        def mean(key, rs=rs):
            vals = [float(r[key]) for r in rs if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        keys = sorted({k2 for r in rs for k2 in r if k2.startswith(("joint_", "alone_", "e", "evals_"))})
        agg[cond] = {"n": len(rs), **{k2: mean(k2) for k2 in keys},
                     "greedy_failures": sum(1 for r in rs if "evals_greedy" in r and r["evals_greedy"] is None)}
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--templates")
    ap.add_argument("--oracle-root")
    ap.add_argument("--lookahead-root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = []
    for inst_dir in sorted(p for p in Path(args.runs).iterdir() if p.is_dir()):
        if args.templates:
            truth = template_truth(Path(args.templates) / inst_dir.name)
        else:
            la = Path(args.lookahead_root) / f"{inst_dir.name}.json" if args.lookahead_root else None
            truth = miplib_truth(Path(args.oracle_root) / inst_dir.name / "oracle.json", la)
        if truth is None:
            continue
        for f in sorted(inst_dir.glob("*.json")):
            rows.append(score(json.loads(f.read_text(encoding="utf-8")), truth))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agg = aggregate(rows)
    agg["paired_joint_recall"] = {f"names_only_minus_{b}": paired(rows, "names_only", b)
                                  for b in ("permuted", "anonymized")}
    agg["samples_per_instance_condition"] = sorted({sum(1 for r in rows if (r["instance"], r["condition"]) == ic)
                                                    for ic in {(r["instance"], r["condition"]) for r in rows}})
    (out / "support_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "support_summary.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
