"""Result tables: prediction, paired differences, search cost, census, reliability and the sample of record, written

Paper: Section 6 and Appendix F: every result table.
as CSV files and summarized in RESULTS.md; with --large-models and --rich-models also the tables of llmdw.scoring.large_models
and llmdw.scoring.rich_models. No request is sent to a language model.

Sample of record
- the answer files under <work>/work/support_templates/<config> and <work>/work/support_miplib/<config> for the
  configurations named by --configs; the MIPLIB-NL instances are those of work/miplib_set (the scorable set)
- every planned cell (instance, presentation, sample) without an answer file counts as a failed answer: it
  predicts no joint support and no candidate. An unparseable answer is scored the same way (its file has no supports)

Prediction (analyze_support.score per answer): hit@1 and recall on instances with a true joint support (templates:
the classes clsp, ndp and cfl, also per class; MIPLIB-NL: instances with a certain joint group without linking
variables); the false-alarm rate on the assignment templates (gap); on the hidden-group templates (needle) the hit
rate (the first predicted joint support is the hidden group) against chance 1/C(k, w) and against the stricter level
1/(number of family sets of size w whose row counts equal those of the hidden group, as multisets). Every rate is the
mean over instances of the per-instance mean over samples, with a 95% percentile bootstrap interval that resamples
instances (10,000 resamples, seed 0). The lookahead detects a true joint support W when W is a zero-marginal group of
the lookahead values alone (positive at W, zero on every proper subset); its detection rate is the share of the true
joint supports it detects, per instance, averaged over the same instances.

Search cost (search_sim, on instances with a true joint support; templates: every class with one): evaluations
until a best set is evaluated for the random order (expectation), greedy selection (failures counted), the covering
order, the lookahead order, the robust staged search anchored at GCG's decomposition (one row per GCG baseline whose
decomposition is representable) and JointDW, the same robust staged search anchored, as in Theorem 2, at the first
joint support listed in the answer (at the families convexified by the top-ranked candidate when it lists none; per
sample, then the mean over the samples of an instance); the prediction error e uses the same anchor. The top-candidate
anchor gives alternative tables (search_cost_top_candidate, search_by_size_top_candidate, and
e_top_candidate in search_points). An answer without a joint support and without a candidate gives JointDW an empty
anchor, which leaves the covering order.

Relabeling (llmdw.search.relabel): every search order that depends on an order of the families (covering, greedy and
lookahead tie-breaking, the staged search and its fallback, the interleavings, the shortlist orders, the GCG anchors)
is run under 1,000 uniformly random orders of the families of each instance (one stream per instance, keyed by its
name), and the per-instance value is the mean over these orders; an instance fails when any order fails. The hit
tables of the lookahead order are expectations over the same orders (ties broken at random). The random order keeps
its closed form. --family-orders identity runs every order on the families sorted by name instead.

Reasoning (--pair: Grok 4.20 (Reasoning) minus Grok 4.20 (Non-Reasoning)): per phase and presentation, the paired
differences a minus b over instances of hit@1 and recall (the prediction instances) and of JointDW's evaluations (the
search-cost instances; per instance the mean over samples), with paired bootstrap intervals (instances resampled,
10,000 resamples, seed 0). `paired_configs.csv`.

Inputs under --work: work/ (instances, gain tables, lookahead values, answers), runs/llm_calls.jsonl and
runs/llm_retries.jsonl (call times and retries), and prompt_groups.csv, which gives the prompt groups when it is
present (otherwise they are computed from the prompts, which needs the MIPLIB-NL descriptions in
<work>/MIPLIB-NL/dataset).

Usage
-----
python -m llmdw.scoring.results --work data --census-corrections data/census_corrections.json \
    --large-models --rich-models --out results [--configs MODEL,...] [--family-orders 1000|identity]
"""
from __future__ import annotations

from llmdw.constants import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, FAMILY_ORDERS

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from llmdw.scoring import analyze_support, cluster, structure
from llmdw.prompts import counterfactual
from llmdw.search import relabel, search_sim
from llmdw.providers import settings, usage
from llmdw.instances import summaries
from llmdw.instances.families import read_model, summarize
from llmdw.instances.rich_models_build import NAME_PREFIX
from llmdw.providers.support_run import result_path

RICH_PREFIX = f"{NAME_PREFIX}_"            # instance names of the rich models, whose retries are not counted here
RICH_PHASE = "rich_models"                 # the phase label of the rich models in the tables

CONDITIONS = usage.CONDITIONS
SAMPLES = 3
TRUE_JOINT_CLASSES = ("clsp", "ndp", "cfl")
RESAMPLES = BOOTSTRAP_RESAMPLES


# ---------------------------------------------------------------- statistics

def boot_stat_se(values, stat=np.mean, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """(lo, hi, se): the 95% percentile bootstrap interval of stat(values) and the bootstrap standard error, the standard
    deviation (ddof = 1) of the same replicates. values: {instance: value} or (instance, value) pairs; prompt groups
    are resampled (llmdw.scoring.cluster), which is the instance bootstrap when every group is one instance."""
    return cluster.boot(values, stat, resamples, seed)


def boot_ci_se(values, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED):
    return boot_stat_se(values, np.mean, resamples, seed)


def boot_ci(values, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """95% percentile bootstrap interval of the mean (values with instance names; prompt groups resampled)."""
    lo, hi, _ = cluster.boot(values, np.mean, resamples, seed)
    return lo, hi


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ---------------------------------------------------------------- instances, truth, answers

def template_class(name: str) -> str:
    return name.split("_")[0]


def lookahead_joint_groups(la: dict, scale, tol: float) -> list:
    """Zero-marginal groups of size >= 2 of the lookahead values alone (base: the LP; without the full family set)."""
    tab = {frozenset(): {"gain": 0.0, "censored": False}}
    k = len(la["families"])
    for row in la["table"].values():
        if row.get("lookahead") is not None and len(row["families"]) < k:   # as structure.instance_tables
            tab[frozenset(row["families"])] = {"gain": row["lookahead"] / scale if scale else row["lookahead"],
                                               "censored": False}
    return [g["group"] for g in structure.zero_marginal_groups(tab, tol=tol) if g["size"] >= 2]


def needle_levels(summary: dict, hidden: frozenset) -> dict:
    rows = {f["family"]: f["rows"] for f in summary["constraint_families"]}
    fams, w = sorted(rows), len(hidden)
    target = sorted(rows[f] for f in hidden)
    matching = sum(1 for c in itertools.combinations(fams, w) if sorted(rows[f] for f in c) == target)
    return {"k": len(fams), "w": w, "chance": 1 / math.comb(len(fams), w), "strict_chance": 1 / matching,
            "matching_sets": matching}


def phase_prompts(work: Path, phase: str, instances: dict | None = None) -> dict:
    """{instance: {presentation: messages}}: the messages support_run sends (usage.planned_messages) in every
    presentation the phase uses. templates and miplib: the four presentations; large_models (no LLM call; the
    MIPLIB-NL presentations they would get); RICH_PHASE (the rich models): the original presentation, the only one
    sent."""
    out = {}
    if phase in ("templates", "miplib"):
        for n, inst in sorted(instances.items()):
            d = work / "work/templates" / n if phase == "templates" else work / "MIPLIB-NL/dataset" / n
            out[n] = {c: usage.planned_messages(inst["summary"], str(d), c) for c in CONDITIONS
                      if counterfactual.applicable(inst["summary"], c)}
    elif phase == "large_models":
        for n in [x.strip() for x in (work / "work/large_models/instances.txt").read_text().splitlines() if x.strip()]:
            summ = summarize(read_model(str(work / "work/mps" / f"{n}.mps.gz")))
            out[n] = {c: usage.planned_messages(summ, str(work / "MIPLIB-NL/dataset" / n), c) for c in CONDITIONS
                      if counterfactual.applicable(summ, c)}
    elif phase == RICH_PHASE:
        root = work / "work/rich_models"
        for n in [x for x in (root / "instances.txt").read_text().split() if x]:
            d = root / "instances" / n
            out[n] = {"original": usage.planned_messages(summarize(read_model(str(d / "model.mps"))), str(d),
                                                         "original")}
    return out


def use_prompt_groups(work: Path, phase: str, instances: dict | None = None) -> list:
    """The prompt groups of one phase, registered for the cluster bootstrap (llmdw.scoring.cluster), as the rows of
    prompt_groups.csv (phase, instance, group). They are read from <work>/prompt_groups.csv when that file exists
    (the released data, which do not include the MIPLIB-NL descriptions), and otherwise computed from the prompts."""
    table = work / "prompt_groups.csv"
    if table.exists():
        with open(table, newline="", encoding="utf-8") as fh:
            pg = {r["instance"]: r["group"] for r in csv.DictReader(fh) if r["phase"] == phase}
    else:
        pg = cluster.prompt_groups(phase_prompts(work, phase, instances))
    cluster.GROUPS.update({n: f"{phase}:{g}" for n, g in pg.items()})
    return [{"phase": phase, "instance": n, "group": pg[n]} for n in sorted(pg)]


def load_instances(work: Path, phase: str) -> dict:
    """{name: {"summary", "truth", "class", "lookahead_groups"}} of one phase."""
    out = {}
    if phase == "templates":
        for d in sorted(p for p in (work / "work/templates").iterdir() if (p / "ground_truth.json").exists()):
            truth = analyze_support.template_truth(d)
            gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
            la_path = work / "work/lookahead_templates" / f"{d.name}.json"
            la = json.loads(la_path.read_text(encoding="utf-8")) if la_path.exists() else None
            if la:
                truth["lookahead"] = sorted(((frozenset(r["families"]), r["lookahead"] or 0.0)
                                             for r in la["table"].values()), key=lambda kv: -kv[1])
            tol = 1e-6 * max(1.0, abs(float(Fraction(gt["lp_bound"]))))
            model = summaries.load(work, "templates", d.name, d / "model.mps")
            truth["family_order"] = relabel.file_order(model, truth["families"])
            out[d.name] = {"summary": model.summary, "truth": truth, "model": model,
                           "class": template_class(d.name),
                           "group_sources": {frozenset(k.split("|")): v for k, v in gt.get("group_sources", {}).items()},
                           "lookahead_groups": lookahead_joint_groups(la, None, tol) if la else None}
    else:
        for d in sorted(p for p in (work / "work/miplib_set").iterdir() if (p / "oracle.json").exists()):
            la_path = work / "work/lookahead" / f"{d.name}.json"
            truth = analyze_support.miplib_truth(d / "oracle.json", la_path)
            if truth is None:
                continue
            o = json.loads((d / "oracle.json").read_text(encoding="utf-8"))
            la = json.loads(la_path.read_text(encoding="utf-8")) if la_path.exists() else None
            scale = structure.instance_tables(o, None)["scale"]
            tol = 1e-4 if scale else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
            model = summaries.load(work, "miplib", d.name, work / "work/mps" / f"{d.name}.mps.gz")
            truth["family_order"] = relabel.file_order(model, truth["families"])
            out[d.name] = {"summary": model.summary, "model": model,
                           "truth": truth, "class": "miplib",
                           "group_sources": None,
                           "lookahead_groups": lookahead_joint_groups(la, scale, tol) if la else None}
    return out


def load_answers(run_dir: Path, instances: dict) -> list:
    """Every planned cell of one configuration and phase; a cell without an answer file is a failed answer."""
    recs = []
    for name, inst in instances.items():
        for cond in CONDITIONS:
            if not counterfactual.applicable(inst["summary"], cond):
                continue
            for s in range(SAMPLES):
                p = result_path(run_dir, name, cond, s)
                if p.exists():
                    rec = json.loads(p.read_text(encoding="utf-8"))
                    rec["status"] = "answered" if rec.get("answer") is not None else "unparseable"
                else:
                    rec = {"instance": name, "condition": cond, "sample": s, "supports": [], "candidates": [],
                           "family_info": {}, "status": "missing"}
                recs.append(rec)
    return recs


def score_all(recs: list, instances: dict) -> list:
    rows = []
    for rec in recs:
        truth = instances[rec["instance"]]["truth"]
        row = analyze_support.score(rec, truth)
        row["status"] = rec["status"]
        table = truth["table"]
        if table and max(table.values()) > 0:
            fams = truth["families"]
            P_top, P = anchor(rec, fams), support_anchor(rec, fams)
            best_sets = [T for T, v in table.items() if T and v >= max(table.values()) - 1e-9]
            row["e_top_candidate"] = row.get("e")
            row["e"] = min(len(B - P) for B in best_sets)                       # Theorem 2's anchor (default)
            s, n = len(fams) - 1, rec["instance"]
            row["evals_jointdw"] = run_value(search_runs(n, truth, s, ("model", P), full=True)[0])
            row["evals_jointdw_top_candidate"] = run_value(search_runs(n, truth, s, ("model", P_top), full=True)[0])
            if "evals_covering" in row:
                row["evals_covering"] = run_value(search_runs(n, truth, s, ("covering",), full=True)[0])
        rows.append(row)
    return rows


def per_instance(rows: list, key: str, cond: str, names) -> dict:
    acc = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r["instance"] in names and r.get(key) is not None:
            acc[r["instance"]].append(float(r[key]))
    return {i: sum(v) / len(v) for i, v in acc.items()}


# ---------------------------------------------------------------- tables

def prediction_tables(phase, config, rows, instances):
    pred, alarm, needle = [], [], []
    joint = {n for n, i in instances.items() if i["truth"]["joint"]
             and (phase == "miplib" or i["class"] in TRUE_JOINT_CLASSES)}
    groups = [("all", joint)] + ([(c, {n for n in joint if instances[n]["class"] == c}) for c in TRUE_JOINT_CLASSES]
                                 if phase == "templates" else [])
    for cond in CONDITIONS:
        for cls, names in groups:
            h = per_instance(rows, "joint_hit1", cond, names)
            r = per_instance(rows, "joint_recall", cond, names)
            pred.append({"phase": phase, "config": config, "condition": cond, "class": cls, "n_instances": len(r),
                         "hit1": _mean(h.values()), **_ci("hit1", h), "recall": _mean(r.values()), **_ci("recall", r)})
        if phase == "templates":
            gap = {n for n, i in instances.items() if i["class"] == "gap"}
            fa = per_instance(rows, "joint_false", cond, gap)
            alarm.append({"config": config, "condition": cond, "n_instances": len(fa), "false_alarm": _mean(fa.values()),
                          **_ci("false_alarm", fa)})
            nd = {n for n, i in instances.items() if i["class"] == "needle"}
            hit = per_instance(rows, "joint_hit1", cond, nd)
            lv = {n: needle_levels(instances[n]["summary"], instances[n]["truth"]["joint"][0]) for n in nd}
            for n in sorted(hit):
                needle.append({"config": config, "condition": cond, "instance": n, "k": lv[n]["k"], "w": lv[n]["w"],
                               "hit_rate": hit[n], "hit_lo": None, "hit_hi": None, "hit_se": None,
                               "chance": lv[n]["chance"],
                               "strict_chance": lv[n]["strict_chance"], "matching_sets": lv[n]["matching_sets"]})
            lo, hi, se = boot_ci_se(hit)
            needle.append({"config": config, "condition": cond, "instance": "all", "n_instances": len(hit), "k": None, "w": None,
                           "hit_rate": _mean(hit.values()), "hit_lo": lo, "hit_hi": hi, "hit_se": se,
                           "chance": _mean(lv[n]["chance"] for n in hit),
                           "strict_chance": _mean(lv[n]["strict_chance"] for n in hit), "matching_sets": None})
    return pred, alarm, needle


def _ci(key, d):
    lo, hi, se = boot_ci_se(d)
    return {f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_se": se}


def lookahead_detection(phase, instances):
    out = []
    joint = {n for n, i in instances.items() if i["truth"]["joint"]
             and (phase == "miplib" or i["class"] in TRUE_JOINT_CLASSES)}
    groups = [("all", joint)] + ([(c, {n for n in joint if instances[n]["class"] == c}) for c in TRUE_JOINT_CLASSES]
                                 if phase == "templates" else [])
    for cls, names in groups:
        det = {}
        for n in sorted(names):                    # sorted: the bootstrap must not depend on string hashing
            g = instances[n]["lookahead_groups"]
            if g is not None:
                tj = instances[n]["truth"]["joint"]
                det[n] = sum(1 for W in tj if W in g) / len(tj)
        lo, hi, se = boot_ci_se(det)
        out.append({"phase": phase, "class": cls, "n_instances": len(det), "n_without_lookahead": len(names) - len(det),
                    "detection_rate": _mean(det.values()), "detection_lo": lo, "detection_hi": hi,
                    "detection_se": se})
    return out


def paired_tables(phase, config, rows, instances):
    out = []
    joint = {n for n, i in instances.items() if i["truth"]["joint"]
             and (phase == "miplib" or i["class"] in TRUE_JOINT_CLASSES)}
    for a, b in (("names_only", "permuted"), ("names_only", "anonymized"), ("original", "anonymized")):
        ma = per_instance(rows, "joint_recall", a, joint)
        mb = per_instance(rows, "joint_recall", b, joint)
        pairs = [(i, ma[i] - mb[i]) for i in sorted(ma) if i in mb]
        diffs = [v for _, v in pairs]
        lo, hi, se = boot_ci_se(pairs)
        out.append({"phase": phase, "config": config, "contrast": f"{a} minus {b}", "n_instances": len(diffs),
                    "mean_difference": _mean(diffs), "lo": lo, "hi": hi, "se_difference": se})
    return out


def paired_configs(phase, a, b, rows_a, rows_b, instances):
    """a minus b per presentation, paired over instances: hit@1, recall and JointDW's evaluations."""
    out = []
    pred = {n for n, i in instances.items() if i["truth"]["joint"]
            and (phase == "miplib" or i["class"] in TRUE_JOINT_CLASSES)}
    search = {n for n, i in instances.items() if i["truth"]["joint"]}
    for cond in CONDITIONS:
        for key, names in (("joint_hit1", pred), ("joint_recall", pred), ("evals_jointdw", search)):
            ma, mb = per_instance(rows_a, key, cond, names), per_instance(rows_b, key, cond, names)
            both = sorted(set(ma) & set(mb))
            diffs = [ma[i] - mb[i] for i in both]
            lo, hi, se = boot_ci_se(list(zip(both, diffs)))
            out.append({"phase": phase, "contrast": f"{a} minus {b}", "condition": cond, "measure": key,
                        "n_instances": len(both), "mean_a": _mean(ma[i] for i in both),
                        "mean_b": _mean(mb[i] for i in both), "mean_difference": _mean(diffs), "lo": lo, "hi": hi,
                        "se_difference": se})
    return out


def baseline_search(phase, instances):
    """Search cost of the prediction-free orders and of the GCG anchors (no configuration)."""
    out, per = [], defaultdict(dict)
    names = [n for n, i in instances.items() if i["truth"]["joint"]]
    for n in names:
        t = instances[n]["truth"]
        table, s = t["table"], len(t["families"]) - 1
        if not table or max(table.values()) <= 0:
            continue
        per["random"][n] = search_sim.random_expected(table)
        per["greedy"][n] = run_value(search_runs(n, t, s, ("greedy",), full=True)[0])
        per["covering"][n] = run_value(search_runs(n, t, s, ("covering",), full=True)[0])
        if t.get("lookahead"):
            per["lookahead"][n] = run_value(search_runs(n, t, s, ("lookahead",), full=True)[0])
        for g, Pg in (t.get("gcg") or {}).items():
            per[f"gcg_anchor_{g}"][n] = run_value(search_runs(n, t, s, ("model", frozenset(Pg)), full=True)[0])
    for method, d in per.items():
        vals = [v for v in d.values() if v is not None]
        out.append({"phase": phase, "method": method, "config": "", "condition": "", "n_instances": len(d),
                    "n_failures": sum(1 for v in d.values() if v is None), "mean": _mean(vals), "median": _median(vals)})
    return out, per


def jointdw_search(phase, config, rows, instances):
    """JointDW rows (first-support anchor) and, separately, the top-candidate alternative; per-sample points."""
    out, alt, points = [], [], []
    names = {n for n, i in instances.items() if i["truth"]["joint"]}
    for cond in CONDITIONS:
        for key, method, dest in (("evals_jointdw", "jointdw", out),
                                  ("evals_jointdw_top_candidate", "jointdw_top_candidate", alt)):
            acc, fails = defaultdict(list), 0
            for r in rows:
                if r["condition"] != cond or r["instance"] not in names or key not in r:
                    continue
                if r[key] is None:
                    fails += 1
                else:
                    acc[r["instance"]].append(r[key])
                if key == "evals_jointdw":
                    points.append({"phase": phase, "config": config, "condition": cond, "instance": r["instance"],
                                   "sample": r["sample"], "k": len(instances[r["instance"]]["truth"]["families"]),
                                   "status": r["status"], "e": r.get("e"), "evals_jointdw": r["evals_jointdw"],
                                   "e_top_candidate": r.get("e_top_candidate"),
                                   "evals_jointdw_top_candidate": r.get("evals_jointdw_top_candidate"),
                                   "evals_covering": r.get("evals_covering")})
            means = [sum(v) / len(v) for v in acc.values()]
            dest.append({"phase": phase, "method": method, "config": config, "condition": cond,
                         "n_instances": len(acc), "n_failures": fails, "mean": _mean(means), "median": _median(means)})
    return out, alt, points


# ---------------------------------------------------------------- census

def census(work: Path):
    rows = json.loads((work / "work/structure/structure_rows.json").read_text(encoding="utf-8"))
    with_kind = {"certain": {r["instance"] for r in rows if r["joint_certain"]},
                 "possible": {r["instance"] for r in rows if r["joint_possible"]}}
    kind = {i: "certain" if i in with_kind["certain"] else "possible" if i in with_kind["possible"] else "none"
            for i in {r["instance"] for r in rows}}
    per_inst, pos_any, pos_oracle = [], set(), set()
    for p in sorted((work / "work/oracle").glob("*/oracle.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        la_path = work / "work/lookahead" / f"{p.parent.name}.json"
        la = json.loads(la_path.read_text(encoding="utf-8")) if la_path.exists() else None
        info = structure.instance_tables(o, la)
        tol = 1e-4 if info["scale"] else 1e-6 * max(1.0, abs(o.get("lp_bound") or 0.0))
        cells = [(T, v) for tab in info["tables"].values() for T, v in tab.items() if T]
        if any(v["gain"] > tol for _, v in cells):
            pos_any.add(p.parent.name)
        oracle_only = structure.instance_tables(o, None)["tables"].values()
        if any(v["gain"] > tol for tab in oracle_only for T, v in tab.items() if T):
            pos_oracle.add(p.parent.name)
        single = [v["gain"] for T, v in cells if len(T) == 1]
        per_inst.append({"instance": p.parent.name, "k": len(info["families"]),
                         "gains_as_share_of_lp_gap": info["scale"] is not None,
                         "max_single_family_gain": max(single) if single and info["scale"] else None,
                         "max_set_gain": max(v["gain"] for _, v in cells) if cells and info["scale"] else None,
                         "group_kind": kind.get(p.parent.name, "none")})
    shares = []
    for basis, pos in (("positive gain, oracle or lookahead", pos_any), ("positive gain, oracle only", pos_oracle)):
        for gk in ("certain", "possible"):
            k = len(pos & with_kind[gk])                     # an instance may have both kinds of group
            lo, hi = wilson(k, len(pos))
            shares.append({"group_kind": gk, "denominator": basis, "k": k, "n": len(pos),
                           "share": k / len(pos) if pos else None, "wilson_lo": lo, "wilson_hi": hi})
    return shares, per_inst


def census_figure(work: Path, corrections: dict) -> tuple[list, list]:
    """Per-instance gains for the census figure (largest single-family gain and largest set gain as shares of the LP
    gap) after corrections: {"optimum": {instance: value or null (no optimum)}, "drop_bounds": {instance: [subset keys]},
    "notes": {instance: text}}. Returns (rows, models left out with the reason)."""
    struct = json.loads((work / "work/structure/structure_rows.json").read_text(encoding="utf-8"))
    kind = {}
    for r in struct:
        k = "certain" if r["joint_certain"] else "possible" if r["joint_possible"] else "none"
        prev = kind.get(r["instance"], "none")
        kind[r["instance"]] = min((prev, k), key=["certain", "possible", "none"].index)
    opt_fix, drop, notes = corrections.get("optimum", {}), corrections.get("drop_bounds", {}), corrections.get("notes", {})
    rows, left = [], []
    for p in sorted((work / "work/oracle").glob("*/oracle.json")):
        n = p.parent.name
        o = json.loads(p.read_text(encoding="utf-8"))
        changes = []
        if n in opt_fix:
            changes.append(f"optimum {o.get('reference_optimal_value')} -> {opt_fix[n]}")
            o["reference_optimal_value"] = opt_fix[n]
        if n in drop:
            before = len(o.get("subsets") or [])
            o["subsets"] = [x for x in o.get("subsets") or [] if x.get("key") not in set(drop[n])]
            changes.append(f"{before - len(o['subsets'])} invalid bound(s) dropped")
        la_path = work / "work/lookahead" / f"{n}.json"
        la = json.loads(la_path.read_text(encoding="utf-8")) if la_path.exists() else None
        info = structure.instance_tables(o, la)
        cells = [(T, v) for tab in info["tables"].values() for T, v in tab.items() if T]
        if info["scale"] is None:
            why = ("reference optimum removed: " + notes[n]) if n in opt_fix and opt_fix[n] is None and n in notes \
                else "no reference optimum" if o.get("reference_optimal_value") is None else "LP bound equals the optimum"
            left.append({"instance": n, "reason": why})
            continue
        if not cells:
            left.append({"instance": n, "reason": "no oracle evaluation or lookahead value with a bound"})
            continue
        single = [v["gain"] for T, v in cells if len(T) == 1]
        rows.append({"instance": n, "k": len(info["families"]), "max_single_family_gain": max(single) if single else None,
                     "max_set_gain": max(v["gain"] for _, v in cells), "group_kind": kind.get(n, "none"),
                     "corrections": "; ".join(changes) + (f" ({notes[n]})" if n in notes and changes else "")})
    return rows, left


def census_counts(work: Path) -> list:
    with open(work / "work/mps/export_summary.csv", newline="", encoding="utf-8") as fh:
        exp = Counter(r["status"] for r in csv.DictReader(fh))
    with open(work / "work/oracle/oracle_plan.csv", newline="", encoding="utf-8") as fh:
        plan = list(csv.DictReader(fh))
    skipped = Counter()
    for r in plan:
        if r["planned"] == "True":
            continue
        m = re.match(r"constraint_families=(\d+)", r["reason"])
        if m:
            skipped["fewer than 2 families" if int(m.group(1)) < 2 else "more than 6 families"] += 1
        elif r["reason"].startswith("num_genconstrs"):
            skipped["general constraints"] += 1
        else:
            skipped[r["reason"] or "other"] += 1
    out = [{"count": f"export status {s}", "n": n} for s, n in sorted(exp.items())]
    out.append({"count": "exported (status ok)", "n": exp.get("ok", 0)})
    out.append({"count": "kept (oracle planned)", "n": sum(1 for r in plan if r["planned"] == "True")})
    out += [{"count": f"skipped: {k}", "n": n} for k, n in sorted(skipped.items())]
    out.append({"count": "oracle results (census)", "n": len(list((work / "work/oracle").glob("*/oracle.json")))})
    out.append({"count": "scorable (work/miplib_set)", "n": len(list((work / "work/miplib_set").glob("*/oracle.json")))})
    return out


# ---------------------------------------------------------------- reliability and answer times

def reliability(phase, config, recs, run_dir: Path, retry_rows: list, template_names: set):
    cells = defaultdict(dict)
    for r in recs:
        if r["status"] != "missing":
            sup = r.get("supports") or []
            cells[(r["instance"], r["condition"])][r["sample"]] = frozenset(sup[0]["families"]) if sup else None
    full = [v for v in cells.values() if len(v) == SAMPLES]
    agree = sum(1 for v in full if len(set(v.values())) == 1)
    answered = [r for r in recs if r["status"] != "missing"]
    usage_rows = [r.get("usage") or {} for r in answered]
    ret = Counter()
    for x in retry_rows:
        inst = (x.get("tag") or "").split(":")[1] if (x.get("tag") or "").startswith("support:") else None
        if (x.get("model") == config and inst is not None and not inst.startswith(RICH_PREFIX)
                and ((inst in template_names) == (phase == "templates"))):
            ret[x.get("reason") or "other"] += 1
    ident = usage.identical_samples(run_dir) if run_dir.exists() else {"identical": 0}
    return {"phase": phase, "config": config, "cells_planned": len(recs), "answers": len(answered),
            "cells_without_answer": sum(1 for r in recs if r["status"] == "missing"),
            "unparseable": sum(1 for r in recs if r["status"] == "unparseable"),
            "cells_with_three_answers": len(full), "first_support_agree": agree,
            "first_support_agree_share": agree / len(full) if full else None,
            "cells_with_identical_samples": ident["identical"],
            **{f"retries_{k}": ret.get(k, 0) for k in usage.RETRY_REASONS},
            "median_latency_s": _median(r.get("latency") for r in answered),
            "prompt_tokens": sum(int(u.get("prompt_tokens") or 0) for u in usage_rows),
            "completion_tokens": sum(int(u.get("completion_tokens") or 0) for u in usage_rows)}


def call_times(work: Path) -> dict:
    """{(tag, model, prompt tokens, completion tokens): [(cached, time)]} of the answered calls in
    runs/llm_calls.jsonl. An answer file's time is that of the call that produced it, not the time of the file; the
    token counts tell apart configurations that share a model and send identical prompts."""
    out = defaultdict(list)
    path = work / "runs/llm_calls.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        u = r.get("usage") or {}
        if "error" in r or not str(r.get("tag", "")).startswith("support:"):
            continue
        out[(r["tag"], r.get("model_requested"), u.get("prompt_tokens"), u.get("completion_tokens"))].append(
            (r.get("cached") is not False, r["time"]))
    return out


def answer_times(work: Path, phase: str, config: str, calls: dict | None = None) -> dict:
    """The times of the first and the last call that produced an answer file of one configuration and phase (the
    latest paid call of a cell, since a cell sent again replaces an earlier answer)."""
    d = work / f"work/support_{phase}/{config}"
    files = list(d.glob("*/*.json")) if d.exists() else []
    calls = call_times(work) if calls is None else calls
    times = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        u = r.get("usage") or {}
        tag = f"support:{r['instance']}:{r['condition']}:s{r.get('sample', 0)}"
        hits = calls.get((tag, r.get("model"), u.get("prompt_tokens"), u.get("completion_tokens")), [])
        paid = [t for cached, t in hits if not cached]
        if hits:
            times.append(max(paid) if paid else max(t for _, t in hits))
    times.sort()
    return {"first_answer": times[0].replace("T", " ") if times else None,
            "last_answer": times[-1].replace("T", " ") if times else None}


# ---------------------------------------------------------------- offline analyses (section 7)
#
# Lookahead detection (section 1): a true joint support W counts as detected when W is a zero-marginal group of the
# lookahead values alone: l(W) > tol and l(S) <= tol for every proper subset S of W (the LP itself has value 0), over
# the family sets the lookahead evaluated without the full family set; tol as in the census (1e-4 in closed-gap units
# when the optimum is known, else 1e-6 times max(1, |LP|)). Section 7 also treats the lookahead order like a model's
# answer: the evaluable sets sorted by lookahead value (ties: fewer families first, then by name).

SIZE_LIMITS = ("2", "3", "k-1")


def lookahead_order(truth: dict) -> list:
    """The lookahead order: sets with at most k - 1 families by decreasing lookahead value, values within
    relabel.TIE_TOL tied (ties: smaller, then name)."""
    k = len(truth["families"])
    la = [(T, v) for T, v in (truth.get("lookahead") or []) if T and len(T) < k]
    cls = relabel.tie_classes(v for _, v in la)
    return [T for T, _ in sorted(la, key=lambda tv: (cls[tv[1]], len(tv[0]), sorted(tv[0])))]


def lookahead_tops(name, truth, m) -> list:
    """The first m sets of the lookahead order under each family order of llmdw.search.relabel."""
    k = len(truth["families"])
    key = ("lookahead_tops", name, m)
    if key not in relabel._memo:
        relabel._memo[key] = [relabel.lookahead_list(truth.get("lookahead"), order, k - 1)[:m]
                              for order in relabel.family_orders(name, base_order(truth))]
    return relabel._memo[key]


def lookahead_hit1(name, truth) -> float:
    """Share of the family orders under which the lookahead order names a true joint support first."""
    return _mean(float(bool(top) and top[0] in truth["joint"]) for top in lookahead_tops(name, truth, 1))


def search_runs(name, truth, s, spec, goal_kind="max", times=None, full=False):
    """Evaluations and seconds of one search order on one instance under every family order of llmdw.search.relabel
    (two arrays over the orders; nan where an order reaches no target). spec: ("covering",), ("greedy",),
    ("lookahead",), ("model", P), ("interleaved", P), ("shortlist", C) or ("shortlist+lookahead", C). full: the
    whole table (the lookahead with every set) instead of the sets with at most s families."""
    fams = truth["families"]
    tab = truth["table"] if full else restricted(truth["table"], s)
    goal = targets(tab, goal_kind)
    la_size = len(fams) if full else s

    def la(order):
        return relabel.lookahead_list(truth.get("lookahead"), order, la_size)

    def fn(order):
        kind = spec[0]
        if kind == "greedy":
            return relabel.greedy_hit(tab, goal, order, times, EVAL_LIMIT_S)
        if kind == "covering":
            it = relabel.covering_iter(order, s)
        elif kind == "lookahead":
            it = la(order)
        elif kind == "model":
            it = relabel.model_iter(order, s, spec[1])
        elif kind == "interleaved":
            it = relabel.alternate(relabel.model_iter(order, s, spec[1]), la(order))
        elif kind == "shortlist":
            it = relabel.shortlist_iter(spec[1], order, s)
        elif kind == "shortlist+lookahead":
            it = relabel.alternate(relabel.shortlist_iter(spec[1], order, s), la(order))
        elif kind == "ranked":
            it = relabel.ranked_iter(order, s, spec[1])
        elif kind == "ranked+lookahead":
            it = relabel.alternate(relabel.ranked_iter(order, s, spec[1]), la(order))
        elif kind == "narrow":
            it = relabel.narrow_iter(truth.get("lookahead"), order, la_size, spec[1])
        elif kind == "narrow+lookahead":
            it = relabel.alternate(relabel.narrow_iter(truth.get("lookahead"), order, la_size, spec[1]), la(order))
        else:
            raise ValueError(spec)
        return relabel.first_hit(it, tab, goal, times, EVAL_LIMIT_S)
    return relabel.runs((name, s, full, goal_kind, spec, times is not None), name, base_order(truth), fn)


def base_order(truth) -> list:
    """The families in the order the permutations of llmdw.search.relabel apply to (their order in the model file)."""
    return truth.get("family_order") or sorted(truth["families"])


def cand_list(rec, fams) -> list:
    """L_cand: the convexified family sets of the answer's candidates, in the order given, without repeats."""
    F, out = frozenset(fams), []
    for c in rec.get("candidates") or []:
        T = F - frozenset(c.get("master_families") or [])
        if T and T not in out:
            out.append(T)
    return out


def vote_list(cell_recs, fams) -> list:
    """L_vote: the joint supports listed in the samples of one cell, by the number of samples that list them, then by
    their mean position (0 = first), then by name; the same list serves every sample of the cell."""
    F, count, pos = frozenset(fams), Counter(), defaultdict(list)
    for r in cell_recs:
        seen = []
        for x in r.get("supports") or []:
            T = frozenset(x.get("families") or []) & F
            if T and T not in seen:
                seen.append(T)
        for i, T in enumerate(seen):
            count[T] += 1
            pos[T].append(i)
    return sorted(count, key=lambda T: (-count[T], sum(pos[T]) / len(pos[T]), sorted(T)))


def union_sets(sets) -> frozenset:
    out = frozenset()
    for T in sets:
        out |= frozenset(T)
    return out


def ranked_and_narrow(phase, recs_by, instances, limits=SIZE_LIMITS, conditions=CONDITIONS):
    """Ranked lists (L_cand, L_vote) and "narrow then lookahead" (C_cand, C_sup) from the existing answers: per
    configuration, presentation, size limit and list: coverage@1, 3, 5 (share of samples whose first m sets of the
    list with at most s families contain a maximal-gain set or a superset of one), mean list length, and mean
    evaluations of the ranked-list order and its interleaving with the lookahead order (relabel.ranked_iter); per C:
    mean |C| / k, the share of samples whose C contains a maximal-gain set, and mean evaluations of "narrow then
    lookahead" and its interleaving (relabel.narrow_iter); next to lookahead, covering and JointDW; paired
    differences over instances. Per instance the mean over its samples; L_vote is one list per cell (a pilot, not an
    independent sample). Returns (ranked rows, ranked paired, narrow rows, narrow paired)."""
    rk_rows, rk_pair, nw_rows, nw_pair = [], [], [], []
    names = sorted(n for n, i in instances.items() if i["truth"]["joint"])

    def pair(out, base, A, B, a, b):
        both = [n for n in A if A[n] is not None and B.get(n) is not None]
        diffs = [A[n] - B[n] for n in both]
        lo, hi, se = boot_ci_se(list(zip(both, diffs)))
        out.append({**base, "contrast": f"{a} minus {b}", "n_instances": len(both),
                    "mean_a": _mean(A[n] for n in both), "mean_b": _mean(B[n] for n in both),
                    "mean_difference": _mean(diffs), "lo": lo, "hi": hi, "se_difference": se,
                    "a_better": sum(1 for x in diffs if x < 0), "b_better": sum(1 for x in diffs if x > 0)})

    def stat_cols(prefix, d):
        lo, hi, se = boot_stat_se(d, np.mean)
        vals = [v for v in d.values() if v is not None]
        return {f"{prefix}": _mean(vals), f"{prefix}_lo": lo, f"{prefix}_hi": hi, f"{prefix}_se": se}

    for limit in limits:
        for cfg, recs in recs_by.items():
            for cond in conditions:
                ev = defaultdict(dict)                       # method -> {instance: mean evaluations}
                cov = defaultdict(dict)                      # (list, m) -> {instance: share}
                length = defaultdict(list)                   # list -> lengths over samples
                csize = defaultdict(dict)                    # C -> {instance: mean |C| / k}
                cin = defaultdict(dict)                      # C -> {instance: share containing a best set}
                for n in names:
                    t = instances[n]["truth"]
                    fams, k = sorted(t["families"]), len(t["families"])
                    s = s_value(limit, k)
                    tab = restricted(t["table"], s)
                    if not tab or max(tab.values()) <= 0:
                        continue
                    goal = targets(tab, "max")
                    has_la = bool([T for T in lookahead_order(t) if len(T) <= s])
                    ev["lookahead"][n] = run_value(search_runs(n, t, s, ("lookahead",))[0]) if has_la else None
                    ev["covering"][n] = run_value(search_runs(n, t, s, ("covering",))[0])
                    cell = [r for r in recs if r["instance"] == n and r["condition"] == cond]
                    if not cell:
                        continue
                    vote = vote_list(cell, fams)
                    per, pc, ps, pi = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                    for r in cell:
                        per["jointdw"].append(run_value(search_runs(n, t, s, ("model", support_anchor(r, fams)))[0]))
                        for lname, L in (("cand", cand_list(r, fams)), ("vote", vote)):
                            length[lname].append(len(L))
                            Ls = [T for T in L if len(T) <= s]
                            for m in (1, 3, 5):
                                pc[(lname, m)].append(float(any(B <= T for T in Ls[:m] for B in goal)))
                            per[f"ranked_{lname}"].append(run_value(search_runs(n, t, s, ("ranked", tuple(L)))[0]))
                            if has_la:
                                per[f"ranked_{lname}+lookahead"].append(
                                    run_value(search_runs(n, t, s, ("ranked+lookahead", tuple(L)))[0]))
                        for cname, C in (("cand", union_sets(cand_list(r, fams))),
                                         ("sup", union_sets(frozenset(x.get("families") or []) & frozenset(fams)
                                                            for x in r.get("supports") or []))):
                            ps[cname].append(len(C) / k)
                            pi[cname].append(float(any(B <= C for B in goal)))
                            if has_la:
                                per[f"narrow_{cname}"].append(run_value(search_runs(n, t, s, ("narrow", C))[0]))
                                per[f"narrow_{cname}+lookahead"].append(
                                    run_value(search_runs(n, t, s, ("narrow+lookahead", C))[0]))
                    for m, vals in per.items():
                        ok = [v for v in vals if v is not None]
                        ev[m][n] = sum(ok) / len(ok) if ok else None
                    for key, vals in pc.items():
                        cov[key][n] = _mean(vals)
                    for key, vals in ps.items():
                        csize[key][n] = _mean(vals)
                    for key, vals in pi.items():
                        cin[key][n] = _mean(vals)
                base = {"phase": phase, "limit": limit, "config": cfg, "condition": cond}
                for lname, label in (("cand", "L_cand"), ("vote", "L_vote (pilot: one list per cell)")):
                    row = {**base, "list": label, "n_instances": len(ev[f"ranked_{lname}"]),
                           "mean_list_length": _mean(length[lname])}
                    for m in (1, 3, 5):
                        row.update(stat_cols(f"coverage_at_{m}", cov[(lname, m)]))
                    lab = {f"ranked_{lname}": "ranked", f"ranked_{lname}+lookahead": "ranked+lookahead",
                           "lookahead": "lookahead", "covering": "covering", "jointdw": "jointdw"}
                    for meth, mlabel in lab.items():
                        row.update(stat_cols(f"evals_{mlabel}", ev.get(meth, {})))
                    rk_rows.append(row)
                    for a in (f"ranked_{lname}", f"ranked_{lname}+lookahead"):
                        for b in ("lookahead", "covering"):
                            pair(rk_pair, {**base, "list": label}, ev.get(a, {}), ev.get(b, {}), lab[a], b)
                for cname, label in (("cand", "C_cand"), ("sup", "C_sup")):
                    row = {**base, "C": label, "n_instances": len(csize[cname]),
                           "mean_C_size_over_k": _mean(csize[cname].values()),
                           **stat_cols("share_C_contains_best", cin[cname])}
                    lab = {f"narrow_{cname}": "narrow", f"narrow_{cname}+lookahead": "narrow+lookahead",
                           "lookahead": "lookahead", "covering": "covering", "ranked_cand": "ranked_L_cand",
                           "jointdw": "jointdw"}
                    for meth, mlabel in lab.items():
                        row.update(stat_cols(f"evals_{mlabel}", ev.get(meth, {})))
                    nw_rows.append(row)
                    for a in (f"narrow_{cname}", f"narrow_{cname}+lookahead"):
                        pair(nw_pair, {**base, "C": label}, ev.get(a, {}), ev.get("lookahead", {}), lab[a], "lookahead")
    return rk_rows, rk_pair, nw_rows, nw_pair


RANKED_PHASES = {"templates": "templates", "miplib": "miplib", RICH_PHASE: "rich_models"}     # work/ranked_set/<dir>


def minimal_best(table: dict) -> set:
    goal = targets(table, "max")
    return {B for B in goal if not any(S < B for S in goal)}


def ranked_prompt_tables(work: Path, configs: list, phase_inst: dict, orig_recs: dict):
    """The ranked-groups prompt arm (prompts/user_ranked.txt; answers in work/support_ranked/<config>), per phase
    (templates, miplib, rich models), configuration and size limit (rich models: 2 and 3): with L_rank = the answer's ranked_groups in
    order (an answer whose ranked_groups do not parse gives an empty list), coverage@1, 3, 5, the mean rank of the
    first set of L_rank (with at most s families) that is a maximal-gain set or a superset of one (over samples that
    have one) with the share of such samples, and evaluations of the ranked-list order and its interleaving with the
    lookahead order; "narrow then lookahead" with C_rel = relevant_families when given, else the union of
    ranked_groups; next to lookahead, covering and JointDW with the original prompt (its original-
    presentation answers on the same instance). Paired differences over instances with the prompt-group bootstrap;
    the primary comparison ("primary" = True) is ranked+lookahead minus lookahead on miplib and the rich models at s = 2 and 3.
    Calibration: every ranked group with a stated probability, in bins of 0.2, against the share that is a minimal
    maximal-gain set of the whole table. Returns (rows, paired, calibration)."""
    rows, paired, calib = [], [], []
    root = work / "work/support_ranked"
    for phase, sub_dir in RANKED_PHASES.items():
        inst = phase_inst.get(phase) or {}
        sdir = work / "work/ranked_set" / sub_dir
        names = sorted(p.name for p in sdir.iterdir()) if sdir.exists() else []
        names = [n for n in names if n in inst]
        limits = ("2", "3") if phase == RICH_PHASE else SIZE_LIMITS
        for cfg in configs:
            d = root / cfg
            recs = [json.loads(p.read_text(encoding="utf-8")) for n in names for p in sorted((d / n).glob("original*.json"))
                    if not p.name.endswith(".part")] if d.exists() else []
            if not recs:
                continue
            bins = defaultdict(lambda: [0, 0.0, 0])            # bin -> [groups, sum of probabilities, hits]
            no_prob = 0
            for r in recs:
                best = minimal_best(inst[r["instance"]]["truth"]["table"])
                for g in r.get("ranked_groups") or []:
                    if g.get("probability") is None:
                        no_prob += 1
                        continue
                    b = min(int(g["probability"] / 0.2), 4)
                    bins[b][0] += 1
                    bins[b][1] += g["probability"]
                    bins[b][2] += frozenset(g["families"]) in best
            for b in range(5):
                n_g, ps, hits = bins[b]
                calib.append({"phase": phase, "config": cfg, "bin": f"[{0.2 * b:.1f}, {0.2 * (b + 1):.1f}{']' if b == 4 else ')'}",
                              "n_groups": n_g, "mean_probability": ps / n_g if n_g else None,
                              "hit_rate": hits / n_g if n_g else None, "groups_without_probability": no_prob})
            for limit in limits:
                ev, cov, frank, fhit = defaultdict(dict), defaultdict(dict), {}, {}
                csize, cin, length = {}, {}, []
                for n in names:
                    t = inst[n]["truth"]
                    fams, k = sorted(t["families"]), len(t["families"])
                    s = s_value(limit, k)
                    tab = restricted(t["table"], s)
                    if not tab or max(tab.values()) <= 0:
                        continue
                    goal = targets(tab, "max")
                    has_la = bool([T for T in lookahead_order(t) if len(T) <= s])
                    ev["lookahead"][n] = run_value(search_runs(n, t, s, ("lookahead",))[0]) if has_la else None
                    ev["covering"][n] = run_value(search_runs(n, t, s, ("covering",))[0])
                    jd = [run_value(search_runs(n, t, s, ("model", support_anchor(r, fams)))[0])
                          for r in orig_recs.get(phase, {}).get(cfg, []) if r["instance"] == n and r["condition"] == "original"]
                    jd = [v for v in jd if v is not None]
                    ev["jointdw_original_prompt"][n] = sum(jd) / len(jd) if jd else None
                    per, pc, rk, ps, pi = defaultdict(list), defaultdict(list), [], [], []
                    for r in (x for x in recs if x["instance"] == n):
                        L = [frozenset(g["families"]) for g in r.get("ranked_groups") or []]
                        length.append(len(L))
                        Ls = [T for T in L if len(T) <= s]
                        hit_at = next((i + 1 for i, T in enumerate(Ls) if any(B <= T for B in goal)), None)
                        for m in (1, 3, 5):
                            pc[m].append(float(hit_at is not None and hit_at <= m))
                        rk.append(hit_at)
                        per["ranked"].append(run_value(search_runs(n, t, s, ("ranked", tuple(L)))[0]))
                        rel = r.get("relevant_families")
                        C = (frozenset(rel) if rel is not None else union_sets(L)) & frozenset(fams)
                        ps.append(len(C) / k)
                        pi.append(float(any(B <= C for B in goal)))
                        if has_la:
                            per["ranked+lookahead"].append(
                                run_value(search_runs(n, t, s, ("ranked+lookahead", tuple(L)))[0]))
                            per["narrow_rel"].append(run_value(search_runs(n, t, s, ("narrow", C))[0]))
                            per["narrow_rel+lookahead"].append(run_value(search_runs(n, t, s, ("narrow+lookahead", C))[0]))
                    for m, vals in per.items():
                        ok = [v for v in vals if v is not None]
                        ev[m][n] = sum(ok) / len(ok) if ok else None
                    for m, vals in pc.items():
                        cov[m][n] = _mean(vals)
                    hits = [x for x in rk if x is not None]
                    frank[n] = _mean(hits) if hits else None
                    fhit[n] = len(hits) / len(rk) if rk else None
                    csize[n], cin[n] = _mean(ps), _mean(pi)
                base = {"phase": phase, "limit": limit, "config": cfg}
                row = {**base, "n_instances": len(ev["ranked"]), "mean_list_length": _mean(length),
                       "mean_rank_first_best": _mean(v for v in frank.values() if v is not None),
                       "share_with_best_in_list": _mean(v for v in fhit.values() if v is not None),
                       "mean_C_rel_size_over_k": _mean(csize.values()),
                       "share_C_rel_contains_best": _mean(cin.values())}
                for m in (1, 3, 5):
                    lo, hi, se = boot_stat_se(cov[m], np.mean)
                    row.update({f"coverage_at_{m}": _mean(v for v in cov[m].values() if v is not None),
                                f"coverage_at_{m}_lo": lo, f"coverage_at_{m}_hi": hi, f"coverage_at_{m}_se": se})
                for meth in ("ranked", "ranked+lookahead", "narrow_rel", "narrow_rel+lookahead", "lookahead",
                             "covering", "jointdw_original_prompt"):
                    d_ = ev.get(meth, {})
                    lo, hi, se = boot_stat_se(d_, np.mean)
                    row.update({f"evals_{meth}": _mean(v for v in d_.values() if v is not None), f"evals_{meth}_lo": lo,
                                f"evals_{meth}_hi": hi, f"evals_{meth}_se": se})
                rows.append(row)
                for a, b in (("ranked+lookahead", "lookahead"), ("ranked+lookahead", "covering"),
                             ("ranked+lookahead", "jointdw_original_prompt"), ("ranked", "lookahead"),
                             ("ranked", "covering"), ("ranked", "jointdw_original_prompt"),
                             ("narrow_rel", "lookahead"), ("narrow_rel+lookahead", "lookahead")):
                    A, B = ev.get(a, {}), ev.get(b, {})
                    both = [n for n in A if A[n] is not None and B.get(n) is not None]
                    diffs = [A[n] - B[n] for n in both]
                    lo, hi, se = boot_ci_se(list(zip(both, diffs)))
                    paired.append({**base, "contrast": f"{a} minus {b}",
                                   "primary": a == "ranked+lookahead" and b == "lookahead" and phase in ("miplib", RICH_PHASE)
                                   and limit in ("2", "3"), "n_instances": len(both),
                                   "mean_a": _mean(A[n] for n in both), "mean_b": _mean(B[n] for n in both),
                                   "mean_difference": _mean(diffs), "lo": lo, "hi": hi, "se_difference": se,
                                   "a_better": sum(1 for x in diffs if x < 0), "b_better": sum(1 for x in diffs if x > 0)})
    return rows, paired, calib


def run_value(arr):
    """Per-instance value of one order: the mean over the family orders; None (a failure) when any order fails."""
    arr = np.asarray(arr, dtype=float)
    return None if len(arr) == 0 or np.isnan(arr).any() else float(arr.mean())


def prediction_instances(phase, instances):
    return {n for n, i in instances.items() if i["truth"]["joint"]
            and (phase == "miplib" or i["class"] in TRUE_JOINT_CLASSES)}


def lookahead_like_models(phase, recs_by, instances, rows_by):
    """Hit@1 and recall over the top m sets of the lookahead order, m from each model's original answers."""
    out = []
    names = sorted(prediction_instances(phase, instances))
    for cfg, recs in recs_by.items():
        orig = [r for r in recs if r["condition"] == "original" and r["instance"] in instances]
        m_mean = _mean(len(r.get("supports") or []) for r in orig) or 0.0
        m = max(1, round(m_mean))
        hit = {n: lookahead_hit1(n, instances[n]["truth"]) for n in names}
        rec = {n: _mean(sum(1 for W in instances[n]["truth"]["joint"] if W in top) / len(instances[n]["truth"]["joint"])
                        for top in lookahead_tops(n, instances[n]["truth"], m)) for n in names}
        mh = per_instance(rows_by[cfg], "joint_hit1", "original", set(names))
        mr = per_instance(rows_by[cfg], "joint_recall", "original", set(names))
        out.append({"phase": phase, "config": cfg, "m_mean": m_mean, "m": m, "n_instances": len(names),
                    "lookahead_hit1": _mean(hit.values()), **_ci("lookahead_hit1", hit),
                    "lookahead_recall_at_m": _mean(rec.values()), **_ci("lookahead_recall_at_m", rec),
                    "model_hit1": _mean(mh.values()), "model_recall": _mean(mr.values())})
    return out


def boot_stat(values, stat, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """95% percentile bootstrap interval of stat (values with instance names; prompt groups resampled)."""
    lo, hi, _ = cluster.boot(values, stat, resamples, seed)
    return lo, hi


def restricted(table: dict, s: int) -> dict:
    return {T: v for T, v in table.items() if len(T) <= s}


def s_value(limit: str, k: int) -> int:
    """The size limit as used: 2, 3 or k - 1, never above k - 1 (the full set is not a decomposition)."""
    return k - 1 if limit == "k-1" else min(int(limit), k - 1)


def model_order(fams, s, P):
    """JointDW's order (search_sim.robust): the staged order around P interleaved with the covering order."""
    a, b = search_sim.staged_order(fams, s, P), search_sim.covering_order(fams, s)
    return [x for pair in itertools.zip_longest(a, b) for x in pair if x is not None]


def interleave(a, b):
    out, seen = [], set()
    for pair in itertools.zip_longest(a, b):
        for x in pair:
            if x is not None and x not in seen:
                seen.add(x)
                out.append(x)
    return out


def anchor(rec, fams):
    top = (rec.get("candidates") or [None])[0]
    return frozenset(fams) - frozenset(top["master_families"]) if top else frozenset()


def search_by_size(phase, recs_by, instances):
    """Section 3 for each size limit: per method the evaluations until a best set (among sets with at most s families)
    is evaluated; JointDW per sample, then the mean over samples. Also JointDW interleaved with the lookahead order."""
    rows, covers, inter = [], [], []
    names = sorted(n for n, i in instances.items() if i["truth"]["joint"])
    for limit in SIZE_LIMITS:
        per = defaultdict(dict)
        for n in names:
            t = instances[n]["truth"]
            fams, k = sorted(t["families"]), len(t["families"])
            s = s_value(limit, k)
            tab = restricted(t["table"], s)
            covers.append({"phase": phase, "instance": n, "k": k, "limit": limit, "s": s,
                           "covering_number": len(search_sim.covering_order(fams, s))})
            if not tab or max(tab.values()) <= 0:
                continue
            la = [T for T in lookahead_order(t) if len(T) <= s]
            per[("random", "")][n] = search_sim.random_expected(tab)
            per[("greedy", "")][n] = run_value(search_runs(n, t, s, ("greedy",))[0])
            per[("covering", "")][n] = run_value(search_runs(n, t, s, ("covering",))[0])
            if la:
                per[("lookahead", "")][n] = run_value(search_runs(n, t, s, ("lookahead",))[0])
            for g, Pg in (t.get("gcg") or {}).items():
                per[(f"gcg_anchor_{g}", "")][n] = run_value(search_runs(n, t, s, ("model", frozenset(Pg)))[0])
            for cfg, recs in recs_by.items():
                for cond in CONDITIONS:
                    for method, afun in (("jointdw", support_anchor), ("jointdw_top_candidate", anchor)):
                        vals = [run_value(search_runs(n, t, s, ("model", afun(r, fams)))[0]) for r in recs
                                if r["instance"] == n and r["condition"] == cond]
                        ok = [v for v in vals if v is not None]
                        per[(method, f"{cfg}|{cond}")][n] = sum(ok) / len(ok) if ok else None
                    if cond == "original" and la:
                        lav_runs = search_runs(n, t, s, ("lookahead",))[0]
                        lav = run_value(lav_runs)
                        both = []
                        for r in recs:
                            if r["instance"] == n and r["condition"] == cond:
                                P = support_anchor(r, fams)
                                iv_runs = search_runs(n, t, s, ("interleaved", P))[0]
                                mv_runs = search_runs(n, t, s, ("model", P))[0]
                                bound_runs = 2 * np.fmin(mv_runs, lav_runs)      # 2 min(model, lookahead) per order
                                within = bool(np.all(iv_runs <= bound_runs))    # False where a value is missing
                                both.append((run_value(iv_runs), run_value(mv_runs), run_value(bound_runs), within))
                        if both:
                            inter.append({"phase": phase, "limit": limit, "config": cfg, "instance": n, "s": s,
                                          "interleaved": _mean(b[0] for b in both), "model": _mean(b[1] for b in both),
                                          "lookahead": lav, "two_min": _mean(b[2] for b in both),
                                          "within_two_min": all(b[3] for b in both)})
        for (method, key), d in per.items():
            vals = [v for v in d.values() if v is not None]
            cfg, cond = key.split("|") if key else ("", "")
            mlo, mhi, mse = boot_stat_se(d, np.mean)
            dlo, dhi, dse = boot_stat_se(d, np.median)
            rows.append({"phase": phase, "limit": limit, "method": method, "config": cfg, "condition": cond,
                         "n_instances": len(d), "n_failures": sum(1 for v in d.values() if v is None),
                         "mean": _mean(vals), "mean_lo": mlo, "mean_hi": mhi, "mean_se": mse,
                         "median": _median(vals), "median_lo": dlo, "median_hi": dhi, "median_se": dse})
    summary = []
    for limit in SIZE_LIMITS:
        for cfg in recs_by:
            xs = [r for r in inter if r["limit"] == limit and r["config"] == cfg]
            if xs:
                summary.append({"phase": phase, "limit": limit, "config": cfg, "n_instances": len(xs),
                                "interleaved_mean": _mean(r["interleaved"] for r in xs),
                                "model_mean": _mean(r["model"] for r in xs),
                                "lookahead_mean": _mean(r["lookahead"] for r in xs),
                                "two_min_mean": _mean(r["two_min"] for r in xs),
                                "instances_within_two_min": sum(1 for r in xs if r["within_two_min"])})
    return rows, covers, inter, summary


def variable_disjoint(model, W) -> bool:
    """No column appears in rows of two different families of W (model: a summary of llmdw.instances.summaries)."""
    cols = {f: model.columns(f) for f in W}
    fs = sorted(W)
    return all(not (cols[a] & cols[b]) for i, a in enumerate(fs) for b in fs[i + 1:])


def support_kinds(instances, names) -> dict:
    """{(instance, W): {"source", "variable_disjoint"}} for the true joint supports."""
    out = {}
    for n in names:
        i = instances[n]
        for W in i["truth"]["joint"]:
            src = (i.get("group_sources") or {}).get(frozenset(W))
            out[(n, frozenset(W))] = {"source": src or "not determined",
                                      "variable_disjoint": variable_disjoint(i["model"], W)}
    return out


def complementarity(phase, rows_by, instances, kinds):
    """Per instance: does the lookahead rank a true support first, does each model (2 of 3 original samples)."""
    names = sorted(prediction_instances(phase, instances))
    la_hit = {n: lookahead_hit1(n, instances[n]["truth"]) for n in names}      # expectation over the family orders
    # expected counts, with one decimal when a tie at the top of the lookahead order makes the first set depend on
    # the family order on some instance
    fractional = any(len({top[0] if top else None for top in lookahead_tops(n, instances[n]["truth"], 1)}) > 1
                     for n in names)
    table, misses = [], []
    for cfg, rows in rows_by.items():
        hits = defaultdict(int)
        for r in rows:
            if r["condition"] == "original" and r["instance"] in la_hit and r.get("joint_hit1"):
                hits[r["instance"]] += 1
        c = Counter()
        for n in names:
            c[(hits[n] >= 2, True)] += la_hit[n]
            c[(hits[n] >= 2, False)] += 1 - la_hit[n]
        cnt = (lambda x: f"{x:.1f}") if fractional else (lambda x: int(round(x)))
        table.append({"phase": phase, "config": cfg, "n_instances": len(names), "both": cnt(c[(True, True)]),
                      "model_only": cnt(c[(True, False)]), "lookahead_only": cnt(c[(False, True)]),
                      "neither": cnt(c[(False, False)])})
    for n in names:
        if la_hit[n] < 1:
            joint = instances[n]["truth"]["joint"]
            firsts = sorted({"|".join(sorted(top[0])) if top else "" for top in lookahead_tops(n, instances[n]["truth"], 1)
                             if not (top and top[0] in joint)})
            for W in joint:
                k = kinds[(n, frozenset(W))]
                misses.append({"phase": phase, "instance": n, "true_support": "|".join(sorted(W)),
                               "source": k["source"], "variable_disjoint": k["variable_disjoint"],
                               "lookahead_first": " / ".join(firsts)})
    return table, misses


def by_group_kind(phase, recs_by, rows_by, instances, kinds):
    """Recall (per true support, mean over samples) and hit@1 (per instance) by kind of the true supports, original
    presentation; the lookahead with recall over its top m sets (m as in lookahead_like_models) and hit@1."""
    out = []
    names = sorted(prediction_instances(phase, instances))
    groups = [("source", lambda k: k["source"]),
              ("variable_disjoint", lambda k: "disjoint" if k["variable_disjoint"] else "sharing variables")]
    for dim, key in groups:
        for value in sorted({key(k) for (n, W), k in kinds.items() if n in names}):
            sup = [(n, W) for (n, W), k in kinds.items() if n in names and key(k) == value]
            inst = sorted({n for n, _ in sup if all(key(kinds[(n, frozenset(V))]) == value
                                                     for V in instances[n]["truth"]["joint"])})
            for cfg, recs in recs_by.items():
                orig = [r for r in recs if r["condition"] == "original"]
                m = max(1, round(_mean(len(r.get("supports") or []) for r in orig) or 0))
                rec_pairs = []
                for n, W in sup:
                    smp = [frozenset(W) in [frozenset(x["families"]) for x in (r.get("supports") or [])]
                           for r in orig if r["instance"] == n]
                    rec_pairs.append((n, _mean(map(float, smp))))
                rec_vals = [v for _, v in rec_pairs]
                h = per_instance(rows_by[cfg], "joint_hit1", "original", set(inst))
                la_rec = [_mean(float(W in top) for top in lookahead_tops(n, instances[n]["truth"], m)) for n, W in sup]
                la_hit = [lookahead_hit1(n, instances[n]["truth"]) for n in inst]
                out.append({"phase": phase, "dimension": dim, "kind": value, "config": cfg, "n_supports": len(sup),
                            "n_instances": len(inst), "model_recall": _mean(rec_vals), **_ci_list("model_recall", rec_pairs),
                            "model_hit1": _mean(h.values()), "lookahead_recall_at_m": _mean(la_rec), "m": m,
                            "lookahead_hit1": _mean(la_hit)})
    return out


# ---------------------------------------------------------------- section 9: Theorem 2 anchor and target
#
# Anchor: besides the families convexified by the top-ranked candidate (the anchor of sections 3 and 7), the first
# joint support listed in the answer (the top candidate's anchor when the answer lists none). Target: besides "a set of
# maximal gain" among the sets with at most s families, "a minimal set whose gain is at least 0.95 of that best gain"
# (no proper subset in the table reaches 0.95 of it); a search reaches the second target only by evaluating such a
# minimal set itself. Time (MIPLIB-NL): the recorded wall-clock seconds of each evaluated set, capped at the 120 s limit
# (censored and failed evaluations count at most 120 s); a set with a lookahead value only counts 120 s.

EVAL_LIMIT_S = 120.0


def support_anchor(rec, fams):
    sup = rec.get("supports") or []
    return frozenset(sup[0]["families"]) & frozenset(fams) if sup else anchor(rec, fams)


def targets(tab: dict, kind: str) -> set:
    best = max(tab.values())
    if kind == "max":
        return {T for T, v in tab.items() if T and v >= best - 1e-9}
    reach = {T for T, v in tab.items() if T and v >= 0.95 * best - 1e-12}
    return {T for T in reach if not any(S < T for S in reach)}


def first_hit_set(order, tab, goal):
    seen = set()
    for T in order:
        if T in seen or T not in tab:
            continue
        seen.add(T)
        if T in goal:
            return len(seen), seen
    return None, seen


def greedy_hit(tab, goal):
    fams = sorted(set().union(*tab))
    cur, seen = frozenset(), []
    while True:
        nbrs = [cur | {f} for f in fams if f not in cur and (cur | {f}) in tab]
        for T in nbrs:
            if T not in seen:
                seen.append(T)
        if any(T in goal for T in nbrs):                # counted like search_sim.greedy: the whole step
            return len(seen), set(seen)
        if not nbrs:
            return None, set(seen)
        nxt = max(nbrs, key=lambda T: tab[T])
        if tab[nxt] <= tab[cur] + 1e-9:
            return None, set(seen)
        cur = nxt


def oracle_times(work: Path, name: str) -> dict:
    """{frozenset of convexified families: seconds} of the evaluations without linking variables."""
    o = json.loads((work / "work/oracle" / name / "oracle.json").read_text(encoding="utf-8"))
    fams = set(o["families"])
    out = {}
    for s in o.get("subsets") or []:
        if s.get("linking_var_families"):
            continue
        t = s.get("wall_seconds") if s.get("wall_seconds") is not None else s.get("solving_time")
        out[frozenset(fams - set(s.get("master_families") or []))] = min(EVAL_LIMIT_S, t if t is not None else EVAL_LIMIT_S)
    return out


def theorem2(phase, recs_by, instances, work: Path):
    """Section 9 tables: anchors (e = 0 shares), search cost and time to both targets, target sizes per instance."""
    anchor_rows, cost, sizes = [], [], []
    names = sorted(n for n, i in instances.items() if i["truth"]["joint"])
    # 1. anchors
    for cfg, recs in recs_by.items():
        for cond in CONDITIONS:
            n_s, e0_top, e0_sup, differ = 0, 0, 0, 0
            for r in recs:
                if r["condition"] != cond or r["instance"] not in names:
                    continue
                t = instances[r["instance"]]["truth"]
                tab = t["table"]
                if not tab or max(tab.values()) <= 0:
                    continue
                best = [T for T in targets(tab, "max")]
                Pt, Ps = anchor(r, t["families"]), support_anchor(r, t["families"])
                n_s += 1
                e0_top += min(len(B - Pt) for B in best) == 0
                e0_sup += min(len(B - Ps) for B in best) == 0
                differ += Pt != Ps
            anchor_rows.append({"phase": phase, "config": cfg, "condition": cond, "samples": n_s,
                                "e0_share_top_candidate": e0_top / n_s if n_s else None,
                                "e0_share_first_support": e0_sup / n_s if n_s else None,
                                "anchors_differ_share": differ / n_s if n_s else None})
    # 2.-4. search cost and time
    for limit in SIZE_LIMITS:
        for goal_kind in ("max", "min95"):
            per, tper = defaultdict(dict), defaultdict(dict)
            for n in names:
                t = instances[n]["truth"]
                fams, k = sorted(t["families"]), len(t["families"])
                s = s_value(limit, k)
                tab = restricted(t["table"], s)
                if not tab or max(tab.values()) <= 0:
                    continue
                goal = targets(tab, goal_kind)
                if goal_kind == "max":
                    mx = targets(tab, "max")
                    m95 = targets(tab, "min95")
                    sizes.append({"phase": phase, "limit": limit, "instance": n, "k": k, "s": s,
                                  "max_sets": len(mx), "max_set_size_min": min(map(len, mx)),
                                  "max_set_size_max": max(map(len, mx)), "min95_sets": len(m95),
                                  "min95_size_min": min(map(len, m95)), "min95_size_max": max(map(len, m95))})
                times = oracle_times(work, n) if phase == "miplib" else None
                tm = (lambda T: times.get(T, EVAL_LIMIT_S)) if times is not None else None
                evaluable = [T for T in tab if T]
                M, N = len(goal), len(evaluable)
                per[("random", "", "")][n] = (N + 1) / (M + 1)
                if tm:
                    tper[("random", "", "")][n] = (relabel.seconds(times, [T for T in evaluable if T not in goal],
                                                                   EVAL_LIMIT_S) / (M + 1)
                                                    + relabel.seconds(times, goal, EVAL_LIMIT_S) / M)
                specs = [("covering", "", "")] + ([("lookahead", "", "")] if lookahead_order(t) and
                                                  [T for T in lookahead_order(t) if len(T) <= s] else [])
                specs.append(("greedy", "", ""))
                for key in specs:
                    ev, sec = search_runs(n, t, s, (key[0],), goal_kind, times)
                    per[key][n] = run_value(ev)
                    if tm:
                        tper[key][n] = run_value(sec) if per[key][n] is not None else None
                for cfg, recs in recs_by.items():
                    for cond in CONDITIONS:
                        for aname, afun in (("top_candidate", anchor), ("first_support", support_anchor)):
                            cs, ts = [], []
                            for r in recs:
                                if r["instance"] == n and r["condition"] == cond:
                                    ev, sec = search_runs(n, t, s, ("model", afun(r, fams)), goal_kind, times)
                                    c = run_value(ev)
                                    if c is not None:
                                        cs.append(c)
                                        if tm:
                                            ts.append(run_value(sec))
                            key = ("jointdw", f"{cfg}|{cond}", aname)
                            per[key][n] = sum(cs) / len(cs) if cs else None
                            if tm:
                                tper[key][n] = sum(ts) / len(ts) if ts else None
            for (method, cc, aname), d in per.items():
                cfg, cond = cc.split("|") if cc else ("", "")
                vals = [v for v in d.values() if v is not None]
                mlo, mhi, mse = boot_stat_se(d, np.mean)
                dlo, dhi, dse = boot_stat_se(d, np.median)
                row = {"phase": phase, "limit": limit, "target": goal_kind, "method": method, "anchor": aname,
                       "config": cfg, "condition": cond, "n_instances": len(d),
                       "n_failures": sum(1 for v in d.values() if v is None), "mean": _mean(vals), "mean_lo": mlo,
                       "mean_hi": mhi, "mean_se": mse, "median": _median(vals), "median_lo": dlo, "median_hi": dhi,
                       "median_se": dse}
                if phase == "miplib":
                    tv = [v for v in tper[(method, cc, aname)].values() if v is not None]
                    tlo, thi, tse = boot_stat_se(tper[(method, cc, aname)], np.mean)
                    row.update({"seconds_mean": _mean(tv), "seconds_lo": tlo, "seconds_hi": thi, "seconds_se": tse,
                                "seconds_median": _median(tv)})
                cost.append(row)
    return anchor_rows, cost, sizes


def paired_search(phase, recs_by, instances, work: Path):
    """Paired over instances (maximal-gain target, first-support anchor, original presentation): JointDW and JointDW
    interleaved with the lookahead order, each minus the covering order and minus the lookahead order, for every s;
    in evaluations and, for MIPLIB-NL, in seconds (oracle times capped at 120 s, as in section 9)."""
    out = []
    names = sorted(n for n, i in instances.items() if i["truth"]["joint"])
    for limit in SIZE_LIMITS:
        for cfg, recs in recs_by.items():
            vals = defaultdict(dict)                     # (method, unit) -> {instance: value}
            for n in names:
                t = instances[n]["truth"]
                fams, k = sorted(t["families"]), len(t["families"])
                s = s_value(limit, k)
                tab = restricted(t["table"], s)
                if not tab or max(tab.values()) <= 0:
                    continue
                goal = targets(tab, "max")
                times = oracle_times(work, n) if phase == "miplib" else None
                secs = (lambda seen: relabel.seconds(times, seen, EVAL_LIMIT_S)) if times is not None else None
                for m in ("covering", "lookahead"):
                    ev, sec = search_runs(n, t, s, (m,), "max", times)
                    c = run_value(ev)
                    vals[(m, "evaluations")][n] = c
                    if secs:
                        vals[(m, "seconds")][n] = run_value(sec) if c is not None else None
                for m in ("jointdw", "interleaved"):
                    cs, ts = [], []
                    for r in recs:
                        if r["instance"] != n or r["condition"] != "original":
                            continue
                        P = support_anchor(r, fams)
                        ev, sec = search_runs(n, t, s, ("model" if m == "jointdw" else "interleaved", P), "max", times)
                        c = run_value(ev)
                        if c is not None:
                            cs.append(c)
                            if secs:
                                ts.append(run_value(sec))
                    vals[(m, "evaluations")][n] = sum(cs) / len(cs) if cs else None
                    if secs:
                        vals[(m, "seconds")][n] = sum(ts) / len(ts) if ts else None
            for unit in ("evaluations", "seconds") if phase == "miplib" else ("evaluations",):
                for a in ("jointdw", "interleaved"):
                    for b in ("covering", "lookahead"):
                        A, B = vals[(a, unit)], vals[(b, unit)]
                        both = [n for n in A if A[n] is not None and B.get(n) is not None]
                        diffs = [A[n] - B[n] for n in both]
                        lo, hi, se = boot_ci_se(list(zip(both, diffs)))
                        out.append({"phase": phase, "limit": limit, "config": cfg, "unit": unit,
                                    "contrast": f"{a} minus {b}", "n_instances": len(both),
                                    "mean_a": _mean(A[n] for n in both), "mean_b": _mean(B[n] for n in both),
                                    "mean_difference": _mean(diffs), "lo": lo, "hi": hi, "se_difference": se,
                                    "a_better": sum(1 for d in diffs if d < 0), "b_better": sum(1 for d in diffs if d > 0)})
    return out


def _ci_list(key, pairs):
    """Interval and SE of the mean of (instance, value) pairs (several values may share an instance)."""
    lo, hi, se = boot_ci_se(pairs)
    return {f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_se": se}


# Section 10: shortlist anchors. S1, S2, S3 are the unions of the families of the
# first 1, 2 and 3 joint supports listed in the answer (all of them when it lists fewer), Sall the union of all listed
# joint supports; an answer that lists none uses the default anchor (support_anchor) for every shortlist.
SHORTLISTS = ("S1", "S2", "S3", "Sall")


def shortlists(rec, fams) -> dict:
    F = frozenset(fams)
    sup = [frozenset(x.get("families") or []) & F for x in rec.get("supports") or []]
    if not sup:
        a = support_anchor(rec, fams)
        return {name: a for name in SHORTLISTS}
    union = lambda xs: frozenset().union(*xs)
    return {"S1": union(sup[:1]), "S2": union(sup[:2]), "S3": union(sup[:3]), "Sall": union(sup)}


def shortlist_order(C, fams, s) -> list:
    """The covering order of the sets of at most s families inside C, then the covering order of all families,
    without repeats."""
    C = sorted(frozenset(C) & frozenset(fams))
    inner = search_sim.covering_order(C, min(s, len(C))) if C else []
    out, seen = [], set()
    for T in inner + search_sim.covering_order(fams, s):
        if T not in seen:
            seen.add(T)
            out.append(T)
    return out


def shortlist_search(phase, recs_by, instances, limits=SIZE_LIMITS, conditions=CONDITIONS):
    """Section 10: per configuration, presentation and size limit, the shortlist orders (alone and interleaved with
    the lookahead order) next to JointDW and the lookahead order: mean shortlist size, share of samples whose best
    set lies inside the shortlist, evaluations until a set of maximal gain (per sample, then the mean over the samples
    of an instance), and paired differences over instances against JointDW and against the lookahead order."""
    rows, paired = [], []
    names = sorted(n for n, i in instances.items() if i["truth"]["joint"])
    methods = [m for C in SHORTLISTS for m in (C, f"{C}+lookahead")]
    for limit in limits:
        for cfg, recs in recs_by.items():
            for cond in conditions:
                ev = defaultdict(dict)                      # method -> {instance: mean evaluations}
                size = defaultdict(list)                    # shortlist -> sizes over samples
                inside = defaultdict(list)                  # shortlist -> best set inside, over samples
                for n in names:
                    t = instances[n]["truth"]
                    fams, k = sorted(t["families"]), len(t["families"])
                    s = s_value(limit, k)
                    tab = restricted(t["table"], s)
                    if not tab or max(tab.values()) <= 0:
                        continue
                    goal = targets(tab, "max")
                    la = [T for T in lookahead_order(t) if len(T) <= s]
                    ev["lookahead"][n] = run_value(search_runs(n, t, s, ("lookahead",))[0]) if la else None
                    per = defaultdict(list)
                    for r in recs:
                        if r["instance"] != n or r["condition"] != cond:
                            continue
                        per["jointdw"].append(run_value(search_runs(n, t, s, ("model", support_anchor(r, fams)))[0]))
                        for C, members in shortlists(r, fams).items():
                            size[C].append(len(members))
                            inside[C].append(any(B <= members for B in goal))
                            per[C].append(run_value(search_runs(n, t, s, ("shortlist", frozenset(members)))[0]))
                            per[f"{C}+lookahead"].append(
                                run_value(search_runs(n, t, s, ("shortlist+lookahead", frozenset(members)))[0]))
                    for m, vals in per.items():
                        ok = [v for v in vals if v is not None]
                        ev[m][n] = sum(ok) / len(ok) if ok else None
                for m in ["jointdw", "lookahead"] + methods:
                    d = ev.get(m, {})
                    vals = [v for v in d.values() if v is not None]
                    lo, hi, se = boot_stat_se(d, np.mean)
                    C = m.split("+")[0]
                    rows.append({"phase": phase, "limit": limit, "config": cfg, "condition": cond, "method": m,
                                 "n_instances": len(d), "n_failures": sum(1 for v in d.values() if v is None),
                                 "mean_size": _mean(size[C]) if C in SHORTLISTS else None,
                                 "best_inside_share": _mean(map(float, inside[C])) if C in SHORTLISTS else None,
                                 "samples": len(inside[C]) if C in SHORTLISTS else None,
                                 "mean": _mean(vals), "mean_lo": lo, "mean_hi": hi, "mean_se": se,
                                 "median": _median(vals)})
                for m in methods:
                    for b in ("jointdw", "lookahead"):
                        A, B = ev.get(m, {}), ev.get(b, {})
                        both = [n for n in A if A[n] is not None and B.get(n) is not None]
                        diffs = [A[n] - B[n] for n in both]
                        lo, hi, se = boot_ci_se(list(zip(both, diffs)))
                        paired.append({"phase": phase, "limit": limit, "config": cfg, "condition": cond,
                                       "contrast": f"{m} minus {b}", "n_instances": len(both),
                                       "mean_a": _mean(A[n] for n in both), "mean_b": _mean(B[n] for n in both),
                                       "mean_difference": _mean(diffs), "lo": lo, "hi": hi, "se_difference": se,
                                       "a_better": sum(1 for x in diffs if x < 0),
                                       "b_better": sum(1 for x in diffs if x > 0)})
    return rows, paired


# ---------------------------------------------------------------- output

NAMES: dict = {}                           # model identifier -> official name (set by main from the settings)
COLUMN_NAMES = {"condition": "presentation"}


def public_row(r: dict) -> dict:
    """A row as written: official names of the configurations, "presentation" for the presentation column."""
    out = {}
    for k, v in r.items():
        if isinstance(v, str) and NAMES:
            if k == "config" and v in NAMES:
                v = NAMES[v]
            elif k == "contrast":
                for m in sorted(NAMES, key=len, reverse=True):
                    v = v.replace(m, NAMES[m])
        out[COLUMN_NAMES.get(k, k)] = v
    return out


def write_csv(path: Path, rows: list):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    rows = [public_row(r) for r in rows]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def md_table(rows: list, keys: list) -> str:
    def f(v):
        if v is None:
            return ""
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    rows = [public_row(r) for r in rows]
    keys = [COLUMN_NAMES.get(k, k) for k in keys]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(f(r.get(k)) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the work root (holds work/, runs/ and llm_cache/)")
    ap.add_argument("--configs", help="model identifiers of the configurations, comma-separated (default: every "
                                      "configuration of --settings, in the order of the paper)")
    ap.add_argument("--pair", default=",".join(settings.pair()),
                    help="two configurations compared per instance (a minus b)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default=str(settings.DEFAULT),
                    help="the configurations (config/public_settings.json): official names and request settings")
    ap.add_argument("--large-models", action="store_true",
                    help="add the section on models with more than 6 families (llmdw.scoring.large_models)")
    ap.add_argument("--rich-models", action="store_true",
                    help="add the rich models (llmdw.scoring.rich_models): their tables for both prompts and their rows of "
                         "the answer-based tables")
    ap.add_argument("--census-corrections", help="JSON of corrections for the census figure data (census_figure)")
    ap.add_argument("--family-orders", default=str(FAMILY_ORDERS),
                    help="family orders per instance (llmdw.search.relabel; the first n of the stream), or 'identity' for "
                         "the families sorted by name only (validation)")
    args = ap.parse_args(argv)
    if args.family_orders == "identity":
        relabel.configure(identity=True)
    else:
        relabel.configure(int(args.family_orders))
    work, out = Path(args.work).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from llmdw.scoring import results as imported          # the module that llmdw.scoring.rich_models imports
    for names in (NAMES, imported.NAMES):
        names.clear()
        names.update(settings.names(args.settings))
    configs = [c for c in args.configs.split(",") if c] if args.configs else list(NAMES)
    retry_path = work / "runs/llm_retries.jsonl"
    retry_rows = [json.loads(ln) for ln in retry_path.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if retry_path.exists() else []
    T = {"prediction": [], "false_alarm": [], "needle": [], "lookahead_detection": [], "paired": [],
         "search_cost": [], "search_points": [], "reliability": [], "paired_configs": [],
         "lookahead_like_models": [], "search_by_size": [], "covering_numbers": [], "interleaved": [],
         "interleaved_summary": [], "complementarity": [], "lookahead_misses": [], "by_group_kind": [],
         "theorem2_anchors": [], "theorem2_search": [], "theorem2_target_sizes": [], "search_cost_top_candidate": [],
         "search_by_size_top_candidate": [], "paired_search": [], "census_figure": [], "census_figure_left_out": [],
         "shortlist_search": [], "shortlist_paired": [], "prompt_groups": [], "ranked_lists": [],
         "ranked_lists_paired": [], "narrow_lookahead": [], "narrow_lookahead_paired": [], "ranked_prompt": [],
         "ranked_prompt_paired": [], "ranked_prompt_calibration": []}
    phase_inst, orig_recs = {}, {}
    cluster.set_groups({})
    pair = [c for c in args.pair.split(",") if c] if args.pair else []
    pair = pair if len(pair) == 2 and all(c in configs for c in pair) else []
    template_names, times = set(), {}
    calls = call_times(work)
    for phase in ("templates", "miplib"):
        instances = load_instances(work, phase)
        T["prompt_groups"] += use_prompt_groups(work, phase, instances)
        if phase == "templates":
            template_names = set(instances)
        T["lookahead_detection"] += lookahead_detection(phase, instances)
        base, _ = baseline_search(phase, instances)
        T["search_cost"] += base
        rows_by, recs_by = {}, {}
        for cfg in configs:
            run_dir = work / f"work/support_{phase}/{cfg}"
            recs = load_answers(run_dir, instances)
            rows = score_all(recs, instances)
            rows_by[cfg], recs_by[cfg] = rows, recs
            p, a, nd = prediction_tables(phase, cfg, rows, instances)
            T["prediction"] += p
            T["false_alarm"] += a
            T["needle"] += nd
            T["paired"] += paired_tables(phase, cfg, rows, instances)
            jd, jd_top, pts = jointdw_search(phase, cfg, rows, instances)
            T["search_cost"] += jd
            T["search_cost_top_candidate"] += jd_top
            T["search_points"] += pts
            T["reliability"].append(reliability(phase, cfg, recs, run_dir, retry_rows, template_names))
            times[(phase, cfg)] = answer_times(work, phase, cfg, calls)
        if pair:
            T["paired_configs"] += paired_configs(phase, pair[0], pair[1], rows_by[pair[0]], rows_by[pair[1]], instances)
        T["lookahead_like_models"] += lookahead_like_models(phase, recs_by, instances, rows_by)
        sz, cov, inter, inter_sum = search_by_size(phase, recs_by, instances)
        T["search_by_size"] += [r for r in sz if r["method"] != "jointdw_top_candidate"]
        T["search_by_size_top_candidate"] += [r for r in sz if r["method"] == "jointdw_top_candidate"]
        T["paired_search"] += paired_search(phase, recs_by, instances, work)
        T["covering_numbers"] += cov
        T["interleaved"] += inter
        T["interleaved_summary"] += inter_sum
        kinds = support_kinds(instances, sorted(prediction_instances(phase, instances)))
        comp, miss = complementarity(phase, rows_by, instances, kinds)
        T["complementarity"] += comp
        T["lookahead_misses"] += miss
        T["by_group_kind"] += by_group_kind(phase, recs_by, rows_by, instances, kinds)
        an, co, sz = theorem2(phase, recs_by, instances, work)
        T["theorem2_anchors"] += an
        T["theorem2_search"] += co
        T["theorem2_target_sizes"] += sz
        sl, slp = shortlist_search(phase, recs_by, instances)
        T["shortlist_search"] += sl
        T["shortlist_paired"] += slp
        phase_inst[phase] = instances
        orig_recs[phase] = {c: [r for r in recs if r["condition"] == "original"] for c, recs in recs_by.items()}
        rk, rkp, nw, nwp = ranked_and_narrow(phase, recs_by, instances)
        T["ranked_lists"] += rk
        T["ranked_lists_paired"] += rkp
        T["narrow_lookahead"] += nw
        T["narrow_lookahead_paired"] += nwp
    shares, per_inst = census(work)
    T["census_shares"], T["census_instances"], T["census_counts"] = shares, per_inst, census_counts(work)
    corr = json.loads(Path(args.census_corrections).read_text(encoding="utf-8")) if args.census_corrections else {}
    T["census_figure"], T["census_figure_left_out"] = census_figure(work, corr)
    T["templates_table"] = templates_table(work)
    T["settings"] = [r for r in settings.table_rows(args.settings) if r["config"] in configs]
    T["sample_of_record"] = sample_of_record(T, configs, times)
    if args.large_models:
        T["prompt_groups"] += use_prompt_groups(work, "large_models")
    if args.rich_models:
        from llmdw.scoring import rich_models
        T["prompt_groups"] += use_prompt_groups(work, rich_models.PHASE)
        r_inst, r_recs = rich_models.llm_context(work, configs)
        phase_inst[RICH_PHASE], orig_recs[RICH_PHASE] = r_inst, r_recs
        if r_recs:                           # rich-model rows (original presentation, s = 2 and 3)
            sl, slp = shortlist_search(RICH_PHASE, r_recs, r_inst, limits=("2", "3"), conditions=("original",))
            T["shortlist_search"] += sl
            T["shortlist_paired"] += slp
            rk, rkp, nw, nwp = ranked_and_narrow(RICH_PHASE, r_recs, r_inst, limits=("2", "3"), conditions=("original",))
            T["ranked_lists"] += rk
            T["ranked_lists_paired"] += rkp
            T["narrow_lookahead"] += nw
            T["narrow_lookahead_paired"] += nwp
    if (work / "work/support_ranked").exists():
        rp, rpp, rpc = ranked_prompt_tables(work, configs, phase_inst, orig_recs)
        T["ranked_prompt"] += rp
        T["ranked_prompt_paired"] += rpp
        T["ranked_prompt_calibration"] += rpc
    for name, rows in T.items():
        write_csv(out / f"{name}.csv", rows)
    text = render(T, configs)
    if args.large_models:
        from llmdw.scoring import large_models
        names = [ln.strip() for ln in (work / large_models.LARGE_DIR / "instances.txt").read_text().splitlines()
                 if ln.strip()]
        md, _ = large_models.report(names, out, work)
        marker = "## 9. "
        text = text.replace(marker, md + "\n" + marker, 1) if marker in text else text + "\n" + md
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    if args.rich_models:
        rich_models.rich_tables(work, out, configs)
    print(f"wrote the tables and RESULTS.md to {out}")
    return 0


def sample_of_record(T: dict, configs: list, times: dict) -> list:
    """Per configuration and phase: answer files used, the times of the calls that produced them, failed answers."""
    rel = {(r["phase"], r["config"]): r for r in T["reliability"]}
    out = []
    for cfg in configs:
        for phase in ("templates", "miplib"):
            a, b = rel.get((phase, cfg), {}), times.get((phase, cfg), {})
            out.append({"config": cfg, "phase": phase, "answer_files": a.get("answers"),
                        "first_answer": b.get("first_answer"), "last_answer": b.get("last_answer"),
                        "unparseable": a.get("unparseable"), "no_answer": a.get("cells_without_answer"),
                        "cells_planned": a.get("cells_planned")})
    return out


TEMPLATE_CLASSES = [("clsp", "Lot sizing"), ("ndp", "Network design"), ("cfl", "Facility location"),
                    ("gap", "Generalized assignment"), ("needle", "Hidden group")]
FAMILY_LETTERS = {"FlowBalance_#_#": "N", "ArcCapacity_#": "U", "ArcLink_#_#": "L"}   # network design


def family_letter(fam: str) -> str:
    """One letter per template family: the letter after Conflict (hidden-group templates), a fixed letter for the
    network-design families, otherwise the first letter of the name."""
    if fam in FAMILY_LETTERS:
        return FAMILY_LETTERS[fam]
    m = re.fullmatch(r"Conflict([A-Z])_#", fam)
    return m.group(1) if m else fam[0]


def _round_half_up(x: Fraction, digits: int) -> str:
    q = Fraction(10) ** digits
    v = math.floor(x * q + Fraction(1, 2))
    return f"{v}" if digits == 0 else f"{Fraction(v) / q:.{digits}f}"


def templates_table(work: Path) -> list:
    """One row per zero-marginal group of each template (its joint supports; group "none" when there is none),
    from ground_truth.json in exact arithmetic. The integer optimum is the bound with every family convexified
    (the LP bound plus gain(all families) when minimizing, minus it when maximizing), so the LP gap is
    gain(all families) / |integer optimum| and a group closes gain(group) / gain(all families) of it (percent,
    rounded half up)."""
    rows = []
    for prefix, cls in TEMPLATE_CLASSES:
        root = work / "work/templates"
        for d in sorted(p for p in root.iterdir() if p.name.split("_")[0] == prefix and (p / "ground_truth.json").exists()):
            gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
            fams = gt["families"]
            gain = {k: Fraction(v) for k, v in gt["gain"].items()}
            full = gain["|".join(sorted(fams))]
            sign = -1 if summaries.load(work, "templates", d.name, d / "model.mps").sense == "maximize" else 1
            ip = Fraction(gt["lp_bound"]) + sign * full
            base = {"class": cls, "instance": d.name, "k": len(fams),
                    "lp_gap_percent": _round_half_up(100 * full / abs(ip), 1) if ip else ""}
            groups = sorted(gt["joint_supports"], key=lambda g: "".join(sorted(map(family_letter, g))))
            if not groups:
                rows.append({**base, "group": "none", "gap_closed_percent": ""})
            for g in groups:
                closed = gain["|".join(sorted(g))] / full if full else Fraction(0)
                rows.append({**base, "group": "".join(sorted(map(family_letter, g))),
                             "gap_closed_percent": _round_half_up(100 * closed, 0)})
    return rows


def render(T: dict, configs: list) -> str:
    s = ["# Results", "",
         f"Sample of record: configurations {', '.join(NAMES.get(c, c) for c in configs)}; template answers and MIPLIB-NL answers after the "
         "one rerun of each phase. A planned cell without an answer counts as a failed answer. Written by "
         "`python -m llmdw.scoring.results`; every table is also a CSV file in this directory. Rates are means over instances "
         "of the per-instance mean over samples; intervals are 95% percentile bootstrap intervals resampling "
         "instances (10,000 resamples, seed 0), Wilson intervals for the census.", "",
         "## Sample of record", "",
         "Answer files used per configuration and phase, the times of the calls that produced them, and the failed "
         "answers (unparseable; no answer after the one rerun). `sample_of_record.csv`. Answer times are those of the "
         "calls in `runs/llm_calls.jsonl` that produced them (answered calls; answers read from the cache at the time "
         "count with the original call).", "",
         md_table(T["sample_of_record"], ["config", "phase", "answer_files", "first_answer", "last_answer",
                                          "unparseable", "no_answer"]), ""]
    s += ["## 1. Prediction", "", "Templates: classes clsp, ndp and cfl; MIPLIB-NL: instances with a certain joint group "
          "(no linking variables). `prediction.csv`.", "",
          md_table(T["prediction"], ["phase", "config", "condition", "class", "n_instances", "hit1", "hit1_lo",
                                     "hit1_hi", "hit1_se", "recall", "recall_lo", "recall_hi", "recall_se"]), "",
          "False alarms on the assignment templates (gap). `false_alarm.csv`.", "",
          md_table(T["false_alarm"], ["config", "condition", "n_instances", "false_alarm", "false_alarm_lo",
                                      "false_alarm_hi", "false_alarm_se"]), "",
          "Hidden-group templates (needle): hit rate against chance 1/C(k,w) and the stricter level (family sets of "
          "size w with the row counts of the hidden group). `needle.csv` also has every instance.", "",
          md_table([r for r in T["needle"] if r["instance"] == "all"],
                   ["config", "condition", "n_instances", "hit_rate", "hit_lo", "hit_hi", "hit_se", "chance", "strict_chance"]), "",
          md_table([r for r in T["needle"] if r["instance"] != "all" and r["config"] == configs[0]
                    and r["condition"] == "original"], ["instance", "k", "w", "chance", "strict_chance",
                                                         "matching_sets"]), "",
          "Lookahead detection on the same instances. `lookahead_detection.csv`.", "",
          md_table(T["lookahead_detection"], ["phase", "class", "n_instances", "n_without_lookahead",
                                              "detection_rate", "detection_lo", "detection_hi", "detection_se"]), ""]
    s += ["## 2. Paired differences in recall", "", "`paired.csv`.", "",
          md_table(T["paired"], ["phase", "config", "contrast", "n_instances", "mean_difference", "lo", "hi", "se_difference"]), ""]
    if T["paired_configs"]:
        a, b = public_row(T["paired_configs"][0])["contrast"].split(" minus ")
        s += [f"## 2b. Reasoning: {a} minus {b} (the same model with and without reasoning)", "",
              "Paired over instances per presentation: hit@1 and recall on the prediction instances, JointDW's "
              "evaluations on the search-cost instances (a negative difference: fewer evaluations with reasoning). "
              "`paired_configs.csv`.", "",
              md_table(T["paired_configs"], ["phase", "condition", "measure", "n_instances", "mean_a", "mean_b",
                                             "mean_difference", "lo", "hi", "se_difference"]), ""]
    s += ["## 3. Search cost", "", "Evaluations until a best set is evaluated, on instances with a true joint support "
          "(templates: every class with one, including needle). Mean and median over instances (JointDW: of the "
          "per-instance mean over samples); failures are not in the mean. `search_cost.csv`; per instance and sample, "
          "e and JointDW's evaluations with the covering cost in `search_points.csv`.", "",
          md_table(T["search_cost"], ["phase", "method", "config", "condition", "n_instances", "n_failures", "mean",
                                      "median"]), "",
          "JointDW's anchor in the main tables is the first joint support of the answer (as in Theorem 2; the "
          "top-ranked candidate's convexified families when the answer lists no joint support); the prediction error "
          "e in `search_points.csv` uses the same anchor.", "",
          "### 3b. Alternative: JointDW anchored at the top-ranked candidate", "",
          "The families convexified by the top-ranked candidate reformulation. "
          "`search_cost_top_candidate.csv`; per s in `search_by_size_top_candidate.csv`; e_top_candidate in "
          "`search_points.csv`.", "",
          md_table(T["search_cost_top_candidate"], ["phase", "method", "config", "condition", "n_instances",
                                                    "n_failures", "mean", "median"]), "",
          "### 3c. Paired comparisons (maximal-gain target, first-support anchor, original presentation)", "",
          "Per instance JointDW (and JointDW interleaved with the lookahead order) minus the covering order and minus "
          "the lookahead order, paired over instances where both reach the target; 95% paired bootstrap intervals "
          "(10,000 resamples, seed 0); MIPLIB-NL also in seconds (oracle times capped at 120 s, a set with a "
          "lookahead value only at 120 s). a_better / b_better: instances where the first / second order is cheaper. "
          "`paired_search.csv`.", "",
          md_table(T["paired_search"], ["phase", "limit", "config", "unit", "contrast", "n_instances", "mean_a",
                                        "mean_b", "mean_difference", "lo", "hi", "se_difference", "a_better", "b_better"]), ""]
    s += ["## 4. Census", "", "`census_shares.csv`, `census_instances.csv`, `census_counts.csv`.", "",
          md_table(T["census_shares"], ["group_kind", "denominator", "k", "n", "share", "wilson_lo", "wilson_hi"]), "",
          md_table(T["census_counts"], ["count", "n"]), "",
          "### 4b. Census figure data (corrected)", "",
          "Largest single-family gain and largest set gain as shares of the LP gap, per instance, after the "
          "corrections listed in the corrections column (`census_figure.csv`); the census counts above do not change "
          "under them. Models left out of the figure, with the reason (`census_figure_left_out.csv`):", "",
          md_table(T["census_figure_left_out"], ["instance", "reason"]), "",
          md_table([r for r in T["census_figure"] if r["corrections"]], ["instance", "k", "max_single_family_gain",
                                                                       "max_set_gain", "group_kind", "corrections"]),
          ""]
    s += ["## 5. Reliability", "", "`reliability.csv`. Retries from `runs/llm_retries.jsonl`.", "",
          md_table(T["reliability"], ["phase", "config", "cells_planned", "answers", "cells_without_answer",
                                      "unparseable", "cells_with_three_answers", "first_support_agree_share",
                                      "cells_with_identical_samples"] + [f"retries_{k}" for k in usage.RETRY_REASONS]
                   + ["median_latency_s", "prompt_tokens", "completion_tokens"]), ""]
    s += ["## 6. Settings", "", "`settings.csv`.", ""]
    if T["settings"]:
        keys = list(dict.fromkeys(k for r in T["settings"] for k in r))
        s += [md_table(T["settings"], keys), ""]
    s += ["## 7. Offline analyses", "",
          "No LLM request. Lookahead detection in section 1 is defined as follows: a true joint support W counts as "
          "detected when W is a zero-marginal group of the lookahead values alone, that is l(W) > tol and l(S) <= tol "
          "for every proper subset S of W (the LP has value 0), over the family sets the lookahead evaluated without "
          "the full family set; tol is the census tolerance (1e-4 in closed-gap units when the optimum is known, else "
          "1e-6 max(1, |LP|)). Below, the lookahead order (sets with at most k - 1 families by decreasing lookahead "
          "value; ties: fewer families first, then by name) is scored like a model's answer.", "",
          "### 7.1 The lookahead order on the models' terms", "",
          "Hit@1: the top-ranked set is a true joint support. Recall over the top m sets, m the mean number of joint "
          "supports in that model's answers (original presentation, rounded, at least 1); the same instances and "
          "intervals as the prediction table, with the model's own hit@1 and recall (original) beside them. "
          "`lookahead_like_models.csv`.", "",
          md_table(T["lookahead_like_models"], ["phase", "config", "m_mean", "m", "n_instances", "lookahead_hit1",
                                                "lookahead_hit1_lo", "lookahead_hit1_hi", "lookahead_hit1_se",
                                                "lookahead_recall_at_m", "lookahead_recall_at_m_lo",
                                                "lookahead_recall_at_m_hi", "lookahead_recall_at_m_se", "model_hit1",
                                                "model_recall"]), "",
          "### 7.2 Search cost by size limit s", "",
          "Only sets with at most s families are evaluable, and the best set is the best among them; s = 2, 3 and k - 1 "
          "(never above k - 1). Instances with a true joint support (templates: every class with one); JointDW per "
          "sample, then the mean over the samples of an instance; intervals: bootstrap over instances (10,000, seed "
          "0). Covering numbers per instance and s in `covering_numbers.csv`. `search_by_size.csv`.", "",
          md_table(T["search_by_size"], ["phase", "limit", "method", "config", "condition", "n_instances", "n_failures",
                                         "mean", "mean_lo", "mean_hi", "mean_se", "median", "median_lo", "median_hi", "median_se"]), "",
          "### 7.3 Complementarity", "",
          "Prediction instances; a model hits when the first joint support is a true one in at least two of the three "
          "original samples, the lookahead when its top-ranked set is a true joint support. `complementarity.csv`.", "",
          md_table(T["complementarity"], ["phase", "config", "n_instances", "both", "model_only", "lookahead_only",
                                          "neither"]), "",
          "Instances the lookahead misses, with the kind of their true supports (source from the exact ground truth "
          "of the templates; not determined for MIPLIB-NL, where it needs exact hull computations; variable-disjoint: "
          "no column in rows of two of its families). `lookahead_misses.csv`.", "",
          md_table(T["lookahead_misses"], ["phase", "instance", "true_support", "source", "variable_disjoint",
                                           "lookahead_first"]), "",
          "JointDW's order interleaved with the lookahead order (alternately, model first, repeats skipped), "
          "original presentation, against 2 min(model, lookahead) per instance. `interleaved_summary.csv`, per instance "
          "`interleaved.csv`.", "",
          md_table(T["interleaved_summary"], ["phase", "limit", "config", "n_instances", "interleaved_mean",
                                              "model_mean", "lookahead_mean", "two_min_mean",
                                              "instances_within_two_min"]), "",
          "### 7.4 By kind of joint support", "",
          "Original presentation. Model recall per true support (mean over samples) and hit@1 per instance whose true "
          "supports are all of that kind; the lookahead with recall over its top m sets and hit@1. "
          "`by_group_kind.csv`.", "",
          md_table(T["by_group_kind"], ["phase", "dimension", "kind", "config", "n_supports", "n_instances",
                                        "model_recall", "model_recall_lo", "model_recall_hi", "model_recall_se", "model_hit1", "m",
                                        "lookahead_recall_at_m", "lookahead_hit1"]), ""]
    orig = [r for r in T["theorem2_search"] if r["method"] != "jointdw" or r["condition"] == "original"]
    s += ["## 9. The anchor and the target of Theorem 2", "",
          "Offline (no LLM request). Anchor: besides the families convexified by "
          "the top-ranked candidate (sections 3 and 7), the first joint support listed in the answer (the top "
          "candidate's anchor when the answer lists none). Target: besides a set of maximal gain among the sets with at "
          "most s families, a minimal set whose gain is at least 0.95 of that best gain (no proper subset in the table "
          "reaches it); only evaluating such a minimal set itself reaches this target. Instances with a true joint "
          "support (templates: every class with one); JointDW per sample, then the mean over the samples of an "
          "instance; random: exact expectations. Time (MIPLIB-NL): the recorded wall-clock seconds of each evaluated "
          "set, capped at the 120 s limit (censored and failed evaluations count at most 120 s; a set with a lookahead "
          "value only counts 120 s). CSVs: `theorem2_anchors.csv`, `theorem2_search.csv` (every presentation), "
          "`theorem2_target_sizes.csv`.", "",
          "### 9.1 Anchors: share of samples with e = 0, and how often the two anchors differ", "",
          md_table(T["theorem2_anchors"], ["phase", "config", "condition", "samples", "e0_share_top_candidate",
                                           "e0_share_first_support", "anchors_differ_share"]), "",
          "### 9.2 Search cost and time to the target (JointDW: original presentation)", "",
          md_table(orig, ["phase", "limit", "target", "method", "anchor", "config", "n_instances", "n_failures",
                          "mean", "mean_lo", "mean_hi", "mean_se", "median", "seconds_mean", "seconds_lo", "seconds_hi", "seconds_se"]), "",
          "### 9.3 Target sizes per instance", "",
          "Number and sizes of the maximal-gain sets and of the minimal 0.95 sets among the sets with at most s "
          "families.", "",
          md_table(T["theorem2_target_sizes"], ["phase", "limit", "instance", "k", "s", "max_sets", "max_set_size_min",
                                                "max_set_size_max", "min95_sets", "min95_size_min",
                                                "min95_size_max"]), ""]
    if T.get("shortlist_search"):
        sl_orig = [r for r in T["shortlist_search"] if r["condition"] == "original"]
        slp_orig = [r for r in T["shortlist_paired"] if r["condition"] == "original"]
        s += ["## 10. Shortlist anchors", "",
              "Offline (no LLM request). Shortlists: S1, S2, S3 are the unions of the "
              "families of the first 1, 2 and 3 joint supports listed in the answer (all of them when it lists fewer), "
              "Sall the union of all listed joint supports; an answer that lists none uses the default anchor (the "
              "first joint support, else the top candidate) for every shortlist. The shortlist order of C is the "
              "covering order of the sets of at most s families inside C, then the covering order of all families, "
              "without repeats; `+lookahead` interleaves it with the lookahead order. Target: a set of maximal gain "
              "among the sets with at most s families; instances with a true joint support (templates: every class "
              "with one); per sample, then the mean over the samples of an instance. mean_size: mean number of "
              "families in the shortlist over samples; best_inside_share: share of samples in which a set of maximal "
              "gain lies inside the shortlist. Paired differences over instances (negative: fewer evaluations than the "
              "comparison). CSVs (every presentation): `shortlist_search.csv`, `shortlist_paired.csv`.", "",
              "### 10.1 Evaluations, shortlist size and coverage (original presentation)", "",
              md_table(sl_orig, ["phase", "limit", "config", "method", "n_instances", "n_failures", "mean_size",
                                 "best_inside_share", "mean", "mean_lo", "mean_hi", "mean_se", "median"]), "",
              "### 10.2 Paired differences (original presentation)", "",
              md_table(slp_orig, ["phase", "limit", "config", "contrast", "n_instances", "mean_a", "mean_b",
                                  "mean_difference", "lo", "hi", "se_difference", "a_better", "b_better"]), ""]
    return "\n".join(s)


if __name__ == "__main__":
    raise SystemExit(main())
