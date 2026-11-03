"""Scoring and search on the rich models (llmdw.instances.rich_models_build).

Paper: Section 6 and Appendix F: the rich models (Tables 5, 11 and 12, Figure 4).

Files under the work root:
  work/rich_models/instances.txt, instances/<name>/   the instances with exact gains, lookahead values and GCG anchors
  work/rich_models/draws.json                          every drawn candidate (LP gap, IP optimum, groups)
  work/support_rich_models/<config>/                   answers to the original prompt (original presentation)
  work/rich_models_with_data/<config>/                 answers to the prompt with the full instance

Every search order that depends on an order of the families is run under the family orders of llmdw.search.relabel, and
the per-instance value is the mean over these orders. Bootstraps resample prompt groups (llmdw.scoring.cluster): the groups
of the original prompt for its own tables and for the paired differences with minus without data, single instances
for the tables of the prompt with data (every such prompt is distinct).
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from llmdw.scoring import analyze_support, cluster, results
from llmdw.search import relabel
from llmdw.instances import summaries
from llmdw.instances.rich_models_build import CLASSES, NAME_PREFIX

PHASE = results.RICH_PHASE
RICH_DIR = "work/rich_models"
ORIGINAL_DIR = "work/support_rich_models"
WITH_DATA_DIR = "work/rich_models_with_data"


def instance_names(work: Path) -> list:
    return [x for x in (work / RICH_DIR / "instances.txt").read_text().split() if x]


def load_instances(work: Path) -> list:
    out = []
    for n in instance_names(work):
        d = work / RICH_DIR / "instances" / n
        truth = analyze_support.template_truth(d)
        la = json.loads((d / "lookahead.json").read_text(encoding="utf-8"))
        truth["lookahead"] = [(frozenset(r["families"]), r["lookahead"] or 0.0) for r in la["table"].values()]
        gp = d / "gcg_anchors.json"
        truth["gcg"] = {b: frozenset(r["convexified"]) for b, r in json.loads(gp.read_text()).items()
                        if r.get("convexified")} if gp.exists() else {}
        model = summaries.load(work, PHASE, n, d / "model.mps")
        truth["family_order"] = relabel.file_order(model, truth["families"])
        cls = next(c for c in CLASSES if n.startswith(f"{NAME_PREFIX}_{c}_"))
        out.append({"name": n, "class": cls, "dir": d, "truth": truth, "model": model})
    return out


def search(instances: list, answers: dict | None = None) -> dict:
    """Evaluations until a set of maximal gain (among the sets of at most s families; supersets count) for s = 2, 3;
    every order except random is the mean over the family orders of llmdw.search.relabel (results.search_runs; an instance
    fails when any order fails)."""
    from llmdw.search import search_sim

    per, inst_rows = [], []
    for inst in instances:
        t, fams = inst["truth"], sorted(inst["truth"]["families"])
        row = {"instance": inst["name"], "class": inst["class"], "k": len(fams)}
        for s in (2, 3):
            tab = results.restricted(t["table"], s)
            goal = results.targets(tab, "max")
            la = [T for T in results.lookahead_order(t) if len(T) <= s]
            row[f"covering_number_s{s}"] = len(search_sim.covering_order(fams, s))
            row[f"best_sets_s{s}"] = " ".join("|".join(sorted(T)) for T in sorted(goal, key=lambda T: (len(T), sorted(T))))
            row[f"lookahead_first_s{s}"] = "|".join(sorted(la[0])) if la else ""
            row[f"lookahead_first_wrong_s{s}"] = bool(la) and la[0] not in goal
            N, M = len([T for T in tab if T]), len(goal)
            nm = inst["name"]
            run = lambda spec: results.run_value(results.search_runs(nm, t, s, spec)[0])  # noqa: E731
            res = {("random", ""): (N + 1) / (M + 1), ("greedy", ""): run(("greedy",)),
                   ("covering", ""): run(("covering",)), ("lookahead", ""): run(("lookahead",)) if la else None}
            for b, P in t.get("gcg", {}).items():
                res[(f"gcg_anchor_{b}", "")] = run(("model", frozenset(P)))
            for (meth, _), v in res.items():
                per.append({"instance": inst["name"], "class": inst["class"], "k": len(fams), "s": s, "method": meth,
                            "config": "", "condition": "", "sample": "", "evaluations": v})
            for cfg, recs in (answers or {}).items():
                for r in recs:
                    if r["instance"] != inst["name"]:
                        continue
                    P = results.support_anchor(r, fams)
                    for meth, spec in (("jointdw", ("model", P)), ("interleaved", ("interleaved", P))):
                        per.append({"instance": inst["name"], "class": inst["class"], "k": len(fams), "s": s,
                                    "method": meth, "config": cfg, "condition": r["condition"],
                                    "sample": r.get("sample", 0), "evaluations": run(spec)})
        inst_rows.append(row)
    return {"per": per, "instances": inst_rows}


def llm_context(work: Path, configs) -> tuple[dict, dict]:
    """(instances in the format of llmdw.scoring.results, {configuration: answer records of the original prompt, original
    presentation}) for the configurations with answers; a planned cell without a file is a failed answer."""
    insts = {}
    for i in load_instances(work):
        insts[i["name"]] = {"truth": i["truth"], "summary": i["model"].summary, "class": i["class"]}
    return insts, answers(work / ORIGINAL_DIR, insts, configs)


def answers(root: Path, insts: dict, configs) -> dict:
    out = {}
    for c in configs:
        d = root / c
        if d.exists() and any(d.rglob("*.json")):
            out[c] = [r for r in results.load_answers(d, insts) if r["condition"] == "original"]
    return out


def use_prompt_groups(work: Path) -> list:
    """Register the prompt groups of the rich models (instances with byte-identical requests) for the cluster
    bootstrap, in place of any other groups."""
    cluster.set_groups({})
    return results.use_prompt_groups(work, PHASE)


def relabel_orders(name, truth):
    return relabel.family_orders(name, results.base_order(truth))


def relabel_first(truth, order, s):
    """The first set of the lookahead order (sets of at most s families) in one family order, or None."""
    la = relabel.lookahead_list(truth.get("lookahead"), order, s)
    return la[0] if la else None


def _paired_row(base: dict, a: str, b: str, A: dict, B: dict) -> dict:
    both = [n for n in sorted(A) if A[n] is not None and B.get(n) is not None]
    diffs = [A[n] - B[n] for n in both]
    lo, hi, se = results.boot_ci_se(list(zip(both, diffs)))
    return {**base, "contrast": f"{a} minus {b}", "n_instances": len(both), "mean_a": results._mean(A[n] for n in both),
            "mean_b": results._mean(B[n] for n in both), "mean_difference": results._mean(diffs),
            "se_difference": se, "lo": lo, "hi": hi, "asterisk": "*" if lo is not None and (lo > 0 or hi < 0) else ""}


def _per_instance_evals(per_llm, cfg, method, s, names) -> dict:
    acc = {}
    for x in per_llm:
        if x["config"] == cfg and x["method"] == method and x["s"] == s and x["instance"] in names:
            acc.setdefault(x["instance"], []).append(x["evaluations"])
    out = {}
    for n, v in acc.items():
        ok = [y for y in v if y is not None]
        out[n] = sum(ok) / len(ok) if ok else None
    return out


def rich_tables(work: Path, out: Path, configs) -> dict:
    """The tables of the rich models, all after relabeling (llmdw.search.relabel):
    rich_models_search.csv     per class and "all", s in {2, 3}, order (random, covering, lookahead, each GCG anchor
                               that exists; JointDW and JointDW interleaved with the lookahead per configuration):
                               n_instances, n_failures, mean evaluations with the bootstrap interval and SE
    rich_models_instances.csv  per instance: class, k, LP gap, the minimal maximal-gain sets (sets of at most 3
                               families), the share of the LP gap the best such set closes, and the share of the family
                               orders under which the lookahead order names a maximal-gain set first, for s = 2 and 3
    rich_models_groups.csv     every zero-marginal group: size, source, gain, family pairs and pairs sharing a variable
    rich_models_prediction.csv hit@1 and recall of the original prompt per configuration and class
    rich_models_paired.csv     JointDW and its interleaving minus covering and minus lookahead, paired over instances
    and the four tables of the prompt with data (with_data_tables)."""
    insts = load_instances(work)
    use_prompt_groups(work)
    draws = {d["name"]: d for d in json.loads((work / RICH_DIR / "draws.json").read_text())["draws"]}
    per = search(insts)["per"]
    methods = ["random", "covering", "lookahead"] + sorted({r["method"] for r in per if r["method"].startswith("gcg_anchor_")})
    srch = []
    for cls in list(CLASSES) + ["all"]:
        for s in (2, 3):
            for m in methods:
                d = {r["instance"]: r["evaluations"] for r in per if r["s"] == s and r["method"] == m
                     and (cls == "all" or r["class"] == cls)}
                if not d:
                    continue
                vals = [v for v in d.values() if v is not None]
                lo, hi, se = results.boot_stat_se(d, np.mean)
                srch.append({"class": cls, "s": s, "method": m, "config": "", "n_instances": len(d),
                             "n_failures": sum(v is None for v in d.values()), "mean": results._mean(vals),
                             "mean_lo": lo, "mean_hi": hi, "mean_se": se})
    rows_i, rows_g = [], []
    for inst in insts:
        n, d, t = inst["name"], inst["dir"], inst["truth"]
        gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
        tab = {frozenset(x for x in k.split("|") if x): Fraction(v) for k, v in gt["gain"].items()}
        best = max(tab.values())
        goal = [T for T, v in tab.items() if T and v == best]
        minimal = sorted((T for T in goal if not any(S < T for S in goal)), key=lambda T: (len(T), sorted(T)))
        dr = draws[n]
        lp, ip = Fraction(dr["lp_bound"]), Fraction(dr["ip_scip"]).limit_denominator(10 ** 6)
        row = {"instance": n, "class": inst["class"], "k": len(gt["families"]), "lp_gap_percent": 100 * dr["lp_gap"],
               "minimal_best_sets": " ".join("|".join(sorted(T)) for T in minimal),
               "gap_closed_by_best_percent": float(100 * best / abs(ip - lp)) if ip != lp else None}
        for s in (2, 3):
            g = results.targets(results.restricted(t["table"], s), "max")
            tops = [relabel_first(t, order, s) for order in relabel_orders(n, t)]
            row[f"lookahead_first_is_best_s{s}"] = results._mean(float(x is not None and x in g) for x in tops)
        la2 = [v for T, v in (t.get("lookahead") or []) if T and len(T) <= 2]
        row["lookahead_all_within_1e-9_of_zero_s2"] = bool(la2) and all(abs(v) <= 1e-9 for v in la2)
        rows_i.append(row)
        cols = {f: inst["model"].columns(f, nonzero=True) for f in gt["families"]}
        for W in gt["joint_supports"]:
            pairs = list(itertools.combinations(sorted(W), 2))
            key = "|".join(sorted(W))
            rows_g.append({"instance": n, "class": inst["class"], "group": key, "size": len(W),
                           "source": gt.get("group_sources", {}).get(key, ""), "gain": float(tab[frozenset(W)]),
                           "family_pairs": len(pairs),
                           "pairs_sharing_a_variable": sum(1 for a, b in pairs if cols.get(a, set()) & cols.get(b, set()))})
    pred, paired = [], []
    r_inst, recs_by = llm_context(work, configs)
    if recs_by:
        per_llm = search(insts, recs_by)["per"]
        for cfg, recs in recs_by.items():
            scores = [analyze_support.score(r, r_inst[r["instance"]]["truth"]) for r in recs]
            for cls in list(CLASSES) + ["all"]:
                names = {n for n, i in r_inst.items() if cls == "all" or i["class"] == cls}
                h = results.per_instance(scores, "joint_hit1", "original", names)
                r_ = results.per_instance(scores, "joint_recall", "original", names)
                pred.append({"class": cls, "config": cfg, "condition": "original", "n_instances": len(r_),
                             "hit1": results._mean(h.values()), **results._ci("hit1", h),
                             "recall": results._mean(r_.values()), **results._ci("recall", r_)})
                for s in (2, 3):
                    vals = {}
                    for m in ("jointdw", "interleaved"):
                        acc = defaultdict(list)
                        for x in per_llm:
                            if (x["config"] == cfg and x["method"] == m and x["s"] == s and x["instance"] in names):
                                acc[x["instance"]].append(x["evaluations"])
                        d = {}
                        for n_, v in acc.items():
                            ok = [y for y in v if y is not None]
                            d[n_] = sum(ok) / len(ok) if ok else None
                        vals[m] = d
                        lo, hi, se = results.boot_stat_se(d, np.mean)
                        srch.append({"class": cls, "s": s, "method": m, "config": cfg, "n_instances": len(d),
                                     "n_failures": sum(v is None for v in d.values()),
                                     "mean": results._mean(v for v in d.values() if v is not None),
                                     "mean_lo": lo, "mean_hi": hi, "mean_se": se})
                    for b in ("covering", "lookahead"):
                        vals[b] = {x["instance"]: x["evaluations"] for x in per_llm if x["s"] == s and x["method"] == b
                                   and x["instance"] in names}
                    for a in ("jointdw", "interleaved"):
                        for b in ("covering", "lookahead"):
                            paired.append(_paired_row({"class": cls, "s": s, "config": cfg}, a, b, vals[a], vals[b]))
    out.mkdir(parents=True, exist_ok=True)
    tables = [("rich_models_search", srch), ("rich_models_instances", rows_i), ("rich_models_groups", rows_g)]
    if recs_by:
        tables += [("rich_models_prediction", pred), ("rich_models_paired", paired)]
    for name, rows in tables:
        results.write_csv(out / f"{name}.csv", rows)
    wd = with_data_tables(work, out, insts, r_inst, recs_by, configs)
    return {"search": srch, "instances": rows_i, "groups": rows_g, "prediction": pred, "paired": paired,
            "configs": sorted(recs_by), "with_data_configs": wd}


def with_data_tables(work: Path, out: Path, insts: list, r_inst: dict, recs_by_without: dict, configs) -> list:
    """The prompt with the full instance (prompts/user_with_data.txt), scored as the original prompt:
    rich_models_with_data_prediction.csv  hit@1 and recall per configuration, class and all
    rich_models_with_data_search.csv      JointDW and its interleaving with the lookahead order at s = 2, 3
    rich_models_with_data_paired.csv      JointDW and interleaved minus covering and minus lookahead
    with the bootstrap over single instances, and rich_models_with_data_vs_without.csv: with data minus without
    data for hit@1, recall and evaluations, with the bootstrap over the prompt groups of the original prompt.
    Returns the configurations with answers."""
    recs_by = answers(work / WITH_DATA_DIR, r_inst, configs)
    if not recs_by:
        return []
    groups_without = dict(cluster.GROUPS)
    cluster.set_groups({})
    per_wd = search(insts, recs_by)["per"]
    per_wo = search(insts, {c: r for c, r in recs_by_without.items() if c in recs_by})["per"]
    pred, srch, paired, vs = [], [], [], []
    for cfg, recs in recs_by.items():
        sc_wd = [analyze_support.score(r, r_inst[r["instance"]]["truth"]) for r in recs]
        sc_wo = [analyze_support.score(r, r_inst[r["instance"]]["truth"]) for r in recs_by_without.get(cfg, [])]
        for cls in list(CLASSES) + ["all"]:
            names = {n for n, i in r_inst.items() if cls == "all" or i["class"] == cls}
            h = results.per_instance(sc_wd, "joint_hit1", "original", names)
            r_ = results.per_instance(sc_wd, "joint_recall", "original", names)
            cluster.set_groups({})
            pred.append({"class": cls, "config": cfg, "condition": "original", "n_instances": len(r_),
                         "hit1": results._mean(h.values()), **results._ci("hit1", h),
                         "recall": results._mean(r_.values()), **results._ci("recall", r_)})
            measures = {"hit1": (h, results.per_instance(sc_wo, "joint_hit1", "original", names)),
                        "recall": (r_, results.per_instance(sc_wo, "joint_recall", "original", names))}
            for s in (2, 3):
                vals = {m: _per_instance_evals(per_wd, cfg, m, s, names) for m in ("jointdw", "interleaved")}
                for m in ("jointdw", "interleaved"):
                    d = vals[m]
                    lo, hi, se = results.boot_stat_se(d, np.mean)
                    srch.append({"class": cls, "s": s, "method": m, "config": cfg, "n_instances": len(d),
                                 "n_failures": sum(v is None for v in d.values()),
                                 "mean": results._mean(v for v in d.values() if v is not None),
                                 "mean_lo": lo, "mean_hi": hi, "mean_se": se})
                    measures[f"evals_{m}_s{s}"] = (d, _per_instance_evals(per_wo, cfg, m, s, names))
                for b in ("covering", "lookahead"):
                    vals[b] = {x["instance"]: x["evaluations"] for x in per_wd if x["s"] == s and x["method"] == b
                               and x["instance"] in names}
                for a in ("jointdw", "interleaved"):
                    for b in ("covering", "lookahead"):
                        paired.append(_paired_row({"class": cls, "s": s, "config": cfg}, a, b, vals[a], vals[b]))
            cluster.set_groups(groups_without)
            for key, (A, B) in measures.items():
                both = [n for n in sorted(A) if A[n] is not None and B.get(n) is not None]
                diffs = [A[n] - B[n] for n in both]
                lo, hi, se = results.boot_ci_se(list(zip(both, diffs)))
                vs.append({"class": cls, "config": cfg, "measure": key, "n_instances": len(both),
                           "mean_with_data": results._mean(A[n] for n in both),
                           "mean_without_data": results._mean(B[n] for n in both),
                           "mean_difference": results._mean(diffs), "se_difference": se, "lo": lo, "hi": hi,
                           "asterisk": "*" if lo is not None and (lo > 0 or hi < 0) else ""})
    cluster.set_groups(groups_without)
    for name, rows in (("rich_models_with_data_prediction", pred), ("rich_models_with_data_search", srch),
                       ("rich_models_with_data_paired", paired), ("rich_models_with_data_vs_without", vs)):
        results.write_csv(out / f"{name}.csv", rows)
    return sorted(recs_by)
