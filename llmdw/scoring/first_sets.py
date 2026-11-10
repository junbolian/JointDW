"""First sets of the language models against the first sets of the lookahead orders, and the cost of the interleaved
order against each of its two sources, from the released data only (no solver, no request).

Paper: Table 31 (tab:app-first) and the appendix paragraph "First sets of the language models and of the lookahead".

Collections: the templates, the 102 scored MIPLIB-NL models, the rich models with 7 to 9 families (24) and with 12
to 16 families (60), and the 59 benchmark instances. Prompts: "text" (the original presentation) and "every
coefficient" (the data variant), for the configurations recorded on each collection.

first_sets.csv, one row per collection, configuration, prompt and lookahead order (the lookahead order at the HiGHS
dual and the five-dual lookahead order):
- units: the instances with a true joint support (templates: the classes clsp, ndp and cfl; benchmark instances: the
  instances with a best pair);
- hit@1 of the language model: llmdw.scoring.analyze_support.score (benchmark instances: the anchor of the sample is a
  bound-optimal set, llmdw.benchmark.tables.instance_values), averaged over the samples of an instance and then over
  the instances;
- hit@1 of a lookahead order: llmdw.scoring.results.lookahead_hit1, the share of the family orders of
  llmdw.search.relabel under which its first set is a true joint support (benchmark instances: the set of the method
  in bnp_plan.json is a bound-optimal set);
- the joint outcome of one sample combines the language model's outcome (0 or 1) and the lookahead order's hit@1:
  both = x L, only the language model = x (1 - L), only the lookahead = (1 - x) L, neither = (1 - x)(1 - L); the share
  of an instance is the mean over its samples, then the mean over the instances; at least one = 1 - neither;
- the hit@1 of the language model on the instances where the lookahead order's hit@1 is 0, and of the lookahead order
  where every sample of the language model misses, with their numbers of instances (no rate below five instances);
- mean, standard error and 95 percent interval from the two-stage bootstrap of llmdw.scoring.cluster (prompt groups,
  then the instances within each drawn group; 10,000 resamples, seed 0) through llmdw.scoring.experiments._se_row and
  _paired, with the prompt groups of the text prompt; every instance is its own group for every coefficient.

evaluations_paired.csv, one row per collection, configuration, prompt, budget s and contrast: JointDW interleaved with
the five-dual lookahead order minus the five-dual lookahead order, minus the staged order alone, and minus the smaller,
for each answer, of the cost of the staged order alone and of the five-dual lookahead order. Costs are the evaluations
until a set of maximal gain among the sets of at most s families is evaluated, the mean over the 1,000 family orders
(llmdw.scoring.results.search_runs; the staged order alone with llmdw.search.relabel.staged_iter, as
llmdw.scoring.rich_models_large.misses_per_prediction). Instances and budgets as the tables of the paper: templates and
MIPLIB-NL as llmdw.scoring.experiments.five_duals_tables (s capped at k - 1; per instance a failure when one sample
fails), rich models as llmdw.scoring.rich_models_large (the sizes of the gain table; per instance the mean of the
samples with a value), benchmark instances as llmdw.scoring.cost_accounting.values_bench with s = 2.

tab_app_first.tex: the body of Table 31 (the rows with the five-dual lookahead order; the lookahead's own hit@1 in
the heading of each collection, from the text prompt).

Usage
-----
python -m llmdw.scoring.first_sets --data data --out results/first_sets --tex-dir results/paper/experiments
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from llmdw.constants import FAMILY_ORDERS
from llmdw.scoring import analyze_support, cluster, experiments, results
from llmdw.scoring import rich_models_large as ee
from llmdw.search import relabel

CONFIGS = ["claude-sonnet-5-5", "gpt-6.1-sol", "grok-4-20-reasoning", "grok-4-20-non-reasoning", "deepseek-v4-pro"]
PROMPT = {"original": "text", "data": "every coefficient"}
LA = {"highs": "lookahead order (HiGHS dual)", "five": "five-dual lookahead order"}
COLL = {"templates": "templates", "miplib": "scored MIPLIB-NL models", "rich_7_9": "rich models, 7 to 9 families",
        "rich_12_16": "rich models, 12 to 16 families", "benchmark": "benchmark instances"}
ORDERS = ("staged order alone", LA["highs"], LA["five"], "JointDW", "JointDW interleaved with the five-dual lookahead order",
          "covering order", "smaller of the staged order alone and the five-dual lookahead order")
ST, JD, IN, MIN = ORDERS[0], ORDERS[3], ORDERS[4], ORDERS[6]
ANSWER_ORDERS = (ST, JD, IN, MIN)


# ---------------------------------------------------------------- output

def fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return ""
        return repr(round(float(x), 10))
    return str(x)


def write_csv(path: Path, rows: list) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(keys)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in keys])


def se_row(values: dict, prefix: str) -> dict:
    return experiments._se_row({n: v for n, v in values.items() if v is not None}, prefix)


# ---------------------------------------------------------------- collections
#
# Each collection: insts {name: {"truth", "five", "class", "k", ...}}; answers {(configuration, prompt): [answer]},
# an answer {"instance", "sample", "anchor", "lm_hit"}; groups {prompt: {instance: prompt group}}; hit_names (the
# instances of first_sets.csv), search_names (the instances of evaluations_paired.csv).

def phase_collection(D: Path, phase: str, official: dict) -> dict:
    """Templates or scored MIPLIB-NL models, as llmdw.scoring.experiments.five_duals_tables (original presentation)."""
    W = D / "work"
    insts0 = experiments._phase_instances(D, phase)
    cluster.set_groups({})
    results.use_prompt_groups(D, phase, insts0)
    groups = dict(cluster.GROUPS)
    cluster.set_groups({})
    insts = {}
    for n, i in insts0.items():
        t = i["truth"]
        p = W / "five_duals" / phase / n / "five_duals.json"
        five = {**t, "lookahead": experiments._five_list(p, len(t["families"]), phase == "templates")} if p.exists() else None
        k = len(t["families"])
        insts[n] = {"truth": t, "five": five, "class": i["class"], "k": k}
    answers = {}
    for cfg in CONFIGS:
        d = W / f"support_{phase}" / cfg
        if not d.exists():
            continue
        rows = []
        for r in results.load_answers(d, insts0):
            if r["condition"] != "original":
                continue
            t = insts[r["instance"]]["truth"]
            sc = analyze_support.score(r, t)
            rows.append({"instance": r["instance"], "sample": r.get("sample", 0),
                         "anchor": results.support_anchor(r, t["families"]),
                         "lm_hit": (1.0 if sc.get("joint_hit1") else 0.0) if t["joint"] else None})
        answers[(official[cfg], "original")] = rows
    search_names = sorted(n for n, i in insts.items() if i["truth"]["joint"] and i["five"] is not None
                          and i["truth"]["table"] and max(i["truth"]["table"].values()) > 0)
    return {"key": phase, "insts": insts, "answers": answers, "groups": {"original": groups},
            "hit_names": sorted(results.prediction_instances(phase, insts0)), "search_names": search_names,
            "s_cap": True, "per_instance": "all samples"}


def rich_collection(D: Path, key: str, official: dict) -> dict:
    """Rich models with 7 to 9 (published) or 12 to 16 families, as llmdw.scoring.rich_models_large.score_set."""
    W = D / "work"
    if key == "rich_7_9":
        root, five_root, sw, phase = W / "rich_models", W / "five_duals/rich_models", D, "rich_models"
        dirs = {"original": W / "support_rich_models", "data": W / "rich_models_with_data"}
        configs = CONFIGS
    else:
        root, five_root, sw, phase = (W / "rich_models_large", W / "five_duals/rich_models_large",
                                      W / "rich_models_large", "rich_models_large")
        dirs = {"original": W / "support_rich_models_large", "data": W / "rich_models_large_with_data"}
        configs = ["gpt-6.1-sol", "claude-sonnet-5-5"]
    il = ee.load_set(root, five_root, sw, phase)
    recs, _ = ee.load_answers(dirs, il, configs, False)
    groups = ee.published_groups(D) if key == "rich_7_9" else ee.prompt_groups(il)
    insts = {i["name"]: {"truth": i["truth"], "five": i["five"], "class": i["class"], "k": i["k"],
                         "sizes": i["sizes"]} for i in il}
    answers = {}
    for (cfg, pres), rs in recs.items():
        rows = []
        for r in rs:
            t = insts[r["instance"]]["truth"]
            sc = analyze_support.score(r, {**t, "table": {}})          # as rich_models_large.prediction_rows
            rows.append({"instance": r["instance"], "sample": r.get("sample", 0),
                         "anchor": results.support_anchor(r, t["families"]),
                         "lm_hit": (1.0 if sc.get("joint_hit1") else 0.0) if t["joint"] else None})
        answers[(official[cfg], pres)] = rows
    return {"key": key, "insts": insts, "answers": answers, "groups": {"original": groups, "data": {}},
            "hit_names": sorted(n for n, i in insts.items() if i["truth"]["joint"]), "search_names": sorted(insts),
            "s_cap": False, "per_instance": "samples with a value"}


def bench_collection(D: Path, official: dict) -> dict:
    """Benchmark instances: hit@1 as llmdw.benchmark.tables.instance_values; search truths as
    llmdw.scoring.cost_accounting.values_bench (gain table of oracle.json, lookahead values of lookahead.json and
    five_duals.json, anchors of bnp_plan.json and bnp_plan_data.json)."""
    from llmdw.benchmark import bnp, tables

    root = D / "work" / "benchmark_std"
    names = (root / "full_set/all_instances.txt").read_text().split()
    names_of = {official[c]: c for c in ("gpt-6.1-sol", "claude-sonnet-5-5")}
    prompts = {}
    for n in names:
        prompts[n] = {}
        for cfg in names_of.values():
            f = root / "answers" / cfg / n / "original.json"
            if f.exists():
                prompts[n][cfg] = json.loads(f.read_text())["messages"]
    groups = cluster.prompt_groups(prompts)
    cluster.set_groups(groups)
    configs = list(names_of) + [f"{c}, data" for c in names_of]
    vals = {n: tables.instance_values(root, n, configs, ("bnp_plan.json", "bnp_plan_data.json")) for n in names}
    cluster.set_groups({})
    insts, answers = {}, defaultdict(list)
    for n in names:
        d = root / "instances" / n
        t = bnp._truth(d)
        k = len(t["families"])
        base = {"families": t["families"], "table": t["table"]}
        insts[n] = {"truth": {**base, "lookahead": experiments._five_list(d / "lookahead.json", k, True)},
                    "five": {**base, "lookahead": experiments._five_list(d / "five_duals.json", k, True)},
                    "class": vals[n]["class"], "k": k,
                    "la_hit": {"highs": float(vals[n]["methods"]["lookahead (single dual)"][0]["hit"]),
                               "five": float(vals[n]["methods"]["lookahead (five duals)"][0]["hit"])}}
        fams = frozenset(t["families"])
        for name in names_of:
            for pres, key in (("original", name), ("data", f"{name}, data")):
                for x in vals[n]["methods"].get(key) or []:
                    answers[(name, pres)].append({"instance": n, "sample": x["sample"],
                                                  "anchor": frozenset(x["set"].split("|")) & fams if x["set"] else frozenset(),
                                                  "lm_hit": float(x["hit"])})
    return {"key": "benchmark", "insts": insts, "answers": dict(answers), "groups": {"original": groups, "data": {}},
            "hit_names": sorted(names), "search_names": sorted(names), "s_cap": True, "per_instance": "all samples"}


def lookahead_hits(C: dict) -> dict:
    """hit@1 of each lookahead order per instance (the memo of llmdw.search.relabel is cleared between the two orders,
    whose cache keys do not hold the lookahead values)."""
    if C["key"] == "benchmark":
        return {o: {n: C["insts"][n]["la_hit"][o] for n in C["hit_names"]} for o in LA}
    out = {}
    for o in LA:
        relabel.clear()
        out[o] = {}
        for n in C["hit_names"]:
            t = C["insts"][n]["truth"] if o == "highs" else C["insts"][n]["five"]
            if t is not None and t.get("lookahead"):
                out[o][n] = results.lookahead_hit1(n, t)
    relabel.clear()
    return out


def collections(D: Path) -> dict:
    from llmdw.providers import settings

    official = settings.names()
    colls = {"templates": phase_collection(D, "templates", official), "miplib": phase_collection(D, "miplib", official),
             "rich_7_9": rich_collection(D, "rich_7_9", official), "rich_12_16": rich_collection(D, "rich_12_16", official),
             "benchmark": bench_collection(D, official)}
    for C in colls.values():
        C["la_hit"] = lookahead_hits(C)
    return colls


# ---------------------------------------------------------------- first sets

def joint_shares(lm_samples: list, L: float) -> dict:
    """The four joint outcomes of one instance: the mean over its samples of the products of the language model's
    outcome x (0 or 1) and the lookahead order's hit@1 L."""
    m = len(lm_samples)
    return {"both": sum(x * L for x in lm_samples) / m, "only_lm": sum(x * (1 - L) for x in lm_samples) / m,
            "only_la": sum((1 - x) * L for x in lm_samples) / m,
            "neither": sum((1 - x) * (1 - L) for x in lm_samples) / m}


def first_set_rows(colls: dict) -> list:
    rows = []
    for ck, C in colls.items():
        lah = C["la_hit"]
        for (cfg, pres), ans in C["answers"].items():
            cluster.set_groups(C["groups"].get(pres, {}))
            per = defaultdict(list)
            for a in ans:
                if a["instance"] in C["hit_names"] and a["lm_hit"] is not None:
                    per[a["instance"]].append(a)
            for o in LA:
                names = sorted(n for n in per if n in lah[o])
                lm = {n: sum(a["lm_hit"] for a in per[n]) / len(per[n]) for n in names}
                la = {n: lah[o][n] for n in names}
                cells = {c: {} for c in ("both", "only_lm", "only_la", "neither")}
                for n in names:
                    for c, v in joint_shares([a["lm_hit"] for a in per[n]], la[n]).items():
                        cells[c][n] = v
                one = {n: 1 - cells["neither"][n] for n in names}
                la_miss = [n for n in names if la[n] == 0]
                lm_miss = [n for n in names if lm[n] == 0]
                row = {"collection": COLL[ck], "configuration": cfg, "prompt": PROMPT[pres], "lookahead": LA[o],
                       **{f"lm_{k}": v for k, v in se_row(lm, "hit1").items()},
                       **{f"la_{k}": v for k, v in se_row(la, "hit1").items() if k not in ("n_instances", "n_groups")},
                       **{f"diff_{k}": v for k, v in experiments._paired(lm, la).items() if k not in ("mean_a", "mean_b")}}
                for c in cells:
                    r = se_row(cells[c], c)
                    row.update({f"{c}_mean": r[f"{c}_mean"], f"{c}_se": r[f"{c}_se"]})
                r = se_row(one, "at_least_one")
                row.update({"at_least_one_mean": r["at_least_one_mean"], "at_least_one_se": r["at_least_one_se"]})
                for tag, cond, miss in (("lm_hit1_where_la_misses", {n: lm[n] for n in la_miss}, la_miss),
                                        ("la_hit1_where_lm_misses", {n: la[n] for n in lm_miss}, lm_miss)):
                    row[f"{tag}_n"] = len(miss)
                    if len(miss) >= 5:
                        r = se_row(cond, "x")
                        row.update({f"{tag}_mean": r["x_mean"], f"{tag}_se": r["x_se"]})
                row["n_mixed_la"] = sum(1 for n in names if 0 < la[n] < 1)
                row["n_mixed_lm"] = sum(1 for n in names if 0 < lm[n] < 1)
                rows.append(row)
    cluster.set_groups({})
    return rows


# ---------------------------------------------------------------- evaluations

def staged_runs(n, t, s, P):
    """The staged order alone under the family orders of llmdw.search.relabel (array over the orders, nan where it
    evaluates no set of maximal gain), as llmdw.scoring.rich_models_large.misses_per_prediction."""
    tab = results.restricted(t["table"], s)
    goal = results.targets(tab, "max")
    return relabel.runs((n, s, "staged order alone", P), n, results.base_order(t),
                        lambda order: relabel.first_hit(relabel.staged_iter(order, s, P), tab, goal, None,
                                                        results.EVAL_LIMIT_S))[0]


def sizes_of(C: dict, n: str) -> list:
    i = C["insts"][n]
    if C["key"] == "benchmark":                  # k = 3: s = 2 only
        return [(2, 2)]
    if C["s_cap"]:
        return [(lab, min(lab, i["k"] - 1)) for lab in (2, 3)]
    return [(lab, lab) for lab in (2, 3) if lab in i["sizes"]]


def per_inst(values: list, rule: str):
    if rule == "all samples":
        return None if any(v is None for v in values) or not values else sum(values) / len(values)
    ok = [v for v in values if v is not None]
    return sum(ok) / len(ok) if ok else None


def rv(arr):
    return None if arr is None else results.run_value(arr)


def search_values(C: dict) -> dict:
    """Per instance the arrays of the covering order and the two lookahead orders, and per answer those of JointDW,
    the staged order alone and JointDW interleaved with the five-dual lookahead order (the memo of
    llmdw.search.relabel is cleared between the passes with the two lookahead values)."""
    inst_vals, ans_rows = defaultdict(dict), []
    names = C["search_names"]
    by = defaultdict(list)
    for (cfg, pres), rs in C["answers"].items():
        for a in rs:
            if a["instance"] in names:
                by[a["instance"]].append((cfg, pres, a))
    relabel.clear()
    for n in names:
        t = C["insts"][n]["truth"]
        for lab, s in sizes_of(C, n):
            inst_vals[("covering order", lab)][n] = results.search_runs(n, t, s, ("covering",))[0]
            la = [T for T in results.lookahead_order(t) if len(T) <= s] if t.get("lookahead") else []
            inst_vals[(LA["highs"], lab)][n] = results.search_runs(n, t, s, ("lookahead",))[0] if la else None
            for cfg, pres, a in by[n]:
                P = a["anchor"]
                nan = np.full(len(relabel.family_orders(n, results.base_order(t))), np.nan)
                ans_rows.append({"instance": n, "configuration": cfg, "prompt": pres, "sample": a["sample"], "s": lab,
                                 "anchor": P, JD: results.search_runs(n, t, s, ("model", P))[0],
                                 ST: staged_runs(n, t, s, P) if P else nan})
    relabel.clear()
    idx = {(r["instance"], r["configuration"], r["prompt"], r["sample"], r["s"]): r for r in ans_rows}
    for n in names:
        t = C["insts"][n]["five"]
        if t is None:
            continue
        for lab, s in sizes_of(C, n):
            la = [T for T in results.lookahead_order(t) if len(T) <= s]
            inst_vals[(LA["five"], lab)][n] = results.search_runs(n, t, s, ("lookahead",))[0] if la else None
            for cfg, pres, a in by[n]:
                idx[(n, cfg, pres, a["sample"], lab)][IN] = results.search_runs(n, t, s, ("interleaved", a["anchor"]))[0]
    relabel.clear()
    return {"inst": inst_vals, "ans": ans_rows}


def paired_rows(colls: dict) -> list:
    rows = []
    for ck, C in colls.items():
        X = search_values(C)
        rule = C["per_instance"]
        labs = sorted({lab for (_, lab) in X["inst"]})
        for a in X["ans"]:
            five = X["inst"].get((LA["five"], a["s"]), {}).get(a["instance"])
            vs, vf = rv(a[ST]), rv(five)
            a["v_" + ST], a["v_" + JD], a["v_" + IN] = vs, rv(a[JD]), rv(a.get(IN))
            cands = [v for v in (vs, vf) if v is not None]
            a["v_" + MIN] = min(cands) if cands else None
        for lab in labs:
            five_d = {n: rv(v) for n, v in X["inst"].get((LA["five"], lab), {}).items()}
            for (cfg, pres) in C["answers"]:
                cluster.set_groups(C["groups"].get(pres, {}))
                rs = [a for a in X["ans"] if a["configuration"] == cfg and a["prompt"] == pres and a["s"] == lab]
                if not rs:
                    continue
                per = {}
                for o in ANSWER_ORDERS:
                    acc = defaultdict(list)
                    for a in rs:
                        acc[a["instance"]].append(a["v_" + o])
                    per[o] = {n: per_inst(v, rule) for n, v in acc.items()}
                b5 = {n: five_d.get(n) for n in per[IN]}
                for contrast, A, B in ((f"{IN} minus {LA['five']}", per[IN], b5), (f"{IN} minus {ST}", per[IN], per[ST]),
                                       (f"{IN} minus {MIN}", per[IN], per[MIN])):
                    rows.append({"collection": COLL[ck], "configuration": cfg, "prompt": PROMPT[pres], "s": lab,
                                 "contrast": contrast, **experiments._paired(A, B)})
        cluster.set_groups({})
    return rows


# ---------------------------------------------------------------- Table 31

TABLE_SETS = (("miplib", "MIPLIB-NL, {n} models"), ("rich_7_9", "Rich models with 7 to 9 families, {n} models"),
              ("rich_12_16", "Rich models with 12 to 16 families, {n} models"), ("benchmark", "Benchmark, {n} instances"))
TABLE_CONFIGS = ("GPT-6.1 Sol", "Claude Sonnet 5.5")


def tex_first_sets(rows: list) -> str:
    """The body of Table 31 from the rows of first_sets.csv with the five-dual lookahead order: per collection a heading
    with the number of instances and the lookahead's hit@1 (text prompt), then hit@1 of the language model and the
    shares of instances on which only the language model, only the lookahead, and at least one of the two names a true
    group first. MIPLIB-NL: every configuration (text); the other collections: GPT-6.1 Sol and Claude Sonnet 5.5, text
    and every coefficient."""
    from llmdw.tables.paper_v1 import pm

    five = [r for r in rows if r["lookahead"] == LA["five"]]
    out = []
    for ck, head in TABLE_SETS:
        rs = [r for r in five if r["collection"] == COLL[ck]]
        text = [r for r in rs if r["prompt"] == "text"]
        out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{head.format(n=text[0]['lm_n_instances'])}; five-dual lookahead "
                   f"{pm(text[0]['la_hit1_mean'], text[0]['la_hit1_se'], 2)}}}}} \\\\")
        if ck == "miplib":
            sel = [(r["configuration"], r) for r in text]
        else:
            sel = [(f"{c}, {'text' if p == 'text' else 'coefficients'}", r) for c in TABLE_CONFIGS
                   for p in ("text", "every coefficient") for r in rs if r["configuration"] == c and r["prompt"] == p]
        for label, r in sel:
            out.append(f"{label} & {pm(r['lm_hit1_mean'], r['lm_hit1_se'], 2)} & {pm(r['only_lm_mean'], r['only_lm_se'], 2)} & "
                       f"{pm(r['only_la_mean'], r['only_la_se'], 2)} & {pm(r['at_least_one_mean'], r['at_least_one_se'], 2)} \\\\")
    return "\n".join(out) + "\n"


def read_rows(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def tex_from_csv(out: Path, tex_dir: Path) -> list:
    """tab_app_first.tex rebuilt from first_sets.csv in out."""
    Path(tex_dir).mkdir(parents=True, exist_ok=True)
    (Path(tex_dir) / "tab_app_first.tex").write_text(tex_first_sets(read_rows(Path(out) / "first_sets.csv")),
                                                    encoding="utf-8")
    return ["tab_app_first.tex"]


def run(data: Path, out: Path, tex_dir: Path) -> list:
    relabel.configure(FAMILY_ORDERS)
    out.mkdir(parents=True, exist_ok=True)
    colls = collections(data)
    write_csv(out / "first_sets.csv", first_set_rows(colls))
    write_csv(out / "evaluations_paired.csv", paired_rows(colls))
    return ["first_sets.csv", "evaluations_paired.csv"] + tex_from_csv(out, tex_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the released data (holds work/ and prompt_groups.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex-dir", required=True)
    a = ap.parse_args(argv)
    print(" ".join(run(Path(a.data), Path(a.out), Path(a.tex_dir))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
