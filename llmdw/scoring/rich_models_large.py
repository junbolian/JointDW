"""Scoring of the rich models with 12 to 16 families (llmdw.instances.rich_models_large).

Instances: the 60 models of llmdw.instances.rich_models_large (lot sizing and facility location with k = 12 and 16
families, network design with k = 12 and 14; ten per class and k). The generator and its seeds are in that module:
candidate j of a class and k draws its data from random.Random(SEED_BASE + 100 k + 10 (class index) + j) with
SEED_BASE = 7000, and the instance rich<k>_<class>_<j> is the j-th draw; draws.json records every draw. Each instance
has the exact gain of every set of at most table_complete_to families (3; 2 for network design with k = 14).

Everything is scored as the published rich models with 7 to 9 families (llmdw.scoring.rich_models,
llmdw.scoring.experiments), with the same functions:
- best sets: llmdw.scoring.results.targets(restricted(table, s), "max"), the sets of at most s families within 1e-9 of
  the largest gain among them (supersets of a best set count);
- hit@1 and recall of the predicted joint supports: llmdw.scoring.analyze_support.score, per instance the mean over
  the samples (results.per_instance);
- search orders, each the mean over the 1,000 family orders of llmdw.search.relabel (results.search_runs,
  results.run_value): the covering order; the lookahead order at the HiGHS dual (lookahead.json); the lookahead order
  at the best of five optimal duals (llmdw.lookahead.five_duals; the values enter only the order, as in
  llmdw.scoring.experiments); the staged search anchored at GCG's own decomposition, one order per GCG baseline whose
  decomposition is representable (spec ("model", P), as llmdw.scoring.rich_models.search); JointDW, the staged search
  around the first predicted joint support of an answer (the families convexified by the top candidate when it lists
  none; results.support_anchor) alternated with the covering order; and JointDW alternated with the five-dual
  lookahead order (spec ("interleaved", P) on the five-dual values). Per instance, JointDW is the mean over the
  samples with a value (llmdw.scoring.rich_models.rich_tables);
- s = 2 and s = 3: every order is restricted to the sets of at most s families. A size s above table_complete_to (s = 3
  for network design with k = 14) is not computed: its rows say so (status "not computed", counted in
  n_not_computed) and the pooled rows leave those instances out;
- a planned answer cell (instance, presentation, sample 0 to 2) without a file is a failed answer (no joint support,
  no candidate: JointDW then runs the covering order), as for the published rich models; --answered-only scores the
  cells with a file only (for a run while answers are still arriving).

Presentations: "original" (the original prompt, original presentation) and "data" (the prompt with the complete
instance), for the configurations of --configs; the display label of "data" is "coefficients". Bootstrap: 95 percent
percentile interval and standard error over 10,000 resamples (seed 0), two stages (llmdw.scoring.cluster): the prompt
groups of the set, then the instances within each drawn group, rows passed in sorted instance order
(llmdw.scoring.experiments._se_row and _paired). The prompt groups are those of the original prompt (prompt_groups:
instances whose original prompts are byte-identical; for the published rich models, prompt_groups.csv of the released
data); they apply to the rows of the original presentation, of the orders without a language model and of the misses
(which pool both presentations). In the rows of the coefficient presentation, whose prompts hold the complete
instance, every instance is its own group, as in llmdw.scoring.rich_models.

Groups: every class and k, every class pooled over k, every k pooled over the classes, and all instances.

The published rich models with 7 to 9 families (the released data, --published) are scored by the same code with the
same configurations and presentations, so that the cost of every order can be put against k
(figures/data/experiment_E_cost_vs_k.csv), and they enter the analysis of the misses of the predicted sets.

Outputs (--out):
  instances.csv        per instance: class, k, draw and seed, size, LP gap, table size, joint supports, minimal best
                       sets for s = 2, 3, covering numbers, the share of the family orders under which the first set of
                       each lookahead order is a best set, GCG anchors, prompt group of the original prompt
  gain_tables/<instance>.csv   every set of the gain table: exact gain, lookahead at the HiGHS dual and at the best of
                       five duals, best set for s = 2 and 3
  answers.csv          per set, configuration and presentation: planned cells, answer files, unparseable, missing,
                       answers without a predicted set
  predictions.csv      hit@1 and recall per configuration, presentation and group
  search_costs.csv     evaluations until a best set is evaluated, per group, s and order
  paired_five_dual.csv every order minus the five-dual lookahead order, paired over instances
  paired_covering.csv  every order minus the covering order, paired over instances
--tex-dir (default <out>/tables/new): experiment_E_instances.tex, experiment_E_predictions.tex,
  experiment_E_search.tex, experiment_E_paired.tex (table bodies).
--figure-dir (default <out>/figures/data):
  experiment_E_cost_vs_k.csv   per set, order, configuration, presentation, class, k and s: mean evaluations with the
                       standard error over instances (both sets)
  search_cost_vs_misses.csv    per set, class, k, s, number of misses e and size w of the best set: for the first
                       predicted set P of every JointDW answer with a predicted set (both sets), e = min over the best
                       sets B for s of |B minus P|, w the smallest |B| attaining it; the number of predictions and of
                       instances; the mean JointDW evaluations (per instance the mean over its predictions in the row,
                       then the mean over instances) with its bootstrap standard error; the number of distinct queries
                       in stages 0 to e of the staged search as implemented (llmdw.search.relabel.staged_iter, with the
                       greedy coverings of llmdw.search.search_sim) for k, s and a predicted set of w families; and the
                       mean evaluations of the covering order over the same instances.
With --answered-only a file PARTIAL.txt in --out records that only the cells with an answer file were scored.

Usage
-----
python -m llmdw.scoring.rich_models_large --work data/work --published data --out results/experiment_E \
    [--tex-dir results/paper/experiments] [--figure-dir results/paper/figures] [--configs gpt-6.1-sol,claude-sonnet-5-5] \
    [--answered-only] [--family-orders 1000]
--work holds rich_models_large (instances), five_duals/rich_models_large (five duals),
support_rich_models_large/<config> and rich_models_large_with_data/<config> (answers);
--published holds work/rich_models, work/five_duals/rich_models, work/support_rich_models/<config>,
work/rich_models_with_data/<config>, work/model_summaries and prompt_groups.csv (read only). No solver and no
language model is called.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

from llmdw.instances import summaries
from llmdw.instances.rich_models_build import CLASSES
from llmdw.scoring import analyze_support, cluster, experiments, results
from llmdw.search import relabel, search_sim
from llmdw.tables import experiments as tx

SIZES = (2, 3)
CONFIGS = ("gpt-6.1-sol", "claude-sonnet-5-5")
PRESENTATIONS = ("original", "data")
SET_E = "rich models, 12 to 16 families"
SET_PUBLISHED = "rich models, 7 to 9 families"
E_ROOT = "rich_models_large"                          # relative to --work (data/work of the released data)
E_FIVE = "five_duals/rich_models_large"
E_ANSWERS = {"original": "support_rich_models_large", "data": "rich_models_large_with_data"}
P_ROOT = "work/rich_models"
P_FIVE = "work/five_duals/rich_models"
P_ANSWERS = {"original": "work/support_rich_models", "data": "work/rich_models_with_data"}
NAME_RE = re.compile(r"^(?:rich(\d+)|e6c)_(" + "|".join(CLASSES) + r")_(\d+)$")
NOT_COMPUTED = "not computed"

COVERING = "covering"
LOOKAHEAD = "lookahead, HiGHS dual"
FIVE = "lookahead, five duals"
JOINTDW = "JointDW"
INTER_FIVE = "JointDW interleaved with the five-dual order"
GCG_LABELS = {"gcg_spfwh": "GCG anchor (SPFWH score)", "gcg_strong": "GCG anchor (strong score)",
              "gcg_spfwh_names": "GCG anchor (SPFWH score, names)"}
ANSWER_ORDERS = (JOINTDW, INTER_FIVE)
CLASS_LABELS = {"lot_sizing": "Lot sizing", "network_design": "Network design",
                "facility_location": "Facility location"}
PRESENTATION_LABELS = {"original": "original", "data": "coefficients"}
SET_GROUPS: dict = {}          # prompt groups of the set being scored (set_groups_of_set)


def set_groups_of_set(mapping: dict) -> None:
    SET_GROUPS.clear()
    SET_GROUPS.update(mapping)
    cluster.set_groups(mapping)


@contextmanager
def groups_for(pres: str):
    """The bootstrap groups of rows of one presentation: every instance its own group for "data", else the prompt
    groups of the set."""
    cluster.set_groups({} if pres == "data" else SET_GROUPS)
    try:
        yield
    finally:
        cluster.set_groups(SET_GROUPS)


# ---------------------------------------------------------------- instances and answers

def class_and_k(name: str, k_families: int) -> tuple:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"not a rich-model name: {name}")
    return m.group(2), k_families


def load_set(root: Path, five_root: Path | None, summary_work: Path, phase: str) -> list:
    """The instances of one rich-model directory (instances.txt, instances/<name>/), with the truth of
    llmdw.scoring.rich_models.load_instances (gain table, lookahead at the HiGHS dual, GCG anchors, family order) and,
    when five_root/<name>/five_duals.json exists, a copy of the truth with the five-dual lookahead values
    (llmdw.scoring.experiments._five_list)."""
    names = [x for x in (root / "instances.txt").read_text().split() if x]
    out = []
    for n in names:
        d = root / "instances" / n
        gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
        truth = analyze_support.template_truth(d)
        la = json.loads((d / "lookahead.json").read_text(encoding="utf-8"))
        truth["lookahead"] = [(frozenset(r["families"]), r["lookahead"] or 0.0) for r in la["table"].values()]
        gp = d / "gcg_anchors.json"
        truth["gcg"] = {b: frozenset(r["convexified"]) for b, r in json.loads(gp.read_text()).items()
                        if r.get("convexified")} if gp.exists() else {}
        model = summaries.load(summary_work, phase, n, d / "model.mps")
        truth["family_order"] = relabel.file_order(model, truth["families"])
        k = len(truth["families"])
        cls, _ = class_and_k(n, k)
        complete_to = int(gt.get("table_complete_to") or k - 1)
        five = None
        fp = five_root / n / "five_duals.json" if five_root else None
        if fp is not None and fp.exists():
            five = {**truth, "lookahead": experiments._five_list(fp, k)}
        out.append({"name": n, "class": cls, "k": k, "dir": d, "truth": truth, "five": five, "model": model,
                    "gt": gt, "complete_to": complete_to, "sizes": [s for s in SIZES if s <= complete_to]})
    return out


def load_answers(answer_roots: dict, insts: list, configs, answered_only: bool) -> tuple[dict, list]:
    """({(config, presentation): records of the original prompt}, status rows). answer_roots: {presentation: folder
    with one subfolder per configuration}. A configuration folder without any answer file is left out."""
    by = {i["name"]: {"summary": i["model"].summary} for i in insts}
    recs, status = {}, []
    for pres, root in answer_roots.items():
        for cfg in configs:
            d = Path(root) / cfg
            if not (d.exists() and any(d.rglob("*.json"))):
                status.append({"configuration": cfg, "presentation": pres, "planned_cells": 3 * len(insts),
                               "answer_files": 0, "unparseable": 0, "missing": 3 * len(insts),
                               "without_predicted_set": None, "scored": 0})
                continue
            rs = [r for r in results.load_answers(d, by) if r["condition"] == "original"]
            fams = {i["name"]: i["truth"]["families"] for i in insts}
            scored = [r for r in rs if not (answered_only and r["status"] == "missing")]
            status.append({"configuration": cfg, "presentation": pres, "planned_cells": len(rs),
                           "answer_files": sum(r["status"] != "missing" for r in rs),
                           "unparseable": sum(r["status"] == "unparseable" for r in rs),
                           "missing": sum(r["status"] == "missing" for r in rs),
                           "without_predicted_set": sum(1 for r in scored
                                                        if not results.support_anchor(r, fams[r["instance"]])),
                           "scored": len(scored)})
            recs[(cfg, pres)] = scored
    return recs, status


# ---------------------------------------------------------------- search orders

def _run(name, truth, s, spec):
    return results.run_value(results.search_runs(name, truth, s, spec)[0])


def search_values(insts: list, recs: dict) -> list:
    """One row per instance, s, order (and answer for JointDW and its interleaving): the evaluations until a best set
    is evaluated (mean over the family orders; None when an order fails). The memo of llmdw.search.relabel is cleared
    before the pass with the five-dual values (its keys do not hold the lookahead values)."""
    rows = []

    def add(inst, s, order, v, cfg="", pres="", rec=None, P=None):
        rows.append({"instance": inst["name"], "class": inst["class"], "k": inst["k"], "s": s, "order": order,
                     "config": cfg, "presentation": pres, "sample": "" if rec is None else rec.get("sample", 0),
                     "anchor": None if P is None else P, "evaluations": v})

    by_inst = defaultdict(list)
    for (cfg, pres), rs in recs.items():
        for r in rs:
            by_inst[r["instance"]].append((cfg, pres, r))
    relabel.clear()
    for inst in insts:
        n, t = inst["name"], inst["truth"]
        fams = sorted(t["families"])
        for s in inst["sizes"]:
            la = [T for T in results.lookahead_order(t) if len(T) <= s]
            add(inst, s, COVERING, _run(n, t, s, ("covering",)))
            add(inst, s, LOOKAHEAD, _run(n, t, s, ("lookahead",)) if la else None)
            for b, P in sorted(t["gcg"].items()):
                add(inst, s, GCG_LABELS.get(b, f"GCG anchor ({b})"), _run(n, t, s, ("model", frozenset(P))))
            for cfg, pres, r in by_inst[n]:
                P = results.support_anchor(r, fams)
                add(inst, s, JOINTDW, _run(n, t, s, ("model", P)), cfg, pres, r, P)
    relabel.clear()
    for inst in insts:
        n, t = inst["name"], inst["five"]
        if t is None:
            continue
        fams = sorted(t["families"])
        for s in inst["sizes"]:
            la = [T for T in results.lookahead_order(t) if len(T) <= s]
            add(inst, s, FIVE, _run(n, t, s, ("lookahead",)) if la else None)
            for cfg, pres, r in by_inst[n]:
                P = results.support_anchor(r, fams)
                add(inst, s, INTER_FIVE, _run(n, t, s, ("interleaved", P)), cfg, pres, r, P)
    relabel.clear()
    return rows


def per_instance_values(rows: list) -> dict:
    """{(order, config, presentation, s): {instance: value}}: answers averaged over the samples with a value
    (None when none has one), as llmdw.scoring.rich_models.rich_tables."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["order"], r["config"], r["presentation"], r["s"])][r["instance"]].append(r["evaluations"])
    out = {}
    for key, d in acc.items():
        out[key] = {}
        for n, v in d.items():
            ok = [x for x in v if x is not None]
            out[key][n] = sum(ok) / len(ok) if ok else None
    return out


def order_keys(vals: dict) -> list:
    """The (order, config, presentation) keys in table order."""
    gcg = sorted({k[0] for k in vals if k[0].startswith("GCG anchor")})
    fixed = [(o, "", "") for o in [COVERING, LOOKAHEAD, FIVE] + gcg]
    ans = sorted({k[:3] for k in vals if k[0] in ANSWER_ORDERS},
                 key=lambda x: (ANSWER_ORDERS.index(x[0]), x[1], PRESENTATIONS.index(x[2])))
    return [k for k in fixed if any(v[:3] == k for v in vals)] + ans


# ---------------------------------------------------------------- groups and rows

def groups(insts: list) -> list:
    """(class label, k label, names): every class and k, every class, every k, all instances."""
    out = []
    ks = sorted({i["k"] for i in insts})
    for c in CLASSES:
        for k in ks:
            names = [i["name"] for i in insts if i["class"] == c and i["k"] == k]
            if names:
                out.append((c, str(k), names))
    for c in CLASSES:
        names = [i["name"] for i in insts if i["class"] == c]
        if names and len({i["k"] for i in insts if i["class"] == c}) > 1:
            out.append((c, "all", names))
    for k in ks:
        names = [i["name"] for i in insts if i["k"] == k]
        if names and len({i["class"] for i in insts if i["k"] == k}) > 1:
            out.append(("all", str(k), names))
    out.append(("all", "all", [i["name"] for i in insts]))
    return out


def _sel(d: dict, names) -> dict:
    return {n: d[n] for n in names if n in d}


def _applies(vals: dict, key: tuple, names) -> bool:
    """Whether the order key has a value (or a failure) on an instance of names at some s: orders without one (a GCG
    baseline without a representable decomposition, a configuration without answers) get no row."""
    return any(n in vals.get(key + (s,), {}) for s in SIZES for n in names)


def search_rows(insts: list, vals: dict, group_list: list) -> list:
    by = {i["name"]: i for i in insts}
    rows = []
    for cls, k, names in group_list:
        for s in SIZES:
            nc = [n for n in names if s not in by[n]["sizes"]]
            for o, cfg, pres in order_keys(vals):
                if not _applies(vals, (o, cfg, pres), names):
                    continue
                d = _sel(vals.get((o, cfg, pres, s), {}), names)
                ok = {n: v for n, v in d.items() if v is not None}
                base = {"class": cls, "k": k, "s": s, "order": o, "config": cfg, "presentation": pres,
                        "status": NOT_COMPUTED if len(nc) == len(names) else "computed",
                        "n_group": len(names), "n_not_computed": len(nc), "n_failed": len(d) - len(ok)}
                with groups_for(pres):
                    rows.append({**base, **experiments._se_row(ok, "evaluations")})
    return rows


def paired_rows(insts: list, vals: dict, group_list: list, base_order: str) -> list:
    by = {i["name"]: i for i in insts}
    rows = []
    for cls, k, names in group_list:
        for s in SIZES:
            nc = [n for n in names if s not in by[n]["sizes"]]
            B = _sel(vals.get((base_order, "", "", s), {}), names)
            for o, cfg, pres in order_keys(vals):
                if o == base_order:
                    continue
                if not _applies(vals, (o, cfg, pres), names):
                    continue
                A = _sel(vals.get((o, cfg, pres, s), {}), names)
                with groups_for(pres):
                    rows.append({"class": cls, "k": k, "s": s, "order": o, "config": cfg, "presentation": pres,
                                 "contrast": f"{o} minus {base_order}",
                                 "status": NOT_COMPUTED if len(nc) == len(names) else "computed",
                                 "n_group": len(names), "n_not_computed": len(nc), **experiments._paired(A, B)})
    return rows


def prediction_rows(insts: list, recs: dict, group_list: list) -> list:
    """hit@1 and recall (analyze_support.score) per configuration, presentation and group; the gain table is not
    passed to the score (its search fields use s = k - 1 on a complete table and are not used here)."""
    by = {i["name"]: i for i in insts}
    rows = []
    for (cfg, pres), rs in recs.items():
        sc = [analyze_support.score(r, {**by[r["instance"]]["truth"], "table": {}}) for r in rs]
        for cls, k, names in group_list:
            joint = {n for n in names if by[n]["truth"]["joint"]}
            h = results.per_instance(sc, "joint_hit1", "original", joint)
            rc = results.per_instance(sc, "joint_recall", "original", joint)
            with groups_for(pres):
                rows.append({"config": cfg, "presentation": pres, "class": cls, "k": k, "n_group": len(names),
                             "n_answers": sum(1 for r in rs if r["instance"] in joint),
                             **experiments._se_row(h, "hit1"),
                             **{a: b for a, b in experiments._se_row(rc, "recall").items()
                                if a not in ("n_instances", "n_groups")}})
    return rows


# ---------------------------------------------------------------- per-instance tables

def best_sets(truth: dict, s: int) -> set:
    return results.targets(results.restricted(truth["table"], s), "max")


def minimal(sets) -> list:
    return sorted((T for T in sets if not any(S < T for S in sets)), key=lambda T: (len(T), sorted(T)))


def _key(T) -> str:
    return "|".join(sorted(T))


def instance_rows(insts: list, draws: dict, prompt_group: dict) -> list:
    rows = []
    for inst in insts:
        n, t, gt = inst["name"], inst["truth"], inst["gt"]
        dr = draws.get(n, {})
        m = NAME_RE.match(n)
        j = int(m.group(3))
        lp, ip = Fraction(gt["lp_bound"]), Fraction(gt["ip_scip"]).limit_denominator(10 ** 6)
        best = max(t["table"].values())
        row = {"instance": n, "class": inst["class"], "k": inst["k"], "draw_index": j,
               "stream_seed": _stream_seed(inst["class"], inst["k"], j), "rows": dr.get("rows"),
               "columns": dr.get("columns"), "binaries": dr.get("binaries"), "lp_bound": float(lp),
               "ip_optimum": float(ip), "lp_gap_percent": 100 * gt["lp_gap"], "table_complete_to": inst["complete_to"],
               "sets_in_table": len(gt["gain"]), "n_joint_supports": len(t["joint"]),
               "joint_supports": " ".join(_key(W) for W in sorted(t["joint"], key=lambda W: (len(W), sorted(W)))),
               "gap_closed_by_best_percent": float(100 * Fraction(best) / abs(ip - lp)) if ip != lp else None}
        for s in SIZES:
            if s not in inst["sizes"]:
                for c in ("minimal_best_sets", "n_best_sets", "covering_number", "lookahead_first_is_best",
                          "five_dual_first_is_best"):
                    row[f"{c}_s{s}"] = NOT_COMPUTED
                continue
            g = best_sets(t, s)
            row[f"minimal_best_sets_s{s}"] = " ".join(_key(T) for T in minimal(g))
            row[f"n_best_sets_s{s}"] = len(g)
            row[f"covering_number_s{s}"] = len(search_sim.covering_order(sorted(t["families"]), s))
            orders = relabel.family_orders(n, results.base_order(t))
            tops = [relabel.lookahead_list(t["lookahead"], o, s) for o in orders]
            row[f"lookahead_first_is_best_s{s}"] = results._mean(float(bool(x) and x[0] in g) for x in tops)
            if inst["five"] is not None:
                tops = [relabel.lookahead_list(inst["five"]["lookahead"], o, s) for o in orders]
                row[f"five_dual_first_is_best_s{s}"] = results._mean(float(bool(x) and x[0] in g) for x in tops)
            else:
                row[f"five_dual_first_is_best_s{s}"] = None
        row["gcg_anchors"] = " ".join(sorted(t["gcg"]))
        row["prompt_group_original"] = prompt_group.get(n, "")
        rows.append(row)
    return rows


def _stream_seed(cls: str, k: int, j: int) -> int:
    from llmdw.instances import rich_models_large as rl

    return rl.SEED_BASE + 100 * k + 10 * CLASSES.index(cls) + j


def gain_table_rows(inst: dict) -> list:
    t, gt = inst["truth"], inst["gt"]
    la = {T: v for T, v in t["lookahead"]}
    five = {T: v for T, v in inst["five"]["lookahead"]} if inst["five"] else {}
    goals = {s: best_sets(t, s) for s in inst["sizes"]}
    rows = []
    for key, v in sorted(gt["gain"].items(), key=lambda kv: (len([x for x in kv[0].split("|") if x]), kv[0])):
        T = frozenset(x for x in key.split("|") if x)
        row = {"set": key, "size": len(T), "gain": v, "gain_value": float(Fraction(v)),
               "lookahead_highs_dual": la.get(T), "lookahead_five_duals": five.get(T)}
        for s in SIZES:
            row[f"best_s{s}"] = (T in goals[s]) if s in goals else NOT_COMPUTED
        rows.append(row)
    return rows


def prompt_groups(insts: list) -> dict:
    """Prompt groups of the original prompt (llmdw.scoring.cluster.prompt_groups on the planned messages of
    llmdw.providers.usage); the groups of the bootstrap (module docstring)."""
    from llmdw.providers import usage

    prompts = {i["name"]: {"original": usage.planned_messages(i["model"].summary, str(i["dir"]), "original")}
               for i in insts}
    return cluster.prompt_groups(prompts)


# ---------------------------------------------------------------- figure data

def published_groups(pub: Path) -> dict:
    """The prompt groups of the published rich models (prompt_groups.csv of the released data, phase rich_models)."""
    with open(pub / "prompt_groups.csv", newline="", encoding="utf-8") as fh:
        return {r["instance"]: r["group"] for r in csv.DictReader(fh) if r["phase"] == "rich_models"}


def cost_vs_k_rows(set_label: str, insts: list, vals: dict) -> list:
    rows = []
    by = {i["name"]: i for i in insts}
    cells = [(c, k) for c in CLASSES for k in sorted({i["k"] for i in insts if i["class"] == c})]
    for o, cfg, pres in order_keys(vals):
        for c, k in cells:
            names = [i["name"] for i in insts if i["class"] == c and i["k"] == k]
            for s in SIZES:
                if not _applies(vals, (o, cfg, pres), names):
                    continue
                nc = [n for n in names if s not in by[n]["sizes"]]
                d = _sel(vals.get((o, cfg, pres, s), {}), names)
                ok = {n: v for n, v in d.items() if v is not None}
                with groups_for(pres):
                    r = experiments._se_row(ok, "evaluations")
                rows.append({"set": set_label, "order": o, "config": cfg, "presentation": pres, "class": c, "k": k,
                             "s": s, "status": NOT_COMPUTED if len(nc) == len(names) else "computed",
                             "n_instances": r["n_instances"], "n_groups": r["n_groups"], "n_failed": len(d) - len(ok),
                             "mean_evaluations": r["evaluations_mean"], "se": r["evaluations_se"]})
    return rows


def stage_counts(k: int, s: int, w: int) -> list:
    """Queries of each stage e = 0, ..., w of the staged search (llmdw.search.relabel.staged_iter) for k families and
    a predicted set of w <= s families: the split a with the fewest queries, ties to the larger a."""
    out = []
    for e in range(w + 1):
        best = None
        for a in range(min(w, s - e), w - e - 1, -1):
            A = search_sim.greedy_covering_positions(w, a, w - e)
            R = ((),) if e == 0 else search_sim.greedy_covering_positions(k - w, s - a, e)
            if best is None or len(A) * len(R) < best:
                best = len(A) * len(R)
        out.append(best or 0)
    return out


def staged_queries(k: int, s: int, w: int, e: int) -> int:
    """Distinct sets among the queries of stages 0 to e of staged_iter (a repeated set is evaluated once)."""
    counts = stage_counts(k, s, w)
    order = list(relabel.staged_iter(list(range(k)), s, list(range(w))))
    return len(set(order[:sum(counts[:e + 1])]))


def misses(P: frozenset, goal: set) -> tuple:
    """(e, w): e = min over the best sets B of |B minus P|, w the smallest |B| with |B minus P| = e."""
    e = min(len(B - P) for B in goal)
    return e, min(len(B) for B in goal if len(B - P) == e)


def misses_rows(set_label: str, insts: list, rows: list, vals: dict) -> list:
    by = {i["name"]: i for i in insts}
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["order"] != JOINTDW or not r["anchor"]:
            continue
        inst = by[r["instance"]]
        e, w = misses(r["anchor"], best_sets(inst["truth"], r["s"]))
        acc[(inst["class"], inst["k"], r["s"], e, w)][r["instance"]].append(r["evaluations"])
    out = []
    for (c, k, s, e, w) in sorted(acc, key=lambda x: (CLASSES.index(x[0]), x[1], x[2], x[3], x[4])):
        d = acc[(c, k, s, e, w)]
        per = {}
        for n, v in d.items():
            ok = [x for x in v if x is not None]
            per[n] = sum(ok) / len(ok) if ok else None
        r = experiments._se_row(per, "evaluations")
        cov = vals.get((COVERING, "", "", s), {})
        out.append({"set": set_label, "class": c, "k": k, "s": s, "e": e, "best_set_size": w,
                    "n_predictions": sum(len(v) for v in d.values()), "n_instances": len(d), "n_groups": r["n_groups"],
                    "mean_evaluations": r["evaluations_mean"], "se": r["evaluations_se"],
                    "staged_queries_stages_0_to_e": staged_queries(k, s, w, e),
                    "mean_evaluations_covering": results._mean(cov.get(n) for n in d)})
    # a class and k whose gain table stops below s: one row that says so
    for inst_k in sorted({(i["class"], i["k"]) for i in insts if any(s not in i["sizes"] for s in SIZES)}):
        c, k = inst_k
        for s in SIZES:
            if all(s not in i["sizes"] for i in insts if i["class"] == c and i["k"] == k):
                out.append({"set": set_label, "class": c, "k": k, "s": s, "e": NOT_COMPUTED,
                            "best_set_size": NOT_COMPUTED, "n_predictions": 0, "n_instances": 0, "n_groups": 0})
    return out


# ---------------------------------------------------------------- table bodies

def _f(x, d=2):
    return tx.f2(x, d)


def _pm(m, se, star=""):
    return tx.pm(m, se, 2, star)


def _group_label(cls: str, k: str) -> str:
    c = "All classes" if cls == "all" else CLASS_LABELS[cls]
    return f"{c}, $k={k}$" if k != "all" else (f"{c}, all $k$" if cls != "all" else "All instances")


def _cfg_label(cfg: str, pres: str) -> str:
    return f"{results.NAMES.get(cfg, cfg)}, {PRESENTATION_LABELS[pres]}"


def _order_label(o: str, cfg: str, pres: str) -> str:
    return o if not cfg else f"{_cfg_label(cfg, pres)}, {o}"


def tex_instances(inst_rows: list, insts: list, partial: bool) -> str:
    head = ("% Experiment E instances. Columns: class; k; instances; sets in the gain table (sets of at most t "
            "families, t given); LP gap in percent (median); joint supports per instance (mean); minimal best sets "
            "for s = 2 and s = 3 per instance (mean); instances with a GCG anchor. Source: "
            "results/experiment_E/instances.csv" + (" (partial)" if partial else ""))
    lines = [head]
    for c in CLASSES:
        for k in sorted({i["k"] for i in insts if i["class"] == c}):
            rs = [r for r in inst_rows if r["class"] == c and r["k"] == k]
            t = rs[0]["table_complete_to"]
            gaps = sorted(r["lp_gap_percent"] for r in rs)
            med = gaps[len(gaps) // 2] if len(gaps) % 2 else (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2]) / 2
            cells = [CLASS_LABELS[c], str(k), str(len(rs)), f"{rs[0]['sets_in_table']} ($t={t}$)", _f(med, 1),
                     _f(sum(r["n_joint_supports"] for r in rs) / len(rs))]
            for s in SIZES:
                v = [r[f"minimal_best_sets_s{s}"] for r in rs]
                cells.append("n.c." if any(x == NOT_COMPUTED for x in v) else
                             _f(sum(len(x.split()) for x in v) / len(v)))
            cells.append(str(sum(1 for r in rs if r["gcg_anchors"])))
            lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def tex_predictions(pred: list, group_list: list, partial: bool) -> str:
    head = ("% Experiment E: first predicted joint support. Columns: configuration and presentation; hit@1; recall "
            "(mean $\\pm$ standard error, two-stage bootstrap over prompt groups and instances); instances. Source: results/experiment_E/predictions.csv"
            + (" (partial)" if partial else ""))
    lines = [head]
    for cls, k, _ in group_list:
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{_group_label(cls, k)}}}}} \\\\")
        for r in pred:
            if r["class"] == cls and r["k"] == k:
                lines.append(f"{_cfg_label(r['config'], r['presentation'])} & {_pm(r['hit1_mean'], r['hit1_se'])} & "
                             f"{_pm(r['recall_mean'], r['recall_se'])} & {r['n_instances']} \\\\")
    return "\n".join(lines) + "\n"


def _cell(rows, s, **key):
    r = next((x for x in rows if x["s"] == s and all(x[a] == b for a, b in key.items())), None)
    if r is None:
        return "--"
    if r["status"] == NOT_COMPUTED:
        return "n.c."
    m = r.get("evaluations_mean", r.get("mean_difference"))
    se = r.get("evaluations_se", r.get("se_difference"))
    return _pm(m, se, r.get("asterisk", ""))


def tex_search(srch: list, group_list: list, partial: bool) -> str:
    head = ("% Experiment E: evaluations until a best set is evaluated. Columns: order; s = 2; s = 3 (mean $\\pm$ "
            "standard error, two-stage bootstrap over prompt groups and instances; n.c.: not computed, the gain table holds the sets of at most 2 families; "
            "a pooled row leaves those instances out at s = 3, see n_not_computed); instances with a value at s = 2. "
            "Source: results/experiment_E/search_costs.csv"
            + (" (partial)" if partial else ""))
    lines = [head]
    for cls, k, _ in group_list:
        rows = [r for r in srch if r["class"] == cls and r["k"] == k]
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{_group_label(cls, k)}}}}} \\\\")
        keys = list(dict.fromkeys((r["order"], r["config"], r["presentation"]) for r in rows))
        for o, cfg, pres in keys:
            kk = dict(order=o, config=cfg, presentation=pres)
            n2 = next((r["n_instances"] for r in rows if r["s"] == 2 and r["order"] == o and r["config"] == cfg
                       and r["presentation"] == pres), "--")
            lines.append(f"{_order_label(o, cfg, pres)} & {_cell(rows, 2, **kk)} & {_cell(rows, 3, **kk)} & {n2} \\\\")
    return "\n".join(lines) + "\n"


def order_keys_of(rows: list) -> list:
    """The (order, config, presentation) keys of table rows in table order (order_keys)."""
    return order_keys({(r["order"], r["config"], r["presentation"], r["s"]): None for r in rows})


def tex_paired(p5: list, pc: list, group_list: list, partial: bool) -> str:
    head = ("% Experiment E: paired differences in evaluations until a best set. Columns: order; minus the five-dual "
            "lookahead order at s = 2, s = 3; minus the covering order at s = 2, s = 3 (mean $\\pm$ standard error, "
            "two-stage bootstrap over prompt groups and instances; asterisk: 95 percent interval excludes 0; n.c.: not computed; a pooled row "
            "leaves the instances not computed out). Sources: "
            "results/experiment_E/paired_five_dual.csv, paired_covering.csv" + (" (partial)" if partial else ""))
    lines = [head]
    for cls, k, _ in group_list:
        a = [r for r in p5 if r["class"] == cls and r["k"] == k]
        b = [r for r in pc if r["class"] == cls and r["k"] == k]
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{_group_label(cls, k)}}}}} \\\\")
        present = {(r["order"], r["config"], r["presentation"]) for r in a + b}
        keys = [x for x in order_keys_of(a + b) if x in present]
        for o, cfg, pres in keys:
            kk = dict(order=o, config=cfg, presentation=pres)
            cells = [_cell(a, 2, **kk), _cell(a, 3, **kk), _cell(b, 2, **kk), _cell(b, 3, **kk)]
            lines.append(f"{_order_label(o, cfg, pres)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- driver

def score_set(root: Path, five_root: Path, answer_roots: dict, summary_work: Path, phase: str, configs,
              answered_only: bool) -> dict:
    insts = load_set(root, five_root, summary_work, phase)
    recs, status = load_answers(answer_roots, insts, configs, answered_only)
    rows = search_values(insts, recs)
    return {"insts": insts, "recs": recs, "status": status, "rows": rows, "vals": per_instance_values(rows)}


def run(args) -> dict:
    from llmdw.providers import settings

    if args.family_orders == "identity":
        relabel.configure(identity=True)
    else:
        relabel.configure(int(args.family_orders))
    results.NAMES.clear()
    results.NAMES.update(settings.names())
    set_groups_of_set({})
    work, out = Path(args.work).expanduser(), Path(args.out).expanduser()
    configs = [c for c in args.configs.split(",") if c]
    partial = bool(args.answered_only)
    E = score_set(work / E_ROOT, work / E_FIVE, {p: work / d for p, d in E_ANSWERS.items()}, work / E_ROOT,
                  "rich_models_large", configs, partial)
    insts, recs, vals = E["insts"], E["recs"], E["vals"]
    E_groups = prompt_groups(insts)
    set_groups_of_set(E_groups)
    gl = groups(insts)
    draws = {d["name"]: d for d in json.loads((work / E_ROOT / "draws.json").read_text())["draws"]
             if d.get("selected")}
    inst_rows = instance_rows(insts, draws, E_groups)
    pred = prediction_rows(insts, recs, gl)
    srch = search_rows(insts, vals, gl)
    p5 = paired_rows(insts, vals, gl, FIVE)
    pc = paired_rows(insts, vals, gl, COVERING)
    status = [{"set": SET_E, **r} for r in E["status"]]
    fig = cost_vs_k_rows(SET_E, insts, vals)
    mis = misses_rows(SET_E, insts, E["rows"], vals)
    if args.published:
        pub = Path(args.published).expanduser()
        set_groups_of_set(published_groups(pub))
        P = score_set(pub / P_ROOT, pub / P_FIVE, {p: pub / d for p, d in P_ANSWERS.items()}, pub, "rich_models",
                      configs, False)
        status += [{"set": SET_PUBLISHED, **r} for r in P["status"]]
        fig = cost_vs_k_rows(SET_PUBLISHED, P["insts"], P["vals"]) + fig
        mis = misses_rows(SET_PUBLISHED, P["insts"], P["rows"], P["vals"]) + mis
    out.mkdir(parents=True, exist_ok=True)
    gdir = out / "gain_tables"
    gdir.mkdir(exist_ok=True)
    for inst in insts:
        results.write_csv(gdir / f"{inst['name']}.csv", gain_table_rows(inst))
    for name, rows in (("instances", inst_rows), ("answers", status), ("predictions", pred), ("search_costs", srch),
                       ("paired_five_dual", p5), ("paired_covering", pc)):
        results.write_csv(out / f"{name}.csv", rows)
    marker = out / "PARTIAL.txt"
    if partial:
        marker.write_text("Only the answer cells with a file were scored (--answered-only); the other cells are left "
                          "out instead of counting as failed answers.\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()
    tex = Path(args.tex_dir).expanduser() if args.tex_dir else out / "tables/new"
    figd = Path(args.figure_dir).expanduser() if args.figure_dir else out / "figures/data"
    tex.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    (tex / "experiment_E_instances.tex").write_text(tex_instances(inst_rows, insts, partial), encoding="utf-8")
    (tex / "experiment_E_predictions.tex").write_text(tex_predictions(pred, gl, partial), encoding="utf-8")
    (tex / "experiment_E_search.tex").write_text(tex_search(srch, gl, partial), encoding="utf-8")
    (tex / "experiment_E_paired.tex").write_text(tex_paired(p5, pc, gl, partial), encoding="utf-8")
    results.write_csv(figd / "experiment_E_cost_vs_k.csv", fig)
    results.write_csv(figd / "search_cost_vs_misses.csv", mis)
    return {"status": status, "instances": inst_rows, "predictions": pred, "search": srch, "paired_five_dual": p5,
            "paired_covering": pc, "cost_vs_k": fig, "misses": mis}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="data/work of the released data (holds rich_models_large/)")
    ap.add_argument("--published", help="the released data (holds work/rich_models and prompt_groups.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex-dir", help="folder of the table bodies (default <out>/tables/new)")
    ap.add_argument("--figure-dir", help="folder of the figure data (default <out>/figures/data)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="model identifiers, comma-separated")
    ap.add_argument("--answered-only", action="store_true",
                    help="score only the answer cells with a file (missing cells are left out, not failed)")
    ap.add_argument("--family-orders", default=str(relabel.PERMUTATIONS),
                    help="family orders per instance (llmdw.search.relabel), or 'identity'")
    args = ap.parse_args(argv)
    res = run(args)
    for r in res["status"]:
        print(json.dumps(r))
    print(f"wrote the tables of the rich models with 12 to 16 families to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def tex_from_csv(out: Path, tex: Path) -> list:
    """The four table bodies rebuilt from the result CSVs (instances.csv, predictions.csv, search_costs.csv,
    paired_five_dual.csv, paired_covering.csv); the class and k groups in the order of predictions.csv."""
    from llmdw.tables.experiments import typed_rows

    out, tex = Path(out), Path(tex)
    inst_rows = typed_rows(out / "instances.csv")
    insts = [{"class": r["class"], "k": r["k"]} for r in inst_rows]
    pred = typed_rows(out / "predictions.csv")
    gl = [(c, k, None) for c, k in dict.fromkeys((r["class"], r["k"]) for r in pred)]
    srch, p5, pc = (typed_rows(out / f) for f in ("search_costs.csv", "paired_five_dual.csv", "paired_covering.csv"))
    for r in pred + srch + p5 + pc:                  # text columns written empty are "" in the tables, not None
        for k in ("order", "config", "presentation", "contrast", "status"):
            if k in r and r[k] is None:
                r[k] = ""
    bodies = {"experiment_E_instances.tex": tex_instances(inst_rows, insts, False),
              "experiment_E_predictions.tex": tex_predictions(pred, gl, False),
              "experiment_E_search.tex": tex_search(srch, gl, False),
              "experiment_E_paired.tex": tex_paired(p5, pc, gl, False)}
    for name, body in bodies.items():
        (tex / name).write_text(body, encoding="utf-8")
    return list(bodies)
