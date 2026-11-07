"""Scoring of the two added experiments on the existing instances.

A (control)    the data variant (the complete instance in CPLEX LP format) against its two controls, data permuted and
               data anonymized (llmdw.prompts.counterfactual maps applied to the tables and to the row and column names
               of the LP section), on the 24 rich models, for GPT-6.1 Sol and Claude Sonnet 5.5: hit@1, recall and the
               JointDW search cost at s = 2 and 3 (llmdw.scoring.rich_models.search), per class and pooled, and the
               paired differences against the data variant. Every prompt of the data variant and of its controls is
               distinct, so the bootstrap clusters are single instances (as for the paper's tables of the data variant).
B (five duals) on the templates, the 102 scored MIPLIB-NL models and the 24 rich models: hit@1 and recall of the first
               set of the lookahead order at the HiGHS dual and at the best of five optimal duals
               (llmdw.lookahead.five_duals), their search cost at s = 2, 3 and k - 1, the paired difference five-dual
               minus single-dual, and JointDW (each configuration, original presentation) and the interleaved orders
               (with the single-dual and with the five-dual lookahead) minus the five-dual lookahead. The gain tables
               are those of the paper (for MIPLIB-NL with the single-dual lookahead values merged as lower bounds, as in
               llmdw.scoring.analyze_support.miplib_truth); the five-dual values enter only the order. Bootstraps
               resample the prompt groups of each set (llmdw.scoring.cluster).

Per-instance values are means over the three samples and over the 1,000 family orders (llmdw.search.relabel); hit@1 of a
lookahead order is the share of family orders under which its first set is a true joint support, and the recall of that
first set is the same share divided by the number of true joint supports. The detection rate of the paper (true joint
supports that are zero-marginal groups of the lookahead table) is also given for both duals.

Usage
-----
python -m llmdw.scoring.experiments control --work DATA --out results ...
python -m llmdw.scoring.experiments five_duals --work DATA --five-duals work/five_duals --out results ...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from llmdw.scoring import analyze_support, cluster, results

CONTROL = {"original": "data", "permuted": "data, permuted", "anonymized": "data, anonymized"}
SIZES = ("2", "3", "k-1")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _se_row(values: dict, prefix: str) -> dict:
    lo, hi, se = results.boot_ci_se(sorted(values.items()))
    return {f"{prefix}_mean": _mean(values.values()), f"{prefix}_se": se, f"{prefix}_lo": lo, f"{prefix}_hi": hi,
            "n_instances": len(values)}


def _paired(a: dict, b: dict) -> dict:
    both = sorted(n for n in a if a[n] is not None and b.get(n) is not None)
    diffs = [(n, a[n] - b[n]) for n in both]
    lo, hi, se = results.boot_ci_se(diffs)
    return {"n_instances": len(both), "mean_a": _mean(a[n] for n in both), "mean_b": _mean(b[n] for n in both),
            "mean_difference": _mean(d for _, d in diffs), "se_difference": se, "lo": lo, "hi": hi,
            "asterisk": "*" if lo is not None and (lo > 0 or hi < 0) else ""}


def _by_instance(rows, key, cond=None):
    acc = defaultdict(list)
    for r in rows:
        if (cond is None or r["condition"] == cond) and r.get(key) is not None:
            acc[r["instance"]].append(float(r[key]))
    return {n: sum(v) / len(v) for n, v in acc.items()}


# ---------------------------------------------------------------- A

def control_tables(work: Path, data_dirs: dict, control_dirs: dict) -> dict:
    """data_dirs, control_dirs: {official name: answer folder}. Returns {table name: rows}."""
    from llmdw.scoring import rich_models

    cluster.set_groups({})
    insts = rich_models.load_instances(work)
    by = {i["name"]: i for i in insts}
    rinst = {i["name"]: {"summary": i["model"].summary, "truth": i["truth"], "class": i["class"]} for i in insts}
    recs = {}
    for cfg in data_dirs:
        r = [x for x in results.load_answers(Path(data_dirs[cfg]), rinst) if x["condition"] == "original"]
        r += [x for x in results.load_answers(Path(control_dirs[cfg]), rinst) if x["condition"] in ("permuted",
                                                                                                    "anonymized")]
        recs[cfg] = r
    scored = {cfg: [{**analyze_support.score(r, by[r["instance"]]["truth"]), "status": r["status"],
                     "n_supports": len(r.get("supports") or [])} for r in rs] for cfg, rs in recs.items()}
    srch = rich_models.search(insts, recs)["per"]
    classes = [("pooled", None)] + [(c, c) for c in rich_models.CLASSES]
    pred, cost, paired, nojoint = [], [], [], []
    for cfg in recs:
        joint = [n for n, i in by.items() if i["truth"]["joint"]]
        for cls, c in classes:
            names = [n for n in joint if c is None or by[n]["class"] == c]
            all_names = [n for n in by if c is None or by[n]["class"] == c]
            vals = {}
            for cond in CONTROL:
                h = {n: v for n, v in _by_instance(scored[cfg], "joint_hit1", cond).items() if n in names}
                rc = {n: v for n, v in _by_instance(scored[cfg], "joint_recall", cond).items() if n in names}
                vals[cond] = {"hit1": h, "recall": rc}
                pred.append({"configuration": cfg, "presentation": CONTROL[cond], "class": cls,
                             **{k: v for k, v in _se_row(h, "hit1").items()},
                             **{k: v for k, v in _se_row(rc, "recall").items() if k != "n_instances"}})
                for s in (2, 3):
                    for meth in ("jointdw", "interleaved"):
                        e = _by_instance([x for x in srch if x["config"] == cfg and x["method"] == meth and x["s"] == s
                                          and x["instance"] in all_names], "evaluations", cond)
                        vals[cond][(meth, s)] = e
                        cost.append({"configuration": cfg, "presentation": CONTROL[cond], "class": cls, "s": s,
                                     "method": meth, **_se_row(e, "evaluations"),
                                     "n_failed": sum(1 for x in srch if x["config"] == cfg and x["method"] == meth
                                                     and x["s"] == s and x["instance"] in all_names
                                                     and x["condition"] == cond and x["evaluations"] is None)})
            for cond in ("permuted", "anonymized"):
                for metric in ["hit1", "recall"] + [(m, s) for s in (2, 3) for m in ("jointdw", "interleaved")]:
                    label = metric if isinstance(metric, str) else f"evaluations {metric[0]} s={metric[1]}"
                    paired.append({"configuration": cfg, "class": cls, "metric": label,
                                   "contrast": f"data minus {CONTROL[cond]}",
                                   **_paired(vals["original"][metric], vals[cond][metric])})
        for cond in CONTROL:
            rs = [r for r in scored[cfg] if r["condition"] == cond]
            answered = [r for r in rs if r["status"] == "answered"]
            nojoint.append({"configuration": cfg, "presentation": CONTROL[cond], "answers": len(rs),
                            "answered": len(answered), "unparseable": sum(1 for r in rs if r["status"] == "unparseable"),
                            "missing": sum(1 for r in rs if r["status"] == "missing"),
                            "no_joint_support": sum(1 for r in answered if r["n_supports"] == 0),
                            "no_joint_support_share": (sum(1 for r in answered if r["n_supports"] == 0) / len(answered))
                            if answered else None})
    return {"control_prediction": pred, "control_search": cost, "control_paired": paired,
            "control_no_joint_support": nojoint}


# ---------------------------------------------------------------- B

def _five_list(path: Path, k: int, full_k: bool = False):
    t = json.loads(path.read_text())["table"]
    return sorted(((frozenset(r["families"]), r["lookahead"] or 0.0) for r in t.values()
                   if r["lookahead"] is not None and (full_k or len(r["families"]) < k)), key=lambda kv: -kv[1])


def _phase_instances(work: Path, phase: str) -> dict:
    if phase == "rich_models":
        from llmdw.scoring import rich_models

        return {i["name"]: {"summary": i["model"].summary, "truth": i["truth"], "class": i["class"]}
                for i in rich_models.load_instances(work)}
    return results.load_instances(work, phase)


def five_duals_tables(work: Path, five_root: Path, answer_dirs: dict) -> dict:
    """answer_dirs: {phase: {official name: folder of the original-presentation answers}}."""
    from llmdw.scoring import rich_models

    first, cost, paired, duals = [], [], [], []
    for phase in ("templates", "miplib", "rich_models"):
        insts = _phase_instances(work, phase)
        if phase == "rich_models":
            rich_models.use_prompt_groups(work)
        else:
            results.use_prompt_groups(work, phase, insts)
        five = {}
        for n, i in insts.items():
            p = five_root / phase / n / "five_duals.json"
            if p.exists():
                five[n] = {**i["truth"], "lookahead": _five_list(p, len(i["truth"]["families"]), phase == "templates")}
        joint = sorted(n for n, i in insts.items() if i["truth"]["joint"]
                       and (phase != "templates" or i["class"] in results.TRUE_JOINT_CLASSES))
        names = [n for n in joint if n in five]
        search_names = [n for n in insts if insts[n]["truth"]["joint"] and n in five] if phase != "rich_models" \
            else [n for n in insts if n in five]
        recs_by = {}
        for cfg, adir in answer_dirs[phase].items():
            if Path(adir).exists():
                recs_by[cfg] = [r for r in results.load_answers(Path(adir), insts) if r["condition"] == "original"]

        def one_pass(truth_of, label, methods):
            """Values under one lookahead (the memo of llmdw.search.relabel is cleared first; the family orders
            depend only on the instance name and are the same in both passes)."""
            from llmdw.search import relabel

            relabel.clear()
            h = {n: results.lookahead_hit1(n, truth_of(n)) for n in names}
            first.append({"set": phase, "order": label, **_se_row(h, "hit1"),
                          **{k: v for k, v in _se_row({n: h[n] / len(insts[n]["truth"]["joint"]) for n in names},
                                                      "recall").items() if k != "n_instances"}})
            for n in search_names:
                t = truth_of(n)
                k = len(t["families"])
                if not t["table"] or max(t["table"].values()) <= 0:
                    continue
                for size in SIZES:
                    s, full = (k - 1, True) if size == "k-1" else (min(int(size), k - 1), False)
                    per[(label, size)][n] = results.run_value(results.search_runs(n, t, s, ("lookahead",),
                                                                                  full=full)[0])
            for cfg, recs in recs_by.items():
                acc = defaultdict(lambda: defaultdict(list))
                for r in recs:
                    n = r["instance"]
                    if n not in search_names:
                        continue
                    t = truth_of(n)
                    k = len(t["families"])
                    if not t["table"] or max(t["table"].values()) <= 0:
                        continue
                    P = results.support_anchor(r, t["families"])
                    for size in SIZES:
                        s, full = (k - 1, True) if size == "k-1" else (min(int(size), k - 1), False)
                        for meth, kind in methods:
                            acc[(meth, size)][n].append(results.run_value(results.search_runs(n, t, s, (kind, P),
                                                                                              full=full)[0]))
                for key, d in acc.items():
                    cfg_vals[(cfg,) + key] = {n: (None if any(x is None for x in v) else sum(v) / len(v))
                                              for n, v in d.items()}

        per, cfg_vals = defaultdict(dict), {}
        one_pass(lambda n: insts[n]["truth"], "lookahead, HiGHS dual",
                 (("JointDW", "model"), ("interleaved, HiGHS dual", "interleaved")))
        one_pass(lambda n: five[n], "lookahead, best of five duals", (("interleaved, best of five duals", "interleaved"),))
        for (label, size), d in per.items():
            cost.append({"set": phase, "order": label, "configuration": "", "size_limit": size,
                         **_se_row({n: v for n, v in d.items() if v is not None}, "evaluations"),
                         "n_failed": sum(1 for v in d.values() if v is None)})
        for (cfg, meth, size), d in sorted(cfg_vals.items()):
            cost.append({"set": phase, "order": meth, "configuration": cfg, "size_limit": size,
                         **_se_row({n: v for n, v in d.items() if v is not None}, "evaluations"),
                         "n_failed": sum(1 for v in d.values() if v is None)})
        for size in SIZES:
            five_d = per.get(("lookahead, best of five duals", size), {})
            paired.append({"set": phase, "size_limit": size, "configuration": "",
                           "contrast": "lookahead, best of five duals minus lookahead, HiGHS dual",
                           **_paired(five_d, per.get(("lookahead, HiGHS dual", size), {}))})
            for (cfg, meth, sz), d in sorted(cfg_vals.items()):
                if sz == size:
                    paired.append({"set": phase, "size_limit": size, "configuration": cfg,
                                   "contrast": f"{meth} minus lookahead, best of five duals", **_paired(d, five_d)})
        # dual multiplicity and limits
        for n in sorted(five):
            p = json.loads((five_root / phase / n / "five_duals.json").read_text())
            t = insts[n]["truth"]
            tab = t["table"]
            best = results.targets(tab, "max") if tab and max(tab.values()) > 0 else set()
            vals_differ = False
            for key, row in p["table"].items():
                if frozenset(row["families"]) in best:
                    v = [x for x in row["per_dual"] if x is not None]
                    if v and max(v) - min(v) > 1e-9 * max(1.0, abs(max(v))):
                        vals_differ = True
            limits = [x for x in p["limits"] if x["truncated"] or x["incomplete_sets"] or x["missing_sets"]
                      or x["status"] != "Optimal"]
            duals.append({"set": phase, "instance": n, "k": len(p["families"]),
                          "duals_differing_from_highs": sum(1 for x in p["duals"][1:] if x.get("differs_from_highs")),
                          "components_at_box": max((x.get("components_at_box") or 0) for x in p["duals"][1:]),
                          "best_set_values_differ": vals_differ,
                          "lp_bound_agrees": p["lp_bound_agrees"], "recorded_lookahead_truncated": p["recorded_truncated"],
                          "time_limit_hit": bool(limits),
                          "sets_without_optimal_blocks": sum(x["sets_without_optimal_blocks"] for x in p["limits"]),
                          "limits": json.dumps(limits) if limits else ""})
        missing = sorted(set(insts) - set(five))
        if missing:
            duals.append({"set": phase, "instance": "(no five-dual result)", "limits": " ".join(missing)})
    return {"five_duals_first_set": first, "five_duals_search": cost, "five_duals_paired": paired,
            "five_duals_duals": duals}


def write(tables: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        results.write_csv(out / f"{name}.csv", rows)


def _pairs(xs):
    return dict(x.split("=", 1) for x in xs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("control", "five_duals"))
    ap.add_argument("--work", required=True, help="the root that holds work/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", nargs="*", default=[], help="control: NAME=folder of the data-variant answers")
    ap.add_argument("--controls", nargs="*", default=[], help="control: NAME=folder of the control answers")
    ap.add_argument("--five-duals", help="five_duals: root of the five-dual results")
    ap.add_argument("--answers", nargs="*", default=[], help="five_duals: SET:NAME=folder (SET templates, miplib, "
                                                              "rich_models)")
    args = ap.parse_args(argv)
    work = Path(args.work).expanduser()
    if args.cmd == "control":
        write(control_tables(work, _pairs(args.data), _pairs(args.controls)), Path(args.out))
    else:
        dirs = defaultdict(dict)
        for x in args.answers:
            ph, rest = x.split(":", 1)
            k, v = rest.split("=", 1)
            dirs[ph][k] = v
        write(five_duals_tables(work, Path(args.five_duals), dirs), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
