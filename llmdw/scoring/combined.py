"""Two offline combinations of the language model, the five-dual lookahead and the learned structural predictor
(no calls, no new evaluations; every value comes from the recorded answers and gain tables).

union-first. For an answer (GPT-6.1 Sol, one configuration) with JointDW anchor P (results.support_anchor) and a
family order, F5 is the first set of the five-dual lookahead order and U = P | F5. If 0 < |U| <= s, U is evaluated
first, then the interleaved order of JointDW and the five-dual lookahead (relabel.alternate of the staged order around
P and the five-dual lookahead order), repeats skipped; otherwise the interleaved order unchanged. The three-source
version adds L1, the first set of the structural learned predictor (llmdw.scoring.learned, variant structural), to U.
L1 comes from the split that holds the instance out: leave_one_class_out for the rich models and the templates of the
three classes (clsp, ndp, cfl), miplibnl_leave_one_out for MIPLIB-NL, and for the other templates a leave-one-out over
all other templates and rich models.

stacked. The structural+lookahead learned predictor with two features added per set T, from the answers of the
configuration (three samples, averaged; a missing or unparseable answer counts 0): the share of samples whose first
joint support is contained in T, and the share of samples that propose a reformulation (a candidate's convexified
set, results.cand_list) containing T. Trained in the four splits of llmdw.scoring.learned (HYPERPARAMETERS). The
answers of an instance are those of the configuration under evaluation: on the rich models the text or the data
variant; elsewhere the original presentation (rows of templates and MIPLIB-NL are labelled with the rich-model
answers that the training sets of that run used).

Size limits s = 2, 3 and k - 1 (k - 1: the whole table, as in Experiment B). s = k - 1 is defined only for the
templates and MIPLIB-NL: the gain tables of the rich models hold the sets of at most three families, so the rich models
have no k - 1 rows and are left out of every training set at k - 1. References: the five-dual lookahead order,
JointDW's interleaved order (staged order around P alternated with the five-dual lookahead order), and the best single
source of the cell: the single-source order with the lowest mean search cost over the instances of the cell (union-
first: JointDW staged order and five-dual lookahead order; three-source: these and the structural learned order; stacked:
JointDW staged order, five-dual lookahead order and the structural+lookahead learned order). The best single source is
chosen per cell after the fact, so the comparison against it favors the single source. Values are means over the family
orders of llmdw.search.relabel and over the samples; paired differences with the cluster bootstrap. An instance whose
order reaches no target in some family order counts as a failure (n_failed) and is left out of that mean.

Usage
-----
python -m llmdw.scoring.combined --work DATA --five-duals DIR --out OUT [--sets templates,rich_models,miplib] [--threads 2]
    [--mps-dir DIR] --jointdw SET:LABEL=FOLDER ...
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from llmdw.scoring import learned, results
from llmdw.search import relabel

SIZES = ("2", "3", "k-1")
ALL_SETS = 99            # features of every set of at most k - 1 families


def defined(it: dict, size: str) -> bool:
    """Whether the size limit is defined for the instance (no s = k - 1 for the rich models)."""
    return not (size == "k-1" and it["set"] == "rich_models")


def has_target(it: dict, size: str) -> bool:
    """Whether the table of the instance has a positive gain among the sets of the size limit (otherwise the instance
    has no target there and is left out, as in Experiment B)."""
    s, full = size_spec(len(it["truth"]["families"]), size)
    return _table(it["truth"], s, full)[0] is not None


def size_spec(k: int, size: str):
    """(s, full) as in Experiment B: k - 1 is the whole table."""
    return (k - 1, True) if size == "k-1" else (min(int(size), k - 1), False)


def _table(t, s, full):
    tab = t["table"] if full else results.restricted(t["table"], s)
    if not tab or max(tab.values()) <= 0:
        return None, None
    return tab, results.targets(tab, "max")


def _value(name, t, key, fn):
    return results.run_value(relabel.runs(("combined",) + key, name, results.base_order(t), fn)[0])


def _first(values, order, la_size):
    la = relabel.lookahead_list(values, order, la_size)
    return la[0] if la else frozenset()


# ---------------------------------------------------------------- learned models of any size limit

def _xrow(it, T, variant, extra):
    row = [it["features"][T][i] for i in learned._columns(variant)]
    if extra is not None:
        row += extra.get(T, [0.0, 0.0])
    return row


def fit_size(train: list, size: str, variant: str, threads: int, extra: dict | None = None):
    """XGBoost (learned.HYPERPARAMETERS) on the labelled sets of at most s families of each training instance
    (s of the instance for size "k-1"); extra: {name: {T: [two stacked features]}} or None."""
    import xgboost

    X, y = [], []
    for it in train:
        if not defined(it, size):
            continue
        s, _ = size_spec(len(it["truth"]["families"]), size)
        for T, v in learned.labels(it["truth"], s).items():
            if T in it["features"]:
                X.append(_xrow(it, T, variant, extra.get(it["name"]) if extra is not None else None))
                y.append(v)
    if not X:
        raise ValueError(f"no labelled sets to train on at size limit {size}")
    m = xgboost.XGBRegressor(**learned.HYPERPARAMETERS, n_jobs=threads)
    m.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return m


def predict_size(m, it, size: str, variant: str, extra: dict | None = None) -> list:
    s, _ = size_spec(len(it["truth"]["families"]), size)
    sets = [T for T in it["features"] if len(T) <= s]
    if not sets:
        return []
    ex = extra.get(it["name"]) if extra is not None else None
    p = m.predict(np.asarray([_xrow(it, T, variant, ex) for T in sets], dtype=float))
    return [(T, float(v)) for T, v in zip(sets, p)]


def stacked_features(it: dict, recs: list) -> dict:
    """{T: [share of samples whose first joint support is in T, share proposing a reformulation that contains T]}."""
    fams = it["truth"]["families"]
    rs = [r for r in recs if r["instance"] == it["name"] and r.get("condition") == "original"]
    n = len(rs) or 1
    firsts = [frozenset((r.get("supports") or [{}])[0].get("families") or ()) if r.get("supports") else None
              for r in rs]
    reforms = [results.cand_list(r, fams) for r in rs]
    out = {}
    for T in it["features"]:
        a = sum(1 for f in firsts if f and f <= T) / n
        b = sum(1 for R in reforms if any(T <= x for x in R)) / n
        out[T] = [a, b]
    return out


def third_source_folds(items: dict):
    """(test names, training names) giving the structural predictor of the three-source version (module docstring)."""
    for _, test, train in learned.split_folds(items, "leave_one_class_out"):
        yield test, train
    for _, test, train in learned.split_folds(items, "miplibnl_leave_one_out"):
        yield test, train
    in_classes = set(learned.TEMPLATE_OF_CLASS.values())
    pool = [n for n, it in items.items() if it["set"] in ("templates", "rich_models")]
    for n, it in sorted(items.items()):
        if it["set"] == "templates" and it["class"] not in in_classes:
            yield [n], [m for m in pool if m != n]


# ---------------------------------------------------------------- per-instance values

def answer_values(it, recs, size, pred3=None) -> dict:
    """Means over the samples of one configuration: JointDW staged order, JointDW interleaved with the five-dual
    lookahead, union-first (two and three sources) and the shares of family orders with |U| <= s."""
    t, name = it["truth"], it["name"]
    k = len(t["families"])
    s, full = size_spec(k, size)
    la_size = len(t["families"]) if full else s
    tab, goal = _table(t, s, full)
    five = it["five"]
    rs = [r for r in recs if r["instance"] == name and r.get("condition") == "original"]
    if tab is None or five is None or not rs:
        return {}
    acc = defaultdict(list)
    orders = relabel.family_orders(name, results.base_order(t))
    for r in rs:
        P = results.support_anchor(r, t["families"])
        pk = tuple(sorted(P))

        def five_order(order):
            return relabel.lookahead_list(five, order, la_size)

        def inter(order):
            return relabel.alternate(relabel.model_iter(order, s, P), five_order(order))

        def union(order, extra_first):
            U = P | _first(five, order, la_size)
            if extra_first is not None:
                U = U | extra_first(order)
            if 0 < len(U) <= s:
                return itertools.chain([U], inter(order))
            return inter(order)

        hit = lambda it_: relabel.first_hit(it_, tab, goal, None, results.EVAL_LIMIT_S)  # noqa: E731
        acc["jointdw staged"].append(_value(name, t, ("staged", name, size, pk),
                                            lambda o: hit(relabel.model_iter(o, s, P))))
        acc["jointdw interleaved"].append(_value(name, t, ("inter", name, size, pk), lambda o: hit(inter(o))))
        acc["union-first"].append(_value(name, t, ("union2", name, size, pk), lambda o: hit(union(o, None))))
        acc["share |U|<=s"].append(float(np.mean([0 < len(P | _first(five, o, la_size)) <= s for o in orders])))
        if pred3 is not None:
            l1 = lambda o: _first(pred3, o, la_size)  # noqa: E731
            acc["union-first three-source"].append(_value(name, t, ("union3", name, size, pk),
                                                          lambda o: hit(union(o, l1))))
            acc["share |U3|<=s"].append(float(np.mean([0 < len(P | _first(five, o, la_size) | l1(o)) <= s
                                                       for o in orders])))
    return {m: (None if any(x is None for x in v) else float(np.mean(v))) for m, v in acc.items()}


def order_value(it, values, size, kind, tag, five=None):
    """Search cost of a learned order (kind "order"), of the staged search around its first set of two or more
    families ("staged"), or of that staged search interleaved with the five-dual lookahead ("interleaved")."""
    t, name = it["truth"], it["name"]
    s, full = size_spec(len(t["families"]), size)
    la_size = len(t["families"]) if full else s
    tab, goal = _table(t, s, full)
    if tab is None or values is None:
        return None

    def fn(order):
        la = relabel.lookahead_list(values, order, la_size)
        if kind == "order":
            it_ = la
        else:
            P = next((T for T in la if len(T) >= 2), la[0] if la else frozenset())
            st = relabel.model_iter(order, s, P)
            it_ = st if kind == "staged" else relabel.alternate(st, relabel.lookahead_list(five, order, la_size))
        return relabel.first_hit(it_, tab, goal, None, results.EVAL_LIMIT_S)
    return _value(name, t, ("order", kind, name, size, tag), fn)


# ---------------------------------------------------------------- run

def _cells(items, names, st):
    classes = sorted({items[n]["class"] for n in names})
    for cls in ["all"] + classes:
        yield cls, [n for n in names if cls == "all" or items[n]["class"] == cls]


def _best(cands: dict, names: list):
    """The single source with the lowest mean over the instances of the cell (ties: first listed)."""
    means = {}
    for lab, v in cands.items():
        xs = [v[n] for n in names if v.get(n) is not None]
        means[lab] = float(np.mean(xs)) if xs else None
    ok = {k: m for k, m in means.items() if m is not None}
    return (min(ok, key=lambda k: ok[k]) if ok else None), means


def run(work: Path, five_root: Path | None, out: Path, sets: list, threads: int, jointdw: dict,
        mps_dir: Path | None = None, log=print, features: Path | None = None, write_features: bool = False) -> dict:
    items = learned.load_items(work, sets, five_root, mps_dir)
    learned.add_features(items, ALL_SETS, features, write_features)
    log(f"{len(items)} instances: " + ", ".join(f"{st} {sum(1 for i in items.values() if i['set'] == st)}"
                                                for st in sets))
    answers = {}
    for (st, lab), folder in jointdw.items():
        insts = {n: {"summary": it["summary"]} for n, it in items.items() if it["set"] == st}
        answers[(st, lab)] = results.load_answers(Path(folder), insts) if Path(folder).exists() else []
    rich_cfgs = [lab for (st, lab) in answers if st == "rich_models"] or [None]
    tables = defaultdict(list)

    # third source: structural learned predictor of the split that holds the instance out
    pred3 = {}
    for size in SIZES:
        for test, train in third_source_folds(items):
            test = [n for n in test if defined(items[n], size)]
            if not test or not any(defined(items[n], size) for n in train):
                continue
            m = fit_size([items[n] for n in train], size, "structural", threads)
            for n in test:
                pred3[(n, size)] = predict_size(m, items[n], size, "structural")
    log("third source trained")

    # union-first
    for st in sets:
        names_st = [n for n, it in items.items() if it["set"] == st]
        learned._group_rows(work, st, {n: items[n] for n in names_st})
        for (ast, cfg), recs in answers.items():
            if ast != st:
                continue
            for size in SIZES:
                if not all(defined(items[n], size) for n in names_st):
                    continue
                per = {n: answer_values(items[n], recs, size, pred3.get((n, size))) for n in names_st}
                five = {n: order_value(items[n], items[n]["five"], size, "order", ("five",)) for n in names_st}
                l3 = {n: order_value(items[n], pred3.get((n, size)), size, "order", ("third",))
                      if pred3.get((n, size)) is not None else None for n in names_st}
                for cls, names in _cells(items, [n for n in names_st if per[n]], st):
                    base = {"set": st, "class": cls, "configuration": cfg, "size_limit": size}
                    meth = {m: {n: per[n].get(m) for n in names} for m in
                            ("union-first", "union-first three-source", "jointdw interleaved", "jointdw staged")}
                    meth["five-dual lookahead order"] = {n: five[n] for n in names}
                    meth["structural learned order (third source)"] = {n: l3[n] for n in names}
                    for m, v in meth.items():
                        tables["union_search"].append({**base, "method": m, **learned._se(v, "evaluations"),
                                                       "n_failed": sum(1 for x in v.values() if x is None)})
                    for share in ("share |U|<=s", "share |U3|<=s"):
                        tables["union_search"].append({**base, "method": share, **learned._se(
                            {n: per[n].get(share) for n in names}, "share")})
                    for combo, singles in (("union-first", ("jointdw staged", "five-dual lookahead order")),
                                           ("union-first three-source", ("jointdw staged", "five-dual lookahead order",
                                                                         "structural learned order (third source)"))):
                        best, means = _best({x: meth[x] for x in singles}, names)
                        refs = [("five-dual lookahead order", meth["five-dual lookahead order"]),
                                ("jointdw interleaved", meth["jointdw interleaved"])]
                        if best:
                            refs.append((f"best single source ({best})", meth[best]))
                        for lab, ref in refs:
                            tables["union_paired"].append({**base, "method": combo, "contrast": f"{combo} minus {lab}",
                                                           **learned._pair(meth[combo], ref)})
        log(f"union-first: {st}")

    # stacked, in the four splits
    for split in learned.SPLITS:
        folds = list(learned.split_folds(items, split))
        if not folds:
            continue
        for rcfg in rich_cfgs:
            extra = {}
            cfg_of = {}
            for n, it in items.items():
                key = ("rich_models", rcfg) if it["set"] == "rich_models" else \
                    next(((st, lab) for (st, lab) in answers if st == it["set"]), None)
                cfg_of[n] = key
                extra[n] = stacked_features(it, answers.get(key, [])) if key else \
                    {T: [0.0, 0.0] for T in it["features"]}
            per = defaultdict(dict)
            curve = defaultdict(list)
            for size in SIZES:
                for label, test, train in folds:
                    test = [n for n in test if defined(items[n], size)]
                    tr = [items[n] for n in train if defined(items[n], size)]
                    if not test or not tr:
                        continue
                    ms = fit_size(tr, size, "structural+lookahead", threads, extra)
                    mp = fit_size(tr, size, "structural+lookahead", threads)
                    for n in test:
                        it = items[n]
                        ps = predict_size(ms, it, size, "structural+lookahead", extra)
                        pp = predict_size(mp, it, size, "structural+lookahead")
                        tag = (split, str(rcfg), str(label), "stacked")
                        v = {"stacked order": order_value(it, ps, size, "order", tag),
                             "stacked staged": order_value(it, ps, size, "staged", tag),
                             "stacked interleaved with five-dual lookahead":
                                 order_value(it, ps, size, "interleaved", tag, it["five"])
                                 if it["five"] is not None else None,
                             "structural+lookahead learned order": order_value(
                                 it, pp, size, "order", (split, str(rcfg), str(label), "plain"))}
                        per[(size, n)] = v
                    if split == "training_size":
                        c, k_size, seed = label
                        xs = [per[(size, n)]["stacked order"] for n in test if has_target(items[n], size)]
                        xs = [x for x in xs if x is not None]
                        curve[(c, k_size, size)].append(float(np.mean(xs)) if xs else None)
            if split == "training_size":
                log(f"stacked: {split}, rich-model answers {rcfg}")
                for (c, k_size, size), xs in sorted(curve.items()):
                    ok = [x for x in xs if x is not None]
                    tables["stacked_training_size"].append({
                        "split": split, "rich_model_answers": rcfg, "class": c, "training_size": k_size,
                        "size_limit": size, "draws": len(xs), "draws_with_value": len(ok),
                        "cost_mean": float(np.mean(ok)) if ok else None,
                        "cost_se": float(np.std(ok, ddof=1) / np.sqrt(len(ok))) if len(ok) > 1 else None})
                continue
            tested = sorted({n for (_, n) in per})
            for st in sets:
                names_st = [n for n in tested if items[n]["set"] == st]
                if not names_st:
                    continue
                learned._group_rows(work, st, {n: items[n] for n in items if items[n]["set"] == st})
                key = cfg_of[names_st[0]]
                recs = answers.get(key, [])
                for size in SIZES:
                    names_sz = [n for n in names_st if defined(items[n], size) and (size, n) in per
                                and has_target(items[n], size)]
                    if not names_sz:
                        continue
                    av = {n: answer_values(items[n], recs, size) for n in names_sz}
                    five = {n: order_value(items[n], items[n]["five"], size, "order", ("five",)) for n in names_sz}
                    for cls, names in _cells(items, names_sz, st):
                        base = {"split": split, "set": st, "class": cls, "configuration": key[1] if key else None,
                                "rich_model_answers": rcfg, "size_limit": size}
                        meth = {m: {n: per[(size, n)].get(m) for n in names} for m in per[(size, names[0])]}
                        meth["five-dual lookahead order"] = {n: five[n] for n in names}
                        meth["jointdw interleaved"] = {n: av[n].get("jointdw interleaved") for n in names}
                        meth["jointdw staged"] = {n: av[n].get("jointdw staged") for n in names}
                        for m, v in meth.items():
                            tables["stacked_search"].append({**base, "method": m, **learned._se(v, "evaluations"),
                                                             "n_failed": sum(1 for x in v.values() if x is None)})
                        best, _ = _best({x: meth[x] for x in ("jointdw staged", "five-dual lookahead order",
                                                               "structural+lookahead learned order")}, names)
                        for m in ("stacked order", "stacked staged", "stacked interleaved with five-dual lookahead"):
                            refs = [("five-dual lookahead order", meth["five-dual lookahead order"]),
                                    ("jointdw interleaved", meth["jointdw interleaved"])]
                            if best:
                                refs.append((f"best single source ({best})", meth[best]))
                            for lab, ref in refs:
                                tables["stacked_paired"].append({**base, "method": m, "contrast": f"{m} minus {lab}",
                                                                 **learned._pair(meth[m], ref)})
            log(f"stacked: {split}, rich-model answers {rcfg}")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if name.startswith("stacked"):
            for split in learned.SPLITS:
                sub = [r for r in rows if r["split"] == split]
                if sub:
                    results.write_csv(out / f"{name}_{split}.csv", sub)
        else:
            results.write_csv(out / f"{name}.csv", rows)
    (out / "REPORT.md").write_text(report(dict(tables), items, sets))
    return dict(tables)


def report(tables: dict, items: dict, sets: list) -> str:
    pm = learned._pm
    L = ["# Combined orders: union-first and stacked", "",
         "Instances: " + ", ".join(f"{st} {sum(1 for i in items.values() if i['set'] == st)}" for st in sets) + ".", "",
         "## Definitions (fixed before any result was computed)", "", __doc__.split("Usage")[0].strip(), "",
         "Note: the best single source is chosen per cell after the fact, so the comparison against it favors the "
         "single source.", "", "## union-first: search cost (evaluations) and |U| <= s", "",
         "| set | class | configuration | s | method | mean ± s.e. | instances | failed |",
         "|---|---|---|---|---|---|---|---|"]
    for r in tables.get("union_search", []):
        pre = "share" if r["method"].startswith("share") else "evaluations"
        L.append(f"| {r['set']} | {r['class']} | {r['configuration']} | {r['size_limit']} | {r['method']} | "
                 f"{pm(r.get(pre + '_mean'), r.get(pre + '_se'))} | {r.get('n_instances')} | {r.get('n_failed', '')} |")
    L += ["", "## union-first: paired differences", "",
          "| set | class | configuration | s | contrast | difference | instances |", "|---|---|---|---|---|---|---|"]
    for r in tables.get("union_paired", []):
        L.append(f"| {r['set']} | {r['class']} | {r['configuration']} | {r['size_limit']} | {r['contrast']} | "
                 f"{pm(r.get('mean_difference'), r.get('se_difference'))}{r.get('asterisk', '')} | {r.get('n_instances')} |")
    for split in learned.SPLITS:
        srch = [r for r in tables.get("stacked_search", []) if r["split"] == split]
        par = [r for r in tables.get("stacked_paired", []) if r["split"] == split]
        ts = [r for r in tables.get("stacked_training_size", []) if r["split"] == split]
        if not (srch or par or ts):
            continue
        L += ["", f"## stacked, {split}"]
        if srch:
            L += ["", "| set | class | configuration | rich-model answers | s | method | mean ± s.e. | instances | failed |",
                  "|---|---|---|---|---|---|---|---|---|"]
            L += [f"| {r['set']} | {r['class']} | {r['configuration']} | {r['rich_model_answers']} | {r['size_limit']} | "
                  f"{r['method']} | {pm(r.get('evaluations_mean'), r.get('evaluations_se'))} | {r.get('n_instances')} | "
                  f"{r.get('n_failed', '')} |" for r in srch]
        if par:
            L += ["", "| set | class | configuration | rich-model answers | s | contrast | difference | instances |",
                  "|---|---|---|---|---|---|---|---|"]
            L += [f"| {r['set']} | {r['class']} | {r['configuration']} | {r['rich_model_answers']} | {r['size_limit']} | "
                  f"{r['contrast']} | {pm(r.get('mean_difference'), r.get('se_difference'))}{r.get('asterisk', '')} | "
                  f"{r.get('n_instances')} |" for r in par]
        if ts:
            L += ["", "| rich-model answers | class | s | training instances | stacked order cost | draws |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {r['rich_model_answers']} | {r['class']} | {r['size_limit']} | {r['training_size']} | "
                  f"{pm(r['cost_mean'], r['cost_se'], 2)} | {r['draws_with_value']}/{r['draws']} |" for r in ts]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--five-duals")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="templates,rich_models,miplib")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mps-dir")
    ap.add_argument("--features", help="feature files (llmdw.scoring.learned.feature_file): read instead of the models")
    ap.add_argument("--write-features", action="store_true", help="compute the features and store them in --features")
    ap.add_argument("--jointdw", nargs="*", default=[], help="SET:LABEL=FOLDER (GPT-6.1 Sol answers)")
    args = ap.parse_args(argv)
    jd = {}
    for x in args.jointdw:
        st, rest = x.split(":", 1)
        lab, folder = rest.split("=", 1)
        jd[(st, lab)] = folder
    run(Path(args.work), Path(args.five_duals) if args.five_duals else None, Path(args.out), args.sets.split(","),
        args.threads, jd, Path(args.mps_dir) if args.mps_dir else None,
        features=Path(args.features) if args.features else None, write_features=args.write_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
