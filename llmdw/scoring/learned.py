"""Learned structural predictor: a non-language-model baseline that ranks family sets by a regression of their gain.

Adapted from the learned decomposition detector of Basso and Ceselli (2023, Mathematical Programming Computation,
doi 10.1007/s12532-022-00230-4): a set T of constraint families is scored through the decomposition that convexifies
T (the rows of the families in T form the blocks, split into connected components over shared variables as in
llmdw.gcg.dec_io.from_master_families; every other row is in the master). The features (FEATURES, computed without
solving) are the static decomposition features of Basso and Ceselli (2023) (blocks, block sizes and densities, border,
right-hand-side and objective statistics, a total-unimodularity score) and the family features |T|, rows and
variables of the families in T, variables shared by them, the share of integer variables, and the single-dual
lookahead value of T. Two variants (VARIANTS): structural (every feature but the lookahead value) and
structural+lookahead (every feature). The learner is an XGBoost regressor with the hyperparameters of Basso and
Ceselli (2023) (HYPERPARAMETERS: max_depth 13, eta 0.1, 100 boosting rounds, seed 0; no other tuning) of the gain share
of T: its gain divided by the best gain of the instance among the sets of at most s families (0 when the instance has
no positive gain). One regressor per variant and size limit s, trained on the sets of at most s families.

The predicted values order the sets of at most s families exactly as lookahead values do (llmdw.search.relabel.
lookahead_list: decreasing value, ties within TIE_TOL to fewer families, then to the family order of the run). The
order is used as an answer with one joint support (its first set of two or more families) for hit@1, as a search order,
as the anchor of the staged search (relabel.model_iter, as JointDW), and that staged search interleaved with the
five-dual lookahead order (relabel.alternate, as the interleaved method). Every value is a mean over the family orders
of llmdw.search.relabel, as for the other orders.

Training and test splits (SPLITS): within_class, leave one instance out (rich models; templates, by template class);
leave_one_class_out (lot sizing, network design, facility location: rich models of the class and the templates clsp,
ndp, cfl), trained on the rich models and templates of the other two classes and on every scored MIPLIB-NL model;
miplibnl_leave_one_out; training_size, within class on the rich models (training sets of 1, 2, 4, 7 instances, 20
draws, seeds 0 to 19, tested on the rest of the class). Labeling cost: the sets of at most s families of every
training instance.

Usage
-----
python -m llmdw.scoring.learned --work DATA --five-duals DIR --out OUT [--sets templates,rich_models,miplib]
    [--splits within_class,leave_one_class_out,miplibnl_leave_one_out,training_size] [--mps-dir DIR] [--threads 2] [--jointdw SET:LABEL=FOLDER ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from llmdw.scoring import cluster, results
from llmdw.search import relabel

SIZE_LIMITS = (2, 3)
CURVE_SIZES = (1, 2, 4, 7)
CURVE_DRAWS = 20
RICH_CLASSES = ("lot_sizing", "network_design", "facility_location")
TEMPLATE_OF_CLASS = {"lot_sizing": "clsp", "network_design": "ndp", "facility_location": "cfl"}
SPLITS = ("within_class", "leave_one_class_out", "miplibnl_leave_one_out", "training_size")

FEATURES = [
    ("size", "|T|, the number of families convexified"),
    ("n_blocks", "number of blocks: connected components of the rows of T over shared variables"),
    ("block_rows_mean", "mean number of rows per block"),
    ("block_rows_std", "standard deviation (population) of the rows per block"),
    ("block_rows_min", "smallest number of rows of a block"),
    ("block_rows_max", "largest number of rows of a block"),
    ("block_cols_mean", "mean number of variables per block (variables with a coefficient in its rows)"),
    ("block_cols_std", "standard deviation (population) of the variables per block"),
    ("block_cols_min", "smallest number of variables of a block"),
    ("block_cols_max", "largest number of variables of a block"),
    ("block_density_mean", "mean block density: nonzeros / (rows x variables) of the block"),
    ("block_density_min", "smallest block density"),
    ("block_density_max", "largest block density"),
    ("border_rows", "number of master rows (rows of the families not in T, and empty rows)"),
    ("border_rows_share", "master rows / all rows"),
    ("border_nonzero_share", "nonzeros in master rows / all nonzeros"),
    ("linking_cols_share", "variables with a coefficient both in a master row and in a block row / all variables"),
    ("block_cols_share", "variables with a coefficient in a block row / all variables"),
    ("rhs_block_mean", "mean over block rows of the right-hand-side magnitude (largest finite |side| of the row; 0 "
                       "when both sides are infinite)"),
    ("rhs_block_std", "standard deviation (population) of the block right-hand-side magnitudes"),
    ("rhs_block_max", "largest block right-hand-side magnitude"),
    ("rhs_master_mean", "mean over master rows of the right-hand-side magnitude"),
    ("rhs_master_std", "standard deviation (population) of the master right-hand-side magnitudes"),
    ("obj_block_mean", "mean |objective coefficient| over the block variables"),
    ("obj_block_std", "standard deviation (population) of |objective coefficient| over the block variables"),
    ("obj_block_max", "largest |objective coefficient| of a block variable"),
    ("obj_block_nonzero_share", "block variables with a nonzero objective coefficient / block variables"),
    ("obj_outside_nonzero_share", "variables outside the blocks with a nonzero objective coefficient / variables "
                                  "outside the blocks (0 when there are none)"),
    ("tu_score", "block rows in blocks that pass the sufficient total-unimodularity test of Heller and Tompkins "
                 "(coefficients in {-1, 0, 1}, at most two nonzeros per column, rows split into two classes so that "
                 "two nonzeros of a column with equal signs are in different classes and with opposite signs in the "
                 "same class) / block rows"),
    ("family_rows_sum", "rows of the families in T, summed"),
    ("family_rows_min", "rows of the smallest family in T"),
    ("family_rows_max", "rows of the largest family in T"),
    ("family_vars_sum", "variables with a coefficient in the rows of each family of T, summed over the families"),
    ("family_vars_min", "fewest variables of a family in T"),
    ("family_vars_max", "most variables of a family in T"),
    ("shared_vars", "variables with a coefficient in the rows of at least two families of T"),
    ("shared_vars_share", "shared_vars / variables with a coefficient in the rows of T (0 when T has one family)"),
    ("int_share", "integer (binary or general integer) variables / variables with a coefficient in the rows of T"),
    ("lookahead_rel", "single-dual lookahead value of T / the largest lookahead value of the instance (0 when that "
                      "is not positive or T has no lookahead value)"),
]
FEATURE_NAMES = [f for f, _ in FEATURES]
VARIANTS = {"structural": [f for f in FEATURE_NAMES if f != "lookahead_rel"],
            "structural+lookahead": list(FEATURE_NAMES)}
HYPERPARAMETERS = {"max_depth": 13, "learning_rate": 0.1, "n_estimators": 100, "random_state": 0}
FEATURE_NOTE = ("The feature list follows the groups of Basso and Ceselli (2023): number of blocks, block shape "
                "(variables over constraints per block), block density, right-hand-side and objective statistics "
                "(mean, standard deviation, range), constraint statistics, and the total unimodularity score; the full "
                "list of their 2018 paper (Annals of Operations Research, doi 10.1007/s10479-018-3067-9) was not "
                "reproduced item by item.")


def _columns(variant: str) -> list:
    return [FEATURE_NAMES.index(f) for f in VARIANTS[variant]]


# ---------------------------------------------------------------- features

def _stats(xs, std=True):
    if not xs:
        return 0.0, 0.0, 0.0, 0.0
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std()) if std else 0.0, float(a.min()), float(a.max())


def _side(row) -> float:
    vals = [abs(x) for x in (row.lhs, row.rhs) if x is not None and math.isfinite(x)]
    return max(vals) if vals else 0.0


def heller_tompkins(rows) -> bool:
    """Sufficient total-unimodularity test of Heller and Tompkins on the matrix of rows (Row objects): every coefficient
    in {-1, 0, 1}, at most two nonzeros per column, and a split of the rows into two classes such that the two nonzeros
    of a column are in different classes when they have the same sign and in the same class when their signs differ."""
    cols = defaultdict(list)
    for i, r in enumerate(rows):
        for v, c in r.coefs.items():
            if c == 0:
                continue
            if c not in (1, -1, 1.0, -1.0):
                return False
            cols[v].append((i, 1 if c > 0 else -1))
    adj = defaultdict(list)
    for entries in cols.values():
        if len(entries) > 2:
            return False
        if len(entries) == 2:
            (i, a), (j, b) = entries
            same_class = a != b                        # opposite signs: same class; equal signs: different classes
            adj[i].append((j, same_class))
            adj[j].append((i, same_class))
    colour = {}
    for start in range(len(rows)):
        if start in colour:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            i = stack.pop()
            for j, same in adj[i]:
                want = colour[i] if same else 1 - colour[i]
                if j not in colour:
                    colour[j] = want
                    stack.append(j)
                elif colour[j] != want:
                    return False
    return True


def set_features(model, fam_rows: dict, T: frozenset, lookahead_rel: float) -> dict:
    """FEATURES of the set T (fam_rows: {family: [row indices]} of the model)."""
    from llmdw.instances.families import components

    block_idx = sorted(i for f in T for i in fam_rows.get(f, ()) if model.rows[i].coefs)
    in_block = set(block_idx)
    master_idx = [i for i in range(len(model.rows)) if i not in in_block]
    comps = components(model, block_idx) if block_idx else []
    rows_b, cols_b, dens, tu_rows = [], [], [], 0
    block_vars = set()
    for comp in comps:
        vs = set()
        nnz = 0
        for i in comp:
            vs.update(model.rows[i].coefs)
            nnz += sum(1 for c in model.rows[i].coefs.values() if c != 0)
        rows_b.append(len(comp))
        cols_b.append(len(vs))
        dens.append(nnz / (len(comp) * len(vs)) if vs else 0.0)
        block_vars |= vs
        if heller_tompkins([model.rows[i] for i in comp]):
            tu_rows += len(comp)
    master_vars = set()
    nnz_master = nnz_all = 0
    for i, r in enumerate(model.rows):
        k = sum(1 for c in r.coefs.values() if c != 0)
        nnz_all += k
        if i not in in_block:
            nnz_master += k
            master_vars.update(r.coefs)
    nvars = len(model.var_type) or 1
    rb = _stats(rows_b)
    cb = _stats(cols_b)
    db = _stats(dens)
    rhs_b = _stats([_side(model.rows[i]) for i in block_idx])
    rhs_m = _stats([_side(model.rows[i]) for i in master_idx])
    ob = [abs(model.obj.get(v, 0.0)) for v in block_vars]
    obs = _stats(ob)
    outside = [v for v in model.var_type if v not in block_vars]
    fr = [len(fam_rows.get(f, ())) for f in T]
    fam_vars = []
    seen = defaultdict(int)
    for f in T:
        vs = set()
        for i in fam_rows.get(f, ()):
            vs.update(model.rows[i].coefs)
        fam_vars.append(len(vs))
        for v in vs:
            seen[v] += 1
    tvars = set(seen)
    shared = sum(1 for c in seen.values() if c >= 2)
    return {
        "size": len(T), "n_blocks": len(comps),
        "block_rows_mean": rb[0], "block_rows_std": rb[1], "block_rows_min": rb[2], "block_rows_max": rb[3],
        "block_cols_mean": cb[0], "block_cols_std": cb[1], "block_cols_min": cb[2], "block_cols_max": cb[3],
        "block_density_mean": db[0], "block_density_min": db[2], "block_density_max": db[3],
        "border_rows": len(master_idx), "border_rows_share": len(master_idx) / max(1, len(model.rows)),
        "border_nonzero_share": nnz_master / max(1, nnz_all),
        "linking_cols_share": len(block_vars & master_vars) / nvars, "block_cols_share": len(block_vars) / nvars,
        "rhs_block_mean": rhs_b[0], "rhs_block_std": rhs_b[1], "rhs_block_max": rhs_b[3],
        "rhs_master_mean": rhs_m[0], "rhs_master_std": rhs_m[1],
        "obj_block_mean": obs[0], "obj_block_std": obs[1], "obj_block_max": obs[3],
        "obj_block_nonzero_share": (sum(1 for x in ob if x != 0) / len(ob)) if ob else 0.0,
        "obj_outside_nonzero_share": (sum(1 for v in outside if model.obj.get(v, 0.0) != 0) / len(outside))
        if outside else 0.0,
        "tu_score": tu_rows / len(block_idx) if block_idx else 0.0,
        "family_rows_sum": float(sum(fr)), "family_rows_min": float(min(fr)), "family_rows_max": float(max(fr)),
        "family_vars_sum": float(sum(fam_vars)), "family_vars_min": float(min(fam_vars)),
        "family_vars_max": float(max(fam_vars)),
        "shared_vars": float(shared), "shared_vars_share": shared / len(tvars) if tvars and len(T) > 1 else 0.0,
        "int_share": (sum(1 for v in tvars if model.var_type.get(v) in ("B", "I")) / len(tvars)) if tvars else 0.0,
        "lookahead_rel": float(lookahead_rel),
    }


def candidate_sets(fams, s: int) -> list:
    """Every set of 1 to min(s, k - 1) families (the full set is the MILP itself)."""
    k = len(fams)
    return [frozenset(c) for m in range(1, min(s, k - 1) + 1) for c in itertools.combinations(sorted(fams), m)]


def instance_features(model, truth: dict, s_max: int) -> dict:
    """{T: feature vector (FEATURE_NAMES order)} for every candidate set of at most s_max families."""
    from llmdw.instances.families import family_of

    fam_rows = defaultdict(list)
    for i, r in enumerate(model.rows):
        fam_rows[family_of(r.name)].append(i)
    la = {T: v for T, v in (truth.get("lookahead") or [])}
    top = max((v for v in la.values() if v is not None), default=0.0)
    out = {}
    for T in candidate_sets(truth["families"], s_max):
        v = la.get(T)
        rel = (v / top) if (v is not None and top > 0) else 0.0
        f = set_features(model, fam_rows, T, rel)
        out[T] = [f[n] for n in FEATURE_NAMES]
    return out


def labels(truth: dict, s: int) -> dict:
    """{T: gain share} over the sets of at most s families with a value in the gain table."""
    tab = {T: v for T, v in truth["table"].items() if T and len(T) <= s}
    best = max(tab.values(), default=0.0)
    return {T: (v / best if best > 0 else 0.0) for T, v in tab.items()}


# ---------------------------------------------------------------- learner

def fit(train: list, s: int, threads: int, variant: str = "structural+lookahead"):
    """XGBoost regressor (HYPERPARAMETERS) on the features of the variant of the labelled sets of the training
    instances (items with "features" and "truth"). Returns (model, labeling cost = number of labelled sets)."""
    import xgboost

    cols = _columns(variant)
    X, y = [], []
    for it in train:
        lab = labels(it["truth"], s)
        for T, v in lab.items():
            if T in it["features"]:
                X.append([it["features"][T][i] for i in cols])
                y.append(v)
    m = xgboost.XGBRegressor(**HYPERPARAMETERS, n_jobs=threads)
    m.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return m, len(y)


def predict(m, it: dict, s: int, variant: str = "structural+lookahead") -> list:
    """[(T, predicted gain share)] over the candidate sets of at most s families."""
    sets = [T for T in it["features"] if len(T) <= s]
    if not sets:
        return []
    cols = _columns(variant)
    p = m.predict(np.asarray([[it["features"][T][i] for i in cols] for T in sets], dtype=float))
    return [(T, float(v)) for T, v in zip(sets, p)]


def importance(m, variant: str = "structural+lookahead") -> dict:
    """Gain importance per feature of the variant, normalized to sum 1 (features never used: 0)."""
    names = VARIANTS[variant]
    sc = m.get_booster().get_score(importance_type="gain")
    vals = {names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v for k, v in sc.items()}
    tot = sum(vals.values()) or 1.0
    return {n: vals.get(n, 0.0) / tot for n in names}


# ---------------------------------------------------------------- splits

def labeling_count(it: dict, s: int) -> int:
    return len(labels(it["truth"], s))


def split_folds(items: dict, split: str):
    """Yields (label, test names, training names) of one split (training_size: label = (class, size, seed))."""
    rich = {n: it for n, it in items.items() if it["set"] == "rich_models"}
    tmpl = {n: it for n, it in items.items() if it["set"] == "templates"}
    mip = {n: it for n, it in items.items() if it["set"] == "miplib"}
    if split not in SPLITS:
        raise ValueError(split)
    if split == "within_class":
        for group in (rich, tmpl):
            for n, it in sorted(group.items()):
                train = [m for m, jt in group.items() if m != n and jt["class"] == it["class"]]
                if train:
                    yield "within class", [n], train
    elif split == "leave_one_class_out":
        for c in RICH_CLASSES:
            held = [n for n, it in items.items() if it["class"] in (c, TEMPLATE_OF_CLASS[c])
                    and it["set"] in ("rich_models", "templates")]
            train = [n for n, it in items.items() if it["set"] in ("rich_models", "templates")
                     and it["class"] in tuple(x for x in RICH_CLASSES if x != c)
                     + tuple(TEMPLATE_OF_CLASS[x] for x in RICH_CLASSES if x != c)] + sorted(mip)
            if held and train:
                yield f"held-out class {c}", sorted(held), sorted(train)
    elif split == "miplibnl_leave_one_out":
        for n in sorted(mip):
            train = [m for m in mip if m != n]
            if train:
                yield "leave one model out", [n], sorted(train)
    elif split == "training_size":
        for c in RICH_CLASSES:
            names = sorted(n for n, it in rich.items() if it["class"] == c)
            for size in CURVE_SIZES:
                if size >= len(names):
                    continue
                for seed in range(CURVE_DRAWS):
                    train = sorted(random.Random(seed).sample(names, size))
                    yield (c, size, seed), [n for n in names if n not in train], train


# ---------------------------------------------------------------- evaluation

def _first_joint(order_sets):
    return next((T for T in order_sets if len(T) >= 2), None)


def evaluate(it: dict, pred: list, five: list | None, s: int, tag) -> dict:
    """hit@1 and search costs (means over the family orders) of the learned order of one instance at size limit s;
    the five-dual lookahead order on the same instance for the paired differences."""
    t = it["truth"]
    name = it["name"]
    k = len(t["families"])
    s = min(s, k - 1)
    tab = results.restricted(t["table"], s)
    out = {}
    joint = set(t["joint"])

    def hit(values, key):
        orders = relabel.family_orders(name, results.base_order(t))
        return float(np.mean([float(_first_joint(relabel.lookahead_list(values, o, s)) in joint) for o in orders]))

    if joint:
        out["hit1"] = hit(pred, ("learned", tag))
        if five is not None:
            out["hit1_five"] = hit(five, ("five", tag))
    if not tab or max(tab.values()) <= 0:
        return out
    goal = results.targets(tab, "max")

    def run(kind, values, key):
        def fn(order):
            la = relabel.lookahead_list(values, order, s)
            if kind == "order":
                it_ = la
            else:
                P = _first_joint(la) or (la[0] if la else frozenset())
                staged = relabel.model_iter(order, s, P)
                it_ = staged if kind == "staged" else relabel.alternate(staged, relabel.lookahead_list(five, order, s))
            return relabel.first_hit(it_, tab, goal, None, results.EVAL_LIMIT_S)
        return results.run_value(relabel.runs(("learned", name, s, kind, key), name, results.base_order(t), fn)[0])

    out["cost_order"] = run("order", pred, tag)
    out["cost_staged"] = run("staged", pred, tag)
    if five is not None:
        out["cost_interleaved_five"] = run("interleaved", pred, tag)
        out["cost_five"] = run("order", five, ("five",))
    return out


def jointdw_values(it: dict, recs: list, s: int) -> dict:
    """hit@1 (first joint support a true one) and JointDW search cost of the answers of one configuration (original
    presentation), means over the samples."""
    t = it["truth"]
    k = len(t["families"])
    s = min(s, k - 1)
    tab = results.restricted(t["table"], s)
    hits, costs = [], []
    for r in recs:
        if r["instance"] != it["name"] or r.get("condition") != "original":
            continue
        sup = r.get("supports") or []
        hits.append(float(bool(sup) and frozenset(sup[0]["families"]) in set(t["joint"])))
        if tab and max(tab.values()) > 0:
            P = results.support_anchor(r, t["families"])
            costs.append(results.run_value(results.search_runs(it["name"], t, s, ("model", P))[0]))
    out = {}
    if hits and t["joint"]:
        out["hit1"] = sum(hits) / len(hits)
    if costs and all(c is not None for c in costs):
        out["cost"] = sum(costs) / len(costs)
    return out


# ---------------------------------------------------------------- loading

def load_items(work: Path, sets: list, five_root: Path | None, mps_dir: Path | None) -> dict:
    """{name: {"name", "set", "class", "truth", "model_path", "five"}} of the requested instance sets."""
    from llmdw.scoring import experiments, rich_models

    items = {}
    for st in sets:
        if st == "rich_models":
            for i in rich_models.load_instances(work):
                items[i["name"]] = {"name": i["name"], "set": st, "class": i["class"], "truth": i["truth"],
                                    "summary": i["model"].summary, "model_path": i["dir"] / "model.mps"}
        else:
            for n, i in results.load_instances(work, st).items():
                mp = (work / "work/templates" / n / "model.mps") if st == "templates" else \
                    ((mps_dir or work / "work/mps") / f"{n}.mps.gz")
                items[n] = {"name": n, "set": st, "class": i["class"], "truth": i["truth"], "summary": i["summary"],
                            "model_path": mp}
    for n, it in items.items():
        p = five_root / it["set"] / n / "five_duals.json" if five_root else None
        k = len(it["truth"]["families"])
        it["five"] = experiments._five_list(p, k, it["set"] == "templates") if p and p.exists() else None
    return items


def feature_file(cache: Path, it: dict, s_max: int) -> Path:
    """<cache>/s_max_<s_max>/<set>/<name>.json: the feature vectors of one instance, as add_features writes them."""
    return Path(cache) / f"s_max_{s_max}" / it["set"] / f"{it['name']}.json"


def add_features(items: dict, s_max: int, cache: Path | None = None, write: bool = False) -> None:
    """it["features"] = instance_features(...) for every item. With cache, a stored feature file (feature_file) is read
    instead of the model, so the tables can be rebuilt without the model files and without a solver library; with
    write, the computed features are stored there (sets in their order, values as recorded)."""
    for it in items.values():
        p = feature_file(cache, it, s_max) if cache else None
        if p is not None and not write and p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            if rec["feature_names"] != FEATURE_NAMES:
                raise SystemExit(f"{p}: feature names differ from FEATURE_NAMES")
            it["features"] = {frozenset(T): v for T, v in rec["sets"]}
            continue
        from llmdw.instances.families import read_model

        model = read_model(str(it["model_path"]))
        it["features"] = instance_features(model, it["truth"], s_max)
        del model
        if p is not None and write:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"feature_names": FEATURE_NAMES,
                                     "sets": [[sorted(T), [float(x) for x in v]] for T, v in it["features"].items()]}),
                         encoding="utf-8")


# ---------------------------------------------------------------- tables

def _group_rows(work: Path, st: str, insts: dict):
    from llmdw.scoring import rich_models

    if st == "rich_models":
        rich_models.use_prompt_groups(work)
    else:
        cluster.set_groups({})
        results.use_prompt_groups(work, st, insts)


def _se(values: dict, prefix: str) -> dict:
    from llmdw.scoring import experiments

    return experiments._se_row({n: v for n, v in values.items() if v is not None}, prefix)


def _pair(a: dict, b: dict) -> dict:
    from llmdw.scoring import experiments

    return experiments._paired(a, b)


def run(work: Path, five_root: Path | None, out: Path, sets: list, splits: list, threads: int, jointdw: dict,
        mps_dir: Path | None = None, log=print, features: Path | None = None, write_features: bool = False) -> dict:
    items = load_items(work, sets, five_root, mps_dir)
    add_features(items, max(SIZE_LIMITS), features, write_features)
    log(f"{len(items)} instances: " + ", ".join(f"{st} {sum(1 for i in items.values() if i['set'] == st)}"
                                                for st in sets))
    answers = {}
    for key, folder in jointdw.items():          # key: (set, label)
        st = key[0]
        insts = {n: {"summary": it["summary"]} for n, it in items.items() if it["set"] == st}
        answers[key] = results.load_answers(Path(folder), insts) if Path(folder).exists() else []
    tables = defaultdict(list)
    for split in splits:
        folds = list(split_folds(items, split))
        for variant in VARIANTS:
            relabel.clear()
            _run_split(work, items, sets, split, variant, folds, threads, answers, tables)
            log(f"{split}, {variant}: {len(folds)} folds")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        for split in splits:
            for variant in VARIANTS:
                sub = [r for r in rows if r["split"] == split and r["variant"] == variant]
                if sub:
                    results.write_csv(out / f"{name}_{split}_{variant}.csv", sub)
    (out / "REPORT.md").write_text(report(dict(tables), splits, items, sets))
    return dict(tables)


def _run_split(work, items, sets, split, variant, folds, threads, answers, tables) -> None:
    per = defaultdict(dict)          # (s, test name) -> evaluate() values (every split but training_size)
    lab = defaultdict(dict)          # (s, test name) -> labeling cost
    curve = defaultdict(list)        # (class, size, s) -> [(draw mean cost, labeling cost)]
    imp = defaultdict(list)
    for s in SIZE_LIMITS:
        for label, test, train in folds:
            m, cost = fit([items[n] for n in train], s, threads, variant)
            imp[s].append(importance(m, variant))
            vals = {}
            for n in test:
                tag = (split, variant, label, s)
                vals[n] = evaluate(items[n], predict(m, items[n], s, variant), items[n]["five"], s, tag)
            if split == "training_size":
                c, size, seed = label
                costs = [v.get("cost_order") for v in vals.values()]
                costs = [x for x in costs if x is not None]
                curve[(c, size, s)].append((sum(costs) / len(costs) if costs else None, cost))
            else:
                for n in test:
                    per[(s, n)] = vals[n]
                    lab[(s, n)] = cost
    head = {"split": split, "variant": variant}
    for s in SIZE_LIMITS:
        names = VARIANTS[variant]
        mean_imp = {f: float(np.mean([d[f] for d in imp[s]])) for f in names} if imp[s] else {}
        for rank, (f, v) in enumerate(sorted(mean_imp.items(), key=lambda kv: -kv[1])[:5], 1):
            tables["learned_importance"].append({**head, "s": s, "rank": rank, "feature": f,
                                                 "mean_gain_importance": v, "models": len(imp[s])})
    if split == "training_size":
        for (c, size, s), xs in sorted(curve.items()):
            ok = [x for x, _ in xs if x is not None]
            tables["learned_training_size"].append({
                **head, "class": c, "training_size": size, "s": s, "draws": len(xs), "draws_with_value": len(ok),
                "cost_mean": float(np.mean(ok)) if ok else None,
                "cost_se": float(np.std(ok, ddof=1) / math.sqrt(len(ok))) if len(ok) > 1 else None,
                "labeling_cost_mean": float(np.mean([y for _, y in xs]))})
        return
    tested = sorted({n for (_, n) in per})
    for st in sets:
        names_st = [n for n in tested if items[n]["set"] == st]
        if not names_st:
            continue
        _group_rows(work, st, {n: items[n] for n in items if items[n]["set"] == st})
        classes = sorted({items[n]["class"] for n in names_st})
        for cls in ["all"] + classes:
            names = [n for n in names_st if cls == "all" or items[n]["class"] == cls]
            for s in SIZE_LIMITS:
                v = {n: per[(s, n)] for n in names}
                base = {**head, "set": st, "class": cls, "s": s}
                tables["learned_prediction"].append({**base, "order": "learned", **_se(
                    {n: x.get("hit1") for n, x in v.items() if "hit1" in x}, "hit1")})
                tables["learned_prediction"].append({**base, "order": "five-dual lookahead", **_se(
                    {n: x.get("hit1_five") for n, x in v.items() if "hit1_five" in x}, "hit1")})
                for meth, key in METHODS:
                    tables["learned_search"].append({**base, "method": meth, **_se(
                        {n: x.get(key) for n, x in v.items() if key in x}, "evaluations"),
                        "n_without_target": sum(1 for x in v.values() if key in x and x[key] is None)})
                tables["learned_labeling"].append({**base, "test_instances": len(names),
                                                   "labeling_cost_mean": float(np.mean([lab[(s, n)] for n in names])),
                                                   "labeling_cost_min": min(lab[(s, n)] for n in names),
                                                   "labeling_cost_max": max(lab[(s, n)] for n in names)})
                learned = {"hit1": {n: x.get("hit1") for n, x in v.items() if "hit1" in x}}
                for key in ("cost_order", "cost_staged", "cost_interleaved_five"):
                    learned[key] = {n: x.get(key) for n, x in v.items() if key in x}
                five = {"hit1": {n: x.get("hit1_five") for n, x in v.items() if "hit1_five" in x},
                        "cost": {n: x.get("cost_five") for n, x in v.items() if "cost_five" in x}}
                for key, lv in learned.items():
                    ref = five["hit1"] if key == "hit1" else five["cost"]
                    tables["learned_paired"].append({**base, "metric": key,
                                                     "contrast": "learned minus five-dual lookahead", **_pair(lv, ref)})
                for (ast, alabel), recs in answers.items():
                    if ast != st:
                        continue
                    jd = {n: jointdw_values(items[n], recs, s) for n in names}
                    for key, lv in learned.items():
                        ref = {n: d.get("hit1" if key == "hit1" else "cost") for n, d in jd.items()}
                        tables["learned_paired"].append({**base, "metric": key,
                                                         "contrast": f"learned minus JointDW {alabel}",
                                                         **_pair(lv, ref)})


METHODS = (("learned order", "cost_order"), ("staged search anchored at the learned first set", "cost_staged"),
           ("staged learned interleaved with five-dual lookahead", "cost_interleaved_five"),
           ("five-dual lookahead order", "cost_five"))


def _pm(m, se, d=3):
    if m is None:
        return ""
    return f"{m:.{d}f}" + (f" ± {se:.{d}f}" if se is not None else "")


def report(tables: dict, splits: list, items: dict, sets: list) -> str:
    """The report: per split and variant, per instance set, class and size limit, the numbers of the tables (mean ±
    standard error of the cluster bootstrap; * when the 95% interval of a paired difference excludes 0), without
    interpretation."""
    hp = ", ".join(f"{k} = {v}" for k, v in HYPERPARAMETERS.items())
    L = ["# Learned structural predictor", "",
         "Instances: " + ", ".join(f"{st} {sum(1 for i in items.values() if i['set'] == st)}" for st in sets) + ".", "",
         "## Learner", "",
         f"XGBoost regression (XGBRegressor) of the gain share with the hyperparameters of Basso and Ceselli (2023): {hp} "
         "(learning_rate is eta, n_estimators the number of boosting rounds), 2 threads; no other tuning.", "",
         "## Variants", "",
         "- structural: " + ", ".join(VARIANTS["structural"]) + ".",
         "- structural+lookahead: the structural features and lookahead_rel.", "",
         "## Features", "", FEATURE_NOTE, "", "| feature | definition |", "|---|---|"] + \
        [f"| {f} | {d} |" for f, d in FEATURES]
    for split in splits:
        L += ["", f"## {split}"]
        for variant in VARIANTS:
            sel = lambda name: [r for r in tables.get(name, []) if r["split"] == split and r["variant"] == variant]  # noqa: E731
            pred, srch, par, ts, imp = (sel(x) for x in ("learned_prediction", "learned_search", "learned_paired",
                                                          "learned_training_size", "learned_importance"))
            lab = {(r["set"], r["class"], r["s"]): r for r in sel("learned_labeling")}
            if not (pred or srch or par or ts or imp):
                continue
            L += ["", f"### {split}, {variant}"]
            if pred:
                L += ["", "#### hit@1", "", "| set | class | s | learned | five-dual lookahead | instances |",
                      "|---|---|---|---|---|---|"]
                for r in pred:
                    if r["order"] != "learned":
                        continue
                    f = next((x for x in pred if x["order"] != "learned" and (x["set"], x["class"], x["s"]) ==
                              (r["set"], r["class"], r["s"])), {})
                    L.append(f"| {r['set']} | {r['class']} | {r['s']} | {_pm(r.get('hit1_mean'), r.get('hit1_se'))} | "
                             f"{_pm(f.get('hit1_mean'), f.get('hit1_se'))} | {r.get('n_instances')} |")
            if srch:
                meths = [m for m, _ in METHODS]
                L += ["", "#### Search cost (evaluations) and labeling cost", "",
                      "| set | class | s | " + " | ".join(meths) + " | labeling cost (mean per test instance) |",
                      "|---|---|---|" + "---|" * (len(meths) + 1)]
                for key in dict.fromkeys((r["set"], r["class"], r["s"]) for r in srch):
                    cells = []
                    for m in meths:
                        r = next((x for x in srch if (x["set"], x["class"], x["s"]) == key and x["method"] == m), {})
                        cells.append(_pm(r.get("evaluations_mean"), r.get("evaluations_se"), 2) +
                                     (f" (n={r.get('n_instances')})" if r else ""))
                    lc = lab.get(key, {}).get("labeling_cost_mean")
                    L.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells) +
                             (f" | {lc:.1f} |" if lc is not None else " | |"))
            if par:
                L += ["", "#### Paired differences", "",
                      "| set | class | s | metric | contrast | difference | instances |", "|---|---|---|---|---|---|---|"]
                for r in par:
                    L.append(f"| {r['set']} | {r['class']} | {r['s']} | {r['metric']} | {r['contrast']} | "
                             f"{_pm(r.get('mean_difference'), r.get('se_difference'))}{r.get('asterisk', '')} | "
                             f"{r.get('n_instances')} |")
            if ts:
                L += ["", "#### Search cost of the learned order against training size", "",
                      "| class | s | training instances | cost (mean over draws ± s.e.) | draws | labeling cost |",
                      "|---|---|---|---|---|---|"]
                for r in ts:
                    L.append(f"| {r['class']} | {r['s']} | {r['training_size']} | {_pm(r['cost_mean'], r['cost_se'], 2)}"
                             f" | {r['draws_with_value']}/{r['draws']} | {r['labeling_cost_mean']:.1f} |")
            if imp:
                L += ["", "#### Five features of largest gain importance", "",
                      "| s | rank | feature | mean normalized gain |", "|---|---|---|---|"]
                L += [f"| {r['s']} | {r['rank']} | {r['feature']} | {r['mean_gain_importance']:.3f} |" for r in imp]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the data root that holds work/")
    ap.add_argument("--five-duals", help="root of the five-dual lookahead results (<set>/<instance>/five_duals.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="templates,rich_models,miplib")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mps-dir", help="MPS files of the MIPLIB-NL models (default <work>/work/mps)")
    ap.add_argument("--features", help="feature files (feature_file): read instead of the models when present")
    ap.add_argument("--write-features", action="store_true", help="compute the features and store them in --features")
    ap.add_argument("--jointdw", nargs="*", default=[],
                    help="SET:LABEL=FOLDER of the answers of one configuration (original presentation), e.g. "
                         "rich_models:GPT-6.1 Sol, text=.../support_rich_models/gpt-6.1-sol")
    args = ap.parse_args(argv)
    jd = {}
    for x in args.jointdw:
        st, rest = x.split(":", 1)
        lab, folder = rest.split("=", 1)
        jd[(st, lab)] = folder
    run(Path(args.work), Path(args.five_duals) if args.five_duals else None, Path(args.out), args.sets.split(","),
        args.splits.split(","), args.threads, jd, Path(args.mps_dir) if args.mps_dir else None,
        features=Path(args.features) if args.features else None, write_features=args.write_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
