import itertools
import json
import math
import statistics

import numpy as np

from llmdw.benchmark import eval_time
from llmdw.constants import FAMILY_ORDERS
from llmdw.scoring import results
from llmdw.search import relabel


def _toy_truth():
    fams = ["A", "B", "C", "D"]
    sets = [frozenset(c) for r in range(0, 4) for c in itertools.combinations(fams, r)]
    gain = {frozenset(): 0.0, frozenset("A"): 0.1, frozenset("B"): 0.0, frozenset("C"): 0.2, frozenset("D"): 0.0,
            frozenset("AB"): 0.1, frozenset("AC"): 0.3, frozenset("AD"): 0.1, frozenset("BC"): 0.2,
            frozenset("BD"): 0.6, frozenset("CD"): 0.2}
    table = {T: gain.get(T, 0.7 if T == frozenset("BCD") else 0.4) for T in sets}
    la = [(T, v + 0.05 * len(T)) for T, v in table.items() if T]
    return {"families": fams, "family_order": ["C", "A", "D", "B"], "table": table, "lookahead": la,
            "joint": [frozenset("BD")]}


def _seen_first_hit(order, tab, goal):
    seen = []
    for T in order:
        if T in seen or T not in tab:
            continue
        seen.append(T)
        if T in goal:
            return seen
    return None


def _seen_greedy(tab, goal, fams):
    cur, seen = frozenset(), []
    while True:
        nbrs = [cur | {f} for f in fams if f not in cur and (cur | {f}) in tab]
        seen += [T for T in nbrs if T not in seen]
        if any(T in goal for T in nbrs):
            return seen
        if not nbrs:
            return None
        nxt = max(nbrs, key=lambda T: tab[T])
        if tab[nxt] <= tab[cur] + 1e-9:
            return None
        cur = nxt


def test_recorder_records_the_evaluated_sets_and_changes_nothing():
    relabel.configure(7)
    try:
        t = _toy_truth()
        specs = [("covering",), ("lookahead",), ("greedy",), ("model", frozenset("A")),
                 ("interleaved", frozenset("BC")), ("shortlist", frozenset("BD"))]
        plain = {(s, sp, g): results.search_runs("toy", t, s, sp, g) for s in (2, 3) for sp in specs
                 for g in ("max", "min95")}
        rec = eval_time.Recorder()
        with rec.installed():
            got = {(s, sp, g): results.search_runs("toy", t, s, sp, g) for s in (2, 3) for sp in specs
                   for g in ("max", "min95")}
        for key in plain:                                     # same evaluations, seconds still None
            assert np.array_equal(plain[key][0], got[key][0], equal_nan=True)
            assert np.isnan(got[key][1]).all()
        assert relabel.first_hit.__module__ == "llmdw.search.relabel"          # restored
        expect = set()
        for s in (2, 3):
            tab = results.restricted(t["table"], s)
            for g in ("max", "min95"):
                goal = results.targets(tab, g)
                for order in relabel.family_orders("toy", results.base_order(t)):
                    its = {"covering": relabel.covering_iter(order, s),
                           "lookahead": relabel.lookahead_list(t["lookahead"], order, s),
                           "model": relabel.model_iter(order, s, frozenset("A")),
                           "interleaved": relabel.alternate(relabel.model_iter(order, s, frozenset("BC")),
                                                            relabel.lookahead_list(t["lookahead"], order, s)),
                           "shortlist": relabel.shortlist_iter(frozenset("BD"), order, s)}
                    for it in its.values():
                        seen = _seen_first_hit(it, tab, goal)
                        expect |= set(seen or [])
                    expect |= set(_seen_greedy(tab, goal, order) or [])
        assert set(rec.sets["toy"]) == expect
        assert frozenset() not in expect
        assert {"covering", "lookahead", "model", "interleaved", "shortlist"} <= set().union(*rec.sets["toy"].values())
        # greedy stalls on this table: its runs are counted as failed and record no set
        assert rec.runs["toy"]["greedy"] == rec.failed["toy"]["greedy"] == 2 * 2 * 7
    finally:
        relabel.configure(FAMILY_ORDERS)


def test_recorder_keeps_given_times():
    relabel.configure(3)
    try:
        t = _toy_truth()
        times = {T: 1.5 for T in t["table"]}
        plain = results.search_runs("toy", t, 2, ("covering",), "max", times)
        rec = eval_time.Recorder()
        with rec.installed():
            got = results.search_runs("toy", t, 2, ("covering",), "max", times)
        assert np.array_equal(plain[1], got[1])
    finally:
        relabel.configure(FAMILY_ORDERS)


def _fake_paths(tmp_path):
    insts = {
        "t1": {"collection": "templates", "class": "cfl", "families": ["A", "B", "C"], "k": 3,
               "visited": ["A", "B", "A|C"], "table_size": 6},
        "t2": {"collection": "templates", "class": "cfl", "families": ["A", "B", "C"], "k": 3,
               "visited": ["B", "C"], "table_size": 6},
        "r1": {"collection": "rich_models", "class": "lot_sizing", "families": ["X", "Y"], "k": 2,
               "visited": ["X", "Y"], "table_size": 2},
    }
    paths = {"settings": {}, "instances": insts}
    runs = tmp_path / "runs"
    times = {("t1", "A"): {"status": "nodelimit", "solving_time": 2.0, "wall_seconds": 2.5},
             ("t1", "B"): {"status": "nodelimit", "solving_time": 4.0, "wall_seconds": None},
             ("t1", "A|C"): {"status": "timelimit", "solving_time": 120.01, "wall_seconds": 300.0},
             ("t2", "B"): {"status": "degenerate_single_block"},
             ("t2", "C"): {"ok": False, "status": "failed", "wall_seconds": 1.0},     # killed: no GCG time
             ("r1", "X"): {"status": "optimal", "solving_time": 0.5, "wall_seconds": 0.75}}   # r1 Y has no record
    for (n, key), row in times.items():
        p = eval_time.record_path(runs, n, insts[n]["families"], key.split("|"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row))
    return paths, runs


def test_tables_on_fake_records(tmp_path):
    paths, runs = _fake_paths(tmp_path)
    assert [j[0] for j in eval_time.jobs(paths, runs)] == ["r1"]
    rows, inst_rows = eval_time.time_tables(paths, runs)
    by = {(r["collection"], r["class"]): r for r in rows}
    t1, t2 = [2.5, 4.0, 120.0], [1.0]                                  # census definition: wall, else GCG; t2 B not run
    g1, g2 = [2.0, 4.0, 120.0], [120.0]                                # GCG's reported time, a missing value at the limit
    mean_t1, mean_t2 = sum(t1) / 3, sum(t2) / 1
    cfl = by[("templates", "cfl")]
    assert math.isclose(cfl["mean_of_instance_means"], (mean_t1 + mean_t2) / 2)
    assert math.isclose(cfl["se_over_instances"], statistics.stdev([mean_t1, mean_t2]) / math.sqrt(2))
    assert math.isclose(cfl["pooled_mean"], sum(t1 + t2) / 4)
    assert math.isclose(cfl["pooled_se"], statistics.stdev(t1 + t2) / math.sqrt(4))
    assert math.isclose(cfl["gcg_mean_of_instance_means"], (sum(g1) / 3 + sum(g2)) / 2)
    assert math.isclose(cfl["gcg_pooled_mean"], sum(g1 + g2) / 4)
    assert cfl["n_sets_at_limit"] == 1 and cfl["n_sets_not_run"] == 1 and cfl["n_sets_without_gcg_time"] == 1
    assert cfl["n_sets_with_time"] == 4 and by[("templates", "all")]["n_instances"] == 2
    rich = by[("rich_models", "lot_sizing")]
    assert rich["se_over_instances"] is None
    assert rich["mean_of_instance_means"] == 0.75 and rich["gcg_mean_of_instance_means"] == 0.5
    assert rich["n_sets"] == 2 and rich["n_sets_with_time"] == 1 and rich["n_sets_missing"] == 1
    assert rich["n_sets_at_limit"] == 0
    r1 = next(r for r in inst_rows if r["instance"] == "r1")
    assert r1["missing"] == 1
    t2r = next(r for r in inst_rows if r["instance"] == "t2")
    assert t2r["not_run"] == 1 and t2r["at_limit"] == 0 and t2r["failed"] == 1


def test_summary_rows():
    paths = {"instances": {
        "a": {"collection": "templates", "class": "x", "k": 3, "visited": ["A"], "table_size": 6},
        "b": {"collection": "templates", "class": "x", "k": 3, "visited": ["A", "B", "C"], "table_size": 6}}}
    rows = eval_time.summary_rows(paths)
    cls = next(r for r in rows if r["level"] == "class")
    assert cls["visited"] == 4 and cls["table_size"] == 12 and cls["visited_median"] == 2


def test_eval_seconds_is_the_census_definition():
    from llmdw.scoring import results

    rows = [{"wall_seconds": 3.0, "solving_time": 1.0}, {"wall_seconds": None, "solving_time": 5.0},
            {"solving_time": 130.0}, {"ok": False}, {"wall_seconds": 0.2}]
    for r in rows:
        assert eval_time.eval_seconds(r)[0] == results.oracle_seconds(r)
    assert [eval_time.eval_seconds(r)[1] for r in rows] == [False, False, True, True, False]
    assert eval_time.eval_seconds({"status": "no_blocks"}) == (None, False)
