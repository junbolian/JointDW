import csv
import json
import math

import pytest

from llmdw.benchmark import eval_time
from llmdw.scoring import eval_growth as eg


def _rec(master, wall=None, solving=None, status="nodelimit", linking=()):
    return {"master_families": list(master), "linking_var_families": list(linking), "status": status,
            "wall_seconds": wall, "solving_time": solving, "key": "|".join(master)}


def _census(tmp_path):
    root = tmp_path / "oracle"
    fams = ["A", "B", "C", "D"]
    subsets = [
        _rec(["B", "C", "D"], wall=1.0),                       # {A}
        _rec(["A", "C", "D"], wall=3.0),                       # {B}
        _rec(["A", "B", "D"], wall=None, solving=None),        # {C}: missing -> at the limit
        _rec(["C", "D"], wall=4.0),                            # {A,B}
        _rec(["B", "D"], wall=200.0, status="failed"),         # {A,C}: capped at 120 -> at the limit
        _rec(["B", "C"], wall=None, solving=16.0),             # {A,D}: solving_time
        _rec(["D"], wall=8.0),                                 # {A,B,C}
        _rec(["A"], wall=5.0, linking=["x"]),                  # linking variables: skipped
        _rec(["C"], status="no_blocks"),                       # {A,B,D}: not run
    ]
    (root / "m1").mkdir(parents=True)
    (root / "m1" / "oracle.json").write_text(json.dumps({"families": fams, "subsets": subsets}))
    (root / "bad").mkdir()
    (root / "bad" / "oracle.json").write_text("{not json")
    return root


def test_census_items_follow_the_census_time(tmp_path):
    items, problems = eg.census_items(_census(tmp_path))
    assert len(problems) == 1 and "bad" in problems[0]
    assert len(items) == 8                                         # the linking record is skipped
    by = {(x["size"], x["t"], x["at_limit"], x["not_run"]) for x in items}
    assert (1, 1.0, False, False) in by and (1, None, True, False) in by
    assert (2, 16.0, False, False) in by and (2, None, True, False) in by
    assert (3, None, False, True) in by and (3, 8.0, False, False) in by
    assert sum(x["failed"] for x in items) == 1


def test_by_size_counts_and_quartiles(tmp_path):
    items, _ = eg.census_items(_census(tmp_path))
    rows = {r["size"]: r for r in eg.by_size_rows(items)}
    r1 = rows[1]
    assert (r1["n_sets"], r1["n_at_limit"], r1["n_below_limit"], r1["n_not_run"]) == (3, 1, 2, 0)
    assert r1["median"] == 2.0 and r1["q25"] == 1.5 and r1["q75"] == 2.5
    r2 = rows[2]
    assert (r2["n_sets"], r2["n_at_limit"], r2["median"], r2["n_failed"]) == (3, 1, 10.0, 1)
    r3 = rows[3]
    assert (r3["n_sets"], r3["n_not_run"], r3["median"]) == (1, 1, 8.0)


def test_slope_of_log_median_on_log_size(tmp_path):
    items, _ = eg.census_items(_census(tmp_path))
    rows = eg.slope_rows(items)
    m = [r for r in rows if r["level"] == "model"][0]
    xs, ys = [math.log(s) for s in (1, 2, 3)], [math.log(v) for v in (2.0, 10.0, 8.0)]
    mx, my = sum(xs) / 3, sum(ys) / 3
    want = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    assert m["slope"] == pytest.approx(want)
    g = [r for r in rows if r["level"] == "group"][0]
    assert (g["n_models"], g["n_models_used"], g["median"]) == (1, 1, pytest.approx(want))


def test_slope_drops_zero_medians_and_needs_three_sizes():
    items = [eg._item("templates", "c", "m", s, "census", t) for s, t in ((1, 0.0), (2, 1.0), (3, 2.0), (4, 4.0))]
    r = eg.model_slope(items)
    assert r["n_sizes_zero_median"] == 1 and r["sizes"] == "2;3;4"
    assert r["slope"] == pytest.approx(eg.ols_slope([math.log(2), math.log(3), math.log(4)],
                                                    [0.0, math.log(2), math.log(4)]))
    assert eg.model_slope(items[:3])["slope"] is None               # two sizes left
    exact = [eg._item("templates", "c", "m", s, "census", 3.0 * s ** 2) for s in (1, 2, 4)]
    assert eg.model_slope(exact)["slope"] == pytest.approx(2.0)


def test_eval_items_two_variants(tmp_path):
    fams = ["A", "B", "C"]
    paths = {"instances": {"t1": {"collection": "rich_models", "class": "lot_sizing", "families": fams,
                                  "visited": ["A", "B", "A|B", "A|C"]}}}
    runs = tmp_path / "runs"
    rows = {frozenset("A"): {"status": "nodelimit", "wall_seconds": 0.5, "solving_time": 0.1},
            frozenset("B"): {"status": "nodelimit", "wall_seconds": 130.0, "solving_time": 119.0},
            frozenset("AB"): {"status": "no_blocks"}}
    for T, row in rows.items():
        p = eval_time.record_path(runs, "t1", fams, T)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row))
    items, problems = eg.eval_items(paths, runs)
    assert problems == ["rich_models t1 A|C: no record"]
    c = {(x["size"], x["model"], x["variant"], x["t"], x["at_limit"], x["not_run"]) for x in items}
    assert (1, "t1", "census", 0.5, False, False) in c and (1, "t1", "gcg", 0.1, False, False) in c
    assert (1, "t1", "census", None, True, False) in c and (1, "t1", "gcg", 119.0, False, False) in c
    assert (2, "t1", "census", None, False, True) in c and (2, "t1", "gcg", None, False, True) in c
    groups = {(r["collection"], r["class"], r["time"]) for r in eg.by_size_rows(items)}
    assert ("rich_models", "all", "gcg") in groups and ("rich_models", "lot_sizing", "census") in groups


def test_seed_median_censoring():
    assert eg.seed_median([3.0, 1.0, 2.0]) == 2.0
    assert eg.seed_median([3.0, None, 1.0]) == 3.0                  # censored ordered last
    assert eg.seed_median([None, None, 1.0]) is None
    assert eg.seed_median([1.0, None]) is None
    assert eg.seed_median([1.0, 2.0, 3.0, None]) == 2.5
    assert eg.seed_median([]) is None


def _d_rows():
    out = []

    def add(n, cls, key, seed, t, solved, root, pricing, failed=False):
        out.append({"instance": n, "class": cls, "set": key, "seed": str(seed), "status": "x",
                    "solved": str(solved), "time": "" if t is None else str(t), "root_time": "" if root is None else
                    str(root), "pricing_time": "" if pricing is None else str(pricing), "failed_twice": str(failed)})
    for s in (0, 1, 2):
        add("i1", "lot_sizing", "A", s, 10 + s, True, 1.0 + s, 0.5)
        add("i1", "lot_sizing", "B", s, 30, True, 3.0, 1.5)
        add("i1", "lot_sizing", "A|B", s, 1800, False, 8.0, 4.0 + s)            # unsolved: total at the limit
        add("i1", "lot_sizing", "A|C", s, 50, s != 0, None if s else 6.0, 2.0)   # root missing on 2 seeds
        add("i1", "lot_sizing", "A|B|C", s, 1800, False, 16.0, 0.0, failed=s == 1)
        add("i1", "lot_sizing", "gcg_default", s, 5, True, 1.0, 0.1)
        add("i1", "lot_sizing", "scip", s, 5, True, 1.0, None)
        add("i2", "network_design", "A", s, 2, True, 0.0, 1.0)
        add("i2", "network_design", "A|B", s, 4, True, 2.0, 2.0)
    return out


def test_bnp_items_seed_medians_and_limits():
    items, problems = eg.bnp_items(_d_rows())
    assert problems == []
    by = {(x["model"], x["size"], x["variant"], x["t"], x["at_limit"], x["missing"], x["failed"]) for x in items}
    assert not any(x["model"] == "i1" and x["size"] == 1 and x["t"] == 5 for x in items)   # references excluded
    assert len({(x["model"], x["size"], x["variant"]) for x in items if x["model"] == "i1"}) == 9
    assert ("i1", 2, "total", None, True, False, False) in by                 # unsolved median seed
    assert ("i1", 2, "pricing", 5.0, False, False, False) in by
    assert ("i1", 2, "root", None, True, True, False) in by                   # A|C: root missing on 2 of 3 seeds
    assert ("i1", 2, "total", 50.0, False, False, False) in by                # A|C: 2 of 3 solved
    assert ("i1", 3, "root", 16.0, False, False, True) in by
    rows = {(r["class"], r["time"], r["size"]): r for r in eg.by_size_rows(items)}
    r = rows[("lot_sizing", "total", 2)]
    assert (r["n_sets"], r["n_at_limit"], r["n_below_limit"], r["median"]) == (2, 1, 1, 50.0)
    r = rows[("lot_sizing", "root", 2)]
    assert (r["n_at_limit"], r["n_at_limit_missing_value"], r["median"]) == (1, 1, 8.0)
    assert rows[("all", "root", 1)]["n_below_limit_zero"] == 1


def test_d_ratios():
    items, _ = eg.bnp_items(_d_rows())
    rows = eg.ratio_rows(items)
    inst = {(r["instance"], r["time"], r["sizes"]): r for r in rows if r["level"] == "instance"}
    # i1 root: size 1 seed medians A = 2, B = 3 -> median 2.5; size 2 below the limit: A|B = 8 -> 8 / 2.5
    assert inst[("i1", "root", "2/1")]["ratio"] == pytest.approx(8.0 / 2.5)
    assert inst[("i1", "root", "3/2")]["ratio"] == pytest.approx(16.0 / 8.0)
    # i1 pricing: size 1 medians 0.5 and 1.5 -> 1.0; size 2: A|B 5, A|C 2 -> 3.5; size 3: 0 -> 0 / 3.5
    assert inst[("i1", "pricing", "2/1")]["ratio"] == pytest.approx(3.5)
    assert inst[("i1", "pricing", "3/2")]["ratio"] == 0.0
    assert inst[("i2", "root", "2/1")]["ratio"] is None                 # zero denominator
    assert ("i2", "root", "3/2") not in inst
    cl = {(r["class"], r["time"], r["sizes"]): r for r in rows if r["level"] == "class"}
    a = cl[("all", "root", "2/1")]
    assert (a["n_instances_both_sizes"], a["n_zero_denominator"], a["n_ratios"]) == (2, 1, 1)
    assert a["median"] == pytest.approx(3.2)
    assert cl[("all", "pricing", "2/1")]["median"] == pytest.approx((3.5 + 2.0) / 2)


def test_command_line_writes_every_output(tmp_path):
    root = _census(tmp_path)
    d = tmp_path / "bnp_runs.csv"
    rows = _d_rows()
    with open(d, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out, tex, fig = tmp_path / "out", tmp_path / "t" / "eval_growth.tex", tmp_path / "f" / "eval_growth.csv"
    assert eg.main(["--oracle-root", str(root), "--out", str(out), "--tex", str(tex), "--figure", str(fig)]) == 0
    assert (out / "by_size.csv").exists() and (out / "slopes.csv").exists() and not (out / "d_ratios.csv").exists()
    body = tex.read_text()
    assert "MIPLIB-NL; census time" in body and "$1$ & 3 & 1 & 2.00 & [1.50, 2.50] \\\\" in body
    assert eg.main(["--oracle-root", str(root), "--d-runs", str(d), "--out", str(out), "--tex", str(tex),
                    "--figure", str(fig)]) == 0
    assert (out / "d_ratios.csv").exists()
    body = tex.read_text()
    assert "Branch-and-price instances, lot sizing; total time" in body and "root time ratio" in body
    with open(fig) as fh:
        f = list(csv.DictReader(fh))
    assert {r["collection"] for r in f} == {"MIPLIB-NL", "benchmark"}


def test_bnp_source_starts_no_run(monkeypatch):
    from llmdw.benchmark import bnp, time_truth

    def boom(*a, **k):
        raise AssertionError("a solver run was started")
    monkeypatch.setattr(time_truth, "stats_rows", boom)
    monkeypatch.setattr(time_truth, "run_one", boom)
    monkeypatch.setattr(bnp, "_bnp", boom)
    items, _ = eg.bnp_items(_d_rows())
    assert items


def test_needs_a_source(tmp_path):
    with pytest.raises(SystemExit):
        eg.main(["--out", str(tmp_path)])
