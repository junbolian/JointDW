"""Experiment C: data readers and checks, standard formulations, branch-and-price helpers, recorded sources."""
import json
from fractions import Fraction
from pathlib import Path

from llmdw.benchmark import bnp, build, sources, standard, trigeiro

ROOT = Path(__file__).resolve().parents[1]

CAP = """2 3
 100 50.
 80 20.
 10 1 2
 5 3 4
 7 5 6
"""

CANAD = """3 2 1
1 2 4 10 100 1 1
2 3 5 20 200 1 1
1 3 7
"""


def test_read_cap_orlib_format():
    d = build.read_cap(CAP)
    assert d["facilities"] == 2 and d["customers"] == 3
    assert d["capacity"] == [100, 80] and d["fixed"] == [50, 20]
    assert d["demand"] == [10, 5, 7] and d["alloc"] == [[1, 3, 5], [2, 4, 6]]
    assert all(isinstance(x, Fraction) for x in d["demand"])


def test_read_canad_commalab_format():
    d = build.read_canad(CANAD)
    assert d["nodes"] == 3 and len(d["arcs"]) == 2 and len(d["commodities"]) == 1
    assert d["arcs"][0] == {"from": 0, "to": 1, "unit": 4, "capacity": 10, "fixed": 100}
    assert d["commodities"][0] == {"origin": 0, "destination": 2, "demand": 7}


def test_name_key_orders_numbers_numerically():
    names = ["cap101", "cap41", "cap62"]
    assert sorted(names, key=build.name_key) == ["cap41", "cap62", "cap101"]


def test_trigeiro_helpers():
    assert trigeiro.ttm_name("F10N") == "F10" and trigeiro.ttm_name("G5") == "G5"
    assert trigeiro._nums("1 2.5 -3") == [1.0, 2.5, -3.0]


def test_letters_are_spreadsheet_names():
    assert [standard.letters(k) for k in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


def _families(m):
    return sorted({r.family for r in m.rows})


def test_standard_lot_sizing_formulation():
    d = {"items": 2, "periods": 3, "demand": [[1, 2, 3], [0, 4, 1]], "setup_time": [1, 2], "capacity": [9, 9, 9],
         "holding": [1, 1], "setup_cost": [10, 20]}
    m = standard.lot_sizing(d, "ls_test")
    assert _families(m) == ["Balance_#_#", "Capacity_#", "SetupLink_#_#"]
    assert len(m.rows) == 2 * 3 + 2 * 3 + 3 and len(m.vars) == 3 * 2 * 3
    assert sum(v.integer for v in m.vars) == 6


def test_standard_network_design_formulation():
    m = standard.network_design(build.read_canad(CANAD), "nd_test")
    assert _families(m) == ["ArcCapacity_#", "ArcLink_#_#", "FlowBalance_#_#"]
    assert sum(r.family == "FlowBalance_#_#" for r in m.rows) == 3
    assert sum(r.family == "ArcLink_#_#" for r in m.rows) == 2 and sum(v.integer for v in m.vars) == 2


def test_standard_facility_location_formulation():
    m = standard.facility_location(build.read_cap(CAP), "fl_test")
    assert _families(m) == ["Assign_#", "Capacity_#", "Link_#_#"]
    assert sum(r.family == "Link_#_#" for r in m.rows) == 6 and sum(v.integer for v in m.vars) == 2


def test_branch_and_price_helpers():
    assert bnp.set_key({"B", "A"}) == "A|B"
    log = "Original Program statistics:\n ...\nRoot Node          :\n  Final Dual Bound : 12.5\n"
    assert bnp._root_bound(log) == 12.5
    assert bnp._root_bound("Original Program statistics:\nRoot Node :\n Final Dual Bound : -\n") is None
    table = {"A": {"families": ["A"], "lookahead": 0.0}, "A|B": {"families": ["A", "B"], "lookahead": 2.0},
             "B": {"families": ["B"], "lookahead": None}}
    assert bnp._first_of_order(["A", "B", "C"], table) == ["A", "B"]


def test_recorded_sources_cover_every_url_with_a_checksum():
    rec = json.loads((ROOT / "config" / "benchmark_sources.json").read_text())
    assert rec["gcg_commit"] == sources.GCG_COMMIT
    urls = {u for us in sources.FIXED.values() for u in us}
    assert {f["url"] for f in rec["files"]} == urls
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in rec["files"])


def test_check_compares_built_model_files_with_recorded_sha256(tmp_path):
    import hashlib
    import json

    from llmdw.benchmark import standard

    rec, out = tmp_path / "recorded", tmp_path / "built"
    for name, text in (("a", b"model a"), ("b", b"model b"), ("c", b"model c")):
        (rec / "instances" / name).mkdir(parents=True)
        (rec / "instances" / name / "meta.json").write_text(json.dumps({"mps_sha256": hashlib.sha256(text).hexdigest()}))
    for name, text in (("a", b"model a"), ("b", b"changed")):
        (out / "instances" / name).mkdir(parents=True)
        (out / "instances" / name / "model.mps").write_bytes(text)
    res = standard.cmd_check(out, rec)
    assert res == {"instances": 3, "identical": 1, "missing": ["c"], "different": ["b"]}
    assert standard.main(["check", "--out", str(out), "--recorded", str(rec)]) == 1
