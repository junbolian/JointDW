"""Regression: the census setting reproduces the exact gains of the 15 templates.

Every family set T with 1 <= |T| <= k - 1 of every template (master = the other families, no linking variables) is
evaluated with GCG in the census setting (llmdw.gcg.oracle.evaluate_subset with its defaults: root node, no presolve,
cuts or heuristics, SCIP propagation off, 120 s, seed 0, one thread), and its gain over the exact LP bound must equal
the exact gain of ground_truth.json (llmdw.instances.export, rational certificates) within the census tolerance
1e-6 max(1, |z_LP|). With SCIP's default propagation the lot-sizing (clsp) and network-design (ndp) templates deviate,
so this test fails if propagation is switched back on in the bound evaluations; the second test checks that it does.

The templates come from data/work/templates (scripts/fetch_data.sh) when present, otherwise they are rebuilt by
llmdw.instances.export (about two minutes). Without GCG (pygcgopt), the tests are skipped.
"""
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import pytest

pytest.importorskip("pygcgopt")

from llmdw.gcg.oracle import evaluate_subset  # noqa: E402
from llmdw.instances import export  # noqa: E402
from llmdw.instances.families import family_of, objective_integral, read_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "data" / "work" / "templates"
TIME_LIMIT = 120.0
JOBS = 8


@pytest.fixture(scope="module")
def templates(tmp_path_factory) -> Path:
    if RELEASED.is_dir() and len(list(RELEASED.glob("*/ground_truth.json"))) == len(export.DEFAULT):
        return RELEASED
    out = tmp_path_factory.mktemp("templates")
    export.main(["--out", str(out)])
    return out


def mismatches(td: Path, out: Path, **kwargs) -> list:
    """The sets of template td whose GCG gain differs from the exact gain by more than the census tolerance."""
    gt = json.loads((td / "ground_truth.json").read_text(encoding="utf-8"))
    mps = str(td / "model.mps")
    model = read_model(mps)
    integral = objective_integral(model)
    fams = sorted({family_of(r.name) for r in model.rows})
    sets = [frozenset(c) for r in range(1, len(fams)) for c in itertools.combinations(fams, r)]
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=JOBS) as pool:
        rows = list(pool.map(lambda T: evaluate_subset(model, mps, out, set(fams) - T, TIME_LIMIT, 0,
                                                       integral=integral, **kwargs), sets))
    lp = float(Fraction(gt["lp_bound"]))
    exact = {frozenset(k.split("|")): float(Fraction(v)) for k, v in gt["gain"].items() if k}
    sign = 1.0 if model.sense == "minimize" else -1.0
    tol = 1e-6 * max(1.0, abs(lp))
    bad = []
    for T, r in zip(sets, rows):
        assert r.get("ok") and r.get("dual_bound") is not None, f"{td.name} {sorted(T)}: {r.get('status')}"
        gain = sign * (r["dual_bound"] - lp)
        if abs(gain - exact.get(T, 0.0)) > tol:
            bad.append(("|".join(sorted(T)), gain, exact.get(T, 0.0)))
    return bad


def test_census_setting_reproduces_exact_template_gains(templates, tmp_path):
    names = [n for n, _ in export.DEFAULT]
    assert sorted(p.name for p in templates.iterdir() if (p / "ground_truth.json").exists()) == sorted(names)
    bad = {n: mismatches(templates / n, tmp_path / n) for n in names}
    assert not any(bad.values()), {n: b for n, b in bad.items() if b}


def test_scip_default_propagation_breaks_the_check(templates, tmp_path):
    """Sensitivity: with SCIP's default propagation the same check fails on a lot-sizing and a network-design
    template, so the first test detects propagation switched back on."""
    for n in ("clsp_2x2", "ndp"):
        assert mismatches(templates / n, tmp_path / n, propagation="default"), n
