"""The table bodies of the paper rebuilt from the results CSVs and compared with the reference bodies in tests/reference:
every row, its label and every number.

tests/reference/paper holds the bodies that llmdw.tables.paper writes (table_01.tex, table_02.tex and table_06.tex to
table_15.tex), tests/reference/experiments the bodies that llmdw.tables.experiments writes (the control presentations,
the five-dual lookahead, the benchmark instances, the learned predictor and the combined orders), both of the sample of
record. Paper Tables 18, 23 and 24 (NEW_TABLES) were added in the revision; their bodies are checked by name, and every
other reference body is unchanged by their addition. The bodies of the time-optimal decompositions, the rich models
with 12 and 16 families, the classes without a joint group, the cost accounting and the evaluation growth
(LATER_BODIES, in tests/reference/experiments as well) are rebuilt from their result CSVs by each module's
tex_from_csv; a number printed from a CSV value may differ from the reference by one unit in its last printed digit
when the value lies on a rounding boundary (the CSV keeps the value rounded, the reference was printed from the value
in memory), every other number and every label must be equal. The results come from data/reference_results (the
released results, fetched by scripts/fetch_data.sh) and, when it exists, from results/ (written by
scripts/reproduce.sh). Without the released data the tests are skipped.

The benchmark tables in the layout of the paper, the rich-model tables of the misses and the instance collections
(PAPER_V1_NAMES, written by llmdw.tables.paper_v1) and the body of the search cost against the misses after the cut
(written by llmdw.scoring.rich_models_large) are compared with their reference bodies byte for byte.

When JOINTDW_PAPER_TEX names the LaTeX source of the paper, every table body of the paper is compared with these
results (llmdw.tables.paper_v1.compare_with_paper): the tables whose layout the code writes number by number at each
position against the reference body, the other tables by looking for every printed number, at its printed precision,
among the values of the result CSVs it comes from. Every difference is written to the report and fails the test,
except the cells declared as derived from the instance lists (paper_v1.DERIVED). Nothing of the paper is kept in this
repository. For paper v1.0, set JOINTDW_PAPER_VERSION=1.0: its benchmark tables print earlier standard errors, which
are then reported but allowed.
"""
import os
import re
from pathlib import Path

import pytest

from llmdw.tables import experiments, paper

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference"
SOURCES = [p for p in (ROOT / "data" / "reference_results", ROOT / "results") if (p / "prediction.csv").exists()]
EXPERIMENT_SOURCES = [p for p in SOURCES if (p / "experiment_C").is_dir()]
NUMBER = re.compile(r"\d+\.\d+|\d+")


def _compare(built: str, expected: str, name: str) -> None:
    assert NUMBER.findall(built) == NUMBER.findall(expected), f"numbers of {name}"
    assert built == expected, name


def _close(built: str, expected: str, name: str) -> None:
    """Equal labels, and equal numbers up to one unit in the last printed digit."""
    nb, ne = NUMBER.findall(built), NUMBER.findall(expected)
    assert NUMBER.sub("#", built) == NUMBER.sub("#", expected), f"labels of {name}"
    assert len(nb) == len(ne), f"numbers of {name}"
    for a, b in zip(nb, ne):
        if a == b:
            continue
        digits = len(b.split(".")[1]) if "." in b else 0
        assert len(a) == len(b) and abs(float(a) - float(b)) <= 1.000001 * 10 ** -digits, f"{name}: {a} against {b}"


LATER_BODIES = {
    "experiment_D": ("llmdw.benchmark.time_truth", ["experiment_D_methods.tex", "experiment_D_paired.tex",
                                                    "experiment_D_tradeoff.tex"]),
    "experiment_E": ("llmdw.scoring.rich_models_large", ["experiment_E_instances.tex", "experiment_E_predictions.tex",
                                                         "experiment_E_search.tex", "experiment_E_paired.tex"]),
    "experiment_F": ("llmdw.benchmark.no_joint_group", ["experiment_F_methods.tex", "experiment_F_paired.tex"]),
    "experiment_G": ("llmdw.scoring.cost_accounting", ["experiment_G_net_benefit.tex"]),
    "eval_growth": ("llmdw.scoring.eval_growth", ["eval_growth.tex"]),
}
PAPER_V1_NAMES = ["tab_bench.tex", "tab_app_bench.tex", "tab_app_bench_time.tex", "tab_app_tradeoff.tex",
                  "tab_app_bench_data.tex", "tab_app_misses.tex", "tab_app_larger.tex", "tab_instances.tex"]
MISSES_NAMES = ["search_cost_vs_misses_after_cut.tex"]
LATER_NAMES = sorted([n for _, names in LATER_BODIES.values() for n in names] + PAPER_V1_NAMES + MISSES_NAMES)
LATER_SOURCES = [p for p in SOURCES if all((p / d).is_dir() for d in LATER_BODIES)
                 and (p / "paper" / "figure_data" / "search_cost_vs_misses_after_cut.csv").exists()]
PAPER_TEX = os.environ.get("JOINTDW_PAPER_TEX")


@pytest.mark.skipif(not SOURCES, reason="no released data: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_paper_table_bodies(source):
    built = paper.tables(source)
    assert sorted(built) == [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for n, body in built.items():
        _compare(body, (REFERENCE / "paper" / f"table_{n:02d}.tex").read_text(encoding="utf-8"), f"table_{n:02d}")


@pytest.mark.skipif(not EXPERIMENT_SOURCES, reason="no released data: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", EXPERIMENT_SOURCES, ids=lambda p: p.name)
def test_experiment_table_bodies(source, tmp_path):
    names = experiments.write_tables(source, tmp_path)
    assert sorted(names) == sorted(p.name for p in (REFERENCE / "experiments").glob("*.tex") if p.name not in LATER_NAMES)
    for name in names:
        _compare((tmp_path / name).read_text(encoding="utf-8"),
                 (REFERENCE / "experiments" / name).read_text(encoding="utf-8"), name)


# paper Tables 18, 23 and 24 of the revised paper: their bodies (llmdw.tables.experiments)
NEW_TABLES = {18: ["rich_models_data_vs_five_dual.tex"], 23: ["experiment_C_paired_hit1.tex"],
              24: ["experiment_C_data_methods.tex", "experiment_C_data_paired.tex"]}


@pytest.mark.skipif(not EXPERIMENT_SOURCES, reason="no released data: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", EXPERIMENT_SOURCES, ids=lambda p: p.name)
@pytest.mark.parametrize("table", sorted(NEW_TABLES), ids=lambda n: f"table_{n}")
def test_new_paper_tables(source, table, tmp_path):
    names = experiments.write_tables(source, tmp_path)
    for name in NEW_TABLES[table]:
        assert name in names, f"Table {table}: {name} not written"
        _compare((tmp_path / name).read_text(encoding="utf-8"),
                 (REFERENCE / "experiments" / name).read_text(encoding="utf-8"), f"Table {table}: {name}")


def test_display_labels_and_counts():
    from llmdw.tables.cells import count, display_label

    assert display_label("data") == "coefficients"
    assert display_label("data, permuted") == "coefficients, names permuted"
    assert display_label("data anonymized") == "coefficients, names anonymized"
    assert display_label("GPT-6.1 Sol, data") == "GPT-6.1 Sol, coefficients"
    assert display_label("GPT-6.1 Sol, data, anonymized") == "GPT-6.1 Sol, coefficients, names anonymized"
    assert display_label("GPT-6.1 Sol, original") == "GPT-6.1 Sol, original"
    assert display_label("## Model with data") == "## Model with data"
    assert count("17.0") == "17" and count(16.333333) == "16.33" and count(0) == "0"


@pytest.mark.skipif(not EXPERIMENT_SOURCES, reason="no released data: run scripts/fetch_data.sh first")
def test_bodies_have_no_bracketed_intervals_and_no_fractions_of_counts(tmp_path):
    names = experiments.write_tables(EXPERIMENT_SOURCES[0], tmp_path)
    paper_cells = {}
    bodies = [(tmp_path / n).read_text(encoding="utf-8") for n in names] + list(paper.tables(SOURCES[0], paper_cells).values())
    for body in bodies:
        assert not re.search(r"\[[^]]*,[^]]*\]", body)
        assert not re.search(r"\d/\d", body)
        assert not re.search(r"(^|, )data(,|\s|$)", body, re.M)
    for name in names:
        cells = (tmp_path / "cells" / (name[:-len(".tex")] + ".csv")).read_text(encoding="utf-8").splitlines()
        assert cells[0] == "table,section,row,column,n,n_groups,mean,se,asterisk" and len(cells) > 1
    assert sorted(paper_cells) == [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert all(c.rows for c in paper_cells.values())


@pytest.mark.skipif(not LATER_SOURCES, reason="no released data v1.2: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", LATER_SOURCES, ids=lambda p: p.name)
@pytest.mark.parametrize("folder", sorted(LATER_BODIES), ids=str)
def test_later_table_bodies_from_their_csvs(source, folder, tmp_path):
    import importlib

    module, names = LATER_BODIES[folder]
    mod = importlib.import_module(module)
    if folder == "experiment_D":
        mod.tex_bodies(source / folder, tmp_path)
    else:
        mod.tex_from_csv(source / folder, tmp_path)
    for name in names:
        _close((tmp_path / name).read_text(encoding="utf-8"),
               (REFERENCE / "experiments" / name).read_text(encoding="utf-8"), name)


@pytest.mark.skipif(not LATER_SOURCES, reason="no released data v1.2: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", LATER_SOURCES, ids=lambda p: p.name)
def test_later_table_bodies_as_written(source):
    """The bodies written by scripts/reproduce.sh (results/paper/experiments) are the reference bodies byte for byte."""
    written = source / "paper" / "experiments"
    for name in LATER_NAMES:
        _compare((written / name).read_text(encoding="utf-8"),
                 (REFERENCE / "experiments" / name).read_text(encoding="utf-8"), name)


@pytest.mark.skipif(not LATER_SOURCES, reason="no released data v1.2: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", LATER_SOURCES, ids=lambda p: p.name)
def test_paper_layout_bodies_from_their_csvs(source, tmp_path):
    from llmdw.tables import paper_v1

    assert sorted(paper_v1.write_tables(source, tmp_path)) == sorted(PAPER_V1_NAMES)
    for name in PAPER_V1_NAMES:
        _compare((tmp_path / name).read_text(encoding="utf-8"),
                 (REFERENCE / "experiments" / name).read_text(encoding="utf-8"), name)


@pytest.mark.skipif(not PAPER_TEX, reason="JOINTDW_PAPER_TEX does not name the LaTeX source of the paper")
@pytest.mark.skipif(not LATER_SOURCES, reason="no released data v1.2: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", LATER_SOURCES, ids=lambda p: p.name)
def test_every_table_body_of_the_paper(source, tmp_path):
    from llmdw.tables import paper_v1

    tex = Path(PAPER_TEX).read_text(encoding="utf-8")
    report = paper_v1.compare_with_paper(tex, source, REFERENCE)
    out = Path(os.environ.get("JOINTDW_PAPER_REPORT", tmp_path / "paper_tables_report.json"))
    paper_v1.write_report(report, out)
    allowed = paper_v1.STANDARD_ERRORS_BEFORE_V1_1 if os.environ.get("JOINTDW_PAPER_VERSION") == "1.0" else ()
    for lab, r in report.items():
        for d in r.get("differences", []) + r.get("not_found", []):
            print(lab, d)
    assert paper_v1.unexpected(report, allowed) == [], f"differences against the paper: see {out}"
