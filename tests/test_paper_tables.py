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
"""
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
LATER_NAMES = sorted(n for _, names in LATER_BODIES.values() for n in names)
LATER_SOURCES = [p for p in SOURCES if all((p / d).is_dir() for d in LATER_BODIES)]


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
