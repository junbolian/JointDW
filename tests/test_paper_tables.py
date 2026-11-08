"""The table bodies of the paper rebuilt from the results CSVs and compared with the reference bodies in tests/reference:
every row, its label and every number.

tests/reference/paper holds the bodies that llmdw.tables.paper writes (table_01.tex, table_02.tex and table_06.tex to
table_15.tex), tests/reference/experiments the bodies that llmdw.tables.experiments writes (the control presentations,
the five-dual lookahead, the benchmark instances, the learned predictor and the combined orders), both of the sample of
record. Paper Tables 18, 23 and 24 (NEW_TABLES) were added in the revision; their bodies are checked by name, and every
other reference body is unchanged by their addition. The results come from data/reference_results (the released results, fetched by scripts/fetch_data.sh) and, when
it exists, from results/ (written by scripts/reproduce.sh). Without the released data the tests are skipped.
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
    assert sorted(names) == sorted(p.name for p in (REFERENCE / "experiments").glob("*.tex"))
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
