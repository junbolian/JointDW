"""The table bodies of the paper (Tables 1, 2 and 6 to 15) rebuilt from the results CSVs, compared with the bodies of
the paper in the released data (data/reference_results/paper): every row, its label and every number.

The results come from data/reference_results (the released results of the sample of record, fetched by
scripts/fetch_data.sh) and, when it exists, from results/ (written by scripts/reproduce.sh). Without the released
data, the test is skipped.
"""
import re
from pathlib import Path

import pytest

from llmdw.tables import paper

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference_results" / "paper"
SOURCES = [p for p in (ROOT / "data" / "reference_results", ROOT / "results") if (p / "prediction.csv").exists()]
NUMBER = re.compile(r"\d+\.\d+|\d+")


@pytest.mark.skipif(not (SOURCES and REFERENCE.is_dir()), reason="no released data: run scripts/fetch_data.sh first")
@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_table_bodies_match_the_paper(source):
    built = paper.tables(source)
    assert sorted(built) == [1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for n, body in built.items():
        expected = (REFERENCE / f"table_{n:02d}.tex").read_text(encoding="utf-8")
        assert NUMBER.findall(body) == NUMBER.findall(expected), f"numbers of Table {n}"
        assert body == expected, f"Table {n}"
