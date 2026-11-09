"""Cell records of the table bodies, for typesetting: per cell of a table body its section, row and column labels, the
instances (n) and the prompt groups among them (n_groups), the mean, the standard error and the asterisk flag (1 when
the 95 percent interval of a paired difference, or of the mean log ratio, excludes zero; 0 when it does not; empty for a
cell without a comparison). Values keep the full precision of the result CSVs; a count is an integer or a mean over the
three samples; a cell without a value is left out.

Display labels: the presentation with every coefficient is named "coefficients" ("coefficients, names permuted",
"coefficients, names anonymized") in the table bodies and these records (display_label); the result CSVs keep their
labels ("data", "data, permuted", "data, anonymized").
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

FIELDS = ["table", "section", "row", "column", "n", "n_groups", "mean", "se", "asterisk"]

_DISPLAY = {"data": "coefficients", "data (every coefficient)": "coefficients",
            "data, permuted": "coefficients, names permuted",
            "data, anonymized": "coefficients, names anonymized", "data permuted": "coefficients, names permuted",
            "data anonymized": "coefficients, names anonymized"}
_SUFFIX = re.compile(r"(^|, )data(,? (permuted|anonymized))?$")


def display_label(s: str) -> str:
    """The display label of a presentation, of a label that ends with one ("GPT-6.1 Sol, data") and of a contrast of
    such labels ("data minus data, permuted")."""
    if " minus " in s:
        return " minus ".join(display_label(x) for x in s.split(" minus "))
    if s in _DISPLAY:
        return _DISPLAY[s]
    m = _SUFFIX.search(s)
    if not m:
        return s
    return s[:m.start()] + m.group(1) + ("coefficients, names " + m.group(3) if m.group(3) else "coefficients")


def count(x) -> str:
    """A count: an integer, or a mean over the samples with two decimals."""
    f = float(x)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.2f}"


def _blank(x):
    return "" if x is None else x


class Cells:
    """The cell records of one table body, in the order of the body."""

    def __init__(self, table: str):
        self.table, self.section, self.rows = table, "", []

    def heading(self, text: str) -> None:
        self.section = text

    def add(self, row: str, column: str, src: dict | None = None, mean=None, se=None, star=None, n=None,
            groups=None, n_key: str = "n_instances", groups_key: str = "n_groups") -> None:
        if mean in (None, ""):
            return
        src = src or {}
        self.rows.append({"table": self.table, "section": self.section, "row": row, "column": column,
                          "n": _blank(n if n is not None else src.get(n_key)),
                          "n_groups": _blank(groups if groups is not None else src.get(groups_key)),
                          "mean": mean, "se": _blank(se),
                          "asterisk": "" if star is None else int(star in ("*", True))})

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(self.rows)
