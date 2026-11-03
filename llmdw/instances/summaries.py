"""Model summaries: the facts about a model that the scoring reads (Section 6 and Appendix F of the paper, every table
built from them), stored per model so that the tables can be rebuilt without the MPS files and without a solver.

A summary holds the objective sense, whether the objective is integral, the family-level summary of
families.summarize (as in the prompt), the constraint families in the order of their first row in the model file
(the base order of the random family orders of llmdw.search.relabel), and the columns of each constraint family (all
columns, and the columns with a nonzero coefficient). Files: <work>/work/model_summaries/<phase>/<instance>.json.
load() reads the file when it exists and otherwise computes the summary from the MPS file (which needs PySCIPOpt).

Usage
-----
python -m llmdw.instances.summaries --work data --phase templates --instance clsp_2x2 --mps data/work/templates/clsp_2x2/model.mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARY_DIR = "work/model_summaries"


def compute(model) -> dict:
    from llmdw.instances.families import family_of, objective_integral, summarize

    sequence, cols, nonzero = [], {}, {}
    for r in model.rows:
        f = family_of(r.name)
        if f not in cols:
            sequence.append(f)
            cols[f], nonzero[f] = set(), set()
        cols[f].update(r.coefs)
        nonzero[f].update(v for v, c in r.coefs.items() if c != 0)
    return {"sense": model.sense, "objective_integral": objective_integral(model), "summary": summarize(model),
            "family_sequence": sequence, "family_columns": {f: sorted(v) for f, v in cols.items()},
            "family_columns_nonzero": {f: sorted(v) for f, v in nonzero.items()}}


class ModelView:
    """The stored facts of one model, with the attributes the scoring uses in place of a model."""

    def __init__(self, data: dict):
        self.data = data
        self.sense = data["sense"]
        self.objective_integral = data["objective_integral"]
        self.summary = data["summary"]
        self.family_sequence = data["family_sequence"]

    def columns(self, family: str, nonzero: bool = False) -> set:
        key = "family_columns_nonzero" if nonzero else "family_columns"
        return set(self.data[key].get(family, ()))


def path(work: Path, phase: str, name: str) -> Path:
    return Path(work) / SUMMARY_DIR / phase / f"{name}.json"


def load(work: Path, phase: str, name: str, mps: str) -> ModelView:
    p = path(work, phase, name)
    if p.exists():
        return ModelView(json.loads(p.read_text(encoding="utf-8")))
    from llmdw.instances.families import read_model
    return ModelView(json.loads(json.dumps(compute(read_model(str(mps))))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--phase", required=True, help="templates, miplib, rich_models or large_models")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--mps", required=True)
    args = ap.parse_args(argv)
    from llmdw.instances.families import read_model
    p = path(Path(args.work), args.phase, args.instance)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(compute(read_model(args.mps)), indent=1), encoding="utf-8")
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
