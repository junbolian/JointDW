"""Write the template instances used in the support experiments, with exact ground truth.

Paper: Section 6 and Appendix C: the templates (Table 4, Figure 1).

For each instance: <out>/<name>/model.mps, instance.json (description), model.md (one sentence per
constraint family, no values), ground_truth.json (exact gain of every family set, the zero-marginal
groups at the LP with at least two families ("joint_supports") and the source of each (polyhedral or
LP-geometric, hull.group_source), families that strengthen alone, the best set). Only instances small enough for
exact computation are written; the default set is fixed here so that runs are comparable.

Usage
-----
python -m llmdw.instances.export --out work/templates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmdw.exact import hull
from llmdw.instances import templates

# Single-item lot sizing is left out: there, convexifying both families is the whole MILP (no master).
DEFAULT = [
    ("clsp_2x2", lambda: templates.clsp(name="clsp_2x2")),
    ("clsp_3x2", lambda: templates.clsp(demand=((2, 1), (1, 2), (2, 2)), capacity=(6, 4), holding=(1, 1, 1),
                                        setup=(6, 5, 7), name="clsp_3x2")),
    ("clsp_2x2_b", lambda: templates.clsp(demand=((3, 2), (1, 3)), capacity=(5, 4), holding=(1, 2), setup=(7, 5),
                                          name="clsp_2x2_b")),
    ("gap_2x3", lambda: templates.gap(name="gap_2x3")),
    ("gap_2x3_b", lambda: templates.gap(weights=((4, 5, 3), (5, 4, 3)), capacity=(8, 7), cost=((2, 1, 9), (9, 8, 1)),
                                        name="gap_2x3_b")),
    ("needle_k4", lambda: templates.needle(k=4, hidden=(1, 2), name="needle_k4")),
    ("needle_k5", lambda: templates.needle(k=5, hidden=(0, 3), cycle_len=3, name="needle_k5")),
    # needle instances with a pair among six families and a triple among five, and the two classes with joint
    # supports outside lot sizing (network design, facility location)
    ("needle_k6", lambda: templates.needle(k=6, hidden=(2, 4), cycle_len=3, name="needle_k6")),
    ("needle_w3_k5", lambda: templates.needle(k=5, hidden=(0, 2, 4), cycle_len=3, name="needle_w3_k5")),
    ("ndp", lambda: templates.network_design(name="ndp")),
    ("ndp_b", lambda: templates.network_design(capacity=(2, 2, 2, 2, 2), fixed=(6, 6, 5, 6, 5), name="ndp_b")),
    ("ndp_c", lambda: templates.network_design(capacity=(3, 2, 2, 2, 3), fixed=(5, 4, 2, 4, 5),
                                               commodities=((0, 3, 2), (1, 3, 1)), name="ndp_c")),
    ("cfl", lambda: templates.facility_location(name="cfl")),
    ("cfl_b", lambda: templates.facility_location(demand=(2, 3, 2), capacity=(5, 4, 4), open_cost=(6, 5, 5),
                                                  assign_cost=((1, 2, 3), (3, 1, 2), (2, 3, 1)), name="cfl_b")),
    ("cfl_c", lambda: templates.facility_location(demand=(3, 2, 2, 1), capacity=(5, 5), open_cost=(7, 6),
                                                  assign_cost=((1, 2, 3, 2), (3, 2, 1, 1)), name="cfl_c")),
]


def ground_truth(m, max_size: int | None = None) -> dict:
    skip = (lambda T: len(T) > max_size) if max_size else None
    tab = hull.gain_table(m, skip=skip)
    groups = hull.zero_marginal_groups(tab)
    joint = [W for W in groups if len(W) >= 2]
    best = max(tab.values())
    best_sets = sorted((sorted(T) for T, v in tab.items() if v == best and T), key=len)
    return {"families": m.families, "lp_bound": str(m.report_bound(hull.lp_bound(m))),
            "gain": {"|".join(sorted(T)): str(v) for T, v in tab.items()},
            "joint_supports": [sorted(W) for W in joint],
            "group_sources": {"|".join(sorted(W)): hull.group_source(m, W) for W in joint},
            "strengthens_alone": sorted(f for f in m.families if tab[frozenset([f])] > 0),
            "best_sets": best_sets}


def write_instance(m, out: Path) -> Path:
    d = out / m.name
    d.mkdir(parents=True, exist_ok=True)
    m.to_scip().writeProblem(str(d / "model.mps"), trans=False, verbose=False)
    (d / "instance.json").write_text(json.dumps({"abstract_problem": m.description}, indent=1), encoding="utf-8")
    lines = ["# Formulation", "", "Constraint families:"]
    for f in m.families:
        lines.append(f"- {f}: {m.family_text.get(f, '')}")
    (d / "model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        gt = ground_truth(m)
    except hull.TooLarge:            # the full family set is the MILP itself; it is never a decomposition
        gt = ground_truth(m, max_size=len(m.families) - 1)
    (d / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    for name, make in DEFAULT:
        d = write_instance(make(), out)
        gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
        print(f"{name:12s} joint supports {gt['joint_supports']}  strengthens alone {gt['strengthens_alone']}",
              flush=True)


if __name__ == "__main__":
    main()
