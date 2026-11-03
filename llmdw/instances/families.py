"""Group the constraints and variables of an MPS model into named families.

Paper: Section 3 and Appendix E: constraint and variable families.

A family is the set of rows (or columns) whose names coincide after every maximal digit
run is replaced by ``#``: ``Item_3_Assignment`` and ``Item_17_Assignment`` both belong
to ``Item_#_Assignment``; ``cap[2,5]`` belongs to ``cap[#,#]``. This is the grouping
that GCG's ``consnamenonumbers`` classifier uses, so an LLM proposal over families and
GCG's class-based detection range over the same candidate decompositions.

For each constraint family the summary reports what an LLM needs to judge whether
convexifying the family can strengthen the bound: row count, sense, row length, the
variable families it touches, coefficient pattern, and the MIPLIB constraint type
(set partitioning, knapsack, bin packing, ...), following the type list of
Gamrath et al. (MIPLIB 2017) as implemented in ``miplib_type`` below.

Usage
-----
python -m llmdw.instances.families work/mps/gmu-35-40.mps.gz --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_DIGITS = re.compile(r"\d+")


def family_of(name: str) -> str:
    return _DIGITS.sub("#", name)


def index_of(name: str) -> tuple[int, ...]:
    return tuple(int(t) for t in _DIGITS.findall(name))


def var_family_map(var_type: dict) -> dict:
    """Variable -> family key. A name family whose columns have more than one type
    (binary, integer, continuous) is split by type, e.g. ``C#:B`` and ``C#:C``, so that an
    LLM can designate, say, only the binary columns of a family as linking variables
    (the analogue of GCG's variable-type classifier)."""
    by_name = defaultdict(set)
    for v, t in var_type.items():
        by_name[family_of(v)].add(t)
    out = {}
    for v, t in var_type.items():
        f = family_of(v)
        out[v] = f"{f}:{t}" if len(by_name[f]) > 1 else f
    return out


@dataclass
class Row:
    name: str
    lhs: float
    rhs: float
    coefs: dict  # var name -> coefficient


@dataclass
class Model:
    """Plain-Python view of a linear MILP read from an MPS/LP file."""

    rows: list[Row]
    var_type: dict  # name -> 'B' | 'I' | 'C'
    var_lb: dict
    var_ub: dict
    obj: dict
    sense: str  # 'minimize' | 'maximize'
    obj_offset: float = 0.0
    var_rows: dict = field(default_factory=dict)  # var name -> list of row indices

    def build_var_rows(self):
        vr = defaultdict(list)
        for i, r in enumerate(self.rows):
            for v in r.coefs:
                vr[v].append(i)
        self.var_rows = dict(vr)


def read_model(path: str) -> Model:
    """Read a model with PySCIPOpt without presolving and return a plain view."""
    import pyscipopt

    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(path)
    rows = []
    for c in m.getConss():
        if c.getConshdlrName() != "linear":
            raise ValueError(f"constraint {c.name} of type {c.getConshdlrName()} is not linear")
        coefs = m.getValsLinear(c)
        rows.append(Row(c.name, m.getLhs(c), m.getRhs(c), dict(coefs)))
    var_type, lb, ub, obj = {}, {}, {}, {}
    for v in m.getVars():
        t = v.vtype()
        var_type[v.name] = {"BINARY": "B", "INTEGER": "I", "IMPLINT": "I", "CONTINUOUS": "C"}[t]
        lb[v.name], ub[v.name], obj[v.name] = v.getLbOriginal(), v.getUbOriginal(), v.getObj()
    sense = m.getObjectiveSense()
    model = Model(rows, var_type, lb, ub, obj, sense, m.getObjoffset())
    model.build_var_rows()
    return model


def objective_integral(model: Model) -> bool:
    """True if every feasible point has an integral objective value: all columns with a nonzero
    objective coefficient are integer and their coefficients and the offset are integers."""
    for v, c in model.obj.items():
        if c != 0.0 and (model.var_type[v] == "C" or not float(c).is_integer()):
            return False
    return float(model.obj_offset).is_integer()


def miplib_type(row: Row, var_type: dict, inf: float = 1e19) -> str:
    """Classify a linear row by the MIPLIB 2017 constraint types (simplified).

    Types: empty, free, singleton, aggregation, precedence, variable_bound,
    set_partitioning, set_packing, set_covering, cardinality, invariant_knapsack,
    equation_knapsack, bin_packing, knapsack, mixed_binary, integer_knapsack,
    general_integer, mixed, continuous.
    """
    coefs, lhs, rhs = row.coefs, row.lhs, row.rhs
    n = len(coefs)
    if n == 0:
        return "empty"
    if lhs <= -inf and rhs >= inf:
        return "free"
    if n == 1:
        return "singleton"
    vals = list(coefs.values())
    if all(x < 0 for x in vals):  # flip the row so that coefficients are positive
        coefs = {v: -c for v, c in coefs.items()}
        lhs, rhs = -rhs, -lhs
        vals = list(coefs.values())
    lo_inf, hi_inf = lhs <= -inf, rhs >= inf
    is_eq = (not lo_inf) and (not hi_inf) and math.isclose(lhs, rhs, abs_tol=1e-9)
    vts = [var_type[v] for v in coefs]
    types = set(vts)
    if n == 2:
        if is_eq:
            return "aggregation"
        if vts[0] == vts[1] and math.isclose(abs(vals[0]), abs(vals[1])) and vals[0] * vals[1] < 0:
            return "precedence"
        if "C" in types and len(types) == 2:
            return "variable_bound"
    if types == {"B"}:
        if all(math.isclose(x, 1.0) for x in vals):
            if is_eq:
                return "set_partitioning" if math.isclose(rhs, 1.0) else "cardinality"
            if lo_inf and math.isclose(rhs, 1.0):
                return "set_packing"
            if hi_inf and math.isclose(lhs, 1.0):
                return "set_covering"
            return "invariant_knapsack"
        if is_eq:
            return "equation_knapsack"
        neg = [x for x in vals if x < 0]
        pos = [x for x in vals if x > 0]
        if len(neg) == 1 and pos and lo_inf and math.isclose(rhs, 0.0) and -neg[0] >= max(pos):
            return "bin_packing"
        if all(x > 0 for x in vals) and lo_inf:
            return "knapsack"
        return "mixed_binary"
    if types <= {"B", "I"}:
        if all(x > 0 for x in vals) and lo_inf:
            return "integer_knapsack"
        return "general_integer"
    if types == {"C"}:
        return "continuous"
    return "mixed"


def summarize(model: Model, max_examples: int = 2) -> dict:
    """Family-level summary of a model, ready to be serialized into the LLM prompt."""
    var_fam = var_family_map(model.var_type)
    vfam_members = defaultdict(list)
    for v, f in var_fam.items():
        vfam_members[f].append(v)

    cfam_rows = defaultdict(list)
    for i, r in enumerate(model.rows):
        cfam_rows[family_of(r.name)].append(i)

    cons_families = []
    for fam, idx in sorted(cfam_rows.items(), key=lambda kv: -len(kv[1])):
        rows = [model.rows[i] for i in idx]
        lengths = [len(r.coefs) for r in rows]
        touched = Counter()
        for r in rows:
            for v in r.coefs:
                touched[var_fam[v]] += 1
        types = Counter(miplib_type(r, model.var_type) for r in rows)
        senses = Counter(
            "=" if (r.lhs > -1e19 and r.rhs < 1e19 and math.isclose(r.lhs, r.rhs)) else
            ("<=" if r.lhs <= -1e19 else (">=" if r.rhs >= 1e19 else "ranged")) for r in rows)
        coef_vals = [abs(c) for r in rows for c in r.coefs.values()]
        n_index = Counter(len(index_of(r.name)) for r in rows)
        # how many other rows of the same family share a variable with a row (coupling within family)
        cons_families.append({
            "family": fam,
            "rows": len(rows),
            "sense": dict(senses),
            "miplib_types": dict(types.most_common()),
            "row_length": {"min": min(lengths), "median": sorted(lengths)[len(lengths) // 2], "max": max(lengths)},
            "variable_families": dict(touched.most_common()),
            "all_coefficients_unit": all(math.isclose(c, 1.0) for c in coef_vals),
            "distinct_abs_coefficients": len(set(round(c, 9) for c in coef_vals)),
            "index_positions": dict(n_index),
            "examples": [rows[k].name for k in range(min(max_examples, len(rows)))],
        })

    var_families = []
    for fam, members in sorted(vfam_members.items(), key=lambda kv: -len(kv[1])):
        tcount = Counter(model.var_type[v] for v in members)
        var_families.append({
            "family": fam,
            "columns": len(members),
            "types": dict(tcount),
            "objective_nonzero": sum(1 for v in members if model.obj.get(v, 0.0) != 0.0),
            "examples": members[:max_examples],
        })
    return {
        "num_rows": len(model.rows),
        "num_columns": len(model.var_type),
        "objective_sense": model.sense,
        "constraint_families": cons_families,
        "variable_families": var_families,
    }


def components(model: Model, block_rows: list[int], linking_vars: set | None = None) -> list[list[int]]:
    """Connected components of the given rows, where two rows are adjacent when they share
    a variable that is not a designated linking variable. Variables that also appear in
    master rows do not split components. Designated linking variables (``linking_vars``)
    may appear in several blocks; GCG then treats them as linking variables."""
    parent = {i: i for i in block_rows}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    block_set = set(block_rows)
    linking_vars = linking_vars or set()
    for v, rows in model.var_rows.items():
        if v in linking_vars:
            continue
        rs = [i for i in rows if i in block_set]
        for a, b in zip(rs, rs[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    comps = defaultdict(list)
    for i in block_rows:
        comps[find(i)].append(i)
    return sorted(comps.values(), key=lambda c: (-len(c), c[0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    s = summarize(read_model(args.model))
    text = json.dumps(s, indent=1)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
