"""Small MILPs with exact rational data and named constraint families.

Paper: Appendices B and C: the templates and the rich models.

Rows are lo <= a.x <= up (either side may be None). The objective is always minimized; a
maximization model is stored negated and flagged, so that every bound in this package is a
lower bound of a minimization problem and a gain is a nonnegative increase of that bound.
Row names follow the convention <Family>_<index> so that families.family_of recovers the
family (digit runs replaced by #) after an MPS round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


def F(v) -> Fraction | None:
    return None if v is None else Fraction(v)


@dataclass
class Var:
    name: str
    lb: Fraction | None
    ub: Fraction | None
    integer: bool


@dataclass
class Row:
    name: str
    family: str
    coefs: dict
    lo: Fraction | None
    up: Fraction | None


@dataclass
class MILP:
    name: str
    vars: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    obj: dict = field(default_factory=dict)
    maximize: bool = False
    description: str = ""
    family_text: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)

    def add_var(self, name: str, lb=0, ub=None, integer: bool = False) -> int:
        self.vars.append(Var(name, F(lb), F(ub), integer))
        return len(self.vars) - 1

    def add_row(self, name: str, family: str, coefs: dict, lo=None, up=None) -> int:
        c = {j: Fraction(v) for j, v in coefs.items() if Fraction(v) != 0}
        self.rows.append(Row(name, family, c, F(lo), F(up)))
        return len(self.rows) - 1

    def set_objective(self, coefs: dict, maximize: bool = False) -> None:
        self.maximize = maximize
        sign = -1 if maximize else 1
        self.obj = {j: sign * Fraction(v) for j, v in coefs.items() if Fraction(v) != 0}

    @property
    def families(self) -> list:
        seen = []
        for r in self.rows:
            if r.family not in seen:
                seen.append(r.family)
        return seen

    def rows_of(self, T) -> list:
        T = set(T)
        return [r for r in self.rows if r.family in T]

    def vars_of(self, T) -> list:
        js = set()
        for r in self.rows_of(T):
            js.update(r.coefs)
        return sorted(js)

    def report_bound(self, z: Fraction) -> Fraction:
        """Bound in the sense of the original model (max models are stored negated)."""
        return -z if self.maximize else z

    def to_scip(self):
        import pyscipopt

        m = pyscipopt.Model(self.name)
        xs = []
        for v in self.vars:
            xs.append(m.addVar(name=v.name, lb=None if v.lb is None else float(v.lb),
                               ub=None if v.ub is None else float(v.ub), vtype="I" if v.integer else "C"))
            if v.integer and v.lb == 0 and v.ub == 1:
                m.chgVarType(xs[-1], "B")
        for r in self.rows:
            expr = pyscipopt.quicksum(float(a) * xs[j] for j, a in r.coefs.items())
            if r.lo is not None and r.up is not None and r.lo == r.up:
                m.addCons(expr == float(r.lo), name=r.name)
            else:
                if r.lo is not None and r.up is not None:
                    raise ValueError(f"ranged row {r.name}: write it as two rows of the same family")
                elif r.lo is not None:
                    m.addCons(expr >= float(r.lo), name=r.name)
                else:
                    m.addCons(expr <= float(r.up), name=r.name)
        sign = -1 if self.maximize else 1
        m.setObjective(pyscipopt.quicksum(sign * float(a) * xs[j] for j, a in self.obj.items()),
                       "maximize" if self.maximize else "minimize")
        return m

    def write_mps(self, path: str) -> None:
        m = self.to_scip()
        m.writeProblem(path, trans=False, verbose=False)
