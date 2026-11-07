"""Instance generators with known family-level structure.

Paper: Appendix C: the templates and the rich models.

Each generator returns a MILP whose rows carry family names (letters identify families, digit
runs are indices, so families.family_of maps row names back to the family), a natural-language
description and one sentence per family for the LLM experiments, and `facts` with the
structure that theory predicts. The exact computations in tests/test_support.py check every fact.

uls        single-item uncapacitated lot sizing: flow balance and setup families; (l,S)
           inequalities need both families (joint support of size 2)
clsp       multi-item lot sizing with a shared continuous capacity: every single family has zero
           gain, gains appear only for pairs (zero-marginal state at the LP)
needle     stable set on an odd cycle split into w families plus k-w distractor path families:
           gain is c * [W subset of T] (the hidden-group instances of the covering theorem)
geometric  two families with singleton supports whose gains are complementary through the LP
gap        generalized assignment: assignment family has the integrality property, capacity
           family alone strengthens the LP (singleton support)
network_design  multicommodity fixed-charge network design: flow balance (continuous flows), arc linking
           x_ka <= d_k y_a and arc capacity sum_k x_ka <= u_a y_a; no family strengthens alone, and on the
           exported data balance with capacity is the only joint support (cut-set inequalities)
facility_location  capacitated facility location with continuous assignment fractions: assignment,
           capacity sum_j d_j x_ij <= C_i y_i and linking x_ij <= y_i; no family strengthens alone,
           assignment with capacity is the joint support (cover inequalities on the opened capacity)
"""
from __future__ import annotations

import string
from fractions import Fraction

from llmdw.instances.milp import MILP


def uls(demand=(3, 2, 4), holding=(1, 1, 1), setup=(8, 8, 8), unit=(0, 0, 0), name="uls") -> MILP:
    T = len(demand)
    d = [Fraction(v) for v in demand]
    m = MILP(name)
    rem = [sum(d[t:]) for t in range(T)]
    x = [m.add_var(f"Produce_{t + 1}", 0, rem[t]) for t in range(T)]
    s = [m.add_var(f"Stock_{t + 1}", 0, sum(d)) for t in range(T)]
    y = [m.add_var(f"Setup_{t + 1}", 0, 1, integer=True) for t in range(T)]
    for t in range(T):
        coefs = {x[t]: 1, s[t]: -1}
        if t > 0:
            coefs[s[t - 1]] = 1
        m.add_row(f"Balance_{t + 1}", "Balance_#", coefs, lo=d[t], up=d[t])
    for t in range(T):
        m.add_row(f"SetupLink_{t + 1}", "SetupLink_#", {x[t]: 1, y[t]: -rem[t]}, up=0)
    obj = {}
    for t in range(T):
        obj[x[t]] = unit[t]
        obj[s[t]] = holding[t]
        obj[y[t]] = setup[t]
    m.set_objective(obj)
    m.description = (f"A single product is made over {T} periods to meet a known demand in each period. "
                     "Producing in a period requires paying a fixed setup cost in that period; stock carried "
                     "to the next period pays a holding cost. Minimize total setup and holding cost.")
    m.family_text = {"Balance_#": "inventory balance in each period: previous stock plus production equals "
                                  "demand plus next stock",
                     "SetupLink_#": "production in a period is allowed only if the setup in that period is paid"}
    m.facts = {"joint_support": [["Balance_#", "SetupLink_#"]], "vars": {"x": x, "s": s, "y": y},
               "demand": [str(v) for v in d]}
    return m


def uls_ls_inequality(m: MILP, l: int, S) -> tuple:
    """(l,S) inequality sum_{t in S} x_t <= sum_{t in S} d_{tl} y_t + s_l as alpha.x >= beta (1-based)."""
    v = m.facts["vars"]
    d = [Fraction(z) for z in m.facts["demand"]]
    alpha = {v["s"][l - 1]: Fraction(1)}
    for t in S:
        alpha[v["x"][t - 1]] = alpha.get(v["x"][t - 1], 0) - 1
        alpha[v["y"][t - 1]] = alpha.get(v["y"][t - 1], 0) + sum(d[t - 1:l])
    return alpha, Fraction(0)


def clsp(demand=((2, 3), (2, 2)), capacity=(5, 4), holding=(1, 1), setup=(6, 6), name="clsp") -> MILP:
    """Items i, periods t; continuous production and stock; shared capacity per period."""
    I, T = len(demand), len(demand[0])
    m = MILP(name)
    letters = string.ascii_uppercase
    x, s, y = {}, {}, {}
    for i in range(I):
        rem = [sum(Fraction(v) for v in demand[i][t:]) for t in range(T)]
        for t in range(T):
            ub = min(Fraction(capacity[t]), rem[t])
            x[i, t] = m.add_var(f"Produce_{letters[i]}_{t + 1}", 0, ub)
            s[i, t] = m.add_var(f"Stock_{letters[i]}_{t + 1}", 0, sum(Fraction(v) for v in demand[i]))
            y[i, t] = m.add_var(f"Setup_{letters[i]}_{t + 1}", 0, 1, integer=True)
    for i in range(I):
        for t in range(T):
            coefs = {x[i, t]: 1, s[i, t]: -1}
            if t > 0:
                coefs[s[i, t - 1]] = 1
            m.add_row(f"Balance_{i + 1}_{t + 1}", "Balance_#_#", coefs, lo=demand[i][t], up=demand[i][t])
    for i in range(I):
        for t in range(T):
            ub = m.vars[x[i, t]].ub
            m.add_row(f"SetupLink_{i + 1}_{t + 1}", "SetupLink_#_#", {x[i, t]: 1, y[i, t]: -ub}, up=0)
    for t in range(T):
        m.add_row(f"Capacity_{t + 1}", "Capacity_#", {x[i, t]: 1 for i in range(I)}, up=capacity[t])
    obj = {}
    for i in range(I):
        for t in range(T):
            obj[s[i, t]] = holding[i]
            obj[y[i, t]] = setup[i]
    m.set_objective(obj)
    m.description = (f"{I} products are made on one machine over {T} periods to meet known demands. Each period "
                     "the machine has a limited production capacity shared by all products. Producing a product "
                     "in a period requires a setup for that product in that period. Minimize setup and holding cost.")
    m.family_text = {"Balance_#_#": "inventory balance of each product in each period",
                     "SetupLink_#_#": "a product can be produced in a period only if its setup is paid",
                     "Capacity_#": "total production of all products in a period is at most the capacity"}
    m.facts = {"joint_support": [["Balance_#_#", "SetupLink_#_#"]]}
    return m


def needle(k: int = 4, hidden=(0, 1), cycle_len: int = 5, name="needle") -> MILP:
    """Maximum stable set on an odd cycle whose edges are split into len(hidden) contiguous paths, one
    family each, plus k - len(hidden) distractor families (paths on fresh vertices, same lengths).
    The gain of convexifying T is 1/2 if every hidden family is in T and 0 otherwise."""
    w = len(hidden)
    assert cycle_len % 2 == 1 and cycle_len >= max(3, w)
    letters = string.ascii_uppercase
    m = MILP(name)
    sizes = [cycle_len // w + (1 if r < cycle_len % w else 0) for r in range(w)]
    cyc = [m.add_var(f"Node_{v + 1}", 0, 1, integer=True) for v in range(cycle_len)]
    fam_names = [f"Conflict{letters[f]}_#" for f in range(k)]
    edge_no = 0
    start = 0
    for piece, f in zip(sizes, hidden):
        for e in range(piece):
            a, b = (start + e) % cycle_len, (start + e + 1) % cycle_len
            edge_no += 1
            m.add_row(f"Conflict{letters[f]}_{edge_no}", fam_names[f], {cyc[a]: 1, cyc[b]: 1}, up=1)
        start += piece
    others = [f for f in range(k) if f not in hidden]
    nxt = cycle_len
    for r, f in enumerate(others):
        length = sizes[r % w]
        verts = [m.add_var(f"Node_{nxt + q + 1}", 0, 1, integer=True) for q in range(length + 1)]
        nxt += length + 1
        for q in range(length):
            edge_no += 1
            m.add_row(f"Conflict{letters[f]}_{edge_no}", fam_names[f], {verts[q]: 1, verts[q + 1]: 1}, up=1)
    m.set_objective({j: 1 for j in range(len(m.vars))}, maximize=True)
    m.rows.sort(key=lambda r: r.family)
    m.description = ("Select as many items as possible; some pairs of items conflict and cannot both be selected.")
    m.family_text = {fn: "pairs of items that cannot both be selected" for fn in fam_names}
    m.facts = {"hidden": [fam_names[f] for f in hidden], "gain": "1/2"}
    return m


def geometric(name="geometric") -> MILP:
    """max y  s.t. y <= x1 + x2, y <= 1/2, 2 x1 <= 1, 2 x2 <= 1, x binary. Convexifying either single
    family fixes one x to 0 and the LP moves weight to the other; both are needed for a gain."""
    m = MILP(name)
    x1 = m.add_var("Use_1", 0, 1, integer=True)
    x2 = m.add_var("Use_2", 0, 1, integer=True)
    yv = m.add_var("Level_1", 0, 1)
    m.add_row("LimitA_1", "LimitA_#", {x1: 2}, up=1)
    m.add_row("LimitB_1", "LimitB_#", {x2: 2}, up=1)
    m.add_row("Cover_1", "Cover_#", {yv: 1, x1: -1, x2: -1}, up=0)
    m.add_row("Cap_1", "Cap_#", {yv: 1}, up=Fraction(1, 2))
    m.set_objective({yv: 1}, maximize=True)
    m.facts = {"singleton_supports": [["LimitA_#"], ["LimitB_#"]], "joint_gain": [["LimitA_#", "LimitB_#"]]}
    return m


def gap(weights=((5, 5, 4), (5, 5, 4)), capacity=(9, 9), cost=((1, 1, 10), (10, 10, 1)), name="gap") -> MILP:
    I, J = len(weights), len(weights[0])
    letters = string.ascii_uppercase
    m = MILP(name)
    x = {(i, j): m.add_var(f"Assign_{letters[i]}_{j + 1}", 0, 1, integer=True) for i in range(I) for j in range(J)}
    for j in range(J):
        m.add_row(f"Job_{j + 1}", "Job_#", {x[i, j]: 1 for i in range(I)}, lo=1, up=1)
    for i in range(I):
        m.add_row(f"Capacity_{i + 1}", "Capacity_#", {x[i, j]: weights[i][j] for j in range(J)}, up=capacity[i])
    m.set_objective({x[i, j]: cost[i][j] for i in range(I) for j in range(J)})
    m.description = ("Assign each job to exactly one machine. A job uses machine-dependent capacity and has a "
                     "machine-dependent cost; each machine has a capacity. Minimize total cost.")
    m.family_text = {"Job_#": "each job is assigned to exactly one machine",
                     "Capacity_#": "the jobs assigned to a machine fit its capacity"}
    m.facts = {"integral_alone": ["Job_#"], "singleton_support": [["Capacity_#"]]}
    return m


def network_design(n_nodes=4, arcs=((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)), commodities=((0, 3, 2), (0, 2, 1)),
                   capacity=(2, 2, 2, 2, 2), fixed=(4, 4, 3, 4, 3), unit=(1, 1, 1, 1, 1), name="ndp") -> MILP:
    """Multicommodity fixed-charge network design. Commodity k = (origin, destination, demand) is routed by
    continuous flows x_ka in [0, d_k]; opening arc a costs fixed[a] and allows flow up to capacity[a]."""
    letters = string.ascii_uppercase
    m = MILP(name)
    x = {(k, a): m.add_var(f"Flow_{letters[k]}_{a + 1}", 0, commodities[k][2])
         for k in range(len(commodities)) for a in range(len(arcs))}
    y = [m.add_var(f"Open_{a + 1}", 0, 1, integer=True) for a in range(len(arcs))]
    for k, (o, dest, dem) in enumerate(commodities):
        for v in range(n_nodes):
            coefs = {}
            for a, (i, j) in enumerate(arcs):
                if i == v:
                    coefs[x[k, a]] = coefs.get(x[k, a], 0) + 1
                if j == v:
                    coefs[x[k, a]] = coefs.get(x[k, a], 0) - 1
            rhs = dem if v == o else (-dem if v == dest else 0)
            m.add_row(f"FlowBalance_{k + 1}_{v + 1}", "FlowBalance_#_#", coefs, lo=rhs, up=rhs)
    for k, (_, _, dem) in enumerate(commodities):
        for a in range(len(arcs)):
            m.add_row(f"ArcLink_{k + 1}_{a + 1}", "ArcLink_#_#", {x[k, a]: 1, y[a]: -dem}, up=0)
    for a in range(len(arcs)):
        m.add_row(f"ArcCapacity_{a + 1}", "ArcCapacity_#",
                  {**{x[k, a]: 1 for k in range(len(commodities))}, y[a]: -capacity[a]}, up=0)
    obj = {y[a]: fixed[a] for a in range(len(arcs))}
    for (k, a), j in x.items():
        obj[j] = unit[a]
    m.set_objective(obj)
    m.description = (f"{len(commodities)} commodities must be shipped from their origins to their destinations "
                     f"through a network of {n_nodes} nodes and {len(arcs)} directed arcs. An arc can carry flow "
                     "only if it is opened, which has a fixed cost; an opened arc has a capacity shared by all "
                     "commodities, and every unit of flow has a per-arc cost. Minimize fixed plus flow cost.")
    m.family_text = {"FlowBalance_#_#": "flow conservation of each commodity at each node",
                     "ArcLink_#_#": "a commodity can use an arc only if the arc is opened, up to its demand",
                     "ArcCapacity_#": "the total flow on an arc is at most its capacity if the arc is opened"}
    m.facts = {"joint_support": [["ArcCapacity_#", "FlowBalance_#_#"]]}     # for the default data (exact check)
    return m


def facility_location(demand=(2, 2, 3), capacity=(4, 4), open_cost=(5, 6), assign_cost=((1, 3, 2), (3, 1, 2)),
                      name="cfl") -> MILP:
    """Capacitated facility location with continuous assignment fractions x_ij in [0, 1] and binary y_i."""
    I, J = len(capacity), len(demand)
    letters = string.ascii_uppercase
    m = MILP(name)
    x = {(i, j): m.add_var(f"Serve_{letters[i]}_{j + 1}", 0, 1) for i in range(I) for j in range(J)}
    y = [m.add_var(f"Open_{letters[i]}", 0, 1, integer=True) for i in range(I)]
    for j in range(J):
        m.add_row(f"Assign_{j + 1}", "Assign_#", {x[i, j]: 1 for i in range(I)}, lo=1, up=1)
    for i in range(I):
        m.add_row(f"Capacity_{i + 1}", "Capacity_#", {**{x[i, j]: demand[j] for j in range(J)}, y[i]: -capacity[i]},
                  up=0)
    for i in range(I):
        for j in range(J):
            m.add_row(f"Link_{i + 1}_{j + 1}", "Link_#_#", {x[i, j]: 1, y[i]: -1}, up=0)
    obj = {y[i]: open_cost[i] for i in range(I)}
    for (i, j), v in x.items():
        obj[v] = assign_cost[i][j] * demand[j]
    m.set_objective(obj)
    m.description = (f"{J} customers with known demands are served from {I} candidate facilities. A facility "
                     "can serve customers only if it is opened, which has a fixed cost, and the demand it serves "
                     "is at most its capacity; a customer's demand may be split among facilities, and serving "
                     "costs depend on the facility and the customer. Minimize opening plus serving cost.")
    m.family_text = {"Assign_#": "the demand of each customer is fully served",
                     "Capacity_#": "the demand served by a facility is at most its capacity if it is opened",
                     "Link_#_#": "a customer can be served by a facility only if the facility is opened"}
    m.facts = {"joint_support": [["Assign_#", "Capacity_#"]]}
    return m


# ---------------------------------------------------------------- single connected models with many families

def rich_lot_sizing(demand=((2, 2, 2), (0, 3, 3)), capacity=(7, 7, 7), setup_time=(1, 1), min_lot=(2, 2),
                    storage=(6, 6, 6), safety=(1, 0), overtime=(2, 2, 2), setup_budget=3, holding=(1, 1),
                    setup=(6, 7), overtime_cost=3, max_setups=None, name="rich_ls") -> MILP:
    """Two-item multi-period lot sizing with setup times, minimum lots, storage, safety stock, one setup per period,
    overtime and a setup budget (9 constraint families on shared variables; I * T binary setups). max_setups gives the
    limit of MaxSetupsPerPeriod per period (default 1 in every period, the rich models of the paper)."""
    I, T = len(demand), len(demand[0])
    letters = string.ascii_uppercase
    m = MILP(name)
    x, s, y, o = {}, {}, {}, {}
    tot = [sum(Fraction(v) for v in demand[i]) for i in range(I)]
    for t in range(T):
        o[t] = m.add_var(f"Overtime_{t + 1}", 0, overtime[t])
    for i in range(I):
        for t in range(T):
            x[i, t] = m.add_var(f"Produce_{letters[i]}_{t + 1}", 0, min(Fraction(capacity[t] + overtime[t]),
                                                                       tot[i] + safety[i]))
            s[i, t] = m.add_var(f"Stock_{letters[i]}_{t + 1}", 0, tot[i] + safety[i])
            y[i, t] = m.add_var(f"Setup_{letters[i]}_{t + 1}", 0, 1, integer=True)
    for i in range(I):
        for t in range(T):
            coefs = {x[i, t]: 1, s[i, t]: -1}
            if t > 0:
                coefs[s[i, t - 1]] = 1
            m.add_row(f"Balance_{i + 1}_{t + 1}", "Balance_#_#", coefs, lo=demand[i][t], up=demand[i][t])
    for i in range(I):
        for t in range(T):
            m.add_row(f"SetupLink_{i + 1}_{t + 1}", "SetupLink_#_#", {x[i, t]: 1, y[i, t]: -m.vars[x[i, t]].ub}, up=0)
    for t in range(T):
        m.add_row(f"Capacity_{t + 1}", "Capacity_#",
                  {**{x[i, t]: 1 for i in range(I)}, **{y[i, t]: setup_time[i] for i in range(I)}, o[t]: -1},
                  up=capacity[t])
    for i in range(I):
        for t in range(T):
            m.add_row(f"MinLot_{i + 1}_{t + 1}", "MinLot_#_#", {x[i, t]: 1, y[i, t]: -min_lot[i]}, lo=0)
    for t in range(T):
        m.add_row(f"StorageCap_{t + 1}", "StorageCap_#", {s[i, t]: 1 for i in range(I)}, up=storage[t])
    for i in range(I):
        for t in range(T - 1):
            m.add_row(f"SafetyStock_{i + 1}_{t + 1}", "SafetyStock_#_#", {s[i, t]: 1}, lo=safety[i])
    for t in range(T):
        m.add_row(f"MaxSetupsPerPeriod_{t + 1}", "MaxSetupsPerPeriod_#", {y[i, t]: 1 for i in range(I)},
                  up=1 if max_setups is None else max_setups[t])
    for t in range(T):
        m.add_row(f"OvertimeLimit_{t + 1}", "OvertimeLimit_#", {o[t]: 1}, up=overtime[t])
    m.add_row("SetupBudget", "SetupBudget", {y[i, t]: 1 for i in range(I) for t in range(T)}, up=setup_budget)
    obj = {o[t]: overtime_cost for t in range(T)}
    for i in range(I):
        for t in range(T):
            obj[s[i, t]] = holding[i]
            obj[y[i, t]] = setup[i]
    m.set_objective(obj)
    m.description = (f"{I} products are made on one machine over {T} periods to meet known demands. Each period the "
                     "machine has a limited production capacity shared by all products, which overtime can extend at "
                     "a cost up to a limit. Producing a product in a period requires a setup, which uses part of the "
                     "capacity, and at least a minimum lot; at most one product can be set up per period, and the "
                     "total number of setups is limited. Stock is limited by the storage space, and a safety stock "
                     "must be kept at the end of every period but the last. Minimize setup, holding and overtime cost.")
    if max_setups is not None:
        m.description = m.description.replace("at most one product can be set up per period",
                                              "the number of products set up in a period is limited")
    m.family_text = {"Balance_#_#": "inventory balance of each product in each period",
                     "SetupLink_#_#": "a product can be produced in a period only if its setup is paid",
                     "Capacity_#": "production plus setup times in a period are at most the capacity plus overtime",
                     "MinLot_#_#": "a product that is set up in a period is produced at least its minimum lot",
                     "StorageCap_#": "the total stock at the end of a period is at most the storage space",
                     "SafetyStock_#_#": "the stock of each product at the end of a period is at least its safety stock",
                     "MaxSetupsPerPeriod_#": "at most one product is set up in a period",
                     "OvertimeLimit_#": "the overtime in a period is at most its limit",
                     "SetupBudget": "the total number of setups is at most the budget"}
    if max_setups is not None:
        m.family_text["MaxSetupsPerPeriod_#"] = "at most a given number of products is set up in a period"
    return m


def rich_network_design(n_nodes=4, arcs=((0, 1), (1, 0), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)),
                        commodities=((0, 3, 2), (0, 2, 1)), capacity=(2, 2, 2, 2, 2, 2, 2),
                        fixed=(4, 4, 4, 3, 3, 4, 3), unit=(1, 1, 1, 1, 1, 1, 1), budget=16, max_out=(2, 2, 2, 1),
                        name="rich_nd") -> MILP:
    """Multicommodity fixed-charge network design with a build budget, out-degree limits, opposite arcs and an
    opened incoming arc at every destination (7 constraint families on shared variables; one binary per arc)."""
    letters = string.ascii_uppercase
    m = MILP(name)
    x = {(k, a): m.add_var(f"Flow_{letters[k]}_{a + 1}", 0, commodities[k][2])
         for k in range(len(commodities)) for a in range(len(arcs))}
    y = [m.add_var(f"Open_{a + 1}", 0, 1, integer=True) for a in range(len(arcs))]
    for k, (org, dest, dem) in enumerate(commodities):
        for v in range(n_nodes):
            coefs = {}
            for a, (i, j) in enumerate(arcs):
                if i == v:
                    coefs[x[k, a]] = coefs.get(x[k, a], 0) + 1
                if j == v:
                    coefs[x[k, a]] = coefs.get(x[k, a], 0) - 1
            rhs = dem if v == org else (-dem if v == dest else 0)
            m.add_row(f"FlowBalance_{k + 1}_{v + 1}", "FlowBalance_#_#", coefs, lo=rhs, up=rhs)
    for k, (_, _, dem) in enumerate(commodities):
        for a in range(len(arcs)):
            m.add_row(f"ArcLink_{k + 1}_{a + 1}", "ArcLink_#_#", {x[k, a]: 1, y[a]: -dem}, up=0)
    for a in range(len(arcs)):
        m.add_row(f"ArcCapacity_{a + 1}", "ArcCapacity_#",
                  {**{x[k, a]: 1 for k in range(len(commodities))}, y[a]: -capacity[a]}, up=0)
    m.add_row("BuildBudget", "BuildBudget", {y[a]: fixed[a] for a in range(len(arcs))}, up=budget)
    for v in range(n_nodes):
        out = [a for a, (i, _) in enumerate(arcs) if i == v]
        if out:
            m.add_row(f"NodeOutDegree_{v + 1}", "NodeOutDegree_#", {y[a]: 1 for a in out}, up=max_out[v])
    for a, (i, j) in enumerate(arcs):
        b = arcs.index((j, i)) if (j, i) in arcs else None
        if b is not None and a < b:
            m.add_row(f"OppositeArcs_{a + 1}", "OppositeArcs_#", {y[a]: 1, y[b]: 1}, up=1)
    sources = {org for org, _, _ in commodities}
    for v in sorted({dest for _, dest, _ in commodities} - sources):
        m.add_row(f"InArcCover_{v + 1}", "InArcCover_#", {y[a]: 1 for a, (_, j) in enumerate(arcs) if j == v}, lo=1)
    obj = {y[a]: fixed[a] for a in range(len(arcs))}
    for (k, a), j in x.items():
        obj[j] = unit[a]
    m.set_objective(obj)
    m.description = (f"{len(commodities)} commodities must be shipped from their origins to their destinations "
                     f"through a network of {n_nodes} nodes and {len(arcs)} directed arcs. An arc can carry flow only "
                     "if it is opened, which has a fixed cost, and the fixed costs of the opened arcs are limited by "
                     "a budget; an opened arc has a capacity shared by all commodities, and every unit of flow has a "
                     "per-arc cost. At most a given number of arcs may be opened out of each node, an arc and its "
                     "opposite arc cannot both be opened, and every destination needs an opened incoming arc. "
                     "Minimize fixed plus flow cost.")
    m.family_text = {"FlowBalance_#_#": "flow conservation of each commodity at each node",
                     "ArcLink_#_#": "a commodity can use an arc only if the arc is opened, up to its demand",
                     "ArcCapacity_#": "the total flow on an arc is at most its capacity if the arc is opened",
                     "BuildBudget": "the fixed costs of the opened arcs are at most the budget",
                     "NodeOutDegree_#": "at most a given number of arcs is opened out of each node",
                     "OppositeArcs_#": "an arc and its opposite arc are not both opened",
                     "InArcCover_#": "every destination that is not an origin has an opened incoming arc"}
    return m


def rich_facility_location(demand=(2, 2, 3), capacity=(4, 4, 5), open_cost=(5, 6, 6),
                           assign_cost=((1, 3, 2), (3, 1, 2), (2, 2, 1)), budget=12, min_open=2,
                           min_use=(2, 2, 3), conflict=(0, 1), name="rich_fl") -> MILP:
    """Capacitated facility location with an opening budget, a minimum number of open facilities, minimum
    utilization of an opened facility and one conflicting pair (7 constraint families; one binary per facility)."""
    I, J = len(capacity), len(demand)
    letters = string.ascii_uppercase
    m = MILP(name)
    x = {(i, j): m.add_var(f"Serve_{letters[i]}_{j + 1}", 0, 1) for i in range(I) for j in range(J)}
    y = [m.add_var(f"Open_{letters[i]}", 0, 1, integer=True) for i in range(I)]
    for j in range(J):
        m.add_row(f"Assign_{j + 1}", "Assign_#", {x[i, j]: 1 for i in range(I)}, lo=1, up=1)
    for i in range(I):
        m.add_row(f"Capacity_{i + 1}", "Capacity_#", {**{x[i, j]: demand[j] for j in range(J)}, y[i]: -capacity[i]},
                  up=0)
    for i in range(I):
        for j in range(J):
            m.add_row(f"Link_{i + 1}_{j + 1}", "Link_#_#", {x[i, j]: 1, y[i]: -1}, up=0)
    m.add_row("OpenBudget", "OpenBudget", {y[i]: open_cost[i] for i in range(I)}, up=budget)
    m.add_row("OpenCount", "OpenCount", {y[i]: 1 for i in range(I)}, lo=min_open)
    for i in range(I):
        m.add_row(f"MinUtilization_{i + 1}", "MinUtilization_#",
                  {**{x[i, j]: demand[j] for j in range(J)}, y[i]: -min_use[i]}, lo=0)
    a, b = conflict
    m.add_row("FacilityConflict", "FacilityConflict", {y[a]: 1, y[b]: 1}, up=1)
    obj = {y[i]: open_cost[i] for i in range(I)}
    for (i, j), v in x.items():
        obj[v] = assign_cost[i][j] * demand[j]
    m.set_objective(obj)
    m.description = (f"{J} customers with known demands are served from {I} candidate facilities. A facility can "
                     "serve customers only if it is opened, which has a fixed cost, and the demand it serves is at "
                     "most its capacity and, if it is opened, at least a minimum utilization; a customer's demand may "
                     "be split among facilities, and serving costs depend on the facility and the customer. The "
                     "opening costs are limited by a budget, at least a given number of facilities must be opened, "
                     f"and facilities {letters[a]} and {letters[b]} cannot both be opened. Minimize opening plus "
                     "serving cost.")
    m.family_text = {"Assign_#": "the demand of each customer is fully served",
                     "Capacity_#": "the demand served by a facility is at most its capacity if it is opened",
                     "Link_#_#": "a customer can be served by a facility only if the facility is opened",
                     "OpenBudget": "the opening costs of the opened facilities are at most the budget",
                     "OpenCount": "at least a given number of facilities is opened",
                     "MinUtilization_#": "an opened facility serves at least its minimum utilization",
                     "FacilityConflict": "the two conflicting facilities are not both opened"}
    return m
