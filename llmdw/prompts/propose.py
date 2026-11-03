"""Build the decomposition prompt, call the LLM, and validate its proposals.

Paper: Appendix D: the prompt, its ranked and data variants, and the answer format.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from llmdw.providers.llm_client import extract_json

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_PATH = PROMPTS / "decomp_v2.md"
MAX_FORMULATION_CHARS = 12000
MAX_DESCRIPTION_CHARS = 8000


def load_template(path: Path = PROMPT_PATH) -> dict:
    text = path.read_text(encoding="utf-8")
    parts, current = {}, None
    for line in text.splitlines(keepends=True):
        if line.startswith("# ") and line.strip()[2:] in ("SYSTEM", "USER", "FEEDBACK"):
            current = line.strip()[2:]
            parts[current] = ""
        elif current:
            parts[current] += line
    return {k: v.strip("\n") + "\n" for k, v in parts.items()}


_LEAK_HEADING = re.compile(r"(optimal|best[ -]known|objective value|solution value|reference value)", re.I)


def strip_answer_sections(markdown: str) -> str:
    """Remove every markdown section whose heading mentions an optimal / best-known / objective
    value (some MIPLIB-NL model.md files state the optimum, e.g. square23-47), and every other
    line that does. The reference answer must never reach a prompt."""
    out, skip_level = [], None
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and _LEAK_HEADING.search(m.group(2)):
                skip_level = level
                continue
        if skip_level is not None:
            continue
        if re.search(r"optimal (objective )?value|best[ -]known", line, re.I):
            continue
        out.append(line)
    return "\n".join(out)


def instance_text(instance_dir: str | None) -> tuple[str, str]:
    """Description and formulation text of a MIPLIB-NL instance directory (answer-free)."""
    if not instance_dir:
        return "(not available)", "(not available)"
    d = Path(instance_dir)
    desc = "(not available)"
    js = d / "instance.json"
    if js.exists():
        with open(js, encoding="utf-8") as fh:
            meta = json.load(fh)
        lines = [meta.get("abstract_problem", "").strip()]
        if meta.get("parameters"):
            lines.append("Parameters: " + json.dumps(meta["parameters"]))
        for f in (meta.get("files") or {}).values():
            lines.append(f"Data file {f.get('path')}: {f.get('description')}")
        desc = strip_answer_sections("\n".join(x for x in lines if x))[:MAX_DESCRIPTION_CHARS]
    form = "(not available)"
    md = d / "model.md"
    if md.exists():
        form = strip_answer_sections(md.read_text(encoding="utf-8", errors="replace"))[:MAX_FORMULATION_CHARS]
    return desc, form


def constraint_table(summary: dict) -> str:
    rows = []
    for f in summary["constraint_families"]:
        types = ", ".join(f"{k}:{v}" for k, v in f["miplib_types"].items())
        sense = ", ".join(f"{k}:{v}" for k, v in f["sense"].items())
        vf = ", ".join(f"{k}:{v}" for k, v in f["variable_families"].items())
        rl = f["row_length"]
        rows.append(f"{f['family']} | {f['rows']} | {sense} | {types} | {rl['min']}/{rl['median']}/{rl['max']} | "
                    f"{vf} | {'yes' if f['all_coefficients_unit'] else 'no'}")
    return "\n".join(rows)


def variable_table(summary: dict) -> str:
    rows = []
    for f in summary["variable_families"]:
        types = ", ".join(f"{k}:{v}" for k, v in f["types"].items())
        rows.append(f"{f['family']} | {f['columns']} | {types} | {f['objective_nonzero']}")
    return "\n".join(rows)


def feedback_rows(history: list[dict]) -> str:
    out = []
    for h in history:
        out.append(" | ".join([
            ", ".join(h.get("master_families") or []) or "-",
            ", ".join(h.get("linking_var_families") or []) or "-",
            str(h.get("n_blocks")), str(h.get("largest_block_rows")), str(h.get("status")),
            _fmt(h.get("cmp_bound", h.get("dual_bound"))), _fmt(h.get("primal_bound")), _fmt(h.get("solving_time")),
            _fmt(h.get("pricing_seconds"))]))
    return "\n".join(out)


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def anonymize_maps(summary: dict) -> dict:
    """Name maps for the ablation that hides problem semantics (LLMDW_ANONYMIZE=1): constraint
    families become F1..Fk and variable families V1..Vm in table order."""
    c = {f["family"]: f"F{i + 1}" for i, f in enumerate(summary["constraint_families"])}
    v = {f["family"]: f"V{i + 1}" for i, f in enumerate(summary["variable_families"])}
    return {"cons": c, "vars": v, "cons_inv": {b: a for a, b in c.items()},
            "vars_inv": {b: a for a, b in v.items()}}


def _anon_summary(summary: dict, maps: dict) -> dict:
    out = json.loads(json.dumps(summary))
    for f in out["constraint_families"]:
        f["family"] = maps["cons"][f["family"]]
        f["variable_families"] = {maps["vars"].get(k, k): n for k, n in f["variable_families"].items()}
        f["examples"] = []
    for f in out["variable_families"]:
        f["family"] = maps["vars"][f["family"]]
        f["examples"] = []
    return out


def _anon_history(history: list[dict], maps: dict) -> list[dict]:
    out = []
    for h in history:
        h2 = dict(h)
        h2["master_families"] = [maps["cons"].get(x, x) for x in h.get("master_families") or []]
        h2["linking_var_families"] = [maps["vars"].get(x, x) for x in h.get("linking_var_families") or []]
        out.append(h2)
    return out


def _deanon_answer(answer: dict, maps: dict) -> dict:
    a = json.loads(json.dumps(answer))
    for c in a.get("candidates") or []:
        if isinstance(c, dict):
            c["master_constraint_families"] = [maps["cons_inv"].get(x, x) for x in c.get("master_constraint_families") or []]
            c["linking_variable_families"] = [maps["vars_inv"].get(x, x) for x in c.get("linking_variable_families") or []]
    if isinstance(a.get("families"), dict):
        a["families"] = {maps["cons_inv"].get(k2, k2): v for k2, v in a["families"].items()}
    for sup in a.get("joint_supports") or []:
        if isinstance(sup, dict) and isinstance(sup.get("families"), list):
            sup["families"] = [maps["cons_inv"].get(x, x) if isinstance(x, str) else x for x in sup["families"]]
    return a


def anonymize_enabled() -> bool:
    return os.environ.get("LLMDW_ANONYMIZE") == "1"


def model_lp_text(instance_dir: str) -> str:
    """The complete instance (instance_dir/model.mps) in CPLEX LP format, written by SCIP with the model's own row
    and column names (the names behind the constraint and variable families); SCIP's statistics comment, which
    names the problem, is removed."""
    import tempfile

    import pyscipopt
    m = pyscipopt.Model()
    m.hideOutput()
    m.readProblem(str(Path(instance_dir) / "model.mps"))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.lp"
        m.writeProblem(str(path), trans=False, verbose=False)
        text = path.read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines() if not line.startswith("\\")).strip("\n")


def build_messages(summary: dict, instance_dir: str | None, k: int, time_limit: float,
                   history: list[dict] | None = None, lp_bound: float | None = None,
                   template: dict | None = None, maps: dict | None = None,
                   withhold: bool | None = None) -> list[dict]:
    t = template or load_template()
    desc, form = instance_text(instance_dir)
    if maps is not None:  # renamed families (anonymized or permuted)
        summary = _anon_summary(summary, maps)
        history = _anon_history(history, maps) if history else history
    if withhold if withhold is not None else maps is not None:
        desc, form = "(withheld)", "(withheld)"
    feedback = ""
    if history:
        feedback = "\n" + t["FEEDBACK"].format(lp_bound=_fmt(lp_bound), rows=feedback_rows(history), k=k)
    user = t["USER"].format(
        time_limit=int(time_limit), description=desc, formulation=form,
        num_rows=summary["num_rows"], num_columns=summary["num_columns"], sense=summary["objective_sense"],
        constraint_table=constraint_table(summary), variable_table=variable_table(summary),
        feedback=feedback, k=k, **({"model_lp": model_lp_text(instance_dir)} if "{model_lp}" in t["USER"] else {}))
    return [{"role": "system", "content": t["SYSTEM"].strip()}, {"role": "user", "content": user}]


def validate(answer: dict, summary: dict, k: int) -> tuple[list[dict], list[str]]:
    """Return (valid candidates, problems). A valid candidate keeps at least one and not all
    constraint families in the master and names only existing families."""
    cfams = {f["family"] for f in summary["constraint_families"]}
    vfams = {f["family"] for f in summary["variable_families"]}
    problems, valid, seen = [], [], set()
    if not isinstance(answer, dict):
        return [], ["answer is not a JSON object"]
    cands = answer.get("candidates") or []
    if not isinstance(cands, list):
        return [], ["candidates is not a list"]
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            problems.append(f"candidate {i}: not an object")
            continue
        master = c.get("master_constraint_families") or []
        linking = c.get("linking_variable_families") or []
        if (not isinstance(master, list) or not isinstance(linking, list)
                or not all(isinstance(x, str) for x in master + linking)):
            problems.append(f"candidate {i}: family lists malformed")
            continue
        bad = [m for m in master if m not in cfams] + [v for v in linking if v not in vfams]
        if bad:
            problems.append(f"candidate {i}: unknown families {bad}")
            continue
        ms, ls = frozenset(master), frozenset(linking)
        if not ms and not ls:
            problems.append(f"candidate {i}: empty master and no linking variables")
            continue
        if ms == cfams:
            problems.append(f"candidate {i}: every family in the master, nothing convexified")
            continue
        if (ms, ls) in seen:
            problems.append(f"candidate {i}: duplicate")
            continue
        seen.add((ms, ls))
        valid.append({"master_families": sorted(ms), "linking_var_families": sorted(ls),
                      "rank": len(valid) + 1,
                      "block_problem": c.get("block_problem"),
                      "integrality_property": c.get("integrality_property"),
                      "expected_pricing_difficulty": c.get("expected_pricing_difficulty")})
        if len(valid) == k:
            break
    return valid, problems


def validate_supports(answer: dict, summary: dict) -> tuple[list[dict], dict, list[str]]:
    """The fields of the support prompt: joint supports (sets of >= 2 known constraint families) and per-family answers."""
    cfams = {f["family"] for f in summary["constraint_families"]}
    problems, supports, seen = [], [], set()
    for i, sup in enumerate(answer.get("joint_supports") or []):
        fams = sup.get("families") if isinstance(sup, dict) else None
        if not isinstance(fams, list) or not all(isinstance(x, str) for x in fams):
            problems.append(f"joint support {i}: malformed")
            continue
        bad = [x for x in fams if x not in cfams]
        if bad:
            problems.append(f"joint support {i}: unknown families {bad}")
            continue
        key = frozenset(fams)
        if len(key) < 2 or key in seen:
            problems.append(f"joint support {i}: fewer than two families or duplicate")
            continue
        seen.add(key)
        supports.append({"families": sorted(key), "rank": len(supports) + 1,
                         "inequality_class": sup.get("inequality_class"), "reason": sup.get("reason")})
    info = {}
    fam = answer.get("families") if isinstance(answer.get("families"), dict) else {}
    for f, v in fam.items():
        if f in cfams and isinstance(v, dict):
            info[f] = {"integrality_property": v.get("integrality_property"),
                       "strengthens_alone": v.get("strengthens_alone")}
    return supports, info, problems


def propose(llm, summary: dict, instance_dir: str | None, k: int, time_limit: float,
            history: list[dict] | None = None, lp_bound: float | None = None, tag: str = "",
            prompt_version: str = "v2", condition: str | None = None, seed: int = 0, sample: int = 0) -> dict:
    """One LLM call. Returns {"messages", "response", "answer", "candidates", "problems", "supports",
    "family_info"}. `condition` (counterfactual.CONDITIONS) overrides LLMDW_ANONYMIZE.
    `sample` >= 1 requests another independent answer to the identical prompt (LLMClient.chat)."""
    from llmdw.prompts import counterfactual

    if condition is not None:
        maps = counterfactual.maps_for(summary, condition, seed)
        withhold = counterfactual.withhold_text(condition)
    else:
        maps = anonymize_maps(summary) if anonymize_enabled() else None
        withhold = None
    template = prompt_template(prompt_version)
    messages = build_messages(summary, instance_dir, k, time_limit, history, lp_bound, template=template,
                              maps=maps, withhold=withhold)
    resp = llm.chat(messages, tag=tag, sample=sample) if sample else llm.chat(messages, tag=tag)
    return {"messages": messages, "response": resp,
            **interpret(resp["content"], summary, k, maps, prompt_version)}


# keys an answer object must have (llm_client.extract_json): the fields that the scoring reads
REQUIRED_KEYS = {"v2": ("families", "joint_supports", "candidates"),
                 "ranked": ("families", "joint_supports", "candidates"),
                 "with_data": ("families", "joint_supports", "candidates")}
RANKED_USER = "user_ranked.txt"
WITH_DATA_USER = "user_with_data.txt"
USER_FILES = {"ranked": RANKED_USER, "with_data": WITH_DATA_USER}


def prompt_template(prompt_version: str) -> dict:
    """The prompt template of a version. "ranked": the system prompt and feedback section of v2 with the user prompt
    of prompts/user_ranked.txt (v2's user prompt plus "ranked_groups" and "relevant_families"); "with_data": the same
    with prompts/user_with_data.txt (v2's user prompt plus a section "## Model with data", the instance in CPLEX LP
    format, model_lp_text)."""
    if prompt_version in USER_FILES:
        t = load_template(PROMPTS / "decomp_v2.md")
        t["USER"] = (PROMPTS / USER_FILES[prompt_version]).read_text(encoding="utf-8")
        return t
    return load_template(PROMPTS / f"decomp_{prompt_version}.md")


def validate_ranked(answer: dict, summary: dict) -> tuple[list[dict], list | None, list[str]]:
    """Prompt "ranked": ranked_groups, validated like joint_supports except that a set may have one family (sets of
    one or more known constraint families, no duplicates), in the order given, each with its probability (a number in [0, 1], else None); and
    relevant_families (known constraint families; None when the field is missing). A missing or malformed
    ranked_groups field is an empty list."""
    cfams = {f["family"] for f in summary["constraint_families"]}
    problems, groups, seen = [], [], set()
    raw = answer.get("ranked_groups")
    for i, g in enumerate(raw if isinstance(raw, list) else []):
        fams = g.get("families") if isinstance(g, dict) else None
        if not isinstance(fams, list) or not all(isinstance(x, str) for x in fams):
            problems.append(f"ranked group {i}: malformed")
            continue
        bad = [x for x in fams if x not in cfams]
        if bad:
            problems.append(f"ranked group {i}: unknown families {bad}")
            continue
        key = frozenset(fams)
        if len(key) < 1 or key in seen:
            problems.append(f"ranked group {i}: empty or duplicate")
            continue
        seen.add(key)
        p = g.get("probability")
        p = float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1 else None
        groups.append({"families": sorted(key), "rank": len(groups) + 1, "probability": p})
    if raw is not None and not isinstance(raw, list):
        problems.append("ranked_groups: not a list")
    rel = answer.get("relevant_families")
    if isinstance(rel, list):
        bad = [x for x in rel if not isinstance(x, str) or x not in cfams]
        if bad:
            problems.append(f"relevant_families: unknown or malformed entries {bad}")
        rel = sorted({x for x in rel if isinstance(x, str) and x in cfams})
    else:
        if rel is not None:
            problems.append("relevant_families: not a list")
        rel = None
    return groups, rel, problems


def interpret(content: str, summary: dict, k: int, maps: dict | None, prompt_version: str = "v2") -> dict:
    """The parsed fields of one answer text: {"answer", "candidates", "problems", "supports", "family_info"}, and
    for the ranked prompt "ranked_groups" and "relevant_families". `maps` are the presentation maps of the prompt
    (None: true names)."""
    try:
        answer = extract_json(content, REQUIRED_KEYS.get(prompt_version, ()))
    except (ValueError, json.JSONDecodeError) as e:
        return {"answer": None, "candidates": [], "problems": [f"unparseable answer: {e}"]}
    if maps is not None:
        answer = _deanon_answer(answer, maps)
    cands, problems = validate(answer, summary, k)
    supports, info, p2 = validate_supports(answer, summary)
    out = {"answer": answer, "candidates": cands, "problems": problems + p2, "supports": supports,
           "family_info": info}
    if prompt_version == "ranked":
        groups, rel, p3 = validate_ranked(answer, summary)
        out.update({"ranked_groups": groups, "relevant_families": rel})
        out["problems"] = out["problems"] + p3
    return out


class ScriptedLLM:
    """Stand-in for LLMClient used in tests: returns pre-written answers in order."""

    def __init__(self, answers: list[dict | str]):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages, tag="", json_mode=True, sample=0):
        self.calls.append({"tag": tag, "messages": messages, "sample": sample})
        a = self.answers.pop(0) if self.answers else {"families": {}, "joint_supports": [], "candidates": []}
        content = a if isinstance(a, str) else json.dumps(a)
        return {"content": content, "model": os.environ.get("LLMDW_MODEL", "scripted"), "usage": None,
                "cached": False, "latency": 0.0, "finish_reason": "stop"}
