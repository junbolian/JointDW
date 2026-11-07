"""Re-read recorded answers with the current parser (llmdw.prompts.propose.interpret, name rule included), no calls.

Every answer file <root>/<configuration>/<instance>/<condition>[.s<n>].json keeps the answer object as extracted and
mapped back to the true names. The object is renamed to the presentation's names again (the inverse of
propose._deanon_answer, which is one to one on the shown names and leaves any other string as it is), so interpret sees
the names the prompt showed, and the parsed fields (supports, candidates, problems, family_info, and for the ranked
prompt ranked_groups and relevant_families) are replaced by its result. An answer without an object (unparseable) is
left as it is. Presentation maps: <instance>/labels.json when present (per-instance label rule), else
counterfactual.maps_for with seed 0 (the rule of the recorded runs). Summaries: <summaries>/<set>/<instance>.json
("summary", as written with the data).

Usage: python -m llmdw.prompts.reparse ROOT [ROOT ...] --summaries DIR [--out OUT]
  (without --out the files are rewritten in place; with --out the tree of each ROOT is written under OUT/<ROOT name>)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llmdw.prompts import counterfactual, propose

FIELDS = ("supports", "candidates", "problems", "family_info", "answer")
RANKED_FIELDS = ("ranked_groups", "relevant_families")
K = 3


def _summary(summaries: Path, name: str, cache: dict) -> dict:
    if name not in cache:
        hits = sorted(summaries.glob(f"*/{name}.json"))
        if not hits:
            raise FileNotFoundError(f"no recorded summary for {name} under {summaries}")
        cache[name] = json.loads(hits[0].read_text(encoding="utf-8"))["summary"]
    return cache[name]


def _maps(summary: dict, cond: str, labels: Path) -> dict | None:
    if labels.exists():
        rec = json.loads(labels.read_text(encoding="utf-8")).get(cond)
        if rec:
            c, v = rec["cons"], rec["vars"]
            return {"cons": c, "vars": v, "cons_inv": {b: a for a, b in c.items()},
                    "vars_inv": {b: a for a, b in v.items()}}
    return counterfactual.maps_for(summary, cond, 0)


def reparse_record(rec: dict, summary: dict, maps: dict | None) -> tuple[dict, bool]:
    """(record with the parsed fields of the current parser, whether any parsed field changed)."""
    if rec.get("answer") is None:
        return rec, False
    obj = rec["answer"] if maps is None else propose._deanon_answer(
        rec["answer"], {"cons_inv": maps["cons"], "vars_inv": maps["vars"]})
    pv = rec.get("prompt_version") or "v2"
    res = propose.interpret(json.dumps(obj), summary, K, maps, pv)
    keys = FIELDS + (RANKED_FIELDS if pv == "ranked" else ())
    new = dict(rec)
    new.update({x: res.get(x) for x in keys})
    changed = any(json.dumps(rec.get(x), sort_keys=True) != json.dumps(new.get(x), sort_keys=True)
                  for x in keys if x not in ("problems", "answer"))
    return new, changed


def reparse_tree(root: Path, summaries: Path, out: Path | None) -> Counter:
    cache, n = {}, Counter()
    for f in sorted(root.glob("*/*/*.json")):
        if f.name == "labels.json":
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        summary = _summary(summaries, rec["instance"], cache)
        maps = _maps(summary, rec.get("condition") or "original", f.parent / "labels.json")
        new, changed = reparse_record(rec, summary, maps)
        n["answers"] += 1
        n["changed"] += changed
        dst = f if out is None else out / f.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if out is not None and (f.parent / "labels.json").exists():
            (dst.parent / "labels.json").write_text((f.parent / "labels.json").read_text(encoding="utf-8"),
                                                    encoding="utf-8")
        tmp = dst.with_suffix(".json.part")
        tmp.write_text(json.dumps(new, indent=1), encoding="utf-8")
        tmp.replace(dst)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    for r in args.roots:
        root = Path(r)
        n = reparse_tree(root, Path(args.summaries), Path(args.out) / root.name if args.out else None)
        print(f"{root}: {n['answers']} answers, parsed fields changed in {n['changed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
