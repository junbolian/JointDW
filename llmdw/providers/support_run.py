"""Support predictions: one call per instance, presentation, prompt and sample, no bound evaluation. The answer

Paper: Section 6 and Appendix D: every answer that the paper scores.
(joint supports, per-family answers, ranked candidates) is stored with the true family names restored. Each answer
file also stores content_sha256, the SHA-256 of the model's answer text as returned, so that byte-identical samples
can be found (llmdw.providers.usage).

Prompts (--prompt): original (prompts/decomp_v2.md); ranked (the same system prompt with prompts/user_ranked.txt,
which also asks for ranked groups and relevant families); with_data (the same system prompt with
prompts/user_with_data.txt, which adds the instance in CPLEX LP format). The ranked and with_data prompts have their
own cache keys (the user prompt file is part of the key) and call tags.

Files: <out>/<instance>/<condition>.json for sample 0 and <condition>.s<i>.json for sample i >= 1. Every sample
sends the identical prompt; samples differ only through the model's own sampling (LLMClient.chat(sample=i)).
Finished files are skipped, so an interrupted run resumes without new calls for them. Each answer file is
written through a temporary file and renamed, and every answered call is also in the LLM cache, so a call that
was answered but not yet written to its file is not paid for again. When one call fails (a rejected key, a
used-up quota, a model mismatch), no thread starts another call, and the reason is printed on a line that starts
with STOPPED. A call that returns no answer after its retries (llm_client.IncompleteResponse) or whose every
attempt failed with a gateway or network error (llm_client.GatewayExhausted) writes no file and prints a line
that starts with INCOMPLETE; the run goes on, and a later run tries that call again. With --stop-file, no thread
starts a new call once one of the named files exists (a line starting with STOP FILE says so); calls already in
flight finish and are written, so a later run resumes without asking for any answer twice.

Usage
-----
python -m llmdw.providers.support_run --templates data/work/templates --conditions original,names_only,permuted,anonymized \
    --samples 3 --parallel 4 --out data/work/support_templates/<config>
python -m llmdw.providers.support_run --mps-dir data/work/mps --dataset MIPLIB-NL/dataset --instances-from data/work/oracle \
    --conditions original,names_only,permuted,anonymized --samples 3 --parallel 4 --out data/work/support_miplib/<config>
python -m llmdw.providers.support_run --rich-models data/work/rich_models --prompt with_data --conditions original \
    --samples 3 --parallel 4 --out data/work/rich_models_with_data/<config>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llmdw.prompts import counterfactual
from llmdw.instances.families import read_model, summarize
from llmdw.providers.llm_client import GatewayExhausted, IncompleteResponse
from llmdw.prompts.propose import propose


PROMPTS = {"original": "v2", "ranked": "ranked", "with_data": "with_data"}     # --prompt -> propose's prompt version


class RunStopped(Exception):
    """Raised by run_instance when should_stop() is true before a call that is still needed."""


def result_path(out: Path, name: str, cond: str, sample: int) -> Path:
    return out / name / (f"{cond}.json" if sample == 0 else f"{cond}.s{sample}.json")


def run_instance(llm, name: str, mps: str, instance_dir: str | None, conditions: list, out: Path,
                 k: int = 3, time_limit: float = 120.0, seed: int = 0, samples: int = 1,
                 should_stop=None, prompt_version: str = "v2", tag_prefix: str = "support", log=None,
                 labels: str = "fixed") -> list:
    """Answers of one instance; files already present are read, not requested. should_stop() is checked before
    every new call, so a stop set by another thread ends this instance at its next call (RunStopped). log, when given,
    receives the INCOMPLETE line of a call without an answer."""
    summary = summarize(read_model(mps))
    results = []
    groups = []
    if labels == "per-instance":     # the true joint supports, kept apart by the permuted presentation
        gt = Path(instance_dir) / "ground_truth.json"
        groups = json.loads(gt.read_text(encoding="utf-8"))["joint_supports"] if gt.exists() else []
    for cond in conditions:
        if not counterfactual.applicable(summary, cond):
            continue
        maps = None
        if labels == "per-instance":
            maps, used = counterfactual.instance_maps(summary, cond, name, groups)
            if maps is not None:
                lp = out / name / "labels.json"
                lp.parent.mkdir(parents=True, exist_ok=True)
                rec = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
                rec[cond] = {"seed": used, "seed_base": counterfactual.instance_seed(name), "cons": maps["cons"],
                             "vars": maps["vars"], "groups_kept_apart": groups if cond == "permuted" else []}
                lp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        for sample in range(samples):
            path = result_path(out, name, cond, sample)
            if path.exists():
                results.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            if should_stop is not None and should_stop():
                raise RunStopped(name)
            try:
                res = propose(llm, summary, instance_dir, k, time_limit, prompt_version=prompt_version, condition=cond,
                              seed=seed, tag=f"{tag_prefix}:{name}:{cond}:s{sample}", sample=sample, maps=maps)
            except (IncompleteResponse, GatewayExhausted) as e:
                if log is not None:
                    log(f"INCOMPLETE {name} {cond} s{sample}: {e}")
                continue
            rec = {"instance": name, "condition": cond, "sample": sample, "model": res["response"].get("model"),
                   "latency": res["response"].get("latency"), "usage": res["response"].get("usage"),
                   "finish_reason": res["response"].get("finish_reason"),
                   "supports": res.get("supports", []), "family_info": res.get("family_info", {}),
                   "candidates": res["candidates"], "problems": res["problems"], "answer": res["answer"],
                   "messages": res["messages"],
                   "content_sha256": hashlib.sha256((res["response"].get("content") or "").encode("utf-8")).hexdigest()}
            if prompt_version == "ranked":
                rec.update({"prompt_version": prompt_version, "ranked_groups": res.get("ranked_groups", []),
                            "relevant_families": res.get("relevant_families")})
            elif prompt_version != "v2":
                rec["prompt_version"] = prompt_version
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.part")
            tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            tmp.replace(path)
            results.append(rec)
    return results


def main(argv=None):
    from llmdw.providers.llm_client import LLMClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mps-dir")
    ap.add_argument("--dataset")
    ap.add_argument("--instances-from", help="directory whose subdirectories name the instances (e.g. the oracle root)")
    ap.add_argument("--templates", action="append", default=[],
                    help="directory written by llmdw.instances.export (repeatable)")
    ap.add_argument("--rich-models", action="append", default=[],
                    help="directory written by llmdw.instances.rich_models_build: the instances listed in its instances.txt "
                         "(repeatable)")
    ap.add_argument("--prompt", default="original", choices=list(PROMPTS),
                    help="original, ranked or with_data (see above)")
    ap.add_argument("--config", help="model identifier of a configuration of config/public_settings.json: the request "
                                     "settings, and the key and base URL from the variables the entry names")
    ap.add_argument("--conditions", default="original,names_only,permuted,anonymized")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="fixed", choices=("fixed", "per-instance"),
                    help="label rule of the permuted and anonymized presentations (llmdw.prompts.counterfactual)")
    ap.add_argument("--samples", type=int, default=1, help="independent answers per instance and condition")
    ap.add_argument("--parallel", type=int, default=1, help="instances processed at the same time")
    ap.add_argument("--stop-file", action="append", default=[],
                    help="no new call starts once this file exists (calls in flight finish and are kept); repeatable")
    args = ap.parse_args(argv)
    conds = [c for c in args.conditions.split(",") if c]
    if args.config:
        import os

        from llmdw.providers import settings
        os.environ.update(settings.environment(args.config))
    prompt_version = PROMPTS[args.prompt]
    if prompt_version != "v2":                         # the cache key includes the user prompt file
        import os

        from llmdw.prompts.propose import USER_FILES
        os.environ["LLMDW_PROMPT_TAG"] = USER_FILES[prompt_version]
    tag_prefix = "support" if prompt_version == "v2" else prompt_version
    llm, out = LLMClient(), Path(args.out)
    items = []
    for tdir in args.templates:
        for d in sorted(p for p in Path(tdir).iterdir() if (p / "model.mps").exists()):
            items.append((d.name, str(d / "model.mps"), str(d)))
    for rdir in args.rich_models:
        for n in (Path(rdir) / "instances.txt").read_text(encoding="utf-8").split():
            d = Path(rdir) / "instances" / n
            items.append((n, str(d / "model.mps"), str(d)))
    if args.instances_from:
        dirs = sorted(p for p in Path(args.instances_from).iterdir() if p.is_dir())
        if any((p / "oracle.json").exists() for p in dirs):     # an oracle root: only instances with ground truth
            dirs = [p for p in dirs if (p / "oracle.json").exists()]
        for d in dirs:
            mps = Path(args.mps_dir) / f"{d.name}.mps.gz"
            if mps.exists():
                items.append((d.name, str(mps), str(Path(args.dataset) / d.name)))

    stop = threading.Event()
    stop_files = [Path(x) for x in args.stop_file]

    def should_stop():
        if not stop.is_set() and any(f.exists() for f in stop_files):
            stop.set()
            print(f"STOP FILE: no new calls ({', '.join(str(f) for f in stop_files if f.exists())})", flush=True)
        return stop.is_set()

    def work(item):
        name, mps, inst = item
        if should_stop():
            return f"{name:40s} not started (run stopped)"
        try:
            recs = run_instance(llm, name, mps, inst, conds, out, args.k, seed=args.seed, samples=args.samples,
                                should_stop=should_stop, prompt_version=prompt_version, tag_prefix=tag_prefix,
                                log=lambda line: print(line, flush=True), labels=args.labels)
        except RunStopped:
            return f"{name:40s} not finished (run stopped)"
        except BaseException as e:
            stop.set()      # ModelMismatch or a failed request stops the whole run, as with --parallel 1
            print(f"STOPPED at {name}: {type(e).__name__}: {str(e)[:400]}", flush=True)
            raise
        counts = {}
        for r in recs:
            counts.setdefault(r["condition"], []).append(len(r["supports"]))
        return f"{name:40s} " + " ".join(f"{c}:{v}" for c, v in counts.items())

    pool = ThreadPoolExecutor(max_workers=max(1, args.parallel))
    try:
        for line in pool.map(work, items):
            print(line, flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(f"all {len(items)} instances finished", flush=True)


if __name__ == "__main__":
    main()
