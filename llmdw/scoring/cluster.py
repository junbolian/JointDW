"""Prompt groups and the cluster bootstrap over them.

Paper: Section 6 (measures): the standard errors and intervals of every table.

Prompt groups: instances whose request bodies are byte-identical in at least one presentation (the same messages,
hence the same request for every configuration and sample index, and the same cache key), closed transitively across
presentations; an instance without such a partner is its own group. Instances of one group may share a cached
answer, so they are not independent observations.

Cluster bootstrap: every bootstrap over instances resamples prompt groups instead: each replicate draws as many groups
as there are among the instances of the row, with replacement, and keeps every value of every drawn group (rows for
one class contain only that class's instances, so they resample the groups within the class). The random draws use
numpy default_rng(seed); when every group of a row is a single value, the draws and the replicates are exactly those
of the instance bootstrap. Point estimates are not computed here.

Values are passed with their instance names: a dict {instance: value} or a list of (instance, value) pairs; None
values are dropped, and several values may share an instance (they then always stay together).
"""
from __future__ import annotations

from llmdw.constants import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

import hashlib
import json

import numpy as np

GROUPS: dict = {}            # instance name -> group label (set by set_groups); an unknown name is its own group


def set_groups(mapping: dict) -> None:
    GROUPS.clear()
    GROUPS.update(mapping)


def messages_hash(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prompt_groups(prompts: dict) -> dict:
    """prompts: {instance: {presentation: messages}}. Returns {instance: group}, the group named by its
    alphabetically first instance; instances with byte-identical messages in one presentation share a group,
    closed transitively (union-find) across presentations."""
    parent = {n: n for n in prompts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for n in sorted(prompts):
        for pres, msgs in sorted(prompts[n].items()):
            h = (pres, messages_hash(msgs))
            if h in seen:
                a, b = find(seen[h]), find(n)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                seen[h] = n
    members = {}
    for n in prompts:
        members.setdefault(find(n), []).append(n)
    return {n: min(ms) for ms in members.values() for n in ms}


def _labelled(values):
    if isinstance(values, dict):
        items = list(values.items())
    else:
        items = list(values)
        if items and not (isinstance(items[0], tuple) and len(items[0]) == 2):
            raise TypeError("the cluster bootstrap needs instance names: pass {instance: value} or (instance, value) pairs")
    return [(n, v) for n, v in items if v is not None]


def replicates(values, stat=np.mean, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """The bootstrap replicates of stat over prompt groups (None without values)."""
    items = _labelled(values)
    if not items:
        return None
    vals = np.asarray([v for _, v in items], dtype=float)
    labels = [GROUPS.get(n, n) for n, _ in items]
    order = list(dict.fromkeys(labels))
    rng = np.random.default_rng(seed)
    if len(order) == len(vals):                       # every group a single value: the instance bootstrap
        idx = rng.integers(0, len(vals), size=(resamples, len(vals)))
        return stat(vals[idx], axis=1)
    pos = {g: i for i, g in enumerate(order)}
    members = [[] for _ in order]
    for i, g in enumerate(labels):
        members[pos[g]].append(i)
    gi = rng.integers(0, len(order), size=(resamples, len(order)))
    if stat is np.mean:
        sums = np.array([vals[m].sum() for m in members])
        counts = np.array([len(m) for m in members], dtype=float)
        return sums[gi].sum(axis=1) / counts[gi].sum(axis=1)
    groups = [vals[m] for m in members]
    return np.array([stat(np.concatenate([groups[g] for g in row])) for row in gi])


def boot(values, stat=np.mean, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """(lo, hi, se): the 95% percentile interval and the standard deviation (ddof = 1) of the replicates."""
    reps = replicates(values, stat, resamples, seed)
    if reps is None:
        return None, None, None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(reps, ddof=1))
