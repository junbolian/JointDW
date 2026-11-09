"""Prompt groups and the two-stage bootstrap over them.

Paper: Section 6 (measures): the standard errors and intervals of every table.

Prompt groups: instances whose request bodies are byte-identical in at least one presentation (the same messages,
hence the same request for every configuration and sample index, and the same cache key), closed transitively across
presentations; an instance without such a partner is its own group. Instances of one group may share a cached
answer, so they are not independent observations.

Two-stage bootstrap: every bootstrap over instances resamples in two stages. Stage 1 draws as many prompt groups as
there are among the instances of the row, with replacement; stage 2 draws, within each drawn group, as many of its
instances as it has, with replacement, and each drawn instance contributes all its values (rows for one class contain
only that class's instances, so they resample the groups within the class). The random draws use numpy
default_rng(seed). When every prompt group of a row has one instance, stage 2 is the identity and the replicates are
exactly those of the earlier one-stage bootstrap over groups (and of the instance bootstrap when every instance has
one value). Point estimates are not computed here.

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
            raise TypeError("the bootstrap needs instance names: pass {instance: value} or (instance, value) pairs")
    return [(n, v) for n, v in items if v is not None]


def _structure(items):
    """Values, the value indices of each instance and the instance indices of each prompt group."""
    vals = np.asarray([v for _, v in items], dtype=float)
    inst = list(dict.fromkeys(n for n, _ in items))
    ipos = {n: i for i, n in enumerate(inst)}
    of_inst = [[] for _ in inst]
    for j, (n, _) in enumerate(items):
        of_inst[ipos[n]].append(j)
    labels = [GROUPS.get(n, n) for n in inst]
    order = list(dict.fromkeys(labels))
    gpos = {g: i for i, g in enumerate(order)}
    of_group = [[] for _ in order]
    for i, g in enumerate(labels):
        of_group[gpos[g]].append(i)
    return vals, of_inst, of_group


def counts(values) -> tuple:
    """(instances, prompt groups) among the instances of values (None values dropped)."""
    items = _labelled(values)
    if not items:
        return 0, 0
    _, of_inst, of_group = _structure(items)
    return len(of_inst), len(of_group)


def n_groups(values) -> int:
    """Prompt groups among the instances of values: {instance: value} (every key, None values included),
    (instance, value) pairs or instance names."""
    if isinstance(values, dict):
        names = list(values)
    else:
        names = [x[0] if isinstance(x, tuple) else x for x in values]
    return len({GROUPS.get(n, n) for n in names})


def _one_stage(vals, labels, stat, resamples, seed):
    """The bootstrap over prompt groups that keeps every value of every drawn group (one stage)."""
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
        counts_ = np.array([len(m) for m in members], dtype=float)
        return sums[gi].sum(axis=1) / counts_[gi].sum(axis=1)
    groups = [vals[m] for m in members]
    return np.array([stat(np.concatenate([groups[g] for g in row])) for row in gi])


def replicates(values, stat=np.mean, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """The bootstrap replicates of stat (None without values): two stages, prompt groups and then the instances
    within each drawn group; the one-stage bootstrap when every group has one instance."""
    items = _labelled(values)
    if not items:
        return None
    vals, of_inst, of_group = _structure(items)
    if all(len(m) == 1 for m in of_group):           # every group one instance: stage 2 is the identity
        return _one_stage(vals, [GROUPS.get(n, n) for n, _ in items], stat, resamples, seed)
    rng = np.random.default_rng(seed)
    G = len(of_group)
    gi = rng.integers(0, G, size=(resamples, G))                         # stage 1: prompt groups
    size = np.array([len(m) for m in of_group])
    start = np.concatenate([[0], np.cumsum(size)[:-1]])
    flat = np.array([i for m in of_group for i in m])
    drawn = size[gi].ravel()                                             # instances to draw per drawn group
    local = rng.integers(0, np.repeat(drawn, drawn))                     # stage 2: instances within each group
    inst = flat[np.repeat(start[gi].ravel(), drawn) + local]
    rep = np.repeat(np.arange(resamples), size[gi].sum(axis=1))
    if stat is np.mean:
        isum = np.array([vals[m].sum() for m in of_inst])
        icnt = np.array([len(m) for m in of_inst], dtype=float)
        return (np.bincount(rep, weights=isum[inst], minlength=resamples)
                / np.bincount(rep, weights=icnt[inst], minlength=resamples))
    ivals = [vals[m] for m in of_inst]
    cuts = np.cumsum(size[gi].sum(axis=1))[:-1]
    return np.array([stat(np.concatenate([ivals[i] for i in row])) for row in np.split(inst, cuts)])


def boot(values, stat=np.mean, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """(lo, hi, se): the 95% percentile interval and the standard deviation (ddof = 1) of the replicates."""
    reps = replicates(values, stat, resamples, seed)
    if reps is None:
        return None, None, None
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(reps, ddof=1))
