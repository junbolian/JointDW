"""Decompositions as master-row sets, and GCG .dec files.

Paper: Appendix E: the decompositions that GCG evaluates.

Representation used throughout the package: a decomposition is the set of master rows;
the remaining rows are split into blocks by connected components (two rows are in the
same block when they share a variable). With this representation the candidate space is
the same as GCG's class-based detection (``consclass`` + ``connectedbase``): choose which
constraint classes stay in the master, let the rest fall apart into components.

A *family-aligned* decomposition puts every row of a family on the same side. An LLM
proposal is a set of master families, which this module materializes.

.dec format written here (GCG "own .dec file" format, original problem):

    PRESOLVED
    0
    NBLOCKS
    <k>
    BLOCK 1
    <row name>
    ...
    MASTERCONSS
    <row name>
"""
from __future__ import annotations

from llmdw.instances.families import Model, components, family_of, var_family_map


def from_master_families(model: Model, master_families: set[str],
                         linking_var_families: set[str] | None = None) -> tuple[list[list[str]], list[str]]:
    """Blocks (lists of row names) and master rows for a set of master constraint families
    and, optionally, a set of variable families allowed to link blocks."""
    master_idx = [i for i, r in enumerate(model.rows) if family_of(r.name) in master_families or not r.coefs]
    master_set = set(master_idx)
    block_idx = [i for i in range(len(model.rows)) if i not in master_set]
    linking = None
    if linking_var_families:
        vf = var_family_map(model.var_type)
        linking = {v for v in model.var_type if vf[v] in linking_var_families}
    comps = components(model, block_idx, linking) if block_idx else []
    blocks = [[model.rows[i].name for i in comp] for comp in comps]
    master = [model.rows[i].name for i in master_idx]
    return blocks, master


def write_dec(path: str, blocks: list[list[str]], master: list[str], header: str | None = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            for line in header.splitlines():
                fh.write(f"\\ {line}\n")
        fh.write("PRESOLVED\n0\n")
        fh.write(f"NBLOCKS\n{len(blocks)}\n")
        for b, rows in enumerate(blocks, start=1):
            fh.write(f"BLOCK {b}\n")
            for name in rows:
                fh.write(name + "\n")
        fh.write("MASTERCONSS\n")
        for name in master:
            fh.write(name + "\n")


def read_dec(path: str) -> tuple[list[list[str]], list[str], bool]:
    """Read a .dec file. Returns (blocks, master rows, presolved flag).

    Variable sections (LINKINGVARS, MASTERVARS, BLOCKVARS ...) are not returned; use
    ``read_dec_full`` when they matter.
    """
    d = read_dec_full(path)
    return d["blocks"], d["master"], d["presolved"]


def read_dec_full(path: str) -> dict:
    """Read a .dec file including its variable sections."""
    blocks: list[list[str]] = []
    master: list[str] = []
    var_sections: dict[str, list[str]] = {}
    presolved = False
    section = None
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("\\")]
    i = 0
    while i < len(lines):
        tok = lines[i]
        up = tok.upper()
        if up in ("PRESOLVED", "CONSDEFAULTMASTER", "NBLOCKS"):
            if up == "PRESOLVED":
                presolved = lines[i + 1] == "1"
            i += 2
            section = None
            continue
        if up.startswith("BLOCKVARS") or up in ("LINKINGVARS", "MASTERVARS", "STAIRLINKINGVARS", "BLOCKVARS"):
            section = up
            var_sections.setdefault(up, [])
        elif up.startswith("BLOCK ") or up.startswith("BLOCKCONSS"):
            blocks.append([])
            section = "block"
        elif up == "MASTERCONSS":
            section = "master"
        elif section == "block":
            blocks[-1].append(tok)
        elif section == "master":
            master.append(tok)
        elif section in var_sections:
            var_sections[section].append(tok)
        i += 1
    return {"blocks": blocks, "master": master, "presolved": presolved, "var_sections": var_sections}


def is_component_partition(model: Model, blocks: list[list[str]], master: list[str]) -> bool:
    """True if the blocks are exactly the connected components of the non-master rows,
    i.e. the decomposition has no linking variables and is representable by its master set."""
    name_idx = {r.name: i for i, r in enumerate(model.rows)}
    ms = set(master)
    block_idx = [i for i, r in enumerate(model.rows) if r.name not in ms]
    comps = components(model, block_idx) if block_idx else []
    want = sorted(sorted(c) for c in comps)
    got = sorted(sorted(name_idx[n] for n in b) for b in blocks)
    return want == got


def family_profile(master_rows: list[str], all_rows: list[str]) -> dict:
    """Per-family share of rows placed in the master (1.0 = whole family in master)."""
    total, in_master = {}, {}
    ms = set(master_rows)
    for name in all_rows:
        f = family_of(name)
        total[f] = total.get(f, 0) + 1
        if name in ms:
            in_master[f] = in_master.get(f, 0) + 1
    return {f: in_master.get(f, 0) / n for f, n in sorted(total.items())}


def aligned_master_families(profile: dict, tol: float = 1e-9) -> set[str] | None:
    """The master-family set if the decomposition is family-aligned, else None."""
    out = set()
    for f, share in profile.items():
        if share >= 1.0 - tol:
            out.add(f)
        elif share > tol:
            return None
    return out
