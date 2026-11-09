"""Download the public benchmark data of the benchmark experiment, byte-identical, with URL and SHA-256.

Sources (fixed before any instance is built):
  trigeiro_suerie   the Trigeiro, Thomas and McClain test set as distributed at suerie.de (751 instances in .prn files)
                    and its map to Trigeiro's instance names, conv_ttm_new.txt
  trigeiro_gcg      the test list and the proved optima of the GCG repository for the Trigeiro instances
                    (check/testset/ls-trigeiro.test and .solu, pinned to one commit); the LP files the list names belong
                    to strIPlib and are not in the repository. Used to select and to check the suerie.de data
  canad_r           the Canad R instances of Crainic, Frangioni and Gendron, CommaLAB (R.tgz)
  orlib_cap         OR-Library capacitated warehouse location, cap41 to cap134 (Beasley)
  lotsizelib_ampl   AMPL copies of four LOTSIZELIB Trigeiro instances (tr6_15, tr6_30, tr12_30, tr24_30), used only to
                    check the suerie.de data value by value
  orlib_gap         OR-Library generalized assignment, gap1 to gap12, and gapinfo.html (the published optima)
  orlib_binpack     OR-Library bin packing, binpack1, and binpackinfo.html

Every file is written under <out>/<source>/ and listed in <out>/sources.json with URL, bytes, SHA-256 and the time of
the download. A file already present with the recorded SHA-256 is not downloaded again.

Usage
-----
python -m llmdw.benchmark.sources --out work/benchmark/raw
python -m llmdw.benchmark.sources --out data/benchmark_raw --recorded config/benchmark_sources.json

With --recorded, every downloaded data file is compared with the SHA-256 recorded when the instances were built, and
the command fails when one is missing or differs; the two web pages (kind "page") are reported, not verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

GCG_COMMIT = "c0e230a34ab836f9c344c60269fbd65fe4713f5b"
GCG_RAW = f"https://raw.githubusercontent.com/scipopt/gcg/{GCG_COMMIT}/check/"
CAP_NAMES = ([f"cap{i}" for i in (41, 42, 43, 44, 51)] + [f"cap{g}{j}" for g in (6, 7, 8, 9) for j in (1, 2, 3, 4)]
             + [f"cap{g}{j}" for g in (10, 11, 12, 13) for j in (1, 2, 3, 4)])
FIXED = {
    "trigeiro_suerie": ["https://suerie.de/testsets/clspl/data.tar.bz2", "https://suerie.de/testsets/clspl/conv_ttm_new.txt"],
    "canad_r": ["https://commalab.di.unipi.it/files/Data/MMCF/R.tgz",
                "https://commalab.di.unipi.it/datasets/mmcf/"],
    "orlib_cap": ["https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html"]
                 + [f"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{n}.txt" for n in CAP_NAMES],
    "trigeiro_gcg": [GCG_RAW + "testset/ls-trigeiro.test", GCG_RAW + "testset/ls-trigeiro.solu"],
    "lotsizelib_ampl": [f"https://www.dcc.fc.up.pt/~jpp/lsize/INSTANCES/{n}" for n in
                        ("tr6_15.dat", "tr6_15.mod", "tr6_30.dat", "tr6_30.mod", "tr12_30.dat", "tr12_30.mod",
                         "tr24_30.dat", "tr24_30.mod", "trigeiro.mod")],
    "orlib_gap": ["https://people.brunel.ac.uk/~mastjjb/jeb/orlib/gapinfo.html"]
                 + [f"https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/gap{g}.txt" for g in range(1, 13)],
    "orlib_binpack": ["https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html",
                      "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack1.txt"],
}
PAGE_NAMES = {"https://commalab.di.unipi.it/datasets/mmcf/": "mmcf_page.html"}


def fetch(url: str, path: Path, attempts: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    for a in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            break
        except OSError:
            if a == attempts - 1:
                raise
            time.sleep(10)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recorded", help="the recorded files (config/benchmark_sources.json): verify every SHA-256")
    args = ap.parse_args(argv)
    out = Path(args.out)
    man_path = out / "sources.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"files": []}
    have = {(f["source"], f["file"]): f for f in manifest["files"]}
    pending = [(s, u) for s, urls in FIXED.items() for u in urls]
    test = out / "trigeiro_gcg" / "ls-trigeiro.test"

    def one(source, url):
        name = PAGE_NAMES.get(url) or url.rstrip("/").split("/")[-1]
        rel = name
        if source == "trigeiro_gcg" and "/instances/" in url:
            rel = url.split("/check/")[1]
        p = out / source / rel
        rec = have.get((source, rel))
        if rec and p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]:
            return rec
        data = fetch(url, p)
        rec = {"source": source, "file": rel, "url": url, "bytes": len(data),
               "sha256": hashlib.sha256(data).hexdigest(), "downloaded": time.strftime("%Y-%m-%d %H:%M:%S %z")}
        have[(source, rel)] = rec
        return rec

    try:
        for s, u in pending:
            one(s, u)
    finally:
        manifest = {"gcg_commit": GCG_COMMIT, "files": sorted(have.values(), key=lambda f: (f["source"], f["file"]))}
        man_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    manifest = {"gcg_commit": GCG_COMMIT, "files": sorted(have.values(), key=lambda f: (f["source"], f["file"]))}
    man_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    by = {}
    for f in manifest["files"]:
        by[f["source"]] = by.get(f["source"], 0) + 1
    print(json.dumps(by))
    if args.recorded:
        rec = json.loads(Path(args.recorded).read_text(encoding="utf-8"))
        got = {(f["source"], f["file"]): f["sha256"] for f in manifest["files"]}
        def differs(f):
            return got.get((f["source"], f["file"])) != f["sha256"]
        data = [f for f in rec["files"] if f.get("kind", "data") == "data"]
        pages = [f for f in rec["files"] if f.get("kind") == "page"]
        bad = [f"{f['source']}/{f['file']}" for f in data if differs(f)]
        # web pages (the dataset descriptions) are kept for provenance; their content may change and is not verified
        print(json.dumps({"data_files": len(data), "verified": len(data) - len(bad), "differ_or_missing": bad,
                          "pages_changed_since_recorded": [f"{f['source']}/{f['file']}" for f in pages if differs(f)]}))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
