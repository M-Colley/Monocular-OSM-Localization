"""How much of each clip's route the archived retrieval references cover.

For every clip of a cold benchmark run, reads the reference lists the run
archived (``data/<slug>/_benchmark_cold_vpr/<run>/<clip>/rep0/{mapillary,
mapillary_c2f}/mly_ref_meta_*.json``; ids and positions only) and reports the
fraction of ground-truth waypoints with a reference within 50 m and 250 m, per
pass, plus the run's own listing record (``result["vpr_listing"]``: tiles
split / lost) where it has one. Ground truth is used for scoring only.

    python scripts/pass1_coverage.py --runs v4 v5 --runs-dir .claude/worktrees/bench/output/benchmark_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def coverage(ref_ll: np.ndarray, route_ll: np.ndarray, d_m: float) -> float:
    """Fraction of route points with a reference within ``d_m`` metres."""
    from monocular_osm.geo import LocalFrame

    if not len(ref_ll):
        return 0.0
    frame = LocalFrame(float(route_ll[:, 0].mean()), float(route_ll[:, 1].mean()))
    rx, ry = frame.to_xy(ref_ll[:, 0], ref_ll[:, 1])
    qx, qy = frame.to_xy(route_ll[:, 0], route_ll[:, 1])
    d, _ = cKDTree(np.c_[rx, ry]).query(np.c_[qx, qy])
    return float(np.mean(d <= d_m))


def archived_pass(rep: Path, sub: str):
    """(signature, latlon[N,2]) of one archived pass, or None."""
    metas = sorted((rep / sub).glob("mly_ref_meta_*.json"))
    if not metas:
        return None
    blob = json.loads(metas[0].read_text(encoding="utf-8"))
    ll = np.array([[r["lat"], r["lon"]] for r in blob.get("refs") or []], float)
    return blob.get("signature", {}), ll.reshape(-1, 2)


def main(argv=None) -> int:
    import benchmark as B

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / B.RUNS_DIR)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--archive-root", type=Path, default=Path("E:/monocular-osm-archive"))
    ap.add_argument("--out", type=Path, default=None, help="write the rows as JSON here")
    args = ap.parse_args(argv)

    manifest = B.load_manifest(ROOT / "benchmarks" / "manifest.json")
    rows = []
    print("| Clip | Run | Pass-1 disc | Kept | Route within 50 m / 250 m (pass 1) "
          "| Pass 2: kept, within 50 m | Tiles split / lost |")
    print("|---|---|---|---|---|---|---|")
    for e in manifest["clips"]:
        if e["split"] != "dev" or not e["available_locally"]:
            continue
        wps = json.loads((ROOT / e["gt"]).read_text(encoding="utf-8"))["waypoints"]
        route = np.array([[w["lat"], w["lon"]] for w in wps], float)
        slug_dirs = B.candidate_slug_dirs(e, e["city"], args.data_dir)
        for run in args.runs:
            rep = next((base / "_benchmark_cold_vpr" / run / e["id"] / "rep0"
                        for d in slug_dirs for base in (d, args.archive_root / d.name)
                        if (base / "_benchmark_cold_vpr" / run / e["id"] / "rep0").is_dir()),
                       None)
            if rep is None:
                continue
            p1, p2 = archived_pass(rep, "mapillary"), archived_pass(rep, "mapillary_c2f")
            listing = {}
            res = args.runs_dir / run / e["id"] / "rep0" / "result.json"
            if res.exists():
                listing = json.loads(res.read_text(encoding="utf-8")).get("vpr_listing") or {}
            row = {"clip": e["id"], "run": run}
            if p1 is not None:
                row.update(p1_radius_m=p1[0].get("radius_m"), p1_kept=len(p1[1]),
                           p1_cov50=coverage(p1[1], route, 50.0),
                           p1_cov250=coverage(p1[1], route, 250.0))
            if p2 is not None:
                row.update(p2_kept=len(p2[1]), p2_cov50=coverage(p2[1], route, 50.0))
            row["listing"] = listing
            rows.append(row)
            lst = " ; ".join(f"{k} {v.get('split', '?')}/{v.get('lost', '?')}"
                             for k, v in listing.items()) or "n/a"
            p1s = (f"{row['p1_radius_m'] / 1000:.1f} km | {row['p1_kept']} | "
                   f"{row['p1_cov50']:.0%} / {row['p1_cov250']:.0%}") if p1 else "- | - | -"
            p2s = f"{row['p2_kept']}, {row['p2_cov50']:.0%}" if p2 else "-"
            print(f"| {e['id']} | {run} | {p1s} | {p2s} | {lst} |", flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
