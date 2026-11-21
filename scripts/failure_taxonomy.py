"""Where does the retrieval track fail? A search-coverage taxonomy (uses ground truth).

For every clip of one or more scored benchmark runs this reports how much of
the ground-truth route lay inside the two discs the track method actually
searched -- the pass-1 disc (the title-derived prior when the log names one,
else the 8 km city disc) and the coarse-to-fine pass-2 disc -- next to the
answer's symmetric error, and sorts each miss into one mode:

    A  route mostly outside the pass-1 disc (< 50 % of GT points inside)
    B  route inside, but pass-1 picked the wrong area (< 25 % inside pass 2)
    C  right area, but pass 2 covers only part of the route (25-75 %)
    D  anything else

It is an offline diagnostic: ground truth is applied here only, never fed to
a run. Discs are read from each run's log, errors from its summary.json
(``benchmark.py aggregate`` must have run).

    python scripts/failure_taxonomy.py \\
        --run dev=output/benchmark_runs/v6 --run test1=output/benchmark_runs/test1 \\
        --manifest benchmarks/manifest.json [--manifest other_manifest.json] \\
        [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.geo import geodesic_distance_m  # noqa: E402

PASS1_CITY_DISC_M = 8000.0  # stages/prior.py _PASS1_COVER_MAX_M
_RE_SEED = re.compile(r"video-named prior ([-\d.]+),([-\d.]+) r=(\d+) m")
_RE_FINE = re.compile(r"re-fetching a (\d+) m disc around the pass-1 centre "
                      r"\(snapped ([-\d.]+), ([-\d.]+)\)")


def share_inside(points: np.ndarray, lat: float, lon: float, radius_m: float) -> float:
    """Fraction of ``points`` (N x [lat, lon]) within ``radius_m`` of the centre."""
    return float(np.mean([geodesic_distance_m(lat, lon, p[0], p[1]) <= radius_m
                          for p in points]))


def failure_mode(error_m: float, pass1_share: float, fine_share: float | None,
                 *, threshold_m: float = 200.0) -> str:
    if error_m <= threshold_m:
        return "ok"
    if pass1_share < 0.5:
        return "A"
    if fine_share is not None and fine_share < 0.25:
        return "B"
    if fine_share is not None and fine_share < 0.75:
        return "C"
    return "D"


def clip_row(split: str, clip_id: str, clip: dict, manifest: dict, run_dir: Path,
             gt_root: Path) -> dict:
    a, b = (float(x) for x in clip["vo_segments"][0].split(":"))
    gt = json.loads((gt_root / manifest[clip_id]["gt"]).read_text(encoding="utf-8"))
    waypoints = [w for w in gt["waypoints"] if a - 1e-6 <= w["t_sec"] <= b + 1e-6]
    points = np.array([[w["lat"], w["lon"]] for w in waypoints])
    length = sum(geodesic_distance_m(*points[i], *points[i + 1]) for i in range(len(points) - 1))
    log = (run_dir / clip_id / "rep0" / "log.txt").read_text(encoding="utf-8", errors="replace")
    seed = _RE_SEED.search(log)
    if seed:
        lat, lon, radius = float(seed.group(1)), float(seed.group(2)), float(seed.group(3))
    else:
        lat, lon = list(clip["city_centroid"].values())[0]["prediction"]["route_latlon"][0]
        radius = PASS1_CITY_DISC_M
    fine = _RE_FINE.search(log)
    fine_share = (share_inside(points, float(fine.group(2)), float(fine.group(3)),
                               float(fine.group(1))) if fine else None)
    method = clip["methods"].get("answer") or clip["methods"]["vpr_track_only"]
    error = float(method["symmetric_mean_m"]["median"])
    confidence = ((clip.get("selective") or {}).get("confidence") or [None])[0]
    pass1_share = share_inside(points, lat, lon, radius)
    return {"split": split, "clip": clip_id, "name": clip["name"], "route_m": round(length),
            "pass1_radius_m": radius, "title_seeded": bool(seed),
            "pass1_share": round(pass1_share, 2),
            "fine_share": None if fine_share is None else round(fine_share, 2),
            "confidence": confidence, "error_m": round(error, 1),
            "mode": failure_mode(error, pass1_share, fine_share)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, metavar="SPLIT=DIR",
                    help="a scored benchmark run directory, labelled with its split")
    ap.add_argument("--manifest", action="append", required=True,
                    help="manifest(s) naming each clip's ground-truth file")
    ap.add_argument("--gt-root", default=str(ROOT),
                    help="directory the manifests' 'gt' paths are relative to")
    ap.add_argument("--json", help="also write the rows here")
    args = ap.parse_args(argv)
    manifest = {}
    for path in args.manifest:
        manifest.update({c["id"]: c for c in json.loads(Path(path).read_text(encoding="utf-8"))["clips"]})
    rows = []
    for spec in args.run:
        split, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows += [clip_row(split, cid, clip, manifest, run_dir, Path(args.gt_root))
                 for cid, clip in summary["clips"].items()]
    print("| Split | Clip | Route (km) | Pass-1 disc (km) | Route in pass-1 disc | "
          "Route in pass-2 disc | Confidence | Answer (m) | Mode |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        fine = "-" if r["fine_share"] is None else f"{r['fine_share']:.2f}"
        conf = "-" if r["confidence"] is None else f"{r['confidence']:.3f}"
        disc = f"{r['pass1_radius_m'] / 1000:.0f}{' (title)' if r['title_seeded'] else ''}"
        print(f"| {r['split']} | {r['name']} | {r['route_m'] / 1000:.1f} | {disc} | "
              f"{r['pass1_share']:.2f} | {fine} | {conf} | {r['error_m']:,.0f} | {r['mode']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
