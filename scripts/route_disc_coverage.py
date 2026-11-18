"""How much of each clip's true route lies inside the searched pass-1 disc (diagnostic).

    python scripts/route_disc_coverage.py SPLIT=DATA_DIR [SPLIT=DATA_DIR ...] [--clips ID ...] [--out OUT.json]

Reads each clip's pass-1 Mapillary cache signature (``mly_ref_meta_*.json``
under ``DATA_DIR/<slug>/mapillary``: centre, radius, cap; the largest cap if
several) and its ground-truth route. It reads ground truth, so it is a
post-hoc diagnostic on scored clips only.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.geo import geodesic_distance_m  # noqa: E402


def route(gt_path: Path) -> np.ndarray:
    g = json.loads(gt_path.read_text(encoding="utf-8"))
    w = g["waypoints"]
    w = ast.literal_eval(w) if isinstance(w, str) else w
    return np.array([[p["lat"], p["lon"]] for p in w], float)


def coverage(clip: dict, data_dir: Path) -> dict | None:
    key = clip["id"].replace("overlay_", "").lower().replace("_", "-")
    metas = glob.glob(str(data_dir / f"{key}*" / "mapillary" / "mly_ref_meta_*.json"))
    if not metas:
        return None
    sig = max((json.loads(Path(p).read_text())["signature"] for p in metas), key=lambda s: s["cap"])
    gt = route(ROOT / clip["gt"])
    d = geodesic_distance_m(sig["center"][0], sig["center"][1], gt[:, 0], gt[:, 1])
    return {"clip": clip["id"], "name": clip["name"], "split": clip["split"],
            "disc_radius_m": sig["radius_m"], "route_inside": round(float(np.mean(d <= sig["radius_m"])), 3),
            "route_from_disc_centre_km": [round(float(d.min()) / 1000, 1), round(float(d.max()) / 1000, 1)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("split_dirs", nargs="+", help="SPLIT=DATA_DIR")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    dirs = dict(s.split("=", 1) for s in args.split_dirs)
    manifest = json.loads((ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8"))["clips"]
    rows = []
    for clip in manifest:
        if clip["split"] not in dirs or (args.clips and clip["id"] not in args.clips):
            continue
        row = coverage(clip, Path(dirs[clip["split"]]))
        if row:
            rows.append(row)
            print(f"{row['split']:5s} {row['name'][:30]:30s} disc {row['disc_radius_m'] / 1000:4.1f} km  "
                  f"route inside {row['route_inside']:4.0%}  {row['route_from_disc_centre_km']} km from centre")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
