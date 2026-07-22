"""Berlin tile3d harness: render LoD2 skylines at the GT poses.

Validates the whole 3D-tile geometry path against REAL data without
needing the video: for the first N Berlin GT waypoints it fetches the
open LoD2 model, renders the building silhouette a dashcam would see
(heading = direction to the next waypoint), and reports per-pose
skyline coverage. Silhouette PNGs land in output/tile3d_berlin/ for
eyeballing against the YouTube frames at the same timestamps.

    python scripts/test_tile3d_berlin.py                # first 5 waypoints
    python scripts/test_tile3d_berlin.py --n 10 --radius 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.citygml_lod2 import fetch_lod2_mesh                      # noqa: E402
from src.tile3d_match import (                                    # noqa: E402
    render_building_mask, skyline_from_mask,
)

GT = ROOT / "ground_truth" / "berlin_lBlKR2ek0w4.json"
CRS = "EPSG:32633"  # UTM 33N — what osmnx projects Berlin to


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5,
                    help="number of leading GT waypoints to render")
    ap.add_argument("--radius", type=float, default=500.0,
                    help="LoD2 disc radius around the rendered waypoints")
    ap.add_argument("--hfov", type=float, default=70.0,
                    help="assumed horizontal FOV (matches default_intrinsics)")
    opts = ap.parse_args(argv)

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

    wps = json.loads(GT.read_text(encoding="utf-8"))["waypoints"]
    n = min(opts.n + 1, len(wps))          # +1: last pose needs a successor
    xy = np.array([tr.transform(w["lon"], w["lat"]) for w in wps[:n]])

    center = xy[:-1].mean(axis=0)
    inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    c_lon, c_lat = inv.transform(center[0], center[1])
    span = float(np.linalg.norm(xy[:-1] - center[None, :], axis=1).max())
    mesh = fetch_lod2_mesh(c_lat, c_lon, span + opts.radius, dst_crs=CRS)
    if mesh is None or not len(mesh.triangles):
        print("FAIL: no LoD2 mesh")
        sys.exit(1)

    w, h = 480, 270
    fx = (w / 2.0) / np.tan(np.radians(opts.hfov) / 2.0)
    K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])
    out = ROOT / "output" / "tile3d_berlin"
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'t':>6s} {'ground_z':>8s} {'fill':>6s} {'skyline_cols':>12s}")
    for i in range(n - 1):
        fwd = xy[i + 1] - xy[i]
        norm = np.linalg.norm(fwd)
        if norm < 1e-6:
            print(f"{wps[i]['t_sec']:6.0f}  (stationary — skipped)")
            continue
        fwd = fwd / norm
        gz = mesh.local_ground_z(xy[i])
        mask = render_building_mask(
            mesh.triangles, xy[i], gz + 2.2, fwd, K, (w, h))
        rows = skyline_from_mask(mask)
        cov = float(np.isfinite(rows).mean())
        cv2.imwrite(str(out / f"gt_t{int(wps[i]['t_sec']):04d}.png"), mask)
        print(f"{wps[i]['t_sec']:6.0f} {gz:8.1f} "
              f"{float(mask.mean()) / 255.0:6.3f} {cov:12.3f}")
    print(f"\nsilhouettes -> {out}  (compare with the video at the "
          f"same timestamps: https://www.youtube.com/watch?v=lBlKR2ek0w4)")


if __name__ == "__main__":
    main()
