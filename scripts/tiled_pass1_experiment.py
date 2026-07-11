"""R4 experiment: does a TILED pass-1 beat the single diluted city-extent disc
on a mega-city (London)?

A city-name deployable run sizes the coarse VPR disc to ~8 km around the
centroid. In a dense metro that disc is heavily diluted (cap 3000 spread over
8 km), so pass-1 may not find the drive. This tiles the same region into
moderate discs, runs VPR per tile, and picks the tile whose queries match best
(mean per-frame top-1 similarity). Reports each tile's confidence + robust
centre distance to the GT drive, so we can see whether the best-confidence tile
lands on the drive better than the single disc does — WITHOUT touching the
pipeline. No GT is used to pick the tile (only the confidence signal).

    MLY_TOKEN=... python scripts/tiled_pass1_experiment.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import osmnx as ox  # noqa: E402
from src import kartaview_vpr as kv  # noqa: E402

VIDEO = ROOT / "data/london_T4wTL3LpLqU/input.mp4"
CACHE = ROOT / "data/local-73200bdd8068-input-london-uk"
CITY = "London, UK"
GT_DRIVE = (51.5223, -0.1267)
SEG = (0, 295)
TILE_R = 3000.0     # per-tile radius (denser: cap 1500 over 3 km vs 8 km)
COVER_R = 5000.0    # region radius around centroid (drive is 1.66 km out)
CAP = 1500


def _dist(a, b):
    return math.hypot((a[0] - b[0]) * 111320,
                      (a[1] - b[1]) * 111320 * math.cos(math.radians(b[0])))


def _frames(n=80):
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lo, hi = int(SEG[0] * fps), int(SEG[1] * fps)
    idx = np.linspace(lo, hi - 1, n).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def _tile_centers(clat, clon):
    step = TILE_R * 1.3 / 111320.0
    stepx = step / max(math.cos(math.radians(clat)), 0.2)
    r = COVER_R / 111320.0
    rx = r / max(math.cos(math.radians(clat)), 0.2)
    centers = []
    la = clat - r
    while la <= clat + r + 1e-9:
        lo = clon - rx
        while lo <= clon + rx + 1e-9:
            centers.append((la, lo))
            lo += stepx
        la += step
    return centers


def main():
    tok = __import__("os").environ.get("MLY_TOKEN")
    frames = _frames()
    clat, clon = ox.geocode(CITY)
    print(f"centroid {clat:.4f},{clon:.4f}; GT drive {_dist((clat,clon),GT_DRIVE)/1000:.2f} km away")

    # single 8 km disc baseline
    single = kv.kartaview_vpr_prior(
        frames, (clat, clon), radius_m=COVER_R, cache_dir=str(CACHE / "mapillary_tileexp_single"),
        source="mapillary", token=tok, cap=3000)
    if single:
        print(f"SINGLE 8km disc  prior {single[0]:.5f},{single[1]:.5f}  "
              f"-> {_dist(single, GT_DRIVE)/1000:.2f} km from drive")

    # tiled: per-tile confidence = mean per-query top-1 similarity
    rows = []
    for k, (tlat, tlon) in enumerate(_tile_centers(clat, clon)):
        out = kv._prepare_refs_and_query(
            frames, (tlat, tlon), TILE_R, str(CACHE / f"mapillary_tileexp_{k}"),
            80, None, "megaloc", "mapillary", tok, CAP)
        if out is None:
            continue
        idx, sims, ref_xy = out
        maxsim = sims.max(axis=1)
        top1 = sims.argmax(axis=1)
        centre = kv._robust_center(ref_xy[top1], maxsim)
        conf = float(np.mean(maxsim))
        rows.append((conf, centre, _dist(centre, GT_DRIVE), (tlat, tlon), len(ref_xy)))
        print(f"  tile {k:2d} @ {tlat:.4f},{tlon:.4f}  conf {conf:.3f}  "
              f"nref {len(ref_xy):4d}  centre->drive {_dist(centre, GT_DRIVE)/1000:.2f} km")
    if rows:
        best = max(rows, key=lambda r: r[0])
        print(f"\nBEST-CONFIDENCE tile: conf {best[0]:.3f}, centre {best[1][0]:.5f},{best[1][1]:.5f} "
              f"-> {best[2]/1000:.2f} km from drive")
        near = min(rows, key=lambda r: r[2])
        print(f"NEAREST tile to drive: {near[2]/1000:.2f} km (conf {near[0]:.3f}) "
              f"— does best-confidence == nearest? {best[3] == near[3]}")


if __name__ == "__main__":
    main()
