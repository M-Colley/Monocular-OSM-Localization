"""CPU-only deployable coarse-prior probe (no GPU / no Gemma VLM).

Measures the video-derived priors that run on CPU -- the CITY geocode (you know
the city name, not the coords) and the LICENCE-PLATE registration district --
against the true route per GT clip. This is the coarse centre a fair (no-GT)
run would seed the VPR disc + OSM graph with. Run with CUDA hidden so nothing
contends with other GPU jobs:

    CUDA_VISIBLE_DEVICES= python scripts/probe_deployable_cpu.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plate_anchor import plate_district_anchor  # noqa: E402
from src.text_anchor import default_geocode_fn  # noqa: E402

MPD = 111320.0

CLIPS = [
    ("Ulm 4K", "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input_4k.webm",
     "Ulm, Germany", "ground_truth/ulm_ULl8s4qydrk.json", 420.0),
    ("KITTI 0009", "data/kitti/drive_0009.mp4", "Karlsruhe, Germany",
     "ground_truth/kitti_drive_0009.json", 47.0),
    ("KITTI 0033", "data/kitti/drive_0033.mp4", "Karlsruhe, Germany",
     "ground_truth/kitti_drive_0033.json", 166.0),
    ("comma2k19", "data/comma/route_148.mp4", "Daly City, California, USA",
     "ground_truth/comma_148.json", 240.0),
    ("Ulm #2", "data/lc9sa-u5ke-ulm-de-centre-ville-dashcam-4k-zhiroad-deutschland-ulmcity-ulm-german/input.mp4",
     "Ulm, Germany", "ground_truth/ulm_LC9Sa--u5KE.json", 500.0),
    ("London", "data/london_T4wTL3LpLqU/input_4k.webm", "London, UK",
     "ground_truth/london_T4wTL3LpLqU.json", 295.0),
]


def err_m(a, b):
    return math.hypot((a[0] - b[0]) * MPD,
                      (a[1] - b[1]) * MPD * math.cos(math.radians(b[0])))


def sample_frames(video, end_s, n=60):
    cap = cv2.VideoCapture(str(ROOT / video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_end = int(min(end_s * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idx = np.linspace(0, max(n_end - 1, 0), n).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def main():
    geo = default_geocode_fn(ROOT / "data" / "geocode_cache.json")
    rows = []
    for name, video, city, gtf, end_s in CLIPS:
        vp = ROOT / video
        if not vp.exists():
            print(f"{name:12s} SKIP (no video)"); continue
        gt = json.load(open(ROOT / gtf))["waypoints"]
        gt0 = (gt[0]["lat"], gt[0]["lon"])
        gtc = (float(np.mean([w["lat"] for w in gt])),
               float(np.mean([w["lon"] for w in gt])))
        # 1) City geocode (know the name, not the coords)
        try:
            cc = geo(city)
            city_err0 = err_m(cc, gt0) if cc else None
            city_errc = err_m(cc, gtc) if cc else None
        except Exception as e:
            cc, city_err0, city_errc = None, None, str(e)
        # 2) Licence-plate registration district (CPU ALPR)
        t0 = time.time()
        try:
            frames = sample_frames(video, end_s, n=60)
            pa = plate_district_anchor(str(vp), frames=frames, geocode_fn=geo)
        except Exception as e:
            pa = None
            print(f"  {name} plate error: {e}")
        dt = time.time() - t0
        p_lbl = (f"{pa.code}={pa.district} ({pa.votes}/{pa.total_unique}u x{pa.margin:.1f}) "
                 f"@{err_m((pa.lat, pa.lon), gt0):.0f}m/start" if pa else "no confident plate")
        print(f"{name:12s}  CITY '{city}' -> "
              f"{f'{city_err0:.0f}m/start {city_errc:.0f}m/centroid' if cc else 'geocode fail'}"
              f"   |  PLATE: {p_lbl}   ({dt:.0f}s)")
        rows.append(dict(name=name, city=city, city_center=cc,
                         city_start_err=city_err0, city_centroid_err=city_errc,
                         plate=[pa.lat, pa.lon, pa.code, pa.district] if pa else None,
                         plate_start_err=err_m((pa.lat, pa.lon), gt0) if pa else None,
                         plate_radius_m=pa.radius_m if pa else None))
    json.dump(rows, open(ROOT / "output" / "deployable_cpu.json", "w"), indent=1)
    print("\nsaved output/deployable_cpu.json")


if __name__ == "__main__":
    main()
