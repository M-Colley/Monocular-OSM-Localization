"""Measure the DEPLOYABLE coarse-location priors per GT clip (no GT coords).

For each clip we derive a coarse centre from the VIDEO alone -- VLM district /
street reading (Gemma) and the licence-plate registration district -- and
report how far each lands from the true route start. This is the seed a fair
(non-GT) run would give the VPR reference disc. The only non-video input is the
CITY NAME, used solely as a sanity bound (knowing you are in Ulm is not the GT
coordinates).

    python scripts/probe_deployable_priors.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plate_anchor import plate_district_anchor  # noqa: E402
from src.text_anchor import default_geocode_fn  # noqa: E402
from src.vlm_anchor import vlm_district_anchor  # noqa: E402

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


def sample_frames(video, end_s, n=40):
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
            print(f"{name:12s} SKIP (no video {video})")
            continue
        gt = json.load(open(ROOT / gtf))["waypoints"]
        gt0 = (gt[0]["lat"], gt[0]["lon"])
        gtc = (float(np.mean([w["lat"] for w in gt])),
               float(np.mean([w["lon"] for w in gt])))
        frames = sample_frames(video, end_s, n=40)
        # VLM (Gemma) district/street reader
        try:
            va = vlm_district_anchor(frames, city, geocode_fn=geo, n_query=6,
                                     min_votes=2, min_street_votes=1,
                                     use_text_fallback=False, max_km_from_city=15.0)
        except Exception as e:
            va = None
            print(f"  {name} VLM error: {e}")
        # Licence-plate registration district
        try:
            pa = plate_district_anchor(str(vp), frames=frames, geocode_fn=geo)
        except Exception as e:
            pa = None
            print(f"  {name} plate error: {e}")
        v_lbl = (f"{va.label} @{err_m((va.lat, va.lon), gt0):.0f}m/start "
                 f"{err_m((va.lat, va.lon), gtc):.0f}m/centroid" if va else "-")
        p_lbl = (f"{pa.code}={pa.district} @{err_m((pa.lat, pa.lon), gt0):.0f}m/start"
                 if pa else "-")
        print(f"{name:12s}  VLM: {v_lbl}   PLATE: {p_lbl}")
        rows.append(dict(name=name, gt0=gt0,
                         vlm=[va.lat, va.lon, va.label] if va else None,
                         vlm_start_err=err_m((va.lat, va.lon), gt0) if va else None,
                         plate=[pa.lat, pa.lon, pa.district] if pa else None,
                         plate_start_err=err_m((pa.lat, pa.lon), gt0) if pa else None))
    json.dump(rows, open(ROOT / "output" / "deployable_priors.json", "w"), indent=1)
    print("\nsaved output/deployable_priors.json")


if __name__ == "__main__":
    main()
