"""LIVE validation of the Gemma-4 VLM district/street anchor.

Loads google/gemma-4-E2B-it ONCE, reads ~6 frames spread across a clip,
prints the raw per-frame model response + the votes + the geocoded anchor,
and measures the anchor's distance to ground truth.

Run (Python312 GPU env):
  C:/Users/localadmin/AppData/Local/Programs/Python/Python312/python.exe \
      scripts/test_vlm_live.py

Two clips are validated:
  * Ulm 4K   -> distance to the GT route polyline (ground_truth/ulm_*.json)
  * Erbach   -> distance to the town centre (~48.3211, 9.8887)
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

import src.vlm_anchor as va  # noqa: E402

ULM_VIDEO = ROOT / "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input_4k.webm"
ULM_GT = ROOT / "ground_truth/ulm_ULl8s4qydrk.json"
ERBACH_VIDEO = Path(
    "C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
    "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
    "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad/erbach.mp4"
)
ERBACH_CENTER = (48.3211, 9.8887)


def extract_frames(path: Path, n: int = 6) -> list:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, n).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    else:  # streaming container with no frame count: read sequentially
        step = 30 * 60  # ~ every minute at 30 fps
        i = 0
        while len(frames) < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            i += step
    cap.release()
    return frames


def _dist_km(a_lat, a_lon, b_lat, b_lon) -> float:
    dy = (a_lat - b_lat) * 111.32
    dx = (a_lon - b_lon) * 111.32 * math.cos(math.radians(0.5 * (a_lat + b_lat)))
    return math.hypot(dx, dy)


def dist_to_polyline_km(lat, lon, waypoints) -> float:
    best = float("inf")
    for wp in waypoints:
        best = min(best, _dist_km(lat, lon, wp["lat"], wp["lon"]))
    return best


def run_clip(name, video, city, gt_kind, gt_data, n=6):
    print(f"\n{'='*70}\n{name}  ({city})\n{'='*70}")
    if not video.exists():
        print(f"  MISSING VIDEO: {video}")
        return None
    frames = extract_frames(video, n)
    print(f"  extracted {len(frames)} frames from {video.name}")

    # Debug hook: capture the raw reply per frame while still voting normally.
    raw = []
    orig_ask = va._ask

    def _spy(pil, c):
        r = orig_ask(pil, c)
        raw.append(r)
        return r

    va._ask = _spy
    try:
        anchor = va.vlm_district_anchor(frames, city, n_query=n)
    finally:
        va._ask = orig_ask

    hits = 0
    for i, r in enumerate(raw):
        st, di, tx = va._parse(r)
        parsed_ok = bool(st or di or tx)
        hits += parsed_ok
        print(f"\n  --- frame {i} raw ---\n{r.strip()}")
        print(f"  parsed: STREET={st!r} DISTRICT={di!r} TEXT={tx!r} "
              f"[{'OK' if parsed_ok else 'EMPTY'}]")
    rate = hits / max(1, len(raw))
    print(f"\n  PARSER HIT-RATE: {hits}/{len(raw)} = {rate:.0%}")

    if anchor is None:
        print("  ANCHOR: None (nothing voted >= threshold geocoded in bound)")
        return {"clip": name, "hit_rate": rate, "anchor": None}

    print(f"  ANCHOR: label={anchor.label!r}  ({anchor.lat:.5f}, {anchor.lon:.5f})")
    print(f"    street_votes={anchor.street_votes}")
    print(f"    district_votes={anchor.district_votes}")
    print(f"    text_votes={anchor.text_votes}")
    if gt_kind == "polyline":
        d = dist_to_polyline_km(anchor.lat, anchor.lon, gt_data)
        print(f"  DISTANCE to GT route polyline: {d*1000:.0f} m")
    else:
        d = _dist_km(anchor.lat, anchor.lon, gt_data[0], gt_data[1])
        print(f"  DISTANCE to town centre: {d*1000:.0f} m")
    return {"clip": name, "hit_rate": rate, "label": anchor.label,
            "lat": anchor.lat, "lon": anchor.lon, "dist_m": d * 1000}


def main():
    print(f"Loading model {va.MODEL_ID} ...")
    va._load()
    print("Model loaded.")

    results = []
    gt = json.loads(ULM_GT.read_text(encoding="utf-8"))
    results.append(run_clip("ULM 4K", ULM_VIDEO, "Ulm, Germany",
                            "polyline", gt["waypoints"]))
    results.append(run_clip("ERBACH", ERBACH_VIDEO,
                            "Erbach an der Donau, Germany",
                            "center", ERBACH_CENTER))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        if r:
            print(f"  {r['clip']}: hit-rate {r['hit_rate']:.0%}, "
                  f"anchor={r.get('label')}, "
                  f"dist={r.get('dist_m', float('nan')):.0f} m"
                  if r.get("anchor") is not False else f"  {r['clip']}: {r}")

    # Free VRAM for the next task.
    import torch
    va._model = None
    va._proc = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
