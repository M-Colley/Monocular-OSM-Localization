"""Build pipeline clips (video + ground_truth JSON) from the fleet-extension
datasets. One function per dataset; each writes data/<slug>/input.mp4 and
ground_truth/<name>.json, then prints the main.py command to test it.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ext_datasets import (  # noqa: E402
    enu_to_latlon, frames_to_video, malaga_track, brno_track, GpsFix,
    track_to_ground_truth,
)


def _write_gt(fixes, name, video_id, url, city, n_waypoints=14):
    gt = track_to_ground_truth(fixes, video_id=video_id, video_url=url,
                               city=city, n_waypoints=n_waypoints)
    out = ROOT / "ground_truth" / name
    out.write_text(json.dumps(gt, indent=1), encoding="utf-8")
    print(f"  wrote {out}  ({len(gt['waypoints'])} waypoints, "
          f"vo_segment {gt['vo_segment']})")
    return gt


# --- Boreas (Toronto / Vaughan) --------------------------------------------

def build_boreas(t0=300.0, t1=480.0, out_fps=10.0):
    """Window [t0,t1] real-seconds of the Glen Shields drive.

    raw_video.mp4 is a ~10 Hz re-encode played at 30 fps, NOT frame-aligned to
    the 20 Hz camera_poses.csv, so we map each raw_video frame to real time by
    its fraction of the full span and rebuild a real-time-paced clip; GT comes
    from camera_poses (ENU→lat/lon about the WGS84 origin) at real times.
    """
    B = ROOT / "data/ext_raw/boreas/boreas-2020-11-26-13-58"
    r0 = next(csv.DictReader(open(B / "applanix/gps_post_process.csv")))
    olat, olon = math.degrees(float(r0["latitude"])), math.degrees(float(r0["longitude"]))
    ts, e, n = [], [], []
    for row in csv.DictReader(open(B / "applanix/camera_poses.csv")):
        ts.append(float(row["GPSTime"])); e.append(float(row["easting"])); n.append(float(row["northing"]))
    ts = np.array(ts); tp = (ts - ts[0]) / 1e6
    latlon = enu_to_latlon(np.array(e), np.array(n), olat, olon)

    # video frame i -> real time via fraction of the full span
    cap = cv2.VideoCapture(str(B / "raw_video.mp4"))
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = tp[-1]
    f0, f1 = int(t0 / span * (nvid - 1)), int(t1 / span * (nvid - 1))
    slug = "boreas-glenshields-vaughan-canada"
    out_mp4 = ROOT / "data" / slug / "input.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = None
    for i in range(f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if i < f0:
            continue
        if vw is None:
            h, w = fr.shape[:2]
            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
        vw.write(fr)
    cap.release()
    if vw is not None:
        vw.release()
    print(f"  wrote {out_mp4}  (frames {f0}..{f1} @ {out_fps}fps = {(f1-f0)/out_fps:.0f}s)")

    # GT: camera poses at real times within [t0,t1], t_sec relative to t0
    m = (tp >= t0) & (tp <= t1)
    fixes = [GpsFix(float(tp[k] - t0), float(latlon[k, 0]), float(latlon[k, 1]))
             for k in np.where(m)[0]]
    _write_gt(fixes, "boreas_glenshields.json", slug,
              "https://www.boreas.utias.utoronto.ca/", "Vaughan, Ontario, Canada")
    return slug


# --- Málaga Urban (Spain) --------------------------------------------------

def build_malaga(extract_dir: Path, max_seconds=200.0, out_fps=20.0):
    """Left rectified images → mp4; GPS.txt (radians) → GT. Clip capped."""
    extract_dir = Path(extract_dir)
    gps_txt = next(extract_dir.glob("*_all-sensors_GPS.txt"))
    fixes = malaga_track(gps_txt)
    img_dir = next(d for d in extract_dir.iterdir()
                   if d.is_dir() and "rectified" in d.name and "1024x768" in d.name)
    lefts = sorted(img_dir.glob("*_left.jpg"))
    # image filename: img_CAMERA1_<unixtime>_left.jpg -> real time for capping
    def ftime(p):
        return float(p.name.split("_")[2])
    t0img = ftime(lefts[0])
    lefts = [p for p in lefts if ftime(p) - t0img <= max_seconds]
    slug = "malaga-urban-extract07-spain"
    out_mp4 = ROOT / "data" / slug / "input.mp4"
    frames_to_video(lefts, out_mp4, fps=out_fps)
    print(f"  wrote {out_mp4} ({len(lefts)} frames)")
    fixes = [f for f in fixes if f.t_sec <= max_seconds]
    _write_gt(fixes, "malaga_extract07.json", slug,
              "https://www.mrpt.org/MalagaUrbanDataset", "Málaga, Spain")
    return slug


# --- Brno Urban (Czechia) --------------------------------------------------

def build_brno(rec_dir: Path):
    """camera_left_front/video.mp4 (H265, 10Hz) + gnss/pose.txt (WGS84)."""
    rec_dir = Path(rec_dir)
    pose = rec_dir / "gnss" / "pose.txt"
    # detect the system-timestamp scale from the first/last vs the video length
    fixes = brno_track(pose)  # default ns scale; validated in caller
    slug = "brno-urban-1231-suburb-czechia"
    out_mp4 = ROOT / "data" / slug / "input.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # transcode H265-in-mp4 to a plain mp4v the pipeline reads uniformly
    cap = cv2.VideoCapture(str(rec_dir / "camera_left_front" / "video.mp4"))
    vw = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if vw is None:
            h, w = fr.shape[:2]
            vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        vw.write(fr)
    cap.release()
    if vw is not None:
        vw.release()
    print(f"  wrote {out_mp4} (fps {fps:.1f})")
    _write_gt(fixes, "brno_1231.json", slug,
              "https://github.com/Robotics-BUT/Brno-Urban-Dataset", "Brno, Czechia")
    return slug


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "boreas"
    if which == "boreas":
        build_boreas()
    elif which == "malaga":
        build_malaga(ROOT / "data/ext_raw/malaga/malaga-urban-dataset-extract-07")
    elif which == "brno":
        build_brno(ROOT / "data/ext_raw/brno/1_2_3_1")
