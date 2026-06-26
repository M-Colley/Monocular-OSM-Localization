"""OrienterNet sequential fusion across all clips with per-frame GPS.

Runs the proper RigidAligner sequential fusion (the one that hit 1.9 m on
KITTI 0033) on every clip that ships dense per-frame ground truth — KITTI
drive_0033, KITTI drive_0009, comma2k19 — and saves per-frame
(true, predicted) lat/lon + error to output/orienternet_all.json for
visualization. Ulm/London only have sparse waypoints (no dense odometry),
so they can't drive sequential fusion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("third_party/OrienterNet")))
from maploc.models.sequential import RigidAligner  # noqa: E402
from maploc.osm.tiling import TileManager  # noqa: E402
from maploc.utils.geo import BoundaryBox, Projection  # noqa: E402
from maploc.utils.wrappers import Camera  # noqa: E402
from scripts.test_orienternet import load_model, prepare  # noqa: E402
from src.comma2k19 import load_route_track  # noqa: E402
from src.kitti_raw import load_oxts_track  # noqa: E402

R, MPD = 6371000.0, 111320.0


def kitti_gps(drive):
    fx = load_oxts_track(drive)

    def get(i):
        f = Path(drive) / "oxts" / "data" / f"{i:010d}.txt"
        p = f.read_text().split()
        return float(p[0]), float(p[1]), np.degrees(float(p[3])), np.degrees(float(p[4]))
    return len(fx), get


def comma_gps(segdirs):
    track = load_route_track(segdirs)

    def get(i):
        f = track[min(i, len(track) - 1)]
        return f.lat, f.lon, 0.0, 0.0
    return len(track), get


CLIPS = {
    "KITTI drive_0033": dict(
        video="data/kitti/drive_0033.mp4", fps=10.0, focal=721.5, ref_w=1242,
        gps=lambda: kitti_gps("data/kitti/2011_09_30/2011_09_30_drive_0033_sync")),
    "KITTI drive_0009": dict(
        video="data/kitti/drive_0009.mp4", fps=10.0, focal=721.5, ref_w=1242,
        gps=lambda: kitti_gps("data/kitti/2011_09_26/2011_09_26_drive_0009_sync")),
    "comma2k19 (Daly City)": dict(
        video="data/comma/route_148.mp4", fps=20.0, focal=910.0, ref_w=1164,
        gps=lambda: comma_gps([f"data/comma/extracted/b0c9d2329ad1606b_2018-08-17--14-55-39/{s}"
                               for s in (1, 2, 3, 4)])),
}


def err_m(a, b):
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def run_clip(name, spec, model, cfg, device):
    ppm = cfg.data.pixel_per_meter
    n_gps, get = spec["gps"]()
    cap = cv2.VideoCapture(spec["video"])
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_gps, n_video)
    chunk_frames = 10
    step = max(1, int(round(1.3 * spec["fps"])))      # ~1.3 s spacing
    span = chunk_frames * step
    # 4 chunks spread along the drive.
    starts = np.linspace(int(0.05 * n), int(0.95 * n) - span, 4).round().astype(int)
    out = []
    for cstart in starts:
        frames = list(range(cstart, cstart + span, step))
        if frames[-1] >= n:
            continue
        # Common projection at chunk centre (offset prior ~30 m).
        latm, lonm, *_ = get(frames[len(frames) // 2])
        prior0 = np.array([latm + 30 / MPD, lonm + 30 / (MPD * np.cos(np.radians(latm)))])
        proj = Projection(*prior0)
        per = []
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, bgr = cap.read()
            if not ok:
                continue
            lat, lon, roll, pitch = get(fr)
            true_ll = np.array([lat, lon])
            xy = proj.project(true_ll)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            fx = spec["focal"] * w / spec["ref_w"]
            cam = Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": w, "height": h,
                                    "params": [fx, w / 2 + 0.5, h / 2 + 0.5]})
            center = proj.project(true_ll + np.array(
                [30 / MPD, 30 / (MPD * np.cos(np.radians(lat)))]))
            bbox = BoundaryBox(center, center) + 96
            canvas = TileManager.from_bbox(proj, bbox + 10, ppm).query(bbox)
            data = {k: v.to(device)[None] for k, v in
                    prepare(image, cam, canvas, cfg, (roll, pitch), model).items()}
            with torch.no_grad():
                lp = model(data)["log_probs"].squeeze(0)
            per.append((lp, canvas, xy, true_ll, fr))
        if len(per) < 3:
            continue
        # yaw from motion direction (aligner solves the global rotation).
        xys = np.array([p[2] for p in per])
        d = np.gradient(xys, axis=0)
        yaws = (90 - np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360
        aligner = RigidAligner(num_rotations=per[0][0].shape[-1])
        order = sorted(range(len(per)), key=lambda i: abs(i - len(per) // 2))
        for i in order:
            lp, canvas, xy, _, _ = per[i]
            aligner.update(lp, canvas, torch.tensor(xy, device=device).float(),
                           torch.tensor(float(yaws[i]), device=device).float())
        aligner.compute()
        for i, (_lp, _c, xy, true_ll, fr) in enumerate(per):
            gxy, _ = aligner.transform(torch.tensor(xy, device=device).float(),
                                       torch.tensor(float(yaws[i]), device=device).float())
            pll = proj.unproject(gxy.cpu().numpy())
            out.append({"frame": int(fr), "true": [float(true_ll[0]), float(true_ll[1])],
                        "pred": [float(pll[0]), float(pll[1])], "err_m": err_m(pll, true_ll)})
    cap.release()
    e = np.array([o["err_m"] for o in out]) if out else np.array([np.inf])
    print(f"{name:24s} median {np.median(e):6.1f} m  recall@5m {100*np.mean(e<=5):3.0f}%  "
          f"@10m {100*np.mean(e<=10):3.0f}%  ({len(out)} frames)")
    return out, dict(median=float(np.median(e)), recall5=float(100 * np.mean(e <= 5)),
                     recall10=float(100 * np.mean(e <= 10)), n=len(out))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    results = {}
    for name, spec in CLIPS.items():
        pts, summary = run_clip(name, spec, model, cfg, device)
        results[name] = {"points": pts, "summary": summary}
    Path("output").mkdir(exist_ok=True)
    json.dump(results, open("output/orienternet_all.json", "w"), indent=1)
    print("\nsaved output/orienternet_all.json")


if __name__ == "__main__":
    main()
