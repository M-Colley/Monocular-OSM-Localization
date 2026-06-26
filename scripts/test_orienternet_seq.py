"""OrienterNet with PROPER sequential fusion (RigidAligner) on KITTI 0033.

Single-frame OrienterNet is ~2 m on good frames but ambiguous in repetitive
residential layout. The paper's fix is to accumulate per-frame probability
VOLUMES across a short sequence via the odometry (RigidAligner), which
sharpens to the true pose. This replicates that on our data, using the
exact KITTI convention from maploc (yaw = 90 - OXTS_yaw_deg; camera xy in
the projection frame; odometry from OXTS, which our VO supplies).

    python scripts/test_orienternet_seq.py
"""

from __future__ import annotations

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

DRIVE = Path("data/kitti/2011_09_30/2011_09_30_drive_0033_sync")
KITTI_FX = 721.5


def oxts(frame: int):
    """(lat, lon, roll_deg, pitch_deg, yaw_deg) for a frame index."""
    f = DRIVE / "oxts" / "data" / f"{frame:010d}.txt"
    p = f.read_text().split()
    lat, lon = float(p[0]), float(p[1])
    roll, pitch, yaw = (np.degrees(float(p[i])) for i in (3, 4, 5))
    return lat, lon, roll, pitch, yaw


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    ppm = cfg.data.pixel_per_meter
    cap = cv2.VideoCapture("data/kitti/drive_0033.mp4")
    R, MPD = 6371000.0, 111320.0
    OFF = 30.0  # constant GPS-bias offset (m) so the prior is realistic

    def run_chunk(frames: list[int]):
        # Common projection at the chunk centre (camera xy + canvases share it).
        mid = oxts(frames[len(frames) // 2])
        prior0 = np.array([mid[0] + OFF / MPD,
                           mid[1] + OFF / (MPD * np.cos(np.radians(mid[0])))])
        proj = Projection(*prior0)
        per = []
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, bgr = cap.read()
            if not ok:
                continue
            lat, lon, roll, pitch, yaw = oxts(fr)
            true_ll = np.array([lat, lon])
            xy = proj.project(true_ll)                  # camera xy (odometry)
            yaw_m = (90.0 - yaw) % 360.0                # maploc convention
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            cam = Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": w, "height": h,
                                    "params": [KITTI_FX * w / 1242.0, w / 2 + 0.5, h / 2 + 0.5]})
            # Per-frame canvas centred on the (offset) prior, in proj coords.
            center = proj.project(true_ll + np.array(
                [OFF / MPD, OFF / (MPD * np.cos(np.radians(lat)))]))
            bbox = BoundaryBox(center, center) + 96
            canvas = TileManager.from_bbox(proj, bbox + 10, ppm).query(bbox)
            data = {k: v.to(device)[None] for k, v in
                    prepare(image, cam, canvas, cfg, (roll, pitch), model).items()}
            with torch.no_grad():
                lp = model(data)["log_probs"].squeeze(0)
            per.append((lp, canvas, torch.tensor(xy), torch.tensor(yaw_m), true_ll))
        if not per:
            return []
        aligner = RigidAligner(num_rotations=per[0][0].shape[-1])
        for lp, canvas, xy, yaw_m, _ in per:
            aligner.update(lp, canvas, xy.to(device).float(), yaw_m.to(device).float())
        aligner.compute()
        errs = []
        for _lp, _c, xy, yaw_m, true_ll in per:
            gxy, _ = aligner.transform(xy.to(device).float(), yaw_m.to(device).float())
            pll = proj.unproject(gxy.cpu().numpy())
            dlat = np.radians(pll[0] - true_ll[0]); dlon = np.radians(pll[1] - true_ll[1])
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(true_ll[0])) ** 2 * np.sin(dlon / 2) ** 2
            errs.append(float(2 * R * np.arcsin(np.sqrt(a))))
        return errs

    all_errs = []
    for start in [250, 550, 850, 1150]:        # 4 chunks along the drive
        chunk = list(range(start, start + 130, 13))  # ~13 s @ 10 Hz, 10 frames
        e = run_chunk(chunk)
        if e:
            all_errs += e
            print(f"  chunk @{start}: median fused {np.median(e):.1f} m  "
                  f"(min {min(e):.1f}, max {max(e):.1f})")
    cap.release()
    if all_errs:
        e = np.array(all_errs)
        print(f"\nOrienterNet SEQUENTIAL on KITTI 0033 ({len(e)} frames):")
        print(f"  median {np.median(e):.1f} m  recall@3m {100*np.mean(e<=3):.0f}%  "
              f"@5m {100*np.mean(e<=5):.0f}%  @10m {100*np.mean(e<=10):.0f}%")


if __name__ == "__main__":
    main()
