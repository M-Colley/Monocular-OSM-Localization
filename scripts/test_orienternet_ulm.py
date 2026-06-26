"""OrienterNet on Ulm (urban city-centre = OrienterNet's domain) at the GT
waypoint frames. Tests whether a more urban scene than KITTI residential
gives better neural BEV->OSM localization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("third_party/OrienterNet")))
from maploc.models.voting import argmax_xyr, fuse_gps  # noqa: E402
from maploc.osm.tiling import TileManager  # noqa: E402
from maploc.utils.geo import BoundaryBox, Projection  # noqa: E402
from maploc.utils.wrappers import Camera  # noqa: E402
from scripts.test_orienternet import load_model, prepare  # noqa: E402

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
GT = "ground_truth/ulm_ULl8s4qydrk.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    ppm = cfg.data.pixel_per_meter
    wps = json.load(open(GT))["waypoints"]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    R, MPD = 6371000.0, 111320.0
    errs = []
    for wp in wps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(wp["t_sec"] * fps))
        ok, bgr = cap.read()
        if not ok:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        f = w / (2 * np.tan(np.deg2rad(70) / 2))   # ~70deg dashcam HFOV
        camera = Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": w, "height": h,
                                   "params": [f, w / 2 + 0.5, h / 2 + 0.5]})
        true_ll = np.array([wp["lat"], wp["lon"]])
        prior = true_ll + np.array([30.0 / MPD, 30.0 / (MPD * np.cos(np.radians(wp["lat"])))])
        proj = Projection(*prior)
        bbox = BoundaryBox(proj.project(prior), proj.project(prior)) + 128
        try:
            canvas = TileManager.from_bbox(proj, bbox + 10, ppm).query(bbox)
            data = {k: v.to(device)[None] for k, v in
                    prepare(image, camera, canvas, cfg, (0.0, 0.0), model).items()}
            with torch.no_grad():
                pred = model(data)
            lp = pred["log_probs"].squeeze(0)
            lp = fuse_gps(lp, torch.from_numpy(canvas.to_uv(bbox.center)).to(lp), ppm, sigma=108)
            xyr = argmax_xyr(lp).cpu().numpy()
            pll = proj.unproject(canvas.to_xy(xyr[:2]))
            dlat = np.radians(pll[0] - true_ll[0]); dlon = np.radians(pll[1] - true_ll[1])
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(true_ll[0])) ** 2 * np.sin(dlon / 2) ** 2
            e = float(2 * R * np.arcsin(np.sqrt(a)))
            errs.append(e)
            print(f"  t={wp['t_sec']:>4}s: {e:6.1f} m")
        except Exception as ex:
            print(f"  t={wp['t_sec']}s FAILED: {type(ex).__name__}: {ex}")
    cap.release()
    if errs:
        e = np.array(errs)
        print(f"\nOrienterNet Ulm ({len(e)} waypoints): median {np.median(e):.1f} m  "
              f"recall@5m {100*np.mean(e<=5):.0f}%  @10m {100*np.mean(e<=10):.0f}%")


if __name__ == "__main__":
    main()
