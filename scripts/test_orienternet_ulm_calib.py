"""Find Ulm's camera FOV by sweeping it through OrienterNet.

OrienterNet needs the right focal to build a correct BEV. Ulm is an unknown
YouTube dashcam, so we sweep the horizontal FOV and, for several waypoints,
run a short OrienterNet fusion window — the FOV that minimises localization
error IS the camera's calibration. Also sweeps a small pitch (gravity).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform

sys.path.insert(0, str(Path("third_party/OrienterNet")))
from maploc.models.sequential import RigidAligner  # noqa: E402
from maploc.osm.tiling import TileManager  # noqa: E402
from maploc.utils.geo import BoundaryBox, Projection  # noqa: E402
from maploc.utils.wrappers import Camera  # noqa: E402
from scripts.test_orienternet import load_model, prepare  # noqa: E402

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
STRIDE = 3
R, MPD = 6371000.0, 111320.0


def err_m(a, b):
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def main() -> None:
    device = torch.device("cuda")
    model, cfg = load_model(device)
    ppm = cfg.data.pixel_per_meter
    xz = np.load(NPZ)["xz"]
    wps = json.load(open(GT))["waypoints"]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STRIDE / fps
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    proj = Projection(lat0, lon0)
    wp_xy = np.array([proj.project(np.array([w["lat"], w["lon"]])) for w in wps])
    wp_vo = np.clip([int(round(w["t_sec"] / dt)) for w in wps], 0, len(xz) - 1)
    tf = SimilarityTransform(); tf.estimate(xz[wp_vo], wp_xy)
    s = tf.scale; Rm = tf.params[:2, :2] / s

    # Use the 5 central waypoints (best VO + map coverage).
    ks = list(range(2, min(7, len(wps))))

    import time

    def fetch_tiler(bbox):
        for attempt in range(6):
            try:
                return TileManager.from_bbox(proj, bbox, ppm)
            except ValueError as e:
                if "509" in str(e) and attempt < 5:
                    print(f"      OSM 509 rate-limited; waiting 30 s (try {attempt+1})")
                    time.sleep(30)
                    continue
                raise

    # Pre-fetch ONE OSM tile per waypoint (covering its whole window) and
    # crop per-frame from it — 5 requests, not 45 — then reuse across FOVs.
    cache = {}
    for k in ks:
        c = int(wp_vo[k])
        win = [i for i in range(c - 52, c + 53, 13) if 0 <= i < len(xz)]
        if len(win) < 4:
            continue
        geo = wp_xy[k] + s * (xz[win] - xz[c]) @ Rm.T
        lo = geo.min(0) - 140 + 30; hi = geo.max(0) + 140 + 30
        tiler = fetch_tiler(BoundaryBox(lo, hi))
        frames = []
        for j, i in enumerate(win):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * STRIDE))
            ok, bgr = cap.read()
            if not ok:
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            canvas = tiler.query(BoundaryBox(geo[j] + 30, geo[j] + 30) + 110)
            frames.append((image, canvas, geo[j], j))
        cache[k] = (frames, len(win))
        time.sleep(2)
    cap.release()

    def eval_fov(fov_deg, pitch_deg):
        errs, confs = [], []
        for k in ks:
            if k not in cache:
                continue
            frames, nwin = cache[k]
            per = []
            for image, canvas, g, j in frames:
                h, ww = image.shape[:2]
                f = ww / (2 * np.tan(np.deg2rad(fov_deg) / 2))
                cam = Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": ww, "height": h,
                                        "params": [f, ww / 2 + 0.5, h / 2 + 0.5]})
                data = {kk: v.to(device)[None] for kk, v in
                        prepare(image, cam, canvas, cfg, (0.0, pitch_deg), model).items()}
                with torch.no_grad():
                    lp = model(data)["log_probs"].squeeze(0)
                # GT-FREE confidence: how peaked is the per-frame belief?
                # log_softmax over the whole (u,v,rot) volume, take the max.
                confs.append(float(torch.log_softmax(lp.flatten(), 0).max()))
                per.append((lp, canvas, g, j))
            if len(per) < 3:
                continue
            win = list(range(nwin))
            gxy = np.array([p[2] for p in per]); d = np.gradient(gxy, axis=0)
            yaws = (90 - np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360
            al = RigidAligner(num_rotations=per[0][0].shape[-1])
            for i in sorted(range(len(per)), key=lambda i: abs(per[i][3] - len(win) // 2)):
                lp, canvas, xy, _ = per[i]
                al.update(lp, canvas, torch.tensor(xy, device=device).float(),
                          torch.tensor(float(yaws[i]), device=device).float())
            al.compute()
            ci = min(range(len(per)), key=lambda i: abs(per[i][3] - len(win) // 2))
            g, _ = al.transform(torch.tensor(per[ci][2], device=device).float(),
                                torch.tensor(float(yaws[ci]), device=device).float())
            errs.append(err_m(proj.unproject(g.cpu().numpy()), np.array([wps[k]["lat"], wps[k]["lon"]])))
        return (np.median(errs) if errs else np.inf,
                float(np.mean(confs)) if confs else -np.inf)

    print("FOV sweep:   GT-free confidence (no waypoints) vs actual error")
    by_conf = (-np.inf, None); by_err = (np.inf, None)
    for fov in [80, 95, 110, 125, 140, 155]:
        m, conf = eval_fov(fov, 0.0)
        star = ""
        print(f"  HFOV {fov:3d} deg -> confidence {conf:7.3f}   |  actual median {m:6.1f} m{star}")
        if conf > by_conf[0]:
            by_conf = (conf, fov)
        if m < by_err[0]:
            by_err = (m, fov)
    print(f"\n  GT-FREE pick (max confidence): {by_conf[1]} deg")
    print(f"  GT pick (min actual error):    {by_err[1]} deg  "
          f"-> they {'MATCH' if by_conf[1] == by_err[1] else 'differ'}")
    cap.release()


if __name__ == "__main__":
    main()
