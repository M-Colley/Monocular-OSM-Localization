"""A/B: does AnyCalib's focal beat the confidence sweep? (GravCal/CalibAnyView
have no public code; AnyCalib, arXiv 2503.12701, is the available newer
model-agnostic single-view calibrator — the GeoCalib successor.)

Same protocol as test_geocalib_ab.py: per Ulm GT window, OrienterNet fusion with
  A  : auto FOV sweep + gravity (0,-4)   (current)
  B  : AnyCalib per-window focal + gravity (0,-4)
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform

from anycalib import AnyCalib
from src.orienternet_localizer import refine_route

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
    xz = np.load(NPZ)["xz"]
    wps = json.load(open(GT))["waypoints"]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STRIDE / fps
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    wp_xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    wp_vo = np.clip([int(round(w["t_sec"] / dt)) for w in wps], 0, len(xz) - 1)
    tf = SimilarityTransform(); tf.estimate(xz[wp_vo], wp_xy)
    s = tf.scale; Rm = tf.params[:2, :2] / s

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ac = AnyCalib(model_id="anycalib_pinhole")
    if hasattr(ac, "to"):
        ac = ac.to(dev)
    probed = {}

    def anycalib_focal(frames):
        fs = []
        for bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = torch.from_numpy(rgb).permute(2, 0, 1).float().to(dev) / 255.0
            res = ac.predict(im, "pinhole")
            if not probed:
                probed["keys"] = list(res.keys())
            intr = res.get("intrinsics", res.get("pred_intrinsics"))
            arr = np.atleast_1d(intr.detach().cpu().numpy().ravel())
            fs.append(float(arr[0]))            # pinhole: [f, cx, cy]
        return float(np.median(fs))

    fixed, anyc = [], []
    for k in (4, 6):
        c = int(wp_vo[k])
        win = [i for i in range(c - 52, c + 53, 13) if 0 <= i < len(xz)]
        if len(win) < 4:
            continue
        geo = wp_xy[k] + s * (xz[win] - xz[c]) @ Rm.T
        prior_ll = np.c_[lat0 + geo[:, 1] / MPD, lon0 + geo[:, 0] / (MPD * cl)]
        frames = []
        for i in win:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * STRIDE))
            ok, bgr = cap.read()
            frames.append(bgr if ok else frames[-1])
        gt = np.array([wps[k]["lat"], wps[k]["lon"]])

        f_ac = anycalib_focal(frames)
        hfov = np.degrees(2 * np.arctan(frames[0].shape[1] / (2 * f_ac)))
        eA = refine_route(frames, prior_ll, None, tile_m=130.0, gravity=(0.0, -4.0))
        eB = refine_route(frames, prior_ll, f_ac, tile_m=130.0, gravity=(0.0, -4.0))
        eA = None if eA is None else err_m(eA[len(eA) // 2], gt)
        eB = None if eB is None else err_m(eB[len(eB) // 2], gt)
        if eA is not None: fixed.append(eA)
        if eB is not None: anyc.append(eB)
        print(f"t={wps[k]['t_sec']:>4}s AnyCalib HFOV={hfov:5.1f} | A(sweep)={eA} B(AnyCalib)={eB}")
    cap.release()
    print("AnyCalib output keys:", probed.get("keys"))
    if fixed and anyc:
        print(f"\n  A  sweep+pitch-4 : {np.median(fixed):.1f} m")
        print(f"  B  AnyCalib focal: {np.median(anyc):.1f} m")


if __name__ == "__main__":
    main()
