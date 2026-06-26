"""Does sweeping PITCH (jointly with FOV) by confidence beat fixed pitch?

  FIXED : auto FOV sweep, hardcoded gravity (0, -4)   -> the previous best
  JOINT : auto FOV + pitch sweep (gravity=None)        -> the new auto-cal
Reported as metric error vs the Ulm GT waypoints, fully GT-free calibration.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

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

    fixed, joint = [], []
    for k in (3, 4, 6, 9):
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

        rf = refine_route(frames, prior_ll, None, tile_m=130.0, gravity=(0.0, -4.0))
        rj = refine_route(frames, prior_ll, None, tile_m=130.0)  # gravity=None -> joint
        ef = None if rf is None else err_m(rf[len(rf) // 2], gt)
        ej = None if rj is None else err_m(rj[len(rj) // 2], gt)
        if ef is not None:
            fixed.append(ef)
        if ej is not None:
            joint.append(ej)
        print(f"t={wps[k]['t_sec']:>4}s  FIXED(pitch -4)={ef}  JOINT(auto pitch)={ej}")
    cap.release()
    if fixed and joint:
        print(f"\nFIXED  median: {np.median(fixed):.1f} m")
        print(f"JOINT  median: {np.median(joint):.1f} m")


if __name__ == "__main__":
    main()
