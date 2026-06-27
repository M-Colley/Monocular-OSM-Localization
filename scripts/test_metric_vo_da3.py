"""DA3 metric-depth speed profile -> re-scale the VO (#1). DA3 outputs metric
extrinsics, so the camera-centre step magnitudes between keyframes are a METRIC
speed profile (more principled than IPM's flat-ground guess). Keep VO step
directions, replace magnitudes with the DA3 speed, measure drift vs Ulm GT.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

from src.da3_reconstruction import da3_trajectory_xy, reconstruct_with_da3

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
STRIDE = 3
MPD = 111320.0
N_SAMPLE = 130
N_KEYFRAMES = 64


def fit_residual(traj, traj_t, wp_t, wp_xy):
    px = np.interp(wp_t, traj_t, traj[:, 0]); py = np.interp(wp_t, traj_t, traj[:, 1])
    tf = SimilarityTransform()
    if not tf.estimate(np.c_[px, py], wp_xy):
        return None
    r = tf(np.c_[px, py]) - wp_xy
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def rescale_by_speed(xz, vo_t, speed_t, speed):
    d = np.diff(xz, axis=0)
    unit = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
    spd = np.clip(np.interp(vo_t[:-1], speed_t, speed), 0, None)
    steps = unit * (spd * np.diff(vo_t))[:, None]
    return np.vstack([xz[0], xz[0] + np.cumsum(steps, axis=0)])


def main():
    wps = json.load(open(GT))["waypoints"]
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    wp_xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    wp_t = np.array([w["t_sec"] for w in wps])

    xz = np.load(NPZ)["xz"]
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vo_t = np.arange(len(xz)) * STRIDE / fps
    vo_res = fit_residual(xz, vo_t, wp_t, wp_xy)

    times = np.linspace(0, 420, N_SAMPLE)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000); ok, f = cap.read()
        frames.append(f if ok else frames[-1])
    cap.release()
    print(f"VO baseline drift = {vo_res:.1f} m. Running DA3 on {N_SAMPLE} frames "
          f"({N_KEYFRAMES} keyframes)...", flush=True)

    rec = reconstruct_with_da3(frames, n_keyframes=N_KEYFRAMES, device="cuda")
    kf = np.asarray(rec.keyframe_indices)
    kf_t = times[kf]
    da3_xy = np.asarray(da3_trajectory_xy(rec))
    order = np.argsort(kf_t); kf_t = kf_t[order]; da3_xy = da3_xy[order]
    seg = np.linalg.norm(np.diff(da3_xy, axis=0), axis=1)
    seg_dt = np.diff(kf_t)
    da3_speed = seg / np.clip(seg_dt, 1e-6, None)
    da3_speed_t = (kf_t[:-1] + kf_t[1:]) / 2
    da3_xz = rescale_by_speed(xz, vo_t, da3_speed_t, da3_speed)
    da3_res = fit_residual(da3_xz, vo_t, wp_t, wp_xy)

    print(f"\n=== drift proxy (global similarity-fit RMS vs Ulm GT) ===")
    print(f"  VO (baseline)         : {vo_res:.1f} m")
    print(f"  VO + IPM metric speed : 227.4 m  (from test_metric_vo.py)")
    print(f"  VO + DA3 metric speed : {da3_res:.1f} m   ({len(kf)} keyframes)")


if __name__ == "__main__":
    main()
