"""Metric-scale-corrected VO: keep the VO's step DIRECTIONS (heading) but replace its
step MAGNITUDES with a metric speed profile (IPM ground-flow, and DA3 metric depth).

Monocular VO normalizes each step (~uniform magnitude), so when the trajectory is
sampled at the GT waypoint TIMES it lands at the wrong along-track position (the car's
real speed varied). A metric speed profile fixes that. Measure the drift proxy (global
similarity-fit RMS vs GT) for: VO baseline vs IPM-rescaled vs DA3-rescaled.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

from src.speed_scale import estimate_route_length_from_flow

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
STRIDE = 3
MPD = 111320.0


def fit_residual(traj, traj_t, wp_t, wp_xy):
    px = np.interp(wp_t, traj_t, traj[:, 0]); py = np.interp(wp_t, traj_t, traj[:, 1])
    tf = SimilarityTransform()
    if not tf.estimate(np.c_[px, py], wp_xy):
        return None
    r = tf(np.c_[px, py]) - wp_xy
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def rescale_by_speed(xz, vo_t, speed_t, speed):
    """Rebuild the path: VO unit step directions, magnitudes = metric speed*dt."""
    d = np.diff(xz, axis=0)
    mag = np.linalg.norm(d, axis=1, keepdims=True)
    unit = d / np.clip(mag, 1e-9, None)
    spd = np.interp(vo_t[:-1], speed_t, speed)              # per-step metric speed
    dt = np.diff(vo_t)
    steps = unit * (spd * dt)[:, None]
    out = np.vstack([xz[0], xz[0] + np.cumsum(steps, axis=0)])
    return out


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

    # VO step-magnitude uniformity (is there a speed profile to recover?)
    mag = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    print(f"VO baseline drift = {vo_res:.1f} m | VO step-magnitude cv = {mag.std()/mag.mean():.2f} "
          f"(low => uniform steps, speed profile lost)")

    # ---- IPM metric speed profile (every 0.5 s) ----
    fr_stride = 15
    frames, ftimes = [], []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fi in range(0, min(n, int(420 * fps)), fr_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
        if ok:
            frames.append(f); ftimes.append(fi / fps)
    h, w = frames[0].shape[:2]
    K = np.array([[900., 0, w / 2], [0, 900., h / 2], [0, 0, 1]])
    _tot, motions = estimate_route_length_from_flow(frames, K, fps=fps, frame_stride=fr_stride)
    ipm_t = np.array(ftimes[1:1 + len(motions)])
    ipm_speed = motions / (fr_stride / fps)
    cap.release()
    ipm_xz = rescale_by_speed(xz, vo_t, ipm_t, ipm_speed)
    ipm_res = fit_residual(ipm_xz, vo_t, wp_t, wp_xy)

    print(f"\n=== drift proxy (global similarity-fit RMS vs Ulm GT) ===")
    print(f"  VO (baseline)        : {vo_res:.1f} m")
    print(f"  VO + IPM metric speed: {ipm_res:.1f} m   ({len(motions)} speed samples)")
    np.savez("data/kartaview_ulm/metric_vo.npz", ipm_t=ipm_t, ipm_speed=ipm_speed)


if __name__ == "__main__":
    main()
