"""GLOMAP (global SfM, built into COLMAP/pycolmap) as a drift-free trajectory.

Our monocular VO drifts over a 7-min clip — the wall that capped the combined chain.
GLOMAP does GLOBAL bundle adjustment, so its camera path should drift less. Extract
SIFT, match sequentially, run the global mapper, read the registered camera centres,
and measure trajectory SHAPE vs the Ulm GT (global similarity-fit RMS) — head-to-head
with monocular VO (258 m). Risk: forward dashcam motion has low parallax, so SfM may
register only a subset / fail on straight stretches.
"""

from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np
import pycolmap
from skimage.transform import SimilarityTransform

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
STRIDE = 3
MPD = 111320.0
N = int(os.environ.get("GL_N", "100"))
T0, T1 = 0.0, 420.0


def fit_residual(traj_xy, traj_t, wp_t, wp_xy):
    px = np.interp(wp_t, traj_t, traj_xy[:, 0]); py = np.interp(wp_t, traj_t, traj_xy[:, 1])
    tf = SimilarityTransform()
    if not tf.estimate(np.c_[px, py], wp_xy):
        return None
    res = tf(np.c_[px, py]) - wp_xy
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def main():
    wps = json.load(open(GT))["waypoints"]
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    wp_xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    wp_t = np.array([w["t_sec"] for w in wps])

    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    xz = np.load(NPZ)["xz"]
    vo_res = fit_residual(xz, np.arange(len(xz)) * STRIDE / fps, wp_t, wp_xy)

    times = np.linspace(T0, T1, N)
    img_dir = tempfile.mkdtemp()
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000); ok, f = cap.read()
        if ok:
            cv2.imwrite(os.path.join(img_dir, f"{i:04d}.jpg"), f)
    cap.release()
    print(f"VO residual = {vo_res:.1f} m. Running GLOMAP on {N} frames...", flush=True)

    db = os.path.join(tempfile.mkdtemp(), "db.db")
    out = tempfile.mkdtemp()
    pycolmap.extract_features(db, img_dir)
    print("  features extracted; matching...", flush=True)
    pycolmap.match_sequential(db)
    print("  matched; global mapping...", flush=True)
    recs = pycolmap.global_mapping(db, img_dir, out)
    if not recs:
        print("  GLOMAP registered NOTHING (low-parallax forward motion).")
        return
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"  GLOMAP registered {rec.num_reg_images()}/{N} images", flush=True)

    pos = {}
    for _id, im in rec.images.items():
        try:
            c = np.asarray(im.projection_center())
        except Exception:
            c = np.asarray(im.cam_from_world.inverse().translation)
        pos[int(im.name.split(".")[0])] = c
    fids = sorted(pos)
    traj = np.array([pos[i] for i in fids]); tt = times[fids]
    c = traj - traj.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    traj_xy = c @ vt[:2].T
    res = fit_residual(traj_xy, tt, wp_t, wp_xy)
    span = np.linalg.norm(traj.max(0) - traj.min(0))
    print(f"\n=== trajectory drift vs Ulm GT (global similarity-fit RMS) ===")
    print(f"  monocular VO   : {vo_res:.1f} m")
    print(f"  GLOMAP ({len(fids)} reg): {res:.1f} m   (PCA span {span:.0f} units)")


if __name__ == "__main__":
    main()
