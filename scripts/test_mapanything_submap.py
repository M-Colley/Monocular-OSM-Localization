"""MapAnything as a SUBMAP-STITCHED metric odometry (VGGT-SLAM-lite, no gtsam).

A single feed-forward pass over a 7-min drive collapses the scale (no co-visibility
between distant frames). Instead we run MapAnything on short, high-overlap sliding
windows -- where its metric reconstruction is reliable -- and chain consecutive
windows by a Sim(3) (Umeyama) fit over their shared frames. That propagates metric
scale + orientation along the route without a global bundle/factor-graph solver.

Compared head-to-head with our monocular VO on the Ulm GT (global similarity-fit RMS).
Tunables via env: MA_DT (s between frames), MA_W (window len), MA_S (window step).
"""

from __future__ import annotations

import json
import os
import tempfile

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
STRIDE = 3
MPD = 111320.0
DT = float(os.environ.get("MA_DT", "2.0"))
W = int(os.environ.get("MA_W", "16"))
S = int(os.environ.get("MA_S", "8"))
T0, T1 = 0.0, 420.0


def fit_residual(traj_xy, traj_t, wp_t, wp_xy):
    px = np.interp(wp_t, traj_t, traj_xy[:, 0])
    py = np.interp(wp_t, traj_t, traj_xy[:, 1])
    src = np.c_[px, py]
    tf = SimilarityTransform()
    if not tf.estimate(src, wp_xy):
        return None
    res = tf(src) - wp_xy
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def umeyama(src, dst):
    """Similarity (s,R,t) mapping src->dst, 3D point sets (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(H)
    S_ = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_[2, 2] = -1
    R = U @ S_ @ Vt
    var = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S_) / max(var, 1e-9)
    t = mu_d - s * R @ mu_s
    return s, R, t


def main() -> None:
    wps = json.load(open(GT))["waypoints"]
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    wp_xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    wp_t = np.array([w["t_sec"] for w in wps])

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    xz = np.load(NPZ)["xz"]
    vo_t = np.arange(len(xz)) * STRIDE / fps
    vo_res = fit_residual(xz, vo_t, wp_t, wp_xy)

    times = np.arange(T0, T1, DT)
    M = len(times)
    print(f"VO residual={vo_res:.1f} m. MapAnything submaps: M={M} frames, "
          f"W={W} S={S} dt={DT}s", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()

    def infer_positions(ids):
        folder = tempfile.mkdtemp()
        for j, fid in enumerate(ids):
            cap.set(cv2.CAP_PROP_POS_MSEC, times[fid] * 1000)
            ok, bgr = cap.read()
            if ok:
                cv2.imwrite(os.path.join(folder, f"{j:03d}.jpg"), bgr)
        views = load_images(folder)
        with torch.no_grad():
            preds = model.infer(views, memory_efficient_inference=True,
                                use_amp=True, amp_dtype="bf16")
        out = []
        for p in preds:
            if "cam_trans" in p:
                out.append(np.asarray(p["cam_trans"].detach().cpu().numpy()).ravel()[:3])
            else:
                cp = np.asarray(p["camera_poses"].detach().cpu().numpy()).reshape(-1, 4, 4)[0]
                out.append(cp[:3, 3])
        return np.array(out)

    G = {}                                   # global frame id -> 3D position (aligned)
    starts = list(range(0, max(1, M - W + 1), S))
    if starts[-1] != M - W:
        starts.append(max(0, M - W))
    for wk, st in enumerate(starts):
        ids = list(range(st, min(st + W, M)))
        pos = infer_positions(ids)
        if wk == 0:
            for fid, p in zip(ids, pos):
                G[fid] = p
            continue
        ov = [(j, fid) for j, fid in enumerate(ids) if fid in G]
        if len(ov) < 3:                      # not enough shared frames to align
            continue
        src = np.array([pos[j] for j, _ in ov])
        dst = np.array([G[fid] for _, fid in ov])
        s, R, t = umeyama(src, dst)
        # MapAnything is metric: the inter-window scale must be ~1. A wild scale
        # means a degenerate (collinear / low-parallax) overlap -> discard the
        # estimated scale and align rigidly (scale fixed to 1) instead.
        if not (0.6 <= s <= 1.7):
            mu_s, mu_d = src.mean(0), dst.mean(0)
            H = (dst - mu_d).T @ (src - mu_s)
            U, _, Vt = np.linalg.svd(H)
            Dz = np.eye(3)
            if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                Dz[2, 2] = -1
            R = U @ Dz @ Vt; s = 1.0; t = mu_d - R @ mu_s
        for j, fid in enumerate(ids):
            if fid not in G:
                G[fid] = s * R @ pos[j] + t
        print(f"  win {wk+1}/{len(starts)} aligned ({len(ov)} overlap, s={s:.2f})", flush=True)
    cap.release()

    fids = sorted(G)
    traj = np.array([G[f] for f in fids])
    tt = times[fids]
    c = traj - traj.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    traj_xy = c @ vt[:2].T
    res = fit_residual(traj_xy, tt, wp_t, wp_xy)
    span = np.linalg.norm(traj.max(0) - traj.min(0))
    print(f"\n=== trajectory drift vs Ulm GT (global similarity-fit RMS) ===")
    print(f"  monocular VO              : {vo_res:.1f} m")
    print(f"  MapAnything submap-stitch : {res:.1f} m   (metric span {span:.0f} m, "
          f"{len(starts)} windows)")


if __name__ == "__main__":
    main()
