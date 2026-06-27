"""VGGT-SLAM 2.0 trajectory drift vs the monocular-VO baseline, measured against
the Ulm GT waypoints.

VGGT-SLAM 2.0 (built with the custom Windows SL4 gtsam) produces a globally-optimised
trajectory on the SL(4) manifold. We compare its drift to the plain VO trajectory using
the SAME drift proxy used elsewhere: the RMS residual after an optimal similarity
(rotation+scale+translation) alignment to the GT route. Lower = less drift.

  - VGGT-SLAM poses:  frame_id x y z qx qy qz qw   (3D; frame_id = extracted index)
  - extraction was 3 fps  ->  time_sec = frame_id / 3
  - GT: lat/lon/t_sec  ->  local metric XY
"""

from __future__ import annotations

import json

import numpy as np

POSES = "C:/vggt_frames/ulm_full_poses.txt"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
VO_NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
FPS_EXTRACT = 3.0          # frames extracted at 3 fps
VO_STRIDE = 3
MPD = 111320.0


def umeyama_sim(src, dst):
    """Similarity transform (scale, R, t) mapping src->dst (least squares). Nx D."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s = src.mean(0); mu_d = dst.mean(0)
    s = src - mu_s; d = dst - mu_d
    cov = d.T @ s / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (s ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - scale * R @ mu_s
    out = (scale * (R @ src.T).T) + t
    rms = float(np.sqrt(np.mean(np.sum((out - dst) ** 2, axis=1))))
    return rms, out


def gt_local_xy(wps):
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    t = np.array([w["t_sec"] for w in wps])
    return xy, t


def main():
    wps = json.load(open(GT))["waypoints"]
    gt_xy, gt_t = gt_local_xy(wps)
    order = np.argsort(gt_t); gt_t = gt_t[order]; gt_xy = gt_xy[order]

    # ---- VGGT-SLAM trajectory ----
    rows = np.loadtxt(POSES)
    fid = rows[:, 0]; pos3 = rows[:, 1:4]
    vt = fid / FPS_EXTRACT
    # keep keyframes whose time falls inside the GT span
    m = (vt >= gt_t.min()) & (vt <= gt_t.max())
    vt_k = vt[m]; pos3_k = pos3[m]
    gx = np.interp(vt_k, gt_t, gt_xy[:, 0]); gy = np.interp(vt_k, gt_t, gt_xy[:, 1])
    gt_k = np.c_[gx, gy, np.zeros_like(gx)]                  # embed GT at z=0
    rms_vggt, _ = umeyama_sim(pos3_k, gt_k)

    # ---- VO baseline (same proxy) ----
    xz = np.load(VO_NPZ)["xz"]
    # video fps for VO timestamps
    import cv2
    cap = cv2.VideoCapture("data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
    vo_t = np.arange(len(xz)) * VO_STRIDE / fps
    mv = (vo_t >= gt_t.min()) & (vo_t <= gt_t.max())
    vo_t_k = vo_t[mv]; xz_k = xz[mv]
    gx2 = np.interp(vo_t_k, gt_t, gt_xy[:, 0]); gy2 = np.interp(vo_t_k, gt_t, gt_xy[:, 1])
    rms_vo, _ = umeyama_sim(xz_k, np.c_[gx2, gy2])

    print("=== drift proxy: RMS after optimal similarity-fit to Ulm GT (0-420 s) ===")
    print(f"  monocular VO baseline : {rms_vo:6.1f} m   ({len(xz_k)} pts)")
    print(f"  VGGT-SLAM 2.0 (SL4)   : {rms_vggt:6.1f} m   ({len(vt_k)} keyframes)")
    print(f"  change                : {100*(rms_vggt-rms_vo)/rms_vo:+.0f} %")
    np.savez("C:/vggt_frames/vggt_slam_drift.npz",
             vt=vt_k, pos3=pos3_k, gt_k=gt_k, rms_vggt=rms_vggt, rms_vo=rms_vo)


if __name__ == "__main__":
    main()
