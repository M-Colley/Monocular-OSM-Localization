"""MapAnything (3DV'26) as a low-drift metric trajectory source vs our VO.

Feeds keyframes spanning the Ulm GT clip to MapAnything's feed-forward model,
reads the regressed metric camera_poses, reduces them to the 2D driving plane,
and measures how well the trajectory SHAPE matches the GT waypoints (residual
of a global similarity fit). Compared head-to-head with the monocular VO that
currently feeds the matcher. Lower residual = less accumulated drift.
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
import os as _os
N = int(_os.environ.get("MA_N", "32"))
T0, T1 = 0.0, float(_os.environ.get("MA_T1", "420.0"))


def fit_residual(traj_xy, traj_t, wp_t, wp_xy):
    """RMS residual (m) of a global similarity fit of traj (sampled at wp_t) to wp_xy."""
    px = np.interp(wp_t, traj_t, traj_xy[:, 0])
    py = np.interp(wp_t, traj_t, traj_xy[:, 1])
    src = np.c_[px, py]
    tf = SimilarityTransform()
    if not tf.estimate(src, wp_xy):
        return None
    res = tf(src) - wp_xy
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def main() -> None:
    wps = json.load(open(GT))["waypoints"]
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    wp_xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    wp_t = np.array([w["t_sec"] for w in wps])

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- our VO baseline (already 2D xz) ---
    xz = np.load(NPZ)["xz"]
    vo_t = np.arange(len(xz)) * STRIDE / fps
    vo_res = fit_residual(xz, vo_t, wp_t, wp_xy)

    # --- MapAnything ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"VO baseline residual = {vo_res:.1f} m; loading MapAnything weights...", flush=True)
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained("facebook/map-anything").to(device).eval()
    print("MapAnything model loaded; sampling + inferring...", flush=True)

    times = np.linspace(T0, T1, N)
    folder = tempfile.mkdtemp()
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, bgr = cap.read()
        if ok:
            cv2.imwrite(os.path.join(folder, f"{i:03d}.jpg"), bgr)
    cap.release()
    views = load_images(folder)
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True,
                            use_amp=True, amp_dtype="bf16")

    def _trans(p):                                          # camera position in world
        if "cam_trans" in p:
            return np.asarray(p["cam_trans"].detach().cpu().numpy()).ravel()[:3]
        cp = np.asarray(p["camera_poses"].detach().cpu().numpy()).reshape(-1, 4, 4)[0]
        return cp[:3, 3]

    pos = np.array([_trans(p) for p in preds])             # (N,3) metric positions
    print(f"MapAnything returned {len(preds)} poses", flush=True)

    # reduce to the 2D driving plane via PCA
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ma_xy = c @ vt[:2].T
    ma_t = times[: len(ma_xy)]
    ma_res = fit_residual(ma_xy, ma_t, wp_t, wp_xy)

    span = np.linalg.norm(pos.max(0) - pos.min(0))
    print(f"\n=== trajectory drift vs Ulm GT (global similarity-fit RMS) ===")
    print(f"  monocular VO        : {vo_res:.1f} m")
    print(f"  MapAnything ({N} kf) : {ma_res:.1f} m   (metric span {span:.0f} m)")


if __name__ == "__main__":
    main()
