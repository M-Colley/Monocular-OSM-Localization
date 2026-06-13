"""Metric scale from ground-plane optical flow (idea 3).

Monocular VO loses metric scale (unit-norm steps). If we assume the
road in front of the camera is a flat plane at a known height and tilt
(the same assumption the IPM module already makes), then features on
that plane move, between consecutive frames, by the vehicle's real
displacement. Tracking road-surface features and back-projecting them
onto the ground plane therefore yields a per-frame *metric* speed, and
summing speed x dt gives the real route length — an absolute scale that
needs no GPS, no anchors, and no second model.

Caveats this inherits from the flat-ground assumption: it's only valid
where the road really is planar (degrades on hills/ramps), and it's
sensitive to the camera-height/pitch/intrinsics guesses (an
uncalibrated YouTube dashcam has none of these for real). So treat the
output as an approximate scale, gated like the other scale sources.

The geometry (:func:`image_to_ground`) is exact and unit-tested; the
flow estimator (:func:`estimate_route_length_from_flow`) is the noisy
empirical part.
"""

from __future__ import annotations

import numpy as np

from .ipm import _rotation_pitch_roll


def image_to_ground(
    pts_uv: np.ndarray,
    K: np.ndarray,
    *,
    camera_height_m: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Back-project image pixels onto the flat road plane.

    Returns Nx2 ground coordinates ``(X right, Z forward)`` in metres
    for each input pixel ``(u, v)``, assuming the road is a plane at
    ``camera_height_m`` below the camera (OpenCV frame: +x right, +y
    down, +z forward; the IPM convention). Pixels whose ray points at or
    above the horizon (no ground intersection) come back as ``nan``.
    """
    pts_uv = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)
    Kinv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    R = _rotation_pitch_roll(pitch_deg, roll_deg)   # camera→vehicle
    hom = np.column_stack([pts_uv, np.ones(len(pts_uv))])    # N x 3
    rays_cam = (Kinv @ hom.T).T                              # N x 3
    rays_veh = (R @ rays_cam.T).T                            # N x 3, +y down
    out = np.full((len(pts_uv), 2), np.nan)
    dy = rays_veh[:, 1]
    hits = dy > 1e-6                                          # ray goes downward
    lam = np.where(hits, camera_height_m / np.where(hits, dy, 1.0), np.nan)
    out[hits, 0] = lam[hits] * rays_veh[hits, 0]             # X (right)
    out[hits, 1] = lam[hits] * rays_veh[hits, 2]             # Z (forward)
    return out


def estimate_route_length_from_flow(
    frames: list,
    K: np.ndarray,
    *,
    camera_height_m: float = 1.4,
    pitch_deg: float = 6.0,
    fps: float = 30.0,
    frame_stride: int = 1,
    roi_top_frac: float = 0.6,
    near_clip_m: float = 4.0,
    far_clip_m: float = 25.0,
    min_features: int = 12,
    max_speed_mps: float = 40.0,
) -> tuple[float, np.ndarray]:
    """Estimate total route length (m) from ground-plane optical flow.

    For each consecutive pair, tracks features in the lower ``1 -
    roi_top_frac`` of the frame (the road), back-projects matches onto
    the ground plane, and takes the median displacement of points within
    ``[near_clip_m, far_clip_m]`` forward as the inter-frame metric
    motion. Per-frame speed = motion x fps / stride; total length = sum
    of motions. Returns ``(total_length_m, per_pair_motion_m)``.

    Robust by construction (median over features, clip on implausible
    speeds), but only as accurate as the flat-ground + calibration
    assumptions.
    """
    import cv2

    if len(frames) < 2:
        return 0.0, np.zeros(0)
    dt = frame_stride / max(fps, 1e-6)
    h, w = frames[0].shape[:2]
    roi_y0 = int(h * roi_top_frac)
    motions: list[float] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[roi_y0:, :] = 255
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                     minDistance=8, mask=mask)
        if p0 is not None and len(p0) >= min_features:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
            ok = (st.ravel() == 1)
            a = p0.reshape(-1, 2)[ok]
            b = p1.reshape(-1, 2)[ok]
            if len(a) >= min_features:
                g0 = image_to_ground(a, K, camera_height_m=camera_height_m,
                                     pitch_deg=pitch_deg)
                g1 = image_to_ground(b, K, camera_height_m=camera_height_m,
                                     pitch_deg=pitch_deg)
                valid = (
                    np.isfinite(g0).all(axis=1) & np.isfinite(g1).all(axis=1)
                    & (g0[:, 1] >= near_clip_m) & (g0[:, 1] <= far_clip_m)
                )
                if valid.sum() >= min_features:
                    disp = np.linalg.norm(g1[valid] - g0[valid], axis=1)
                    m = float(np.median(disp))
                    if 0.0 <= m <= max_speed_mps * dt:
                        motions.append(m)
        prev_gray = gray
    motions_arr = np.asarray(motions, dtype=np.float64)
    # Total = per-pair median motion, scaled up for pairs we skipped
    # (low-feature frames) so the estimate isn't biased short.
    n_pairs = len(frames) - 1
    if len(motions_arr) == 0:
        return 0.0, motions_arr
    total = float(motions_arr.sum() * n_pairs / len(motions_arr))
    return total, motions_arr
