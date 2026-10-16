"""Road-plane metric step lengths (monocular_osm.plane_speed).

The algebra is checked on exact synthetic correspondences, the whole chain
(corners -> Lucas-Kanade -> RANSAC -> smoothing) on images rendered from a
textured road plane with a known camera height and speed. KITTI accuracy
against OXTS is in docs/graph_filter_kitti.md, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from monocular_osm.plane_speed import (
    _model_for,
    plane_step_lengths,
    solve_plane,
    transfer_error,
)
from monocular_osm.visual_odometry import Trajectory


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _ground_pair(n=200, d=1.5, R=None, t=(0.05, 0.0, -1.0), seed=0):
    """Ground points y = d (camera y axis points down) seen from two poses."""
    rng = np.random.default_rng(seed)
    X1 = np.column_stack([rng.uniform(-4, 4, n), np.full(n, d), rng.uniform(4, 25, n)])
    R = np.eye(3) if R is None else R
    t = np.asarray(t, float)
    X2 = X1 @ R.T + t
    return X1 / X1[:, 2:3], X2 / X2[:, 2:3], R, t


def test_plane_is_recovered_exactly_from_ground_correspondences() -> None:
    x1, x2, R, t = _ground_pair(R=_rot_y(2.0))
    m = solve_plane(x1, x2, R, t)
    assert m == pytest.approx([0.0, 1 / 1.5, 0.0], abs=1e-9)
    assert transfer_error(m, x1, x2, R, t).max() < 1e-9


def test_ransac_model_rejects_off_plane_points() -> None:
    from skimage.measure import ransac

    x1, x2, R, t = _ground_pair(n=150, d=2.0)
    rng = np.random.default_rng(1)
    bad = rng.choice(len(x2), 40, replace=False)
    x2 = x2.copy()
    x2[bad, :2] += rng.normal(0, 0.02, (40, 2))          # a car, a wall, a bad track
    model, inl = ransac((x1, x2), _model_for(R, t), min_samples=3,
                        residual_threshold=1e-3, max_trials=200, rng=rng)
    assert model and not inl[bad].any() and inl.sum() >= 105
    m = solve_plane(x1[inl], x2[inl], R, t)
    assert np.linalg.norm(m) == pytest.approx(0.5, rel=1e-6)


def _render_drive(n_frames=12, step_m=1.0, cam_h=1.5, size=(240, 320), seed=0):
    """Frames of a textured road from a camera ``cam_h`` m up, driving straight
    ahead ``step_m`` metres per frame; poses in VO convention (unit steps)."""
    import cv2

    h, w = size
    K = np.array([[300.0, 0, w / 2], [0, 300.0, 100.0], [0, 0, 1]])
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 1.2)
    d_vo = cam_h / step_m                     # plane distance in VO units (1 unit = 1 step)
    m0 = np.array([0.0, 1.0 / d_vo, 0.0])
    Kinv = np.linalg.inv(K)
    frames, rots, trans, centers = [], [], [], []
    for i in range(n_frames):
        t0i = np.array([0.0, 0.0, -float(i)])
        H = K @ (np.eye(3) + np.outer(t0i, m0)) @ Kinv
        frames.append(cv2.warpPerspective(base, H, (w, h), borderMode=cv2.BORDER_REFLECT))
        rots.append(np.eye(3))
        trans.append(t0i)
        centers.append(-t0i)
    centers = np.array(centers)
    traj = Trajectory(centers=centers, xz=centers[:, [0, 2]], valid=np.ones(n_frames, bool),
                      n_inliers=[100] * n_frames, rotations=np.array(rots),
                      translations=np.array(trans))
    return frames, traj, K


def test_rendered_drive_gives_the_true_speed() -> None:
    frames, traj, K = _render_drive(step_m=1.0, cam_h=1.5)
    ps = plane_step_lengths(frames, traj, K, camera_height_m=1.5, smooth_steps=3)
    assert ps.coverage > 0.8
    assert np.median(ps.lengths_m) == pytest.approx(1.0, rel=0.05)


def test_camera_height_only_scales_the_answer() -> None:
    frames, traj, K = _render_drive(step_m=0.8, cam_h=1.5)
    a = plane_step_lengths(frames, traj, K, camera_height_m=1.5, smooth_steps=3)
    b = plane_step_lengths(frames, traj, K, camera_height_m=1.2, smooth_steps=3)
    assert np.median(a.lengths_m) == pytest.approx(0.8, rel=0.05)
    assert b.lengths_m == pytest.approx(a.lengths_m * 1.2 / 1.5, rel=1e-9)


def test_stationary_and_invalid_steps_stay_zero() -> None:
    frames, traj, K = _render_drive(n_frames=8)
    traj.valid[4] = False                     # VO lost the pair 3->4: pose held
    ps = plane_step_lengths(frames, traj, K, smooth_steps=3)
    assert ps.lengths_m[3] == 0.0 and np.isnan(ps.raw_m[3])
    assert (ps.lengths_m[[0, 1, 5, 6]] > 0).all()
