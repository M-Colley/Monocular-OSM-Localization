"""Tests for visual_odometry helpers and synthetic VO sanity checks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.visual_odometry import (
    _fit_plane_projection,
    bearing_signature,
    default_intrinsics,
    estimate_trajectory,
    resample_uniform,
    trajectory_arc_length,
)


def test_default_intrinsics_basic() -> None:
    K = default_intrinsics(1280, 720, hfov_deg=70.0)
    assert K.shape == (3, 3)
    # Principal point at image center.
    assert K[0, 2] == pytest.approx(640.0)
    assert K[1, 2] == pytest.approx(360.0)
    # fx > 0, square pixels.
    assert K[0, 0] > 0
    assert K[0, 0] == pytest.approx(K[1, 1])


def test_arc_length_monotone() -> None:
    pts = np.array([[0, 0], [3, 4], [3, 8]])  # legs of 5 and 4
    s = trajectory_arc_length(pts)
    assert s.tolist() == pytest.approx([0, 5, 9])


def test_resample_uniform_endpoints() -> None:
    pts = np.array([[0, 0], [10, 0], [10, 10]])
    r = resample_uniform(pts, n_samples=11)
    assert len(r) == 11
    assert r[0].tolist() == pytest.approx([0, 0])
    assert r[-1].tolist() == pytest.approx([10, 10])


def test_bearing_signature_invariant_to_translation_and_rotation() -> None:
    # An L-shape: go east 10, then north 10.
    pts = np.array([[i, 0] for i in range(11)] + [[10, j] for j in range(1, 11)])
    sig = bearing_signature(pts, n_samples=64)

    # Translate by some offset.
    sig_trans = bearing_signature(pts + np.array([100.0, -50.0]), n_samples=64)
    assert np.allclose(sig, sig_trans, atol=1e-6)

    # Rotate by 30 degrees about origin.
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    sig_rot = bearing_signature(pts @ R.T, n_samples=64)
    assert np.allclose(sig, sig_rot, atol=1e-6)

    # Uniform scale (5x).
    sig_scaled = bearing_signature(pts * 5, n_samples=64)
    assert np.allclose(sig, sig_scaled, atol=1e-6)


def test_bearing_signature_distinguishes_left_vs_right_turn() -> None:
    left_turn = np.array([[i, 0] for i in range(11)] + [[10, j] for j in range(1, 11)])
    right_turn = np.array([[i, 0] for i in range(11)] + [[10, -j] for j in range(1, 11)])
    a = bearing_signature(left_turn, n_samples=64)
    b = bearing_signature(right_turn, n_samples=64)
    # The signed turn should be opposite-sign in the two halves.
    assert np.sign(a.sum()) == -np.sign(b.sum())


def _l_turn_centers(rng: np.random.Generator) -> np.ndarray:
    """3D camera centers for an L-turn: +x then +z, tiny vertical noise."""
    straight = np.c_[np.arange(20.0), np.zeros(20), np.zeros(20)]
    turn = np.c_[np.full(15, 19.0), np.zeros(15), np.arange(1.0, 16.0)]
    centers = np.vstack([straight, turn])
    centers[:, 1] += rng.normal(size=len(centers)) * 0.01
    return centers


def _turn_sign(xy: np.ndarray) -> float:
    """Sign of the cross product between the two legs of an L-path."""
    d1 = xy[8] - xy[0]
    d2 = xy[-1] - xy[-9]
    return float(np.sign(d1[0] * d2[1] - d1[1] * d2[0]))


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_fit_plane_projection_preserves_turn_sign_under_yaw() -> None:
    """The PCA plane projection must never MIRROR the path: on the old
    arbitrary-handedness SVD roughly half of these seeds came back with
    the L-turn's sign flipped (left turn became a right turn)."""
    centers = _l_turn_centers(np.random.default_rng(0))
    ref_sign = _turn_sign(centers[:, [0, 2]])  # raw top-down drop
    assert ref_sign != 0.0
    for seed in range(20):
        a = np.random.default_rng(seed).uniform(0.0, 2 * np.pi)
        projected = _fit_plane_projection(centers @ _rot_y(a).T)
        assert _turn_sign(projected) == ref_sign, f"mirrored at seed {seed}"


def test_fit_plane_projection_up_hint_handles_full_so3() -> None:
    """With the camera-up hint (what estimate_trajectory passes), the turn
    sign survives arbitrary SO(3) rotations of the world frame."""
    centers = _l_turn_centers(np.random.default_rng(1))
    ref_sign = _turn_sign(centers[:, [0, 2]])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        up = Q @ np.array([0.0, -1.0, 0.0])
        projected = _fit_plane_projection(centers @ Q.T, up=up)
        assert _turn_sign(projected) == ref_sign, f"mirrored at seed {seed}"


def test_fit_plane_projection_discards_axis_nearest_vertical() -> None:
    """Accumulated pitch drift can give the vertical axis more variance
    than the lateral one; the kept plane must still be the ground plane
    (drop the axis nearest camera-y), not (forward, vertical-drift)."""
    # Mostly-straight drive along x with a gentle lateral (z) wiggle of
    # amplitude ~1 and a large vertical (y) drift bow of amplitude 8.
    # Orthogonalize the three signals so the principal axes are exactly
    # the coordinate axes and the variance ordering is unambiguous:
    # var(x) >> var(y drift) > var(z wiggle).
    x = np.arange(0.0, 60.0)
    xc = x - x.mean()
    y = -np.sin(x / 60.0 * np.pi) * 8.0
    y = y - y.mean()
    y -= (y @ xc) / (xc @ xc) * xc
    z = np.cos(x / 60.0 * 2 * np.pi) * 1.0
    z = z - z.mean()
    z -= (z @ xc) / (xc @ xc) * xc
    z -= (z @ y) / (y @ y) * y
    centers = np.c_[x, y, z]
    projected = _fit_plane_projection(centers)
    # The lateral wiggle must survive the projection: correlate the
    # projected minor axis with the true z, not with the vertical drift.
    minor = projected[:, 1] - projected[:, 1].mean()
    corr_z = abs(np.corrcoef(minor, z)[0, 1])
    corr_y = abs(np.corrcoef(minor, y)[0, 1])
    assert corr_z > 0.99 and corr_y < 0.1


def _render_textured_scene(
    width: int = 640,
    height: int = 480,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 3D point cloud and a corresponding base image to project from.

    The scene is a textured plane at z = 5 (in front of the camera). Each
    "world point" is associated with a colored speck — when we re-project
    from a moved camera we get a new image with the specks in their new
    pixel locations, which is what ORB-based VO needs to recover motion.
    """
    rng = np.random.default_rng(seed)
    n = 500
    # Spread points in a roughly square 10m x 10m region at z=5.
    X = rng.uniform(-5, 5, size=n)
    Y = rng.uniform(-3, 3, size=n)
    Z = rng.uniform(8, 14, size=n)
    pts3d = np.stack([X, Y, Z], axis=1)
    colors = rng.integers(50, 256, size=(n, 3), dtype=np.uint8)
    return pts3d, colors


def _project(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project Nx3 world points into pixel coordinates for camera with
    world-to-camera (R, t)."""
    cam = pts3d @ R.T + t
    z = cam[:, 2]
    valid = z > 0.1
    cam = cam[valid]
    z = z[valid]
    pix = (cam @ K.T)
    pix = pix[:, :2] / pix[:, 2:3]
    return pix, valid


def _render_view(
    pts3d: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    pix, valid = _project(pts3d, K, R, t)
    cols = colors[valid]
    for (x, y), c in zip(pix, cols):
        ix, iy = int(round(x)), int(round(y))
        if 2 <= ix < w - 2 and 2 <= iy < h - 2:
            cv2.circle(img, (ix, iy), 2, c.tolist(), -1)
    return img


def test_visual_odometry_recovers_forward_motion_direction() -> None:
    """Two frames, camera moves forward 1 unit between them. The recovered
    translation direction (after recoverPose normalizes) should point along
    +Z."""
    w, h = 640, 480
    K = default_intrinsics(w, h)
    pts3d, colors = _render_textured_scene(w, h, seed=7)

    R0 = np.eye(3)
    t0 = np.zeros(3)
    R1 = np.eye(3)
    t1 = np.array([0.0, 0.0, -1.0])  # moving camera +z forward → world point's
                                     # z in camera coords decreases by 1

    img0 = _render_view(pts3d, colors, K, R0, t0, w, h)
    img1 = _render_view(pts3d, colors, K, R1, t1, w, h)

    traj = estimate_trajectory([img0, img1], K, enforce_planar=False)
    assert traj.valid[1], "VO failed on synthetic forward translation"
    # Camera 1 should be displaced from camera 0 along world +z.
    delta = traj.centers[1] - traj.centers[0]
    assert np.linalg.norm(delta) == pytest.approx(1.0, rel=0.05)
    # And the dominant component should be along z.
    assert abs(delta[2]) > 0.85, f"expected forward motion, got {delta}"


def test_stationary_frames_inject_zero_arc_length() -> None:
    """A stopped car (red light) must contribute ZERO steps.

    The scene is static except for a small moving patch (a pedestrian
    crossing) — enough coherent flow for recoverPose to clear its inlier
    gate, so the old code normalized the spurious translation to a full
    unit step per pair. The median matched-keypoint displacement is ~0
    (the static background dominates), which is what the stationary
    guard keys on."""
    w, h = 640, 480
    K = default_intrinsics(w, h)
    rng = np.random.default_rng(3)
    tex = rng.integers(0, 255, (h // 2, w // 2, 3)).astype(np.uint8)
    base = cv2.resize(tex, (w, h), interpolation=cv2.INTER_LINEAR)
    patch = np.random.default_rng(5).integers(0, 255, (80, 80, 3)).astype(np.uint8)

    frames = []
    for i in range(6):
        f = base.copy()
        y = 100 + 25 * i
        f[y:y + 80, 120:200] = patch
        frames.append(f)

    traj = estimate_trajectory(frames, K, enforce_planar=False)
    s = trajectory_arc_length(traj.xz)
    assert s[-1] == pytest.approx(0.0, abs=1e-9)
    # Stationary pairs are flagged invalid (pose held), not fake motion.
    assert not traj.valid[1:].any()


def test_visual_odometry_parallel_matches_sequential() -> None:
    """The parallel-VO path must produce identical trajectory shape and
    valid-mask to the sequential path on the same input frames.

    Pose chaining is deterministic and only fans out per-pair computation
    across threads, so the output of n_workers=1 and n_workers=8 should
    be byte-for-byte equal.
    """
    w, h = 320, 240
    K = default_intrinsics(w, h)
    pts3d, colors = _render_textured_scene(w, h, seed=3)

    # Build a short walking-camera sequence so we have multiple consecutive pairs.
    frames = []
    for i in range(6):
        R = np.eye(3)
        t = np.array([0.0, 0.0, -float(i) * 0.5])
        frames.append(_render_view(pts3d, colors, K, R, t, w, h))

    seq = estimate_trajectory(frames, K, enforce_planar=False, n_workers=1)
    par = estimate_trajectory(frames, K, enforce_planar=False, n_workers=8)

    assert seq.valid.tolist() == par.valid.tolist()
    np.testing.assert_allclose(seq.centers, par.centers, atol=1e-9)
    np.testing.assert_allclose(seq.rotations, par.rotations, atol=1e-9)
    np.testing.assert_allclose(seq.translations, par.translations, atol=1e-9)
