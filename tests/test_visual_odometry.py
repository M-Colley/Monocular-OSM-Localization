"""Tests for visual_odometry helpers and synthetic VO sanity checks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.visual_odometry import (
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
