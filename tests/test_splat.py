"""Tests for the sparse-splat reconstruction module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.splat import (
    build_splat_points,
    render_topdown_splat,
    save_ply,
)
from src.visual_odometry import Trajectory, default_intrinsics


def _make_textured_scene(seed: int = 0, n: int = 600):
    """3-D points roughly 8–14 m in front of camera-0, plus per-point colors."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=n)
    Y = rng.uniform(-3, 3, size=n)
    Z = rng.uniform(8, 14, size=n)
    pts = np.stack([X, Y, Z], axis=1)
    colors = rng.integers(50, 256, size=(n, 3), dtype=np.uint8)
    return pts, colors


def _project_into(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    cam = pts3d @ R.T + t
    z = cam[:, 2]
    mask = z > 0.1
    cam = cam[mask]
    pix = cam @ K.T
    pix = pix[:, :2] / pix[:, 2:3]
    return pix, mask, cam[:, 2]


def _render(pts3d, colors, K, R, t, w, h):
    pix, mask, _ = _project_into(pts3d, K, R, t)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cs = colors[mask]
    for (x, y), c in zip(pix, cs):
        ix, iy = int(round(x)), int(round(y))
        if 2 <= ix < w - 2 and 2 <= iy < h - 2:
            cv2.circle(img, (ix, iy), 2, c.tolist(), -1)
    return img


def _trajectory_from_world_to_cam(centers, rotations_w2c, translations_w2c):
    n = len(centers)
    return Trajectory(
        centers=np.asarray(centers),
        xz=np.asarray(centers)[:, [0, 2]],
        valid=np.ones(n, dtype=bool),
        n_inliers=[100] * n,
        rotations=np.asarray(rotations_w2c),
        translations=np.asarray(translations_w2c),
    )


def test_build_splat_recovers_3d_geometry() -> None:
    """Two frames with known poses → triangulated 3-D points should match
    the synthetic ground-truth distribution (within reason — ORB drops a
    lot of points and triangulation is noisy)."""
    w, h = 640, 480
    K = default_intrinsics(w, h)
    pts3d_gt, colors_gt = _make_textured_scene(seed=11, n=800)

    R0 = np.eye(3); t0 = np.zeros(3)
    R1 = np.eye(3); t1 = np.array([0.0, 0.0, -1.0])  # camera moves +z forward

    img0 = _render(pts3d_gt, colors_gt, K, R0, t0, w, h)
    img1 = _render(pts3d_gt, colors_gt, K, R1, t1, w, h)

    centers = [
        -R0.T @ t0,
        -R1.T @ t1,
    ]
    traj = _trajectory_from_world_to_cam(centers, [R0, R1], [t0, t1])

    pts, cols = build_splat_points(
        [img0, img1], traj, K, baseline_frames=1, max_pairs=1
    )
    assert len(pts) > 20, f"expected non-trivial reconstruction, got {len(pts)}"
    assert pts.shape[1] == 3
    assert cols.shape[1] == 3 and cols.dtype == np.uint8

    # Recovered points should land in the same z range as the synthetic scene.
    z = pts[:, 2]
    mid = float(np.median(z))
    assert 6 < mid < 16, f"median triangulated z ({mid:.1f}) is far from 8-14 ground truth"


def test_render_topdown_splat_produces_image() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(200, 3))
    cols = rng.integers(0, 256, size=(200, 3), dtype=np.uint8)

    img = render_topdown_splat(pts, cols, resolution=256)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    # Not entirely background.
    assert img.sum() > 0


def test_render_topdown_splat_handles_empty_input() -> None:
    img = render_topdown_splat(
        np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), resolution=128
    )
    assert img.shape == (128, 128, 3)
    # All-background.
    assert img.sum() == 0


def test_save_ply_roundtrips_via_open3d(tmp_path: Path) -> None:
    """Round-trip points through Open3D's PLY writer/reader to confirm
    we're producing a file the standard library can load."""
    import open3d as o3d

    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, 0.0, 1.5]])
    cols = np.array([[10, 20, 30], [255, 0, 0], [128, 128, 128]], dtype=np.uint8)
    out = tmp_path / "test.ply"
    save_ply(pts, cols, out)

    pcd = o3d.io.read_point_cloud(str(out))
    loaded_pts = np.asarray(pcd.points)
    loaded_cols_f = np.asarray(pcd.colors)
    assert loaded_pts.shape == (3, 3)
    assert loaded_cols_f.shape == (3, 3)
    # Colors come back as floats in [0, 1]; check ratios match input.
    expected_f = cols.astype(float) / 255.0
    assert np.allclose(loaded_cols_f, expected_f, atol=1.5 / 255)


def test_save_interactive_html_writes_file(tmp_path: Path) -> None:
    from src.splat import save_interactive_html

    pts = np.random.default_rng(0).uniform(-5, 5, size=(50, 3))
    cols = np.random.default_rng(0).integers(0, 256, size=(50, 3), dtype=np.uint8)
    out = tmp_path / "splat.html"
    save_interactive_html(pts, cols, out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # Plotly emits a <div id="..."> with the figure JSON in a script tag.
    assert "plotly" in text.lower()
    assert "Scatter3d" in text or "scatter3d" in text
