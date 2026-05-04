"""IPM unit tests — pure geometry, no dependencies on heavy models."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ipm import (
    IPMCalibration,
    compute_ipm_homography,
    render_ipm_canvas,
    stitch_bev_along_trajectory,
    warp_to_bev,
)
from src.visual_odometry import default_intrinsics


def test_compute_homography_returns_valid_3x3() -> None:
    K = default_intrinsics(1280, 720, hfov_deg=70.0)
    cal = IPMCalibration(K=K, camera_height_m=1.4, pitch_deg=6.0,
                         bev_width_m=20.0, bev_depth_m=30.0,
                         bev_resolution_pix_per_m=8.0)
    H, (h, w) = compute_ipm_homography(cal)
    assert H.shape == (3, 3)
    assert h == 240 and w == 160
    # Non-singular.
    assert abs(np.linalg.det(H)) > 1e-9


def test_warp_runs_on_synthetic_image() -> None:
    K = default_intrinsics(1280, 720)
    cal = IPMCalibration(K=K)
    H, bev_size = compute_ipm_homography(cal)

    img = np.full((720, 1280, 3), 64, dtype=np.uint8)
    # Paint a horizontal "road line" at the y where the road plane projects.
    cv2.line(img, (0, 540), (1280, 540), (255, 255, 255), 4)

    bev = warp_to_bev(img, H, bev_size)
    assert bev.shape == (bev_size[0], bev_size[1], 3)
    # Most of the BEV should be non-empty.
    assert (bev.sum(axis=2) > 0).mean() > 0.3


def test_stitch_bev_returns_canvas() -> None:
    """Stitch 5 synthetic frames along a small trajectory."""
    K = default_intrinsics(1280, 720)
    cal = IPMCalibration(K=K, bev_width_m=20.0, bev_depth_m=20.0,
                         bev_resolution_pix_per_m=4.0)

    # Synthetic frames: each a different solid color, bordered.
    frames = []
    for i in range(5):
        f = np.full((720, 1280, 3), [40 + i * 30, 60, 100], dtype=np.uint8)
        cv2.rectangle(f, (10, 10), (1270, 710), (255, 255, 255), 8)
        frames.append(f)

    # Trajectory: car drives along +y for a while.
    traj = np.array([[0.0, float(i)] for i in range(5)])
    canvas = stitch_bev_along_trajectory(
        frames, traj, cal, keyframe_stride=1, canvas_resolution_pix_per_m=2.0,
    )
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    assert (canvas.sum(axis=2) > 0).mean() > 0.05


def test_stitch_bev_handles_helper_wrapper() -> None:
    K = default_intrinsics(1280, 720)
    frames = [
        np.full((720, 1280, 3), 80, dtype=np.uint8) for _ in range(4)
    ]
    traj = np.array([[0.0, float(i) * 0.5] for i in range(4)])
    canvas = render_ipm_canvas(frames, traj, K, keyframe_stride=1)
    assert canvas.ndim == 3
    # Helper succeeds with default calibration.


def test_stitch_bev_rejects_mismatched_lengths() -> None:
    K = default_intrinsics(1280, 720)
    cal = IPMCalibration(K=K)
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    traj = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        stitch_bev_along_trajectory(frames, traj, cal)
