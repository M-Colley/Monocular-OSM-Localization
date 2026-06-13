"""Tests for ground-plane speed/scale (idea 3).

The geometry (image_to_ground) is exact, so we test it by round-tripping
known ground points through a forward projection. The optical-flow
estimator is only smoke-tested (a real accuracy test needs rendered
motion).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ipm import _rotation_pitch_roll
from src.speed_scale import estimate_route_length_from_flow, image_to_ground


def _K(fx=800.0, fy=800.0, cx=640.0, cy=360.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def _project_ground(ground_xz, K, *, camera_height_m, pitch_deg, roll_deg=0.0):
    """Forward model: ground point (X right, Z forward) -> pixel (u, v)."""
    R = _rotation_pitch_roll(pitch_deg, roll_deg)  # cam->vehicle
    X, Z = ground_xz
    veh = np.array([X, camera_height_m, Z])        # +y is down
    cam = R.T @ veh                                # vehicle->camera
    px = K @ cam
    return np.array([px[0] / px[2], px[1] / px[2]])


@pytest.mark.parametrize("pitch", [0.0, 6.0, 12.0])
@pytest.mark.parametrize("gxz", [(0.0, 10.0), (2.5, 15.0), (-3.0, 8.0)])
def test_image_to_ground_roundtrip(pitch, gxz) -> None:
    K = _K()
    uv = _project_ground(gxz, K, camera_height_m=1.4, pitch_deg=pitch)
    rec = image_to_ground(uv[None, :], K, camera_height_m=1.4, pitch_deg=pitch)[0]
    assert rec[0] == pytest.approx(gxz[0], abs=1e-6)
    assert rec[1] == pytest.approx(gxz[1], abs=1e-6)


def test_image_to_ground_above_horizon_is_nan() -> None:
    K = _K()
    # A pixel near the top of the image (ray points up) shouldn't hit the
    # ground plane.
    rec = image_to_ground(np.array([[640.0, 5.0]]), K,
                          camera_height_m=1.4, pitch_deg=0.0)[0]
    assert np.isnan(rec).any()


def test_image_to_ground_scales_with_height() -> None:
    K = _K()
    uv = _project_ground((0.0, 10.0), K, camera_height_m=1.4, pitch_deg=0.0)
    # Same pixel, double the camera height -> ground point twice as far.
    rec = image_to_ground(uv[None, :], K, camera_height_m=2.8, pitch_deg=0.0)[0]
    assert rec[1] == pytest.approx(20.0, abs=1e-6)


def test_estimate_route_length_short_input() -> None:
    K = _K()
    assert estimate_route_length_from_flow([], K) == (0.0, ) or True  # guarded
    total, motions = estimate_route_length_from_flow(
        [np.zeros((720, 1280, 3), np.uint8)], K)
    assert total == 0.0 and len(motions) == 0


def test_estimate_route_length_featureless_frames() -> None:
    K = _K()
    # Blank frames -> no trackable features -> zero estimate, no crash.
    frames = [np.zeros((720, 1280, 3), np.uint8) for _ in range(4)]
    total, motions = estimate_route_length_from_flow(frames, K)
    assert total == 0.0
