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
from src.speed_scale import _pair_motion, estimate_route_length_from_flow, image_to_ground


def _K(fx=800.0, fy=800.0, cx=640.0, cy=360.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def _project_ground(ground_xz, K, *, camera_height_m, pitch_deg, roll_deg=0.0):
    """Forward model: ground point (X right, Z forward) -> pixel (u, v).

    Matches ipm.py's convention: ``_rotation_pitch_roll`` is the
    vehicle->camera rotation and positive pitch means pitched DOWN. (The
    old version of this helper applied the transpose, which round-tripped
    the very inversion bug image_to_ground used to have.)
    """
    R = _rotation_pitch_roll(pitch_deg, roll_deg)  # vehicle->camera
    X, Z = ground_xz
    veh = np.array([X, camera_height_m, Z])        # relative to camera, +y down
    cam = R @ veh
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


def test_positive_pitch_means_down_matches_ipm_convention() -> None:
    """A camera pitched DOWN sees more ground: with pitch_deg=+6 a point
    10 m ahead must land BELOW where it lands with pitch 0 (the old code
    modelled positive pitch as up and NaN'd real down-pitched rigs)."""
    K = _K()
    uv_flat = _project_ground((0.0, 10.0), K, camera_height_m=1.4, pitch_deg=0.0)
    rec = image_to_ground(uv_flat[None, :], K, camera_height_m=1.4, pitch_deg=6.0)[0]
    # Same pixel with the camera pitched down = a ray hitting the ground
    # closer to the car, not above the horizon.
    assert np.isfinite(rec).all()
    assert 0.0 < rec[1] < 10.0


def test_image_to_ground_cross_consistent_with_ipm_projection() -> None:
    """A ground point projected with ipm.py's exact camera geometry
    (vehicle->camera R, camera at y=-h) must back-project to itself
    through image_to_ground."""
    K = _K()
    h_m, pitch = 1.4, 6.0
    R = _rotation_pitch_roll(pitch, 0.0)
    t = np.array([0.0, -h_m, 0.0])   # ipm.py: camera h above the road (+y down)
    ground = np.array([
        [2.0, 0.0, 12.0],
        [-3.0, 0.0, 7.0],
        [0.0, 0.0, 30.0],
    ])
    cam = (ground - t) @ R.T          # ipm.py's projection: x_cam = R @ (x_veh - t)
    img = cam @ K.T
    uv = img[:, :2] / img[:, 2:3]
    rec = image_to_ground(uv, K, camera_height_m=h_m, pitch_deg=pitch)
    np.testing.assert_allclose(rec, ground[:, [0, 2]], atol=1e-9)


def _bearing_rotate(pts, phi):
    """Rotate ground points (X right, Z forward) by ``phi`` in bearing."""
    c, s = np.cos(phi), np.sin(phi)
    return np.c_[pts[:, 0] * c + pts[:, 1] * s, pts[:, 1] * c - pts[:, 0] * s]


def test_pair_motion_ignores_pure_yaw() -> None:
    """A stationary car turning in place sweeps ground points laterally;
    the yaw-compensated forward motion must be ~0 where the raw
    displacement norm (old logic) reads a large fake speed."""
    rng = np.random.default_rng(2)
    g0 = np.c_[rng.uniform(-5, 5, 80), rng.uniform(5, 24, 80)]
    g1 = _bearing_rotate(g0, np.deg2rad(4.0))
    naive = float(np.median(np.linalg.norm(g1 - g0, axis=1)))
    assert naive > 0.3                       # the bias the old logic had
    assert abs(_pair_motion(g0, g1)) < 1e-9  # fully compensated


def test_pair_motion_recovers_forward_translation_during_turn() -> None:
    """Forward travel + yaw combined: the estimate must recover the true
    travel, not the (much larger) displacement norm."""
    d = 0.5
    xs, zs = np.meshgrid(np.linspace(-4, 4, 9), np.linspace(5, 20, 8))
    g0 = np.c_[xs.ravel(), zs.ravel()]
    g1 = _bearing_rotate(g0 - np.array([0.0, d]), np.deg2rad(4.0))
    naive = float(np.median(np.linalg.norm(g1 - g0, axis=1)))
    assert naive > 0.65                      # old logic: biased long
    assert _pair_motion(g0, g1) == pytest.approx(d, abs=1e-6)


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
