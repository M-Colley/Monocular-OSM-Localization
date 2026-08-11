"""Tests for per-step speed from road-plane optical flow.

The module exists because the VO's unit-normalised step imposes a
CONSTANT-SPEED assumption on the trajectory shape. So the properties worth
testing are not "is the metre value right" — absolute scale is provably
worth zero to the matcher — but: does it track the relative profile, does it
refuse to answer when the road plane is not visible, and can it ever make
the trajectory worse than leaving it alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from monocular_osm.ground_flow_scale import (
    PlaneCalibration,
    RoadTracks,
    _confidence,
    calibrate_from_flow,
    rescale_steps,
    step_lengths,
)
from monocular_osm.visual_odometry import default_intrinsics

W, H = 960, 540
K = default_intrinsics(W, H)


# ---------------------------------------------------------------------------
# rescale_steps — direction preserved, length replaced
# ---------------------------------------------------------------------------


def test_rescale_keeps_direction_and_applies_length() -> None:
    xz = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])  # unit steps
    out = rescale_steps(xz, np.array([2.0, 5.0, 0.5]))
    steps = np.diff(out, axis=0)
    assert np.linalg.norm(steps, axis=1) == pytest.approx([2.0, 5.0, 0.5])
    # Directions unchanged: each step is parallel to the original.
    for a, b in zip(np.diff(xz, axis=0), steps):
        assert a[0] * b[1] - a[1] * b[0] == pytest.approx(0.0, abs=1e-9)  # parallel
        assert np.dot(a, b) > 0                                           # same way


def test_rescale_leaves_stopped_steps_at_zero() -> None:
    # A held pose (VO found no motion) has a zero step; it must stay zero
    # rather than acquiring a direction out of floating-point noise.
    xz = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = rescale_steps(xz, np.array([3.0, 3.0, 3.0]))
    assert np.linalg.norm(out[2] - out[1]) == pytest.approx(0.0)


def test_rescale_is_a_noop_under_unit_lengths() -> None:
    rng = np.random.default_rng(0)
    steps = rng.normal(size=(30, 2))
    steps /= np.linalg.norm(steps, axis=1)[:, None]
    xz = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    np.testing.assert_allclose(rescale_steps(xz, np.ones(30)), xz, atol=1e-9)


# ---------------------------------------------------------------------------
# abstention — the property that makes this safe to enable
# ---------------------------------------------------------------------------


def _flat_cal(sharpness=1.0):
    return PlaneCalibration(v_horizon=H * 0.5, pitch_deg=0.0,
                            static_mask=np.zeros((18, 24), bool),
                            sharpness=sharpness)


def test_confidence_is_zero_when_points_disagree() -> None:
    # Wide relative spread: the tracked points cannot agree how far the car
    # moved, so the flat-plane model is not describing this footage.
    motion = np.full(60, 2.0)
    spread = np.full(60, 0.9)
    assert _confidence(motion, spread, _flat_cal(), 0.25, 0.55, 0.30, 0.45) == 0.0


def test_confidence_is_one_when_points_agree() -> None:
    motion = np.full(60, 2.0)
    spread = np.full(60, 0.05)
    assert _confidence(motion, spread, _flat_cal(), 0.25, 0.55, 0.30, 0.45) == 1.0


def test_confidence_is_zero_when_the_horizon_was_never_pinned_down() -> None:
    # Tidy residuals do not rescue a meaningless metric conversion.
    motion = np.full(60, 2.0)
    spread = np.full(60, 0.05)
    assert _confidence(motion, spread, _flat_cal(sharpness=0.0),
                       0.25, 0.55, 0.30, 0.45) == 0.0


def test_no_tracks_returns_the_fallback_untouched() -> None:
    empty = RoadTracks(np.zeros((0, 4), np.float32), np.array([0]), 30.0, (W, H))
    moving = np.ones(10, bool)
    fallback = np.full(10, 3.0)
    lengths, conf = step_lengths(empty, K, step_directions_valid=moving,
                                 fallback_lengths=fallback)
    assert conf == 0.0
    np.testing.assert_allclose(lengths, fallback)


def test_abstaining_reproduces_the_fallback_exactly() -> None:
    """The whole safety argument: a clip the model cannot describe must come
    back byte-for-byte as the trajectory the VO already produced, so enabling
    this can never cost anything on such a clip."""
    from monocular_osm.ground_flow_scale import _blend

    moving = np.ones(20, bool)
    fallback = np.linspace(1.0, 4.0, 20)
    nonsense = np.full(20, 99.0)
    np.testing.assert_allclose(_blend(nonsense, fallback, 0.0, moving), fallback)


def test_blend_preserves_total_distance() -> None:
    # Partial confidence must change the SHAPE of the profile, not the route
    # length — otherwise it would fight the scale-lock machinery.
    from monocular_osm.ground_flow_scale import _blend

    moving = np.ones(50, bool)
    fallback = np.full(50, 2.0)
    estimate = np.abs(np.random.default_rng(1).normal(5.0, 2.0, 50))
    for conf in (0.25, 0.5, 0.75):
        out = _blend(estimate, fallback, conf, moving)
        assert out.sum() == pytest.approx(fallback.sum(), rel=1e-9)


# ---------------------------------------------------------------------------
# horizon self-calibration
# ---------------------------------------------------------------------------


def _synthetic_tracks(v_horizon: float, travel_per_step=2.0, n_steps=60,
                      n_pts=80, seed=0) -> RoadTracks:
    """Points on a flat road plane, seen before and after a forward move.

    Depth from image row is Z = c / (v - v_h); moving forward by d takes a
    point from Z to Z - d, which is the row v' satisfying c/(v'-v_h) = Z - d.
    """
    rng = np.random.default_rng(seed)
    c = 1000.0
    rows, offs = [], [0]
    for _ in range(n_steps):
        v0 = rng.uniform(v_horizon + 40.0, H * 0.95, n_pts)
        u0 = rng.uniform(W * 0.2, W * 0.8, n_pts)
        z0 = c / (v0 - v_horizon)
        z1 = np.maximum(z0 - travel_per_step, 0.5)
        v1 = c / z1 + v_horizon
        # u drifts outward as the point approaches, as perspective requires
        u1 = W / 2 + (u0 - W / 2) * (z0 / z1)
        rows.append(np.column_stack([u0, v0, u1, v1]).astype(np.float32))
        offs.append(offs[-1] + n_pts)
    return RoadTracks(np.concatenate(rows), np.asarray(offs, np.int64),
                      30.0, (W, H), stride=6)


@pytest.mark.parametrize("true_vh", [220.0, 270.0, 320.0])
def test_calibrate_recovers_the_horizon_row(true_vh) -> None:
    tracks = _synthetic_tracks(true_vh)
    cal = calibrate_from_flow(tracks, float(K[0, 0]), float(K[1, 2]))
    assert cal.v_horizon == pytest.approx(true_vh, abs=8.0)
    assert cal.sharpness > 0.0


def test_calibrated_pitch_is_consistent_with_the_horizon() -> None:
    tracks = _synthetic_tracks(240.0)
    cal = calibrate_from_flow(tracks, float(K[0, 0]), float(K[1, 2]))
    expected = np.degrees(np.arctan((float(K[1, 2]) - cal.v_horizon) / float(K[0, 0])))
    assert cal.pitch_deg == pytest.approx(expected, abs=1e-6)


def test_recovered_profile_tracks_relative_speed() -> None:
    """The thing that actually matters: does a faster step come out longer?

    Absolute metres are irrelevant (the matcher's similarity fit absorbs any
    global factor), so this asserts CORRELATION with the truth, not equality.
    """
    rng = np.random.default_rng(5)
    true = np.abs(rng.normal(2.0, 1.0, 60)) + 0.2
    c, v_h = 1000.0, 250.0
    rows, offs = [], [0]
    for d in true:
        v0 = rng.uniform(v_h + 40.0, H * 0.95, 80)
        u0 = rng.uniform(W * 0.2, W * 0.8, 80)
        z0 = c / (v0 - v_h)
        z1 = np.maximum(z0 - d, 0.5)
        v1 = c / z1 + v_h
        u1 = W / 2 + (u0 - W / 2) * (z0 / z1)
        rows.append(np.column_stack([u0, v0, u1, v1]).astype(np.float32))
        offs.append(offs[-1] + 80)
    tracks = RoadTracks(np.concatenate(rows), np.asarray(offs, np.int64),
                        30.0, (W, H), stride=6)
    lengths, conf = step_lengths(tracks, K, step_directions_valid=np.ones(60, bool),
                                 near_m=0.0, far_m=1e6, lat_m=1e6)
    assert conf > 0.0
    assert np.corrcoef(lengths, true)[0, 1] > 0.9
