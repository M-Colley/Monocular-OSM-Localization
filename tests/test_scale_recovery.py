"""Tests for anchor-based scale recovery + georeferencing (ideas 1 & 2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.scale_recovery import (
    apply_transform,
    estimate_anchor_scale,
    fit_similarity_ransac,
    scaled_length,
    vo_positions_at_times,
)


def _make_world(vo, scale, deg, t):
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return scale * vo @ R.T + t


# ---------------------------------------------------------------------------
# vo_positions_at_times
# ---------------------------------------------------------------------------


def test_vo_positions_picks_nearest_frame() -> None:
    traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    out = vo_positions_at_times(traj, ts, np.array([0.1, 2.9]))
    assert np.allclose(out, [[0, 0], [3, 0]])


# ---------------------------------------------------------------------------
# estimate_anchor_scale
# ---------------------------------------------------------------------------


def test_estimate_scale_recovers_known_factor() -> None:
    vo = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    world = _make_world(vo, scale=7.0, deg=33, t=np.array([500.0, -200.0]))
    s = estimate_anchor_scale(vo, world)
    assert s == pytest.approx(7.0, rel=1e-6)


def test_estimate_scale_median_resists_one_outlier() -> None:
    # Median-of-pairs is robust once outliers are well under half the
    # pairs: one bad anchor among 6 corrupts 5/15 pairs (33%), so the
    # median ratio stays at the true scale. (With only 4 anchors one
    # outlier corrupts 50% of pairs — use the RANSAC fit there instead.)
    vo = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]], dtype=float)
    world = _make_world(vo, scale=5.0, deg=0, t=np.zeros(2))
    world[5] += np.array([400.0, 400.0])  # corrupt one anchor
    s = estimate_anchor_scale(vo, world)
    assert s == pytest.approx(5.0, rel=0.05)  # median unaffected


def test_estimate_scale_none_when_degenerate() -> None:
    vo = np.array([[0, 0], [0, 0]], dtype=float)  # same point
    world = np.array([[0, 0], [100, 0]], dtype=float)
    assert estimate_anchor_scale(vo, world) is None


# ---------------------------------------------------------------------------
# fit_similarity_ransac
# ---------------------------------------------------------------------------


def test_ransac_recovers_transform_clean() -> None:
    rng = np.random.default_rng(0)
    vo = rng.uniform(-50, 50, size=(8, 2))
    world = _make_world(vo, scale=4.0, deg=25, t=np.array([1000.0, 2000.0]))
    res = fit_similarity_ransac(vo, world, thresh_m=10, min_inliers=3,
                                min_world_baseline_m=10)
    assert res is not None
    assert res.scale == pytest.approx(4.0, rel=1e-3)
    assert len(res.inlier_idx) == 8
    assert res.rms_m < 1.0
    # Applying the transform reproduces world.
    assert np.allclose(apply_transform(vo, res.transform), world, atol=1.0)


def test_ransac_rejects_outlier_anchors() -> None:
    rng = np.random.default_rng(1)
    vo = rng.uniform(-100, 100, size=(7, 2))
    world = _make_world(vo, scale=3.0, deg=10, t=np.array([0.0, 0.0]))
    # Corrupt two anchors badly (bad geocodes).
    world[2] += np.array([3000.0, -2000.0])
    world[5] += np.array([-4000.0, 1500.0])
    res = fit_similarity_ransac(vo, world, thresh_m=50, min_inliers=4,
                                min_world_baseline_m=10)
    assert res is not None
    assert res.scale == pytest.approx(3.0, rel=0.05)
    assert set(res.inlier_idx).isdisjoint({2, 5})  # outliers excluded


def test_ransac_declines_short_baseline() -> None:
    # Three good but tightly-clustered anchors → unreliable scale.
    vo = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    world = _make_world(vo, scale=5.0, deg=0, t=np.array([0.0, 0.0]))
    res = fit_similarity_ransac(vo, world, thresh_m=10, min_inliers=3,
                                min_world_baseline_m=250.0)
    assert res is None  # baseline ~5 m << 250 m


def test_ransac_declines_too_few_anchors() -> None:
    vo = np.array([[0, 0], [10, 0]], dtype=float)
    world = _make_world(vo, scale=2.0, deg=0, t=np.zeros(2))
    assert fit_similarity_ransac(vo, world, min_inliers=3) is None


# ---------------------------------------------------------------------------
# scaled_length
# ---------------------------------------------------------------------------


def test_scaled_length() -> None:
    traj = np.array([[0, 0], [0, 100]], dtype=float)  # 100 VO units
    assert scaled_length(traj, 5.0) == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# da3_metric_scale (idea 4)
# ---------------------------------------------------------------------------


def test_da3_metric_scale_ratio() -> None:
    from src.scale_recovery import da3_metric_scale
    vo = np.array([[0, 0], [0, 100]], dtype=float)        # 100 VO units
    da3 = np.array([[0, 0], [0, 530]], dtype=float)       # 530 m
    assert da3_metric_scale(da3, vo) == pytest.approx(5.3)


def test_da3_metric_scale_none_degenerate() -> None:
    from src.scale_recovery import da3_metric_scale
    vo = np.array([[0, 0], [0, 0]], dtype=float)          # zero length
    da3 = np.array([[0, 0], [0, 100]], dtype=float)
    assert da3_metric_scale(da3, vo) is None
