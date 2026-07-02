"""Tests for the VPR-prior study's measurement logic
(scripts/test_vpr_prior_accuracy.py).

The old script measured 'distance to GT route' as distance to the nearest
SPARSE WAYPOINT — up to ~250 m of quantization noise on the Ulm GT file,
the same magnitude as the model/aggregation differences being compared.
These tests pin the corrected polyline metric and the production-mirroring
query sampling.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "vpr_prior_accuracy_under_test",
        ROOT / "scripts" / "test_vpr_prior_accuracy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


# ---------------------------------------------------------------------------
# route_polyline_dist_m — point-to-SEGMENT, not point-to-nearest-waypoint
# ---------------------------------------------------------------------------


def test_on_route_point_between_sparse_waypoints_is_near_zero() -> None:
    # Two waypoints ~1.1 km apart (Ulm-scale sparsity). A prior sitting
    # exactly ON the route midway between them must read ~0 m — the old
    # nearest-waypoint metric reported ~556 m for this point.
    wps = [{"lat": 48.400, "lon": 9.980}, {"lat": 48.410, "lon": 9.980}]
    d = M.route_polyline_dist_m(48.405, 9.980, wps)
    assert d < 1.0


def test_off_route_point_measures_perpendicular_distance() -> None:
    wps = [{"lat": 48.400, "lon": 9.980}, {"lat": 48.410, "lon": 9.980}]
    # 0.001 deg of longitude east of the segment's midpoint.
    expected = 0.001 * 111320.0 * math.cos(math.radians(48.405))
    d = M.route_polyline_dist_m(48.405, 9.981, wps)
    assert d == pytest.approx(expected, rel=0.02)


def test_point_beyond_endpoint_clamps_to_endpoint() -> None:
    wps = [{"lat": 48.400, "lon": 9.980}, {"lat": 48.410, "lon": 9.980}]
    d = M.route_polyline_dist_m(48.412, 9.980, wps)
    assert d == pytest.approx(0.002 * 111320.0, rel=0.02)


def test_polyline_uses_all_consecutive_segments() -> None:
    # An L-shaped route: the point sits on the second leg.
    wps = [{"lat": 48.400, "lon": 9.980},
           {"lat": 48.410, "lon": 9.980},
           {"lat": 48.410, "lon": 9.995}]
    assert M.route_polyline_dist_m(48.410, 9.990, wps) < 1.0


# ---------------------------------------------------------------------------
# query sampling — mirror the production vo-segment window
# ---------------------------------------------------------------------------


def test_parse_segment() -> None:
    assert M._parse_segment("0:420") == (0.0, 420.0)
    assert M._parse_segment("30:") == (30.0, None)
    assert M._parse_segment(None) == (0.0, None)


def test_query_times_stay_inside_segment() -> None:
    # Production samples n_query frames from the ANALYZED segment only;
    # the old script spread queries across the whole video (0.02..0.98).
    times = M._query_times("0:420", duration_sec=600.0, n_query=40)
    assert len(times) == 40
    assert times.min() >= 0.0
    assert times.max() <= 420.0


def test_query_times_clamp_open_end_to_duration() -> None:
    times = M._query_times("30:", duration_sec=100.0, n_query=10)
    assert times.min() >= 30.0
    assert times.max() <= 100.0


def test_n_query_matches_production_default() -> None:
    # kartaview_vpr_prior samples n_query=40 frames; the study must
    # measure the same prior the pipeline computes.
    assert M.N_QUERY == 40
