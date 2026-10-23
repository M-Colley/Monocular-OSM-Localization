"""Tests for monocular_osm.geo (pyproj-backed WGS84 geodesy)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from monocular_osm.geo import (
    LocalFrame,
    degree_half_extent,
    geodesic_distance_m,
    metres_per_degree,
)

ULM = (48.3984, 9.9916)


def _haversine(lat1, lon1, lat2, lon2, r=6_371_000.0):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# geodesic_distance_m
# ---------------------------------------------------------------------------


def test_geodesic_known_values() -> None:
    # One degree of longitude on the equator = a * pi / 180 (WGS84 a).
    assert geodesic_distance_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_319.4908, abs=1e-3)
    # One degree of latitude at the equator (meridian arc 0..1 deg).
    assert geodesic_distance_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(110_574.39, abs=0.1)
    assert geodesic_distance_m(*ULM, *ULM) == 0.0


def test_geodesic_close_to_haversine() -> None:
    # The replaced spherical haversine was 0.1-0.5 % off the ellipsoid.
    pairs = [(48.40, 9.99, 48.41, 10.00), (51.5, -0.13, 51.52, -0.10),
             (36.7, -4.4, 36.6, -4.5), (60.0, 0.0, 60.0, 0.05)]
    for p in pairs:
        g, h = geodesic_distance_m(*p), _haversine(*p)
        assert abs(g - h) / h < 0.005


def test_geodesic_scalar_returns_float_and_arrays_broadcast() -> None:
    d = geodesic_distance_m(48.4, 9.99, 48.41, 9.99)
    assert isinstance(d, float)
    lats = np.array([48.40, 48.41, 48.42])
    out = geodesic_distance_m(48.40, 9.99, lats, 9.99)       # broadcast
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(out) > 0)
    grid = geodesic_distance_m(lats[:, None], 9.99, 48.40, np.array([9.99, 10.0]))
    assert grid.shape == (3, 2)
    assert geodesic_distance_m([], [], [], []).shape == (0,)


# ---------------------------------------------------------------------------
# LocalFrame
# ---------------------------------------------------------------------------


def test_local_frame_origin_axes_and_roundtrip() -> None:
    f = LocalFrame(*ULM)
    x0, y0 = f.to_xy(*ULM)
    assert abs(x0) < 1e-6 and abs(y0) < 1e-6
    # A point due north has x == 0 and y == its geodesic distance.
    x, y = f.to_xy(ULM[0] + 0.01, ULM[1])
    assert abs(x) < 1e-6
    assert y == pytest.approx(geodesic_distance_m(*ULM, ULM[0] + 0.01, ULM[1]), abs=1e-6)
    # Due east -> positive x.
    x, _ = f.to_xy(ULM[0], ULM[1] + 0.01)
    assert x > 0
    # Vectorised round trip.
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(-5000, 5000, 50), rng.uniform(-5000, 5000, 50)
    lat, lon = f.to_latlon(xs, ys)
    assert isinstance(lat, np.ndarray) and lat.shape == (50,)
    x2, y2 = f.to_xy(lat, lon)
    assert np.allclose(x2, xs, atol=1e-6) and np.allclose(y2, ys, atol=1e-6)


def test_local_frame_is_distance_preserving_from_origin() -> None:
    # Azimuthal equidistant: |xy| == geodesic distance from the origin.
    f = LocalFrame(*ULM)
    lat = np.array([48.42, 48.37, 48.40, 48.45])
    lon = np.array([10.03, 9.95, 9.90, 10.10])
    x, y = f.to_xy(lat, lon)
    assert np.allclose(np.hypot(x, y), geodesic_distance_m(*ULM, lat, lon), atol=1e-6)
    # ...and pairwise distances agree with geodesics to <1e-5 relative at
    # the few-km scale of a drive.
    d_xy = math.hypot(x[0] - x[1], y[0] - y[1])
    d_g = geodesic_distance_m(lat[0], lon[0], lat[1], lon[1])
    assert abs(d_xy - d_g) / d_g < 1e-5


def test_local_frame_scalar_types() -> None:
    f = LocalFrame(*ULM)
    xy = f.to_xy(48.41, 10.0)
    ll = f.to_latlon(100.0, 200.0)
    assert all(isinstance(v, float) for v in (*xy, *ll))


# ---------------------------------------------------------------------------
# metres_per_degree / degree_half_extent
# ---------------------------------------------------------------------------


def test_metres_per_degree_known_values() -> None:
    m_lat, m_lon = metres_per_degree(0.0)
    assert m_lat == pytest.approx(110_574.28, abs=0.05)
    assert m_lon == pytest.approx(111_319.49, abs=0.05)
    m_lat, m_lon = metres_per_degree(60.0)
    assert m_lat == pytest.approx(111_412.29, abs=0.05)
    assert m_lon == pytest.approx(55_800.00, abs=0.05)
    # Agrees with a short geodesic at Ulm.
    m_lat, m_lon = metres_per_degree(ULM[0])
    d = 1e-3
    assert m_lat * d == pytest.approx(
        geodesic_distance_m(ULM[0] - d / 2, ULM[1], ULM[0] + d / 2, ULM[1]), rel=1e-6)
    assert m_lon * d == pytest.approx(
        geodesic_distance_m(ULM[0], ULM[1], ULM[0], ULM[1] + d), rel=1e-6)
    arr = metres_per_degree(np.array([0.0, 60.0]))
    assert arr[0].shape == (2,)


def test_degree_half_extent_matches_metric_radius() -> None:
    dlat, dlon = degree_half_extent(*ULM, 1000.0)
    m_lat, m_lon = metres_per_degree(ULM[0])
    assert dlat == pytest.approx(1000.0 / m_lat, rel=1e-4)
    assert dlon == pytest.approx(1000.0 / m_lon, rel=1e-4)
    # within 0.3 % of the former flat 111 320 m/deg conversion
    assert dlat == pytest.approx(1000.0 / 111_320.0, rel=3e-3)
    assert dlon == pytest.approx(
        1000.0 / (111_320.0 * math.cos(math.radians(ULM[0]))), rel=3e-3)
    # antimeridian: still a small positive half-width
    _, dlon_am = degree_half_extent(0.0, 179.9999, 1000.0)
    assert 0 < dlon_am < 0.01
