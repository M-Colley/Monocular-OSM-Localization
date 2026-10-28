"""Bounded, order-sensitive Fréchet diagnostics; no models or benchmark labels."""
from functools import lru_cache

import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point

from monocular_osm import metrics


@pytest.mark.parametrize("seed", range(8))
def test_iterative_recurrence_agrees_with_geos_on_small_curves(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3 + seed, 2)).cumsum(axis=0)
    b = rng.normal(size=(2 + seed, 2)).cumsum(axis=0)
    expected = shapely.frechet_distance(LineString(a), LineString(b))
    assert metrics._discrete_frechet_distance(a, b) == pytest.approx(expected, rel=1e-14)
    assert metrics._discrete_frechet_distance(b, a) == pytest.approx(expected, rel=1e-14)


def test_matches_exhaustive_monotone_couplings_including_stationary_vertices():
    a = np.array([[0., 0.], [2., 1.], [2., 1.], [5., 0.]])
    b = np.array([[0., 1.], [1., 0.], [5., 1.]])
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=2)

    @lru_cache(None)
    def coupling_maxima(i, j):
        if i == 0 and j == 0:
            return (distances[0, 0],)
        values = []
        for pi, pj in ((i - 1, j), (i, j - 1), (i - 1, j - 1)):
            if pi >= 0 and pj >= 0:
                values.extend(max(value, distances[i, j]) for value in coupling_maxima(pi, pj))
        return tuple(values)

    expected = min(coupling_maxima(len(a) - 1, len(b) - 1))
    assert metrics._discrete_frechet_distance(a, b) == expected


def test_points_reversal_and_zero_distance_are_hand_computed():
    route = np.array([[0., 0.], [3., 4.], [6., 8.]])
    assert metrics._discrete_frechet_distance(route, route) == 0.
    assert metrics._discrete_frechet_distance(route, route[::-1]) == 10.
    assert metrics._discrete_frechet_distance(route[:1], route) == 10.
    assert metrics._bounded_frechet_distance(Point(0., 0.), Point(3., 4.)) == 5.


def test_fixed_vertex_cap_is_applied_before_dynamic_program(monkeypatch):
    a = LineString(np.column_stack([np.arange(5000.), np.zeros(5000)]))
    b = LineString(np.column_stack([np.arange(7000.), np.ones(7000)]))
    observed = []

    def observe(first, second):
        observed.append((first.shape, second.shape))
        np.testing.assert_array_equal(first[[0, -1]], [[0., 0.], [4999., 0.]])
        np.testing.assert_array_equal(second[[0, -1]], [[0., 1.], [6999., 1.]])
        return 17.

    monkeypatch.setattr(metrics, "_discrete_frechet_distance", observe)
    assert metrics._bounded_frechet_distance(a, b) == 17.
    assert observed == [((metrics.FRECHET_MAX_VERTICES, 2),) * 2]


def test_full_budget_folded_curve_finishes_without_geos_recursion(monkeypatch):
    def native_forbidden(*args, **kwargs):
        raise AssertionError("diagnostic must not enter native recursive Fréchet")

    monkeypatch.setattr(shapely, "frechet_distance", native_forbidden)
    t = np.linspace(0., 1., metrics.FRECHET_MAX_VERTICES)
    xy = np.column_stack([15000 * t, 1000 * np.sin(12 * np.pi * t)])
    # Every coupling of opposite orders must include endpoints 15 km apart;
    # the diagonal coupling attains exactly that bound for this geometry.
    actual = metrics._bounded_frechet_distance(LineString(xy), LineString(xy[::-1]))
    assert actual == pytest.approx(15000., abs=1e-9)


def test_score_route_preserves_all_non_frechet_values_exactly(monkeypatch):
    inverse = Transformer.from_crs(
        "+proj=aeqd +lat_0=48.4 +lon_0=9.99 +datum=WGS84", 4326, always_xy=True)

    def ll(xy):
        xy = np.asarray(xy, float)
        lon, lat = inverse.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lat, lon])

    pred = ll([[0, 0], [100, 30], [200, 0], [200, 300], [0, 300]])
    gt = ll([[0, 0], [100, 0], [200, 0]])
    actual = metrics.score_route(pred, gt).to_dict()

    def old_native(a, b):
        return float(shapely.frechet_distance(metrics._cap_vertices(a), metrics._cap_vertices(b)))

    monkeypatch.setattr(metrics, "_bounded_frechet_distance", old_native)
    historical = metrics.score_route(pred, gt).to_dict()
    assert actual.pop("frechet_m") == pytest.approx(historical.pop("frechet_m"), abs=1e-10)
    assert actual == historical
