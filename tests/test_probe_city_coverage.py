"""Municipal coverage is determined by geometry, without route labels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from monocular_osm.geo import geodesic_distance_m

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("city_probe_tested", ROOT / "scripts/probe_city_coverage.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_hex_cells_cover_a_concave_municipality_with_geodesic_discs():
    # Synthetic L-shaped municipality; no route, city cache, or GT is supplied.
    city = Polygon([(-0.12, -0.1), (0.12, -0.1), (0.12, 0),
                    (0, 0), (0, 0.1), (-0.12, 0.1)])
    regions, audit = M.municipality_grid(city, "Synthetic municipality")
    assert 1 < len(regions) <= 64
    assert audit["uncovered_polygon_area_m2"] < 1.0
    assert all(r["radius_m"] == 3000 for r in regions)
    # Independent geodesic check over interior and edge samples.
    lonlat = np.array([(lo, la) for lo in np.linspace(-0.12, 0.12, 41)
                       for la in np.linspace(-0.1, 0.1, 41) if city.covers(Point(lo, la))])
    nearest = np.min([geodesic_distance_m(lonlat[:, 1], lonlat[:, 0], *r["center"])
                      for r in regions], axis=0)
    assert float(nearest.max()) <= 3000
    again, _ = M.municipality_grid(city, "Synthetic municipality")
    assert regions == again


def test_city_cover_refuses_truncation_when_management_budget_is_too_small():
    city = Polygon([(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)])
    with pytest.raises(ValueError, match="no truncated plan"):
        M.municipality_grid(city, "Synthetic municipality", max_cells=1)


def test_city_cover_requires_a_real_polygon():
    with pytest.raises(ValueError, match="municipality polygon"):
        M.municipality_grid(Point(0, 0), "Missing boundary")
