"""Tests for the real-RGB satellite tile module.

The network fetch (``contextily.bounds2img``) is monkeypatched so these
run offline; we verify the bbox-crop geometry and the candidate→lon/lat
routing rather than the tile contents.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import networkx as nx
import pytest
from shapely.geometry import LineString

from src import satellite
from src.osm_data import _build_polyline_view
from src.trajectory_matching import MatchCandidate


def _road():
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"  # UTM 32N (Ulm)
    coords = {0: (573000, 5361000), 1: (573300, 5361000)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))
    g.add_edge(0, 1, length=300.0, geometry=LineString([coords[0], coords[1]]))
    return _build_polyline_view(g)


def _candidate():
    walk = np.array([[573000.0, 5361000.0], [573300.0, 5361000.0]])
    return MatchCandidate(
        score=1.0, bearing_corr=0.5, start_node=0, walk=[(0, 1, 0)],
        walk_xy=walk, aligned_traj_xy=walk.copy(), walk_length_m=300.0,
    )


def test_crop_to_bbox_3857_selects_interior() -> None:
    # 100x100 mosaic spanning Web-Mercator extent (left,right,bottom,top).
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255  # bright square in the centre
    extent = (0.0, 1000.0, 0.0, 1000.0)  # 1000 m mercator span, 10 m/px

    # Request the central 400..600 bbox in mercator → reproject to lon/lat
    # then back is lossy, so instead test the helper directly with a bbox
    # whose mercator coords we control via the transformer round-trip.
    from pyproj import Transformer
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = to_ll.transform(400.0, 400.0)
    east, north = to_ll.transform(600.0, 600.0)

    crop = satellite._crop_to_bbox_3857(img, extent, west, south, east, north)
    # Central 200m → 20px window; should land on the bright square.
    assert crop.shape[0] >= 2 and crop.shape[1] >= 2
    assert crop.shape[0] <= 40 and crop.shape[1] <= 40
    assert int(crop.max()) == 255


def test_crop_handles_degenerate_extent() -> None:
    img = np.ones((10, 10, 3), dtype=np.uint8)
    out = satellite._crop_to_bbox_3857(img, (0.0, 0.0, 0.0, 0.0), 0, 0, 1, 1)
    assert out.shape == img.shape  # returns input unchanged on zero-span


def test_fetch_satellite_tile_monkeypatched(monkeypatch) -> None:
    """fetch_satellite_tile returns (size,size,3) uint8 RGB and converts RGBA."""
    captured = {}

    def fake_bounds2img(w, s, e, n, ll=False, source=None, **kw):
        captured["bounds"] = (w, s, e, n)
        captured["ll"] = ll
        # RGBA mosaic; extent generously around the requested bbox in 3857.
        from pyproj import Transformer
        to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        wx, sy = to_merc.transform(w, s)
        ex, ny = to_merc.transform(e, n)
        pad = (ex - wx)
        img = np.zeros((128, 128, 4), dtype=np.uint8)
        img[..., 3] = 255
        img[..., 0] = 200
        return img, (wx - pad, ex + pad, sy - pad, ny + pad)

    fake_cx = types.SimpleNamespace(
        bounds2img=fake_bounds2img,
        providers=types.SimpleNamespace(
            Esri=types.SimpleNamespace(WorldImagery="ESRI_SRC")
        ),
    )
    monkeypatch.setitem(sys.modules, "contextily", fake_cx)

    out = satellite.fetch_satellite_tile(9.99, 48.40, half_extent_m=60.0, size=256)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    assert captured["ll"] is True
    # Requested a tiny bbox around the centre lon/lat.
    w, s, e, n = captured["bounds"]
    assert w < 9.99 < e and s < 48.40 < n


def test_satellite_tile_for_candidate_routes_lonlat(monkeypatch) -> None:
    seen = {}

    def fake_fetch(lon, lat, *, half_extent_m, size, provider):
        seen.update(lon=lon, lat=lat, half=half_extent_m, size=size, provider=provider)
        return np.zeros((size, size, 3), dtype=np.uint8)

    monkeypatch.setattr(satellite, "fetch_satellite_tile", fake_fetch)
    out = satellite.satellite_tile_for_candidate(
        _road(), _candidate(), half_extent_m=60.0, size=128, provider="esri",
    )
    assert out.shape == (128, 128, 3)
    # Ulm is around lon 9.9-10.1, lat 48.3-48.5 — confirms CRS transform ran.
    assert 9.5 < seen["lon"] < 10.5
    assert 48.0 < seen["lat"] < 48.8
    assert seen["provider"] == "esri"


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unsupported satellite provider"):
        satellite._provider_source("googlemaps")
