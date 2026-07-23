"""Offline tests for the worldwide OSM LoD1 building extrusion (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from src.osm_buildings3d import (_grid_centers, extrude_footprint,
                                 parse_osm_height_m)


def test_parse_height_from_height_tag() -> None:
    assert parse_osm_height_m({"height": "12"}) == 12.0
    assert parse_osm_height_m({"height": "12.5 m"}) == 12.5
    assert parse_osm_height_m({"building:height": "30m"}) == 30.0


def test_parse_height_from_levels() -> None:
    assert parse_osm_height_m({"building:levels": "5"}) == 16.0     # 5*3 + 1
    assert parse_osm_height_m({"levels": "2"}) == 7.0
    # explicit height wins over levels
    assert parse_osm_height_m({"height": "40", "building:levels": "3"}) == 40.0


def test_parse_height_default_and_garbage() -> None:
    assert parse_osm_height_m({}) == 8.0
    assert parse_osm_height_m({"height": "tall"}) == 8.0            # unparseable
    assert parse_osm_height_m({"height": "9999"}) == 8.0           # out of range
    assert parse_osm_height_m({}, default_height_m=5.0) == 5.0


def test_extrude_square_box() -> None:
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
    tris = extrude_footprint(sq, 20.0)
    # 4 edges x 2 wall tris = 8, + roof (quad -> 2) = 10
    assert len(tris) == 10
    assert tris[:, :, 2].min() == 0.0
    assert tris[:, :, 2].max() == 20.0
    # walls are vertical: each wall tri spans z 0..20
    walls = tris[:8]
    assert (walls[:, :, 2].max(axis=1) == 20.0).all()
    assert (walls[:, :, 2].min(axis=1) == 0.0).all()


def test_extrude_open_ring_is_closed() -> None:
    # ring given without the closing repeat still extrudes
    tri = np.array([[0, 0], [10, 0], [5, 8]], dtype=float)
    tris = extrude_footprint(tri, 15.0)
    assert len(tris) >= 6            # 3 edges x 2 + roof
    assert tris[:, :, 2].max() == 15.0


def test_extrude_degenerate_ring_empty() -> None:
    assert len(extrude_footprint(np.array([[0, 0], [1, 1]]), 10.0)) == 0
    assert len(extrude_footprint(np.zeros((0, 2)), 10.0)) == 0


def test_grid_centers_small_disc_is_single_query() -> None:
    assert _grid_centers(51.5, -0.1, 400.0, 700.0) == [(51.5, -0.1)]


def test_grid_centers_large_disc_tiles_and_covers() -> None:
    centres = _grid_centers(51.5, -0.13, 1500.0, 700.0)
    assert len(centres) > 1                       # tiled
    # every sub-centre lies within radius+tile of the disc centre
    for la, lo in centres:
        dm = np.hypot((la - 51.5) * 111320.0,
                      (lo + 0.13) * 111320.0 * np.cos(np.radians(51.5)))
        assert dm <= 1500.0 + 700.0 + 1.0
    # a point near the disc edge is within one tile-radius of some centre
    import numpy as _np
    edge_lat = 51.5 + (1400.0 / 111320.0)
    covered = any(
        _np.hypot((la - edge_lat) * 111320.0, 0.0) <= 700.0
        for la, lo in centres if abs(lo + 0.13) < 1e-6)
    assert covered


def test_extruded_mesh_renders_a_skyline() -> None:
    # a ring of tall boxes around a camera must produce a skyline
    from src.citygml_lod2 import Lod2Mesh
    from src.tile3d_match import (render_building_mask, skyline_from_mask,
                                  scale_intrinsics)
    from src.visual_odometry import default_intrinsics
    tris = []
    for cx, cy in [(40, 0), (-40, 0), (0, 40), (0, -40), (30, 30), (-30, 30)]:
        ring = np.array([[cx - 8, cy - 8], [cx + 8, cy - 8],
                         [cx + 8, cy + 8], [cx - 8, cy + 8]], dtype=float)
        tris.append(extrude_footprint(ring, 25.0))
    mesh = Lod2Mesh(triangles=np.concatenate(tris).astype(np.float32),
                    building_ground=np.zeros((6, 3), dtype=np.float32),
                    crs="EPSG:32633", provider="osm", n_buildings=6)
    K = scale_intrinsics(default_intrinsics(1280, 720), (1280, 720), (480, 270))
    mask = render_building_mask(mesh.triangles, np.array([0.0, 0.0]), 2.2,
                                np.array([0.0, 1.0]), K, (480, 270))
    assert mask.any()
    assert np.isfinite(skyline_from_mask(mask)).any()
