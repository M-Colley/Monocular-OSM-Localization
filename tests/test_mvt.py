"""monocular_osm.mvt: point features of a Mapbox Vector Tile."""

from __future__ import annotations

import mercantile
import numpy as np
import pytest

from monocular_osm.mvt import decode_points

from ._mvt_fixtures import encode_point_tile

DETROIT = (42.3316, -83.0473)


@pytest.mark.parametrize("id_kind", ["int_value", "uint_value"])
def test_points_round_trip_to_within_tile_resolution(id_kind) -> None:
    t = mercantile.tile(DETROIT[1], DETROIT[0], 14)
    b = mercantile.bounds(t)
    rng = np.random.default_rng(0)
    pts = [(10 ** 15 + k, float(rng.uniform(b.south, b.north)), float(rng.uniform(b.west, b.east)))
           for k in range(50)]
    ids, ll = decode_points(encode_point_tile(pts, t.z, t.x, t.y, id_kind=id_kind),
                            "image", t.z, t.x, t.y)
    assert ids == [p[0] for p in pts]                      # the LINESTRING is skipped
    want = np.array([[p[1], p[2]] for p in pts])
    err_m = np.hypot(ll[:, 0] - want[:, 0],
                     (ll[:, 1] - want[:, 1]) * np.cos(np.radians(DETROIT[0]))) * 111_000
    assert err_m.max() < 0.5                               # 1.8 km / 4096 px


def test_a_missing_layer_or_property_gives_empty_or_none() -> None:
    t = mercantile.tile(DETROIT[1], DETROIT[0], 14)
    data = encode_point_tile([(7, *DETROIT)], t.z, t.x, t.y, extra_layers=("sequence",))
    assert decode_points(data, "overview", t.z, t.x, t.y)[0] == []
    ids, ll = decode_points(data, "image", t.z, t.x, t.y, prop="creator_id")
    assert ids == [None] and ll.shape == (1, 2)
