"""Encode small Mapbox Vector Tiles for tests (the inverse of monocular_osm.mvt)."""

from __future__ import annotations

import math

import mercantile

from monocular_osm.mvt import Tile


def _zz(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def encode_point_tile(points, z: int, x: int, y: int, *, layer: str = "image",
                      extent: int = 4096, id_kind: str = "int_value",
                      extra_layers: tuple[str, ...] = ()) -> bytes:
    """``points``: iterable of ``(id, lat, lon)`` inside tile z/x/y. Each becomes
    a POINT feature with an ``id`` property (stored as ``id_kind``) plus a
    ``is_pano`` bool; one LINESTRING feature is added to check it is skipped."""
    b = mercantile.bounds(mercantile.Tile(x, y, z))
    t = Tile()
    lyr = t.layers.add(name=layer, version=2, extent=extent)
    lyr.keys.extend(["is_pano", "id"])
    lyr.values.add(bool_value=False)
    for pid, lat, lon in points:
        fx = (lon - b.west) / (b.east - b.west)
        my = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
        mn = math.log(math.tan(math.pi / 4 + math.radians(b.north) / 2))
        ms = math.log(math.tan(math.pi / 4 + math.radians(b.south) / 2))
        fy = (mn - my) / (mn - ms)
        px, py = int(round(fx * extent)), int(round(fy * extent))
        lyr.values.add(**{id_kind: pid})
        f = lyr.features.add(type=1)
        f.tags.extend([0, 0, 1, len(lyr.values) - 1])
        f.geometry.extend([(1 << 3) | 1, _zz(px), _zz(py)])
    line = lyr.features.add(type=2)
    line.geometry.extend([(1 << 3) | 1, _zz(10), _zz(10), (1 << 3) | 2, _zz(5), _zz(5)])
    for name in extra_layers:
        t.layers.add(name=name, version=2, extent=extent)
    return t.SerializeToString()
