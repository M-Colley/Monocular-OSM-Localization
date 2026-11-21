"""Point features of a Mapbox Vector Tile (MVT 2.1), decoded with ``protobuf``.

Mapillary's coverage tiles (``mly1_public``, zoom 14) list every image in a
tile with its id and position in one response; this reads them. GDAL's MVT
driver (via pyogrio) would too, but it refuses tiles over 10 MiB, and a dense
downtown z14 tile is larger. The message classes are built at import from the
schema in the MVT specification (github.com/mapbox/vector-tile-spec, 2.1), so
no generated code or ``protoc`` is needed.
"""

from __future__ import annotations

import math

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_F = descriptor_pb2.FieldDescriptorProto


def _build_tile_class():
    fdp = descriptor_pb2.FileDescriptorProto(name="vector_tile_2_1.proto",
                                             package="vector_tile", syntax="proto2")
    tile = fdp.message_type.add(name="Tile")
    gt = tile.enum_type.add(name="GeomType")
    for i, n in enumerate(("UNKNOWN", "POINT", "LINESTRING", "POLYGON")):
        gt.value.add(name=n, number=i)

    def field(msg, name, num, typ, label=_F.LABEL_OPTIONAL, type_name=None, packed=False):
        f = msg.field.add(name=name, number=num, type=typ, label=label)
        if type_name:
            f.type_name = type_name
        if packed:
            f.options.packed = True

    value = tile.nested_type.add(name="Value")
    for name, num, typ in (("string_value", 1, _F.TYPE_STRING), ("float_value", 2, _F.TYPE_FLOAT),
                           ("double_value", 3, _F.TYPE_DOUBLE), ("int_value", 4, _F.TYPE_INT64),
                           ("uint_value", 5, _F.TYPE_UINT64), ("sint_value", 6, _F.TYPE_SINT64),
                           ("bool_value", 7, _F.TYPE_BOOL)):
        field(value, name, num, typ)
    feat = tile.nested_type.add(name="Feature")
    field(feat, "id", 1, _F.TYPE_UINT64)
    field(feat, "tags", 2, _F.TYPE_UINT32, _F.LABEL_REPEATED, packed=True)
    field(feat, "type", 3, _F.TYPE_ENUM, type_name=".vector_tile.Tile.GeomType")
    field(feat, "geometry", 4, _F.TYPE_UINT32, _F.LABEL_REPEATED, packed=True)
    layer = tile.nested_type.add(name="Layer")
    field(layer, "version", 15, _F.TYPE_UINT32)
    field(layer, "name", 1, _F.TYPE_STRING)
    field(layer, "features", 2, _F.TYPE_MESSAGE, _F.LABEL_REPEATED, ".vector_tile.Tile.Feature")
    field(layer, "keys", 3, _F.TYPE_STRING, _F.LABEL_REPEATED)
    field(layer, "values", 4, _F.TYPE_MESSAGE, _F.LABEL_REPEATED, ".vector_tile.Tile.Value")
    field(layer, "extent", 5, _F.TYPE_UINT32)
    field(tile, "layers", 3, _F.TYPE_MESSAGE, _F.LABEL_REPEATED, ".vector_tile.Tile.Layer")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("vector_tile.Tile"))


Tile = _build_tile_class()


def _value(v):
    """The one field a Tile.Value carries (the schema has no oneof)."""
    for name in ("int_value", "uint_value", "sint_value", "string_value",
                 "double_value", "float_value", "bool_value"):
        if v.HasField(name):
            return getattr(v, name)
    return None


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def decode_points(data: bytes, layer: str, z: int, x: int, y: int,
                  prop: str = "id") -> tuple[list, np.ndarray]:
    """``(values, latlon[N, 2])`` for every point of ``layer`` in tile z/x/y,
    where ``values[i]`` is the feature's ``prop`` property (None if absent).
    A layer missing from the tile gives empty results."""
    t = Tile()
    t.ParseFromString(data)
    lyr = next((l for l in t.layers if l.name == layer), None)
    if lyr is None:
        return [], np.zeros((0, 2))
    extent = lyr.extent or 4096
    try:
        key = list(lyr.keys).index(prop)
    except ValueError:
        key = None
    vals_cache: dict[int, object] = {}
    out_v, px, py = [], [], []
    for f in lyr.features:
        if f.type != 1:                       # POINT only
            continue
        val = None
        if key is not None:
            tags = f.tags
            for i in range(0, len(tags) - 1, 2):
                if tags[i] == key:
                    vi = tags[i + 1]
                    if vi not in vals_cache:
                        vals_cache[vi] = _value(lyr.values[vi])
                    val = vals_cache[vi]
                    break
        g = f.geometry
        cx = cy = 0
        i = 0
        while i < len(g):
            cmd, count = g[i] & 7, g[i] >> 3
            i += 1
            if cmd != 1:                      # points are MoveTo only
                i += 2 * count if cmd == 2 else 0
                continue
            for _ in range(count):
                cx += _zigzag(g[i])
                cy += _zigzag(g[i + 1])
                i += 2
                out_v.append(val)
                px.append(cx)
                py.append(cy)
    if not px:
        return [], np.zeros((0, 2))
    n = 2.0 ** z
    fx = (x + np.asarray(px, float) / extent) / n
    fy = (y + np.asarray(py, float) / extent) / n
    lon = fx * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * fy))))
    return out_v, np.c_[lat, lon]
