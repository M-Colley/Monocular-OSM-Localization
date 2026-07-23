"""Global LoD1 building meshes from OpenStreetMap footprints + heights.

The CityGML providers in :mod:`src.citygml_lod2` cover only four German
states. This module is the WORLDWIDE fallback for the tile3d skyline
channel: query OSM building polygons around a point, extrude each by its
tagged height (or an estimate from ``building:levels``) into a flat-roofed
LoD1 box, and assemble a :class:`~src.citygml_lod2.Lod2Mesh` in the road
CRS. LoD-Loc v2 (ICCV'25) showed LoD1 silhouettes are enough for skyline
localization, and OSM building coverage is dense in most cities — so this
makes the channel usable far beyond the open-CityGML footprint (London,
Málaga, US/Canada clips, …), wherever OSM has building heights or levels.

Heights: the ``height`` / ``building:height`` tag (metres) when present,
else ``building:levels`` x a storey height, else a default. Missing-height
buildings are common, so the estimate is deliberately conservative.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .citygml_lod2 import Lod2Mesh, _triangulate_ring

__all__ = ["fetch_osm_building_mesh", "parse_osm_height_m", "extrude_footprint"]

_DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "tiles3d" / "osm"
_MESH_VERSION = 1
_NUM = re.compile(r"[-+]?\d*\.?\d+")


def parse_osm_height_m(tags: dict, *, default_height_m: float = 8.0,
                       level_height_m: float = 3.0) -> float:
    """Best-effort building height (metres) from OSM tags.

    Prefers an explicit ``height`` / ``building:height`` (strip units), then
    ``building:levels`` * storey height (+ a nominal roof), else a default.
    """
    for key in ("height", "building:height"):
        v = tags.get(key)
        if v is not None:
            m = _NUM.search(str(v))
            if m:
                h = float(m.group())
                if 1.0 <= h <= 700.0:
                    return h
    for key in ("building:levels", "levels"):
        v = tags.get(key)
        if v is not None:
            m = _NUM.search(str(v))
            if m:
                lv = float(m.group())
                if 0.5 <= lv <= 200.0:
                    return lv * level_height_m + 1.0
    return default_height_m


def extrude_footprint(ring_xy: np.ndarray, height: float) -> np.ndarray:
    """Extrude a 2D footprint ring (N,2) to flat-roofed LoD1 triangles.

    Vertical wall quads (0..height) for every edge plus a roof cap at
    ``height``. Returns (T, 3, 3) float64; empty for a degenerate ring.
    """
    r = np.asarray(ring_xy, dtype=np.float64)
    if len(r) >= 2 and np.allclose(r[0], r[-1]):
        r = r[:-1]
    if len(r) < 3:
        return np.empty((0, 3, 3), dtype=np.float64)
    tris: list[np.ndarray] = []
    n = len(r)
    for i in range(n):
        a = r[i]
        b = r[(i + 1) % n]
        p0 = [a[0], a[1], 0.0]
        p1 = [b[0], b[1], 0.0]
        p2 = [b[0], b[1], height]
        p3 = [a[0], a[1], height]
        tris.append(np.array([p0, p1, p2], dtype=np.float64))
        tris.append(np.array([p0, p2, p3], dtype=np.float64))
    roof_ring = np.column_stack([r[:, 0], r[:, 1], np.full(n, height)])
    roof = _triangulate_ring(roof_ring)
    walls = np.asarray(tris, dtype=np.float64)
    if len(roof):
        return np.concatenate([walls, roof], axis=0)
    return walls


def _iter_exteriors(geom):
    """Yield exterior coordinate rings of a (Multi)Polygon geometry."""
    gt = getattr(geom, "geom_type", None)
    if gt == "Polygon":
        yield np.asarray(geom.exterior.coords)
    elif gt == "MultiPolygon":
        for part in geom.geoms:
            yield np.asarray(part.exterior.coords)


def _grid_centers(lat: float, lon: float, radius_m: float,
                  tile_r: float) -> list[tuple[float, float]]:
    """Sub-query centres (lat, lon) tiling a disc — a single Overpass bbox
    over a large dense area truncates ('Response ended prematurely'), so
    big discs are fetched as overlapping ~tile_r sub-queries and merged."""
    if radius_m <= tile_r * 1.3:
        return [(lat, lon)]
    step = tile_r                      # overlap: tiles are tile_r-radius
    dlat = step / 111_320.0
    dlon = step / (111_320.0 * max(0.2, np.cos(np.radians(lat))))
    n = int(np.ceil(radius_m / step))
    centres: list[tuple[float, float]] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if np.hypot(i * step, j * step) <= radius_m + tile_r:
                centres.append((lat + i * dlat, lon + j * dlon))
    return centres


def _fetch_osm_buildings(lat: float, lon: float, radius_m: float,
                         tile_r: float = 700.0):
    """OSM building features around a point as one GeoDataFrame, tiling the
    Overpass query for large discs and deduping buildings across tiles."""
    import osmnx as ox
    import pandas as pd

    frames = []
    for clat, clon in _grid_centers(lat, lon, radius_m, tile_r):
        for attempt in range(2):
            try:
                g = ox.features_from_point((clat, clon),
                                           tags={"building": True},
                                           dist=tile_r)
                if g is not None and not g.empty:
                    frames.append(g)
                break
            except Exception:
                if attempt == 1:
                    break              # skip this tile, keep the rest
    if not frames:
        return None
    gdf = pd.concat(frames)
    return gdf[~gdf.index.duplicated(keep="first")]


def fetch_osm_building_mesh(
    lat: float,
    lon: float,
    radius_m: float,
    *,
    dst_crs: str,
    cache_dir: Path | None = None,
    default_height_m: float = 8.0,
    level_height_m: float = 3.0,
    max_buildings: int = 40000,
) -> Lod2Mesh | None:
    """Worldwide LoD1 building mesh from OSM, in ``dst_crs``.

    Returns None (with a printed reason) if OSM has no buildings here. The
    assembled mesh caches as an npz keyed by (rounded centre, radius,
    dst_crs, height params) so reruns skip the Overpass fetch + extrusion.
    """
    from pyproj import Transformer

    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha1((
        f"{lat:.4f},{lon:.4f},{radius_m:.0f},{dst_crs},"
        f"{default_height_m},{level_height_m},v{_MESH_VERSION}"
    ).encode()).hexdigest()[:12]
    mesh_cache = cache / f"osm_{sig}.npz"
    if mesh_cache.exists():
        try:
            with np.load(mesh_cache) as z:
                return Lod2Mesh(
                    triangles=z["triangles"].reshape(-1, 3, 3),
                    building_ground=z["building_ground"].reshape(-1, 3),
                    crs=dst_crs, provider="osm",
                    n_buildings=int(z["n_buildings"]))
        except (ValueError, EOFError, OSError, KeyError):
            mesh_cache.unlink(missing_ok=True)

    try:
        gdf = _fetch_osm_buildings(float(lat), float(lon), float(radius_m))
    except Exception as e:  # Overpass unreachable / import failure
        print(f"      -> OSM buildings query failed at "
              f"{lat:.4f},{lon:.4f} ({e}); tile3d(osm) inactive")
        return None
    if gdf is None or gdf.empty:
        print(f"      -> OSM has no buildings at {lat:.4f},{lon:.4f}; "
              f"tile3d(osm) inactive")
        return None

    to_dst = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    all_tris: list[np.ndarray] = []
    grounds: list[tuple[float, float, float]] = []
    n_tag = 0
    cols = gdf.columns
    for geom, row in zip(gdf.geometry, gdf.to_dict("records")):
        if geom is None or geom.is_empty:
            continue
        tags = {k: row[k] for k in cols
                if k in row and row[k] is not None and k != "geometry"}
        h = parse_osm_height_m(tags, default_height_m=default_height_m,
                               level_height_m=level_height_m)
        if any(k in tags for k in ("height", "building:height",
                                   "building:levels", "levels")):
            n_tag += 1
        for ring in _iter_exteriors(geom):
            if len(ring) < 4:
                continue
            x, y = to_dst.transform(ring[:, 0], ring[:, 1])
            ring_xy = np.column_stack([x, y])
            t = extrude_footprint(ring_xy, h)
            if len(t):
                all_tris.append(t)
                c = ring_xy.mean(axis=0)
                grounds.append((float(c[0]), float(c[1]), 0.0))
        if len(grounds) >= max_buildings:
            print(f"      -> OSM buildings capped at {max_buildings} "
                  f"(raise for a wider disc)")
            break

    if not all_tris:
        print(f"      -> OSM buildings had no usable footprints here; "
              f"tile3d(osm) inactive")
        return None
    triangles = np.concatenate(all_tris, axis=0).astype(np.float32)
    ground = np.asarray(grounds, dtype=np.float32)
    frac = n_tag / max(1, len(grounds))
    print(f"      -> OSM LoD1 [osm] {len(ground)} buildings, "
          f"{len(triangles)} triangles ({100 * frac:.0f}% with a "
          f"height/levels tag; rest estimated) [license: ODbL, OSM]")

    mesh = Lod2Mesh(triangles=triangles, building_ground=ground,
                    crs=dst_crs, provider="osm", n_buildings=len(ground))
    import os
    tmp = mesh_cache.with_suffix(f".part{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, triangles=mesh.triangles.reshape(-1, 9),
                            building_ground=mesh.building_ground.reshape(-1, 3),
                            n_buildings=np.int64(mesh.n_buildings))
        tmp.replace(mesh_cache)
    finally:
        tmp.unlink(missing_ok=True)
    return mesh
