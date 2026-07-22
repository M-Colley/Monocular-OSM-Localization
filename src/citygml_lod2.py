"""Open-data LoD2 CityGML city models: fetch, parse, triangulate.

Turns the free official German LoD2 building models into a triangle mesh
(in the pipeline's projected road CRS) that :mod:`tile3d_match` renders
for the 3D-tile skyline channel.

Providers (URLs verified live 2026-07-22):

- ``berlin`` — Senatsverwaltung LoD2, 1x1 km tiles, EPSG:25833,
  license dl-de/zero-2.0 (no attribution required)::

      https://gdi.berlin.de/data/a_lod2/atom/LoD2_<E_km>_<N_km>.zip

- ``bw`` — Baden-Wuerttemberg (covers the Ulm and Karlsruhe GT clips),
  2x2 km blocks of four 1 km GML tiles, EPSG:25832, license
  dl-de/by-2.0 (attribution "LGL, www.lgl-bw.de")::

      https://opengeodata.lgl-bw.de/data/lod2/LoD2_32_<E_km odd>_<N_km even>_2_bw.zip

  The LGL server 403s python's default User-Agent; we send a browser UA.

Documented but not implemented (extend PROVIDERS when needed): Bavaria
(download1.bayernwolke.de, CC BY 4.0), NRW (opengeodata.nrw.de,
dl-de/zero-2.0), and the Germany-wide basemap.de 3D Gebaeude OGC 3D
Tiles stream (CC BY 4.0, glb+DRACO — needs a glTF decoder).

Google Photorealistic 3D Tiles was evaluated and REJECTED: the Map
Tiles API policies explicitly prohibit "image analysis", "machine
interpretation", "geodata extraction" and offline storage — precisely
this module's use. The open state models above cover every German GT
clip, so Google is neither needed nor permitted here.

Datum note: the state models are ETRS89-based UTM (EPSG:2583x) while
the road graph is WGS84 UTM (EPSG:326xx); pyproj handles the transform,
whose datum component is <1 m — negligible against LoD2 generalization.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

__all__ = [
    "Lod2Mesh", "PROVIDERS", "provider_for_latlon", "tiles_for_disc",
    "parse_citygml_lod2", "fetch_lod2_mesh",
]

# The LGL-BW server rejects requests without a browser-looking UA.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
       "Monocular-OSM-Localization/1.0 (open-data LoD2 research fetch)")

_DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "tiles3d"

# Bump when parse_citygml_lod2/_triangulate_ring semantics change: the
# assembled-mesh npz cache key includes it, so stale meshes parsed by
# an older version are never silently reused.
_PARSE_VERSION = 2

PROVIDERS: dict[str, dict] = {
    "berlin": {
        "crs": "EPSG:25833",
        "step_km": 1,
        "url": "https://gdi.berlin.de/data/a_lod2/atom/LoD2_{e}_{n}.zip",
        "license": "dl-de/zero-2.0",
        # generous lat/lon box around the state of Berlin
        "bbox": (52.30, 52.70, 13.05, 13.80),
    },
    "bw": {
        "crs": "EPSG:25832",
        "step_km": 2,
        "url": ("https://opengeodata.lgl-bw.de/data/lod2/"
                "LoD2_32_{e}_{n}_2_bw.zip"),
        "license": "dl-de/by-2.0 (attribution: LGL, www.lgl-bw.de)",
        "bbox": (47.50, 49.85, 7.40, 10.55),
    },
}


def provider_for_latlon(lat: float, lon: float) -> str | None:
    """Name of the open-LoD2 provider covering (lat, lon), or None."""
    for name, p in PROVIDERS.items():
        lat0, lat1, lon0, lon1 = p["bbox"]
        if lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
            return name
    return None


def _snap_tile(e_km: int, n_km: int, provider: str) -> tuple[int, int]:
    """Snap a 1 km cell to the provider's tile-name grid."""
    if PROVIDERS[provider]["step_km"] == 2:
        # BW blocks are named by ODD easting-km and EVEN northing-km.
        e_km = e_km if e_km % 2 == 1 else e_km - 1
        n_km = n_km if n_km % 2 == 0 else n_km - 1
    return e_km, n_km


def tiles_for_disc(easting: float, northing: float, radius_m: float,
                   provider: str) -> list[tuple[int, int]]:
    """Provider tile names (E_km, N_km) whose square intersects the disc.

    Sorted by tile-center distance to the disc center so a tile cap
    keeps the closest tiles.
    """
    step = PROVIDERS[provider]["step_km"]
    e0 = int(np.floor((easting - radius_m) / 1000.0))
    e1 = int(np.floor((easting + radius_m) / 1000.0))
    n0 = int(np.floor((northing - radius_m) / 1000.0))
    n1 = int(np.floor((northing + radius_m) / 1000.0))
    tiles: set[tuple[int, int]] = set()
    for e in range(e0, e1 + 1):
        for n in range(n0, n1 + 1):
            tiles.add(_snap_tile(e, n, provider))

    def _dist(t: tuple[int, int]) -> float:
        ce = (t[0] + step / 2.0) * 1000.0
        cn = (t[1] + step / 2.0) * 1000.0
        return float(np.hypot(ce - easting, cn - northing))

    return sorted(tiles, key=_dist)


# --------------------------------------------------------------------------
# CityGML parsing
# --------------------------------------------------------------------------

def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _newell_normal(pts: np.ndarray) -> np.ndarray:
    """Newell's method: robust polygon normal for near-planar 3D rings."""
    nxt = np.roll(pts, -1, axis=0)
    n = np.array([
        np.sum((pts[:, 1] - nxt[:, 1]) * (pts[:, 2] + nxt[:, 2])),
        np.sum((pts[:, 2] - nxt[:, 2]) * (pts[:, 0] + nxt[:, 0])),
        np.sum((pts[:, 0] - nxt[:, 0]) * (pts[:, 1] + nxt[:, 1])),
    ])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _ear_clip(pts2d: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple (possibly concave) 2D polygon by ear clipping.

    Falls back to a fan when numerics defeat the ear test — LoD2 rings
    are small and near-convex, so the fallback is rare and harmless.
    """
    n = len(pts2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
    area2 = float(np.sum(pts2d[:, 0] * np.roll(pts2d[:, 1], -1)
                         - np.roll(pts2d[:, 0], -1) * pts2d[:, 1]))
    idx = list(range(n))
    if area2 < 0:
        idx.reverse()
    tris: list[tuple[int, int, int]] = []
    while len(idx) > 3:
        m = len(idx)
        clipped = False
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = pts2d[i0], pts2d[i1], pts2d[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:      # reflex or degenerate corner
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts2d[j]
                d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
                d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
                if d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                del idx[k]
                clipped = True
                break
        if not clipped:             # numeric stalemate: fan the remainder
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            return tris
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def _triangulate_ring(ring: np.ndarray) -> np.ndarray:
    """(M,3) closed 3D ring -> (T,3,3) triangles (empty on degenerate)."""
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        return np.empty((0, 3, 3), dtype=np.float64)
    normal = _newell_normal(ring)
    # 2D basis in the ring's plane
    seed = np.array([1.0, 0.0, 0.0])
    if abs(normal[0]) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    pts2d = np.stack([ring @ u, ring @ v], axis=1)
    tri_idx = _ear_clip(pts2d)
    if not tri_idx:
        return np.empty((0, 3, 3), dtype=np.float64)
    tris = ring[np.asarray(tri_idx, dtype=np.int64)]
    # drop zero-area output (colinear rings survive ear clipping as
    # degenerate fans that cv2 would rasterize as spurious 1-px lines)
    areas = 0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    return tris[areas > 1e-6]


def parse_citygml_lod2(
    gml_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse one CityGML document into triangles + per-building ground.

    Returns ``(triangles (N,3,3) float64 in the source CRS,
    building_ground (M,3) [x, y, z_min] float64)``. Only exterior
    LinearRings are used (interior rings = holes are rare in LoD2 and
    only cost tiny silhouette artifacts). Terrain-intersection curves
    carry no LinearRing, so they are skipped naturally.
    """
    tris: list[np.ndarray] = []
    grounds: list[tuple[float, float, float]] = []
    interior_depth = 0
    ring_depth = 0
    bldg_depth = 0
    bldg_pts: list[np.ndarray] = []

    for event, elem in ET.iterparse(BytesIO(gml_bytes), events=("start", "end")):
        name = _localname(elem.tag)
        if event == "start":
            if name == "interior":
                interior_depth += 1
            elif name == "LinearRing":
                ring_depth += 1
            elif name in ("Building", "BuildingPart"):
                bldg_depth += 1
                if bldg_depth == 1:
                    bldg_pts = []
            continue
        # end events
        if name == "interior":
            interior_depth = max(0, interior_depth - 1)
        elif name == "LinearRing":
            ring_depth = max(0, ring_depth - 1)
        elif name == "posList":
            if (ring_depth > 0 and interior_depth == 0 and elem.text
                    and elem.get("srsDimension", "3") == "3"):
                vals = np.array(elem.text.split(), dtype=np.float64)
                if vals.size >= 9 and vals.size % 3 == 0:
                    ring = vals.reshape(-1, 3)
                    if np.all(np.isfinite(ring)):
                        t = _triangulate_ring(ring)
                        if len(t):
                            tris.append(t)
                            if bldg_depth > 0:
                                bldg_pts.append(ring)
            elem.clear()
        elif name in ("Building", "BuildingPart"):
            bldg_depth = max(0, bldg_depth - 1)
            if bldg_depth == 0 and bldg_pts:
                allp = np.vstack(bldg_pts)
                grounds.append((float(allp[:, 0].mean()),
                                float(allp[:, 1].mean()),
                                float(allp[:, 2].min())))
                bldg_pts = []
            elem.clear()

    triangles = (np.concatenate(tris, axis=0) if tris
                 else np.empty((0, 3, 3), dtype=np.float64))
    ground = (np.asarray(grounds, dtype=np.float64) if grounds
              else np.empty((0, 3), dtype=np.float64))
    return triangles, ground


# --------------------------------------------------------------------------
# Fetch + mesh assembly
# --------------------------------------------------------------------------

@dataclass
class Lod2Mesh:
    """LoD2 triangles in a target projected CRS (x east, y north, z abs)."""
    triangles: np.ndarray        # (N, 3, 3) float32, dst CRS + absolute z
    building_ground: np.ndarray  # (M, 3) [x, y, z_min] float32, dst CRS
    crs: str                     # CRS of the coordinates above
    provider: str
    n_buildings: int

    def local_ground_z(self, xy, radius_m: float = 200.0) -> float:
        """Robust terrain height near ``xy``: median building base z.

        LoD2 has no terrain layer; building bases track it closely.
        Falls back to the mesh-wide median when no building is nearby.
        """
        if not len(self.building_ground):
            return 0.0
        d = np.hypot(self.building_ground[:, 0] - float(xy[0]),
                     self.building_ground[:, 1] - float(xy[1]))
        near = self.building_ground[d < radius_m, 2]
        pool = near if len(near) >= 3 else self.building_ground[:, 2]
        return float(np.median(pool))


def _download_tile(url: str, dest: Path) -> bool:
    """Fetch one tile zip (atomic write). False = tile does not exist
    (HTTP 404 — disc corner outside state coverage), which callers may
    cache; transient errors raise so they are never cached. The temp
    name is per-process so concurrent fleet runs sharing the cache
    cannot interleave writes, and the payload must actually BE a zip
    before it is committed — a captive portal / error page served with
    HTTP 200 must never poison the cache."""
    import os

    import requests

    tmp = dest.with_suffix(f".part{os.getpid()}")
    try:
        with requests.get(url, headers={"User-Agent": _UA}, stream=True,
                          timeout=120) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        if not zipfile.is_zipfile(tmp):
            raise RuntimeError(f"{url} returned a non-zip body "
                               f"(proxy/error page?) — not caching it")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def _parse_tile_zip(zip_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse every GML/XML member of a tile zip."""
    tris: list[np.ndarray] = []
    grounds: list[np.ndarray] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith((".gml", ".xml")):
                continue
            t, g = parse_citygml_lod2(zf.read(member))
            if len(t):
                tris.append(t)
            if len(g):
                grounds.append(g)
    triangles = (np.concatenate(tris, axis=0) if tris
                 else np.empty((0, 3, 3), dtype=np.float64))
    ground = (np.concatenate(grounds, axis=0) if grounds
              else np.empty((0, 3), dtype=np.float64))
    return triangles, ground


def fetch_lod2_mesh(
    lat: float,
    lon: float,
    radius_m: float,
    *,
    dst_crs: str,
    provider: str = "auto",
    cache_dir: Path | None = None,
    max_tiles: int = 80,
) -> Lod2Mesh | None:
    """Open-data LoD2 mesh for a disc, transformed into ``dst_crs``.

    Returns None (with a printed reason) when no open provider covers
    the location — the caller should treat the channel as inactive.
    Tile zips cache under ``data/tiles3d/<provider>/`` (404s cached as
    ``.missing`` markers); the assembled per-disc mesh caches as an npz
    keyed by (provider, tiles, dst_crs) so reruns skip the XML parse.
    """
    from pyproj import Transformer

    if provider == "auto":
        provider = provider_for_latlon(lat, lon)  # type: ignore[assignment]
        if provider is None:
            print(f"      -> no open LoD2 provider covers "
                  f"{lat:.4f},{lon:.4f} (open data exists for Berlin/BW; "
                  f"Google 3D Tiles is ToS-prohibited for analysis use)")
            return None
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LoD2 provider: {provider!r}")
    spec = PROVIDERS[provider]

    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE / provider
    cache.mkdir(parents=True, exist_ok=True)

    to_prov = Transformer.from_crs("EPSG:4326", spec["crs"], always_xy=True)
    e, n = to_prov.transform(lon, lat)
    tiles = tiles_for_disc(float(e), float(n), radius_m, provider)
    if len(tiles) > max_tiles:
        print(f"      -> LoD2 disc needs {len(tiles)} tiles; capping at "
              f"the {max_tiles} closest (raise max_tiles to cover all)")
        tiles = tiles[:max_tiles]

    sig = hashlib.sha1(json.dumps(
        {"provider": provider, "tiles": sorted(tiles), "dst": dst_crs,
         "parse_version": _PARSE_VERSION},
        sort_keys=True).encode()).hexdigest()[:12]
    mesh_cache = cache / f"mesh_{sig}.npz"
    if mesh_cache.exists():
        z = np.load(mesh_cache)
        return Lod2Mesh(
            triangles=z["triangles"].reshape(-1, 3, 3),
            building_ground=z["building_ground"].reshape(-1, 3),
            crs=dst_crs, provider=provider,
            n_buildings=int(z["n_buildings"]),
        )

    all_tris: list[np.ndarray] = []
    all_ground: list[np.ndarray] = []
    n_fetched = n_missing = 0
    for e_km, n_km in tiles:
        name = spec["url"].format(e=e_km, n=n_km).rsplit("/", 1)[-1]
        zip_path = cache / name
        missing_marker = cache / (name + ".missing")
        if missing_marker.exists():
            n_missing += 1
            continue
        url = spec["url"].format(e=e_km, n=n_km)
        if not zip_path.exists():
            if not _download_tile(url, zip_path):
                missing_marker.touch()   # genuine 404: outside coverage
                n_missing += 1
                continue
            n_fetched += 1
        try:
            t, g = _parse_tile_zip(zip_path)
        except zipfile.BadZipFile:
            # a corrupt zip from an older run must self-heal, not
            # permanently kill the channel: drop it, refetch once
            print(f"      -> corrupt cached tile {zip_path.name}; "
                  f"refetching")
            zip_path.unlink(missing_ok=True)
            if not _download_tile(url, zip_path):
                missing_marker.touch()
                n_missing += 1
                continue
            t, g = _parse_tile_zip(zip_path)
        if len(t):
            all_tris.append(t)
        if len(g):
            all_ground.append(g)

    triangles = (np.concatenate(all_tris, axis=0) if all_tris
                 else np.empty((0, 3, 3), dtype=np.float64))
    ground = (np.concatenate(all_ground, axis=0) if all_ground
              else np.empty((0, 3), dtype=np.float64))
    print(f"      -> LoD2 [{provider}] {len(tiles)} tile(s) "
          f"({n_fetched} fetched, {n_missing} outside coverage): "
          f"{len(ground)} buildings, {len(triangles)} triangles "
          f"[license: {spec['license']}]")

    if len(triangles):
        # provider CRS -> road CRS (x,y only; absolute z passes through)
        to_dst = Transformer.from_crs(spec["crs"], dst_crs, always_xy=True)
        flat = triangles.reshape(-1, 3)
        x, y = to_dst.transform(flat[:, 0].tolist(), flat[:, 1].tolist())
        flat[:, 0] = x
        flat[:, 1] = y
        triangles = flat.reshape(-1, 3, 3)
        if len(ground):
            gx, gy = to_dst.transform(ground[:, 0].tolist(),
                                      ground[:, 1].tolist())
            ground[:, 0] = gx
            ground[:, 1] = gy

    mesh = Lod2Mesh(
        triangles=triangles.astype(np.float32),
        building_ground=ground.astype(np.float32),
        crs=dst_crs, provider=provider, n_buildings=len(ground),
    )
    tmp = mesh_cache.with_suffix(".part.npz")
    np.savez_compressed(
        tmp, triangles=mesh.triangles.reshape(-1, 9),
        building_ground=mesh.building_ground.reshape(-1, 3),
        n_buildings=np.int64(mesh.n_buildings),
    )
    tmp.replace(mesh_cache)
    return mesh
