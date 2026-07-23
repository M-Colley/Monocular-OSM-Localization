"""Offline tests for the LoD2 CityGML fetch/parse module (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from src.citygml_lod2 import (
    Lod2Mesh,
    PROVIDERS,
    _ear_clip,
    _triangulate_ring,
    parse_citygml_lod2,
    provider_for_latlon,
    tiles_for_disc,
)

# A minimal CityGML 2.0 document: one building with one wall polygon
# (with an interior hole that must be ignored) and one roof polygon,
# plus a non-ring posList (terrain-intersection style) that must NOT
# become geometry.
_GML = """<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"
    xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
    xmlns:gml="http://www.opengis.net/gml">
  <core:cityObjectMember>
    <bldg:Building gml:id="B1">
      <bldg:lod2TerrainIntersection>
        <gml:MultiCurve><gml:curveMember><gml:LineString>
          <gml:posList srsDimension="3">0 0 34 10 0 34</gml:posList>
        </gml:LineString></gml:curveMember></gml:MultiCurve>
      </bldg:lod2TerrainIntersection>
      <bldg:boundedBy><bldg:WallSurface><bldg:lod2MultiSurface>
        <gml:MultiSurface><gml:surfaceMember><gml:Polygon>
          <gml:exterior><gml:LinearRing>
            <gml:posList srsDimension="3">
              0 0 34 10 0 34 10 0 46 0 0 46 0 0 34
            </gml:posList>
          </gml:LinearRing></gml:exterior>
          <gml:interior><gml:LinearRing>
            <gml:posList srsDimension="3">
              4 0 38 6 0 38 6 0 40 4 0 40 4 0 38
            </gml:posList>
          </gml:LinearRing></gml:interior>
        </gml:Polygon></gml:surfaceMember></gml:MultiSurface>
      </bldg:lod2MultiSurface></bldg:WallSurface></bldg:boundedBy>
      <bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface>
        <gml:MultiSurface><gml:surfaceMember><gml:Polygon>
          <gml:exterior><gml:LinearRing>
            <gml:posList srsDimension="3">
              0 0 46 10 0 46 10 8 46 0 8 46 0 0 46
            </gml:posList>
          </gml:LinearRing></gml:exterior>
        </gml:Polygon></gml:surfaceMember></gml:MultiSurface>
      </bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>
"""


def test_parse_citygml_extracts_exterior_rings_only() -> None:
    tris, ground = parse_citygml_lod2(_GML.encode())
    # wall quad -> 2 tris, roof quad -> 2 tris; hole + terrain curve ignored
    assert len(tris) == 4
    assert tris.shape[1:] == (3, 3)
    assert np.all(np.isfinite(tris))
    # one building, base at z=34, centroid within the footprint bbox
    assert len(ground) == 1
    assert ground[0, 2] == pytest.approx(34.0)
    assert 0.0 <= ground[0, 0] <= 10.0


def test_parse_citygml_empty_document() -> None:
    tris, ground = parse_citygml_lod2(b"<root></root>")
    assert len(tris) == 0 and len(ground) == 0


def test_ear_clip_concave_polygon_preserves_area() -> None:
    # L-shaped (concave) hexagon, area 3
    poly = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
                    dtype=np.float64)
    tris = _ear_clip(poly)
    assert len(tris) == 4  # n-2 triangles for a simple polygon
    area = 0.0
    for i0, i1, i2 in tris:
        a, b, c = poly[i0], poly[i1], poly[i2]
        area += abs((b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
    assert area == pytest.approx(3.0)


def test_triangulate_ring_vertical_wall() -> None:
    # vertical rectangle in the x-z plane (y constant) — the case a
    # naive 2D (x, y) projection would collapse to zero area
    ring = np.array([[0, 5, 0], [4, 5, 0], [4, 5, 10], [0, 5, 10], [0, 5, 0]],
                    dtype=np.float64)
    tris = _triangulate_ring(ring)
    assert len(tris) == 2
    n = np.cross(tris[0, 1] - tris[0, 0], tris[0, 2] - tris[0, 0])
    assert abs(n[1]) == pytest.approx(np.linalg.norm(n))  # normal is +-y


def test_tiles_for_disc_berlin_1km_grid() -> None:
    # disc centered mid-tile, radius small: exactly that tile, then a
    # bigger radius pulls in the 8 neighbours
    tiles = tiles_for_disc(392500.0, 5820500.0, 100.0, "berlin")
    assert tiles == [(392, 5820)]
    tiles = tiles_for_disc(392500.0, 5820500.0, 600.0, "berlin")
    assert set(tiles) == {(e, n) for e in (391, 392, 393)
                          for n in (5819, 5820, 5821)}
    assert tiles[0] == (392, 5820)  # closest first


def test_tiles_for_disc_bw_odd_even_snapping() -> None:
    # Ulm-ish coordinates: 574 (even) snaps to block 573 (odd);
    # 5361 (odd) snaps to 5360 (even) — the verified LGL naming grid.
    tiles = tiles_for_disc(574500.0, 5361500.0, 100.0, "bw")
    assert tiles == [(573, 5360)]
    # radius reaching the next block east adds 575, not 574
    tiles = tiles_for_disc(574900.0, 5361500.0, 300.0, "bw")
    assert (575, 5360) in tiles and (574, 5360) not in tiles


def test_provider_for_latlon() -> None:
    assert provider_for_latlon(52.5230, 13.4171) == "berlin"   # Alexanderplatz
    assert provider_for_latlon(48.4059, 9.9837) == "bw"        # Ulm
    assert provider_for_latlon(49.0093, 8.4371) == "bw"        # Karlsruhe
    assert provider_for_latlon(50.9375, 6.9603) == "nrw"       # Cologne
    assert provider_for_latlon(48.1372, 11.5755) == "bavaria"  # Munich
    assert provider_for_latlon(49.4521, 11.0767) == "bavaria"  # Nuremberg
    assert provider_for_latlon(51.5270, -0.1318) is None       # London
    assert provider_for_latlon(37.6750, -122.4609) is None     # Daly City


def test_tiles_for_disc_nrw_1km_gml_naming() -> None:
    # NRW: 1 km tiles, no snapping; Cologne-ish easting/northing in km.
    tiles = tiles_for_disc(355500.0, 5644500.0, 100.0, "nrw")
    assert tiles == [(355, 5644)]
    name = PROVIDERS["nrw"]["url"].format(e=355, n=5644).rsplit("/", 1)[-1]
    assert name == "LoD2_32_355_5644_1_NW.gml"


def test_tiles_for_disc_bavaria_even_snapping() -> None:
    # Bavaria: 2 km tiles named by EVEN easting/northing km.
    tiles = tiles_for_disc(705500.0, 5323500.0, 100.0, "bavaria")
    assert tiles == [(704, 5322)]
    name = PROVIDERS["bavaria"]["url"].format(e=704, n=5322).rsplit("/", 1)[-1]
    assert name == "704_5322.gml"


def test_provider_formats() -> None:
    assert PROVIDERS["berlin"]["format"] == "zip"
    assert PROVIDERS["bw"]["format"] == "zip"
    assert PROVIDERS["nrw"]["format"] == "gml"
    assert PROVIDERS["bavaria"]["format"] == "gml"


def test_looks_like_citygml_rejects_html() -> None:
    from src.citygml_lod2 import _looks_like_citygml
    assert _looks_like_citygml(b'<?xml version="1.0"?><core:CityModel>')
    assert _looks_like_citygml(b'\xef\xbb\xbf<?xml version="1.0"?>')
    assert not _looks_like_citygml(b"<!DOCTYPE html><html><body>404")
    assert not _looks_like_citygml(b"<html>Not Found</html>")


def test_per_tile_cache_reused_across_discs(tmp_path, monkeypatch) -> None:
    """A per-tile parsed cache must be reused across DIFFERENT discs, so a
    later run (or the prefetch) never re-parses a tile another disc
    already parsed — the fix for the prefetch-cache-never-hit bug.
    Everything is offline: a fake provider + a synthetic zip."""
    import zipfile

    import src.citygml_lod2 as mod

    monkeypatch.setitem(mod.PROVIDERS, "test", {
        "crs": "EPSG:25833", "step_km": 1,
        "url": "http://test.invalid/LoD2_{e}_{n}.zip",
        "license": "test", "bbox": (52.0, 53.0, 13.0, 14.0)})

    def fake_download(url, dest, fmt="zip"):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("tile.gml", _GML)
        return True

    calls = {"n": 0}
    real_parse = mod._parse_tile_zip

    def counting_parse(zip_path):
        calls["n"] += 1
        return real_parse(zip_path)

    monkeypatch.setattr(mod, "_download_tile", fake_download)
    monkeypatch.setattr(mod, "_parse_tile_zip", counting_parse)

    big = mod.fetch_lod2_mesh(52.5, 13.4, 2500.0, dst_crs="EPSG:32633",
                              provider="test", cache_dir=tmp_path, max_tiles=60)
    n_big = calls["n"]
    assert n_big >= 4 and len(big.triangles) > 0
    assert list(tmp_path.glob("tile_*.npz"))          # per-tile caches written

    # A smaller, DIFFERENT disc (a strict subset of tiles) must reuse every
    # tile from the big disc's per-tile caches -> zero new parses.
    small = mod.fetch_lod2_mesh(52.5, 13.4, 100.0, dst_crs="EPSG:32633",
                                provider="test", cache_dir=tmp_path, max_tiles=60)
    assert 0 < len(small.triangles) < len(big.triangles)  # genuinely a new disc
    assert calls["n"] == n_big                            # all tiles from cache


def test_per_tile_cache_self_heals_corrupt(tmp_path, monkeypatch) -> None:
    """A corrupt per-tile npz must be dropped and re-parsed, not fatal."""
    import zipfile

    import src.citygml_lod2 as mod

    monkeypatch.setitem(mod.PROVIDERS, "test", {
        "crs": "EPSG:25833", "step_km": 1,
        "url": "http://test.invalid/LoD2_{e}_{n}.zip",
        "license": "test", "bbox": (52.0, 53.0, 13.0, 14.0)})

    def fake_download(url, dest, fmt="zip"):
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("t.gml", _GML)
        return True

    monkeypatch.setattr(mod, "_download_tile", fake_download)

    m1 = mod.fetch_lod2_mesh(52.5, 13.4, 100.0, dst_crs="EPSG:32633",
                             provider="test", cache_dir=tmp_path, max_tiles=10)
    n1 = len(m1.triangles)
    assert n1 > 0
    # Corrupt the per-tile caches AND drop the per-disc mesh cache so the
    # next call re-enters the tile loop and must self-heal the bad npz.
    for p in tmp_path.glob("tile_*.npz"):
        p.write_bytes(b"not an npz")
    for p in tmp_path.glob("mesh_*.npz"):
        p.unlink()
    m2 = mod.fetch_lod2_mesh(52.5, 13.4, 100.0, dst_crs="EPSG:32633",
                             provider="test", cache_dir=tmp_path, max_tiles=10)
    assert len(m2.triangles) == n1            # rebuilt, not crashed


def test_local_ground_z_prefers_nearby_buildings() -> None:
    ground = np.array([
        [0.0, 0.0, 30.0], [50.0, 0.0, 31.0], [80.0, 0.0, 32.0],
        [5000.0, 0.0, 90.0], [5100.0, 0.0, 91.0], [5200.0, 0.0, 92.0],
    ], dtype=np.float32)
    mesh = Lod2Mesh(
        triangles=np.zeros((0, 3, 3), dtype=np.float32),
        building_ground=ground, crs="EPSG:32633",
        provider="berlin", n_buildings=6,
    )
    assert mesh.local_ground_z((10.0, 0.0)) == pytest.approx(31.0)
    assert mesh.local_ground_z((5100.0, 0.0)) == pytest.approx(91.0)
    # far from everything: falls back to the global median
    assert 30.0 <= mesh.local_ground_z((99999.0, 99999.0)) <= 92.0
