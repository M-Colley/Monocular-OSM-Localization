from __future__ import annotations

import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from src.embedding_retrieval import (
    _crop_embedding_cube,
    _embedding_cube_to_rgb,
    _embedding_cubes_to_rgb_shared,
    score_candidates_by_embeddings,
)
from src.osm_data import _build_polyline_view
from src.trajectory_matching import MatchCandidate


class FakeEmbedder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        feats = []
        for image in images:
            feats.append(np.array([float(image.mean())], dtype=np.float32))
        return np.vstack(feats)


def _road_graph():
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    coords = {0: (0, 0), 1: (300, 0), 2: (300, 300)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))

    def add(a: int, b: int) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]), name=f"{a}-{b}")
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]), name=f"{a}-{b}")

    add(0, 1)
    add(1, 2)
    return _build_polyline_view(g)


def test_embedding_cube_to_rgb_returns_uint8_rgb() -> None:
    cube = np.random.default_rng(0).normal(size=(32, 32, 8)).astype(np.float32)
    rgb = _embedding_cube_to_rgb(cube, size=64)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_score_candidates_by_embeddings_scores_each_candidate(tmp_path: Path) -> None:
    road = _road_graph()
    candidates = [
        MatchCandidate(
            score=1.0,
            bearing_corr=0.2,
            start_node=0,
            walk=[(0, 1, 0)],
            walk_xy=np.array([[0.0, 0.0], [300.0, 0.0]]),
            aligned_traj_xy=np.array([[0.0, 0.0], [300.0, 0.0]]),
            walk_length_m=300.0,
        ),
        MatchCandidate(
            score=2.0,
            bearing_corr=0.1,
            start_node=1,
            walk=[(1, 2, 0)],
            walk_xy=np.array([[300.0, 0.0], [300.0, 300.0]]),
            aligned_traj_xy=np.array([[300.0, 0.0], [300.0, 300.0]]),
            walk_length_m=300.0,
        ),
    ]
    query = np.full((64, 64, 3), 200, dtype=np.uint8)

    results = score_candidates_by_embeddings(
        query,
        road,
        candidates,
        output_dir=tmp_path,
        sources=("osm",),
        embedder=FakeEmbedder(),
        size=64,
    )

    assert list(results) == ["osm"]
    assert len(results["osm"]) == 2
    assert all(r.image_path is not None and r.image_path.exists() for r in results["osm"])
    assert all(np.isfinite(r.cosine_similarity) for r in results["osm"])


# ---------------------------------------------------------------------------
# GeoTessera crop + shared-PCA rendering
# ---------------------------------------------------------------------------

# Affine mapping pixel (col, row) -> (x, y) in EPSG:32632, 10 m pixels,
# tile origin chosen so the test road graph (x, y in 0..300) sits inside.
_FAKE_AFFINE = types.SimpleNamespace(a=10.0, b=0.0, c=-1000.0, d=0.0, e=-10.0, f=2000.0)


def _fake_cube(h: int = 400, w: int = 400, c: int = 8) -> np.ndarray:
    # Spatially-varying content so different crops produce different images.
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cube = np.stack([(rows * w + cols + ch * 0.1) for ch in range(c)], axis=-1)
    return cube.astype(np.float32)


def test_crop_embedding_cube_centers_on_candidate() -> None:
    """The crop must be a candidate-centred window, not the whole tile —
    two points in the same 0.1-degree cell must yield DIFFERENT crops."""
    from pyproj import Transformer

    cube = _fake_cube()
    to_ll = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    lon1, lat1 = to_ll.transform(150.0, 0.0)
    lon2, lat2 = to_ll.transform(300.0, 150.0)

    crop1 = _crop_embedding_cube(
        cube, "EPSG:32632", _FAKE_AFFINE, lon1, lat1, half_extent_m=200.0)
    crop2 = _crop_embedding_cube(
        cube, "EPSG:32632", _FAKE_AFFINE, lon2, lat2, half_extent_m=200.0)

    # A 200 m half-extent at 10 m/px is a ~41 px window, not the 400 px tile.
    assert crop1.shape[0] < cube.shape[0] and crop1.shape[1] < cube.shape[1]
    assert crop1.shape == crop2.shape
    assert not np.array_equal(crop1, crop2)
    # The crop is centred on the candidate: (150, 0) -> col 115, row 200.
    assert np.isclose(crop1[crop1.shape[0] // 2, crop1.shape[1] // 2, 0],
                      cube[200, 115, 0])


def test_crop_embedding_cube_degrades_to_full_tile_without_transform() -> None:
    cube = _fake_cube(50, 50, 4)
    out = _crop_embedding_cube(cube, None, None, 9.0, 48.0, half_extent_m=100.0)
    assert out is cube


def test_shared_pca_basis_preserves_cross_tile_differences() -> None:
    """A constant embedding offset between two tiles is real signal.  A
    per-tile basis + per-tile mean-subtraction (the old path) erases it;
    the shared basis must preserve it."""
    rng = np.random.default_rng(0)
    cube_a = rng.normal(size=(32, 32, 8)).astype(np.float32)
    cube_b = cube_a + 5.0    # same texture, offset feature space

    # Old per-tile rendering maps both to the byte-identical image.
    rgb_a_solo = _embedding_cube_to_rgb(cube_a, size=32)
    rgb_b_solo = _embedding_cube_to_rgb(cube_b, size=32)
    assert np.array_equal(rgb_a_solo, rgb_b_solo)

    # Shared basis + shared normalization keeps them distinguishable.
    rgb_a, rgb_b = _embedding_cubes_to_rgb_shared([cube_a, cube_b], size=32)
    assert rgb_a.shape == (32, 32, 3) and rgb_a.dtype == np.uint8
    assert not np.array_equal(rgb_a, rgb_b)


def test_geotessera_source_same_cell_candidates_get_distinct_images(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end through score_candidates_by_embeddings: two candidates
    whose centroids fall in the SAME GeoTessera tile must render distinct,
    candidate-centred images (the old code returned the whole ~11 km tile
    for both -> byte-identical images -> similarity ties)."""
    cube = _fake_cube()

    class FakeGeoTessera:
        def fetch_embedding(self, *, lon, lat, year):
            # Whole-tile cube regardless of the point — like the real client.
            return cube, "EPSG:32632", _FAKE_AFFINE

    fake_mod = types.ModuleType("geotessera")
    fake_mod.GeoTessera = FakeGeoTessera
    monkeypatch.setitem(sys.modules, "geotessera", fake_mod)

    road = _road_graph()
    candidates = [
        MatchCandidate(
            score=1.0, bearing_corr=0.2, start_node=0, walk=[(0, 1, 0)],
            walk_xy=np.array([[0.0, 0.0], [300.0, 0.0]]),
            aligned_traj_xy=np.array([[0.0, 0.0], [300.0, 0.0]]),
            walk_length_m=300.0,
        ),
        MatchCandidate(
            score=2.0, bearing_corr=0.1, start_node=1, walk=[(1, 2, 0)],
            walk_xy=np.array([[300.0, 0.0], [300.0, 300.0]]),
            aligned_traj_xy=np.array([[300.0, 0.0], [300.0, 300.0]]),
            walk_length_m=300.0,
        ),
    ]
    query = np.full((64, 64, 3), 200, dtype=np.uint8)

    results = score_candidates_by_embeddings(
        query, road, candidates,
        output_dir=tmp_path,
        sources=("geotessera",),
        embedder=FakeEmbedder(),
        size=64,
    )

    geo = results["geotessera"]
    assert len(geo) == 2
    assert all(r.error is None for r in geo)
    assert all(r.image_path is not None and r.image_path.exists() for r in geo)

    import cv2
    img1 = cv2.imread(str(geo[0].image_path))
    img2 = cv2.imread(str(geo[1].image_path))
    assert img1 is not None and img2 is not None
    assert not np.array_equal(img1, img2), (
        "same-cell candidates rendered the byte-identical tile"
    )
