from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from src.embedding_retrieval import (
    _embedding_cube_to_rgb,
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
