"""Tests for the aerial / OSM-patch feature-matching channel.

We use a small hand-built road graph and a synthetic "splat" rendering
so the test runs offline. The properties checked are:

  * `render_osm_patch` returns a same-size grayscale image.
  * `feature_match_score` returns more inliers when matched against
    itself than against an unrelated random image.
  * `match_splat_against_candidates` produces one result per candidate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.aerial_match import (
    feature_match_score,
    match_splat_against_candidates,
    render_osm_patch,
)
from src.osm_data import _build_polyline_view
from src.trajectory_matching import MatchCandidate


def _two_road_graph() -> "RoadGraph":
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    coords = {0: (0, 0), 1: (300, 0), 2: (300, 300), 3: (0, 300)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))

    def add(a, b):
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        L = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=L, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=L, geometry=LineString([(bx, by), (ax, ay)]))

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        add(a, b)

    return _build_polyline_view(g)


def test_render_osm_patch_size() -> None:
    road = _two_road_graph()
    img = render_osm_patch(road, (150.0, 150.0), half_extent_m=200.0, resolution=256)
    assert img.shape == (256, 256)
    assert img.dtype == np.uint8
    # Image isn't entirely white — there should be roads drawn.
    assert int(img.min()) < 200


def test_feature_match_self_returns_high_inliers() -> None:
    """Matching a textured image against itself should produce a high
    inlier count. This exercises the ORB + RANSAC path end-to-end."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    # Add some structure so ORB can latch onto features.
    for _ in range(40):
        x, y = rng.integers(20, 236, size=2)
        cv2.rectangle(img, (x - 10, y - 10), (x + 10, y + 10),
                      tuple(rng.integers(0, 256, size=3).tolist()), -1)

    n_match, n_in = feature_match_score(img, img)
    assert n_match >= 8
    # Self-match should give many inliers.
    assert n_in >= 8
    assert n_in / max(1, n_match) > 0.3


def test_feature_match_unrelated_images_lower_score() -> None:
    """Unrelated random images should score worse than self-match."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    self_in = feature_match_score(a, a)[1]
    cross_in = feature_match_score(a, b)[1]
    assert self_in > cross_in


def test_match_splat_against_candidates_returns_one_per_input(tmp_path: Path) -> None:
    road = _two_road_graph()

    # Two synthetic candidates centered at different points in the graph.
    walks_xy = [
        np.array([[50.0, 50.0], [50.0, 250.0]]),
        np.array([[200.0, 200.0], [280.0, 200.0]]),
    ]
    candidates = []
    for w in walks_xy:
        candidates.append(MatchCandidate(
            score=10.0,
            bearing_corr=0.5,
            start_node=0,
            walk=[(0, 1, 0)],
            walk_xy=w,
            aligned_traj_xy=w.copy(),
            walk_length_m=200.0,
        ))

    splat_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(splat_rgb, (10, 10), (110, 110), (255, 255, 255), -1)
    cv2.circle(splat_rgb, (64, 64), 30, (0, 0, 0), 3)

    results = match_splat_against_candidates(
        splat_rgb, road, candidates,
        output_dir=tmp_path,
        resolution=128,
        half_extent_m=200.0,
    )
    assert len(results) == 2
    assert all(r.osm_render_path is not None and r.osm_render_path.exists() for r in results)
    # Indices are preserved.
    assert [r.candidate_index for r in results] == [0, 1]
