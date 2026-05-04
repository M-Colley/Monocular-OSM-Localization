"""Tests for OSM data utilities using a hand-built tiny graph.

We don't hit the network here — we build a `nx.MultiDiGraph` with exactly
the structure OSMnx-projected graphs have (nodes carry `x`/`y`, edges
carry `length` and optionally `geometry`) and test the polyline view and
walk enumerator against it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.osm_data import (
    RoadGraph,
    _build_polyline_view,
    walk_to_polyline,
    walks_from_node,
)


def _grid_graph(size: int = 4, spacing: float = 100.0) -> nx.MultiDiGraph:
    """A `size x size` regular grid as a directed multigraph in meters."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"

    for i in range(size):
        for j in range(size):
            node_id = i * size + j
            g.add_node(node_id, x=j * spacing, y=i * spacing)

    def add_edge(a: int, b: int) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]))

    for i in range(size):
        for j in range(size):
            n = i * size + j
            if j + 1 < size:
                add_edge(n, n + 1)
            if i + 1 < size:
                add_edge(n, n + size)
    return g


def test_build_polyline_view_extracts_geometry() -> None:
    g = _grid_graph(size=3, spacing=50.0)
    view: RoadGraph = _build_polyline_view(g)
    # 3x3 grid → 12 undirected edges → 24 directed in our construction.
    assert len(view.polylines) == 24
    assert all(len(p) >= 2 for p in view.polylines)
    # Each polyline is 2 points (pure-segment grid).
    assert all(p.shape == (2, 2) for p in view.polylines)


def test_walks_from_node_meets_target_length() -> None:
    g = _grid_graph(size=5, spacing=100.0)
    walks = walks_from_node(g, start=0, target_length_m=350.0, max_walks=4, max_depth=10)
    assert walks, "expected at least one walk"
    for walk in walks:
        polyline = walk_to_polyline(g, walk)
        # Cumulative geometric length should reach the target.
        diffs = np.diff(polyline, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        assert seg.sum() >= 350.0 - 1e-6


def test_walks_prefer_straight_continuation() -> None:
    """When a walk passes through an interior intersection, the
    heading-deviation prior in `walks_from_node` should keep going
    straight rather than turning."""
    # Build a graph that's a straight horizontal road from (0,0) → (400,0)
    # with one north-south branch crossing at (200, 0). At the crossing,
    # straight-ahead and a 90° turn are both available, so the prior is
    # what determines the choice.
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"

    nodes = {
        "W": (0, 0), "M": (200, 0), "E": (400, 0),       # west, mid, east
        "N": (200, 100), "S": (200, -100),                # branches at M
    }
    for k, (x, y) in nodes.items():
        g.add_node(k, x=float(x), y=float(y))

    def add(a: str, b: str) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        L = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=L, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=L, geometry=LineString([(bx, by), (ax, ay)]))

    add("W", "M")
    add("M", "E")
    add("M", "N")
    add("M", "S")

    # Start at W, target 350 m. The first edge has only one option (W→M).
    # At M with prior heading east, straight (M→E) competes against
    # left (M→N) and right (M→S). The prior should pick M→E.
    walks = walks_from_node(g, start="W", target_length_m=350.0, max_walks=1, max_depth=5)
    assert walks
    walk = walks[0]
    nodes_in_walk = [walk[0][0]] + [v for u, v, k in walk]
    assert nodes_in_walk == ["W", "M", "E"], (
        f"expected straight walk W→M→E, got {nodes_in_walk}"
    )


def test_walk_to_polyline_dedupes_join_points() -> None:
    g = _grid_graph(size=3, spacing=100.0)
    # Walk from node 0 → 1 → 2 should give 3 points, not 4.
    walk = [(0, 1, 0), (1, 2, 0)]
    poly = walk_to_polyline(g, walk)
    assert poly.shape == (3, 2)
    # Coords match grid construction.
    assert poly[0].tolist() == [0.0, 0.0]
    assert poly[1].tolist() == [100.0, 0.0]
    assert poly[2].tolist() == [200.0, 0.0]
