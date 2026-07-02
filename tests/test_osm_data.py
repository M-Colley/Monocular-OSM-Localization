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
    _build_walk,
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


def test_walks_include_two_turn_routes() -> None:
    """Real urban routes routinely need two non-greedy turns (the Ulm GT
    route does). The enumeration must produce at least one walk whose
    geometry contains two real turns; the previous single-turn scheme
    structurally could not."""
    g = _grid_graph(size=6, spacing=100.0)
    # Start at the SW corner (node 0). With enough budget, some walk must
    # turn twice — detectable as a polyline whose heading changes by
    # ~90 deg at two separate interior vertices.
    walks = walks_from_node(g, start=0, target_length_m=400.0, max_walks=60, max_depth=8)
    assert walks

    def n_turns(poly: np.ndarray) -> int:
        d = np.diff(poly, axis=0)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        cos = (d[:-1] * d[1:]).sum(axis=1)
        return int((cos < 0.5).sum())  # heading change > 60 deg

    turn_counts = [n_turns(walk_to_polyline(g, w)) for w in walks]
    assert max(turn_counts) >= 2, (
        f"no two-turn walk enumerated (turn counts: {sorted(set(turn_counts))})"
    )


def test_enumerator_produces_closed_rectangular_loop() -> None:
    """A route that returns through its starting intersection (the KITTI
    drive_0033 pattern) must be enumerable. On a grid, the greedy walk
    from a corner follows the perimeter — a closed rectangular loop —
    but the old node-based visited set truncated it one edge before the
    revisit of the start node, so no closed walk could ever exist."""
    g = _grid_graph(size=4, spacing=100.0)
    # Perimeter of the 4x4 grid: 4 * 3 * 100 = 1200 m.
    walks = walks_from_node(g, start=0, target_length_m=1200.0, max_walks=8, max_depth=20)
    assert walks
    closed = [w for w in walks if w[-1][1] == w[0][0]]
    assert closed, (
        "no closed loop enumerated; walks end at "
        f"{sorted({w[-1][1] for w in walks})}"
    )
    # The closed walk really is the full rectangle, not a short-circuit.
    poly = walk_to_polyline(g, closed[0])
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    assert seg.sum() == pytest.approx(1200.0)


def test_build_walk_does_not_lap_roundabout() -> None:
    """Edge-based visited still prevents lapping a (one-way) roundabout
    forever: each edge is consumed at most once, so the walk breaks
    after a single lap even with a much larger length target."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    ring = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    for n, (x, y) in enumerate(ring):
        g.add_node(n, x=x, y=y)
    for n in range(4):
        m = (n + 1) % 4
        ax, ay = ring[n]
        bx, by = ring[m]
        g.add_edge(n, m, length=100.0, geometry=LineString([(ax, ay), (bx, by)]))

    walk, length = _build_walk(g, (0, 1, 0), target_length_m=5000.0, max_depth=50)
    assert len(walk) == 4          # one lap: each ring edge exactly once
    assert length == pytest.approx(400.0)


def test_walks_cover_all_four_directions_at_crossroads() -> None:
    """At a standard 4-way intersection every out-direction must appear
    among the enumerated walks. The old first-edge cap ([:3], sorted by
    edge length) never tried the 4th direction at all."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    nodes = {
        "M": (0, 0),
        "N": (0, 400), "E": (400, 0), "S": (0, -400), "W": (-400, 0),
    }
    for k, (x, y) in nodes.items():
        g.add_node(k, x=float(x), y=float(y))

    def add(a: str, b: str) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        L = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=L, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=L, geometry=LineString([(bx, by), (ax, ay)]))

    for arm in ("N", "E", "S", "W"):
        add("M", arm)

    walks = walks_from_node(g, start="M", target_length_m=400.0, max_walks=8, max_depth=4)
    first_targets = {w[0][1] for w in walks}
    assert first_targets == {"N", "E", "S", "W"}, (
        f"enumeration missed a start direction; first edges reach {first_targets}"
    )


def test_walk_budget_round_robins_across_first_edges() -> None:
    """With a budget smaller than one first edge's full expansion, the
    remaining budget must still be spread over ALL first edges (greedy
    walks first) instead of letting the first direction consume it."""
    g = _grid_graph(size=6, spacing=100.0)
    # Interior node: out-degree 4. Budget of 4 → exactly the four greedy
    # walks, one per direction. The old sequential scheme burned all 4
    # on the first direction's greedy + single-turn variants.
    start = 2 * 6 + 2
    walks = walks_from_node(g, start=start, target_length_m=300.0, max_walks=4, max_depth=8)
    assert len(walks) == 4
    first_targets = {w[0][1] for w in walks}
    assert len(first_targets) == 4, (
        f"budget was not round-robined: first edges reach only {first_targets}"
    )


def test_walks_single_turn_both_directions_reachable() -> None:
    """At a 4-way crossing, single-turn walks must include BOTH left and
    right turns (rank 1 alone only covers one of them)."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    nodes = {
        "W": (0, 0), "M": (200, 0), "E": (500, 0),
        "N": (200, 300), "S": (200, -300),
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

    walks = walks_from_node(g, start="W", target_length_m=400.0, max_walks=30, max_depth=4)
    end_nodes = {w[-1][1] for w in walks}
    assert "N" in end_nodes and "S" in end_nodes, (
        f"single-turn enumeration missed a turn direction; walks end at {end_nodes}"
    )
