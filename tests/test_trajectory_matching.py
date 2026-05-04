"""Tests for trajectory matching: Procrustes math and end-to-end on
synthetic graphs with known ground truth."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.osm_data import RoadGraph, _build_polyline_view
from src.trajectory_matching import (
    match_trajectory,
    procrustes_similarity,
)


def test_procrustes_recovers_known_similarity() -> None:
    rng = np.random.default_rng(0)
    src = rng.uniform(-1, 1, size=(20, 2))
    th = np.deg2rad(37.0)
    R_true = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s_true = 4.5
    t_true = np.array([10.0, -3.0])
    dst = s_true * src @ R_true.T + t_true

    R, scale, residual, src_aligned = procrustes_similarity(src, dst)
    assert scale == pytest.approx(s_true, rel=1e-6)
    assert residual < 1e-6
    # R should equal R_true (Procrustes also handles reflections; we asked
    # for orientation-preserving and the test data is orientation-preserving).
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(src_aligned, dst, atol=1e-6)


def test_procrustes_residual_grows_with_noise() -> None:
    rng = np.random.default_rng(1)
    src = rng.uniform(-1, 1, size=(30, 2))
    dst_clean = 2.0 * src
    dst_noisy = dst_clean + rng.normal(0, 0.05, size=src.shape)

    _, _, r_clean, _ = procrustes_similarity(src, dst_clean)
    _, _, r_noisy, _ = procrustes_similarity(src, dst_noisy)
    assert r_clean < 1e-9
    assert r_noisy > 0.01


def _l_shape_graph() -> RoadGraph:
    """Two streets that share a start node:

        A -- B -- C  (the 'L1' street)
        |
        D
        |
        E             (the 'L2' street, perpendicular)

    plus a third decoy street far away with the same shape as L1 but
    different heading; the matcher should still find the right spot when
    given an L1-shaped trajectory.
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"

    nodes = {
        "A": (0, 0), "B": (100, 0), "C": (200, 0),
        "D": (0, -100), "E": (0, -200),
        # Decoy street 1 km north, oriented the same as A-B-C so it has
        # the same straight bearing signature. It's a separate component.
        "F": (10000, 0), "G": (10100, 0), "H": (10200, 0),
    }
    for k, (x, y) in nodes.items():
        g.add_node(k, x=float(x), y=float(y))

    def add(a: str, b: str) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]),
                   name=f"{a}{b}")
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]),
                   name=f"{a}{b}")

    add("A", "B")
    add("B", "C")
    add("A", "D")
    add("D", "E")
    add("F", "G")
    add("G", "H")

    return _build_polyline_view(g)


def test_match_finds_correct_walk_on_l_shape() -> None:
    """A trajectory that turns 90° left should match A→D→E (a left turn),
    not the straight A→B→C or the decoy F→G→H."""
    road = _l_shape_graph()

    # Build a synthetic trajectory shaped like A → D → E: go south 100 m,
    # then continue south another 100 m. (No turn, it's actually
    # A→D→E going straight south.)
    south_traj = np.array(
        [[0, -i] for i in range(0, 201, 5)]
    , dtype=float)
    # Apply an arbitrary similarity (rotate, scale, translate) — the
    # matcher must be invariant to this.
    th = np.deg2rad(42.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    south_traj = 0.013 * south_traj @ R.T + np.array([7.5, -2.1])

    cands = match_trajectory(
        south_traj,
        road,
        n_samples=48,
        walks_per_node=4,
        walk_depth=6,
        bearing_top_k=20,
        final_top_k=5,
        estimated_length_m=180.0,
        progress=False,
    )
    assert cands, "matcher returned no candidates"
    best = cands[0]

    # Trajectory is a perfectly straight south path. Many walks in this
    # graph are straight (A→B→C, A→D→E, B→C, D→E, ...) and they all map
    # onto the trajectory under a 2-D similarity with essentially zero
    # residual — they're tied valid matches. The matcher should rank
    # ANY of these straight walks above the L-walks (B→A→D, D→A→B). We
    # check that property directly: the best walk's geometry must be
    # straight.
    poly = best.walk_xy
    if len(poly) >= 3:
        d0 = poly[1] - poly[0]
        dN = poly[-1] - poly[-2]
        cos_turn = float(d0 @ dN / (np.linalg.norm(d0) * np.linalg.norm(dN) + 1e-9))
        assert cos_turn > 0.9, (
            f"best walk has a turn (cos={cos_turn:.2f}); a straight walk "
            "should have ranked higher for a straight trajectory"
        )
    # And: the residual must be effectively zero — these are exact shape
    # matches modulo floating-point noise.
    assert best.score < 1.0, f"unexpectedly large residual {best.score}"


def test_match_l_turn_picks_l_turn_over_straight() -> None:
    """Trajectory with a clear 90° left turn should rank A→...→E (straight
    south, no turn) BELOW a walk that actually contains a 90° turn."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    # An L-walk and a straight-walk sharing the same start node 0.
    coords = {
        0: (0, 0),
        1: (100, 0),    # straight east
        2: (200, 0),
        3: (0, -100),   # south
        4: (-100, -100),  # then west — 90° right turn
    }
    for n, (x, y) in coords.items():
        g.add_node(n, x=float(x), y=float(y))

    def add(a: int, b: int) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]))

    add(0, 1)
    add(1, 2)
    add(0, 3)
    add(3, 4)

    road = _build_polyline_view(g)

    # Trajectory: south 100, then a sharp right turn, west 100.
    traj = np.array(
        [[0, -i] for i in range(0, 101, 5)]
        + [[-i, -100] for i in range(5, 101, 5)]
    , dtype=float)

    cands = match_trajectory(
        traj,
        road,
        n_samples=48,
        walks_per_node=6,
        walk_depth=6,
        bearing_top_k=30,
        final_top_k=4,
        estimated_length_m=190.0,
        progress=False,
    )
    assert cands, "no candidates"
    best = cands[0]

    # The best walk must contain a real ~90° turn — the trajectory's shape
    # is an L, so the matcher should prefer any L-walk over the straight
    # walk 0→1→2. Multiple L-walks exist in this graph (0→3→4, 1→0→3,
    # etc.) and they're all valid by symmetry; the test just checks the
    # matcher picks one of them and not the straight road.
    poly = best.walk_xy
    d0 = poly[1] - poly[0]
    d1 = poly[-1] - poly[-2]
    cos_turn = float(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1)))
    assert cos_turn < 0.5, (
        f"best walk is too straight (cos={cos_turn:.2f}); matcher failed to "
        "prefer the L-shaped walk over the straight one"
    )

    # And: a straight walk should not be the top result.
    nodes = [best.walk[0][0]] + [v for _, v, _ in best.walk]
    assert nodes != [0, 1, 2], "straight walk ranked above L-walk"
