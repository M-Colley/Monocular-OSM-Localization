"""Tests for the ground-truth evaluator."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.evaluator import (
    _polyline_to_polyline_distance,
    _segment_to_polyline_distance,
    best_rank_for_gt,
    evaluate_candidates,
)
from src.osm_data import _build_polyline_view
from src.trajectory_matching import MatchCandidate


def _named_graph() -> "RoadGraph":
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    nodes = {
        "A": (0.0, 0.0),    "B": (100.0, 0.0),    "C": (200.0, 0.0),    # NorthRoad
        "D": (0.0, -100.0), "E": (100.0, -100.0),                       # SouthRoad
    }
    for k, (x, y) in nodes.items():
        g.add_node(k, x=x, y=y)

    def add(a: str, b: str, name: str) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        L = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=L, geometry=LineString([(ax, ay), (bx, by)]), name=name)
        g.add_edge(b, a, length=L, geometry=LineString([(bx, by), (ax, ay)]), name=name)

    add("A", "B", "NorthRoad")
    add("B", "C", "NorthRoad")
    add("D", "E", "SouthRoad")
    return _build_polyline_view(g)


def _candidate_for_walk(road: "RoadGraph", walk: list[tuple], score: float = 1.0) -> MatchCandidate:
    from src.osm_data import walk_to_polyline
    poly = walk_to_polyline(road.graph, walk)
    return MatchCandidate(
        score=score, bearing_corr=0.5,
        start_node=walk[0][0],
        walk=walk,
        walk_xy=poly,
        aligned_traj_xy=poly.copy(),
        walk_length_m=200.0,
    )


def test_segment_to_polyline_distance_zero_when_on_polyline() -> None:
    poly = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert _segment_to_polyline_distance(np.array([5.0, 0.0]), poly) == pytest.approx(0.0)
    # 3-4-5 triangle.
    assert _segment_to_polyline_distance(np.array([5.0, 4.0]), poly) == pytest.approx(4.0)


def test_polyline_to_polyline_distance_via_proxy() -> None:
    a = np.array([[5.0, 4.0], [50.0, 50.0]])
    b = np.array([[0.0, 0.0], [10.0, 0.0]])
    # Closest point of a to b is (5, 4) → distance 4.0.
    assert _polyline_to_polyline_distance(a, b) == pytest.approx(4.0)


def test_evaluate_candidates_marks_on_gt_street() -> None:
    road = _named_graph()
    cand_on_north = _candidate_for_walk(road, [("A", "B", 0), ("B", "C", 0)], score=10.0)
    cand_on_south = _candidate_for_walk(road, [("D", "E", 0)], score=20.0)

    results = evaluate_candidates([cand_on_north, cand_on_south], road, ["NorthRoad"])
    assert results[0].on_gt_street is True
    assert "NorthRoad" in results[0].matching_gt_names
    assert results[0].nearest_distance_m == pytest.approx(0.0)

    assert results[1].on_gt_street is False
    # SouthRoad is 100m south of NorthRoad.
    assert results[1].nearest_distance_m == pytest.approx(100.0)


def test_evaluate_candidates_handles_unknown_street_name() -> None:
    road = _named_graph()
    cand = _candidate_for_walk(road, [("A", "B", 0)])
    results = evaluate_candidates([cand], road, ["StreetThatDoesNotExist"])
    assert results[0].on_gt_street is False
    # No GT polylines → all distances are inf.
    assert results[0].nearest_distance_m == float("inf")


def test_best_rank_for_gt_picks_closest_and_first_named() -> None:
    road = _named_graph()
    on_north = _candidate_for_walk(road, [("A", "B", 0)])
    on_south = _candidate_for_walk(road, [("D", "E", 0)])
    results = evaluate_candidates([on_south, on_north], road, ["NorthRoad"])
    best_dist_rank, name_rank = best_rank_for_gt(results)
    assert best_dist_rank == 2  # NorthRoad walk is at index 1 → rank 2
    assert name_rank == 2
