"""Tests for the ground-truth evaluator."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString

from src.evaluator import (
    _normalize_street_name,
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


def test_polyline_distance_mid_segment_approach() -> None:
    """The closest approach can be mid-segment on BOTH polylines — long
    straight OSM edges carry only 2 vertices. The old vertex-sampled min
    reported the distance from the far segment ENDPOINTS (~190 m here)
    instead of the true ~15 m mid-segment distance."""
    a = np.array([[0.0, 0.0], [400.0, 0.0]])        # 400 m two-vertex edge
    b = np.array([[200.0, 15.0], [210.0, 15.0]])    # short GT street near a's middle
    assert _polyline_to_polyline_distance(a, b) == pytest.approx(15.0, abs=0.5)


def test_polyline_distance_crossing_is_zero() -> None:
    """Two crossing polylines have distance ~0 even though every vertex
    of one is far from the other."""
    a = np.array([[-200.0, 0.0], [200.0, 0.0]])
    b = np.array([[0.0, -200.0], [0.0, 200.0]])
    assert _polyline_to_polyline_distance(a, b) == pytest.approx(0.0, abs=5.5)


def test_evaluate_candidates_uses_segment_distance_on_long_edges() -> None:
    """End-to-end: a candidate walk on a long 2-vertex edge passing 15 m
    from a GT street mid-segment must evaluate at ~15 m, so
    best_rank_for_gt names the truly closest candidate."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    nodes = {
        "A": (0.0, 0.0), "B": (400.0, 0.0),          # long unsimplified edge
        "G": (200.0, 15.0), "H": (210.0, 15.0),      # the GT street
    }
    for k, (x, y) in nodes.items():
        g.add_node(k, x=x, y=y)

    def add(a: str, b: str, name: str) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        L = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=L, geometry=LineString([(ax, ay), (bx, by)]), name=name)
        g.add_edge(b, a, length=L, geometry=LineString([(bx, by), (ax, ay)]), name=name)

    add("A", "B", "LongRoad")
    add("G", "H", "GtRoad")
    road = _build_polyline_view(g)

    cand = _candidate_for_walk(road, [("A", "B", 0)])
    results = evaluate_candidates([cand], road, ["GtRoad"])
    assert results[0].nearest_distance_m == pytest.approx(15.0, abs=0.5)


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


def test_normalize_street_name_handles_german_eszett() -> None:
    # The case that bit us on the live Ulm run: --ground-truth Olgastrasse
    # was being compared against OSM's 'Olgastraße' and failing.
    assert _normalize_street_name("Olgastraße") == _normalize_street_name("Olgastrasse")
    assert _normalize_street_name("STRAẞE") == _normalize_street_name("Strasse")


def test_normalize_street_name_handles_diacritics() -> None:
    assert _normalize_street_name("Café") == _normalize_street_name("Cafe")
    assert _normalize_street_name("Bülowstraße") == _normalize_street_name("Bulowstrasse")
    assert _normalize_street_name("Champs-Élysées") == _normalize_street_name("Champs-Elysees")


def test_evaluate_candidates_matches_eszett_with_ascii_groundtruth() -> None:
    """A walk along an OSM-named 'Olgastraße' must be detected when the
    user supplies the ASCII transliteration 'Olgastrasse' on the CLI."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("B", x=100.0, y=0.0)
    geom = LineString([(0.0, 0.0), (100.0, 0.0)])
    g.add_edge("A", "B", length=100.0, geometry=geom, name="Olgastraße")
    g.add_edge("B", "A", length=100.0, geometry=LineString([(100.0, 0.0), (0.0, 0.0)]), name="Olgastraße")
    road = _build_polyline_view(g)

    cand = _candidate_for_walk(road, [("A", "B", 0)])
    results = evaluate_candidates([cand], road, ["Olgastrasse"])  # ASCII spelling
    assert results[0].on_gt_street is True
    assert results[0].matching_gt_names == ["Olgastrasse"]
    assert results[0].nearest_distance_m == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GPS-waypoint ground truth
# ---------------------------------------------------------------------------


def test_load_gt_waypoints_valid_file(tmp_path) -> None:
    import json

    from src.evaluator import load_gt_waypoints

    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "city": "Ulm, Germany",
        "waypoints": [
            {"t_sec": 0, "lat": 48.405933, "lon": 9.983683},
            {"t_sec": 415, "lat": 48.400353, "lon": 10.002622},
        ],
    }), encoding="utf-8")
    wps = load_gt_waypoints(path)
    assert wps.shape == (2, 2)
    assert wps[0, 0] == pytest.approx(48.405933)
    assert wps[1, 1] == pytest.approx(10.002622)


@pytest.mark.parametrize(
    "payload",
    [
        {},                                            # no waypoints key
        {"waypoints": []},                             # empty
        {"waypoints": [{"lat": 48.0}]},                # missing lon
        {"waypoints": [{"lat": "x", "lon": 9.0}]},     # non-numeric
        {"waypoints": [{"lat": 95.0, "lon": 9.0}]},    # out of range
    ],
)
def test_load_gt_waypoints_rejects_malformed(tmp_path, payload) -> None:
    import json

    from src.evaluator import load_gt_waypoints

    path = tmp_path / "gt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gt_waypoints(path)


def _waypoints_on_polyline(road, poly: np.ndarray) -> np.ndarray:
    """GT lat/lon fixes lying exactly on a projected polyline."""
    from src.position import xy_to_latlon

    return xy_to_latlon(poly, road.crs)


def test_waypoint_eval_zero_for_exact_route() -> None:
    from src.evaluator import evaluate_candidates_against_waypoints

    road = _named_graph()
    cand = _candidate_for_walk(road, [("A", "B", 0), ("B", "C", 0)])
    gt = _waypoints_on_polyline(road, cand.aligned_traj_xy)

    evals = evaluate_candidates_against_waypoints([cand], road, gt)
    assert len(evals) == 1
    assert evals[0].start_error_m == pytest.approx(0.0, abs=0.01)
    assert evals[0].mean_route_error_m == pytest.approx(0.0, abs=0.01)
    assert evals[0].max_route_error_m == pytest.approx(0.0, abs=0.01)


def test_waypoint_eval_measures_known_offset() -> None:
    from src.evaluator import evaluate_candidates_against_waypoints

    road = _named_graph()
    cand = _candidate_for_walk(road, [("A", "B", 0), ("B", "C", 0)])
    # Shift the GT 100 m north of the candidate's path (which runs along y=0).
    shifted = cand.aligned_traj_xy + np.array([0.0, 100.0])
    gt = _waypoints_on_polyline(road, shifted)

    evals = evaluate_candidates_against_waypoints([cand], road, gt)
    assert evals[0].start_error_m == pytest.approx(100.0, abs=0.5)
    assert evals[0].mean_route_error_m == pytest.approx(100.0, abs=0.5)
    assert evals[0].max_route_error_m == pytest.approx(100.0, abs=0.5)


def test_waypoint_eval_ranks_closer_candidate_first() -> None:
    from src.evaluator import (
        best_rank_for_waypoints,
        evaluate_candidates_against_waypoints,
    )

    road = _named_graph()
    near = _candidate_for_walk(road, [("A", "B", 0), ("B", "C", 0)])   # y=0
    far = _candidate_for_walk(road, [("D", "E", 0)])                   # y=-100
    gt = _waypoints_on_polyline(road, near.aligned_traj_xy)

    evals = evaluate_candidates_against_waypoints([far, near], road, gt)
    assert evals[1].mean_route_error_m < evals[0].mean_route_error_m
    assert best_rank_for_waypoints(evals) == 2  # 1-based: 'near' is second


def test_waypoint_eval_empty_trajectory_is_inf() -> None:
    from src.evaluator import evaluate_candidates_against_waypoints

    road = _named_graph()
    cand = _candidate_for_walk(road, [("A", "B", 0)])
    cand.aligned_traj_xy = np.zeros((0, 2))
    gt = _waypoints_on_polyline(road, np.array([[50.0, 0.0]]))

    evals = evaluate_candidates_against_waypoints([cand], road, gt)
    assert not np.isfinite(evals[0].mean_route_error_m)


def test_best_rank_for_waypoints_empty() -> None:
    from src.evaluator import best_rank_for_waypoints

    assert best_rank_for_waypoints([]) is None
