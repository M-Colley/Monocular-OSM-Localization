"""Tests for src/position.py: projected-meters -> WGS84 position reports.

Ground truth strategy: instead of hardcoding UTM coordinates, every
round-trip test *forward*-projects known WGS84 points with pyproj and
asserts our code recovers the original lat/lon. That validates our
column ordering and CRS handling against pyproj itself.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from src.osm_data import RoadGraph, _build_polyline_view
from src.position import (
    build_position_report,
    candidate_center_latlon,
    classify_confidence,
    format_position_summary,
    google_maps_url,
    openstreetmap_url,
    xy_to_latlon,
)
from src.trajectory_matching import MatchCandidate

# Ulm Münster — squarely inside UTM zone 32N.
ULM_LAT, ULM_LON = 48.3984, 9.9916
UTM32N = "EPSG:32632"


def _project(latlons: list[tuple[float, float]], crs: str = UTM32N) -> np.ndarray:
    """Forward-project (lat, lon) pairs to projected (x, y) via pyproj."""
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = t.transform([lon for _, lon in latlons], [lat for lat, _ in latlons])
    return np.column_stack([xs, ys]).astype(np.float64)


def _road_with_one_street(crs: str = UTM32N, street: str = "Olgastraße") -> RoadGraph:
    """A two-node graph holding a single named street near Ulm."""
    xy = _project([(ULM_LAT, ULM_LON), (ULM_LAT + 0.002, ULM_LON)], crs)
    g = nx.MultiDiGraph()
    g.graph["crs"] = crs
    g.add_node("A", x=float(xy[0, 0]), y=float(xy[0, 1]))
    g.add_node("B", x=float(xy[1, 0]), y=float(xy[1, 1]))
    g.add_edge(
        "A", "B",
        length=float(np.linalg.norm(xy[1] - xy[0])),
        geometry=LineString([tuple(xy[0]), tuple(xy[1])]),
        name=street,
    )
    return _build_polyline_view(g)


def _make_candidate(
    road: RoadGraph,
    aligned_traj_xy: np.ndarray,
    *,
    score: float = 100.0,
    bearing_corr: float = 0.4,
) -> MatchCandidate:
    return MatchCandidate(
        score=score,
        bearing_corr=bearing_corr,
        start_node="A",
        walk=[("A", "B", 0)],
        walk_xy=road.polylines[0],
        aligned_traj_xy=np.asarray(aligned_traj_xy, dtype=np.float64),
        walk_length_m=float(np.linalg.norm(road.polylines[0][-1] - road.polylines[0][0])),
    )


# ---------------------------------------------------------------------------
# xy_to_latlon
# ---------------------------------------------------------------------------


def test_xy_to_latlon_roundtrips_known_point() -> None:
    xy = _project([(ULM_LAT, ULM_LON)])
    latlon = xy_to_latlon(xy, UTM32N)
    assert latlon.shape == (1, 2)
    # Column order must be (lat, lon) — a swap would land near (10, 48).
    assert latlon[0, 0] == pytest.approx(ULM_LAT, abs=1e-6)
    assert latlon[0, 1] == pytest.approx(ULM_LON, abs=1e-6)


def test_xy_to_latlon_accepts_wkt_crs() -> None:
    """RoadGraph.crs may be a WKT blob (graphml round-trips do that)."""
    wkt = CRS.from_epsg(32632).to_wkt()
    xy = _project([(ULM_LAT, ULM_LON)])
    latlon = xy_to_latlon(xy, wkt)
    assert latlon[0, 0] == pytest.approx(ULM_LAT, abs=1e-6)
    assert latlon[0, 1] == pytest.approx(ULM_LON, abs=1e-6)


# ---------------------------------------------------------------------------
# build_position_report
# ---------------------------------------------------------------------------


def test_position_report_recovers_route_endpoints() -> None:
    road = _road_with_one_street()
    waypoints = [
        (ULM_LAT, ULM_LON),
        (ULM_LAT + 0.001, ULM_LON + 0.001),
        (ULM_LAT + 0.002, ULM_LON + 0.002),
    ]
    cand = _make_candidate(road, _project(waypoints))

    pos = build_position_report(cand, road)
    assert pos is not None

    # Headline = route start = camera at first analyzed frame.
    assert pos["latitude"] == pytest.approx(waypoints[0][0], abs=1e-5)
    assert pos["longitude"] == pytest.approx(waypoints[0][1], abs=1e-5)
    assert pos["start"]["latitude"] == pos["latitude"]
    assert pos["end"]["latitude"] == pytest.approx(waypoints[-1][0], abs=1e-5)
    assert pos["end"]["longitude"] == pytest.approx(waypoints[-1][1], abs=1e-5)
    # Center must sit between start and end.
    assert (
        min(w[0] for w in waypoints)
        <= pos["center"]["latitude"]
        <= max(w[0] for w in waypoints)
    )
    assert pos["street_names"] == ["Olgastraße"]
    assert pos["ranking"] == "shape"


def test_position_report_route_is_subsampled_with_endpoints_kept() -> None:
    road = _road_with_one_street()
    n = 500
    lats = np.linspace(ULM_LAT, ULM_LAT + 0.01, n)
    waypoints = [(float(la), ULM_LON) for la in lats]
    cand = _make_candidate(road, _project(waypoints))

    pos = build_position_report(cand, road, max_route_points=50)
    assert pos is not None
    route = pos["route_latlon"]
    assert 2 <= len(route) <= 50
    assert route[0] == [pos["start"]["latitude"], pos["start"]["longitude"]]
    assert route[-1] == [pos["end"]["latitude"], pos["end"]["longitude"]]


def test_position_report_short_route_not_padded() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(
        road, _project([(ULM_LAT, ULM_LON), (ULM_LAT + 0.001, ULM_LON)])
    )
    pos = build_position_report(cand, road)
    assert pos is not None
    assert len(pos["route_latlon"]) == 2


@pytest.mark.parametrize("bad_crs", ["", "None", "not-a-crs"])
def test_position_report_returns_none_for_unusable_crs(bad_crs: str) -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 3))
    broken = RoadGraph(
        graph=road.graph,
        polylines=road.polylines,
        edge_keys=road.edge_keys,
        crs=bad_crs,
    )
    assert build_position_report(cand, broken) is None


def test_position_report_returns_none_for_nonfinite_trajectory() -> None:
    road = _road_with_one_street()
    traj = _project([(ULM_LAT, ULM_LON), (ULM_LAT + 0.001, ULM_LON)])
    traj[1, 0] = np.nan
    cand = _make_candidate(road, traj)
    assert build_position_report(cand, road) is None


def test_position_report_returns_none_for_empty_trajectory() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, np.zeros((0, 2)))
    assert build_position_report(cand, road) is None


def test_position_report_includes_map_urls_with_coords() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 2))
    pos = build_position_report(cand, road)
    assert pos is not None
    lat_str = f"{pos['latitude']:.6f}"
    assert lat_str in pos["google_maps_url"]
    assert lat_str in pos["openstreetmap_url"]
    assert pos["google_maps_url"].startswith("https://www.google.com/maps?q=")
    assert pos["openstreetmap_url"].startswith("https://www.openstreetmap.org/?mlat=")


def test_position_report_counts_candidates_and_ranking() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 2))
    matches = [{"score_rms_m": 1.0}, {"score_rms_m": 2.0}, {"score_rms_m": 3.0}]
    pos = build_position_report(cand, road, matches=matches, ranking="consensus")
    assert pos is not None
    assert pos["n_candidates"] == 3
    assert pos["ranking"] == "consensus"


def test_position_report_is_json_serializable() -> None:
    import json

    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 5))
    pos = build_position_report(cand, road)
    assert pos is not None
    json.dumps(pos)  # raises TypeError on stray numpy scalars


# ---------------------------------------------------------------------------
# candidate_center_latlon
# ---------------------------------------------------------------------------


def test_candidate_center_latlon_is_walk_centroid() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 2))
    latlon = candidate_center_latlon(cand, road)
    assert latlon is not None
    lat, lon = latlon
    # Walk spans ULM_LAT .. ULM_LAT+0.002 at constant lon.
    assert lat == pytest.approx(ULM_LAT + 0.001, abs=1e-4)
    assert lon == pytest.approx(ULM_LON, abs=1e-4)


def test_candidate_center_latlon_none_without_crs() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 2))
    broken = RoadGraph(
        graph=road.graph,
        polylines=road.polylines,
        edge_keys=road.edge_keys,
        crs="",
    )
    assert candidate_center_latlon(cand, broken) is None


# ---------------------------------------------------------------------------
# classify_confidence
# ---------------------------------------------------------------------------


def _cand_with_scores(score: float, corr: float) -> MatchCandidate:
    poly = np.array([[0.0, 0.0], [1.0, 1.0]])
    return MatchCandidate(
        score=score,
        bearing_corr=corr,
        start_node="A",
        walk=[],
        walk_xy=poly,
        aligned_traj_xy=poly,
        walk_length_m=1.0,
    )


@pytest.mark.parametrize(
    ("rms", "corr", "expected"),
    [
        (150.0, 0.50, "high"),     # both thresholds inclusive
        (100.0, 0.80, "high"),
        (151.0, 0.50, "medium"),   # rms just over the high bar
        (150.0, 0.49, "medium"),   # corr just under the high bar
        (400.0, 0.25, "medium"),   # low boundary is exclusive
        (100.0, 0.20, "low"),      # bad corr alone is enough
        (500.0, 0.80, "low"),      # bad rms alone is enough
        (500.0, 0.10, "low"),
    ],
)
def test_confidence_levels(rms: float, corr: float, expected: str) -> None:
    out = classify_confidence(_cand_with_scores(rms, corr))
    assert out["level"] == expected
    assert out["score_rms_m"] == pytest.approx(rms, abs=0.1)
    assert out["bearing_corr"] == pytest.approx(corr, abs=1e-3)


def test_confidence_attaches_consensus_margin_and_sliding_ratio() -> None:
    matches = [
        {"consensus_score": 3.5, "sliding_window_support_ratio": 0.75},
        {"consensus_score": 6.0},
    ]
    out = classify_confidence(_cand_with_scores(100.0, 0.6), matches)
    assert out["consensus_margin"] == pytest.approx(2.5)
    assert out["sliding_window_support_ratio"] == pytest.approx(0.75)


def test_confidence_omits_margin_without_consensus_scores() -> None:
    out = classify_confidence(
        _cand_with_scores(100.0, 0.6), [{"score_rms_m": 1.0}, {"score_rms_m": 2.0}]
    )
    assert "consensus_margin" not in out
    assert "sliding_window_support_ratio" not in out


def test_confidence_single_match_has_no_margin() -> None:
    out = classify_confidence(
        _cand_with_scores(100.0, 0.6), [{"consensus_score": 3.5}]
    )
    assert "consensus_margin" not in out


# ---------------------------------------------------------------------------
# format_position_summary
# ---------------------------------------------------------------------------


def test_format_summary_contains_the_essentials() -> None:
    road = _road_with_one_street()
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 3))
    pos = build_position_report(cand, road, matches=[{}, {}], ranking="consensus")
    assert pos is not None
    text = format_position_summary(pos)
    assert f"{pos['latitude']:.6f}" in text
    assert f"{pos['longitude']:.6f}" in text
    assert "Olgastraße" in text
    assert pos["confidence"]["level"] in text
    assert pos["google_maps_url"] in text
    assert "consensus" in text


def test_format_summary_handles_unnamed_roads() -> None:
    road = _road_with_one_street(street="x")
    cand = _make_candidate(road, _project([(ULM_LAT, ULM_LON)] * 3))
    pos = build_position_report(cand, road)
    assert pos is not None
    pos["street_names"] = []
    assert "(unnamed roads)" in format_position_summary(pos)


# ---------------------------------------------------------------------------
# Output contract: anchored answer is the HEADLINE, matcher pick is always
# reported alongside (src.pipeline._final_position_reports)
# ---------------------------------------------------------------------------


def _matcher_and_anchored(road, shift_deg: float = 0.005):
    """A matcher candidate plus an anchor-primary translated copy."""
    import dataclasses

    matcher = _make_candidate(
        road,
        _project([
            (ULM_LAT, ULM_LON),
            (ULM_LAT + 0.001, ULM_LON + 0.001),
            (ULM_LAT + 0.002, ULM_LON + 0.002),
        ]),
    )
    anchored = dataclasses.replace(
        matcher,
        aligned_traj_xy=_project([
            (ULM_LAT + shift_deg, ULM_LON),
            (ULM_LAT + shift_deg + 0.001, ULM_LON + 0.001),
            (ULM_LAT + shift_deg + 0.002, ULM_LON + 0.002),
        ]),
    )
    return matcher, anchored


def test_headline_is_anchored_route_when_anchor_fired() -> None:
    """CRITICAL contract: when anchor-primary fired, result['position']
    (headline lat/lon, google_maps_url) must describe the ANCHORED
    route, not the raw matcher pick — the old code reported the matcher
    pick and buried the accuracy win in a side field."""
    from src.pipeline import _final_position_reports

    road = _road_with_one_street()
    matcher, anchored = _matcher_and_anchored(road)
    prior = (ULM_LAT + 0.006, ULM_LON + 0.001)

    headline, matcher_pos = _final_position_reports(
        [matcher], road, matches=[{}], ranking="consensus(shape+vpr)",
        anchored_cand=anchored, anchor_origin="vpr", prior_latlon=prior,
    )
    assert headline is not None and matcher_pos is not None
    # Headline = anchored start, NOT the matcher start.
    assert headline["latitude"] == pytest.approx(ULM_LAT + 0.005, abs=1e-4)
    assert matcher_pos["latitude"] == pytest.approx(ULM_LAT, abs=1e-4)
    assert headline["latitude"] != matcher_pos["latitude"]
    # Source labels per the contract.
    assert headline["source"] == "anchor_primary_vpr"
    assert matcher_pos["source"] == "matcher"
    assert headline["prior_latlon"] == [pytest.approx(prior[0]),
                                        pytest.approx(prior[1])]
    # The shareable link must reflect the headline coordinates.
    assert f"{headline['latitude']:.6f}" in headline["google_maps_url"]


def test_headline_source_reflects_vlm_origin() -> None:
    from src.pipeline import _final_position_reports

    road = _road_with_one_street()
    matcher, anchored = _matcher_and_anchored(road)
    headline, _ = _final_position_reports(
        [matcher], road, matches=[{}], ranking="shape",
        anchored_cand=anchored, anchor_origin="vlm",
        prior_latlon=(ULM_LAT, ULM_LON),
    )
    assert headline is not None
    assert headline["source"] == "anchor_primary_vlm"
    assert headline["ranking"] == "anchored(vlm)"


def test_headline_is_matcher_pick_without_anchor() -> None:
    from src.pipeline import _final_position_reports

    road = _road_with_one_street()
    matcher, _ = _matcher_and_anchored(road)
    headline, matcher_pos = _final_position_reports(
        [matcher], road, matches=[{}], ranking="shape",
    )
    assert headline is matcher_pos
    assert headline is not None
    assert headline["source"] == "matcher"
    assert headline["latitude"] == pytest.approx(ULM_LAT, abs=1e-4)


def test_headline_falls_back_when_anchored_report_fails() -> None:
    """An anchored candidate whose trajectory can't be converted (NaN)
    must not lose the answer: fall back to the matcher report."""
    import dataclasses

    from src.pipeline import _final_position_reports

    road = _road_with_one_street()
    matcher, anchored = _matcher_and_anchored(road)
    bad_traj = np.asarray(anchored.aligned_traj_xy, dtype=np.float64).copy()
    bad_traj[0, 0] = np.nan
    anchored = dataclasses.replace(anchored, aligned_traj_xy=bad_traj)
    headline, matcher_pos = _final_position_reports(
        [matcher], road, matches=[{}], ranking="shape",
        anchored_cand=anchored, anchor_origin="vpr",
        prior_latlon=(ULM_LAT, ULM_LON),
    )
    assert headline is matcher_pos
    assert headline is not None
    assert headline["source"] == "matcher"
