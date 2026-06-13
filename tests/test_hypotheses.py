"""Tests for the calibrated multi-hypothesis output."""

from __future__ import annotations

import numpy as np

from src.hypotheses import (
    cluster_candidates,
    distinct_hypotheses,
    hypothesis_confidence,
)
from src.trajectory_matching import MatchCandidate


class _FakeGraph:
    """Minimal stand-in so candidate_geographic_summary returns names."""
    def __init__(self):
        self.nodes = {0: {"x": 0.0, "y": 0.0}}
        self._edges = {}

    @property
    def edges(self):
        return self._edges


class _Road:
    # A CRS pyproj accepts; xy here are tiny offsets so lat/lon stay valid.
    crs = "EPSG:32632"  # UTM 32N (metres)

    def __init__(self):
        self.graph = _FakeGraph()


def _cand(start_xy, score, *, names=("X",)) -> MatchCandidate:
    # aligned_traj_xy: a short path starting at start_xy. UTM 32N metres
    # near Karlsruhe so xy_to_latlon yields valid WGS84.
    base = np.array([456000.0, 5428000.0])  # ~Karlsruhe in UTM 32N
    p0 = base + np.asarray(start_xy, dtype=float)
    traj = np.array([p0, p0 + [10.0, 0.0], p0 + [20.0, 5.0]])
    c = MatchCandidate(
        score=score, bearing_corr=0.3, start_node=0, walk=[],
        walk_xy=traj.copy(), aligned_traj_xy=traj, walk_length_m=100.0)
    c._names = list(names)  # not used by the graph stub; summary returns []
    return c


def test_cluster_groups_nearby_starts() -> None:
    cands = [
        _cand((0, 0), 10.0),
        _cand((50, 0), 12.0),       # within 150 m of #0 -> same cluster
        _cand((1000, 0), 11.0),     # far -> own cluster
        _cand((1020, 10), 13.0),    # near #2 -> joins it
    ]
    clusters = cluster_candidates(cands, radius_m=150.0)
    assert len(clusters) == 2
    # Clusters ordered by best-ranked (lowest-index) member.
    assert clusters[0][0] == 0 and 1 in clusters[0]
    assert clusters[1][0] == 2 and 3 in clusters[1]


def test_distinct_hypotheses_dedup_and_rank() -> None:
    cands = [
        _cand((0, 0), 10.0),
        _cand((30, 0), 11.0),       # same place as #0
        _cand((900, 0), 12.0),      # second place
    ]
    road = _Road()
    hyps = distinct_hypotheses(cands, road, radius_m=150.0, top_n=5)
    assert len(hyps) == 2                      # two distinct places
    assert hyps[0].rank == 1 and hyps[0].support == 2
    assert hyps[1].rank == 2 and hyps[1].support == 1
    for h in hyps:
        assert -90 <= h.lat <= 90 and -180 <= h.lon <= 180


def test_confidence_high_when_concentrated() -> None:
    # 9 of 10 candidates start within 150 m of the top hypothesis.
    cands = [_cand((i * 10, 0), 10.0 + i) for i in range(9)]
    cands.append(_cand((3000, 0), 50.0))
    road = _Road()
    hyps = distinct_hypotheses(cands, road, radius_m=150.0)
    conf = hypothesis_confidence(cands, hyps, top_k=10, radius_m=150.0)
    assert conf["level"] == "high"
    assert conf["concentration"] >= 0.5
    assert conf["spread_m"] <= 300.0


def test_confidence_low_when_scattered() -> None:
    # Every candidate starts far from the others -> a guess.
    cands = [_cand((i * 1000, (i % 2) * 1000), 10.0 + i) for i in range(8)]
    road = _Road()
    hyps = distinct_hypotheses(cands, road, radius_m=150.0)
    conf = hypothesis_confidence(cands, hyps, top_k=8, radius_m=150.0)
    assert conf["level"] == "low"
    assert conf["spread_m"] > 800.0


def test_confidence_empty_hyps() -> None:
    conf = hypothesis_confidence([], [])
    assert conf["level"] == "low"
