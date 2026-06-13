"""Calibrated multi-hypothesis output for trajectory localization.

The multi-clip benchmark (scripts/bench_matching.py) established that
trajectory-shape fit (RMS, bearing correlation) is *uncorrelated* with
geographic correctness: a walk can match the VO shape perfectly and sit
on the wrong (parallel / repeated-grid) street. The true route is almost
always *in* the candidate pool, but not reliably at rank #1, and the
winner's own RMS/corr therefore say nothing trustworthy about whether
it's right.

So instead of presenting one over-confident pick, this module:

* collapses the ranked candidate pool into **distinct location
  hypotheses** (candidates whose start positions cluster within a few
  hundred metres are the same answer — many "different" walks just run
  along the same street), and
* derives a **calibrated confidence** from *spatial agreement* — do the
  independently-good-shape candidates concentrate on one place
  (trustworthy) or scatter across the region (a guess)? — rather than
  from the winner's own shape score, which the benchmark showed is not
  predictive.

The pipeline reports the top-N hypotheses with this confidence, turning
"confidently wrong" into "honestly narrowed".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .osm_data import RoadGraph
from .position import xy_to_latlon
from .trajectory_matching import MatchCandidate, candidate_geographic_summary


@dataclass
class LocationHypothesis:
    rank: int                 # 1-based, by the candidates' incoming order
    candidate_index: int      # representative candidate (best-ranked in cluster)
    lat: float
    lon: float
    score: float              # representative candidate's shape RMS (lower=better)
    support: int              # how many pool candidates fall in this cluster
    street_names: list[str]


def _starts_xy(candidates: list[MatchCandidate]) -> np.ndarray:
    """Start (first aligned-trajectory point) of each candidate, metric xy."""
    return np.array([np.asarray(c.aligned_traj_xy, dtype=np.float64)[0]
                     for c in candidates])


def cluster_candidates(
    candidates: list[MatchCandidate], *, radius_m: float = 150.0
) -> list[list[int]]:
    """Greedily cluster candidates by start position, in incoming order.

    Candidates must already be in ranked (best-first) order; each cluster
    is therefore seeded by its best-ranked member. Returns a list of
    clusters (lists of candidate indices), ordered by best-ranked member.
    """
    if not candidates:
        return []
    starts = _starts_xy(candidates)
    centers: list[np.ndarray] = []
    clusters: list[list[int]] = []
    for i, s in enumerate(starts):
        placed = False
        for c_idx, center in enumerate(centers):
            if np.linalg.norm(s - center) <= radius_m:
                clusters[c_idx].append(i)
                placed = True
                break
        if not placed:
            centers.append(s)
            clusters.append([i])
    clusters.sort(key=min)   # by best-ranked (lowest index) member
    return clusters


def distinct_hypotheses(
    candidates: list[MatchCandidate],
    road: RoadGraph,
    *,
    radius_m: float = 150.0,
    top_n: int = 5,
) -> list[LocationHypothesis]:
    """Top-N distinct location hypotheses from a ranked candidate pool."""
    crs = road.crs
    hyps: list[LocationHypothesis] = []
    for rank, members in enumerate(cluster_candidates(candidates, radius_m=radius_m), 1):
        rep = members[0]                       # best-ranked member of the cluster
        cand = candidates[rep]
        start_xy = np.asarray(cand.aligned_traj_xy, dtype=np.float64)[:1]
        try:
            latlon = xy_to_latlon(start_xy, crs)[0]
        except Exception:
            continue
        hyps.append(LocationHypothesis(
            rank=rank,
            candidate_index=rep,
            lat=float(latlon[0]),
            lon=float(latlon[1]),
            score=float(cand.score),
            support=len(members),
            street_names=candidate_geographic_summary(cand, road.graph)["street_names"][:3],
        ))
        if len(hyps) >= top_n:
            break
    return hyps


def _median_pairwise(starts: np.ndarray) -> float:
    if len(starts) < 2:
        return 0.0
    d = []
    for i in range(len(starts)):
        diff = starts[i + 1:] - starts[i]
        if len(diff):
            d.append(np.linalg.norm(diff, axis=1))
    return float(np.median(np.concatenate(d))) if d else 0.0


def hypothesis_confidence(
    candidates: list[MatchCandidate],
    hyps: list[LocationHypothesis],
    *,
    top_k: int = 10,
    radius_m: float = 150.0,
) -> dict:
    """Confidence from spatial AGREEMENT of the top-k candidates.

    Signals (none derived from the winner's own RMS/corr, which the
    benchmark showed don't predict correctness):

    * ``concentration`` — fraction of the top-k candidates whose start
      falls within ``radius_m`` of the #1 hypothesis. High = the good
      candidates agree on one place.
    * ``spread_m`` — median pairwise distance among the top-k starts.
      Low = tight consensus; high = scattered guesses.
    * ``support`` — how many pool candidates back the #1 hypothesis.

    Level: ``high`` when the top candidates concentrate tightly on #1,
    ``low`` when they scatter across the region, else ``medium``.
    """
    if not hyps:
        return {"level": "low", "concentration": 0.0, "spread_m": None, "support": 0}
    top = candidates[:top_k]
    starts = _starts_xy(top)
    h1 = np.array(_starts_xy([candidates[hyps[0].candidate_index]])[0])
    within = np.linalg.norm(starts - h1, axis=1) <= radius_m
    concentration = float(within.mean())
    spread = _median_pairwise(starts)

    if concentration >= 0.5 and spread <= 300.0:
        level = "high"
    elif concentration < 0.3 or spread > 800.0:
        level = "low"
    else:
        level = "medium"

    return {
        "level": level,
        "concentration": round(concentration, 3),
        "spread_m": round(spread, 1),
        "support": hyps[0].support,
        "n_hypotheses": len(hyps),
    }
