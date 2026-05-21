"""Quantitative evaluation of localization candidates against ground
truth — a list of street names that the video is known to traverse.

We define the "ground-truth distance" of a candidate walk as the
minimum Euclidean distance (in projected meters) from any point on the
candidate's polyline to any edge geometry of any ground-truth street
in the OSM road graph. A candidate that traverses one of the
ground-truth streets gets a distance of ~0; one that's a few hundred
meters off gets a proportional distance.

We also report whether the candidate's named edges *overlap* the GT
street name list — name-overlap is a stricter test that rewards
walks that explicitly include any of the GT streets, even if the
geometric distance happens to be similar.

This module is consumed by the pipeline when `--ground-truth` is
provided, and prints a per-candidate evaluation table after the
shape/aerial comparison.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .osm_data import RoadGraph
from .trajectory_matching import MatchCandidate


# German ß ↔ ss is one of the few mappings ASCII folding doesn't handle
# (NFKD leaves ß intact). Most other Latin-1 diacritics fold cleanly
# via NFKD + ASCII-encode, so this short translation table is the only
# special case we need for the OSM street names we'll see in practice.
_ASCII_FOLD_MAP = str.maketrans({
    "ß": "ss", "ẞ": "SS",
    "œ": "oe", "Œ": "OE",
    "æ": "ae", "Æ": "AE",
    "ø": "o",  "Ø": "O",
    "ł": "l",  "Ł": "L",
})


def _normalize_street_name(name: str) -> str:
    """Casefold + strip diacritics + ASCII-fold so 'Olgastrasse' matches
    'Olgastraße', 'Cœur' matches 'Coeur', etc.

    Without this, ``--ground-truth Olgastrasse`` does not match an OSM
    edge named 'Olgastraße' and the entire ``on_gt_street`` flag is
    always False — yielding the misleading "first_named_rank=None" we
    saw on the Ulm run even though six of ten candidates physically
    traversed Olgastraße (distance = 0 m).
    """
    if not name:
        return ""
    folded = name.translate(_ASCII_FOLD_MAP)
    nfkd = unicodedata.normalize("NFKD", folded)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_only.casefold()


@dataclass
class GroundTruthEval:
    """Result of evaluating one candidate against the GT street set."""
    candidate_index: int
    nearest_distance_m: float        # min distance walk → any GT geometry
    on_gt_street: bool               # candidate has any edge whose name is GT
    matching_gt_names: list[str]     # GT street names hit (if any)


def _gt_polylines(road: RoadGraph, gt_streets: list[str]) -> list[np.ndarray]:
    """Collect every edge geometry whose `name` matches any GT entry.

    Name matching uses :func:`_normalize_street_name` so the user can
    pass ASCII transliterations (``Olgastrasse``) and still hit OSM
    edges named with the original diacritics (``Olgastraße``).
    """
    gt_normalized = [_normalize_street_name(s) for s in gt_streets]
    polys: list[np.ndarray] = []
    for (u, v, k), poly in zip(road.edge_keys, road.polylines):
        d = road.graph.edges[u, v, k]
        name = d.get("name")
        if isinstance(name, list):
            names = [_normalize_street_name(str(n)) for n in name]
        elif name:
            names = [_normalize_street_name(str(name))]
        else:
            continue
        for gt in gt_normalized:
            if gt and any(gt in n for n in names):
                polys.append(poly)
                break
    return polys


def _segment_to_polyline_distance(p: np.ndarray, polyline: np.ndarray) -> float:
    """Min distance from a single point to any segment of the polyline."""
    if len(polyline) < 2:
        if len(polyline) == 1:
            return float(np.linalg.norm(polyline[0] - p))
        return float("inf")
    seg_a = polyline[:-1]
    seg_b = polyline[1:]
    seg_v = seg_b - seg_a
    seg_len_sq = (seg_v ** 2).sum(axis=1)
    seg_len_sq = np.maximum(seg_len_sq, 1e-9)
    pa = p - seg_a
    t = np.clip((pa * seg_v).sum(axis=1) / seg_len_sq, 0.0, 1.0)
    proj = seg_a + t[:, None] * seg_v
    d = np.linalg.norm(proj - p, axis=1)
    return float(d.min())


def _polyline_to_polyline_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Min point-to-segment distance from any point of `a` to any segment of `b`."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return min(_segment_to_polyline_distance(p, b) for p in a)


def _candidate_gt_names(
    cand: MatchCandidate, road: RoadGraph, gt_streets: list[str]
) -> list[str]:
    """Return the GT street names (in the user's exact spelling) that
    appear in the candidate's walk, matched via normalized comparison.
    """
    gt_normalized = [_normalize_street_name(s) for s in gt_streets]
    hits: set[str] = set()
    for (u, v, k) in cand.walk:
        d = road.graph.edges[u, v, k]
        name = d.get("name")
        if isinstance(name, list):
            names = [str(n) for n in name]
        elif name:
            names = [str(name)]
        else:
            continue
        for n in names:
            ln = _normalize_street_name(n)
            for gt, raw in zip(gt_normalized, gt_streets):
                if gt and gt in ln:
                    hits.add(raw)
    return sorted(hits)


def evaluate_candidates(
    candidates: list[MatchCandidate],
    road: RoadGraph,
    gt_streets: list[str],
) -> list[GroundTruthEval]:
    """For each candidate, compute distance to GT and name-overlap."""
    gt_polys = _gt_polylines(road, gt_streets)
    if not gt_polys:
        # Useful diagnostic — caller may have misspelled the street names.
        return [
            GroundTruthEval(i, float("inf"), False, [])
            for i in range(len(candidates))
        ]

    results: list[GroundTruthEval] = []
    for i, cand in enumerate(candidates):
        # Distance from the walk polyline to each GT polyline; take the min.
        dmin = float("inf")
        for gt_poly in gt_polys:
            d = _polyline_to_polyline_distance(cand.walk_xy, gt_poly)
            if d < dmin:
                dmin = d
        names = _candidate_gt_names(cand, road, gt_streets)
        results.append(GroundTruthEval(
            candidate_index=i,
            nearest_distance_m=dmin,
            on_gt_street=bool(names),
            matching_gt_names=names,
        ))
    return results


def best_rank_for_gt(
    results: list[GroundTruthEval],
) -> tuple[int | None, int | None]:
    """Among all candidates, return:
      - rank (1-based) of the closest-to-GT candidate by *distance*
      - rank of the first candidate that *names* a GT street (None if none)
    """
    if not results:
        return None, None
    by_dist = sorted(range(len(results)), key=lambda i: results[i].nearest_distance_m)
    best_dist_rank = by_dist[0] + 1

    name_rank = None
    for i, r in enumerate(results):
        if r.on_gt_street:
            name_rank = i + 1
            break
    return best_dist_rank, name_rank
