"""Match a VO trajectory against the OSM road graph of a known city.

Strategy: at every candidate start node we enumerate a handful of
forward walks long enough to cover the trajectory, then score each
walk by how well its shape aligns with the trajectory under a
similarity transform (Procrustes).

Procrustes is the right scoring function here: monocular VO loses
metric scale and we don't know which way the car was pointing at
frame 0, so the trajectory and the road-walk are equal up to a 2D
similarity (rotation + translation + uniform scale). Procrustes finds
that transform optimally and returns the residual — RMS distance per
sample after alignment.

We do a cheap pre-filter on bearing-signature correlation so we don't
do the full Procrustes on every walk in the city.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np
from skimage.transform import SimilarityTransform
from tqdm import tqdm

from .osm_data import RoadGraph, walk_to_polyline, walks_from_node
from .visual_odometry import bearing_signature, resample_uniform, trajectory_arc_length


@dataclass
class MatchCandidate:
    score: float                 # lower is better — RMS residual after alignment
    bearing_corr: float          # higher is better — pre-filter score
    start_node: int
    walk: list[tuple]            # list of (u, v, k) edges
    walk_xy: np.ndarray          # the walk as a polyline in metric coords
    aligned_traj_xy: np.ndarray  # the trajectory after similarity alignment
    walk_length_m: float


@dataclass
class SlidingWindowMatchResult:
    candidate_index: int
    n_windows: int
    support_count: int
    support_ratio: float
    mean_rank: float
    mean_score_rms_m: float
    sliding_score: float


def procrustes_similarity(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Best similarity transform `src → dst`.

    Thin wrapper around `skimage.transform.SimilarityTransform.estimate`,
    which solves the same Procrustes problem (rotation + uniform scale +
    translation, no reflection) using a numerically-stable SVD.

    Returns `(R, s, residual_rms, src_aligned)` where `R` is the 2x2
    rotation, `s` is the uniform scale, `residual_rms` is the RMS
    Euclidean error per point after alignment, and `src_aligned` is
    `src` after the estimated transform was applied. The first three
    are returned for backwards compatibility with callers that expect
    them; new code can call `SimilarityTransform` directly.
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 2

    # `from_estimate` is the post-0.26 scikit-image API; falls back to
    # the older `estimate()` if it isn't available yet.
    if hasattr(SimilarityTransform, "from_estimate"):
        tform = SimilarityTransform.from_estimate(src, dst)
        if tform is None:
            return np.eye(2), 1.0, float("inf"), src.copy()
    else:  # pragma: no cover - older scikit-image
        tform = SimilarityTransform()
        if not tform.estimate(src, dst):
            return np.eye(2), 1.0, float("inf"), src.copy()

    src_aligned = tform(src)
    residual = float(np.sqrt(((src_aligned - dst) ** 2).sum(axis=1).mean()))

    # Decompose the 3x3 homogeneous matrix into the (R, s) the rest of
    # the pipeline expects.
    M = tform.params  # 3x3
    # SimilarityTransform's upper-left 2x2 is `s * R`. Recover scale and R.
    scale = float(np.sqrt(np.linalg.det(M[:2, :2])))
    if scale < 1e-12:
        return np.eye(2), 1.0, float("inf"), src.copy()
    R = M[:2, :2] / scale
    return R, scale, residual, src_aligned


def procrustes_fixed_scale(
    src: np.ndarray, dst: np.ndarray, scale: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Best rotation+translation of `src → dst` at a *fixed* uniform scale.

    Unlike :func:`procrustes_similarity`, the scale is not free — it's
    pinned to ``scale`` (e.g. the metric-per-VO-unit factor implied by a
    trustworthy route-length prior). Only rotation and translation are
    optimised (orientation-preserving Kabsch; the optimal rotation is
    independent of the fixed scale). This stops the alignment from
    *shrinking* a drifty VO path onto a compact decoy walk — the
    compression that left the localized route unable to reach its far
    end — by forcing it to span the prescribed metric extent.

    Returns ``(residual_rms, src_aligned, R)`` where ``R`` is the 2x2
    orientation-preserving rotation (callers can reuse it to re-place the
    path under a different translation, e.g. an anchor pin).
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 2
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                      # forbid reflection
        Vt = Vt.copy(); Vt[-1] *= -1
        R = Vt.T @ U.T
    src_aligned = scale * (src_c @ R.T) + dst.mean(axis=0)
    residual = float(np.sqrt(((src_aligned - dst) ** 2).sum(axis=1).mean()))
    return residual, src_aligned, R


def anchor_pinned_route(
    vo_xy: np.ndarray,
    walk_xy: np.ndarray,
    locked_scale: float,
    anchor_vo_xy: np.ndarray,
    anchor_world_xy: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    n_samples: int = 128,
) -> np.ndarray:
    """Scale-locked route whose translation is fit to the anchors.

    Combines the three trustworthy pieces: the metric **scale** (locked),
    the **rotation** from the shape fit of the VO path onto the matched
    walk, and the **position** from the OCR anchors. With the scale and
    rotation fixed, the only free parameter is the translation ``t``; its
    least-squares optimum over the anchor correspondences is the
    (confidence-)weighted mean of ``world_i - s·R·vo_i``. One anchor →
    an exact pin; several → a balanced fit that spreads the residual
    across them instead of nailing one point and letting the rest drift.

    ``anchor_vo_xy`` / ``anchor_world_xy`` are ``(N, 2)`` (a single
    ``(2,)`` pair is accepted too). Returns the full ``vo_xy`` mapped into
    world (projected-metre) coordinates.
    """
    vo_xy = np.asarray(vo_xy, dtype=np.float64)
    a_vo = np.asarray(anchor_vo_xy, dtype=np.float64).reshape(-1, 2)
    a_world = np.asarray(anchor_world_xy, dtype=np.float64).reshape(-1, 2)
    vr = resample_uniform(vo_xy, n_samples)
    wr = resample_uniform(np.asarray(walk_xy, dtype=np.float64), n_samples)
    _resid, _aligned, R = procrustes_fixed_scale(vr, wr, locked_scale)
    # Each anchor implies a translation t_i = world_i - s·R·vo_i. A valid
    # (clean) correspondence — the car was actually at the geocoded spot
    # at that time — agrees with the others; a bad one (a direction sign,
    # a distant/coarse POI) gives an outlier t_i. So keep the densest
    # cluster of t_i and average that, rejecting the rest. This is what
    # makes the pin robust to the noisy anchors that naive averaging let
    # in (Ulm: naive 307 m, this ~140 m).
    residuals = a_world - locked_scale * (a_vo @ R.T)        # (N, 2) candidate t's
    w = (np.asarray(weights, dtype=np.float64)
         if weights is not None and len(weights) == len(residuals) else None)
    if len(residuals) >= 3:
        from .text_anchor import select_anchor_cluster
        keep = select_anchor_cluster(residuals, w, radius_m=150.0)
        residuals = residuals[keep]
        w = w[keep] if w is not None else None
    if w is not None and w.sum() > 1e-9:
        t = (residuals * w[:, None]).sum(axis=0) / w.sum()
    else:
        t = residuals.mean(axis=0)
    return locked_scale * (vo_xy @ R.T) + t


def _bearing_corr(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Pearson correlation of two bearing signatures, clipped to [-1, 1]."""
    if len(sig_a) != len(sig_b) or len(sig_a) < 2:
        return 0.0
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip((a @ b) / (na * nb), -1.0, 1.0))


def _candidate_starts(road: RoadGraph, every: int = 1) -> list[int]:
    """Subset of nodes worth treating as walk roots.

    Drop degree-1 dead ends (cars don't usually start a journey there in
    a city graph). With `every>1`, additionally subsample for speed.
    """
    nodes = []
    for n, deg in road.graph.degree():
        if deg >= 2:
            nodes.append(n)
    nodes.sort()
    if every > 1:
        nodes = nodes[::every]
    return nodes


def match_trajectory(
    traj_xz: np.ndarray,
    road: RoadGraph,
    *,
    n_samples: int = 128,
    # Walk budget per start node. The enumeration generates greedy +
    # one-turn + two-turn walks (~29 per first edge); a budget much
    # smaller than that silently discards the multi-turn walks that real
    # urban routes need (the Ulm GT route is a two-turn walk).
    walks_per_node: int = 40,
    walk_depth: int = 40,
    bearing_top_k: int = 500,
    final_top_k: int = 5,
    sample_every: int = 1,
    estimated_length_m: float = 1500.0,
    progress: bool = True,
    bearing_corr_weight: float = 400.0,
    extra_start_nodes: list | None = None,
    restrict_to_start_nodes: bool = False,
    locked_scale: float | None = None,
) -> list[MatchCandidate]:
    """Localize the trajectory: return up to `final_top_k` best walks.

    Parameters
    ----------
    traj_xz:
        VO trajectory in arbitrary 2D coords (Nx2).
    road:
        Projected OSM graph (in metric units).
    estimated_length_m:
        How long, in meters, we think the driven path is. The VO
        trajectory has unknown scale, so we prescribe this as the
        length of the OSM walks we generate. A few hundred meters too
        long or short still scores correctly because Procrustes
        rescales — this is mainly a knob to keep walk enumeration
        bounded.
    extra_start_nodes:
        Additional walk-root nodes to enumerate from, on top of the
        graph-wide candidate starts. Used to *seed* enumeration around
        absolute anchors (e.g. OCR-geocoded POIs) so the anchored area
        is in the pool even when trajectory drift would exclude it.
    restrict_to_start_nodes:
        When True, enumerate *only* from ``extra_start_nodes`` (the
        anchor vicinity) instead of unioning them into the graph-wide
        scan. This turns a trustworthy absolute anchor into a hard
        spatial gate: the wrong-district walks that drift makes
        shape-match better are excluded from the pool entirely, so shape
        ranks only *within* the anchored area. No-op (falls back to the
        full scan) if no start nodes are supplied.
    """
    # Degenerate input — empty, a single point, or a stationary segment
    # (VO emitted identical poses, zero total arc length) — cannot be
    # matched; return "no candidates" instead of crashing in
    # resample_uniform. This must run unconditionally: the previous
    # guard only fired for len(traj_xz) < n_samples, so a 300-identical-
    # pose trajectory blew up after the expensive VO/graph stages.
    if len(traj_xz) < 2 or trajectory_arc_length(traj_xz)[-1] <= 0:
        return []

    traj_resampled = resample_uniform(traj_xz, n_samples)
    traj_sig = bearing_signature(traj_xz, n_samples=n_samples)

    # Scale the walk-depth (edge count) cap with the length prior. A
    # fixed cap of 40 edges silently empties the pool once the prior
    # exceeds ~5 km in a dense grid: with ~60-80 m edges, 40 edges cover
    # only 2.4-3.2 km, so every walk fails the 0.5*estimated_length_m
    # filter below and dense-center start nodes contribute NOTHING —
    # truncated walks are dropped, not scored. Derive the depth needed
    # to physically reach 1.5x the target at the graph's median edge
    # length (computed once), never below the caller's `walk_depth` and
    # capped at 150 to keep enumeration bounded.
    edge_lengths = np.array([
        float(d.get("length", 0.0))
        for _, _, d in road.graph.edges(data=True)
    ], dtype=np.float64)
    edge_lengths = edge_lengths[edge_lengths > 0]
    if len(edge_lengths):
        median_edge_m = float(np.median(edge_lengths))
        needed = int(np.ceil(1.5 * estimated_length_m / median_edge_m))
        walk_depth = int(min(150, max(walk_depth, needed)))

    seed = [n for n in (extra_start_nodes or []) if n in road.graph]
    if restrict_to_start_nodes and seed:
        # Hard spatial gate: enumerate only from the anchor vicinity.
        starts = list(dict.fromkeys(seed))
    elif seed:
        # Union in the seed nodes (dedup, keep graph nodes only).
        starts = list(dict.fromkeys(list(_candidate_starts(road, every=sample_every)) + seed))
    else:
        starts = _candidate_starts(road, every=sample_every)
    if progress:
        starts_iter = tqdm(starts, desc="enumerating walks", unit="node")
    else:
        starts_iter = starts

    # ---- Stage 1: cheap bearing-correlation filter ----
    rough: list[tuple[float, int, list[tuple]]] = []
    for start in starts_iter:
        walks = walks_from_node(
            road.graph,
            start,
            target_length_m=estimated_length_m,
            max_walks=walks_per_node,
            max_depth=walk_depth,
        )
        for walk in walks:
            poly = walk_to_polyline(road.graph, walk)
            if len(poly) < 2:
                continue
            walk_len = float(trajectory_arc_length(poly)[-1])
            if walk_len < 0.5 * estimated_length_m:
                continue
            try:
                sig = bearing_signature(poly, n_samples=n_samples)
            except ValueError:
                continue
            corr = _bearing_corr(traj_sig, sig)
            rough.append((corr, start, walk))

    rough.sort(key=lambda r: -r[0])
    rough = rough[:bearing_top_k]

    # ---- Stage 2: Procrustes alignment on the survivors ----
    candidates: list[MatchCandidate] = []
    for corr, start, walk in rough:
        poly = walk_to_polyline(road.graph, walk)
        try:
            poly_resampled = resample_uniform(poly, n_samples)
        except ValueError:
            continue
        if locked_scale is not None:
            residual, traj_aligned, _R = procrustes_fixed_scale(
                traj_resampled, poly_resampled, locked_scale)
        else:
            _, _, residual, traj_aligned = procrustes_similarity(
                traj_resampled, poly_resampled)
        # A failed/degenerate alignment reports an inf (or NaN) residual
        # and an UNALIGNED trajectory. Letting it through corrupts the
        # ranking and every downstream consumer of aligned_traj_xy
        # (hypothesis starts, spread_m, evaluator start error) — drop it.
        if not np.isfinite(residual):
            continue
        walk_len = float(trajectory_arc_length(poly)[-1])
        candidates.append(
            MatchCandidate(
                score=residual,
                bearing_corr=corr,
                start_node=start,
                walk=walk,
                walk_xy=poly,
                aligned_traj_xy=traj_aligned,
                walk_length_m=walk_len,
            )
        )

    # Combined ranking: Procrustes RMS is the primary geometric fit but
    # bearing correlation captures *orientation* agreement, which matters
    # when several candidate walks have similar RMS but different
    # heading-pattern alignment with the trajectory. Empirically (from
    # GT-evaluated runs on the Ulm clip) the correct walk is consistently
    # among the highest-bearing-correlation candidates even when its RMS
    # is mid-pack — multiple parallel streets fit the shape similarly,
    # but only one matches the turn pattern. A composite score
    #
    #     score = RMS - bearing_corr_weight * bearing_corr
    #
    # promotes high-correlation candidates without ignoring RMS. The
    # default weight (bearing_corr_weight=400 m per unit corr, above) makes
    # going from corr=0.20 to corr=0.35 worth 0.15*400 = 60 m of RMS — the
    # typical spread of "right-area" candidates.
    def _combined(c: MatchCandidate) -> float:
        return c.score - bearing_corr_weight * c.bearing_corr

    candidates.sort(key=_combined)
    return candidates[:final_top_k]


def candidate_geographic_summary(
    cand: MatchCandidate, graph: nx.MultiDiGraph
) -> dict:
    """Pull a small dict describing the candidate (street names + lat/lon
    of start node) that's safe to dump to JSON."""
    start_data = graph.nodes[cand.start_node]
    streets: list[str] = []
    for (u, v, k) in cand.walk:
        d = graph.edges[u, v, k]
        name = d.get("name")
        if isinstance(name, list):
            for n in name:
                if n and n not in streets:
                    streets.append(str(n))
        elif name and str(name) not in streets:
            streets.append(str(name))

    # OSMnx-projected graphs keep `x`/`y` in projected meters and stash
    # original lon/lat under `lon`/`lat` only in some versions; fall back
    # by un-projecting via graph CRS only if needed. For the JSON we just
    # report the projected coords plus any street names — they're
    # geocodable and unambiguous.
    return {
        "score_rms_m": cand.score,
        "bearing_corr": cand.bearing_corr,
        "walk_length_m": cand.walk_length_m,
        "start_node_xy": [float(start_data.get("x", 0.0)), float(start_data.get("y", 0.0))],
        "street_names": streets[:10],
        "n_edges": len(cand.walk),
    }


def _window_slices(n_points: int, window_size: int, step: int) -> list[tuple[int, int]]:
    if n_points < 2 or window_size < 2 or step < 1:
        return []
    if n_points <= window_size:
        return [(0, n_points)]
    windows = []
    start = 0
    while start + window_size <= n_points:
        windows.append((start, start + window_size))
        start += step
    if windows[-1][1] < n_points:
        windows.append((n_points - window_size, n_points))
    return windows


def _candidate_street_names(cand: MatchCandidate, graph: nx.MultiDiGraph) -> set[str]:
    summary = candidate_geographic_summary(cand, graph)
    return {
        str(name).casefold()
        for name in summary.get("street_names", [])
        if str(name).strip()
    }


def _min_polyline_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Min point-to-segment distance between two polylines (symmetric)."""
    # Lazy import: evaluator imports MatchCandidate from this module, so
    # a top-level import would be circular.
    from .evaluator import _polyline_to_polyline_distance
    return min(
        _polyline_to_polyline_distance(a, b),
        _polyline_to_polyline_distance(b, a),
    )


def _candidates_overlap(
    ref: MatchCandidate,
    other: MatchCandidate,
    graph: nx.MultiDiGraph,
    *,
    support_radius_m: float,
    name_radius_m: float = 500.0,
) -> bool:
    """Does the window-match walk `other` support the full-route candidate `ref`?

    Geometric support is the minimum point-to-polyline distance between
    the two walks — NOT centroid-to-centroid, which structurally denied
    support to long routes (a window covering the first quarter of a
    1.5 km route has its centroid ~550 m from the full route's centroid
    *on the same street*). A street-name match only counts when the
    named window walk also lies within ``name_radius_m`` of the
    candidate polyline; bare name equality granted support to any
    same-named street city-wide.
    """
    dist = _min_polyline_distance(ref.walk_xy, other.walk_xy)
    if dist <= support_radius_m:
        return True

    ref_names = _candidate_street_names(ref, graph)
    other_names = _candidate_street_names(other, graph)
    if ref_names and other_names and ref_names.intersection(other_names):
        return dist <= name_radius_m
    return False


def score_candidates_with_sliding_windows(
    traj_xz: np.ndarray,
    road: RoadGraph,
    candidates: list[MatchCandidate],
    *,
    window_size: int = 64,
    step: int = 32,
    resample_points: int | None = None,
    window_top_k: int = 5,
    estimated_length_m: float = 1500.0,
    support_radius_m: float = 250.0,
    match_fn: Callable[..., list[MatchCandidate]] | None = None,
    target_n_windows: int = 12,
) -> list[SlidingWindowMatchResult]:
    """Re-score full-route candidates by their support across trajectory windows.

    Parameters
    ----------
    resample_points:
        Number of points the trajectory is resampled to before slicing
        into windows. ``None`` (default) auto-sizes so we get roughly
        ``target_n_windows`` windows of length ``window_size``. The
        previous fixed default of 128 produced only 3 windows for a
        7-minute trajectory (``(128 - 64) / 32 + 1 = 3``), which can't
        discriminate between candidates that pass through one part of
        the city vs another. Auto-sizing keeps short trajectories
        cheap while letting long trajectories actually use sliding
        windows for what they're for.
    target_n_windows:
        When ``resample_points`` is ``None``, pick a resample size that
        yields about this many windows. Per-window matching is
        expensive (one full city scan), so this caps the cost; 12
        windows ≈ one per 35 s of a 7-minute clip, comparable to the
        natural rate of intersections in a driven route.
    """
    if not candidates:
        return []
    if match_fn is None:
        match_fn = match_trajectory
    if len(traj_xz) < 2 or trajectory_arc_length(traj_xz)[-1] <= 0:
        return []

    if resample_points is None:
        # Solve (n_points - window_size) / step + 1 ≈ target_n_windows for n_points.
        auto = window_size + max(0, target_n_windows - 1) * step
        # Don't oversample a tiny trajectory.
        n_points = max(window_size, min(auto, len(traj_xz)))
    else:
        n_points = max(resample_points, window_size)
    traj_resampled = resample_uniform(traj_xz, n_points)
    windows = _window_slices(len(traj_resampled), window_size, step)
    if not windows:
        return []

    full_length = float(trajectory_arc_length(traj_resampled)[-1])
    ranks: list[list[int]] = [[] for _ in candidates]
    scores: list[list[float]] = [[] for _ in candidates]

    for start, end in windows:
        window = traj_resampled[start:end]
        window_length = float(trajectory_arc_length(window)[-1])
        if window_length <= 0:
            continue
        length_scale = window_length / max(full_length, 1e-9)
        window_matches = match_fn(
            window,
            road,
            final_top_k=window_top_k,
            estimated_length_m=max(100.0, estimated_length_m * length_scale),
            progress=False,
        )
        if not window_matches:
            continue
        for idx, candidate in enumerate(candidates):
            best_rank: int | None = None
            best_score: float | None = None
            for rank, window_match in enumerate(window_matches, start=1):
                if not _candidates_overlap(
                    candidate,
                    window_match,
                    road.graph,
                    support_radius_m=support_radius_m,
                ):
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_score = window_match.score
            if best_rank is not None and best_score is not None:
                ranks[idx].append(best_rank)
                scores[idx].append(best_score)

    results: list[SlidingWindowMatchResult] = []
    n_windows = len(windows)
    for idx, _candidate in enumerate(candidates):
        support_count = len(ranks[idx])
        support_ratio = support_count / max(1, n_windows)
        mean_rank = float(np.mean(ranks[idx])) if ranks[idx] else float("inf")
        mean_score = float(np.mean(scores[idx])) if scores[idx] else float("inf")
        rank_bonus = 0.0 if not np.isfinite(mean_rank) else (window_top_k - mean_rank + 1) / max(1, window_top_k)
        sliding_score = support_ratio + 0.25 * max(0.0, rank_bonus)
        results.append(
            SlidingWindowMatchResult(
                candidate_index=idx,
                n_windows=n_windows,
                support_count=support_count,
                support_ratio=support_ratio,
                mean_rank=mean_rank,
                mean_score_rms_m=mean_score,
                sliding_score=sliding_score,
            )
        )

    return results
