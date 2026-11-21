"""Experimental evidence/continuity selection across dense retrieval regions.

The constants below are fixed physical/noise heuristics, not calibrated success
probabilities. This prototype is opt-in and must earn adoption on fresh clips.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import operator

import numpy as np

from .geo import LocalFrame, geodesic_distance_m


def _mst_region_edges(centers):
    """Deterministic Prim tree; WGS84 distances avoid longitude-wrap errors."""
    n = len(centers)
    distances = geodesic_distance_m(centers[:, 0, None], centers[:, 1, None],
                                    centers[None, :, 0], centers[None, :, 1])
    visited, edges = {0}, []
    while len(visited) < n:
        _distance, a, b = min((float(distances[a, b]), a, b)
                              for a in sorted(visited) for b in range(n) if b not in visited)
        edges.append((a, b))
        visited.add(b)
    return edges


def _farthest_corridor_point(a, b, centers, radii):
    """Maximise distance outside the disc union on one local metric segment.

    Along a line, each distance-to-disc-boundary function is convex. A maximum
    of their lower envelope therefore occurs at an endpoint or where two
    functions meet. Solving those (at most quadratic) intersections avoids a
    sampling grid that could silently skip a short uncovered interval.
    """
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return None
    if np.any(np.maximum(np.linalg.norm(centers - a, axis=1),
                         np.linalg.norm(centers - b, axis=1)) <= radii):
        return None                         # A single convex disc covers this path.
    relative = (centers - a) / length
    direction = delta / length
    u = relative @ direction
    v2 = np.maximum(np.sum(relative ** 2, axis=1) - u ** 2, 0.0)
    r = radii / length
    c = u ** 2 + v2
    candidates = [0.0, 1.0]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            radius_delta = r[i] - r[j]
            linear = 2.0 * (u[j] - u[i])
            constant = c[i] - c[j]
            if abs(radius_delta) < 1e-12:
                roots = [-constant / linear] if abs(linear) > 1e-12 else []
            else:
                constant -= radius_delta ** 2
                coeff = np.array([
                    linear ** 2 - 4.0 * radius_delta ** 2,
                    2.0 * linear * constant + 8.0 * radius_delta ** 2 * u[j],
                    constant ** 2 - 4.0 * radius_delta ** 2 * c[j],
                ])
                scale = float(np.max(np.abs(coeff)))
                if scale == 0:
                    continue
                coeff[np.abs(coeff) < scale * 1e-12] = 0.0
                roots = np.roots(np.trim_zeros(coeff, "f"))
            for root in roots:
                if abs(np.imag(root)) > 1e-8 or not -1e-9 <= np.real(root) <= 1.0 + 1e-9:
                    continue
                fraction = float(np.clip(np.real(root), 0.0, 1.0))
                left = np.sqrt((fraction - u[i]) ** 2 + v2[i]) - r[i]
                right = np.sqrt((fraction - u[j]) ** 2 + v2[j]) - r[j]
                if abs(left - right) <= 1e-8:
                    candidates.append(fraction)
    fractions = np.unique(candidates)
    clearance = (np.sqrt((fractions[:, None] - u) ** 2 + v2) - r).min(axis=1) * length
    pick = int(clearance.argmax())
    return a + fractions[pick] * delta, float(clearance[pick]), float(fractions[pick])


def fill_region_corridors(regions, max_regions=6, radius_m=3000):
    """Use spare dense-disc budget to bridge video-named/seed regions.

    Original regions, their order and radii are preserved. A deterministic MST
    connects their centers using geodesic edge weights. Its paths are straight
    segments in a local metric projection about the first region; every added
    center lies on one of those fixed paths. At each step choose the point
    farthest outside the existing disc union, accounting for each disc's own
    radius. Stop once all paths are covered (1 cm numerical tolerance), or the
    budget is full. No geocoder, imagery, route labels or GT are accepted.

    If the input already fills/exceeds the budget it is returned intact. This
    city-scale planner rejects invalid WGS84 coordinates/radii and projections
    that cannot represent the provided centers finitely.
    """
    try:
        budget = operator.index(max_regions)
    except TypeError as exc:
        raise ValueError("max_regions must be a positive integer") from exc
    if isinstance(max_regions, (bool, np.bool_)) or budget < 1:
        raise ValueError("max_regions must be a positive integer")
    try:
        added_radius = float(radius_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius_m must be finite and positive") from exc
    if not np.isfinite(added_radius) or added_radius <= 0:
        raise ValueError("radius_m must be finite and positive")

    try:
        originals = list(regions)
    except TypeError as exc:
        raise ValueError("regions must be an iterable of region mappings") from exc
    coords, radii = [], []
    for region in originals:
        if not isinstance(region, Mapping):
            raise ValueError("each region must contain a WGS84 center and positive radius_m")
        try:
            ll, radius = np.asarray(region["center"], dtype=float), float(region["radius_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each region must contain a WGS84 center and positive radius_m") from exc
        if (ll.shape != (2,) or not np.isfinite(ll).all()
                or abs(ll[0]) > 90 or abs(ll[1]) > 180
                or not np.isfinite(radius) or radius <= 0):
            raise ValueError("region centers must be finite WGS84 coordinates and radii positive")
        coords.append(ll)
        radii.append(radius)
    output = deepcopy(originals)
    if len(originals) < 2 or len(originals) >= budget:
        return output

    coords = np.asarray(coords)
    frame = LocalFrame(*coords[0])
    centers = np.column_stack(frame.to_xy(coords[:, 0], coords[:, 1]))
    if not np.isfinite(centers).all():
        raise ValueError("region centers must admit a finite local projection")
    edges = _mst_region_edges(coords)
    fixed_centers = centers.copy()
    radii = np.asarray(radii, dtype=float)
    while len(output) < budget:
        best = None
        for edge_index, (a, b) in enumerate(edges):
            candidate = _farthest_corridor_point(fixed_centers[a], fixed_centers[b], centers, radii)
            if candidate is None:
                continue
            point, clearance, fraction = candidate
            if best is None or clearance > best[0]:
                best = (clearance, point, fraction, edge_index, a, b)
        if best is None or best[0] <= 0.01:
            break
        _clearance, point, fraction, edge_index, a, b = best
        lat, lon = frame.to_latlon(*point)
        if not np.isfinite([lat, lon]).all():
            raise ValueError("corridor center could not be converted to finite WGS84")
        names = [str(originals[i].get("name", f"region {i + 1}")) for i in (a, b)]
        output.append({"name": f"corridor {edge_index + 1}.{len(output) - len(originals) + 1}",
                       "center": [float(lat), float(lon)], "radius_m": added_radius,
                       "origin": "named_corridor", "corridor_between": names,
                       "corridor_fraction": fraction})
        centers = np.vstack([centers, point])
        radii = np.append(radii, added_radius)
    return output


def merge_region_tracks(tracks, timestamps, *, max_speed_mps: float = 55.0,
                        position_tolerance_m: float = 150.0,
                        jump_penalty_per_km: float = 0.05,
                        switch_penalty: float = 0.005):
    """Select a regional candidate per query by cosine and temporal continuity.

    Inputs are the same-frame tracks produced by the same descriptor in each
    region. Dynamic programming maximises summed cosine, minus excess travel
    beyond a generous road-speed/noise allowance and a small switching cost.
    Returning the selected candidates preserves their own similarity evidence.
    Missing/misaligned/invalid tracks are rejected instead of pairing unrelated
    frame indices. A genuine route can cross regions through overlapping discs.
    """
    if not tracks:
        return None, {"reason": "no usable regional tracks"}
    indices = np.asarray(tracks[0][0], dtype=int)
    times = np.asarray(timestamps, dtype=float)
    if (indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0)
            or indices[0] < 0 or indices[-1] >= len(times)):
        raise ValueError("regional query indices must be ordered within the video")
    coords, similarities = [], []
    for idx, ll, sim in tracks:
        ll, sim = np.asarray(ll, dtype=float), np.asarray(sim, dtype=float)
        if (not np.array_equal(indices, np.asarray(idx, dtype=int))
                or ll.shape != (len(indices), 2) or sim.shape != (len(indices),)):
            raise ValueError("regional tracks must describe the same query frames")
        coords.append(ll)
        similarities.append(sim)
    coords, similarities = np.asarray(coords), np.asarray(similarities)
    valid = (np.isfinite(coords).all(axis=2) & np.isfinite(similarities)
             & (np.abs(coords[:, :, 0]) <= 90) & (np.abs(coords[:, :, 1]) <= 180))
    if not valid.any(axis=0).all():
        return None, {"reason": "no finite regional candidate for a query frame"}
    evidence = np.where(valid, similarities, -np.inf)
    n_regions, n_queries = evidence.shape
    score = evidence[:, 0].copy()
    parent = np.zeros((n_queries, n_regions), dtype=int)
    for t in range(1, n_queries):
        dt = max(float(times[indices[t]] - times[indices[t - 1]]), 0.0)
        distance = geodesic_distance_m(
            coords[:, t - 1, 0, None], coords[:, t - 1, 1, None],
            coords[None, :, t, 0], coords[None, :, t, 1])
        excess = np.maximum(distance - (max_speed_mps * dt + position_tolerance_m), 0.0)
        penalty = jump_penalty_per_km * excess / 1000.0
        penalty += switch_penalty * (1 - np.eye(n_regions))
        transition = score[:, None] - np.where(np.isfinite(penalty), penalty, np.inf)
        parent[t] = transition.argmax(axis=0)
        score = evidence[:, t] + transition[parent[t], np.arange(n_regions)]
    selected = np.zeros(n_queries, dtype=int)
    selected[-1] = int(score.argmax())
    for t in range(n_queries - 1, 0, -1):
        selected[t - 1] = parent[t, selected[t]]
    q = np.arange(n_queries)
    track = (indices, coords[selected, q], similarities[selected, q])
    diagnostics = {
        "method": "regional_cosine_continuity",
        "selected_regions": selected.tolist(),
        "region_frame_counts": np.bincount(selected, minlength=n_regions).tolist(),
        "median_similarity": float(np.median(track[2])),
        "max_speed_mps": max_speed_mps,
        "position_tolerance_m": position_tolerance_m,
        "jump_penalty_per_km": jump_penalty_per_km,
        "switch_penalty": switch_penalty,
    }
    return track, diagnostics
