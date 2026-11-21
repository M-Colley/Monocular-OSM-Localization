"""Experimental bounded sequence decoding of a complete retrieval matrix.

No models, files, network, VO or labels are accessed. Both declared variants
are returned. With more than64 references this is an approximation: a good
global path can be pruned before continuity is considered. Callers must freeze
the complete raw matrix, not just the retained candidates, before evaluation.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np

from .geo import LocalFrame


PROFILE = {
    "version": "sparse-sequence-top64-v1",
    "validation_status": "posthoc_dev_unvalidated",
    "variants": ["raw", "reference_mean"],
    "candidates_per_query": 64,
    "candidate_ranking": "each variant's emission score before continuity",
    "max_queries": 80,
    "max_references": 100000,
    "max_matrix_values": 8000000,
    "transition_base_free_m": 30.,
    "transition_free_speed_mps": 40.,
    "transition_dt_floor_s": .5,
    "transition_excess_penalty": .02,
    "timing": "each actual consecutive query interval",
    "score_arithmetic": "float64; reference means over every supplied query",
    "score_layout": "C-contiguous; original float32/float64 input precision retained",
    "ties": "lexicographic ID at rank boundary, predecessor and terminal",
    "metric_frame": "AEQD at all-reference mean latitude and circular-mean longitude; ID-ordered sum",
    "median_window": 5,
    "smoothing": "running median in metric coordinates",
    "search_claim": "exact within retained per-frame lattice; not full-reference DP when pruned",
}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


PROFILE_SHA256 = _json_sha(PROFILE)
# A fresh JSON copy is exposed by decode; freezing the nested variants avoids
# changing the registered profile through a mutable list in this mapping.
PROFILE["variants"] = tuple(PROFILE["variants"])
PROFILE = MappingProxyType(PROFILE)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _inputs(raw_cosines, ref_ids, ref_latlon, query_times):
    scores = np.asarray(raw_cosines)
    refs = np.asarray(ref_latlon)
    ids = np.asarray(ref_ids)
    times = np.asarray(query_times)
    if (scores.ndim != 2 or scores.dtype.kind != "f" or scores.dtype.itemsize not in {4, 8}
            or not 2 <= scores.shape[0] <= PROFILE["max_queries"]
            or not 1 <= scores.shape[1] <= PROFILE["max_references"]
            or scores.size > PROFILE["max_matrix_values"]):
        raise ValueError("full float32/float64 matrix exceeds the fixed query/reference budget")
    n_query, n_ref = scores.shape
    if (refs.shape != (n_ref, 2) or refs.dtype.kind not in "fiu" or not np.isfinite(refs).all()
            or np.any(np.abs(refs[:, 0]) > 90) or np.any(np.abs(refs[:, 1]) > 180)
            or ids.shape != (n_ref,) or ids.dtype.kind not in "US"
            or times.shape != (n_query,) or times.dtype.kind not in "fiu"
            or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times.astype(float)) <= 0)
            or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.00001)):
        raise ValueError("finite raw cosines, WGS84 references, IDs and ordered actual times must agree")
    identifiers = ids.astype(str)
    if len(set(identifiers.tolist())) != n_ref or any(not value for value in identifiers):
        raise ValueError("reference IDs must be nonempty and unique")
    # NumPy uses different reduction orders for C/F arrays. Canonical layout
    # makes near-tied reference means independent of the caller's memory view.
    scores = np.ascontiguousarray(scores)
    intervals = np.diff(times.astype(np.float64))
    if not np.isfinite(intervals).all() or intervals.max() > np.finfo(float).max / 40.:
        raise ValueError("actual query intervals overflow the transition calculation")
    order = np.argsort(identifiers, kind="stable")
    refs64 = np.asarray(refs, dtype=np.float64)
    latitude = float(np.mean(refs64[order, 0]))
    longitudes = np.deg2rad(refs64[order, 1])
    sin, cos = float(np.mean(np.sin(longitudes))), float(np.mean(np.cos(longitudes)))
    if np.hypot(sin, cos) < 1e-8:
        raise ValueError("reference longitudes have no stable local metric origin")
    longitude = float(np.rad2deg(np.arctan2(sin, cos)))
    frame = LocalFrame(latitude, longitude)
    xy = np.column_stack(frame.to_xy(refs64[:, 0], refs64[:, 1])).astype(np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("reference positions cannot form a finite local metric frame")
    return scores, identifiers, refs64, times.astype(np.float64), frame, xy, order


def _candidates(emissions, id_order):
    """TopK with an ID-stable boundary; returned states are ID-ordered for DP."""
    k = min(PROFILE["candidates_per_query"], len(emissions))
    if k == len(emissions):
        return id_order.copy()
    threshold = np.partition(emissions, len(emissions) - k)[len(emissions) - k]
    higher = emissions > threshold
    tied = emissions == threshold
    keep = higher.copy()
    need = k - int(higher.sum())
    keep[id_order[tied[id_order]][:need]] = True
    return id_order[keep[id_order]]


def _decode(scores, means, times, xy, id_order):
    candidates, emissions = [], []
    for row in scores:
        effective = row.astype(np.float64) - means
        retained = _candidates(effective, id_order)
        candidates.append(retained)
        emissions.append(effective[retained])
    back = [None]
    value = emissions[0].copy()
    largest = 0
    for index, interval in enumerate(np.diff(times), 1):
        previous, current = xy[candidates[index - 1]], xy[candidates[index]]
        distance = np.hypot(previous[:, None, 0] - current[None, :, 0],
                            previous[:, None, 1] - current[None, :, 1])
        largest = max(largest, int(distance.size))
        free = PROFILE["transition_base_free_m"] + PROFILE["transition_free_speed_mps"] * max(
            float(interval), PROFILE["transition_dt_floor_s"])
        objective = value[:, None] - PROFILE["transition_excess_penalty"] * np.maximum(distance - free, 0.)
        predecessor = objective.argmax(axis=0)
        value = objective[predecessor, np.arange(len(current))] + emissions[index]
        back.append(predecessor)
    state = int(value.argmax())
    objective = float(value[state])
    path = np.empty(len(times), dtype=np.int64)
    for index in range(len(times) - 1, -1, -1):
        path[index] = candidates[index][state]
        if index:
            state = int(back[index][state])
    return path, objective, candidates, largest


def decode_sparse_tracks(raw_cosines, ref_ids, ref_latlon, query_times):
    """Return both fixed variants without a selected-method policy.

    Inputs include every reference and query row. Only transition hypotheses
    are pruned, separately by each variant's emissions. Input arrays are never
    modified; typed hashes bind the original complete inputs. No quadratic
    all-reference distance matrix is constructed.
    """
    scores, ids, refs, times, frame, xy, order = _inputs(raw_cosines, ref_ids, ref_latlon, query_times)
    means = scores.mean(axis=0, dtype=np.float64)
    outputs = {}
    for variant in PROFILE["variants"]:
        normalization = np.zeros(len(ids), dtype=np.float64) if variant == "raw" else means
        path, objective, retained, largest = _decode(scores, normalization, times, xy, order)
        selected = xy[path]
        smooth = np.array([np.median(selected[max(0, i - 2):i + 3], axis=0) for i in range(len(times))])
        lat, lon = frame.to_latlon(smooth[:, 0], smooth[:, 1])
        _, counts = np.unique(refs[path], axis=0, return_counts=True)
        candidates = [[str(ids[column]) for column in row] for row in retained]
        outputs[variant] = {"status": "complete", "variant": variant, "objective": objective,
            "route_latlon": np.column_stack([lat, lon]).tolist(), "route_timestamps": times.tolist(),
            "raw_route_latlon": refs[path].tolist(), "chosen_ref_indices": path.tolist(),
            "chosen_ref_ids": ids[path].tolist(), "selected_raw_cosines": scores[np.arange(len(times)), path].tolist(),
            "selected_emissions": (scores[np.arange(len(times)), path].astype(np.float64) - normalization[path]).tolist(),
            "unique_coordinate_count": len(counts), "dominant_coordinate_fraction": float(counts.max() / len(times)),
            "candidate_pruning": {"limit_per_query": 64, "full_reference_count": len(ids),
                "retained_counts": [len(row) for row in retained], "candidate_ids": candidates,
                "candidate_ids_sha256": _json_sha(candidates), "pruned": len(ids) > 64,
                "largest_transition_values": largest, "max_transition_values": 64 * 64,
                "claim": PROFILE["search_claim"]}}
    return {"status": "complete", "experimental": True, "validation_status": "posthoc_dev_unvalidated",
        "confidence_calibrated": False, "selected_variant": None,
        "profile": json.loads(json.dumps(dict(PROFILE))), "profile_sha256": PROFILE_SHA256,
        "input_identity": {"raw_cosines_sha256": _array_sha(raw_cosines), "ref_ids_sha256": _json_sha(ids.tolist()),
            "ref_latlon_sha256": _array_sha(ref_latlon), "query_times_sha256": _array_sha(query_times),
            "raw_matrix_shape": list(scores.shape), "raw_matrix_dtype": scores.dtype.str,
            "reference_means_sha256": _array_sha(means), "metric_origin": [frame.lat0, frame.lon0]},
        "outputs": outputs}
