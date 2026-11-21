"""Exact all-reference sequence DP in bounded blocks; no models/files/labels.

This is a posthoc algorithm diagnostic of the same fixed sparse objective.
It changes neither the sparse prototype nor any pipeline/default policy.
"""
from __future__ import annotations
from types import MappingProxyType
import numpy as np
from .sparse_sequence import _inputs, _array_sha, _json_sha


PROFILE = {"version": "full-sequence-bounded-v1", "validation_status": "posthoc_dev_algorithm_diagnostic",
    "variants": ["raw", "reference_mean"], "max_queries": 80, "max_references": 10000,
    "max_matrix_values": 800000, "current_state_block": 128, "scratch_columns": 64,
    "max_distance_objective_temporary_values": 2000000,
    "transition_base_free_m": 30., "transition_free_speed_mps": 40.,
    "transition_dt_floor_s": .5, "transition_excess_penalty": .02,
    "timing": "each actual consecutive query interval", "score_arithmetic": "float64; all-query reference means",
    "score_layout": "C-contiguous; original input precision retained", "ties": "lexicographic ID at predecessor and terminal",
    "metric_frame": "AEQD at all-reference mean latitude and circular-mean longitude; ID-ordered sum",
    "median_window": 5, "smoothing": "running median in metric coordinates",
    "candidate_pruning": False, "search_claim": "exact full-reference max-product objective",
    "selected_variant": None, "default_policy_adopted": False}
PROFILE_SHA256 = _json_sha(PROFILE)
PROFILE["variants"] = tuple(PROFILE["variants"])
PROFILE = MappingProxyType(PROFILE)


def _decode_all(scores, times, xy, order):
    """Both variants share distances; scratch becomes each variant's objective."""
    q, n = scores.shape
    block, scratch_width = min(n, PROFILE["current_state_block"]), min(n, PROFILE["scratch_columns"])
    temporary_values = n * (block + scratch_width)
    if temporary_values > PROFILE["max_distance_objective_temporary_values"]:
        raise ValueError("full DP block allocation exceeds fixed budget")
    # All predecessor/current states use ID order, so argmax implements ties.
    xy = xy[order]
    means = scores.mean(axis=0, dtype=np.float64)[order]
    first = scores[0, order].astype(np.float64)
    values = [first.copy(), first - means]
    back = [np.empty((q, n), dtype=np.int32) for _ in range(2)]
    for table in back: table[0] = -1
    distance = np.empty((n, block), dtype=np.float64)
    scratch = np.empty((n, scratch_width), dtype=np.float64)
    largest_values = 0
    for row, interval in enumerate(np.diff(times), 1):
        free = PROFILE["transition_base_free_m"] + PROFILE["transition_free_speed_mps"] * max(float(interval), PROFILE["transition_dt_floor_s"])
        emission = scores[row, order].astype(np.float64)
        centered = emission - means
        following = [np.empty(n, np.float64), np.empty(n, np.float64)]
        for start in range(0, n, block):
            stop = min(start + block, n)
            for sub in range(start, stop, scratch_width):
                end = min(sub + scratch_width, stop); width = end - sub
                tile = distance[:, sub-start:sub-start+width]
                work = scratch[:, :width]
                np.subtract(xy[:, None, 0], xy[None, sub:end, 0], out=tile)
                np.subtract(xy[:, None, 1], xy[None, sub:end, 1], out=work)
                np.hypot(tile, work, out=tile)
                np.subtract(tile, free, out=tile)
                np.maximum(tile, 0., out=tile)
                np.multiply(tile, -PROFILE["transition_excess_penalty"], out=tile)
                columns = np.arange(width)
                for variant, emit in enumerate((emission, centered)):
                    np.add(values[variant][:, None], tile, out=work)
                    predecessor = work.argmax(axis=0)
                    following[variant][sub:end] = work[predecessor, columns] + emit[sub:end]
                    back[variant][row, sub:end] = predecessor
                largest_values = max(largest_values, n * width)
        values = following
    outputs = []
    for variant in range(2):
        state = int(values[variant].argmax()); objective = float(values[variant][state])
        path = np.empty(q, dtype=np.int64)
        for row in range(q - 1, -1, -1):
            path[row] = order[state]
            if row: state = int(back[variant][row, state])
        outputs.append((path, objective))
    return outputs, {"allocated_distance_values": int(distance.size), "allocated_scratch_values": int(scratch.size),
        "distance_objective_temporary_values": temporary_values, "largest_objective_tile_values": largest_values,
        "backpointer_tables": 2, "backpointer_shape_per_variant": list(back[0].shape), "backpointer_dtype": back[0].dtype.str,
        "transition_pairs_per_variant": int((q-1)*n*n), "distances_shared_between_variants": True}


def decode_full_tracks(raw_cosines, ref_ids, ref_latlon, query_times):
    """Return both fixed variants, preserving every state at every query."""
    shape = np.shape(raw_cosines)
    if len(shape) != 2 or not 2 <= shape[0] <= PROFILE["max_queries"] or not 1 <= shape[1] <= PROFILE["max_references"] or shape[0]*shape[1] > PROFILE["max_matrix_values"]:
        raise ValueError("complete matrix exceeds exact full-reference budget")
    scores, ids, refs, times, frame, xy, order = _inputs(raw_cosines, ref_ids, ref_latlon, query_times)
    decoded, allocation = _decode_all(scores, times, xy, order)
    means = scores.mean(axis=0, dtype=np.float64)
    outputs = {}
    for variant, (path, objective) in zip(PROFILE["variants"], decoded):
        selected = xy[path]
        smooth = np.array([np.median(selected[max(0, i-2):i+3], axis=0) for i in range(len(times))])
        lat, lon = frame.to_latlon(smooth[:, 0], smooth[:, 1])
        _, counts = np.unique(refs[path], axis=0, return_counts=True)
        normalization = np.zeros(len(ids), np.float64) if variant == "raw" else means
        outputs[variant] = {"status": "complete", "variant": variant, "objective": objective,
            "route_latlon": np.column_stack([lat, lon]).tolist(), "route_timestamps": times.tolist(),
            "raw_route_latlon": refs[path].tolist(), "chosen_ref_indices": path.tolist(), "chosen_ref_ids": ids[path].tolist(),
            "selected_raw_cosines": scores[np.arange(len(times)), path].tolist(),
            "selected_emissions": (scores[np.arange(len(times)), path].astype(np.float64)-normalization[path]).tolist(),
            "unique_coordinate_count": len(counts), "dominant_coordinate_fraction": float(counts.max()/len(times)),
            "candidate_pruning": False, "full_reference_count": len(ids), "all_states_per_query": len(ids)}
    return {"status": "complete", "experimental": True, "confidence_calibrated": False,
        "validation_status": PROFILE["validation_status"], "selected_variant": None, "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
        "inputs": {"raw_cosines_sha256": _array_sha(scores), "ref_ids_sha256": _json_sha(ids.tolist()),
            "ref_latlon_sha256": _array_sha(refs), "query_times_sha256": _array_sha(times),
            "metric_origin_latlon": [frame.lat0, frame.lon0]}, "allocation": allocation, "outputs": outputs}
