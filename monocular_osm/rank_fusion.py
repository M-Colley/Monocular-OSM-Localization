"""Fixed posthoc DEV rank calibration and equal three-encoder fusion.

Pure bounded numerical code. Ranking diversity is not retrieval correctness;
this module makes no policy, calibration, or generalization claim.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from . import heading_sequence as heading

MODELS = ("megaloc", "image", "selavpr")
PROFILE = {
    "version": "equal-three-encoder-average-rank-v1",
    "validation_status": "posthoc_DEV_after_encoder_outcomes_unvalidated",
    "models": MODELS, "weights": (1/3, 1/3, 1/3),
    "ranking": "ascending average exact-tie rank per RAW query row",
    "percentile": "(average_rank-0.5)/N; float64 C-contiguous",
    "fusion": "(rank_megaloc+rank_image+rank_selavpr)/3 in declared order",
    "reference_mean": "AFTER rank calibration/fusion; unchanged decoder reference mean",
    "outputs": ("rank_megaloc", "rank_image", "rank_selavpr", "equal3"),
    "max_queries": 80, "max_references": 10000, "max_values": 800000,
    "ID_itemsize_ceiling": 512, "candidate_pruning": False,
    "heading_profile_sha256": heading.PROFILE_SHA256,
    "heading_mode": "erased only; two variants share unchanged physical geometry",
    "heading_metadata": "unknown cameras; NaN headings; one pooled view offset0; no VO",
    "selected_variant": None, "confidence": None, "default_policy_adopted": False,
}
PROFILE_SHA256 = heading._json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)


def _matrix(value):
    value = np.asarray(value)
    heading._require(value.ndim == 2 and value.dtype.kind == "f" and value.dtype.itemsize in (4, 8)
        and 2 <= value.shape[0] <= PROFILE["max_queries"]
        and 1 <= value.shape[1] <= PROFILE["max_references"]
        and value.size <= PROFILE["max_values"], "bounded float32/float64 [query,reference] matrix required")
    heading._require(np.isfinite(value).all() and np.max(np.abs(value)) <= 1.00001,
                     "finite raw cosine matrix required")
    return value


def average_rank_percentiles(raw_cosines):
    """Exact ties share their average rank; every supplied reference survives."""
    scores = _matrix(raw_cosines)
    result = np.empty(scores.shape, np.float64)
    n = scores.shape[1]
    for row, values in enumerate(scores):
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
        ends = np.r_[starts[1:], n]
        result[row, order] = np.repeat((starts+ends)/(2.*n), ends-starts)
    return result


def calibrate_evidence(evidence):
    """Accept exactly three independently declared but identical identity grids.

    Each model maps to scores/ref_ids/ref_latlon/query_times. Identity equality
    includes dtype and ordering; a shared gallery cannot silently be intersected.
    """
    heading._require(isinstance(evidence, dict) and set(evidence) == set(MODELS),
                     "exact three declared encoders required")
    matrices, identities, first = {}, {}, None
    for model in MODELS:
        item = evidence[model]
        heading._require(isinstance(item, dict) and set(item) == {"scores", "ref_ids", "ref_latlon", "query_times"},
                         "exact score and identity roles required")
        scores = _matrix(item["scores"])
        q, n = scores.shape
        # Reuse the unchanged decoder's coordinate, ID and physical-grid guards.
        data = heading._inputs(scores[:, None, :], item["ref_ids"], item["ref_latlon"], item["query_times"],
            [None]*n, np.full(n, np.nan), np.zeros(1), None, None,
            heading.PROFILE["default_temp_bytes"], heading.PROFILE["max_joint_backpointer_bytes"])
        identity = {key: heading._array_sha(np.asarray(item[key])) for key in ("ref_ids", "ref_latlon", "query_times")}
        identity.update(shape=list(scores.shape), dtype=scores.dtype.str)
        heading._require(first is None or identity == first, "encoder matrix dtype/shape or exact ID/time/coordinate grid differs")
        first = identity
        matrices["rank_"+model] = average_rank_percentiles(scores)
        identities[model] = {**identity, "raw_scores_sha256": heading._array_sha(scores)}
        del data
    matrices["equal3"] = (matrices["rank_megaloc"] + matrices["rank_image"] + matrices["rank_selavpr"])/3.
    return {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "matrices": matrices,
            "input_identity": identities, "output_sha256": {k: heading._array_sha(v) for k, v in matrices.items()},
            "selected_variant": None, "confidence": None}


def decode_erased_tracks(scores, ref_ids, ref_latlon, query_times, *,
                         max_temp_bytes=heading.PROFILE["default_temp_bytes"],
                         max_backpointer_bytes=heading.PROFILE["max_joint_backpointer_bytes"], progress=None):
    """Exactly the old six-way decoder's raw/erased and centered/erased paths.

    Only two state/backpointer tables are allocated. The old profile, globals,
    input validator, reference means and blocked physical transition are intact.
    The caller may use progress solely for resource/watchdog checks.
    """
    scores = _matrix(scores)
    q, n = scores.shape
    required = 2*q*n*16*np.dtype(np.int32).itemsize
    heading._require(type(max_backpointer_bytes) is int and 0 < max_backpointer_bytes
        <= heading.PROFILE["max_joint_backpointer_bytes"] and required <= max_backpointer_bytes,
        "two erased backpointer tables exceed byte budget")
    heading._require(progress is None or callable(progress), "progress must be a callable or None")
    data = heading._inputs(scores[:, None, :], ref_ids, ref_latlon, query_times, [None]*n,
        np.full(n, np.nan), np.zeros(1), None, None, max_temp_bytes,
        heading.PROFILE["max_joint_backpointer_bytes"])
    ordered, times, xy = data["scores"], data["times"], data["xy"]
    means = heading._reference_means(ordered, max_temp_bytes)/heading.PROFILE["appearance_temperature"]
    pooled = ordered[0, 0].astype(np.float64)/heading.PROFILE["appearance_temperature"]
    previous = np.stack([np.broadcast_to(pooled[:, None], (n, 16))]*2)
    previous[1] -= means[:, None]
    back = np.empty((2, q, n, 16), np.int32)
    back[:, 0] = -1
    tree, steps = cKDTree(xy), []
    for row, dt in enumerate(np.diff(times), 1):
        if progress is not None: progress(row, q)
        following, predecessors, allocation = heading._advance(previous, tree, xy, float(dt),
            np.nan, 0., max_temp_bytes)
        back[:, row] = predecessors
        pooled = ordered[row, 0].astype(np.float64)/heading.PROFILE["appearance_temperature"]
        following += pooled[None, :, None]
        following[1] -= means[:, None]
        previous = following
        steps.append(allocation)
    outputs = {}
    for method, variant in enumerate(("raw", "reference_mean")):
        state = int(previous[method].argmax())
        objective = float(previous[method].reshape(-1)[state])
        references, headings = np.empty(q, np.int64), np.empty(q, np.int64)
        for row in range(q-1, -1, -1):
            reference, angle = divmod(state, 16)
            references[row], headings[row] = reference, angle
            if row: state = int(back[method, row, reference, angle])
        outputs[variant] = {"status": "complete", "variant": variant, "mode": "erased", "objective": objective,
            "route_latlon": data["refs"][references].tolist(), "route_timestamps": times.tolist(),
            "chosen_ref_ids": data["ids"][references].tolist(), "chosen_ref_indices": data["order"][references].tolist(),
            "heading_bin_indices": headings.tolist(), "camera_heading_deg": np.asarray(heading.HEADINGS_DEG)[headings].tolist(),
            "selected_view_indices": [0]*q, "reference_heading_available": [False]*q,
            "reference_heading_active": [False]*q,
            "selected_raw_view_cosines": ordered[np.arange(q), 0, references].tolist(),
            "score_semantics": "input percentile ranks; legacy cosine field names retained",
            "candidate_pruning": False, "confidence": None, "smoothed": False, "selected_answer": None}
    return {"profile_sha256": PROFILE_SHA256, "heading_profile_sha256": heading.PROFILE_SHA256,
        "inputs": data["hashes"], "outputs": outputs, "selected_variant": None, "confidence": None,
        "allocation": {"joint_backpointer_bytes": back.nbytes, "state_methods": 2, "heading_states": n*16,
            "temporary_ceiling_bytes": max_temp_bytes, "steps": steps}}
