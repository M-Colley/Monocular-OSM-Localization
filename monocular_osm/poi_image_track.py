"""Pure single-branch POI image-track prototype, without a VO interface.

Post-hoc DEV; not a validated default. Acquisition and real native OCR happen
outside this module. No models, files, network or labels are accessed here.
"""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
from types import MappingProxyType

import numpy as np
from .geo import LocalFrame

PROFILE = {"version": "single-poi-image-track-v1", "radius_m": 3000, "cap": 1500,
           "query_mode": "crops", "score_normalization": "reference_mean",
           "model": "megaloc", "max_queries": 80, "ocr_samples": 24,
           "transition_base_free_m": 30., "transition_free_speed_mps": 40.,
           "transition_dt_floor_s": .5, "transition_excess_penalty": .02,
           "median_window": 5, "cosine_tolerance": 1e-5, "radius_tolerance_m": 1.,
           "timing": "each actual query interval", "ties": "lexicographic source-prefixed reference ID",
           "smoothing": "local metric coordinate running median"}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


PROFILE_SHA256 = _json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def plan_single_branch(clues, ocr_provenance):
    """Plan the entire sole literal branch; zero/ambiguous branches refuse."""
    if not isinstance(clues, Mapping) or clues.get("selected_answer") is not None:
        raise ValueError("supply unselected literal clue-builder output")
    branches = clues.get("candidate_clusters")
    if not isinstance(branches, list) or len(branches) != 1:
        raise ValueError("image-only POI prototype requires exactly one literal branch")
    if not isinstance(ocr_provenance, Mapping):
        raise ValueError("native same-video OCR provenance is required")
    native = json.loads(json.dumps(dict(ocr_provenance), allow_nan=False))
    inventory = clues.get("inventory") or {}
    if not isinstance(inventory, Mapping):
        raise ValueError("fingerprinted city inventory provenance is required")
    if (native.get("source") != "same_video_native_frames" or native.get("native_bbox_grid") is not True
            or native.get("sample_count") != PROFILE["ocr_samples"]
            or not _digest_string(native.get("video_sha256"))
            or not _digest_string(clues.get("detections_sha256"))
            or not _digest_string(inventory.get("snapshot_sha256"))
            or inventory.get("scope") != "named_city_bbox" or not inventory.get("city")):
        raise ValueError("real native same-video OCR and fingerprinted city inventory are required")
    branch = branches[0]
    if not isinstance(branch, Mapping) or not isinstance(branch.get("pois"), list):
        raise ValueError("branch must retain its OSM POI inventory")
    positions = np.array([[p["lat"], p["lon"]] for p in branch["pois"]], dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or not len(positions)
            or not np.isfinite(positions).all() or np.any(np.abs(positions[:, 0]) > 90)
            or np.any(np.abs(positions[:, 1]) > 180)):
        raise ValueError("branch POIs must have finite WGS84 coordinates")
    lon0 = float(np.degrees(np.angle(np.exp(1j * np.radians(positions[:, 1])).mean())))
    frame = LocalFrame(float(positions[:, 0].mean()), lon0)
    xy = np.column_stack(frame.to_xy(positions[:, 0], positions[:, 1]))
    if not np.isfinite(xy).all():
        raise ValueError("branch metric projection is not finite")
    center_xy = xy.mean(axis=0)
    lat, lon = frame.to_latlon(*center_xy)
    if np.linalg.norm(xy - center_xy, axis=1).max() > PROFILE["radius_m"]:
        raise ValueError("sole branch exceeds the fixed neighborhood; no crop or branch was chosen")
    plan = {"experimental": True, "validation_status": "posthoc_dev_unvalidated",
            "profile_sha256": PROFILE_SHA256, "profile": dict(PROFILE),
            "branch_count": 1, "hypothesis_id": str(branch.get("id", "hypothesis_0")),
            "center": [float(lat), float(lon)], "clues": json.loads(json.dumps(clues, allow_nan=False)),
            "ocr_provenance": native}
    plan["plan_sha256"] = _json_sha(plan)
    return plan


def _validated_evidence(evidence, plan, query_video_sha256):
    if (not isinstance(plan, Mapping) or plan.get("profile_sha256") != PROFILE_SHA256
            or plan.get("profile") != PROFILE or plan.get("branch_count") != 1
            or plan.get("plan_sha256") != _json_sha({k: v for k, v in plan.items() if k != "plan_sha256"})):
        raise ValueError("single-branch plan/profile content changed")
    if not _digest_string(query_video_sha256) or query_video_sha256 != plan["ocr_provenance"]["video_sha256"]:
        raise ValueError("native OCR and query frames must identify the same video")
    if not isinstance(evidence, Mapping):
        raise ValueError("full reference evidence is unavailable")
    provenance = evidence.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ValueError("full evidence requires declared encoder provenance")
    backbone, preprocessing = provenance.get("backbone") or {}, provenance.get("preprocessing") or {}
    if not isinstance(backbone, Mapping) or not isinstance(preprocessing, Mapping):
        raise ValueError("full evidence requires declared encoder provenance")
    center = np.asarray(provenance.get("center"), float)
    if (center.shape != (2,) or not np.allclose(center, plan["center"], rtol=0, atol=1e-9)
            or provenance.get("radius_m") != PROFILE["radius_m"] or provenance.get("cap") != PROFILE["cap"]
            or provenance.get("query_mode") != PROFILE["query_mode"]
            or backbone.get("name") != PROFILE["model"] or not backbone.get("hub")
            or preprocessing.get("descriptor_normalization") != "L2"):
        raise ValueError("full evidence does not match the fixed acquisition/encoder profile")
    indices = np.asarray(evidence.get("query_indices"))
    times = np.asarray(evidence.get("query_times"))
    refs = np.asarray(evidence.get("ref_latlon"))
    scores = np.asarray(evidence.get("similarities"))
    ids = np.asarray(evidence.get("ref_ids"))
    source = provenance.get("source")
    if (not isinstance(source, str) or not source or ids.ndim != 1
            or not 1 <= len(ids) <= PROFILE["cap"]):
        raise ValueError("reference IDs must be unique and source-prefixed within the fixed cap")
    ids = ids.astype(str)
    if (len(set(ids.tolist())) != len(ids)
            or any(not key.startswith(source + ":") or len(key) == len(source) + 1 for key in ids)):
        raise ValueError("reference IDs must be unique and source-prefixed within the fixed cap")
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or not 2 <= len(indices) <= PROFILE["max_queries"] or indices[0] < 0
            or np.any(np.diff(indices) <= 0) or times.shape != indices.shape
            or any(array.dtype.kind not in "fiu" for array in (times, refs, scores))
            or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0)
            or refs.shape != (len(ids), 2) or not np.isfinite(refs).all()
            or np.any(refs[:, 0] < -90) or np.any(refs[:, 0] > 90)
            or np.any(refs[:, 1] < -180) or np.any(refs[:, 1] > 180)
            or scores.shape != (len(times), len(ids)) or not np.isfinite(scores).all()
            or np.any(scores < -1 - PROFILE["cosine_tolerance"])
            or np.any(scores > 1 + PROFILE["cosine_tolerance"])):
        raise ValueError("query/reference shapes, finite cosines and actual ordered timestamps must agree")
    times, refs = np.asarray(times, float), np.asarray(refs, float)
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    intervals = np.diff(times)
    if (not np.isfinite(intervals).all() or np.any(intervals <= 0)
            or intervals.max() > np.finfo(float).max / PROFILE["transition_free_speed_mps"]):
        raise ValueError("actual query intervals must remain finite and strictly positive")
    frame = LocalFrame(*plan["center"])
    xy = np.column_stack(frame.to_xy(refs[:, 0], refs[:, 1])).astype(np.float32)
    if (not np.isfinite(xy).all()
            or np.linalg.norm(xy, axis=1).max() > PROFILE["radius_m"] + PROFILE["radius_tolerance_m"]):
        raise ValueError("reference coordinates exceed the declared fixed neighborhood")
    return indices, times, refs, scores, ids, frame, xy, provenance


def _decode(centered, xy, times):
    """Fixed existing Viterbi penalties, with each observed time interval."""
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    np.hypot(dx, dy, out=dx)
    del dy
    distance = dx
    score = centered[0].copy()
    back = np.zeros(centered.shape, dtype=np.int32)
    columns = np.arange(len(xy))
    for row, interval in enumerate(np.diff(times), 1):
        free = PROFILE["transition_base_free_m"] + PROFILE["transition_free_speed_mps"] * max(
            float(interval), PROFILE["transition_dt_floor_s"])
        transition = -PROFILE["transition_excess_penalty"] * np.maximum(distance - free, 0.)
        candidates = score[:, None] + transition
        back[row] = candidates.argmax(axis=0)
        score = candidates[back[row], columns] + centered[row]
    path = np.zeros(len(times), dtype=np.int32)
    path[-1] = score.argmax()
    for row in range(len(times) - 2, -1, -1):
        path[row] = back[row + 1, path[row + 1]]
    return path, float(score[path[-1]])


def decode_single_branch_image_track(evidence, plan, *, query_video_sha256):
    """Decode full evidence with no VO input and no acquisition fallback."""
    indices, times, refs, scores, ids, frame, xy, provenance = _validated_evidence(
        evidence, plan, query_video_sha256)
    order = np.argsort(ids, kind="stable")
    centered = scores - scores.mean(axis=0, keepdims=True)
    path, objective = _decode(np.ascontiguousarray(centered[:, order]), xy[order], times)
    chosen = order[path]
    selected_xy = xy[chosen]
    half = PROFILE["median_window"] // 2
    smooth = np.array([np.median(selected_xy[max(0, i - half):i + half + 1], axis=0)
                       for i in range(len(times))])
    lat, lon = frame.to_latlon(smooth[:, 0], smooth[:, 1])
    route = np.column_stack([lat, lon])
    _, counts = np.unique(refs[chosen], axis=0, return_counts=True)
    result = {"experimental": True, "validation_status": "posthoc_dev_unvalidated", "confidence_calibrated": False,
        "status": "complete", "uses_rigid_vo": False, "retained_branch_count": 1,
        "successful_branch_count": 1, "hypothesis_id": plan["hypothesis_id"],
        "radius_m": PROFILE["radius_m"], "cap": PROFILE["cap"], "query_mode": PROFILE["query_mode"],
        "score_normalization": PROFILE["score_normalization"], "point_count": len(times), "route_valid": True,
        "route_latlon": route.tolist(), "route_timestamps": times.tolist(), "query_indices": indices.tolist(),
        "chosen_ref_indices": chosen.tolist(), "chosen_ref_ids": ids[chosen].tolist(),
        "raw_route_latlon": refs[chosen].tolist(), "selected_raw_cosines": scores[np.arange(len(times)), chosen].tolist(),
        "selected_centered_scores": centered[np.arange(len(times)), chosen].tolist(), "objective": objective,
        "dominant_coordinate_fraction": float(counts.max() / len(times)), "unique_coordinate_count": len(counts),
        "degenerate": bool(np.all(route == route[0]))}
    result["provenance"] = _provenance(plan, query_video_sha256, indices, times, refs, scores, ids, provenance)
    return result


def _provenance(plan, video_sha, indices, times, refs, scores, ids, source):
    return {"profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
            "plan_sha256": plan["plan_sha256"], "query_video_sha256": video_sha,
            "query_times_sha256": _array_sha(times), "query_indices_sha256": _array_sha(indices),
            "similarities_sha256": _array_sha(scores), "ref_latlon_sha256": _array_sha(refs),
            "ref_ids_sha256": _json_sha(ids.tolist()), "reference_count": len(ids),
            "normalization": "raw cosine minus each reference's mean over the supplied query frames",
            "source": json.loads(json.dumps(source, allow_nan=False))}
