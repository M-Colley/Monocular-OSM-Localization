"""Pure preparation of optional local yaw and actual crop optical bearings.

Uses camera rotations only. VO translations and held invalid poses never
provide metric distance, a measured stop, or a global ground-plane heading.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np


PROFILE = MappingProxyType({
    "version": "local-camera-yaw-preparation-v1",
    "rotation_convention": "world to camera; clockwise yaw = -atan2(Rrel[0,2],Rrel[2,2]); Rrel=R_current @ R_previous.T",
    "minimum_inliers": 12, "max_3d_angular_speed_deg_s": 60.,
    "max_native_interval_s": 1., "minimum_interval_time_coverage": .8,
    "max_missing_gap_s": 1.,
    "rotation_validation": "finite proper orthonormal matrices, absolute tolerance 1e-6",
    "interval_aggregation": "sum accepted local yaw then wrap to [-180,180); weight by accepted elapsed-time fraction",
    "missing": "NaN yaw and zero quality; invalid steps are never zero-turn measurements",
    "query_grid": "exact ordinal subset with bitwise equal float64 times",
    "crop_bearing": "three actual evenly spaced square crops; inverse-K ray through integer-pixel crop center",
    "translations_used": False, "ground_plane_used": False,
    "posthoc_DEV_unvalidated": True, "policy_adopted": False,
})
PROFILE_SHA256 = hashlib.sha256(json.dumps(dict(PROFILE), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def crop_bearings_deg(K, image_shape):
    """Horizontal optical offsets for the exact three square crop centers."""
    matrix = np.asarray(K)
    _require(matrix.shape == (3, 3) and matrix.dtype.kind == "f" and np.isfinite(matrix).all(),
             "finite floating camera intrinsic matrix required")
    _require(np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0)
             and matrix[0, 0] > 0 and matrix[1, 1] > 0,
             "positive pinhole intrinsics required")
    _require(len(image_shape) == 2 and all(type(v) is int and 1 <= v <= 16384 for v in image_shape),
             "bounded integer native height and width required")
    height, width = image_shape
    edge = min(height, width)
    points = []
    for view in range(3):
        offset = round((max(height, width)-edge)*view/2.)
        points.append([offset+(edge-1)/2. if width >= height else (edge-1)/2.,
            (edge-1)/2. if width >= height else offset+(edge-1)/2., 1.])
    rays = np.linalg.solve(matrix.astype(np.float64), np.asarray(points).T).T
    _require(np.all(rays[:, 2] > 0), "crop rays must face forward")
    return np.degrees(np.arctan2(rays[:, 0], rays[:, 2]))


def local_yaw_evidence(rotations, valid, n_inliers, timestamps, frame_ordinals,
                       query_times, query_ordinals):
    """Return [Q-1] optional clockwise yaw increments and explicit quality.

    Every query must be an actual native motion frame. Each interval's
    evidence uses only consecutive local rotations, avoiding accumulated
    global pose tilt. A low-coverage interval is wholly missing.
    """
    rotation, good, inliers, times, ordinals, qtimes, qord = map(np.asarray,
        (rotations, valid, n_inliers, timestamps, frame_ordinals, query_times, query_ordinals))
    count = len(times)
    _require(2 <= count <= 100000 and times.shape == (count,) and times.dtype == np.float64
             and np.isfinite(times).all() and times[0] >= 0 and np.all(np.diff(times) > 0),
             "bounded strictly increasing float64 native times required")
    _require(rotation.shape == (count, 3, 3) and rotation.dtype.kind == "f" and np.isfinite(rotation).all(),
             "finite native rotations required")
    _require(good.shape == (count,) and good.dtype == np.bool_, "native Boolean validity required")
    _require(inliers.shape == (count,) and inliers.dtype.kind in "iu" and np.all(inliers >= 0),
             "nonnegative integer inlier counts required")
    _require(ordinals.shape == (count,) and ordinals.dtype.kind in "iu"
             and ordinals[0] >= 0 and np.all(ordinals[1:] > ordinals[:-1]), "ordered native frame ordinals required")
    _require(qtimes.ndim == 1 and 2 <= len(qtimes) <= 80 and qtimes.dtype == np.float64
             and np.isfinite(qtimes).all() and np.all(np.diff(qtimes) > 0), "increasing float64 query times required")
    _require(qord.shape == qtimes.shape and qord.dtype.kind in "iu"
             and np.all(qord[1:] > qord[:-1]), "ordered query frame ordinals required")
    indices = np.searchsorted(ordinals, qord)
    _require(np.all(indices < count) and np.array_equal(ordinals[indices], qord),
             "query ordinals must be an exact native subset")
    _require(np.array_equal(times[indices].view(np.uint64), qtimes.view(np.uint64)),
             "query times differ from actual native times")
    r = rotation.astype(np.float64)
    gram = r @ r.transpose(0, 2, 1)
    _require(np.allclose(gram, np.eye(3), atol=1e-6, rtol=0)
             and np.allclose(np.linalg.det(r), 1., atol=1e-6, rtol=0), "proper orthonormal rotations required")
    dt = np.diff(times)
    relative = r[1:] @ r[:-1].transpose(0, 2, 1)
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1.)/2., -1., 1.)))
    yaw = -np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 2, 2]))
    accepted = good[1:] & (inliers[1:] >= PROFILE["minimum_inliers"])
    accepted &= (dt <= PROFILE["max_native_interval_s"])
    accepted &= (angle <= PROFILE["max_3d_angular_speed_deg_s"]*dt)
    result = np.full(len(qtimes)-1, np.nan, np.float64)
    quality = np.zeros(len(qtimes)-1, np.float64)
    intervals = []
    for row, (start, stop) in enumerate(zip(indices[:-1], indices[1:])):
        duration = float(times[stop]-times[start])
        mask, elapsed = accepted[start:stop], dt[start:stop]
        coverage = min(1., float(elapsed[mask].sum()/duration))
        missing_gap = max_gap = 0.
        for ok, step in zip(mask, elapsed):
            missing_gap = 0. if ok else missing_gap+float(step)
            max_gap = max(max_gap, missing_gap)
        active = (coverage >= PROFILE["minimum_interval_time_coverage"]
            and max_gap <= PROFILE["max_missing_gap_s"])
        if active:
            result[row] = (float(yaw[start:stop][mask].sum())+180.) % 360.-180.
            quality[row] = coverage
        intervals.append({"native_step_count": int(stop-start), "accepted_steps": int(mask.sum()),
            "elapsed_s": duration, "accepted_time_fraction": coverage,
            "longest_missing_gap_s": max_gap, "active": bool(active)})
    return {"vo_yaw_delta_deg": result, "vo_quality": quality,
        "query_native_indices": indices, "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
        "intervals": intervals, "accepted_native_steps": int(accepted.sum()),
        "native_steps": count-1, "translations_used": False}
