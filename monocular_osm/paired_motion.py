"""Pure correspondence-to-pose control for paired experimental frontends.

All rejection means UNKNOWN motion, never a verified stop. Accepted essential
translations have arbitrary unit scale. No files, models, locations or policy.
"""
from __future__ import annotations
import hashlib
import json
from types import MappingProxyType

import cv2
import numpy as np

from .visual_odometry import _fit_plane_projection

PROFILE = {"version": "paired-native-motion-geometry-v1", "posthoc_DEV": True,
    "minimum_features_each": 16, "minimum_matches": 16, "parallax_percentile": 90,
    "minimum_parallax_focal_fraction": .006, "moving_displacement_px": .5,
    "minimum_moving_matches": 16, "selected_match_cap": 300,
    "selection": "stable ascending supplied matcher distance, original matcher order on ties",
    "essential": "cv2.findEssentialMat RANSAC prob.999 threshold1px required3x3",
    "recover_pose_minimum_inliers": 8, "minimum_translation_norm": 1e-6,
    "rng_seed_per_pair": 0, "max_features": 16384, "max_matches": 16384, "max_frames": 2950,
    "feature_bound": "resource refusal only; ORB nfeatures2500 is a soft target; never trim",
    "initial_pose": "identity bootstrap; not a measured motion edge",
    "invalid_pair": "UNKNOWN; held pose is bookkeeping, not stop evidence",
    "translation": "unit scale only; never metres or velocity",
    "chain": "world-to-camera; Rnew=Rrel@Rold,tnew=Rrel@told+trel",
    "default_adopted": False, "selected_frontend": None, "confidence": None}


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    result = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    if a.size: result.update(memoryview(a).cast("B"))
    return result.hexdigest()


PROFILE_SHA256 = json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)


def require(condition, message):
    if not condition: raise ValueError(message)


def _rotation(value):
    return (value.shape == (3, 3) and np.isfinite(value).all()
        and np.allclose(value.T@value, np.eye(3), rtol=0, atol=1e-5)
        and abs(float(np.linalg.det(value))-1.) <= 1e-5)


def estimate_pair(points_first, points_second, matches, distances, K):
    """Apply the unchanged ORB VO gates to full supplied correspondences.

    Features and reciprocal matches must be bounded. Save this entire result:
    essential_mask is copied BEFORE recoverPose can mutate its input mask.
    All match/selection indices refer to original feature/reciprocal ordering.
    """
    first, second, indices, distance, matrix = map(np.asarray,
        (points_first, points_second, matches, distances, K))
    for points in (first, second):
        require(points.ndim == 2 and points.shape[1] == 2 and len(points) <= 16384
            and points.dtype.kind == "f" and points.dtype.itemsize in (4, 8) and np.isfinite(points).all(),
            "bounded finite feature coordinates required")
    require(indices.ndim == 2 and indices.shape[1] == 2 and len(indices) <= 16384
        and indices.dtype.kind in "iu", "bounded integer match indices required")
    require(distance.shape == (len(indices),) and distance.dtype.kind in "fiu" and np.isfinite(distance).all(),
            "one finite matcher distance per reciprocal match required")
    require(matrix.shape == (3, 3) and matrix.dtype.kind in "fiu" and np.isfinite(matrix).all()
        and matrix[0, 0] > 0 and matrix[1, 1] > 0 and np.array_equal(matrix[2], [0., 0., 1.]), "finite native pinhole K required")
    require(not len(indices) or (indices.min() >= 0 and indices[:, 0].max() < len(first)
        and indices[:, 1].max() < len(second)), "match index outside features")
    require(len(set(indices[:, 0].tolist())) == len(indices) and len(set(indices[:, 1].tolist())) == len(indices),
            "one-to-one reciprocal indices required")
    a = np.asarray(first[indices[:, 0]], np.float32)
    b = np.asarray(second[indices[:, 1]], np.float32)
    displacement = np.linalg.norm(b-a, axis=1)
    result = {"status": "unknown", "reason": "unprocessed", "n_inliers": 0,
        "n_features_first": len(first), "n_features_second": len(second), "match_count": len(indices),
        "matches": indices.astype(np.int32), "distances": distance.copy(), "points_first": a, "points_second": b,
        "displacement": displacement, "moving_mask": displacement >= .5,
        "selected_match_indices": np.empty(0, np.int32), "E": np.empty((0, 0), np.float64),
        "essential_mask": np.empty(0, np.uint8), "pose_mask": np.empty(0, np.uint8),
        "R": np.empty((0, 0), np.float64), "t": np.empty(0, np.float64),
        "t_before_normalization": np.empty(0, np.float64),
        "recover_pose_inliers": 0, "parallax_p90_px": None, "translation_norm": None,
        "failure_is_stop": False, "metric_scale": False}
    def decline(reason):
        result["reason"] = reason
        return result
    if min(len(first), len(second)) < 16: return decline("insufficient_features")
    if len(indices) < 16: return decline("insufficient_matches")
    result["parallax_p90_px"] = float(np.percentile(displacement, 90))
    if result["parallax_p90_px"] < .006*float(matrix[0, 0]): return decline("low_parallax_unknown")
    moving = displacement >= .5
    result["moving_mask"] = moving
    if int(moving.sum()) < 16: return decline("insufficient_moving_matches")
    moving_indices = np.flatnonzero(moving)
    chosen = moving_indices[np.argsort(distance[moving_indices], kind="stable")[:300]]
    result["selected_match_indices"] = chosen.astype(np.int32)
    cv2.setRNGSeed(0)
    E, mask = cv2.findEssentialMat(a[chosen], b[chosen], matrix, method=cv2.RANSAC, prob=.999, threshold=1.)
    if E is not None: result["E"] = np.asarray(E).copy()
    if mask is not None: result["essential_mask"] = np.asarray(mask).copy().reshape(-1)
    if E is None or E.shape != (3, 3) or not np.isfinite(E).all(): return decline("invalid_essential")
    if mask is None or mask.size != len(chosen): return decline("invalid_essential_mask")
    n, rotation, translation, pose_mask = cv2.recoverPose(E, a[chosen], b[chosen], matrix, mask=mask)
    result.update(recover_pose_inliers=int(n), R=np.asarray(rotation).copy(), t=np.asarray(translation).reshape(-1).copy())
    result["t_before_normalization"] = result["t"].copy()
    if pose_mask is not None: result["pose_mask"] = np.asarray(pose_mask).copy().reshape(-1)
    if n < 8: return decline("insufficient_cheirality_inliers")
    if not _rotation(result["R"]) or result["t"].shape != (3,) or not np.isfinite(result["t"]).all(): return decline("invalid_pose")
    norm = float(np.linalg.norm(result["t"]))
    result["translation_norm"] = norm
    if norm < 1e-6: return decline("degenerate_translation")
    if len(result["pose_mask"]) != len(chosen): return decline("invalid_pose_mask")
    result.update(status="complete", reason="accepted_unit_scale_pose", n_inliers=int(n), t=result["t"]/norm)
    return result


def chain_relative_poses(rotations, translations, valid, n_inliers, *, enforce_planar=True):
    """Small pose arrays only; invalid pairs hold pose with validity False."""
    rotations, translations, valid, n_inliers = map(np.asarray, (rotations, translations, valid, n_inliers))
    n = len(valid)
    require(1 <= n < 2950 and valid.dtype == np.bool_ and rotations.shape == (n, 3, 3)
        and translations.shape == (n, 3) and n_inliers.shape == (n,) and n_inliers.dtype.kind in "iu", "bounded relative pose package required")
    require(np.all(n_inliers >= 0), "nonnegative inlier counts required")
    for rotation, translation in zip(rotations[valid], translations[valid]):
        require(_rotation(rotation) and np.isfinite(translation).all()
            and abs(float(np.linalg.norm(translation))-1) <= 1e-6, "valid relative pose must be proper and unit-scale")
    R, t = np.eye(3), np.zeros(3)
    centers, all_R, all_t = [np.zeros(3)], [R.copy()], [t.copy()]
    for i in range(n):
        if valid[i]:
            R = rotations[i]@R
            t = rotations[i]@t+translations[i]
        centers.append(-R.T@t if valid[i] else centers[-1].copy())
        all_R.append(R.copy()); all_t.append(t.copy())
    centers, all_R, all_t = np.asarray(centers), np.asarray(all_R), np.asarray(all_t)
    xz = centers[:, [0, 2]].copy()
    if enforce_planar:
        down = all_R[:, 1, :].mean(axis=0); norm = float(np.linalg.norm(down))
        xz = _fit_plane_projection(centers, up=-down/norm if norm > 1e-9 else None)
    return {"centers": centers, "xz": xz, "rotations": all_R, "translations": all_t,
        "valid": np.r_[True, valid], "n_inliers": np.r_[0, np.where(valid, n_inliers, 0)]}
