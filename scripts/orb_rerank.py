"""Bounded, frozen-input ORB geometry diagnostic; no learned model or GPS input.

This experiment never changes the assisted policy or the source score matrix.
Its constants must be frozen before actual-image execution.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import os
import zipfile

import cv2
import numpy as np

PROFILE = {
    "version": "orb-fundamental-coverage-v1", "experimental": True,
    "top_k": 8, "candidate_order": "reference_mean_centered_cosine_stable",
    "query_views": "three_long_axis_squares_offsets_0_half_full",
    "image_size_wh": [512, 512], "resize": "opencv_INTER_LINEAR",
    "orb": {"nfeatures": 2500, "scaleFactor": 1.2, "nlevels": 8,
            "edgeThreshold": 31, "firstLevel": 0, "WTA_K": 2,
            "scoreType": "HARRIS_SCORE", "patchSize": 31, "fastThreshold": 20},
    "matching": {"norm": "HAMMING", "reciprocal": True,
                 "ratio_both_directions": .75},
    "geometry": {"method": "USAC_MAGSAC", "threshold_px": 3.,
                 "confidence": .999, "max_iterations": 10000, "rng_seed": 0,
                 "minimum_matches": 8},
    "coverage": "convex_hull_area_divided_by_image_area",
    "score": "inlier_count_times_minimum_query_reference_coverage",
    "view_pooling": "maximum", "tie_breaks": "centered_cosine_then_reference_index",
    "zero_evidence": "original_centered_top1", "temporal_decoder": None,
    "route_status": "diagnostic_uncalibrated_not_assisted_policy",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_topk(scores, k=8):
    values = np.asarray(scores)
    if (values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all()
            or isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError("finite nonempty score matrix and positive integer top-k required")
    return np.argsort(-values.astype(np.float64), axis=1, kind="stable")[:, :min(int(k), values.shape[1])]


def mutual_ratio_matches(desc_a, desc_b, ratio=.75):
    if not 0 < ratio < 1:
        raise ValueError("ratio must lie strictly between zero and one")
    if desc_a is None or desc_b is None:
        return np.empty((0, 2), dtype=np.int64)
    a, b = np.asarray(desc_a), np.asarray(desc_b)
    if (a.ndim != 2 or b.ndim != 2 or a.dtype != np.uint8 or b.dtype != np.uint8
            or a.shape[1] != b.shape[1] or not a.shape[1]):
        raise ValueError("equal-width uint8 descriptor matrices required")
    if len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), dtype=np.int64)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    def accepted(source, target):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(source, target, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance}
    forward, reverse = accepted(a, b), accepted(b, a)
    pairs = sorted((i, j) for i, j in forward.items() if reverse.get(j) == i)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def coverage(points, shape_hw):
    pts = np.asarray(points, dtype=np.float32)
    shape = np.asarray(shape_hw, dtype=float)
    if (shape.shape != (2,) or not np.isfinite(shape).all() or np.any(shape <= 0)
            or pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all()):
        raise ValueError("finite point coordinates and positive image shape required")
    if np.any(pts < 0) or (len(pts) and np.any(pts > shape[::-1])):
        raise ValueError("point lies outside its declared image")
    if len(pts) < 3:
        return 0.
    return float(np.clip(cv2.contourArea(cv2.convexHull(pts)) / float(np.prod(shape)), 0., 1.))


def verify_geometry(pa, pb, shape_a, shape_b, *, find_fn=None):
    a, b = np.asarray(pa, np.float32), np.asarray(pb, np.float32)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite equal-shaped correspondence arrays required")
    coverage(a, shape_a)
    coverage(b, shape_b)
    result = {"status": "no_geometry", "match_count": len(a), "inlier_count": 0,
              "query_coverage": 0., "reference_coverage": 0., "score": 0.}
    if len(a) < PROFILE["geometry"]["minimum_matches"]:
        return result
    if not hasattr(cv2, "USAC_MAGSAC"):
        raise RuntimeError("this OpenCV build lacks USAC_MAGSAC")
    cv2.setRNGSeed(PROFILE["geometry"]["rng_seed"])
    try:
        F, mask = (find_fn or cv2.findFundamentalMat)(a, b, method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=PROFILE["geometry"]["threshold_px"],
            confidence=PROFILE["geometry"]["confidence"],
            maxIters=PROFILE["geometry"]["max_iterations"])
    except cv2.error:
        return result
    # OpenCV may return a noninitialised mask when F is None: never read it.
    if F is None or mask is None:
        return result
    F = np.asarray(F)
    mask = np.asarray(mask).reshape(-1)
    if (F.shape != (3, 3) or not np.isfinite(F).all() or np.linalg.norm(F) == 0
            or np.linalg.matrix_rank(F) != 2 or mask.shape != (len(a),)
            or not np.isin(mask, [0, 1]).all()):
        return result
    keep = mask.astype(bool)
    count = int(keep.sum())
    if count < PROFILE["geometry"]["minimum_matches"]:
        return result
    qa, rb = coverage(a[keep], shape_a), coverage(b[keep], shape_b)
    result.update(status="ok", inlier_count=count, query_coverage=qa,
                  reference_coverage=rb, score=float(count * min(qa, rb)))
    return result


def rank_candidates(score_row, top_indices, records):
    row, indices = np.asarray(score_row, float), np.asarray(top_indices)
    if (row.ndim != 1 or not len(row) or not np.isfinite(row).all() or indices.ndim != 1
            or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(indices >= len(row))):
        raise ValueError("finite scores and distinct in-range candidate indices required")
    best = {int(index): (0., None) for index in indices}
    for record in records:
        index, view, score = int(record["ref_index"]), int(record["view"]), float(record["score"])
        if index not in best or view not in (0, 1, 2) or not np.isfinite(score) or score < 0:
            raise ValueError("invalid candidate geometry record")
        if score > best[index][0] or (score == best[index][0] and best[index][1] is not None and view < best[index][1]):
            best[index] = score, view
    chosen = min(best, key=lambda index: (-best[index][0], -row[index], index))
    original = min(best, key=lambda index: (-row[index], index))
    return {"selected_ref_index": chosen, "selected_view": best[chosen][1],
            "geometry_score": best[chosen][0], "original_ref_index": original,
            "reason": "positive_geometry" if best[chosen][0] > 0 else "no_positive_geometry"}


def stream_raw_store(npz_path, dest_path, *, expected_sha256, expected_ids, allowed_temp_root):
    """Validate the source/ID vector and stream only raw.npy into a readonly mmap."""
    source = Path(npz_path).resolve(strict=True)
    root = Path(allowed_temp_root).resolve(strict=True)
    destination = Path(dest_path).resolve()
    partial = destination.with_name(destination.name + ".partial")
    ids_expected = [str(value) for value in expected_ids]
    if (not root.is_dir() or destination == root or not destination.is_relative_to(root)
            or destination.exists() or partial.exists()):
        raise ValueError("new mmap destination must stay within its existing checked temp root")
    if not ids_expected or len(ids_expected) > 1500 or len(set(ids_expected)) != len(ids_expected):
        raise ValueError("distinct bounded expected reference IDs required")
    if file_sha256(source) != expected_sha256:
        raise ValueError("raw reference NPZ SHA mismatch")
    with np.load(source, allow_pickle=False) as saved:
        ids = saved["ids"]
        if ids.ndim != 1 or [str(value) for value in ids] != ids_expected:
            raise ValueError("raw reference ID vector/order mismatch")
    digest = hashlib.sha256()
    total = 0
    try:
        with zipfile.ZipFile(source) as archive:
            info = archive.getinfo("raw.npy")
            if info.file_size > 2 * 1024**3:
                raise ValueError("raw array exceeds the fixed 2GiB file budget")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as reader, partial.open("xb") as writer:
                for block in iter(lambda: reader.read(16 * 1024**2), b""):
                    writer.write(block)
                    digest.update(block)
                    total += len(block)
            if total != info.file_size:
                raise ValueError("truncated raw array extraction")
        if file_sha256(source) != expected_sha256:
            raise ValueError("raw NPZ changed during extraction")
        os.replace(partial, destination)
        raw = np.load(destination, mmap_mode="r", allow_pickle=False)
        if (raw.ndim != 4 or raw.shape[0] != len(ids_expected) or raw.shape[-1] != 3
                or raw.dtype != np.uint8 or not all(raw.shape) or raw.flags.writeable):
            del raw
            raise ValueError("raw pixels are not a readonly uint8 N,H,W,3 array")
        return raw, {"source": str(source), "source_sha256": expected_sha256,
            "path": str(destination), "sha256": digest.hexdigest(), "bytes": total,
            "shape": list(raw.shape), "dtype": raw.dtype.str, "readonly": True}
    except Exception:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
