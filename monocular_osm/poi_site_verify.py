"""Sign-site verification for literal shop-sign branches with weak motion.

When native OCR reads shop signs whose names occur at several places in the
named city (each a literal branch from the clue builder), the image-only
decodes search 3 km around each place with a 1,500-image sample, which can
leave the signed shops themselves almost unsampled: on the Atlanta night
clip the true site had one sampled image within 200 m, the false one 25, and
the union decode chose the false one (11 km). Here every site is instead
verified against a dense local pool around its shops: the frames in which
its signs were read (each read time +- one OCR sampling interval) are
compared with that pool, and the site whose imagery they match best is
kept. The whole window is then decoded over that site's dense pool plus the
route-scale pool (3 km, 1,500 images) around it, with the fixed image-only
decoder (reference-mean emission, the existing Viterbi penalties, a 5-sample
metric median).

Pure functions: no I/O, models or ground truth. The worker child supplies the
retrieval evidence; the audit replays these functions on the saved matrices.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .geo import LocalFrame

PROFILE = {"version": "poi-site-verify-v1", "site_radius_m": 300, "site_cap": 600,
           "route_radius_m": 3000, "route_cap": 1500, "queries": 80, "query_mode": "warp",
           "score_normalization": "reference_mean", "min_sites": 1, "max_sites": 6,
           "selection": "mean over sign-window queries of the best raw cosine to the site pool",
           "sign_margin": "one native OCR sampling interval", "median_window": 5,
           "decoder": "poi_image_track._decode", "uses_rigid_vo": False}


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


PROFILE_SHA256 = _json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)


def _centre(pois):
    positions = np.array([[float(p["lat"]), float(p["lon"])] for p in pois], dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or not len(positions)
            or not np.isfinite(positions).all() or np.any(np.abs(positions[:, 0]) > 90)
            or np.any(np.abs(positions[:, 1]) > 180)):
        raise ValueError("site POIs must have finite WGS84 coordinates")
    lon0 = float(np.degrees(np.angle(np.exp(1j * np.radians(positions[:, 1])).mean())))
    frame = LocalFrame(float(positions[:, 0].mean()), lon0)
    xy = np.column_stack(frame.to_xy(positions[:, 0], positions[:, 1]))
    lat, lon = frame.to_latlon(*xy.mean(axis=0))
    return [float(lat), float(lon)]


def plan_sites(clues, ocr_provenance):
    """One site per literal branch: its shops' centroid and the times its
    signs were read. Refuses zero or more than ``max_sites`` branches."""
    if not isinstance(clues, Mapping) or clues.get("selected_answer") is not None:
        raise ValueError("supply unselected literal clue-builder output")
    branches = clues.get("candidate_clusters")
    if (not isinstance(branches, list)
            or not PROFILE["min_sites"] <= len(branches) <= PROFILE["max_sites"]):
        raise ValueError("sign-site verification needs one to six literal branches")
    if not isinstance(ocr_provenance, Mapping):
        raise ValueError("native same-video OCR provenance is required")
    window = ocr_provenance.get("window")
    samples = ocr_provenance.get("sample_count")
    if (not isinstance(window, list) or len(window) != 2 or type(samples) is not int or samples < 2
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in window)
            or not window[1] > window[0]):
        raise ValueError("native OCR window and sample count are required")
    margin = float(window[1] - window[0]) / (samples - 1)
    observations = clues.get("matched_observations") or []
    sites = []
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping) or not isinstance(branch.get("pois"), list) or not branch["pois"]:
            raise ValueError("each branch must retain its OSM POIs")
        ids = sorted(str(p["id"]) for p in branch["pois"])
        times = sorted({round(float(o["t_sec"]), 6) for o in observations
                        if isinstance(o, Mapping) and str(o.get("poi_id")) in ids})
        sites.append({"index": index, "hypothesis_id": str(branch.get("id", f"hypothesis_{index}")),
                      "center": _centre(branch["pois"]), "poi_ids": ids, "sign_times": times})
    return {"experimental": True, "validation_status": "posthoc_dev_unvalidated",
            "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "sign_margin_s": margin,
            "sites": sites}


def sign_rows(query_times, sign_times, margin_s):
    """Query rows within ``margin_s`` of a sign read; every row when none."""
    times = np.asarray(query_times, float)
    if not len(sign_times):
        return np.arange(len(times))
    near = np.zeros(len(times), dtype=bool)
    for t in sign_times:
        near |= np.abs(times - float(t)) <= float(margin_s)
    rows = np.flatnonzero(near)
    return rows if len(rows) else np.arange(len(times))


def site_score(similarities, rows):
    """Mean over the sign rows of each row's best raw cosine to the site pool."""
    s = np.asarray(similarities, dtype=np.float64)
    if s.ndim != 2 or not s.shape[1] or not np.isfinite(s).all():
        raise ValueError("site evidence must be a finite query x reference matrix")
    return float(s[np.asarray(rows, int)].max(axis=1).mean())


def select_site(scores):
    """Index of the best-scoring site (ties: the lowest index)."""
    values = np.asarray(scores, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("site scores must be finite")
    return int(np.argmax(values))


def merge_pools(*pools):
    """Concatenate (ids, latlon, similarities) pools, keeping each id once, in
    order of first appearance. All pools must share the same query rows."""
    ids, latlon, columns, seen = [], [], [], set()
    rows = None
    for pool_ids, pool_ll, pool_sims in pools:
        pool_sims = np.asarray(pool_sims, np.float32)
        if rows is None:
            rows = pool_sims.shape[0]
        if pool_sims.shape[0] != rows or pool_sims.shape[1] != len(pool_ids) or len(pool_ll) != len(pool_ids):
            raise ValueError("pools must share query rows and match their reference lists")
        for j, ref in enumerate(pool_ids):
            if str(ref) in seen:
                continue
            seen.add(str(ref))
            ids.append(str(ref))
            latlon.append([float(pool_ll[j][0]), float(pool_ll[j][1])])
            columns.append(pool_sims[:, j])
    if not ids:
        raise ValueError("no references to decode")
    return np.asarray(ids), np.asarray(latlon, float), np.column_stack(columns).astype(np.float32)


def decode(similarities, ref_latlon, times):
    """The fixed image-only decode: reference-mean emission, the existing
    Viterbi penalties per observed interval, a 5-sample metric median."""
    from .poi_image_track import _decode

    scores = np.asarray(similarities, np.float64)
    ll = np.asarray(ref_latlon, float)
    t = np.asarray(times, float)
    if scores.ndim != 2 or scores.shape != (len(t), len(ll)) or not np.isfinite(scores).all():
        raise ValueError("decode needs a finite query x reference matrix")
    centred = scores - scores.mean(axis=0, keepdims=True)
    frame = LocalFrame(float(ll[:, 0].mean()), float(ll[:, 1].mean()))
    xy = np.column_stack(frame.to_xy(ll[:, 0], ll[:, 1]))
    path, objective = _decode(np.ascontiguousarray(centred), xy, t)
    half = PROFILE["median_window"] // 2
    chosen = xy[path]
    smooth = np.array([np.median(chosen[max(0, i - half):i + half + 1], axis=0) for i in range(len(t))])
    lat, lon = frame.to_latlon(smooth[:, 0], smooth[:, 1])
    _, counts = np.unique(ll[path], axis=0, return_counts=True)
    return {"route_latlon": np.column_stack([lat, lon]).tolist(), "route_timestamps": t.tolist(),
            "chosen_ref_indices": path.tolist(), "objective": float(objective),
            "selected_raw_cosines": scores[np.arange(len(t)), path].tolist(),
            "dominant_coordinate_fraction": float(counts.max() / len(t)),
            "unique_coordinate_count": int(len(counts))}
