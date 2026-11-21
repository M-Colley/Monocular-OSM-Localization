"""Pure all-literal-branch image retrieval; no VO, files, models or network.

Persist the complete plan before acquisition. The injected callback returns
full appearance evidence and actual native-query/encoder proofs, documented
in localize_image_union. This module checks content and regional consistency;
it cannot authenticate pixels from hash claims alone. Crop scores are not
compared with warp confidence statistics. The existing single-branch method
and its decoder/policy remain separate and unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .geo import LocalFrame, degree_half_extent
from .full_sequence import PROFILE_SHA256 as FULL_PROFILE_SHA256, decode_full_tracks
from .reference_inventory import recorded_url_missing_refs


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


PROFILE = {
    "version": "all-poi-image-union-v1", "validation_status": "posthoc_dev_unvalidated",
    "min_branches": 1, "max_branches": 6, "radius_m": 3000, "cap_per_branch": 1500,
    "max_references": 9000, "queries": 80, "query_mode": "crops", "query_views": 3,
    "crop_aggregation": "maximum cosine over three crop descriptors for each reference",
    "model": "megaloc", "image_size": 512, "ocr_samples": 24,
    "ocr_min_len": 2, "ocr_min_confidence": .3, "native_pixels": True,
    "score_normalization": "reference_mean", "score_dtype": "float32",
    "reference_extent": "WGS84 geodesic-disc bounding box; wrapped longitude",
    "radius_tolerance_m": 1., "duplicate_cosine_atol": 1e-6,
    "duplicate_canonical_occurrence": "lexicographic request identity, then hypothesis ID",
    "all_branch_completion_required": True, "candidate_pruning": False,
    "listing_completeness": "known Mapillary tiles listing; zero lost tiles; exact served/unserved partition; graph completeness unknown",
    "full_sequence_profile_sha256": FULL_PROFILE_SHA256,
    "score_arithmetic": "full_sequence float64; full-gallery metric frame; median5",
    "max_provenance_bytes_per_region": 4_000_000,
}
PROFILE_SHA256 = _json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)
_AGGREGATION = PROFILE["crop_aggregation"]
_MEAN, _STD = [.485, .456, .406], [.229, .224, .225]


class _Invalid(ValueError):
    """A contract error whose fixed message is safe for diagnostics."""


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    if value.size:
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _window(native):
    value = native.get("window")
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) for v in value)):
        raise _Invalid("native OCR needs finite ordered window limits")
    try:
        limits = np.asarray(value, float)
    except (TypeError, ValueError):
        raise _Invalid("native OCR needs finite ordered window limits") from None
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] < 0 or limits[1] <= limits[0]:
        raise _Invalid("native OCR needs finite ordered window limits")
    return limits


def _native(clues, native):
    if not isinstance(clues, Mapping) or clues.get("selected_answer") is not None:
        raise _Invalid("unselected literal clue-builder output is required")
    if not isinstance(native, Mapping):
        raise _Invalid("native OCR provenance is required")
    native = _json_copy(dict(native))
    _window(native)
    inventory = clues.get("inventory") or {}
    shape = native.get("frame_shape_hw")
    if (native.get("source") != "same_video_native_frames" or native.get("native_bbox_grid") is not True
            or native.get("sample_count") != PROFILE["ocr_samples"]
            or native.get("min_len") != PROFILE["ocr_min_len"] or native.get("min_confidence") != PROFILE["ocr_min_confidence"]
            or native.get("super_res") is not False or native.get("use_gpu") is not False
            or native.get("position_stamp_filter") != "scene_text.is_position_stamp"
            or native.get("position_stamps_rejected") is not True
            or not _digest(native.get("video_sha256")) or not _digest(clues.get("detections_sha256"))
            or not isinstance(inventory, Mapping) or inventory.get("scope") != "named_city_bbox"
            or not isinstance(inventory.get("city"), str) or not inventory["city"] or not _digest(inventory.get("snapshot_sha256"))
            or not isinstance(shape, (list, tuple)) or len(shape) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)):
        raise _Invalid("native same-video OCR and named-city inventory identities are required")
    return native


def _region(branch, index):
    if not isinstance(branch, Mapping) or not isinstance(branch.get("pois"), list):
        raise _Invalid("each literal branch must retain all public POIs")
    pois, identifiers = branch["pois"], branch.get("poi_ids")
    if (not pois or len(pois) > 100_000 or not isinstance(identifiers, list)
            or any(not isinstance(v, str) or not v for v in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(p, Mapping) or not isinstance(p.get("id"), str) for p in pois)
            or set(identifiers) != {p["id"] for p in pois} or len(pois) != len(identifiers)):
        raise _Invalid("branch POI identities must be complete and unique")
    positions = np.asarray([[p["lat"], p["lon"]] for p in sorted(pois, key=lambda p: p["id"])], float)
    if (positions.shape != (len(pois), 2) or not np.isfinite(positions).all()
            or np.any(np.abs(positions[:, 0]) > 90) or np.any(np.abs(positions[:, 1]) > 180)):
        raise _Invalid("branch POIs must be finite WGS84")
    longitude = float(np.degrees(np.angle(np.exp(1j * np.radians(positions[:, 1])).mean())))
    frame = LocalFrame(float(positions[:, 0].mean()), longitude)
    xy = np.column_stack(frame.to_xy(positions[:, 0], positions[:, 1]))
    center = xy.mean(axis=0)
    if not np.isfinite(xy).all() or np.linalg.norm(xy - center, axis=1).max() > PROFILE["radius_m"]:
        raise _Invalid("whole literal branch exceeds the fixed neighborhood")
    lat, lon = frame.to_latlon(*center)
    identifier = branch.get("id", f"hypothesis_{index}")
    if not isinstance(identifier, str) or not identifier or len(identifier) > 256:
        raise _Invalid("literal hypothesis IDs must be nonempty strings")
    request = {"origin": "literal_poi_branch_centroid", "center": [float(lat), float(lon)],
               "poi_ids": sorted(identifiers), "radius_m": PROFILE["radius_m"], "cap": PROFILE["cap_per_branch"]}
    return {"hypothesis_id": identifier, **request, "request_sha256": _json_sha(request)}


def plan_image_union(clues, ocr_provenance):
    """Hash-bind every equal-budget branch; zero or >6 decline intact.

    Malformed input raises ValueError. An inadmissible branch count returns an
    inactive plan retaining every clue, without choosing a subset. Persist the
    complete plan before calling localization; no callback is called here.
    """
    native = _native(clues, ocr_provenance)
    branches = clues.get("candidate_clusters")
    if not isinstance(branches, list):
        raise _Invalid("all literal branches must be supplied together")
    active = PROFILE["min_branches"] <= len(branches) <= PROFILE["max_branches"]
    regions = [_region(branch, index) for index, branch in enumerate(branches)] if active else []
    if len({r["hypothesis_id"] for r in regions}) != len(regions):
        raise _Invalid("literal hypothesis IDs must be unique")
    plan = {"status": "ready" if active else "inactive", "reason": None if active else "branch_count_outside_fixed_budget",
            "experimental": True, "confidence_calibrated": False,
            "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
            "branch_count": len(branches), "region_plans": regions,
            "clues": _json_copy(dict(clues)), "ocr_provenance": native}
    plan["plan_sha256"] = _json_sha(plan)
    return plan


def _validate_plan(plan):
    if not isinstance(plan, Mapping):
        raise _Invalid("a frozen all-branch plan is required")
    rebuilt = plan_image_union(plan.get("clues"), plan.get("ocr_provenance"))
    if _json_sha(dict(plan)) != _json_sha(rebuilt):
        raise _Invalid("all-branch plan or profile changed")
    return rebuilt


def _reference_claims(provenance, ids):
    """Unknown legacy identity is not upgraded by a cache filename."""
    claims = provenance.get("reference_identities", {})
    if not isinstance(claims, Mapping) or set(claims) - set(ids):
        raise _Invalid("per-reference identity claims do not match actual embedded IDs")
    result = {}
    for identifier in ids:
        claim = claims.get(identifier, {})
        if not isinstance(claim, Mapping) or set(claim) - {"pixel_sha256", "embedding_sha256", "encoding_identity"}:
            raise _Invalid("invalid per-reference pixel/embedding identity claim")
        for key in ("pixel_sha256", "embedding_sha256"):
            if claim.get(key) is not None and not _digest(claim[key]):
                raise _Invalid("invalid per-reference pixel/embedding digest")
        encoding = claim.get("encoding_identity")
        if encoding is not None and (not isinstance(encoding, str) or not encoding or len(encoding) > 512):
            raise _Invalid("invalid per-reference encoding identity")
        result[identifier] = {key: claim.get(key) for key in ("pixel_sha256", "embedding_sha256", "encoding_identity")}
    return result


def _acquisition(provenance, ids, refs, region):
    """Matrix completeness does not imply a known complete provider listing."""
    record, fetch = provenance.get("acquisition"), provenance.get("fetch")
    if (not isinstance(record, Mapping) or record.get("status") != "complete"
            or record.get("listing_complete") is not True or record.get("cap") != PROFILE["cap_per_branch"]
            or provenance["source"] != "mapillary" or not isinstance(fetch, Mapping)
            or fetch.get("listing_stats_known") is not True or fetch.get("listing") != "tiles"
            or type(fetch.get("lost")) is not int or fetch["lost"] != 0):
        raise _Invalid("bounded provider listing completeness is missing, unknown or partial")
    listed = record.get("listed_references")
    if not isinstance(listed, list) or not 1 <= len(listed) <= PROFILE["cap_per_branch"]:
        raise _Invalid("full capped metadata listing is required")
    if any(not isinstance(row, Mapping) or not isinstance(row.get("id"), (str, int))
           or isinstance(row.get("id"), bool) or not str(row["id"]) or ":" in str(row["id"]) for row in listed):
        raise _Invalid("listed provider IDs must be unprefixed and nonempty")
    prefixed = ["mapillary:" + str(row["id"]) for row in listed]
    coordinates = np.asarray([[row["lat"], row["lon"]] for row in listed], dtype=np.float64)
    if (len(set(prefixed)) != len(prefixed) or coordinates.shape != (len(listed), 2)
            or not np.isfinite(coordinates).all() or np.any(np.abs(coordinates[:, 0]) > 90)
            or np.any(np.abs(coordinates[:, 1]) > 180)):
        raise _Invalid("listed reference identities/coordinates must be finite and unique")
    triples = [[str(row["id"]), float(point[0]), float(point[1])] for row, point in zip(listed, coordinates)]
    snapshot_sha = hashlib.sha256(json.dumps(triples, separators=(",", ":")).encode()).hexdigest()
    snapshot = fetch.get("snapshot")
    if (not isinstance(snapshot, Mapping) or type(snapshot.get("n_refs")) is not int
            or snapshot["n_refs"] != len(listed) or snapshot.get("refs_sha256") != snapshot_sha
            or record.get("listing_snapshot_sha256") != snapshot_sha):
        raise _Invalid("listing snapshot does not bind the exact capped metadata")
    if ("listed_ref_ids" in record and record["listed_ref_ids"] != prefixed
            or "listed_ref_latlon" in record and not np.array_equal(np.asarray(record["listed_ref_latlon"]), coordinates)):
        raise _Invalid("redundant listed-ID/coordinate provenance disagrees")
    by_id = dict(zip(prefixed, coordinates))
    if any(identifier not in by_id or not np.array_equal(by_id[identifier], point) for identifier, point in zip(ids, refs)):
        raise _Invalid("embedded IDs/coordinates are not an exact subset of the public listing")
    unserved = record.get("unserved_references")
    reasons = {"thumbnail_url_unavailable", "download_failed", "not_in_frozen_pixel_snapshot", "not_in_embedded_reference_snapshot"}
    if (not isinstance(unserved, list) or any(not isinstance(row, Mapping) or not isinstance(row.get("ref_id"), str)
            or row.get("reason") not in reasons for row in unserved)):
        raise _Invalid("every unserved listing member needs an explicit acquisition outcome")
    absent = [row["ref_id"] for row in unserved]
    if len(set(absent)) != len(absent) or set(ids) & set(absent) or set(ids) | set(absent) != set(prefixed):
        raise _Invalid("embedded and unserved IDs do not exactly partition the complete listing")
    # When the fetch recorded which listed IDs had no thumbnail URL, that
    # outcome is claimed only for them (older fetch records name no IDs).
    try:
        url_less = recorded_url_missing_refs(fetch)
    except ValueError as error:
        raise _Invalid("recorded URL-less references are malformed") from error
    if url_less is not None:
        url_less = {"mapillary:" + row["id"] for row in url_less}
        if (not url_less <= set(prefixed) or any(row["reason"] == "thumbnail_url_unavailable"
                and row["ref_id"] not in url_less for row in unserved)):
            raise _Invalid("URL-unavailable outcome is not a recorded URL-less listing member")
    dlat, dlon = degree_half_extent(*region["center"], PROFILE["radius_m"] + PROFILE["radius_tolerance_m"])
    lon_offset = (coordinates[:, 1] - region["center"][1] + 180) % 360 - 180
    if (np.any(np.abs(coordinates[:, 0] - region["center"][0]) > dlat+1e-12)
            or np.any(np.abs(lon_offset) > dlon+1e-12)):
        raise _Invalid("complete listing exceeds the planned provider bounding box")


def _evidence(raw, region, indices, times, video_sha):
    if not isinstance(raw, Mapping) or raw.get("status") != "complete":
        raise _Invalid("region response is not explicitly complete")
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _Invalid("region encoder/query provenance is required")
    if len(json.dumps(dict(provenance), allow_nan=False).encode()) > PROFILE["max_provenance_bytes_per_region"]:
        raise _Invalid("region provenance exceeds fixed budget")
    provenance = _json_copy(dict(provenance))
    backbone, prep, query = (provenance.get(key) for key in ("backbone", "preprocessing", "query_identity"))
    if not all(isinstance(value, dict) for value in (backbone, prep, query)):
        raise _Invalid("actual backbone, preprocessing and query identity are required")
    center = np.asarray(provenance.get("center"), float)
    if (center.shape != (2,) or not np.allclose(center, region["center"], rtol=0, atol=1e-9)
            or provenance.get("radius_m") != PROFILE["radius_m"] or provenance.get("cap") != PROFILE["cap_per_branch"]
            or provenance.get("query_mode") != PROFILE["query_mode"]
            or backbone.get("name") != PROFILE["model"] or not backbone.get("hub")
            or prep.get("reference_color") != "RGB" or prep.get("reference_size") != [512, 512]
            or prep.get("reference_resize") != "opencv INTER_LINEAR" or prep.get("scale") != "divide by255"
            or prep.get("mean") != _MEAN or prep.get("std") != _STD or prep.get("descriptor_normalization") != "L2"
            or isinstance(prep.get("dimensions"), bool) or not isinstance(prep.get("dimensions"), int)
            or prep["dimensions"] < 1):
        raise _Invalid("region does not match fixed3km/1500/crops/MegaLoc512 profile")
    expected = {"query_video_sha256": video_sha, "query_indices_sha256": _array_sha(indices),
                "query_times_sha256": _array_sha(times), "query_mode": "crops", "views": 3, "aggregation": _AGGREGATION}
    digest_keys = ("query_pixels_sha256", "query_descriptors_sha256", "actual_model_state_sha256")
    if (any(query.get(key) != value for key, value in expected.items())
            or any(not _digest(query.get(key)) for key in digest_keys)):
        raise _Invalid("actual query pixel/descriptor/model/grid proof is missing or stale")
    actual_indices = np.asarray(raw.get("query_indices"))
    actual_times = np.asarray(raw.get("query_times"))
    if (actual_indices.dtype.kind not in "iu" or not np.array_equal(actual_indices, indices)
            or actual_times.dtype.kind not in "fiu" or not np.array_equal(actual_times, times)):
        raise _Invalid("region query indices/times do not match the native80 grid")
    ids = np.asarray(raw.get("ref_ids"))
    refs, scores = np.asarray(raw.get("ref_latlon")), np.asarray(raw.get("similarities"))
    source = provenance.get("source")
    if (ids.ndim != 1 or ids.dtype.kind not in "US" or not 1 <= len(ids) <= PROFILE["cap_per_branch"]
            or not isinstance(source, str) or source not in {"mapillary", "kartaview", "panoramax"}):
        raise _Invalid("actual source-prefixed reference IDs must obey the per-branch cap")
    ids = ids.astype(str).tolist()
    if (len(set(ids)) != len(ids) or any(not key.startswith(source + ":") or len(key) <= len(source)+1 or len(key) > 256 for key in ids)
            or refs.shape != (len(ids), 2) or refs.dtype.kind not in "fiu" or not np.isfinite(refs).all()
            or np.any(np.abs(refs[:, 0]) > 90) or np.any(np.abs(refs[:, 1]) > 180)
            or scores.shape != (len(times), len(ids)) or scores.dtype != np.float32
            or not np.isfinite(scores).all() or np.any(np.abs(scores) > 1.00001)):
        raise _Invalid("finite float32 full matrix, reference IDs and coordinates must agree")
    dlat, dlon = degree_half_extent(*region["center"], PROFILE["radius_m"] + PROFILE["radius_tolerance_m"])
    lon_offset = (refs[:, 1] - region["center"][1] + 180) % 360 - 180
    if (np.any(np.abs(refs[:, 0] - region["center"][0]) > dlat+1e-12) or np.any(np.abs(lon_offset) > dlon+1e-12)):
        raise _Invalid("reference coordinates exceed the planned provider bounding box")
    refs, scores = np.asarray(refs, dtype=np.float64).copy(), np.ascontiguousarray(scores).copy()
    _acquisition(provenance, ids, refs, region)
    claims = _reference_claims(provenance, ids)
    hashes = {"query_indices_sha256": _array_sha(indices), "query_times_sha256": _array_sha(times),
              "ref_ids_sha256": _json_sha(ids), "ref_latlon_sha256": _array_sha(refs), "similarities_sha256": _array_sha(scores)}
    declared = provenance.get("evidence_identity", {})
    if not isinstance(declared, Mapping):
        raise _Invalid("invalid declared full-matrix identity")
    for key in hashes:
        if key in declared and declared[key] != hashes[key]:
            raise _Invalid("declared full-matrix identity is stale")
    for key, actual in (("query_indices", indices.tolist()), ("query_times", times.tolist()), ("ref_ids", ids)):
        if key in declared and declared[key] != actual:
            raise _Invalid("declared full-matrix membership is stale")
    inventory = provenance.get("reference_inventory")
    if inventory is not None:
        triples = [[key, float(point[0]), float(point[1])] for key, point in zip(ids, refs)]
        snapshot = hashlib.sha256(json.dumps(triples, separators=(",", ":")).encode()).hexdigest()
        if not isinstance(inventory, Mapping) or inventory.get("n_refs") != len(ids) or inventory.get("refs_sha256") != snapshot:
            raise _Invalid("actual reference inventory snapshot is stale")
    common = {"backbone": backbone, "preprocessing": prep,
              "query_identity": {key: query[key] for key in (*expected, *digest_keys)}}
    return {"ids": ids, "refs": refs, "scores": scores, "claims": claims,
            "provenance": provenance, "hashes": hashes, "common": common}


def _readonly(array):
    """Immutable bytes backing prevents accidental re-enabling of writes."""
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _union(regions):
    columns, claims, memberships, canonical, envelopes = {}, {}, {}, {}, {}
    for region, data in regions:
        key = (region["request_sha256"], region["hypothesis_id"])
        for column, identifier in enumerate(data["ids"]):
            current = data["refs"][column], data["scores"][:, column]
            if identifier in columns:
                previous = columns[identifier]
                low, high = envelopes[identifier]
                low, high = np.minimum(low, current[1]), np.maximum(high, current[1])
                if (not np.array_equal(previous[0], current[0])
                        or np.any(high.astype(float) - low > PROFILE["duplicate_cosine_atol"])):
                    raise _Invalid("duplicate reference ID has conflicting coordinates or score columns")
                envelopes[identifier] = low, high
            else:
                columns[identifier] = current
                claims[identifier] = {k: None for k in ("pixel_sha256", "embedding_sha256", "encoding_identity")}
                memberships[identifier], canonical[identifier] = [], key
                envelopes[identifier] = current[1].copy(), current[1].copy()
            for name, value in data["claims"][identifier].items():
                if value is not None and claims[identifier][name] not in (None, value):
                    raise _Invalid("duplicate reference ID has conflicting declared pixel/embedding/encoding provenance")
                if value is not None:
                    claims[identifier][name] = value
            memberships[identifier].append(region["hypothesis_id"])
            if key < canonical[identifier]:
                columns[identifier], canonical[identifier] = current, key
    ids = sorted(columns)
    if not 1 <= len(ids) <= PROFILE["max_references"]:
        raise _Invalid("deduplicated complete union exceeds fixed reference budget")
    refs = np.array([columns[key][0] for key in ids], dtype=np.float64)
    scores = np.ascontiguousarray(np.column_stack([columns[key][1] for key in ids]), dtype=np.float32)
    return ids, refs, scores, claims, {key: sorted(value) for key, value in memberships.items()}, canonical


def localize_image_union(plan, *, timestamps, query_video_sha256, retrieve_evidence, query_cache=None):
    """Acquire every pool and return only the reference_mean full-union route.

    Callback: ``retrieve_evidence(region, *, n_query, query_mode, query_cache)``.
    Every response must have status='complete', full query_indices/query_times/
    similarities/ref_latlon/ref_ids and provenance. Its required query_identity
    keys are query_video_sha256, query_pixels_sha256, query_descriptors_sha256,
    actual_model_state_sha256, query_indices_sha256, query_times_sha256,
    query_mode='crops', views=3, aggregation=PROFILE['crop_aggregation']. Hash
    indices as int64 and times as float64 with _array_sha. Optional per-ID
    reference_identities has pixel_sha256, embedding_sha256, encoding_identity.

    Mandatory acquisition provenance contains status='complete',
    listing_complete=True, cap=1500, listed_references=[{id:unprefixed,lat,lon}],
    listing_snapshot_sha256 and unserved_references=[{ref_id:source-prefixed,
    reason:explicit}]. These must exactly partition the complete metadata;
    Mapillary fetch must have known tiles stats, lost=0 and the same recomputed
    snapshot. Graph paging/truncation and other provider completeness remain
    unknown, so they decline instead of being inferred from a valid matrix.

    All callbacks are attempted once and outcomes retained. Partial/unknown/
    failed regions decline the entire candidate. A cap is a maximum, not an
    assertion that every listed photo downloaded. Arrays use immutable bytes
    backing for external NPZ freezing; this function writes nothing. Original
    per-occurrence provenance is retained even when a duplicate contributes a
    known claim. A known claim does not authenticate unknown occurrences.
    """
    result = {"status": "inactive", "experimental": True, "confidence_calibrated": False,
              "uses_rigid_vo": False, "route_valid": False, "profile": dict(PROFILE),
              "profile_sha256": PROFILE_SHA256, "regions": [], "failed_regions": []}
    try:
        plan = _validate_plan(plan)
        result.update(plan_sha256=plan["plan_sha256"], region_plans=deepcopy(plan["region_plans"]),
                      retained_branch_count=plan["branch_count"], successful_branch_count=0)
        if plan["status"] != "ready":
            result["reason"] = plan["reason"]
            return result
        native = np.asarray(timestamps)
        start, end = _window(plan["ocr_provenance"])
        if (native.ndim != 1 or native.dtype.kind not in "fiu" or len(native) < PROFILE["queries"]
                or not np.isfinite(native).all() or np.any(np.diff(native.astype(float)) <= 0)
                or native[0] < start or native[-1] > end
                or not _digest(query_video_sha256) or query_video_sha256 != plan["ocr_provenance"]["video_sha256"]):
            raise _Invalid("query video/native timestamps do not match the OCR horizon")
        indices = np.linspace(0, len(native)-1, PROFILE["queries"]).astype(np.int64)
        times = np.asarray(native[indices], dtype=np.float64)
        if query_cache is None:
            query_cache = {}
        if not isinstance(query_cache, dict) or not callable(retrieve_evidence):
            raise _Invalid("an explicit query cache and evidence callback are required")
        accepted, common = [], None
        for region in plan["region_plans"]:
            outcome = {"hypothesis_id": region["hypothesis_id"], "request_sha256": region["request_sha256"]}
            try:
                raw = retrieve_evidence(deepcopy(region), n_query=PROFILE["queries"],
                                        query_mode=PROFILE["query_mode"], query_cache=query_cache)
            except Exception as error:
                outcome.update(status="failed", reason=f"retrieval callback raised {type(error).__name__}")
                result["failed_regions"].append(deepcopy(outcome))
                result["regions"].append(outcome)
                continue
            if isinstance(raw, Mapping):
                reported_status = raw.get("status")
                outcome["reported_status"] = reported_status if isinstance(reported_status, str) and reported_status in {"complete", "partial", "failed", "unavailable"} else "unknown"
                reported = raw.get("provenance")
                if isinstance(reported, Mapping):
                    try:
                        encoded = json.dumps(dict(reported), allow_nan=False)
                        if len(encoded.encode()) <= PROFILE["max_provenance_bytes_per_region"]:
                            outcome["provenance"] = json.loads(encoded)
                    except (TypeError, ValueError):
                        pass
            try:
                data = _evidence(raw, region, indices, times, query_video_sha256)
                if common is not None and data["common"] != common:
                    raise _Invalid("regions have different actual query/model/preprocessing identities")
                common = data["common"]
                accepted.append((region, data))
                outcome.update(status="complete", reference_count=len(data["ids"]),
                               evidence_hashes=data["hashes"], provenance=data["provenance"])
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                outcome.update(status="invalid", reason=str(error) if isinstance(error, _Invalid) else f"invalid evidence: {type(error).__name__}")
                result["failed_regions"].append(deepcopy(outcome))
            result["regions"].append(outcome)
        result["successful_branch_count"] = len(accepted)
        if len(accepted) != plan["branch_count"]:
            result["reason"] = "every planned branch must complete with compatible evidence"
            return result
        ids, refs, scores, claims, memberships, canonical = _union(accepted)
        # Core computes both variants internally, but only this fixed mean
        # output is exposed/admitted; no score-based normalization choice.
        decoded = decode_full_tracks(scores, ids, refs, times)
        chosen = decoded["outputs"]["reference_mean"]
        union_hashes = {"query_indices_sha256": _array_sha(indices), "query_times_sha256": _array_sha(times),
                        "ref_ids_sha256": _json_sha(ids), "ref_latlon_sha256": _array_sha(refs), "similarities_sha256": _array_sha(scores)}
        result.update(status="complete", route_valid=True, reason=None, point_count=len(times),
                      score_normalization="reference_mean", query_mode="crops", radius_m=3000, cap=1500,
                      **deepcopy({key: value for key, value in chosen.items() if key != "status"}))
        result["degenerate"] = bool(np.all(np.asarray(result["route_latlon"]) == result["route_latlon"][0]))
        result["reference_union"] = {"reference_count": len(ids), "occurrence_count": sum(len(d["ids"]) for _, d in accepted),
            "ref_ids": ids, "memberships": memberships, "canonical_occurrences": {k: list(v) for k, v in canonical.items()},
            "reference_identities": claims, "hashes": union_hashes,
            "query_indices": _readonly(indices), "query_times": _readonly(times),
            "ref_latlon": _readonly(refs), "similarities": _readonly(scores)}
        result["provenance"] = {"common_query_encoder_identity": common, "decoder_profile_sha256": decoded["profile_sha256"],
            "decoder_inputs": decoded["inputs"], "decoder_allocation": decoded["allocation"],
            "typed_union_hashes": union_hashes, "legacy_reference_encoding_unknown_preserved": True,
            "comparison_limit": "Crop-union scores are not calibrated probabilities or comparable warp confidence statistics"}
        return result
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        result["reason"] = str(error) if isinstance(error, _Invalid) else f"invalid inputs: {type(error).__name__}"
        return result
