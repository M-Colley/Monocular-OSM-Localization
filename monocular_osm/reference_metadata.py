"""Pure planning/parsing of bounded public Mapillary reference metadata.

No network, models, query coordinates, scores or labels. This experimental
availability probe preserves every frozen gallery ID, including unknowns.
Metadata acquisition does not select a camera position or a route.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from types import MappingProxyType
import numpy as np
from .geo import geodesic_distance_m

FIELDS = ("id", "sequence", "captured_at", "camera_type", "compass_angle", "computed_compass_angle", "computed_rotation")
PROFILE = MappingProxyType({"version": "mapillary-all-gallery-metadata-v1", "experimental": True,
    "endpoint": "https://graph.mapillary.com/images", "fields": FIELDS,
    "batch_size": 50, "workers": 2, "requests_per_second": 1., "max_attempts_per_batch": 2,
    "max_requests": 390, "new_request_deadline_s": 1200., "connect_timeout_s": 10., "read_timeout_s": 30.,
    "max_decoded_body_bytes": 2*1024**2, "max_total_decoded_body_bytes": 64*1024**2,
    "retry_statuses": (429, 500, 502, 503, 504), "retry_delay_s": 2., "max_retry_after_s": 60.,
    "fatal_statuses": (400, 401, 403), "first_batch_compatibility": True,
    "order": "sha256(version+NUL+provider-qualified-ID), then ID; never location or retrieval scores",
    "missingness": "all supplied IDs retained; absent/null/invalid fields distinguished; no inference from absence",
    "auth": "MLY_TOKEN via Authorization: OAuth header only; no SDK validation request",
    "schema_or_endpoint_fallback": False, "images_or_models_or_query_gps": False,
    "selected_answer": None, "validation_status": "metadata_availability_only_posthoc_DEV"})


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def profile_record():
    return json.loads(json.dumps(dict(PROFILE)))


def plan_requests(ref_ids, ref_latlon, *, provenance=None):
    ids, ll = np.asarray(ref_ids), np.asarray(ref_latlon)
    maximum = (PROFILE["max_requests"]//PROFILE["max_attempts_per_batch"])*PROFILE["batch_size"]
    if (ids.ndim != 1 or not 1 <= len(ids) <= maximum or ids.dtype.kind not in "US" or ids.dtype.itemsize > 512
            or ll.shape != (len(ids), 2) or ll.dtype.kind not in "fiu"):
        raise ValueError("complete bounded provider-qualified Mapillary IDs and frozen reference geotags required")
    ll64 = ll.astype(np.float64)
    if not np.isfinite(ll64).all() or np.any(np.abs(ll64[:, 0]) > 90) or np.any(np.abs(ll64[:, 1]) > 180):
        raise ValueError("reference geotags must be finite WGS84")
    text = ids.astype(str).tolist()
    if len(set(text)) != len(text) or any(re.fullmatch(r"mapillary:[1-9][0-9]{0,31}", value) is None for value in text):
        raise ValueError("distinct public Mapillary image IDs required")
    references = [{"reference_id": value, "image_id": value.split(":", 1)[1],
        "lat": float(position[0]), "lon": float(position[1])} for value, position in zip(text, ll64)]
    shuffled = sorted(text, key=lambda value: (hashlib.sha256((PROFILE["version"]+"\0"+value).encode()).hexdigest(), value))
    batches = []
    for start in range(0, len(shuffled), PROFILE["batch_size"]):
        batch = shuffled[start:start+PROFILE["batch_size"]]
        batches.append({"index": len(batches), "reference_ids": batch,
            "params": {"image_ids": ",".join(value.split(":", 1)[1] for value in batch), "fields": ",".join(FIELDS)}})
    return {"profile": profile_record(), "profile_sha256": json_sha(profile_record()), "reference_count": len(text),
        "source_ids_sha256": array_sha(ids), "source_latlon_sha256": array_sha(ll),
        "references": references, "batches": batches, "provenance": dict(provenance or {}),
        "all_ids_planned": True, "selected_answer": None}


def unknown_record(reference_id, *, request_status="not_requested", reason=None):
    return {"reference_id": reference_id, "image_id": reference_id.split(":", 1)[1],
        "image_status": "unknown", "request_status": request_status, "reason": reason,
        "fields": {name: None for name in FIELDS if name != "id"},
        "field_status": {name: "unknown" for name in FIELDS if name != "id"}}


def _real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _field(name, value):
    if value is None:
        return None, "null"
    if name == "sequence":
        value = value.get("id") if isinstance(value, dict) else value
        if isinstance(value, str) and 0 < len(value) <= 256:
            return value, "present"
    elif name == "captured_at":
        if _real(value) and 0 <= value <= 2**53 and int(value) == value:
            return int(value), "present"
    elif name == "camera_type":
        if isinstance(value, str) and 0 < len(value) <= 64:
            return value, "present"
    elif name in ("compass_angle", "computed_compass_angle"):
        if _real(value) and 0 <= value <= 360:
            return float(value) % 360., "present"
    elif name == "computed_rotation":
        if isinstance(value, list) and len(value) == 3 and all(_real(v) for v in value):
            return [float(v) for v in value], "present"
    return None, "invalid"


def parse_response(body, reference_ids):
    """Strict batch envelope/ID membership; optional missing fields are valid.

    A partial successful response identifies omitted images as unknown. An
    unexpected/duplicate ID or unsupported pagination invalidates the entire
    batch, rather than accepting a hand-picked subset.
    """
    if not isinstance(body, bytes) or len(body) > PROFILE["max_decoded_body_bytes"]:
        raise ValueError("bounded decoded response bytes required")
    ids = list(reference_ids)
    expected = {value.split(":", 1)[1]: value for value in ids}
    if not 1 <= len(ids) <= PROFILE["batch_size"] or len(expected) != len(ids):
        raise ValueError("distinct planned batch required")
    value = json.loads(body)
    if (not isinstance(value, dict) or not isinstance(value.get("data"), list) or len(value["data"]) > len(ids)
            or isinstance(value.get("paging"), dict) and value["paging"].get("next")):
        raise ValueError("unsupported metadata response envelope or pagination")
    result = {key: unknown_record(key, request_status="complete", reason="omitted_from_successful_response") for key in ids}
    seen = set()
    for item in value["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] not in expected or item["id"] in seen:
            raise ValueError("unexpected, duplicate or invalid response image ID")
        seen.add(item["id"])
        key = expected[item["id"]]
        record = unknown_record(key, request_status="complete")
        record.update(image_status="available", reason=None)
        for field in FIELDS[1:]:
            parsed, status = _field(field, item[field]) if field in item else (None, "absent")
            record["fields"][field], record["field_status"][field] = parsed, status
        result[key] = record
    return result


def summarize(plan, records):
    references = {row["reference_id"]: row for row in plan["references"]}
    if set(records) != set(references):
        raise ValueError("metadata summary must retain exactly every planned reference")
    available = [row for row in records.values() if row["image_status"] == "available"]
    field_counts = {field: {status: sum(row["field_status"][field] == status for row in records.values())
        for status in ("present", "absent", "null", "invalid", "unknown")} for field in FIELDS[1:]}
    groups = {}
    for row in available:
        sequence = row["fields"]["sequence"]
        if sequence is not None:
            groups.setdefault(sequence, []).append(row)
    gaps, ties, ordered_edges, ambiguous_edges = [], 0, 0, 0
    for rows in groups.values():
        timed = [row for row in rows if row["fields"]["captured_at"] is not None]
        timed.sort(key=lambda row: (row["fields"]["captured_at"], row["reference_id"]))
        frequencies = {}
        for row in timed:
            value = row["fields"]["captured_at"]
            frequencies[value] = frequencies.get(value, 0)+1
        for first, second in zip(timed, timed[1:]):
            if first["fields"]["captured_at"] == second["fields"]["captured_at"]:
                ties += 1  # not treated as evidence of an ordered edge
                continue
            if frequencies[first["fields"]["captured_at"]] != 1 or frequencies[second["fields"]["captured_at"]] != 1:
                ambiguous_edges += 1
                continue  # never choose one member of a tied timestamp group by ID
            a, b = references[first["reference_id"]], references[second["reference_id"]]
            gaps.append(float(geodesic_distance_m(a["lat"], a["lon"], b["lat"], b["lon"])))
            ordered_edges += 1
    return {"reference_count": len(records), "available_images": len(available),
        "unknown_images": len(records)-len(available), "field_status_counts": field_counts,
        "sequence_count": len(groups), "singleton_sequences": sum(len(rows) == 1 for rows in groups.values()),
        "repeated_sequences": sum(len(rows) >= 2 for rows in groups.values()),
        "references_in_repeated_sequences": sum(len(rows) for rows in groups.values() if len(rows) >= 2),
        "strictly_timed_sampled_edges": ordered_edges, "equal_time_adjacent_pairs": ties,
        "adjacencies_touching_tied_timestamps": ambiguous_edges,
        "sampled_gap_m_quantiles": dict(zip(("min", "p25", "median", "p75", "max"), np.quantile(gaps, [0, .25, .5, .75, 1]).tolist())) if gaps else None,
        "spherical_images": sum(row["fields"]["camera_type"] == "spherical" for row in available),
        "selected_answer": None, "query_or_GT_comparison": False,
        "scope": "only frozen gallery members; sampled sequence edges do not prove complete traverses or road geometry"}
