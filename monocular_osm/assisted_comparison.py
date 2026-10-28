"""Truthful raw comparisons for opt-in v3; original v1 controls stay intact.

Decoder provenance is retained separately from the NEW encoder/query identity.
Only the union decision uses that new identity. Legacy gates keep their exact
original identity, including the original decoder.
"""
from __future__ import annotations
from copy import deepcopy
from collections import Counter
import numpy as np
from . import metadata_union as union

SCHEMA = "assisted-raw-reference-comparison-v3"
PROJECTION_SCHEMA = "original-baseline-raw-appearance-projection-v1"
PROFILE = {"schema": SCHEMA, "baseline_projection": PROJECTION_SCHEMA,
    "identity": "raw encoder/state/warp/native pixels/query descriptors/typed grid and times; excludes decoder",
    "legacy_gate": "unchanged original identity and decoder before new union",
    "union_primary": "reference_mean", "maximum_queries": 80,
    "metadata_union_profile_sha256": union.PROFILE_SHA256,
    "full_sequence_profile_sha256": union.full.PROFILE_SHA256}
PROFILE_SHA256 = union.json_sha(PROFILE)


def baseline_projection(comparison, original_summary):
    """A separate, verifiable projection; never changes the original fields."""
    validated, identity, indices, _ = union.legacy_summary(comparison, original_summary)
    return {"schema": PROJECTION_SCHEMA, "original_comparison": deepcopy(comparison),
        "original_summary": deepcopy(original_summary), "original_comparison_sha256": union.json_sha(comparison),
        "original_score_identity": validated["score_identity"], "original_similarity_median": validated["similarity_median"],
        "appearance_identity": identity, "appearance_identity_sha256": union.json_sha(identity), "query_count": len(indices)}


def validated_projection(projection, baseline):
    union.require(isinstance(projection, dict), "original baseline raw projection missing")
    expected = baseline_projection(projection["original_comparison"], projection["original_summary"])
    union.require(expected == projection, "baseline appearance projection changed")
    original = projection["original_summary"]
    union.require(all(baseline.get(key) == original.get(key) for key in
        ("status", "route_valid", "point_count", "score_kind", "score_identity", "similarity_median")),
        "baseline original v1 evidence was replaced or is stale")
    return expected


def augment_baseline_summary(comparison, original_summary):
    return {**deepcopy(original_summary), "metadata_union_baseline": baseline_projection(comparison, original_summary)}


def legacy_summary(comparison, route_latlon, *, expanded=False):
    """Derive original v1 evidence; no appearance-only legacy admission."""
    _, indices, _ = union.appearance(comparison)
    scores = np.asarray(comparison["decoded_reference_cosines"], np.float64)
    route = np.asarray(route_latlon, np.float64)
    union.require(route.shape == (len(indices), 2) and np.isfinite(route).all()
        and np.all(abs(route[:, 0]) <= 90) and np.all(abs(route[:, 1]) <= 180), "complete original route required")
    value = {"status": "complete", "route_valid": True, "point_count": len(route), "score_kind": "raw_cosine",
        "score_identity": union.json_sha({key: comparison[key] for key in ("statistic", "model", "query", "inner_decoder")}),
        "similarity_median": float(np.median(scores))}
    if expanded:
        coords = np.asarray(comparison["decoded_reference_latlon"], np.float64)
        value["dominant_coordinate_fraction"] = max(Counter(map(tuple, coords)).values())/len(coords)
    union.legacy_summary(comparison, value)
    return value


def union_comparison(result, original_baseline_comparison, original_baseline_summary,
                     legacy_metadata_comparison, legacy_metadata_summary, *, requested_pool_count):
    union.require(result.get("status") == "complete" and result.get("profile_sha256") == union.PROFILE_SHA256
        and set(result["outputs"]) == {"raw", "reference_mean"}, "both unchanged full-union results required")
    projection = baseline_projection(original_baseline_comparison, original_baseline_summary)
    output, proof = result["outputs"]["reference_mean"], result["comparisons"]["reference_mean"]
    value = {"schema": SCHEMA, "candidate_kind": "metadata_union_reference_mean", "statistic": union.STATISTIC,
        "baseline_projection": projection, "appearance_identity": deepcopy(proof["appearance_identity"]),
        "appearance_identity_sha256": proof["appearance_identity_sha256"],
        "actual_decoder": {"family": "full_sequence", "profile_sha256": union.full.PROFILE_SHA256,
            "variant": "reference_mean", "candidate_pruning": False},
        "metadata_union_profile_sha256": union.PROFILE_SHA256, "requested_pool_count": requested_pool_count,
        "complete_pool_count": requested_pool_count, "full_reference_count": len(result["union"]["ref_ids"]),
        "query_indices": result["union"]["query_indices"].tolist(), "query_times": result["union"]["query_times"].tolist(),
        "chosen_ref_ids": deepcopy(output["chosen_ref_ids"]), "raw_column_proof": deepcopy(proof),
        "route_latlon": deepcopy(output["route_latlon"]), "route_timestamps": deepcopy(output["route_timestamps"]),
        "legacy_metadata": {"comparison": deepcopy(legacy_metadata_comparison), "summary": deepcopy(legacy_metadata_summary)},
        "reference_encoding_identity": "UNVERIFIED legacy cache fingerprints retained", "confidence": None}
    union_summary(value, value["route_latlon"], value["route_timestamps"])
    return value


def union_summary(comparison, route_latlon, route_timestamps):
    """Policy summary from RAW selected columns and UNSMOOTHED coordinates."""
    c = comparison
    union.require(c.get("schema") == SCHEMA and c.get("candidate_kind") == "metadata_union_reference_mean"
        and c.get("statistic") == union.STATISTIC and c.get("metadata_union_profile_sha256") == union.PROFILE_SHA256,
        "fixed v3 union comparison required")
    decoder = {"family": "full_sequence", "profile_sha256": union.full.PROFILE_SHA256,
        "variant": "reference_mean", "candidate_pruning": False}
    union.require(c.get("actual_decoder") == decoder and "inner_decoder" not in c,
        "truthful reference-mean full-DP provenance required; no legacy decoder forgery")
    p = c["baseline_projection"]; validated_projection(p, p["original_summary"])
    identity = p["appearance_identity"]; identity_hash = union.json_sha(identity)
    union.require(c["appearance_identity"] == identity and c["appearance_identity_sha256"] == identity_hash,
        "new candidate and actual baseline raw appearance differ")
    indices = np.asarray(c["query_indices"]); times = np.asarray(c["query_times"], np.float64)
    union.require(indices.shape == times.shape == (80,) and indices.dtype.kind in "iu"
        and np.array_equal(indices, np.arange(80)) and np.isfinite(times).all() and np.all(np.diff(times) > 0)
        and union.full._array_sha(indices) == identity["query_indices_sha256"]
        and union.full._array_sha(times) == identity["query_times_sha256"], "complete fixed80 actual query grid required")
    count, complete, n = c["requested_pool_count"], c["complete_pool_count"], c["full_reference_count"]
    union.require(type(count) is int and 1 <= count <= 6 and type(complete) is int and complete == count
        and type(n) is int and 1 <= n <= 10000, "all requested bounded pools/references required")
    proof = c["raw_column_proof"]
    union.require(proof["variant"] == "reference_mean" and proof["actual_decoder_profile_sha256"] == union.full.PROFILE_SHA256
        and proof["appearance_identity"] == identity and proof["appearance_identity_sha256"] == identity_hash,
        "raw selected-column proof identity differs")
    raw = np.asarray(proof["raw_cosines"], np.float64); coords = np.asarray(proof["unsmoothed_reference_latlon"], np.float64)
    route, output_times = np.asarray(route_latlon, np.float64), np.asarray(route_timestamps, np.float64)
    union.require(raw.shape == (80,) and np.isfinite(raw).all() and np.max(abs(raw)) <= 1
        and coords.shape == route.shape == (80, 2) and np.isfinite(coords).all() and np.isfinite(route).all()
        and np.all(abs(coords[:, 0]) <= 90) and np.all(abs(coords[:, 1]) <= 180)
        and np.all(abs(route[:, 0]) <= 90) and np.all(abs(route[:, 1]) <= 180)
        and route.tolist() == c["route_latlon"] and np.array_equal(output_times, times)
        and c["route_timestamps"] == times.tolist(), "complete actual full-DP answer and raw proof required")
    rows, ids = proof["selected_source_rows"], c["chosen_ref_ids"]
    union.require(len(rows) == len(ids) == 80, "full path raw column proof missing")
    for q, row in enumerate(rows):
        union.require(row["query_row"] == q and row["query_index"] == int(indices[q]) and row["query_time"] == float(times[q])
            and type(row["global_column"]) is int and 0 <= row["global_column"] < n
            and isinstance(row["ref_id"], str) and row["ref_id"].startswith("mapillary:") and row["ref_id"] == ids[q]
            and isinstance(row["region"], str) and row["region"] and type(row["region_column"]) is int
            and 0 <= row["region_column"] < 1500 and row["raw_cosine"] == float(raw[q]), "selected global/pool/query raw column proof differs")
    median = float(np.median(raw)); dominance = max(Counter(map(tuple, coords)).values())/80
    union.require(proof["similarity_median"] == median and proof["dominant_coordinate_fraction"] == dominance,
        "raw admission scalar differs from actual selected evidence")
    return {"status": "complete", "route_valid": True, "point_count": 80, "score_kind": "raw_cosine",
        "score_identity": identity_hash, "similarity_median": median, "dominant_coordinate_fraction": dominance,
        "candidate_kind": c["candidate_kind"], "comparison_schema": SCHEMA, "actual_decoder": decoder,
        "metadata_union_profile_sha256": union.PROFILE_SHA256, "requested_pool_count": count, "complete_pool_count": complete,
        "full_reference_count": n, "baseline_projection_sha256": union.json_sha(p), "legacy_metadata": deepcopy(c["legacy_metadata"]),
        "comparison_evidence": deepcopy(c)}
