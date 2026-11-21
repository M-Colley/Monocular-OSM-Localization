"""Pure observed-reference alternatives, without identity upgrading.

Every physical source passes unchanged strict per-pool gates. Distinct
descriptor/raw-column versions remain alternatives; only exact keys fold.
No IO, model, field, decoder, selection, or confidence admission occurs here.
"""
from __future__ import annotations
from copy import deepcopy
from types import MappingProxyType
import numpy as np
from . import raw_vpr_bank as strict

PROFILE = MappingProxyType({
    "method": "all-physical-native80-reference-snapshot-mixture-v1",
    "strict_regional_validation_profile_sha256": strict.legacy.PROFILE_SHA256,
    "strict_input_primitives_profile_sha256": strict.PROFILE_SHA256,
    "maximum_calls": 64, "queries": 80, "maximum_pool_references": 1500,
    "maximum_reference_occurrences": 64 * 1500, "maximum_snapshots": 96000,
    "maximum_provider_ids": 30000, "maximum_score_values": 80 * 96000,
    "maximum_raw_matrix_bytes": 80 * 96000 * 4,
    "maximum_output_npz_bytes": 128 * 1024**2, "maximum_field_values": 16_000_000,
    "maximum_pool_descriptor_bytes": 1500 * 8448 * 4,
    "snapshot_key": "provider ID, computed protected F32 descriptor-row SHA, typed ORIGINAL F32 raw80-column SHA",
    "ordering": "lexical(provider ID, descriptor-row SHA, raw-column SHA)",
    "canonical_source": "lexical physical call key then original column",
    "same_provider_coordinates": "exact across ALL occurrences and snapshots",
    "provider_total_weight": 1., "snapshot_weight": "uniform Float64(1/K) for K distinct provider snapshots",
    "alias_weight": "exact snapshot key folds only after EVERY physical source validated",
    "actual_descriptor_bodies_required": True, "consumed_whole_tensor_sha_exact": True,
    "descriptor_vectors_retained": False, "descriptor_equivalence_assumed": False,
    "cross_version_L2_or_raw_envelope_test": False, "raw_score_averaging_or_normalization": False,
    "all_physical_aliases_required": True, "query_observations_per_time": 1,
    "reference_encoding_identity": "UNVERIFIED", "historical_encoder_verified": False,
    "old_strict_or_numeric_status_upgraded": False, "research_only": True,
    "field_built": False, "decoder_run": False, "policy_admission": False,
})
PROFILE_SHA256 = strict.legacy.json_sha(dict(PROFILE))

def _need(value, message):
    if not value: raise ValueError(message)

def _body(region, occurrences):
    _need(isinstance(region, dict), "complete loaded physical region required")
    ids = region.get("arrays", {}).get("ref_ids")
    _need(isinstance(ids, np.ndarray) and ids.ndim == 1, "saved provider ID rows required")
    n = len(ids)
    _need(1 <= n <= 1500 and occurrences + n <= PROFILE["maximum_reference_occurrences"],
          "whole bank exceeds physical occurrence bound")
    body = region.get("descriptor_rows")
    _need(isinstance(body, np.ndarray) and body.shape == (n, 8448)
          and body.dtype == np.float32 and body.flags.c_contiguous
          and body.nbytes <= PROFILE["maximum_pool_descriptor_bytes"],
          "EVERY physical pool requires actual bounded native C-F32 descriptor body")

def admit_bank(bank_key, records, common_comparison, native_facts, load_region):
    """Admit ALL declared snapshots or return whole-bank UNAVAILABLE.

    Loader ABI is {key,audit,arrays,descriptor_rows} from authenticated copies.
    Every physical alias passes full per-pool validation before its key folds.
    Only hashes and copied F32 columns survive into the next loader callback.
    Float64(1/K) represents provider total one mathematically; weight sums keep
    ordinary F64 reduction rounding, without a nonuniform final correction.
    """
    try:
        identity, indices, times, approved = strict._common(common_comparison, native_facts, records, bank_key)
        records = deepcopy(records)
        _need(callable(load_region), "physical-region loader required")
        providers, snapshots, inputs, occurrences = {}, {}, {}, 0
        for record in sorted(records, key=lambda r: r["key"]):
            region = load_region(deepcopy(record))
            _body(region, occurrences)
            ids, refs, scores, row_hashes, whole_sha = strict._region(
                record, region, identity, indices, times, approved)
            _need(row_hashes is not None and whole_sha is not None, "actual protected descriptor proof required")
            occurrences += len(ids)
            inputs[record["key"]] = {
                "source": deepcopy(record),
                "typed_arrays": deepcopy(region["audit"]["array_artifact"]["arrays"]),
                "metadata_array_pointer": {k: region["audit"]["array_artifact"][k] for k in ("path", "sha256")},
                "actual_descriptor_tensor_sha256": whole_sha,
                "reference_encoding_identity": "UNVERIFIED"}
            for col in range(len(ids)):
                provider = str(ids[col])
                previous = providers.get(provider)
                if previous is None:
                    _need(len(providers) < PROFILE["maximum_provider_ids"], "whole bank exceeds provider-ID bound")
                    providers[provider] = {"coordinate": refs[col].astype(np.float64, copy=True), "snapshot_count": 0}
                else:
                    _need(np.array_equal(previous["coordinate"], refs[col]),
                          "provider ID coordinates conflict across snapshot occurrences")
                column_sha = strict.legacy.full._array_sha(scores[:, col])
                key = (provider, row_hashes[col], column_sha)
                source = {"key": record["key"], "column": col}
                snapshot = snapshots.get(key)
                if snapshot is None:
                    _need(len(snapshots) < PROFILE["maximum_snapshots"], "whole bank exceeds distinct-snapshot bound")
                    snapshots[key] = {"raw": scores[:, col].copy(), "occurrences": [source]}
                    providers[provider]["snapshot_count"] += 1
                else:
                    snapshot["occurrences"].append(source)
            del region, ids, refs, scores, row_hashes
        ordered = sorted(snapshots)
        _need(80 * len(ordered) <= PROFILE["maximum_score_values"], "whole snapshot matrix exceeds value bound")
        raw = np.empty((80, len(ordered)), np.float32)
        coordinates = np.empty((len(ordered), 2), np.float64)
        weights = np.empty(len(ordered), np.float64)
        mapping = []
        for col, (provider, descriptor_sha, column_sha) in enumerate(ordered):
            snapshot = snapshots[(provider, descriptor_sha, column_sha)]
            raw[:, col] = snapshot["raw"]
            coordinates[col] = providers[provider]["coordinate"]
            weights[col] = 1. / providers[provider]["snapshot_count"]
            mapping.append({"provider_id": provider, "descriptor_row_sha256": descriptor_sha,
                "raw_column_sha256": column_sha, "canonical": deepcopy(snapshot["occurrences"][0]),
                "occurrences": snapshot["occurrences"]})
        return {"status": "complete_snapshot_research_evidence", "profile": dict(PROFILE),
            "profile_sha256": PROFILE_SHA256, "bank_key": bank_key,
            "provider_ids": np.asarray([key[0] for key in ordered]), "ref_latlon": coordinates,
            "raw_similarities": raw, "snapshot_weights": weights,
            "query_indices": indices, "query_times": times, "mapping": mapping, "inputs": inputs,
            "unique_provider_count": len(providers), "snapshot_count": len(ordered),
            "reference_occurrences": occurrences, "physical_calls_validated": len(records),
            "appearance_identity": identity, "appearance_identity_sha256": strict.legacy.json_sha(identity),
            "query_observations_per_time": 1, "reference_encoding_identity": "UNVERIFIED",
            "historical_encoder_verified": False, "descriptor_equivalence_assumed": False,
            "old_strict_or_numeric_status_upgraded": False, "field_built": False,
            "decoder_run": False, "policy_admission": False}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "bank_key": bank_key, "profile_sha256": PROFILE_SHA256,
            "reason": type(exc).__name__ + ": " + str(exc),
            "reference_encoding_identity": "UNVERIFIED", "historical_encoder_verified": False,
            "descriptor_equivalence_assumed": False, "old_strict_or_numeric_status_upgraded": False,
            "field_built": False, "decoder_run": False, "policy_admission": False}
