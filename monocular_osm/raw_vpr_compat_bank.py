"""Separate numerical-compatibility study of complete raw VPR banks.

The caller authenticates all inputs. Original per-pool gates and consumed
whole-tensor hashes remain strict. Cross-pool distance is measured to one fixed
lexical canonical F32 row, without attesting bitwise identity or a common
historical encoder. No IO, field, decoder, model, or policy runs here.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
import numpy as np
from . import raw_vpr_bank as strict


PROFILE = MappingProxyType({
    "method": "all-physical-native80-numeric-compatible-raw-vpr-bank-v1",
    "strict_bank_profile_sha256": strict.PROFILE_SHA256,
    "regional_validation_profile_sha256": strict.legacy.PROFILE_SHA256,
    "maximum_calls": 64, "queries": 80, "maximum_pool_references": 1500,
    "maximum_references": 30000, "maximum_reference_occurrences": 64 * 1500,
    "maximum_score_values": 2400000,
    "descriptor_dimensions": 8448, "descriptor_dtype": "float32 C-contiguous",
    "maximum_canonical_descriptor_bytes": 1013760000,
    "maximum_pool_descriptor_bytes": 50688000,
    "maximum_validation_finite_mask_bytes": 12672000,
    "distance_workspace_bytes": 67584,
    "descriptor_distance": "Float64 L2 to fixed lexical canonical F32 row for EVERY occurrence",
    "maximum_descriptor_l2": 1e-6,
    "canonical": "lexical physical call key then original column; lexical provider IDs",
    "canonical_storage": "independent C-F32 row copies; validation-only, never returned",
    "duplicate_coordinates": "exact", "duplicate_raw_envelope": 1e-6,
    "all_physical_aliases_required": True, "actual_descriptor_bodies_required": True,
    "original_consumed_whole_tensor_sha_required": True,
    "query_observations_per_time": 1, "compatibility_not_bitwise_identity": True,
    "gallery_budget": "linear unique vectors and 80xN raw matrix; no quadratic decoder",
    "reference_encoding_identity": "UNVERIFIED", "historical_encoder_verified": False,
    "research_only": True, "field_built": False, "decoder_run": False,
    "policy_admission": False,
})
PROFILE_SHA256 = strict.legacy.json_sha(dict(PROFILE))


def _need(value, message):
    if not value:
        raise ValueError(message)


def _pool_body(region, occurrences):
    """Cheap shape/accounting checks before genuine per-pool typed hashing."""
    _need(isinstance(region, dict), "complete loaded region required")
    ids = region.get("arrays", {}).get("ref_ids")
    _need(isinstance(ids, np.ndarray) and ids.ndim == 1,
          "one-dimensional saved reference IDs required")
    n = len(ids)
    _need(1 <= n <= PROFILE["maximum_pool_references"]
          and occurrences + n <= PROFILE["maximum_reference_occurrences"],
          "whole bank exceeds pool/declared occurrence budget")
    body = region.get("descriptor_rows")
    _need(isinstance(body, np.ndarray) and body.dtype == np.float32
          and body.shape == (n, PROFILE["descriptor_dimensions"])
          and body.flags.c_contiguous
          and body.nbytes <= PROFILE["maximum_pool_descriptor_bytes"],
          "EVERY pool needs actual bounded contiguous F32 descriptor rows")
    return body


def admit_bank(bank_key, records, common_comparison, native_facts, load_region):
    """Same source/loader ABI as strict admission, with a distinct study status.

    All records/aliases are required. The loader supplies authenticated
    {key,audit,arrays,descriptor_rows} and must not retain earlier pools. Only
    independent canonical F32 copies survive each callback; loaded arrays,
    views, and original descriptor bodies are released before the next one.
    """
    try:
        identity, indices, times, approved = strict._common(
            common_comparison, native_facts, records, bank_key)
        records = deepcopy(records)
        _need(callable(load_region), "region loader required")
        entries, inputs, occurrences, maximum_l2 = {}, {}, 0, 0.
        for record in sorted(records, key=lambda r: r["key"]):
            region = load_region(deepcopy(record))
            descriptors = _pool_body(region, occurrences)
            ids, refs, score, hashes, whole_sha = strict._region(
                record, region, identity, indices, times, approved)
            _need(whole_sha is not None and hashes is not None,
                  "actual original tensor and row hashes required for EVERY pool")
            occurrences += len(ids)
            inputs[record["key"]] = {
                "source": deepcopy(record),
                "typed_arrays": deepcopy(region["audit"]["array_artifact"]["arrays"]),
                "metadata_array_pointer": {k: region["audit"]["array_artifact"][k]
                                           for k in ("path", "sha256")},
                "actual_descriptor_tensor_sha256": whole_sha,
                "reference_encoding_identity": "UNVERIFIED"}
            for col in range(len(ids)):
                identifier, row_sha = str(ids[col]), hashes[col]
                current = entries.get(identifier)
                if current is None:
                    count = len(entries) + 1
                    _need(count <= PROFILE["maximum_references"]
                          and 80 * count <= PROFILE["maximum_score_values"]
                          and count * 8448 * 4 <= PROFILE["maximum_canonical_descriptor_bytes"],
                          "whole bank exceeds linear canonical/gallery budget; never truncate")
                    # A view would keep the complete prior pool alive.
                    canonical = descriptors[col].copy(order="C")
                    current = entries[identifier] = {
                        "descriptor": canonical,
                        "coordinate": refs[col].astype(np.float64, copy=True),
                        "score": score[:, col].copy(),
                        "minimum": score[:, col].astype(np.float64),
                        "maximum": score[:, col].astype(np.float64),
                        "occurrences": [], "maximum_descriptor_l2_from_canonical": 0.}
                else:
                    _need(np.array_equal(current["coordinate"], refs[col]),
                          "duplicate provider ID coordinate conflict")
                # One F64 row workspace; no full-pool cast or norm temporary.
                # Square/reduce matches the earlier F64 L2 study, not a chain.
                delta = descriptors[col].astype(np.float64)
                delta -= current["descriptor"]
                np.square(delta, out=delta)
                distance = float(np.sqrt(np.sum(delta, dtype=np.float64)))
                del delta
                _need(np.isfinite(distance) and distance <= PROFILE["maximum_descriptor_l2"],
                      "descriptor L2 to fixed canonical exceeds numerical compatibility bound")
                np.minimum(current["minimum"], score[:, col], out=current["minimum"])
                np.maximum(current["maximum"], score[:, col], out=current["maximum"])
                _need(np.max(current["maximum"] - current["minimum"]) <= PROFILE["duplicate_raw_envelope"],
                      "duplicate raw-score envelope conflict")
                occurrence = {"key": record["key"], "column": col,
                              "descriptor_sha256": row_sha,
                              "descriptor_l2_from_canonical": distance}
                current["occurrences"].append(occurrence)
                current["maximum_descriptor_l2_from_canonical"] = max(
                    current["maximum_descriptor_l2_from_canonical"], distance)
                maximum_l2 = max(maximum_l2, distance)
            del region, descriptors, ids, refs, score, hashes
        # Descriptor validation is finished; do not overlap the large gallery
        # with output construction or expose validation vectors as evidence.
        for entry in entries.values():
            del entry["descriptor"]
        del canonical
        ordered = sorted(entries)
        matrix = np.empty((80, len(ordered)), np.float32)
        coordinates = np.empty((len(ordered), 2), np.float64)
        mapping = []
        for col, identifier in enumerate(ordered):
            entry = entries[identifier]
            matrix[:, col], coordinates[col] = entry["score"], entry["coordinate"]
            mapping.append({"ref_id": identifier, "canonical": deepcopy(entry["occurrences"][0]),
                "occurrences": entry["occurrences"],
                "maximum_descriptor_l2_from_canonical": entry["maximum_descriptor_l2_from_canonical"],
                "maximum_raw_score_spread": float(np.max(entry["maximum"] - entry["minimum"]))})
        return {"status": "complete_compatible_research_bank", "profile": dict(PROFILE),
            "profile_sha256": PROFILE_SHA256, "bank_key": bank_key,
            "ref_ids": np.asarray(ordered), "ref_latlon": coordinates,
            "raw_similarities": matrix, "query_indices": indices, "query_times": times,
            "mapping": mapping, "inputs": inputs, "reference_occurrences": occurrences,
            "physical_calls_validated": len(records), "appearance_identity": identity,
            "appearance_identity_sha256": strict.legacy.json_sha(identity),
            "maximum_descriptor_l2_from_canonical": maximum_l2,
            "compatibility_not_bitwise_identity": True,
            "historical_encoder_verified": False, "reference_encoding_identity": "UNVERIFIED",
            "query_observations_per_time": 1, "field_built": False,
            "decoder_run": False, "policy_admission": False}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "bank_key": bank_key, "profile_sha256": PROFILE_SHA256,
            "reason": type(exc).__name__ + ": " + str(exc),
            "compatibility_not_bitwise_identity": True, "historical_encoder_verified": False,
            "reference_encoding_identity": "UNVERIFIED", "field_built": False,
            "decoder_run": False, "policy_admission": False}
