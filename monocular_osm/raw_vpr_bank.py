"""Pure admission of a complete, native-verified frozen raw-VPR bank.

The caller authenticates metadata/NPZ/descriptor copies before supplying a
region. This module performs no IO and confers no IO authority. Every physical
alias is validated; aliases never create additional query observations. No
decoder, likelihood field, model, or selection policy is run here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

import numpy as np

from . import metadata_union as legacy
from . import raw_vpr_native as native


ARRAYS = frozenset(("query_indices", "query_times", "ref_ids", "ref_latlon",
                   "raw_similarities", "decoded_ref_indices", "decoded_cosines",
                   "returned_indices", "returned_latlon", "returned_cosines"))
PROFILE = MappingProxyType({
    "method": "all-physical-native80-raw-vpr-bank-v1", "maximum_calls": 64,
    "queries": 80, "maximum_pool_references": 1500, "maximum_references": 10000,
    "maximum_reference_occurrences": 96000, "duplicate_raw_envelope": 1e-6,
    "canonical": "lexical physical call key then original column; lexical provider IDs",
    "overlap": "exact coordinates and computed protected descriptor row SHA",
    "all_physical_aliases_required": True, "query_observations_per_time": 1,
    "nested_alias_pointer": "identical file SHA; both array roles in full authenticated native inventory",
    "reference_encoding_identity": "UNVERIFIED", "research_only": True,
    "regional_validation_profile_sha256": legacy.PROFILE_SHA256,
    "encoder_only_causal_claim": False, "field_built": False,
    "decoder_run": False, "policy_admission": False,
})
PROFILE_SHA256 = legacy.json_sha(dict(PROFILE))


def _need(value, message):
    if not value:
        raise ValueError(message)


def tensor_sha(rows):
    """The original measured Torch-F32 tensor digest, without importing Torch."""
    _need(isinstance(rows, np.ndarray) and rows.dtype == np.float32
          and rows.flags.c_contiguous, "native contiguous float32 descriptor tensor required")
    h = hashlib.sha256(json.dumps([list(rows.shape), "torch.float32"], separators=(",", ":")).encode())
    h.update(memoryview(rows).cast("B"))
    return h.hexdigest()


def _common(comparison, facts, records, bank_key):
    _need(isinstance(bank_key, str) and bank_key and isinstance(records, list)
          and 1 <= len(records) <= PROFILE["maximum_calls"], "bounded named bank required")
    for record in records:
        _need(isinstance(record, dict) and set(record) == {"key", "request_ref", "metadata_ref", "array_ref"}
              and isinstance(record["key"], str) and record["key"], "singular physical call record required")
    # This genuine seam checks the COMPLETE named-bank membership, all three
    # source roles per alias, and its own replayed pixels/typed query clock.
    flat = native.validate_native_facts(facts, records, bank_key)
    indices = np.asarray(comparison["query"]["indices"])
    times = np.asarray(comparison["query"]["times"], np.float64)
    _need(indices.shape == (80,) and indices.dtype == np.dtype("int64")
          and indices[0] >= 0 and np.all(indices[1:] > indices[:-1])
          and times.shape == (80,) and np.isfinite(times).all()
          and np.all(times[1:] > times[:-1]), "exact signed-int64/native float64 ordered80 query grid required")
    identity, indices, times = legacy.appearance(comparison)
    expected = {"backbone": {k: v for k, v in comparison["model"].items() if k != "weights_sha256"},
                "actual_model_state_sha256": comparison["model"]["weights_sha256"],
                "preprocessing": comparison["query"]["preprocessing"], "query_mode": comparison["query"]["mode"],
                "query_pixels_sha256": comparison["query"]["pixels_sha256"],
                "query_descriptors_sha256": comparison["query"]["descriptors_sha256"]}
    for name, value in (("query_indices", indices), ("query_times", times)):
        expected[name + "_header"] = {"shape": list(value.shape), "dtype": value.dtype.str,
                                      "sha256": legacy.full._array_sha(value)}
    _need(flat == expected, "native facts and actual common model/query identity differ")
    approved_arrays = {}
    for call in facts["all_physical_calls_and_aliases"]:
        path, expected_sha = native.ref(call["array_ref"])
        _need(approved_arrays.setdefault(path, expected_sha) == expected_sha,
              "native inventory has conflicting array source hashes")
    return identity, indices.copy(), times.copy(), approved_arrays


def _region(record, region, identity, indices, times, approved_arrays):
    _need(isinstance(region, dict) and region.get("key") == record["key"], "loader returned another physical call")
    audit, arrays = region["audit"], region["arrays"]
    _need(isinstance(arrays, dict) and set(arrays) == ARRAYS, "all ten saved array roles required")
    artifact = audit["array_artifact"]
    pointer = native.ref({"path": artifact["path"], "sha256": artifact["sha256"]})
    physical = native.ref(record["array_ref"])
    # A nested byte-copy may retain its original metadata pointer. Both array
    # roles must occur in the complete authenticated native inventory and have
    # identical file SHA; this never authorizes an arbitrary protected path.
    _need(pointer[1] == physical[1] and approved_arrays.get(pointer[0]) == pointer[1],
          "metadata array pointer is not an authenticated same-SHA physical source")
    headers = artifact["arrays"]
    _need(isinstance(headers, dict) and set(headers) == ARRAYS, "all ten typed array headers required")
    _need(isinstance(arrays["ref_ids"], np.ndarray) and arrays["ref_ids"].ndim == 1,
          "one-dimensional saved reference IDs required")
    n = len(arrays["ref_ids"])
    _need(1 <= n <= 1500, "bounded pool required before typed hashing")
    shapes = {"query_indices": (80,), "query_times": (80,), "ref_ids": (n,), "ref_latlon": (n, 2),
              "raw_similarities": (80, n), "decoded_ref_indices": (80,), "decoded_cosines": (80,),
              "returned_indices": (80,), "returned_latlon": (80, 2), "returned_cosines": (80,)}
    for name, value in arrays.items():
        _need(isinstance(value, np.ndarray) and not value.dtype.hasobject
              and value.shape == shapes[name] and value.dtype.itemsize <= (512 if name == "ref_ids" else 8),
              "bounded plain saved ndarray required")
        _need(headers[name] == {"shape": list(value.shape), "dtype": value.dtype.str,
                              "sha256": legacy.full._array_sha(value)}, "actual array differs from saved typed header: " + name)
    # Keep the existing per-pool acquisition/identity/returned-track semantics.
    ids, refs, score, hashes = legacy.validate_region(region, identity, indices, times)
    descriptors = region.get("descriptor_rows")
    whole_sha = None
    if descriptors is not None:
        whole_sha = tensor_sha(descriptors)
        _need(whole_sha == audit.get("ref_embeddings_sha256"),
              "actual protected descriptor tensor differs from original measured SHA")
    return ids, refs, score, hashes, whole_sha


def admit_bank(bank_key, records, common_comparison, native_facts, load_region):
    """Return complete research evidence or whole-bank UNAVAILABLE, never a subset.

    ``records`` is the exact named-bank list of singular physical call records.
    ``load_region(record)`` supplies ``{key,audit,arrays,descriptor_rows}`` from
    authenticated frozen copies, with the ten saved array roles. Optional
    descriptors have native contiguous F32 shape (N,8448); every overlap needs
    actual descriptor rows for ALL its occurrences, including physical aliases.
    The loader must not retain earlier pools. No descriptor arrays or views are
    retained by this function when the next loader is invoked.
    """
    try:
        identity, indices, times, approved_arrays = _common(common_comparison, native_facts, records, bank_key)
        records = deepcopy(records)
        _need(callable(load_region), "region loader required")
        entries, mapping_inputs, occurrences = {}, {}, 0
        for record in sorted(records, key=lambda r: r["key"]):
            region = load_region(deepcopy(record))
            ids, refs, score, hashes, whole_sha = _region(record, region, identity, indices, times, approved_arrays)
            occurrences += len(ids)
            _need(occurrences <= PROFILE["maximum_reference_occurrences"], "bank occurrence cap exceeded")
            mapping_inputs[record["key"]] = {
                "source": deepcopy(record), "typed_arrays": deepcopy(region["audit"]["array_artifact"]["arrays"]),
                "metadata_array_pointer": {k: region["audit"]["array_artifact"][k] for k in ("path", "sha256")},
                "actual_descriptor_tensor_sha256": whole_sha,
                "reference_encoding_identity": "UNVERIFIED"}
            for col in range(len(ids)):
                identifier, row_sha = str(ids[col]), hashes[col] if hashes is not None else None
                occurrence = {"key": record["key"], "column": col, "descriptor_sha256": row_sha}
                current = entries.get(identifier)
                if current is None:
                    _need(len(entries) < PROFILE["maximum_references"], "whole bank exceeds unique-reference cap")
                    entries[identifier] = {
                        "coordinate": refs[col].astype(np.float64, copy=True), "score": score[:, col].copy(),
                        "minimum": score[:, col].astype(np.float64), "maximum": score[:, col].astype(np.float64),
                        "descriptor_sha256": row_sha, "occurrences": [occurrence]}
                else:
                    _need(np.array_equal(current["coordinate"], refs[col]), "duplicate provider ID coordinate conflict")
                    _need(row_sha is not None and row_sha == current["descriptor_sha256"],
                          "overlap requires identical computed protected descriptor rows")
                    np.minimum(current["minimum"], score[:, col], out=current["minimum"])
                    np.maximum(current["maximum"], score[:, col], out=current["maximum"])
                    _need(np.max(current["maximum"] - current["minimum"]) <= 1e-6,
                          "duplicate raw-score envelope conflict")
                    current["occurrences"].append(occurrence)
            # Do not leave even a row/block view retaining the previous tensor.
            del region, ids, refs, score, hashes
        ordered = sorted(entries)
        matrix = np.empty((80, len(ordered)), np.float32)
        coordinates = np.empty((len(ordered), 2), np.float64)
        mapping = []
        for col, identifier in enumerate(ordered):
            entry = entries[identifier]
            matrix[:, col], coordinates[col] = entry["score"], entry["coordinate"]
            mapping.append({"ref_id": identifier, "canonical": deepcopy(entry["occurrences"][0]),
                            "occurrences": entry["occurrences"],
                            "maximum_raw_score_spread": float(np.max(entry["maximum"] - entry["minimum"]))})
        return {"status": "complete_research_bank", "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
                "bank_key": bank_key, "ref_ids": np.asarray(ordered), "ref_latlon": coordinates,
                "raw_similarities": matrix, "query_indices": indices, "query_times": times,
                "mapping": mapping, "inputs": mapping_inputs, "reference_occurrences": occurrences,
                "physical_calls_validated": len(records), "appearance_identity": identity,
                "appearance_identity_sha256": legacy.json_sha(identity),
                "reference_encoding_identity": "UNVERIFIED", "query_observations_per_time": 1,
                "field_built": False, "policy_admission": False}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "bank_key": bank_key, "profile_sha256": PROFILE_SHA256,
                "reason": type(exc).__name__ + ": " + str(exc), "reference_encoding_identity": "UNVERIFIED",
                "field_built": False, "policy_admission": False}
