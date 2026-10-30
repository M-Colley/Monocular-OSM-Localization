"""Fixed posthoc full-support metadata replay; no IO, models or policy adoption.

The new comparison attests raw encoder/query identity and selected matrix
columns. It never labels the full DP as the legacy regional decoder.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType
import numpy as np
from . import full_sequence as full, assisted_policy as policy


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


PROFILE = {"version": "metadata-full-support-replay-v1", "posthoc_DEV": True,
    "proposal_file_sha256": "9e6ee0236d3f9969fd76d1683bcf8bb856b02df6dc0529335a824c61642a371f",
    "proposal_profile_sha256": "21abf71b66d2ca2821e48635672626160407abc3a2a73296da06171784377516",
    "research_outputs": ["raw", "reference_mean"], "sole_new_primary": "reference_mean",
    "full_sequence_profile_sha256": full.PROFILE_SHA256, "admission_policy_config_sha256": policy.CONFIG_SHA256,
    "comparison_schema": "raw-appearance-selected-column-proof-v1",
    "maximum_regions": 6, "maximum_queries": 80, "maximum_references": 10000,
    "maximum_pool_references": 1500, "raw_dtype": "float32", "query_mode": "warp",
    "backbone_hub": "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db",
    "dedup": "lexical providerID; canonical lexical region key/column; exact coordinates; score envelope1e-6; identical declared per-ID descriptors for overlap",
    "duplicate_score_atol": 1e-6, "all_regions_required": True, "candidate_pruning": False,
    "score_for_admission": "RAW cosine at selected full-DP reference, never centered emission/objective",
    "dominance": "exact unsmoothed chosen-reference coordinates before median5/interpolation",
    "precedence": "already admitted legacy metadata wins; raw diagnostic never selects",
    "original_comparison_proof": "baseline selected_audit_rows maps ALL own frozen audit pools; metadata maps full union and original regional winners",
    "maximum_baseline_audits": 8,
    "reference_encoding_identity": "UNVERIFIED legacy fingerprints retained, no encoder-only causal claim",
    "default_adopted": False, "confidence": None}
PROFILE_SHA256 = json_sha(PROFILE)
PROFILE = MappingProxyType(PROFILE)
STATISTIC = "raw_cosine_at_decoded_reference"


def require(value, message):
    if not value: raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def appearance(comparison):
    require(comparison.get("statistic") == STATISTIC, "raw decoded-reference statistic required")
    model, query = comparison["model"], comparison["query"]
    require(model.get("name") == "megaloc" and model.get("hub") == PROFILE["backbone_hub"] and digest(model.get("weights_sha256"))
        and query.get("mode") == "warp" and isinstance(query.get("preprocessing"), dict)
        and bool(query["preprocessing"]) and digest(query.get("pixels_sha256")) and digest(query.get("descriptors_sha256")),
        "declared raw encoder/warp/native query identity required")
    indices, times = np.asarray(query["indices"]), np.asarray(query["times"], np.float64)
    require(indices.ndim == 1 and indices.dtype.kind in "iu" and 2 <= len(indices) <= 80
        and indices[0] >= 0 and np.all(np.diff(indices) > 0) and times.shape == indices.shape
        and np.isfinite(times).all() and np.all(np.diff(times) > 0), "actual ordered query grid required")
    identity = {"schema": "raw-encoder-query-v1", "statistic": STATISTIC, "model": deepcopy(model),
        "query": {k: deepcopy(query[k]) for k in ("mode", "preprocessing", "pixels_sha256", "descriptors_sha256")},
        "query_indices_sha256": full._array_sha(indices), "query_times_sha256": full._array_sha(times)}
    return identity, indices, times


def legacy_summary(comparison, supplied):
    identity, indices, times = appearance(comparison)
    decoder = comparison["inner_decoder"]
    require(decoder.get("actual") == "viterbi" and decoder.get("fallback") is False
        and isinstance(decoder.get("settings"), dict) and decoder["settings"], "truthful complete legacy Viterbi comparison required")
    legacy_id = json_sha({k: comparison[k] for k in ("statistic", "model", "query", "inner_decoder")})
    values = np.asarray(comparison["decoded_reference_cosines"], np.float64)
    coords = np.asarray(comparison["decoded_reference_latlon"], np.float64)
    require(values.shape == indices.shape and np.isfinite(values).all() and np.max(abs(values)) <= 1
        and coords.shape == (len(indices), 2) and np.isfinite(coords).all()
        and np.all(abs(coords[:, 0]) <= 90) and np.all(abs(coords[:, 1]) <= 180), "finite original raw evidence required")
    expected = {"status": "complete", "route_valid": True, "point_count": len(indices), "score_kind": "raw_cosine",
        "score_identity": legacy_id, "similarity_median": float(np.median(values))}
    require(all(supplied.get(k) == v for k, v in expected.items()), "original summary differs from actual raw comparison")
    expected["dominant_coordinate_fraction"] = max(Counter(map(tuple, coords)).values())/len(coords)
    if "dominant_coordinate_fraction" in supplied:
        require(supplied["dominant_coordinate_fraction"] == expected["dominant_coordinate_fraction"], "original unsmoothed dominance differs")
    return expected, identity, indices, times


def legacy_gate(baseline, candidate):
    """Reuse the EXACT existing expansion gate; no new threshold arithmetic."""
    return policy._expanded({"expanded": candidate}, {}, True, baseline["similarity_median"], baseline["score_identity"])


def validate_region(region, identity, indices, times):
    require(isinstance(region, dict) and isinstance(region.get("key"), str) and region["key"], "named complete region required")
    a, z = region["audit"], region["arrays"]
    require(a.get("status") == "complete" and not a.get("issues"), "all regional audits must be complete")
    ids, refs, score = np.asarray(z["ref_ids"]), np.asarray(z["ref_latlon"]), np.asarray(z["raw_similarities"])
    n = len(ids)
    require(ids.shape == (n,) and ids.dtype.kind in "US" and ids.dtype.itemsize <= 512 and 1 <= n <= 1500
        and len(set(ids.tolist())) == n and all(isinstance(v, str) and v.startswith("mapillary:") for v in ids.tolist()), "bounded unique provider IDs required")
    require(refs.shape == (n, 2) and refs.dtype.kind == "f" and np.isfinite(refs).all()
        and np.all(abs(refs[:, 0]) <= 90) and np.all(abs(refs[:, 1]) <= 180)
        and score.shape == (len(indices), n) and score.dtype == np.float32
        and np.isfinite(score).all() and np.max(abs(score)) <= 1.0001, "bounded finite complete raw matrix required")
    require(full._array_sha(z["query_indices"]) == full._array_sha(indices)
        and full._array_sha(z["query_times"]) == full._array_sha(times), "regional actual query grid differs")
    comparison = {"statistic": a["score_kind"], "model": {**a["backbone"], "weights_sha256": a["actual_model_state_sha256"]},
        "query": {"indices": np.asarray(z["query_indices"]).tolist(), "times": np.asarray(z["query_times"]).tolist(),
            "mode": a["query_mode"], "preprocessing": a["preprocessing"], "pixels_sha256": a["query_pixels_sha256"],
            "descriptors_sha256": a["query_descriptors_sha256"]}}
    require(appearance(comparison)[0] == identity, "regional actual appearance identity differs")
    original = {k: a[k] for k in ("query_pixels_sha256", "query_descriptors_sha256", "actual_model_state_sha256",
        "backbone", "preprocessing", "query_mode", "score_kind", "decoder")}
    original.update(query_indices_sha256=full._array_sha(indices), query_times_sha256=full._array_sha(times))
    require(a.get("comparison_identity") == original and a.get("comparison_identity_sha256") == json_sha(original), "regional identity proof differs")
    chosen = np.asarray(z["decoded_ref_indices"])
    require(chosen.shape == indices.shape and chosen.dtype.kind in "iu" and chosen.min() >= 0 and chosen.max() < n,
            "original regional winners missing")
    require(np.array_equal(z["decoded_cosines"], score[np.arange(len(indices)), chosen])
        and np.array_equal(z["returned_indices"], indices) and np.array_equal(z["returned_latlon"], refs[chosen])
        and np.array_equal(z["returned_cosines"], score[np.arange(len(indices)), chosen]), "original regional raw controls differ")
    pool = a["reference_pool"]; fetch = pool["fetch"]
    require(pool.get("source") == "mapillary" and pool.get("count") == n and fetch.get("listing") == "tiles"
        and fetch.get("listing_stats_known") is True and type(fetch.get("lost")) is int and fetch["lost"] == 0
        and fetch.get("snapshot", {}).get("n_refs") == n, "complete capped served tile inventory required")
    row_hashes = a.get("reference_row_embeddings_sha256")
    require(row_hashes is None or (isinstance(row_hashes, list) and len(row_hashes) == n and all(digest(v) for v in row_hashes)),
            "invalid per-reference descriptor evidence")
    # A string claim alone is insufficient for a future overlapping pool.
    # Callers must supply protected actual descriptor rows when dedup needs it.
    descriptors = region.get("descriptor_rows")
    verified_hashes = None
    if descriptors is not None:
        descriptors = np.asarray(descriptors)
        require(descriptors.shape == (n, 8448) and descriptors.dtype == np.float32
            and np.isfinite(descriptors).all(), "protected actual MegaLoc descriptor rows required")
        verified_hashes = [full._array_sha(descriptors[i:i+1]) for i in range(n)]
    return ids, refs, score, verified_hashes


def build_union(regions, baseline_comparison):
    identity, indices, times = appearance(baseline_comparison)
    require(isinstance(regions, list) and 1 <= len(regions) <= 6 and all(isinstance(r, dict) for r in regions)
        and len({r.get("key") for r in regions}) == len(regions), "all unique requested metadata regions required")
    entries = {}; occurrences = 0
    for region in sorted(regions, key=lambda r: r["key"]):
        ids, refs, score, hashes = validate_region(region, identity, indices, times)
        occurrences += len(ids)
        for col, identifier in enumerate(ids.tolist()):
            entry = {"region": region["key"], "column": col, "coordinate": refs[col].copy(), "scores": score[:, col].copy(),
                     "descriptor_sha256": hashes[col] if hashes else None}
            entries.setdefault(identifier, []).append(entry)
    require(1 <= len(entries) <= 10000 and len(indices)*len(entries) <= 800000, "full union exceeds fixed decoder budget")
    ids = np.asarray(sorted(entries)); refs = []; columns = []; mapping = []
    for identifier in ids.tolist():
        rows = entries[identifier]; canonical = rows[0]
        require(all(np.array_equal(r["coordinate"], canonical["coordinate"]) for r in rows), "duplicate provider ID has conflicting coordinates")
        if len(rows) > 1:
            require(all(r["descriptor_sha256"] is not None and r["descriptor_sha256"] == canonical["descriptor_sha256"] for r in rows),
                    "overlapping ID needs identical protected descriptor evidence")
            all_scores = np.stack([r["scores"].astype(np.float64) for r in rows])
            require(np.max(all_scores.max(axis=0)-all_scores.min(axis=0)) <= 1e-6, "duplicate raw-score envelope conflicts")
        refs.append(canonical["coordinate"]); columns.append(canonical["scores"])
        mapping.append({"ref_id": identifier, "canonical_region": canonical["region"], "canonical_column": canonical["column"],
            "occurrences": [{"region": r["region"], "column": r["column"], "descriptor_sha256": r["descriptor_sha256"]} for r in rows]})
    matrix = np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
    return {"ref_ids": ids, "ref_latlon": np.asarray(refs, np.float64), "raw_similarities": matrix,
        "query_indices": indices, "query_times": times, "mapping": mapping, "appearance_identity": identity,
        "appearance_identity_sha256": json_sha(identity), "reference_occurrences": occurrences,
        "input_sha256": {r["key"]: {k: full._array_sha(np.asarray(v)) for k, v in r["arrays"].items()} for r in regions}}


def prove_baseline_selection(baseline_regions, comparison):
    """Validate every original control pool, then its explicit per-query rows."""
    identity, indices, times = appearance(comparison)
    require(isinstance(baseline_regions, list) and 1 <= len(baseline_regions) <= PROFILE["maximum_baseline_audits"]
        and all(isinstance(r, dict) for r in baseline_regions)
        and len({r.get("key") for r in baseline_regions}) == len(baseline_regions), "all independent original baseline audit pools required")
    validated = [validate_region(r, identity, indices, times) for r in baseline_regions]
    selected = comparison.get("selected_audit_rows")
    require(isinstance(selected, list) and len(selected) == len(indices), "explicit baseline selected_audit_rows required")
    ids = comparison.get("decoded_reference_ids")
    require(isinstance(ids, list) and len(ids) == len(indices), "baseline selected IDs required")
    mapping = []
    for q, pair in enumerate(selected):
        require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int and v >= 0 for v in pair), "invalid baseline selected audit row")
        audit_index, row = pair
        require(audit_index < len(validated) and row < len(indices), "baseline selected audit row outside complete pools")
        region = baseline_regions[audit_index]; ref_ids, refs, scores, _ = validated[audit_index]
        require(int(region["arrays"]["query_indices"][row]) == int(indices[q])
            and float(region["arrays"]["query_times"][row]) == float(times[q]), "baseline selected audit row differs from actual query")
        col = int(region["arrays"]["decoded_ref_indices"][row])
        require(ids[q] == str(ref_ids[col]) and comparison["decoded_reference_latlon"][q] == refs[col].tolist()
            and comparison["decoded_reference_cosines"][q] == float(scores[row, col]),
            "original baseline selected raw evidence differs from its own audit pool")
        mapping.append({"query_row": q, "audit_index": audit_index, "audit_query_row": row,
            "region": region["key"], "region_column": col, "ref_id": str(ref_ids[col]), "raw_cosine": float(scores[row, col])})
    return mapping


def replay_metadata_union(regions, baseline_comparison, baseline_summary, metadata_comparison, metadata_summary, baseline_regions=None):
    baseline, identity, indices, times = legacy_summary(baseline_comparison, baseline_summary)
    metadata, metadata_identity, _, _ = legacy_summary(metadata_comparison, metadata_summary)
    require(identity == metadata_identity, "original baseline/metadata appearance differs")
    previous = legacy_gate(baseline, metadata)
    if previous.get("selected_candidate") == "expanded":
        return {"status": "preserved", "selection": "legacy_metadata", "legacy_gate": previous,
            "outputs": {}, "union_decoded": False, "profile_sha256": PROFILE_SHA256}
    require(baseline["similarity_median"] < policy.policy_config()["baseline_median_min"], "baseline already meets existing keep cutoff")
    baseline_mapping = prove_baseline_selection(baseline_regions, baseline_comparison)
    union = build_union(regions, baseline_comparison)
    by_id = {value: i for i, value in enumerate(union["ref_ids"].tolist())}
    original_mappings = {"baseline": baseline_mapping}
    for label, comparison in (("metadata", metadata_comparison),):
        # A legacy winner belongs to its explicitly recorded regional matrix.
        # Duplicate IDs use one deterministic column in the new union, but
        # otherwise identical descriptor rows can have pool-dependent float32
        # GEMM rounding within the already-validated duplicate score envelope.
        # Prove the original scalar EXACTLY against its source occurrence;
        # never replace it with, or demand bit equality to, another occurrence.
        original_rows = prove_baseline_selection(regions, comparison)
        mapped = []
        for q, source in enumerate(original_rows):
            identifier = source["ref_id"]
            require(identifier in by_id, "original selected reference missing from complete metadata union")
            col = by_id[identifier]
            entry = union["mapping"][col]
            require(union["ref_latlon"][col].tolist() == comparison["decoded_reference_latlon"][q],
                    "original selected coordinate differs from full evidence")
            require(any(r["region"] == source["region"] and r["column"] == source["region_column"]
                        for r in entry["occurrences"]), "original selected occurrence missing from complete metadata union")
            mapped.append({**source, "global_column": col, "canonical_union_source": {
                "region": entry["canonical_region"], "region_column": entry["canonical_column"],
                "raw_cosine": float(union["raw_similarities"][q, col])}})
        original_mappings[label] = mapped
    result = full.decode_full_tracks(union["raw_similarities"], union["ref_ids"], union["ref_latlon"], times)
    require(result["profile_sha256"] == full.PROFILE_SHA256 and set(result["outputs"]) == {"raw", "reference_mean"}, "both unchanged full-DP outputs required")
    comparisons = {}
    for variant, output in result["outputs"].items():
        chosen = np.asarray(output["chosen_ref_indices"], np.int64)
        require(chosen.shape == indices.shape and chosen.min() >= 0 and chosen.max() < len(union["ref_ids"]), "decoded indices outside full union")
        scores = union["raw_similarities"][np.arange(len(indices)), chosen].astype(np.float64)
        coords = union["ref_latlon"][chosen]
        require(output["chosen_ref_ids"] == union["ref_ids"][chosen].tolist() and output["raw_route_latlon"] == coords.tolist()
            and output["selected_raw_cosines"] == scores.tolist(), "new output differs from mapped raw reference evidence")
        comparisons[variant] = {"schema": PROFILE["comparison_schema"], "appearance_identity": union["appearance_identity"],
            "appearance_identity_sha256": union["appearance_identity_sha256"], "actual_decoder_profile_sha256": full.PROFILE_SHA256,
            "variant": variant, "raw_cosines": scores.tolist(), "unsmoothed_reference_latlon": coords.tolist(),
            "similarity_median": float(np.median(scores)), "dominant_coordinate_fraction": max(Counter(map(tuple, coords)).values())/len(coords),
            "selected_source_rows": [{"query_row": q, "query_index": int(indices[q]), "query_time": float(times[q]),
                "global_column": int(col), "ref_id": union["mapping"][col]["ref_id"],
                "region": union["mapping"][col]["canonical_region"], "region_column": union["mapping"][col]["canonical_column"],
                "raw_cosine": float(scores[q])} for q, col in enumerate(chosen)]}
    proof = comparisons["reference_mean"]
    new_baseline = {**baseline, "score_identity": union["appearance_identity_sha256"]}
    candidate = {"status": "complete", "route_valid": True, "point_count": len(indices), "score_kind": "raw_cosine",
        "score_identity": union["appearance_identity_sha256"], "similarity_median": proof["similarity_median"],
        "dominant_coordinate_fraction": proof["dominant_coordinate_fraction"]}
    decision = legacy_gate(new_baseline, candidate)
    return {"status": "complete", "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256,
        "full_sequence_profile_sha256": full.PROFILE_SHA256, "union": union, "outputs": result["outputs"],
        "full_decoder_evidence": {k: v for k, v in result.items() if k != "outputs"}, "comparisons": comparisons,
        "admission": {"baseline_original": baseline_summary, "baseline_new_comparison": new_baseline,
            "metadata_original": metadata_summary, "legacy_gate": previous, "sole_primary_summary": candidate,
            "sole_primary": "reference_mean", "decision": decision, "decoder_identity_not_forged": True,
            "original_selected_sources": original_mappings},
        "selection": "metadata_union_reference_mean" if decision.get("selected_candidate") == "expanded" else "baseline_or_existing_municipality_fallback",
        "confidence": None, "default_adopted": False, "union_decoded": True}
