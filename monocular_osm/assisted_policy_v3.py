"""Opt-in DEV v3: original v1 precedence plus one full metadata candidate.

No routes/IO/models occur here. Original legacy identities remain authoritative
for every old admission. Only the new reference-mean union uses a separately
validated raw-appearance projection of the unchanged baseline evidence.
"""
from __future__ import annotations
from copy import deepcopy
from . import assisted_policy as base, metadata_union as union, assisted_comparison as comparison

POLICY_VERSION = "assisted-policy-posthoc-dev-v3"
_CONFIG = base.policy_config()
_CONFIG.update(policy_version=POLICY_VERSION, base_policy_config_sha256=base.CONFIG_SHA256,
    comparison_profile_sha256=comparison.PROFILE_SHA256, metadata_union_profile_sha256=union.PROFILE_SHA256,
    metadata_union_primary="reference_mean", metadata_union_legacy_priority="exact original v1 identity/decoder gates first",
    metadata_union_incomplete="decline whole candidate then original municipality fallback",
    metadata_union_max_pools=6, metadata_union_queries=80, metadata_union_max_references=10000)
CONFIG_SHA256 = union.json_sha(_CONFIG)


def policy_config():
    return deepcopy(_CONFIG)


def _emit(value):
    result = deepcopy(value)
    result.update(policy_version=POLICY_VERSION, config_sha256=CONFIG_SHA256)
    return result


def _decline(evidence, earlier, reason):
    baseline = base._mapping(evidence.get("baseline"))
    valid = base._route(baseline); median, identity = base._score(baseline) if valid else (None, None)
    summary = deepcopy(earlier["evidence_summary"])
    summary["metadata_union"] = {"admitted": False, "decline_reason": reason}
    return _emit(base._expanded({"expanded": {"status": "unavailable"}}, summary, valid, median, identity))


def decide_assisted_policy(evidence):
    evidence = base._mapping(evidence)
    candidate = base._mapping(evidence.get("expanded"))
    kind = candidate.get("candidate_kind")
    union_marker = (candidate.get("comparison_schema") == comparison.SCHEMA
        or "comparison_evidence" in candidate or "metadata_union_profile_sha256" in candidate)
    if kind is None and not union_marker:
        # This includes every original legacy expansion. Excluding the decoder
        # from the new identity must NEVER admit an old incomparable track.
        return _emit(base.decide_assisted_policy(evidence))
    earlier = base.decide_assisted_policy({**evidence, "expanded": {"status": "not_run"}})
    if earlier["next_action"] != "request_expanded_retrieval":
        return _emit(earlier)  # Strong/earlier stages ignore all future fields.
    if kind != "metadata_union_reference_mean":
        return _decline(evidence, earlier, "unsupported_union_kind_or_raw_diagnostic")
    try:
        baseline = base._mapping(evidence.get("baseline"))
        projection = comparison.validated_projection(baseline.get("metadata_union_baseline"), baseline)
        original, identity, _, _ = union.legacy_summary(projection["original_comparison"], projection["original_summary"])
        legacy = candidate["legacy_metadata"]
        old_candidate, old_identity, _, _ = union.legacy_summary(legacy["comparison"], legacy["summary"])
        union.require(identity == old_identity, "legacy and baseline actual raw appearance differs")
        legacy_gate = union.legacy_gate(original, old_candidate)
        union.require(legacy_gate["selected_candidate"] != "expanded", "already_admitted_legacy_must_precede_union")
        proof = candidate["comparison_evidence"]
        derived = comparison.union_summary(proof,proof["route_latlon"],proof["route_timestamps"])
        union.require(all(candidate.get(k) == v for k,v in derived.items()), "union summary differs from full selected raw-column evidence")
        count = candidate["requested_pool_count"]; complete = candidate["complete_pool_count"]
        decoder = {"family": "full_sequence", "profile_sha256": union.full.PROFILE_SHA256,
            "variant": "reference_mean", "candidate_pruning": False}
        union.require(base._route(candidate) and candidate.get("point_count") == 80
            and candidate.get("comparison_schema") == comparison.SCHEMA and candidate.get("actual_decoder") == decoder
            and candidate.get("metadata_union_profile_sha256") == union.PROFILE_SHA256
            and type(count) is int and 1 <= count <= 6 and type(complete) is int and complete == count
            and type(candidate.get("full_reference_count")) is int and 1 <= candidate["full_reference_count"] <= 10000,
            "incomplete_or_mismatched_full_union")
        identity_hash = union.json_sha(identity)
        union.require(candidate.get("baseline_projection_sha256") == union.json_sha(projection)
            and candidate.get("score_kind") == "raw_cosine" and candidate.get("score_identity") == identity_hash,
            "new_raw_comparison_is_not_actual_baseline_projection")
        # This local copy is deliberately separate from original fields and
        # original v1 gate evidence retained above and in the decision trace.
        comparable_baseline = {**original, "score_identity": identity_hash}
        summary = deepcopy(earlier["evidence_summary"])
        summary["metadata_union"] = {"sole_primary": "reference_mean", "actual_decoder": decoder,
            "original_baseline_score_identity": original["score_identity"], "new_raw_appearance_identity": identity_hash,
            "legacy_gate": legacy_gate, "requested_pool_count": count, "full_reference_count": candidate["full_reference_count"]}
        decision = union.legacy_gate(comparable_baseline, candidate)
        decision["evidence_summary"] = {**summary, **decision["evidence_summary"]}
        return _emit(decision)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return _decline(evidence, earlier, "missing_invalid_or_conflicting_union_proof")
