"""Fixed experimental DEV policy with a single all-branch image union.

Baseline/joint/PF/expanded admission follows v1 unchanged. Weak VO admits
one common-query image decode across every literal branch (one to six).
This is a new, unvalidated policy version, never a retrospective clip oracle.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from . import assisted_policy as base
from .poi_image_union import PROFILE as UNION_PROFILE, PROFILE_SHA256 as UNION_SHA256

POLICY_VERSION = "assisted-policy-posthoc-dev-v2"
_CONFIG = base.policy_config()
_CONFIG.pop("image_only_branch_count")
_CONFIG.pop("image_only_profile")
_CONFIG.update(policy_version=POLICY_VERSION, base_policy_config_sha256=base.CONFIG_SHA256,
    image_union_profile=dict(UNION_PROFILE), image_union_profile_sha256=UNION_SHA256,
    incomplete_union="decline entire union, then ordinary expanded fallback; no branch winner")
CONFIG_SHA256 = hashlib.sha256(json.dumps(_CONFIG, sort_keys=True, separators=(",", ":"),
    allow_nan=False).encode()).hexdigest()


def policy_config():
    return deepcopy(_CONFIG)


def _emit(value):
    result = deepcopy(value)
    result.update(policy_version=POLICY_VERSION, config_sha256=CONFIG_SHA256)
    return result


def _complete(candidate, branches):
    return (base._joint_complete(candidate, branches)
        and base._count(candidate.get("point_count")) == UNION_PROFILE["queries"]
        and candidate.get("uses_rigid_vo") is False
        and candidate.get("profile_sha256") == UNION_SHA256
        and candidate.get("candidate_pruning") is False
        and candidate.get("score_normalization") == "reference_mean"
        and candidate.get("query_mode") == "crops"
        and base._number(candidate.get("radius_m"), low=0) == UNION_PROFILE["radius_m"]
        and base._count(candidate.get("cap")) == UNION_PROFILE["cap_per_branch"])


def decide_assisted_policy(evidence):
    evidence = base._mapping(evidence)
    baseline = base._mapping(evidence.get("baseline"))
    median, _ = base._score(baseline) if base._route(baseline) else (None, None)
    cues = base._mapping(evidence.get("cues"))
    # Strong baseline and incomplete/unreached cues retain their original
    # precedence. Later candidate fields cannot influence either decision.
    if (median is not None and median >= _CONFIG["baseline_median_min"]
            or base._status(cues) != "complete"):
        return _emit(base.decide_assisted_policy(evidence))
    vo, ocr = (base._mapping(cues.get(name)) for name in ("vo", "ocr"))
    valid = base._number(vo.get("valid_fraction"), low=0, high=1) if base._status(vo) == "complete" else None
    branches = base._count(ocr.get("clue_branch_count")) if base._status(ocr) == "complete" else None
    # A legacy single-branch result can never substitute for this union.
    fallback_inputs = {**evidence, "poi_image_track": {"status": "unavailable"}}
    fallback = _emit(base.decide_assisted_policy(fallback_inputs))
    eligible = (valid is not None and valid < _CONFIG["joint_vo_valid_min"]
        and branches is not None and UNION_PROFILE["min_branches"] <= branches <= UNION_PROFILE["max_branches"])
    if not eligible:
        return fallback
    candidate = base._mapping(evidence.get("poi_image_union"))
    brief = base._candidate_summary(candidate)
    brief.update(retained_branch_count=base._count(candidate.get("retained_branch_count")),
        successful_branch_count=base._count(candidate.get("successful_branch_count")),
        profile_sha256=candidate.get("profile_sha256"))
    early = {key: deepcopy(fallback["evidence_summary"][key]) for key in ("baseline", "cues")}
    early["poi_image_union"] = brief
    if base._status(candidate) == "not_run":
        return _emit({"experimental": True, "validation_status": "posthoc_dev_unvalidated",
            "next_action": "request_poi_image_union", "selected_candidate": None,
            "selection_reason": "all_literal_branches_with_weak_vo_admit_common_query_image_union",
            "evidence_summary": early, "request_config": {
                "profile_sha256": UNION_SHA256, "radius_m": UNION_PROFILE["radius_m"],
                "cap_per_branch": UNION_PROFILE["cap_per_branch"], "branches": branches,
                "query_mode": "crops", "queries": 80, "score_normalization": "reference_mean",
                "uses_rigid_vo": False}})
    if _complete(candidate, branches):
        return _emit({"experimental": True, "validation_status": "posthoc_dev_unvalidated",
            "next_action": "select_candidate", "selected_candidate": "poi_image_union",
            "selection_reason": "complete_fixed_all_branch_image_union", "evidence_summary": early})
    brief["decline_reason"] = "incomplete_invalid_or_mismatched_all_branch_image_union"
    fallback["evidence_summary"] = {**early, **{key: value for key, value in fallback["evidence_summary"].items()
        if key not in {"baseline", "cues", "poi_image_track"}}}
    return fallback
