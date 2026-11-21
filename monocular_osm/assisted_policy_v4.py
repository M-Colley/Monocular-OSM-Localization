"""Opt-in DEV v4: v3 unchanged, plus sign-site verification for weak motion.

When the cues stage reads one to six literal shop-sign branches but native VO
is too weak for the joint POI attempt (valid fraction below the v1
``joint_vo_valid_min``), v4 requests one ``poi_site`` child
(:mod:`monocular_osm.poi_site_verify`) before anything else, and selects its
complete route. In every other case, and when that child is incomplete, the
decision is exactly v3's. A posthoc development rule, not a calibrated one.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from . import assisted_policy as base
from . import assisted_policy_v3 as v3
from .poi_site_verify import PROFILE as SITE_PROFILE, PROFILE_SHA256 as SITE_SHA256

POLICY_VERSION = "assisted-policy-posthoc-dev-v4"
_CONFIG = v3.policy_config()
_CONFIG.update(policy_version=POLICY_VERSION, base_policy_config_sha256=v3.CONFIG_SHA256,
    site_profile=dict(SITE_PROFILE), site_profile_sha256=SITE_SHA256,
    site_admission="cues complete, weak baseline, native VO below joint_vo_valid_min, 1-6 literal branches",
    incomplete_site="decline the site candidate, then the unchanged v3 decision")
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
        and base._count(candidate.get("point_count")) == SITE_PROFILE["queries"]
        and candidate.get("uses_rigid_vo") is False
        and candidate.get("profile_sha256") == SITE_SHA256
        and candidate.get("candidate_pruning") is False
        and candidate.get("score_normalization") == SITE_PROFILE["score_normalization"]
        and candidate.get("query_mode") == SITE_PROFILE["query_mode"]
        and base._number(candidate.get("radius_m"), low=0) == SITE_PROFILE["route_radius_m"]
        and base._count(candidate.get("cap")) == SITE_PROFILE["route_cap"])


def decide_assisted_policy(evidence):
    evidence = base._mapping(evidence)
    baseline = base._mapping(evidence.get("baseline"))
    median, _ = base._score(baseline) if base._route(baseline) else (None, None)
    cues = base._mapping(evidence.get("cues"))
    if (median is not None and median >= base.policy_config()["baseline_median_min"]
            or base._status(cues) != "complete"):
        return _emit(v3.decide_assisted_policy(evidence))
    vo, ocr = (base._mapping(cues.get(name)) for name in ("vo", "ocr"))
    valid = base._number(vo.get("valid_fraction"), low=0, high=1) if base._status(vo) == "complete" else None
    branches = base._count(ocr.get("clue_branch_count")) if base._status(ocr) == "complete" else None
    eligible = (valid is not None and valid < base.policy_config()["joint_vo_valid_min"]
        and branches is not None and SITE_PROFILE["min_sites"] <= branches <= SITE_PROFILE["max_sites"])
    if not eligible:
        return _emit(v3.decide_assisted_policy(evidence))
    # The single-branch image track is superseded here; its absence must not
    # stall the v3 fallback.
    fallback = _emit(v3.decide_assisted_policy({**evidence, "poi_image_track": {"status": "unavailable"}}))
    candidate = base._mapping(evidence.get("poi_site"))
    brief = base._candidate_summary(candidate)
    brief.update(retained_branch_count=base._count(candidate.get("retained_branch_count")),
        successful_branch_count=base._count(candidate.get("successful_branch_count")),
        profile_sha256=candidate.get("profile_sha256"))
    early = {key: deepcopy(fallback["evidence_summary"][key]) for key in ("baseline", "cues")}
    early["poi_site"] = brief
    if base._status(candidate) == "not_run":
        return _emit({"experimental": True, "validation_status": "posthoc_dev_unvalidated",
            "next_action": "request_poi_site", "selected_candidate": None,
            "selection_reason": "literal_branches_with_weak_vo_admit_sign_site_verification",
            "evidence_summary": early, "request_config": {
                "profile_sha256": SITE_SHA256, "site_radius_m": SITE_PROFILE["site_radius_m"],
                "site_cap": SITE_PROFILE["site_cap"], "route_radius_m": SITE_PROFILE["route_radius_m"],
                "route_cap": SITE_PROFILE["route_cap"], "branches": branches, "query_mode": "warp",
                "queries": SITE_PROFILE["queries"], "score_normalization": "reference_mean",
                "uses_rigid_vo": False}})
    if _complete(candidate, branches):
        return _emit({"experimental": True, "validation_status": "posthoc_dev_unvalidated",
            "next_action": "select_candidate", "selected_candidate": "poi_site",
            "selection_reason": "complete_sign_site_verified_route", "evidence_summary": early})
    brief["decline_reason"] = "incomplete_invalid_or_mismatched_sign_site_candidate"
    fallback["evidence_summary"] = {**early, **{key: value for key, value in fallback["evidence_summary"].items()
        if key not in {"baseline", "cues", "poi_image_track"}}}
    return fallback
