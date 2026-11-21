"""Pure provisional input-only policy; POST-HOC DEV, not a validated default.

No routes, models, IO, evaluation or per-call threshold overrides occur here.
Callers validate route artifacts and supply scalar summaries. Stage status is
not_run/complete/partial/failed/unavailable; unreached stages are ignored.
The completed cues bundle contains independently status-tagged vo/ocr/plane
measurements. Missing measured plane quality declines PF eligibility.
Raw-cosine identities must encode backbone, weights and query preprocessing;
expanded evidence must match the declared baseline identity exactly.
"""
from __future__ import annotations
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json
import math
from numbers import Integral, Real

POLICY_VERSION = "assisted-policy-posthoc-dev-v1"
_CONFIG = {
    "policy_version": POLICY_VERSION, "validation_status": "posthoc_dev_unvalidated",
    "baseline_median_min": .162, "joint_vo_valid_min": .80, "image_only_branch_count": 1,
    "image_only_profile": {"radius_m": 3000, "cap": 1500, "query_mode": "crops",
                           "score_normalization": "reference_mean"},
    "pf_duration_min_s": 120., "pf_vo_valid_min": .95,
    "pf_plane_coverage_min": .50, "pf_plane_rel_spread_max": .15,
    "expanded_median_improvement_min": .02, "expanded_dominant_coordinate_fraction_max": .25,
    "comparison_score_kind": "raw_cosine",
}
CONFIG_SHA256 = hashlib.sha256(json.dumps(_CONFIG, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def policy_config():
    """Detached configuration copy; thresholds have no per-call override API."""
    return json.loads(json.dumps(_CONFIG))


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _status(stage):
    value = _mapping(stage).get("status", "not_run")
    return value if isinstance(value, str) and value in {
        "not_run", "complete", "partial", "failed", "unavailable"} else "invalid"


def _number(value, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    return None if (low is not None and value < low) or (high is not None and value > high) else value


def _count(value):
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0 else None


def _route(stage):
    stage = _mapping(stage)
    count = _count(stage.get("point_count"))
    return _status(stage) == "complete" and stage.get("route_valid") is True and count is not None and count >= 2


def _score(stage):
    stage = _mapping(stage)
    identity = stage.get("score_identity")
    valid = (stage.get("score_kind") == _CONFIG["comparison_score_kind"]
             and isinstance(identity, str) and bool(identity.strip()))
    return (_number(stage.get("similarity_median"), low=-1, high=1), identity) if valid else (None, None)


def _candidate_summary(stage):
    stage = _mapping(stage)
    if _status(stage) != "complete":
        return {"status": _status(stage), "route_valid": False, "point_count": None}
    return {"status": _status(stage), "route_valid": _route(stage),
            "point_count": _count(stage.get("point_count"))}


def _emit(action, selected, reason, summary, request_config=None):
    result = {"experimental": True, "validation_status": "posthoc_dev_unvalidated",
              "policy_version": POLICY_VERSION, "config_sha256": CONFIG_SHA256,
              "next_action": action, "selected_candidate": selected,
              "selection_reason": reason, "evidence_summary": summary}
    if request_config is not None:
        result["request_config"] = request_config
    return result


def _joint_complete(candidate, branches):
    candidate = _mapping(candidate)
    return (_route(candidate) and _count(candidate.get("retained_branch_count")) == branches
            and _count(candidate.get("successful_branch_count")) == branches)


def _image_complete(candidate):
    candidate = _mapping(candidate)
    profile = _CONFIG["image_only_profile"]
    return (_joint_complete(candidate, 1) and candidate.get("uses_rigid_vo") is False
            and _number(candidate.get("radius_m"), low=0) == profile["radius_m"]
            and _count(candidate.get("cap")) == profile["cap"]
            and candidate.get("query_mode") == profile["query_mode"]
            and candidate.get("score_normalization") == profile["score_normalization"])


def decide_assisted_policy(evidence):
    """Return a request or candidate NAME, never a route.

    Route summaries: status, route_valid, point_count. Baseline/expanded add
    similarity_median, score_kind='raw_cosine', score_identity (registered
    backbone/weights/query-preprocessing identity). Cues contain status,
    duration_s, vo={status,valid_fraction}, ocr={status,clue_branch_count},
    plane={status,measured,coverage,rel_spread}. POI candidates attest
    retained_branch_count/successful_branch_count; image-only also attests its
    exact profile and uses_rigid_vo=False. PF attests metric_only=True,
    uses_user_spatial_prior=False, position_observation_count=0. Expanded adds
    dominant_coordinate_fraction. The caller validates finite usable routes.
    """
    evidence = _mapping(evidence)
    baseline = _mapping(evidence.get("baseline"))
    baseline_valid = _route(baseline)
    median, identity = _score(baseline) if baseline_valid else (None, None)
    summary = {"baseline": {**_candidate_summary(baseline), "similarity_median": median,
                             "score_identity": identity}}
    if median is not None and median >= _CONFIG["baseline_median_min"]:
        return _emit("keep_baseline", "baseline", "baseline_median_meets_fixed_cutoff", summary)
    cues = _mapping(evidence.get("cues"))
    cue_status = _status(cues)
    summary["cues"] = {"status": cue_status}
    if cue_status == "not_run":
        return _emit("request_ocr_vo", None, "baseline_needs_assistance", summary)
    vo, ocr = (_mapping(cues.get(name)) if cue_status == "complete" else {} for name in ("vo", "ocr"))
    valid = _number(vo.get("valid_fraction"), low=0, high=1) if _status(vo) == "complete" else None
    branches = _count(ocr.get("clue_branch_count")) if _status(ocr) == "complete" else None
    duration = _number(cues.get("duration_s"), low=0) if cue_status == "complete" else None
    summary["cues"].update(duration_s=duration, vo_status=_status(vo), vo_valid_fraction=valid,
                            ocr_status=_status(ocr), clue_branch_count=branches)
    return _poi_or_later(evidence, summary, cues, cue_status, baseline_valid, median, identity,
                         duration, valid, branches)


def _poi_or_later(evidence, summary, cues, cue_status, baseline_valid, median, identity,
                  duration, valid, branches):
    if branches is not None and branches > 0 and valid is not None:
        if valid >= _CONFIG["joint_vo_valid_min"]:
            candidate = _mapping(evidence.get("poi_joint"))
            summary["poi_joint"] = _candidate_summary(candidate)
            if _status(candidate) == "not_run":
                return _emit("request_poi_joint", None, "literal_branches_and_vo_admit_joint_attempt", summary)
            summary["poi_joint"].update(
                retained_branch_count=_count(candidate.get("retained_branch_count")),
                successful_branch_count=_count(candidate.get("successful_branch_count")))
            if _joint_complete(candidate, branches):
                return _emit("select_candidate", "poi_joint", "complete_joint_candidate_with_sufficient_vo", summary)
            summary["poi_joint"]["decline_reason"] = "incomplete_invalid_or_missing_literal_branches"
        elif branches == _CONFIG["image_only_branch_count"]:
            candidate = _mapping(evidence.get("poi_image_track"))
            summary["poi_image_track"] = _candidate_summary(candidate)
            if _status(candidate) == "not_run":
                request = {**_CONFIG["image_only_profile"], "uses_rigid_vo": False}
                return _emit("request_poi_image_track", None, "single_branch_with_weak_vo", summary, request)
            if _image_complete(candidate):
                return _emit("select_candidate", "poi_image_track", "complete_fixed_image_only_candidate", summary)
            summary["poi_image_track"]["decline_reason"] = "incomplete_invalid_or_mismatched_image_only_profile"
    return _after_poi(evidence, summary, cues if cue_status == "complete" else {},
                      baseline_valid, median, identity, duration, valid)


def _after_poi(evidence, summary, cues, baseline_valid, median, identity, duration, valid):
    eligible = (duration is not None and duration >= _CONFIG["pf_duration_min_s"]
                and valid is not None and valid >= _CONFIG["pf_vo_valid_min"])
    summary["metric_pf_eligibility"] = {"duration_and_vo_qualify": eligible, "eligible": False}
    if eligible:
        plane = _mapping(cues.get("plane"))
        measured = _status(plane) == "complete" and plane.get("measured") is True
        coverage = _number(plane.get("coverage"), low=0, high=1) if measured else None
        spread = _number(plane.get("rel_spread"), low=0) if measured else None
        eligible = (coverage is not None and coverage >= _CONFIG["pf_plane_coverage_min"]
                    and spread is not None and spread <= _CONFIG["pf_plane_rel_spread_max"])
        summary["metric_pf_eligibility"].update(measured_plane=measured,
            plane_coverage=coverage, plane_rel_spread=spread, eligible=eligible)
        if eligible:
            candidate = _mapping(evidence.get("metric_pf"))
            summary["metric_pf"] = _candidate_summary(candidate)
            if _status(candidate) == "not_run":
                return _emit("request_metric_pf", None, "measured_video_motion_admits_metric_pf_attempt", summary,
                             {"metric_only": True, "uses_user_spatial_prior": False, "position_observation_count": 0})
            summary["metric_pf"].update(
                metric_only=candidate.get("metric_only") if isinstance(candidate.get("metric_only"), bool) else None,
                uses_user_spatial_prior=(candidate.get("uses_user_spatial_prior")
                    if isinstance(candidate.get("uses_user_spatial_prior"), bool) else None),
                position_observation_count=_count(candidate.get("position_observation_count")))
            if (_route(candidate) and candidate.get("metric_only") is True
                    and candidate.get("uses_user_spatial_prior") is False
                    and _count(candidate.get("position_observation_count")) == 0):
                return _emit("select_candidate", "metric_pf", "complete_metric_only_pf_without_spatial_observations", summary)
            summary["metric_pf"]["decline_reason"] = "incomplete_invalid_or_nonmetric_spatially_assisted_pf"
    return _expanded(evidence, summary, baseline_valid, median, identity)


def _expanded(evidence, summary, baseline_valid, median, identity):
    candidate = _mapping(evidence.get("expanded"))
    summary["expanded"] = _candidate_summary(candidate)
    if _status(candidate) == "not_run":
        return _emit("request_expanded_retrieval", None, "no_admissible_poi_or_metric_pf", summary,
                     {"coverage": "metadata_municipality_expansion"})
    score, candidate_identity = _score(candidate)
    dominant = _number(candidate.get("dominant_coordinate_fraction"), low=0, high=1)
    # Decimal string arithmetic preserves inclusive decimal threshold boundaries
    # without introducing a new data-dependent tolerance.
    threshold = (max(Decimal(str(_CONFIG["baseline_median_min"])),
                     Decimal(str(median)) + Decimal(str(_CONFIG["expanded_median_improvement_min"])))
                 if median is not None and identity is not None else None)
    summary["expanded"].update(similarity_median=score, score_identity=candidate_identity,
        identity_matches=identity is not None and candidate_identity == identity,
        required_median=float(threshold) if threshold is not None else None,
        dominant_coordinate_fraction=dominant)
    if (_route(candidate) and threshold is not None and score is not None
            and candidate_identity == identity and Decimal(str(score)) >= threshold
            and dominant is not None and dominant <= _CONFIG["expanded_dominant_coordinate_fraction_max"]):
        return _emit("select_candidate", "expanded", "expanded_track_improves_comparable_score_without_coordinate_collapse", summary)
    summary["expanded"]["decline_reason"] = "incomplete_invalid_incomparable_or_insufficient_expanded_evidence"
    return _emit("keep_baseline" if baseline_valid else "no_selection",
                 "baseline" if baseline_valid else None,
                 "no_admissible_assistance" if baseline_valid else "no_valid_baseline_or_admissible_assistance", summary)
