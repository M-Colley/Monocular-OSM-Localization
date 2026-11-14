"""Synthetic policy branches only: no candidate acquisition or label scoring."""
from copy import deepcopy
import json

import pytest

from monocular_osm.assisted_policy import CONFIG_SHA256, decide_assisted_policy, policy_config


def baseline(median=.15, **changes):
    return {"status": "complete", "route_valid": True, "point_count": 80,
            "similarity_median": median, "score_kind": "raw_cosine",
            "score_identity": "fixed-backbone-weights-warp-v1", **changes}


def cues(valid=.9, branches=2, duration=180.):
    return {"status": "complete", "duration_s": duration,
            "vo": {"status": "complete", "valid_fraction": valid},
            "ocr": {"status": "complete", "clue_branch_count": branches},
            "plane": {"status": "complete", "measured": True, "coverage": .7, "rel_spread": .1}}


def joint(branches=2, **changes):
    return {"status": "complete", "route_valid": True, "point_count": 149,
            "retained_branch_count": branches, "successful_branch_count": branches, **changes}


def image_track(**changes):
    return {**joint(1), "uses_rigid_vo": False, "radius_m": 3000, "cap": 1500,
            "query_mode": "crops", "score_normalization": "reference_mean", **changes}


def pf(**changes):
    return {"status": "complete", "route_valid": True, "point_count": 100,
            "metric_only": True, "uses_user_spatial_prior": False,
            "position_observation_count": 0, "confidence": 0., **changes}


def expanded(median=.17, **changes):
    return {**baseline(median), "dominant_coordinate_fraction": .1, **changes}


def test_high_baseline_keeps_inclusive_cutoff_and_ignores_all_future_stages():
    initial = {"baseline": baseline(.162)}
    expected = decide_assisted_policy(initial)
    assert expected["next_action"] == "keep_baseline" and expected["selected_candidate"] == "baseline"
    assert expected == decide_assisted_policy({**initial, "cues": cues(), "poi_joint": joint(),
        "poi_image_track": image_track(), "metric_pf": pf(confidence=1.), "expanded": expanded(.99)})
    assert set(expected["evidence_summary"]) == {"baseline"}


@pytest.mark.parametrize("change", [{"similarity_median": .161999}, {"similarity_median": None},
    {"similarity_median": float("nan")}, {"similarity_median": float("inf")},
    {"similarity_median": True}, {"similarity_median": 2.}, {"score_kind": "reference_mean"},
    {"score_identity": ""}, {"status": "partial"}, {"route_valid": False}, {"point_count": 1}])
def test_missing_bad_or_partial_baseline_requests_cues_without_future_leak(change):
    value = {"baseline": baseline(.9, **change)}
    expected = decide_assisted_policy(value)
    assert expected["next_action"] == "request_ocr_vo"
    assert expected == decide_assisted_policy({**value, "poi_joint": joint(), "metric_pf": pf(), "expanded": expanded(.99)})


def test_joint_requires_cues_first_then_complete_all_branch_output_at_point80_vo_boundary():
    value = {"baseline": baseline(), "cues": cues(.8)}
    expected = decide_assisted_policy(value)
    assert expected["next_action"] == "request_poi_joint"
    assert expected == decide_assisted_policy({**value, "metric_pf": pf(), "expanded": expanded(.99)})
    decision = decide_assisted_policy({**value, "poi_joint": joint()})
    assert decision["selected_candidate"] == "poi_joint"
    assert "metric_pf" not in decision["evidence_summary"]


@pytest.mark.parametrize("change", [{"status": "partial"}, {"status": "failed"},
    {"route_valid": False}, {"point_count": None}, {"retained_branch_count": 1},
    {"successful_branch_count": 1}])
def test_incomplete_joint_candidate_is_declined(change):
    decision = decide_assisted_policy({"baseline": baseline(), "cues": cues(.8), "poi_joint": joint(**change)})
    assert decision["next_action"] == "request_expanded_retrieval"


def test_weak_vo_single_branch_requests_exact_image_profile_without_rigid_vo():
    value = {"baseline": baseline(), "cues": cues(.799999, 1)}
    decision = decide_assisted_policy({**value, "poi_joint": joint(1)})
    assert decision["next_action"] == "request_poi_image_track"
    assert decision["request_config"] == {"radius_m": 3000, "cap": 1500, "query_mode": "crops",
                                           "score_normalization": "reference_mean", "uses_rigid_vo": False}
    assert decide_assisted_policy({**value, "poi_image_track": image_track()})["selected_candidate"] == "poi_image_track"


@pytest.mark.parametrize("valid,branches", [(.7, 0), (.7, 2), (None, 1), (float("nan"), 1), (.7, True)])
def test_weak_or_unknown_vo_does_not_promote_joint_or_ambiguous_image_track(valid, branches):
    decision = decide_assisted_policy({"baseline": baseline(), "cues": cues(valid, branches),
                                       "poi_joint": joint(), "poi_image_track": image_track()})
    assert decision["next_action"] == "request_expanded_retrieval"


@pytest.mark.parametrize("change", [{"radius_m": 1000}, {"cap": 500}, {"query_mode": "warp"},
    {"score_normalization": "raw"}, {"uses_rigid_vo": True}, {"uses_rigid_vo": None},
    {"retained_branch_count": 2}, {"successful_branch_count": 0}, {"status": "partial"}])
def test_image_candidate_must_be_complete_and_attest_exact_fixed_profile(change):
    decision = decide_assisted_policy({"baseline": baseline(), "cues": cues(.2, 1),
                                       "poi_image_track": image_track(**change)})
    assert decision["next_action"] == "request_expanded_retrieval"


def test_metric_pf_requires_measured_quality_and_complete_metric_stage_not_confidence():
    measured = cues(.95, 0, 120.)
    measured["plane"].update(coverage=.50, rel_spread=.15)
    value = {"baseline": baseline(), "cues": measured}
    request = decide_assisted_policy(value)
    assert request["next_action"] == "request_metric_pf"
    assert request["request_config"] == {"metric_only": True, "uses_user_spatial_prior": False,
                                           "position_observation_count": 0}
    assert request == decide_assisted_policy({**value, "expanded": expanded(.99)})
    assert decide_assisted_policy({**value, "metric_pf": pf(confidence=0.)})["selected_candidate"] == "metric_pf"


@pytest.mark.parametrize("field,value", [("duration_s", 119.999), ("valid_fraction", .949999),
    ("coverage", .499999), ("rel_spread", .150001), ("rel_spread", float("inf")),
    ("rel_spread", float("nan")), ("coverage", None), ("measured", False), ("status", "partial")])
def test_pf_quality_failures_or_missing_measurements_decline_before_future_pf(field, value):
    measurement = cues(.95, 0, 120.)
    if field == "duration_s": measurement[field] = value
    elif field == "valid_fraction": measurement["vo"][field] = value
    else: measurement["plane"][field] = value
    decision = decide_assisted_policy({"baseline": baseline(), "cues": measurement,
                                       "metric_pf": pf(confidence=1.)})
    assert decision["next_action"] == "request_expanded_retrieval"
    assert "metric_pf" not in decision["evidence_summary"]


def test_no_plane_measurement_means_no_pf_request_and_bad_cue_bundle_is_not_trusted():
    measurement = cues(.99, 0)
    measurement.pop("plane")
    value = {"baseline": baseline(), "cues": measurement, "metric_pf": pf()}
    assert decide_assisted_policy(value)["next_action"] == "request_expanded_retrieval"
    measurement["status"] = "partial"
    assert decide_assisted_policy(value)["next_action"] == "request_expanded_retrieval"


@pytest.mark.parametrize("change", [{"metric_only": False}, {"metric_only": None},
    {"uses_user_spatial_prior": True}, {"uses_user_spatial_prior": None},
    {"position_observation_count": 1}, {"position_observation_count": None},
    {"status": "partial"}, {"route_valid": False}, {"point_count": 1}])
def test_pf_with_priors_observations_partial_or_bad_route_is_not_admitted(change):
    decision = decide_assisted_policy({"baseline": baseline(), "cues": cues(.99, 0),
                                       "metric_pf": pf(confidence=1., **change)})
    assert decision["next_action"] == "request_expanded_retrieval"


def _expanded_state(base=.15):
    return {"baseline": baseline(base), "cues": cues(.5, 0, 30.)}


@pytest.mark.parametrize("base,score", [(.1, .162), (.15, .17), (.157, .177)])
def test_expanded_admission_uses_exact_inclusive_declared_thresholds(base, score):
    decision = decide_assisted_policy({**_expanded_state(base),
                                       "expanded": expanded(score, dominant_coordinate_fraction=.25)})
    assert decision["selected_candidate"] == "expanded"
    assert decision["evidence_summary"]["expanded"]["required_median"] == score


@pytest.mark.parametrize("change", [{"similarity_median": .169999}, {"similarity_median": None},
    {"similarity_median": float("nan")}, {"dominant_coordinate_fraction": .250001},
    {"dominant_coordinate_fraction": None}, {"dominant_coordinate_fraction": -1.},
    {"score_kind": "reference_mean"}, {"score_identity": "different-query-profile"},
    {"status": "partial"}, {"point_count": 1}, {"route_valid": False}])
def test_bad_collapsed_or_incomparable_expanded_track_retains_baseline(change):
    decision = decide_assisted_policy({**_expanded_state(), "expanded": expanded(**change)})
    assert decision["next_action"] == "keep_baseline" and decision["selected_candidate"] == "baseline"


def test_expanded_cannot_claim_improvement_when_baseline_comparison_metrics_missing():
    value = _expanded_state()
    value["baseline"].pop("similarity_median")
    assert decide_assisted_policy({**value, "expanded": expanded(.99)})["selected_candidate"] == "baseline"
    value["baseline"]["route_valid"] = False
    result = decide_assisted_policy({**value, "expanded": expanded(.99)})
    assert result["next_action"] == "no_selection" and result["selected_candidate"] is None


def test_successful_earlier_stage_wins_without_consulting_later_candidates():
    value = {"baseline": baseline(), "cues": cues(.99, 2), "poi_joint": joint(),
             "metric_pf": pf(), "expanded": expanded(.99)}
    decision = decide_assisted_policy(value)
    assert decision["selected_candidate"] == "poi_joint"
    assert "metric_pf" not in decision["evidence_summary"] and "expanded" not in decision["evidence_summary"]
    value["poi_joint"]["status"] = "failed"
    decision = decide_assisted_policy(value)
    assert decision["selected_candidate"] == "metric_pf" and "expanded" not in decision["evidence_summary"]


@pytest.mark.parametrize("stage,value,action", [
    ("poi_joint", {"baseline": baseline(), "cues": cues(.8)}, "request_poi_joint"),
    ("poi_image_track", {"baseline": baseline(), "cues": cues(.2, 1)}, "request_poi_image_track"),
    ("metric_pf", {"baseline": baseline(), "cues": cues(.99, 0)}, "request_metric_pf"),
    ("expanded", _expanded_state(), "request_expanded_retrieval"),
])
def test_not_run_stage_does_not_leak_stale_candidate_metrics(stage, value, action):
    expected = decide_assisted_policy(value)
    stale = {"status": "not_run", "route_valid": True, "point_count": 999,
             "similarity_median": .99, "retained_branch_count": 99, "confidence": 1.}
    actual = decide_assisted_policy({**value, stage: stale})
    assert expected["next_action"] == action and actual == expected


def test_unreached_plane_and_future_payloads_are_not_read():
    class Forbidden(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("unreached future stage was inspected")
    value = {"baseline": baseline(), "cues": cues(.8), "poi_joint": joint(),
             "metric_pf": Forbidden(), "expanded": Forbidden()}
    value["cues"]["plane"] = Forbidden()
    assert decide_assisted_policy(value)["selected_candidate"] == "poi_joint"


def test_policy_has_no_io_and_never_returns_or_changes_routes(monkeypatch):
    value = {"baseline": baseline(), "cues": cues(), "poi_joint": joint()}
    value["baseline"]["route_latlon"] = [[1., 2.], [3., 4.]]
    value["poi_joint"]["route_latlon"] = [[5., 6.], [7., 8.]]
    original = deepcopy(value)
    def forbidden(*args, **kwargs):
        raise AssertionError("policy attempted IO")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    result = decide_assisted_policy(value)
    assert result["selected_candidate"] == "poi_joint" and value == original
    assert "route_latlon" not in json.dumps(result, allow_nan=False)


def test_configuration_and_requests_are_detached_and_hash_is_stable():
    settings = policy_config()
    assert settings["baseline_median_min"] == .162 and settings["joint_vo_valid_min"] == .8
    settings["baseline_median_min"] = 0
    settings["image_only_profile"]["cap"] = 1
    assert decide_assisted_policy({"baseline": baseline()})["next_action"] == "request_ocr_vo"
    value = {"baseline": baseline(), "cues": cues(.2, 1)}
    first = decide_assisted_policy(value)
    first["request_config"]["cap"] = 1
    second = decide_assisted_policy(value)
    assert second["request_config"]["cap"] == 1500
    assert second["config_sha256"] == CONFIG_SHA256 and len(CONFIG_SHA256) == 64
    assert second["experimental"] and second["validation_status"] == "posthoc_dev_unvalidated"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "0.95", 10**1000])
def test_bad_numeric_metrics_are_declined_with_json_safe_evidence(bad):
    value = {"baseline": baseline(), "cues": cues(bad, 0), "expanded": expanded(.17)}
    decision = decide_assisted_policy(value)
    assert decision["selected_candidate"] == "expanded"
    assert decision["evidence_summary"]["cues"]["vo_valid_fraction"] is None
    json.dumps(decision, allow_nan=False)
