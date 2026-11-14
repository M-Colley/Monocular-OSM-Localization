from copy import deepcopy

import pytest

from monocular_osm import assisted_policy as v1, assisted_policy_v2 as v2
from monocular_osm.assisted_profiles import resolve_policy
from monocular_osm.poi_image_union import PROFILE_SHA256


def state(branches=2, valid=.5):
    return {"baseline": {"status": "complete", "route_valid": True, "point_count": 80,
        "similarity_median": .15, "score_kind": "raw_cosine", "score_identity": "fixed-warp"},
        "cues": {"status": "complete", "duration_s": 180.,
            "vo": {"status": "complete", "valid_fraction": valid},
            "ocr": {"status": "complete", "clue_branch_count": branches}}}


def union(branches=2, **changes):
    return {"status": "complete", "route_valid": True, "point_count": 80,
        "retained_branch_count": branches, "successful_branch_count": branches,
        "uses_rigid_vo": False, "profile_sha256": PROFILE_SHA256, "candidate_pruning": False,
        "score_normalization": "reference_mean", "query_mode": "crops", "radius_m": 3000,
        "cap": 1500, **changes}


@pytest.mark.parametrize("branches", range(1, 7))
def test_every_eligible_literal_branch_is_requested_and_all_must_complete(branches):
    data = state(branches)
    before = deepcopy(data)
    result = v2.decide_assisted_policy(data)
    assert result["next_action"] == "request_poi_image_union"
    assert result["request_config"]["branches"] == branches
    assert result["request_config"]["profile_sha256"] == PROFILE_SHA256
    assert data == before
    assert v2.decide_assisted_policy({**data, "poi_image_union": union(branches)})["selected_candidate"] == "poi_image_union"


@pytest.mark.parametrize("branches", [0, 7, 1000, True, None])
def test_no_truncation_or_coercion_of_branch_count(branches):
    data = state(branches)
    data["poi_image_union"] = union()
    assert v2.decide_assisted_policy(data)["next_action"] == "request_expanded_retrieval"


@pytest.mark.parametrize("changes", [{"status": "partial"}, {"status": "inactive"},
    {"route_valid": False}, {"point_count": 79}, {"point_count": True},
    {"retained_branch_count": 1}, {"successful_branch_count": 1}, {"uses_rigid_vo": True},
    {"profile_sha256": "a"*64}, {"candidate_pruning": True}, {"candidate_pruning": None},
    {"query_mode": "warp"}, {"score_normalization": "raw"}, {"radius_m": 3001}, {"cap": 1499}])
def test_incomplete_or_profile_mismatched_union_falls_through(changes):
    data = {**state(), "poi_image_union": union(**changes)}
    decision = v2.decide_assisted_policy(data)
    assert decision["next_action"] == "request_expanded_retrieval"
    assert "decline_reason" in decision["evidence_summary"]["poi_image_union"]


def test_legacy_single_branch_result_cannot_replace_union():
    data = {**state(1), "poi_image_track": union(1), "poi_image_union": {"status": "failed"}}
    assert v2.decide_assisted_policy(data)["next_action"] == "request_expanded_retrieval"
    assert v1.decide_assisted_policy({**state(1), "poi_image_track": union(1)})["selected_candidate"] == "poi_image_track"


def test_strong_baseline_precedes_cues_and_ignores_future_union():
    data = state()
    data["baseline"]["similarity_median"] = .162
    expected = v2.decide_assisted_policy({"baseline": data["baseline"]})
    assert expected["selected_candidate"] == "baseline"
    assert expected == v2.decide_assisted_policy({**data, "poi_image_union": union()})


def test_native_cues_and_high_vo_joint_keep_original_precedence():
    data = state(valid=.8)
    assert v2.decide_assisted_policy(data)["next_action"] == "request_poi_joint"
    assert v2.decide_assisted_policy({**data, "cues": {"status": "not_run"},
        "poi_image_union": union()})["next_action"] == "request_ocr_vo"


def test_later_score_or_legacy_candidate_does_not_influence_union_request():
    data = state()
    expected = v2.decide_assisted_policy(data)
    assert expected == v2.decide_assisted_policy({**data, "expanded": {"status": "complete",
        "similarity_median": .99}, "poi_image_track": union(1), "metric_pf": union()})


def test_failed_union_and_failed_expanded_can_keep_valid_baseline():
    data = {**state(), "poi_image_union": {"status": "failed"}, "expanded": {"status": "failed"}}
    result = v2.decide_assisted_policy(data)
    assert result["selected_candidate"] == "baseline"
    assert result["policy_version"] == v2.POLICY_VERSION
    assert result["config_sha256"] == v2.CONFIG_SHA256 != v1.CONFIG_SHA256


def test_config_is_detached_and_profile_resolution_is_explicit():
    config = v2.policy_config()
    config["image_union_profile"]["cap_per_branch"] = 2
    assert v2.policy_config()["image_union_profile"]["cap_per_branch"] == 1500
    assert resolve_policy("v1") is v1 and resolve_policy("v2") is v2
    with pytest.raises(ValueError): resolve_policy("latest")
