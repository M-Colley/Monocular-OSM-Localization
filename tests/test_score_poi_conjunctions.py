import hashlib
import json

import pytest

from scripts.score_poi_conjunctions import interpolate_gt, load_frozen, score_record


def test_frozen_artifact_tamper_is_rejected(tmp_path):
    source = tmp_path / "record.json"
    source.write_text(json.dumps({"id": "clip", "script_sha256": "script", "selected_answer": None}))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"ground_truth_used": False, "script_sha256": "script",
                                 "artifacts": [{"id": "clip", "path": str(source),
                                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}))
    assert load_frozen(frozen)[1][0]["id"] == "clip"
    source.write_text("{}")
    with pytest.raises(ValueError, match="fingerprint changed"):
        load_frozen(frozen)


def test_time_interpolation_refuses_extrapolation():
    waypoints = [{"t_sec": 0, "lat": 0, "lon": 0}, {"t_sec": 10, "lat": 0, "lon": 0.01}]
    assert interpolate_gt(waypoints, 5) == pytest.approx((0, 0.005))
    assert interpolate_gt(waypoints, 10) == pytest.approx((0, 0.01))
    with pytest.raises(ValueError, match="observation must lie"):
        interpolate_gt(waypoints, 11)


def test_offline_visibility_scores_every_branch_without_selecting():
    waypoints = [{"t_sec": 0, "lat": 0, "lon": 0}, {"t_sec": 10, "lat": 0, "lon": 0.001}]
    def cluster(lon):
        return {"poi_ids": ["a", "b"], "centre_latlon": [0, lon], "conjunctions": [{
            "pois": [{"id": "a", "lat": 0, "lon": lon}, {"id": "b", "lat": 0, "lon": lon}],
            "witnesses": [{"observations": [{"poi_id": "a", "t_sec": 0},
                                               {"poi_id": "b", "t_sec": 10}]}]}]}
    record = {"id": "clip", "bounds": {"visibility_m": 100},
              "candidate_clusters": [cluster(0.0005), cluster(1)]}
    result = score_record(record, waypoints)
    assert result["selected_answer"] is None
    assert [c["candidate_index"] for c in result["candidates"]] == [0, 1]
    assert result["candidates"][0]["witnesses"][0]["both_within_frozen_visibility"]
    assert not result["candidates"][1]["witnesses"][0]["both_within_frozen_visibility"]
