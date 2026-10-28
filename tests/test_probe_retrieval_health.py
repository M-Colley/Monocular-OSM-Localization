"""Prediction-only diagnostics cannot consult evaluation values or label files."""

from argparse import Namespace
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_retrieval_health as M


def _source():
    return {"n_frames": 5, "n_valid_poses": 4,
            "provenance": {"config": {"vo_start_sec": 0., "vo_end_sec": 4.}},
            "vpr_prior": {"track": [[0., 0., 0., .2], [1., 0., .01, .3], [2., 0., 0., .4],
                                      [3., 0., 0., .5], [4., 0., 0., .6]]}}


def test_known_hub_outlier_features_use_raw_and_exact_running_median5():
    f = M.derive_features(_source())
    assert f["status"] == "ok" and f["valid_pose_fraction"] == .8
    assert f["similarity_median"] == pytest.approx(.4)
    assert f["similarity_std"] == pytest.approx(math.sqrt(.02))
    assert f["unique_coordinate_count"] == 2 and f["dominant_coordinate_fraction"] == .8
    assert f["raw"]["arc_m"] == pytest.approx(2226.39, abs=.1)
    assert f["running_median5"]["arc_m"] == 0
    assert f["raw"]["speed_mps"]["max"] == pytest.approx(1113.195, abs=.1)
    assert f["raw"]["speed_mps"]["p95"] == f["raw"]["speed_mps"]["max"]
    assert f["running_median5"]["speed_mps"]["p95"] == 0
    assert f["selected_track_length_over_5_5_duration"] == 0


def test_disallowed_json_values_are_never_decoded(monkeypatch):
    source = _source()
    source["scores"] = {"symmetric_error": "FORBIDDEN_VALUE"}
    source["answer"] = {"route_latlon": "FORBIDDEN_VALUE"}
    source["vpr_prior"]["evaluation"] = "FORBIDDEN_VALUE"
    source["provenance"]["config"]["ground_truth_waypoints"] = "FORBIDDEN_VALUE"
    source["other"] = 'escaped string } [ \\" , and nested values'
    decoder = json.JSONDecoder()
    def guarded(text, index):
        value, end = decoder.raw_decode(text, index)
        assert value != "FORBIDDEN_VALUE"
        assert not (isinstance(value, dict) and "symmetric_error" in value)
        return value, end
    monkeypatch.setattr(M, "DECODER", SimpleNamespace(raw_decode=guarded))
    projected = M.project_inputs(json.dumps(source).encode())
    assert projected == _source()


def test_missing_track_never_substitutes_answer_or_prior_center():
    source = _source()
    source["vpr_prior"] = {"lat": 10, "lon": 20}
    source["answer"] = {"route_latlon": [[10, 20], [11, 21]]}
    f = M.derive_features(M.project_inputs(json.dumps(source).encode()))
    assert f["status"] == "unavailable" and f["track_count"] == 0
    assert "raw" not in f


def test_stationary_retrieval_can_have_high_cosine_without_motion():
    source = _source()
    for row in source["vpr_prior"]["track"]:
        row[1:] = [1., 2., .9]
    f = M.derive_features(source)
    assert f["similarity_median"] == .9 and f["dominant_coordinate_fraction"] == 1
    assert f["raw"]["arc_m"] == f["running_median5"]["arc_m"] == 0
    assert f["raw"]["speed_mps"]["max"] == 0


@pytest.mark.parametrize("bad", ["repeated_time", "nonfinite", "short_row"])
def test_invalid_tracks_remain_explicit_and_json_safe(bad):
    source = _source()
    if bad == "repeated_time": source["vpr_prior"]["track"][1][0] = 0
    elif bad == "nonfinite": source["vpr_prior"]["track"][1][3] = float("nan")
    else: source["vpr_prior"]["track"][1].pop()
    f = M.derive_features(source)
    assert f["status"] == "invalid"
    json.dumps(f, allow_nan=False)


def test_features_do_not_depend_on_id_or_unselected_result_fields():
    a, b = _source(), _source()
    a.update(id="one", scores={"error": 1}, answer="first")
    b.update(id="another", scores={"error": 9999}, answer="second")
    assert M.derive_features(M.project_inputs(json.dumps(a).encode())) == M.derive_features(M.project_inputs(json.dumps(b).encode()))


def test_probe_records_all_members_without_reading_gt_or_emitting_coordinates(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "benchmarks").mkdir(parents=True)
    forbidden = root / "never_open_labels.json"
    forbidden.write_text('"FORBIDDEN_VALUE"', encoding="utf-8")
    manifest = {"clips": [{"id": cid, "split": "dev", "available_locally": True, "gt": str(forbidden)}
                          for cid in ("a", "b")]}
    (root / "benchmarks/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    v6, baseline, out = tmp_path / "v6", tmp_path / "baseline", tmp_path / "features"
    for folder in (v6 / "a/rep0", baseline / "a/rep0"):
        folder.mkdir(parents=True)
        source = _source()
        source["scores"] = "FORBIDDEN_VALUE"
        (folder / "result.json").write_text(json.dumps(source), encoding="utf-8")
    original = Path.read_bytes
    def guarded(path):
        assert path != forbidden
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    args = Namespace(root=root, v6_root=v6, baseline_root=baseline, baseline_clips=["a"], out=out)
    result = M.run(args)
    assert len(result["records"]) == 3 and result["records"][1]["features"]["reason"] == "result_file_missing"
    assert "FORBIDDEN_VALUE" not in json.dumps(result)
    assert "route_latlon" not in json.dumps(result) and "track\": [" not in json.dumps(result)
    assert result["policy"]["route_selection_policy"] is None
    assert result["policy"]["ground_truth_opened"] is False
    assert result["records"][0]["source_sha256"]
    with pytest.raises(SystemExit, match="already exists"):
        M.run(args)
