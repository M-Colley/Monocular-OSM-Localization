"""Synthetic timed image evidence only; no VO, models, network or labels."""
from copy import deepcopy
import json

import numpy as np
import pytest

from monocular_osm.poi_image_track import (
    PROFILE, PROFILE_SHA256, decode_single_branch_image_track, plan_single_branch,
)

VIDEO = "b" * 64


def _plan():
    clues = {"selected_answer": None, "detections_sha256": "a" * 64,
        "inventory": {"city": "Synthetic City", "scope": "named_city_bbox", "snapshot_sha256": "c" * 64},
        "candidate_clusters": [{"pois": [{"id": "p", "lat": 0., "lon": 0.},
                                            {"id": "q", "lat": 0., "lon": .0002}],
            "conjunctions": [{"witnesses": [{"observations": [
                {"poi_id": "p", "text": "Aster", "t_sec": 0., "confidence": .99},
                {"poi_id": "q", "text": "Beryl", "t_sec": 1., "confidence": .99}]}]}]}]}
    native = {"source": "same_video_native_frames", "native_bbox_grid": True, "sample_count": 24,
              "video_sha256": VIDEO, "frame_shape_hw": [480, 640], "window": [0., 100.]}
    return plan_single_branch(clues, native)


def _evidence(plan):
    return {"query_indices": np.array([0, 5, 10]), "query_times": np.array([0., 1., 2.]),
        "ref_ids": np.array(["mapillary:a", "mapillary:b", "mapillary:c", "mapillary:hub"]),
        "ref_latlon": np.array([[0., 0.], [0., .0002], [0., .0004], [0., .009]]),
        "similarities": np.array([[.8, .1, .1, .95], [.1, .8, .1, .95], [.1, .1, .8, .95]], np.float32),
        "provenance": {"source": "mapillary", "center": plan["center"], "radius_m": 3000., "cap": 1500,
            "query_mode": "crops", "backbone": {"name": "megaloc", "hub": "fixed-repository"},
            "preprocessing": {"descriptor_normalization": "L2", "dimensions": 4}}}


def test_reference_mean_removes_constant_hub_without_any_vo_input():
    plan = _plan()
    evidence = _evidence(plan)
    output = decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    assert output["chosen_ref_ids"] == ["mapillary:a", "mapillary:b", "mapillary:c"]
    assert output["route_timestamps"] == [0., 1., 2.] and output["query_indices"] == [0, 5, 10]
    assert not output["uses_rigid_vo"] and output["retained_branch_count"] == 1
    assert output["provenance"]["profile_sha256"] == PROFILE_SHA256
    np.testing.assert_allclose(output["selected_raw_cosines"], [.8, .8, .8])
    assert min(output["selected_centered_scores"]) > .4
    assert output["dominant_coordinate_fraction"] == pytest.approx(1 / 3)
    json.dumps(output, allow_nan=False)


def test_each_actual_time_interval_controls_transitions():
    plan = _plan()
    evidence = _evidence(plan)
    evidence.update(query_times=np.array([0., 1., 100.]), ref_ids=np.array(["mapillary:a", "mapillary:b"]),
                    ref_latlon=np.array([[0., 0.], [0., .009]]),
                    similarities=np.array([[.95, .1], [.7, .8], [.1, .95]], np.float32))
    output = decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    assert output["chosen_ref_ids"] == ["mapillary:a", "mapillary:a", "mapillary:b"]
    assert output["route_timestamps"] == [0., 1., 100.]


def test_tie_order_is_reference_identity_based_not_input_column_order():
    plan = _plan()
    evidence = _evidence(plan)
    evidence["similarities"][:] = .5
    first = decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    permuted = deepcopy(evidence)
    order = [3, 2, 1, 0]
    for key in ("ref_ids", "ref_latlon"):
        permuted[key] = permuted[key][order]
    permuted["similarities"] = permuted["similarities"][:, order]
    second = decode_single_branch_image_track(permuted, plan, query_video_sha256=VIDEO)
    assert first["chosen_ref_ids"] == second["chosen_ref_ids"] == ["mapillary:a"] * 3
    assert first["degenerate"] and first["dominant_coordinate_fraction"] == 1.


@pytest.mark.parametrize("count", [0, 2])
def test_unique_literal_branch_is_required_before_any_reference_decoding(count):
    source = _plan()
    clues = deepcopy(source["clues"])
    clues["candidate_clusters"] = clues["candidate_clusters"] * count
    with pytest.raises(ValueError, match="exactly one"):
        plan_single_branch(clues, source["ocr_provenance"])


@pytest.mark.parametrize("fault", ["external_ocr", "non_native", "wrong_count", "missing_video", "missing_inventory"])
def test_plan_requires_native_ocr_and_city_inventory_fingerprints(fault):
    source = _plan()
    clues, native = deepcopy(source["clues"]), deepcopy(source["ocr_provenance"])
    if fault == "external_ocr": native["source"] = "other_video"
    elif fault == "non_native": native["native_bbox_grid"] = False
    elif fault == "wrong_count": native["sample_count"] = 12
    elif fault == "missing_video": native["video_sha256"] = None
    else: clues["inventory"]["snapshot_sha256"] = None
    with pytest.raises(ValueError, match="native same-video OCR"):
        plan_single_branch(clues, native)


@pytest.mark.parametrize("fault", ["mode", "radius", "cap", "backbone", "normalization", "center"])
def test_decoding_refuses_wrong_acquisition_or_encoder_profile(fault):
    plan = _plan()
    evidence = _evidence(plan)
    p = evidence["provenance"]
    if fault == "mode": p["query_mode"] = "warp"
    elif fault == "radius": p["radius_m"] = 1000
    elif fault == "cap": p["cap"] = 500
    elif fault == "backbone": p["backbone"]["name"] = "other"
    elif fault == "normalization": p["preprocessing"]["descriptor_normalization"] = "none"
    else: p["center"] = [1., 1.]
    with pytest.raises(ValueError, match="fixed acquisition/encoder profile"):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)


@pytest.mark.parametrize("fault", ["times", "indices", "nan", "cosine", "ids", "prefix", "outside"])
def test_invalid_matrix_inventory_or_timestamps_cannot_fall_back_to_a_track(fault):
    plan = _plan()
    evidence = _evidence(plan)
    if fault == "times": evidence["query_times"] = np.array([0., 0., 2.])
    elif fault == "indices": evidence["query_indices"] = np.array([0., 5., 10.])
    elif fault == "nan": evidence["similarities"][0, 0] = np.nan
    elif fault == "cosine": evidence["similarities"][0, 0] = 2.
    elif fault == "ids": evidence["ref_ids"][1] = evidence["ref_ids"][0]
    elif fault == "prefix": evidence["ref_ids"][0] = "wrong:a"
    else: evidence["ref_latlon"][0, 1] = .05
    with pytest.raises(ValueError):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)


def test_changed_plan_or_query_video_refuses_stale_evidence():
    plan = _plan()
    evidence = _evidence(plan)
    with pytest.raises(ValueError, match="same video"):
        decode_single_branch_image_track(evidence, plan, query_video_sha256="d" * 64)
    plan["center"][0] += .001
    with pytest.raises(ValueError, match="content changed"):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)


@pytest.mark.parametrize("fault", ["queries", "references"])
def test_budgets_are_checked_before_cast_or_quadratic_allocation(monkeypatch, fault):
    plan = _plan()
    evidence = _evidence(plan)
    if fault == "queries":
        evidence.update(query_indices=np.arange(81), query_times=np.arange(81, dtype=float),
                        similarities=np.zeros((81, 4), np.float32))
    else:
        evidence.update(ref_ids=np.array([f"mapillary:{i}" for i in range(1501)]),
                        ref_latlon=np.zeros((1501, 2)), similarities=np.zeros((3, 1501)))
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized evidence reached allocation/decoding")
    monkeypatch.setattr("monocular_osm.poi_image_track.np.ascontiguousarray", forbidden)
    monkeypatch.setattr("monocular_osm.poi_image_track._decode", forbidden)
    with pytest.raises(ValueError):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)


def test_pure_decoder_ignores_vo_and_preserves_inputs(monkeypatch):
    plan = _plan()
    evidence = _evidence(plan)
    original_plan, original_scores = deepcopy(plan), evidence["similarities"].copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("pure decoder attempted IO")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    first = decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    evidence["vo"] = {"path": "must never be read", "trajectory": object()}
    second = decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    assert first == second and plan == original_plan
    np.testing.assert_array_equal(evidence["similarities"], original_scores)


def test_fixed_profile_cannot_be_overridden_or_silently_mutated():
    with pytest.raises(TypeError):
        PROFILE["cap"] = 10
    plan = _plan()
    evidence = _evidence(plan)
    plan["profile"]["cap"] = 10
    with pytest.raises(ValueError, match="profile content changed"):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)


@pytest.mark.parametrize("window", [None, [0.], [-1., 100.], [2., 2.], [0., "later"], [False, 100.]])
def test_native_ocr_window_must_be_declared_and_finite(window):
    original = _plan()
    native = original["ocr_provenance"]
    native["window"] = window
    with pytest.raises(ValueError, match="window"):
        plan_single_branch(original["clues"], native)


@pytest.mark.parametrize("times", [[1000., 1001., 1002.], [0., 1., 101.]])
def test_same_video_evidence_outside_native_ocr_window_is_refused(times):
    plan = _plan()
    evidence = _evidence(plan)
    evidence["query_times"] = np.array(times)
    with pytest.raises(ValueError, match="native OCR video window"):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)


def test_query_cannot_precede_native_ocr_window_and_open_ended_window_is_allowed():
    original = _plan()
    native = original["ocr_provenance"]
    native["window"] = [1., None]
    plan = plan_single_branch(original["clues"], native)
    evidence = _evidence(plan)
    with pytest.raises(ValueError, match="native OCR video window"):
        decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)
    evidence["query_times"] += 1.
    assert decode_single_branch_image_track(evidence, plan, query_video_sha256=VIDEO)["status"] == "complete"
