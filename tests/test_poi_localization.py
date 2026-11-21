"""Generic input-only orchestration: no model, network, GT, or default answer."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from pyproj import Transformer

from monocular_osm import poi_localization as M


def _case():
    inverse = Transformer.from_crs(3857, 4326, always_xy=True)
    pois = []
    for identifier, brand, x in [("a", "Example Grocery", 40), ("b", "Example Pharmacy", 70),
                                  ("c", "Example Grocery", 1040), ("d", "Example Pharmacy", 1070)]:
        lon, lat = inverse.transform(x, -20)
        pois.append({"id": identifier, "lat": lat, "lon": lon, "tags": {"brand": brand}})
    observations = [{"text": "Example Grocery", "t_sec": 2., "confidence": .95,
                     "bbox": [90., 40., 100., 50.]},
                    {"text": "Example Pharmacy", "t_sec": 8., "confidence": .9,
                     "bbox": [90., 40., 100., 50.]}]
    times = np.arange(11, dtype=float)
    args = dict(frames=[None] * len(times), vo_xy=np.c_[times, np.zeros(11)], timestamps=times,
                intrinsics=np.array([[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]]),
                city="Example city", road_xy=np.c_[np.arange(0, 1101, 100), np.zeros(12)],
                map_crs="EPSG:3857", detections=observations,
                inventory={"city": "Example city", "pois": pois},
                config=M.POILocalizationConfig(radius_m=1000, cap_per_branch=2, n_query=3))
    evidence = []
    for unique, x in [("mapillary:west", 80), ("mapillary:east", 1080)]:
        points = np.array([inverse.transform(50, -20), inverse.transform(x, -20)])[:, ::-1]
        evidence.append({"query_indices": np.array([0, 5, 10]), "query_times": times[[0, 5, 10]],
            "similarities": np.array([[.2, .8], [.3, .5], [.4, .7]], dtype=np.float32),
            "ref_latlon": points, "ref_ids": ["mapillary:shared", unique],
            "provenance": {"backbone": {"name": "synthetic", "hub": "synthetic@fixed"},
                "query_mode": "warp",
                "preprocessing": {"reference": "warp512", "query": "warp512", "dimensions": 2},
                "cache": unique, "encoding_identity": "legacy_unverified"}})
    return args, evidence


def _mock_alignment(monkeypatch):
    captured = []

    def align(vo, timestamps, K, clues, **kwargs):
        captured.append({"vo": vo, "clues": clues, **kwargs})
        return {"selected": {"hypothesis_id": "hypothesis_1", "route_latlon": [[0., 0.], [0., .001]],
                             "route_timestamps": [float(timestamps[0]), float(timestamps[-1])]},
                "hypotheses": deepcopy(clues["candidate_clusters"]), "ranked_hypotheses": []}

    monkeypatch.setattr(M.core, "align_candidate_clusters", align)
    return captured


def test_equal_branch_budgets_dedup_and_one_query_cache_preserve_every_clue(monkeypatch):
    args, evidence = _case()
    original = deepcopy(args)
    aligned = _mock_alignment(monkeypatch)
    calls, cache = [], {}

    def retrieve(frames, timestamps, center, **kwargs):
        calls.append((frames, center, kwargs))
        kwargs["query_cache"]["query"] = "shared"
        return evidence[len(calls) - 1]

    result = M.localize_with_poi(**args, retrieve=retrieve, query_cache=cache)
    assert result["status"] == "ok" and result["selected_answer"] is None
    assert len(calls) == len(result["region_plans"]) == len(result["clues"]["candidate_clusters"]) == 2
    assert all(c[2] == {"radius_m": 1000., "cap": 2, "n_query": 3, "query_cache": cache} for c in calls)
    assert all(c[2]["query_cache"] is cache for c in calls)
    assert result["reference_union"]["ids"] == ["mapillary:shared", "mapillary:west", "mapillary:east"]
    assert result["regions"][1]["duplicate_reference_count"] == 1
    assert result["regions"][1]["provenance"]["encoding_identity"] == "legacy_unverified"
    assert len(aligned) == 1 and aligned[0]["config"] == M.core.POIAlignmentConfig()
    assert aligned[0]["appearance_field"].provenance["reference_mean"] is False
    assert len(aligned[0]["clues"]["candidate_clusters"]) == 2
    assert aligned[0]["appearance_field"].values.dtype == np.float32
    assert result["diagnostic_position"]["source"] == "experimental_poi_alignment"
    assert args["inventory"] == original["inventory"] and args["detections"] == original["detections"]
    np.testing.assert_array_equal(args["vo_xy"], original["vo_xy"])
    json.dumps(result, allow_nan=False)


def test_no_literal_clues_returns_inactive_before_retrieval():
    args, _ = _case()
    args["detections"] = []
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: pytest.fail("no clues"))
    assert result["status"] == "inactive" and result["reason"] == "no_literal_poi_conjunctions"
    assert result["diagnostic_position"] is None and result["selected_answer"] is None


def test_failed_region_does_not_drop_its_branch_or_promote_partial_alignment(monkeypatch):
    args, evidence = _case()
    aligned = _mock_alignment(monkeypatch)
    queue = iter([{"reason": "unavailable", "provenance": {"fetch": "failed"}}, evidence[1]])
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: next(queue))
    assert result["status"] == "inactive" and result["reason"] == "incomplete_retrieval"
    assert len(result["regions"]) == len(aligned[0]["clues"]["candidate_clusters"]) == 2
    assert result["regions"][0]["provenance"] == {"fetch": "failed"}
    assert result["diagnostic_position"] is not None and result["selected_answer"] is None


@pytest.mark.parametrize("mismatch", ["query", "backbone", "query_mode", "preprocessing", "coordinates", "scores", "nonfinite", "cap"])
def test_incompatible_region_stays_failed_and_contributes_no_partial_columns(monkeypatch, mismatch):
    args, evidence = _case()
    _mock_alignment(monkeypatch)
    second = evidence[1]
    if mismatch == "query":
        second["query_indices"] = np.array([0, 3, 10])
        second["query_times"] = args["timestamps"][[0, 3, 10]]
    elif mismatch == "backbone": second["provenance"]["backbone"]["hub"] = "other"
    elif mismatch == "query_mode": second["provenance"]["query_mode"] = "crops"
    elif mismatch == "preprocessing": second["provenance"]["preprocessing"]["query"] = "crops"
    elif mismatch == "coordinates": second["ref_latlon"][0, 0] += .001
    elif mismatch == "scores": second["similarities"][0, 0] += .01
    elif mismatch == "nonfinite": second["similarities"][0, 0] = np.nan
    elif mismatch == "cap":
        second["ref_ids"].append("mapillary:over_cap")
        second["ref_latlon"] = np.r_[second["ref_latlon"], second["ref_latlon"][:1]]
        second["similarities"] = np.c_[second["similarities"], second["similarities"][:, :1]]
    queue = iter(evidence)
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: next(queue))
    assert result["status"] == "inactive" and result["failed_regions"] == ["hypothesis_1"]
    assert result["reference_union"]["ids"] == ["mapillary:shared", "mapillary:west"]
    assert "error" in result["regions"][1]


def test_all_failed_evidence_leaves_ordinary_result_inactive():
    args, _ = _case()
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: None)
    assert result["status"] == "inactive" and result["reason"] == "no_compatible_reference_evidence"
    assert len(result["failed_regions"]) == 2 and result["diagnostic_position"] is None


def test_core_receives_real_raw_fields_and_full_city_starts_without_any_model():
    args, evidence = _case()
    queue = iter(evidence)
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: next(queue))
    assert result["status"] == "ok", result.get("error")
    assert result["alignment"]["search"]["hypothesis_count"] == 2
    assert len(result["alignment"]["selected"]["route_latlon"]) == len(args["timestamps"])
    assert result["selected_answer"] is None and result["confidence_calibrated"] is False


def test_matrix_budget_refuses_before_any_retrieval_or_branch_dropping():
    args, _ = _case()
    args["config"] = replace(args["config"], max_evidence_values=1)
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: pytest.fail("budget refused"))
    assert result["status"] == "inactive" and "before requesting evidence" in result["error"]
    assert len(result["clues"]["candidate_clusters"]) == 2


@pytest.mark.parametrize("extent,queries,road_limit,error", [
    (1000, 2, 1, "road raster"), (1_000_000, 80, 4_000_000, "appearance raster")])
def test_map_grids_refuse_before_allocating_large_arrays(monkeypatch, extent, queries, road_limit, error):
    import scipy.ndimage  # Finish optional imports before guarding allocations.

    def forbidden(*args, **kwargs):
        raise AssertionError("grid allocated before budget check")

    monkeypatch.setattr(M.np, "zeros", forbidden)
    with pytest.raises(ValueError, match=error):
        M._map_fields(np.array([[0., 0.], [extent, extent]]), np.array([[0., 0.]]),
                      1., queries, M.core.POIAlignmentConfig(),
                      M.POILocalizationConfig(radius_m=1000, cap_per_branch=500,
                                              max_road_grid_cells=road_limit), None)


def test_provided_full_city_starts_are_shared_without_poi_based_pruning(monkeypatch):
    args, evidence = _case()
    aligned = _mock_alignment(monkeypatch)
    starts = np.array([[0., 0.], [1000., 0.]])
    queue = iter(evidence)
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: next(queue), start_samples=starts,
                                map_provenance={"city": args["city"], "scope": "whole_city", "sha256": "synthetic"})
    np.testing.assert_array_equal(aligned[0]["start_samples"], starts)
    assert result["grid"]["start_policy"] == "caller-supplied full-city starts"
    assert len(aligned[0]["clues"]["candidate_clusters"]) == 2


@pytest.mark.parametrize("invalid", ["timeline", "K", "stationary", "geographic_crs", "cropped_map"])
def test_invalid_metric_or_video_inputs_fail_before_retrieval(invalid):
    args, _ = _case()
    extra = {}
    if invalid == "timeline": args["detections"][0]["t_sec"] = -1
    elif invalid == "K": args["intrinsics"][0, 0] = 0
    elif invalid == "stationary": args["vo_xy"][:] = 0
    elif invalid == "geographic_crs": args["map_crs"] = "EPSG:4326"
    elif invalid == "cropped_map": extra["map_provenance"] = {"scope": "clip_bbox"}
    result = M.localize_with_poi(**args, **extra, retrieve=lambda *a, **k: pytest.fail("invalid inputs"))
    assert result["status"] == "inactive" and result["reason"] == "poi_localization_failed"


def test_orchestration_itself_does_not_open_files_or_network(monkeypatch):
    import scipy.ndimage
    args, evidence = _case()
    _mock_alignment(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("orchestration attempted external access")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    queue = iter(evidence)
    result = M.localize_with_poi(**args, retrieve=lambda *a, **k: next(queue))
    assert result["status"] == "ok", result.get("error")


@pytest.mark.parametrize("values", [dict(radius_m=True, cap_per_branch=5),
                                    dict(radius_m=1000, cap_per_branch=0),
                                    dict(radius_m=1000, cap_per_branch=500, n_query=81)])
def test_experimental_budgets_are_explicit_and_bounded(values):
    with pytest.raises(ValueError):
        M.POILocalizationConfig(**values)
