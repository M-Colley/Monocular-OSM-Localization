"""Synthetic motion/cues test alignment without imagery, accounts or labels."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from monocular_osm.poi_alignment import (
    POIAlignmentConfig, RasterField, align_candidate_clusters, align_poi_hypotheses,
    build_appearance_field,
)


def _case():
    times = np.arange(11, dtype=float)
    path = np.c_[times, np.zeros(11)]
    K = np.array([[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]])
    observations = [{"id": "first", "text": "Alpha", "t_sec": 2., "confidence": .9,
                     "bbox": [90., 40., 100., 50.]},
                    {"id": "second", "text": "Beta", "t_sec": 8., "confidence": .95,
                     "bbox": [90., 40., 100., 50.]}]
    pois = [{"id": "a0", "xy": [50., -20.]}, {"id": "b0", "xy": [80., -20.]},
            {"id": "a1", "xy": [1050., -20.]}, {"id": "b1", "xy": [1080., -20.]}]
    branches = [{"id": "west", "assignments": {"first": ["a0"], "second": ["b0"]}},
                {"id": "east", "assignments": {"first": ["a1"], "second": ["b1"]}}]
    config = POIAlignmentConfig(average_speed_range_mps=(5., 5.), length_count=1, yaw_step_deg=360.,
                                basins_per_hypothesis=1, seed_top_scores=4,
                                refine_xy_offsets_m=(0.,), refine_yaw_offsets_deg=(0.,),
                                refine_length_multipliers=(1.,), retained_candidates=8,
                                max_coarse_poses=100, max_total_coarse_scores=200)
    values = np.zeros((80, 140), dtype=np.float32)
    values[:, 100:] = 3.
    appearance = RasterField(values, (-100., -400.), 10., -6., provenance={"snapshot": "synthetic"})
    arguments = dict(vo_xy=path, timestamps=times, intrinsics=K, observations=observations,
                     pois=pois, hypotheses=branches, config=config,
                     start_samples=np.array([[0., 0.], [1000., 0.]]))
    return arguments, appearance


def test_appearance_resolves_identical_literal_branches_and_every_branch_is_preserved():
    arguments, appearance = _case()
    result = align_poi_hypotheses(**arguments, appearance_field=appearance)
    assert result["selected"]["hypothesis_id"] == "east"
    np.testing.assert_allclose(result["selected"]["route_xy"][0], [1000., 0.])
    assert [h["id"] for h in result["hypotheses"]] == ["west", "east"]
    assert [h["id"] for h in result["ranked_hypotheses"]] == ["east", "west"]
    assert all(h["refinement_basins"] == 1 for h in result["hypotheses"])
    assert result["hypothesis_objective_margin"] > 0
    assert result["confidence_calibrated"] is False
    assert result["search"]["coarse_pose_count"] == 2
    assert result["provenance"]["appearance"]["sources"] == {"snapshot": "synthetic"}
    assert sum(o["weighted_loss"] for o in result["selected"]["observations"]) == pytest.approx(result["selected"]["poi_loss"])


def test_absent_appearance_is_explicit_and_does_not_hide_branch_ambiguity():
    arguments, _appearance = _case()
    result = align_poi_hypotheses(**arguments)
    assert result["provenance"]["appearance"] is None
    assert result["hypothesis_objective_margin"] == pytest.approx(0)
    assert len(result["ranked_hypotheses"]) == 2
    assert result["ranked_hypotheses"][0]["id"] == result["selected"]["hypothesis_id"]


def test_image_right_is_clockwise_from_forward_in_the_map_plane():
    arguments, _ = _case()
    arguments["observations"] = arguments["observations"][:1]
    arguments["pois"] = [{"id": "above", "xy": [50., 20.]}, {"id": "below", "xy": [50., -20.]}]
    arguments["hypotheses"] = [{"id": "above", "assignments": {"first": ["above"]}},
                               {"id": "below", "assignments": {"first": ["below"]}}]
    arguments["start_samples"] = np.array([[0., 0.]])
    result = align_poi_hypotheses(**arguments)
    assert result["selected"]["hypothesis_id"] == "below"
    assert result["selected"]["observations"][0]["bearing_error_deg"] < 3


def test_missing_bbox_skips_bearing_without_fabricating_camera_information():
    arguments, _ = _case()
    for observation in arguments["observations"]:
        observation["bbox"] = None
    result = align_poi_hypotheses(**arguments)
    assert all(not o["bearing_available"] and o["bearing_error_deg"] is None
               for o in result["selected"]["observations"])


def test_metric_alignment_is_pure_and_sources_are_bound_to_actual_arrays(monkeypatch):
    arguments, appearance = _case()
    original = deepcopy(arguments)
    def forbidden(*args, **kwargs):
        raise AssertionError("alignment attempted file or network access")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    result = align_poi_hypotheses(**arguments, appearance_field=appearance,
                                  provenance={"inventory_digest": "synthetic-source"})
    np.testing.assert_array_equal(arguments["vo_xy"], original["vo_xy"])
    assert arguments["observations"] == original["observations"] and arguments["pois"] == original["pois"]
    assert len(result["provenance"]["vo_sha256"]) == 64
    assert result["provenance"]["caller_sources"] == {"inventory_digest": "synthetic-source"}
    json.dumps(result, allow_nan=False)


def test_timed_anchor_proposals_are_bounded_and_use_every_hypothesis():
    arguments, appearance = _case()
    arguments["start_samples"] = None
    result = align_poi_hypotheses(**arguments, appearance_field=appearance)
    assert result["search"]["start_mode"] == "timed_poi_grid"
    assert result["search"]["coarse_pose_count"] <= arguments["config"].max_coarse_poses
    assert {h["id"] for h in result["hypotheses"]} == {"west", "east"}


def test_field_keeps_query_identity_instead_of_pairing_unrelated_rows():
    field = RasterField(np.stack([np.ones((3, 3)), np.full((3, 3), 7.)]), (0., 0.), 1., -9., np.array([0., 2.]))
    positions = np.array([[[.5, .5], [1.5, 1.5]]])
    np.testing.assert_array_equal(field.sample(positions, [0., 2.]), [[1., 7.]])
    with pytest.raises(ValueError, match="query times"):
        field.sample(positions, [0., 1.])


def test_appearance_builder_records_pool_identity_and_checks_bounds():
    config = POIAlignmentConfig(field_cell_m=10., appearance_sigma_m=10.)
    refs = np.array([[10., 10.], [20., 20.]])
    field = build_appearance_field(refs, [[.1, .3], [.2, .4]], [0., 1.],
                                    origin_xy=(0., 0.), shape=(5, 5), config=config)
    assert field.provenance["reference_count"] == 2
    assert len(field.provenance["similarities_sha256"]) == 64
    with pytest.raises(ValueError, match="include every"):
        build_appearance_field(refs, [[.1, .3]], [0.], origin_xy=(0., 0.), shape=(2, 2), config=config)


@pytest.mark.parametrize("change", [
    {"max_hypotheses": 1}, {"max_coarse_poses": 1}, {"max_total_coarse_scores": 1}, {"max_fine_poses": 1},
])
def test_budgets_reject_the_search_instead_of_dropping_hypotheses(change):
    arguments, _ = _case()
    arguments["config"] = replace(arguments["config"], **change)
    with pytest.raises(ValueError, match="budget|counts"):
        align_poi_hypotheses(**arguments)


@pytest.mark.parametrize("change", [{"transform_batch": 1.5}, {"visibility_m": -1},
                                    {"distance_sigma_m": 0}, {"bearing_sigma_deg": np.nan},
                                    {"refine_xy_offsets_m": (5,)}])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        POIAlignmentConfig(**change)


def test_unknown_poi_and_nonfinite_motion_are_rejected():
    arguments, _ = _case()
    arguments["hypotheses"][0]["assignments"]["first"] = ["missing"]
    with pytest.raises(ValueError, match="unknown POI"):
        align_poi_hypotheses(**arguments)
    arguments, _ = _case()
    arguments["vo_xy"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        align_poi_hypotheses(**arguments)


def _clusters(arguments):
    from pyproj import Transformer
    inverse = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    locations = {p["id"]: inverse.transform(*p["xy"]) for p in arguments["pois"]}
    result = []
    for branch in arguments["hypotheses"]:
        observations, pois = [], []
        for observation in arguments["observations"]:
            for identifier in branch["assignments"].get(observation["id"], []):
                observations.append({**observation, "poi_id": identifier})
                lon, lat = locations[identifier]
                pois.append({"id": identifier, "lat": lat, "lon": lon, "name": observation["text"]})
        result.append({"id": branch["id"], "pois": pois,
                       "conjunctions": [{"witnesses": [{"observations": observations}]}]})
    return result


def test_literal_cluster_adapter_replays_every_branch_and_preserves_inventory_evidence():
    arguments, appearance = _case()
    clusters = _clusters(arguments)
    source = {"candidate_clusters": clusters, "inventory_sha256": "frozen-synthetic-inventory"}
    original = deepcopy(source)
    result = align_candidate_clusters(
        arguments["vo_xy"], arguments["timestamps"], arguments["intrinsics"], source,
        map_crs="EPSG:3857", start_samples=arguments["start_samples"],
        appearance_field=appearance, config=arguments["config"])
    assert result["selected"]["hypothesis_id"] == "east"
    assert result["candidate_clusters"] == clusters
    assert source == original
    assert result["provenance"]["literal_clue_sources"] == {"inventory_sha256": "frozen-synthetic-inventory"}
    assert len(result["provenance"]["candidate_clusters_sha256"]) == 64
    assert len(result["selected"]["route_latlon"]) == len(arguments["timestamps"])
    assert all(len(h["best"]["observations"]) == 2 for h in result["hypotheses"])
    json.dumps(result, allow_nan=False)


def test_cluster_adapter_rejects_geographic_map_units_and_conflicting_poi_identity():
    arguments, _ = _case()
    clusters = _clusters(arguments)
    with pytest.raises(ValueError, match="metres"):
        align_candidate_clusters(arguments["vo_xy"], arguments["timestamps"], arguments["intrinsics"],
                                 clusters, map_crs="EPSG:4326")
    clusters[1]["pois"].append({**clusters[0]["pois"][0], "lon": 1.})
    with pytest.raises(ValueError, match="conflicting"):
        align_candidate_clusters(arguments["vo_xy"], arguments["timestamps"], arguments["intrinsics"],
                                 clusters, map_crs="EPSG:3857")


def test_incomplete_branch_gets_explicit_missing_observation_penalty():
    arguments, appearance = _case()
    arguments["hypotheses"][1]["assignments"].pop("second")
    result = align_poi_hypotheses(**arguments, appearance_field=appearance)
    east = result["hypotheses"][1]["best"]
    assert east["observations"][1]["poi_id"] is None
    assert east["observations"][1]["weighted_loss"] > 0
    assert east["poi_loss"] == pytest.approx(sum(o["weighted_loss"] for o in east["observations"]))


def test_temporal_field_mismatch_and_negative_road_distances_are_rejected():
    arguments, _ = _case()
    temporal = RasterField(np.zeros((2, 3, 3)), (0., 0.), 1., -6., [0., 2.])
    with pytest.raises(ValueError, match="query times"):
        align_poi_hypotheses(**arguments, appearance_field=temporal, query_times=[0., 1.])
    road = RasterField(np.full((3, 3), -1.), (0., 0.), 1., 200.)
    with pytest.raises(ValueError, match="nonnegative"):
        align_poi_hypotheses(**arguments, road_distance_field=road)


def test_raster_final_row_and_column_are_inclusive():
    field = RasterField(np.arange(9).reshape(3, 3), (0., 0.), 1., -99.)
    positions = np.array([[[2., 2.], [2., 1.5], [2.00001, 2.]]])
    np.testing.assert_array_equal(field.sample(positions, [0., 1., 2.]), [[8., 6.5, -99.]])


@pytest.mark.parametrize("change", [{"yaw_step_deg": 1e-100}, {"length_count": 10**9}])
def test_search_cardinalities_are_checked_before_grid_allocation(monkeypatch, change):
    arguments, _ = _case()
    arguments["config"] = replace(arguments["config"], **change)
    def forbidden(*args, **kwargs):
        raise AssertionError("search grid allocated before budget refusal")
    monkeypatch.setattr("monocular_osm.poi_alignment.np.arange", forbidden)
    monkeypatch.setattr("monocular_osm.poi_alignment.np.geomspace", forbidden)
    with pytest.raises(ValueError, match="before grid allocation"):
        align_poi_hypotheses(**arguments)


def test_anchor_grid_cardinality_is_checked_before_offset_allocation(monkeypatch):
    arguments, _ = _case()
    arguments["start_samples"] = None
    arguments["config"] = replace(arguments["config"], anchor_search_radius_m=1e100)
    original = np.arange
    def bounded(*args, **kwargs):
        if abs(args[0]) > 1e10:
            raise AssertionError("anchor grid allocated before budget refusal")
        return original(*args, **kwargs)
    monkeypatch.setattr("monocular_osm.poi_alignment.np.arange", bounded)
    with pytest.raises(ValueError, match="before grid allocation"):
        align_poi_hypotheses(**arguments)
