from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from monocular_osm.poi_alignment import RasterField
from monocular_osm.road_target_alignment import RoadTargetConfig, _loss, _prepare, align_road_targets


def case():
    times = np.array([0., 5., 10., 15., 20.])
    xy = np.column_stack([times, np.zeros(5)])
    valid = np.ones(5, bool)
    observations = [
        {"id": "first", "ref": "12A", "t_sec": 5., "confidence": .99,
         "candidates": [{"id": "node/a", "xy": [100., 0.]}, {"id": "node/b", "xy": [100., 600.]}]},
        {"id": "second", "ref": "13", "t_sec": 15., "confidence": .95,
         "candidates": [{"id": "node/c", "xy": [200., 0.]}]},
    ]
    config = RoadTargetConfig(near_radius_m=50, advance_radius_m=200, anchor_spacing_m=200,
        mean_speed_range_mps=(5., 10.), length_count=2, yaw_step_deg=90, basins_per_branch=2,
        refine_xy_m=(-50., 0., 50.), refine_yaw_deg=(0.,), refine_length_factors=(1.,), retained_per_branch=2)
    return xy, times, valid, observations, config


def run(*, field=None, mutate=None, config=None):
    xy, times, valid, observations, cfg = case()
    if mutate:
        mutate(xy, times, valid, observations)
    return align_road_targets(xy, times, valid, observations, appearance_field=field, config=config or cfg)


def test_all_assignments_nulls_and_routes_are_preserved():
    result = run()
    assert len(result["branches"]) == (2+1)*(1+1)-1
    assignments = [branch["assignments"] for branch in result["branches"]]
    assert {row["first"] for row in assignments} == {None, "node/a", "node/b"}
    assert {row["second"] for row in assignments} == {None, "node/c"}
    assert result["unlocalized_alternative"]["assignments"] == {"first": None, "second": None}
    assert result["unlocalized_alternative"]["route_xy"] is None
    assert result["selection_scope"] == "anchored diagnostic hypotheses only"
    assert result["policy_adopted"] is False and result["confidence_calibrated"] is False
    assert all(len(branch["best"]["route_xy"]) == 5 for branch in result["branches"])
    assert result["selected"]["branch_id"] == result["ranked_branch_ids"][0]
    assert len({branch["coarse_pose_count"] for branch in result["branches"]}) == 1


def test_actual_times_and_input_hashes_and_immutability():
    xy, times, valid, observations, config = case()
    saved = copy.deepcopy(observations)
    result = align_road_targets(xy, times, valid, observations, config=config, query_times=[0., 7., 20.], provenance={"source": "frozen"})
    assert result["selected"]["route_timestamps"] == [0., 7., 20.]
    assert result["provenance"]["caller_sources"] == {"source": "frozen"}
    assert observations == saved and np.array_equal(xy[:, 0], times)
    assert result["provenance"]["profile_sha256"] and result["provenance"]["motion_valid_sha256"]
    json.dumps(result, allow_nan=False)


def test_input_permutation_ties_are_stable():
    original = run()
    permuted = run(mutate=lambda xy, times, valid, obs: (obs.reverse(), obs[-1]["candidates"].reverse()))
    assert original == permuted


def test_appearance_field_is_used_for_every_branch():
    values = np.zeros((15, 15), np.float32)
    values[7:9, 4:10] = 4.
    result = run(field=RasterField(values, (-400., -400.), 100., -6.))
    assert result["selected"]["appearance"] > 0
    assert result["provenance"]["appearance"]["values_sha256"]
    assert all(branch["best"]["appearance"] > -6 for branch in result["branches"])


def test_target_is_a_soft_advance_hint_not_camera_equality():
    xy, times, valid, observations, config = case()
    data = _prepare(xy, times, valid, observations, times, config)
    target = {"id": "forward", "xy": [1000., 0.]}
    loss, evidence = _loss(np.array([[0., 0., 0., 200.]]), (target, None), data, RoadTargetConfig(), details=True)
    assert evidence[0]["distance_m"] == 950.
    assert evidence[0]["range_mode"] in {"near", "advance"}
    assert evidence[0]["radial_loss"] < 1 and evidence[0]["bearing_loss"] == 0
    assert 0 < loss[0] < 2
    assert "no camera fix" in evidence[0]["constraint"]


def test_backward_hint_penalized_softly_not_excluded():
    xy, times, valid, observations, config = case()
    data = _prepare(xy, times, valid, observations, times, config)
    _, evidence = _loss(np.array([[0., 0., 0., 200.]]), ({"id": "behind", "xy": [-100., 0.]}, None), data, config, details=True)
    assert 0 < evidence[0]["bearing_loss"] <= config.bearing_loss_cap


def test_zero_distance_and_stopped_heading_do_not_invent_bearing():
    xy, times, valid, observations, config = case()
    data = _prepare(xy, times, valid, observations, times, config)
    _, evidence = _loss(np.array([[0., 0., 0., 200.]]), ({"id": "at-camera", "xy": [50., 0.]}, None), data, config, details=True)
    assert evidence[0]["bearing_used"] is False
    xy[1:4] = xy[2]
    observations[0]["t_sec"] = 10.
    data = _prepare(xy, times, valid, observations, times, config)
    assert not data["heading_valid"][0]


def test_weak_motion_is_reported_without_confidence_promotion():
    result = run(mutate=lambda xy, times, valid, obs: valid.__setitem__(slice(1, None), False))
    assert result["motion_quality"]["valid_fraction"] == .2
    assert result["confidence_calibrated"] is False


@pytest.mark.parametrize("change", [
    {"yaw_step_deg": 1e-300}, {"anchor_spacing_m": 1e-300}, {"max_coarse_poses": 1},
    {"max_fine_poses": 1}, {"max_branches": 2},
])
def test_allocation_bounds_refuse_before_search(change):
    with pytest.raises(ValueError, match="budget|exceeds"):
        run(config=replace(case()[-1], **change))


@pytest.mark.parametrize("change", [
    {"null_loss": np.nan}, {"batch_size": 1.5}, {"near_radius_m": -1},
    {"advance_radius_m": 10}, {"refine_xy_m": (1.,)}, {"mean_speed_range_mps": (35., 2.)},
])
def test_invalid_configs_refused(change):
    with pytest.raises(ValueError):
        replace(RoadTargetConfig(), **change)


@pytest.mark.parametrize("change", [
    lambda xy, times, valid, obs: times.__setitem__(1, 0),
    lambda xy, times, valid, obs: xy.__setitem__((0, 0), np.nan),
    lambda xy, times, valid, obs: obs[0].__setitem__("t_sec", -1),
    lambda xy, times, valid, obs: obs[0].__setitem__("confidence", np.nan),
    lambda xy, times, valid, obs: obs[0]["candidates"].append(copy.deepcopy(obs[0]["candidates"][0])),
    lambda xy, times, valid, obs: obs[1]["candidates"][0].update(id="node/a", xy=[3., 4.]),
])
def test_malformed_evidence_refused(change):
    with pytest.raises(ValueError):
        run(mutate=change)


def test_query_rows_cannot_silently_extrapolate():
    xy, times, valid, observations, config = case()
    with pytest.raises(ValueError, match="query times"):
        align_road_targets(xy, times, valid, observations, query_times=[0., 21.], config=config)
    field = RasterField(np.zeros((2, 3, 3)), (0., 0.), 100., -6., np.array([0., 20.]))
    with pytest.raises(ValueError, match="query times"):
        align_road_targets(xy, times, valid, observations, appearance_field=field, config=config)


def test_routes_are_bounded_by_query_budget_and_global_mean_scale():
    result = run()
    for branch in result["branches"]:
        assert 20*5 <= branch["best"]["length_m"] <= 20*10
        assert len(branch["best"]["route_xy"]) <= RoadTargetConfig().max_queries
    assert "stops are allowed" in result["search"]["scale_scope"]


@pytest.mark.parametrize("radius", [1e-12, 1e-100])
def test_tiny_metric_scales_cannot_bypass_integer_grid_budget(radius):
    xy, times, valid, observations, config = case()
    observations = [observations[0]]
    observations[0]["candidates"] = observations[0]["candidates"][:1]
    config = replace(config, near_radius_m=0., advance_radius_m=radius, anchor_spacing_m=radius,
        length_count=1, yaw_step_deg=360., basins_per_branch=1, max_coarse_poses=9,
        refine_xy_m=(0.,), refine_yaw_deg=(0.,), refine_length_factors=(1.,))
    result = align_road_targets(xy, times, valid, observations, config=config)
    assert result["search"]["coarse_poses"] == 5
    assert result["search"]["fine_poses"] == 1
