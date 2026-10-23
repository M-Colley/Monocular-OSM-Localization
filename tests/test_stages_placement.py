"""Evaluation, anchor-primary placement and the final position report."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.stages.evaluation as evaluation
import monocular_osm.stages.placement as placement
from monocular_osm.position import latlon_to_xy, xy_to_latlon
from monocular_osm.stages.context import RunContext

from ._stage_fixtures import UTM32N, make_ctx, prime_ranks


def _line(road, street: int, n: int = 41, dy: float = 0.0, dx: float = 0.0) -> np.ndarray:
    poly = np.asarray(road.polylines[street], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return poly[0] + t * (poly[-1] - poly[0]) + np.array([dx, dy])


def _fine(ctx, dy: float = 0.0):
    """Give every candidate a dense aligned route along its street."""
    for i, c in enumerate(ctx.candidates):
        c.aligned_traj_xy = _line(ctx.road, i)
    return ctx


def _track(ctx, street: int = 0, *, dy: float = 0.0, n: int = 30):
    """A noise-free per-frame VPR track along ``street``, displaced by ``dy`` m."""
    xy = _line(ctx.road, street, n, dy=dy)
    ll = xy_to_latlon(xy, UTM32N)
    return np.arange(n), ll, np.full(n, 0.7), xy


# ---------------------------------------------------------------------------
# evaluate_ground_truth_streets / evaluate_waypoints
# ---------------------------------------------------------------------------


def test_street_ground_truth_marks_the_matching_candidate(tmp_path):
    ctx = make_ctx(tmp_path, ground_truth_streets=("Street 1",))
    evaluation.evaluate_ground_truth_streets(ctx)
    gt = ctx.result["ground_truth"]
    assert gt["streets"] == ["Street 1"] and gt["best_distance_rank"] == 2
    assert [m["gt_on_street"] for m in ctx.result["matches"]] == [False, True, False]
    assert ctx.result["matches"][1]["gt_matching_names"] == ["Street 1"]


def test_no_street_ground_truth_means_no_evaluation(tmp_path):
    ctx = make_ctx(tmp_path)
    evaluation.evaluate_ground_truth_streets(ctx)
    assert "ground_truth" not in ctx.result


def _gt_file(tmp_path, xy: np.ndarray):
    ll = xy_to_latlon(xy, UTM32N)
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"waypoints": [
        {"t_sec": float(i), "lat": float(a), "lon": float(b)} for i, (a, b) in enumerate(ll)]}),
        encoding="utf-8")
    return path


def test_waypoint_errors_are_attached_to_every_candidate(tmp_path):
    ctx = _fine(make_ctx(tmp_path))
    ctx.cfg.ground_truth_waypoints = _gt_file(tmp_path, _line(ctx.road, 0, 12))
    evaluation.evaluate_waypoints(ctx)
    gt = ctx.result["ground_truth_waypoints"]
    assert gt["n_waypoints"] == 12 and gt["best_mean_error_rank"] == 1
    assert gt["final_pick_start_error_m"] == pytest.approx(0.0, abs=0.5)
    assert ctx.result["matches"][0]["gt_waypoint_start_error_m"] == pytest.approx(0.0, abs=0.5)
    assert ctx.result["matches"][2]["gt_waypoint_start_error_m"] > 9000      # 2 streets north
    assert ctx.waypoint_evals is not None and ctx.anchored_eval is None


def test_the_anchored_route_is_evaluated_on_its_own(tmp_path):
    ctx = _fine(make_ctx(tmp_path))
    ctx.cfg.ground_truth_waypoints = _gt_file(tmp_path, _line(ctx.road, 0, 12))
    ctx.anchored_cand = SimpleNamespace(**{**vars(ctx.candidates[0]),
                                           "aligned_traj_xy": _line(ctx.road, 0, dy=60.0)})
    ctx.result["anchored_position"] = {}
    evaluation.evaluate_waypoints(ctx)
    assert ctx.anchored_eval.start_error_m == pytest.approx(60.0, abs=1.0)
    assert ctx.result["anchored_position"]["gt_start_error_m"] == pytest.approx(60.0, abs=1.0)


def test_an_unreadable_waypoint_file_is_reported_not_raised(tmp_path):
    ctx = make_ctx(tmp_path)
    bad = tmp_path / "gt.json"
    bad.write_text("{}", encoding="utf-8")
    ctx.cfg.ground_truth_waypoints = bad
    evaluation.evaluate_waypoints(ctx)
    assert "non-empty 'waypoints' list" in ctx.result["ground_truth_waypoints_error"]
    assert ctx.waypoint_evals is None


# ---------------------------------------------------------------------------
# anchor_primary_placement
# ---------------------------------------------------------------------------


def test_a_vpr_track_pins_the_route_start_to_the_retrieved_start(tmp_path):
    ctx = _fine(make_ctx(tmp_path))
    idx, ll, sims, xy = _track(ctx, 0, dy=250.0)                # truth is 250 m north of the pick
    ctx.vpr_track, ctx.vpr_center, ctx.anchor_origin = (idx, ll, sims), tuple(ll.mean(axis=0)), "vpr"
    placement.anchor_primary_placement(ctx)
    ap = ctx.result["anchored_position"]
    assert ap["source"].startswith("vpr_startpin") and ap["candidate_rank"] == 1
    assert ap["prior_latlon"] == pytest.approx(list(ll.mean(axis=0)))
    # the start was moved onto the retrieved start (within the robust-centre window)
    start_xy = latlon_to_xy(np.array([[ap["lat"], ap["lon"]]]), UTM32N)[0]
    assert np.linalg.norm(start_xy - xy[0]) < 40.0
    assert ctx.anchored_cand is not None and ctx.anchored_cand is not ctx.candidates[0]
    # the matcher's own candidate is untouched
    assert np.allclose(ctx.candidates[0].aligned_traj_xy, _line(ctx.road, 0))


def test_a_vlm_prior_only_translates_the_centroid(tmp_path):
    ctx = _fine(make_ctx(tmp_path))
    cen = np.asarray(ctx.candidates[0].aligned_traj_xy).mean(axis=0)
    prior = xy_to_latlon(np.array([cen + [0.0, 200.0]]), UTM32N)[0]
    ctx.vpr_center, ctx.anchor_origin = (float(prior[0]), float(prior[1])), "vlm"
    placement.anchor_primary_placement(ctx)
    ap = ctx.result["anchored_position"]
    assert ap["source"] == "vlm_geocode"
    placed = np.asarray(ctx.anchored_cand.aligned_traj_xy)
    assert np.allclose(placed.mean(axis=0), cen + [0.0, 200.0], atol=0.5)


def test_without_a_prior_there_is_no_anchored_position(tmp_path):
    ctx = _fine(make_ctx(tmp_path))
    placement.anchor_primary_placement(ctx)
    assert ctx.anchored_cand is None and "anchored_position" not in ctx.result


def test_a_failing_pin_falls_back_to_the_centroid_and_is_recorded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("degenerate track")

    monkeypatch.setattr(placement, "_place_candidate_on_track", boom)
    ctx = _fine(make_ctx(tmp_path))
    idx, ll, sims, _ = _track(ctx, 0, dy=250.0)
    ctx.vpr_track, ctx.vpr_center, ctx.anchor_origin = (idx, ll, sims), tuple(ll.mean(axis=0)), "vpr"
    placement.anchor_primary_placement(ctx)
    assert ctx.result["anchored_position"]["source"] == "vpr_centroid"
    assert ctx.channel_status["start_pin_placement"]["status"] == "failed"


# ---------------------------------------------------------------------------
# final_position
# ---------------------------------------------------------------------------


def _final_ctx(tmp_path, **kw) -> RunContext:
    ctx = _fine(make_ctx(tmp_path, **kw))
    prime_ranks(ctx)
    ctx.ranking_mode = "consensus(shape)"
    ctx.result["ranking_mode"] = ctx.ranking_mode
    return ctx


def test_the_matcher_pick_is_the_headline_when_nothing_else_placed_it(tmp_path):
    ctx = _final_ctx(tmp_path)
    placement.final_position(ctx)
    pos = ctx.result["position"]
    assert pos["source"] == "matcher" and ctx.result["matcher_position"]["source"] == "matcher"
    assert "pinned_position" not in ctx.result
    assert len(pos["hypotheses"]) >= 1 and "spatial_confidence" in pos
    assert all("center_latlon" in m for m in ctx.result["matches"])
    assert "google_maps_url" in pos


def test_the_anchored_route_becomes_the_headline(tmp_path):
    ctx = _final_ctx(tmp_path)
    ctx.vpr_center, ctx.anchor_origin = (48.4, 9.99), "vpr"
    placed = SimpleNamespace(**{**vars(ctx.candidates[0]),
                                "aligned_traj_xy": _line(ctx.road, 0, dy=80.0)})
    ctx.anchored_cand = placed
    placement.final_position(ctx)
    pos = ctx.result["position"]
    assert pos["source"] == "anchor_primary_vpr" and pos["prior_latlon"] == [48.4, 9.99]
    assert ctx.result["matcher_position"]["source"] == "matcher"      # raw pick kept apart
    assert pos["latitude"] != ctx.result["matcher_position"]["latitude"]


def test_scale_locked_anchor_pin_reports_a_pinned_position(tmp_path):
    ctx = _final_ctx(tmp_path)
    ctx.locked_scale = 400.0 / 399.0                                   # m per VO unit
    ctx.match_xz = np.stack([np.arange(0.0, 400.0), np.zeros(400)], axis=1) / ctx.locked_scale
    ctx.match_ts = np.linspace(0.0, 29.0, len(ctx.match_xz))
    lat, lon = xy_to_latlon(_line(ctx.road, 0, 5)[2:3] + [0.0, 30.0], UTM32N)[0]
    ctx.ocr_anchors = [SimpleNamespace(name="Rathaus", lat=float(lat), lon=float(lon),
                                       confidence=0.9, t_sec=14.0)]
    placement.final_position(ctx)
    pinned = ctx.result["pinned_position"]
    assert pinned["source"] == "ocr_anchor_pin"
    assert "anchor-pin(1poi+0street)" in pinned["ranking"]
    assert ctx.result["position"]["source"] == "ocr_anchor_pin"        # headline = pinned here


def test_a_failing_anchor_pin_is_recorded_and_the_matcher_pick_survives(tmp_path):
    ctx = _final_ctx(tmp_path)
    ctx.locked_scale = 1.0
    ctx.match_ts = np.zeros(3)                                         # desynced from match_xz (30)
    ctx.ocr_anchors = [SimpleNamespace(name="x", lat=48.4, lon=9.99, confidence=1.0, t_sec=1.0)]
    placement.final_position(ctx)
    assert ctx.channel_status["anchor_pin"]["status"] == "failed"
    assert "timestamp axis desynced" in ctx.channel_status["anchor_pin"]["error"]
    assert ctx.result["position"]["source"] == "matcher"


def test_final_position_needs_candidates(tmp_path):
    ctx = _final_ctx(tmp_path)
    ctx.candidates = []
    placement.final_position(ctx)
    assert "position" not in ctx.result and "position_error" not in ctx.result
