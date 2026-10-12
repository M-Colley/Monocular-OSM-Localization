"""The consensus-fusion stage on synthetic candidates.

Three candidates sit on three east-west streets 5 km apart, ranked by shape
(candidate 0 best). Each channel is then given a prior that favours candidate 1,
and the test checks that the consensus reorders ``ctx.candidates`` and
``result['matches']`` together and reports which channels contributed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import monocular_osm.stages.consensus as consensus
from monocular_osm.position import xy_to_latlon
from monocular_osm.stages.context import RunContext

from ._stage_fixtures import make_ctx, prime_ranks


def _ctx(tmp_path, **kw) -> RunContext:
    ctx = make_ctx(tmp_path, **kw)
    prime_ranks(ctx)
    return ctx


def _centroid_latlon(c):
    cen = np.asarray(c.aligned_traj_xy, dtype=np.float64).mean(axis=0)
    la, lo = xy_to_latlon(cen[None, :], "EPSG:32632")[0]
    return float(la), float(lo)


def _order(ctx):
    return [m["street_names"][0] for m in ctx.result["matches"]]


def test_without_any_prior_the_shape_order_stands(tmp_path):
    ctx = _ctx(tmp_path)
    before = list(ctx.candidates)
    consensus.fuse_channels(ctx)
    assert ctx.candidates == before
    assert _order(ctx) == ["Street 0", "Street 1", "Street 2"]
    assert ctx.ranking_mode == "shape" and ctx.result["ranking_mode"] == "shape"
    assert [m["final_rank"] for m in ctx.result["matches"]] == [1, 2, 3]
    assert [m["shape_rank"] for m in ctx.result["matches"]] == [1, 2, 3]
    assert all("consensus_score" in m for m in ctx.result["matches"])


def test_a_vpr_prior_demotes_a_candidate_kilometres_away(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.vpr_center = _centroid_latlon(ctx.candidates[1])        # the prior sits on candidate 1
    ctx.anchor_origin = "vpr"
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 1"
    assert ctx.candidates[0].start_node == "A1"                 # candidates reordered in step
    assert ctx.ranking_mode == "consensus(shape+vpr)"
    assert ctx.result["ranking_mode"] == ctx.ranking_mode


def test_the_vpr_prior_is_free_inside_its_own_error_bar(tmp_path):
    ctx = _ctx(tmp_path)
    la, lo = _centroid_latlon(ctx.candidates[0])
    ctx.vpr_center = (la + 0.0005, lo)                          # ~55 m off candidate 0
    ctx.anchor_origin = "vpr"
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 0"                         # no penalty: the pick is kept


def test_a_vlm_prior_gets_the_wider_district_free_radius(tmp_path):
    ctx = _ctx(tmp_path)
    la, lo = _centroid_latlon(ctx.candidates[0])
    ctx.vpr_center = (la + 0.012, lo)                           # ~1.3 km: inside 1.5 km, outside 150 m
    ctx.anchor_origin = "vlm"
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 0"
    ctx2 = _ctx(tmp_path)
    ctx2.vpr_center, ctx2.anchor_origin = (la + 0.012, lo), "vpr"
    consensus.fuse_channels(ctx2)
    assert ctx2.result["matches"][0]["consensus_score"] != ctx.result["matches"][0]["consensus_score"] \
        or _order(ctx2) != _order(ctx)


def test_a_plate_district_prior_demotes_candidates_outside_it(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.plate_center = _centroid_latlon(ctx.candidates[1])
    ctx.result_plate_anchor = {"margin": 2.0}
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 1"
    assert ctx.result_plate_anchor["applied"] is True
    assert "plate" in ctx.ranking_mode


def test_a_confident_sun_heading_demotes_a_contradicting_route(tmp_path):
    ctx = _ctx(tmp_path)
    # candidates run due east (bearing 90); a sun heading of 270 contradicts all of
    # them equally, so make candidate 0 head WEST instead and the sun say east
    c0 = ctx.candidates[0]
    c0.aligned_traj_xy = np.asarray(c0.aligned_traj_xy)[::-1].copy()
    ctx.result_sun_heading = {"available": True, "median_heading": 90.0, "confidence": 0.9}
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 1"
    assert ctx.result["matches"][-1]["sun_heading_diff_deg"] == pytest.approx(180.0, abs=1.0) \
        or any(m.get("sun_heading_diff_deg", 0) > 150 for m in ctx.result["matches"])
    assert "sun" in ctx.ranking_mode


def test_a_low_confidence_sun_heading_is_ignored(tmp_path):
    ctx = _ctx(tmp_path)
    c0 = ctx.candidates[0]
    c0.aligned_traj_xy = np.asarray(c0.aligned_traj_xy)[::-1].copy()
    ctx.result_sun_heading = {"available": True, "median_heading": 90.0, "confidence": 0.3}
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 0" and ctx.ranking_mode == "shape"


def test_ocr_anchors_pull_the_pick_toward_the_anchored_street(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.anchor_xy = np.asarray(ctx.candidates[1].walk_xy)[:1] + np.array([[20.0, 5.0]])
    ctx.anchor_gated = False                                    # un-gated: strong weight
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 1"
    assert ctx.result["matches"][0]["anchor_rank"] == 1
    assert "anchor" in ctx.ranking_mode


def test_gated_enumeration_spends_the_anchors_so_shape_keeps_the_lead(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.anchor_xy = np.asarray(ctx.candidates[1].walk_xy)[:1] + np.array([[20.0, 5.0]])
    ctx.anchor_gated = True                                     # weight 0.25, not 2.0
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 0"


def test_an_ablated_channel_weight_changes_the_outcome(tmp_path):
    ctx = _ctx(tmp_path, fusion_weights={"anchor_ungated": 0.0})
    ctx.anchor_xy = np.asarray(ctx.candidates[1].walk_xy)[:1] + np.array([[20.0, 5.0]])
    consensus.fuse_channels(ctx)
    assert _order(ctx)[0] == "Street 0"                         # the anchor channel is muted


def test_the_feature_table_is_persisted_in_matcher_order(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.vpr_center = _centroid_latlon(ctx.candidates[1])
    ctx.anchor_origin = "vpr"
    ctx.channel_status["vpr"] = {"status": "ok"}
    consensus.fuse_channels(ctx)
    table = json.loads((ctx.cfg.output_dir / "fusion_features.json").read_text(encoding="utf-8"))
    assert ctx.result["fusion"]["features_file"] == "fusion_features.json"
    assert set(ctx.result["fusion"]["weights_effective"]) >= {"shape", "anchor", "aerial"}
    # matcher order, not consensus order: candidate 0 (worst vpr penalty) comes first
    feats = table["candidates"]
    assert [f["matcher_index"] for f in feats] == [0, 1, 2]
    assert feats[0]["vpr_penalty"] > feats[1]["vpr_penalty"]
    assert all(len(f["route_latlon"]) >= 2 for f in feats)
    assert table["channels"]["vpr"]["status"] == "ok"             # what ran, for offline ablation


def test_an_unwritable_feature_table_does_not_stop_the_fusion(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(consensus, "write_feature_table", boom)
    ctx = _ctx(tmp_path)
    consensus.fuse_channels(ctx)
    assert "fusion feature table not written" in capsys.readouterr().out
    assert _order(ctx) == ["Street 0", "Street 1", "Street 2"]


def test_without_candidates_the_stage_is_a_noop(tmp_path):
    ctx = RunContext.create(make_ctx(tmp_path).cfg)
    ctx.result = {"city": "Ulm", "matches": []}
    consensus.fuse_channels(ctx)
    assert ctx.result == {"city": "Ulm", "matches": []} and ctx.ranking_mode == "shape"


def test_turn_channel_ranks_are_stored_but_not_fused(tmp_path):
    ctx = _ctx(tmp_path)
    consensus.fuse_channels(ctx)
    assert all("turn_match_rank" in m for m in ctx.result["matches"])
    assert math.isclose(ctx.fusion_w.turn, 0.0)                 # diagnostics only by default
