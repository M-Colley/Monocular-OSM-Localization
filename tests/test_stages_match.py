"""The matching stages (scale lock, VGGT gate, candidate matching, result summary)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.stages.match as stage_match
import monocular_osm.vggt_trajectory as vggt_trajectory
from monocular_osm.position import xy_to_latlon

from ._stage_fixtures import make_cfg, make_ctx, make_road
from monocular_osm.stages.context import RunContext


def _bare_ctx(tmp_path, **kw) -> RunContext:
    ctx = RunContext.create(make_cfg(tmp_path, **kw))
    ctx.road = make_road()
    ctx.match_xz = np.stack([np.linspace(0.0, 100.0, 21), np.zeros(21)], axis=1)
    ctx.estimated_length_m = 500.0
    return ctx


# ---------------------------------------------------------------------------
# lock_alignment_scale
# ---------------------------------------------------------------------------


def test_scale_lock_is_metres_per_vo_unit(tmp_path):
    ctx = _bare_ctx(tmp_path)
    stage_match.lock_alignment_scale(ctx)
    assert ctx.locked_scale == pytest.approx(5.0)          # 500 m over 100 VO units


def test_scale_lock_can_be_disabled(tmp_path):
    ctx = _bare_ctx(tmp_path, scale_lock=False)
    stage_match.lock_alignment_scale(ctx)
    assert ctx.locked_scale is None


def test_scale_lock_declines_a_motionless_trajectory(tmp_path):
    ctx = _bare_ctx(tmp_path)
    ctx.match_xz = np.zeros((10, 2))
    stage_match.lock_alignment_scale(ctx)
    assert ctx.locked_scale is None


# ---------------------------------------------------------------------------
# vggt_gate
# ---------------------------------------------------------------------------


def test_vggt_gate_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(vggt_trajectory, "vggt_camera_trajectory",
                        lambda *a, **k: pytest.fail("not enabled"))
    ctx = _bare_ctx(tmp_path)
    stage_match.vggt_gate(ctx)
    assert ctx.vggt_seed == [] and ctx.result_vggt is None


def test_vggt_gate_seeds_the_start_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(vggt_trajectory, "vggt_camera_trajectory",
                        lambda frames, n_keyframes: np.zeros((6, 2)))
    monkeypatch.setattr(vggt_trajectory, "vggt_seed_nodes",
                        lambda xy, road, estimated_length_m: ["A0", "A1"])
    ctx = _bare_ctx(tmp_path, use_vggt_gating=True)
    ctx.frames = SimpleNamespace(frames=[0, 1, 2])
    stage_match.vggt_gate(ctx)
    assert ctx.vggt_seed == ["A0", "A1"]
    assert ctx.result_vggt == {"poses": 6, "seed_nodes": 2}


def test_vggt_gate_without_the_package_is_a_quiet_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(vggt_trajectory, "vggt_camera_trajectory", lambda *a, **k: None)
    ctx = _bare_ctx(tmp_path, use_vggt_gating=True)
    ctx.frames = SimpleNamespace(frames=[0])
    stage_match.vggt_gate(ctx)
    assert ctx.vggt_seed == [] and ctx.result_vggt is None and ctx.channel_status == {}


def test_a_failing_vggt_gate_is_recorded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("cuda oom")

    monkeypatch.setattr(vggt_trajectory, "vggt_camera_trajectory", boom)
    ctx = _bare_ctx(tmp_path, use_vggt_gating=True)
    ctx.frames = SimpleNamespace(frames=[0])
    stage_match.vggt_gate(ctx)
    assert ctx.channel_status["vggt"]["status"] == "failed"


# ---------------------------------------------------------------------------
# match_candidates
# ---------------------------------------------------------------------------


def _fake_matcher(pool, calls, *, gated_hits=True):
    def fake(match_xz, road, **kw):
        calls.append(kw)
        if kw.get("restrict_to_start_nodes") and not gated_hits:
            return []
        return list(pool)
    return fake


def _pool(n):
    return [SimpleNamespace(score=float(i), aligned_traj_xy=np.zeros((4, 2)), tag=i)
            for i in range(n)]


def test_the_geometric_pool_is_kept_and_the_headline_runs_on_top_k(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage_match, "match_trajectory", _fake_matcher(_pool(60), calls))
    ctx = _bare_ctx(tmp_path, top_k=7)
    ctx.locked_scale = 5.0
    stage_match.match_candidates(ctx)
    assert [c.tag for c in ctx.candidates] == list(range(7))
    assert len(ctx.geom_pool) == 60 and ctx.geom_pool[:7] == ctx.candidates
    assert len(calls) == 1
    assert calls[0]["final_top_k"] == 50                    # max(top_k, hypothesis pool)
    assert calls[0]["locked_scale"] == 5.0 and calls[0]["restrict_to_start_nodes"] is False
    assert calls[0]["extra_start_nodes"] is None
    assert ctx.anchor_gated is False and ctx.vpr_scale_applied is None


def test_ocr_anchor_seeds_gate_enumeration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage_match, "match_trajectory", _fake_matcher(_pool(12), calls))
    ctx = _bare_ctx(tmp_path, enable_ocr_anchor=True)
    ctx.anchor_seed = ["A1", "A2"]
    ctx.ocr_anchors = [object()]
    stage_match.match_candidates(ctx)
    assert ctx.anchor_gated is True
    assert calls[0]["restrict_to_start_nodes"] is True
    assert calls[0]["extra_start_nodes"] == ["A1", "A2"]


def test_a_vggt_seed_alone_also_gates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage_match, "match_trajectory", _fake_matcher(_pool(12), calls))
    ctx = _bare_ctx(tmp_path)
    ctx.vggt_seed = ["A0"]
    stage_match.match_candidates(ctx)
    assert ctx.anchor_gated is True and calls[0]["extra_start_nodes"] == ["A0"]


def test_an_empty_gated_match_falls_back_to_the_full_scan(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage_match, "match_trajectory",
                        _fake_matcher(_pool(12), calls, gated_hits=False))
    ctx = _bare_ctx(tmp_path)
    ctx.vggt_seed = ["A0"]
    stage_match.match_candidates(ctx)
    assert len(calls) == 2
    assert calls[0]["restrict_to_start_nodes"] is True
    assert "restrict_to_start_nodes" not in calls[1]        # the unrestricted scan
    assert len(ctx.candidates) == 10


def test_no_matches_leaves_empty_candidate_lists(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_match, "match_trajectory", lambda *a, **k: [])
    ctx = _bare_ctx(tmp_path)
    stage_match.match_candidates(ctx)
    assert ctx.candidates == [] and ctx.geom_pool == []


def _two_pass_ctx(tmp_path, monkeypatch, base_dist, alt_dist):
    base, alt = _pool(5), _pool(5)

    def fake_match(xz, road, *, locked_scale, estimated_length_m, **kw):
        return alt if locked_scale == 99.0 else base

    monkeypatch.setattr(stage_match, "match_trajectory", fake_match)
    monkeypatch.setattr(stage_match, "_vpr_alt_scale",
                        lambda *a, **k: (99.0, 800.0, 1.7))
    ids = {id(c): base_dist for c in base} | {id(c): alt_dist for c in alt}
    monkeypatch.setattr(stage_match, "_vpr_sequence_median_m",
                        lambda traj, *a: ids[id(next(c for c in base + alt
                                                     if c.aligned_traj_xy is traj))])
    ctx = _bare_ctx(tmp_path, vpr_two_pass_scale=True)
    ctx.locked_scale = 5.0
    ctx.vpr_center = (48.4, 9.99)
    ll = xy_to_latlon(np.asarray([[570000.0, 5360000.0]]), "EPSG:32632")
    ctx.vpr_track = (np.arange(3), np.repeat(ll, 3, axis=0), np.ones(3))
    ctx.frames = SimpleNamespace(frames=[0] * 10)
    return ctx, base, alt


def test_two_pass_scale_adopts_the_alternate_when_it_explains_the_track_better(
        tmp_path, monkeypatch):
    ctx, base, alt = _two_pass_ctx(tmp_path, monkeypatch, base_dist=300.0, alt_dist=100.0)
    stage_match.match_candidates(ctx)
    assert ctx.candidates[0] in alt
    assert ctx.locked_scale == 99.0 and ctx.estimated_length_m == 800.0
    assert ctx.vpr_scale_applied == 1.7


def test_two_pass_scale_keeps_the_base_without_a_clear_win(tmp_path, monkeypatch):
    ctx, base, alt = _two_pass_ctx(tmp_path, monkeypatch, base_dist=300.0, alt_dist=280.0)
    stage_match.match_candidates(ctx)                       # 280 is not < 0.9 * 300
    assert ctx.candidates[0] in base
    assert ctx.locked_scale == 5.0 and ctx.vpr_scale_applied is None


# ---------------------------------------------------------------------------
# summarise_matches
# ---------------------------------------------------------------------------


def test_the_result_summarises_every_candidate(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.result = {}
    ctx.n_valid, ctx.estimated_length_m, ctx.result_traj_source = 27, 1234.5, "openvo"
    stage_match.summarise_matches(ctx)
    r = ctx.result
    assert r["city"] == "Ulm, Germany" and r["n_frames"] == 30 and r["n_valid_poses"] == 27
    assert r["estimated_length_m"] == 1234.5 and r["trajectory_source"] == "openvo"
    assert [m["street_names"] for m in r["matches"]] == [["Street 0"], ["Street 1"], ["Street 2"]]
    assert (ctx.cfg.output_dir / "match.png").exists()
    assert "vpr_scale_correction" not in r and "ocr_anchors" not in r


def test_the_result_records_a_vpr_scale_correction_and_the_anchors(tmp_path):
    ctx = make_ctx(tmp_path, enable_ocr_anchor=True)
    ctx.result = {}
    ctx.n_valid, ctx.estimated_length_m, ctx.result_traj_source = 27, 1000.0, "vo"
    ctx.vpr_scale_applied = 1.234
    ctx.ocr_anchor_error = "geocoder down"
    stage_match.summarise_matches(ctx)
    assert ctx.result["vpr_scale_correction"] == 1.234
    assert ctx.result["ocr_anchors"] == [] and ctx.result["ocr_street_anchors"] == []
    assert ctx.result["ocr_anchor_error"] == "geocoder down"


def test_a_run_without_matches_still_gets_a_minimal_result(tmp_path):
    ctx = make_ctx(tmp_path)
    ctx.candidates, ctx.result = [], {}
    stage_match.summarise_matches(ctx)
    assert ctx.result == {"city": "Ulm, Germany", "matches": []}
