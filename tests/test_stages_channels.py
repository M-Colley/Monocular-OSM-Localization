"""The optional channel stages: each scores/renders the candidates, records its
outcome in ``ctx.result`` and leaves a neutral (uniform-rank, zero-weight) result
when it declines, so the consensus can always run."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.osm_buildings3d as osm_buildings3d
import monocular_osm.stages.channels as channels
import monocular_osm.tile3d_match as tile3d_match
from monocular_osm.aerial_match import AerialMatchResult
from monocular_osm.stages.context import RunContext
from monocular_osm.trajectory_matching import SlidingWindowMatchResult

from ._stage_fixtures import make_ctx


def _no_candidates(tmp_path, **kw) -> RunContext:
    ctx = make_ctx(tmp_path, **kw)
    ctx.candidates, ctx.result = [], {"city": "Ulm", "matches": []}
    return ctx


# ---------------------------------------------------------------------------
# sliding window
# ---------------------------------------------------------------------------


def _sliding(i, support, score):
    return SlidingWindowMatchResult(
        candidate_index=i, n_windows=4, support_count=support, support_ratio=support / 4,
        mean_rank=1.0 + i, mean_score_rms_m=10.0, sliding_score=score)


def test_sliding_window_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "score_candidates_with_sliding_windows",
                        lambda *a, **k: pytest.fail("not enabled"))
    ctx = make_ctx(tmp_path)
    channels.sliding_window_channel(ctx)
    assert "sliding_window" not in ctx.result


def test_sliding_window_support_is_ranked_and_recorded(tmp_path, monkeypatch):
    seen = {}

    def fake(xz, road, cands, **kw):
        seen.update(kw)
        return [_sliding(0, 1, 0.2), _sliding(1, 4, 0.9), _sliding(2, 2, 0.5)]

    monkeypatch.setattr(channels, "score_candidates_with_sliding_windows", fake)
    ctx = make_ctx(tmp_path, enable_sliding_window=True, sliding_window_size=50,
                   sliding_window_step=25, top_k=2)
    ctx.estimated_length_m = 800.0
    channels.sliding_window_channel(ctx)
    assert ctx.result["sliding_window"] == {"window_size": 50, "step": 25, "n_windows": 4}
    assert [m["sliding_window_rank"] for m in ctx.result["matches"]] == [3, 1, 2]
    assert ctx.result["matches"][1]["sliding_window_support_count"] == 4
    assert seen["window_size"] == 50 and seen["step"] == 25
    assert seen["window_top_k"] == 3 and seen["estimated_length_m"] == 800.0


def test_sliding_window_without_a_usable_result_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "score_candidates_with_sliding_windows", lambda *a, **k: [])
    ctx = make_ctx(tmp_path, enable_sliding_window=True)
    ctx.estimated_length_m = 800.0
    channels.sliding_window_channel(ctx)
    assert "sliding_window" not in ctx.result


# ---------------------------------------------------------------------------
# splat / ipm
# ---------------------------------------------------------------------------


def _patch_splat(monkeypatch, n_points=5):
    pts = np.zeros((n_points, 3))
    monkeypatch.setattr(channels, "build_splat_points", lambda frames, traj, K, max_pairs: (pts, pts))
    written = []
    monkeypatch.setattr(channels, "save_ply", lambda p, c, path: written.append(path))
    monkeypatch.setattr(channels, "save_interactive_html", lambda p, c, path: written.append(path))
    monkeypatch.setattr(channels, "render_topdown_to_file", lambda p, c, path: written.append(path))
    monkeypatch.setattr(channels, "render_topdown_splat", lambda p, c: np.full((4, 4, 3), 7, np.uint8))
    return written


def test_splat_produces_the_topdown_image_for_the_aerial_channel(tmp_path, monkeypatch):
    written = _patch_splat(monkeypatch)
    ctx = make_ctx(tmp_path)
    channels.splat_channel(ctx)
    assert ctx.splat_img_rgb.shape == (4, 4, 3)
    assert [p.name for p in written] == ["splat.ply", "splat.html", "splat_topdown.png"]


def test_splat_with_no_triangulated_points_leaves_no_image(tmp_path, monkeypatch):
    _patch_splat(monkeypatch, n_points=0)
    ctx = make_ctx(tmp_path)
    channels.splat_channel(ctx)
    assert ctx.splat_img_rgb is None


def test_splat_needs_candidates_and_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "build_splat_points",
                        lambda *a, **k: pytest.fail("must not run"))
    channels.splat_channel(make_ctx(tmp_path, enable_splat=False))
    channels.splat_channel(_no_candidates(tmp_path))


def test_ipm_writes_the_stitched_canvas_and_records_it(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "render_ipm_canvas",
                        lambda frames, xz, K, **kw: np.full((6, 8, 3), 100, np.uint8))
    ctx = make_ctx(tmp_path, enable_ipm=True)
    ctx.traj = SimpleNamespace(xz=np.zeros((30, 2)))
    channels.ipm_channel(ctx)
    assert ctx.ipm_canvas.shape == (6, 8, 3)
    assert ctx.result["ipm"] == "ipm_bev.png" and (ctx.cfg.output_dir / "ipm_bev.png").exists()


def test_an_ipm_failure_is_recorded_and_leaves_no_canvas(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("homography failed")

    monkeypatch.setattr(channels, "render_ipm_canvas", boom)
    ctx = make_ctx(tmp_path, enable_ipm=True)
    ctx.traj = SimpleNamespace(xz=np.zeros((30, 2)))
    channels.ipm_channel(ctx)
    assert ctx.ipm_canvas is None and ctx.result["ipm_error"] == "homography failed"
    assert ctx.channel_status["ipm"]["status"] == "failed"


def test_ipm_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "render_ipm_canvas", lambda *a, **k: pytest.fail("off"))
    ctx = make_ctx(tmp_path)
    channels.ipm_channel(ctx)
    assert ctx.ipm_canvas is None and "ipm" not in ctx.result


# ---------------------------------------------------------------------------
# aerial
# ---------------------------------------------------------------------------


def _aerial(i, score):
    return AerialMatchResult(candidate_index=i, traj_iou=0.1 * i, traj_coverage=0.5,
                             aerial_score=score, n_orb_matches=0, n_inliers=0,
                             inlier_ratio=0.0, osm_render_path=None)


def test_aerial_disabled_leaves_a_neutral_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "match_splat_against_candidates",
                        lambda *a, **k: pytest.fail("disabled"))
    ctx = make_ctx(tmp_path, enable_aerial_match=False)
    channels.aerial_channel(ctx)
    assert ctx.aerial_rank == {0: 1, 1: 2, 2: 3} and ctx.w_aerial == 0.0
    assert ctx.ranking_mode == "shape"


def test_aerial_ranks_candidates_by_descending_score(tmp_path, monkeypatch):
    seen = {}

    def fake(image, road, cands, output_dir):
        seen["image"] = image
        return [_aerial(0, 0.1), _aerial(1, 0.9), _aerial(2, 0.5)]

    monkeypatch.setattr(channels, "match_splat_against_candidates", fake)
    ctx = make_ctx(tmp_path)
    channels.aerial_channel(ctx)
    assert ctx.aerial_rank == {0: 3, 1: 1, 2: 2}
    assert ctx.w_aerial == ctx.fusion_w.aerial
    assert ctx.result["aerial_input"] == "traj_iou_only" and seen["image"] is None
    assert [m["aerial_rank"] for m in ctx.result["matches"]] == [3, 1, 2]
    assert ctx.result["matches"][1]["aerial_score"] == 0.9


def test_aerial_prefers_the_ipm_canvas_over_the_splat_image(tmp_path, monkeypatch):
    seen = {}

    def fake(image, *a, **k):
        seen["image"] = image
        return [_aerial(i, 0.5) for i in range(3)]

    monkeypatch.setattr(channels, "match_splat_against_candidates", fake)
    ctx = make_ctx(tmp_path)
    ctx.ipm_canvas = np.zeros((4, 4, 3), np.uint8)
    ctx.ipm_canvas[..., 0] = 255                                # blue in BGR
    ctx.splat_img_rgb = np.zeros((4, 4, 3), np.uint8)
    channels.aerial_channel(ctx)
    assert ctx.result["aerial_input"] == "ipm_bev"
    assert seen["image"][0, 0].tolist() == [0, 0, 255]          # converted to RGB


def test_aerial_falls_back_to_the_splat_image(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "match_splat_against_candidates",
                        lambda *a, **k: [_aerial(i, 0.5) for i in range(3)])
    ctx = make_ctx(tmp_path)
    ctx.splat_img_rgb = np.zeros((4, 4, 3), np.uint8)
    channels.aerial_channel(ctx)
    assert ctx.result["aerial_input"] == "splat_topdown"


def test_aerial_without_candidates_only_resets_the_ranking_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "match_splat_against_candidates",
                        lambda *a, **k: pytest.fail("no candidates"))
    ctx = _no_candidates(tmp_path)
    ctx.ranking_mode = "stale"
    channels.aerial_channel(ctx)
    assert ctx.ranking_mode == "shape" and ctx.aerial_rank == {}


# ---------------------------------------------------------------------------
# tile3d
# ---------------------------------------------------------------------------


def test_tile3d_is_off_by_default_and_leaves_a_neutral_channel(tmp_path):
    ctx = make_ctx(tmp_path)
    channels.tile3d_channel(ctx)
    assert ctx.tile3d_rank == {0: 1, 1: 2, 2: 3} and ctx.w_tile3d == 0.0
    assert ctx.t3_mesh_ref is None and "tile3d" not in ctx.result


def _mesh(n_triangles=10):
    return SimpleNamespace(triangles=list(range(n_triangles)), provider="osm", n_buildings=4,
                           building_ground=np.zeros((1, 2)))


def _t3_result(score, err=2.0):
    return SimpleNamespace(tile3d_score=score, skyline_err_deg=err, pitch_offset_deg=0.5,
                           coverage=0.9, n_informative=7)


def test_an_empty_mesh_leaves_the_channel_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(osm_buildings3d, "fetch_osm_building_mesh", lambda *a, **k: _mesh(0))
    ctx = make_ctx(tmp_path, use_tile3d=True, tile3d_source="osm")
    channels.tile3d_channel(ctx)
    assert ctx.w_tile3d == 0.0 and "tile3d" not in ctx.result
    assert ctx.channel_status == {}                             # inactive is not a failure


def test_a_failing_tile3d_fetch_is_recorded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("overpass unreachable")

    monkeypatch.setattr(osm_buildings3d, "fetch_osm_building_mesh", boom)
    ctx = make_ctx(tmp_path, use_tile3d=True, tile3d_source="osm")
    channels.tile3d_channel(ctx)
    assert ctx.w_tile3d == 0.0
    assert ctx.channel_status["tile3d"]["status"] == "failed"


def test_tile3d_publishes_ranks_weight_and_the_refinement_inputs(tmp_path, monkeypatch):
    mesh = _mesh()
    monkeypatch.setattr(osm_buildings3d, "fetch_osm_building_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(tile3d_match, "walk_coverage_fractions",
                        lambda ground, walks: [1.0] * len(walks))
    monkeypatch.setattr(tile3d_match, "match_tiles3d_against_candidates",
                        lambda *a, **k: [_t3_result(0.2), _t3_result(0.8), _t3_result(0.5)])
    ctx = make_ctx(tmp_path, use_tile3d=True, tile3d_source="osm", tile3d_samples=4)
    channels.tile3d_channel(ctx)
    assert ctx.tile3d_rank == {0: 3, 1: 1, 2: 2}
    assert ctx.w_tile3d == ctx.fusion_w.tile3d
    assert ctx.t3_mesh_ref is mesh and len(ctx.t3_samples_ref) == 4
    assert ctx.result["tile3d"]["provider"] == "osm"
    assert ctx.result["matches"][1]["tile3d_rank"] == 1


def test_tile3d_with_a_coverage_gap_stays_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(osm_buildings3d, "fetch_osm_building_mesh", lambda *a, **k: _mesh())
    monkeypatch.setattr(tile3d_match, "walk_coverage_fractions",
                        lambda ground, walks: [1.0, 0.4, 1.0])
    monkeypatch.setattr(tile3d_match, "match_tiles3d_against_candidates",
                        lambda *a, **k: pytest.fail("a capped fetch must not score"))
    ctx = make_ctx(tmp_path, use_tile3d=True, tile3d_source="osm")
    channels.tile3d_channel(ctx)
    assert ctx.w_tile3d == 0.0 and ctx.tile3d_rank == {0: 1, 1: 2, 2: 3}


def test_tile3d_that_matched_nothing_keeps_the_uniform_rank(tmp_path, monkeypatch):
    monkeypatch.setattr(osm_buildings3d, "fetch_osm_building_mesh", lambda *a, **k: _mesh())
    monkeypatch.setattr(tile3d_match, "walk_coverage_fractions",
                        lambda ground, walks: [1.0] * len(walks))
    monkeypatch.setattr(tile3d_match, "match_tiles3d_against_candidates",
                        lambda *a, **k: [_t3_result(0.0)] * 3)
    ctx = make_ctx(tmp_path, use_tile3d=True, tile3d_source="osm")
    channels.tile3d_channel(ctx)
    assert ctx.w_tile3d == 0.0 and ctx.t3_mesh_ref is None
