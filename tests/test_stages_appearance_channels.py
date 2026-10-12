"""The late appearance channels: embedding retrieval, BevSplat fusion, DA3 dense reconstruction.

Each records its outcome in ``ctx.result`` and, when it fails or finds nothing,
says so (``channel_status`` / ``*_error``) instead of silently changing the ranking.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.da3_reconstruction as da3_reconstruction
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
# embedding / BevSplat / DA3
# ---------------------------------------------------------------------------


def test_embedding_retrieval_needs_a_topdown_query(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "score_candidates_by_embeddings",
                        lambda *a, **k: pytest.fail("no query image"))
    ctx = make_ctx(tmp_path, embedding_sources=("osm",))
    channels.embedding_channel(ctx)
    assert "needs IPM or splat" in ctx.result["embedding_retrieval_error"]


def test_embedding_scores_become_per_candidate_ranks(tmp_path, monkeypatch):
    def emb(sim):
        return SimpleNamespace(cosine_similarity=sim, image_path=None, error=None)

    monkeypatch.setattr(channels, "score_candidates_by_embeddings",
                        lambda *a, **k: {"osm": [emb(0.1), emb(0.9), emb(0.5)]})
    ctx = make_ctx(tmp_path, embedding_sources=("osm",))
    ctx.splat_img_rgb = np.zeros((4, 4, 3), np.uint8)
    channels.embedding_channel(ctx)
    assert ctx.result["embedding_query"] == "splat_topdown"
    assert [m["osm_embedding_rank"] for m in ctx.result["matches"]] == [3, 1, 2]


def _bev(score, error=None):
    return SimpleNamespace(score=score, pred_shift_u=0.0, pred_shift_v=0.0, pred_heading=0.0,
                           satellite_path=None, error=error)


def test_bevsplat_appearance_can_reorder_the_shortlist(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "score_candidates_with_bevsplat",
                        lambda *a, **k: [_bev(0.1), _bev(0.2), _bev(0.9)])
    ctx = make_ctx(tmp_path, enable_bev_splat=True, bev_fusion_cap=3)
    for i, m in enumerate(ctx.result["matches"]):
        m["consensus_score"] = 2.0 + 0.2 * i                    # near-tie, geometry slightly ahead
    ctx.ranking_mode = "consensus(shape)"
    channels.bevsplat_channel(ctx)
    assert ctx.result["bev_splat"]["fused"] is True
    assert ctx.result["matches"][0]["street_names"] == ["Street 2"]
    assert ctx.candidates[0].start_node == "A2"                 # candidates reordered in step
    assert ctx.ranking_mode == "consensus(shape)+bev" == ctx.result["ranking_mode"]


def test_bevsplat_with_mostly_failed_tiles_is_not_fused(tmp_path, monkeypatch):
    monkeypatch.setattr(channels, "score_candidates_with_bevsplat",
                        lambda *a, **k: [_bev(0.0, "no tile"), _bev(0.0, "no tile"), _bev(0.9)])
    ctx = make_ctx(tmp_path, enable_bev_splat=True)
    before = list(ctx.candidates)
    channels.bevsplat_channel(ctx)
    assert ctx.result["bev_splat"]["fused"] is False and ctx.candidates == before


def test_da3_dense_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(da3_reconstruction, "reconstruct_with_da3", boom)
    ctx = make_ctx(tmp_path, enable_da3=True)
    ctx.traj = SimpleNamespace(valid=np.ones(30, bool))
    channels.da3_dense_channel(ctx)
    assert ctx.result["da3_error"] == "no CUDA"
    assert ctx.channel_status["da3"]["status"] == "failed"


def test_da3_dense_reuses_the_trajectory_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setattr(da3_reconstruction, "reconstruct_with_da3",
                        lambda *a, **k: pytest.fail("must reuse ctx.da3_rec"))
    for name in ("save_ply", "save_interactive_html", "render_topdown_to_file"):
        monkeypatch.setattr(channels, name, lambda *a, **k: None)
    rec = SimpleNamespace(points_world=np.zeros((9, 3)), colors_rgb=np.zeros((9, 3)),
                          extrinsics_w2c=np.zeros((5, 3, 4)))
    ctx = make_ctx(tmp_path, enable_da3=True)
    ctx.da3_rec = rec
    ctx.traj = SimpleNamespace(valid=np.ones(30, bool))
    channels.da3_dense_channel(ctx)
    assert ctx.result["da3"]["n_points"] == 9 and ctx.result["da3"]["n_keyframes"] == 5
