"""Retrieval answers survive geometry failures and frames are always released."""
import json

import numpy as np
import pytest

import monocular_osm.pipeline as pipeline
from monocular_osm.frame_extraction import ExtractedFrames, FrameSequence


def _setup(tmp_path, monkeypatch, **kwargs):
    cfg = pipeline.PipelineConfig(url="local", city="Ulm, Germany",
                                  data_dir=tmp_path / "d", output_dir=tmp_path / "o",
                                  **kwargs)
    frames = FrameSequence()
    for _ in range(3):
        frames.append(np.zeros((8, 8, 3), np.uint8))
    extracted = ExtractedFrames(frames, [0., 1., 2.], 1., 3)
    monkeypatch.setattr(pipeline, "acquire_video", lambda ctx: None)
    monkeypatch.setattr(pipeline, "extract_video_frames",
                        lambda ctx, **kw: setattr(ctx, "frames", extracted))
    monkeypatch.setattr(pipeline, "estimate_length_prior", lambda ctx: None)
    monkeypatch.setattr(pipeline, "estimate_flow_scale", lambda ctx: None)
    monkeypatch.setattr(pipeline, "coarse_seed_from_frames", lambda ctx: None)
    monkeypatch.setattr(pipeline, "plate_district_prior", lambda ctx: None)
    monkeypatch.setattr(pipeline, "vlm_district_prior", lambda ctx: None)

    def retrieve(ctx):
        ctx.vpr_center = (48.4, 10.)
        ctx.vpr_source_used = "mapillary"
        ctx.anchor_origin = "vpr"
        ctx.vpr_track = (np.arange(3), np.array([[48.4, 10.], [48.401, 10.001],
                                               [48.402, 10.002]]), np.full(3, 0.25))
    monkeypatch.setattr(pipeline, "vpr_coarse_prior", retrieve)
    return cfg, frames


def test_retrieval_only_needs_no_vo_or_graph(tmp_path, monkeypatch):
    cfg, frames = _setup(tmp_path, monkeypatch, retrieval_only=True)
    monkeypatch.setattr(pipeline, "run_visual_odometry", lambda ctx: pytest.fail("VO ran"))
    result = pipeline.run_pipeline(cfg)
    saved = json.loads((cfg.output_dir / "result.json").read_text())
    assert result["answer"]["method"] == "vpr_track_only"
    assert saved["answer"] == result["answer"]
    assert result["execution"]["geometry"] == "skipped"
    assert frames._closed


def test_geometry_failure_keeps_checkpointed_answer(tmp_path, monkeypatch):
    cfg, frames = _setup(tmp_path, monkeypatch)

    def fail(ctx):
        checkpoint = json.loads((cfg.output_dir / "result.json").read_text())
        assert checkpoint["answer"]["route_latlon"]
        raise RuntimeError("Overpass offline")
    monkeypatch.setattr(pipeline, "_run_geometry_stages", fail)
    result = pipeline.run_pipeline(cfg)
    assert result["answer"]["route_latlon"]
    assert result["channels"]["geometry"]["status"] == "failed"
    assert result["execution"]["geometry"] == "failed"
    assert frames._closed


def test_without_retrieval_a_geometry_failure_remains_an_error(tmp_path, monkeypatch):
    cfg, frames = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "vpr_coarse_prior", lambda ctx: None)

    def fail(ctx):
        raise RuntimeError("no geometry")
    monkeypatch.setattr(pipeline, "_run_geometry_stages", fail)
    with pytest.raises(RuntimeError, match="no geometry"):
        pipeline.run_pipeline(cfg)
    assert frames._closed


def test_no_track_releases_sparse_frames_and_reextracts_dense_before_geometry(tmp_path, monkeypatch):
    cfg, sparse = _setup(tmp_path, monkeypatch, retrieval_only=True)
    dense = FrameSequence()
    for _ in range(12):
        dense.append(np.zeros((8, 8, 3), np.uint8))
    calls = []

    def extract(ctx, *, query_only=False):
        calls.append(query_only)
        if query_only:
            ctx.frames = ExtractedFrames(sparse, [0., 5., 11.], 1., 12,
                                          sampling={"mode": "linspace", "dense_count": 12})
        else:
            assert sparse._closed
            ctx.frames = ExtractedFrames(dense, list(map(float, range(12))), 1., 12,
                                          sampling={"mode": "dense", "dense_count": 12})

    def geometry(ctx):
        assert ctx.frames.frames is dense and len(ctx.frames.frames) == 12
        ctx.result["dense_geometry"] = True
        return ctx.result

    monkeypatch.setattr(pipeline, "extract_video_frames", extract)
    monkeypatch.setattr(pipeline, "vpr_coarse_prior", lambda ctx: None)
    monkeypatch.setattr(pipeline, "_run_geometry_stages", geometry)
    result = pipeline.run_pipeline(cfg)
    assert calls == [True, False] and result["dense_geometry"]
    assert sparse._closed and dense._closed


def test_flow_scale_keeps_dense_acquisition_in_retrieval_only_mode(tmp_path, monkeypatch):
    cfg, frames = _setup(tmp_path, monkeypatch, retrieval_only=True, use_ipm_scale=True)
    calls = []
    extracted = ExtractedFrames(frames, [0., 1., 2.], 1., 3)

    def extract(ctx, *, query_only=False):
        calls.append(query_only)
        ctx.frames = extracted

    monkeypatch.setattr(pipeline, "extract_video_frames", extract)
    pipeline.run_pipeline(cfg)
    assert calls == [False]
    assert frames._closed


def test_checkpoint_precedes_optional_models_and_retains_their_output(tmp_path, monkeypatch):
    cfg, frames = _setup(tmp_path, monkeypatch, retrieval_only=True, use_plate_anchor=True)

    def plate(ctx):
        saved = json.loads((cfg.output_dir / "result.json").read_text())
        assert saved["answer"]["route_latlon"]
        ctx.result["plate_stage_ran"] = True

    monkeypatch.setattr(pipeline, "plate_district_prior", plate)
    result = pipeline.run_pipeline(cfg)
    assert result["plate_stage_ran"] and result["execution"]["geometry"] == "skipped"
    assert frames._closed
