"""The acquisition and route-length-prior stages, run on a hand-built context.

Both used to be inline blocks of ``run_pipeline`` whose only way to be exercised
was a full end-to-end run; now a stage takes a :class:`RunContext`, so a test can
set up exactly the state it needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import monocular_osm.speed_scale as speed_scale
import monocular_osm.stages.acquire as acquire
from monocular_osm.frame_extraction import ExtractedFrames
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext
from monocular_osm.stages.length_prior import (
    _length_sane_here,
    estimate_flow_scale,
    estimate_length_prior,
)


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    return PipelineConfig(url="https://example.com/v", city="Ulm, Germany",
                          data_dir=tmp_path / "data", output_dir=tmp_path / "out", **kw)


def _frames(n: int, dt: float = 1.0, fps: float = 30.0) -> ExtractedFrames:
    return ExtractedFrames(
        frames=[np.zeros((12, 16, 3), np.uint8) for _ in range(n)],
        timestamps=[i * dt for i in range(n)], fps=fps, total_frames=n * 6)


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


def test_acquire_video_sets_the_local_path(tmp_path):
    video = tmp_path / "drive.mp4"
    video.write_bytes(b"\x00")
    ctx = RunContext.create(_cfg(tmp_path, video_path=video))
    acquire.acquire_video(ctx)
    assert ctx.video_path == video


def test_extract_video_frames_passes_the_analysed_window_through(tmp_path, monkeypatch):
    seen = {}

    def fake_extract(path, **kw):
        seen["path"], seen["kw"] = path, kw
        return _frames(4)

    monkeypatch.setattr(acquire, "extract_frames", fake_extract)
    ctx = RunContext.create(_cfg(tmp_path, frame_stride=3, max_frames=50,
                                 vo_start_sec=5.0, vo_end_sec=20.0))
    ctx.video_path = tmp_path / "v.mp4"
    acquire.extract_video_frames(ctx)
    assert len(ctx.frames.frames) == 4
    assert seen["path"] == ctx.video_path
    assert seen["kw"] == dict(stride=3, max_frames=50, start_sec=5.0, end_sec=20.0)


# ---------------------------------------------------------------------------
# estimate_length_prior
# ---------------------------------------------------------------------------


def test_length_prior_is_duration_at_urban_speed(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path))
    ctx.frames = _frames(101, dt=1.0)                 # a 100 s analysed window
    estimate_length_prior(ctx)
    assert ctx.duration_prior_m == pytest.approx(100 * 5.5)
    assert ctx.estimated_length_m == ctx.duration_prior_m


def test_an_explicit_length_overrides_the_estimate_but_not_the_reference(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path, estimated_length_m=2000.0))
    ctx.frames = _frames(101, dt=1.0)
    estimate_length_prior(ctx)
    assert ctx.estimated_length_m == 2000.0
    # the sanity reference every noisier scale source is gated against stays
    # the duration-based one
    assert ctx.duration_prior_m == pytest.approx(550.0)


def test_length_prior_is_clamped_for_a_tiny_clip(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path))
    ctx.frames = _frames(1)
    estimate_length_prior(ctx)
    assert ctx.duration_prior_m == 500.0              # floor: walks degenerate below


def test_length_sane_here_uses_the_context_reference(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path))
    ctx.duration_prior_m = 1000.0
    assert _length_sane_here(ctx, 1900.0) and _length_sane_here(ctx, 520.0)
    assert not _length_sane_here(ctx, 2100.0) and not _length_sane_here(ctx, 400.0)


def test_length_sane_here_also_accepts_the_users_explicit_length(tmp_path):
    ctx = RunContext.create(_cfg(tmp_path, estimated_length_m=5000.0))
    ctx.duration_prior_m = 1000.0
    assert _length_sane_here(ctx, 4000.0)             # within 2x of the user's figure


# ---------------------------------------------------------------------------
# estimate_flow_scale
# ---------------------------------------------------------------------------


def _flow_ctx(tmp_path, **kw) -> RunContext:
    ctx = RunContext.create(_cfg(tmp_path, **kw))
    ctx.frames = _frames(30)
    ctx.duration_prior_m = 1000.0
    ctx.estimated_length_m = 1000.0
    return ctx


def test_flow_scale_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(speed_scale, "estimate_route_length_from_flow",
                        lambda *a, **k: pytest.fail("must not run"))
    ctx = _flow_ctx(tmp_path)
    estimate_flow_scale(ctx)
    assert ctx.result_ipm_scale is None and ctx.estimated_length_m == 1000.0


def test_flow_length_within_3x_of_the_prior_is_adopted(tmp_path, monkeypatch):
    # a faster-than-assumed drive legitimately measures ~2x the 5.5 m/s prior
    monkeypatch.setattr(speed_scale, "estimate_route_length_from_flow",
                        lambda *a, **k: (1950.0, []))
    ctx = _flow_ctx(tmp_path, use_ipm_scale=True)
    estimate_flow_scale(ctx)
    assert ctx.estimated_length_m == 1950.0
    assert ctx.result_ipm_scale == {"status": "ok", "length_m": 1950.0}


def test_flow_length_outside_the_band_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(speed_scale, "estimate_route_length_from_flow",
                        lambda *a, **k: (4000.0, []))
    ctx = _flow_ctx(tmp_path, use_ipm_scale=True)
    estimate_flow_scale(ctx)
    assert ctx.estimated_length_m == 1000.0           # prior kept
    assert ctx.result_ipm_scale == {"status": "rejected", "length_m": 4000.0}


def test_flow_scale_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("flow exploded")

    monkeypatch.setattr(speed_scale, "estimate_route_length_from_flow", boom)
    ctx = _flow_ctx(tmp_path, use_ipm_scale=True)
    estimate_flow_scale(ctx)
    assert ctx.estimated_length_m == 1000.0
    assert ctx.result_ipm_scale["status"] == "error"
    assert ctx.channel_status["ipm_scale"]["status"] == "failed"
    assert "flow exploded" in ctx.channel_status["ipm_scale"]["error"]
