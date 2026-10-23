"""The visual-odometry stage (with its on-disk cache), run on a hand-built context.

VO is the slowest CPU stage, so its cache decides whether a re-run is minutes or
seconds. The cache rules (version in the key, frame-count shape check, never
loading another clip's file) were only reachable through a full run before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import monocular_osm.stages.trajectory as trajectory
from monocular_osm.frame_extraction import ExtractedFrames
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext
from monocular_osm.stages.trajectory import _vo_cache_candidates, run_visual_odometry
from monocular_osm.visual_odometry import VO_ALGO_VERSION, Trajectory


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    kw.setdefault("vo_start_sec", 0.0)
    kw.setdefault("vo_end_sec", 30.0)
    kw.setdefault("frame_stride", 3)
    cfg = PipelineConfig(url="u", city="Ulm, Germany", data_dir=tmp_path / "data",
                         output_dir=tmp_path / "out", **kw)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _traj(n: int, step: float = 1.0) -> Trajectory:
    xz = np.stack([np.arange(n) * step, np.sin(np.arange(n) / 3.0)], axis=1)
    valid = np.ones(n, dtype=bool)
    valid[0] = False
    return Trajectory(
        centers=np.c_[xz[:, 0], np.zeros(n), xz[:, 1]], xz=xz, valid=valid,
        n_inliers=[10] * n, rotations=np.tile(np.eye(3), (n, 1, 1)),
        translations=np.zeros((n, 3)))


def _save(path: Path, traj: Trajectory) -> None:
    np.savez(path, centers=traj.centers, xz=traj.xz, valid=traj.valid,
             n_inliers=np.asarray(traj.n_inliers), rotations=traj.rotations,
             translations=traj.translations, vo_algo_version=np.int64(VO_ALGO_VERSION))


def _ctx(tmp_path: Path, n: int, **kw) -> RunContext:
    ctx = RunContext.create(_cfg(tmp_path, **kw))
    ctx.video_path = tmp_path / "v.mp4"
    ctx.frames = ExtractedFrames(
        frames=[np.zeros((8, 8, 3), np.uint8) for _ in range(n)],
        timestamps=[float(i) for i in range(n)], fps=30.0, total_frames=n * 3)
    return ctx


def _no_vo(*a, **k):
    pytest.fail("visual odometry must not run when a valid cache exists")


def test_a_matching_cache_is_loaded_without_running_vo(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(12))
    monkeypatch.setattr(trajectory, "estimate_trajectory", _no_vo)
    run_visual_odometry(ctx)
    assert ctx.vo_cache == canonical
    assert ctx.traj.xz.shape == (12, 2)
    assert ctx.n_valid == 11                          # the first pose is flagged invalid
    assert (ctx.cfg.output_dir / "trajectory.png").exists()


def test_a_cache_miss_runs_vo_and_writes_a_versioned_cache(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 9, vo_workers=1)
    seen = {}

    def fake_vo(frames, n_workers=None):
        seen["n"], seen["workers"] = len(frames), n_workers
        return _traj(9)

    monkeypatch.setattr(trajectory, "estimate_trajectory", fake_vo)
    run_visual_odometry(ctx)
    assert seen == {"n": 9, "workers": 1}
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    assert ctx.vo_cache == canonical and canonical.exists()
    with np.load(canonical) as z:
        assert int(z["vo_algo_version"]) == VO_ALGO_VERSION
        assert z["xz"].shape == (9, 2)
    assert ctx.n_valid == 8


def test_a_cache_for_another_frame_count_is_recomputed_not_loaded(tmp_path, monkeypatch):
    """Same key, different stream (fps changed on a re-download): loading it
    would misassociate poses with frames."""
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(30))
    monkeypatch.setattr(trajectory, "estimate_trajectory", lambda *a, **k: _traj(12))
    run_visual_odometry(ctx)
    assert ctx.traj.xz.shape == (12, 2)
    with np.load(canonical) as z:                     # overwritten with the right shape
        assert z["xz"].shape == (12, 2)


def test_a_sibling_cache_with_the_right_frame_count_is_reused(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 12, max_frames=None)
    _, siblings_before, _ = _vo_cache_candidates(ctx.cfg)
    assert siblings_before == []
    seg = f"trajectory_v2_0-30.0_s3_f400_vo{VO_ALGO_VERSION}.npz"
    sibling = ctx.cfg.data_dir / seg
    _save(sibling, _traj(12))
    monkeypatch.setattr(trajectory, "estimate_trajectory", _no_vo)
    run_visual_odometry(ctx)
    assert ctx.vo_cache == sibling


def test_caches_from_an_older_vo_are_reported_and_ignored(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path, 12)
    stale = ctx.cfg.data_dir / f"trajectory_v2_0-30.0_s3_fauto_vo{VO_ALGO_VERSION - 1}.npz"
    _save(stale, _traj(12))
    monkeypatch.setattr(trajectory, "estimate_trajectory", lambda *a, **k: _traj(12))
    run_visual_odometry(ctx)
    out = capsys.readouterr().out
    assert "Ignoring 1 VO cache(s)" in out and stale.name in out
    assert ctx.vo_cache != stale


def test_ground_flow_scale_is_applied_only_when_asked(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(12))
    calls = []

    def fake_scale(traj, video_path, frames, cfg):
        calls.append(video_path)
        traj.xz = traj.xz * 2.0
        return traj

    monkeypatch.setattr(trajectory, "_apply_ground_flow_scale", fake_scale)
    run_visual_odometry(ctx)
    assert calls == []
    assert ctx.traj.xz[1, 0] == pytest.approx(1.0)

    ctx2 = _ctx(tmp_path, 12, use_ground_flow_scale=True)
    run_visual_odometry(ctx2)
    assert calls == [ctx2.video_path]
    assert ctx2.traj.xz[1, 0] == pytest.approx(2.0)
