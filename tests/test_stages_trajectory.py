"""The visual-odometry stage (with its on-disk cache), run on a hand-built context.

VO is the slowest CPU stage, so its cache decides whether a re-run is minutes or
seconds. The cache rules (version in the key, frame-count shape check, never
loading another clip's file) were only reachable through a full run before.
"""

from __future__ import annotations

import os
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


def _save(path: Path, traj: Trajectory, ctx: RunContext | None = None) -> None:
    arrays = dict(centers=traj.centers, xz=traj.xz, valid=traj.valid,
                  n_inliers=np.asarray(traj.n_inliers), rotations=traj.rotations,
                  translations=traj.translations, vo_algo_version=np.int64(VO_ALGO_VERSION))
    if ctx is not None:
        arrays["cache_signature"] = np.array(trajectory._vo_input_signature(
            ctx.cfg, ctx.video_path, ctx.frames))
    np.savez(path, **arrays)


def _ctx(tmp_path: Path, n: int, **kw) -> RunContext:
    ctx = RunContext.create(_cfg(tmp_path, **kw))
    ctx.video_path = tmp_path / "v.mp4"
    if not ctx.video_path.exists():
        ctx.video_path.write_bytes(b"source video")
    ctx.frames = ExtractedFrames(
        frames=[np.zeros((8, 8, 3), np.uint8) for _ in range(n)],
        timestamps=[float(i) for i in range(n)], fps=30.0, total_frames=n * 3)
    return ctx


def _no_vo(*a, **k):
    pytest.fail("visual odometry must not run when a valid cache exists")


def test_a_matching_cache_is_loaded_without_running_vo(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(12), ctx)
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
        assert str(z["cache_signature"]) == trajectory._vo_input_signature(
            ctx.cfg, ctx.video_path, ctx.frames)
    assert ctx.n_valid == 8


def test_a_cache_for_another_frame_count_is_recomputed_not_loaded(tmp_path, monkeypatch):
    """Same key, different stream (fps changed on a re-download): loading it
    would misassociate poses with frames."""
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(30), ctx)
    monkeypatch.setattr(trajectory, "estimate_trajectory", lambda *a, **k: _traj(12))
    run_visual_odometry(ctx)
    assert ctx.traj.xz.shape == (12, 2)
    with np.load(canonical) as z:                     # overwritten with the right shape
        assert z["xz"].shape == (12, 2)


def test_a_sibling_cache_with_different_extraction_params_is_recomputed(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, 12, max_frames=400)
    sibling, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(sibling, _traj(12), ctx)
    ctx.cfg.max_frames = None
    _, siblings_before, _ = _vo_cache_candidates(ctx.cfg)
    assert siblings_before == [sibling]
    monkeypatch.setattr(trajectory, "estimate_trajectory", lambda *a, **k: _traj(12, step=2))
    run_visual_odometry(ctx)
    assert ctx.vo_cache != sibling
    assert ctx.traj.xz[1, 0] == 2


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
    _save(canonical, _traj(12), ctx)
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


@pytest.mark.parametrize("changed", ["pixels", "dimensions", "timestamps", "fps",
                                      "source", "path", "window"])
def test_same_frame_count_does_not_make_changed_inputs_cache_compatible(
        tmp_path, monkeypatch, changed):
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(12), ctx)
    if changed == "pixels":
        ctx.frames.frames[6][2, 3, 0] = 1
    elif changed == "dimensions":
        ctx.frames.frames = [np.zeros((4, 16, 3), np.uint8) for _ in range(12)]
    elif changed == "timestamps":
        ctx.frames.timestamps[6] += 0.001
    elif changed == "fps":
        ctx.frames.fps = 60.0
    elif changed == "source":
        stat = ctx.video_path.stat()
        ctx.video_path.write_bytes(b"other source video")
        os.utime(ctx.video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif changed == "path":
        ctx.video_path = tmp_path / "other.mp4"
        ctx.video_path.write_bytes(b"source video")
    else:
        ctx.cfg.vo_start_sec = 0.125
        # A renamed cache must still fail its exact window signature.
        renamed, _, _ = _vo_cache_candidates(ctx.cfg)
        canonical.replace(renamed)
    monkeypatch.setattr(trajectory, "estimate_trajectory", lambda *a, **k: _traj(12, step=2))
    run_visual_odometry(ctx)
    assert ctx.traj.xz[1, 0] == 2


def test_frame_digest_detects_same_size_source_replaced_with_preserved_stat(tmp_path):
    ctx = _ctx(tmp_path, 12)
    before = trajectory._vo_input_signature(ctx.cfg, ctx.video_path, ctx.frames)
    stat = ctx.video_path.stat()
    ctx.video_path.write_bytes(b"different!!!")      # same size as the original
    os.utime(ctx.video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    ctx.frames.frames[-1][0, 0, 0] = 17
    after = trajectory._vo_input_signature(ctx.cfg, ctx.video_path, ctx.frames)
    assert before != after


@pytest.mark.parametrize("damage", ["unsigned", "missing_array", "bad_shape", "nonfinite"])
def test_unverifiable_or_damaged_cache_is_recomputed(tmp_path, monkeypatch, damage):
    ctx = _ctx(tmp_path, 12)
    canonical, _, _ = _vo_cache_candidates(ctx.cfg)
    _save(canonical, _traj(12), None if damage == "unsigned" else ctx)
    if damage != "unsigned":
        with np.load(canonical) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if damage == "missing_array":
            del arrays["xz"]
        elif damage == "bad_shape":
            arrays["rotations"] = arrays["rotations"][:-1]
        else:
            arrays["xz"][0, 0] = np.nan
        np.savez(canonical, **arrays)
    monkeypatch.setattr(trajectory, "estimate_trajectory", lambda *a, **k: _traj(12, step=2))
    run_visual_odometry(ctx)
    assert ctx.traj.xz[1, 0] == 2


def test_fractional_start_times_have_distinct_cache_paths(tmp_path):
    cfg = _cfg(tmp_path, vo_start_sec=0.1)
    first, _, _ = _vo_cache_candidates(cfg)
    cfg.vo_start_sec = 0.2
    second, _, _ = _vo_cache_candidates(cfg)
    assert first != second


def test_interrupted_cache_write_preserves_previous_archive(tmp_path, monkeypatch):
    target = tmp_path / "trajectory.npz"
    target.write_bytes(b"previous complete archive")

    def fail_save(path, **arrays):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(trajectory.np, "savez", fail_save)
    with pytest.raises(OSError, match="disk full"):
        trajectory._save_vo_cache(target, _traj(12), "signature")
    assert target.read_bytes() == b"previous complete archive"
    assert list(tmp_path.iterdir()) == [target]
