"""Which trajectory the shape matcher consumes, and loop-closure drift correction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import monocular_osm.da3_reconstruction as da3_reconstruction
import monocular_osm.loop_closure as loop_closure
import monocular_osm.mapanything_trajectory as mapanything_trajectory
import monocular_osm.stages.trajectory as trajectory
from monocular_osm.frame_extraction import ExtractedFrames
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext
from monocular_osm.visual_odometry import Trajectory

N_FRAMES = 40


def _ctx(tmp_path: Path, **kw) -> RunContext:
    cfg = PipelineConfig(url="u", city="Ulm, Germany", data_dir=tmp_path / "data",
                         output_dir=tmp_path / "out", **kw)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext.create(cfg)
    ctx.video_path = tmp_path / "v.mp4"
    ctx.frames = ExtractedFrames(
        frames=[np.zeros((8, 8, 3), np.uint8) for _ in range(N_FRAMES)],
        timestamps=[float(i) for i in range(N_FRAMES)], fps=30.0, total_frames=N_FRAMES * 3)
    xz = np.stack([np.linspace(0, 80, N_FRAMES), np.zeros(N_FRAMES)], axis=1)
    ctx.traj = Trajectory(centers=np.c_[xz[:, 0], 0 * xz[:, 0], xz[:, 1]], xz=xz,
                          valid=np.ones(N_FRAMES, bool), n_inliers=[9] * N_FRAMES,
                          rotations=np.tile(np.eye(3), (N_FRAMES, 1, 1)),
                          translations=np.zeros((N_FRAMES, 3)))
    ctx.duration_prior_m = 1000.0
    ctx.estimated_length_m = 1000.0
    return ctx


def _pose_file(path: Path, n: int, cols: int = 12, scale: float = 1.0) -> Path:
    rows = np.zeros((n, cols))
    rows[:, 3] = np.linspace(0.0, 50.0, n) * scale          # tx
    rows[:, 11] = np.linspace(0.0, 20.0, n) * scale         # tz (col 11 in both layouts)
    np.savetxt(path, rows)
    return path


# ---------------------------------------------------------------------------
# select_matcher_trajectory
# ---------------------------------------------------------------------------


def test_the_vo_path_is_the_default_matcher_input(tmp_path):
    ctx = _ctx(tmp_path)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.match_xz is ctx.traj.xz and ctx.result_traj_source == "vo"
    assert ctx.da3_rec is None and ctx.result_scale_recovery is None
    assert np.array_equal(ctx.match_ts, np.arange(N_FRAMES, dtype=float))


def test_a_staged_openvo_path_replaces_vo_and_gets_its_own_time_axis(tmp_path):
    pose = _pose_file(tmp_path / "ov.txt", 25)
    ctx = _ctx(tmp_path, openvo_trajectory_path=pose)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "openvo" and ctx.match_xz.shape == (25, 2)
    assert ctx.match_xz[-1].tolist() == [50.0, 20.0]
    # fewer poses than frames: the time axis spans the clip over the pose rows
    assert len(ctx.match_ts) == 25 and ctx.match_ts[0] == 0.0 and ctx.match_ts[-1] == N_FRAMES - 1
    assert (ctx.cfg.output_dir / "trajectory_openvo.png").exists()


def test_an_openvo_file_next_to_the_data_is_picked_up_unless_disabled(tmp_path):
    ctx = _ctx(tmp_path)
    _pose_file(ctx.cfg.data_dir / "openvo_trajectory.txt", 10)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "openvo"
    ctx2 = _ctx(tmp_path, prefer_openvo_trajectory=False)
    trajectory.select_matcher_trajectory(ctx2)
    assert ctx2.result_traj_source == "vo"


def test_an_unreadable_openvo_file_falls_back_to_vo(tmp_path):
    bad = tmp_path / "ov.txt"
    bad.write_text("not numbers", encoding="utf-8")
    ctx = _ctx(tmp_path, openvo_trajectory_path=bad)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "vo" and ctx.match_xz is ctx.traj.xz


def test_vggt_long_takes_precedence_over_openvo(tmp_path):
    ov = _pose_file(tmp_path / "ov.txt", 25)
    vl = _pose_file(tmp_path / "vl.txt", 30, cols=16, scale=2.0)
    ctx = _ctx(tmp_path, openvo_trajectory_path=ov, vggt_long_trajectory_path=vl)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "vggt_long" and ctx.match_xz.shape == (30, 2)
    assert ctx.match_xz[-1].tolist() == [100.0, 40.0]


def test_mapanything_is_used_when_it_returns_a_valid_path(tmp_path, monkeypatch):
    path = np.stack([np.linspace(0, 30, 12), np.zeros(12)], axis=1)
    monkeypatch.setattr(mapanything_trajectory, "mapanything_trajectory_xy",
                        lambda video, start, end: (path, None))
    ctx = _ctx(tmp_path, use_mapanything_trajectory=True)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "mapanything" and ctx.match_xz is path


def test_mapanything_that_returns_nothing_keeps_vo(tmp_path, monkeypatch):
    monkeypatch.setattr(mapanything_trajectory, "mapanything_trajectory_xy",
                        lambda *a: None)
    ctx = _ctx(tmp_path, use_mapanything_trajectory=True)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "vo"


def _da3(monkeypatch, xy):
    monkeypatch.setattr(da3_reconstruction, "reconstruct_with_da3", lambda *a, **k: "rec")
    monkeypatch.setattr(da3_reconstruction, "da3_trajectory_xy", lambda rec: xy)


def test_a_plausible_da3_path_is_adopted_with_its_metric_length(tmp_path, monkeypatch):
    xy = np.stack([np.linspace(0, 900, 20), np.zeros(20)], axis=1)     # 900 m: sane vs 1000 m
    _da3(monkeypatch, xy)
    ctx = _ctx(tmp_path, use_da3_trajectory=True)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "da3" and ctx.match_xz is xy and ctx.da3_rec == "rec"
    assert ctx.estimated_length_m == pytest.approx(900.0)
    assert ctx.result_scale_recovery["source"] == "da3"


def test_a_da3_length_far_from_the_prior_is_not_adopted(tmp_path, monkeypatch):
    xy = np.stack([np.linspace(0, 5000, 20), np.zeros(20)], axis=1)    # 5x the prior
    _da3(monkeypatch, xy)
    ctx = _ctx(tmp_path, use_da3_trajectory=True)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "da3" and ctx.estimated_length_m == 1000.0
    assert ctx.result_scale_recovery is None


def test_a_zigzag_da3_path_is_rejected_and_vo_kept(tmp_path, monkeypatch):
    xy = np.array([[(i % 2) * 10.0, float(i)] for i in range(40)])
    xy[:, 1] = np.where(np.arange(40) % 2 == 0, 0.0, 1.0) * 10 + (np.arange(40) // 2) * 0.01
    _da3(monkeypatch, xy)
    ctx = _ctx(tmp_path, use_da3_trajectory=True)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "vo" and ctx.match_xz is ctx.traj.xz
    assert (ctx.cfg.output_dir / "trajectory_da3_rejected.png").exists()


def test_a_failing_da3_is_recorded_and_vo_kept(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("No module named 'depth_anything_3'")

    monkeypatch.setattr(da3_reconstruction, "reconstruct_with_da3", boom)
    ctx = _ctx(tmp_path, use_da3_trajectory=True)
    trajectory.select_matcher_trajectory(ctx)
    assert ctx.result_traj_source == "vo" and ctx.da3_rec is None
    assert ctx.channel_status["da3_trajectory"]["status"] == "failed"


# ---------------------------------------------------------------------------
# close_loop
# ---------------------------------------------------------------------------


def test_loop_closure_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_closure, "detect_end_to_start_loop",
                        lambda *a: pytest.fail("not enabled"))
    ctx = _ctx(tmp_path)
    trajectory.select_matcher_trajectory(ctx)
    trajectory.close_loop(ctx)
    assert ctx.result_loop_closure is None


def test_a_route_that_does_not_revisit_its_start_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_closure, "detect_end_to_start_loop", lambda frames: None)
    ctx = _ctx(tmp_path, enable_loop_closure=True)
    trajectory.select_matcher_trajectory(ctx)
    before = ctx.match_xz.copy()
    trajectory.close_loop(ctx)
    assert ctx.result_loop_closure == {"closed": False} and np.array_equal(ctx.match_xz, before)


def test_a_detected_loop_redistributes_the_drift_in_pose_space(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_closure, "detect_end_to_start_loop", lambda frames: (0, N_FRAMES - 1))
    ctx = _ctx(tmp_path, enable_loop_closure=True)
    # a lap that ends ~2% of a turn short of its start: drift, not a false loop
    ang = np.linspace(0.0, 2 * np.pi * 0.98, N_FRAMES)
    ctx.traj.xz = 50.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    trajectory.select_matcher_trajectory(ctx)
    start_end_gap = np.linalg.norm(ctx.match_xz[-1] - ctx.match_xz[0])
    trajectory.close_loop(ctx)
    lc = ctx.result_loop_closure
    assert lc["closed"] is True and (lc["i"], lc["j"]) == (0, N_FRAMES - 1)
    assert lc["gap_units"] == pytest.approx(round(float(start_end_gap), 2))
    assert np.linalg.norm(ctx.match_xz[-1] - ctx.match_xz[0]) < 1e-6      # the gap is closed


def test_the_loop_pair_is_remapped_when_the_matcher_path_is_shorter_than_the_frames(
        tmp_path, monkeypatch):
    monkeypatch.setattr(loop_closure, "detect_end_to_start_loop", lambda frames: (0, N_FRAMES - 1))
    ctx = _ctx(tmp_path, enable_loop_closure=True,
               openvo_trajectory_path=_pose_file(tmp_path / "ov.txt", 20))
    trajectory.select_matcher_trajectory(ctx)
    trajectory.close_loop(ctx)
    lc = ctx.result_loop_closure
    assert (lc["frame_i"], lc["frame_j"]) == (0, N_FRAMES - 1)
    assert (lc["i"], lc["j"]) == (0, 19)                                   # pose rows, not frames


def test_a_failing_loop_detector_is_recorded_and_the_path_kept(tmp_path, monkeypatch):
    def boom(frames):
        raise RuntimeError("descriptor model missing")

    monkeypatch.setattr(loop_closure, "detect_end_to_start_loop", boom)
    ctx = _ctx(tmp_path, enable_loop_closure=True)
    trajectory.select_matcher_trajectory(ctx)
    trajectory.close_loop(ctx)
    assert ctx.channel_status["loop_closure"]["status"] == "failed"
    assert ctx.match_xz is ctx.traj.xz
