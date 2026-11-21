"""Synthetic native preflight checks; no actual video VO, plane fit or PF."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pytest

from monocular_osm import assisted_motion as motion
from monocular_osm.plane_speed import PlaneSpeed
from monocular_osm.visual_odometry import Trajectory


def _case(tmp_path, monkeypatch, *, fraction=1., coverage=.6, spread=.1):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic native video identity")
    n = 1300
    times = np.arange(n, dtype=float) * .1
    frames = [np.zeros((8, 12, 3), np.uint8) + index % 255 for index in range(n)]
    closed, observed = [], {}
    extracted = SimpleNamespace(frames=frames, timestamps=times, fps=30., close=lambda: closed.append(True))
    class Capture:
        def isOpened(self): return True
        def get(self, prop): return 30. if prop == motion.cv2.CAP_PROP_FPS else 3900.
        def release(self): pass
    monkeypatch.setattr(motion.cv2, "VideoCapture", lambda path: Capture())
    def extract(path, **kwargs):
        observed["extract"] = kwargs
        return extracted
    monkeypatch.setattr(motion, "extract_frames", extract)
    valid = np.arange(n) < int(n * fraction)
    trajectory = Trajectory(np.zeros((n, 3)), np.c_[np.zeros(n), np.arange(n)], valid,
        [20] * n, np.repeat(np.eye(3)[None], n, axis=0), np.zeros((n, 3)))
    def estimate(gray, K, **kwargs):
        observed["vo"] = kwargs
        observed["K"] = K.copy()
        assert gray[0].shape == (8, 12)
        return trajectory
    monkeypatch.setattr(motion, "estimate_trajectory", estimate)
    speed = PlaneSpeed(np.ones(n-1), np.ones(n-1), np.full(n-1, 20), coverage, spread, 1.5)
    def plane(gray, traj, K, **kwargs):
        observed["plane"] = kwargs
        assert traj is trajectory
        return speed
    monkeypatch.setattr(motion, "plane_step_lengths", plane)
    auto_file = tmp_path / "auto.npz"
    np.savez(auto_file, valid=np.ones(390, bool))
    auto = {"status": "complete", "valid_fraction": 1., "duration_s": 129.66666666666666,
        "n_valid": 390, "n_frames": 390, "stride": 10, "fps": 30., "window": [0., 130.],
        "timestamps_sha256": "a" * 64, "video_sha256": motion.file_sha256(video),
        "vo_artifact": {"path": str(auto_file), "sha256": motion.file_sha256(auto_file)}}
    return SimpleNamespace(video=video, cfg=SimpleNamespace(vo_start_sec=0., vo_end_sec=130.),
        output=tmp_path / "native", data=tmp_path / "data", auto=auto,
        observed=observed, closed=closed, trajectory=trajectory, times=times, speed=speed)


def _prepare(case):
    return motion.prepare_metric_inputs(case.cfg, case.video, case.output, case.data, auto_quality=case.auto)


def test_complete_metric_preflight_freezes_both_grids_and_reuses_without_frames(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    original = deepcopy(case.auto)
    result = _prepare(case)
    assert result["summary"]["eligible"] and result["summary"]["auto_quality"] == original
    native = result["summary"]["native_quality"]
    assert native["stride"] == 3 and native["n_frames"] == 1300 and native["full_declared_grid"]
    assert result["summary"]["plane"]["measured"] and result["summary"]["plane"]["coverage"] == .6
    assert case.closed and case.auto == original
    assert case.observed["extract"] == dict(stride=3, max_frames=None, start_sec=0., end_sec=130.,
        memory_mb=128., max_width=None, sample_count=None)
    assert case.observed["vo"] == {"n_workers": 2, "enforce_planar": True}
    assert case.observed["plane"] == {"camera_height_m": 1.5, "horizon": "principal", "mask_static": False, "seed": 0}
    trajectory, speed, times, K = motion.load_metric_inputs(result, case.video, window=[0., 130.])
    np.testing.assert_array_equal(trajectory.valid, case.trajectory.valid)
    np.testing.assert_array_equal(times, case.times)
    np.testing.assert_array_equal(speed.raw_m, case.speed.raw_m)
    np.testing.assert_array_equal(K, case.observed["K"])


@pytest.mark.parametrize("fault", ["weak", "short", "missing", "video", "window", "counts", "grid", "artifact"])
def test_ineligible_auto_grid_declines_before_native_extraction(tmp_path, monkeypatch, fault):
    case = _case(tmp_path, monkeypatch)
    if fault == "weak": case.auto["valid_fraction"] = .94
    elif fault == "short": case.auto["duration_s"] = 119.99
    elif fault == "missing": case.auto = {}
    elif fault == "video": case.auto["video_sha256"] = "b" * 64
    elif fault == "window": case.auto["window"] = [1., 130.]
    elif fault == "counts": case.auto["n_valid"] = 389
    elif fault == "grid": case.auto["timestamps_sha256"] = None
    else: Path(case.auto["vo_artifact"]["path"]).write_bytes(b"changed")
    result = _prepare(case)
    assert not result["summary"]["eligible"] and result["summary"]["reasons"]
    assert result["summary"]["native_quality"]["status"] == "not_run"
    assert not result["summary"]["plane"]["measured"] and case.observed == {}
    assert not (case.output / "native_metric_inputs.npz").exists()


def test_weak_native_vo_still_truthfully_measures_plane_and_declines_before_pf(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, fraction=.94)
    result = _prepare(case)
    summary = result["summary"]
    assert summary["status"] == "complete" and not summary["eligible"]
    assert summary["native_quality"]["valid_fraction"] == .94
    assert summary["plane"]["status"] == "complete" and summary["plane"]["measured"]
    assert summary["plane"]["eligible"] and summary["plane"]["coverage"] == .6
    assert summary["reasons"] == ["native_vo_below_fixed_pf_minimum"]
    with pytest.raises(ValueError, match="ineligible"):
        motion.load_metric_inputs(result, case.video, window=[0., 130.])


@pytest.mark.parametrize("coverage,spread,eligible", [(.5, .15, True), (.499, .1, False), (.6, .151, False), (.6, np.nan, False)])
def test_plane_quality_uses_fixed_inclusive_policy_and_unknown_spread_declines(tmp_path, monkeypatch, coverage, spread, eligible):
    case = _case(tmp_path, monkeypatch, coverage=coverage, spread=spread)
    result = _prepare(case)
    assert result["summary"]["eligible"] is eligible
    plane = result["summary"]["plane"]
    assert plane["measured"] and plane["coverage"] == coverage
    assert plane["rel_spread"] == (None if not np.isfinite(spread) else spread)


def test_native_validity_boundary_is_inclusive_and_profile_has_no_overrides(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, fraction=.95)
    result = _prepare(case)
    assert result["summary"]["eligible"] and result["native_valid_fraction"] == .95
    with pytest.raises(TypeError): motion.PROFILE["camera_height_m"] = 2.


def test_plane_failure_preserves_native_arrays_and_restores_scoped_resources(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    previous_temp, previous_threads = tempfile.tempdir, motion.cv2.getNumThreads()
    def fail(*args, **kwargs): raise RuntimeError("synthetic plane failure")
    monkeypatch.setattr(motion, "plane_step_lengths", fail)
    result = _prepare(case)
    assert case.closed and tempfile.tempdir == previous_temp and motion.cv2.getNumThreads() == previous_threads
    summary = result["summary"]
    assert summary["status"] == "failed" and not summary["eligible"]
    assert summary["native_quality"]["status"] == "complete"
    assert summary["plane"]["status"] == "failed" and not summary["plane"]["measured"]
    with np.load(case.output / "native_metric_inputs.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["valid"], case.trajectory.valid)


@pytest.mark.parametrize("target", ["native_metric_inputs.npz", "metric_summary.json", "metric_registration.json", "source_snapshot/plane_speed.py"])
def test_reuse_checks_every_frozen_array_config_and_source_before_pf(tmp_path, monkeypatch, target):
    case = _case(tmp_path, monkeypatch)
    result = _prepare(case)
    (case.output / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        motion.load_metric_inputs(result, case.video, window=[0., 130.])


@pytest.mark.parametrize("change", ["video", "window"])
def test_reuse_is_bound_to_actual_video_bytes_and_registered_window(tmp_path, monkeypatch, change):
    case = _case(tmp_path, monkeypatch)
    result = _prepare(case)
    if change == "video": case.video.write_bytes(b"other video")
    window = [1., 130.] if change == "window" else [0., 130.]
    with pytest.raises(ValueError, match="different video/window"):
        motion.load_metric_inputs(result, case.video, window=window)


def test_native_preflight_never_rewrites_or_reblesses_a_completed_package(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    _prepare(case)
    with pytest.raises(FileExistsError, match="never re-bless"):
        _prepare(case)
