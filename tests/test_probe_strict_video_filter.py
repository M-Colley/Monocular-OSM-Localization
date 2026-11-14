"""The strict-video probe has no sensor timing or initial-position inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probe_strict_video_filter import metric_config, parser, preserve_sources, sha256, video_odometry, write_json


def test_video_pts_define_filter_time_and_aggregation():
    n = 11
    centers = np.column_stack([np.zeros(n), np.zeros(n), np.arange(n)])
    traj = SimpleNamespace(centers=centers, xz=centers[:, [2, 0]],
                           valid=np.ones(n, bool), rotations=np.repeat(np.eye(3)[None], n, axis=0))
    intervals = np.array([.10, .12, .09, .11, .10, .09, .10, .12, .10, .10])
    times = np.r_[13.0, 13.0 + np.cumsum(intervals)]
    odom, factor = video_odometry(traj, times, np.ones(n - 1))
    assert factor == 5
    assert np.allclose(odom.dt, [intervals[:5].sum(), intervals[5:].sum()])
    assert np.array_equal(odom.frame_index, [5, 10])
    assert np.allclose(odom.times, times[np.r_[0, odom.frame_index]] - times[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        video_odometry(traj, np.zeros(n), np.ones(n - 1))


def test_metric_noise_comes_from_estimator_and_no_position_prior():
    config = metric_config(600_000, 0, .21, 256)
    assert config.metric_speed_dev_sigma == .21
    assert config.metric_logk_sigma == .25
    assert config.init_disc is None
    assert config.path_budget_mb == 256
    assert metric_config(10, 1, float("nan"), 128).metric_speed_dev_sigma == .15
    assert metric_config(10, 1, 5.0, 128).metric_speed_dev_sigma == .3


def test_arguments_exclude_external_calibration_and_position_inputs():
    options = {option for action in parser()._actions for option in action.option_strings}
    assert not options.intersection({"--gt", "--calibration", "--oxts", "--timestamps", "--init-disc", "--vpr"})


def test_json_artifacts_do_not_emit_nonfinite_values(tmp_path):
    path = tmp_path / "prediction.json"
    write_json(path, {"metric": float("nan"), "nested": [float("inf"), 2.0]})
    assert json.loads(path.read_text()) == {"metric": None, "nested": [None, 2.0]}


def test_replay_source_snapshots_preserve_the_hashed_bytes(tmp_path):
    root, output = tmp_path / "code", tmp_path / "experiment"
    source = root / "package" / "filter.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# exact original source\r\n")
    hashes = preserve_sources([source], output, root)
    relative = str(source.relative_to(root))
    snapshot = output / "source_snapshot" / relative
    assert hashes[relative] == sha256(snapshot) == sha256(source)
    source.write_bytes(b"# later checkout change\n")
    assert hashes[relative] == sha256(snapshot) != sha256(source)
