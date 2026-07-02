"""Tests for the comma2k19 → video/ground-truth adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.comma2k19 import (
    COMMA_FPS,
    comma_ground_truth,
    ecef_to_latlon,
    load_route_track,
    load_segment_track,
    render_route_to_video,
)


def _ecef(latlons, alt=50.0) -> np.ndarray:
    """Forward WGS84 -> ECEF for building synthetic poses."""
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    out = []
    for lat, lon in latlons:
        x, y, z = fwd.transform(lon, lat, alt)
        out.append([x, y, z])
    return np.array(out)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    # comma2k19 stores arrays under an extension-less filename.
    with open(path, "wb") as fh:
        np.save(fh, arr)


def _make_segment(seg_dir: Path, latlons, times=None) -> Path:
    pose = seg_dir / "global_pose"
    pose.mkdir(parents=True)
    _save_npy(pose / "frame_positions", _ecef(latlons))
    if times is not None:
        _save_npy(pose / "frame_times", np.asarray(times, dtype=float))
    return seg_dir


# ---------------------------------------------------------------------------
# ecef_to_latlon
# ---------------------------------------------------------------------------


def test_ecef_to_latlon_roundtrip() -> None:
    latlons = [(37.46, -122.16), (37.47, -122.15), (37.48, -122.14)]
    out = ecef_to_latlon(_ecef(latlons))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [p[0] for p in latlons], atol=1e-6)
    np.testing.assert_allclose(out[:, 1], [p[1] for p in latlons], atol=1e-6)


# ---------------------------------------------------------------------------
# load_segment_track
# ---------------------------------------------------------------------------


def test_load_segment_track_uses_frame_times(tmp_path: Path) -> None:
    latlons = [(37.46 + i * 1e-4, -122.16 + i * 1e-4) for i in range(5)]
    # Boot-time clock that does NOT start at zero -> must be re-referenced.
    times = [1000.0 + i * 0.05 for i in range(5)]
    seg = _make_segment(tmp_path / "seg0", latlons, times=times)
    track = load_segment_track(seg)
    assert len(track) == 5
    assert track[0].t_sec == pytest.approx(0.0)
    assert track[1].t_sec == pytest.approx(0.05)
    assert track[0].lat == pytest.approx(37.46, abs=1e-6)
    assert track[0].lon == pytest.approx(-122.16, abs=1e-6)


def test_load_segment_track_synthesises_time_without_file(tmp_path: Path) -> None:
    latlons = [(37.46, -122.16), (37.461, -122.159)]
    seg = _make_segment(tmp_path / "seg0", latlons)  # no frame_times
    track = load_segment_track(seg)
    assert track[1].t_sec == pytest.approx(1.0 / COMMA_FPS)


# ---------------------------------------------------------------------------
# load_route_track — concatenation across consecutive segments
# ---------------------------------------------------------------------------


def test_load_route_track_concatenates_with_monotonic_time(tmp_path: Path) -> None:
    seg0 = _make_segment(
        tmp_path / "0", [(37.46, -122.16), (37.461, -122.159)],
        times=[100.0, 100.05])
    seg1 = _make_segment(
        tmp_path / "1", [(37.462, -122.158), (37.463, -122.157)],
        times=[200.0, 200.05])
    track = load_route_track([seg0, seg1])
    assert len(track) == 4
    ts = [f.t_sec for f in track]
    assert ts == sorted(ts)                 # strictly increasing across the seam
    assert ts[0] == pytest.approx(0.0)
    # Second segment starts one frame-interval after the first one ended.
    assert ts[2] == pytest.approx(0.05 + 1.0 / COMMA_FPS)


# ---------------------------------------------------------------------------
# comma_ground_truth — emits the project schema
# ---------------------------------------------------------------------------


def test_comma_ground_truth_schema(tmp_path: Path) -> None:
    route = tmp_path / "route_abc"
    segs = []
    for s in range(3):
        latlons = [(37.46 + (s * 10 + i) * 1e-4, -122.16 + (s * 10 + i) * 1e-4)
                   for i in range(10)]
        segs.append(_make_segment(route / str(s), latlons,
                                   times=[i * 0.05 for i in range(10)]))
    gt = comma_ground_truth(segs, n_waypoints=10)
    assert gt["city"].startswith("San Francisco")
    assert gt["source"] == "comma2k19_global_pose"
    assert gt["video_id"] == "comma2k19_route_abc"
    assert 2 <= len(gt["waypoints"]) <= 10
    for w in gt["waypoints"]:
        assert set(w) == {"t_sec", "lat", "lon"}
    import json
    json.dumps(gt)


def test_comma_ground_truth_empty_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    (empty / "global_pose").mkdir(parents=True)
    _save_npy(empty / "global_pose" / "frame_positions", np.zeros((0, 3)))
    with pytest.raises(ValueError):
        comma_ground_truth([empty])


# ---------------------------------------------------------------------------
# render_route_to_video — video/GT alignment must fail loudly, not silently
# ---------------------------------------------------------------------------


def _make_video_segment(seg_dir: Path, n_video_frames: int, n_poses: int) -> Path:
    """A segment with a decodable 'video.hevc' (mp4-encoded; cv2 sniffs the
    container by content, not extension) and `n_poses` pose rows."""
    import cv2

    seg_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(seg_dir / "video.hevc"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this platform")
    for i in range(n_video_frames):
        writer.write(np.full((48, 64, 3), i % 256, dtype=np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(seg_dir / "video.hevc"))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        pytest.skip("OpenCV cannot re-read the synthetic segment video")
    pose = seg_dir / "global_pose"
    pose.mkdir(exist_ok=True)
    _save_npy(pose / "frame_positions", np.zeros((n_poses, 3)))
    return seg_dir


def test_render_route_matching_counts_succeeds(tmp_path: Path) -> None:
    seg = _make_video_segment(tmp_path / "0", n_video_frames=10, n_poses=10)
    out = render_route_to_video([seg], tmp_path / "out.mp4")
    assert out.exists()


def test_render_route_raises_on_frame_pose_mismatch(tmp_path: Path) -> None:
    # A truncated video.hevc decodes far fewer frames than the segment has
    # poses — silently continuing would shift every later frame vs GT time.
    seg = _make_video_segment(tmp_path / "0", n_video_frames=10, n_poses=40)
    with pytest.raises(RuntimeError, match="misalign"):
        render_route_to_video([seg], tmp_path / "out.mp4")


def test_render_route_raises_on_unopenable_segment(tmp_path: Path) -> None:
    # Old behavior: an unopenable second segment was skipped silently and
    # the mp4 quietly lost a whole segment of footage.
    seg0 = _make_video_segment(tmp_path / "0", n_video_frames=10, n_poses=10)
    seg1 = tmp_path / "1"
    (seg1 / "global_pose").mkdir(parents=True)
    (seg1 / "video.hevc").write_bytes(b"this is not a video stream")
    _save_npy(seg1 / "global_pose" / "frame_positions", np.zeros((10, 3)))
    with pytest.raises(RuntimeError):
        render_route_to_video([seg0, seg1], tmp_path / "out.mp4")
