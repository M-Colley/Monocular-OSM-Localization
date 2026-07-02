"""Tests for the KITTI raw → video/ground-truth adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.kitti_raw import (
    KITTI_FPS,
    kitti_ground_truth,
    load_oxts_track,
    osm_around_for_track,
    render_images_to_video,
)
from src.gps_overlay import GpsFix


# A short real-shaped OXTS snippet: lat lon alt roll pitch yaw ... (30 cols).
# Only the first two matter to us; the rest are filler.
def _oxts_line(lat: float, lon: float) -> str:
    rest = " ".join(["0.0"] * 28)
    return f"{lat:.9f} {lon:.9f} {rest}\n"


def _make_drive(tmp: Path, latlons, timestamps=None) -> Path:
    drive = tmp / "2011_09_26_drive_0009_sync"
    data = drive / "oxts" / "data"
    data.mkdir(parents=True)
    for i, (lat, lon) in enumerate(latlons):
        (data / f"{i:010d}.txt").write_text(_oxts_line(lat, lon), encoding="utf-8")
    if timestamps is not None:
        (drive / "oxts" / "timestamps.txt").write_text(
            "\n".join(timestamps) + "\n", encoding="utf-8")
    return drive


# ---------------------------------------------------------------------------
# load_oxts_track
# ---------------------------------------------------------------------------


def test_load_oxts_reads_latlon_and_synthesises_time(tmp_path: Path) -> None:
    latlons = [(49.0 + i * 1e-4, 8.4 + i * 1e-4) for i in range(5)]
    drive = _make_drive(tmp_path, latlons)  # no timestamps file
    track = load_oxts_track(drive)
    assert len(track) == 5
    assert track[0].lat == pytest.approx(49.0)
    assert track[0].lon == pytest.approx(8.4)
    # Without a timestamps file, time falls back to index / 10 Hz.
    assert track[1].t_sec == pytest.approx(1.0 / KITTI_FPS)
    assert track[-1].t_sec == pytest.approx(4.0 / KITTI_FPS)


def test_load_oxts_uses_timestamps_file(tmp_path: Path) -> None:
    latlons = [(49.0, 8.4), (49.001, 8.401), (49.002, 8.402)]
    ts = [
        "2011-09-26 13:02:25.000000000",
        "2011-09-26 13:02:25.500000000",
        "2011-09-26 13:02:26.100000000",
    ]
    drive = _make_drive(tmp_path, latlons, timestamps=ts)
    track = load_oxts_track(drive)
    assert [round(f.t_sec, 2) for f in track] == [0.0, 0.5, 1.1]


def test_load_oxts_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_oxts_track(tmp_path / "nope")


def test_load_oxts_skips_blank_lines(tmp_path: Path) -> None:
    latlons = [(49.0, 8.4), (49.001, 8.401)]
    drive = _make_drive(tmp_path, latlons)
    # Corrupt one file to be empty — should be skipped, not crash.
    (drive / "oxts" / "data" / "0000000001.txt").write_text("", encoding="utf-8")
    track = load_oxts_track(drive)
    assert len(track) == 1


# ---------------------------------------------------------------------------
# kitti_ground_truth — emits the project schema
# ---------------------------------------------------------------------------


def test_kitti_ground_truth_schema(tmp_path: Path) -> None:
    latlons = [(49.0 + i * 1e-4, 8.4 + i * 1e-4) for i in range(40)]
    drive = _make_drive(tmp_path, latlons)
    gt = kitti_ground_truth(drive, n_waypoints=10)
    assert gt["city"] == "Karlsruhe, Germany"
    assert gt["source"] == "kitti_raw_oxts"
    assert gt["video_id"] == "2011_09_26_drive_0009_sync"
    assert 2 <= len(gt["waypoints"]) <= 10
    for w in gt["waypoints"]:
        assert set(w) == {"t_sec", "lat", "lon"}
    import json
    json.dumps(gt)  # serialisable


# ---------------------------------------------------------------------------
# osm_around_for_track — the region prior for the OSM fetch
# ---------------------------------------------------------------------------


def test_osm_around_bounds_track(tmp_path: Path) -> None:
    # ~ a few hundred metres of travel around Karlsruhe.
    fixes = [GpsFix(i / KITTI_FPS, 49.0 + i * 1e-4, 8.4 + i * 1e-4) for i in range(20)]
    clat, clon, radius = osm_around_for_track(fixes, margin_m=500.0)
    assert clat == pytest.approx(np.mean([f.lat for f in fixes]), abs=1e-6)
    assert clon == pytest.approx(np.mean([f.lon for f in fixes]), abs=1e-6)
    # Radius covers the half-diagonal of the span plus the margin.
    assert radius > 500.0
    # The farthest fix must lie inside the disc.
    far = max(fixes, key=lambda f: (f.lat - clat) ** 2 + (f.lon - clon) ** 2)
    dlat_m = (far.lat - clat) * 111320.0
    dlon_m = (far.lon - clon) * 111320.0 * np.cos(np.radians(clat))
    assert np.hypot(dlat_m, dlon_m) <= radius


# ---------------------------------------------------------------------------
# render_images_to_video — corrupt PNGs must not crash or shift silently
# ---------------------------------------------------------------------------


def _make_image_drive(tmp: Path, n_frames: int, corrupt: set[int]) -> Path:
    import cv2

    drive = tmp / "2011_09_26_drive_0009_sync"
    data = drive / "image_02" / "data"
    data.mkdir(parents=True)
    for i in range(n_frames):
        fp = data / f"{i:010d}.png"
        if i in corrupt:
            fp.write_bytes(b"")  # zero-byte PNG (interrupted zip extraction)
        else:
            cv2.imwrite(str(fp), np.full((48, 64, 3), i % 256, dtype=np.uint8))
    return drive


def test_render_skips_corrupt_first_png_and_warns(tmp_path: Path) -> None:
    # Old behavior: AttributeError on `first.shape` for a corrupt frame 0.
    drive = _make_image_drive(tmp_path, n_frames=5, corrupt={0, 2})
    with pytest.warns(RuntimeWarning, match="unreadable"):
        out = render_images_to_video(drive, tmp_path / "out.mp4")
    assert out.exists()


def test_render_no_warning_when_all_readable(tmp_path: Path) -> None:
    import warnings

    drive = _make_image_drive(tmp_path, n_frames=4, corrupt=set())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = render_images_to_video(drive, tmp_path / "out.mp4")
    assert out.exists()


def test_render_all_corrupt_raises_descriptive(tmp_path: Path) -> None:
    drive = _make_image_drive(tmp_path, n_frames=3, corrupt={0, 1, 2})
    with pytest.raises(RuntimeError, match="no readable PNG"):
        render_images_to_video(drive, tmp_path / "out.mp4")
