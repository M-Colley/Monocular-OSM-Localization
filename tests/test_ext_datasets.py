"""Unit tests for the fleet-extension dataset adapters (GT-critical paths).

All fixtures are synthetic — no network, no real dataset files. These lock the
parsing/conversion invariants the round-4 audit flagged: radians conversion,
clock rebasing, the WGS84 metre-per-degree scales, the Boreas ENU-origin
guard, and the frame-drop warning.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.ext_datasets import (
    GpsFix,
    boreas_pose_track,
    brno_track,
    enu_to_latlon,
    frames_to_video,
    malaga_track,
    subsample_fixes,
)


# --- Málaga ------------------------------------------------------------------


def _write_malaga_gps(path: Path, rows):
    lines = ["% Time Lat Lon Alt fix sats speed dir ..."]
    for t, lat_rad, lon_rad in rows:
        lines.append(f"{t:.6f} {lat_rad:.9f} {lon_rad:.9f} 60.0 1 8 3.0 90.0")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_malaga_track_converts_radians_and_rebases(tmp_path: Path) -> None:
    lat_deg, lon_deg = 36.72424, -4.47621
    gps = tmp_path / "x_all-sensors_GPS.txt"
    _write_malaga_gps(gps, [
        (1000.0, math.radians(lat_deg), math.radians(lon_deg)),
        (1001.0, math.radians(lat_deg + 1e-4), math.radians(lon_deg)),
    ])
    fixes = malaga_track(gps)
    assert len(fixes) == 2
    assert fixes[0].t_sec == 0.0                      # rebased to GPS row 0
    # 1e-7 deg ~= 1 cm (the fixture writes radians at 9 decimals)
    assert abs(fixes[0].lat - lat_deg) < 1e-7         # radians -> degrees
    assert abs(fixes[0].lon - lon_deg) < 1e-7


def test_malaga_track_t0_abs_uses_video_clock(tmp_path: Path) -> None:
    """The GPS log starts AFTER the first video frame (0.42 s on extract-07);
    rebasing to the video clock must shift every t_sec by that offset."""
    gps = tmp_path / "x_all-sensors_GPS.txt"
    _write_malaga_gps(gps, [(1000.42, 0.64, -0.078), (1001.42, 0.64, -0.078)])
    fixes = malaga_track(gps, t0_abs=1000.0)          # first frame at 1000.0
    assert abs(fixes[0].t_sec - 0.42) < 1e-9
    assert abs(fixes[1].t_sec - 1.42) < 1e-9


# --- Brno --------------------------------------------------------------------


def test_brno_track_ts_scale_and_range_guard(tmp_path: Path) -> None:
    pose = tmp_path / "pose.txt"
    pose.write_text(
        "# header\n"
        "1000000000 49.19 16.60 220.0 0.5 0.5\n"      # ns clock
        "2000000000 49.191 16.601 220.0 0.5 0.5\n"
        "3000000000 999.0 16.6 220.0 0.5 0.5\n",       # lat out of range -> drop
        encoding="utf-8")
    fixes = brno_track(pose)
    assert len(fixes) == 2
    assert abs(fixes[1].t_sec - 1.0) < 1e-9            # 1e9 ns -> 1 s
    assert fixes[0].lat == 49.19


# --- ENU / WGS84 -------------------------------------------------------------


def test_enu_to_latlon_uses_latitude_dependent_scales() -> None:
    """Regression (audit round-4 EXT-1): the spherical 111320 constant put a
    2.3-3.6 m systematic error on the Boreas GT 1.8 km from the origin. The
    fix must agree with the analytic WGS84 meridian scale at 43.78N
    (111108 m/deg, NOT 111320) to sub-decimetre."""
    lat0, lon0 = 43.78215, -79.46611
    ll = enu_to_latlon(np.array([0.0]), np.array([1800.0]), lat0, lon0)
    la = math.radians(lat0)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * la) + 1.175 * math.cos(4 * la)
    expected_lat = lat0 + 1800.0 / m_per_deg_lat
    assert abs(ll[0, 0] - expected_lat) * 111120 < 0.1        # <0.1 m
    # the OLD constant would land ~3.4 m away — make sure we're not on it
    old_lat = lat0 + 1800.0 / 111320.0
    assert abs(ll[0, 0] - old_lat) * 111120 > 2.0


def test_boreas_pose_track_origin_guard_and_units(tmp_path: Path) -> None:
    ad = tmp_path / "applanix"
    ad.mkdir()
    lat0, lon0 = 43.78215, -79.46611
    (ad / "gps_post_process.csv").write_text(
        "GPSTime,easting,northing,altitude,latitude,longitude\n"
        f"1.0,0.0001,0.0002,180.0,{math.radians(lat0)},{math.radians(lon0)}\n",
        encoding="utf-8")
    (ad / "camera_poses.csv").write_text(
        "GPSTime,easting,northing,altitude\n"
        "1000000,0.0,0.0,180.0\n"
        "1050000,10.0,20.0,180.0\n",                   # +0.05 s, +10 m E +20 m N
        encoding="utf-8")
    tp, latlon = boreas_pose_track(ad)
    assert abs(tp[1] - 0.05) < 1e-9                    # microseconds -> s
    assert abs(latlon[0, 0] - lat0) < 1e-9             # frame 0 at the origin
    assert latlon[1, 0] > lat0 and latlon[1, 1] > lon0

    # non-zero origin row must REFUSE (a shifted reference frame would move
    # the whole GT silently)
    (ad / "gps_post_process.csv").write_text(
        "GPSTime,easting,northing,altitude,latitude,longitude\n"
        f"1.0,500.0,0.0,180.0,{math.radians(lat0)},{math.radians(lon0)}\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="not the ENU origin"):
        boreas_pose_track(ad)


# --- shared helpers ----------------------------------------------------------


def test_subsample_fixes_keeps_endpoints() -> None:
    fixes = [GpsFix(float(i), 48.0 + i * 1e-5, 9.0) for i in range(10)]
    out = subsample_fixes(fixes, every_n=4)
    assert out[0] is fixes[0] and out[-1] is fixes[-1]
    assert [f.t_sec for f in out] == [0.0, 4.0, 8.0, 9.0]
    assert subsample_fixes(fixes, every_n=1) == fixes


def test_frames_to_video_warns_on_dropped_frames(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    good = tmp_path / "a.png"
    bad = tmp_path / "b.png"
    good2 = tmp_path / "c.png"
    cv2.imwrite(str(good), np.full((32, 48, 3), 100, np.uint8))
    bad.write_bytes(b"not an image")
    cv2.imwrite(str(good2), np.full((16, 24, 3), 50, np.uint8))   # resized
    out = tmp_path / "out.mp4"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames_to_video([good, bad, good2], out, fps=10.0)
    assert any("unreadable" in str(x.message) for x in w)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2             # bad dropped
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 48            # first size
    cap.release()
