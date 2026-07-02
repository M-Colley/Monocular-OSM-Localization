"""Tests for burned-in GPS-overlay extraction (auto ground truth)."""

from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from src.gps_overlay import (
    GpsFix,
    _reject_jumps,
    extract_gps_track,
    osm_around_for_track,
    parse_latlon,
    track_to_ground_truth,
)


# ---------------------------------------------------------------------------
# parse_latlon — the deterministic core, across real overlay formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lat", "lon"),
    [
        ("N:53.8235 E:10.5033", 53.8235, 10.5033),         # VIOFO style
        ("N 53.8235 E 10.5033", 53.8235, 10.5033),         # space separator
        ("102KM/H N:53.8235 E:10.5033", 53.8235, 10.5033),  # with speed prefix
        ("S:33.8688 E:151.2093", -33.8688, 151.2093),      # southern hemisphere
        ("N:51.5270 W:0.1318", 51.5270, -0.1318),          # western (negative lon)
        ("N53,8235 E10,5033", 53.8235, 10.5033),           # comma decimals (EU OCR)
        ("51.527047, -0.131824", 51.527047, -0.131824),    # signed decimal pair
        ("48.3984 9.9916", 48.3984, 9.9916),               # space-separated pair
    ],
)
def test_parse_latlon_formats(text, lat, lon) -> None:
    out = parse_latlon(text)
    assert out is not None
    assert out[0] == pytest.approx(lat, abs=1e-4)
    assert out[1] == pytest.approx(lon, abs=1e-4)


def test_parse_latlon_dms() -> None:
    out = parse_latlon("51°31'37.4\"N 0°07'54.6\"W")
    assert out is not None
    assert out[0] == pytest.approx(51.5270, abs=1e-3)
    assert out[1] == pytest.approx(-0.1318, abs=1e-3)


@pytest.mark.parametrize(
    "text",
    [
        "", "102 KM/H", "12:45:03 2024-06-13",     # no coords / time only
        "N:200.0 E:10.0",                           # lat out of range
        "0.0 0.0",                                  # null island rejected
        "REC  FHD  speed 60",                       # junk
    ],
)
def test_parse_latlon_rejects_non_coords(text) -> None:
    assert parse_latlon(text) is None


# ---------------------------------------------------------------------------
# extract_gps_track — injected OCR, no easyocr/video needed
# ---------------------------------------------------------------------------


class _ScriptedReader:
    """Returns a scripted overlay string per frame."""

    def __init__(self, lines):
        self._lines = lines
        self._i = 0

    def readtext(self, image):
        line = self._lines[self._i % len(self._lines)]
        self._i += 1
        return [([], line, 0.9)] if line else []


def _frames(times):
    import numpy as np

    def _r(video_path, start, end, interval):
        return [(t, np.zeros((720, 1280, 3), dtype=np.uint8)) for t in times]
    return _r


def test_extract_track_parses_each_frame(tmp_path: Path) -> None:
    lines = [
        "N:51.5270 W:0.1318",
        "N:51.5260 W:0.1300",
        "N:51.5250 W:0.1280",
    ]
    track = extract_gps_track(
        tmp_path / "v.mp4", ocr_reader=_ScriptedReader(lines),
        frame_reader=_frames([0.0, 2.0, 4.0]),
    )
    assert len(track) == 3
    assert track[0].lat == pytest.approx(51.5270, abs=1e-3)
    assert track[0].lon == pytest.approx(-0.1318, abs=1e-3)
    assert [f.t_sec for f in track] == [0.0, 2.0, 4.0]


def test_extract_track_rejects_jump(tmp_path: Path) -> None:
    # Middle fix has an OCR digit error putting it ~7 km away → dropped.
    lines = [
        "N:51.5270 W:0.1318",
        "N:51.5265 W:0.1310",
        "N:51.5870 W:0.1300",   # bad: 51.58 instead of 51.526
        "N:51.5255 W:0.1290",
        "N:51.5250 W:0.1280",
    ]
    track = extract_gps_track(
        tmp_path / "v.mp4", ocr_reader=_ScriptedReader(lines),
        frame_reader=_frames([0.0, 2.0, 4.0, 6.0, 8.0]), max_jump_m=400.0,
    )
    lats = [round(f.lat, 4) for f in track]
    assert 51.587 not in lats           # the jump was rejected
    assert len(track) == 4


def test_extract_track_handles_frames_without_overlay(tmp_path: Path) -> None:
    lines = ["N:51.5270 W:0.1318", "", "N:51.5260 W:0.1300"]
    track = extract_gps_track(
        tmp_path / "v.mp4", ocr_reader=_ScriptedReader(lines),
        frame_reader=_frames([0.0, 2.0, 4.0]),
    )
    assert len(track) == 2   # the blank frame contributes nothing, no crash


# ---------------------------------------------------------------------------
# track_to_ground_truth
# ---------------------------------------------------------------------------


def test_track_to_ground_truth_schema_and_subsample() -> None:
    fixes = [GpsFix(float(i), 51.527 - i * 1e-4, -0.131 + i * 1e-4) for i in range(50)]
    gt = track_to_ground_truth(
        fixes, video_id="abc", video_url="http://x", city="London, UK",
        n_waypoints=10)
    assert gt["city"] == "London, UK"
    assert gt["source"] == "gps_overlay_ocr"
    assert 2 <= len(gt["waypoints"]) <= 10
    wps = gt["waypoints"]
    assert wps[0]["t_sec"] == 0.0 and wps[-1]["t_sec"] == 49.0
    for w in wps:
        assert set(w) == {"t_sec", "lat", "lon"}
    import json
    json.dumps(gt)  # JSON-serializable


def test_track_to_ground_truth_empty_raises() -> None:
    with pytest.raises(ValueError):
        track_to_ground_truth([], video_id="a", video_url="b", city="c")


# ---------------------------------------------------------------------------
# _reject_jumps — a wrong FIRST fix must not poison the track's start
# ---------------------------------------------------------------------------


def test_reject_jumps_drops_bad_first_fix() -> None:
    # Fix #0 has a single-digit OCR error putting it ~1.1 km north — inside
    # the 30 km median gate, but far from where the clip actually starts.
    # It must be dropped and the correct successors kept, not the reverse.
    bad = GpsFix(0.0, 51.5370, -0.1318)
    good = [GpsFix(2.0 + 2 * i, 51.5270 - i * 2e-4, -0.1318 + i * 2e-4)
            for i in range(5)]
    out = _reject_jumps([bad] + good, max_jump_m=400.0)
    lats = [round(f.lat, 4) for f in out]
    assert 51.5370 not in lats              # bogus first fix rejected
    assert len(out) == 5                    # every correct fix survives
    assert out[0].lat == pytest.approx(51.5270, abs=1e-4)


def test_reject_jumps_keeps_consistent_first_fix() -> None:
    fixes = [GpsFix(2.0 * i, 51.5270 - i * 2e-4, -0.1318 + i * 2e-4)
             for i in range(6)]
    assert _reject_jumps(fixes, max_jump_m=400.0) == fixes


# ---------------------------------------------------------------------------
# osm_around_for_track — the disc must cover every fix
# ---------------------------------------------------------------------------


def _dist_to_center_m(f: GpsFix, clat: float, clon: float) -> float:
    dlat = (f.lat - clat) * 111320.0
    dlon = (f.lon - clon) * 111320.0 * np.cos(np.radians(clat))
    return float(np.hypot(dlat, dlon))


def test_osm_around_covers_time_clumped_track() -> None:
    # Stop-and-go: 9 fixes piled at one end (stopped at a light), 1 fix
    # ~2.2 km away. The mean-centred/bbox-half-diagonal disc would leave
    # the far fix outside; the fixed disc must cover every fix.
    fixes = [GpsFix(float(i), 49.0, 8.4) for i in range(9)]
    fixes.append(GpsFix(9.0, 49.02, 8.4))
    clat, clon, radius = osm_around_for_track(fixes, margin_m=100.0)
    for f in fixes:
        assert _dist_to_center_m(f, clat, clon) <= radius
