"""Tests for burned-in GPS-overlay extraction (auto ground truth)."""

from __future__ import annotations

from pathlib import Path

import pytest

import numpy as np

from monocular_osm.gps_overlay import (
    GpsFix,
    _reject_jumps,
    extract_gps_track,
    longest_continuous_run,
    normalize_overlay_text,
    osm_around_for_track,
    parse_latlon,
    split_on_discontinuities,
    track_to_ground_truth,
)

# The injected-reader tests script ONE overlay string per frame, so they
# must run a single preprocessing variant — the production default reads
# each band three times (see gps_overlay._VARIANTS).
_ONE_VARIANT = (("native", 0.18, 1.0),)


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
        "2016/08/16 15:03:47 DOD LS460W",           # date + camera model only
        "OOOMPH SUBSCRIBE HDR 12-21-2023 16:02:58",  # overlay minus the coords
    ],
)
def test_parse_latlon_rejects_non_coords(text) -> None:
    assert parse_latlon(text) is None


# --- verbatim easyocr output from the overlay-clip fleet -------------------
#
# Every string below was captured by running easyocr over the real overlay
# band of the named clip (see scripts/overlay_clips.py). They are the
# regression contract for normalize_overlay_text + the DMS dialects: each
# one broke the parser before, and each breaks a different rule.


@pytest.mark.parametrize(
    ("clip", "text", "lat", "lon"),
    [
        # VIOFO A229 Pro — decimal point dropped between the boxes.
        ("vCSEG6KaFng", "OOOMPH N:41.8933 W:87 6216 SUBSCRIBE HDR 12-21-2023 16:02:58",
         41.8933, -87.6216),
        # Same camera, a stray space before the point.
        ("LmIHvLMLqFk", "OOOMPH N:41.8856 W:87 .6250 SUBSCRIBE HDR 12-16-2023 22:51:11",
         41.8856, -87.6250),
        # WolfBox i07 — separators spaced out, fraction split across boxes.
        ("kxEDNj5L_yQ", "2023/07 /25 04 : 08 29 65 mp h N : 33 . 853474 W : 84 . 43164 1",
         33.853474, -84.431641),
        ("y66ZkRpUh4k", "2023707 | 24 08 : 57 : 16 0 nnn N : 33 773609 W 84, 371269",
         33.773609, -84.371269),
        # ROVE R2-4K — clean hemisphere-prefixed decimals, 3-digit longitude.
        ("wJsEQTCAg1c", "30 28MPH N33.49497 W111.91312", 33.49497, -111.91312),
        # DOD LS460W — hemisphere-PREFIX DMS. The W's degree sign is read as
        # an 8 ("W838"), and the trailing W must not be stolen by the N triple.
        ("ZhGb8q1kliY", "2016/08/16 15:03:47 DOD LS460W N42 20' 18. 07\" W838 3' 23. 70\"",
         42.33835, -83.05658),
        ("1nF_7l07i-E", "2016/08/15 21:36 : 08 DOD LS460W N42 19' 53. 68 w83\" 3' 11. 80",
         42.33158, -83.05328),
        # YouTube-Capture overlay — hemisphere-SUFFIX DMS with no degree sign
        # at all, and the latitude's apostrophe read as a 1 ("4222159\"N").
        ("g5lnpYCk1Ec", "4222159\"N 83 10\"35\"W E 15.9 kmlh", 42.38306, -83.17639),
        # Nextbase NBDVR402G, Newcastle UK — the only SINGLE-DIGIT degree
        # longitude in the fleet. Every US clip sits at 83-111 degrees west, so
        # a pattern that assumed 2-3 degree digits would pass all of them and
        # still fail here.
        ("Wrn4_uCxRCQ",
         "14/01/2000 02:10:17 NBDVR402G 2MPH N54°58'55.29\" W1°36'24.77\"",
         54.98202, -1.60688),
        # ...and with the seconds' closing quote dropped, which OCR does often.
        ("Wrn4_uCxRCQ",
         "14/01/2000 02:15:17 NBDVR402G 4MPH N54°59'19.92\" W1°35'22.16",
         54.98887, -1.58949),
    ],
)
def test_parse_latlon_real_ocr_output(clip, text, lat, lon) -> None:
    out = parse_latlon(text)
    assert out is not None, f"{clip}: failed to parse {text!r}"
    assert out[0] == pytest.approx(lat, abs=2e-3)
    assert out[1] == pytest.approx(lon, abs=2e-3)


# --- the g5lnpYCk1Ec regression: a WRONG-but-plausible parse ---------------
#
# This clip's stamp reads "42 22'59\"N  83 10'35\"W" and OCR mangles the
# punctuation differently frame to frame. The first implementation parsed
# several of those spellings into coordinates that looked perfectly ordinary
# and sat 37 km south of the truth — the worst failure mode for auto-extracted
# ground truth, because nothing downstream can tell it is wrong. Every string
# below is verbatim easyocr output from that clip.


@pytest.mark.parametrize(
    ("text", "lat", "lon"),
    [
        # Apostrophe survived: degrees run into minutes ("4222'59" = 422 + 2),
        # and the fix is to shift the digit into the minutes, not drop it.
        ("4222'59\"N 83 10\"35\"W 4F", 42.38306, -83.17639),
        ("4222\"59\"N 83 10\"35\"W 4F", 42.38306, -83.17639),
        # Apostrophe read as a 1.
        ("4222159\"N 83 10\"35\"W E 15.9 kml h", 42.38306, -83.17639),
        ("4222126\"N 83 08\"47\"W S 104.8 kml h", 42.37389, -83.14639),
        ("4223122\"N 83 11\"25\"W SE 60.6 km h", 42.38944, -83.19028),
        # Degrees kept their own token, minutes+seconds ran together.
        ("42 23122\"N 83 11\"25\"W", 42.38944, -83.19028),
        ("4223\"34\"N 83 12109\"W N", 42.39278, -83.2025),
    ],
)
def test_parse_latlon_runtogether_dms(text, lat, lon) -> None:
    out = parse_latlon(text)
    assert out is not None, f"failed to parse {text!r}"
    assert out[0] == pytest.approx(lat, abs=2e-4)
    assert out[1] == pytest.approx(lon, abs=2e-4)


def test_degree_overflow_shift_beats_truncation_only_when_legal() -> None:
    # Two causes of overflowing degrees need opposite repairs, and the
    # discriminator is whether the shifted minutes stay under 60.
    shift = parse_latlon("4222'59\"N 83 10'35\"W")        # 422 -> 42 deg 22'
    assert shift is not None and shift[0] == pytest.approx(42.38306, abs=2e-4)

    truncate = parse_latlon("N42 20' 18.07\" W838 3' 23.70\"")   # 838 -> 83 deg 3'
    assert truncate is not None
    assert truncate[1] == pytest.approx(-83.05658, abs=2e-4)


def test_lost_decimal_point_repair_does_not_eat_a_dms() -> None:
    # "N 83 12109\"W" is 83 deg 12' 09", not the decimal 83.12109 — the two
    # differ by 1.6 km and the decimal reading looks entirely plausible.
    out = parse_latlon("4223\"34\"N 83 12109\"W N")
    assert out is not None
    assert out[1] == pytest.approx(-83.2025, abs=2e-4)
    # ...while the repair it exists for still works.
    viofo = parse_latlon("OOOMPH N:41.8933 W:87 6216 SUBSCRIBE HDR")
    assert viofo is not None and viofo[1] == pytest.approx(-87.6216, abs=1e-4)


def test_dms_prefix_does_not_steal_the_next_hemisphere() -> None:
    # The bug a single prefix-or-suffix pattern causes: the N triple's
    # trailing quote is followed by " W", so a permissive pattern consumes
    # the W as its own suffix and the longitude vanishes.
    out = parse_latlon("N42 20' 18.07\" W83 3' 23.70\"")
    assert out is not None
    assert out[0] == pytest.approx(42.33835, abs=1e-4)
    assert out[1] == pytest.approx(-83.05658, abs=1e-4)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("N : 33 . 853474", "N:33.853474"),        # spaces around separators
        ("W : 84 . 43164 1", "W:84.431641"),       # fraction split in two
        ("N:87 6216", "N:87.6216"),                # decimal point dropped
        ("4222159\"N", "42 22'59\"N"),             # DMS punctuation eaten
        ("12-21-2023 16:02:58", "12-21-2023 16:02:58"),   # clock/date untouched
        ("N42 20' 18.07\"", "N42 20' 18.07\""),    # already-clean DMS untouched
    ],
)
def test_normalize_overlay_text(raw, expected) -> None:
    assert normalize_overlay_text(raw) == expected
    # Idempotent: a second pass must not corrupt an already-normalised string.
    assert normalize_overlay_text(expected) == expected


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
        frame_reader=_frames([0.0, 2.0, 4.0]), variants=_ONE_VARIANT,
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
        variants=_ONE_VARIANT,
    )
    lats = [round(f.lat, 4) for f in track]
    assert 51.587 not in lats           # the jump was rejected
    assert len(track) == 4


def test_extract_track_handles_frames_without_overlay(tmp_path: Path) -> None:
    lines = ["N:51.5270 W:0.1318", "", "N:51.5260 W:0.1300"]
    track = extract_gps_track(
        tmp_path / "v.mp4", ocr_reader=_ScriptedReader(lines),
        frame_reader=_frames([0.0, 2.0, 4.0]), variants=_ONE_VARIANT,
    )
    assert len(track) == 2   # the blank frame contributes nothing, no crash


class _PerVariantReader:
    """Returns a different string on each successive readtext call.

    Models the production path, where one frame's band is OCR'd once per
    preprocessing variant and the passes disagree.
    """

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0

    def readtext(self, image):
        line = self._lines[self._i % len(self._lines)]
        self._i += 1
        return [([], line, 0.9)] if line else []


def test_variants_outvote_a_single_bad_pass(tmp_path: Path) -> None:
    # Pass 1 misreads a digit (~2.2 km north); passes 2 and 3 agree. The
    # agreeing pair must win rather than the highest-priority variant.
    two_variants_agree = [
        "N:51.5470 W:0.1318",   # variant "native" — wrong
        "N:51.5270 W:0.1318",   # variant "x2"
        "N:51.5270 W:0.1318",   # variant "tight-x3"
    ]
    track = extract_gps_track(
        tmp_path / "v.mp4", ocr_reader=_PerVariantReader(two_variants_agree),
        frame_reader=_frames([0.0]),
    )
    assert len(track) == 1
    assert track[0].lat == pytest.approx(51.5270, abs=1e-4)


def test_variants_fall_back_to_first_reading_when_all_disagree(tmp_path: Path) -> None:
    all_disagree = [
        "N:51.5270 W:0.1318",
        "N:51.5470 W:0.1318",
        "N:51.5670 W:0.1318",
    ]
    track = extract_gps_track(
        tmp_path / "v.mp4", ocr_reader=_PerVariantReader(all_disagree),
        frame_reader=_frames([0.0]),
    )
    assert len(track) == 1
    assert track[0].lat == pytest.approx(51.5270, abs=1e-4)


class _CountingReader(_ScriptedReader):
    """Scripted reader that records how many times it was asked to OCR."""

    calls = 0

    def readtext(self, image):
        type(self).calls += 1
        return super().readtext(image)


def test_track_cache_round_trips_and_skips_the_ocr(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")            # the cache key includes size/mtime
    cache = tmp_path / "track.json"
    lines = ["N:51.5270 W:0.1318", "N:51.5260 W:0.1300"]

    class R1(_CountingReader):
        calls = 0

    first = extract_gps_track(
        video, ocr_reader=R1(lines), frame_reader=_frames([0.0, 2.0]),
        variants=_ONE_VARIANT, cache_path=cache)
    assert R1.calls == 2 and cache.exists()

    class R2(_CountingReader):
        calls = 0

    second = extract_gps_track(
        video, ocr_reader=R2(lines), frame_reader=_frames([0.0, 2.0]),
        variants=_ONE_VARIANT, cache_path=cache)
    assert R2.calls == 0                   # served entirely from cache
    assert [(f.t_sec, f.lat, f.lon) for f in second] == \
        [(f.t_sec, f.lat, f.lon) for f in first]


def test_track_cache_regenerates_when_the_settings_change(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    cache = tmp_path / "track.json"
    lines = ["N:51.5270 W:0.1318", "N:51.5260 W:0.1300"]

    extract_gps_track(video, ocr_reader=_ScriptedReader(lines),
                      frame_reader=_frames([0.0, 2.0]), variants=_ONE_VARIANT,
                      cache_path=cache)

    class R(_CountingReader):
        calls = 0

    # A different band is a different read of the same video — the cached
    # track must NOT be served for it.
    extract_gps_track(video, ocr_reader=R(lines), frame_reader=_frames([0.0, 2.0]),
                      variants=_ONE_VARIANT, cache_path=cache, region="top")
    assert R.calls == 2


def test_track_cache_regenerates_when_the_video_changes(tmp_path: Path) -> None:
    # Re-downloading the same clip at another resolution into the same path
    # must invalidate, not silently serve a track OCR'd from the old file.
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    cache = tmp_path / "track.json"
    lines = ["N:51.5270 W:0.1318"]
    extract_gps_track(video, ocr_reader=_ScriptedReader(lines),
                      frame_reader=_frames([0.0]), variants=_ONE_VARIANT,
                      cache_path=cache)

    video.write_bytes(b"\x00" * 4096)      # different size

    class R(_CountingReader):
        calls = 0

    extract_gps_track(video, ocr_reader=R(lines), frame_reader=_frames([0.0]),
                      variants=_ONE_VARIANT, cache_path=cache)
    assert R.calls == 1


def test_variants_recover_a_frame_the_first_pass_cannot_read(tmp_path: Path) -> None:
    # The native pass reads nothing parseable; a later variant does.
    track = extract_gps_track(
        tmp_path / "v.mp4",
        ocr_reader=_PerVariantReader(["OOOMPH HDR SUBSCRIBE", "N:51.5270 W:0.1318", ""]),
        frame_reader=_frames([0.0]),
    )
    assert len(track) == 1
    assert track[0].lon == pytest.approx(-0.1318, abs=1e-4)


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


# ---------------------------------------------------------------------------
# edit cuts — an uploaded clip is often several drives spliced together
# ---------------------------------------------------------------------------


def _leg(t0: float, lat: float, lon: float, n: int, dlat: float, dt: float) -> list[GpsFix]:
    return [GpsFix(t0 + i * dt, lat + i * dlat, lon) for i in range(n)]


def test_split_on_discontinuities_leaves_a_real_drive_intact() -> None:
    # 5e-4 deg of latitude per 5 s is ~55 m == 40 km/h, ordinary driving.
    drive = _leg(0.0, 33.49, -111.91, 12, 5e-4, 5.0)
    assert len(split_on_discontinuities(drive)) == 1


def test_split_on_discontinuities_finds_the_cut() -> None:
    # The real failure this was written for: the Phoenix upload
    # (wJsEQTCAg1c) is a compilation, and the extracted track jumped
    # 19.5 km in 140 s == 503 km/h between two separate drives.
    first = _leg(0.0, 33.4950, -111.9100, 6, 5e-4, 5.0)
    second = _leg(140.0, 33.6717, -111.9685, 6, 5e-4, 5.0)   # ~19.6 km north
    runs = split_on_discontinuities(first + second)
    assert len(runs) == 2
    assert len(runs[0]) == 6 and len(runs[1]) == 6


def test_longest_continuous_run_keeps_the_longer_drive() -> None:
    short = _leg(0.0, 33.4950, -111.9100, 3, 5e-4, 5.0)       # 10 s
    long_ = _leg(200.0, 33.6717, -111.9685, 20, 5e-4, 5.0)    # 95 s
    out = longest_continuous_run(short + long_)
    assert len(out) == 20
    assert out[0].t_sec == 200.0


def test_longest_continuous_run_measures_time_not_fix_count() -> None:
    # A patchy-OCR stretch covering more of the drive beats a dense short
    # one — the analysis window is what matters, not the sample count.
    dense_short = [GpsFix(i * 2.0, 33.4950 + i * 2e-4, -111.91) for i in range(8)]   # 14 s
    sparse_long = [GpsFix(300.0 + i * 30.0, 33.6717 + i * 3e-3, -111.96)
                   for i in range(5)]                                                # 120 s
    out = longest_continuous_run(dense_short + sparse_long)
    assert len(out) == 5
    assert out[0].t_sec == 300.0


def test_split_on_discontinuities_tolerates_freeway_speed() -> None:
    # 120 km/h between 5 s samples is 167 m — driving, not a cut.
    fast = [GpsFix(i * 5.0, 33.49 + i * 1.5e-3, -111.91) for i in range(10)]
    assert len(split_on_discontinuities(fast)) == 1


def test_split_on_discontinuities_handles_empty_and_single() -> None:
    assert split_on_discontinuities([]) == []
    one = [GpsFix(0.0, 33.49, -111.91)]
    assert split_on_discontinuities(one) == [one]
    assert longest_continuous_run([]) == []


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
