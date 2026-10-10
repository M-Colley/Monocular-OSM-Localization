"""Unit tests for the sun-heading capability (pure logic; no video/GPU)."""

from __future__ import annotations

import datetime
import zoneinfo

import numpy as np

from monocular_osm.sun_heading import _parse_clock, detect_sun_bearing, sun_az_alt


def test_parse_clock_formats():
    assert _parse_clock("speed 50  2023-07-15 13:45:09 cam") == datetime.datetime(2023, 7, 15, 13, 45, 9)
    assert _parse_clock("15/07/2023 13:45:09") == datetime.datetime(2023, 7, 15, 13, 45, 9)
    assert _parse_clock("no clock here") is None
    assert _parse_clock("1999-01-01 00:00:00") is None       # out of plausible range


def test_parse_clock_disambiguates_day_month_order():
    # US month-first clock: day > 12 proves the order (the old day-first-only
    # parse rejected this as month 13 and lost the channel on US clips).
    assert _parse_clock("06/13/2023 16:00:00") == datetime.datetime(2023, 6, 13, 16, 0, 0)
    # EU day-first stays supported (day > 12 proves it, see above).
    assert _parse_clock("25/06/2023 09:30:00") == datetime.datetime(2023, 6, 25, 9, 30, 0)
    # Same value either way -> unambiguous.
    assert _parse_clock("07/07/2023 12:00:00") == datetime.datetime(2023, 7, 7, 12, 0, 0)


def test_parse_clock_rejects_ambiguous_dates():
    # 06/10 is June 10 (US) or 6 October (EU): a wrong guess skews the sun
    # azimuth by tens of degrees, so ambiguity must yield None, not a guess
    # (the old code silently picked day-first).
    assert _parse_clock("06/10/2023 16:00:00") is None
    assert _parse_clock("01/02/2024 08:00:00") is None


def test_sun_az_alt_known():
    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    dt = datetime.datetime(2023, 6, 21, 13, 30, tzinfo=tz)    # near summer solstice noon
    az, alt = sun_az_alt(48.40, 9.99, dt)
    assert 150 < az < 200          # roughly south
    assert 58 < alt < 66           # high summer sun in Ulm
    # early morning -> sun in the east, low
    az2, alt2 = sun_az_alt(48.40, 9.99, datetime.datetime(2023, 6, 21, 6, 0, tzinfo=tz))
    assert 45 < az2 < 120 and alt2 < 30        # low sun in the NE/E at 06:00


def test_detect_sun_bearing_synthetic():
    img = np.zeros((480, 640, 3), np.uint8)
    # bright sun disk in the upper-right of the sky region
    import cv2
    cv2.circle(img, (480, 120), 14, (255, 255, 255), -1)
    det = detect_sun_bearing(img, focal_px=500.0)
    assert det is not None
    rel, alt, conf = det
    assert rel > 0           # sun right of centre -> positive bearing
    assert alt > 0           # above the optical axis
    assert conf > 0
    # no bright blob -> None
    assert detect_sun_bearing(np.zeros((480, 640, 3), np.uint8), 500.0) is None


# ---------------------------------------------------------------------------
# Burned-in clock OCR goes through the project's OCR-engine factory
# ---------------------------------------------------------------------------


class _FakeCapture:
    """cv2.VideoCapture stand-in: every read returns a blank 1080p frame."""

    def __init__(self, path):
        self.path = path

    def get(self, prop):
        return 30.0

    def set(self, prop, value):
        return True

    def read(self):
        return True, np.zeros((1080, 1920, 3), np.uint8)

    def release(self):
        pass


class _ClockReader:
    """Reads a clock that advances with video time on the BOTTOM strip
    (2 readtext calls per frame: bottom, top; frames at t = 5, 30, 90 s)."""

    def __init__(self):
        self.calls = 0

    def readtext(self, image):
        i, self.calls = self.calls, self.calls + 1
        if i % 2:
            return []                                      # top strip: nothing
        t = (5, 30, 90)[i // 2]
        stamp = datetime.datetime(2023, 7, 15, 12, 0, 0) + datetime.timedelta(seconds=t)
        return [([[0, 0], [1, 0], [1, 1], [0, 1]],
                 stamp.strftime("%Y-%m-%d %H:%M:%S"), 0.9)]


def _patch_video(monkeypatch):
    import cv2

    import monocular_osm.sun_heading as sh

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(sh, "TimezoneFinder_at", lambda lat, lon: "Europe/Berlin")
    return sh


def test_ocr_clock_uses_the_configured_engine(monkeypatch):
    """The clock read must come from scene_text's engine factory (rapidocr by
    default, honouring --ocr-engine) -- it used to import easyocr directly and
    silently return None on a default (rapidocr-only) install."""
    sh = _patch_video(monkeypatch)
    import monocular_osm.scene_text as st

    calls = []

    def fake_default_reader(languages, use_gpu, engine="rapidocr"):
        calls.append((tuple(languages), use_gpu, engine))
        return _ClockReader()

    monkeypatch.setattr(st, "_default_reader", fake_default_reader)
    got = sh._ocr_clock_datetime("clip.mp4", 48.4, 9.99)
    # 12:00:05 CEST at t=5 s -> t=0 is 12:00:00 CEST = 10:00:00 UTC.
    assert got == datetime.datetime(2023, 7, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
    assert calls == [(("en",), True, "rapidocr")]

    calls.clear()
    sh._ocr_clock_datetime("clip.mp4", 48.4, 9.99, ocr_engine="easyocr", ocr_use_gpu=False)
    assert calls == [(("en",), False, "easyocr")]


def test_ocr_clock_injected_reader_and_missing_backend(monkeypatch):
    sh = _patch_video(monkeypatch)
    import monocular_osm.scene_text as st

    got = sh._ocr_clock_datetime("clip.mp4", 48.4, 9.99, ocr_reader=_ClockReader())
    assert got == datetime.datetime(2023, 7, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)

    def no_backend(languages, use_gpu, engine="rapidocr"):
        raise ImportError("No OCR backend available")

    monkeypatch.setattr(st, "_default_reader", no_backend)
    assert sh._ocr_clock_datetime("clip.mp4", 48.4, 9.99) is None


def test_estimate_heading_threads_ocr_engine(monkeypatch):
    import inspect

    import monocular_osm.sun_heading as sh

    assert inspect.signature(sh.estimate_heading).parameters["ocr_engine"].default == "rapidocr"
    seen = {}

    def fake_capture_datetime(path, lat, lon, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(sh, "capture_datetime", fake_capture_datetime)
    out = sh.estimate_heading("clip.mp4", (48.4, 9.99), [0.0], ocr_engine="easyocr",
                              ocr_use_gpu=False)
    assert out["available"] is False
    assert seen["ocr_engine"] == "easyocr" and seen["ocr_use_gpu"] is False


def test_sun_heading_does_not_import_easyocr_directly():
    import ast
    from pathlib import Path

    import monocular_osm.sun_heading as sh

    tree = ast.parse(Path(sh.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "easyocr" not in imported
