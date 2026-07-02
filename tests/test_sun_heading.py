"""Unit tests for the sun-heading capability (pure logic; no video/GPU)."""

from __future__ import annotations

import datetime
import zoneinfo

import numpy as np

from src.sun_heading import _parse_clock, detect_sun_bearing, sun_az_alt


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
