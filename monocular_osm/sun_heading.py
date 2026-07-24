"""Absolute camera heading from the sun — a per-video capability that activates
when (and only when) the clip carries a usable capture time.

The shape-match / VO->geo georeference has a FREE rotation DOF (absolute heading
unknown), which lets wrong parallel streets win. If we know the capture instant we
can compute the sun's TRUE azimuth (pysolar, exact) and, by finding the sun's
direction in the image, recover the camera's ABSOLUTE heading — pinning that DOF.

It is gated on information most YouTube re-uploads lack but most RAW dashcam files
have: a capture timestamp, taken from (1) container `creation_time` metadata, or
(2) a burned-in on-screen clock (OCR'd). With no timestamp it returns ``None`` and
the pipeline proceeds unchanged. The sun direction is read from the image (brightest
sky blob) and VALIDATED against the astronomically-known sun elevation, so a white
building/sign is rejected. Returns a per-frame heading + confidence.
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
import zoneinfo

import numpy as np

# Time separators are OCR-tolerant ([:.\s] — a dashcam ':' is often read as '.').
_DT_PATTERNS = [
    re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\D{1,3}(\d{1,2})[:.\s](\d{2})[:.\s](\d{2})"),
    re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\D{1,3}(\d{1,2})[:.\s](\d{2})[:.\s](\d{2})"),
]


def _ffprobe_creation_time(path):
    """Container/stream creation_time -> tz-aware UTC datetime, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format_tags=creation_time:stream_tags=creation_time",
             "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "creation_time=" in line:
            v = line.split("=", 1)[1].strip()
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    d = _dt.datetime.strptime(v, fmt)
                    if d.tzinfo is None:
                        d = d.replace(tzinfo=_dt.timezone.utc)
                    if 2005 < d.year < 2035:
                        return d.astimezone(_dt.timezone.utc)
                except ValueError:
                    continue
    return None


def _parse_clock(text):
    for i, pat in enumerate(_DT_PATTERNS):
        m = pat.search(text)
        if not m:
            continue
        g = list(map(int, m.groups()))
        if i == 0:
            y, mo, da, hh, mm, ss = g
        else:
            # NN/NN/YYYY is ambiguous between EU day-first and US month-first;
            # a wrong date silently skews the sun azimuth by tens of degrees.
            a, b, y, hh, mm, ss = g
            if a > 12:                       # month can't exceed 12 -> day-first
                da, mo = a, b
            elif b > 12:                     # -> month-first (US clock)
                mo, da = a, b
            elif a == b:                     # same date either way
                mo = da = a
            else:
                # Genuinely ambiguous: a guessed date is worse than none (the
                # whole point of the channel is a TRUSTED absolute heading).
                continue
        try:
            if 2005 < y < 2035 and 1 <= mo <= 12 and 1 <= da <= 31 and hh < 24:
                return _dt.datetime(y, mo, da, hh, mm, ss)
        except ValueError:
            return None
    return None


def _ocr_clock_datetime(path, lat, lon):
    """Best-effort burned-in clock -> tz-aware UTC datetime (anchored at t=0), or None."""
    try:
        import cv2
        import easyocr
    except Exception:
        return None
    try:
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        hits = []                                    # (t_sec, parsed_naive_dt)
        for t in (5, 30, 90):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            h, w = f.shape[:2]
            strips = [f[int(0.86 * h):], f[:int(0.14 * h)]]   # clocks live top/bottom
            txt = " ".join(s for st in strips
                           for (_, s, c) in reader.readtext(st) if c > 0.3)
            d = _parse_clock(txt)
            if d:
                hits.append((t, d))
        cap.release()
        if len(hits) < 2:
            return None
        # consistency: the clock must advance ~in step with video time
        (t0, d0), (t1, d1) = hits[0], hits[-1]
        if abs((d1 - d0).total_seconds() - (t1 - t0)) > 60:
            return None
        tz = zoneinfo.ZoneInfo(TimezoneFinder_at(lat, lon))
        return (d0 - _dt.timedelta(seconds=t0)).replace(tzinfo=tz).astimezone(_dt.timezone.utc)
    except Exception:
        return None


def TimezoneFinder_at(lat, lon):
    from timezonefinder import TimezoneFinder
    return TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"


def capture_datetime(path, lat, lon):
    """Absolute capture instant (tz-aware UTC) at t=0 of the clip + source, or None."""
    d = _ffprobe_creation_time(path)
    if d is not None:
        return d, "metadata"
    d = _ocr_clock_datetime(path, lat, lon)
    if d is not None:
        return d, "burned-in-clock"
    return None


def sun_az_alt(lat, lon, dt_utc):
    from pysolar.solar import get_altitude, get_azimuth
    return get_azimuth(lat, lon, dt_utc), get_altitude(lat, lon, dt_utc)


def detect_sun_bearing(bgr, focal_px):
    """Brightest sky blob -> (rel_bearing_deg, image_alt_deg, conf), or None.

    rel_bearing: sun's horizontal angle right of the optical axis.
    image_alt:   sun's angle above the optical axis (≈ elevation for a level cam).
    """
    import cv2
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 248).astype(np.uint8)
    mask[int(0.6 * h):] = 0                          # sky only
    if mask.sum() < 15:
        return None
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    cx, cy = cent[i]
    rel_bearing = np.degrees(np.arctan((cx - w / 2) / focal_px))
    image_alt = np.degrees(np.arctan((h / 2 - cy) / focal_px))
    conf = float(min(1.0, area / (0.0006 * w * h)))
    return float(rel_bearing), float(image_alt), conf


def estimate_heading(video_path, center_latlon, times_sec, focal_px=None,
                     cam_pitch_deg=0.0, elev_tol_deg=18.0, assume_capture_utc=None):
    """Per-frame absolute vehicle heading from the sun, or ``None`` if unavailable.

    ``assume_capture_utc`` (tz-aware datetime) overrides timestamp extraction — for
    when the user knows the capture time, or to validate the image side.
    Returns a dict: available, source, capture_utc, headings (compass deg per time,
    nan where unusable), median_heading, n_used, confidence.
    """
    try:
        import cv2
    except Exception:
        return None
    lat, lon = center_latlon
    if assume_capture_utc is not None:
        dt0, source = assume_capture_utc.astimezone(_dt.timezone.utc), "assumed"
    else:
        cap_dt = capture_datetime(video_path, lat, lon)
        if cap_dt is None:
            return {"available": False, "reason": "no capture timestamp (metadata or clock)"}
        dt0, source = cap_dt
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"available": False, "reason": "cannot open video"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    if focal_px is None:
        focal_px = w / (2 * np.tan(np.deg2rad(70.0) / 2))     # rough default FOV
    headings, confs = [], []
    for t in times_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if not ok:
            headings.append(np.nan); confs.append(0.0); continue
        dt = dt0 + _dt.timedelta(seconds=float(t))
        az, alt = sun_az_alt(lat, lon, dt)
        det = detect_sun_bearing(f, focal_px)
        if det is None or alt < 3:                            # sun down / not found
            headings.append(np.nan); confs.append(0.0); continue
        rel, img_alt, conf = det
        if abs((img_alt + cam_pitch_deg) - alt) > elev_tol_deg:  # not the sun -> reject
            headings.append(np.nan); confs.append(0.0); continue
        headings.append(float((az - rel) % 360.0)); confs.append(conf)
    cap.release()
    headings = np.array(headings); confs = np.array(confs)
    good = np.isfinite(headings) & (confs > 0)
    if good.sum() == 0:
        return {"available": False, "source": source, "capture_utc": dt0.isoformat(),
                "reason": "timestamp ok but sun not detectable in frames (overcast / out of view)"}
    ang = np.radians(headings[good])
    med = float((np.degrees(np.arctan2(np.average(np.sin(ang), weights=confs[good]),
                                       np.average(np.cos(ang), weights=confs[good])))) % 360)
    return {"available": True, "source": source, "capture_utc": dt0.isoformat(),
            "headings": headings, "median_heading": med, "n_used": int(good.sum()),
            "confidence": float(confs[good].mean())}
