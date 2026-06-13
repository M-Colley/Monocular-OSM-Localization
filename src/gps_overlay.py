"""Extract ground truth from dashcam videos with a burned-in GPS overlay.

Consumer dashcams (VIOFO, Garmin, BlackVue, Nextbase, …) routinely stamp
the live position into a corner of the frame — e.g. ``N:53.8235
E:10.5033  102KM/H``. That overlay is, in effect, free per-frame ground
truth: OCR it, parse the coordinates, and you get a GPS track without any
manual labelling. This is how we scale localization validation from a
couple of hand-labelled clips to as many overlay clips as we can find.

This module is two layers:

* :func:`parse_latlon` — a robust, fully-deterministic parser for the
  common overlay coordinate formats (hemisphere-prefixed decimal, signed
  decimal pair, degrees-minutes-seconds), tolerant of the usual OCR
  noise (``:`` vs space, comma decimals, O-for-0). This is the testable
  core.
* :func:`extract_gps_track` — samples frames, OCRs the overlay region,
  parses each, and returns a time-stamped, sanity-filtered track. The
  OCR reader is injectable so it's testable without easyocr, and it
  reuses the same engine as :mod:`scene_text`.

:func:`track_to_ground_truth` then emits the project's standard
``ground_truth/*.json`` schema, so an overlay clip drops straight into
the existing ``--ground-truth-waypoints`` evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .scene_text import OcrReader


@dataclass(frozen=True)
class GpsFix:
    t_sec: float
    lat: float
    lon: float


# --- coordinate parsing ----------------------------------------------------

# Hemisphere-prefixed decimal: "N:53.8235", "N 53,8235", "S53.8235".
# OCR sometimes reads the decimal point as a comma; allow both.
_HEMI = re.compile(
    r"\b([NSEW])\s*[:=]?\s*(\d{1,3}(?:[.,]\d{3,7}))", re.IGNORECASE)
# Signed decimal pair: "51.527047, -0.131824" / "51.5270 -0.1318".
_PAIR = re.compile(
    r"(-?\d{1,2}\.\d{3,7})\s*[,;\s]\s*(-?\d{1,3}\.\d{3,7})")
# Degrees-minutes-seconds: 51°31'37.4"N  (° may OCR as o/º/*, etc.)
_DMS = re.compile(
    r"(\d{1,3})\s*[°ºo*]\s*(\d{1,2})\s*['’′]\s*(\d{1,2}(?:\.\d+)?)\s*[\"”″]?\s*([NSEW])",
    re.IGNORECASE)


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (
        abs(lat) < 1e-6 and abs(lon) < 1e-6)  # reject the 0,0 null island


def _dms_to_deg(d: str, m: str, s: str, hemi: str) -> float:
    deg = float(d) + float(m) / 60.0 + float(s) / 3600.0
    return -deg if hemi.upper() in ("S", "W") else deg


def parse_latlon(text: str) -> tuple[float, float] | None:
    """Parse one overlay string into ``(lat, lon)`` or ``None``.

    Tries hemisphere-prefixed decimal, then DMS, then a signed decimal
    pair. Returns ``None`` if nothing valid is found. Lenient with OCR
    quirks (``:``/space separators, comma decimals) but strict on the
    final WGS84 range check so garbage doesn't slip through.
    """
    if not text:
        return None

    # 1. Hemisphere-prefixed decimals (most common dashcam style).
    lat = lon = None
    for hemi, num in _HEMI.findall(text):
        val = float(num.replace(",", "."))
        h = hemi.upper()
        if h in ("N", "S") and lat is None:
            lat = -val if h == "S" else val
        elif h in ("E", "W") and lon is None:
            lon = -val if h == "W" else val
    if lat is not None and lon is not None and _valid(lat, lon):
        return lat, lon

    # 2. Degrees-minutes-seconds.
    dms = _DMS.findall(text)
    if len(dms) >= 2:
        vals = {}
        for d, m, s, hemi in dms:
            h = hemi.upper()
            axis = "lat" if h in ("N", "S") else "lon"
            vals.setdefault(axis, _dms_to_deg(d, m, s, h))
        if "lat" in vals and "lon" in vals and _valid(vals["lat"], vals["lon"]):
            return vals["lat"], vals["lon"]

    # 3. Signed decimal pair (lat first).
    m = _PAIR.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a) <= 90 and abs(b) <= 180 and _valid(a, b):
            return a, b
    return None


# --- track extraction ------------------------------------------------------


def _crop_region(image, region: str):
    """Crop the overlay strip. Dashcam overlays sit in a thin top or
    bottom band; cropping there both speeds OCR and avoids scene text."""
    h = image.shape[0]
    if region == "bottom":
        return image[int(h * 0.82):, :]
    if region == "top":
        return image[:int(h * 0.18), :]
    return image  # "full"


def extract_gps_track(
    video_path: Path,
    *,
    sample_interval_sec: float = 2.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    region: str = "bottom",
    languages: tuple[str, ...] = ("en",),
    min_confidence: float = 0.2,
    ocr_reader: OcrReader | None = None,
    use_gpu: bool = True,
    frame_reader: Callable[..., list] | None = None,
    max_jump_m: float = 400.0,
) -> list[GpsFix]:
    """OCR the GPS overlay every ``sample_interval_sec`` → a GPS track.

    Each sampled frame's overlay band is OCR'd, all detected strings are
    concatenated, and :func:`parse_latlon` extracts the fix. Fixes are
    sanity-filtered: invalid ranges dropped, and a fix that jumps more
    than ``max_jump_m`` from the running median of recent fixes is
    rejected as an OCR misread (a single wrong digit moves the point
    kilometres). Returns time-ordered :class:`GpsFix` records.
    """
    from .scene_text import _default_reader, _sample_frames

    frames = (frame_reader or _sample_frames)(
        video_path, start_sec, end_sec, sample_interval_sec)
    reader = ocr_reader or _default_reader(tuple(languages), use_gpu)

    raw: list[GpsFix] = []
    for t_sec, image in frames:
        crop = _crop_region(image, region)
        texts = [str(txt) for (_b, txt, conf) in reader.readtext(crop)
                 if float(conf) >= min_confidence]
        # Try the joined line first (coords often span two boxes), then
        # each box alone.
        for candidate in [" ".join(texts), *texts]:
            ll = parse_latlon(candidate)
            if ll is not None:
                raw.append(GpsFix(float(t_sec), ll[0], ll[1]))
                break

    return _reject_jumps(raw, max_jump_m)


def _haversine_m(a: GpsFix, b: GpsFix) -> float:
    R = 6371000.0
    la1, lo1, la2, lo2 = map(np.radians, [a.lat, a.lon, b.lat, b.lon])
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(h)))


def _reject_jumps(fixes: list[GpsFix], max_jump_m: float) -> list[GpsFix]:
    """Drop fixes inconsistent with their neighbours (OCR digit errors).

    Anchors on the median location of all fixes (robust to a minority of
    wild misreads) and drops any fix farther than a generous bound from
    it, then additionally drops point-to-point teleports.
    """
    if len(fixes) < 3:
        return fixes
    med = GpsFix(0.0, float(np.median([f.lat for f in fixes])),
                 float(np.median([f.lon for f in fixes])))
    # Coarse gate: within ~30 km of the median (a single clip stays local).
    near = [f for f in fixes if _haversine_m(f, med) < 30000.0]
    if len(near) < 2:
        return near
    out = [near[0]]
    for f in near[1:]:
        if _haversine_m(out[-1], f) <= max_jump_m * max(1.0, (f.t_sec - out[-1].t_sec)):
            out.append(f)
    return out


# --- ground-truth emission -------------------------------------------------


def osm_around_for_track(
    fixes: list[GpsFix], *, margin_m: float = 600.0
) -> tuple[float, float, float]:
    """``(center_lat, center_lon, radius_m)`` bounding a track + margin.

    This is the coarse *region prior* for the OSM graph fetch — the same
    role a city name plays, but derived from a GPS track's own extent.
    Shared by the dataset adapters (KITTI, comma2k19) whose ground-truth
    tracks tell us roughly where to pull the road graph.
    """
    if not fixes:
        raise ValueError("no fixes to bound")
    lats = np.array([f.lat for f in fixes])
    lons = np.array([f.lon for f in fixes])
    clat, clon = float(lats.mean()), float(lons.mean())
    dlat_m = (lats.max() - lats.min()) * 111320.0
    dlon_m = (lons.max() - lons.min()) * 111320.0 * np.cos(np.radians(clat))
    radius = float(np.hypot(dlat_m, dlon_m) / 2.0 + margin_m)
    return clat, clon, radius


def track_to_ground_truth(
    fixes: list[GpsFix],
    *,
    video_id: str,
    video_url: str,
    city: str,
    n_waypoints: int = 10,
) -> dict:
    """Convert an extracted track into the project's GT-JSON schema.

    Subsamples to ``n_waypoints`` evenly-spaced fixes (endpoints kept),
    matching ``ground_truth/*.json`` so the clip drops into the existing
    ``--ground-truth-waypoints`` evaluation.
    """
    if not fixes:
        raise ValueError("no GPS fixes to write")
    if len(fixes) <= n_waypoints:
        sel = fixes
    else:
        idx = np.unique(np.linspace(0, len(fixes) - 1, n_waypoints).round().astype(int))
        sel = [fixes[i] for i in idx]
    return {
        "video_id": video_id,
        "video_url": video_url,
        "city": city,
        "vo_segment": f"{int(sel[0].t_sec)}:{int(sel[-1].t_sec)}",
        "description": (
            f"Auto-extracted from a burned-in GPS overlay via OCR "
            f"({len(fixes)} fixes → {len(sel)} waypoints). Verify before trusting."
        ),
        "source": "gps_overlay_ocr",
        "waypoints": [
            {"t_sec": round(f.t_sec, 1), "lat": round(f.lat, 6), "lon": round(f.lon, 6)}
            for f in sel
        ],
    }
