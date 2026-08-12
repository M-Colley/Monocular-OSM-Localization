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

import json
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


# --- OCR text normalisation ------------------------------------------------
#
# Real easyocr output off a dashcam band is messy in a small number of
# *systematic* ways (measured on the fleet of overlay clips in
# scripts/overlay_clips.py):
#
#   "N : 33 . 853474"        spaces sprayed around the separators
#   "W : 84 . 43164 1"       the fraction split across two detection boxes
#   "N:87 6216"              the decimal point dropped entirely
#   "W838 3' 23.70\""        the degree sign read as the digit 8
#   "4222159\"N"             the DMS apostrophe read as a 1, spaces gone
#
# Normalising these away up front keeps the actual coordinate grammars
# below simple, and every rule is anchored on context that scene text
# and clock/date stamps do not have.

# Unicode look-alikes easyocr emits for the quote/degree marks.
_GLYPHS = str.maketrans({
    "’": "'", "‘": "'", "′": "'", "´": "'", "`": "'",
    "”": '"', "“": '"', "″": '"',
    "º": "°", "˚": "°", "ᵒ": "°", "∘": "°",
})
# "N : 33" -> "N:33". Only around an explicit separator, so hemisphere-prefix
# DMS ("N42 20' ...") keeps the spacing its own grammar reads.
_HEMI_SEP = re.compile(r"\b([NSEW])\s*[:=]\s*(?=\d)", re.IGNORECASE)
# "33 . 853474" / "87 ,6250" -> "33.853474" / "87,6250"
_SPACED_POINT = re.compile(r"(?<=\d)\s*([.,])\s*(?=\d)")
# "84.43164 1" -> "84.431641": once a decimal point has been seen, a
# following bare digit run is the rest of the fraction, not a new number.
# Anchored on the preceding fraction so "83 10'35\"" is untouched.
_SPLIT_FRACTION = re.compile(r"(\d[.,]\d+)\s+(\d+)(?![\d.,])")
# "W:87 6216" -> "W:87.6216": a hemisphere-prefixed integer followed by a
# bare 3-7 digit run is a dropped decimal point. The >=3-digit floor keeps
# it away from DMS minutes ("N42 20'"), which are 1-2 digits.
# The trailing lookahead also excludes a quote: a digit run closed by ' or "
# is a DMS seconds field, not a fraction (defence in depth — the DMS repairs
# above normally consume those first).
_LOST_POINT = re.compile(
    r"(?<![\d.])([NSEW])\s*[:=]?\s*(\d{1,3})\s+(\d{3,7})(?![\d.,'\"|])", re.IGNORECASE)
# "4222159\"N" -> "42 22'59\"N": DDMMSS run whose punctuation the OCR ate
# (the apostrophe typically comes back as a 1). Requires the trailing quote +
# hemisphere so clock/date runs can't match.
_DDMMSS_RUN = re.compile(r"(?<![\d.])(\d{6,7})\s*['\"|]\s*([NSEW])\b", re.IGNORECASE)
# "83 12109\"W" -> "83 12'09\"W": same damage, but the degrees survived as
# their own token so only minutes+seconds ran together.
_MMSS_RUN = re.compile(
    r"(?<![\d.])(\d{1,3})\s+(\d{4,5})\s*['\"|]\s*([NSEW])\b", re.IGNORECASE)


def _dms_or_original(m: re.Match, dd: str, mm: str, ss: str, hemi: str) -> str:
    if int(mm) < 60 and int(ss) < 60:
        return f"{dd} {mm}'{ss}\"{hemi}"
    return m.group(0)


def _expand_ddmmss(m: re.Match) -> str:
    digits, hemi = m.group(1), m.group(2)
    return _dms_or_original(m, digits[:2], digits[2:4], digits[-2:], hemi)


def _expand_mmss(m: re.Match) -> str:
    dd, digits, hemi = m.group(1), m.group(2), m.group(3)
    return _dms_or_original(m, dd, digits[:2], digits[-2:], hemi)


def normalize_overlay_text(text: str) -> str:
    """Repair the systematic OCR damage listed above. Idempotent."""
    if not text:
        return text
    out = text.translate(_GLYPHS)
    out = _HEMI_SEP.sub(r"\1:", out)
    out = _SPACED_POINT.sub(r"\1", out)
    out = _SPLIT_FRACTION.sub(r"\1\2", out)
    # Restore run-together DMS BEFORE the dropped-decimal-point repair. Both
    # look at a bare digit run after a hemisphere, and _LOST_POINT wins ties
    # in the wrong direction: on "N 83 12109\"W" it produced "N:83.12109",
    # turning 83 deg 12' 09" into the decimal 83.12109 and losing 1.6 km.
    out = _DDMMSS_RUN.sub(_expand_ddmmss, out)
    out = _MMSS_RUN.sub(_expand_mmss, out)
    out = _LOST_POINT.sub(r"\1:\2.\3", out)
    return out


# --- coordinate parsing ----------------------------------------------------

# Hemisphere-prefixed decimal: "N:53.8235", "N 53,8235", "S53.8235".
# OCR sometimes reads the decimal point as a comma; allow both.
_HEMI = re.compile(
    r"\b([NSEW])\s*[:=]?\s*(\d{1,3}(?:[.,]\d{3,7}))", re.IGNORECASE)
# Signed decimal pair: "51.527047, -0.131824" / "51.5270 -0.1318".
_PAIR = re.compile(
    r"(-?\d{1,2}\.\d{3,7})\s*[,;\s]\s*(-?\d{1,3}\.\d{3,7})")

# Degrees-minutes-seconds. Two dialects occur in the wild and they must be
# matched by SEPARATE patterns, not one with optional hemispheres on both
# ends: a single pattern would let "N42d 20' 18.07\" W83d ..." swallow the
# *next* token's W as its own trailing hemisphere and lose the longitude.
#
#   suffix dialect (DOD/most cameras): 51°31'37.4"N 0°07'54.6"W
#   prefix dialect (Beytekin/DOD LS460W): N42° 20' 18.07"  W83° 3' 23.70"
#
# The degree mark is optional and deliberately loose: it is frequently
# absent ("42 22'59\"N"), or misread as " or o. The minute mark, by
# contrast, is required — it is the anchor that stops clock/date stamps
# from matching.
_DMS_BODY = (r"(\d{1,3})\s*[\"°oO*]?\s*"      # degrees (+ optional mark)
             r"(\d{1,2})\s*['\"|]\s*"              # minutes + required mark
             r"(\d{1,2}(?:\.\d+)?)\s*['\"|]?")     # seconds (+ optional mark)
_DMS_SUFFIX = re.compile(_DMS_BODY + r"\s*([NSEW])\b", re.IGNORECASE)
_DMS_PREFIX = re.compile(r"([NSEW])\s*" + _DMS_BODY, re.IGNORECASE)

_MAX_DEG = {"lat": 90.0, "lon": 180.0}


def _valid(lat: float, lon: float) -> bool:
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (
        abs(lat) < 1e-6 and abs(lon) < 1e-6)  # reject the 0,0 null island


def _dms_to_deg(d: str, m: str, s: str, hemi: str) -> float:
    deg = float(d) + float(m) / 60.0 + float(s) / 3600.0
    return -deg if hemi.upper() in ("S", "W") else deg


def _dms_component(d: str, m: str, s: str, hemi: str) -> tuple[str, float] | None:
    """One DMS triple -> ``(axis, signed_degrees)``, or ``None`` if absurd.

    Overflowing degrees have two distinct causes, and picking the wrong
    repair silently produces a *plausible* coordinate tens of kilometres
    away — far worse than rejecting the fix:

    * the degrees ran together with the minutes because the pattern
      matched greedily — ``4222'59"N`` gives d=422, m=2, and the truth is
      42°22'59". The trailing degree digit belongs at the FRONT of the
      minutes.
    * the degree sign was read as a digit — ``W83°`` -> ``W838`` gives
      d=838, m=3, and the truth is 83°3'. The trailing digit is noise.

    Try the shift first and keep it only when it yields legal minutes;
    that discriminates the two, because shifting the ``838`` case would
    make minutes 83. Measured on the Detroit clip g5lnpYCk1Ec, getting
    this backwards moved the whole track 37 km south while still looking
    like a perfectly ordinary drive.
    """
    axis = "lat" if hemi.upper() in ("N", "S") else "lon"
    if float(m) >= 60.0 or float(s) >= 60.0:
        return None
    if float(d) > _MAX_DEG[axis] and len(d) > 1:
        shifted_m = d[-1] + m
        if float(shifted_m) < 60.0 and float(d[:-1]) <= _MAX_DEG[axis]:
            d, m = d[:-1], shifted_m
        else:
            d = d[:-1]
    val = _dms_to_deg(d, m, s, hemi)
    return (axis, val) if abs(val) <= _MAX_DEG[axis] else None


def _parse_dms(text: str) -> tuple[float, float] | None:
    """Try each DMS dialect on its own; the first that yields BOTH axes wins.

    Results are never merged across dialects — a prefix-pattern hit on
    suffix-dialect text lands on the wrong token and would silently swap
    latitude for longitude.
    """
    for pattern, hemi_first in ((_DMS_PREFIX, True), (_DMS_SUFFIX, False)):
        vals: dict[str, float] = {}
        for groups in pattern.findall(text):
            hemi, d, m, s = (groups if hemi_first
                             else (groups[3], groups[0], groups[1], groups[2]))
            comp = _dms_component(d, m, s, hemi)
            if comp is not None:
                vals.setdefault(*comp)
        if "lat" in vals and "lon" in vals and _valid(vals["lat"], vals["lon"]):
            return vals["lat"], vals["lon"]
    return None


def parse_latlon(text: str) -> tuple[float, float] | None:
    """Parse one overlay string into ``(lat, lon)`` or ``None``.

    Normalises the usual OCR damage, then tries hemisphere-prefixed
    decimal, then DMS (both hemisphere dialects), then a signed decimal
    pair. Returns ``None`` if nothing valid is found. Lenient with OCR
    quirks (``:``/space separators, comma decimals, eaten punctuation)
    but strict on the final WGS84 range check so garbage doesn't slip
    through.
    """
    if not text:
        return None
    text = normalize_overlay_text(text)

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
    dms = _parse_dms(text)
    if dms is not None:
        return dms

    # 3. Signed decimal pair (lat first).
    m = _PAIR.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a) <= 90 and abs(b) <= 180 and _valid(a, b):
            return a, b
    return None


# --- track extraction ------------------------------------------------------


def _crop_region(image, region: str, frac: float = 0.18):
    """Crop the overlay strip. Dashcam overlays sit in a thin top or
    bottom band; cropping there both speeds OCR and avoids scene text."""
    h = image.shape[0]
    if region == "bottom":
        return image[int(h * (1.0 - frac)):, :]
    if region == "top":
        return image[:int(h * frac), :]
    return image  # "full"


# Preprocessing variants applied to the overlay band, tried in order until
# one parses. No single variant wins everywhere, and the differences are
# not cosmetic — they flip whole clips between "reads" and "doesn't":
#
#   native   DOD LS460W clips keep their DMS punctuation ("N42 20' 18.07\"")
#            that upscaling smears into "N428 20 18.07\"".
#   x2       recovers the decimal point on VIOFO/WolfBox overlays
#            ("W:87 6216" -> "W:87.6216") and the W that the native pass
#            loses entirely on the WolfBox bottom-right stamp.
#   tight-x3 last resort for small overlays: half the band, 3x upscale.
#
# Running all three costs ~3x the OCR of one frame, which is irrelevant at
# the sparse sampling this does (a few hundred frames per clip).
_VARIANTS: tuple[tuple[str, float, float], ...] = (
    ("native", 0.18, 1.0),
    ("x2", 0.18, 2.0),
    ("tight-x3", 0.10, 3.0),
)

# Two variants landing within this distance are treated as agreeing, and
# their mean is used. Well under the ~30 m a vehicle covers between the
# frames these are read from, and far under the km-scale error a single
# misread digit produces.
_AGREE_M = 60.0


def _prepare_band(image, region: str, frac: float, scale: float):
    if scale <= 1.0:
        return _crop_region(image, region, frac)
    from .scene_text import _upscale_sharpen
    return _upscale_sharpen(_crop_region(image, region, frac), scale)


def _read_fix(reader: OcrReader, image, min_confidence: float) -> tuple[float, float] | None:
    """OCR one prepared band and parse the first coordinate out of it."""
    texts = [str(txt) for (_b, txt, conf) in reader.readtext(image)
             if float(conf) >= min_confidence]
    # Try the joined line first (coords often span two boxes), then each
    # box alone.
    for candidate in [" ".join(texts), *texts]:
        ll = parse_latlon(candidate)
        if ll is not None:
            return ll
    return None


def _consensus(fixes: list[tuple[float, float]]) -> tuple[float, float]:
    """Pick a fix from what the variants read, preferring agreement.

    If any two variants land within ``_AGREE_M``, return the mean of that
    agreeing group — a digit misread by one engine pass is outvoted
    rather than merely filtered later. Otherwise fall back to the first
    (highest-priority variant) reading.
    """
    for i, a in enumerate(fixes):
        group = [b for b in fixes[i:]
                 if _haversine_m(GpsFix(0.0, *a), GpsFix(0.0, *b)) <= _AGREE_M]
        if len(group) >= 2:
            return (float(np.mean([g[0] for g in group])),
                    float(np.mean([g[1] for g in group])))
    return fixes[0]


def _track_cache_signature(
    video_path: Path, sample_interval_sec: float, start_sec: float,
    end_sec: float | None, region: str, min_confidence: float,
    max_jump_m: float, variants: tuple[tuple[str, float, float], ...],
    engine: str = "rapidocr",
) -> dict:
    sig = {
        "sample_interval_sec": sample_interval_sec,
        "start_sec": start_sec, "end_sec": end_sec, "region": region,
        "min_confidence": min_confidence, "max_jump_m": max_jump_m,
        "variants": [list(v) for v in variants],
        # Bump when the parser changes in a way that would read a DIFFERENT
        # track off the same pixels, so stale tracks regenerate.
        #   2: run-together DMS repairs (degree/minute shift, MMSS runs) and
        #      the _LOST_POINT reordering — these changed g5lnpYCk1Ec by 37 km.
        "parser": 2,
        "engine": engine,
    }
    p = Path(video_path)
    try:
        st = p.stat()
        sig["video"] = {"name": p.name, "size": int(st.st_size), "mtime": int(st.st_mtime)}
    except OSError:
        sig["video"] = {"name": p.name, "size": None, "mtime": None}
    return sig


def _load_track_cache(cache_path: Path, sig: dict) -> list[GpsFix] | None:
    try:
        blob = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if blob.get("signature") != sig:
        return None
    return [GpsFix(f["t_sec"], f["lat"], f["lon"]) for f in blob.get("fixes", [])]


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
    variants: tuple[tuple[str, float, float], ...] = _VARIANTS,
    cache_path: Path | None = None,
    engine: str = "rapidocr",
) -> list[GpsFix]:
    """OCR the GPS overlay every ``sample_interval_sec`` → a GPS track.

    Each sampled frame's overlay band is OCR'd under several
    preprocessing ``variants`` (see :data:`_VARIANTS`), all detected
    strings are concatenated, and :func:`parse_latlon` extracts the fix.
    Where two variants agree the mean is taken; otherwise the
    highest-priority reading wins. Fixes are then sanity-filtered:
    invalid ranges dropped, and a fix that jumps more than ``max_jump_m``
    from the running median of recent fixes is rejected as an OCR misread
    (a single wrong digit moves the point kilometres). Returns
    time-ordered :class:`GpsFix` records.

    ``cache_path`` memoises the result keyed by the sampling parameters and
    the video's identity. Reading a ten-minute window costs minutes of OCR,
    and every re-run of the fleet builder would otherwise pay it again for
    tracks that cannot have changed.
    """
    from .scene_text import _default_reader, _sample_frames

    sig = _track_cache_signature(video_path, sample_interval_sec, start_sec,
                                 end_sec, region, min_confidence, max_jump_m,
                                 variants, engine)
    if cache_path is not None:
        cached = _load_track_cache(Path(cache_path), sig)
        if cached is not None:
            return cached

    frames = (frame_reader or _sample_frames)(
        video_path, start_sec, end_sec, sample_interval_sec)
    reader = ocr_reader or _default_reader(tuple(languages), use_gpu, engine)

    raw: list[GpsFix] = []
    for t_sec, image in frames:
        read: list[tuple[float, float]] = []
        for _name, frac, scale in variants:
            ll = _read_fix(reader, _prepare_band(image, region, frac, scale),
                           min_confidence)
            if ll is not None:
                read.append(ll)
        if read:
            lat, lon = _consensus(read)
            raw.append(GpsFix(float(t_sec), lat, lon))

    track = _reject_jumps(raw, max_jump_m)
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(
            {"signature": sig,
             "fixes": [{"t_sec": f.t_sec, "lat": f.lat, "lon": f.lon} for f in track]},
            indent=1), encoding="utf-8")
    return track


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
    # Seed from the median of the first few plausible fixes rather than
    # trusting fix #0 outright: a moderately-wrong first fix (inside the
    # 30 km gate but ~1 km off) would otherwise be kept and cause the
    # genuinely-correct successors to be rejected against it.
    k = min(5, len(near))
    seed = GpsFix(0.0, float(np.median([f.lat for f in near[:k]])),
                  float(np.median([f.lon for f in near[:k]])))
    out: list[GpsFix] = []
    for f in near:
        if not out:
            # Generous gate: the vehicle moves < ~max_jump_m within the
            # few samples the seed median spans.
            if _haversine_m(f, seed) <= 2.0 * max_jump_m:
                out.append(f)
            continue
        if _haversine_m(out[-1], f) <= max_jump_m * max(1.0, (f.t_sec - out[-1].t_sec)):
            out.append(f)
    if not out:
        # Every fix disagrees with the seed (degenerate); keep the old
        # first-fix seeding rather than dropping the track entirely.
        out = [near[0]]
        for f in near[1:]:
            if _haversine_m(out[-1], f) <= max_jump_m * max(1.0, (f.t_sec - out[-1].t_sec)):
                out.append(f)
    return out


# --- ground-truth emission -------------------------------------------------


# A road vehicle. Well above any legal limit (autobahn included) so a fast
# freeway stretch is never mistaken for a cut, and far below the hundreds of
# km/h an edit between two locations implies.
_MAX_ROAD_KMH = 160.0


def split_on_discontinuities(
    fixes: list[GpsFix], *, max_speed_kmh: float = _MAX_ROAD_KMH,
) -> list[list[GpsFix]]:
    """Split a track wherever it implies a speed no car reaches.

    Uploaded dashcam footage is often *edited* — a "Driving in Phoenix,
    Tempe, Scottsdale, Chandler..." upload cuts between separate drives,
    and the extracted track then teleports 19 km between consecutive
    samples. Treating that as one route produces ground truth that no
    trajectory can match, and the jump filter cannot help: it rejects
    outlier *fixes*, whereas both sides of a cut are perfectly correct.

    A cut is equally fatal to the visual odometry the ground truth is
    meant to score, so the answer is to segment rather than to patch.
    """
    if not fixes:
        return []
    runs: list[list[GpsFix]] = [[fixes[0]]]
    for prev, cur in zip(fixes, fixes[1:]):
        dt = cur.t_sec - prev.t_sec
        kmh = (_haversine_m(prev, cur) / dt) * 3.6 if dt > 0 else float("inf")
        if kmh > max_speed_kmh:
            runs.append([])
        runs[-1].append(cur)
    return runs


def longest_continuous_run(
    fixes: list[GpsFix], *, max_speed_kmh: float = _MAX_ROAD_KMH,
) -> list[GpsFix]:
    """The longest cut-free stretch of a track, by elapsed time.

    Duration, not fix count: a stretch with patchy OCR is still the
    better analysis window if it covers more of the drive.
    """
    runs = split_on_discontinuities(fixes, max_speed_kmh=max_speed_kmh)
    if not runs:
        return []
    return max(runs, key=lambda r: (r[-1].t_sec - r[0].t_sec) if len(r) > 1 else -1.0)


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
    # Centre on the bbox midpoint, not the fix mean: fixes are uniform in
    # TIME, so stop-and-go traffic piles them at one end and drags the
    # mean off-centre, leaving part of the route outside the disc.
    clat = float((lats.max() + lats.min()) / 2.0)
    clon = float((lons.max() + lons.min()) / 2.0)
    # Radius = farthest fix from the chosen centre (+margin), so every
    # fix is inside the disc by construction.
    dlat_m = (lats - clat) * 111320.0
    dlon_m = (lons - clon) * 111320.0 * np.cos(np.radians(clat))
    radius = float(np.hypot(dlat_m, dlon_m).max() + margin_m)
    return clat, clon, radius


_NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
# Nominatim spells it out; the repo's --city strings (and its geocode cache)
# use the short form, e.g. "San Francisco, California, USA".
_COUNTRY_SHORT = {"United States": "USA", "United Kingdom": "UK"}


def city_for_track(
    fixes: list[GpsFix], *, cache_path: Path | None = None, timeout: float = 20.0,
) -> str | None:
    """Reverse-geocode a track's midpoint into a ``--city`` string.

    Completes the "video in, ground truth out" loop: the overlay gives
    coordinates, and this turns them into the place name the rest of the
    pipeline wants (``--city``, and the key its OSM graph / geocode caches
    are stored under). One request per clip, memoised to ``cache_path``.

    Returns ``None`` on any failure — the caller is expected to fall back
    to the video title (:mod:`monocular_osm.city_inference`) or an
    explicit override rather than treat this as fatal.
    """
    if not fixes:
        return None
    mid = fixes[len(fixes) // 2]
    key = f"{mid.lat:.3f},{mid.lon:.3f}"   # ~100 m — plenty for a city name

    cache: dict[str, str | None] = {}
    cp = Path(cache_path) if cache_path else None
    if cp is not None and cp.exists():
        try:
            cache = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
    if key in cache:
        return cache[key]

    try:
        import requests

        resp = requests.get(
            _NOMINATIM_REVERSE,
            params={"lat": mid.lat, "lon": mid.lon, "format": "jsonv2", "zoom": 10},
            headers={"User-Agent": "monocular-osm-localization/gt-builder"},
            timeout=timeout,
        )
        resp.raise_for_status()
        addr = resp.json().get("address", {})
    except Exception:
        return None   # transient: do NOT memoise a failure

    place = next((addr[k] for k in
                  ("city", "town", "village", "municipality", "county")
                  if addr.get(k)), None)
    if not place:
        return None
    country = addr.get("country")
    country = _COUNTRY_SHORT.get(country, country)
    parts = [place, addr.get("state"), country]
    city = ", ".join(p for p in parts if p)

    cache[key] = city
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return city


def track_to_ground_truth(
    fixes: list[GpsFix],
    *,
    video_id: str,
    video_url: str,
    city: str,
    n_waypoints: int = 10,
    source: str = "gps_overlay_ocr",
    description: str | None = None,
) -> dict:
    """Convert an extracted track into the project's GT-JSON schema.

    Subsamples to ``n_waypoints`` evenly-spaced fixes (endpoints kept),
    matching ``ground_truth/*.json`` so the clip drops into the existing
    ``--ground-truth-waypoints`` evaluation. ``source``/``description``
    default to this module's OCR-overlay provenance; dataset adapters with
    real (RTK/INS) GT MUST pass their own so the JSON does not claim an
    OCR origin it doesn't have.
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
        "description": description if description is not None else (
            f"Auto-extracted from a burned-in GPS overlay via OCR "
            f"({len(fixes)} fixes → {len(sel)} waypoints). Verify before trusting."
        ),
        "source": source,
        "waypoints": [
            {"t_sec": round(f.t_sec, 1), "lat": round(f.lat, 6), "lon": round(f.lon, 6)}
            for f in sel
        ],
    }
