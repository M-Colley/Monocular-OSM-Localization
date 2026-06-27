"""License-plate registration-district anchor.

European (here: German) number plates encode the *registration district* in the
leading 1-3 letters (UL = Ulm, M = Munchen, B = Berlin, ...). Most vehicles are
registered locally, so the modal district prefix seen across a dashcam clip is a
strong, free, absolute coarse-location prior — exactly the kind of region gate
the OSM shape-matcher needs to resolve its selection ambiguity.

PRIVACY: we read plates only to extract the *district code* (the 1-3 letter
prefix), which is shared by tens of thousands of vehicles and is not an
individual identifier. We never store or use the full plate number. Consecutive
reads of the same full plate are collapsed so one nearby car cannot dominate the
vote — but only the prefix and a transient hash leave the per-frame scope.

Usage:
    from src.plate_anchor import plate_district_anchor
    anchor = plate_district_anchor("clip.mp4")  # -> {lat, lon, radius_m, code, ...}
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .kfz_codes import KFZ_DISTRICTS

_PREFIX_RE = re.compile(r"^([A-ZÄÖÜ]{1,3})[A-ZÄÖÜ]{1,2}\d")


@dataclass
class PlateAnchor:
    lat: float
    lon: float
    radius_m: float
    code: str
    district: str
    votes: int
    total_unique: int
    margin: float          # winner_votes / second_votes  (confidence of the vote)
    tally: dict


def _alpr():
    from fast_alpr import ALPR
    return ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="global-plates-mobile-vit-v2-model",
    )


def _read_prefixes(frames, alpr, min_conf: float = 0.55):
    """Yield (district_prefix, plate_hash) for confident plate reads."""
    for img in frames:
        for r in alpr.predict(img):
            o = getattr(r, "ocr", None)
            if o is None or not o.text:
                continue
            conf = getattr(o, "confidence", None)
            try:
                conf = float(np.mean(conf)) if conf is not None else 0.0
            except Exception:
                conf = float(conf or 0.0)
            t = re.sub(r"[^A-Z0-9]", "", o.text.upper())
            if conf < min_conf or len(t) < 5:
                continue
            m = _PREFIX_RE.match(t)
            if not m:
                continue
            code = m.group(1)
            if code not in KFZ_DISTRICTS:
                continue
            yield code, hashlib.md5(t.encode()).hexdigest()[:8]


def _sample_frames(video_path: str, every_sec: float, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_sec * fps)))
    out = []
    for fi in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if ok:
            out.append(f)
        if len(out) >= max_frames:
            break
    cap.release()
    return out


def plate_district_anchor(
    video_path: str,
    *,
    every_sec: float = 3.0,
    max_frames: int = 200,
    geocode_fn=None,
    frames=None,
) -> PlateAnchor | None:
    """Read plates across the clip, vote on the registration district, and return
    a coarse region anchor (district centroid + radius). None if no confident
    district emerges.

    geocode_fn(name)->(lat,lon) defaults to osmnx/Nominatim.
    """
    if frames is None:
        frames = _sample_frames(video_path, every_sec, max_frames)
    if not frames:
        return None

    alpr = _alpr()
    # Collapse consecutive identical plates (same car tracked) into one vote each,
    # so a single nearby vehicle cannot stuff the ballot.
    seen_plate = None
    votes: Counter = Counter()
    unique_plates: set = set()
    for code, phash in _read_prefixes(frames, alpr):
        if phash == seen_plate:
            continue
        seen_plate = phash
        if phash in unique_plates:
            continue
        unique_plates.add(phash)
        votes[code] += 1

    if not votes:
        return None
    ranked = votes.most_common()
    code, n = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    margin = n / second if second else float(n)
    district = KFZ_DISTRICTS[code]

    if geocode_fn is None:
        import osmnx as ox

        def geocode_fn(name):
            return tuple(ox.geocode(name))

    try:
        lat, lon = geocode_fn(f"{district}, Germany")
    except Exception:
        return None

    # District radius: German Landkreise are ~15-30 km across; use a generous
    # gate so the true area is inside even for a city-edge drive.
    return PlateAnchor(
        lat=float(lat), lon=float(lon), radius_m=20000.0,
        code=code, district=district, votes=int(n),
        total_unique=len(unique_plates), margin=float(margin),
        tally=dict(ranked),
    )
