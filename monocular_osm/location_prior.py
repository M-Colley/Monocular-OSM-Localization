"""Video-derived COARSE location prior — a GPS-free seed from the names the
uploader wrote in the title / description.

The deployable wall is coarse self-localization: seeding the search from a
city centroid misses peripheral / suburban drives, so the fine matcher then
searches the wrong ~1 km. But most dashcam uploads NAME the route — e.g.
"Berlin 4K … Alexanderplatz, Potsdamer Platz, Brandenburg Gate". Geocoding
those named places yields a disc that actually covers the drive, far tighter
than the city centroid, at zero visual cost.

:func:`resolve_coarse_prior` extracts candidate place phrases from the
title + description, geocodes each with the city as context, keeps those that
land near the city, and returns the robust disc (centre + radius) covering
them — a drop-in ``osm_around`` derived from the video instead of GPS truth.
"""

from __future__ import annotations

import math
import re

__all__ = ["resolve_coarse_prior", "extract_place_candidates"]

_PLACE = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*"
_PHRASE = rf"{_PLACE}(?:[ \-]{_PLACE}){{0,3}}"
# split the text into route segments before scanning for place phrases
_SEP = re.compile(r"[,\|/·—–>\n\t;:]+|→|➜|»|\bto\b|\bvia\b|\bthen\b", re.IGNORECASE)

# Generic words that are NOT place names on their own; a phrase made only of
# these is dropped. Real names survive because they carry a proper token.
_GENERIC = {
    "best", "dashcam", "dash", "cam", "drive", "driving", "driver", "ride",
    "street", "streets", "road", "roads", "avenue", "boulevard", "walk",
    "walking", "trip", "travel", "tour", "touring", "night", "day", "morning",
    "evening", "sunset", "rain", "rainy", "snow", "winter", "summer", "autumn",
    "spring", "traffic", "highway", "motorway", "autobahn", "freeway", "city",
    "downtown", "center", "centre", "central", "old", "town", "pov", "hd",
    "hdr", "uhd", "4k", "8k", "60fps", "asmr", "relaxing", "scenic", "route",
    "part", "episode", "vol", "full", "video", "footage", "germany", "france",
    "spain", "italy", "usa", "canada", "uk", "england", "japan", "the", "and",
    "a", "an", "of", "in", "on", "at", "west", "east", "north", "south",
    "gmbh", "official", "channel", "subscribe", "like", "https", "http", "www",
}


def extract_place_candidates(title: str | None, description: str | None,
                             *, max_candidates: int = 14) -> list[str]:
    """Ordered, de-duplicated place-name phrases from the title + description."""
    text = " ".join(t for t in (title, description) if t)
    if not text.strip():
        return []
    text = re.sub(r"https?://\S+", " ", text)      # drop URLs
    out: list[str] = []
    seen: set[str] = set()
    for part in _SEP.split(text):
        if not part or not part.strip():
            continue
        for m in re.finditer(_PHRASE, part):
            phrase = m.group().strip(" -–—.'’")
            toks = [t for t in phrase.split() if t]
            if len(phrase) < 3 or not toks:
                continue
            # keep only phrases with a SUBSTANTIVE token (non-generic, >=3
            # chars) — drops "K Driving Tour" (from "4K"), "Night Drive", etc.
            if not any(t.lower().strip(".'’-") not in _GENERIC and len(t) >= 3
                       for t in toks):
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(phrase)
            if len(out) >= max_candidates:
                return out
    return out


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6_371_000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _default_geocode(query: str):
    import osmnx as ox
    try:
        lat, lon = ox.geocode(query)
        return (float(lat), float(lon))
    except Exception:
        return None


def resolve_coarse_prior(
    city: str,
    title: str | None,
    description: str | None,
    *,
    geocode_fn=None,
    max_km_from_city: float = 60.0,
    margin_m: float = 500.0,
    min_radius_m: float = 800.0,
    max_radius_m: float = 6000.0,
    min_places: int = 1,
) -> tuple[float, float, float, list[str]] | None:
    """Coarse prior ``(lat, lon, radius_m, matched_places)`` from named places
    in the video's title + description, or ``None`` if none resolve usefully.

    Geocodes each candidate as ``"<place>, <city>"``, keeps those within
    ``max_km_from_city`` of the city centroid, and returns the robust
    (median-centred) disc covering them. The result is meant to be used as an
    ``osm_around`` seed — video-derived, not GPS-derived.
    """
    geocode_fn = geocode_fn or _default_geocode
    city_ll = geocode_fn(city)
    if city_ll is None:
        return None
    cands = extract_place_candidates(title, description)
    hits: list[tuple[str, tuple[float, float]]] = []
    for c in cands:
        if c.lower() == city.split(",")[0].strip().lower():
            continue                       # the city itself is not a refinement
        ll = geocode_fn(f"{c}, {city}")
        if ll is None:
            continue
        if _haversine_m(city_ll, ll) / 1000.0 <= max_km_from_city:
            hits.append((c, ll))
    if len(hits) < min_places:
        return None
    lats = [h[1][0] for h in hits]
    lons = [h[1][1] for h in hits]
    centre = (_median(lats), _median(lons))
    radius = max((_haversine_m(centre, h[1]) for h in hits), default=0.0) + margin_m
    radius = min(max(radius, min_radius_m), max_radius_m)
    return (centre[0], centre[1], radius, [h[0] for h in hits])
