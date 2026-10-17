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

import re

from .geo import geodesic_distance_m

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


def _geodesic_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """WGS84 geodesic distance between two ``(lat, lon)`` points (metres)."""
    return geodesic_distance_m(a[0], a[1], b[0], b[1])


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _default_geocode(query: str):
    """Nominatim's first result with what it matched, or ``None``.

    Returns ``{"lat", "lon", "name", "class", "type", "place_rank"}`` via
    osmnx's public ``geocode_to_gdf`` (its cache and throttling), so
    :func:`_is_place_match` can tell a district from a business that merely
    shares a word with the query.
    """
    import osmnx as ox
    try:
        row = ox.geocode_to_gdf(query, which_result=1).iloc[0]
        return {"lat": float(row["lat"]), "lon": float(row["lon"]),
                "name": str(row.get("name") or ""), "class": str(row.get("class") or ""),
                "type": str(row.get("type") or ""),
                "place_rank": int(row.get("place_rank") or 30)}
    except Exception:
        return None


def _latlon(hit) -> tuple[float, float] | None:
    if hit is None:
        return None
    if isinstance(hit, dict):
        return (float(hit["lat"]), float(hit["lon"]))
    return (float(hit[0]), float(hit[1]))


#: OSM classes / amenity types that are businesses: a title word matching one
#: ("China Tours and Travel" for "Tours") names no part of the route.
_BUSINESS_CLASSES = {"shop", "office", "craft"}
_BUSINESS_AMENITIES = {"cafe", "restaurant", "fast_food", "bar", "pub", "bank",
                       "pharmacy", "fuel", "car_rental", "car_wash", "atm"}
#: Nominatim place_rank below this is coarser than a city (state, country).
_MIN_PLACE_RANK = 13
#: rapidfuzz ratio (0-100) the matched object's name must reach.
_MIN_NAME_RATIO = 90.0


def _is_place_match(candidate: str, hit, city: str) -> bool:
    """Does the geocoder's hit actually NAME ``candidate``?

    A title word that is a common noun geocodes to whatever object contains
    it: on the overlay fleet "Tours" -> China Tours and Travel (a shop),
    "Support" -> Signature Flight Support Terminal, "Check" -> Sam's Grocery &
    Check Cashing, "Napoleon" (Ulm) -> Napoleonweg. Accept the hit only when
    the object's name is the candidate (city words removed, fuzzy ratio >= 90),
    it is not a business, and it is not coarser than a city. Rules fixed from
    what the geocoder returns, not from localization errors. Hits without
    details (tests' plain ``(lat, lon)``) are accepted unchanged.
    """
    if not isinstance(hit, dict):
        return True
    from rapidfuzz import fuzz

    city_words = {w.strip().casefold() for part in city.split(",") for w in part.split()}

    def strip_city(text: str) -> str:
        return " ".join(w for w in text.casefold().split() if w not in city_words)

    if not strip_city(candidate):
        return False
    if hit.get("class") in _BUSINESS_CLASSES or (
            hit.get("class") == "amenity" and hit.get("type") in _BUSINESS_AMENITIES):
        return False
    if int(hit.get("place_rank") or 30) < _MIN_PLACE_RANK:
        return False
    name = str(hit.get("name") or "")
    # "Bloomsbury London" names "Bloomsbury"; "Ulm Minster" names "Ulm Minster".
    score = max(fuzz.ratio(candidate.casefold(), name.casefold()),
                fuzz.ratio(strip_city(candidate), strip_city(name)))
    return score >= _MIN_NAME_RATIO


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
    route_length_m: float | None = None,
) -> tuple[float, float, float, list[str]] | None:
    """Coarse prior ``(lat, lon, radius_m, matched_places)`` from named places
    in the video's title + description, or ``None`` if none resolve usefully.

    Geocodes each candidate as ``"<place>, <city>"``, keeps those within
    ``max_km_from_city`` of the city centroid, and returns the robust
    (median-centred) disc covering them. The result is meant to be used as an
    ``osm_around`` seed — video-derived, not GPS-derived.

    ``route_length_m`` (the drive's estimated length) widens the disc to the
    places' spread PLUS that length: if the named places lie on the route, no
    point of the route is farther from them than the route is long. Without it
    the disc only covers the places themselves, and two landmarks 700 m apart
    gave Ulm 4K an 845 m gate holding 38 % of its 5-minute route.
    """
    geocode_fn = geocode_fn or _default_geocode
    city_ll = _latlon(geocode_fn(city))
    if city_ll is None:
        return None
    # Any part of the city string ("Detroit", "Michigan", "USA") is the city
    # itself or coarser: never a refinement ("Michigan" geocoded to a station
    # and became an 800 m gate on a Detroit clip).
    city_parts = {p.strip().casefold() for p in city.split(",") if p.strip()}
    cands = extract_place_candidates(title, description)
    hits: list[tuple[str, tuple[float, float]]] = []
    for c in cands:
        if c.strip().casefold() in city_parts:
            continue
        hit = geocode_fn(f"{c}, {city}")
        ll = _latlon(hit)
        if ll is None or not _is_place_match(c, hit, city):
            continue
        if _geodesic_m(city_ll, ll) / 1000.0 <= max_km_from_city:
            hits.append((c, ll))
    if len(hits) < min_places:
        return None
    lats = [h[1][0] for h in hits]
    lons = [h[1][1] for h in hits]
    centre = (_median(lats), _median(lons))
    radius = (max((_geodesic_m(centre, h[1]) for h in hits), default=0.0)
              + max(margin_m, float(route_length_m or 0.0)))
    radius = min(max(radius, min_radius_m), max_radius_m)
    return (centre[0], centre[1], radius, [h[0] for h in hits])
