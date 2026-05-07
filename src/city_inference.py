"""Infer a likely city from a video title using local heuristics."""

from __future__ import annotations

import re
from hashlib import sha256


# Latin title-case words, including common diacritics found in European city names.
_TITLE_WORD = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*"
_TITLE_PHRASE = rf"{_TITLE_WORD}(?:[\s-]+{_TITLE_WORD}){{0,3}}"
_COUNTRY_HINTS = {
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Austria",
    "Switzerland",
    "Poland",
    "Czechia",
    "Czech",
    "England",
    "Scotland",
    "Ireland",
    "Wales",
    "UK",
    "USA",
    "Canada",
    "Japan",
    "China",
    "India",
    "Korea",
    "Australia",
    "New Zealand",
}
_REJECT_PHRASE_TOKENS = {
    "Best",
    "Dashcam",
    "Drive",
    "Driving",
    "Driver",
    "Street",
    "Streets",
    "Road",
    "Roads",
    "Walk",
    "Walking",
    "Trip",
    "Travel",
    "Tour",
    "Night",
    "Rain",
    "Snow",
    "Traffic",
    "Highway",
    "Motorway",
    "Autobahn",
    "Freeway",
    "City",
    "Downtown",
    "Center",
    "Centre",
    "POV",
    "POV.",
    "HD",
    "HDR",
    "UHD",
    "ASMR",
}
_PREPOSITION_PATTERNS = [
    # Matches "Driving in Ulm, Germany" / "walk through Paris".
    re.compile(
        rf"\b(?:in|through|around|near|from|to|across|exploring|visiting)\s+"
        rf"(?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?",
        re.UNICODE,
    ),
    # Matches "Ulm dashcam" / "Paris drive".
    re.compile(
        rf"\b(?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?\s+"
        rf"(?:dashcam|drive|driving|walk|walking|tour|trip)\b",
        re.UNICODE,
    ),
    # Matches "Tokyo night drive" / "Vienna downtown walk".
    re.compile(
        rf"\b(?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?"
        rf"(?:\s+(?:night|day|morning|evening|downtown|center|centre|city)){{0,2}}\s+"
        rf"(?:dashcam|drive|driving|walk|walking|tour|trip)\b",
        re.UNICODE,
    ),
]
_PARENS_PATTERN = re.compile(
    rf"\((?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?\)",
    re.UNICODE,
)
_MAX_SLUG_PARTS = 3
_MAX_SLUG_LENGTH = 80


def _clean_phrase(text: str | None) -> str | None:
    if not text:
        return None
    phrase = re.sub(r"\s+", " ", text).strip(" -–—:,.|/[]()")
    if not phrase:
        return None
    tokens = phrase.split()
    if not tokens:
        return None
    if tokens[0] in _REJECT_PHRASE_TOKENS:
        return None
    if all(token in _REJECT_PHRASE_TOKENS for token in tokens):
        return None
    return phrase


def _format_city(city: str | None, country: str | None) -> str | None:
    city_clean = _clean_phrase(city)
    if city_clean is None:
        return None
    country_clean = _clean_phrase(country)
    if country_clean and country_clean in _COUNTRY_HINTS and country_clean != city_clean:
        return f"{city_clean}, {country_clean}"
    return city_clean


def guess_city_from_title(title: str | None) -> str | None:
    """Guess a city string from a human-readable video title."""
    if not title:
        return None

    normalized = re.sub(r"\s+", " ", title)
    for pattern in (*_PREPOSITION_PATTERNS, _PARENS_PATTERN):
        match = pattern.search(normalized)
        if not match:
            continue
        guess = _format_city(match.group("city"), match.groupdict().get("country"))
        if guess:
            return guess

    return None


def slugify_submission(*parts: str | None, fallback_seed: str) -> str:
    """Build a stable directory slug for a submission."""
    chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        cleaned = re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-")
        if cleaned:
            chunks.append(cleaned)
    if chunks:
        return "-".join(chunks[:_MAX_SLUG_PARTS])[:_MAX_SLUG_LENGTH].strip("-")
    digest = sha256(fallback_seed.encode("utf-8")).hexdigest()[:12]
    return f"video-{digest}"
