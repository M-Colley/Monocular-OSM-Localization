"""Infer a likely city from a video title using local heuristics."""

from __future__ import annotations

import re
from hashlib import sha256


# Worldwide city words: Extended Latin, Greek, Cyrillic, plus CJK/Hangul scripts.
_CASED_WORD = r"[A-ZÀ-ÖØ-ÞĀ-ŽͰ-ϿЀ-Я][A-Za-zÀ-ÖØ-öø-ÿĀ-žͰ-ϿЀ-я'’.-]*"
_CJK_WORD = r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]{2,10}"
_TITLE_WORD = rf"(?:{_CASED_WORD}|{_CJK_WORD})"
_TITLE_PHRASE = rf"{_TITLE_WORD}(?:[\s-]+{_TITLE_WORD}){{0,3}}"
_COUNTRY_HINTS = {
    "Germany", "France", "Italy", "Spain", "Portugal", "Netherlands", "Belgium",
    "Austria", "Switzerland", "Poland", "Czechia", "Czech", "Czech Republic",
    "Slovakia", "Hungary", "Romania", "Bulgaria", "Serbia", "Croatia", "Slovenia",
    "Greece", "Turkey", "Türkiye", "Norway", "Sweden", "Finland", "Denmark",
    "Estonia", "Latvia", "Lithuania", "Ukraine", "Russia",
    "England", "Scotland", "Ireland", "Wales", "UK", "United Kingdom",
    "USA", "United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
    "Colombia", "Peru", "Uruguay",
    "Japan", "China", "Taiwan", "Hong Kong", "India", "Korea", "South Korea",
    "Thailand", "Vietnam", "Indonesia", "Malaysia", "Singapore", "Philippines",
    "UAE", "Saudi Arabia", "Israel", "Qatar", "Egypt", "Morocco", "South Africa",
    "Kenya", "Nigeria", "Australia", "New Zealand",
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
        rf"\b(?i:in|through|around|near|from|to|across|exploring|visiting)\s+"
        rf"(?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?",
        re.UNICODE,
    ),
    # Matches "Ulm dashcam" / "Paris drive".
    re.compile(
        rf"\b(?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?\s+"
        rf"(?i:dashcam|drive|driving|walk|walking|tour|trip)\b",
        re.UNICODE,
    ),
    # Matches "Tokyo night drive" / "Vienna downtown walk".
    re.compile(
        rf"\b(?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?"
        rf"(?:\s+(?i:night|day|morning|evening|downtown|center|centre|city)){{0,2}}\s+"
        rf"(?i:dashcam|drive|driving|walk|walking|tour|trip)\b",
        re.UNICODE,
    ),
]
_PARENS_PATTERN = re.compile(
    rf"\((?P<city>{_TITLE_PHRASE})(?:,\s*(?P<country>{_TITLE_PHRASE}))?\)",
    re.UNICODE,
)
_CITY_COUNTRY_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?P<city>{_TITLE_PHRASE}),\s*(?P<country>{_TITLE_PHRASE})",
    re.UNICODE,
)
_SEGMENT_SPLIT_RE = re.compile(r"\s+[-–—|:/]\s+|[|()]")
_MAX_SLUG_PARTS = 3
_MAX_SLUG_LENGTH = 80


def _clean_phrase(text: str | None, *, strip_leading_generic: bool = False) -> str | None:
    if not text:
        return None
    phrase = re.sub(r"\s+", " ", text).strip(" -–—:,.|/[]()")
    if not phrase:
        return None
    tokens = phrase.split()
    if strip_leading_generic:
        while tokens and tokens[0] in _REJECT_PHRASE_TOKENS:
            tokens = tokens[1:]
    if not tokens:
        return None
    if tokens[0] in _REJECT_PHRASE_TOKENS:
        return None
    if all(token in _REJECT_PHRASE_TOKENS for token in tokens):
        return None
    return " ".join(tokens)


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

    for seg in _SEGMENT_SPLIT_RE.split(normalized):
        for match in _CITY_COUNTRY_PATTERN.finditer(seg):
            country_raw = match.group("country")
            # Take only the leading country phrase before trailing descriptors (e.g. 'Japan 4K')
            country_tokens = (re.sub(r"\s+", " ", country_raw).strip(" -–—:,.|/[]()")).split()
            matched_country = None
            for length in range(min(3, len(country_tokens)), 0, -1):
                cand = " ".join(country_tokens[:length])
                if cand in _COUNTRY_HINTS:
                    matched_country = cand
                    break
            if matched_country is None:
                continue
            city_clean = _clean_phrase(match.group("city"), strip_leading_generic=True)
            if city_clean and city_clean != matched_country:
                return f"{city_clean}, {matched_country}"

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
