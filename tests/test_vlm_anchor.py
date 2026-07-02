"""Tests for the VLM anchor guards (model + geocoder injected, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

import src.vlm_anchor as va
from src.vlm_anchor import _parse, vlm_district_anchor

CITY = "Erbach"
CITY_CENTER = (48.328, 9.888)          # Erbach an der Donau
FAR_MARKTPLATZ = (49.657, 8.995)       # Marktplatz in Erbach (Odenwald), ~150 km


def _frames(n: int):
    return [np.zeros((4, 4, 3), np.uint8) for _ in range(n)]


def _wire(monkeypatch, replies: list[str]):
    """Bypass model loading; script one reply per queried frame."""
    it = iter(replies)
    monkeypatch.setattr(va, "_load", lambda: None)
    monkeypatch.setattr(va, "_ask", lambda pil, city: next(it))


def _geocoder(db: dict):
    queried: list[str] = []

    def fn(q: str):
        queried.append(q)
        if q in db:
            return db[q]
        raise KeyError(q)

    return fn, queried


def test_parse_extracts_fields() -> None:
    st, di, tx = _parse("TEXT: Bäckerei Müller, Apotheke\nSTREET: Hauptstraße\n"
                        "DISTRICT: unknown")
    assert st == "Hauptstraße" and di is None
    assert tx == ["Bäckerei Müller", "Apotheke"]


def test_single_frame_street_hallucination_is_not_geocoded(monkeypatch) -> None:
    # One frame says 'Marktplatz', the other knows nothing: 1 vote < min 2,
    # so the street must never reach the geocoder (the old code geocoded it
    # and moved the anchor 150 km to the wrong Erbach).
    _wire(monkeypatch, ["TEXT: none\nSTREET: Marktplatz\nDISTRICT: unknown",
                        "TEXT: none\nSTREET: unknown\nDISTRICT: unknown"])
    gc, queried = _geocoder({CITY: CITY_CENTER,
                             f"Marktplatz, {CITY}": FAR_MARKTPLATZ})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is None
    assert f"Marktplatz, {CITY}" not in queried


def test_far_geocode_is_rejected_even_with_votes(monkeypatch) -> None:
    # Consistent answer, but Nominatim resolves it to another federal state:
    # the documented "bounded to the city" promise must reject it.
    _wire(monkeypatch, ["TEXT: none\nSTREET: Marktplatz\nDISTRICT: unknown"] * 2)
    gc, queried = _geocoder({CITY: CITY_CENTER,
                             f"Marktplatz, {CITY}": FAR_MARKTPLATZ})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is None
    assert f"Marktplatz, {CITY}" in queried        # tried, then rejected


def test_local_consensus_street_wins(monkeypatch) -> None:
    _wire(monkeypatch, ["TEXT: none\nSTREET: Hauptstraße\nDISTRICT: Altstadt"] * 2)
    near = (CITY_CENTER[0] + 0.01, CITY_CENTER[1] + 0.01)   # ~1.3 km away
    gc, _ = _geocoder({CITY: CITY_CENTER, f"Hauptstraße, {CITY}": near})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is not None
    assert out.label == "Hauptstraße"
    assert out.lat == pytest.approx(near[0]) and out.lon == pytest.approx(near[1])


def test_no_city_reference_means_no_anchor(monkeypatch) -> None:
    # If the bare city can't geocode we cannot bound anything -> bail out.
    _wire(monkeypatch, ["TEXT: none\nSTREET: Hauptstraße\nDISTRICT: unknown"] * 2)
    gc, _ = _geocoder({f"Hauptstraße, {CITY}": CITY_CENTER})
    assert vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2) is None


def test_district_fallback_respects_min_votes_and_bound(monkeypatch) -> None:
    _wire(monkeypatch, ["TEXT: none\nSTREET: unknown\nDISTRICT: Altstadt"] * 2)
    near = (CITY_CENTER[0] - 0.005, CITY_CENTER[1])
    gc, _ = _geocoder({CITY: CITY_CENTER, f"Altstadt, {CITY}": near})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is not None and out.label == "Altstadt"
