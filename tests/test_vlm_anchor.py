"""Tests for the VLM anchor guards (model + geocoder injected, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

import src.vlm_anchor as va
from src.vlm_anchor import (
    _classify_reply,
    _crop_bbox,
    _parse,
    classify_sign_types,
    vlm_district_anchor,
)

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


def test_single_frame_street_is_geocoded_but_bound_rejects_far(monkeypatch) -> None:
    # A single legible street plate IS now trusted enough to geocode
    # (min_street_votes=1 — LIVE finding: the real 'Salzstadel' plate appeared
    # exactly once and lands 81 m from the route). The safety net against a
    # mis-resolved same-named street in another city is the distance BOUND, not
    # a vote count: here 'Marktplatz' resolves 150 km away and is rejected.
    _wire(monkeypatch, ["TEXT: none\nSTREET: Marktplatz\nDISTRICT: unknown",
                        "TEXT: none\nSTREET: unknown\nDISTRICT: unknown"])
    gc, queried = _geocoder({CITY: CITY_CENTER,
                             f"Marktplatz, {CITY}": FAR_MARKTPLATZ})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is None                             # far geocode → rejected
    assert f"Marktplatz, {CITY}" in queried        # but it WAS tried


def test_single_frame_street_in_bound_wins(monkeypatch) -> None:
    # The Salzstadel case: one legible plate, geocodes INSIDE the city → used.
    _wire(monkeypatch, ["TEXT: none\nSTREET: Salzstadel\nDISTRICT: unknown",
                        "TEXT: none\nSTREET: unknown\nDISTRICT: unknown"])
    near = (CITY_CENTER[0] + 0.005, CITY_CENTER[1])
    gc, _ = _geocoder({CITY: CITY_CENTER, f"Salzstadel, {CITY}": near})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is not None and out.label == "Salzstadel"


def test_city_name_as_district_is_never_the_anchor(monkeypatch) -> None:
    # DISTRICT=<the city> is a tautology (geocodes to the centroid), so even
    # with votes it must be dropped rather than become a fake anchor.
    _wire(monkeypatch, ["TEXT: none\nSTREET: unknown\nDISTRICT: Erbach"] * 2)
    gc, _ = _geocoder({CITY: CITY_CENTER, f"Erbach, {CITY}": CITY_CENTER})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2)
    assert out is None


def test_text_fallback_off_by_default(monkeypatch) -> None:
    # A lone TEXT token must NOT become the anchor by default (the 'WILL'
    # 2.2 km failure).
    _wire(monkeypatch, ["TEXT: Sedelhof\nSTREET: unknown\nDISTRICT: unknown"] * 2)
    near = (CITY_CENTER[0] + 0.004, CITY_CENTER[1])
    gc, _ = _geocoder({CITY: CITY_CENTER, f"Sedelhof, {CITY}": near})
    assert vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2) is None


def test_text_fallback_used_when_enabled(monkeypatch) -> None:
    # Opting in geocodes the text token (for the "any prior beats none" use).
    _wire(monkeypatch, ["TEXT: Sedelhof\nSTREET: unknown\nDISTRICT: unknown"] * 2)
    near = (CITY_CENTER[0] + 0.004, CITY_CENTER[1])
    gc, _ = _geocoder({CITY: CITY_CENTER, f"Sedelhof, {CITY}": near})
    out = vlm_district_anchor(_frames(2), CITY, geocode_fn=gc, n_query=2,
                              use_text_fallback=True)
    assert out is not None and out.label == "Sedelhof"


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


# --- here-vs-direction sign classification -------------------------------


class _Det:
    """Lightweight OCR detection stand-in (text + bbox + frame index)."""

    def __init__(self, text, bbox=None, frame_idx=0):
        self.text = text
        self.bbox = bbox
        self.frame_idx = frame_idx


def test_classify_reply_parses_leniently() -> None:
    assert _classify_reply("HERE") == "here"
    assert _classify_reply("The answer is DIRECTION.") == "direction"
    assert _classify_reply("**Here** — a shopfront") == "here"
    # 'direction' wins ties (conservative: suppress ambiguous anchors).
    assert _classify_reply("HERE or DIRECTION") == "direction"
    assert _classify_reply("I cannot tell") == "other"


def test_classify_sign_types_routes_by_reply() -> None:
    frames = [np.zeros((100, 100, 3), np.uint8)]
    dets = [_Det("Holborn", bbox=(10, 10, 40, 20)),
            _Det("Sedelhofgasse", bbox=(50, 50, 90, 60))]
    # ask_fn keys off the text so we can script HERE vs DIRECTION.
    replies = {"Holborn": "DIRECTION", "Sedelhofgasse": "HERE"}

    def ask_fn(pil, text):
        return replies[text]

    labels = classify_sign_types(frames, dets, ask_fn=ask_fn)
    assert labels == ["direction", "here"]


def test_classify_bad_frame_index_falls_back_and_never_raises() -> None:
    frames = [np.zeros((20, 20, 3), np.uint8)]
    dets = [_Det("X", bbox=(0, 0, 5, 5), frame_idx=99)]  # out of range → frame 0

    def ask_fn(pil, text):
        assert pil is not None  # a crop was produced from the fallback frame
        return "HERE"

    assert classify_sign_types(frames, dets, ask_fn=ask_fn) == ["here"]


def test_classify_swallows_model_errors_as_other() -> None:
    frames = [np.zeros((20, 20, 3), np.uint8)]

    def ask_fn(pil, text):
        raise RuntimeError("model OOM")

    assert classify_sign_types(frames, [_Det("X", bbox=(0, 0, 5, 5))],
                               ask_fn=ask_fn) == ["other"]


def test_crop_bbox_applies_margin_and_clips() -> None:
    frame = np.zeros((100, 200, 3), np.uint8)
    # box near the corner; margin must not push the crop off-frame.
    pil = _crop_bbox(frame, (0, 0, 20, 10), margin=0.6)
    w, h = pil.size
    assert 0 < w <= 200 and 0 < h <= 100


def test_crop_bbox_none_returns_full_frame() -> None:
    frame = np.zeros((30, 40, 3), np.uint8)
    pil = _crop_bbox(frame, None)
    assert pil.size == (40, 30)  # PIL is (w, h)
