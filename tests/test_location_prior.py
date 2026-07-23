"""Offline tests for the video-derived coarse location prior (no network)."""

from __future__ import annotations

from src.location_prior import extract_place_candidates, resolve_coarse_prior


def test_extract_landmarks_from_route_title() -> None:
    title = ("BERLIN - Germany 4K Driving Tour | Alexanderplatz, Potsdamer "
             "Platz, Brandenburg Gate, West-Berlin")
    c = extract_place_candidates(title, None)
    assert "Alexanderplatz" in c
    assert "Potsdamer Platz" in c
    assert "Brandenburg Gate" in c
    # generic junk from "4K Driving Tour" must not survive
    assert not any("driving tour" in x.lower() for x in c)
    assert not any(x.lower() in ("tour", "night drive") for x in c)


def test_extract_drops_generic_only_titles() -> None:
    assert extract_place_candidates("Relaxing 4K Night Drive Dashcam POV", None) == []
    assert extract_place_candidates(None, None) == []
    assert extract_place_candidates("", "   ") == []


def test_extract_uses_description_and_strips_urls() -> None:
    c = extract_place_candidates("City drive",
                                 "Route: Marienplatz to Odeonsplatz. "
                                 "https://maps.google.com/xyz Subscribe!")
    assert "Marienplatz" in c and "Odeonsplatz" in c
    assert not any("http" in x.lower() for x in c)


def test_resolve_prior_covers_named_route() -> None:
    title = "Berlin drive: Alexanderplatz, Potsdamer Platz, Brandenburg Gate"
    coords = {
        "berlin, germany": (52.52, 13.405),
        "alexanderplatz, berlin, germany": (52.5219, 13.4132),
        "potsdamer platz, berlin, germany": (52.5096, 13.3760),
        "brandenburg gate, berlin, germany": (52.5163, 13.3777),
    }
    prior = resolve_coarse_prior("Berlin, Germany", title, None,
                                 geocode_fn=lambda q: coords.get(q.lower()))
    assert prior is not None
    lat, lon, r, places = prior
    assert 52.50 < lat < 52.53 and 13.37 < lon < 13.42
    assert 800 <= r <= 6000
    assert set(places) >= {"Alexanderplatz", "Potsdamer Platz", "Brandenburg Gate"}


def test_resolve_prior_rejects_far_geocodes() -> None:
    # a same-named place in the wrong country must be dropped by the distance gate
    title = "Ulm drive past Springfield"
    coords = {"ulm, germany": (48.40, 9.99),
              "springfield, ulm, germany": (39.80, -89.64)}  # US Springfield
    prior = resolve_coarse_prior("Ulm, Germany", title, None,
                                 geocode_fn=lambda q: coords.get(q.lower()))
    assert prior is None            # nothing plausible near Ulm -> no refinement


def test_resolve_prior_none_without_places() -> None:
    prior = resolve_coarse_prior("Daly City, California, USA",
                                 "comma2k19 route 148 dashcam", None,
                                 geocode_fn=lambda q: (37.7, -122.46))
    assert prior is None


def test_resolve_prior_none_when_city_ungeocodable() -> None:
    assert resolve_coarse_prior("Nowhere", "Foo Bar Baz", None,
                                geocode_fn=lambda q: None) is None
