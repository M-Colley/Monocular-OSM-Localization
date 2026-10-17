"""Offline tests for the video-derived coarse location prior (no network)."""

from __future__ import annotations

import pytest

from monocular_osm.location_prior import extract_place_candidates, resolve_coarse_prior


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


def test_one_title_place_is_not_enough_when_two_are_required() -> None:
    """The title seed requires corroboration (cli passes min_places=2): on
    Ulm #2 the single title word "Napoleon" became an 800 m hard gate centred
    8-9 km from the route (all three benchmark reps answered 7.7 km off)."""
    coords = {"ulm, germany": (48.40, 9.99),
              "napoleon, ulm, germany": (48.451, 10.089),
              "ulm minster, ulm, germany": (48.3985, 9.9925)}
    geo = lambda q: coords.get(q.lower())                        # noqa: E731
    one = "Ulm drive past Napoleon"
    assert resolve_coarse_prior("Ulm, Germany", one, None, geocode_fn=geo) is not None
    assert resolve_coarse_prior("Ulm, Germany", one, None, geocode_fn=geo,
                                min_places=2) is None
    two = "Ulm drive: Napoleon, Ulm Minster"
    assert resolve_coarse_prior("Ulm, Germany", two, None, geocode_fn=geo,
                                min_places=2) is not None


def test_cli_title_seed_requires_two_places() -> None:
    import inspect

    import monocular_osm.cli as cli

    src = inspect.getsource(cli)
    i = src.index("resolve_coarse_prior(city, metadata.title")
    assert "min_places=2" in src[i:i + 200]


def _hit(lat, lon, name, cls, typ, rank=30):
    return {"lat": lat, "lon": lon, "name": name, "class": cls, "type": typ, "place_rank": rank}


# What Nominatim returned (2026-10-17) for title words of the benchmark clips.
_NOMINATIM = {
    "chicago, illinois, usa": _hit(41.88, -87.63, "Chicago", "boundary", "administrative", 16),
    "tours, chicago, illinois, usa": _hit(41.85, -87.63, "China Tours and Travel", "shop", "travel_agency"),
    "support, chicago, illinois, usa": _hit(41.98, -87.90, "Signature Flight Support Terminal", "aeroway", "terminal"),
    "pro, chicago, illinois, usa": _hit(41.90, -87.70, "All Pro Cleaners", "shop", "dry_cleaning"),
    "detroit, michigan, usa": _hit(42.33, -83.05, "Detroit", "boundary", "administrative", 16),
    "michigan, detroit, michigan, usa": _hit(42.33, -83.05, "Michigan", "railway", "station"),
    "northwest, detroit, michigan, usa": _hit(42.40, -83.20, "Northwest", "shop", "convenience"),
    "london, uk": _hit(51.507, -0.128, "London", "boundary", "administrative", 16),
    "bloomsbury, london, uk": _hit(51.523, -0.125, "Bloomsbury", "place", "suburb", 19),
    "bloomsbury london, london, uk": _hit(51.523, -0.125, "Bloomsbury", "place", "suburb", 19),
    "afternoon, london, uk": _hit(51.519, -0.127, "The British Museum Afternoon Tea", "amenity", "cafe"),
    "west end, london, uk": _hit(51.513, -0.136, "West End", "place", "locality", 25),
    "ulm, germany": _hit(48.398, 9.993, "Ulm", "boundary", "administrative", 16),
    "napoleon, ulm, germany": _hit(48.451, 10.089, "Napoleonweg", "highway", "residential", 26),
    "ulm minster, ulm, germany": _hit(48.3985, 9.9925, "Ulm Minster", "amenity", "place_of_worship"),
    "neu-ulm, ulm, germany": _hit(48.392, 10.011, "Neu-Ulm", "boundary", "administrative", 16),
    "bavaria, ulm, germany": _hit(48.9, 11.4, "Bavaria", "boundary", "administrative", 8),
}


def _geo(q):
    return _NOMINATIM.get(q.lower())


def test_title_words_that_only_match_a_business_or_the_city_are_not_places() -> None:
    # Junk only: no seed at all (the run then searches the named city).
    assert resolve_coarse_prior("Chicago, Illinois, USA",
                                "Chicago | Viofo A229 Pro Front Unit Day Time Testing | Support, Tours",
                                None, geocode_fn=_geo) is None
    assert resolve_coarse_prior("Detroit, Michigan, USA",
                                "Driving from Detroit, Michigan to Northwest", None,
                                geocode_fn=_geo) is None
    assert resolve_coarse_prior("Ulm, Germany", "Ulm drive past Napoleon", None,
                                geocode_fn=_geo) is None


def test_real_districts_and_landmarks_still_seed() -> None:
    pr = resolve_coarse_prior("London, UK",
                              "Bloomsbury London | Afternoon drive | West End", None,
                              geocode_fn=_geo, min_places=2)
    assert pr is not None and set(pr[3]) == {"Bloomsbury London", "West End"}   # not "Afternoon"
    pr = resolve_coarse_prior("Ulm, Germany", "Ulm Minster, Neu-Ulm, Bavaria", None,
                              geocode_fn=_geo, min_places=2)
    assert pr is not None and set(pr[3]) == {"Ulm Minster", "Neu-Ulm"}            # a state is coarser


def test_route_length_widens_a_tight_landmark_pair_to_hold_the_drive() -> None:
    """Two landmarks ~1.5 km apart gave Ulm 4K an 845 m gate holding 38 % of
    its route; the disc must allow the route's own length around them."""
    t = "Ulm Minster, Neu-Ulm"
    tight = resolve_coarse_prior("Ulm, Germany", t, None, geocode_fn=_geo, min_places=2)
    wide = resolve_coarse_prior("Ulm, Germany", t, None, geocode_fn=_geo, min_places=2,
                                route_length_m=1650.0)
    assert tight[2] < 1500 and wide[2] == pytest.approx(tight[2] - 500 + 1650, abs=1)
    assert resolve_coarse_prior("Ulm, Germany", t, None, geocode_fn=_geo, min_places=2,
                                route_length_m=50_000)[2] == 6000      # still capped
