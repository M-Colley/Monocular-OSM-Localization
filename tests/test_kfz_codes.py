"""Sanity tests for the KFZ district-code table (audit kfz_codes:71) and its
Wikidata snapshot (monocular_osm/data/kfz_codes.json)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from monocular_osm.geo import geodesic_distance_m
from monocular_osm.kfz_codes import (
    KFZ_DISTRICTS,
    KFZ_SNAPSHOT,
    KFZ_SNAPSHOT_META,
    MANUAL_OVERRIDES,
)


def test_all_keys_are_valid_prefixes() -> None:
    # A key with whitespace/digits can never be matched by the plate parser.
    for k in KFZ_DISTRICTS:
        assert re.fullmatch(r"[A-ZÄÖÜ]{1,3}", k), f"dead key: {k!r}"


def test_no_colliding_manual_geocode_strings() -> None:
    # Two codes mapping to the same query would geocode one of them ~hundreds
    # of km wrong (FB/FDB were both 'Friedberg', 250 km apart).
    dupes = [v for v, n in Counter(MANUAL_OVERRIDES.values()).items() if n > 1]
    assert dupes == []


def test_snapshot_seat_strings_are_not_homonyms() -> None:
    # Several codes may legitimately share a seat (a district's reissued old
    # codes resolve to its seat), but one geocode string must never stand for
    # two different places: every code sharing a string sits at one location.
    by_seat: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in KFZ_SNAPSHOT.values():
        by_seat[r["seat"]].append((r["lat"], r["lon"]))
    for seat, pts in by_seat.items():
        lat0, lon0 = pts[0]
        for lat, lon in pts[1:]:
            assert geodesic_distance_m(lat0, lon0, lat, lon) < 30_000, seat


def test_friedberg_codes_are_state_qualified() -> None:
    assert "Bayern" in KFZ_DISTRICTS["FDB"]
    assert "Hessen" in KFZ_DISTRICTS["FB"]
    # ...and the snapshot resolves them to the two different Friedbergs too.
    assert KFZ_SNAPSHOT["FDB"]["state"] == "Bayern"
    assert KFZ_SNAPSHOT["FB"]["state"] == "Hessen"


def test_corrected_seats() -> None:
    assert KFZ_DISTRICTS["MN"] == "Mindelheim"      # Unterallgäu, not Memmingen
    assert KFZ_DISTRICTS["MM"] == "Memmingen"
    assert KFZ_DISTRICTS["EN"] == "Schwelm"         # Ennepe-Ruhr, not Hagen
    assert KFZ_DISTRICTS["HA"] == "Hagen"
    assert KFZ_DISTRICTS["WÜM"] == "Waldmünchen"
    assert KFZ_DISTRICTS["ER"] == "Erlangen"
    assert " ER" not in KFZ_DISTRICTS               # old dead key


def test_manual_errors_fixed_against_wikidata() -> None:
    # RID is Riedenburg (Landkreis Kelheim), not Rottal-Inn (PAN/EG/GRI);
    # LOS/PM named kreisfreie Staedte outside those districts.
    assert KFZ_DISTRICTS["RID"] == "Riedenburg"
    assert KFZ_SNAPSHOT["RID"]["seat_label"] == "Riedenburg"
    assert KFZ_DISTRICTS["LOS"] == "Beeskow"
    assert KFZ_SNAPSHOT["LOS"]["seat_label"] == "Beeskow"
    assert KFZ_DISTRICTS["PM"] == "Bad Belzig"
    assert KFZ_SNAPSHOT["PM"]["seat_label"] == "Bad Belzig"


# ---------------------------------------------------------------------------
# Wikidata snapshot
# ---------------------------------------------------------------------------


def test_snapshot_loads_with_provenance() -> None:
    assert KFZ_SNAPSHOT_META["source"] == "Wikidata Query Service"
    assert re.match(r"\d{4}-\d{2}-\d{2}T", KFZ_SNAPSHOT_META["retrieved_utc"])
    assert "P395" in KFZ_SNAPSHOT_META["sparql"]["statements"]
    assert KFZ_SNAPSHOT_META["n_codes"] == len(KFZ_SNAPSHOT)


def test_snapshot_is_much_larger_than_the_manual_table() -> None:
    # Germany has ~400 current codes plus several hundred reissued historic
    # ones; the hand-curated table had 311.
    assert len(KFZ_SNAPSHOT) >= 600
    assert len(KFZ_DISTRICTS) >= len(KFZ_SNAPSHOT)
    assert len(KFZ_DISTRICTS) > 2 * len(MANUAL_OVERRIDES)


def test_snapshot_records_are_complete_and_in_germany() -> None:
    for code, r in KFZ_SNAPSHOT.items():
        assert r["seat"] and r["seat_qid"].startswith("Q"), code
        assert isinstance(r["historic"], bool)
        assert 47.2 < r["lat"] < 55.1 and 5.8 < r["lon"] < 15.1, code


def test_overrides_win_over_snapshot() -> None:
    for code, name in MANUAL_OVERRIDES.items():
        assert KFZ_DISTRICTS[code] == name
    # A code where Wikidata only knows the current district seat: the curated
    # historic seat wins.
    assert KFZ_SNAPSHOT["WÜM"]["seat_label"] == "Cham"
    assert KFZ_DISTRICTS["WÜM"] == "Waldmünchen"
    # Codes absent from the manual table come from the snapshot.
    snapshot_only = set(KFZ_SNAPSHOT) - set(MANUAL_OVERRIDES)
    assert len(snapshot_only) > 300
    for code in snapshot_only:
        assert KFZ_DISTRICTS[code] == KFZ_SNAPSHOT[code]["seat"]


def test_known_codes() -> None:
    for code, city in (("UL", "Ulm"), ("KA", "Karlsruhe"), ("B", "Berlin"),
                       ("M", "München")):
        assert KFZ_DISTRICTS[code].startswith(city)
        assert KFZ_SNAPSHOT[code]["seat_label"] == city   # Wikidata agrees
    assert KFZ_SNAPSHOT["UL"]["state"] == "Baden-Württemberg"
    assert KFZ_SNAPSHOT["M"]["state"] == "Bayern"


def test_reissued_and_retired_codes() -> None:
    # EHI (Ehingen, reissued 2013 next to the Ulm clips) was missing from the
    # manual table; Wikidata still records only its 1973 end date.
    assert KFZ_DISTRICTS["EHI"].startswith("Ehingen")
    assert KFZ_SNAPSHOT["EHI"]["historic"] is True
    # GDR-era letters (ended 1990) must not vote: Y is now the Bundeswehr.
    assert "Y" not in KFZ_SNAPSHOT and "YO" not in KFZ_SNAPSHOT
    # L is Leipzig, not Erfurt (whose GDR-era Bezirk letter L is in Wikidata).
    assert KFZ_SNAPSHOT["L"]["seat_label"] == "Leipzig"


def test_plate_parser_sees_snapshot_codes() -> None:
    from monocular_osm.plate_anchor import _candidate_codes

    assert _candidate_codes("EHI-AB 12") == ["EHI"]
    assert _candidate_codes("UL-AB 123") == ["UL"]
