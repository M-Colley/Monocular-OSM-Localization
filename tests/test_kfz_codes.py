"""Sanity tests for the KFZ district-code table (audit kfz_codes:71)."""

from __future__ import annotations

import re
from collections import Counter

from src.kfz_codes import KFZ_DISTRICTS


def test_all_keys_are_valid_prefixes() -> None:
    # A key with whitespace/digits can never be matched by the plate parser.
    for k in KFZ_DISTRICTS:
        assert re.fullmatch(r"[A-ZÄÖÜ]{1,3}", k), f"dead key: {k!r}"


def test_no_colliding_geocode_strings() -> None:
    # Two codes mapping to the same query would geocode one of them ~hundreds
    # of km wrong (FB/FDB were both 'Friedberg', 250 km apart).
    dupes = [v for v, n in Counter(KFZ_DISTRICTS.values()).items() if n > 1]
    assert dupes == []


def test_friedberg_codes_are_state_qualified() -> None:
    assert "Bayern" in KFZ_DISTRICTS["FDB"]
    assert "Hessen" in KFZ_DISTRICTS["FB"]


def test_corrected_seats() -> None:
    assert KFZ_DISTRICTS["MN"] == "Mindelheim"      # Unterallgäu, not Memmingen
    assert KFZ_DISTRICTS["MM"] == "Memmingen"
    assert KFZ_DISTRICTS["EN"] == "Schwelm"         # Ennepe-Ruhr, not Hagen
    assert KFZ_DISTRICTS["HA"] == "Hagen"
    assert KFZ_DISTRICTS["WÜM"] == "Waldmünchen"
    assert KFZ_DISTRICTS["ER"] == "Erlangen"
    assert " ER" not in KFZ_DISTRICTS               # old dead key
