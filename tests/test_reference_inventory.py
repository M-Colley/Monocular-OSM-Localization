"""The one served-versus-sampled rule shared by every served-pool check."""
from copy import deepcopy

import numpy as np
import pytest

from monocular_osm import reference_inventory as inventory
from monocular_osm.kartaview_vpr import _reference_snapshot


def pool(n_refs=200, missing=(3, 17), *, recorded=True):
    """A capped snapshot of ``n_refs`` sorted IDs; ``missing`` rows got no URL."""
    listed = sorted(({"id": f"{1000 + i}", "lat": 48.1 + i * 1e-4, "lon": 11.5 - i * 1e-4}
                     for i in range(n_refs)), key=lambda r: r["id"])
    absent = [listed[i] for i in missing]
    served = [r for i, r in enumerate(listed) if i not in set(missing)]
    fetch = {"listing": "tiles", "listing_stats_known": True, "lost": 0,
             "url_lookups": n_refs, "url_missing": len(absent), "snapshot": _reference_snapshot(listed)}
    if recorded:
        fetch["url_missing_refs"] = deepcopy(absent)
    ids = np.array([f"mapillary:{r['id']}" for r in served])
    latlon = np.array([[r["lat"], r["lon"]] for r in served], np.float64)
    return fetch, ids, latlon


def test_bound_is_five_or_one_percent():
    assert [inventory.url_missing_bound(n) for n in (1, 499, 500, 1200, 1500)] == [5, 5, 5, 12, 15]


def test_snapshot_digest_is_the_fetch_snapshot_identity():
    refs = [{"id": "7", "lat": 48.0, "lon": 11.0, "url": "expiring"}, {"id": "8", "lat": 48.5, "lon": 11}]
    assert inventory.snapshot_sha256(refs) == _reference_snapshot(refs)["refs_sha256"]


@pytest.mark.parametrize("extra", [{}, {"url_missing": 0, "url_missing_refs": []},
                                   {"url_missing": 2, "url_missing_refs": "stale diagnostics"}])
def test_fully_served_pool_is_accepted_exactly_as_before(extra):
    fetch, ids, latlon = pool(missing=())
    fetch.pop("url_missing")
    fetch.update(extra)
    record = inventory.served_inventory(fetch, ids, latlon)
    assert record == {"rule": inventory.RULE, "n_refs": 200, "served": 200,
                      "url_missing": 0, "evidence": "all_served"}


def test_recorded_url_less_rows_must_rebuild_the_exact_snapshot():
    fetch, ids, latlon = pool()
    record = inventory.served_inventory(fetch, ids, latlon)
    assert record["evidence"] == "url_missing_refs" and record["served"] + record["url_missing"] == 200
    assert record["url_missing_ids"] == ["mapillary:1003", "mapillary:1017"]


def test_count_without_ids_is_accepted_as_weaker_evidence():
    fetch, ids, latlon = pool(recorded=False)
    record = inventory.served_inventory(fetch, ids, latlon)
    assert record["evidence"] == "url_missing_count" and "url_missing_ids" not in record


@pytest.mark.parametrize("recorded", [True, False])
def test_bound_holds_at_one_percent_and_refuses_beyond(recorded):
    fetch, ids, latlon = pool(1500, range(15), recorded=recorded)
    assert inventory.served_inventory(fetch, ids, latlon)["url_missing"] == 15
    fetch, ids, latlon = pool(1500, range(16), recorded=recorded)
    assert inventory.served_inventory(fetch, ids, latlon) is None


@pytest.mark.parametrize("fault", [
    "no_count", "bool_count", "text_count", "zero_count", "count_mismatch", "row_count",
    "served_id", "duplicate_id", "substituted_id", "moved_row", "moved_served", "extra_key",
    "nan_coordinate", "bool_coordinate", "rows_not_list", "unprefixed", "repeated_served",
    "float32_coordinates", "no_snapshot", "float_n_refs"])
def test_unaccounted_or_malformed_shortfall_is_refused(fault):
    fetch, ids, latlon = pool()
    rows = fetch["url_missing_refs"]
    if fault == "no_count": fetch.pop("url_missing")
    elif fault == "bool_count": fetch["url_missing"] = True
    elif fault == "text_count": fetch["url_missing"] = "2"
    elif fault == "zero_count": fetch["url_missing"] = 0
    elif fault == "count_mismatch": fetch["url_missing"] = 3
    elif fault == "row_count": rows.pop()
    elif fault == "served_id": rows[0]["id"] = ids[0].split(":", 1)[1]
    elif fault == "duplicate_id": rows[1] = dict(rows[0])
    elif fault == "substituted_id": rows[0]["id"] = "999999"
    elif fault == "moved_row": rows[0]["lat"] += 1e-6
    elif fault == "moved_served": latlon[5, 1] += 1e-6
    elif fault == "extra_key": rows[0]["url"] = None
    elif fault == "nan_coordinate": rows[0]["lon"] = float("nan")
    elif fault == "bool_coordinate": rows[0]["lat"] = True
    elif fault == "rows_not_list": fetch["url_missing_refs"] = tuple(rows)
    elif fault == "unprefixed": ids = np.array([value.split(":", 1)[1] for value in ids])
    elif fault == "repeated_served": ids[1] = ids[0]
    elif fault == "float32_coordinates": latlon = latlon.astype(np.float32)
    elif fault == "no_snapshot": fetch.pop("snapshot")
    else: fetch["snapshot"]["n_refs"] = 200.0
    assert inventory.served_inventory(fetch, ids, latlon) is None


def test_count_only_shortfall_must_equal_the_recorded_count():
    fetch, ids, latlon = pool(recorded=False)
    assert inventory.served_inventory(fetch, ids[:-1], latlon[:-1]) is None
    fetch["url_missing"] = 1
    assert inventory.served_inventory(fetch, ids, latlon) is None


def test_recorded_rows_parse_strictly_and_absent_means_unrecorded():
    fetch, _, _ = pool()
    assert [r["id"] for r in inventory.recorded_url_missing_refs(fetch)] == ["1003", "1017"]
    assert inventory.recorded_url_missing_refs({"url_missing": 2}) is None
    with pytest.raises(ValueError):
        inventory.recorded_url_missing_refs({"url_missing_refs": [{"id": ""}]})
