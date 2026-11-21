"""Served-versus-sampled accounting for one capped Mapillary reference pool.

``kartaview_vpr._fetch_refs_mapillary`` keeps the sha1-lowest ``cap`` listed
IDs (the metadata snapshot, ``n_refs`` rows) and then asks the Graph API for
their thumbnail URLs. An ID without a URL stays in the snapshot but cannot be
downloaded, so the served (embedded) pool can be smaller than ``n_refs``.
Every check that compares a served pool with its snapshot uses
:func:`served_inventory`. The sampling itself is not changed here.

Rule ``RULE``. A served pool is the complete capped inventory when

* every snapshot member was served (``served == n_refs``, as before), or
* the fetch recorded an int ``url_missing`` with
  ``0 < url_missing <= url_missing_bound(n_refs)`` and
  ``served + url_missing == n_refs``.

In the second case, if the fetch also recorded which members had no URL
(``url_missing_refs``: their listed ``id``/``lat``/``lon``), those exact rows
must be disjoint from the served IDs and, merged with the served rows in ID
order, must rebuild the snapshot's ``refs_sha256``. That proves which sampled
IDs were not served. Fetch records without the rows (for example restored from
metadata cached before the rows were kept) are accepted on the count alone and
reported as the weaker ``url_missing_count`` evidence.

Listing completeness (coverage tiles, zero lost tiles, known statistics) is a
separate condition that each caller keeps checking itself.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import numbers

RULE = "mapillary-served-inventory-url-missing-v1"
PREFIX = "mapillary:"


def url_missing_bound(n_refs: int) -> int:
    """Most snapshot members a complete pool may lack for want of a thumbnail URL."""
    return max(5, n_refs // 100)


def snapshot_sha256(refs) -> str:
    """Digest of ordered ``{id, lat, lon}`` rows (``_reference_snapshot``'s identity)."""
    content = [[str(r["id"]), float(r["lat"]), float(r["lon"])] for r in refs]
    return hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()


def _coordinate(value) -> bool:
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and math.isfinite(value))


def recorded_url_missing_refs(fetch):
    """The fetch's recorded URL-less snapshot rows, or ``None`` when not recorded.

    Raises ``ValueError`` for a malformed record: rows must be exactly
    ``{id, lat, lon}`` with a nonempty unique string ID and finite coordinates.
    """
    rows = fetch.get("url_missing_refs") if isinstance(fetch, Mapping) else None
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError("recorded URL-less references must be a list")
    result = []
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {"id", "lat", "lon"}
                or not isinstance(row["id"], str) or not row["id"]
                or not _coordinate(row["lat"]) or not _coordinate(row["lon"])):
            raise ValueError("invalid recorded URL-less reference")
        result.append({"id": row["id"], "lat": row["lat"], "lon": row["lon"]})
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("recorded URL-less references repeat an ID")
    return result


def _account(fetch, served_ids, served_latlon, prefix):
    snapshot = fetch["snapshot"]
    if not isinstance(snapshot, Mapping):
        return None
    n_refs = snapshot.get("n_refs")
    ids = [str(value) for value in served_ids]
    served = len(ids)
    base = {"rule": RULE, "n_refs": n_refs, "served": served}
    if n_refs == served:
        return {**base, "url_missing": 0, "evidence": "all_served"}
    missing = fetch.get("url_missing")
    if (type(n_refs) is not int or type(missing) is not int
            or not 0 < missing <= url_missing_bound(n_refs) or served + missing != n_refs):
        return None
    rows = recorded_url_missing_refs(fetch)
    if rows is None:
        return {**base, "url_missing": missing, "evidence": "url_missing_count"}
    if len(rows) != missing or not all(value.startswith(prefix) for value in ids):
        return None
    plain = [value[len(prefix):] for value in ids]
    absent = [row["id"] for row in rows]
    if len(set(plain)) != served or set(plain) & set(absent):
        return None
    coordinates = [[float(v) for v in point] for point in served_latlon]
    if len(coordinates) != served or any(len(point) != 2 for point in coordinates):
        return None
    rebuilt = sorted([{"id": key, "lat": point[0], "lon": point[1]}
                      for key, point in zip(plain, coordinates)] + rows, key=lambda row: row["id"])
    if snapshot_sha256(rebuilt) != snapshot.get("refs_sha256"):
        return None
    return {**base, "url_missing": missing, "evidence": "url_missing_refs",
            "url_missing_ids": [prefix + key for key in sorted(absent)]}


def served_inventory(fetch, served_ids, served_latlon, *, prefix=PREFIX):
    """Account for a served pool against its fetch snapshot under ``RULE``.

    ``served_ids`` are the embedded provider IDs (``prefix`` + ID) and
    ``served_latlon`` their coordinates, in pool order. Returns a small record
    (``rule``, ``n_refs``, ``served``, ``url_missing``, ``evidence`` and, for
    ID-level evidence, ``url_missing_ids``), or ``None`` when the pool is not
    the complete capped inventory or its evidence is malformed.
    """
    try:
        return _account(fetch, served_ids, served_latlon, prefix)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
