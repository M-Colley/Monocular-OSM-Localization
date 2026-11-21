"""Explicit bounded 2x2 full-city retail fallback; no OCR or route inputs.

Use only after the ordinary single city query fails. Every tile is fixed
before calls, and every tile must complete before an inventory is published.
The original fetch_city_retail cache and its outputs are never changed.
"""
from __future__ import annotations
from copy import deepcopy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import socket
import time
from urllib.parse import urlparse

from .poi_clues import RETAIL_AMENITIES, _digest

ENDPOINT = "https://overpass.private.coffee/api/interpreter"
MAX_ELEMENTS = 100_000
MAX_BODY_BYTES = 32 * 1024**2
MAX_SNAPSHOT_BYTES = 64 * 1024**2
ATTEMPTS_PER_TILE = 2
_ALLOWED_TAGS = {"name", "brand", "operator", "short_name", "alt_name", "shop", "amenity",
                 "branch", "addr:street", "addr:housenumber"}


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024**2): digest.update(block)
    return digest.hexdigest()


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _coordinate(value):
    return f"{value:.17f}".rstrip("0").rstrip(".")


def _query(box):
    west, south, east, north = box
    bbox = ",".join(_coordinate(v) for v in (south, west, north, east))
    return (f'[out:json][timeout:90];(nwr["shop"]({bbox});'
            f'nwr["amenity"~"^({RETAIL_AMENITIES})$"]({bbox}););out center tags;')


def plan_city_retail_tiles(city, bbox_wsen, *, bbox_source, endpoint=ENDPOINT):
    """Plan every quadrant of a public named-city box; never accept scene cues."""
    box = list(bbox_wsen)
    if (not isinstance(city, str) or not city.strip() or len(box) != 4 or not all(map(_number, box))
            or not -180 <= box[0] < box[2] <= 180 or not -90 <= box[1] < box[3] <= 90
            or max(box[2] - box[0], box[3] - box[1]) > 1.):
        raise ValueError("fallback needs a finite bounded full named-city WGS84 box")
    source = deepcopy(bbox_source)
    if (not isinstance(source, dict) or source.get("city") != city
            or source.get("scope") != "named_city_bbox" or source.get("source_kind") != "public_geocode"
            or not isinstance(source.get("source_sha256"), str) or len(source["source_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in source["source_sha256"])):
        raise ValueError("full city box needs fingerprinted public named-city provenance")
    address = urlparse(endpoint)
    if (address.scheme != "https" or not address.hostname or address.username or address.password
            or address.query or address.fragment or not address.path.endswith("/interpreter")):
        raise ValueError("fallback endpoint must be an HTTPS public interpreter without credentials/query")
    west, south, east, north = map(float, box)
    xs, ys = [west, (west + east) / 2, east], [south, (south + north) / 2, north]
    tiles = []
    for row in range(2):
        for column in range(2):
            tile = [xs[column], ys[row], xs[column + 1], ys[row + 1]]
            query = _query(tile)
            tiles.append({"id": f"r{row}c{column}", "bbox_wsen": tile, "query": query,
                          "query_sha256": hashlib.sha256(query.encode()).hexdigest()})
    plan = {"schema": 1, "kind": "full-city-retail-2x2-v1", "city": city,
        "scope": "named_city_bbox", "bbox_wsen": [west, south, east, north], "bbox_source": source,
        "endpoint": endpoint, "transport": "get", "grid": [2, 2], "max_elements_before_merge": MAX_ELEMENTS,
        "attempts_per_tile": ATTEMPTS_PER_TILE, "max_response_bytes": MAX_BODY_BYTES,
        "max_snapshot_bytes": MAX_SNAPSHOT_BYTES, "http_timeout_s": [8, 110], "retry_wait_s": 2.,
        "tiles": tiles, "coverage": "all four closed quadrants of the full parent box; shared boundaries"}
    plan["plan_sha256"] = _digest(plan)
    return plan


def _validate_plan(plan):
    if not isinstance(plan, dict): raise ValueError("supply a fixed full-city tile plan")
    expected = plan_city_retail_tiles(plan.get("city"), plan.get("bbox_wsen", []),
        bbox_source=plan.get("bbox_source"), endpoint=plan.get("endpoint", ""))
    if expected != plan: raise ValueError("fixed full-city tile plan/config/query changed")


def _request_tile(endpoint, query):
    """One GET, with bounded response storage; retries belong to the caller."""
    import requests
    from osmnx._http import _config_dns, _get_http_headers
    from .net import prefer_reachable_host
    previous = socket.getaddrinfo
    try:
        with prefer_reachable_host(endpoint):
            _config_dns(endpoint)
            with requests.get(endpoint, params={"data": query}, headers=_get_http_headers(),
                              timeout=(8, 110), stream=True) as response:
                body, complete = bytearray(), True
                for block in response.iter_content(chunk_size=1024**2):
                    if len(body) + len(block) > MAX_BODY_BYTES:
                        body.extend(block[:MAX_BODY_BYTES - len(body)])
                        complete = False
                        break
                    body.extend(block)
                return {"status": response.status_code, "body": bytes(body), "complete": complete}
    finally:
        socket.getaddrinfo = previous


def _payload(body):
    def invalid(value): raise ValueError("non-finite Overpass JSON constant")
    payload = json.loads(body.decode("utf-8-sig"), parse_constant=invalid)
    if not isinstance(payload, dict) or payload.get("remark") or not isinstance(payload.get("elements"), list):
        raise ValueError("Overpass tile is incomplete or has no element array")
    if len(payload["elements"]) > MAX_ELEMENTS:
        raise ValueError("tile exceeds the pre-merge inventory element budget")
    return payload


def _response_digest(records):
    return _digest([{key: record[key] for key in ("id", "query_sha256", "response_sha256")}
                    for record in records])


def _merge_elements(payloads):
    # Count BEFORE deduplication: overlapping boundaries do not evade the cap.
    if sum(len(payload["elements"]) for payload in payloads) > MAX_ELEMENTS:
        raise ValueError("aggregate tile elements exceed the pre-merge inventory budget")
    by_id = {}
    for payload in payloads:
        for element in payload["elements"]:
            if (not isinstance(element, dict) or element.get("type") not in {"node", "way", "relation"}
                    or isinstance(element.get("id"), bool) or not isinstance(element.get("id"), int)
                    or element["id"] <= 0):
                raise ValueError("retail response has an invalid OSM type/id")
            key = (element["type"], element["id"])
            if key in by_id and _digest(by_id[key]) != _digest(element):
                raise ValueError(f"conflicting duplicate OSM element {key[0]}/{key[1]}")
            by_id[key] = element
    result = []
    order = {"node": 0, "way": 1, "relation": 2}
    for kind, identifier in sorted(by_id, key=lambda key: (order[key[0]], key[1])):
        element = by_id[kind, identifier]
        tags = element.get("tags", {})
        if not isinstance(tags, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in tags.items()):
            raise ValueError("OSM retail tags must be string mappings")
        if not any(tags.get(k) for k in ("name", "brand", "operator")): continue
        location = element if kind == "node" else element.get("center", {})
        if (not isinstance(location, dict) or not _number(location.get("lat")) or not _number(location.get("lon"))
                or abs(location["lat"]) > 90 or abs(location["lon"]) > 180):
            raise ValueError("named retail POI lacks finite WGS84 coordinates")
        result.append({"id": f"{kind}/{identifier}", "lat": float(location["lat"]), "lon": float(location["lon"]),
                       "tags": {k: tags[k] for k in sorted(tags) if k in _ALLOWED_TAGS}})
    return result


def _cached(plan, directory):
    receipt = directory / "complete.json"
    if not receipt.is_file(): return None
    complete = json.loads(receipt.read_text(encoding="utf-8"))
    snapshot_file = directory / "inventory.json"
    if (complete.get("plan_sha256") != plan["plan_sha256"] or not snapshot_file.is_file()
            or snapshot_file.stat().st_size > MAX_SNAPSHOT_BYTES or _sha(snapshot_file) != complete.get("inventory_file_sha256")):
        raise ValueError("completed tiled inventory identity/content changed")
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    tiles = snapshot.get("tile_responses", [])
    if (snapshot.get("snapshot_sha256") != _digest(snapshot.get("pois"))
            or len(snapshot.get("pois", [])) > MAX_ELEMENTS or len(tiles) != 4
            or [tile.get("id") for tile in tiles] != [tile["id"] for tile in plan["tiles"]]
            or snapshot.get("response_sha256") != _response_digest(tiles) or snapshot.get("plan_sha256") != plan["plan_sha256"]
            or complete.get("response_sha256") != snapshot.get("response_sha256")
            or snapshot.get("city") != plan["city"] or snapshot.get("scope") != "named_city_bbox"
            or snapshot.get("bbox_wsen") != plan["bbox_wsen"] or snapshot.get("complete_parent_bbox_coverage") is not True
            or json.loads((directory / "plan.json").read_text(encoding="utf-8")) != plan):
        raise ValueError("completed tiled inventory coverage/digest changed")
    payloads = []
    for tile, planned in zip(tiles, plan["tiles"]):
        path = (directory / tile["response_file"]).resolve(strict=True)
        if (not path.is_relative_to(directory.resolve()) or path.stat().st_size > MAX_BODY_BYTES
                or _sha(path) != tile["response_sha256"] or tile.get("complete") is not True):
            raise ValueError("completed tile response changed or escaped its directory")
        if tile.get("query_sha256") != planned["query_sha256"]:
            raise ValueError("completed tile does not identify the fixed coverage query")
        payloads.append(_payload(path.read_bytes()))
    if _digest(_merge_elements(payloads)) != snapshot["snapshot_sha256"]:
        raise ValueError("cached inventory does not match all complete raw tile responses")
    return snapshot


def _fetch_one(plan, tile, directory):
    attempts = []
    for number in range(1, ATTEMPTS_PER_TILE + 1):
        record = {"attempt": number, "tile_id": tile["id"], "query_sha256": tile["query_sha256"],
                  "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "complete": False}
        started = time.monotonic()
        try:
            response = _request_tile(plan["endpoint"], tile["query"])
            body = response["body"]
            filename = f"{tile['id']}_attempt{number}.response"
            (directory / filename).write_bytes(body)
            record.update(response_file=filename, response_sha256=hashlib.sha256(body).hexdigest(),
                          response_bytes=len(body), http_status=response["status"], complete_body=response["complete"])
            if len(body) > MAX_BODY_BYTES or not response["complete"]:
                raise ValueError("retail response exceeded the bounded body limit")
            if response["status"] != 200: raise RuntimeError(f"Overpass HTTP {response['status']}")
            payload = _payload(body)
            record.update(complete=True, n_elements=len(payload["elements"]),
                          osm_base_timestamp=(payload.get("osm3s") or {}).get("timestamp_osm_base"))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["elapsed_s"] = time.monotonic() - started
        attempts.append(record)
        _write(directory / f"{tile['id']}_attempts.json", attempts)
        if record["complete"]:
            return payload, record
        if number < ATTEMPTS_PER_TILE: time.sleep(2.)
    return None, attempts[-1]


def fetch_tiled_city_retail(plan, directory, *, offline=False):
    """Fetch the frozen 2x2 plan into a NEW family, or validate a complete cache.

    Failures retain their immutable plan, all attempts and any raw responses.
    Partial directories are never resumed/replaced or interpreted as a city.
    """
    _validate_plan(plan)
    directory = Path(directory)
    if directory.exists():
        cached = _cached(plan, directory)
        if cached is not None: return cached
        if any(directory.iterdir()):
            raise FileExistsError("partial tiled inventory is preserved; choose a new output family")
    if offline: raise FileNotFoundError("no complete fixed-plan tiled city inventory is cached")
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "plan.json", plan)  # All four queries frozen BEFORE the first request.
    payloads, records, total_elements, over_budget = [], [], 0, False
    for tile in plan["tiles"]:
        payload, record = _fetch_one(plan, tile, directory)
        records.append({"id": tile["id"], **record})
        if payload is not None:
            total_elements += len(payload["elements"])
            if total_elements > MAX_ELEMENTS:
                over_budget, payloads = True, []
            elif not over_budget:
                payloads.append(payload)
        _write(directory / "progress.json", {"plan_sha256": plan["plan_sha256"], "tiles": records,
                                            "inventory_available": False})
    if over_budget:
        _write(directory / "failed.json", {"reason": "aggregate_pre_merge_element_budget_exceeded",
            "elements_before_merge": total_elements, "inventory_available": False})
        raise ValueError("aggregate tile elements exceed the pre-merge inventory budget")
    if len(payloads) != 4:
        _write(directory / "failed.json", {"reason": "all_four_tiles_required", "tiles": records,
                                          "inventory_available": False})
        raise RuntimeError("full-city retail inventory unknown: all four tiles must complete")
    try:
        pois = _merge_elements(payloads)
    except Exception as error:
        _write(directory / "failed.json", {"reason": f"{type(error).__name__}: {error}",
                                          "inventory_available": False})
        raise
    snapshot = {"schema": 1, "city": plan["city"], "scope": "named_city_bbox", "bbox_wsen": plan["bbox_wsen"],
        "query": "\n".join(tile["query"] for tile in plan["tiles"]), "endpoint": plan["endpoint"],
        "transport": "get", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "response_sha256": _response_digest(records), "tile_responses": records, "plan_sha256": plan["plan_sha256"],
        "response_aggregate_definition": "ordered tile id, query SHA256 and raw complete-response SHA256",
        "complete_parent_bbox_coverage": True, "bbox_source": plan["bbox_source"],
        "elements_before_merge": sum(len(payload["elements"]) for payload in payloads),
        "pois": pois, "snapshot_sha256": _digest(pois)}
    _write(directory / "inventory.json", snapshot)
    if (directory / "inventory.json").stat().st_size > MAX_SNAPSHOT_BYTES:
        (directory / "inventory.json").rename(directory / "oversized_inventory.json")
        raise ValueError("bounded tiled inventory snapshot size exceeded")
    _write(directory / "complete.json", {"plan_sha256": plan["plan_sha256"],
        "inventory_file_sha256": _sha(directory / "inventory.json"), "response_sha256": snapshot["response_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"], "complete_parent_bbox_coverage": True})
    return _cached(plan, directory)
