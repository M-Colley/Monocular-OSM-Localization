"""Exploratory retail-POI conjunctions from frozen literal scene clues.

No GT files, predictions or clip/retailer-specific rules are accepted. All
qualifying branch combinations are retained as hypotheses, never an answer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monocular_osm.geo import LocalFrame, geodesic_distance_m
from monocular_osm.osm_gazetteer import _norm_name

RETAIL_AMENITIES = "cafe|restaurant|fast_food|bar|pub|food_court|ice_cream"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def fetch_city_retail(city, directory, *, offline=False, endpoint_override=None, transport="post"):
    """One bounded city-bbox retail snapshot, shared by every clip in that city."""
    directory.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "-", city.casefold()).strip("-")
    cache = directory / f"{tag}.json"
    if cache.exists():
        snapshot = json.loads(cache.read_text(encoding="utf-8"))
        if snapshot.get("city") == city and snapshot.get("schema") == 1:
            return snapshot
    if offline:
        raise FileNotFoundError(f"No cached retail snapshot for {city}")
    import osmnx as ox
    import requests
    from osmnx._http import _config_dns, _get_http_headers
    from monocular_osm.net import prefer_reachable_host, transient_retry

    boundary = ox.geocode_to_gdf(city)
    west, south, east, north = map(float, boundary.total_bounds)
    if max(east - west, north - south) > 1.0:
        raise ValueError(f"Refusing unexpectedly large city bbox for {city}")
    bbox = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    query = (f'[out:json][timeout:90];(nwr["shop"]({bbox});'
             f'nwr["amenity"~"^({RETAIL_AMENITIES})$"]({bbox}););out center tags;')
    endpoint = endpoint_override or ox.settings.overpass_url.rstrip("/") + "/interpreter"

    @transient_retry(attempts=2, first_wait_s=2, max_wait_s=3)
    def request():
        original_dns = socket.getaddrinfo
        try:
            with prefer_reachable_host(endpoint):
                _config_dns(endpoint)
                kwargs = {"params" if transport == "get" else "data": {"data": query}}
                response = getattr(requests, transport)(
                    endpoint, headers=_get_http_headers(), timeout=(8, 110), **kwargs)
        finally:
            socket.getaddrinfo = original_dns
        response.raise_for_status()
        payload = response.json()
        if payload.get("remark"):
            raise RuntimeError(f"Overpass returned an incomplete response: {payload['remark']}")
        if len(payload.get("elements", [])) > 100_000:
            raise ValueError("Retail response exceeded the bounded inventory limit")
        return response.content, payload

    print(f"Fetching city-wide retail inventory: {city}", flush=True)
    content, payload = request()
    pois = []
    allowed = {"name", "brand", "operator", "short_name", "alt_name", "shop", "amenity",
               "branch", "addr:street", "addr:housenumber"}
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        location = element if element["type"] == "node" else element.get("center", {})
        if not any(tags.get(k) for k in ("name", "brand", "operator")) or "lat" not in location:
            continue
        pois.append({"id": f"{element['type']}/{element['id']}",
                     "lat": float(location["lat"]), "lon": float(location["lon"]),
                     "tags": {k: v for k, v in tags.items() if k in allowed}})
    if not pois:
        raise RuntimeError(f"No named retail POIs returned for {city}; not caching a silent miss")
    snapshot = {"schema": 1, "city": city, "scope": "named_city_bbox",
                "bbox_wsen": [west, south, east, north], "query": query,
                "endpoint": endpoint, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "response_sha256": hashlib.sha256(content).hexdigest(), "pois": pois}
    snapshot["snapshot_sha256"] = _digest(pois)
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache)
    return snapshot


def _aliases(poi):
    tags = poi["tags"]
    aliases = list(poi.get("additional_aliases", []))
    for key in ("name", "brand", "short_name", "alt_name", "operator"):
        for value in str(tags.get(key) or "").split(";"):
            normalized = _norm_name(value)
            if len(normalized) < 2 or not any(c.isalpha() for c in normalized):
                continue
            if len(normalized.replace(" ", "")) < 4 and key != "brand":
                continue       # short initials require an explicit OSM brand
            aliases.append((normalized, value.strip(), key))
    return aliases


def match_literal_clues(detections, pois, *, min_confidence=0.8):
    """Conservative exact/full-name matching; long OCR typos may score >=90%."""
    from rapidfuzz.fuzz import ratio

    matched = []
    indexed = [(poi, _aliases(poi)) for poi in pois]
    for detection in detections:
        if detection["confidence"] < min_confidence:
            continue
        literal = _norm_name(detection["text"])
        if len(literal) < 2 or not any(c.isalpha() for c in literal):
            continue
        for poi, aliases in indexed:
            best = None
            for normalized, label, field in aliases:
                score = 100.0 if literal == normalized else 0.0
                if min(len(literal.replace(" ", "")), len(normalized.replace(" ", ""))) >= 6:
                    score = max(score, ratio(literal, normalized))
                if score >= 90 and (best is None or score > best["name_score"]):
                    best = {"matched_alias": label, "alias_field": field, "name_score": score}
            if best is not None:
                canonical = _norm_name(poi["tags"].get("brand") or poi["tags"].get("name") or
                                       poi["tags"].get("operator"))
                matched.append({"poi_id": poi["id"], "canonical": canonical,
                                "text": detection["text"], "confidence": detection["confidence"],
                                "t_sec": detection["t_sec"], "bbox": detection.get("bbox"), **best})
    return matched


def deduplicate_pois(pois):
    """Collapse near-identical node/way representations, retaining their IDs."""
    unique = []
    indexed = defaultdict(set)
    for poi in pois:
        aliases = _aliases(poi)
        names = {value for value, _, _ in aliases}
        candidates = set().union(*(indexed[name] for name in names)) if names else set()
        duplicate = next((index for index in sorted(candidates)
                          if geodesic_distance_m(unique[index]["lat"], unique[index]["lon"],
                                                  poi["lat"], poi["lon"]) < 25), None)
        if duplicate is not None:
            unique[duplicate]["osm_ids"].append(poi["id"])
            extra = unique[duplicate].setdefault("additional_aliases", [])
            extra.extend(a for a in aliases if a not in _aliases(unique[duplicate]))
            index = duplicate
        else:
            index = len(unique)
            unique.append({**poi, "osm_ids": [poi["id"]]})
        for name in names:
            indexed[name].add(index)
    return unique


def candidate_conjunctions(matches, pois, *, max_gap_s=15.0, visibility_m=100.0,
                           max_speed_mps=20.0):
    """Every feasible distinct-shop conjunction, with explicit temporal bounds."""
    from shapely.geometry import Point

    by_id = {p["id"]: p for p in pois}
    edges = {}
    for a_index, a in enumerate(matches):
        for b in matches[a_index + 1:]:
            if a["poi_id"] == b["poi_id"] or a["canonical"] == b["canonical"]:
                continue
            # One OCR reading matching multiple OSM aliases/representations is
            # still one clue, not independent corroboration of two shops.
            words_a, words_b = set(_norm_name(a["text"]).split()), set(_norm_name(b["text"]).split())
            if words_a <= words_b or words_b <= words_a:
                continue
            gap = abs(float(a["t_sec"]) - float(b["t_sec"]))
            if gap > max_gap_s:
                continue
            pa, pb = by_id[a["poi_id"]], by_id[b["poi_id"]]
            distance = geodesic_distance_m(pa["lat"], pa["lon"], pb["lat"], pb["lon"])
            allowed = 2 * visibility_m + max_speed_mps * gap
            if distance > allowed:
                continue
            centre = ((pa["lat"] + pb["lat"]) / 2, (pa["lon"] + pb["lon"]) / 2)
            frame = LocalFrame(*centre)
            ax, ay = frame.to_xy(pa["lat"], pa["lon"])
            bx, by = frame.to_xy(pb["lat"], pb["lon"])
            # Possible camera position at the observations' midpoint time,
            # conditional on the stated visibility and speed assumptions.
            radius = visibility_m + max_speed_mps * gap / 2
            lens = Point(ax, ay).buffer(radius).intersection(Point(bx, by).buffer(radius))
            if lens.is_empty or lens.geom_type != "Polygon":
                continue
            coordinates = list(lens.exterior.coords)
            latitudes, longitudes = frame.to_latlon([p[0] for p in coordinates],
                                                    [p[1] for p in coordinates])
            key = tuple(sorted((pa["id"], pb["id"])))
            witness = {"observations": [a, b], "gap_s": gap, "poi_separation_m": distance,
                       "maximum_separation_m": allowed,
                       "midpoint_time_sec": (a["t_sec"] + b["t_sec"]) / 2,
                       "conditional_camera_region": {"type": "Polygon", "coordinates": [
                           [[float(lon), float(lat)] for lat, lon in zip(latitudes, longitudes)]]}}
            edge = edges.setdefault(key, {"poi_ids": list(key), "pois": [by_id[k] for k in key],
                                          "witnesses": []})
            edge["witnesses"].append(witness)
    # Connected candidate clusters preserve ambiguous branch alternatives.
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    clusters, seen = [], set()
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen.update(component)
        relevant = [edge for key, edge in edges.items() if set(key) <= component]
        locations = [by_id[k] for k in sorted(component)]
        clusters.append({"poi_ids": sorted(component), "pois": locations,
                         "conjunctions": relevant,
                         "centre_latlon": [sum(p["lat"] for p in locations) / len(locations),
                                           sum(p["lon"] for p in locations) / len(locations)],
                         "distinct_names": sorted({_norm_name(p["tags"].get("brand") or
                                                              p["tags"].get("name") or
                                                              p["tags"].get("operator")) for p in locations}),
                         "n_witnesses": sum(len(e["witnesses"]) for e in relevant)})
    return sorted(clusters, key=lambda c: (-len(c["distinct_names"]), -c["n_witnesses"], c["poi_ids"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clues-dir", type=Path, default=ROOT / "output" / "scene_clue_probe")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "poi_clue_probe")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overpass-endpoint", default=None,
                        help="Optional bounded-query endpoint override from the OSM public-instance list")
    parser.add_argument("--transport", choices=["post", "get"], default="post")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"exploratory": True, "selected_answer": None,
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "bounds": {"min_ocr_confidence": 0.8, "min_name_score": 90,
                          "max_gap_s": 15, "visibility_m": 100, "max_speed_mps": 20}, "clips": []}
    failed_cities = {}
    for path in sorted(args.clues_dir.glob("*.json")):
        clues = json.loads(path.read_text(encoding="utf-8"))
        if "detections" not in clues or "city" not in clues:
            continue
        try:
            if clues["city"] in failed_cities:
                raise RuntimeError(failed_cities[clues["city"]])
            inventory = fetch_city_retail(clues["city"], args.out / "osm", offline=args.offline,
                                          endpoint_override=args.overpass_endpoint, transport=args.transport)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failed_cities.setdefault(clues["city"], error)
            record = {"id": clues["id"], "city": clues["city"], "exploratory": True,
                      "status": "retail_inventory_unavailable", "error": error,
                      "selected_answer": None, "candidate_clusters": None,
                      "created_utc": summary["created_utc"], "script_sha256": summary["script_sha256"],
                      "clue_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (args.out / path.name).write_text(json.dumps(record, indent=2), encoding="utf-8")
            summary["clips"].append({"id": clues["id"], "status": record["status"], "error": error})
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary["clips"][-1]), flush=True)
            continue
        pois = deduplicate_pois(inventory["pois"])
        matches = match_literal_clues(clues["detections"], pois)
        clusters = candidate_conjunctions(matches, pois)
        record = {"id": clues["id"], "city": clues["city"], "exploratory": True,
                  "status": "ok",
                  "created_utc": summary["created_utc"], "script_sha256": summary["script_sha256"],
                  "selected_answer": None, "clue_source": str(path),
                  "clue_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "osm_snapshot_sha256": inventory["snapshot_sha256"],
                  "bounds": summary["bounds"], "matched_observations": matches,
                  "candidate_clusters": clusters}
        (args.out / path.name).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        row = {"id": clues["id"], "n_retail_pois": len(inventory["pois"]),
               "status": "ok",
               "n_distinct_retail_pois": len(pois),
               "n_matched_observations": len(matches), "n_candidate_clusters": len(clusters),
               "candidate_centres": [c["centre_latlon"] for c in clusters]}
        summary["clips"].append(row)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    return 1 if failed_cities else 0


if __name__ == "__main__":
    raise SystemExit(main())
