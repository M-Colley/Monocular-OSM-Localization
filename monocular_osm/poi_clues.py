"""Retail POI hypotheses from literal scene text and a named-city inventory.

Retain all feasible branches. These hypotheses are conditional on the stated
visibility/time assumptions; they are not calibrated position observations.
Network access occurs only in the explicitly called inventory fetch helper.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import socket

from .geo import LocalFrame, geodesic_distance_m
from .osm_gazetteer import _norm_name

RETAIL_AMENITIES = "cafe|restaurant|fast_food|bar|pub|food_court|ice_cream"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def fetch_city_retail(city, directory, *, offline=False, endpoint_override=None, transport="post"):
    """One bounded city-bbox retail snapshot, shared by every clip in that city."""
    directory = Path(directory)
    if transport not in {"post", "get"}:
        raise ValueError("retail transport must be post or get")
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

    # Same patience as the road-graph fetch (osm_data): a busy Overpass
    # (502/504) usually recovers within minutes; two quick tries lost the
    # whole OCR cue on runs that otherwise completed.
    @transient_retry()
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
            unique.append({**deepcopy(poi), "osm_ids": [poi["id"]]})
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


@dataclass(frozen=True)
class PoiClueConfig:
    min_confidence: float = 0.8
    max_gap_s: float = 15.0
    visibility_m: float = 100.0
    max_speed_mps: float = 20.0

    def __post_init__(self):
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("POI OCR confidence must lie in [0,1]")
        for name in ("max_gap_s", "visibility_m", "max_speed_mps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


def build_poi_hypotheses(detections, inventory, config=None):
    """Pure inference over an existing inventory, retaining every branch.

    Detections are dictionaries with text, confidence, t_sec and optional bbox.
    No video, map service, predictions, or ground-truth input is read here.
    A cached inventory with a declared content digest must match that digest.
    """
    config = config or PoiClueConfig()
    source_pois = inventory["pois"]
    snapshot_sha256 = _digest(source_pois)
    if inventory.get("snapshot_sha256", snapshot_sha256) != snapshot_sha256:
        raise ValueError("retail inventory fingerprint changed")
    if len(source_pois) > 100_000:
        raise ValueError("retail inventory exceeds the bounded inventory limit")
    for poi in source_pois:
        if (not math.isfinite(poi["lat"]) or not math.isfinite(poi["lon"])
                or abs(poi["lat"]) > 90 or abs(poi["lon"]) > 180):
            raise ValueError("POI coordinates must be finite WGS84")
    if len({poi["id"] for poi in source_pois}) != len(source_pois):
        raise ValueError("retail inventory contains duplicate source IDs")
    detections = [dict(d) for d in detections]
    for detection in detections:
        if (not math.isfinite(detection["t_sec"]) or not math.isfinite(detection["confidence"])
                or not 0 <= detection["confidence"] <= 1):
            raise ValueError("OCR times and confidence must be finite and confidence in [0,1]")
    # Defend the public boundary even when callers bypass extract_scene_text.
    from .scene_text import is_position_stamp

    accepted = [d for d in detections if not is_position_stamp(d["text"])]
    pois = deduplicate_pois(source_pois)
    matches = match_literal_clues(accepted, pois, min_confidence=config.min_confidence)
    clusters = candidate_conjunctions(matches, pois, max_gap_s=config.max_gap_s,
                                      visibility_m=config.visibility_m,
                                      max_speed_mps=config.max_speed_mps)
    return {"experimental": True, "selected_answer": None, "confidence_calibrated": False,
            "algorithm_version": 1, "config": asdict(config),
            "inventory": {"city": inventory.get("city"), "scope": inventory.get("scope"),
                          "snapshot_sha256": snapshot_sha256,
                          "n_pois": len(source_pois), "n_deduplicated_pois": len(pois)},
            "detections_sha256": _digest(detections), "n_detections": len(detections),
            "n_position_stamps_rejected": len(detections) - len(accepted),
            "matched_observations": matches, "candidate_clusters": clusters}
