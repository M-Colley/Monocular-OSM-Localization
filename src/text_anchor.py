"""Turn recovered scene text into absolute position anchors.

Geocodes OCR detections (:mod:`scene_text`) into lat/lon points, keeps
those that land inside the city, and exposes them to the matcher two
ways:

* :func:`score_candidates_by_anchors` — distance from each candidate
  walk to the nearest anchor, for a consensus re-rank channel.
* :func:`anchor_seed_nodes` — graph nodes near the anchors, to *seed
  enumeration* so the anchored area is represented in the candidate
  pool even when drift would otherwise exclude it. This is the part
  that attacks the enumeration failure (re-ranking alone can't fix a
  pool that doesn't contain the answer).

Geocoding (network, rate-limited) is injected via ``geocode_fn`` and
cached to JSON, so tests and re-runs never hit Nominatim.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .osm_data import RoadGraph
from .scene_text import SceneText

# A geocoder: query string -> (lat, lon) or None if not found.
GeocodeFn = Callable[[str], "tuple[float, float] | None"]


@dataclass(frozen=True)
class PoiAnchor:
    """A geocoded point of interest read from the video."""
    name: str          # the OCR text that geocoded
    lat: float
    lon: float
    confidence: float  # OCR confidence of the source detection
    t_sec: float       # when in the video it was seen


@dataclass(frozen=True)
class StreetAnchor:
    """An OSM street name read off a sign in the video.

    Stronger and more route-relevant than a POI: the camera is *on* (or
    adjacent to) this street, and the anchor is the street's own graph
    geometry, not a nearby landmark. ``node_ids`` are the graph nodes on
    edges carrying this name — ideal enumeration seeds. Only available
    when plates are legible (true-4K source).
    """
    name: str          # canonical OSM street name
    ocr_text: str      # what OCR actually read
    confidence: float  # OCR confidence
    match_ratio: float # fuzzy-match similarity to the OSM name
    node_ids: tuple    # graph nodes on this street
    t_sec: float = 0.0 # when the plate was read — a temporally-valid
                       # "you are here" time (car is ON this street then)


_NON_ALNUM = re.compile(r"[^0-9A-Za-zÀ-ÿ]+")


def is_geocodable_text(text: str, *, min_letters: int = 4) -> bool:
    """Heuristic gate before spending a (rate-limited) geocode call.

    Rejects pure numbers, times/dates, and short or mostly-punctuation
    fragments. We deliberately keep the bar low — the bbox filter in
    :func:`geocode_texts` is what actually rejects bad geocodes; this
    just avoids obviously-useless queries (``"535"``, ``"22-6h"``).
    """
    letters = re.sub(r"[^A-Za-zÀ-ÿ]", "", text)
    if len(letters) < min_letters:
        return False
    # Reject things that are mostly digits/punctuation (e.g. "29.13.9.2023").
    if len(letters) < 0.5 * len(_NON_ALNUM.sub("", text)):
        return False
    return True


def _clean(text: str) -> str:
    return _NON_ALNUM.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------


def default_geocode_fn(cache_path: Path | None = None) -> GeocodeFn:
    """Nominatim-backed geocoder (via osmnx) with on-disk memoization
    and the 1 req/s courtesy delay Nominatim asks for."""
    import time

    import osmnx as ox

    cache: dict[str, list | None] = {}
    cp = Path(cache_path) if cache_path else None
    if cp and cp.exists():
        try:
            cache = json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    def _fn(query: str) -> tuple[float, float] | None:
        if query in cache:
            v = cache[query]
            return (v[0], v[1]) if v else None
        try:
            lat, lon = ox.geocode(query)
            cache[query] = [float(lat), float(lon)]
            result: tuple[float, float] | None = (float(lat), float(lon))
        except Exception:
            cache[query] = None
            result = None
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(1.1)
        return result

    return _fn


def _graph_bbox_latlon(road: RoadGraph) -> tuple[float, float, float, float]:
    """(min_lat, min_lon, max_lat, max_lon) of the graph, with a small
    margin, used to reject geocodes that land outside the city."""
    from pyproj import Transformer

    xs = np.array([d["x"] for _, d in road.graph.nodes(data=True)], dtype=float)
    ys = np.array([d["y"] for _, d in road.graph.nodes(data=True)], dtype=float)
    t = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform([xs.min(), xs.max()], [ys.min(), ys.max()])
    margin = 0.02  # ~2 km
    return (min(lat) - margin, min(lon) - margin,
            max(lat) + margin, max(lon) + margin)


def _select_by_time_strata(
    dets: list[SceneText], max_queries: int, n_buckets: int
) -> list[SceneText]:
    """Pick up to ``max_queries`` detections with temporal COVERAGE.

    Pure global confidence sorting starves the start/end of the clip: a
    prominent sign read repeatedly mid-route crowds the query budget,
    leaving the early route with no geocoded anchor and forcing the
    position to be extrapolated back from a late anchor (the Ulm start
    weakness). Round-robin across equal-time buckets instead, taking the
    most confident still-unused detection from each bucket per round, so
    every part of the route gets a geocode attempt within the budget.
    """
    if n_buckets <= 1 or len(dets) <= max_queries:
        return sorted(dets, key=lambda d: -d.confidence)[:max_queries]
    ts = [d.t_sec for d in dets]
    tmin, tmax = min(ts), max(ts)
    span = max(tmax - tmin, 1e-6)
    buckets: dict[int, list[SceneText]] = {}
    for d in dets:
        b = min(n_buckets - 1, int((d.t_sec - tmin) / span * n_buckets))
        buckets.setdefault(b, []).append(d)
    for b in buckets:
        buckets[b].sort(key=lambda d: -d.confidence)
    idx = {b: 0 for b in buckets}
    out: list[SceneText] = []
    while len(out) < max_queries and any(idx[b] < len(buckets[b]) for b in buckets):
        for b in sorted(buckets):
            if len(out) >= max_queries:
                break
            if idx[b] < len(buckets[b]):
                out.append(buckets[b][idx[b]])
                idx[b] += 1
    return out


def geocode_texts(
    detections: list[SceneText],
    city: str,
    road: RoadGraph,
    *,
    geocode_fn: GeocodeFn,
    min_confidence: float = 0.5,
    max_queries: int = 25,
    time_buckets: int = 0,
) -> list[PoiAnchor]:
    """Geocode the most confident, plausible detections into anchors.

    Each unique cleaned text is queried as ``"<text>, <city>"``. A result
    is kept only if it lands inside the city's bounding box — this is the
    key noise filter: a stray brand name that geocodes to another country
    is dropped. Returns anchors sorted by OCR confidence (best first),
    deduplicated by name.

    ``time_buckets`` > 1 spreads the query budget across that many equal
    time slices of the clip (round-robin by confidence within each), so
    the start/end of the route get anchors instead of being crowded out
    by a prominent mid-route sign. 0/1 keeps the historic global-confidence
    behaviour. Downstream cluster/temporal filtering still rejects any bad
    anchor the wider net pulls in, so coverage is gained without losing
    robustness.
    """
    bbox = _graph_bbox_latlon(road)
    min_lat, min_lon, max_lat, max_lon = bbox

    # Best (highest-confidence) detection per cleaned text, conf-sorted.
    best: dict[str, SceneText] = {}
    for d in detections:
        if d.confidence < min_confidence:
            continue
        cleaned = _clean(d.text)
        if not is_geocodable_text(cleaned):
            continue
        prev = best.get(cleaned.casefold())
        if prev is None or d.confidence > prev.confidence:
            best[cleaned.casefold()] = SceneText(cleaned, d.confidence, d.t_sec)

    ordered = _select_by_time_strata(list(best.values()), max_queries, time_buckets)
    anchors: list[PoiAnchor] = []
    seen: set[str] = set()
    for d in ordered:
        latlon = geocode_fn(f"{d.text}, {city}")
        if latlon is None:
            continue
        lat, lon = latlon
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue  # outside the city → almost certainly a wrong match
        key = d.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        anchors.append(PoiAnchor(name=d.text, lat=lat, lon=lon,
                                 confidence=d.confidence, t_sec=d.t_sec))
    return anchors


# ---------------------------------------------------------------------------
# Using anchors in the matcher (projected-CRS geometry)
# ---------------------------------------------------------------------------


def select_anchor_cluster(
    rep_xy: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    radius_m: float = 1200.0,
) -> np.ndarray:
    """Indices of the densest confidence-weighted cluster of anchors.

    OCR over a city recovers both route-relevant anchors and scattered
    noise (a shop sign, a fuzzy-matched far-off street). A genuine
    location is *corroborated* — several anchors land near each other —
    while noise is isolated. We pick the anchor whose neighborhood
    (within ``radius_m``) carries the most total confidence weight, then
    keep every anchor within that radius and drop the rest.

    On the Ulm 4K run this keeps the central cluster (Sedelhöfe,
    Handwerkskammer, Sedelhofgasse, Polizeipräsidium, …) and rejects the
    outliers (TÖPFER 3 km N, Arena 2.5 km SE, the false Donauhalde match
    9 km out) that otherwise hijacked enumeration. With 0–1 anchors it's
    a no-op (nothing to corroborate against).
    """
    n = len(rep_xy)
    if n <= 1:
        return np.arange(n)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    # Pairwise distances (small n; anchors are few).
    d = np.linalg.norm(rep_xy[:, None, :] - rep_xy[None, :, :], axis=2)
    within = d <= radius_m
    scores = (within * w[None, :]).sum(axis=1)
    center = int(np.argmax(scores))
    return np.where(within[center])[0]


def anchors_to_xy(anchors: list[PoiAnchor], crs: str) -> np.ndarray:
    """Project anchors to the graph CRS → Nx2 (x, y) in meters."""
    if not anchors:
        return np.zeros((0, 2))
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lons = [a.lon for a in anchors]
    lats = [a.lat for a in anchors]
    x, y = t.transform(lons, lats)
    return np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]).astype(float)


def score_candidates_by_anchors(
    candidate_polylines: list[np.ndarray],
    anchor_xy: np.ndarray,
) -> list[float]:
    """Min distance (m) from each candidate walk to the nearest anchor.

    Lower = better (the walk passes close to where we read a sign). With
    no anchors, every candidate gets ``inf`` (channel is a no-op).
    """
    if len(anchor_xy) == 0:
        return [float("inf")] * len(candidate_polylines)
    out: list[float] = []
    for poly in candidate_polylines:
        poly = np.asarray(poly, dtype=np.float64)
        if poly.ndim != 2 or len(poly) == 0:
            out.append(float("inf"))
            continue
        # Min over anchors of min over polyline vertices.
        d = np.min([
            np.linalg.norm(poly - a, axis=1).min() for a in anchor_xy
        ])
        out.append(float(d))
    return out


def anchor_seed_nodes(
    road: RoadGraph,
    anchor_xy: np.ndarray,
    *,
    radius_m: float = 300.0,
    max_nodes: int = 40,
) -> list:
    """Graph nodes within ``radius_m`` of any anchor — enumeration seeds.

    Adding these as walk roots guarantees the anchored area is
    represented in the candidate pool regardless of trajectory drift.
    Capped at ``max_nodes`` (closest first) so a dense anchor doesn't
    explode enumeration cost.
    """
    if len(anchor_xy) == 0:
        return []
    node_ids = list(road.graph.nodes)
    xy = np.array([[road.graph.nodes[n]["x"], road.graph.nodes[n]["y"]]
                   for n in node_ids], dtype=float)
    # Distance of each node to its nearest anchor.
    dmin = np.full(len(node_ids), np.inf)
    for a in anchor_xy:
        dmin = np.minimum(dmin, np.linalg.norm(xy - a, axis=1))
    within = np.where(dmin <= radius_m)[0]
    within = within[np.argsort(dmin[within])][:max_nodes]
    return [node_ids[i] for i in within]


def anchors_to_json(anchors: list[PoiAnchor]) -> list[dict]:
    return [asdict(a) for a in anchors]


# ---------------------------------------------------------------------------
# Street-name matching (true-4K: OCR street plates → OSM graph geometry)
# ---------------------------------------------------------------------------


def build_street_gazetteer(road: RoadGraph) -> dict[str, str]:
    """Map normalized street name → canonical name, from graph edges."""
    from .evaluator import _normalize_street_name

    gaz: dict[str, str] = {}
    for _u, _v, _k, d in road.graph.edges(keys=True, data=True):
        name = d.get("name")
        names = name if isinstance(name, list) else [name]
        for nm in names:
            if nm:
                gaz.setdefault(_normalize_street_name(str(nm)), str(nm))
    return gaz


def _nodes_for_street(road: RoadGraph, canonical_name: str) -> tuple:
    """Graph nodes lying on any edge whose name matches ``canonical_name``."""
    nodes: list = []
    seen = set()
    for u, v, k, d in road.graph.edges(keys=True, data=True):
        name = d.get("name")
        names = name if isinstance(name, list) else [name]
        if any(str(nm) == canonical_name for nm in names if nm):
            for n in (u, v):
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
    return tuple(nodes)


def match_text_to_streets(
    detections: list[SceneText],
    road: RoadGraph,
    *,
    min_confidence: float = 0.5,
    min_ratio: float = 0.85,
    min_letters: int = 5,
) -> list[StreetAnchor]:
    """Fuzzy-match OCR detections against the OSM street gazetteer.

    Returns one :class:`StreetAnchor` per distinct matched street. A match
    requires similarity ``>= min_ratio`` to a real graph street name — a
    strict bar that, combined with the distinctive German street suffixes
    (``-straße`` / ``-gasse`` / ``-weg`` …), keeps shop-name false
    positives out. Needs legible plates (true-4K); at 720p this returns
    nothing (confirmed empirically).
    """
    import difflib

    from .evaluator import _normalize_street_name

    gaz = build_street_gazetteer(road)
    keys = list(gaz)
    best: dict[str, StreetAnchor] = {}
    for d in detections:
        if d.confidence < min_confidence:
            continue
        norm = _normalize_street_name(_clean(d.text))
        if len(norm) < min_letters:
            continue
        hits = difflib.get_close_matches(norm, keys, n=1, cutoff=min_ratio)
        if not hits:
            continue
        ratio = difflib.SequenceMatcher(None, norm, hits[0]).ratio()
        canonical = gaz[hits[0]]
        prev = best.get(canonical)
        if prev is None or d.confidence > prev.confidence:
            best[canonical] = StreetAnchor(
                name=canonical, ocr_text=d.text, confidence=d.confidence,
                match_ratio=ratio, node_ids=_nodes_for_street(road, canonical),
                t_sec=float(d.t_sec),
            )
    return sorted(best.values(), key=lambda a: -a.confidence)


def street_anchor_xy(anchors: list[StreetAnchor], road: RoadGraph) -> np.ndarray:
    """All node coordinates across matched streets → Nx2 projected (x, y)."""
    pts: list[list[float]] = []
    for a in anchors:
        for n in a.node_ids:
            nd = road.graph.nodes[n]
            pts.append([nd["x"], nd["y"]])
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def street_anchor_seed_nodes(anchors: list[StreetAnchor]) -> list:
    """Graph nodes on matched streets — direct enumeration seeds."""
    out: list = []
    seen = set()
    for a in anchors:
        for n in a.node_ids:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def cluster_filter_anchors(
    poi_anchors: list[PoiAnchor],
    street_anchors: list[StreetAnchor],
    road: RoadGraph,
    *,
    radius_m: float = 1200.0,
) -> "tuple[list[PoiAnchor], list[StreetAnchor]]":
    """Drop spatial-outlier anchors, keeping the densest cluster.

    Combines POI points and street centroids into one set, runs
    :func:`select_anchor_cluster`, and returns the surviving POIs and
    streets. This is the noise gate that stops one bad anchor (a far-off
    shop sign or a fuzzy-matched wrong street) from hijacking the
    anchor-gated enumeration.
    """
    items: list[tuple[str, object, float, float, float]] = []  # kind,obj,x,y,w
    for a in poi_anchors:
        xy = anchors_to_xy([a], road.crs)
        if len(xy):
            items.append(("poi", a, xy[0, 0], xy[0, 1], a.confidence))
    for s in street_anchors:
        xy = street_anchor_xy([s], road)
        if len(xy):
            c = xy.mean(axis=0)
            items.append(("street", s, c[0], c[1], s.confidence))
    if len(items) <= 1:
        return poi_anchors, street_anchors
    rep = np.array([[it[2], it[3]] for it in items], dtype=float)
    w = np.array([it[4] for it in items], dtype=float)
    keep = set(int(i) for i in select_anchor_cluster(rep, w, radius_m=radius_m))
    kept_poi = [items[i][1] for i in keep if items[i][0] == "poi"]
    kept_street = [items[i][1] for i in keep if items[i][0] == "street"]
    return kept_poi, kept_street  # type: ignore[return-value]


def street_anchors_to_json(anchors: list[StreetAnchor]) -> list[dict]:
    return [
        {"name": a.name, "ocr_text": a.ocr_text, "confidence": a.confidence,
         "match_ratio": round(a.match_ratio, 3), "n_nodes": len(a.node_ids)}
        for a in anchors
    ]
