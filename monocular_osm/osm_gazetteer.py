"""Local OSM gazetteer for bulk anchor resolution (offline geocoding).

The remote Nominatim path (:func:`src.text_anchor.geocode_texts`) is
throttled to a handful of queries at ~1.1 s each, so most OCR
detections never get a geocode attempt, and it is fragile (one flaky
network run can poison a cache). This module builds a *local* gazetteer
of named OSM features covering the SAME area as the road graph and
fuzzy-matches OCR texts against it — no network at match time, so every
detection can be resolved, for free.

Two entry points:

* :func:`build_gazetteer` — one osmnx download per area (cached to
  JSON), yielding named POIs, transit stops, and building housenames.
* :func:`match_texts` — conservative fuzzy match of OCR texts against
  the gazetteer, returning :class:`~src.text_anchor.PoiAnchor` objects
  (drop-in with the existing anchor-scoring machinery).

Design choices that matter for accuracy:

* Matching is deliberately strict (score >= 0.87, length >= 4,
  token-set style) so a stray brand name doesn't invent an anchor.
* A name with more than two instances in the bbox is DROPPED unless the
  instances cluster within 300 m (then the centroid is used): a
  wrong-instance anchor (one of three 'Boots' pharmacies) is worse than
  no anchor at all.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .osm_data import RoadGraph
    from .text_anchor import PoiAnchor

# OSM tag families we treat as "named landmarks worth anchoring on".
# value True = "any value of this key, as long as the feature has a name".
_POI_TAGS: dict[str, object] = {
    "amenity": True,
    "shop": True,
    "tourism": True,
    "leisure": True,
    "office": True,
}
_TRANSIT_TAGS: dict[str, object] = {
    "highway": ["bus_stop"],
    "railway": ["station", "halt", "tram_stop"],
    "public_transport": ["platform", "station"],
}
_HOUSENAME_TAGS: dict[str, object] = {"addr:housename": True}


_NON_ALNUM = re.compile(r"[^0-9A-Za-zÀ-ſ]+")


def _ascii_fold(text: str) -> str:
    """Casefold + strip diacritics so 'Sedelhöfe' -> 'sedelhofe'."""
    if not text:
        return ""
    # German umlaut expansion first (ö -> oe) so it matches OCR that
    # rendered it either way; then NFKD strip for the rest.
    subs = {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "ae", "Ö": "oe",
            "Ü": "ue", "ß": "ss"}
    for k, v in subs.items():
        text = text.replace(k, v)
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_only.casefold()


def _norm_name(text: str) -> str:
    """Normalized comparison key: ascii-folded, non-alnum -> single space."""
    folded = _ascii_fold(text)
    return _NON_ALNUM.sub(" ", folded).strip()


def _have_rapidfuzz() -> bool:
    try:
        import rapidfuzz  # noqa: F401
        return True
    except Exception:
        return False


def _similarity(a: str, b: str) -> float:
    """Normalized [0,1] similarity between two already-normalized names.

    ``a`` is the OCR text, ``b`` a gazetteer name. token_set_ratio alone
    is too permissive here: a single common OCR word ('will') is a token
    SUBSET of a long name ('ingenieurbuero will') and scores 1.0, minting
    a wrong anchor. So we combine it with token_sort_ratio (order-tolerant
    but length-sensitive) and a token-coverage guard: the OCR text must
    cover a real fraction of the gazetteer name's tokens. Falls back to
    difflib when rapidfuzz is unavailable.
    """
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        tset = fuzz.token_set_ratio(a, b) / 100.0
    except Exception:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()
    # token_set_ratio is subset-tolerant: a single OCR word ('will') is a
    # token subset of a long name ('ingenieurbuero will') and scores 1.0,
    # minting a wrong anchor. Penalize by how little of the gazetteer name
    # the OCR text covers (char-length ratio — a coverage proxy robust to
    # OCR typos within the covered part). A distinctive OCR token that is a
    # real portion of the name ('Russell' -> 'Russell Square', cov 0.50)
    # survives; a short common fragment ('hell' -> 'heaven hell', cov 0.36;
    # 'will' -> 'ingenieurbuero will', cov 0.21) is cut below the 0.87 bar.
    coverage = min(1.0, len(a) / len(b))
    # Hard coverage floor: an OCR fragment shorter than ~45% of the name is
    # too little of it to be a confident anchor ('will'/'hell'/'bahnhof'
    # cases at cov 0.21/0.36/0.39). Above the floor, trust token_set — a
    # distinctive partial token ('Russell' -> 'Russell Square', cov 0.50)
    # is a legitimate match that the downstream cluster/VPR gate can still
    # veto if it is spatially isolated.
    if coverage < 0.45:
        return tset * (coverage / 0.45)  # scaled well below the 0.87 bar
    return tset


# ---------------------------------------------------------------------------
# Building the gazetteer
# ---------------------------------------------------------------------------


def _bbox_from_graph(road: "RoadGraph", margin_deg: float = 0.005
                     ) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) covering the graph, with margin."""
    import numpy as np
    from pyproj import Transformer

    xs = np.array([d["x"] for _, d in road.graph.nodes(data=True)], dtype=float)
    ys = np.array([d["y"] for _, d in road.graph.nodes(data=True)], dtype=float)
    t = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform([xs.min(), xs.max()], [ys.min(), ys.max()])
    return (min(lon) - margin_deg, min(lat) - margin_deg,
            max(lon) + margin_deg, max(lat) + margin_deg)


def _bbox_signature(bbox: tuple[float, float, float, float]) -> str:
    return "_".join(f"{v:.4f}" for v in bbox)


def _centroid_latlon(geom) -> "tuple[float, float] | None":
    """(lat, lon) of a shapely geometry's representative point, or None."""
    try:
        pt = geom.representative_point() if geom.geom_type != "Point" else geom
        return (float(pt.y), float(pt.x))
    except Exception:
        return None


def _kind_for_row(tags: dict) -> str:
    """A coarse category label for an OSM feature row."""
    if tags.get("public_transport") or tags.get("railway") or (
        tags.get("highway") == "bus_stop"
    ):
        return "transit"
    if tags.get("addr:housename"):
        return "housename"
    return "poi"


def build_gazetteer(
    graph_or_bbox,
    cache_path: "str | Path | None" = None,
    *,
    margin_deg: float = 0.005,
) -> dict:
    """Build (or load) a gazetteer of named OSM features for an area.

    ``graph_or_bbox`` is either a :class:`~src.osm_data.RoadGraph` (its
    lon/lat bbox is derived) or an explicit ``(min_lon, min_lat, max_lon,
    max_lat)`` tuple. Downloads POIs, transit stops, and building
    housenames via osmnx and returns::

        {"signature": {"bbox": [...]},
         "entries": [{"name", "norm", "lat", "lon", "kind"}, ...]}

    Cached to ``cache_path`` (JSON) keyed by the bbox signature. Any
    osmnx / network error yields an EMPTY gazetteer (never raises) — the
    pipeline must not crash because a features download failed.
    """
    # Resolve bbox.
    if isinstance(graph_or_bbox, (tuple, list)) and len(graph_or_bbox) == 4:
        bbox = tuple(float(v) for v in graph_or_bbox)  # type: ignore[assignment]
    else:
        try:
            bbox = _bbox_from_graph(graph_or_bbox, margin_deg=margin_deg)
        except Exception:
            return {"signature": {"bbox": None}, "entries": []}

    sig = _bbox_signature(bbox)  # type: ignore[arg-type]
    cp = Path(cache_path) if cache_path else None
    if cp and cp.exists():
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            if cached.get("signature", {}).get("sig") == sig:
                return cached
        except (OSError, ValueError):
            pass

    entries = _download_entries(bbox)  # type: ignore[arg-type]
    gaz = {
        "signature": {"sig": sig, "bbox": list(bbox)},
        "entries": entries,
    }
    if cp is not None:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(gaz, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return gaz


def _download_entries(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Query osmnx features_from_bbox for each tag family; flatten to entries.

    Returns [] on any error (offline, timeout, insufficient response).
    """
    try:
        import osmnx as ox
    except Exception:
        return []

    all_tags: dict[str, object] = {}
    for group in (_POI_TAGS, _TRANSIT_TAGS, _HOUSENAME_TAGS):
        for k, v in group.items():
            if k in all_tags and isinstance(all_tags[k], list) and isinstance(v, list):
                all_tags[k] = list(set(all_tags[k] + v))  # type: ignore[operator]
            else:
                all_tags[k] = v

    try:
        gdf = ox.features.features_from_bbox(tuple(bbox), all_tags)
    except Exception:
        return []

    entries: list[dict] = []
    if gdf is None or len(gdf) == 0:
        return entries
    cols = set(gdf.columns)
    for _idx, row in gdf.iterrows():
        # Prefer 'name'; fall back to housename for pure address features.
        name = None
        if "name" in cols:
            name = row.get("name")
        if (name is None or (isinstance(name, float) and math.isnan(name))) \
                and "addr:housename" in cols:
            name = row.get("addr:housename")
        if name is None or (isinstance(name, float) and math.isnan(name)):
            continue
        name = str(name).strip()
        if not name:
            continue
        geom = row.get("geometry")
        if geom is None:
            continue
        ll = _centroid_latlon(geom)
        if ll is None:
            continue
        lat, lon = ll
        tags = {c: row.get(c) for c in cols if c != "geometry"}
        entries.append({
            "name": name,
            "norm": _norm_name(name),
            "lat": lat,
            "lon": lon,
            "kind": _kind_for_row(tags),
        })
    return entries


# ---------------------------------------------------------------------------
# Matching OCR texts against the gazetteer
# ---------------------------------------------------------------------------


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _resolve_instances(
    instances: list[dict], *, max_instances: int, cluster_radius_m: float
) -> "tuple[float, float] | None":
    """Pick a single (lat, lon) for a name with possibly many instances.

    - 1 instance: use it.
    - <= max_instances that all fall within cluster_radius_m of their
      centroid: use the centroid.
    - otherwise: return None (ambiguous → no anchor).
    """
    if not instances:
        return None
    if len(instances) == 1:
        return (instances[0]["lat"], instances[0]["lon"])
    lat_c = sum(i["lat"] for i in instances) / len(instances)
    lon_c = sum(i["lon"] for i in instances) / len(instances)
    spread = max(_haversine_m(lat_c, lon_c, i["lat"], i["lon"])
                 for i in instances)
    if spread <= cluster_radius_m:
        return (lat_c, lon_c)  # co-located duplicates → safe centroid
    if len(instances) > max_instances:
        return None  # scattered multi-instance → ambiguous, drop
    # 2 scattered instances: still ambiguous which one — drop.
    return None


def match_texts(
    texts_with_times,
    gazetteer: dict,
    *,
    min_score: float = 0.87,
    min_letters: int = 4,
    min_confidence: float = 0.0,
    max_instances: int = 2,
    cluster_radius_m: float = 300.0,
):
    """Fuzzy-match OCR texts against a gazetteer → list of PoiAnchor.

    ``texts_with_times`` is any iterable of items exposing ``.text``,
    ``.confidence`` and ``.t_sec`` (e.g. :class:`~src.scene_text.SceneText`),
    or ``(text, confidence, t_sec)`` tuples.

    A detection matches a gazetteer name when the normalized token-set
    similarity is ``>= min_score`` and the normalized text has at least
    ``min_letters`` letters. Multi-instance names are resolved by
    :func:`_resolve_instances` (co-located duplicates collapse to a
    centroid; scattered ones are dropped). One anchor per matched name,
    keyed by the best (highest-confidence, then highest-score) detection.
    """
    from .text_anchor import PoiAnchor  # local import: avoid cycle

    entries = gazetteer.get("entries", []) if gazetteer else []
    if not entries:
        return []

    # Index gazetteer entries by normalized name; group instances.
    by_norm: dict[str, list[dict]] = {}
    for e in entries:
        norm = e.get("norm") or _norm_name(e.get("name", ""))
        if len(re.sub(r"[^a-z]", "", norm)) < min_letters:
            continue
        by_norm.setdefault(norm, []).append(e)
    gaz_norms = list(by_norm.keys())
    if not gaz_norms:
        return []

    # Normalize detections; keep best detection per normalized text.
    best_det: dict[str, tuple[str, float, float]] = {}
    for item in texts_with_times:
        if hasattr(item, "text"):
            text, conf, t_sec = item.text, float(item.confidence), float(item.t_sec)
        else:
            text, conf, t_sec = item[0], float(item[1]), float(item[2])
        if conf < min_confidence:
            continue
        norm = _norm_name(text)
        if len(re.sub(r"[^a-z]", "", norm)) < min_letters:
            continue
        prev = best_det.get(norm)
        if prev is None or conf > prev[1]:
            best_det[norm] = (text, conf, t_sec)

    anchors: list["PoiAnchor"] = []
    used_names: set[str] = set()
    # Iterate detections best-confidence first for stable dedupe.
    for norm, (text, conf, t_sec) in sorted(
        best_det.items(), key=lambda kv: -kv[1][1]
    ):
        # Find best-scoring gazetteer name.
        best_gn, best_score = None, 0.0
        for gn in gaz_norms:
            s = _similarity(norm, gn)
            if s > best_score:
                best_gn, best_score = gn, s
        if best_gn is None or best_score < min_score:
            continue
        canonical = by_norm[best_gn][0]["name"]
        if canonical.casefold() in used_names:
            continue
        latlon = _resolve_instances(
            by_norm[best_gn], max_instances=max_instances,
            cluster_radius_m=cluster_radius_m,
        )
        if latlon is None:
            continue
        lat, lon = latlon
        used_names.add(canonical.casefold())
        anchors.append(PoiAnchor(
            name=canonical, lat=lat, lon=lon,
            confidence=conf, t_sec=t_sec,
        ))
    return anchors
