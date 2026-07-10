"""Tests for text → geocoded anchor logic (geocoder injected, no network)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import LineString

from src.osm_data import _build_polyline_view
from src.scene_text import SceneText
from src.text_anchor import (
    PoiAnchor,
    anchor_seed_nodes,
    anchors_to_xy,
    geocode_texts,
    is_geocodable_text,
    score_candidates_by_anchors,
)

UTM32N = "EPSG:32632"
ULM = (48.3984, 9.9916)


def _project(latlons, crs=UTM32N):
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = t.transform([lo for _, lo in latlons], [la for la, _ in latlons])
    return np.column_stack([xs, ys]).astype(float)


def _ulm_graph() -> "RoadGraph":
    """A few nodes spanning ~400 m around central Ulm."""
    pts = [(48.3984, 9.9916), (48.4000, 9.9916), (48.3984, 9.9950)]
    xy = _project(pts)
    g = nx.MultiDiGraph()
    g.graph["crs"] = UTM32N
    for i, (x, y) in enumerate(xy):
        g.add_node(i, x=float(x), y=float(y))
    g.add_edge(0, 1, length=180.0,
               geometry=LineString([tuple(xy[0]), tuple(xy[1])]), name="A")
    g.add_edge(1, 2, length=300.0,
               geometry=LineString([tuple(xy[1]), tuple(xy[2])]), name="B")
    return _build_polyline_view(g)


# ---------------------------------------------------------------------------
# is_geocodable_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Sedelhöfe", True),
        ("Polizeipräsidium", True),
        ("535", False),         # pure number
        ("22-6h", False),       # sign fragment
        ("ab", False),          # too short
        ("29.13.9.2023", False),  # date
    ],
)
def test_is_geocodable_text(text: str, ok: bool) -> None:
    assert is_geocodable_text(text) is ok


# ---------------------------------------------------------------------------
# geocode_texts — bbox filtering is the key noise gate
# ---------------------------------------------------------------------------


def test_geocode_keeps_in_city_drops_out_of_city() -> None:
    road = _ulm_graph()
    detections = [
        SceneText("Sedelhöfe", 0.99, 150.0),     # in Ulm
        SceneText("Eiffel Tower", 0.95, 200.0),  # far away → must drop
        SceneText("535", 1.0, 10.0),             # not geocodable → never queried
    ]
    fake_db = {
        "Sedelhöfe, Ulm, Germany": (48.3999, 9.9860),
        "Eiffel Tower, Ulm, Germany": (48.8584, 2.2945),  # Paris
    }
    queried: list[str] = []

    def geocode_fn(q: str):
        queried.append(q)
        return fake_db.get(q)

    anchors = geocode_texts(detections, "Ulm, Germany", road,
                            geocode_fn=geocode_fn, min_confidence=0.5)
    assert [a.name for a in anchors] == ["Sedelhöfe"]
    # The number was filtered before any geocode call.
    assert "535, Ulm, Germany" not in queried


def test_geocode_respects_min_confidence() -> None:
    road = _ulm_graph()
    detections = [SceneText("Sedelhöfe", 0.40, 150.0)]
    called = []
    anchors = geocode_texts(
        detections, "Ulm, Germany", road,
        geocode_fn=lambda q: called.append(q) or (48.3999, 9.9860),
        min_confidence=0.5,
    )
    assert anchors == []
    assert called == []  # below confidence → not even queried


def test_geocode_dedups_repeated_text() -> None:
    road = _ulm_graph()
    detections = [
        SceneText("Sedelhöfe", 0.99, 150.0),
        SceneText("sedelhöfe", 0.95, 156.0),  # same place, lower conf
    ]
    anchors = geocode_texts(
        detections, "Ulm, Germany", road,
        geocode_fn=lambda q: (48.3999, 9.9860), min_confidence=0.5,
    )
    assert len(anchors) == 1
    assert anchors[0].confidence == 0.99  # kept the most confident


def test_geocode_time_buckets_surface_early_anchor() -> None:
    # Three prominent LATE signs + one lower-confidence EARLY sign. With a
    # tight budget, global-confidence selection crowds the early sign out
    # (the Ulm start weakness); temporal stratification reserves the early
    # bucket a query so the start of the route gets an anchor.
    road = _ulm_graph()
    detections = [
        SceneText("Muensterplatz", 0.99, 300.0),
        SceneText("Hauptbahnhof", 0.98, 310.0),
        SceneText("Rathausplatz", 0.97, 320.0),
        SceneText("Sedelhoefe", 0.80, 20.0),     # early, lower confidence
    ]
    db = {
        "Muensterplatz, Ulm, Germany": (48.398, 9.991),
        "Hauptbahnhof, Ulm, Germany": (48.399, 9.982),
        "Rathausplatz, Ulm, Germany": (48.398, 9.992),
        "Sedelhoefe, Ulm, Germany": (48.400, 9.986),
    }
    gc = lambda q: db.get(q)  # noqa: E731

    glob = geocode_texts(detections, "Ulm, Germany", road, geocode_fn=gc,
                         min_confidence=0.5, max_queries=2, time_buckets=0)
    assert "Sedelhoefe" not in [a.name for a in glob]   # crowded out

    strat = geocode_texts(detections, "Ulm, Germany", road, geocode_fn=gc,
                          min_confidence=0.5, max_queries=2, time_buckets=2)
    assert "Sedelhoefe" in [a.name for a in strat]      # early bucket reserved


# ---------------------------------------------------------------------------
# default_geocode_fn — persistent cache must not memoize transient failures
# ---------------------------------------------------------------------------


def _geocode_fn_env(monkeypatch):
    """Silence the Nominatim courtesy delay for the cache tests."""
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_geocode_cache_skips_transient_failures(tmp_path, monkeypatch) -> None:
    """A network hiccup (timeout/429/DNS) must NOT be written to the persistent
    cache — the old code memoized it as 'not found' forever, silently killing
    the OCR-anchor channel on every later run."""
    ox = pytest.importorskip("osmnx")
    from src.text_anchor import default_geocode_fn

    _geocode_fn_env(monkeypatch)
    cp = tmp_path / "geocode_cache.json"
    calls = []

    def flaky(query):
        calls.append(query)
        raise ConnectionError("network down")     # transient, NOT not-found

    monkeypatch.setattr(ox, "geocode", flaky)
    fn = default_geocode_fn(cp)
    assert fn("Sedelhöfe, Ulm") is None
    assert calls == ["Sedelhöfe, Ulm"]

    # Network recovers: a fresh geocoder (new run) must retry and succeed.
    monkeypatch.setattr(ox, "geocode", lambda q: (48.4, 9.99))
    fn2 = default_geocode_fn(cp)
    assert fn2("Sedelhöfe, Ulm") == (48.4, 9.99)


def test_geocode_cache_memoizes_genuine_not_found(tmp_path, monkeypatch) -> None:
    ox = pytest.importorskip("osmnx")
    from osmnx._errors import InsufficientResponseError

    from src.text_anchor import default_geocode_fn

    _geocode_fn_env(monkeypatch)
    cp = tmp_path / "geocode_cache.json"

    def not_found(query):
        raise InsufficientResponseError("no results")

    monkeypatch.setattr(ox, "geocode", not_found)
    fn = default_geocode_fn(cp)
    assert fn("Nowhereplatz, Ulm") is None

    # New run: the miss is served from the cache, network never consulted.
    def boom(query):
        raise AssertionError("cached not-found must not re-query")

    monkeypatch.setattr(ox, "geocode", boom)
    fn2 = default_geocode_fn(cp)
    assert fn2("Nowhereplatz, Ulm") is None


def test_city_extent_radius_bbox_and_cache(tmp_path, monkeypatch) -> None:
    """city_extent_radius returns the OSM bbox half-diagonal (used to size the
    coarse VPR disc so a peripheral-district drive is inside it — the fix for
    Málaga's 5169 m deployable miss) and caches it; a transient lookup failure
    is NOT cached (so it retries next run)."""
    import json
    import math

    ox = pytest.importorskip("osmnx")
    from src.text_anchor import city_extent_radius

    class _Gdf:
        def __init__(self, bounds):
            self.total_bounds = bounds

    calls = []

    def ok(city):
        calls.append(city)
        return _Gdf((-4.5, 36.6, -4.3, 36.8))  # min_lon, min_lat, max_lon, max_lat

    monkeypatch.setattr(ox, "geocode_to_gdf", ok)
    cp = tmp_path / "geocode_cache.json"
    r = city_extent_radius("Málaga, Spain", cp)
    assert r is not None
    clat, clon, half = r
    assert abs(clat - 36.7) < 1e-6 and abs(clon + 4.4) < 1e-6
    exp = math.hypot(0.2 * 111320, 0.2 * 111320 * math.cos(math.radians(36.7))) / 2
    assert abs(half - exp) < 1.0
    # cached: a second call does not re-invoke the (rate-limited) lookup
    city_extent_radius("Málaga, Spain", cp)
    assert calls == ["Málaga, Spain"]

    # transient failure -> None and NOT written to the cache
    def boom(city):
        raise ConnectionError("network down")

    monkeypatch.setattr(ox, "geocode_to_gdf", boom)
    assert city_extent_radius("Nowhere City", cp) is None
    assert "__extent__Nowhere City" not in json.loads(cp.read_text(encoding="utf-8"))


def test_geocode_cache_memoizes_hits(tmp_path, monkeypatch) -> None:
    ox = pytest.importorskip("osmnx")
    from src.text_anchor import default_geocode_fn

    _geocode_fn_env(monkeypatch)
    cp = tmp_path / "geocode_cache.json"
    monkeypatch.setattr(ox, "geocode", lambda q: (48.4, 9.99))
    default_geocode_fn(cp)("Sedelhöfe, Ulm")

    monkeypatch.setattr(ox, "geocode",
                        lambda q: (_ for _ in ()).throw(AssertionError("cached")))
    assert default_geocode_fn(cp)("Sedelhöfe, Ulm") == (48.4, 9.99)


# ---------------------------------------------------------------------------
# geometry: scoring + seed nodes
# ---------------------------------------------------------------------------


def test_score_candidates_by_anchors_ranks_near_walk_first() -> None:
    road = _ulm_graph()
    anchor_xy = anchors_to_xy(
        [PoiAnchor("X", 48.3984, 9.9916, 0.9, 0.0)], road.crs
    )  # exactly node 0
    near = road.polylines[0]              # edge A starts at node 0
    far = np.array([[1e6, 1e6], [1e6 + 10, 1e6]])
    dists = score_candidates_by_anchors([far, near], anchor_xy)
    assert dists[1] < dists[0]
    assert dists[1] == pytest.approx(0.0, abs=1.0)


def test_score_candidates_no_anchors_is_all_inf() -> None:
    road = _ulm_graph()
    dists = score_candidates_by_anchors([road.polylines[0]], np.zeros((0, 2)))
    assert dists == [float("inf")]


def test_anchor_seed_nodes_picks_nearby_only() -> None:
    road = _ulm_graph()
    # Anchor right at node 0 → node 0 (and node 1, 180 m away) within 300 m;
    # node 2 is 300 m down edge B from node 1, ~ at the radius.
    anchor_xy = anchors_to_xy(
        [PoiAnchor("X", 48.3984, 9.9916, 0.9, 0.0)], road.crs
    )
    seeds = anchor_seed_nodes(road, anchor_xy, radius_m=200.0, max_nodes=10)
    assert 0 in seeds          # the anchored node
    assert 2 not in seeds      # ~480 m away, outside radius


def test_anchor_seed_nodes_empty_without_anchors() -> None:
    road = _ulm_graph()
    assert anchor_seed_nodes(road, np.zeros((0, 2))) == []


# ---------------------------------------------------------------------------
# Street-name matching (true-4K plates → OSM graph)
# ---------------------------------------------------------------------------


def _named_street_graph() -> "RoadGraph":
    """Graph with two real-ish street names for fuzzy-match tests."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = UTM32N
    pts = _project([(48.3984, 9.9916), (48.4000, 9.9916), (48.4010, 9.9930)])
    for i, (x, y) in enumerate(pts):
        g.add_node(i, x=float(x), y=float(y))
    g.add_edge(0, 1, length=180.0,
               geometry=LineString([tuple(pts[0]), tuple(pts[1])]), name="Neutorstraße")
    g.add_edge(1, 2, length=160.0,
               geometry=LineString([tuple(pts[1]), tuple(pts[2])]), name="Olgastraße")
    return _build_polyline_view(g)


def test_match_text_to_streets_exact_and_fuzzy() -> None:
    from src.text_anchor import match_text_to_streets

    road = _named_street_graph()
    dets = [
        SceneText("Neutorstraße", 0.95, 30.0),   # exact
        SceneText("Olgastrasse", 0.88, 60.0),    # OCR dropped the umlaut
        SceneText("Bahnhofplatz", 0.92, 90.0),   # not in graph → no match
    ]
    anchors = match_text_to_streets(dets, road, min_confidence=0.5, min_ratio=0.85)
    matched = {a.name for a in anchors}
    assert "Neutorstraße" in matched
    assert "Olgastraße" in matched      # fuzzy: 'Olgastrasse' -> 'Olgastraße'
    assert len(anchors) == 2            # Bahnhofplatz rejected
    for a in anchors:
        assert len(a.node_ids) >= 2     # carries the street's nodes


def test_match_text_to_streets_rejects_low_confidence_and_short() -> None:
    from src.text_anchor import match_text_to_streets

    road = _named_street_graph()
    dets = [
        SceneText("Neutorstraße", 0.30, 30.0),   # below min_confidence
        SceneText("Olg", 0.99, 60.0),            # too short
    ]
    assert match_text_to_streets(dets, road, min_confidence=0.5) == []


def test_street_anchor_xy_and_seed_nodes() -> None:
    from src.text_anchor import (
        match_text_to_streets,
        street_anchor_seed_nodes,
        street_anchor_xy,
    )

    road = _named_street_graph()
    anchors = match_text_to_streets(
        [SceneText("Neutorstraße", 0.95, 30.0)], road, min_confidence=0.5
    )
    xy = street_anchor_xy(anchors, road)
    seeds = street_anchor_seed_nodes(anchors)
    assert xy.shape[0] == len(seeds) >= 2
    assert set(seeds) <= set(road.graph.nodes)


def test_match_text_to_streets_splits_citywide_name_into_components() -> None:
    """A common street name recurring across town must yield one anchor per
    physical instance — the old single anchor put its centroid BETWEEN the
    instances (a phantom point on no street)."""
    from src.text_anchor import match_text_to_streets

    g = nx.MultiDiGraph()
    g.graph["crs"] = UTM32N
    # Two 'Hauptstraße' instances ~5 km apart, two nodes each.
    pts = _project([
        (48.3984, 9.9916), (48.4000, 9.9916),     # instance 1
        (48.4434, 9.9916), (48.4450, 9.9916),     # instance 2, ~5 km north
    ])
    for i, (x, y) in enumerate(pts):
        g.add_node(i, x=float(x), y=float(y))
    g.add_edge(0, 1, length=180.0,
               geometry=LineString([tuple(pts[0]), tuple(pts[1])]), name="Hauptstraße")
    g.add_edge(2, 3, length=180.0,
               geometry=LineString([tuple(pts[2]), tuple(pts[3])]), name="Hauptstraße")
    road = _build_polyline_view(g)

    anchors = match_text_to_streets(
        [SceneText("Hauptstraße", 0.9, 10.0)], road, min_confidence=0.5,
    )
    assert len(anchors) == 2                      # one anchor per instance
    assert {a.name for a in anchors} == {"Hauptstraße"}
    assert sorted(sorted(a.node_ids) for a in anchors) == [[0, 1], [2, 3]]


def test_match_text_to_streets_no_false_positive_from_shop_name() -> None:
    """A shop name dissimilar to any street must not fuzzy-match."""
    from src.text_anchor import match_text_to_streets

    road = _named_street_graph()
    dets = [SceneText("KAAN", 0.95, 30.0), SceneText("PIERCING", 0.9, 40.0)]
    assert match_text_to_streets(dets, road, min_confidence=0.5) == []


# ---------------------------------------------------------------------------
# Anchor cluster filtering (outlier rejection)
# ---------------------------------------------------------------------------


def test_select_anchor_cluster_keeps_dense_drops_outliers() -> None:
    from src.text_anchor import select_anchor_cluster

    # Four points clustered near origin, two far outliers.
    xy = np.array([
        [0, 0], [100, 0], [0, 100], [50, 50],   # central cluster
        [9000, 0], [0, 9000],                    # outliers
    ], dtype=float)
    keep = set(int(i) for i in select_anchor_cluster(xy, radius_m=1200.0))
    assert keep == {0, 1, 2, 3}


def test_select_anchor_cluster_weights_break_ties() -> None:
    from src.text_anchor import select_anchor_cluster

    # Two pairs equally sized; higher-confidence pair should win.
    xy = np.array([[0, 0], [100, 0], [9000, 0], [9100, 0]], dtype=float)
    w = np.array([0.5, 0.5, 0.99, 0.99])
    keep = set(int(i) for i in select_anchor_cluster(xy, w, radius_m=500.0))
    assert keep == {2, 3}


def test_select_anchor_cluster_noop_small() -> None:
    from src.text_anchor import select_anchor_cluster

    assert list(select_anchor_cluster(np.zeros((1, 2)))) == [0]
    assert list(select_anchor_cluster(np.zeros((0, 2)))) == []


def test_cluster_filter_anchors_drops_far_street(monkeypatch) -> None:
    """The Ulm 4K failure: a central POI cluster plus a far false street
    match. The street 9 km out must be dropped."""
    from src.text_anchor import PoiAnchor, StreetAnchor, cluster_filter_anchors

    road = _named_street_graph()  # nodes near 48.40, 9.99 (central)
    # Central POIs near the graph nodes.
    pois = [
        PoiAnchor("Sedelhöfe", 48.3999, 9.9860, 1.0, 0.0),
        PoiAnchor("Handwerkskammer", 48.4005, 9.9864, 1.0, 6.0),
    ]
    # One street anchor that IS central (Neutorstraße, in graph) and one
    # far bogus one we fake by pointing node_ids at... we only have central
    # nodes, so emulate "far" via a POI outlier instead:
    pois.append(PoiAnchor("Töpfer", 48.4600, 10.0500, 1.0, 9.0))  # ~7 km off
    streets = [
        StreetAnchor("Neutorstraße", "Neutorstraße", 0.95, 0.95,
                     _named_street_graph().graph and (0, 1)),
    ]
    kept_poi, kept_street = cluster_filter_anchors(pois, streets, road, radius_m=1500.0)
    names = {p.name for p in kept_poi}
    assert "Sedelhöfe" in names and "Handwerkskammer" in names
    assert "Töpfer" not in names                 # outlier dropped
    assert len(kept_street) == 1                  # central street kept
