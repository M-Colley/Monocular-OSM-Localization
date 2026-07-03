"""Tests for the local OSM gazetteer (osmnx download mocked)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point

from src.osm_data import _build_polyline_view
from src.osm_gazetteer import (
    _norm_name,
    _resolve_instances,
    _similarity,
    build_gazetteer,
    match_texts,
)
from src.scene_text import SceneText

UTM32N = "EPSG:32632"


def _project(latlons, crs=UTM32N):
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = t.transform([lo for _, lo in latlons], [la for la, _ in latlons])
    return np.column_stack([xs, ys]).astype(float)


def _ulm_graph():
    pts = [(48.3984, 9.9916), (48.4010, 9.9916), (48.3984, 9.9970)]
    xy = _project(pts)
    g = nx.MultiDiGraph()
    g.graph["crs"] = UTM32N
    for i, (x, y) in enumerate(xy):
        g.add_node(i, x=float(x), y=float(y))
    g.add_edge(0, 1, length=290.0,
               geometry=LineString([tuple(xy[0]), tuple(xy[1])]), name="A")
    g.add_edge(1, 2, length=420.0,
               geometry=LineString([tuple(xy[1]), tuple(xy[2])]), name="B")
    return _build_polyline_view(g)


# ---------------------------------------------------------------------------
# normalization / similarity
# ---------------------------------------------------------------------------


def test_norm_folds_umlauts():
    assert _norm_name("Sedelhöfe") == "sedelhoefe"
    assert _norm_name("Sedelhoefe") == "sedelhoefe"
    assert _norm_name("Straße") == "strasse"


def test_similarity_umlaut_variants_high():
    # 'Sedelhoefe' (OCR) vs 'Sedelhöfe' (OSM) after normalization.
    assert _similarity("sedelhoefe", "sedelhoefe") == 1.0
    assert _similarity(_norm_name("Sedelhoefe"), _norm_name("Sedelhöfe")) >= 0.87


def test_similarity_multiword_name_full_coverage():
    # OCR text covering (nearly) the whole gazetteer name scores high.
    assert _similarity("polizeipraesidium", "polizeipraesidium ulm") >= 0.87


def test_similarity_bare_fragment_penalized():
    # A single OCR word that is only a small token SUBSET of a long name
    # must NOT mint an anchor (the 'WILL' -> 'Ingenieurbuero Will' bug).
    assert _similarity("will", "ingenieurbuero will") < 0.87
    assert _similarity("hell", "heaven hell") < 0.87


# ---------------------------------------------------------------------------
# _resolve_instances (multi-instance policy)
# ---------------------------------------------------------------------------


def _inst(lat, lon):
    return {"lat": lat, "lon": lon, "name": "X", "norm": "x", "kind": "poi"}


def test_resolve_single_instance():
    out = _resolve_instances([_inst(48.4, 9.99)], max_instances=2,
                             cluster_radius_m=300.0)
    assert out == (48.4, 9.99)


def test_resolve_colocated_duplicates_centroid():
    # Two entries ~30 m apart (same building split into node+way): centroid.
    out = _resolve_instances(
        [_inst(48.4000, 9.9900), _inst(48.40025, 9.9900)],
        max_instances=2, cluster_radius_m=300.0,
    )
    assert out is not None
    assert abs(out[0] - 48.400125) < 1e-4


def test_resolve_three_scattered_instances_dropped():
    # Three 'Boots' >300 m apart -> ambiguous -> None.
    out = _resolve_instances(
        [_inst(48.40, 9.99), _inst(48.41, 9.99), _inst(48.40, 10.00)],
        max_instances=2, cluster_radius_m=300.0,
    )
    assert out is None


def test_resolve_two_scattered_instances_dropped():
    out = _resolve_instances(
        [_inst(48.40, 9.99), _inst(48.41, 9.99)],
        max_instances=2, cluster_radius_m=300.0,
    )
    assert out is None


# ---------------------------------------------------------------------------
# build_gazetteer (mock osmnx.features_from_bbox)
# ---------------------------------------------------------------------------


class _FakeGDF:
    """Minimal GeoDataFrame stand-in: iterrows() + columns."""

    def __init__(self, rows):
        self._rows = rows
        cols = set()
        for r in rows:
            cols.update(r.keys())
        self.columns = list(cols)

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


def _fake_features(monkeypatch, rows):
    import osmnx as ox

    def _fake(bbox, tags):
        return _FakeGDF(rows)

    monkeypatch.setattr(ox.features, "features_from_bbox", _fake)


def test_build_gazetteer_flattens_named_features(monkeypatch, tmp_path):
    rows = [
        {"name": "Sedelhöfe", "shop": "mall",
         "geometry": Point(9.9916, 48.4000)},
        {"name": "Hauptbahnhof", "railway": "station",
         "geometry": Point(9.9820, 48.3990)},
        {"name": float("nan"), "amenity": "bench",
         "geometry": Point(9.99, 48.40)},          # no name -> skipped
        {"addr:housename": "Metzgerturm", "name": float("nan"),
         "geometry": Point(9.995, 48.398)},         # housename fallback
    ]
    _fake_features(monkeypatch, rows)
    gaz = build_gazetteer(_ulm_graph(), cache_path=tmp_path / "gaz.json")
    names = {e["name"] for e in gaz["entries"]}
    assert "Sedelhöfe" in names
    assert "Hauptbahnhof" in names
    assert "Metzgerturm" in names
    # unnamed bench dropped
    assert len(gaz["entries"]) == 3
    # transit classification
    kinds = {e["name"]: e["kind"] for e in gaz["entries"]}
    assert kinds["Hauptbahnhof"] == "transit"


def test_build_gazetteer_caches_by_signature(monkeypatch, tmp_path):
    rows = [{"name": "Sedelhöfe", "shop": "mall",
             "geometry": Point(9.9916, 48.4000)}]
    _fake_features(monkeypatch, rows)
    cp = tmp_path / "gaz.json"
    gaz1 = build_gazetteer(_ulm_graph(), cache_path=cp)
    assert cp.exists()

    # Second call must NOT hit osmnx again (make it raise if called).
    import osmnx as ox

    def _boom(bbox, tags):
        raise AssertionError("should have used cache")

    monkeypatch.setattr(ox.features, "features_from_bbox", _boom)
    gaz2 = build_gazetteer(_ulm_graph(), cache_path=cp)
    assert gaz2["entries"] == gaz1["entries"]


def test_build_gazetteer_osmnx_error_returns_empty(monkeypatch, tmp_path):
    import osmnx as ox

    def _boom(bbox, tags):
        raise RuntimeError("network down")

    monkeypatch.setattr(ox.features, "features_from_bbox", _boom)
    gaz = build_gazetteer(_ulm_graph(), cache_path=tmp_path / "gaz.json")
    assert gaz["entries"] == []


def test_build_gazetteer_accepts_explicit_bbox(monkeypatch, tmp_path):
    rows = [{"name": "Foo", "amenity": "cafe",
             "geometry": Point(9.99, 48.40)}]
    _fake_features(monkeypatch, rows)
    bbox = (9.98, 48.39, 10.00, 48.41)  # lon/lat/lon/lat
    gaz = build_gazetteer(bbox, cache_path=tmp_path / "gaz.json")
    assert [e["name"] for e in gaz["entries"]] == ["Foo"]


# ---------------------------------------------------------------------------
# match_texts
# ---------------------------------------------------------------------------


def _gaz(entries):
    from src.osm_gazetteer import _norm_name as nn
    return {"signature": {}, "entries": [
        {"name": n, "norm": nn(n), "lat": la, "lon": lo, "kind": k}
        for (n, la, lo, k) in entries
    ]}


def test_match_umlaut_ocr_hits_osm_name():
    gaz = _gaz([("Sedelhöfe", 48.4000, 9.9916, "poi")])
    dets = [SceneText("Sedelhoefe", 0.95, 12.0)]
    out = match_texts(dets, gaz)
    assert len(out) == 1
    assert out[0].name == "Sedelhöfe"
    assert out[0].t_sec == 12.0
    assert out[0].confidence == 0.95


def test_match_rejects_low_similarity():
    gaz = _gaz([("Sedelhöfe", 48.4000, 9.9916, "poi")])
    dets = [SceneText("Bakery", 0.95, 1.0)]
    assert match_texts(dets, gaz) == []


def test_match_rejects_short_text():
    gaz = _gaz([("Rex", 48.40, 9.99, "poi")])
    dets = [SceneText("Rex", 0.99, 1.0)]  # < 4 letters
    assert match_texts(dets, gaz) == []


def test_match_drops_multi_instance_scattered():
    gaz = _gaz([
        ("Boots", 48.40, 9.99, "poi"),
        ("Boots", 48.41, 9.99, "poi"),
        ("Boots", 48.40, 10.00, "poi"),
    ])
    dets = [SceneText("Boots", 0.95, 5.0)]
    assert match_texts(dets, gaz) == []


def test_match_uses_centroid_for_colocated():
    gaz = _gaz([
        ("Rathaus", 48.40000, 9.99000, "poi"),
        ("Rathaus", 48.40025, 9.99000, "poi"),
    ])
    dets = [SceneText("Rathaus", 0.9, 3.0)]
    out = match_texts(dets, gaz)
    assert len(out) == 1
    assert abs(out[0].lat - 48.400125) < 1e-4


def test_match_empty_gazetteer():
    assert match_texts([SceneText("Foo", 0.9, 1.0)], {"entries": []}) == []


def test_match_dedupes_by_name():
    gaz = _gaz([("Sedelhöfe", 48.4000, 9.9916, "poi")])
    dets = [SceneText("Sedelhoefe", 0.95, 12.0),
            SceneText("Sedelhöfe", 0.80, 40.0)]
    out = match_texts(dets, gaz)
    assert len(out) == 1
    # keeps the higher-confidence detection's timestamp
    assert out[0].t_sec == 12.0
