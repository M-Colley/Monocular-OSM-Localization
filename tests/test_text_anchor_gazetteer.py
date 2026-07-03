"""Tests for text_anchor.gazetteer_anchors (additive local-gazetteer source)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from src import text_anchor
from src.osm_data import _build_polyline_view
from src.scene_text import SceneText
from src.text_anchor import PoiAnchor, gazetteer_anchors

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


def _patch_build(monkeypatch, entries):
    from src import osm_gazetteer

    def _fake_build(graph_or_bbox, cache_path=None, **kw):
        return {"signature": {}, "entries": entries}

    monkeypatch.setattr(osm_gazetteer, "build_gazetteer", _fake_build)


def _entry(name, lat, lon):
    from src.osm_gazetteer import _norm_name
    return {"name": name, "norm": _norm_name(name), "lat": lat, "lon": lon,
            "kind": "poi"}


def test_gazetteer_anchors_returns_poi_anchors(monkeypatch):
    _patch_build(monkeypatch, [_entry("Sedelhöfe", 48.4000, 9.9930)])
    dets = [SceneText("Sedelhoefe", 0.95, 12.0)]
    out = gazetteer_anchors(dets, _ulm_graph())
    assert len(out) == 1
    assert isinstance(out[0], PoiAnchor)
    assert out[0].name == "Sedelhöfe"


def test_gazetteer_anchors_filters_outside_bbox(monkeypatch):
    # Entry far outside the graph bbox is dropped even if name matches.
    _patch_build(monkeypatch, [_entry("Sedelhöfe", 50.0, 8.0)])
    dets = [SceneText("Sedelhoefe", 0.95, 12.0)]
    assert gazetteer_anchors(dets, _ulm_graph()) == []


def test_gazetteer_anchors_dedupes_against_existing(monkeypatch):
    _patch_build(monkeypatch, [
        _entry("Sedelhöfe", 48.4000, 9.9930),
        _entry("Rathaus", 48.4002, 9.9932),
    ])
    dets = [SceneText("Sedelhoefe", 0.95, 12.0),
            SceneText("Rathaus", 0.9, 20.0)]
    existing = [PoiAnchor(name="Sedelhöfe", lat=48.4, lon=9.99,
                          confidence=0.9, t_sec=1.0)]
    out = gazetteer_anchors(dets, _ulm_graph(), existing=existing)
    names = {a.name for a in out}
    assert "Sedelhöfe" not in names   # already had it via Nominatim
    assert "Rathaus" in names


def test_gazetteer_anchors_swallows_errors(monkeypatch):
    from src import osm_gazetteer

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(osm_gazetteer, "build_gazetteer", _boom)
    out = gazetteer_anchors([SceneText("Foo", 0.9, 1.0)], _ulm_graph())
    assert out == []
