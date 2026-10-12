"""The road-graph and sun-heading stages on a hand-built context."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

import monocular_osm.stages.anchors as anchors
import monocular_osm.stages.graph as stage_graph
import monocular_osm.sun_heading as sun_heading
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext

CRS = "EPSG:32632"


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    return PipelineConfig(url="u", city="Ulm, Germany", data_dir=tmp_path / "data",
                          output_dir=tmp_path / "out", **kw)


def _road(n_nodes: int = 3):
    g = nx.MultiDiGraph()
    g.add_nodes_from(range(n_nodes))
    g.add_edge(0, 1)
    return SimpleNamespace(graph=g, crs=CRS)


# ---------------------------------------------------------------------------
# load_road_graph
# ---------------------------------------------------------------------------


def test_graph_for_the_whole_named_city(tmp_path, monkeypatch):
    seen = {}

    def fake_fetch(city, cache_path, around=None):
        seen.update(city=city, cache_path=cache_path, around=around)
        return _road()

    monkeypatch.setattr(stage_graph, "fetch_city_graph", fake_fetch)
    ctx = RunContext.create(_cfg(tmp_path))
    stage_graph.load_road_graph(ctx)
    assert ctx.road.crs == CRS
    assert ctx.cache_path == tmp_path / "data" / "Ulm_Germany.graphml"
    assert seen == dict(city="Ulm, Germany", cache_path=ctx.cache_path, around=None)


def test_graph_around_the_seed_disc_gets_its_own_cache_file(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(stage_graph, "fetch_city_graph",
                        lambda city, cache_path, around=None: seen.update(
                            around=around, cache_path=cache_path) or _road())
    ctx = RunContext.create(_cfg(tmp_path))
    ctx.osm_around = (48.40123, 9.99456, 1234.6)
    stage_graph.load_road_graph(ctx)
    assert seen["around"] == (48.40123, 9.99456, 1234.6)
    # the disc is part of the cache key, so a re-centred disc never reads a stale graph
    assert ctx.cache_path.name == "Ulm_Germany_around_48.4012_9.9946_1234.graphml"


# ---------------------------------------------------------------------------
# estimate_sun_heading
# ---------------------------------------------------------------------------


def test_sun_heading_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(sun_heading, "estimate_heading",
                        lambda *a, **k: pytest.fail("not enabled"))
    ctx = RunContext.create(_cfg(tmp_path))
    anchors.estimate_sun_heading(ctx)
    assert ctx.result_sun_heading is None


def test_sun_heading_uses_the_seed_disc_centre_and_the_clip_window(tmp_path, monkeypatch):
    seen = {}

    def fake(video, centre, times, ocr_engine=None):
        seen.update(video=video, centre=centre, times=list(times), engine=ocr_engine)
        return {"available": True, "median_heading": 123.0, "source": "exif",
                "n_used": 5, "confidence": 0.9}

    monkeypatch.setattr(sun_heading, "estimate_heading", fake)
    ctx = RunContext.create(_cfg(tmp_path, use_sun_heading=True, vo_start_sec=10.0,
                                 vo_end_sec=50.0))
    ctx.video_path = tmp_path / "v.mp4"
    ctx.osm_around = (48.4, 9.99, 900.0)
    anchors.estimate_sun_heading(ctx)
    assert ctx.result_sun_heading["median_heading"] == 123.0
    assert seen["centre"] == (48.4, 9.99) and seen["engine"] == "rapidocr"
    assert len(seen["times"]) == 30 and seen["times"][0] == 10.0 and seen["times"][-1] == 50.0


def test_sun_heading_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no clock")

    monkeypatch.setattr(sun_heading, "estimate_heading", boom)
    ctx = RunContext.create(_cfg(tmp_path, use_sun_heading=True))
    ctx.video_path = tmp_path / "v.mp4"
    ctx.osm_around = (48.4, 9.99, 900.0)
    anchors.estimate_sun_heading(ctx)
    assert ctx.result_sun_heading == {"available": False, "reason": "no clock"}
    assert ctx.channel_status["sun"]["status"] == "failed"
