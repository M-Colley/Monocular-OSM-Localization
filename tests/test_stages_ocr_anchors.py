"""The OCR scene-text anchor stage: geocoded anchors, scale recovery, enumeration seeds."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import monocular_osm.scene_text as scene_text
import monocular_osm.stages.anchors as anchors
import monocular_osm.stages.graph as stage_graph
import monocular_osm.sun_heading as sun_heading
import monocular_osm.text_anchor as text_anchor
from monocular_osm.frame_extraction import ExtractedFrames
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.position import xy_to_latlon
from monocular_osm.stages.context import RunContext
from monocular_osm.visual_odometry import Trajectory

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
# ocr_scene_text_anchors
# ---------------------------------------------------------------------------


def _anchor(name, xy, t_sec, conf=0.9):
    lat, lon = xy_to_latlon(np.asarray([xy], dtype=np.float64), CRS)[0]
    return SimpleNamespace(name=name, lat=float(lat), lon=float(lon), confidence=conf,
                           t_sec=float(t_sec))


def _ocr_ctx(tmp_path, monkeypatch, poi, **kw) -> RunContext:
    """A context whose OCR / geocoding backends are faked; the scale-recovery
    maths is the real one."""
    monkeypatch.setattr(scene_text, "extract_scene_text", lambda *a, **k: ["det"])
    monkeypatch.setattr(text_anchor, "match_text_to_streets", lambda *a, **k: [])
    monkeypatch.setattr(text_anchor, "geocode_texts", lambda *a, **k: list(poi))
    monkeypatch.setattr(text_anchor, "default_geocode_fn", lambda path: None)
    monkeypatch.setattr(text_anchor, "cluster_filter_anchors", lambda o, s, road: (o, s))
    monkeypatch.setattr(text_anchor, "anchor_seed_nodes", lambda road, xy: [1, 2])
    kw.setdefault("enable_ocr_anchor", True)
    kw.setdefault("use_osm_gazetteer", False)
    ctx = RunContext.create(_cfg(tmp_path, **kw))
    ctx.video_path = tmp_path / "v.mp4"
    ctx.road = _road()
    ctx.duration_prior_m = 900.0
    ctx.estimated_length_m = 900.0
    n = 21                                      # 100 VO units over 20 s
    xz = np.stack([np.linspace(0, 100, n), np.zeros(n)], axis=1)
    ctx.traj = Trajectory(centers=np.c_[xz[:, 0], 0 * xz[:, 0], xz[:, 1]], xz=xz,
                          valid=np.ones(n, bool), n_inliers=[9] * n,
                          rotations=np.tile(np.eye(3), (n, 1, 1)),
                          translations=np.zeros((n, 3)))
    ctx.frames = ExtractedFrames(frames=[], timestamps=[float(i) for i in range(n)],
                                 fps=30.0, total_frames=n)
    return ctx


def _world(vo_x, scale=10.0, theta=np.radians(30.0), origin=(570000.0, 5360000.0)):
    c, s = np.cos(theta), np.sin(theta)
    return (origin[0] + scale * (c * vo_x), origin[1] + scale * (s * vo_x))


def test_ocr_anchors_fit_the_scale_and_seed_enumeration(tmp_path, monkeypatch):
    poi = [_anchor(f"shop{i}", _world(x), t)
           for i, (x, t) in enumerate([(5, 1), (30, 6), (60, 12), (95, 19)])]
    ctx = _ocr_ctx(tmp_path, monkeypatch, poi)
    anchors.ocr_scene_text_anchors(ctx)
    assert len(ctx.ocr_anchors) == 4
    sr = ctx.result_scale_recovery
    assert sr["status"] == "ok" and sr["n_inliers"] == 4
    assert sr["scale_m_per_vo_unit"] == pytest.approx(10.0, rel=0.01)
    assert ctx.estimated_length_m == pytest.approx(1000.0, rel=0.02)   # 100 units x 10 m
    assert ctx.anchor_world_route is not None and ctx.anchor_transform is not None
    assert ctx.anchor_seed == [1, 2] and ctx.anchor_xy.shape == (4, 2)


def test_a_scale_that_disagrees_with_the_duration_prior_is_rejected(tmp_path, monkeypatch):
    poi = [_anchor(f"shop{i}", _world(x, scale=40.0), t)
           for i, (x, t) in enumerate([(5, 1), (30, 6), (60, 12), (95, 19)])]
    ctx = _ocr_ctx(tmp_path, monkeypatch, poi)
    anchors.ocr_scene_text_anchors(ctx)
    assert ctx.result_scale_recovery["status"] == "rejected_sanity"
    assert ctx.estimated_length_m == 900.0 and ctx.anchor_world_route is None


def test_too_few_anchors_decline_scale_recovery(tmp_path, monkeypatch):
    poi = [_anchor(f"shop{i}", _world(x), t) for i, (x, t) in enumerate([(5, 1), (95, 19)])]
    ctx = _ocr_ctx(tmp_path, monkeypatch, poi)
    anchors.ocr_scene_text_anchors(ctx)
    assert ctx.result_scale_recovery is None                  # needs >= 3 anchors
    assert len(ctx.ocr_anchors) == 2 and ctx.estimated_length_m == 900.0


def test_poi_anchors_far_from_a_vpr_prior_are_dropped_as_direction_signs(tmp_path, monkeypatch):
    near = _anchor("Rathaus", (570000.0, 5360000.0), 3)
    far = _anchor("Holborn", (570000.0, 5363000.0), 9)         # 3 km away
    ctx = _ocr_ctx(tmp_path, monkeypatch, [near, far], enable_scale_recovery=False)
    ctx.vpr_center = (near.lat, near.lon)
    ctx.anchor_origin = "vpr"
    anchors.ocr_scene_text_anchors(ctx)
    assert [a.name for a in ctx.ocr_anchors] == ["Rathaus"]


def test_a_vlm_prior_is_too_coarse_to_veto_anchors(tmp_path, monkeypatch):
    near = _anchor("Rathaus", (570000.0, 5360000.0), 3)
    far = _anchor("Holborn", (570000.0, 5363000.0), 9)
    ctx = _ocr_ctx(tmp_path, monkeypatch, [near, far], enable_scale_recovery=False)
    ctx.vpr_center = (near.lat, near.lon)
    ctx.anchor_origin = "vlm"                  # a district centroid can sit km from a correct anchor
    anchors.ocr_scene_text_anchors(ctx)
    assert len(ctx.ocr_anchors) == 2


def test_an_ocr_failure_is_recorded_and_leaves_no_anchors(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ocr model missing")

    monkeypatch.setattr(scene_text, "extract_scene_text", boom)
    ctx = RunContext.create(_cfg(tmp_path, enable_ocr_anchor=True))
    ctx.video_path = tmp_path / "v.mp4"
    anchors.ocr_scene_text_anchors(ctx)
    assert ctx.ocr_anchors == [] and ctx.anchor_seed is None
    assert ctx.ocr_anchor_error == "ocr model missing"
    assert ctx.channel_status["ocr_anchor"]["status"] == "failed"


def test_ocr_anchors_are_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_text, "extract_scene_text",
                        lambda *a, **k: pytest.fail("not enabled"))
    ctx = RunContext.create(_cfg(tmp_path))
    anchors.ocr_scene_text_anchors(ctx)
    assert ctx.ocr_anchors == [] and ctx.anchor_xy.shape == (0, 2)
