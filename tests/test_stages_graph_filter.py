"""The opt-in road-graph particle filter stage (--graph-filter).

Synthetic road graph and a VO-like trajectory: no network, no GPU. What is
checked is the stage's contract (off by default, never touches the headline,
honest labelling, failure recording), not the filter's accuracy — that is
tests/test_graph_filter.py and the KITTI evaluation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import baselines
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext
from monocular_osm.stages.graph_filter import graph_filter_channel
from monocular_osm.visual_odometry import Trajectory, _fit_plane_projection

from tests.test_graph_filter import CRS, ROUTE, grid_graph, jittered_grid  # noqa: F401


def _cfg(tmp_path, **kw) -> PipelineConfig:
    return PipelineConfig(url="u", city="Testville", data_dir=tmp_path / "d",
                          output_dir=tmp_path / "o", **kw)


def _vo_like_trajectory(n: int = 90, turn_every: int = 30) -> Trajectory:
    """Unit steps with a left turn every ``turn_every`` poses (OpenCV camera
    convention: x right, y down, z forward)."""
    yaw = np.cumsum(np.where(np.arange(n) % turn_every == turn_every - 1,
                             np.deg2rad(90.0), 0.0))
    centers = np.zeros((n, 3))
    rot = np.zeros((n, 3, 3))
    for i in range(n):
        fwd = np.array([np.sin(-yaw[i]), 0.0, np.cos(-yaw[i])])
        right = np.array([np.cos(-yaw[i]), 0.0, -np.sin(-yaw[i])])
        rot[i] = np.vstack([right, np.array([0.0, 1.0, 0.0]), fwd])
        if i:
            centers[i] = centers[i - 1] + fwd
    xz = _fit_plane_projection(centers, up=np.array([0.0, -1.0, 0.0]))
    return Trajectory(centers=centers, xz=xz, valid=np.ones(n, bool), n_inliers=[0] * n,
                      rotations=rot, translations=np.zeros((n, 3)))


def _ctx(tmp_path, *, graph_filter: bool, with_road: bool = True, **kw) -> RunContext:
    from monocular_osm.osm_data import _build_polyline_view

    cfg = _cfg(tmp_path, graph_filter=graph_filter, graph_filter_particles=3000, **kw)
    ctx = RunContext(cfg=cfg, fusion_w=None)
    traj = _vo_like_trajectory()
    ctx.traj = traj
    ctx.frames = SimpleNamespace(timestamps=list(np.arange(len(traj.xz)) * 0.5))
    ctx.result = {"position": {"source": "matcher", "latitude": 1.0, "longitude": 2.0}}
    if with_road:
        ctx.road = _build_polyline_view(jittered_grid())
    return ctx


def test_filter_is_off_by_default_and_leaves_no_trace(tmp_path) -> None:
    assert _cfg(tmp_path).graph_filter is False
    ctx = _ctx(tmp_path, graph_filter=False)
    graph_filter_channel(ctx)
    assert "graph_filter" not in ctx.result and "filter_position" not in ctx.result
    assert "graph_filter" not in ctx.channel_status


def test_stage_records_a_labelled_result_and_never_touches_the_headline(tmp_path) -> None:
    ctx = _ctx(tmp_path, graph_filter=True)
    headline_before = dict(ctx.result["position"])
    graph_filter_channel(ctx)
    gf, fp = ctx.result["graph_filter"], ctx.result["filter_position"]
    assert ctx.result["position"] == headline_before            # never the headline
    assert gf["metric_odometry"] is False and gf["confidence_calibrated"] is False
    assert gf["n_particles"] == 3000 and gf["uses_vpr_observations"] is False
    # An ambiguous map may leave no mode holding the minimum mass: 0 modes is a
    # legitimate (and honest) answer, not an error.
    assert 0.0 <= gf["confidence"] <= 1.0 and gf["n_modes"] >= 0
    assert fp["source"] == "graph_filter" and len(fp["route_latlon"]) >= 2
    assert all(-90 <= la <= 90 and -180 <= lo <= 180 for la, lo in fp["route_latlon"])
    assert ctx.channel_status["graph_filter"]["status"] == "ok"


def test_filter_route_is_scorable_as_its_own_method(tmp_path) -> None:
    ctx = _ctx(tmp_path, graph_filter=True)
    graph_filter_channel(ctx)
    pred = baselines.graph_filter_only(ctx.result)
    assert pred is not None and pred.name == "graph_filter_only" and pred.kind == "route"
    assert baselines.graph_filter_only({"position": {}}) is None


def test_missing_inputs_are_inactive_not_a_crash(tmp_path) -> None:
    ctx = _ctx(tmp_path, graph_filter=True, with_road=False)
    graph_filter_channel(ctx)
    assert ctx.channel_status["graph_filter"]["status"] == "inactive"
    assert "filter_position" not in ctx.result


def test_a_failing_filter_is_recorded_and_the_run_continues(tmp_path, monkeypatch) -> None:
    import monocular_osm.graph_filter as gf_mod

    def boom(*a, **k):
        raise RuntimeError("synthetic filter failure")

    monkeypatch.setattr(gf_mod, "run_filter", boom)
    ctx = _ctx(tmp_path, graph_filter=True)
    graph_filter_channel(ctx)                                  # must not raise
    st = ctx.channel_status["graph_filter"]
    assert st["status"] == "failed" and "synthetic filter failure" in st["error"]
    assert "filter_position" not in ctx.result


def test_vpr_observations_are_used_only_when_asked_and_available(tmp_path, monkeypatch) -> None:
    import monocular_osm.graph_filter as gf_mod

    seen = {}
    real = gf_mod.run_filter

    def spy(net, odom, cfg=None, **kw):
        seen["obs"] = kw.get("position_obs")
        return real(net, odom, cfg, **kw)

    monkeypatch.setattr(gf_mod, "run_filter", spy)
    ctx = _ctx(tmp_path, graph_filter=True, graph_filter_vpr=True)
    # three retrieval fixes on the road network (any points inside the map)
    xy = np.array([[ctx.road.polylines[0][0][0], ctx.road.polylines[0][0][1]]] * 3)
    from monocular_osm.position import xy_to_latlon

    latlon = xy_to_latlon(xy, ctx.road.crs)
    ctx.vpr_track = (np.array([5, 30, 60]), latlon, np.array([0.5, 0.6, 0.7]))
    ctx.anchor_origin = "vpr"
    graph_filter_channel(ctx)
    assert seen["obs"] is not None and len(seen["obs"]) == 3
    assert ctx.result["graph_filter"]["uses_vpr_observations"] is True

    seen.clear()
    ctx2 = _ctx(tmp_path, graph_filter=True, graph_filter_vpr=False)
    ctx2.vpr_track = ctx.vpr_track
    ctx2.anchor_origin = "vpr"
    graph_filter_channel(ctx2)
    assert list(seen["obs"]) == []                              # not asked: not used
