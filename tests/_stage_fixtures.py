"""Shared builders for the stage tests: a tiny projected road graph near Ulm,
candidates along it, and a ready-to-use :class:`RunContext`."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

from monocular_osm.frame_extraction import ExtractedFrames
from monocular_osm.osm_data import RoadGraph, _build_polyline_view
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext
from monocular_osm.trajectory_matching import MatchCandidate

UTM32N = "EPSG:32632"
ULM = (48.3984, 9.9916)


def project(lat: float, lon: float) -> np.ndarray:
    t = Transformer.from_crs("EPSG:4326", UTM32N, always_xy=True)
    x, y = t.transform(lon, lat)
    return np.array([x, y], dtype=np.float64)


def make_road(n_streets: int = 3, length_m: float = 400.0, gap_m: float = 5000.0) -> RoadGraph:
    """``n_streets`` parallel east-west streets, ``gap_m`` apart (north)."""
    g = nx.MultiDiGraph()
    g.graph["crs"] = UTM32N
    o = project(*ULM)
    for i in range(n_streets):
        a, b = o + np.array([0.0, i * gap_m]), o + np.array([length_m, i * gap_m])
        g.add_node(f"A{i}", x=float(a[0]), y=float(a[1]))
        g.add_node(f"B{i}", x=float(b[0]), y=float(b[1]))
        g.add_edge(f"A{i}", f"B{i}", length=length_m, name=f"Street {i}",
                   geometry=LineString([tuple(a), tuple(b)]))
    return _build_polyline_view(g)


def make_candidates(road: RoadGraph, scores=None) -> list[MatchCandidate]:
    """One candidate per street, ranked by ``scores`` (lower = better)."""
    out = []
    scores = scores or [10.0 + 5.0 * i for i in range(len(road.polylines))]
    for i, poly in enumerate(road.polylines):
        out.append(MatchCandidate(
            score=float(scores[i]), bearing_corr=0.5, start_node=f"A{i}",
            walk=[(f"A{i}", f"B{i}", 0)], walk_xy=poly,
            aligned_traj_xy=np.asarray(poly, dtype=np.float64).copy(),
            walk_length_m=float(np.linalg.norm(poly[-1] - poly[0]))))
    return out


def make_cfg(tmp_path: Path, **kw) -> PipelineConfig:
    cfg = PipelineConfig(url="u", city="Ulm, Germany", data_dir=tmp_path / "data",
                         output_dir=tmp_path / "out", **kw)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def make_ctx(tmp_path: Path, n_streets: int = 3, **kw) -> RunContext:
    """A context holding a road, ranked candidates and a matching ``result``."""
    ctx = RunContext.create(make_cfg(tmp_path, **kw))
    ctx.road = make_road(n_streets)
    ctx.candidates = make_candidates(ctx.road)
    ctx.geom_pool = list(ctx.candidates)
    ctx.frames = ExtractedFrames(
        frames=[np.zeros((8, 8, 3), np.uint8) for _ in range(30)],
        timestamps=[float(i) for i in range(30)], fps=30.0, total_frames=90)
    ctx.match_xz = np.stack([np.linspace(0.0, 400.0, 30), np.zeros(30)], axis=1)
    ctx.match_ts = np.asarray(ctx.frames.timestamps)
    ctx.result = {"city": ctx.cfg.city, "matches": [
        {"street_names": [f"Street {i}"], "score_rms_m": c.score}
        for i, c in enumerate(ctx.candidates)]}
    return ctx


def prime_ranks(ctx: RunContext) -> None:
    """What the aerial / tile3d channels leave on the context when they no-op."""
    n = len(ctx.candidates)
    ctx.aerial_rank = {i: i + 1 for i in range(n)}
    ctx.tile3d_rank = {i: i + 1 for i in range(n)}
    ctx.w_aerial = ctx.w_tile3d = 0.0
