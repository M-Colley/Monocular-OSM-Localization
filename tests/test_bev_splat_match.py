"""Tests for the BevSplat aerial-channel scaffold.

The upstream model has no released weights and ships CUDA extensions
that are not pip-installable, so these tests exercise the integration
contract end-to-end via injected fakes rather than the real model:

* tile renderer is replaced with a synthetic generator so the test is
  offline and deterministic
* inference is replaced with :class:`MockBevSplatInference` (NCC-based)
  or with custom fakes that exercise specific code paths

The properties verified are:

  * one result per candidate, in input order
  * tile files are written on success
  * mock inference returns scores in [0, 1] and shifts in [-1, 1]
  * unloaded backend produces an ``error`` field and ``score == 0``
    without crashing the pipeline
  * tile-renderer exceptions are caught per-candidate
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from src.bev_splat_match import (
    BevSplatConfig,
    BevSplatMatchResult,
    MockBevSplatInference,
    _build_bev_splat_args,
    _load_bev_splat_inference,
    score_candidates_with_bevsplat,
)
from src.osm_data import _build_polyline_view
from src.trajectory_matching import MatchCandidate


def _road_graph() -> "RoadGraph":
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:32632"
    coords = {0: (0.0, 0.0), 1: (300.0, 0.0), 2: (300.0, 300.0)}
    for n, (x, y) in coords.items():
        g.add_node(n, x=x, y=y)

    def add(a: int, b: int) -> None:
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        length = float(np.hypot(bx - ax, by - ay))
        g.add_edge(a, b, length=length, geometry=LineString([(ax, ay), (bx, by)]))
        g.add_edge(b, a, length=length, geometry=LineString([(bx, by), (ax, ay)]))

    add(0, 1)
    add(1, 2)
    return _build_polyline_view(g)


def _candidates() -> list[MatchCandidate]:
    walks_xy = [
        np.array([[0.0, 0.0], [300.0, 0.0]]),
        np.array([[300.0, 0.0], [300.0, 300.0]]),
    ]
    out = []
    for w in walks_xy:
        out.append(MatchCandidate(
            score=1.0,
            bearing_corr=0.5,
            start_node=0,
            walk=[(0, 1, 0)],
            walk_xy=w,
            aligned_traj_xy=w.copy(),
            walk_length_m=300.0,
        ))
    return out


def _fake_tile_renderer(rng_seed: int = 0):
    """Return a tile renderer that produces deterministic synthetic tiles."""

    def renderer(source, road, cand, *, size, half_extent_m, geotessera_year):
        rng = np.random.default_rng(rng_seed + cand.walk_xy.sum().astype(int))
        # Synthetic "satellite" tile: structured noise so NCC has something to bite on.
        img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        # Add a block of structure so template matching has signal.
        img[size // 4 : size // 2, size // 4 : size // 2] = 220
        return img

    return renderer


def test_mock_inference_returns_normalized_outputs() -> None:
    """MockBevSplatInference should return score in [0,1] and shifts in [-1,1]."""
    rng = np.random.default_rng(0)
    ground = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    sat = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    K = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])

    mock = MockBevSplatInference()
    score, du, dv, dh = mock(ground, sat, K)

    assert 0.0 <= score <= 1.0
    assert -1.0 <= du <= 1.0
    assert -1.0 <= dv <= 1.0
    assert dh == 0.0  # Mock has no heading prediction.


def test_score_candidates_returns_one_per_input(tmp_path: Path) -> None:
    road = _road_graph()
    cands = _candidates()
    rng = np.random.default_rng(0)
    query = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    K = np.eye(3, dtype=np.float64)

    results = score_candidates_with_bevsplat(
        query,
        K,
        road,
        cands,
        output_dir=tmp_path,
        config=BevSplatConfig(satellite_size=128, half_extent_m=200.0),
        inference=MockBevSplatInference(),
        tile_renderer=_fake_tile_renderer(),
    )

    assert len(results) == len(cands)
    assert [r.candidate_index for r in results] == [0, 1]
    assert all(isinstance(r, BevSplatMatchResult) for r in results)
    assert all(r.satellite_path is not None and r.satellite_path.exists() for r in results)
    assert all(r.error is None for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_score_candidates_handles_missing_inference(tmp_path: Path) -> None:
    """No weights/no real model → results carry error, pipeline doesn't crash."""
    road = _road_graph()
    cands = _candidates()
    query = np.full((64, 64, 3), 128, dtype=np.uint8)
    K = np.eye(3)

    # weights_path=None → _load_bev_splat_inference returns (None, err) → recorded per candidate
    results = score_candidates_with_bevsplat(
        query,
        K,
        road,
        cands,
        output_dir=tmp_path,
        config=BevSplatConfig(weights_path=None, satellite_size=128, half_extent_m=200.0),
        tile_renderer=_fake_tile_renderer(),
    )
    assert len(results) == len(cands)
    assert all(r.error is not None for r in results)
    assert all(r.score == 0.0 for r in results)
    # Tiles should still be written for inspection.
    assert all(r.satellite_path is not None and r.satellite_path.exists() for r in results)


def test_score_candidates_handles_tile_render_failure(tmp_path: Path) -> None:
    """A renderer that raises should produce an error result per candidate, not a crash."""
    road = _road_graph()
    cands = _candidates()
    query = np.full((64, 64, 3), 128, dtype=np.uint8)
    K = np.eye(3)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic tile fetch failure")

    results = score_candidates_with_bevsplat(
        query,
        K,
        road,
        cands,
        output_dir=tmp_path,
        config=BevSplatConfig(),
        inference=MockBevSplatInference(),
        tile_renderer=boom,
    )
    assert len(results) == len(cands)
    assert all("synthetic tile fetch failure" in (r.error or "") for r in results)
    assert all(r.satellite_path is None for r in results)


def test_score_candidates_skips_when_query_none(tmp_path: Path) -> None:
    """No query frame → empty result list, no tiles rendered."""
    road = _road_graph()
    cands = _candidates()
    results = score_candidates_with_bevsplat(
        None,
        np.eye(3),
        road,
        cands,
        output_dir=tmp_path,
        config=BevSplatConfig(),
        inference=MockBevSplatInference(),
        tile_renderer=_fake_tile_renderer(),
    )
    assert results == []


def test_score_candidates_records_inference_exceptions(tmp_path: Path) -> None:
    """Inference that raises → error recorded per candidate, others still run."""
    road = _road_graph()
    cands = _candidates()
    query = np.full((64, 64, 3), 128, dtype=np.uint8)
    K = np.eye(3)

    call_count = {"n": 0}

    def flaky(ground, satellite, K):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated OOM")
        return 0.42, 0.0, 0.0, 0.0

    results = score_candidates_with_bevsplat(
        query,
        K,
        road,
        cands,
        output_dir=tmp_path,
        config=BevSplatConfig(satellite_size=64, half_extent_m=200.0),
        inference=flaky,
        tile_renderer=_fake_tile_renderer(),
    )
    assert len(results) == 2
    assert "simulated OOM" in (results[0].error or "")
    assert results[0].score == 0.0
    assert results[1].error is None
    assert results[1].score == 0.42


# ---------- Tests for the real-loader prerequisites ----------------------


def test_load_bev_splat_inference_no_weights_returns_clear_error() -> None:
    inference, err = _load_bev_splat_inference(BevSplatConfig(weights_path=None))
    assert inference is None
    assert err is not None
    assert "weights_path" in err.lower() or "weights" in err.lower()


def test_load_bev_splat_inference_weights_missing_returns_clear_error(tmp_path: Path) -> None:
    bogus = tmp_path / "nonexistent.pth"
    inference, err = _load_bev_splat_inference(BevSplatConfig(weights_path=bogus))
    assert inference is None
    assert err is not None
    assert "not found" in err.lower()


def test_load_bev_splat_inference_no_repo_returns_clear_error(tmp_path: Path) -> None:
    """Once weights exist but repo_path is unset, the loader must explain
    that the user needs to clone the BevSplat repo and build CUDA exts."""
    fake_weights = tmp_path / "weights.pth"
    fake_weights.write_bytes(b"")  # exists but not a real checkpoint
    inference, err = _load_bev_splat_inference(BevSplatConfig(
        weights_path=fake_weights,
        repo_path=None,
    ))
    assert inference is None
    assert err is not None
    assert "repo" in err.lower() and "wangqww" in err.lower()


def test_load_bev_splat_inference_bad_repo_path_returns_clear_error(tmp_path: Path) -> None:
    """A repo_path that doesn't look like a BevSplat checkout must be caught
    BEFORE the actual import, so we don't crash the pipeline with an opaque
    ModuleNotFoundError."""
    fake_weights = tmp_path / "weights.pth"
    fake_weights.write_bytes(b"")
    fake_repo = tmp_path / "not-bev-splat"
    fake_repo.mkdir()
    inference, err = _load_bev_splat_inference(BevSplatConfig(
        weights_path=fake_weights,
        repo_path=fake_repo,
    ))
    assert inference is None
    assert err is not None
    assert "models_kitti_seq" in err or "clone" in err.lower()


def test_build_bev_splat_args_reproduces_upstream_defaults() -> None:
    """The argparse defaults baked into _BEV_SPLAT_DEFAULT_ARGS must match
    what wangqww/BevSplat's train_KITTI_weak_seq.parse_args() produces, so
    the model is constructed against the same hyperparameters it was
    trained with."""
    ns = _build_bev_splat_args({})
    # Spot-check the load-bearing fields the Model.__init__ reads.
    assert ns.level == "0_2"
    assert ns.channels == "32_16_4"
    assert ns.N_iters == 1
    assert ns.share == 1
    assert ns.proj == "geo"
    assert ns.sequence == 2
    assert ns.rotation_range == 10.0
    assert ns.shift_range_lat == 20.0
    assert ns.shift_range_lon == 20.0


def test_build_bev_splat_args_applies_overrides() -> None:
    ns = _build_bev_splat_args({"sequence": 4, "rotation_range": 5.0})
    assert ns.sequence == 4
    assert ns.rotation_range == 5.0
    # Untouched defaults survive.
    assert ns.level == "0_2"
