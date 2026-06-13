"""Tests for pipeline input acquisition and the geocode error wrapper.

These exercise `_resolve_input_video` / `_fetch_road_graph` directly so
no network, video decoding, or OSM access is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.pipeline as pipeline
from src.pipeline import (
    PipelineConfig,
    _auto_estimated_length_m,
    _fetch_road_graph,
    _resolve_input_video,
)


def _cfg(tmp_path: Path, **overrides: object) -> PipelineConfig:
    defaults: dict = dict(
        url="https://example.com/video",
        city="Ulm, Germany",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
    )
    defaults.update(overrides)
    cfg = PipelineConfig(**defaults)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _forbid_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("download_video must not be called")

    monkeypatch.setattr(pipeline, "download_video", _boom)


# ---------------------------------------------------------------------------
# _resolve_input_video
# ---------------------------------------------------------------------------


def test_local_video_used_directly_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_download(monkeypatch)
    video = tmp_path / "drive.mp4"
    video.write_bytes(b"\x00")
    cfg = _cfg(tmp_path, video_path=video)
    assert _resolve_input_video(cfg) == video


def test_local_video_wins_over_skip_download_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --video beats a stale cached input.* in data/."""
    _forbid_download(monkeypatch)
    video = tmp_path / "drive.mp4"
    video.write_bytes(b"\x00")
    cfg = _cfg(tmp_path, video_path=video, skip_download=True)
    (cfg.data_dir / "input.mp4").write_bytes(b"\x01")
    assert _resolve_input_video(cfg) == video


def test_missing_local_video_raises_with_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, video_path=tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        _resolve_input_video(cfg)


def test_skip_download_returns_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_download(monkeypatch)
    cfg = _cfg(tmp_path, skip_download=True)
    cached = cfg.data_dir / "input.mp4"
    cached.write_bytes(b"\x00")
    assert _resolve_input_video(cfg) == cached


def test_skip_download_without_cache_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, skip_download=True)
    with pytest.raises(FileNotFoundError, match="skip-download"):
        _resolve_input_video(cfg)


def test_url_branch_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []
    sentinel = tmp_path / "data" / "input.mp4"

    def fake_download(url: str, out_dir: Path) -> Path:
        calls.append((url, out_dir))
        return sentinel

    monkeypatch.setattr(pipeline, "download_video", fake_download)
    cfg = _cfg(tmp_path)
    assert _resolve_input_video(cfg) == sentinel
    assert calls == [(cfg.url, cfg.data_dir)]


# ---------------------------------------------------------------------------
# _fetch_road_graph
# ---------------------------------------------------------------------------


def test_fetch_road_graph_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        pipeline, "fetch_city_graph", lambda city, cache_path: sentinel
    )
    assert _fetch_road_graph("Ulm, Germany", tmp_path / "g.graphml") is sentinel


def test_fetch_road_graph_wraps_geocode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(city: str, cache_path: Path) -> None:
        raise RuntimeError("Nominatim could not geocode 'Ulmm'")

    monkeypatch.setattr(pipeline, "fetch_city_graph", boom)
    with pytest.raises(ValueError) as exc_info:
        _fetch_road_graph("Ulmm, Germany", tmp_path / "g.graphml")

    msg = str(exc_info.value)
    assert "Ulmm, Germany" in msg
    assert "City, Country" in msg
    # The original exception must be preserved for debugging.
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# _auto_estimated_length_m
# ---------------------------------------------------------------------------


def test_auto_length_matches_urban_speed() -> None:
    # 415 s at ~5.5 m/s — calibrated against the Ulm GT track, whose
    # true 7-minute route is ~2.1-2.4 km (the old fixed 8000 m default
    # was nearly 4x too long and caused a ~880 m start error).
    length = _auto_estimated_length_m(415.0)
    assert 2000.0 <= length <= 2600.0


def test_auto_length_clamps_extremes() -> None:
    assert _auto_estimated_length_m(0.0) == 500.0       # floor
    assert _auto_estimated_length_m(10.0) == 500.0      # below floor
    assert _auto_estimated_length_m(1e6) == 12000.0     # ceiling


def test_estimated_length_default_is_auto() -> None:
    """PipelineConfig must default to None so the duration-derived prior
    kicks in; a fixed default re-introduces the length-mismatch bug."""
    cfg = PipelineConfig(
        url="u", city="c", data_dir=Path("d"), output_dir=Path("o")
    )
    assert cfg.estimated_length_m is None


# ---------------------------------------------------------------------------
# _da3_trajectory_plausible
# ---------------------------------------------------------------------------


def test_da3_plausible_accepts_smooth_drive() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    # An L-shaped drive: 20 steps east, then 20 steps north.
    path = np.array(
        [[i, 0.0] for i in range(20)] + [[19.0, j] for j in range(1, 21)]
    )
    assert _da3_trajectory_plausible(path) is True


def test_da3_plausible_rejects_zigzag_scribble() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    rng = np.random.default_rng(0)
    scribble = rng.uniform(-1, 1, size=(48, 2))  # failed pose solve
    assert _da3_trajectory_plausible(scribble) is False


def test_da3_plausible_rejects_degenerate_paths() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    assert _da3_trajectory_plausible(np.zeros((2, 2))) is False      # too short
    assert _da3_trajectory_plausible(np.zeros((10, 2))) is False     # stationary


def test_da3_plausible_tolerates_single_u_turn() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    # Drive 30 steps east, U-turn, 30 steps back west: exactly one
    # reversal among 59 segment pairs — a real maneuver, must pass.
    path = np.array(
        [[i, 0.0] for i in range(30)] + [[29.0 - i, 0.01] for i in range(1, 31)]
    )
    assert _da3_trajectory_plausible(path) is True


# ---------------------------------------------------------------------------
# _fuse_bev_rank
# ---------------------------------------------------------------------------


def test_fuse_bev_flips_a_near_tie() -> None:
    from src.pipeline import _fuse_bev_rank

    # Geometric scores nearly tied; BevSplat strongly prefers candidate 1.
    base = [3.5, 4.0, 9.0]
    bev_ranks = [3, 1, 2]
    order = _fuse_bev_rank(base, bev_ranks, w_bev=0.75)
    assert order[0] == 1  # 4.0 + 0.75 < 3.5 + 2.25


def test_fuse_bev_cannot_override_a_large_geometric_gap() -> None:
    from src.pipeline import _fuse_bev_rank

    base = [2.0, 12.0, 13.0]
    bev_ranks = [3, 1, 2]   # appearance prefers the geometric losers
    order = _fuse_bev_rank(base, bev_ranks, w_bev=0.75)
    assert order[0] == 0


def test_fuse_bev_ties_break_by_incoming_order() -> None:
    from src.pipeline import _fuse_bev_rank

    base = [5.0, 5.0]
    bev_ranks = [1, 1]
    assert _fuse_bev_rank(base, bev_ranks) == [0, 1]
