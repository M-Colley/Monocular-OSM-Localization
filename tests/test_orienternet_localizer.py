"""Tests for the OrienterNet OSM-canvas fetching strategy.

The full refinement head needs `third_party/OrienterNet` + a GPU, so
these tests exercise only `_fetch_canvases` — the tile-download planner —
via injected fakes.  The property under test: a short route triggers ONE
TileManager (= one Overpass download) queried per keyframe, while a very
long route falls back to per-keyframe downloads.
"""

from __future__ import annotations

import numpy as np

from src.orienternet_localizer import _fetch_canvases


class FakeBBox:
    """Mimics maploc's BoundaryBox: constructed from two corners and
    padded with `+ margin`."""

    def __init__(self, mn, mx):
        self.min_ = np.asarray(mn, dtype=float)
        self.max_ = np.asarray(mx, dtype=float)

    def __add__(self, margin):
        return FakeBBox(self.min_ - margin, self.max_ + margin)


class FakeTileManager:
    from_bbox_calls: list = []

    def __init__(self, bbox):
        self.bbox = bbox
        self.query_calls = []

    @classmethod
    def from_bbox(cls, proj, bbox, ppm, path=None):
        cls.from_bbox_calls.append(bbox)
        return cls(bbox)

    def query(self, bbox):
        self.query_calls.append(bbox)
        return ("canvas", tuple(bbox.min_), tuple(bbox.max_))


def _reset():
    FakeTileManager.from_bbox_calls = []


def test_short_route_downloads_osm_once() -> None:
    """5 keyframes within a ~400 m route bbox -> a single TileManager
    (one OSM download), queried once per keyframe."""
    _reset()
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 50.0],
                   [300.0, 100.0], [400.0, 100.0]])

    canvases = _fetch_canvases(
        FakeTileManager, FakeBBox, proj=None, xy_all=xy, tile_m=160.0, ppm=2,
    )

    assert len(canvases) == len(xy)
    assert len(FakeTileManager.from_bbox_calls) == 1, (
        "expected ONE shared OSM download, got one per keyframe"
    )
    # The shared bbox covers the whole route plus the tile margin.
    shared = FakeTileManager.from_bbox_calls[0]
    assert (shared.min_ <= xy.min(axis=0) - 160.0).all()
    assert (shared.max_ >= xy.max(axis=0) + 160.0).all()
    # Each canvas is centred on its own keyframe (distinct query bboxes).
    assert len({c[1] for c in canvases}) == len(xy)


def test_huge_route_falls_back_to_per_keyframe_downloads() -> None:
    """A route spanning > max_shared_span_m must keep the old per-keyframe
    fetch (a single 5+ km tile download would be enormous)."""
    _reset()
    xy = np.array([[0.0, 0.0], [2500.0, 0.0], [5000.0, 0.0]])

    canvases = _fetch_canvases(
        FakeTileManager, FakeBBox, proj=None, xy_all=xy, tile_m=160.0, ppm=2,
    )

    assert len(canvases) == 3
    assert len(FakeTileManager.from_bbox_calls) == 3


def test_shared_manager_failure_falls_back_per_keyframe() -> None:
    """If the whole-route download fails, per-keyframe fetching still
    produces one canvas per keyframe."""
    _reset()

    class FlakyManager(FakeTileManager):
        first = True

        @classmethod
        def from_bbox(cls, proj, bbox, ppm, path=None):
            if cls.first:
                cls.first = False
                raise ValueError("HTTP 400: bbox too large")
            return super().from_bbox(proj, bbox, ppm, path=path)

    FlakyManager.from_bbox_calls = []
    xy = np.array([[0.0, 0.0], [100.0, 0.0]])

    canvases = _fetch_canvases(
        FlakyManager, FakeBBox, proj=None, xy_all=xy, tile_m=160.0, ppm=2,
    )
    assert len(canvases) == 2
    # The failed shared attempt is followed by two per-keyframe fetches.
    assert len(FlakyManager.from_bbox_calls) == 2


def test_hung_shared_download_hits_deadline_then_falls_back() -> None:
    """A download that streams forever (osm.org's slow-trickle mode, which
    urllib3's per-op timeout never catches) must be abandoned at the hard
    deadline instead of stalling the refine for 35+ minutes."""
    import time as _time

    class HungOnSharedManager(FakeTileManager):
        calls = 0

        @classmethod
        def from_bbox(cls, proj, bbox, ppm, path=None):
            cls.calls += 1
            if cls.calls == 1:          # the shared whole-route fetch hangs
                _time.sleep(30.0)
            return super().from_bbox(proj, bbox, ppm, path=path)

    HungOnSharedManager.from_bbox_calls = []
    xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    t0 = _time.monotonic()
    canvases = _fetch_canvases(
        HungOnSharedManager, FakeBBox, proj=None, xy_all=xy,
        tile_m=160.0, ppm=2, fetch_deadline_s=0.4,
    )
    assert _time.monotonic() - t0 < 10.0     # nowhere near the 30 s hang
    assert len(canvases) == 2                # per-keyframe fallback served


def test_json_cache_makes_second_run_offline(tmp_path) -> None:
    """With cache_dir set, the first run writes the raw OSM JSON per bbox
    and the second run performs ZERO downloads."""
    from types import SimpleNamespace

    class CachingManager(FakeTileManager):
        downloads = 0

        @classmethod
        def from_bbox(cls, proj, bbox, ppm, path=None):
            if path is not None and path.exists():
                return cls(bbox)             # cache hit: no download
            cls.downloads += 1
            if path is not None:
                path.write_text("{}", encoding="utf-8")
            return cls(bbox)

    proj = SimpleNamespace(latlonalt=(48.4, 9.99, 0.0))
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 50.0]])
    for expected_downloads in (1, 1):        # 2nd pass adds none
        _fetch_canvases(CachingManager, FakeBBox, proj=proj, xy_all=xy,
                        tile_m=160.0, ppm=2, cache_dir=tmp_path)
        assert CachingManager.downloads == expected_downloads
    assert len(list(tmp_path.glob("osm_*.json"))) == 1
