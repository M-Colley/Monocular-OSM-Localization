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
    def from_bbox(cls, proj, bbox, ppm):
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
        def from_bbox(cls, proj, bbox, ppm):
            if cls.first:
                cls.first = False
                raise ValueError("HTTP 400: bbox too large")
            return super().from_bbox(proj, bbox, ppm)

    FlakyManager.from_bbox_calls = []
    xy = np.array([[0.0, 0.0], [100.0, 0.0]])

    canvases = _fetch_canvases(
        FlakyManager, FakeBBox, proj=None, xy_all=xy, tile_m=160.0, ppm=2,
    )
    assert len(canvases) == 2
    # The failed shared attempt is followed by two per-keyframe fetches.
    assert len(FlakyManager.from_bbox_calls) == 2
