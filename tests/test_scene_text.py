"""Tests for scene-text OCR extraction (easyocr injected, no models)."""

from __future__ import annotations

from pathlib import Path

from src.scene_text import SceneText, _polygon_to_bbox, extract_scene_text


class _FakeReader:
    """Stand-in for easyocr.Reader: returns a scripted detection per frame."""

    def __init__(self, per_frame: list[list[tuple]]) -> None:
        self._per_frame = per_frame
        self._i = 0

    def readtext(self, image) -> list:
        out = self._per_frame[self._i % len(self._per_frame)]
        self._i += 1
        return out


def _fake_frames(times: list[float]):
    def _reader(video_path, start, end, interval):
        return [(t, object()) for t in times]
    return _reader


def test_extract_filters_by_confidence_and_length(tmp_path: Path) -> None:
    reader = _FakeReader([[
        ([], "Sedelhöfe", 0.99),   # keep
        ([], "ab", 0.99),          # too short
        ([], "noise", 0.10),       # too low conf
    ]])
    out = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=reader,
        frame_reader=_fake_frames([0.0]), min_confidence=0.3, min_len=3,
    )
    assert [s.text for s in out] == ["Sedelhöfe"]
    assert out[0].confidence == 0.99
    assert out[0].t_sec == 0.0


def test_extract_records_time_per_frame(tmp_path: Path) -> None:
    reader = _FakeReader([
        [([], "Alpha", 0.9)],
        [([], "Beta", 0.8)],
    ])
    out = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=reader,
        frame_reader=_fake_frames([0.0, 12.0]),
    )
    assert [(s.text, s.t_sec) for s in out] == [("Alpha", 0.0), ("Beta", 12.0)]


def test_extract_uses_and_writes_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    reader = _FakeReader([[([], "Sedelhöfe", 0.99)]])
    kw = dict(
        ocr_reader=reader, frame_reader=_fake_frames([0.0]),
        cache_path=cache, sample_interval_sec=6.0, min_confidence=0.3, min_len=3,
    )
    first = extract_scene_text(tmp_path / "v.mp4", **kw)
    assert cache.exists()

    # Second call must hit the cache — a reader that would blow up proves
    # OCR was not re-run.
    class _Boom:
        def readtext(self, image):
            raise AssertionError("OCR must not run on a cache hit")

    second = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=_Boom(),
        frame_reader=_fake_frames([0.0]), cache_path=cache,
        sample_interval_sec=6.0, min_confidence=0.3, min_len=3,
    )
    assert first == second == [SceneText("Sedelhöfe", 0.99, 0.0)]


def test_cache_invalidated_by_video_change(tmp_path: Path) -> None:
    """Re-downloading the same slug at a different resolution/format must
    invalidate the cache — the old signature ignored the video entirely and
    silently served detections OCR'd from the previous file."""
    import os

    cache = tmp_path / "cache.json"
    video = tmp_path / "v.mp4"
    video.write_bytes(b"AAA")
    kw = dict(frame_reader=_fake_frames([0.0]), cache_path=cache,
              sample_interval_sec=6.0)
    extract_scene_text(video, ocr_reader=_FakeReader([[([], "Alpha", 0.9)]]), **kw)

    # Same slug, different file (size + mtime change) → must re-run OCR.
    video.write_bytes(b"BBBBBB")
    os.utime(video, (1_700_000_000, 1_700_000_000))
    out = extract_scene_text(
        video, ocr_reader=_FakeReader([[([], "Beta", 0.9)]]), **kw)
    assert [s.text for s in out] == ["Beta"]

    # Unchanged file → cache still hits.
    class _Boom:
        def readtext(self, image):
            raise AssertionError("OCR must not run on a cache hit")

    again = extract_scene_text(video, ocr_reader=_Boom(), **kw)
    assert [s.text for s in again] == ["Beta"]


def test_cache_invalidated_by_param_change(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=_FakeReader([[([], "Alpha", 0.9)]]),
        frame_reader=_fake_frames([0.0]), cache_path=cache,
        sample_interval_sec=6.0,
    )
    # Different interval → signature mismatch → re-run with the new reader.
    out = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=_FakeReader([[([], "Beta", 0.9)]]),
        frame_reader=_fake_frames([0.0]), cache_path=cache,
        sample_interval_sec=3.0,
    )
    assert [s.text for s in out] == ["Beta"]


def test_polygon_to_bbox_axis_aligned() -> None:
    # easyocr 4-point polygon (may be rotated) → axis-aligned min/max box.
    poly = [[10, 20], [90, 25], [88, 60], [12, 55]]
    assert _polygon_to_bbox(poly) == (10.0, 20.0, 90.0, 60.0)


def test_polygon_to_bbox_empty_is_none() -> None:
    assert _polygon_to_bbox([]) is None
    assert _polygon_to_bbox(None) is None


def test_bbox_retained_from_reader(tmp_path: Path) -> None:
    poly = [[10, 20], [90, 20], [90, 60], [10, 60]]
    reader = _FakeReader([[(poly, "Sedelhöfe", 0.99)]])
    out = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=reader,
        frame_reader=_fake_frames([0.0]),
    )
    assert out[0].bbox == (10.0, 20.0, 90.0, 60.0)


def test_bbox_survives_cache_roundtrip(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    poly = [[1, 2], [30, 2], [30, 40], [1, 40]]
    kw = dict(frame_reader=_fake_frames([0.0]), cache_path=cache,
              sample_interval_sec=6.0)
    first = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=_FakeReader([[(poly, "Rathaus", 0.9)]]),
        **kw)

    class _Boom:
        def readtext(self, image):
            raise AssertionError("OCR must not run on a cache hit")

    second = extract_scene_text(tmp_path / "v.mp4", ocr_reader=_Boom(), **kw)
    # bbox is restored to a tuple (JSON has no tuples) so the cache hit ==
    # the freshly-computed result.
    assert first == second
    assert second[0].bbox == (1.0, 2.0, 30.0, 40.0)


def test_old_cache_without_schema_is_invalidated(tmp_path: Path) -> None:
    """A schema-1 cache (no 'schema' key, no bbox) must be ignored so the
    detections regenerate WITH boxes."""
    import json

    cache = tmp_path / "cache.json"
    # Hand-write a plausible old-format cache with no schema/bbox.
    cache.write_text(json.dumps({
        "signature": {"sample_interval_sec": 6.0, "start_sec": 0.0,
                      "end_sec": None, "languages": ["de", "en"],
                      "min_confidence": 0.3, "min_len": 3, "super_res": False},
        "detections": [{"text": "Old", "confidence": 0.9, "t_sec": 0.0}],
    }), encoding="utf-8")
    poly = [[0, 0], [5, 0], [5, 5], [0, 5]]
    out = extract_scene_text(
        tmp_path / "v.mp4", ocr_reader=_FakeReader([[(poly, "New", 0.9)]]),
        frame_reader=_fake_frames([0.0]), cache_path=cache,
        sample_interval_sec=6.0,
    )
    assert [s.text for s in out] == ["New"]
    assert out[0].bbox == (0.0, 0.0, 5.0, 5.0)
