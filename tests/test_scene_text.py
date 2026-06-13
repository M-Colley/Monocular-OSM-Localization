"""Tests for scene-text OCR extraction (easyocr injected, no models)."""

from __future__ import annotations

from pathlib import Path

from src.scene_text import SceneText, extract_scene_text


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
