"""Large clips preserve exact pixels and random access with bounded memory."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from monocular_osm.frame_extraction import FrameSequence, extract_frames
from tests.test_frame_extraction import _FakeCap, _TruncatedCap, _make_video


def test_spill_is_lossless_and_slices_remain_lazy():
    rng = np.random.default_rng(4)
    expected = [rng.integers(0, 256, (40, 60, 3), dtype=np.uint8) for _ in range(12)]
    frames = FrameSequence(memory_mb=0.02)
    try:
        for frame in expected:
            frames.append(frame)
        assert frames.spilled
        view = frames[1::3][::-1]
        assert not isinstance(view, list)
        for got, want in zip(view, expected[1::3][::-1]):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(frames[-1], expected[-1])
        assert frames.resident_bytes <= 0.02 * 1024 ** 2
        with pytest.raises(IndexError):
            _ = frames[len(frames)]
        with pytest.raises(IndexError):
            _ = frames[-len(frames) - 1]
    finally:
        directory = frames._temp.name
        frames.close()
    from pathlib import Path
    assert not Path(directory).exists()


def test_parallel_pair_reads_keep_pixel_identity():
    frames = FrameSequence(memory_mb=0.01)
    try:
        for i in range(20):
            frames.append(np.full((32, 32, 3), i, np.uint8))
        with ThreadPoolExecutor(max_workers=4) as pool:
            sums = list(pool.map(lambda i: int(frames[i][0, 0, 0]), range(20)))
        assert sums == list(range(20))
        assert frames.resident_bytes <= 0.01 * 1024 ** 2
    finally:
        frames.close()


def test_parallel_grayscale_reads_keep_identity_under_eviction():
    from monocular_osm.frame_extraction import GrayscaleFrames

    gray = GrayscaleFrames([np.full((32, 32, 3), i, np.uint8) for i in range(20)])
    indices = list(range(20)) * 20
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda i: int(gray[i][0, 0]), indices))
    assert values == indices
    assert len(gray._cache) == 2


def test_extraction_can_resize_without_changing_sampling(tmp_path):
    video = _make_video(tmp_path / "drive.mp4", n_frames=30, w=320, h=240)
    full = extract_frames(video, stride=5)
    resized = extract_frames(video, stride=5, max_width=80, memory_mb=0.01)
    try:
        assert len(full.frames) == len(resized.frames) == 6
        assert full.timestamps == resized.timestamps
        assert resized.frames[0].shape == (60, 80, 3)
        assert resized.frames.spilled
    finally:
        full.close()
        resized.close()


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_memory_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        FrameSequence(budget)


def test_grayscale_view_reads_only_requested_frames_and_keeps_two():
    import cv2
    from monocular_osm.frame_extraction import GrayscaleFrames

    class Source:
        reads = []
        def __len__(self):
            return 10000
        def __getitem__(self, index):
            self.reads.append(index)
            return np.full((20, 30, 3), (index % 255, 40, 80), np.uint8)

    source = Source()
    gray = GrayscaleFrames(source)
    assert not source.reads
    expected = cv2.cvtColor(source[900], cv2.COLOR_BGR2GRAY)
    source.reads.clear()
    np.testing.assert_array_equal(gray[900], expected)
    np.testing.assert_array_equal(gray[900], expected)
    assert source.reads == [900]
    _ = gray[901], gray[902], gray[-1]
    assert len(gray._cache) == 2
    assert not isinstance(gray[::100], list)


@pytest.mark.parametrize("parameters", [
    dict(stride=3, max_frames=None, start_sec=0.25, end_sec=9.4),
    dict(stride=2, max_frames=105, start_sec=0.0, end_sec=10.0),
    dict(stride=6, max_frames=None, start_sec=0.0, end_sec=None),
])
def test_sparse_queries_match_dense_pixels_and_timestamps_exactly(tmp_path, parameters):
    video = _make_video(tmp_path / "queries.mp4", n_frames=360, w=80, h=60)
    dense = extract_frames(video, memory_mb=0.01, **parameters)
    sparse = extract_frames(video, memory_mb=0.01, sample_count=80, **parameters)
    try:
        indices = np.linspace(0, len(dense.frames) - 1, min(80, len(dense.frames))).astype(int)
        assert sparse.sampling["mode"] == "linspace"
        assert sparse.sampling["dense_count"] == len(dense.frames)
        assert sparse.sampling["dense_indices"] == indices.tolist()
        assert sparse.timestamps == [dense.timestamps[i] for i in indices]
        assert len(sparse.frames) == len(indices)
        for sampled, i in zip(sparse.frames, indices):
            np.testing.assert_array_equal(sampled, dense.frames[i])
    finally:
        dense.close()
        sparse.close()


def test_sparse_queries_decode_the_full_window_despite_a_storage_cap(tmp_path, monkeypatch):
    import monocular_osm.frame_extraction as fe

    video = tmp_path / "queries.mp4"
    video.write_bytes(b"source")
    capture = _FakeCap(300, frame_count=300)
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda path: capture)
    sparse = extract_frames(video, stride=3, max_frames=50, sample_count=80)
    try:
        assert len(sparse.frames) == 50
        assert capture._i == 300
        assert sparse.decoded_until_s == pytest.approx(299 / 30)
    finally:
        sparse.close()


def test_sparse_extraction_keeps_truncation_detection(tmp_path, monkeypatch):
    import monocular_osm.frame_extraction as fe

    video = tmp_path / "truncated.mp4"
    video.write_bytes(b"source")
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda path: _TruncatedCap(9000, 1350))
    with pytest.raises(fe.TruncatedVideoError):
        extract_frames(video, stride=3, max_frames=None, end_sec=300, sample_count=80)


def test_sparse_extraction_recovers_exact_indices_after_small_metadata_shortfall(
        tmp_path, monkeypatch):
    import monocular_osm.frame_extraction as fe

    video = tmp_path / "shortfall.mp4"
    video.write_bytes(b"source")
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda path: _FakeCap(300, frame_count=310))
    dense = extract_frames(video, stride=3, max_frames=None)
    sparse = extract_frames(video, stride=3, max_frames=None, sample_count=80)
    try:
        indices = np.linspace(0, len(dense.frames) - 1, 80).astype(int)
        assert sparse.timestamps == [dense.timestamps[i] for i in indices]
        assert sparse.sampling["metadata_shortfall_recovered"] is True
        assert sparse.sampling["dense_count"] == 100
    finally:
        dense.close()
        sparse.close()


def test_unknown_horizon_retains_the_original_dense_grid(tmp_path, monkeypatch):
    import monocular_osm.frame_extraction as fe

    video = tmp_path / "unknown.mp4"
    video.write_bytes(b"source")
    monkeypatch.setattr(fe.cv2, "VideoCapture", lambda path: _FakeCap(300, frame_count=0))
    frames = extract_frames(video, stride=3, max_frames=None, sample_count=80)
    try:
        assert len(frames.frames) == 100
        assert frames.sampling["mode"] == "dense"
        assert frames.sampling["requested_sample_count"] == 80
    finally:
        frames.close()
