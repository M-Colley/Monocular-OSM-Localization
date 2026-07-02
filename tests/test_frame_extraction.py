"""Frame extraction tests using a synthetic video produced with cv2.VideoWriter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.frame_extraction import FrameExtractionError, extract_frames


def _make_video(path: Path, n_frames: int = 60, fps: float = 30.0,
                w: int = 160, h: int = 120) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this platform")
    for i in range(n_frames):
        # Each frame is a unique solid color so we can verify we picked the
        # frames we think we picked.
        frame = np.full((h, w, 3), (i * 4 % 256, (i * 7) % 256, (i * 11) % 256), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_extract_frames_stride(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "test.mp4", n_frames=30)
    out = extract_frames(video, stride=5, max_frames=100)
    # 30 frames, stride 5 → indices 0, 5, 10, 15, 20, 25 → 6 frames.
    assert len(out.frames) == 6
    assert out.timestamps[0] == 0.0
    assert out.timestamps[-1] == pytest.approx(25 / 30.0, abs=0.01)


def test_extract_frames_window(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "test.mp4", n_frames=60, fps=30.0)
    out = extract_frames(video, stride=1, max_frames=100, start_sec=0.5, end_sec=1.0)
    # 0.5..1.0 s @ 30 fps = 15 frames.
    assert 14 <= len(out.frames) <= 15


def test_extract_frames_max_cap(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "test.mp4", n_frames=50)
    out = extract_frames(video, stride=1, max_frames=10)
    assert len(out.frames) == 10


def test_extract_frames_missing_file() -> None:
    with pytest.raises(FrameExtractionError):
        extract_frames(Path("does-not-exist.mp4"))


def test_extract_frames_uncapped(tmp_path: Path) -> None:
    """max_frames=None must read the whole segment — a fixed cap smaller
    than the segment silently truncates the analyzed window."""
    video = _make_video(tmp_path / "test.mp4", n_frames=50)
    out = extract_frames(video, stride=1, max_frames=None)
    assert len(out.frames) == 50


class _FakeCap:
    """Stand-in VideoCapture: configurable frame count / fps / per-frame
    media times, for containers cv2 can't be made to produce on demand."""

    def __init__(self, n_frames: int, fps: float = 30.0,
                 frame_count: int = 0, pos_msec=None):
        self._n = n_frames
        self._fps = fps
        self._frame_count = frame_count
        self._pos_msec = pos_msec  # list of media times (ms) or None
        self._i = 0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == cv2.CAP_PROP_POS_MSEC:
            # Real backends report the PTS of the frame just read.
            if self._pos_msec is not None and 0 < self._i <= len(self._pos_msec):
                return float(self._pos_msec[self._i - 1])
            return 0.0
        return 0.0

    def set(self, prop, value):
        return True

    def read(self):
        if self._i >= self._n:
            return False, None
        self._i += 1
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        pass


def test_extract_frames_unknown_frame_count_open_ended(tmp_path: Path, monkeypatch) -> None:
    """CAP_PROP_FRAME_COUNT == 0 with an open-ended segment must read
    until EOF instead of raising 'no frames extracted'."""
    video = tmp_path / "stream.mp4"
    video.write_bytes(b"\x00")  # existence check only; capture is faked
    monkeypatch.setattr(cv2, "VideoCapture",
                        lambda p: _FakeCap(n_frames=20, frame_count=0))
    out = extract_frames(video, stride=1, max_frames=None, end_sec=None)
    assert len(out.frames) == 20


def test_extract_frames_vfr_timestamps_from_pos_msec(tmp_path: Path, monkeypatch) -> None:
    """Timestamps come from the container's media clock (CAP_PROP_POS_MSEC),
    not idx/nominal-fps — VFR sources drift under the nominal clock."""
    video = tmp_path / "vfr.mp4"
    video.write_bytes(b"\x00")
    # Nominal 30 fps, but true frame times are wildly non-uniform.
    times_ms = [0.0, 100.0, 350.0, 900.0]
    monkeypatch.setattr(cv2, "VideoCapture",
                        lambda p: _FakeCap(n_frames=4, frame_count=4,
                                           pos_msec=times_ms))
    out = extract_frames(video, stride=1, max_frames=None)
    assert out.timestamps == pytest.approx([0.0, 0.1, 0.35, 0.9])


def test_extract_frames_pos_msec_unavailable_falls_back_to_fps(
        tmp_path: Path, monkeypatch) -> None:
    """Backends that always report POS_MSEC == 0 fall back to idx/fps."""
    video = tmp_path / "nomsec.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr(cv2, "VideoCapture",
                        lambda p: _FakeCap(n_frames=3, fps=10.0, frame_count=3))
    out = extract_frames(video, stride=1, max_frames=None)
    assert out.timestamps == pytest.approx([0.0, 0.1, 0.2])
