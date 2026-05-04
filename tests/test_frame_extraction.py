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
