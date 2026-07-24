"""Extract a sequence of frames from a video at a fixed temporal stride.

Returned in chronological order. Frames are kept in memory as numpy arrays;
the dashcam clips we work with at 720p / ~1 fps effective rate easily fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class FrameExtractionError(RuntimeError):
    pass


@dataclass
class ExtractedFrames:
    frames: list[np.ndarray]
    timestamps: list[float]   # seconds from start of video
    fps: float
    total_frames: int


def extract_frames(
    video_path: Path,
    *,
    stride: int = 6,
    max_frames: int | None = 400,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> ExtractedFrames:
    """Read frames from `video_path`, keeping every `stride`-th frame in
    [start_sec, end_sec).

    `stride=6` at 30 fps gives one frame every 0.2 s, which is a reasonable
    inter-frame baseline for monocular VO on a vehicle moving at urban speed.

    `max_frames=None` removes the cap — the segment bounds the count. A
    fixed cap smaller than the segment silently truncates the analyzed
    window (a 10-minute `--vo-segment` with the old 4200-frame default
    only actually analyzed the first 7 minutes), so callers that honor a
    user-specified segment should pass None.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FrameExtractionError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FrameExtractionError(f"cv2 failed to open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = int(start_sec * fps)
    # Some containers (raw elementary streams, broken duration metadata)
    # report a frame count of 0 even though cap.read() decodes fine. With
    # an open-ended segment we then read until EOF instead of never reading.
    if end_sec is not None:
        end_frame: int | None = int(end_sec * fps)
    elif total > 0:
        end_frame = total
    else:
        end_frame = None  # unknown length: cap.read() termination ends the loop

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    idx = start_frame
    try:
        while (end_frame is None or idx < end_frame) and (
                max_frames is None or len(frames) < max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            # Media time of the frame just decoded (POS_MSEC after read()
            # reports its PTS) — exact on variable-frame-rate sources
            # where idx/fps drifts.
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if (idx - start_frame) % stride == 0:
                frames.append(frame)
                # Backends that can't report POS_MSEC return 0/negative;
                # fall back to the nominal-fps timestamp there.
                if pos_ms is not None and pos_ms > 0:
                    timestamps.append(pos_ms / 1000.0)
                else:
                    timestamps.append(idx / fps)
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise FrameExtractionError(
            f"no frames extracted (start={start_sec}s end={end_sec} stride={stride})"
        )

    return ExtractedFrames(frames=frames, timestamps=timestamps, fps=fps, total_frames=total)
