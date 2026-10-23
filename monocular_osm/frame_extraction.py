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


class TruncatedVideoError(FrameExtractionError):
    """The video stops decoding long before the window / declared duration."""


# A decode failure is only an error when it cuts the analysed span short by
# more than this fraction AND this many seconds (seeking/keyframe rounding and
# containers whose declared length is a little off are normal).
_MAX_MISSING_FRACTION = 0.10
_MIN_MISSING_S = 5.0


def tail_decodes(video_path: Path, *, at_fraction: float = 0.97) -> bool | None:
    """Does a frame near the end of the container actually decode?

    ``None`` when the container reports no usable length (cannot tell).
    A truncated or half-overwritten download usually opens fine and reports
    its full declared duration while everything after some point is garbage.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 30:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * at_fraction))
        ok, _frame = cap.read()
        return bool(ok)
    finally:
        cap.release()


@dataclass
class ExtractedFrames:
    frames: list[np.ndarray]
    timestamps: list[float]   # seconds from start of video
    fps: float
    total_frames: int
    #: Video time of the last frame actually decoded, and where the analysed
    #: span was expected to end (None when unknowable) - kept for diagnostics.
    decoded_until_s: float | None = None
    expected_until_s: float | None = None


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
    decode_failed = False
    try:
        while (end_frame is None or idx < end_frame) and (
                max_frames is None or len(frames) < max_frames):
            ok, frame = cap.read()
            if not ok:
                decode_failed = True
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

    # Where should the analysed span have ended? The earlier of the requested
    # window end and the container's declared length (a video genuinely shorter
    # than the window is fine; one that STOPS DECODING early is not).
    horizon = [h for h in (end_sec, total / fps if total > 0 else None) if h is not None]
    expected_end = min(horizon) if horizon else None
    decoded_until = timestamps[-1]
    if decode_failed and expected_end is not None:
        wanted = expected_end - start_sec
        missing = expected_end - decoded_until
        if (wanted > 0 and missing > _MIN_MISSING_S
                and missing > _MAX_MISSING_FRACTION * wanted):
            raise TruncatedVideoError(
                f"{video_path.name} stops decoding at {decoded_until:.0f} s but the "
                f"analysed window runs to {expected_end:.0f} s (container declares "
                f"{total / fps:.0f} s): the file is truncated or corrupt. Delete it "
                f"and re-download; analysing {decoded_until - start_sec:.0f} s of a "
                f"{wanted:.0f} s window would silently give a wrong answer.")

    return ExtractedFrames(frames=frames, timestamps=timestamps, fps=fps,
                           total_frames=total, decoded_until_s=decoded_until,
                           expected_until_s=expected_end)
