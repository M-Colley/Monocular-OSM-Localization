"""Extract a sequence of frames from a video at a fixed temporal stride.

Returned in chronological order. Large sequences spill losslessly to temporary
storage so memory use does not grow with the duration or resolution of a clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
import tempfile
import threading

import cv2
import numpy as np


class FrameExtractionError(RuntimeError):
    pass


class TruncatedVideoError(FrameExtractionError):
    """The video stops decoding long before the window / declared duration."""


class FrameSequence(Sequence):
    """Random-access BGR frames with a bounded resident cache.

    Small sequences stay in RAM. Once the budget is reached, frames are stored
    as lossless PNGs in an owned temporary directory. Slices are lazy views;
    iterating or sampling a long clip never materializes the entire sequence.
    Call ``close()`` to release storage promptly after a pipeline run.
    """

    def __init__(self, memory_mb: float = 512.0):
        if not np.isfinite(memory_mb) or memory_mb <= 0:
            raise ValueError("frame memory budget must be positive and finite")
        self._limit = max(1, int(memory_mb * 1024 ** 2))
        self._frames: list[np.ndarray] = []
        self._resident = 0
        self._count = 0
        self._temp = None
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False

    @property
    def spilled(self) -> bool:
        return self._temp is not None

    @property
    def resident_bytes(self) -> int:
        return self._resident

    def __len__(self):
        return self._count

    def _path(self, index):
        return Path(self._temp.name) / f"{index:08d}.png"

    def _write(self, index, frame):
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise FrameExtractionError("could not encode frame cache")
        self._path(index).write_bytes(encoded.tobytes())

    def append(self, frame):
        if self._closed:
            raise RuntimeError("frame sequence is closed")
        if self._temp is None and self._resident + frame.nbytes > self._limit:
            self._temp = tempfile.TemporaryDirectory(prefix="osm-frames-")
            for i, old in enumerate(self._frames):
                self._write(i, old)
            self._frames.clear()
            self._resident = 0
        if self._temp is None:
            self._frames.append(frame)
            self._resident += frame.nbytes
        else:
            self._write(self._count, frame)
        self._count += 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FrameView(self, range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self._lock:
            if self._closed:
                raise RuntimeError("frame sequence is closed")
            if self._temp is None:
                return self._frames[index]
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]
            # imdecode handles Windows paths containing non-ASCII characters.
            frame = cv2.imdecode(np.frombuffer(self._path(index).read_bytes(), np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                raise FrameExtractionError(f"could not read cached frame {index}")
            while self._cache and self._resident + frame.nbytes > self._limit:
                _, old = self._cache.popitem(last=False)
                self._resident -= old.nbytes
            if frame.nbytes <= self._limit:
                self._cache[index] = frame
                self._resident += frame.nbytes
            return frame

    def close(self):
        with self._lock:
            self._frames.clear()
            self._cache.clear()
            self._resident = 0
            if self._temp is not None:
                self._temp.cleanup()
            self._closed = True


class _FrameView(Sequence):
    def __init__(self, source, indices):
        self._source, self._indices = source, indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FrameView(self._source, self._indices[index])
        return self._source[self._indices[index]]


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
    frames: Sequence[np.ndarray]
    timestamps: list[float]   # seconds from start of video
    fps: float
    total_frames: int
    #: Video time of the last frame actually decoded, and where the analysed
    #: span was expected to end (None when unknowable) - kept for diagnostics.
    decoded_until_s: float | None = None
    expected_until_s: float | None = None
    #: Describes sparse query frames relative to the original dense stride grid.
    sampling: dict | None = None

    def close(self):
        close = getattr(self.frames, "close", None)
        if close is not None:
            close()


def extract_frames(
    video_path: Path,
    *,
    stride: int = 6,
    max_frames: int | None = 400,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    memory_mb: float = 512.0,
    max_width: int | None = None,
    sample_count: int | None = None,
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

    ``sample_count`` retains only the same linspace query frames that would
    have been selected from the dense stride/window/cap sequence. A known
    horizon is required; unknown-length streams retain dense sampling. Sparse
    extraction still decodes the entire window to detect truncated videos.
    """
    video_path = Path(video_path)
    if stride <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("stride and max_frames must be positive")
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be positive")
    if sample_count is not None and sample_count <= 0:
        raise ValueError("sample_count must be positive")
    frames = FrameSequence(memory_mb)
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

    planned_dense_count = None
    keep_indices = None
    if sample_count is not None and end_frame is not None:
        horizon = min(end_frame, total) if total > 0 else end_frame
        planned_dense_count = max(0, (horizon - start_frame + stride - 1) // stride)
        if max_frames is not None:
            planned_dense_count = min(planned_dense_count, max_frames)
        keep_indices = np.linspace(
            0, max(planned_dense_count - 1, 0),
            min(sample_count, planned_dense_count)).astype(int).tolist()
    keep_set = set(keep_indices) if keep_indices is not None else None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    timestamps: list[float] = []
    idx = start_frame
    decode_failed = False
    dense_count = 0
    decoded_until = None
    try:
        while (end_frame is None or idx < end_frame) and (
                keep_set is not None or max_frames is None or dense_count < max_frames):
            ok, frame = cap.read()
            if not ok:
                decode_failed = True
                break
            # Media time of the frame just decoded (POS_MSEC after read()
            # reports its PTS) — exact on variable-frame-rate sources
            # where idx/fps drifts.
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            decoded_until = pos_ms / 1000.0 if pos_ms is not None and pos_ms > 0 else idx / fps
            if ((idx - start_frame) % stride == 0
                    and (max_frames is None or dense_count < max_frames)):
                if keep_set is None or dense_count in keep_set:
                    if max_width is not None and frame.shape[1] > max_width:
                        height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
                        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
                    frames.append(frame)
                    timestamps.append(decoded_until)
                dense_count += 1
            idx += 1
    except BaseException:
        frames.close()
        raise
    finally:
        cap.release()

    if not frames:
        frames.close()
        raise FrameExtractionError(
            f"no frames extracted (start={start_sec}s end={end_sec} stride={stride})"
        )

    # Where should the analysed span have ended? The earlier of the requested
    # window end and the container's declared length (a video genuinely shorter
    # than the window is fine; one that STOPS DECODING early is not).
    horizon = [h for h in (end_sec, total / fps if total > 0 else None) if h is not None]
    expected_end = min(horizon) if horizon else None
    if decode_failed and expected_end is not None:
        wanted = expected_end - start_sec
        missing = expected_end - decoded_until
        if (wanted > 0 and missing > _MIN_MISSING_S
                and missing > _MAX_MISSING_FRACTION * wanted):
            frames.close()
            raise TruncatedVideoError(
                f"{video_path.name} stops decoding at {decoded_until:.0f} s but the "
                f"analysed window runs to {expected_end:.0f} s (container declares "
                f"{total / fps:.0f} s): the file is truncated or corrupt. Delete it "
                f"and re-download; analysing {decoded_until - start_sec:.0f} s of a "
                f"{wanted:.0f} s window would silently give a wrong answer.")

    if keep_set is not None and dense_count != planned_dense_count:
        # A small metadata shortfall can be valid under the truncation guard.
        # Its true dense length changes linspace indices: reconstruct once from
        # the dense sequence so sparse and ordinary queries remain identical.
        frames.close()
        dense = extract_frames(video_path, stride=stride, max_frames=max_frames,
                               start_sec=start_sec, end_sec=end_sec,
                               memory_mb=memory_mb, max_width=max_width)
        selected = FrameSequence(memory_mb)
        try:
            indices = np.linspace(0, len(dense.frames) - 1,
                                  min(sample_count, len(dense.frames))).astype(int).tolist()
            selected_times = []
            for index in indices:
                selected.append(dense.frames[index])
                selected_times.append(dense.timestamps[index])
            return ExtractedFrames(
                selected, selected_times, dense.fps, dense.total_frames,
                dense.decoded_until_s, dense.expected_until_s,
                sampling={"mode": "linspace", "dense_count": len(dense.frames),
                          "dense_indices": indices, "requested_sample_count": sample_count,
                          "sample_count": len(indices), "metadata_shortfall_recovered": True})
        except BaseException:
            selected.close()
            raise
        finally:
            dense.close()

    return ExtractedFrames(frames=frames, timestamps=timestamps, fps=fps,
                           total_frames=total, decoded_until_s=decoded_until,
                           expected_until_s=expected_end,
                           sampling={"mode": "linspace" if keep_set is not None else "dense",
                                     "dense_count": dense_count, "dense_indices": keep_indices,
                                     "requested_sample_count": sample_count,
                                     "sample_count": len(frames)})
