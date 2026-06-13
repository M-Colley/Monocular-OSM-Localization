"""Scene-text extraction from video frames (OCR).

Reads text off sampled frames — shop/POI/building names, direction
signs — so the pipeline can geocode it into an *absolute* position
anchor (see :mod:`text_anchor`). This is the only channel that injects
absolute geographic information from the video; everything else is
relative shape matching, which is why it's the lever against the VO
drift that otherwise floats the whole solution (see the README's
"Why re-ranking has a ceiling").

OCR is slow, so results are cached to JSON keyed by the sampling
parameters. The heavy dependency (``easyocr``) is imported lazily and
can be injected for tests via ``ocr_reader``.

Resolution note: at the 720p of the reference clip, street-name plates
are not legible, but large signage (``Sedelhöfe``, ``Haus der
Wirtschaft``, ``Polizeipräsidium``) is — and those geocode to useful
anchors. A true-4K source additionally exposes street-name plates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class SceneText:
    """One OCR detection: recovered ``text`` with ``confidence`` in
    [0, 1] read from the frame at ``t_sec`` seconds."""
    text: str
    confidence: float
    t_sec: float


class OcrReader(Protocol):
    """Minimal contract matching ``easyocr.Reader``.

    ``readtext`` returns a list of ``(bbox, text, confidence)`` for one
    image (HxWx3 BGR uint8). Tests inject a stand-in; production wraps
    ``easyocr.Reader([...]).readtext``.
    """

    def readtext(self, image) -> list:  # pragma: no cover - structural
        ...


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_signature(
    sample_interval_sec: float,
    start_sec: float,
    end_sec: float | None,
    languages: tuple[str, ...],
    min_confidence: float,
    min_len: int,
) -> dict:
    return {
        "sample_interval_sec": sample_interval_sec,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "languages": list(languages),
        "min_confidence": min_confidence,
        "min_len": min_len,
    }


def _load_cache(cache_path: Path, sig: dict) -> list[SceneText] | None:
    if not cache_path.exists():
        return None
    try:
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if blob.get("signature") != sig:
        return None
    return [SceneText(**d) for d in blob.get("detections", [])]


def _save_cache(cache_path: Path, sig: dict, detections: list[SceneText]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"signature": sig, "detections": [asdict(d) for d in detections]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def _default_reader(languages: tuple[str, ...], use_gpu: bool) -> OcrReader:
    import easyocr  # lazy: heavy import + model download on first use

    return easyocr.Reader(list(languages), gpu=use_gpu, verbose=False)


def extract_scene_text(
    video_path: Path,
    *,
    sample_interval_sec: float = 6.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    languages: tuple[str, ...] = ("de", "en"),
    min_confidence: float = 0.3,
    min_len: int = 3,
    cache_path: Path | None = None,
    ocr_reader: OcrReader | None = None,
    use_gpu: bool = True,
    frame_reader: Callable[[Path, float, float | None, float], list] | None = None,
) -> list[SceneText]:
    """OCR text off frames of ``video_path`` every ``sample_interval_sec``.

    Returns detections with ``confidence >= min_confidence`` and at least
    ``min_len`` non-space characters, in time order. Results are cached
    to ``cache_path`` keyed by the sampling parameters; a matching cache
    short-circuits the (slow) OCR entirely.

    ``ocr_reader`` / ``frame_reader`` are injection points for tests so
    no real easyocr model or video decode is needed.
    """
    video_path = Path(video_path)
    sig = _cache_signature(
        sample_interval_sec, start_sec, end_sec, tuple(languages),
        min_confidence, min_len,
    )
    if cache_path is not None:
        cached = _load_cache(Path(cache_path), sig)
        if cached is not None:
            return cached

    frames = (frame_reader or _sample_frames)(
        video_path, start_sec, end_sec, sample_interval_sec
    )
    reader = ocr_reader or _default_reader(tuple(languages), use_gpu)

    detections: list[SceneText] = []
    for t_sec, image in frames:
        for item in reader.readtext(image):
            # easyocr returns (bbox, text, confidence).
            _bbox, text, conf = item
            text = str(text).strip()
            if float(conf) >= min_confidence and len(text) >= min_len:
                detections.append(SceneText(text=text, confidence=float(conf),
                                            t_sec=float(t_sec)))

    if cache_path is not None:
        _save_cache(Path(cache_path), sig, detections)
    return detections


def _sample_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float | None,
    interval_sec: float,
) -> list[tuple[float, "object"]]:
    """Yield ``(t_sec, frame_bgr)`` sampled every ``interval_sec``.

    Native resolution (no downscale) — text legibility is the whole
    point, and frames are read sparsely so memory isn't a concern.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = int(end_sec * fps) if end_sec is not None else total
    step = max(1, int(interval_sec * fps))
    out: list[tuple[float, object]] = []
    try:
        idx = int(start_sec * fps)
        while idx < end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            out.append((idx / fps, frame))
            idx += step
    finally:
        cap.release()
    return out
