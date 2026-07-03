"""LIVE validation of here-sign vs direction-sign classification.

Re-runs OCR on ~20 London 4K frames WITH bounding boxes (via the updated
src.scene_text path), then classifies every detection (text length >= 4)
with the SAME loaded Gemma model through src.vlm_anchor.classify_sign_types.

SUCCESS = 'Holborn' and other green/blue directional content classify
'direction', while genuine street nameplates / shopfronts classify 'here'.
Also probes any Ulm direction signs ('Heidenheim', 'TÖPFER') plus known-good
Ulm street plates if present.

Run:
  C:/Users/localadmin/AppData/Local/Programs/Python/Python312/python.exe \
      scripts/test_sign_type.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.vlm_anchor as va  # noqa: E402
from src.scene_text import _polygon_to_bbox  # noqa: E402

LONDON_VIDEO = ROOT / "data/london_T4wTL3LpLqU/input_4k.webm"
ULM_VIDEO = ROOT / "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input_4k.webm"

# Known off-route direction-sign words (should classify 'direction').
KNOWN_DIRECTION = {"holborn", "heidenheim", "töpfer", "topfer"}


@dataclass
class Det:
    text: str
    confidence: float
    bbox: tuple
    frame_idx: int
    t_sec: float


def sample_frames(path: Path, n: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, n).astype(int)
    else:
        idxs = [i * int(fps * 15) for i in range(n)]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append((int(i) / fps, f))
    cap.release()
    return frames


def ocr_frames(frames, reader, min_conf=0.35, min_len=4):
    """Run easyocr on each (t_sec, frame); return frame list + Det records."""
    frame_arrs = [f for _, f in frames]
    dets = []
    for fi, (t, img) in enumerate(frames):
        for item in reader.readtext(img):
            poly, text, conf = item
            text = str(text).strip()
            if float(conf) >= min_conf and len(text) >= min_len:
                dets.append(Det(text=text, confidence=float(conf),
                                bbox=_polygon_to_bbox(poly),
                                frame_idx=fi, t_sec=t))
    return frame_arrs, dets


def make_reader():
    import easyocr
    # London is English; Ulm German. Load both langs (en must pair with latin).
    try:
        return easyocr.Reader(["en", "de"], gpu=True, verbose=False)
    except Exception:
        return easyocr.Reader(["en", "de"], gpu=False, verbose=False)


def run_clip(name, video, reader, n=20):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    if not video.exists():
        print(f"  MISSING VIDEO: {video}")
        return []
    frames = sample_frames(video, n)
    print(f"  sampled {len(frames)} frames")
    frame_arrs, dets = ocr_frames(frames, reader)
    print(f"  OCR detections (len>=4, conf>=0.35): {len(dets)}")
    if not dets:
        return []
    labels = va.classify_sign_types(frame_arrs, dets)
    print(f"\n  {'TYPE':<10} {'CONF':>5}  TEXT")
    rows = []
    for d, lab in zip(dets, labels):
        rows.append((d.text, lab, d.confidence, d.t_sec))
        print(f"  {lab:<10} {d.confidence:>5.2f}  {d.text!r}  @{d.t_sec:.0f}s")

    # Scorecard on known direction words.
    print("\n  --- known direction-sign check ---")
    for d, lab in zip(dets, labels):
        if d.text.strip().casefold() in KNOWN_DIRECTION:
            ok = "PASS" if lab == "direction" else "MISS"
            print(f"  [{ok}] {d.text!r} -> {lab}")
    return rows


def main():
    print(f"Loading model {va.MODEL_ID} ...")
    va._load()
    print("Model loaded.")
    reader = make_reader()

    run_clip("LONDON 4K", LONDON_VIDEO, reader, n=20)
    run_clip("ULM 4K", ULM_VIDEO, reader, n=20)

    import torch
    va._model = None
    va._proc = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
