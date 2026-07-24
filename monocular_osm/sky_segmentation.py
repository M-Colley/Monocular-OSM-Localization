"""Learned sky segmentation for skyline extraction (opt-in).

The default :func:`src.tile3d_match.skyline_from_frame` grows sky from the
top edge by Lab-colour smoothness — cheap and dependency-free, but fragile
under night / glare / heavy occlusion and where the sky is not markedly
smoother than the facades. This module offers a robust alternative: read
the "sky" class from a Cityscapes-pretrained SegFormer and take the
per-column lower boundary of the sky region as the roofline.

Opt-in (needs ``transformers`` + a one-off model download, ~15 MB for the
b0 variant); wired into the tile3d channel via ``--tile3d-skyseg``. The
extractor returns the SAME per-column format as ``skyline_from_frame`` so
the two are interchangeable in the scorer and the placement refiner.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SkySegmenter", "skyline_from_frame_seg"]

# Cityscapes trainId of the "sky" class (road=0 … vegetation=8, terrain=9,
# sky=10, person=11, …). Stable across the standard 19-class SegFormer heads.
_SKY_TRAIN_ID = 10


class SkySegmenter:
    """Lazy Cityscapes-SegFormer sky segmenter (loads weights once)."""

    def __init__(self, model_name: str =
                 "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                 device: str | None = None) -> None:
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                   SegformerImageProcessor)
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = (SegformerForSemanticSegmentation
                      .from_pretrained(model_name).to(self.device).eval())

    def sky_mask(self, frame_bgr: np.ndarray,
                 wh: tuple[int, int]) -> np.ndarray:
        """Boolean (h, w) sky mask at the requested render resolution."""
        import cv2
        w, h = wh
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w, h)), cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits          # (1, 19, h/4, w/4)
        up = self._torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1)[0].cpu().numpy()
        return pred == _SKY_TRAIN_ID


def skyline_from_frame_seg(frame_bgr: np.ndarray, wh: tuple[int, int],
                           segmenter: SkySegmenter) -> np.ndarray:
    """Per-column skyline from a learned sky mask.

    Same contract as :func:`src.tile3d_match.skyline_from_frame`: a finite
    row where a structure edge stops the sky, ``+inf`` where the sky runs to
    the bottom (open horizon), ``NaN`` where the column is unusable (top not
    sky — overpass / canopy / night).
    """
    w, h = wh
    sky = segmenter.sky_mask(frame_bgr, wh)
    rows = np.full(w, np.nan)
    top_sky = sky[0]                       # columns whose top row is sky
    # first non-sky row per column (h if the whole column is sky)
    nonsky = ~sky
    has_struct = nonsky.any(axis=0)
    first = np.where(has_struct, np.argmax(nonsky, axis=0), h)
    for u in range(w):
        if not top_sky[u]:
            continue                       # occluded / night: unusable
        f = int(first[u])
        if f >= h:
            rows[u] = np.inf               # open sky all the way down
        elif f <= 2:
            rows[u] = np.nan               # immediate break at the top
        else:
            rows[u] = float(f)
    return rows
