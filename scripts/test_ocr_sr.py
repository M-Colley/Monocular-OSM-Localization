"""Does super-resolution / sharpening recover more legible street text for OCR?

Compares easyOCR detections on original frames vs an upscaled+sharpened version
(and OpenCV dnn_superres if a model is present), on the clips where OCR currently
under-performs (London 720p, Ulm-Innenstadt). Reports detection counts + the actual
street-name-like tokens recovered, so we can see if SR turns unreadable signage into
geocodable anchors before wiring it into the pipeline.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

SP = ("C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
      "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
      "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad")
CLIPS = [
    ("London 720p", "data/london_T4wTL3LpLqU/input.mp4"),
    ("Ulm Innenstadt", f"{SP}/ulm_innenstadt.mp4"),
]
# street-name-ish tokens: >=4 letters, allow German/English road suffixes
TOK = re.compile(r"^[A-Za-zÄÖÜäöüß]{4,}$")


def upscale_sharpen(bgr, scale=2.5):
    up = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 2.0)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)


def text_tokens(reader, img):
    out = []
    for _, s, c in reader.readtext(img):
        for tok in re.split(r"\s+", s):
            if c > 0.45 and TOK.match(tok):
                out.append((tok, round(float(c), 2)))
    return out


def main():
    import easyocr
    reader = easyocr.Reader(["en", "de"], gpu=True, verbose=False)
    for name, path in CLIPS:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        orig_tok, sr_tok = set(), set()
        no, ns = 0, 0
        for t in np.linspace(5, max(15, dur - 5), 25):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            o = text_tokens(reader, f)
            s = text_tokens(reader, upscale_sharpen(f))
            no += len(o); ns += len(s)
            orig_tok.update(w.lower() for w, _ in o)
            sr_tok.update(w.lower() for w, _ in s)
        cap.release()
        print(f"\n=== {name} ===")
        print(f"  high-conf word detections: original {no}  ->  SR {ns}")
        print(f"  unique street-ish tokens:  original {len(orig_tok)}  ->  SR {len(sr_tok)}")
        gained = sorted(sr_tok - orig_tok)[:25]
        print(f"  recovered ONLY with SR ({len(sr_tok - orig_tok)}): {gained}")


if __name__ == "__main__":
    main()
