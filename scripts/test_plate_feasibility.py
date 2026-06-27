"""Feasibility test for the license-plate district anchor.

Can we read German registration-district PREFIXES off the Ulm dashcam? We sample
high-res frames, OCR them, and regex for German plate patterns. We only ever
extract the leading district code (1-3 letters) -- never a full plate number.

German plate format: <district 1-3 letters> <1-2 letters> <1-4 digits>, e.g.
"UL AB 1234" (UL = Ulm). We tally the district prefixes seen.
"""

from __future__ import annotations

import glob
import re
from collections import Counter

import cv2
import numpy as np

FRAMES = "C:/Users/localadmin/AppData/Local/Temp/claude/C--Users-localadmin-Documents-Monocular-OSM-Localization/5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad/plate_frames"

# District prefix (1-3 letters) + 1-2 letters + 1-4 digits, tolerant of spaces/dashes.
PLATE_RE = re.compile(r"\b([A-ZÄÖÜ]{1,3})[\s\-·.]{0,2}([A-ZÄÖÜ]{1,2})[\s\-·.]{0,2}(\d{1,4})\b")
# A looser prefix-only fallback: a short all-caps token that could be a district code.
PREFIX_RE = re.compile(r"\b([A-ZÄÖÜ]{1,3})\b")


def main():
    import easyocr
    reader = easyocr.Reader(["de", "en"], gpu=True, verbose=False)
    frames = sorted(glob.glob(f"{FRAMES}/*.jpg"))
    print(f"{len(frames)} frames", flush=True)

    plate_hits: Counter = Counter()
    all_texts = []
    for fp in frames:
        img = cv2.imread(fp)
        # OCR full frame; plates are small so allow low text threshold
        res = reader.readtext(img, detail=1, paragraph=False)
        for _box, text, conf in res:
            if conf < 0.3:
                continue
            up = text.upper().replace("O", "0") if any(c.isdigit() for c in text) else text.upper()
            t = text.upper().strip()
            all_texts.append((t, round(float(conf), 2)))
            m = PLATE_RE.search(t)
            if m:
                plate_hits[m.group(1)] += 1
    print("\n=== full German-plate matches (district prefix counted) ===")
    for code, n in plate_hits.most_common():
        print(f"  {code}: {n}")
    print("\n=== sample of all OCR tokens (to gauge plate readability) ===")
    for t, c in all_texts[:60]:
        print(f"  {c}  {t!r}")


if __name__ == "__main__":
    main()
