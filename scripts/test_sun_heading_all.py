"""Re-check the sun-heading capability across EVERY video we have."""

from __future__ import annotations

import cv2
import numpy as np

from src.sun_heading import estimate_heading

SP = ("C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
      "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
      "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad")

VIDEOS = [
    ("Ulm 4K (ULl8s4qydrk)", "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4", (48.40, 9.99)),
    ("Ulm Innenstadt (aQi60unoOKw)", f"{SP}/ulm_innenstadt.mp4", (48.40, 9.99)),
    ("Erbach Garmin (uKbCXuxPnZ8)", f"{SP}/erbach.mp4", (48.33, 9.89)),
    ("London (T4wTL3LpLqU)", "data/london_T4wTL3LpLqU/input.mp4", (51.50, -0.13)),
    ("London 2", "data/local-73200bdd8068-input-london-uk/input.mp4", (51.50, -0.13)),
    ("comma2k19 (route_148)", "data/comma/route_148.mp4", (37.74, -122.45)),
    ("KITTI drive_0009", "data/kitti/drive_0009.mp4", (49.01, 8.40)),
    ("KITTI drive_0033", "data/kitti/drive_0033.mp4", (49.01, 8.40)),
]


def main():
    print(f"{'clip':30s} {'sun heading':>12s}   detail")
    print("-" * 78)
    for name, path, center in VIDEOS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"{name:30s} {'-':>12s}   (file missing)"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
        cap.release()
        times = list(np.linspace(5, max(15, dur - 5), 30))
        r = estimate_heading(path, center, times)
        if r and r.get("available"):
            print(f"{name:30s} {r['median_heading']:>9.0f} deg   "
                  f"via {r['source']}, {r['n_used']} sun-frames, conf {r['confidence']:.2f}, "
                  f"@ {r['capture_utc'][:19]}Z", flush=True)
        else:
            print(f"{name:30s} {'no':>12s}   {(r or {}).get('reason', 'n/a')}", flush=True)


if __name__ == "__main__":
    main()
