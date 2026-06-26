"""Contact sheet of the actual frames used for the two new clips."""

from __future__ import annotations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = ("C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
      "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
      "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad")

CLIPS = [
    ("Ulm  ·  'Mit dem Auto durch Ulm Innenstadt'  ·  youtu.be/aQi60unoOKw  ·  720p",
     f"{SP}/ulm_innenstadt.mp4", [12, 80, 150, 220]),
    ("Erbach a.d. Donau  ·  'Erbach–Ersingen–…–Ehingen'  ·  youtu.be/uKbCXuxPnZ8  ·  1080p Garmin",
     f"{SP}/erbach.mp4", [12, 100, 200, 320]),
]


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.2))
    for r, (title, path, ts) in enumerate(CLIPS):
        cap = cv2.VideoCapture(path)
        for c, t in enumerate(ts):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, bgr = cap.read()
            ax = axes[r, c]
            if ok:
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(f"t = {t}s", fontsize=9)
            ax.axis("off")
        cap.release()
        fig.text(0.5, 0.965 - r * 0.49, title, ha="center", va="top",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.28, top=0.9)
    out = "output/new_clips_frames.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
