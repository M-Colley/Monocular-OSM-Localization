"""One figure per clip-row: sample frames + the OSM map of where it localized."""

from __future__ import annotations

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
from pyproj import Transformer

SP = ("C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
      "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
      "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad")
TF = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

CLIPS = [
    dict(name="Ulm  ·  youtu.be/aQi60unoOKw",
         video=f"{SP}/ulm_innenstadt.mp4", ts=[12, 120, 220],
         graph="data/local-e09f1da470f6-ulm-innenstadt-ulm-germany/Ulm_Germany.graphml",
         result="output/local-e09f1da470f6-ulm-innenstadt-ulm-germany/result.json",
         likely=2, likely_txt="H2 = historic centre\n(Sedelhofgasse)"),
    dict(name="Erbach a.d. Donau  ·  youtu.be/uKbCXuxPnZ8",
         video=f"{SP}/erbach.mp4", ts=[12, 150, 300],
         graph="data/local-d8ccdbb9544b-erbach-erbach-an-der-donau-germany/"
               "Erbach_an_der_Donau_Germany_around_48.3300_9.8900_7000.graphml",
         result="output/local-d8ccdbb9544b-erbach-erbach-an-der-donau-germany/result.json",
         likely=4, likely_txt="H4 = Erbach town\n(Max-Eyth-Str.)"),
]


def draw_map(ax, clip):
    G = ox.load_graphml(clip["graph"])
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            xs, ys = data["geometry"].xy
        else:
            xs = [float(G.nodes[u]["x"]), float(G.nodes[v]["x"])]
            ys = [float(G.nodes[u]["y"]), float(G.nodes[v]["y"])]
        ax.plot(xs, ys, color="0.82", lw=0.5, zorder=1)
    pos = json.load(open(clip["result"], encoding="utf-8"))["position"]
    xs, ys = [], []
    for h in pos["hypotheses"][:5]:
        x, y = TF.transform(h["longitude"], h["latitude"])
        xs.append(x); ys.append(y)
        pick = h["rank"] == 1
        like = h["rank"] == clip["likely"]
        ax.scatter([x], [y], s=460 if pick else 230, marker="*" if pick else "o",
                   c="#d1495b" if pick else "#2e86ab", edgecolors="black",
                   linewidths=1.3, zorder=4)
        if like:
            ax.scatter([x], [y], s=820, facecolors="none", edgecolors="#22aa55",
                       linewidths=2.6, zorder=5)
        ax.annotate(f"H{h['rank']}", (x, y), xytext=(7, 5),
                    textcoords="offset points", fontsize=9, fontweight="bold", zorder=6)
    mx = (max(xs) - min(xs)) * 0.18 + 350
    my = (max(ys) - min(ys)) * 0.18 + 350
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"where it localized  ·  ⭐=pick  ◯={clip['likely_txt'].splitlines()[0]}",
                 fontsize=9)


def main() -> None:
    fig = plt.figure(figsize=(16, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.5])
    for r, clip in enumerate(CLIPS):
        cap = cv2.VideoCapture(clip["video"])
        for c, t in enumerate(clip["ts"]):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, bgr = cap.read()
            ax = fig.add_subplot(gs[r, c])
            if ok:
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(f"t = {t}s", fontsize=9)
            ax.axis("off")
        cap.release()
        draw_map(fig.add_subplot(gs[r, 3]), clip)
        fig.text(0.012, 0.74 - r * 0.485, clip["name"], rotation=90,
                 va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("The two clips I tested, and where the pipeline placed them "
                 "(no GPS)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96])
    out = "output/new_clips_video_and_location.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
