"""Regional 'where are the clips' map: Ulm-Innenstadt vs Erbach a.d. Donau."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from pyproj import Transformer

GRAPHS = [
    "data/local-e09f1da470f6-ulm-innenstadt-ulm-germany/Ulm_Germany.graphml",
    "data/local-d8ccdbb9544b-erbach-erbach-an-der-donau-germany/"
    "Erbach_an_der_Donau_Germany_around_48.3300_9.8900_7000.graphml",
]
# (label, lat, lon, colour) -- the location each clip most likely shows
PINS = [
    ("Ulm — 'Innenstadt' clip\n(youtu.be/aQi60unoOKw)", 48.40014, 9.98588, "#d1495b"),
    ("Erbach a.d. Donau clip\n(youtu.be/uKbCXuxPnZ8)", 48.32114, 9.88869, "#22aa55"),
]
TF = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
R = 6371000.0


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    for gp in GRAPHS:
        G = ox.load_graphml(gp)
        for u, v, data in G.edges(data=True):
            if "geometry" in data:
                xs, ys = data["geometry"].xy
            else:
                xs = [float(G.nodes[u]["x"]), float(G.nodes[v]["x"])]
                ys = [float(G.nodes[u]["y"]), float(G.nodes[v]["y"])]
            ax.plot(xs, ys, color="0.85", lw=0.4, zorder=1)

    pts = []
    for label, lat, lon, col in PINS:
        x, y = TF.transform(lon, lat)
        pts.append((x, y))
        ax.scatter([x], [y], s=560, marker="*", c=col, edgecolors="black",
                   linewidths=1.6, zorder=5)
        ax.annotate(label, (x, y), xytext=(14, 10), textcoords="offset points",
                    fontsize=11, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.92))

    # distance annotation between the two clips
    (la1, lo1), (la2, lo2) = (PINS[0][1], PINS[0][2]), (PINS[1][1], PINS[1][2])
    dlat, dlon = np.radians(la1 - la2), np.radians(lo1 - lo2)
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(la2)) ** 2 * np.sin(dlon / 2) ** 2
    km = 2 * R * np.arcsin(np.sqrt(h)) / 1000
    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], "--", color="0.5", lw=1.2, zorder=4)
    mx, my = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
    ax.annotate(f"{km:.1f} km apart", (mx, my), fontsize=10, color="0.3",
                ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.85))

    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    pad = 4500
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Where the two clips are — Baden-Württemberg, Alb-Donau-Kreis\n"
                 "(Ulm city centre, and Erbach an der Donau ~12 km SW)", fontsize=13)
    # simple scale bar (2 km)
    x0 = min(xs) - pad + 800; y0 = min(ys) - pad + 800
    ax.plot([x0, x0 + 2000], [y0, y0], "k-", lw=3)
    ax.annotate("2 km", (x0 + 1000, y0 + 250), ha="center", fontsize=9)

    out = "output/new_clips_region.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
