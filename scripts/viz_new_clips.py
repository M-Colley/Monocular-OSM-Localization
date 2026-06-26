"""Map the localization hypotheses for the two new clips (Ulm, Erbach).

Plots the OSM road network with the top-5 candidate start hypotheses so the
coordinates can be eyeballed against the real drive. The consensus pick is a
star; the hypothesis whose street names match the known location is ringed.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
from pyproj import Transformer

CLIPS = [
    ("Ulm  -  'Mit dem Auto durch Ulm Innenstadt'",
     "data/local-e09f1da470f6-ulm-innenstadt-ulm-germany/Ulm_Germany.graphml",
     "output/local-e09f1da470f6-ulm-innenstadt-ulm-germany/result.json", 2),
    ("Erbach an der Donau  -  'Erbach - Ersingen - ... - Ehingen'",
     "data/local-d8ccdbb9544b-erbach-erbach-an-der-donau-germany/"
     "Erbach_an_der_Donau_Germany_around_48.3300_9.8900_7000.graphml",
     "output/local-d8ccdbb9544b-erbach-erbach-an-der-donau-germany/result.json", 4),
]
TF = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    for ax, (title, gpath, rpath, likely) in zip(axes, CLIPS):
        G = ox.load_graphml(gpath)
        for u, v, data in G.edges(data=True):
            if "geometry" in data:
                xs, ys = data["geometry"].xy
            else:
                xs = [float(G.nodes[u]["x"]), float(G.nodes[v]["x"])]
                ys = [float(G.nodes[u]["y"]), float(G.nodes[v]["y"])]
            ax.plot(xs, ys, color="0.82", lw=0.5, zorder=1)

        pos = json.load(open(rpath, encoding="utf-8"))["position"]
        hyps = pos["hypotheses"][:5]
        xs, ys = [], []
        for h in hyps:
            x, y = TF.transform(h["longitude"], h["latitude"])
            xs.append(x); ys.append(y)
            is_pick = h["rank"] == 1
            is_likely = h["rank"] == likely
            ax.scatter([x], [y], s=520 if is_pick else 300,
                       marker="*" if is_pick else "o",
                       c="#d1495b" if is_pick else "#2e86ab",
                       edgecolors="black", linewidths=1.4, zorder=4)
            if is_likely:
                ax.scatter([x], [y], s=900, facecolors="none",
                           edgecolors="#22aa55", linewidths=2.8, zorder=5)
            label = f"H{h['rank']}  {', '.join(h['street_names'][:2])}"
            ax.annotate(label, (x, y), xytext=(9, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold" if (is_pick or is_likely) else "normal",
                        zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85))

        mx = (max(xs) - min(xs)) * 0.25 + 300
        my = (max(ys) - min(ys)) * 0.25 + 300
        ax.set_xlim(min(xs) - mx, max(xs) + mx)
        ax.set_ylim(min(ys) - my, max(ys) + my)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal"); ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="*", color="w", markerfacecolor="#d1495b",
                   markeredgecolor="k", markersize=18, label="consensus pick (H1)"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2e86ab",
                   markeredgecolor="k", markersize=12, label="other hypotheses"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#22aa55", markeredgewidth=2.6, markersize=18,
                   label="best match to the named location"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle("OrienterNet pipeline - candidate start hypotheses for two new clips "
                 "(no GPS used)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = "output/new_clips_hypotheses.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
