"""Plot the OrienterNet-predicted course through Ulm on the OSM street map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyproj import Transformer  # noqa: E402

from src.osm_data import fetch_city_graph  # noqa: E402

MPD = 111320.0
ulm = json.load(open("output/orienternet_ulm_london.json"))["Ulm"]
true = np.array([p["true"] for p in ulm])
pred = np.array([p["pred"] for p in ulm])
errs = np.array([p["err"] for p in ulm])
lat0, lon0 = true[:, 0].mean(), true[:, 1].mean()


def to_m(lat, lon):
    return ((lon - lon0) * MPD * np.cos(np.radians(lat0)), (lat - lat0) * MPD)


road = fetch_city_graph("Ulm", cache_path=Path(
    "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/Ulm_Germany.graphml"))
to_ll = Transformer.from_crs(road.crs, "EPSG:4326", always_xy=True)

fig, ax = plt.subplots(figsize=(9, 9))
# OSM streets (clip to a window around the route).
Tx, Ty = to_m(true[:, 0], true[:, 1])
xlim = (min(Tx) - 250, max(Tx) + 250); ylim = (min(Ty) - 250, max(Ty) + 250)
for poly in road.polylines:
    if len(poly) < 2:
        continue
    lon, lat = to_ll.transform(poly[:, 0], poly[:, 1])
    xs, ys = to_m(np.asarray(lat), np.asarray(lon))
    if np.any((xs > xlim[0]) & (xs < xlim[1]) & (ys > ylim[0]) & (ys < ylim[1])):
        ax.plot(xs, ys, "-", color="0.78", lw=0.8, zorder=1)

# True route + predicted course.
ax.plot(Tx, Ty, "-o", color="#2ca02c", lw=2, ms=7, label="true route (GPS waypoints)", zorder=3)
Px, Py = to_m(pred[:, 0], pred[:, 1])
ax.plot(Px, Py, "-", color="#534AB7", lw=1.2, alpha=0.6, zorder=4)
sc = ax.scatter(Px, Py, c=np.clip(errs, 0, 40), cmap="RdYlGn_r", s=85,
                edgecolors="k", linewidths=0.5, vmin=0, vmax=40, zorder=5,
                label="OrienterNet predicted course")
for (tx, ty), (px, py) in zip(zip(Tx, Ty), zip(Px, Py)):
    ax.plot([tx, px], [ty, py], "-", color="0.5", lw=0.7, zorder=2)

ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
ax.set_title(f"OrienterNet predicted course through Ulm (on OSM streets)\n"
             f"median {np.median(errs):.1f} m   ·   {100*np.mean(errs<=25):.0f}% within 25 m   ·   "
             f"shape-matching was 160 m", fontsize=11)
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("error (m)")
plt.tight_layout()
plt.savefig("output/orienternet_ulm_course.png", dpi=120, bbox_inches="tight")
print("saved output/orienternet_ulm_course.png")
