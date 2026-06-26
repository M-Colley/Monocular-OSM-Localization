"""Visualize the Ulm OrienterNet result + the FOV calibration that fixed it."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MPD = 111320.0
data = json.load(open("output/orienternet_ulm_london.json"))
ulm = data["Ulm"]

fov = [70, 95, 110, 125, 140, 155]
med = [32.7, 23.3, 17.9, 12.8, 17.3, 83.3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# FOV calibration curve
ax1.plot(fov, med, "-o", color="#534AB7", lw=2, ms=7)
ax1.scatter([125], [12.8], s=180, facecolors="none", edgecolors="#1D9E75", lw=2.5, zorder=5)
ax1.annotate("camera is ~125° wide\n(not the 70° I assumed)", (125, 12.8),
             textcoords="offset points", xytext=(12, 28), fontsize=10, color="#0F6E56")
ax1.axhline(160, color="#D85A30", ls="--", lw=1.2)
ax1.text(72, 168, "shape-matching baseline 160 m", color="#993C1D", fontsize=10)
ax1.set_xlabel("assumed horizontal field-of-view (deg)")
ax1.set_ylabel("Ulm median error (m)")
ax1.set_title("calibrating the camera fixes Ulm")
ax1.grid(alpha=0.3); ax1.set_ylim(0, 175)

# Ulm map: true waypoints + predictions
true = np.array([p["true"] for p in ulm]); pred = np.array([p["pred"] for p in ulm])
errs = np.array([p["err"] for p in ulm])
lat0, lon0 = true[:, 0].mean(), true[:, 1].mean()
def to_m(ll):
    return np.c_[(ll[:, 1] - lon0) * MPD * np.cos(np.radians(lat0)), (ll[:, 0] - lat0) * MPD]
T, P = to_m(true), to_m(pred)
for t, p in zip(T, P):
    ax2.plot([t[0], p[0]], [t[1], p[1]], "-", color="0.7", lw=0.7, zorder=1)
ax2.plot(T[:, 0], T[:, 1], "-", color="#2ca02c", lw=1.4, label="true route (waypoints)", zorder=2)
sc = ax2.scatter(P[:, 0], P[:, 1], c=np.clip(errs, 0, 40), cmap="RdYlGn_r", s=70,
                 edgecolors="k", linewidths=0.5, vmin=0, vmax=40, label="OrienterNet", zorder=3)
e = errs
ax2.set_title(f"Ulm — median {np.median(e):.1f} m   "
              f"≤25 m {100*np.mean(e<=25):.0f}%   (was 160 m)", fontsize=11)
ax2.set_xlabel("East (m)"); ax2.set_ylabel("North (m)")
ax2.axis("equal"); ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04); cb.set_label("error (m)")

plt.tight_layout()
plt.savefig("output/orienternet_ulm.png", dpi=115, bbox_inches="tight")
print("saved output/orienternet_ulm.png")
