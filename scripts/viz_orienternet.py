"""Visualize OrienterNet localization results from output/orienternet_all.json.

Per clip: an overhead map (local metres) of the true track (green) and the
OrienterNet predictions (dots coloured by error, with error vectors), plus
a panel title with median / recall. Saves output/orienternet_<clip>.png and
a combined output/orienternet_summary.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MPD = 111320.0
data = json.load(open("output/orienternet_all.json"))

fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 6))
if len(data) == 1:
    axes = [axes]

for ax, (name, d) in zip(axes, data.items()):
    pts = d["points"]
    if not pts:
        ax.set_title(f"{name}\n(no data)")
        continue
    true = np.array([p["true"] for p in pts])
    pred = np.array([p["pred"] for p in pts])
    errs = np.array([p["err_m"] for p in pts])
    lat0, lon0 = true[:, 0].mean(), true[:, 1].mean()
    def to_m(ll):
        return np.c_[(ll[:, 1] - lon0) * MPD * np.cos(np.radians(lat0)),
                     (ll[:, 0] - lat0) * MPD]
    T, P = to_m(true), to_m(pred)
    # error vectors
    for t, p in zip(T, P):
        ax.plot([t[0], p[0]], [t[1], p[1]], "-", color="0.7", lw=0.6, zorder=1)
    ax.plot(T[:, 0], T[:, 1], "-", color="#2ca02c", lw=1.5, label="true track", zorder=2)
    sc = ax.scatter(P[:, 0], P[:, 1], c=np.clip(errs, 0, 30), cmap="RdYlGn_r",
                    s=42, edgecolors="k", linewidths=0.4, vmin=0, vmax=30,
                    label="OrienterNet pred", zorder=3)
    s = d["summary"]
    ax.set_title(f"{name}\nmedian {s['median']:.1f} m   "
                 f"recall@5m {s['recall5']:.0f}%   @10m {s['recall10']:.0f}%", fontsize=11)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("error (m)")

plt.tight_layout()
plt.savefig("output/orienternet_summary.png", dpi=110, bbox_inches="tight")
print("saved output/orienternet_summary.png")

# Error CDF across all clips
plt.figure(figsize=(7, 5))
for name, d in data.items():
    e = np.sort([p["err_m"] for p in d["points"]]) if d["points"] else np.array([])
    if len(e):
        plt.plot(e, 100 * np.arange(1, len(e) + 1) / len(e),
                 label=f"{name} (med {np.median(e):.1f} m)", lw=2)
plt.axvline(5, color="0.6", ls="--", lw=1); plt.axvline(50, color="r", ls=":", lw=1)
plt.text(5.3, 8, "5 m", color="0.4"); plt.text(51, 8, "50 m target", color="r")
plt.xscale("log"); plt.xlabel("localization error (m, log)"); plt.ylabel("% of frames ≤ x")
plt.title("OrienterNet sequential fusion — error CDF"); plt.grid(alpha=0.3, which="both")
plt.legend(fontsize=8); plt.xlim(0.5, 1000)
plt.savefig("output/orienternet_cdf.png", dpi=110, bbox_inches="tight")
print("saved output/orienternet_cdf.png")
