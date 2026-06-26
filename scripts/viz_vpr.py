"""Visualize the KartaView+EigenPlaces VPR coarse-prior result on Ulm."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

TF = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
R = 6371000.0


def to_xy(ll):
    x, y = TF.transform(ll[:, 1], ll[:, 0])
    return np.c_[x, y]


def main():
    d = np.load("data/kartaview_ulm/vpr_result.npz")
    qtrue, ref_xy, est1, prior = d["qtrue"], d["ref_xy"], d["est_top1"], d["prior"]
    err_top1 = d["err_top1"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    refs = to_xy(ref_xy); q = to_xy(qtrue); e1 = to_xy(est1)
    pr = to_xy(prior[None])[0]; rc = q.mean(0)
    a = ax[0]
    a.scatter(refs[:, 0], refs[:, 1], s=6, c="0.78", label=f"KartaView refs ({len(refs)})", zorder=1)
    for i in range(len(q)):
        a.plot([q[i, 0], e1[i, 0]], [q[i, 1], e1[i, 1]], color="#e8a25a", lw=0.4, alpha=0.5, zorder=2)
    a.scatter(e1[:, 0], e1[:, 1], s=14, c="#e8862e", label="per-frame top-1 retrieval", zorder=3)
    a.plot(q[:, 0], q[:, 1], "-o", color="#2e86ab", ms=4, lw=1.6, label="true route (GT)", zorder=4)
    a.scatter([pr[0]], [pr[1]], s=480, marker="*", c="#22aa55", edgecolors="k",
              linewidths=1.5, label="VPR prior (median)", zorder=6)
    a.scatter([rc[0]], [rc[1]], s=90, marker="X", c="k", label="route centroid", zorder=6)
    a.set_aspect("equal"); a.axis("off"); a.legend(loc="upper left", fontsize=9)
    a.set_title("KartaView + EigenPlaces VPR on Ulm (blind, no GPS)\n"
                "robust-median prior lands ~53 m from route centroid", fontsize=11)

    b = ax[1]
    xs = np.sort(err_top1); cdf = np.arange(1, len(xs) + 1) / len(xs)
    b.plot(xs, cdf * 100, "-o", color="#2e86ab", ms=3, label="per-frame top-1")
    for thr in (50, 100, 200, 500):
        b.axvline(thr, color="0.85", lw=0.8)
    b.axhline(100 * np.mean(err_top1 <= 500), color="0.7", ls=":")
    b.set_xlabel("per-frame retrieval error (m)"); b.set_ylabel("% of query frames")
    b.set_xlim(0, 1500); b.set_ylim(0, 100); b.grid(alpha=0.3)
    b.set_title("Per-frame retrieval error CDF\n(noisy per frame; the median over the clip is the prior)",
                fontsize=11)
    b.legend(loc="lower right")
    fig.tight_layout()
    out = "output/kartaview_vpr_map.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
