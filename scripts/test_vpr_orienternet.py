"""Blind chain: VPR per-frame retrievals as pseudo-GPS -> georeference the VO ->
OrienterNet metric refine. Mirrors the GPS-georeference that hit 19.7 m on Ulm, but
with NO GPS: the anchors are the noisy KartaView+EigenPlaces per-frame retrievals,
robust-fit (RANSAC similarity) to the VO so the VO's smooth shape carries the global
anchor + heading + scale. Measures (a) the georeferenced prior error vs GT and (b) the
OrienterNet-refined error vs GT.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

from src.orienternet_localizer import refine_route

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
VPR = "data/kartaview_ulm/vpr_result.npz"
STRIDE = 3
MPD = 111320.0


def err_m(a, b):
    R = 6371000.0
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def main():
    wps = json.load(open(GT))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    gla = np.array([w["lat"] for w in wps]); glo = np.array([w["lon"] for w in wps])
    lat0, lon0 = gla.mean(), glo.mean(); cl = np.cos(np.radians(lat0))

    def to_local(lat, lon):
        return np.c_[(lon - lon0) * MPD * cl, (lat - lat0) * MPD]

    def to_geo(xy):
        return np.c_[lat0 + xy[:, 1] / MPD, lon0 + xy[:, 0] / (MPD * cl)]

    xz = np.load(NPZ)["xz"]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STRIDE / fps

    d = np.load(VPR)
    vpr_geo = d["est_top1"]                                  # (40,2) lat,lon per query frame
    qts = np.linspace(ts.min(), ts.max(), len(vpr_geo))      # the query times used to build it
    vpr_local = to_local(vpr_geo[:, 0], vpr_geo[:, 1])
    q_idx = np.clip((qts / dt).round().astype(int), 0, len(xz) - 1)
    vo_at_q = xz[q_idx]

    # robust VO->VPR similarity (reject the ~40% gross VPR outliers)
    model, inl = ransac((vo_at_q, vpr_local), SimilarityTransform,
                        min_samples=3, residual_threshold=250.0, max_trials=2000)
    print(f"RANSAC georeference: {inl.sum()}/{len(inl)} inliers, scale={model.scale:.2f}")

    vo_geo = to_geo(model(xz))                               # georeferenced VO -> lat/lon
    # prior error vs GT (interp GT at every VO time)
    vt = np.arange(len(xz)) * dt
    gt_at_vo = np.c_[np.interp(vt, ts, gla), np.interp(vt, ts, glo)]
    prior_err = np.array([err_m(vo_geo[i], gt_at_vo[i]) for i in range(0, len(xz), 20)])
    print(f"VPR-georeferenced PRIOR error vs GT: median {np.median(prior_err):.0f} m  "
          f"(start {err_m(vo_geo[0], np.array([gla[0], glo[0]])):.0f} m)")

    # OrienterNet refine on a few central windows, prior from the georeferenced VO
    print("\nOrienterNet refine (prior = VPR-georeferenced VO):")
    errs = []
    for frac in (0.35, 0.5, 0.65):
        c = int(frac * len(xz))
        win = [i for i in range(c - 60, c + 61, 15) if 0 <= i < len(xz)]
        prior_ll = vo_geo[win]
        frames = []
        for i in win:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * STRIDE))
            ok, bgr = cap.read()
            frames.append(bgr if ok else frames[-1])
        refined = refine_route(frames, prior_ll, None, tile_m=200.0, gravity=(0.0, -4.0))
        gt_c = np.array([np.interp(c * dt, ts, gla), np.interp(c * dt, ts, glo)])
        pe = err_m(prior_ll[len(prior_ll) // 2], gt_c)
        if refined is None:
            print(f"  t={c*dt:.0f}s prior {pe:.0f} m -> OrienterNet unavailable"); continue
        re = err_m(refined[len(refined) // 2], gt_c)
        errs.append(re)
        print(f"  t={c*dt:>4.0f}s  prior {pe:5.0f} m  ->  OrienterNet {re:5.0f} m")
    cap.release()
    if errs:
        print(f"\nBLIND VPR->OrienterNet median: {np.median(errs):.0f} m   "
              f"(vs blind shape+ON 673 m; GPS-georeferenced ON was 19.7 m)")


if __name__ == "__main__":
    main()
