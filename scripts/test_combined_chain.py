"""Combined chain: VPR position + an absolute HEADING -> georeference -> OrienterNet.

The VPR->OrienterNet run hit 343 m because the heading came from a noisy RANSAC fit
of the VO to the per-frame VPR points. This isolates the heading: we FIX the global
rotation from a heading source and fit only scale+translation to the (blind) VPR
positions. Compare:
  RANSAC heading  : the old free fit (the 343 m baseline)
  ORACLE heading  : the ideal rotation (stand-in for what a sun/OCR heading provides
                    blind -- sun was validated to ~2 deg on Erbach, OCR on Ulm)
If ORACLE-heading + VPR-position reaches metres, then heading is the only blocker and
a blind heading source closes it.
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


def kabsch_R(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, d]) @ U.T


def fit_st(R, P, Q):
    A = (R @ P.T).T
    keep = np.ones(len(P), bool)
    s, t = 1.0, np.zeros(2)
    for _ in range(4):
        a, q = A[keep], Q[keep]
        ac, qc = a - a.mean(0), q - q.mean(0)
        s = float((ac * qc).sum() / max((ac * ac).sum(), 1e-9))
        t = q.mean(0) - s * a.mean(0)
        res = np.linalg.norm(s * A + t - Q, axis=1)
        keep = res < max(200.0, np.median(res) * 1.5)
    return s, t


def main():
    wps = json.load(open(GT))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    gla = np.array([w["lat"] for w in wps]); glo = np.array([w["lon"] for w in wps])
    lat0, lon0 = gla.mean(), glo.mean(); cl = np.cos(np.radians(lat0))
    to_local = lambda la, lo: np.c_[(lo - lon0) * MPD * cl, (la - lat0) * MPD]
    to_geo = lambda xy: np.c_[lat0 + xy[:, 1] / MPD, lon0 + xy[:, 0] / (MPD * cl)]

    xz = np.load(NPZ)["xz"]
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STRIDE / fps
    d = np.load(VPR)
    vpr_geo = d["est_top1"]
    qts = np.linspace(ts.min(), ts.max(), len(vpr_geo))
    q_idx = np.clip((qts / dt).round().astype(int), 0, len(xz) - 1)
    vo_q = xz[q_idx]
    vpr_loc = to_local(vpr_geo[:, 0], vpr_geo[:, 1])
    gt_loc_q = to_local(np.interp(qts, ts, gla), np.interp(qts, ts, glo))

    # heading sources -> a global rotation
    R_oracle = kabsch_R(vo_q, gt_loc_q)
    tf, _ = ransac((vo_q, vpr_loc), SimilarityTransform, min_samples=3,
                   residual_threshold=250.0, max_trials=2000)
    R_ransac = tf.params[:2, :2] / tf.scale

    vt = np.arange(len(xz)) * dt
    gt_at_vo = np.c_[np.interp(vt, ts, gla), np.interp(vt, ts, glo)]

    for name, R in [("RANSAC heading (old)", R_ransac), ("ORACLE heading (sun/OCR stand-in)", R_oracle)]:
        s, t = fit_st(R, vo_q, vpr_loc)              # position+scale from blind VPR
        geo = to_geo(s * (R @ xz.T).T + t)
        pe = np.array([err_m(geo[i], gt_at_vo[i]) for i in range(0, len(xz), 20)])
        errs = []
        for frac in (0.35, 0.5, 0.65):
            c = int(frac * len(xz))
            win = [i for i in range(c - 60, c + 61, 15) if 0 <= i < len(xz)]
            frames = []
            for i in win:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * STRIDE)); ok, b = cap.read()
                frames.append(b if ok else frames[-1])
            ref = refine_route(frames, geo[win], None, tile_m=180.0, gravity=(0.0, -4.0))
            gt_c = np.array([np.interp(c * dt, ts, gla), np.interp(c * dt, ts, glo)])
            if ref is not None:
                errs.append(err_m(ref[len(ref) // 2], gt_c))
        print(f"  [{name}] prior median {np.median(pe):.0f} m  ->  OrienterNet "
              f"{('%.0f m' % np.median(errs)) if errs else 'NA'}", flush=True)
    cap.release()
    print("  (vs VPR-RANSAC chain 343 m, blind shape+ON 673 m, GPS-georef ON 19.7 m)")


if __name__ == "__main__":
    main()
