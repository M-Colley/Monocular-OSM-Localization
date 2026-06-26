"""Diagnostic: would a RELIABLE heading make the heading-filter work?

Reuses the cached SALAD reference DB (data/kartaview_ulm2) and filters references
by the GT driving heading (oracle) instead of the bootstrapped one. If pass2-oracle
beats pass1 a lot, heading-filtering is a real lever and the blocker is just getting
heading (-> pursue sun/shadow heading). If not, drop heading-filtering.
"""

from __future__ import annotations

import os

os.environ["XFORMERS_DISABLED"] = "1"

import json

import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_kartaview_vpr2 import (CACHE, GT, N_QUERY, VIDEO,
                                 consensus, embed, hav, load_salad, prep)


def main():
    wps = json.load(open(GT))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    gla = np.array([w["lat"] for w in wps]); glo = np.array([w["lon"] for w in wps])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = json.load(open(f"{CACHE}/ref_meta.json"))
    d = np.load(f"{CACHE}/ref_imgs.npz", allow_pickle=True)
    raw, keep = d["raw"], d["keep"].tolist()
    refs = [refs[k] for k in keep]
    ref_xy = np.array([[r["lat"], r["lon"]] for r in refs])
    ref_hd = np.array([r["heading"] if r["heading"] is not None else np.nan for r in refs])

    model = load_salad(device)
    ref_emb = embed(model, device, [prep(raw[i]) for i in range(len(raw))])

    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    qts = np.linspace(ts.min(), ts.max(), N_QUERY)
    qimgs, qtrue = [], []
    for t in qts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000); ok, bgr = cap.read()
        if ok:
            qimgs.append(prep(bgr)); qtrue.append([np.interp(t, ts, gla), np.interp(t, ts, glo)])
    cap.release()
    qtrue = np.array(qtrue); q_emb = embed(model, device, qimgs)
    sims = (q_emb @ ref_emb.T).numpy(); order = np.argsort(-sims, 1)

    # ORACLE driving heading from GT (bearing between consecutive interpolated GT fixes)
    dlat = np.gradient(qtrue[:, 0]); dlon = np.gradient(qtrue[:, 1])
    cl = np.cos(np.radians(qtrue[:, 0]))
    q_bearing = np.degrees(np.arctan2(dlon * cl, dlat)) % 360

    def ev(name, est):
        e = np.array([hav(est[i], qtrue[i]) for i in range(len(qtrue))])
        print(f"  [{name}] per-frame median {np.median(e):.0f} m | recall@200 {100*np.mean(e<=200):.0f}% "
              f"@500 {100*np.mean(e<=500):.0f}% | AGG {hav(np.median(est,0), qtrue.mean(0)):.0f} m")

    est1 = np.array([consensus(ref_xy, order[i, :5])[0] for i in range(len(qtrue))])
    ev("pass1 (no heading)", est1)

    def ang(a, b):
        return np.abs((a - b + 180) % 360 - 180)
    for tol in (90, 60, 45):
        est2 = est1.copy()
        used = 0
        for i in range(len(qtrue)):
            ok_hd = np.where(np.isfinite(ref_hd) & (ang(ref_hd, q_bearing[i]) < tol))[0]
            if len(ok_hd) >= 8:
                sub = ok_hd[np.argsort(-sims[i, ok_hd])[:5]]
                est2[i] = consensus(ref_xy, sub)[0]; used += 1
        ev(f"pass2 ORACLE heading +/-{tol} ({used}/{len(qtrue)} filtered)", est2)


if __name__ == "__main__":
    main()
