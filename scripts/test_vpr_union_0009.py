"""Offline A/B for KITTI 0009: Mapillary-only vs Mapillary+KartaView union.

0009's anchored start is stuck at ~301 m because the Mapillary refs are 10x
thinner near the route start than elsewhere (2 refs <=50 m vs 30-41 <=100 m
later), so retrieval mass drags the start-region robust centre downstream.
Panoramax has ZERO images there; KartaView has 23 within 250 m — the only
tokenless densification available. This measures whether adding KartaView
refs to the retrieval pool actually fixes the START-centre before any
production change is made.

    python scripts/test_vpr_union_0009.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.kartaview_vpr as kv  # noqa: E402
from src.kartaview_vpr import _fetch_refs, _robust_center  # noqa: E402

MPD = 111320.0
VIDEO = ROOT / "data/kitti/drive_0009.mp4"
MLY_DIR = ROOT / "data/local-05c0f063c75b-drive-0009-karlsruhe-germany/mapillary"
KV_DIR = ROOT / "data/local-05c0f063c75b-drive-0009-karlsruhe-germany/kartaview"
GT = ROOT / "ground_truth/kitti_drive_0009.json"
CENTER = (49.009340, 8.439418)      # the clip's --osm-around centre
RADIUS = 800.0


def _err_m(a, b):
    return float(np.hypot((a[0] - b[0]) * MPD,
                          (a[1] - b[1]) * MPD * np.cos(np.radians(b[0]))))


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    kv._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if kv._MODEL is None:
        kv._MODEL = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                                   verbose=False).to(device).eval()

    mly_refs = json.load(open(MLY_DIR / "mly_ref_meta.json"))["refs"]
    mly_emb, mly_ll = kv._embed_refs(mly_refs, device, str(MLY_DIR),
                                     model_name="megaloc")
    kv_refs = _fetch_refs(CENTER, RADIUS, str(KV_DIR))
    print(f"mapillary refs: {len(mly_ll)}   kartaview refs fetched: {len(kv_refs)}")
    kv_emb, kv_ll = kv._embed_refs(kv_refs, device, str(KV_DIR),
                                   model_name="megaloc")
    if kv_emb is None or len(kv_ll) == 0:
        print("kartaview embedding failed; aborting")
        return
    print(f"kartaview refs embedded: {len(kv_ll)}")

    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, n - 1, 80).astype(int)
    imgs = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            imgs.append(kv._prep(bgr))
    cap.release()
    q = kv._embed(device, imgs)

    wps = json.load(open(GT))["waypoints"]
    gt_start = (wps[0]["lat"], wps[0]["lon"])
    k = max(5, len(imgs) // 16)
    dt = 47.0 / max(len(imgs) - 1, 1)

    def start_centre(emb, ll, label):
        sims = (q @ emb.T).numpy()
        ll = np.asarray(ll, dtype=np.float64)
        for mode in ("argmax", "viterbi"):
            if mode == "argmax":
                pick = sims.argmax(1)
            else:
                pick = kv._viterbi_decode(sims, ll, dt)
            track = ll[pick]
            w = sims[np.arange(len(pick)), pick]
            sc = _robust_center(track[:k], w[:k])
            print(f"  {label:22s} {mode:8s} START-centre err "
                  f"{_err_m(sc, gt_start):7.1f} m")

    start_centre(mly_emb, mly_ll, "mapillary-only")
    import torch as _t
    u_emb = _t.cat([mly_emb, kv_emb])
    u_ll = np.vstack([np.asarray(mly_ll, float), np.asarray(kv_ll, float)])
    start_centre(u_emb, u_ll, "mapillary+kartaview")


if __name__ == "__main__":
    main()
