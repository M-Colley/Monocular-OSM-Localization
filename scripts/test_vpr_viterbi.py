"""Offline A/B: per-frame argmax VPR track vs a Viterbi sequence decode.

Every placement gate this week exists to survive confident-but-wrong
per-frame retrievals. A continuity-constrained decode (states = cached refs,
transition cost = distance beyond what the vehicle can plausibly drive
between query frames) attacks that noise at the SOURCE. This scores both
tracks against interpolated GT for every clip with a warm Mapillary cache:

  - per-frame error (median / p90)
  - START-region robust-centre error (what the anchor placement consumes)

    python scripts/test_vpr_viterbi.py

Uses the cached refs + ref embeddings (no token, no downloads); only the
80 query frames are embedded fresh (seconds on GPU).
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
from src.kartaview_vpr import _robust_center  # noqa: E402

MPD = 111320.0

CLIPS = [
    ("Ulm 4K", "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input_4k.webm",
     "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/mapillary",
     "ground_truth/ulm_ULl8s4qydrk.json", None),
    ("KITTI 0009", "data/kitti/drive_0009.mp4",
     "data/local-05c0f063c75b-drive-0009-karlsruhe-germany/mapillary",
     "ground_truth/kitti_drive_0009.json", 47.0),
    ("KITTI 0033", "data/kitti/drive_0033.mp4",
     "data/local-36a50c34107a-drive-0033-karlsruhe-germany/mapillary",
     "ground_truth/kitti_drive_0033.json", 166.0),
    ("comma", "data/comma/route_148.mp4",
     "data/local-88d9fe89bc4d-route-148-san-francisco-california-usa/mapillary",
     "ground_truth/comma_148.json", 240.0),
    ("London", "data/london_T4wTL3LpLqU/input.mp4",
     "data/local-73200bdd8068-input-london-uk/mapillary",
     "ground_truth/london_T4wTL3LpLqU.json", 295.0),
]

N_QUERY = 80
SPEED_FREE_MPS = 40.0      # transition free radius grows with gap duration
TRANS_ALPHA = 0.02         # similarity-points penalty per metre beyond free


def _dist_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise metres between (N,2) and (M,2) lat/lon."""
    lat0 = float(np.mean(a[:, 0]))
    ax = np.column_stack([a[:, 1] * MPD * np.cos(np.radians(lat0)),
                          a[:, 0] * MPD])
    bx = np.column_stack([b[:, 1] * MPD * np.cos(np.radians(lat0)),
                          b[:, 0] * MPD])
    return np.linalg.norm(ax[:, None, :] - bx[None, :, :], axis=2)


def _viterbi(sims: np.ndarray, ref_ll: np.ndarray, dt_s: float) -> np.ndarray:
    """Continuity-constrained state sequence over the reference set."""
    n_q, n_r = sims.shape
    d = _dist_m(ref_ll, ref_ll)
    free = 30.0 + SPEED_FREE_MPS * dt_s
    trans = -TRANS_ALPHA * np.maximum(0.0, d - free)     # (from, to)
    score = sims[0].copy()
    back = np.zeros((n_q, n_r), dtype=np.int32)
    for q in range(1, n_q):
        cand = score[:, None] + trans                    # (from, to)
        back[q] = np.argmax(cand, axis=0)
        score = cand[back[q], np.arange(n_r)] + sims[q]
    path = np.zeros(n_q, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for q in range(n_q - 2, -1, -1):
        path[q] = back[q + 1][path[q + 1]]
    return path


def _gt_at(gt_file: str, t: np.ndarray) -> np.ndarray:
    wps = json.load(open(gt_file))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    la = np.array([w["lat"] for w in wps])
    lo = np.array([w["lon"] for w in wps])
    return np.column_stack([np.interp(t, ts, la), np.interp(t, ts, lo)])


def _err_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lat0 = float(np.mean(b[:, 0]))
    return np.hypot((a[:, 0] - b[:, 0]) * MPD,
                    (a[:, 1] - b[:, 1]) * MPD * np.cos(np.radians(lat0)))


def run_clip(name, video, cache_dir, gt_file, seg_end):
    import torch
    meta = Path(cache_dir) / "mly_ref_meta.json"
    if not meta.exists():
        print(f"{name:12s} SKIP (no warm cache)")
        return None
    refs = json.load(open(meta))["refs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kv._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    kv._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if kv._MODEL is None:
        kv._MODEL = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                                   verbose=False).to(device).eval()
    ref_emb, ref_xy = kv._embed_refs(refs, device, str(cache_dir),
                                     model_name="megaloc")
    if ref_emb is None:
        print(f"{name:12s} SKIP (no cached embeddings)")
        return None
    ref_ll = np.asarray(ref_xy, dtype=np.float64)

    cap = cv2.VideoCapture(str(ROOT / video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(n_frames / fps, seg_end) if seg_end else n_frames / fps
    t = np.linspace(0.0, end * 0.999, N_QUERY)
    imgs = []
    for ti in t:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ti) * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            break
        imgs.append(kv._prep(bgr))
    cap.release()
    t = t[: len(imgs)]
    q_emb = kv._embed(device, imgs)
    sims = (q_emb @ ref_emb.T).numpy()

    gt = _gt_at(str(ROOT / gt_file), t)
    base_ll = ref_ll[sims.argmax(1)]
    vit_ll = ref_ll[_viterbi(sims, ref_ll, float(t[1] - t[0]))]

    eb, ev = _err_m(base_ll, gt), _err_m(vit_ll, gt)
    k = max(5, len(t) // 16)
    sc_b = _robust_center(base_ll[:k], sims.max(1)[:k])
    sc_v = _robust_center(vit_ll[:k], sims.max(1)[:k])
    s_b = _err_m(np.array([sc_b]), gt[:1])[0]
    s_v = _err_m(np.array([sc_v]), gt[:1])[0]
    print(f"{name:12s} per-frame median {np.median(eb):6.1f} -> {np.median(ev):6.1f} m"
          f"   p90 {np.percentile(eb, 90):7.1f} -> {np.percentile(ev, 90):7.1f} m"
          f"   START-centre {s_b:6.1f} -> {s_v:6.1f} m")
    return dict(name=name, base_med=float(np.median(eb)),
                vit_med=float(np.median(ev)), base_start=float(s_b),
                vit_start=float(s_v))


def main():
    rows = [r for c in CLIPS if (r := run_clip(*c)) is not None]
    json.dump(rows, open(ROOT / "output" / "vpr_viterbi_ab.json", "w"), indent=1)
    print("\nsaved output/vpr_viterbi_ab.json")


if __name__ == "__main__":
    main()
