"""VPR v2: SALAD descriptor + top-k spatial consensus + two-pass heading filter.

Tightens the per-frame KartaView retrieval (v1 EigenPlaces: ~411 m) so the
VO->VPR georeference heading sharpens. Heading is bootstrapped: pass 1 (no
heading) -> rough RANSAC georeference -> per-query absolute driving heading ->
pass 2 keeps only reference photos facing ~the same way (KartaView stores each
photo's heading), killing the opposite-view confusion. Saves est_top1 (pass-2
top-k-consensus positions) to vpr_result.npz for the OrienterNet chain.
"""

from __future__ import annotations

import os

os.environ["XFORMERS_DISABLED"] = "1"   # SALAD's DINOv2 -> vanilla attention (no triton)

import base64
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import torch
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
CACHE = "data/kartaview_ulm2"
STRIDE = 3
N_QUERY = 60
REF_CAP = 1400
MPD = 111320.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def hav(a, b):
    R = 6371000.0
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def cdn(lth):
    stg = lth.split("/")[0]
    legacy = f"https://{stg}.openstreetcam.org/{lth[len(stg)+1:]}"
    return f"https://cdn.kartaview.org/pr:sharp/{base64.urlsafe_b64encode(legacy.encode()).decode().rstrip('=')}"


def build_db(wps):
    mp = f"{CACHE}/ref_meta.json"
    if os.path.exists(mp):
        return json.load(open(mp))
    os.makedirs(CACHE, exist_ok=True)
    lats = [w["lat"] for w in wps]; lons = [w["lon"] for w in wps]
    la0, la1 = min(lats) - 0.008, max(lats) + 0.008
    lo0, lo1 = min(lons) - 0.012, max(lons) + 0.012
    s = requests.Session(); refs = {}
    grid = [(la, lo) for la in np.arange(la0, la1, 0.0025) for lo in np.arange(lo0, lo1, 0.0035)]
    for la, lo in grid:
        try:
            r = s.post("https://api.openstreetcam.org/1.0/list/nearby-photos/",
                       data={"lat": la, "lng": lo, "radius": 180}, timeout=30)
            for it in r.json().get("currentPageItems", []):
                hd = it.get("heading")
                refs[it["id"]] = {"lat": float(it["lat"]), "lon": float(it["lng"]),
                                  "url": cdn(it.get("lth_name") or it["name"]),
                                  "heading": float(hd) if hd not in (None, "", "null") else None}
        except Exception:
            continue
    refs = [{"id": k, **v} for k, v in refs.items()]
    if len(refs) > REF_CAP:
        refs = [refs[i] for i in np.linspace(0, len(refs) - 1, REF_CAP).astype(int)]
    json.dump(refs, open(mp, "w"))
    print(f"  reference DB: {len(refs)} photos (with heading)")
    return refs


def load_salad(device):
    m = torch.hub.load("serizba/salad", "dinov2_salad", trust_repo=True)
    return m.to(device).eval()


def prep(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (322, 322))
    return (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 - MEAN) / STD


@torch.no_grad()
def embed(model, device, imgs):
    out = []
    for i in range(0, len(imgs), 16):
        b = torch.stack(imgs[i:i + 16]).to(device)
        f = model(b).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def get_ref_images(refs):
    ip = f"{CACHE}/ref_imgs.npz"
    if os.path.exists(ip):
        d = np.load(ip, allow_pickle=True); return d["raw"], d["keep"].tolist()
    s = requests.Session()

    def fetch(r):
        try:
            rr = s.get(r["url"], timeout=25)
            if rr.status_code == 200:
                return cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
        return None
    raws, keep = [], []
    print(f"  fetching {len(refs)} ref images...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        for j, a in enumerate(ex.map(fetch, refs)):
            if a is not None:
                raws.append(cv2.resize(a, (322, 322))); keep.append(j)
            if (j + 1) % 250 == 0:
                print(f"    {j+1}/{len(refs)} ({len(keep)} ok)", flush=True)
    raw = np.stack(raws); np.savez(ip, raw=raw, keep=np.array(keep))
    return raw, keep


def consensus(ref_xy, idx_topk):
    """top-k spatial consensus position + spread (confidence)."""
    pts = ref_xy[idx_topk]
    med = np.median(pts, axis=0)
    spread = np.median([hav(p, med) for p in pts])
    return med, spread


def main():
    wps = json.load(open(GT))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    gla = np.array([w["lat"] for w in wps]); glo = np.array([w["lon"] for w in wps])
    lat0, lon0 = gla.mean(), glo.mean(); cl = np.cos(np.radians(lat0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] DB"); refs = build_db(wps)
    raw, keep = get_ref_images(refs)
    refs = [refs[k] for k in keep]
    ref_xy = np.array([[r["lat"], r["lon"]] for r in refs])
    ref_hd = np.array([r["heading"] if r["heading"] is not None else np.nan for r in refs])
    print(f"  {len(refs)} refs, {np.isfinite(ref_hd).mean()*100:.0f}% have heading")
    print("[2] SALAD"); model = load_salad(device)
    ref_emb = embed(model, device, [prep(raw[i]) for i in range(len(raw))])

    print("[3] queries")
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STRIDE / fps
    qts = np.linspace(ts.min(), ts.max(), N_QUERY)
    qimgs, qtrue = [], []
    for t in qts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000); ok, bgr = cap.read()
        if ok:
            qimgs.append(prep(bgr)); qtrue.append([np.interp(t, ts, gla), np.interp(t, ts, glo)])
    cap.release()
    qtrue = np.array(qtrue); q_emb = embed(model, device, qimgs)
    sims = (q_emb @ ref_emb.T).numpy()
    order = np.argsort(-sims, 1)

    def evaluate(name, est):
        e = np.array([hav(est[i], qtrue[i]) for i in range(len(qtrue))])
        prior = np.median(est, 0); rc = qtrue.mean(0)
        print(f"  [{name}] per-frame median {np.median(e):.0f} m | recall@200 {100*np.mean(e<=200):.0f}% "
              f"@500 {100*np.mean(e<=500):.0f}% | AGG prior {hav(prior,rc):.0f} m")
        return e, prior

    # PASS 1: top-k consensus, no heading
    est1 = np.array([consensus(ref_xy, order[i, :5])[0] for i in range(len(qtrue))])
    evaluate("pass1 SALAD top5", est1)

    # bootstrap heading: RANSAC VO->pass1, then per-query absolute driving bearing
    xz = np.load(NPZ)["xz"]
    q_idx = np.clip((qts / dt).round().astype(int), 0, len(xz) - 1)
    est1_loc = np.c_[(est1[:, 1] - lon0) * MPD * cl, (est1[:, 0] - lat0) * MPD]
    model_t, inl = ransac((xz[q_idx], est1_loc), SimilarityTransform,
                          min_samples=3, residual_threshold=250.0, max_trials=2000)
    geo_dir = np.gradient(model_t(xz), axis=0)[q_idx]          # georeferenced VO direction
    q_bearing = (np.degrees(np.arctan2(geo_dir[:, 0], geo_dir[:, 1]))) % 360  # compass

    # PASS 2: keep refs whose heading is within +/-70 deg of the query bearing
    def ang(a, b):
        d = np.abs((a - b + 180) % 360 - 180); return d
    est2 = est1.copy()
    for i in range(len(qtrue)):
        ok_hd = np.where(np.isfinite(ref_hd) & (ang(ref_hd, q_bearing[i]) < 70))[0]
        if len(ok_hd) < 10:
            continue
        sub = ok_hd[np.argsort(-sims[i, ok_hd])[:5]]
        est2[i] = consensus(ref_xy, sub)[0]
    evaluate("pass2 +heading", est2)

    np.savez(f"{CACHE}/vpr_result.npz", qtrue=qtrue, est_top1=est2, ref_xy=ref_xy, qts=qts)
    # also overwrite the chain's expected file
    np.savez("data/kartaview_ulm/vpr_result.npz", qtrue=qtrue, est_top1=est2,
             ref_xy=ref_xy, est5=est2, err_top1=np.array([hav(est2[i], qtrue[i]) for i in range(len(qtrue))]),
             prior=np.median(est2, 0))
    print(f"  saved -> data/kartaview_ulm/vpr_result.npz (for the OrienterNet chain)")


if __name__ == "__main__":
    main()
