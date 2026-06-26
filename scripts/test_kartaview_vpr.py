"""Visual Place Recognition prior from KartaView street imagery (no API token).

Tests the agents' top idea: a coarse location prior INDEPENDENT of trajectory shape.
Build a reference DB of GPS-tagged KartaView street photos over central Ulm (broader
than the route, so retrieval must actually disambiguate), embed reference + our query
frames with DINOv2, and for each query frame retrieve the nearest reference photo ->
its GPS is the per-frame estimate. Measure error vs the GT waypoints (CDF, recall@X),
plus the aggregated (robust-median) prior the pipeline would consume.

KartaView image URL: 1.0/list/nearby-photos gives `lth_name` (storage path); the live
image is the CDN proxy https://cdn.kartaview.org/pr:sharp/<base64url(legacy_url)>.
"""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import torch

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
CACHE = "data/kartaview_ulm"
N_QUERY = 40
REF_CAP = 1200
R = 6371000.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def haversine(a, b):
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def cdn_url(lth_name: str) -> str:
    stg = lth_name.split("/")[0]
    legacy = f"https://{stg}.openstreetcam.org/{lth_name[len(stg) + 1:]}"
    b64 = base64.urlsafe_b64encode(legacy.encode()).decode().rstrip("=")
    return f"https://cdn.kartaview.org/pr:sharp/{b64}"


def build_reference_db(wps):
    meta_path = f"{CACHE}/ref_meta.json"
    if os.path.exists(meta_path):
        refs = json.load(open(meta_path))
        print(f"  loaded {len(refs)} cached reference photos")
        return refs
    os.makedirs(CACHE, exist_ok=True)
    lats = [w["lat"] for w in wps]; lons = [w["lon"] for w in wps]
    # broaden well beyond the route so retrieval must disambiguate (~+/-900 m)
    la0, la1 = min(lats) - 0.008, max(lats) + 0.008
    lo0, lo1 = min(lons) - 0.012, max(lons) + 0.012
    sess = requests.Session()
    refs = {}
    grid = [(la, lo) for la in np.arange(la0, la1, 0.0025)
            for lo in np.arange(lo0, lo1, 0.0035)]
    print(f"  querying KartaView over {len(grid)} grid points "
          f"({la0:.4f}-{la1:.4f}, {lo0:.4f}-{lo1:.4f})...")
    for i, (la, lo) in enumerate(grid):
        try:
            r = sess.post("https://api.openstreetcam.org/1.0/list/nearby-photos/",
                          data={"lat": la, "lng": lo, "radius": 180}, timeout=30)
            for it in r.json().get("currentPageItems", []):
                refs[it["id"]] = {"lat": float(it["lat"]), "lon": float(it["lng"]),
                                  "url": cdn_url(it.get("lth_name") or it["name"])}
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"    grid {i + 1}/{len(grid)}: {len(refs)} unique photos", flush=True)
    refs = [{"id": k, **v} for k, v in refs.items()]
    if len(refs) > REF_CAP:                       # spatially-uniform subsample
        idx = np.linspace(0, len(refs) - 1, REF_CAP).astype(int)
        refs = [refs[i] for i in idx]
    json.dump(refs, open(meta_path, "w"))
    print(f"  reference DB: {len(refs)} photos -> {meta_path}")
    return refs


def load_model(device):
    # EigenPlaces (ICCV'23): a descriptor trained specifically for cross-city visual
    # place recognition (vs DINOv2-CLS which conflates look-alike streets).
    m = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                       backbone="ResNet50", fc_output_dim=2048, verbose=False)
    return m.to(device).eval()


def prep(bgr):
    # full-frame resize to EigenPlaces' training size (keep side building facades)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512))
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


@torch.no_grad()
def embed(model, device, imgs):
    out = []
    for i in range(0, len(imgs), 24):
        batch = torch.stack(imgs[i:i + 24]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(batch).float()                           # (B, 2048) VPR descriptor
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def embed_reference(model, device, refs):
    # Cache the raw downloaded images (resized) so we can re-embed cheaply with a
    # different descriptor without re-fetching from KartaView.
    img_path = f"{CACHE}/ref_imgs.npz"
    if os.path.exists(img_path):
        d = np.load(img_path, allow_pickle=True)
        raw, keep = d["raw"], d["keep"].tolist()
        print(f"  loaded {len(keep)} cached reference images {raw.shape}")
    else:
        sess = requests.Session()

        def fetch(ref):
            try:
                rr = sess.get(ref["url"], timeout=25)
                if rr.status_code == 200:
                    return cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                return None
            return None

        raws, keep = [], []
        print(f"  fetching {len(refs)} reference images...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            for j, a in enumerate(ex.map(fetch, refs)):
                if a is not None:
                    raws.append(cv2.resize(a, (512, 512))); keep.append(j)
                if (j + 1) % 200 == 0:
                    print(f"    fetched {j + 1}/{len(refs)} ({len(keep)} ok)", flush=True)
        raw = np.stack(raws)
        np.savez(img_path, raw=raw, keep=np.array(keep))
        print(f"  cached {len(keep)} reference images")
    imgs = [prep(raw[i]) for i in range(len(raw))]
    emb = embed(model, device, imgs)
    print(f"  embedded {len(keep)} reference images")
    return emb, keep


def main():
    wps = json.load(open(GT))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    la = np.array([w["lat"] for w in wps]); lo = np.array([w["lon"] for w in wps])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global MEAN, STD
    MEAN, STD = MEAN.to("cpu"), STD.to("cpu")

    print("[1] reference DB"); refs = build_reference_db(wps)
    print("[2] DINOv2"); model = load_model(device)
    ref_emb, keep = embed_reference(model, device, refs)
    refs = [refs[k] for k in keep]
    ref_xy = np.array([[r["lat"], r["lon"]] for r in refs])

    print("[3] query frames")
    cap = cv2.VideoCapture(VIDEO)
    qts = np.linspace(ts.min(), ts.max(), N_QUERY)
    qimgs, qtrue = [], []
    for t in qts:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, bgr = cap.read()
        if not ok:
            continue
        qimgs.append(prep(bgr))
        qtrue.append([np.interp(t, ts, la), np.interp(t, ts, lo)])
    cap.release()
    qtrue = np.array(qtrue)
    q_emb = embed(model, device, qimgs)

    print("[4] retrieve + evaluate")
    sims = (q_emb @ ref_emb.T).numpy()                 # (Nq, Nr)
    top1 = sims.argmax(1)
    topk = np.argsort(-sims, 1)[:, :5]
    err_top1 = np.array([haversine(ref_xy[top1[i]], qtrue[i]) for i in range(len(qtrue))])
    # top-5 robust median position
    est5 = np.array([np.median(ref_xy[topk[i]], axis=0) for i in range(len(qtrue))])
    err_top5 = np.array([haversine(est5[i], qtrue[i]) for i in range(len(qtrue))])

    prior = np.median(ref_xy[top1], axis=0)            # the single coarse prior
    route_centroid = np.array([la.mean(), lo.mean()])
    prior_err = haversine(prior, route_centroid)

    print("\n================ KartaView VPR result (Ulm, blind) ================")
    print(f"  reference photos: {len(refs)}   query frames: {len(qtrue)}")
    for thr in (50, 100, 200, 500):
        print(f"  recall@{thr:>3}m  top1 {100*np.mean(err_top1<=thr):4.0f}%   "
              f"top5med {100*np.mean(err_top5<=thr):4.0f}%")
    print(f"  median per-frame error: top1 {np.median(err_top1):.0f} m   "
          f"top5med {np.median(err_top5):.0f} m")
    print(f"  AGGREGATED PRIOR (median of top1) error vs route centroid: "
          f"{prior_err:.0f} m   [{prior[0]:.5f},{prior[1]:.5f}]")
    print(f"  (vs blind shape-match best-in-pool ~520 m, final 664 m)")

    np.savez(f"{CACHE}/vpr_result.npz", qtrue=qtrue, est_top1=ref_xy[top1],
             est5=est5, err_top1=err_top1, ref_xy=ref_xy, prior=prior)
    print(f"  saved {CACHE}/vpr_result.npz")


if __name__ == "__main__":
    main()
