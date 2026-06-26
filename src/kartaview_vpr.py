"""Blind coarse-location prior via Visual Place Recognition on KartaView imagery.

The pipeline's wall is SELECTION: trajectory shape is non-unique, so blind it picks
the wrong neighbourhood (Ulm 4K: 664 m off). This module supplies the missing piece —
a coarse location prior INDEPENDENT of trajectory shape — by retrieving the video's
frames against GPS-tagged street photos from KartaView (open API, NO token needed) with
a place-recognition descriptor (EigenPlaces, ICCV'23). Per-frame retrieval is noisy, but
the robust median over the clip lands ~53 m from the true route on Ulm 4K (vs ~526 m for
chance) — inside OrienterNet's ~80 m refine window. Used to gate OSM enumeration to the
right neighbourhood (see --use-vpr-prior), turning blind selection into a local problem.

KartaView: ``1.0/list/nearby-photos`` gives metadata; the live image is the CDN proxy
``https://cdn.kartaview.org/pr:sharp/<base64url(legacy_storageNN_url)>`` (legacy host 404s).

Heavy/optional: needs requests + a GPU + the EigenPlaces torch.hub weights. Returns
``None`` on any failure so the pipeline degrades to ungated shape-matching.
"""

from __future__ import annotations

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_MEAN = None
_STD = None
_MODEL = None


def _cdn_url(lth_name: str) -> str:
    stg = lth_name.split("/")[0]
    legacy = f"https://{stg}.openstreetcam.org/{lth_name[len(stg) + 1:]}"
    b64 = base64.urlsafe_b64encode(legacy.encode()).decode().rstrip("=")
    return f"https://cdn.kartaview.org/pr:sharp/{b64}"


def _fetch_refs(center, radius_m, cache_dir, cap=1500):
    import requests
    meta = None
    if cache_dir:
        meta = os.path.join(cache_dir, "ref_meta.json")
        if os.path.exists(meta):
            return json.load(open(meta))
    clat, clon = center
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * np.cos(np.radians(clat)))
    sess = requests.Session()
    refs = {}
    grid = [(la, lo) for la in np.arange(clat - dlat, clat + dlat, 0.0028)
            for lo in np.arange(clon - dlon, clon + dlon, 0.0040)]
    for la, lo in grid:
        try:
            r = sess.post("https://api.openstreetcam.org/1.0/list/nearby-photos/",
                          data={"lat": la, "lng": lo, "radius": 200}, timeout=30)
            for it in r.json().get("currentPageItems", []):
                refs[it["id"]] = {"lat": float(it["lat"]), "lon": float(it["lng"]),
                                  "url": _cdn_url(it.get("lth_name") or it["name"])}
        except Exception:
            continue
    refs = [{"id": k, **v} for k, v in refs.items()]
    if len(refs) > cap:
        refs = [refs[i] for i in np.linspace(0, len(refs) - 1, cap).astype(int)]
    if meta:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(refs, open(meta, "w"))
    return refs


def _prep(bgr):
    import cv2
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512))
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


def _embed(device, imgs):
    import torch
    out = []
    for i in range(0, len(imgs), 24):
        batch = torch.stack(imgs[i:i + 24]).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            f = _MODEL(batch).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def _embed_refs(refs, device, cache_dir):
    import cv2
    import requests
    import torch
    raw = None
    img_cache = os.path.join(cache_dir, "ref_imgs.npz") if cache_dir else None
    if img_cache and os.path.exists(img_cache):
        d = np.load(img_cache, allow_pickle=True)
        raw, keep = d["raw"], d["keep"].tolist()
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
        with ThreadPoolExecutor(max_workers=16) as ex:
            for j, a in enumerate(ex.map(fetch, refs)):
                if a is not None:
                    raws.append(cv2.resize(a, (512, 512))); keep.append(j)
        if not raws:
            return None, None
        raw = np.stack(raws)
        if img_cache:
            np.savez(img_cache, raw=raw, keep=np.array(keep))
    emb = _embed(device, [_prep(raw[i]) for i in range(len(raw))])
    ref_xy = np.array([[refs[k]["lat"], refs[k]["lon"]] for k in keep])
    return emb, ref_xy


def kartaview_vpr_prior(frames_bgr, center, radius_m=3000.0, *,
                        cache_dir=None, n_query=30, device=None):
    """Return a ``(lat, lon)`` coarse prior from KartaView VPR, or ``None``.

    ``center`` is a ``(lat, lon)`` seed (e.g. the city centroid); reference photos are
    fetched within ``radius_m``. The prior is the robust median of the per-frame nearest
    KartaView photo's GPS — a shape-independent estimate of where the clip was filmed.
    """
    global _MEAN, _STD, _MODEL
    try:
        import torch
    except Exception:
        return None
    try:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        refs = _fetch_refs(center, radius_m, cache_dir)
        if len(refs) < 30:
            return None
        if _MODEL is None:
            _MODEL = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                                    backbone="ResNet50", fc_output_dim=2048,
                                    verbose=False).to(device).eval()
        ref_emb, ref_xy = _embed_refs(refs, device, cache_dir)
        if ref_emb is None or len(ref_xy) < 30:
            return None
        idx = np.linspace(0, len(frames_bgr) - 1, min(n_query, len(frames_bgr))).astype(int)
        q_emb = _embed(device, [_prep(frames_bgr[i]) for i in idx])
        sims = (q_emb @ ref_emb.T).numpy()
        top1 = sims.argmax(1)
        prior = np.median(ref_xy[top1], axis=0)
        return float(prior[0]), float(prior[1])
    except Exception:
        return None
