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
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_MEAN = None
_STD = None
_MODEL = None
_MODEL_NAME = None   # backbone actually resident in _MODEL (survives fallback)


def _cdn_url(lth_name: str) -> str:
    stg = lth_name.split("/")[0]
    legacy = f"https://{stg}.openstreetcam.org/{lth_name[len(stg) + 1:]}"
    b64 = base64.urlsafe_b64encode(legacy.encode()).decode().rstrip("=")
    return f"https://cdn.kartaview.org/pr:sharp/{b64}"


def _fetch_signature(center, radius_m, cap) -> dict:
    """Cache key of a metadata fetch: without it, changing --vpr-search-radius
    or the seed centre silently returns the stale refs (audit kartaview:45)."""
    return {"center": [round(float(center[0]), 4), round(float(center[1]), 4)],
            "radius_m": float(radius_m), "cap": int(cap)}


def _refs_fingerprint(refs) -> str:
    """Content hash of a refs list — ties ref_imgs.npz / embeddings to the
    exact metadata they were built from (audit kartaview:124)."""
    blob = json.dumps([[r["id"], r["lat"], r["lon"]] for r in refs])
    return hashlib.sha1(blob.encode()).hexdigest()


def _fetch_refs(center, radius_m, cache_dir, cap=1500):
    import requests
    sig = _fetch_signature(center, radius_m, cap)
    meta = None
    if cache_dir:
        meta = os.path.join(cache_dir, "ref_meta.json")
        if os.path.exists(meta):
            try:
                blob = json.load(open(meta))
            except Exception:
                blob = None
            # Legacy caches (a bare list) carry no fetch params, so they can't
            # be trusted against the requested (center, radius, cap): refetch.
            if isinstance(blob, dict) and blob.get("signature") == sig:
                return blob["refs"]
    clat, clon = center
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * np.cos(np.radians(clat)))
    sess = requests.Session()
    refs = {}
    # Latitude-aware lon step so query discs tile uniformly at any latitude
    # (a fixed degree step leaves coverage strips near the equator).
    step_lat = 0.0028
    step_lon = step_lat / max(np.cos(np.radians(clat)), 0.2)
    grid = [(la, lo) for la in np.arange(clat - dlat, clat + dlat, step_lat)
            for lo in np.arange(clon - dlon, clon + dlon, step_lon)]

    def query(cell):
        la, lo = cell
        try:
            r = sess.post("https://api.openstreetcam.org/1.0/list/nearby-photos/",
                          data={"lat": la, "lng": lo, "radius": 200}, timeout=30)
            return r.json().get("currentPageItems", [])
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=8) as ex:
        for items in ex.map(query, grid):
            for it in items:
                refs[it["id"]] = {"lat": float(it["lat"]), "lon": float(it["lng"]),
                                  "url": _cdn_url(it.get("lth_name") or it["name"])}
    refs = [{"id": k, **v} for k, v in refs.items()]
    if len(refs) > cap:
        refs = [refs[i] for i in np.linspace(0, len(refs) - 1, cap).astype(int)]
    if meta:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump({"signature": sig, "refs": refs}, open(meta, "w"))
    return refs


def _fetch_refs_mapillary(center, radius_m, cache_dir, cap=1500, token=None):
    """Mapillary street-level refs (Graph API v4) as ``[{id,lat,lon,url}]`` —
    a *much* denser source than KartaView on most areas (validated: a MegaLoc
    prior from these lands 3-31 m from the GT route on all clips, incl. the
    ones KartaView could not cover). Needs a free access token (``MLY_TOKEN``
    env var or ``token=``) unless a warm ref cache covers the query. Returns
    [] with no token and no warm cache / on error.

    No metadata cache: Mapillary thumbnail URLs are signed and expire, so we
    re-query metadata each run (cheap) and rely on the fingerprinted image
    cache (``ref_imgs.npz``) to skip the expensive download+embed on warm runs.
    Refs are sorted by id so that fingerprint is stable across runs.
    """
    # Persistent ref cache (id/lat/lon only — NOT the expiring thumb URLs) so
    # the ref set (hence the prior) is REPRODUCIBLE across runs. Without it,
    # Mapillary returns a slightly different id-set each query, the image-cache
    # fingerprint drifts, and a different subsample is used run-to-run (seen:
    # London prior wandered 91 m vs 356 m). On a warm hit with the embedded-
    # image cache present, we reuse the cached refs and never re-download.
    # Checked BEFORE the token guard: a warm cache needs no API access, so
    # offline regression sweeps run without MLY_TOKEN.
    sig = _fetch_signature(center, radius_m, cap)
    meta = os.path.join(cache_dir, "mly_ref_meta.json") if cache_dir else None
    npz = os.path.join(cache_dir, "ref_imgs.npz") if cache_dir else None
    if meta and os.path.exists(meta) and npz and os.path.exists(npz):
        try:
            blob = json.load(open(meta))
            if isinstance(blob, dict) and blob.get("signature") == sig:
                return blob["refs"]  # url=None; the npz image cache is used
        except Exception:
            pass
    import requests
    token = token or os.environ.get("MLY_TOKEN")
    if not token:
        return []
    clat, clon = center
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * np.cos(np.radians(clat)))
    # Small cells: the Graph API rejects a bbox that covers too many images.
    step_lat = 0.0022
    step_lon = step_lat / max(np.cos(np.radians(clat)), 0.2)
    cells = []
    la = clat - dlat
    while la < clat + dlat:
        lo = clon - dlon
        while lo < clon + dlon:
            cells.append((lo, la, min(lo + step_lon, clon + dlon),
                          min(la + step_lat, clat + dlat)))
            lo += step_lon
        la += step_lat
    if len(cells) > 1200:  # bound API cost on large radii
        cells = [cells[i] for i in np.linspace(0, len(cells) - 1, 1200).astype(int)]
    sess = requests.Session()

    def query(cell):
        w, s, e, n = cell
        try:
            r = sess.get("https://graph.mapillary.com/images",
                         params={"access_token": token,
                                 "fields": "id,geometry,thumb_1024_url",
                                 "bbox": f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}",
                                 "limit": 2000}, timeout=30)
            return r.json().get("data", []) if r.status_code == 200 else []
        except Exception:
            return []

    refs = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for data in ex.map(query, cells):
            for d in data:
                g = d.get("geometry", {}).get("coordinates")
                u = d.get("thumb_1024_url")
                if g and u:
                    refs[d["id"]] = {"lat": float(g[1]), "lon": float(g[0]), "url": u}
    out = [{"id": k, **v} for k, v in sorted(refs.items())]
    if len(out) > cap:
        out = [out[i] for i in np.linspace(0, len(out) - 1, cap).astype(int)]
    if meta and out:
        os.makedirs(cache_dir, exist_ok=True)
        # store WITHOUT urls (they expire); the fingerprinted npz holds pixels
        stable = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"]} for r in out]
        json.dump({"signature": sig, "refs": stable}, open(meta, "w"))
    return out


def _fetch_refs_panoramax(center, radius_m, cache_dir, cap=1500):
    """Panoramax street-level refs (federated open imagery, STAC ``/search``)
    as ``[{id,lat,lon,url}]`` — tokenless and openly licensed; the coverage
    complement to Mapillary (105M+ images across 12 instances by mid-2026,
    strongest in France/EU; probe 2026-07-04: 500+ hits on Ulm, both
    Karlsruhe sites and London, zero on Daly City). Same signed metadata
    cache pattern as the other sources (``pnx_ref_meta.json``), so warm
    reruns are reproducible and offline.
    """
    import requests
    sig = _fetch_signature(center, radius_m, cap)
    meta = os.path.join(cache_dir, "pnx_ref_meta.json") if cache_dir else None
    if meta and os.path.exists(meta):
        try:
            blob = json.load(open(meta))
            if isinstance(blob, dict) and blob.get("signature") == sig:
                return blob["refs"]
        except Exception:
            pass
    clat, clon = center
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * np.cos(np.radians(clat)))
    # The meta-catalog serves at most ~500 features per search and (observed)
    # no next-page link, so tile the disc into sub-bboxes small enough that
    # each stays under that ceiling in dense areas.
    n = max(1, int(np.ceil(2.0 * radius_m / 700.0)))
    las = np.linspace(clat - dlat, clat + dlat, n + 1)
    los = np.linspace(clon - dlon, clon + dlon, n + 1)
    cells = [(los[j], las[i], los[j + 1], las[i + 1])
             for i in range(n) for j in range(n)]
    sess = requests.Session()

    def query(cell):
        try:
            r = sess.get("https://api.panoramax.xyz/api/search",
                         params={"bbox": ",".join(f"{v:.6f}" for v in cell),
                                 "limit": 500}, timeout=30)
            return r.json().get("features", [])
        except Exception:
            return []

    refs = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for feats in ex.map(query, cells):
            for f in feats:
                try:
                    lon, lat = f["geometry"]["coordinates"][:2]
                    assets = f.get("assets", {})
                    a = assets.get("sd") or assets.get("thumb") or assets.get("hd")
                    if not a or not a.get("href"):
                        continue
                    refs[str(f["id"])] = {"lat": float(lat), "lon": float(lon),
                                          "url": a["href"]}
                except Exception:
                    continue
    # Sort by id so the subsample (hence the image-cache fingerprint and the
    # prior) is reproducible run-to-run, like the Mapillary fetcher.
    refs = [{"id": k, **refs[k]} for k in sorted(refs)]
    if len(refs) > cap:
        refs = [refs[i] for i in np.linspace(0, len(refs) - 1, cap).astype(int)]
    if meta and refs:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump({"signature": sig, "refs": refs}, open(meta, "w"))
    return refs


def has_mapillary_cache(cache_dir) -> bool:
    """True when a reusable Mapillary ref cache (metadata + image blob)
    exists, so a warm rerun can skip the MLY_TOKEN requirement entirely.
    (Whether it actually COVERS the query is decided by the signature check
    in :func:`_fetch_refs_mapillary`; a mismatch degrades to no refs.)"""
    return bool(cache_dir) and all(
        os.path.exists(os.path.join(cache_dir, f))
        for f in ("mly_ref_meta.json", "ref_imgs.npz"))


def _fetch_refs_for(source, center, radius_m, cache_dir, cap, token):
    """Dispatch to the requested VPR reference source."""
    if source == "mapillary":
        return _fetch_refs_mapillary(center, radius_m, cache_dir, cap=cap, token=token)
    if source == "panoramax":
        return _fetch_refs_panoramax(center, radius_m, cache_dir, cap=cap)
    return _fetch_refs(center, radius_m, cache_dir, cap=cap)


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


def _load_ref_images(refs, cache_dir):
    """Download (or load cached) reference images. Returns
    ``(raw[N,512,512,3], ref_xy[N,2], fingerprint)`` or ``(None, None, fp)``.

    The npz is SELF-CONTAINED: it stores the kept photos' lat/lon plus a
    fingerprint of the refs list it was built from. A stale npz paired with a
    regenerated ref_meta.json would otherwise label every embedding with an
    arbitrary other photo's GPS (audit kartaview:124) — on mismatch we discard
    it and re-download.
    """
    import cv2
    import requests
    fp = _refs_fingerprint(refs)
    img_cache = os.path.join(cache_dir, "ref_imgs.npz") if cache_dir else None
    if img_cache and os.path.exists(img_cache):
        with np.load(img_cache, allow_pickle=True) as d:
            if ("fingerprint" in d.files and str(d["fingerprint"]) == fp
                    and "ref_xy" in d.files):
                return np.asarray(d["raw"]), np.asarray(d["ref_xy"], float), fp
        # legacy or mismatched cache -> refetch rather than mispair coords
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
        return None, None, fp
    raw = np.stack(raws)
    ref_xy = np.array([[refs[k]["lat"], refs[k]["lon"]] for k in keep], float)
    if img_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(img_cache, raw=raw, keep=np.array(keep), ref_xy=ref_xy,
                 fingerprint=np.array(fp))
    return raw, ref_xy, fp


def _embed_refs(refs, device, cache_dir, model_name="model"):
    import torch
    raw, ref_xy, fp = _load_ref_images(refs, cache_dir)
    if raw is None:
        return None, None
    # Embedding cache keyed by (image fingerprint, model): warm reruns skip
    # the GPU embedding pass entirely.
    emb_cache = (os.path.join(cache_dir, f"ref_emb_{model_name}.npz")
                 if cache_dir else None)
    if emb_cache and os.path.exists(emb_cache):
        with np.load(emb_cache, allow_pickle=True) as d:
            if str(d["fingerprint"]) == fp and len(d["emb"]) == len(ref_xy):
                return torch.from_numpy(np.asarray(d["emb"], np.float32)), ref_xy
    emb = _embed(device, [_prep(raw[i]) for i in range(len(raw))])
    if emb_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(emb_cache, emb=emb.numpy(), fingerprint=np.array(fp))
    return emb, ref_xy


def _geometric_median(pts, weights=None, iters=64, eps=1e-9):
    """Weiszfeld geometric median (robust to outlier matches)."""
    pts = np.asarray(pts, float)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, float)
    x = np.average(pts, axis=0, weights=w)
    for _ in range(iters):
        d = np.maximum(np.linalg.norm(pts - x, axis=1), eps)
        wd = w / d
        x_new = (pts * wd[:, None]).sum(0) / wd.sum()
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
    return x


def _robust_center(latlons, sims):
    """Robust coarse centre from per-frame VPR matches: confidence-thresholded
    (top ~40%), similarity-weighted geometric median, then a spatial-MAD outlier
    rejection + refit. On Ulm this lands 91 m from the GT route (vs 109 m without
    the MAD step, 512 m for a plain median). Returns (lat, lon)."""
    import math as _m
    latlons = np.asarray(latlons, float)
    sims = np.asarray(sims, float)
    keep = sims >= float(np.percentile(sims, 60))
    if int(keep.sum()) < 5:
        keep = np.ones(len(sims), bool)
    pts, wts = latlons[keep], sims[keep]
    # Weiszfeld in a LOCAL METRIC frame: one degree of longitude is cos(lat)
    # times shorter than one of latitude, so a raw-degree median is biased
    # along the east-west axis (tens of metres at 48-60N — audit kartaview:155).
    dm = 111320.0
    lat0 = float(np.mean(pts[:, 0]))
    coslat = _m.cos(_m.radians(lat0))
    xy = np.column_stack([(pts[:, 1]) * dm * coslat, (pts[:, 0]) * dm])
    p = _geometric_median(xy, weights=wts)
    d = np.linalg.norm(xy - p, axis=1)
    mad = float(np.median(np.abs(d - np.median(d)))) + 1e-6
    good = d <= np.median(d) + 2.5 * 1.4826 * mad
    if int(good.sum()) >= 5:
        p = _geometric_median(xy[good], weights=wts[good])
    return float(p[1] / dm), float(p[0] / (dm * coslat))


def _resolve_backbone(model_name, device):
    """Load the retrieval backbone into the module cache ONCE, returning the
    name of the backbone actually resident in ``_MODEL``.

    MegaLoc (2024 SOTA retrieval) >> EigenPlaces, but its hub/weights fetch can
    fail; we fall back to EigenPlaces so the channel still works offline-ish.
    The RESOLVED name is remembered in ``_MODEL_NAME`` so a warm ``_MODEL`` is
    never mislabelled on a later call: the embedding cache is keyed on this
    name, and keying ``"megaloc"`` onto resident EigenPlaces weights would
    silently dot two different embedding spaces and return a wrong prior with
    no error (bug found 2026-07-05).
    """
    global _MODEL, _MODEL_NAME
    import torch
    if _MODEL is not None:
        return _MODEL_NAME or model_name
    resolved = model_name
    if model_name == "megaloc":
        try:
            _MODEL = torch.hub.load("gmberton/MegaLoc", "get_trained_model",
                                    verbose=False).to(device).eval()
        except Exception:
            resolved = "eigenplaces"
    if _MODEL is None:
        _MODEL = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                                backbone="ResNet50", fc_output_dim=2048,
                                verbose=False).to(device).eval()
    _MODEL_NAME = resolved
    return resolved


def _prepare_refs_and_query(frames_bgr, center, radius_m, cache_dir, n_query,
                            device, model_name, source, token, cap):
    """Shared front half of both VPR entrypoints: fetch references, load/cache
    the backbone, embed the references + the ``n_query`` sampled query frames.

    Returns ``(idx, sims[n_query, n_ref], ref_xy[n_ref, 2])`` or ``None`` when
    references are unavailable / too sparse. Callers own the tail (single
    robust centre vs. per-frame track).
    """
    global _MEAN, _STD
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    refs = _fetch_refs_for(source, center, radius_m, cache_dir, cap, token)
    if len(refs) < 30:
        return None
    resolved = _resolve_backbone(model_name, device)
    ref_emb, ref_xy = _embed_refs(refs, device, cache_dir, model_name=resolved)
    if ref_emb is None or len(ref_xy) < 30:
        return None
    idx = np.linspace(0, len(frames_bgr) - 1,
                      min(n_query, len(frames_bgr))).astype(int)
    q_emb = _embed(device, [_prep(frames_bgr[i]) for i in idx])
    sims = (q_emb @ ref_emb.T).numpy()
    return idx, sims, np.asarray(ref_xy, dtype=float)


def kartaview_vpr_prior(frames_bgr, center, radius_m=3000.0, *,
                        cache_dir=None, n_query=40, device=None,
                        model_name="megaloc", source="kartaview", token=None,
                        cap=1500):
    """Return a ``(lat, lon)`` coarse prior from street-level VPR, or ``None``.

    ``center`` is a ``(lat, lon)`` seed (e.g. the city centroid); reference photos are
    fetched within ``radius_m`` from ``source`` (``"kartaview"`` or ``"mapillary"``).
    The prior is the robust median of the per-frame nearest photo's GPS — a
    shape-independent estimate of where the clip was filmed.
    """
    try:
        import torch  # noqa: F401 - availability guard; helper re-imports
    except Exception:
        return None
    try:
        out = _prepare_refs_and_query(frames_bgr, center, radius_m, cache_dir,
                                      n_query, device, model_name, source,
                                      token, cap)
        if out is None:
            return None
        _idx, sims, ref_xy = out
        top1 = sims.argmax(1)
        maxsim = sims.max(1)
        return _robust_center(ref_xy[top1], maxsim)
    except Exception:
        return None


def _viterbi_decode(sims: np.ndarray, ref_latlon: np.ndarray,
                    dt_s: float) -> np.ndarray:
    """Continuity-constrained reference sequence for the query frames.

    Per-frame argmax retrieval produces the confident-but-wrong matches that
    every downstream gate exists to survive; a Viterbi decode with a
    transition penalty per metre beyond what a vehicle plausibly drives
    between query frames kills them at the SOURCE. Offline A/B
    (scripts/test_vpr_viterbi.py, all 5 GT clips): per-frame median improves
    everywhere (London 127->52 m, KITTI-0033 505->212 m), the p90 outlier
    tail collapses ~5-8x (Ulm 1198->208 m), the start-region robust centre
    is unchanged. Returns the per-query reference indices.
    """
    n_q, n_r = sims.shape
    lat0 = float(np.mean(ref_latlon[:, 0]))
    xy = np.column_stack([
        ref_latlon[:, 1] * 111320.0 * np.cos(np.radians(lat0)),
        ref_latlon[:, 0] * 111320.0])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    free = 30.0 + 40.0 * max(dt_s, 0.5)          # generous urban speed cap
    trans = -0.02 * np.maximum(0.0, d - free)    # sim-points per excess metre
    score = sims[0].astype(np.float64).copy()
    back = np.zeros((n_q, n_r), dtype=np.int32)
    for q in range(1, n_q):
        cand = score[:, None] + trans
        back[q] = np.argmax(cand, axis=0)
        score = cand[back[q], np.arange(n_r)] + sims[q]
    path = np.zeros(n_q, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for q in range(n_q - 2, -1, -1):
        path[q] = back[q + 1][path[q + 1]]
    return path


def kartaview_vpr_track(frames_bgr, center, radius_m=3000.0, *,
                        cache_dir=None, n_query=80, device=None,
                        model_name="megaloc", source="kartaview", token=None,
                        cap=1500, sequence_decode=True, query_dt_s=4.0):
    """Per-frame VPR positions: a sparse, noisy 'GPS' track to fit the trajectory
    to (anchor-primary v2). Returns ``(query_indices, latlons[N,2], sims[N])`` for
    ``n_query`` frames sampled uniformly across the clip, or ``None``. Unlike the
    single-point prior, this constrains the trajectory's ORIENTATION + start, not
    just its centre. ``source`` selects ``"kartaview"``/``"mapillary"``/
    ``"panoramax"``. ``sequence_decode`` (default on) replaces per-frame argmax
    with the continuity-constrained Viterbi decode; ``query_dt_s`` is the real
    seconds between query frames (sets the transition free radius).
    """
    try:
        import torch  # noqa: F401 - availability guard; helper re-imports
    except Exception:
        return None
    try:
        out = _prepare_refs_and_query(frames_bgr, center, radius_m, cache_dir,
                                      n_query, device, model_name, source,
                                      token, cap)
        if out is None:
            return None
        idx, sims, ref_xy = out
        top1 = sims.argmax(1)
        if sequence_decode and len(idx) >= 3:
            try:
                top1 = _viterbi_decode(sims, ref_xy, float(query_dt_s))
            except Exception:
                pass                      # keep the argmax track
        maxsim = sims[np.arange(len(idx)), top1]
        return idx, ref_xy[top1].astype(float), maxsim.astype(float)
    except Exception:
        return None
