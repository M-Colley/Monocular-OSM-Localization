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


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON via a temp file + os.replace, so an interrupted/concurrent
    write never truncates the canonical cache (which would reset the union
    accumulator to {} and reintroduce the +-100 m cold-fetch drift R1 killed;
    audit CACHE-ATOMIC)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _atomic_savez(path: str, **arrays) -> None:
    """np.savez via temp + os.replace (see _atomic_write_json). np.savez
    appends .npz if missing, so write to a base name then rename the .npz."""
    base = f"{path}.{os.getpid()}.tmp"
    np.savez(base, **arrays)
    os.replace(base + ".npz", path)


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


def _sig_tag(sig: dict) -> str:
    """Short filename tag of a fetch signature, so caches for DIFFERENT query
    shapes coexist in one dir. With a single untagged meta/npz pair, the
    deployable (city-extent 8 km/cap-3000) and GT-seeded (3 km/1500) configs
    of the same clip CLOBBER each other's cache — seen live 2026-07-11: the
    auto-sizing runs overwrote Ulm's GT-seeded cache and every later tokenless
    run lost the VPR channel ("VPR unavailable")."""
    blob = json.dumps(sig, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _legacy_img_paths(cache_dir, fp: str) -> list[str]:
    """Pre-R1 monolithic image-cache paths (fingerprint-tagged, then
    untagged). Read-only fallbacks: _load_ref_images migrates their contents
    into the per-id store on first use, so pre-existing warm caches (the
    whole GT-seeded fleet) keep serving tokenless after the R1 format change."""
    if not cache_dir:
        return []
    return [os.path.join(cache_dir, f"ref_imgs_{fp[:10]}.npz"),
            os.path.join(cache_dir, "ref_imgs.npz")]




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
        _atomic_write_json(meta, {"signature": sig, "refs": refs})
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
    # Signature-TAGGED meta (variants coexist) with the legacy untagged file
    # as a read fallback so pre-existing warm caches keep serving.
    metas = ([os.path.join(cache_dir, f"mly_ref_meta_{_sig_tag(sig)}.json"),
              os.path.join(cache_dir, "mly_ref_meta.json")]
             if cache_dir else [])
    meta = metas[0] if metas else None
    # The cached refs carry NO thumb urls (they expire), so they are only
    # servable when pixels for (nearly) all of them are cached — otherwise
    # _load_ref_images would drop the url-less misses and silently serve a
    # truncated set. Existence of the meta is not enough. Pixels can live in
    # the per-id store OR a legacy monolithic npz (pre-R1 caches — the whole
    # GT-seeded fleet; _load_ref_images migrates those on first use).
    stored_ids: set[str] = set()
    store_path = os.path.join(cache_dir, "ref_img_store.npz") if cache_dir else None
    if store_path and os.path.exists(store_path):
        try:
            with np.load(store_path, allow_pickle=True) as d:
                stored_ids = {str(x) for x in d["ids"]}
        except Exception:
            stored_ids = set()
    for mpath in metas:
        if not os.path.exists(mpath):
            continue
        try:
            blob = json.load(open(mpath))
            if not (isinstance(blob, dict) and blob.get("signature") == sig):
                continue
            refs_m = blob.get("refs") or []
            if not refs_m:
                continue
            n_stored = sum(1 for r in refs_m if str(r["id"]) in stored_ids)
            # A COMPLETE store always serves; a PARTIAL one serves at >=90%
            # and >=30 — a handful of transiently-failed downloads must not
            # kill the offline path (the token run itself served only the
            # stored subset, so serving it warm is IDENTICAL to a rerun with
            # the token; bug B, round-5). _load_ref_images drops the url-less
            # misses.
            if n_stored == len(refs_m) or (
                    n_stored >= 30 and n_stored >= int(0.9 * len(refs_m))):
                return refs_m
            # Legacy monolithic image cache (fingerprint over the FULL meta
            # list): serve if it matches; _load_ref_images migrates it into
            # the per-id store on load (bug A, round-5 — pre-R1 warm caches,
            # i.e. every GT-seeded fleet clip, otherwise die tokenless).
            fp = _refs_fingerprint(refs_m)
            for npz in _legacy_img_paths(cache_dir, fp):
                if not os.path.exists(npz):
                    continue
                try:
                    with np.load(npz, allow_pickle=True) as d:
                        if ("fingerprint" in d.files
                                and str(d["fingerprint"]) == fp):
                            return refs_m
                except Exception:
                    pass
            # not enough pixels anywhere -> fall through to the token fetch
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
    # To bound API cost on large radii we COARSEN the step so the grid stays
    # under the cell budget while still covering the whole disc. (The old
    # code linspace-DROPPED cells instead, which left striped ~0.5-1.2 km
    # unqueried gaps across 70%+ of an 8 km disc — exactly the deployable
    # city-extent mode whose entire point is that the drive is inside the
    # disc; audit round-4 R3.)
    step_lat = 0.0022
    _n_est = lambda s: (int(2 * dlat / s) + 1) * (  # noqa: E731
        int(2 * dlon / (s / max(np.cos(np.radians(clat)), 0.2))) + 1)
    while _n_est(step_lat) > 1200:
        step_lat *= 1.3
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
    sess = requests.Session()

    def query(cell, _retried=False, _depth=0):
        w, s, e, n = cell
        try:
            r = sess.get("https://graph.mapillary.com/images",
                         params={"access_token": token,
                                 "fields": "id,geometry,thumb_1024_url",
                                 "bbox": f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}",
                                 "limit": 2000}, timeout=30)
            data = r.json().get("data", []) if r.status_code == 200 else []
        except Exception:
            # One retry: a single 30 s timeout otherwise silently drops this
            # cell's patch of refs FOR THIS RUN ONLY — a direct source of the
            # run-to-run variance in the deployable numbers.
            if not _retried:
                return query(cell, _retried=True, _depth=_depth)
            return []
        # A FULL page means the (possibly coarsened) cell was truncated —
        # subdivide and union, same pattern as the Panoramax fetcher.
        if len(data) >= 2000 and _depth < 2:
            mx, my = (w + e) / 2.0, (s + n) / 2.0
            out = []
            for sub in [(w, s, mx, my), (mx, s, e, my),
                        (w, my, mx, n), (mx, my, e, n)]:
                out.extend(query(sub, _depth=_depth + 1))
            return out
        return data

    refs = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for data in ex.map(query, cells):
            for d in data:
                g = d.get("geometry", {}).get("coordinates")
                u = d.get("thumb_1024_url")
                if g and u:
                    refs[d["id"]] = {"lat": float(g[1]), "lon": float(g[0]), "url": u}
    # UNION the fetched ids into a persistent per-signature store so the ref
    # set only ever GROWS across cold queries — the Graph API returns a
    # slightly different id-set each time and cells occasionally fail, so
    # without accumulation the served set (hence the prior) drifts +-100 m
    # run-to-run (audit round-4 R1). The store holds id/lat/lon only.
    store = os.path.join(cache_dir, f"mly_ref_store_{_sig_tag(sig)}.json") if cache_dir else None
    union: dict[str, list] = {}
    if store and os.path.exists(store):
        try:
            union = json.load(open(store))
        except Exception:
            union = {}
    for k, v in refs.items():
        union[str(k)] = [v["lat"], v["lon"]]
    if store and union:
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_write_json(store, union)
    # Deterministic, insertion-STABLE subsample: keep the `cap` ids with the
    # lowest sha1(id). Unlike linspace over a sorted list (one inserted/dropped
    # id shifts every kept index -> a different served set each run), a sha1
    # threshold keeps the SAME ids whenever they reappear; only ids near the
    # cap-th sha1 quantile ever swap in/out. Combined with the union store the
    # served set converges to a fixed point.
    ids = list(union.keys())
    if len(ids) > cap:
        ids = sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:cap]
    ids = sorted(ids)   # id order -> stable fingerprint
    out = [{"id": i, "lat": union[i][0], "lon": union[i][1],
            **({"url": refs[i]["url"]} if i in refs else {})} for i in ids]
    if meta and out:
        os.makedirs(cache_dir, exist_ok=True)
        # store WITHOUT urls (they expire); the per-id image store holds pixels
        stable = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"]} for r in out]
        _atomic_write_json(meta, {"signature": sig, "refs": stable})
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

    def query(cell, depth=0):
        try:
            r = sess.get("https://api.panoramax.xyz/api/search",
                         params={"bbox": ",".join(f"{v:.6f}" for v in cell),
                                 "limit": 500}, timeout=30)
            feats = r.json().get("features", [])
        except Exception:
            return []
        # A FULL page means the cell was truncated (the API returns at most
        # ~500 features and no next-page link) — silently keeping it would
        # spatially bias the ref set exactly where coverage is densest.
        # Subdivide the cell (bounded depth) and take the union instead.
        if len(feats) >= 500 and depth < 2:
            w, s, e, nn = cell
            mx, my = (w + e) / 2.0, (s + nn) / 2.0
            sub = [(w, s, mx, my), (mx, s, e, my),
                   (w, my, mx, nn), (mx, my, e, nn)]
            out = []
            for c in sub:
                out.extend(query(c, depth + 1))
            return out
        return feats

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
        _atomic_write_json(meta, {"signature": sig, "refs": refs})
    return refs


def has_mapillary_cache(cache_dir) -> bool:
    """True when a reusable Mapillary ref cache (metadata + image blob)
    exists, so a warm rerun can skip the MLY_TOKEN requirement entirely.
    (Whether it actually COVERS the query is decided by the signature check
    in :func:`_fetch_refs_mapillary`; a mismatch degrades to no refs.)
    Matches both the signature-tagged filenames and the legacy untagged ones."""
    import glob as _glob
    if not cache_dir:
        return False
    return bool(_glob.glob(os.path.join(cache_dir, "mly_ref_meta*.json"))
                and (os.path.exists(os.path.join(cache_dir, "ref_img_store.npz"))
                     or _glob.glob(os.path.join(cache_dir, "ref_imgs*.npz"))))


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
    """Embed a list of PREPPED tensors, or a uint8 BGR array ``[N,H,W,3]``.

    The array form preps lazily PER BATCH: materializing all prepped float32
    tensors up front is ~3.15 MB each — ~9.4 GB at the 3000-ref deployable
    cap, on top of the 2.4 GB raw array (audit round-4 f3/R2). Lazy prep
    holds ~75 MB per 24-batch instead; results are identical.
    """
    import torch
    lazy = not isinstance(imgs, list)
    out = []
    for i in range(0, len(imgs), 24):
        chunk = imgs[i:i + 24]
        if lazy:
            chunk = [_prep(chunk[j]) for j in range(len(chunk))]
        batch = torch.stack(chunk).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            f = _MODEL(batch).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def _load_ref_images(refs, cache_dir):
    """Download (or load cached) reference images. Returns
    ``(raw[N,512,512,3], ref_xy[N,2], fingerprint)`` or ``(None, None, fp)``.

    Per-ID image store (``ref_img_store.npz`` = parallel ``ids``/``raw``):
    each photo's pixels are cached by its own id, so a ref set that only
    grew/shrank by a few ids reuses the rest instead of re-downloading all
    (the old monolithic fingerprint-keyed npz discarded EVERYTHING on a 1-id
    drift — a slow, nondeterministic refetch and the main cold-fetch variance
    source; audit round-4 R1). Refs whose ``url`` is absent (accumulated by
    the union store but not in the current metadata query) are served ONLY if
    already stored. The fingerprint is over the SERVABLE subset, so it is
    stable once the union/sha1 pick converges — warm embed reruns then hit.
    """
    import cv2
    import requests
    store_path = os.path.join(cache_dir, "ref_img_store.npz") if cache_dir else None
    have: dict[str, int] = {}
    store_ids: list[str] = []
    store_raw = store_xy = None
    if store_path and os.path.exists(store_path):
        # mmap_mode is a NO-OP for .npz archives (numpy returns an in-RAM
        # NpzFile), so there is no lazy row access — just load and CLOSE the
        # handle (audit K2/F5). This full path only runs on a cold/partial
        # miss (pixels are needed anyway); the fully-warm case skips it via
        # _peek_store in _embed_refs.
        try:
            with np.load(store_path, allow_pickle=True) as d:
                store_ids = [str(x) for x in d["ids"]]
                store_raw = np.asarray(d["raw"])
                store_xy = np.asarray(d["xy"], float)
            have = {i: k for k, i in enumerate(store_ids)}
        except Exception:
            have, store_ids, store_raw, store_xy = {}, [], None, None

    # MIGRATE a legacy monolithic npz (pre-R1 format) into the per-id model:
    # its fingerprint is over exactly this refs list, so its `keep` indices
    # map each stored image to refs[keep[j]] with coords ref_xy[j]. Without
    # this, every pre-R1 warm cache (the whole GT-seeded fleet) re-downloads
    # — or dies tokenless (round-5 bug A).
    legacy_imgs: dict[str, np.ndarray] = {}
    legacy_xy: dict[str, list] = {}
    if any(str(r["id"]) not in have for r in refs):
        fp0 = _refs_fingerprint(refs)
        for lp in _legacy_img_paths(cache_dir, fp0):
            if not os.path.exists(lp):
                continue
            try:
                with np.load(lp, allow_pickle=True) as d:
                    if ("fingerprint" not in d.files
                            or str(d["fingerprint"]) != fp0
                            or "ref_xy" not in d.files):
                        continue
                    keep = [int(k) for k in d["keep"]] if "keep" in d.files \
                        else list(range(len(d["raw"])))
                    lraw = np.asarray(d["raw"])
                    lxy = np.asarray(d["ref_xy"], float)
                    for j, k in enumerate(keep):
                        if k < len(refs):
                            i = str(refs[k]["id"])
                            if i not in have:
                                legacy_imgs[i] = lraw[j]
                                legacy_xy[i] = [float(lxy[j, 0]), float(lxy[j, 1])]
                break
            except Exception:
                continue

    sess = requests.Session()

    def fetch(ref):
        try:
            rr = sess.get(ref["url"], timeout=25)
            if rr.status_code == 200:
                img = cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
                return None if img is None else cv2.resize(img, (512, 512))
        except Exception:
            return None
        return None

    to_dl = [r for r in refs
             if str(r["id"]) not in have and str(r["id"]) not in legacy_imgs
             and r.get("url")]
    new_imgs: dict[str, np.ndarray] = {}
    if to_dl:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for r, a in zip(to_dl, ex.map(fetch, to_dl)):
                if a is not None:
                    new_imgs[str(r["id"])] = a

    # servable = refs with pixels available (stored, legacy, or just fetched)
    servable = [r for r in refs
                if str(r["id"]) in have or str(r["id"]) in legacy_imgs
                or str(r["id"]) in new_imgs]
    if not servable:
        return None, None, _refs_fingerprint(refs)

    # Each image keeps ITS OWN stored lat/lon (never the incoming ref's), so a
    # cached photo can never be mislabelled with another ref's coordinates —
    # the coordinate-mispairing failure the old fingerprint guard existed for.
    out_raw, out_xy = [], []
    for r in servable:
        i = str(r["id"])
        if i in have:
            out_raw.append(np.asarray(store_raw[have[i]]))
            out_xy.append(store_xy[have[i]])
        elif i in legacy_imgs:
            out_raw.append(legacy_imgs[i])
            out_xy.append(legacy_xy[i])
        else:
            out_raw.append(new_imgs[i])
            out_xy.append([r["lat"], r["lon"]])
    raw = np.stack(out_raw)
    ref_xy = np.array(out_xy, float)
    del out_raw, out_xy
    fp = _refs_fingerprint(servable)

    # Persist the store as EXACTLY the current servable set (bounded at cap;
    # evicts ids that left the sha1 pick) — but only rewrite when it changed,
    # to avoid re-serialising ~2.4 GB every warm run.
    if store_path and set(str(r["id"]) for r in servable) != set(store_ids):
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_savez(store_path,
                      ids=np.array([str(r["id"]) for r in servable]),
                      raw=raw, xy=ref_xy)
    return raw, ref_xy, fp


def _peek_store(refs, cache_dir):
    """If the per-id image store holds EVERY ref, return ``(fp, ref_xy)``
    computed from the stored ids/coords ALONE — no pixels loaded. Lets a
    fully-warm embed-cache hit skip materializing ~2.4 GB of raw pixels that
    are then discarded unembedded (audit K2). np.load on an .npz is lazy, so
    reading d["ids"]/d["xy"] never touches the raw array; the handle is closed
    by the context manager (audit F5)."""
    if not cache_dir:
        return None
    p = os.path.join(cache_dir, "ref_img_store.npz")
    if not os.path.exists(p):
        return None
    try:
        with np.load(p, allow_pickle=True) as d:
            have = {str(x): k for k, x in enumerate(d["ids"])}
            xy = np.asarray(d["xy"], float)
    except Exception:
        return None
    if any(str(r["id"]) not in have for r in refs):
        return None
    ref_xy = np.array([xy[have[str(r["id"])]] for r in refs], float)
    return _refs_fingerprint(refs), ref_xy


def _read_emb_cache(cache_dir, model_name, fp, n):
    """Guarded embedding-cache read: a truncated/corrupt npz (non-atomic write
    interrupted by Ctrl-C / disk-full) is treated as a MISS, not an exception
    that would brick the fingerprint-keyed VPR channel on every warm run
    (audit F1 — the one read in this module that was unguarded)."""
    if not cache_dir:
        return None
    import torch
    for p in [os.path.join(cache_dir, f"ref_emb_{model_name}_{fp[:10]}.npz"),
              os.path.join(cache_dir, f"ref_emb_{model_name}.npz")]:
        if not os.path.exists(p):
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                if str(d["fingerprint"]) == fp and len(d["emb"]) == n:
                    return torch.from_numpy(np.asarray(d["emb"], np.float32))
        except Exception:
            continue   # corrupt npz -> re-embed (cheap, deterministic)
    return None


def _embed_refs(refs, device, cache_dir, model_name="model"):
    # Fully-warm fast path: store holds every ref + embeddings cached -> return
    # without loading a single pixel or hitting the network (audit K2).
    peek = _peek_store(refs, cache_dir)
    if peek is not None:
        fp, ref_xy = peek
        emb = _read_emb_cache(cache_dir, model_name, fp, len(ref_xy))
        if emb is not None:
            return emb, ref_xy
    raw, ref_xy, fp = _load_ref_images(refs, cache_dir)
    if raw is None:
        return None, None
    emb = _read_emb_cache(cache_dir, model_name, fp, len(ref_xy))
    if emb is None:
        emb = _embed(device, raw)   # lazy per-batch prep (see _embed)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            _atomic_savez(
                os.path.join(cache_dir, f"ref_emb_{model_name}_{fp[:10]}.npz"),
                emb=emb.numpy(), fingerprint=np.array(fp))
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
    # O(n_ref^2) matrices: float32 at n_r=6000 is ~144 MB each (d/trans/cand
    # live simultaneously). --vpr-cap makes larger sets reachable from the
    # CLI; beyond this bound a MemoryError could kill the whole run, so bail
    # to the caller's argmax fallback instead.
    if n_r > 6000:
        raise ValueError(f"viterbi decode skipped: {n_r} refs exceeds the "
                         f"O(n^2) memory bound (6000); using argmax track")
    lat0 = float(np.mean(ref_latlon[:, 0]))
    xy = np.column_stack([
        ref_latlon[:, 1] * 111320.0 * np.cos(np.radians(lat0)),
        ref_latlon[:, 0] * 111320.0]).astype(np.float32)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    free = 30.0 + 40.0 * max(dt_s, 0.5)          # generous urban speed cap
    trans = -0.02 * np.maximum(0.0, d - free)    # sim-points per excess metre
    score = sims[0].astype(np.float32).copy()
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
