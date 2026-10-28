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
import datetime as dt
import hashlib
import heapq
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy

import numpy as np

from .geo import LocalFrame, degree_half_extent

_MEAN = None
_STD = None
_MODEL = None
_MODEL_NAME = None   # backbone actually resident in _MODEL (survives fallback)
_MODEL_DEVICE = None  # device the resident model runs on ("cuda" | "cpu")

# torch.hub sources, pinned to commits so an upstream change to a backbone
# cannot silently change results (recorded via backbone_info()).
# trust_repo=True is passed explicitly: without it torch.hub ASKS on stdin
# whether to trust the repo, and an unattended run reads EOF as "no" — the
# model never loads and the VPR channel, the main accuracy lever, silently
# drops out ("VPR unavailable") on every fresh machine.
HUB_MEGALOC = "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
HUB_EIGENPLACES = "gmberton/eigenplaces:a2969f71d5ea31017443af490b15273ca4c50af1"

# Why the last VPR entrypoint returned None (None after a success), so the
# pipeline can record the reason instead of a bare "unavailable".
LAST_FAILURE: str | None = None
LAST_TRACK_STATS: dict = {}   # requested versus actual decoder on the last track call
#: Original listing diagnostics and the exact metadata snapshot for the last
#: Mapillary query, restored on warm hits too. Legacy metadata marks listing
#: completeness unknown instead of implying a complete zero-loss listing.
LAST_FETCH_STATS: dict = {}
#: How often a tile the Graph API rejects as too large ("Please reduce the
#: amount of data") is split into quarters. ~540 m tiles (an 8 km disc) are
#: rejected over a dense downtown; four levels reach ~35 m.
_MLY_MAX_SPLIT_DEPTH = 4
#: The outer reference ring of the last retrieval pass that asked for one
#: (``outer_radius_m``): radius, cap, refs added, its listing record; else None.
LAST_RING: dict | None = None


def backbone_info() -> dict:
    """Which retrieval backbone is resident and where it came from."""
    hub = {"megaloc": HUB_MEGALOC, "eigenplaces": HUB_EIGENPLACES}
    return {"name": _MODEL_NAME, "hub": hub.get(_MODEL_NAME or ""), "device": _MODEL_DEVICE}


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


def _reference_snapshot(refs, *, created_utc=None) -> dict:
    """Identity of the exact ordered metadata set, without expiring URLs."""
    content = [[str(r["id"]), float(r["lat"]), float(r["lon"])] for r in refs]
    digest = hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()
    return {"refs_sha256": digest, "n_refs": len(refs), "created_utc": created_utc}


def _restore_mapillary_fetch_stats(blob: dict, refs) -> None:
    """Warm reads retain the listing's losses and its reference-set identity.

    Older metadata did not record listing completeness. Mark it unknown rather
    than presenting it as a successful zero-loss listing or leaking another
    query's process-global statistics.
    """
    global LAST_FETCH_STATS
    listing_stats = blob.get("listing_stats")
    known = isinstance(listing_stats, dict) and bool(listing_stats)
    snapshot = _reference_snapshot(refs)
    recorded = blob.get("snapshot") or {}
    if recorded.get("refs_sha256") == snapshot["refs_sha256"]:
        snapshot["created_utc"] = recorded.get("created_utc")
    LAST_FETCH_STATS = {
        **(listing_stats if known else {"listing": "unknown"}),
        "listing_stats_known": known,
        "cache_hit": True,
        "snapshot": snapshot,
    }


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
    dlat, dlon = degree_half_extent(clat, clon, radius_m)
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


def _fetch_refs_mapillary(center, radius_m, cache_dir, cap=1500, token=None,
                          listing="tiles"):
    """Mapillary street-level refs (Graph API v4) as ``[{id,lat,lon,url}]`` —
    a *much* denser source than KartaView on most areas (validated: a MegaLoc
    prior from these lands 3-31 m from the GT route on all clips, incl. the
    ones KartaView could not cover). Needs a free access token (``MLY_TOKEN``
    env var or ``token=``) unless a warm ref cache covers the query. Returns
    [] with no token and no warm cache / on error.

    Metadata and its original listing diagnostics are cached without signed
    thumbnail URLs. A warm pixel store serves it offline. Refs are sorted by
    id so their snapshot identity is stable across runs.
    """
    # Persistent ref cache (id/lat/lon only — NOT the expiring thumb URLs) so
    # the ref set (hence the prior) is REPRODUCIBLE across runs. Without it,
    # Mapillary returns a slightly different id-set each query, the image-cache
    # fingerprint drifts, and a different subsample is used run-to-run (seen:
    # London prior wandered 91 m vs 356 m). On a warm hit with the embedded-
    # image cache present, we reuse the cached refs and never re-download.
    # Checked BEFORE the token guard: a warm cache needs no API access, so
    # offline regression sweeps run without MLY_TOKEN.
    global LAST_FETCH_STATS
    LAST_FETCH_STATS = {}
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
                _restore_mapillary_fetch_stats(blob, refs_m)
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
                            _restore_mapillary_fetch_stats(blob, refs_m)
                            return refs_m
                except Exception:
                    pass
            # not enough pixels anywhere -> fall through to the token fetch
        except Exception:
            pass
    token = token or os.environ.get("MLY_TOKEN")
    if not token:
        return []
    import requests
    sess = requests.Session()
    stats = {"listing": listing, "tiles": 0, "split": 0, "lost": 0}
    lock = threading.Lock()

    def count(key, n=1):
        with lock:
            stats[key] += n

    # listed: id -> [lat, lon] (millions over an 8 km disc, so plain lists);
    # urls: id -> thumbnail URL where known (the Graph search returns them).
    listed: dict[str, list] = {}
    urls: dict[str, str] = {}
    if listing == "tiles":
        listed = _mly_tile_listing(center, radius_m, token, sess, count, stats)
        if stats["tiles"] and stats["lost"] == stats["tiles"]:
            print("      -> mapillary coverage tiles unavailable; listing with "
                  "the Graph API bbox search instead")
            stats.update(listing="graph", tiles=0, split=0, lost=0)
            listing = "graph"
    if listing == "graph":
        graph = _mly_graph_listing(center, radius_m, token, sess, count, stats)
        listed = {k: [v["lat"], v["lon"]] for k, v in graph.items()}
        urls = {k: v["url"] for k, v in graph.items()}
        del graph
    if stats["lost"]:
        print(f"      -> mapillary listing: {stats['lost']} of {stats['tiles']} tiles "
              f"(after {stats['split']} splits) returned nothing; the reference set "
              f"is incomplete there")
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
    union.update(listed)
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
        ids = heapq.nsmallest(cap, ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())
    ids = sorted(ids)   # id order -> stable fingerprint
    # Coverage tiles carry no thumbnail URLs: look them up for the kept ids
    # only (this also gives url-less store ids from earlier listings a URL).
    need = [i for i in ids if i not in urls]
    if need:
        found = _mly_lookup_urls(need, token, sess)
        stats["url_lookups"], stats["url_missing"] = len(need), len(need) - len(found)
        urls.update(found)
    out = [{"id": i, "lat": union[i][0], "lon": union[i][1],
            **({"url": urls[i]} if i in urls else {})} for i in ids]
    snapshot = _reference_snapshot(
        out, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    LAST_FETCH_STATS = {**stats, "listing_stats_known": True,
                        "cache_hit": False, "snapshot": snapshot}
    if meta and out:
        os.makedirs(cache_dir, exist_ok=True)
        # store WITHOUT urls (they expire); the per-id image store holds pixels
        stable = [{"id": r["id"], "lat": r["lat"], "lon": r["lon"]} for r in out]
        _atomic_write_json(meta, {"signature": sig, "refs": stable,
                                  "listing_stats": stats, "snapshot": snapshot})
    return out


def _mly_tile_listing(center, radius_m, token, sess, count, stats) -> dict:
    """``{id: [lat, lon]}`` of every image in the disc's bounding box, read from
    Mapillary's zoom-14 coverage tiles (``mly1_public``, layer ``image``): one
    response per ~1.5 km tile lists all its images, where the Graph API bbox
    search times out over dense centres and truncates answers it does give
    (the same Berlin 100 m box returned 339, 34 or 22 images by page size)."""
    import mercantile

    from .mvt import decode_points
    clat, clon = center
    dlat, dlon = degree_half_extent(clat, clon, radius_m)
    w, s, e, n = clon - dlon, clat - dlat, clon + dlon, clat + dlat
    tiles = list(mercantile.tiles(w, s, e, n, 14))
    stats["tiles"] = len(tiles)

    def one(t):
        for attempt in range(3):
            try:
                r = sess.get("https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
                             f"{t.z}/{t.x}/{t.y}", params={"access_token": token},
                             timeout=60)
                if r.status_code == 200:
                    return decode_points(r.content, "image", t.z, t.x, t.y)
                raise RuntimeError(f"HTTP {r.status_code}")
            except Exception:
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
        count("lost")
        return [], np.zeros((0, 2))

    refs = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ids, ll in ex.map(one, tiles):
            if not len(ll):
                continue
            inside = ((ll[:, 0] >= s) & (ll[:, 0] <= n)
                      & (ll[:, 1] >= w) & (ll[:, 1] <= e))
            for i, (la, lo), keep in zip(ids, ll, inside):
                if keep and i is not None:
                    refs[str(i)] = [float(la), float(lo)]
    return refs


def _mly_lookup_urls(ids, token, sess, batch=100) -> dict:
    """``{id: thumb_1024_url}`` for ``ids`` via the Graph API (at most 300 ids
    per call; 100 answer in ~3 s). Ids it cannot resolve are left out."""
    chunks = [ids[k:k + batch] for k in range(0, len(ids), batch)]

    def one(chunk):
        for attempt in range(3):
            try:
                r = sess.get("https://graph.mapillary.com/images",
                             params={"access_token": token, "image_ids": ",".join(chunk),
                                     "fields": "id,thumb_1024_url"}, timeout=60)
                if r.status_code == 200:
                    return {str(d["id"]): d["thumb_1024_url"]
                            for d in r.json().get("data", []) if d.get("thumb_1024_url")}
            except Exception:
                pass
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
        return {}

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for got in ex.map(one, chunks):
            out.update(got)
    return out


def _mly_graph_listing(center, radius_m, token, sess, count, stats) -> dict:
    """``{id: {lat, lon, url}}`` from the Graph API bbox search over a grid of
    cells: the fallback when coverage tiles are unavailable. Incomplete over
    dense centres even with splitting (see ``_mly_tile_listing``)."""
    clat, clon = center
    dlat, dlon = degree_half_extent(clat, clon, radius_m)
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
    stats["tiles"] = len(cells)

    def split(cell, depth):
        w, s, e, n = cell
        mx, my = (w + e) / 2.0, (s + n) / 2.0
        out = []
        for sub in [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]:
            out.extend(query(sub, _depth=depth + 1))
        return out

    def query(cell, _retried=False, _depth=0):
        w, s, e, n = cell
        try:
            r = sess.get("https://graph.mapillary.com/images",
                         params={"access_token": token,
                                 "fields": "id,geometry,thumb_1024_url",
                                 "bbox": f"{w:.5f},{s:.5f},{e:.5f},{n:.5f}",
                                 "limit": 2000}, timeout=30)
            if r.status_code == 200:
                data = r.json().get("data", [])
            elif (r.status_code == 500 and "reduce the amount of data" in r.text.lower()
                  and _depth < _MLY_MAX_SPLIT_DEPTH):
                # The API rejects a tile holding too many images. Dropping it
                # (as this code used to) made every city-wide first pass blind
                # to the densest areas - downtown Detroit had images within
                # 130 m of every route waypoint, and none were listed.
                count("split")
                return split(cell, _depth)
            else:
                raise RuntimeError(f"HTTP {r.status_code}")
        except Exception:
            # One retry after a short pause (rate limits, timeouts); a tile
            # still failing is counted in LAST_FETCH_STATS, never silent.
            if not _retried:
                time.sleep(2.0)
                return query(cell, _retried=True, _depth=_depth)
            count("lost")
            return []
        # A FULL page means the (possibly coarsened) cell was truncated —
        # subdivide and union, same pattern as the Panoramax fetcher.
        if len(data) >= 2000 and _depth < _MLY_MAX_SPLIT_DEPTH:
            count("split")
            return split(cell, _depth)
        return data

    refs = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for data in ex.map(query, cells):
            for d in data:
                g = d.get("geometry", {}).get("coordinates")
                u = d.get("thumb_1024_url")
                if g and u:
                    refs[str(d["id"])] = {"lat": float(g[1]), "lon": float(g[0]), "url": u}
    return refs


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
    dlat, dlon = degree_half_extent(clat, clon, radius_m)
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


def _union_refs(center, radius_m, cache_dir, cap, token, device, model_name, *, batch_size=24):
    """Embed and CONCATENATE reference pools from several street-imagery
    sources — the coverage lever for areas where one source is thin (KITTI's
    start has 2 Mapillary refs <=50 m but 23 KartaView <=250 m). Each source
    keeps its own cache sub-dir (sibling of ``cache_dir``); embeddings share
    the MegaLoc space so pools concatenate directly. Returns
    ``(ref_emb, ref_xy)`` or ``(None, None)``."""
    from pathlib import Path

    import numpy as np
    import torch
    cache_path = Path(cache_dir)
    regional = (cache_path.parent.name in ("union_regions", "union_city")
                or cache_path.name.startswith(("union_region_", "union_city_")))
    parent = cache_path if regional else cache_path.parent
    embs, xys = [], []
    counts = []
    for src in ("mapillary", "kartaview", "panoramax"):
        d = str(parent / src)
        try:
            refs = _fetch_refs_for(src, center, radius_m, d, cap,
                                   token if src == "mapillary" else None)
            if len(refs) < 5:
                continue
            embed_options = {"batch_size": batch_size} if batch_size != 24 else {}
            e, xy = _embed_refs(refs, device, d, model_name=model_name, **embed_options)
            if e is not None and len(xy):
                embs.append(e)
                xys.append(np.asarray(xy, dtype=float))
                counts.append(f"{src}:{len(xy)}")
        except Exception:
            continue
    if not embs:
        return None, None
    print(f"      -> VPR union refs [{', '.join(counts)}]")
    return torch.cat(embs), np.vstack(xys)


def _prep(bgr, size=(512, 512)):
    import cv2
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, size)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


def _embed(device, imgs, *, batch_size=24, prepare=None):
    """Embed a list of PREPPED tensors, or a uint8 BGR array ``[N,H,W,3]``.

    The array form preps lazily PER BATCH: materializing all prepped float32
    tensors up front is ~3.15 MB each — ~9.4 GB at the 3000-ref deployable
    cap, on top of the 2.4 GB raw array (audit round-4 f3/R2). Lazy prep
    holds ~75 MB per 24-batch instead; results are identical.
    """
    import torch
    lazy = prepare is not None or not isinstance(imgs, list)
    prep = prepare or _prep
    out = []
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i:i + batch_size]
        if lazy:
            chunk = [prep(chunk[j]) for j in range(len(chunk))]
        batch = torch.stack(chunk).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                           enabled=str(device).startswith("cuda")):
            f = _MODEL(batch).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out)


def _square_query_crop(bgr, view: int):
    """One of three evenly placed square views along the image's long axis."""
    if view not in (0, 1, 2):
        raise ValueError("query crop index must be 0, 1 or 2")
    h, w = bgr.shape[:2]
    edge = min(h, w)
    offset = round((max(h, w) - edge) * view / 2.0)
    return bgr[:, offset:offset + edge] if w >= h else bgr[offset:offset + edge, :]


def _aspect_query_size(bgr) -> tuple[int, int]:
    """Preserve aspect at roughly the control's 512² pixels, on ViT's grid."""
    h, w = bgr.shape[:2]
    scale = 512.0 / np.sqrt(h * w)
    return max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14)


class _QueryViews:
    """Lazy BGR views: large source frames are held only for one model batch."""

    def __init__(self, frames, indices, mode):
        self.frames, self.indices, self.mode = frames, indices, mode
        self.n_views = 3 if mode == "crops" else 1

    def __len__(self):
        return len(self.indices) * self.n_views

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        frame = self.frames[int(self.indices[index // self.n_views])]
        return _square_query_crop(frame, index % 3) if self.mode == "crops" else frame


def _query_cache_key(frames, indices, device, mode, model_name):
    """Identity of immutable sampled inputs and the actual resident model.

    Lists can cheaply distinguish replaced frame objects. Disk-backed sequences
    use their own identity: decoding the same stored frame creates new arrays,
    and reading every frame again would defeat the bounded lazy acquisition.
    Callers must clear the stage-local cache if they mutate pixels in place.
    """
    indices = tuple(int(i) for i in indices)
    items = tuple(id(frames[i]) for i in indices) if isinstance(frames, (list, tuple)) else None
    resident_device = getattr(_MODEL, "device", None)
    if resident_device is None:
        try:
            resident_device = next(_MODEL.parameters()).device
        except (AttributeError, StopIteration):
            resident_device = _MODEL_DEVICE
    model = (id(_MODEL), _MODEL_NAME, model_name, str(device), str(resident_device))
    return (id(frames), items, indices), model, mode


def _audit_array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _audit_tensor_digest(tensor, digest):
    """Hash existing dense tensors with at most 8 MiB of transfer staging."""
    import torch

    tensor = tensor.detach()
    chunk_bytes = 8 * 1024 ** 2
    if tensor.layout != torch.strided:
        raise ValueError("audited tensors must have a dense strided layout")
    if not tensor.is_contiguous():
        if tensor.numel() * tensor.element_size() > chunk_bytes:
            raise ValueError("large noncontiguous model tensors cannot be audited within the staging bound")
        tensor = tensor.contiguous()
    digest.update(json.dumps([list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode())
    flat = tensor.reshape(-1)
    step = max(1, chunk_bytes // max(1, tensor.element_size()))
    for start in range(0, flat.numel(), step):
        raw = flat[start:start + step].cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))


def _audit_tensor_sha(tensor):
    digest = hashlib.sha256()
    _audit_tensor_digest(tensor, digest)
    return digest.hexdigest()


def _audit_model_state():
    """Identify the actual parameters/buffers, independently of a hub label."""
    digest, count, elements = hashlib.sha256(), 0, 0
    for name, tensor in sorted(_MODEL.state_dict().items()):
        digest.update(json.dumps(str(name), ensure_ascii=False).encode())
        _audit_tensor_digest(tensor, digest)
        count += 1
        elements += int(tensor.numel())
    return {"sha256": digest.hexdigest(), "tensor_count": count,
            "element_count": elements, "transfer_chunk_bytes": 8 * 1024 ** 2}


def _audit_pixels(frames, indices):
    digest = hashlib.sha256()
    for index in indices:
        frame = np.asarray(frames[int(index)])
        if frame.nbytes > 64 * 1024 ** 2:
            raise ValueError("audited native frame exceeds the 64 MiB staging bound")
        array = np.ascontiguousarray(frame)
        digest.update(json.dumps([int(index), list(array.shape), array.dtype.str], separators=(",", ":")).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _audit_issue(audit, error):
    audit.setdefault("issues", []).append(f"{type(error).__name__}: {error}")
    audit["status"] = "incomplete"


def _audit_query(audit, frames, indices, descriptors, mode):
    """Auditing never changes descriptors or the ordinary retrieval result."""
    try:
        import torch

        expected_rows = len(indices) * (3 if mode == "crops" else 1)
        if (descriptors.ndim != 2 or descriptors.shape[0] != expected_rows
                or not bool(torch.isfinite(descriptors).all())
                or not bool(torch.allclose(descriptors.norm(dim=1),
                                          torch.ones(expected_rows, device=descriptors.device),
                                          rtol=0, atol=1e-4))):
            raise ValueError("audited query descriptors must be finite L2-normalized rows")
        audit.update(query_pixels_sha256=_audit_pixels(frames, indices),
                     query_descriptors_sha256=_audit_tensor_sha(descriptors),
                     query_descriptor_shape=list(descriptors.shape), query_mode=mode)
    except Exception as error:
        _audit_issue(audit, error)


def _query_similarities(frames, indices, ref_emb, device, *, mode="warp", batch_size=None,
                        query_cache=None, model_name=None, audit=None):
    """Score the same references with fixed query views; no geographic inputs.

    ``warp`` is the unchanged control. ``crops`` takes the maximum cosine per
    reference over three square views. ``aspect`` keeps the native aspect
    ratio at roughly the control's pixel budget. Reference embeddings and
    preprocessing stay unchanged, and sequence decoding happens afterwards.
    """
    if mode not in {"warp", "crops", "aspect"}:
        raise ValueError(f"unknown VPR query mode: {mode}")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("VPR query batch size must be positive")
    key = (_query_cache_key(frames, indices, device, mode, model_name)
           if query_cache is not None else None)
    q_emb = query_cache.get(key) if query_cache is not None else None
    if q_emb is None:
        if mode == "warp" and batch_size is None:
            q_emb = _embed(device, [_prep(frames[int(i)]) for i in indices])
        else:
            views = _QueryViews(frames, indices, mode)
            prep = (lambda im: _prep(im, _aspect_query_size(im))) if mode == "aspect" else _prep
            q_emb = _embed(device, views, batch_size=batch_size or 4, prepare=prep)
        if query_cache is not None and q_emb.numel() * q_emb.element_size() < 10_000_000:
            # One input grid per model/mode bounds memory even if a caller
            # replaces frames or changes sampling during the stage.
            for previous in list(query_cache):
                if previous[1:] == key[1:]:
                    del query_cache[previous]
            query_cache[key] = q_emb
    if audit is not None:
        _audit_query(audit, frames, indices, q_emb, mode)
    scores = (q_emb @ ref_emb.T).numpy()
    n_views = 3 if mode == "crops" else 1
    return scores if n_views == 1 else scores.reshape(len(indices), n_views, -1).max(axis=1)


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

    if not refs:
        return None, None, _refs_fingerprint(refs)
    shape = (512, 512, 3)
    for ref in refs:
        key = str(ref["id"])
        if key in have:
            shape = store_raw[have[key]].shape
            break
        if key in legacy_imgs:
            shape = legacy_imgs[key].shape
            break
    # Download straight into the final pool: retaining N downloaded arrays and
    # then np.stack used two full pixel pools (~2.4 GB for 1500 references).
    raw = np.empty((len(refs), *shape), dtype=np.uint8)
    ref_xy = np.empty((len(refs), 2), dtype=float)
    keep = np.zeros(len(refs), dtype=bool)
    to_dl = []
    for row, ref in enumerate(refs):
        key = str(ref["id"])
        if key in have:
            raw[row], ref_xy[row] = store_raw[have[key]], store_xy[have[key]]
            keep[row] = True
        elif key in legacy_imgs:
            raw[row], ref_xy[row] = legacy_imgs[key], legacy_xy[key]
            keep[row] = True
        elif ref.get("url"):
            to_dl.append((row, ref))
    # Release loaded/migrated arrays before downloads and model inference.
    store_raw = store_xy = None
    legacy_imgs.clear()
    lraw = None
    if to_dl:
        pending = {}
        jobs = iter(to_dl)
        with ThreadPoolExecutor(max_workers=16) as ex:
            def submit_one():
                item = next(jobs, None)
                if item is not None:
                    row, ref = item
                    pending[ex.submit(fetch, ref)] = (row, ref)
            for _ in range(16):
                submit_one()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    row, ref = pending.pop(future)
                    pixels = future.result()
                    if pixels is not None:
                        raw[row] = pixels
                        ref_xy[row] = [ref["lat"], ref["lon"]]
                        keep[row] = True
                    submit_one()
    indices = np.flatnonzero(keep)
    if not len(indices):
        return None, None, _refs_fingerprint(refs)
    # Stable in-place compaction avoids a second full allocation when a few
    # downloads fail; every destination is <= its source in this ordered pass.
    for destination, source in enumerate(indices):
        if destination != source:
            raw[destination], ref_xy[destination] = raw[source], ref_xy[source]
    raw, ref_xy = raw[:len(indices)], ref_xy[:len(indices)]
    servable = [refs[int(index)] for index in indices]
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


def _embed_refs(refs, device, cache_dir, model_name="model", *, batch_size=24):
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
        options = {"batch_size": batch_size} if batch_size != 24 else {}
        emb = _embed(device, raw, **options)   # lazy per-batch prep (see _embed)
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
    latlons = np.asarray(latlons, float)
    sims = np.asarray(sims, float)
    keep = sims >= float(np.percentile(sims, 60))
    if int(keep.sum()) < 5:
        keep = np.ones(len(sims), bool)
    pts, wts = latlons[keep], sims[keep]
    # Weiszfeld in a LOCAL METRIC frame: one degree of longitude is cos(lat)
    # times shorter than one of latitude, so a raw-degree median is biased
    # along the east-west axis (tens of metres at 48-60N — audit kartaview:155).
    # WGS84 azimuthal-equidistant frame around the cloud's mean (geo.py).
    frame = LocalFrame(float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    xy = np.column_stack(frame.to_xy(pts[:, 0], pts[:, 1]))
    p = _geometric_median(xy, weights=wts)
    d = np.linalg.norm(xy - p, axis=1)
    mad = float(np.median(np.abs(d - np.median(d)))) + 1e-6
    good = d <= np.median(d) + 2.5 * 1.4826 * mad
    if int(good.sum()) >= 5:
        p = _geometric_median(xy[good], weights=wts[good])
    lat, lon = frame.to_latlon(float(p[0]), float(p[1]))
    return float(lat), float(lon)


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
    global _MODEL, _MODEL_NAME, _MODEL_DEVICE
    import torch
    if _MODEL is not None:
        return _MODEL_NAME or model_name
    resolved = model_name
    if model_name == "megaloc":
        try:
            _MODEL = torch.hub.load(HUB_MEGALOC, "get_trained_model",
                                    trust_repo=True,
                                    verbose=False).to(device).eval()
        except Exception as e:
            # Not silent: EigenPlaces is a different (weaker) embedding, and
            # results computed with it are not comparable to MegaLoc ones.
            print(f"      -> MegaLoc unavailable ({type(e).__name__}: {e}); "
                  f"falling back to EigenPlaces")
            resolved = "eigenplaces"
    if _MODEL is None:
        _MODEL = torch.hub.load(HUB_EIGENPLACES, "get_trained_model",
                                backbone="ResNet50", fc_output_dim=2048,
                                trust_repo=True,
                                verbose=False).to(device).eval()
    _MODEL_NAME = resolved
    _MODEL_DEVICE = str(device)
    return resolved


def _prepare_refs_and_query(frames_bgr, center, radius_m, cache_dir, n_query,
                            device, model_name, source, token, cap,
                            outer_radius_m=None, outer_cap=None, query_mode="warp",
                            query_batch_size=None, query_cache=None, reference_identity=None,
                            audit=None):
    """Shared front half of both VPR entrypoints: fetch references, load/cache
    the backbone, embed the references + the ``n_query`` sampled query frames.

    ``outer_radius_m`` (> ``radius_m``) adds a sparser ring: a second listing
    of the wider disc, capped at ``outer_cap``, in its own cache dir
    (``<cache_dir>_outer``), minus the ids already in the core set. The core
    set is exactly what it would be without the ring.

    Returns ``(idx, sims[n_query, n_ref], ref_xy[n_ref, 2])`` or ``None`` when
    references are unavailable / too sparse. Callers own the tail (single
    robust centre vs. per-frame track).
    """
    global _MEAN, _STD, LAST_FAILURE, LAST_RING, LAST_FETCH_STATS
    LAST_RING = None
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    resolved = _resolve_backbone(model_name, device)
    reference_options = ({"batch_size": query_batch_size}
                         if query_batch_size is not None else {})
    if source == "union":
        ref_emb, ref_xy = _union_refs(center, radius_m, cache_dir, cap, token,
                                      device, resolved, **reference_options)
    else:
        refs = _fetch_refs_for(source, center, radius_m, cache_dir, cap, token)
        if len(refs) < 30:
            LAST_FAILURE = (f"only {len(refs)} {source} reference images within "
                            f"{radius_m:.0f} m (need 30)")
            return None
        ref_emb, ref_xy = _embed_refs(refs, device, cache_dir, model_name=resolved,
                                     **reference_options)
        if (outer_radius_m and outer_radius_m > radius_m and cache_dir
                and ref_emb is not None):
            core_stats = dict(LAST_FETCH_STATS)
            outer_dir = f"{cache_dir}_outer"
            outer = _fetch_refs_for(source, center, outer_radius_m, outer_dir,
                                    outer_cap or cap, token)
            ring_stats = dict(LAST_FETCH_STATS)
            LAST_FETCH_STATS = core_stats
            have = {str(r["id"]) for r in refs}
            outer = [r for r in outer if str(r["id"]) not in have]
            added = 0
            if outer:
                o_emb, o_xy = _embed_refs(outer, device, outer_dir, model_name=resolved,
                                         **reference_options)
                if o_emb is not None and len(o_xy):
                    ref_emb = torch.cat([ref_emb, o_emb])
                    ref_xy = np.vstack([np.asarray(ref_xy, dtype=float),
                                        np.asarray(o_xy, dtype=float)])
                    added = int(len(o_xy))
            LAST_RING = {"outer_radius_m": float(outer_radius_m),
                         "outer_cap": int(outer_cap or cap), "n_core": int(len(refs)),
                         "n_outer_added": added, "listing": ring_stats}
            print(f"      -> outer ring to {outer_radius_m:.0f} m: +{added} references "
                  f"(core {len(refs)} within {radius_m:.0f} m)")
    if ref_emb is None or len(ref_xy) < 30:
        LAST_FAILURE = (f"too few embeddable {source} references "
                        f"({0 if ref_xy is None else len(ref_xy)})")
        return None
    if reference_identity is not None:
        if source == "union" or outer_radius_m:
            raise ValueError("full reference identity currently requires a single source and disc")
        reference_identity["ref_ids"] = _embedded_reference_ids(refs, ref_xy, cache_dir, source)
        reference_identity["embedding_dimensions"] = int(ref_emb.shape[1])
        if audit is not None:
            reference_identity["listed_ref_ids"] = [f"{source}:{r['id']}" for r in refs]
            reference_identity["listed_ref_latlon"] = [[float(r["lat"]), float(r["lon"])] for r in refs]
            reference_identity["listing_snapshot"] = _reference_snapshot(refs)
    idx = np.linspace(0, len(frames_bgr) - 1,
                      min(n_query, len(frames_bgr))).astype(int)
    if audit is not None:
        try:
            if source == "union" or outer_radius_m:
                raise ValueError("track audit requires a single source and disc")
            if len(idx) > 80 or len(idx) * len(ref_xy) > 1_600_000:
                raise ValueError("track audit exceeds its 80-query/1.6-million-score bound")
            model_state = _audit_model_state()
            audit.update(query_indices=idx.copy(),
                         ref_ids=_embedded_reference_ids(refs, ref_xy, cache_dir, source),
                         ref_latlon=np.asarray(ref_xy, float).copy(),
                         ref_embeddings_sha256=_audit_tensor_sha(ref_emb),
                         actual_model_state_sha256=model_state["sha256"], model_state=model_state,
                         backbone=deepcopy(backbone_info()),
                         reference_encoding_identity="unverified: cache schema carries reference fingerprint only",
                         preprocessing={"version": 1, "color": "BGR to RGB", "resize": "opencv INTER_LINEAR",
                             "warp_size": [512, 512], "scale": "divide by255",
                             "mean": [.485, .456, .406], "std": [.229, .224, .225],
                             "descriptor_normalization": "L2", "dimensions": int(ref_emb.shape[1])},
                         reference_pool={"source": source, "center": list(map(float, center)),
                             "radius_m": float(radius_m), "cap": int(cap), "count": int(len(ref_xy)),
                             "fetch": deepcopy(LAST_FETCH_STATS)})
            if audit.get("reference_rows_requested"):
                if len(ref_xy) > 1500:
                    raise ValueError("per-reference evidence audit exceeds its 1500-row bound")
                audit["reference_row_embeddings_sha256"] = [
                    _audit_tensor_sha(ref_emb[i:i+1]) for i in range(len(ref_xy))]
        except Exception as error:
            _audit_issue(audit, error)
    query_options = ({"query_cache": query_cache, "model_name": resolved}
                     if query_cache is not None else {})
    if audit is not None:
        query_options["audit"] = audit
    sims = _query_similarities(frames_bgr, idx, ref_emb, device, mode=query_mode,
                              batch_size=query_batch_size, **query_options)
    if audit is not None and len(idx) <= 80 and np.size(sims) <= 1_600_000:
        audit["raw_similarities"] = np.asarray(sims).copy()
    return idx, sims, np.asarray(ref_xy, dtype=float)


def _embedded_reference_ids(refs, ref_latlon, cache_dir, source):
    """Identify the ordered *servable* subset, including partial downloads."""
    if len(refs) == len(ref_latlon):
        result = [str(ref["id"]) for ref in refs]
    else:
        if cache_dir is None:
            raise ValueError("a partial reference pool needs its per-ID cache for identity")
        store = os.path.join(cache_dir, "ref_img_store.npz")
        with np.load(store, allow_pickle=False) as data:
            ids = [str(value) for value in data["ids"]]
            xy = np.asarray(data["xy"], float)
        if len(ids) != len(xy) or len(set(ids)) != len(ids):
            raise ValueError("reference store has invalid ID/coordinate identity")
        locations = dict(zip(ids, xy))
        result = [str(ref["id"]) for ref in refs if str(ref["id"]) in locations]
        expected = np.asarray([locations[key] for key in result], float)
        if expected.shape != np.shape(ref_latlon) or not np.array_equal(expected, ref_latlon):
            raise ValueError("servable reference IDs do not match the embedded coordinate order")
    if len(set(result)) != len(result):
        raise ValueError("embedded reference IDs must be unique")
    return np.array([f"{source}:{identifier}" for identifier in result])


def kartaview_vpr_evidence(frames_bgr, timestamps, center, radius_m=3000.0, *,
                            cache_dir=None, n_query=80, device=None, model_name="megaloc",
                            source="mapillary", token=None, cap=1500, query_mode="warp",
                            query_batch_size=4, query_cache=None, audit=None,
                            query_video_sha256=None):
    """Full appearance evidence for an opt-in joint localizer, without decoding.

    Uses the ordinary bounded acquisition/embedding path. The output preserves
    actual query times, every reference score and source-prefixed IDs so callers
    can merge overlapping discs without counting one photo twice. Cache files
    currently lack model/preprocessing identity; that limitation is explicit.
    No default track/prior call retains these matrices or changes behavior.
    An opt-in audit binds the actual native pixels, descriptors, model and
    reference tensor rows. Its complete status refers to matrix identity;
    provenance.acquisition separately reports bounded-listing completeness.
    """
    global LAST_FAILURE
    LAST_FAILURE = None
    try:
        times = np.asarray(timestamps, float)
        if (times.ndim != 1 or len(times) != len(frames_bgr) or not len(times)
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError("appearance timestamps must match finite, strictly ordered frames")
        if source == "union":
            raise ValueError("full evidence currently requires a single reference source")
        if source not in {"mapillary", "kartaview", "panoramax"}:
            raise ValueError("unknown reference source")
        for name, value in (("query count", n_query), ("reference cap", cap),
                            ("batch size", query_batch_size)):
            if value is None and name == "batch size":
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        seed = np.asarray(center, float)
        if (seed.shape != (2,) or not np.isfinite(seed).all()
                or abs(seed[0]) > 90 or abs(seed[1]) > 180
                or not np.isfinite(radius_m) or radius_m <= 0):
            raise ValueError("reference disc needs finite WGS84 coordinates and positive radius")
        if audit is not None:
            if (not isinstance(audit, dict) or not isinstance(query_video_sha256, str)
                    or len(query_video_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in query_video_sha256)
                    or n_query > 80 or cap > 1500):
                raise ValueError("audited evidence needs a video SHA256 and bounded query/reference counts")
            audit.clear()
            audit.update(status="pending", reference_rows_requested=True)
        identity = {}
        audit_options = {"audit": audit} if audit is not None else {}
        prepared = _prepare_refs_and_query(
            frames_bgr, center, radius_m, cache_dir, n_query, device, model_name,
            source, token, cap, query_mode=query_mode, query_batch_size=query_batch_size,
            query_cache=query_cache, reference_identity=identity, **audit_options)
        if prepared is None:
            return None
        indices, similarities, coordinates = prepared
        ref_ids = identity["ref_ids"]
        if (np.shape(coordinates) != (len(ref_ids), 2)
                or np.shape(similarities) != (len(indices), len(ref_ids))
                or not np.isfinite(similarities).all() or not np.isfinite(coordinates).all()):
            raise ValueError("appearance matrix and reference inventory do not agree")
        result = {"query_indices": indices, "query_times": times[indices],
                "similarities": similarities, "ref_latlon": coordinates, "ref_ids": ref_ids,
                "provenance": {
                    "experimental": True, "source": source, "cache": str(cache_dir),
                    "center": list(map(float, center)), "radius_m": float(radius_m), "cap": int(cap),
                    "backbone": backbone_info(), "query_mode": query_mode,
                    "preprocessing": {"version": 1, "reference_color": "RGB", "reference_size": [512, 512],
                                      "reference_resize": "opencv INTER_LINEAR", "scale": "divide by255",
                                      "mean": [.485, .456, .406], "std": [.229, .224, .225],
                                      "descriptor_normalization": "L2",
                                      "dimensions": identity["embedding_dimensions"]},
                    "reference_encoding_identity": "unverified: cache schema carries reference fingerprint only",
                    "fetch": deepcopy(LAST_FETCH_STATS),
                    "reference_inventory": _reference_snapshot([
                        {"id": key, "lat": coordinate[0], "lon": coordinate[1]}
                        for key, coordinate in zip(ref_ids, coordinates)])}}
        if audit is not None:
            proof, row_identities = _finish_evidence_audit(
                audit, indices, similarities, coordinates, ref_ids, times,
                query_mode, query_video_sha256)
            result["status"] = "complete"
            result["provenance"].update(query_identity=proof, reference_identities=row_identities,
                acquisition=_evidence_acquisition(identity, ref_ids, source, cap, LAST_FETCH_STATS))
        return result
    except Exception as exc:
        if isinstance(audit, dict):
            _audit_issue(audit, exc)
        LAST_FAILURE = f"{type(exc).__name__}: {exc}"
        return None


def _finish_evidence_audit(audit, indices, similarities, coordinates, ref_ids, timestamps,
                           query_mode, video_sha256):
    """Bind an opt-in undecoded matrix to the actual query and reference rows."""
    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    idx, times = np.asarray(indices), np.asarray(timestamps, dtype=float)
    views = 3 if query_mode == "crops" else 1
    required = ("query_pixels_sha256", "query_descriptors_sha256", "actual_model_state_sha256")
    if (audit.get("issues") or any(not valid_sha(audit.get(k)) for k in required)
            or idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer)
            or not np.array_equal(audit.get("query_indices"), idx)
            or not np.array_equal(audit.get("raw_similarities"), similarities)
            or not np.array_equal(audit.get("ref_ids"), ref_ids)
            or not np.array_equal(audit.get("ref_latlon"), coordinates)
            or audit.get("query_mode") != query_mode
            or audit.get("query_descriptor_shape") != [len(idx) * views, audit.get("preprocessing", {}).get("dimensions")]):
        raise ValueError("actual query/reference evidence identity is incomplete or inconsistent")
    row_hashes = audit.get("reference_row_embeddings_sha256")
    if (not isinstance(row_hashes, list) or len(row_hashes) != len(ref_ids)
            or any(not valid_sha(value) for value in row_hashes)):
        raise ValueError("actual per-reference embedding identities are incomplete")
    proof = {"query_video_sha256": video_sha256,
        "query_pixels_sha256": audit["query_pixels_sha256"],
        "query_descriptors_sha256": audit["query_descriptors_sha256"],
        "actual_model_state_sha256": audit["actual_model_state_sha256"],
        "query_indices_sha256": _audit_array_sha(idx),
        "query_times_sha256": _audit_array_sha(times[idx]), "query_mode": query_mode,
        "views": views, "aggregation": ("maximum cosine over three crop descriptors for each reference"
            if views == 3 else "one descriptor per query reference comparison")}
    rows = {str(identifier): {"embedding_sha256": digest, "pixel_sha256": None,
        "encoding_identity": audit["reference_encoding_identity"]}
        for identifier, digest in zip(ref_ids, row_hashes)}
    audit.update(status="complete", query_identity=deepcopy(proof), query_times=times[idx].copy())
    return proof, rows


def _evidence_acquisition(identity, embedded_ids, source, cap, fetch):
    """Account for every capped listing member without inventing download causes."""
    listed = identity["listed_ref_ids"]
    snapshot = identity["listing_snapshot"]
    used = set(map(str, embedded_ids))
    if (len(set(listed)) != len(listed) or not used <= set(listed)
            or len(listed) > cap or snapshot["n_refs"] != len(listed)):
        raise ValueError("embedded references do not partition the bounded public listing")
    known_complete = (source == "mapillary" and fetch.get("listing_stats_known") is True
        # The legacy graph fallback does not exhaust paging.next and may stop
        # at a full max-depth page. Its lost==0 is not a completeness proof.
        and fetch.get("listing") == "tiles"
        and type(fetch.get("lost")) is int and fetch["lost"] == 0
        and fetch.get("snapshot", {}).get("refs_sha256") == snapshot["refs_sha256"]
        and fetch.get("snapshot", {}).get("n_refs") == len(listed))
    return {"status": "complete" if known_complete else "unknown", "listing_complete": bool(known_complete),
        "cap": int(cap), "listed_ref_ids": list(listed),
        "listed_ref_latlon": deepcopy(identity["listed_ref_latlon"]),
        "listed_references": [{"id": value.removeprefix(source + ":"),
            "lat": float(point[0]), "lon": float(point[1])}
            for value, point in zip(listed, identity["listed_ref_latlon"])],
        "unserved_references": [{"ref_id": value, "reason": "not_in_embedded_reference_snapshot"}
            for value in listed if value not in used],
        "listing_snapshot_sha256": snapshot["refs_sha256"],
        "scope": "capped public metadata set; only zero-loss tile listings prove completeness; absent embedded rows have unknown acquisition cause"}


def kartaview_vpr_prior(frames_bgr, center, radius_m=3000.0, *,
                        cache_dir=None, n_query=40, device=None,
                        model_name="megaloc", source="kartaview", token=None,
                        cap=1500, query_mode="warp", query_batch_size=None, query_cache=None):
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
    global LAST_FAILURE
    LAST_FAILURE = None
    try:
        out = _prepare_refs_and_query(frames_bgr, center, radius_m, cache_dir,
                                      n_query, device, model_name, source,
                                      token, cap, query_mode=query_mode,
                                      query_batch_size=query_batch_size, query_cache=query_cache)
        if out is None:
            return None
        _idx, sims, ref_xy = out
        top1 = sims.argmax(1)
        maxsim = sims.max(1)
        return _robust_center(ref_xy[top1], maxsim)
    except Exception as e:
        LAST_FAILURE = f"{type(e).__name__}: {e}"
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
    ref_latlon = np.asarray(ref_latlon, dtype=np.float64)
    frame = LocalFrame(float(np.mean(ref_latlon[:, 0])),
                       float(np.mean(ref_latlon[:, 1])))
    xy = np.column_stack(frame.to_xy(ref_latlon[:, 0], ref_latlon[:, 1])
                         ).astype(np.float32)
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
                        cap=1500, sequence_decode=True, query_dt_s=4.0,
                        outer_radius_m=None, outer_cap=None, query_mode="warp",
                        query_batch_size=None, query_cache=None, audit=None,
                        query_timestamps=None):
    """Per-frame VPR positions: a sparse, noisy 'GPS' track to fit the trajectory
    to (anchor-primary v2). Returns ``(query_indices, latlons[N,2], sims[N])`` for
    ``n_query`` frames sampled uniformly across the clip, or ``None``. Unlike the
    single-point prior, this constrains the trajectory's ORIENTATION + start, not
    just its centre. ``source`` selects ``"kartaview"``/``"mapillary"``/
    ``"panoramax"``. ``sequence_decode`` (default on) replaces per-frame argmax
    with the continuity-constrained Viterbi decode; ``query_dt_s`` is the real
    seconds between query frames (sets the transition free radius).
    ``outer_radius_m``/``outer_cap`` add a sparser outer reference ring (see
    ``_prepare_refs_and_query``).
    ``query_mode`` is the unchanged ``warp`` control, or an opt-in ``crops`` /
    ``aspect`` query-only experiment; reference embeddings are reused.
    ``query_cache`` may be an explicitly owned per-video dictionary, allowing
    multiple regional searches to reuse normalized query descriptors.
    The opt-in ``audit`` sink binds actual pixels/descriptors/model state and
    raw decoded-reference cosines. Supply the full input ``query_timestamps``
    grid. Missing/unsupported audit inputs remain incomparable while ordinary
    retrieval still returns its unchanged route.
    """
    global LAST_FAILURE, LAST_TRACK_STATS
    LAST_FAILURE = None
    LAST_TRACK_STATS = {"sequence_decode_requested": bool(sequence_decode), "decoder": "unavailable"}
    if audit is not None:
        audit.clear()
        audit.update(schema="vpr-track-audit-v1", status="pending", issues=[])
        try:
            times = np.asarray(query_timestamps, float)
            if (times.shape != (len(frames_bgr),) or not np.isfinite(times).all()
                    or np.any(np.diff(times) <= 0)):
                raise ValueError("track audit requires the actual increasing input timestamp grid")
        except Exception as error:
            _audit_issue(audit, error)
    try:
        import torch  # noqa: F401 - availability guard; helper re-imports
    except Exception as exc:
        LAST_FAILURE = f"torch unavailable: {exc}"
        LAST_TRACK_STATS["reason"] = LAST_FAILURE
        return None
    try:
        audit_options = {"audit": audit} if audit is not None else {}
        out = _prepare_refs_and_query(frames_bgr, center, radius_m, cache_dir,
                                      n_query, device, model_name, source,
                                      token, cap, outer_radius_m, outer_cap, query_mode,
                                      query_batch_size, query_cache, **audit_options)
        if out is None:
            LAST_TRACK_STATS["reason"] = LAST_FAILURE or "no reference track"
            if audit is not None:
                audit.update(status="unavailable", reason=LAST_TRACK_STATS["reason"])
            return None
        idx, sims, ref_xy = out
        LAST_TRACK_STATS.update(n_references=int(len(ref_xy)), n_queries=int(len(idx)), decoder="argmax")
        top1 = sims.argmax(1)
        if sequence_decode and len(idx) >= 3:
            try:
                top1 = _viterbi_decode(sims, ref_xy, float(query_dt_s))
                LAST_TRACK_STATS["decoder"] = "viterbi"
            except Exception as exc:
                LAST_TRACK_STATS["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        elif sequence_decode:
            LAST_TRACK_STATS["fallback_reason"] = "fewer than three query frames"
        maxsim = sims[np.arange(len(idx)), top1]
        if audit is not None:
            _finish_track_audit(audit, idx, sims, top1, maxsim, query_timestamps,
                                sequence_decode, query_dt_s)
        return idx, ref_xy[top1].astype(float), maxsim.astype(float)
    except Exception as e:
        LAST_FAILURE = f"{type(e).__name__}: {e}"
        LAST_TRACK_STATS["reason"] = LAST_FAILURE
        if audit is not None:
            _audit_issue(audit, e)
        return None


def _finish_track_audit(audit, indices, similarities, chosen, cosines, timestamps,
                        requested, query_dt_s):
    """Finish an opt-in evidence ledger without changing decoded output."""
    try:
        times = np.asarray(timestamps, float)
        if (times.ndim != 1 or len(times) <= int(np.max(indices)) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or not np.isfinite(query_dt_s) or query_dt_s <= 0):
            raise ValueError("audit timestamps/continuity interval are invalid")
        decoder = {"requested": bool(requested), "actual": LAST_TRACK_STATS["decoder"],
                   "query_dt_s": float(query_dt_s), "transition_base_free_m": 30.,
                   "transition_free_speed_mps": 40., "transition_dt_floor_s": .5,
                   "transition_excess_penalty": .02, "maximum_viterbi_references": 6000,
                   "fallback_reason": LAST_TRACK_STATS.get("fallback_reason")}
        audit.update(query_times=times[indices].copy(), decoded_ref_indices=np.asarray(chosen).copy(),
                     decoded_cosines=np.asarray(cosines).copy(), decoder=decoder,
                     score_kind="raw_cosine_at_decoded_reference")
        required = ("query_indices", "query_pixels_sha256", "query_descriptors_sha256",
                    "actual_model_state_sha256", "backbone", "preprocessing", "ref_ids", "ref_latlon",
                    "ref_embeddings_sha256", "raw_similarities")
        if any(key not in audit for key in required):
            raise ValueError("track audit is missing encoder/reference evidence")
        if (np.shape(audit["raw_similarities"]) != np.shape(similarities)
                or not np.array_equal(audit["query_indices"], indices)
                or not np.array_equal(audit["raw_similarities"], similarities)
                or not np.isfinite(similarities).all()
                or np.any(np.abs(similarities) > 1.0001)
                or np.shape(audit["ref_latlon"]) != (np.shape(similarities)[1], 2)
                or len(audit["ref_ids"]) != np.shape(similarities)[1]):
            raise ValueError("track audit matrix/grid differs from the decoded inputs")
        identity = {key: audit[key] for key in ("query_pixels_sha256", "query_descriptors_sha256",
                    "actual_model_state_sha256", "backbone", "preprocessing", "query_mode", "score_kind", "decoder")}
        identity.update(query_indices_sha256=_audit_array_sha(indices),
                        query_times_sha256=_audit_array_sha(times[indices]))
        audit["comparison_identity"] = identity
        audit["comparison_identity_sha256"] = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        audit["status"] = "incomplete" if audit.get("issues") else "complete"
    except Exception as error:
        _audit_issue(audit, error)
