"""Metro-scale multi-source VPR with spatial micro-cell deduplication and corridor Pass 2.

Addresses the three dominant failure modes measured across ``test1``-``test5``
(``docs/search_area_coverage_2026-11.md``, ``docs/failure_taxonomy_2026-11.md``):

1. **Mode A (suburban / commuter drives outside the 8 km city disc, or 3 km
   seed-clamping trap):** Always searches a full metropolitan disc (default
   20 km around the city geocode) with 25 m micro-cell deduplication so
   single-traverse suburban roads receive the same per-road-kilometre sampling
   rate as multi-traverse downtown boulevards. Video-named places (including
   neighbouring municipalities up to 60 km out) and frame seeds are pooled
   **additively** into the same unified Viterbi matrix rather than clamping the
   search to an unrecoverable 3 km gate.
2. **Mode C (highway / commuter drives longer than the single 2 km Pass-2
   disc):** Covers the coherent Pass-1 trajectory with up to ``corridor_max_discs``
   overlapping dense discs and unions them with the corridor's Pass-1
   references, forming a strict local superset along the entire driven route.
3. **Global provider gaps:** Automatically augments thin Mapillary regions with
   open Panoramax and KartaView street-level imagery in the same MegaLoc
   descriptor space.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .geo import geodesic_distance_m
from . import kartaview_vpr as kv


def _coord_key(lat: float, lon: float) -> tuple[int, int]:
    """~0.11 m coordinate identity key to deduplicate overlapping pools."""
    return ( int(round(float(lat) * 1_000_000)), int(round(float(lon) * 1_000_000)) )


def _dedup_pool(embs: list, xys: list[np.ndarray]):
    """Concatenate ``(emb, xy)`` pairs, dropping duplicate reference locations."""
    import torch

    kept_emb = []
    kept_xy = []
    seen: set[tuple[int, int]] = set()
    for emb, xy in zip(embs, xys):
        if emb is None or xy is None or len(xy) == 0:
            continue
        xy_arr = np.asarray(xy, dtype=np.float64)
        mask = []
        for row in range(len(xy_arr)):
            key = _coord_key(xy_arr[row, 0], xy_arr[row, 1])
            if key not in seen:
                seen.add(key)
                mask.append(row)
        if mask:
            idx_t = torch.tensor(mask, dtype=torch.long)
            kept_emb.append(emb.index_select(0, idx_t))
            kept_xy.append(xy_arr[mask])
    if not kept_emb:
        return None, None
    return torch.cat(kept_emb, dim=0), np.vstack(kept_xy)


def _median_smooth_latlon(latlons: np.ndarray, window: int = 5) -> np.ndarray:
    ll = np.asarray(latlons, dtype=np.float64)
    if len(ll) < 3 or window <= 1:
        return ll.copy()
    h = max(int(window) // 2, 1)
    return np.array([
        np.median(ll[max(0, i - h):min(len(ll), i + h + 1)], axis=0)
        for i in range(len(ll))
    ], dtype=np.float64)


def plan_corridor_discs(
    latlons: np.ndarray,
    sims: np.ndarray,
    *,
    radius_m: float = 2500.0,
    max_discs: int = 4,
    min_sep_m: float = 1800.0,
    min_cluster_frames: int = 3,
    cluster_radius_m: float = 1500.0,
) -> list[tuple[float, float]]:
    """Choose up to ``max_discs`` overlapping disc centres covering a Pass-1 track.

    Disc 0 is the robust similarity-weighted geometric median of the track.
    Additional discs are placed greedily on locally coherent track segments
    (at least ``min_cluster_frames`` in a 5-frame neighbourhood within
    ``cluster_radius_m``) that sit farther than ``min_sep_m`` from every
    already-selected disc centre. Short urban drives yield 1 disc; 6-15 km
    arterial/highway drives yield 2-4 overlapping discs spanning the drive.
    """
    ll = np.asarray(latlons, dtype=np.float64).reshape(-1, 2)
    sm_sims = np.asarray(sims, dtype=np.float64).reshape(-1)
    if len(ll) == 0:
        return []
    max_discs = max(1, int(max_discs))
    grid = 0.0001

    def snap(pt: tuple[float, float]) -> tuple[float, float]:
        return (round(float(pt[0]) / grid) * grid, round(float(pt[1]) / grid) * grid)

    c0 = snap(kv._robust_center(ll, sm_sims))
    centres: list[tuple[float, float]] = [c0]
    if max_discs == 1 or len(ll) < min_cluster_frames:
        return centres

    sm = _median_smooth_latlon(ll, window=5)
    coherent_idx: list[int] = []
    for q in range(len(sm)):
        lo, hi = max(0, q - 2), min(len(sm), q + 3)
        d_win = geodesic_distance_m(sm[q, 0], sm[q, 1], sm[lo:hi, 0], sm[lo:hi, 1])
        if int(np.sum(d_win <= cluster_radius_m)) >= min(min_cluster_frames, hi - lo):
            coherent_idx.append(q)
    if not coherent_idx:
        return centres

    cand_pts = sm[coherent_idx]
    eff_max = max_discs
    if max_discs >= 4 and len(cand_pts) >= 2:
        import math
        min_lat, min_lon = np.min(cand_pts, axis=0)
        max_lat, max_lon = np.max(cand_pts, axis=0)
        span_m = float(geodesic_distance_m(min_lat, min_lon, max_lat, max_lon))
        if span_m > 7500.0:
            eff_max = min(8, max(max_discs, int(math.ceil(span_m / 2500.0)) + 1))

    while len(centres) < eff_max:
        min_dists = np.min(
            np.column_stack([
                geodesic_distance_m(c[0], c[1], cand_pts[:, 0], cand_pts[:, 1])
                for c in centres
            ]),
            axis=1,
        )
        best_i = int(np.argmax(min_dists))
        if float(min_dists[best_i]) < float(min_sep_m):
            break
        nxt = snap((float(cand_pts[best_i, 0]), float(cand_pts[best_i, 1])))
        if any(geodesic_distance_m(nxt[0], nxt[1], c[0], c[1]) < 0.5 * min_sep_m for c in centres):
            break
        centres.append(nxt)
    return centres


def _fetch_and_embed_region(
    source: str,
    center: tuple[float, float],
    radius_m: float,
    cache_dir: Path,
    cap: int,
    token: str | None,
    device: str,
    model_name: str,
    *,
    spatial_dedup_m: float | None = None,
    batch_size: int = 24,
    auto_fallback: bool = True,
    fallback_min_refs: int = 200,
) -> tuple[object, np.ndarray | None, dict]:
    """Fetch + embed one region, augmenting with open providers if primary is sparse."""
    cache_dir = Path(cache_dir)
    embs, xys = [], []
    provider_counts: dict[str, int] = {}
    listings: dict[str, dict] = {}

    if source == "union":
        sources = ("mapillary", "panoramax", "kartaview")
    else:
        sources = (source,)

    def run_one(src: str):
        sub_dir = cache_dir if (len(sources) == 1 and src == source) else (cache_dir / src)
        tok = token if src == "mapillary" else None
        if src == "mapillary" and not tok and not kv.has_mapillary_cache(str(sub_dir)):
            return
        kv.LAST_FETCH_STATS = {}
        refs = kv._fetch_refs_for(
            src, center, radius_m, str(sub_dir), cap, tok,
            spatial_dedup_m=spatial_dedup_m,
        )
        if src == "mapillary" and kv.LAST_FETCH_STATS:
            listings[src] = dict(kv.LAST_FETCH_STATS)
        if not refs:
            provider_counts[src] = 0
            return
        e, xy = kv._embed_refs_streaming(
            refs, device, str(sub_dir), model_name=model_name, batch_size=batch_size,
        )
        if e is not None and xy is not None and len(xy):
            embs.append(e)
            xys.append(np.asarray(xy, dtype=np.float64))
            provider_counts[src] = int(len(xy))
        else:
            provider_counts[src] = 0

    for src in sources:
        try:
            run_one(src)
        except Exception as exc:
            print(f"      -> [metro-dense] {src} fetch/embed failed: {type(exc).__name__}: {exc}")
            provider_counts[src] = 0

    total_primary = sum(provider_counts.values())
    if auto_fallback and source != "union" and total_primary < fallback_min_refs:
        for extra_src in ("panoramax", "kartaview"):
            if extra_src in provider_counts:
                continue
            try:
                run_one(extra_src)
            except Exception as exc:
                print(f"      -> [metro-dense] fallback {extra_src} failed: {type(exc).__name__}: {exc}")
                continue

    emb_out, xy_out = _dedup_pool(embs, xys)
    meta = {
        "center": [round(float(center[0]), 5), round(float(center[1]), 5)],
        "radius_m": float(radius_m),
        "cap": int(cap),
        "spatial_dedup_m": spatial_dedup_m,
        "provider_counts": provider_counts,
        "n_unique": 0 if xy_out is None else int(len(xy_out)),
        "listings": listings,
    }
    return emb_out, xy_out, meta


def _shared_metro_dir(
    shared_root: Path,
    source: str,
    center: tuple[float, float],
    radius_m: float,
    cap: int,
    spatial_dedup_m: float | None,
) -> Path:
    ident = json_key = (
        f"{source}:{round(float(center[0]), 4)}:{round(float(center[1]), 4)}:"
        f"{round(float(radius_m), 1)}:{int(cap)}:{spatial_dedup_m}"
    )
    tag = hashlib.sha256(json_key.encode()).hexdigest()[:14]
    return Path(shared_root) / f"metro_{source}_{tag}"


def _decode_exact(sims: np.ndarray, ref_xy: np.ndarray, dt_s: float,
                  device: str) -> tuple[np.ndarray, str]:
    """Exact Viterbi decode for any reference count on CUDA or CPU."""
    import torch

    n_q, n_r = sims.shape
    if n_q < 3:
        return sims.argmax(axis=1).astype(np.int32), "argmax"
    if n_r <= kv._STREAM_MIN_REFS:
        return kv._viterbi_decode(sims, ref_xy, float(dt_s)), "viterbi"
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dec_dev = "cuda" if use_cuda else "cpu"
    blk = 2048 if use_cuda else 512
    return (
        kv._viterbi_decode_torch(sims, ref_xy, float(dt_s), device=dec_dev, block_size=blk),
        f"viterbi_tiled_{dec_dev}",
    )


def run_metro_dense_vpr(
    frames_bgr,
    timestamps,
    *,
    city: str,
    city_center: tuple[float, float],
    seed_around: tuple[float, float, float] | None = None,
    video_title: str | None = None,
    video_description: str | None = None,
    data_dir: Path,
    shared_cache_root: Path | None = None,
    metro_radius_m: float = 20000.0,
    metro_cap: int = 24000,
    spatial_dedup_m: float = 25.0,
    corridor_radius_m: float = 2500.0,
    corridor_max_discs: int = 4,
    corridor_disc_cap: int = 2500,
    n_query: int = 80,
    source: str = "mapillary",
    token: str | None = None,
    device: str | None = "cuda",
    model_name: str = "megaloc",
    query_mode: str = "warp",
    query_batch_size: int = 4,
    query_cache: dict | None = None,
    sequence_decode: bool = True,
    query_dt_s: float = 4.0,
    auto_fallback_sources: bool = True,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, dict]:
    """Unified metro-scale Pass-1 + multi-disc corridor Pass-2 VPR localizer.

    Returns ``(track, diagnostics)`` where ``track`` is ``(idx, latlons, sims)``
    or ``None`` if fewer than 30 references exist across all queried regions.
    """
    import torch
    from .location_prior import resolve_retrieval_regions

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kv._MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    kv._STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    resolved_model = kv._resolve_backbone(model_name, device)
    batch_sz = max(4, int(query_batch_size or 24))

    shared_root = (Path(shared_cache_root)
                   if shared_cache_root is not None
                   else Path(data_dir).parent / "_metro_shared")
    metro_dir = _shared_metro_dir(
        shared_root, source, city_center, metro_radius_m, metro_cap, spatial_dedup_m
    )

    print(
        f"      -> [metro-dense] pass 1: r={metro_radius_m:.0f} m around {city} "
        f"({city_center[0]:.4f}, {city_center[1]:.4f}), cap={metro_cap}, "
        f"dedup={spatial_dedup_m:.0f} m"
    )
    p1_embs, p1_xys = [], []
    region_records = []

    m_emb, m_xy, m_meta = _fetch_and_embed_region(
        source, city_center, metro_radius_m, metro_dir, metro_cap, token,
        device, resolved_model, spatial_dedup_m=spatial_dedup_m,
        batch_size=max(batch_sz, 24), auto_fallback=auto_fallback_sources,
        fallback_min_refs=300,
    )
    m_meta["role"] = "metro_disc"
    region_records.append(m_meta)
    if m_emb is not None and m_xy is not None:
        p1_embs.append(m_emb)
        p1_xys.append(m_xy)

    # Additive seed / title regions: densify named places (including suburbs up
    # to 60 km out like Howwood, Gresham, Podzamcze) WITHOUT gating out the
    # 20 km metro pool when a seed is partial or noisy.
    extra_targets: list[tuple[str, tuple[float, float], float]] = []
    if seed_around is not None:
        extra_targets.append((
            "coarse_seed",
            (float(seed_around[0]), float(seed_around[1])),
            float(np.clip(seed_around[2], 2500.0, 5000.0)),
        ))
    if video_title or video_description:
        try:
            named = resolve_retrieval_regions(
                city, video_title, video_description,
                center=city_center, radius_m=3500.0, max_regions=6,
                max_km_from_city=60.0,
            )
            for reg in named:
                if reg.get("origin") == "video_named":
                    rc = tuple(map(float, reg["center"]))
                    extra_targets.append((str(reg.get("name", "named")), rc, 3500.0))
        except Exception as exc:
            print(f"      -> [metro-dense] named-region lookup skipped ({exc})")

    deduped_targets: list[tuple[str, tuple[float, float], float, int]] = []
    inner_kept = 0
    for label, rc, r_m in extra_targets:
        if any(geodesic_distance_m(rc[0], rc[1], prev[1][0], prev[1][1]) < 1500.0
               for prev in deduped_targets):
            continue
        dist_city = float(geodesic_distance_m(city_center[0], city_center[1], rc[0], rc[1]))
        is_outer = dist_city > 0.65 * float(metro_radius_m)
        if not is_outer:
            if inner_kept >= 1:
                continue
            inner_kept += 1
        seed_cap = 2000 if is_outer else 1500
        deduped_targets.append((label, rc, r_m, seed_cap))
        if len(deduped_targets) >= 4:
            break

    for idx_extra, (label, rc, r_m, seed_cap) in enumerate(deduped_targets):
        tag = hashlib.sha256(f"{label}:{rc[0]:.4f}:{rc[1]:.4f}:{r_m:.0f}:{seed_cap}".encode()).hexdigest()[:12]
        sub_dir = Path(data_dir) / f"{source}_metro_seeds" / f"seed_{tag}"
        print(f"      -> [metro-dense] additive region {idx_extra + 1}/{len(deduped_targets)}: "
              f"{label} ({rc[0]:.4f}, {rc[1]:.4f}) r={r_m:.0f} m cap={seed_cap}")
        s_emb, s_xy, s_meta = _fetch_and_embed_region(
            source, rc, r_m, sub_dir, seed_cap, token, device, resolved_model,
            spatial_dedup_m=15.0, batch_size=max(batch_sz, 24),
            auto_fallback=auto_fallback_sources, fallback_min_refs=80,
        )
        s_meta["role"] = f"additive:{label}"
        region_records.append(s_meta)
        if s_emb is not None and s_xy is not None:
            p1_embs.append(s_emb)
            p1_xys.append(s_xy)

    p1_emb, p1_xy = _dedup_pool(p1_embs, p1_xys)
    diagnostics: dict = {
        "mode": "metro_dense",
        "backbone": kv.backbone_info(),
        "metro_radius_m": float(metro_radius_m),
        "metro_cap": int(metro_cap),
        "spatial_dedup_m": float(spatial_dedup_m),
        "pass1_regions": region_records,
        "pass1_n_references": 0 if p1_xy is None else int(len(p1_xy)),
    }
    if p1_emb is None or p1_xy is None or len(p1_xy) < 30:
        diagnostics["status"] = "unavailable"
        diagnostics["reason"] = (
            f"only {0 if p1_xy is None else len(p1_xy)} references across metro pool (need 30)"
        )
        return None, diagnostics

    q_idx = np.linspace(0, len(frames_bgr) - 1, min(int(n_query), len(frames_bgr))).astype(int)
    q_opts = {"query_cache": query_cache, "model_name": resolved_model} if query_cache is not None else {}
    sims_p1 = kv._query_similarities(
        frames_bgr, q_idx, p1_emb, device, mode=query_mode,
        batch_size=query_batch_size, **q_opts,
    )
    if sequence_decode:
        path_p1, dec_p1 = _decode_exact(sims_p1, p1_xy, float(query_dt_s), device)
    else:
        path_p1, dec_p1 = sims_p1.argmax(axis=1).astype(np.int32), "argmax"

    p1_track_xy = p1_xy[path_p1].astype(float)
    p1_track_sims = sims_p1[np.arange(len(q_idx)), path_p1].astype(float)
    p1_center = kv._robust_center(p1_track_xy, p1_track_sims)
    p1_med_sim = float(np.median(p1_track_sims)) if len(p1_track_sims) else 0.0

    # When primary Mapillary Pass-1 confidence is weak (< 0.150, e.g. local
    # Mapillary desert in US Sunbelt / Asia / LatAm suburbs), augment the metro
    # pool with open Panoramax and KartaView imagery even if downtown Mapillary
    # had >= 300 references.
    low_sim_augment = bool(auto_fallback_sources and source == "mapillary" and p1_med_sim < 0.150)
    if low_sim_augment:
        aug_added = 0
        n_prev_chunks = len(p1_embs)
        for extra_src, extra_cap in (("panoramax", 5000), ("kartaview", 3000)):
            if (m_meta.get("provider_counts") or {}).get(extra_src, 0) > 0:
                continue
            extra_dir = metro_dir / extra_src
            e_emb, e_xy, e_meta = _fetch_and_embed_region(
                extra_src, city_center, metro_radius_m, extra_dir, extra_cap, None,
                device, resolved_model, spatial_dedup_m=spatial_dedup_m,
                batch_size=max(batch_sz, 24), auto_fallback=False,
            )
            e_meta["role"] = f"low_sim_open:{extra_src}"
            region_records.append(e_meta)
            if e_emb is not None and e_xy is not None and len(e_xy):
                p1_embs.append(e_emb)
                p1_xys.append(e_xy)
                aug_added += int(len(e_xy))
        if aug_added > 0:
            aug_emb, aug_xy = _dedup_pool(p1_embs, p1_xys)
            sims_aug = kv._query_similarities(
                frames_bgr, q_idx, aug_emb, device, mode=query_mode,
                batch_size=query_batch_size, **q_opts,
            )
            if sequence_decode:
                path_aug, dec_aug = _decode_exact(sims_aug, aug_xy, float(query_dt_s), device)
            else:
                path_aug, dec_aug = sims_aug.argmax(axis=1).astype(np.int32), "argmax"
            aug_track_xy = aug_xy[path_aug].astype(float)
            aug_track_sims = sims_aug[np.arange(len(q_idx)), path_aug].astype(float)
            aug_med_sim = float(np.median(aug_track_sims)) if len(aug_track_sims) else 0.0
            primary_sparse = int((m_meta.get("provider_counts") or {}).get(source, 0)) < 5000
            if primary_sparse or aug_med_sim >= p1_med_sim + 0.005:
                p1_emb, p1_xy = aug_emb, aug_xy
                sims_p1, path_p1, dec_p1 = sims_aug, path_aug, dec_aug
                p1_track_xy, p1_track_sims = aug_track_xy, aug_track_sims
                p1_center = kv._robust_center(p1_track_xy, p1_track_sims)
                p1_med_sim = aug_med_sim
                print(f"      -> [metro-dense] accepted open-provider augmentation "
                      f"(+{aug_added} refs, sim {p1_med_sim:.3f}, pool={len(p1_xy)})")
            else:
                # Keep primary p1_embs/p1_xys un-diluted when Mapillary is already dense
                # and open imagery did not raise median similarity by >= 0.005.
                del p1_embs[n_prev_chunks:]
                del p1_xys[n_prev_chunks:]
                p1_emb, p1_xy = _dedup_pool(p1_embs, p1_xys)

    diagnostics["pass1"] = {
        "decoder": dec_p1,
        "n_references": int(len(p1_xy)),
        "n_queries": int(len(q_idx)),
        "center": [float(p1_center[0]), float(p1_center[1])],
        "median_sim": round(p1_med_sim, 4),
    }

    # Step 2: Multi-disc corridor refinement along the Pass-1 track, unioned
    # with local Pass-1 references so Pass 2 is a strict superset on the route.
    corridor_centres = plan_corridor_discs(
        p1_track_xy, p1_track_sims,
        radius_m=corridor_radius_m,
        max_discs=corridor_max_discs,
    )
    print(
        f"      -> [metro-dense] pass 2: {len(corridor_centres)} corridor disc(s) "
        f"(r={corridor_radius_m:.0f} m, cap={corridor_disc_cap}/disc) along pass-1 track"
    )
    c_embs, c_xys = [], []
    # Retain all Pass-1 references near any corridor disc so Pass 2 cannot lose
    # already-matched references.
    if corridor_centres:
        near_mask = np.zeros(len(p1_xy), dtype=bool)
        keep_r = float(corridor_radius_m) + 1000.0
        for cc in corridor_centres:
            near_mask |= geodesic_distance_m(cc[0], cc[1], p1_xy[:, 0], p1_xy[:, 1]) <= keep_r
        if near_mask.any():
            idx_near = torch.tensor(np.flatnonzero(near_mask), dtype=torch.long)
            c_embs.append(p1_emb.index_select(0, idx_near))
            c_xys.append(p1_xy[near_mask])

    corridor_fb_min = (corridor_disc_cap + 1) if low_sim_augment else 50
    corridor_records = []
    for c_i, cc in enumerate(corridor_centres):
        c_tag = hashlib.sha256(
            f"{source}:{cc[0]:.4f}:{cc[1]:.4f}:{corridor_radius_m:.0f}:{corridor_disc_cap}:8.0".encode()
        ).hexdigest()[:12]
        c_dir = shared_root / f"corridor_{source}_{c_tag}"
        ce, cxy, cmeta = _fetch_and_embed_region(
            source, cc, corridor_radius_m, c_dir, corridor_disc_cap, token,
            device, resolved_model, spatial_dedup_m=8.0,
            batch_size=max(batch_sz, 24), auto_fallback=auto_fallback_sources,
            fallback_min_refs=corridor_fb_min,
        )
        cmeta["disc_index"] = c_i
        corridor_records.append(cmeta)
        if ce is not None and cxy is not None:
            c_embs.append(ce)
            c_xys.append(cxy)

    # Retain local Pass-1 references near seed_around before freeing p1_emb in
    # case low-cosine night-drive seed recovery is needed below.
    seed_local_pair = None
    if seed_around is not None:
        s_lat, s_lon, s_rad = float(seed_around[0]), float(seed_around[1]), float(seed_around[2])
        s_mask = geodesic_distance_m(s_lat, s_lon, p1_xy[:, 0], p1_xy[:, 1]) <= max(s_rad, 3500.0) + 1000.0
        if s_mask.any():
            idx_s = torch.tensor(np.flatnonzero(s_mask), dtype=torch.long)
            seed_local_pair = (p1_emb.index_select(0, idx_s), p1_xy[s_mask])

    # Free the large Pass-1 descriptor tensor before Pass-2 similarity matmul.
    del p1_emb
    p2_emb, p2_xy = _dedup_pool(c_embs, c_xys)
    if p2_emb is None or p2_xy is None or len(p2_xy) < 30:
        diagnostics["status"] = "ok"
        diagnostics["kept"] = "pass-1 (corridor unavailable)"
        diagnostics["corridor_discs"] = corridor_records
        return (q_idx, p1_track_xy, p1_track_sims), diagnostics

    sims_p2 = kv._query_similarities(
        frames_bgr, q_idx, p2_emb, device, mode=query_mode,
        batch_size=query_batch_size, **q_opts,
    )
    if sequence_decode:
        path_p2, dec_p2 = _decode_exact(sims_p2, p2_xy, float(query_dt_s), device)
    else:
        path_p2, dec_p2 = sims_p2.argmax(axis=1).astype(np.int32), "argmax"

    p2_track_xy = p2_xy[path_p2].astype(float)
    p2_track_sims = sims_p2[np.arange(len(q_idx)), path_p2].astype(float)
    p2_center = kv._robust_center(p2_track_xy, p2_track_sims)
    p2_med_sim = float(np.median(p2_track_sims)) if len(p2_track_sims) else 0.0

    keep_p2 = p2_med_sim >= p1_med_sim - 0.02
    kept_label = "pass-2-corridor" if keep_p2 else "pass-1 (corridor sim lower)"
    final_xy = p2_track_xy if keep_p2 else p1_track_xy
    final_sims = p2_track_sims if keep_p2 else p1_track_sims
    final_center = p2_center if keep_p2 else p1_center
    final_med_sim = p2_med_sim if keep_p2 else p1_med_sim

    # Step 3: Corroborated-seed recovery on low-cosine (e.g. night) drives.
    # When visual appearance cosines stay weak across the entire metro (< 0.155,
    # typical of pitch-black night footage against daytime street imagery) AND
    # a corroborated multi-vote OCR/plate/title seed exists that the noisy wide
    # decode drifted away from, evaluate a dense local seed corridor and keep it
    # unless the distant track clearly beats it by > 0.025 cosine.
    if seed_around is not None and final_med_sim < 0.155:
        s_lat, s_lon, s_rad = float(seed_around[0]), float(seed_around[1]), float(seed_around[2])
        drift_m = float(geodesic_distance_m(final_center[0], final_center[1], s_lat, s_lon))
        if drift_m > s_rad + 1500.0:
            r_seed_dense = float(np.clip(s_rad, 3500.0, 6000.0))
            s_tag = hashlib.sha256(
                f"{source}:night_seed:{s_lat:.4f}:{s_lon:.4f}:{r_seed_dense:.0f}:6000:10.0".encode()
            ).hexdigest()[:12]
            s_dir = Path(data_dir) / f"{source}_metro_seeds" / f"dense_seed_{s_tag}"
            se, sxy, smeta = _fetch_and_embed_region(
                source, (s_lat, s_lon), r_seed_dense, s_dir, 6000, token,
                device, resolved_model, spatial_dedup_m=10.0,
                batch_size=max(batch_sz, 24), auto_fallback=auto_fallback_sources,
                fallback_min_refs=200,
            )
            seed_embs, seed_xys = [], []
            if seed_local_pair is not None:
                seed_embs.append(seed_local_pair[0])
                seed_xys.append(seed_local_pair[1])
            if se is not None and sxy is not None:
                seed_embs.append(se)
                seed_xys.append(sxy)
            seed_pool_emb, seed_pool_xy = _dedup_pool(seed_embs, seed_xys)
            if seed_pool_emb is not None and seed_pool_xy is not None and len(seed_pool_xy) >= 30:
                sims_seed = kv._query_similarities(
                    frames_bgr, q_idx, seed_pool_emb, device, mode=query_mode,
                    batch_size=query_batch_size, **q_opts,
                )
                if sequence_decode:
                    path_seed, dec_seed = _decode_exact(sims_seed, seed_pool_xy, float(query_dt_s), device)
                else:
                    path_seed, dec_seed = sims_seed.argmax(axis=1).astype(np.int32), "argmax"
                s_track_xy = seed_pool_xy[path_seed].astype(float)
                s_track_sims = sims_seed[np.arange(len(q_idx)), path_seed].astype(float)
                s_med_sim = float(np.median(s_track_sims)) if len(s_track_sims) else 0.0
                s_center = kv._robust_center(s_track_xy, s_track_sims)
                accepted_seed = s_med_sim >= final_med_sim - 0.025
                diagnostics["seed_recovery"] = {
                    "triggered": True,
                    "drift_m": round(drift_m, 1),
                    "seed_radius_m": r_seed_dense,
                    "n_references": int(len(seed_pool_xy)),
                    "decoder": dec_seed,
                    "seed_median_sim": round(s_med_sim, 4),
                    "global_median_sim": round(final_med_sim, 4),
                    "accepted": accepted_seed,
                }
                if accepted_seed:
                    final_xy, final_sims = s_track_xy, s_track_sims
                    p2_center, p2_med_sim = s_center, s_med_sim
                    kept_label = "seed-corridor (low-cosine recovery)"
                    print(
                        f"      -> [metro-dense] low-cosine ({final_med_sim:.3f}) drifted "
                        f"{drift_m:.0f} m from corroborated seed; recovered seed-corridor "
                        f"(sim {s_med_sim:.3f}, refs={len(seed_pool_xy)})"
                    )

    diagnostics.update(
        status="ok",
        kept=kept_label,
        corridor_discs=corridor_records,
        pass2={
            "decoder": dec_p2,
            "n_references": int(len(p2_xy)),
            "n_corridor_discs": len(corridor_centres),
            "center": [float(p2_center[0]), float(p2_center[1])],
            "median_sim": round(p2_med_sim, 4),
        },
    )
    print(
        f"      -> [metro-dense] kept {kept_label} "
        f"(sim p1 {p1_med_sim:.3f} / p2 {p2_med_sim:.3f}, "
        f"p2 refs={len(p2_xy)})"
    )
    return (q_idx, final_xy, final_sims), diagnostics
