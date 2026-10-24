"""Stage 4a: absolute coarse priors that locate the clip independently of shape.

* a GPS-free frame-based coarse seed (licence-plate district / OCR place names /
  optional VLM) that bounds the search when no disc was given;
* blind VPR (Mapillary / KartaView street photos + MegaLoc), the strongest
  shape-independent selection signal, with optional coarse-to-fine refinement;
* the licence-plate registration-district anchor;
* the VLM district/landmark prior (a coverage fallback when VPR finds nothing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geo import geodesic_distance_m
from .context import _record_channel, RunContext


# Coarse-to-fine pass-1 city-extent sizing (deployable, city-name-only mode):
# cap on the enlarged coarse VPR disc, and the ref cap used when it is enlarged.
# 8 km covers the observed peripheral drives (Málaga 5.4 km, Vaughan 4.2 km from
# the centroid) with margin while diluting less than the full municipality bbox
# (which reaches 12-37 km incl. rural/coastal fringe): Málaga 10 km -> 16 m vs
# 8 km -> 11 m, and the central-drive Ulm regression shrinks too.
_PASS1_COVER_MAX_M = 8000.0
_PASS1_COVER_CAP = 3000
# With a Mapillary TOKEN every coarse-pass ref is a real image download, so
# a 3000-ref city-extent disc times out; dense coverage needs far fewer to
# find the right district (the FINE pass still uses full density on its tight
# disc). Cap the tokened coarse pass here so un-seeded/mega-city runs finish.
_PASS1_TOKEN_CAP = 1200
# Pass-2 is kept as the placement prior only if its median top-1 VPR similarity
# is within this margin of pass-1's (else pass 1 already localized better;
# audit deepdive C2F-1 — London pass-1 0.10 km was silently overwritten).
_C2F_SIM_MARGIN = 0.02
# Graph-disc radius slack: covers the ~11 m centre snap plus a little margin so
# a snapped disc never clips a route endpoint against the tight margin (audit
# F2). (Was 400 m for the old ~500 m snap; the snap is now ~11 m.)
_C2F_SNAP_TOL_M = 50.0


# OCR place names must agree before they may seed the search disc. The seed
# becomes a hard gate on both the VPR fetch and the road graph, so a false
# seed is unrecoverable, while a missed one only falls back to the whole city.
# Observed: rapidocr read "KA.GC" (a Karlsruhe licence-plate fragment) on
# KITTI 0009; geocoded as "KA.GC, Karlsruhe" it landed 15 km outside the city
# and the run finished 18.3 km off (city centroid: 2.6 km). Same two-vote rule
# as the plate seed.
_OCR_SEED_MIN_PLACES = 2


def _city_extent_reach_m(city: str, cache_path) -> float | None:
    """How far from the city's geocode centroid its OSM bounding box reaches:
    dist(centroid, bbox centre) + half-diagonal (triangle inequality)."""
    from ..text_anchor import city_extent_radius, default_geocode_fn
    ext = city_extent_radius(city, cache_path)
    cen = default_geocode_fn(cache_path)(city) if ext is not None else None
    if ext is None or cen is None:
        return None
    return geodesic_distance_m(cen[0], cen[1], ext[0], ext[1]) + float(ext[2])


def _city_centre(city: str, cache_path):
    from ..text_anchor import default_geocode_fn
    return default_geocode_fn(cache_path)(city)


def _seed_inside_city(lat: float, lon: float, radius_m: float, city: str, cache_dir) -> bool:
    """Does a frame-derived seed disc overlap the city the user named?

    Every seed is a hard gate, and the user said which city it is: a seed
    outside it contradicts the input. Observed in benchmark run v3: on a
    Detroit clip the licence-plate reader matched the German district code "D"
    (Duesseldorf) and sent the search 6,500 km away. Unknown city extent:
    refuse, as for OCR seeds."""
    if cache_dir is None:
        return False
    cache_path = Path(cache_dir) / "geocode_cache.json"
    try:
        reach = _city_extent_reach_m(city, cache_path)
        cen = _city_centre(city, cache_path) if reach is not None else None
    except Exception:
        return False
    if reach is None or cen is None:
        return False
    return geodesic_distance_m(cen[0], cen[1], lat, lon) <= float(reach) + float(radius_m)


def _frame_coarse_seed(video_path, frames, city: str, *, want_vlm: bool = False,
                       vo_start: float = 0.0, vo_end: float | None = None,
                       ocr_engine: str = "rapidocr", cache_dir=None):
    """GPS-free coarse seed from the video FRAMES. Fast deterministic signals
    first — license-plate registration district, then legible place names
    read by OCR (geocoded through the same resolver as the title) — and only
    if ``want_vlm`` a slow VLM read of the scene. Returns
    ``(lat, lon, radius_m, source)`` or None; every backend is best-effort."""
    try:                                       # 1) license-plate district
        from ..plate_anchor import plate_district_anchor
        pa = plate_district_anchor(str(video_path))
        if pa is not None:
            rad = max(1500.0, float(getattr(pa, "radius_m", 0.0) or 0.0))
            if _seed_inside_city(float(pa.lat), float(pa.lon), rad, city, cache_dir):
                return (float(pa.lat), float(pa.lon), rad,
                        f"plate:{getattr(pa, 'district', '?')}")
            print(f"      -> [coarse] plate district {getattr(pa, 'district', '?')} lies "
                  f"outside {city}; ignored")
    except Exception as e:
        print(f"      -> [coarse] plate seed unavailable ({e})")
    try:                                       # 2) OCR place names -> geocode
        from ..location_prior import resolve_coarse_prior
        from ..scene_text import extract_scene_text
        st = extract_scene_text(video_path, start_sec=vo_start, end_sec=vo_end,
                                engine=ocr_engine)
        ocr_text = " , ".join(s.text for s in st if getattr(s, "text", None))
        # Scene text is noisy (plates, shop signs, direction signs naming
        # OTHER places), unlike an uploader's title. Accept only names inside
        # the named city's own extent - the user said which city it is - and
        # require two of them (_OCR_SEED_MIN_PLACES). Unknown extent: refuse
        # rather than fall back to the title resolver's 60 km.
        reach_m = None
        if cache_dir is not None:
            try:
                reach_m = _city_extent_reach_m(city, Path(cache_dir) / "geocode_cache.json")
            except Exception as e:
                print(f"      -> [coarse] city extent unavailable ({e})")
        from .length_prior import _auto_estimated_length_m
        seg_s = (vo_end - vo_start) if vo_end is not None else None
        pr = (resolve_coarse_prior(city, ocr_text, None,
                                   max_km_from_city=reach_m / 1000.0,
                                   min_places=_OCR_SEED_MIN_PLACES,
                                   route_length_m=(_auto_estimated_length_m(seg_s)
                                                   if seg_s else None))
              if reach_m is not None else None)
        if pr is not None:
            return (pr[0], pr[1], pr[2],
                    f"ocr:{','.join(pr[3][:3])}")
    except Exception as e:
        print(f"      -> [coarse] OCR seed unavailable ({e})")
    if want_vlm:                               # 3) VLM geoguess (slow; opt-in)
        try:
            from ..vlm_anchor import vlm_district_anchor
            va = vlm_district_anchor(frames, city)
            if va is not None:
                rad = max(1500.0, float(getattr(va, "radius_m", 0.0) or 2500.0))
                if _seed_inside_city(float(va.lat), float(va.lon), rad, city, cache_dir):
                    return (float(va.lat), float(va.lon), rad, "vlm")
                print(f"      -> [coarse] VLM guess lies outside {city}; ignored")
        except Exception as e:
            print(f"      -> [coarse] VLM seed unavailable ({e})")
    return None


def coarse_seed_from_frames(ctx: RunContext) -> None:
    """GPS-free coarse seed (plate district / OCR / VLM) -> ``ctx.osm_around``.

    Bounds the graph + VPR near the drive when no disc and no title-derived
    prior were given.
    """
    cfg = ctx.cfg
    video_path = ctx.video_path
    frames = ctx.frames

    ctx.osm_around = cfg.osm_around
    ctx.frame_seed_source = None
    # GPS-free frame-based coarse seed (deployable): plate district / VLM read
    # of the scene, bounding the graph + VPR near the drive when no disc and
    # no title-derived prior were given.
    if ctx.osm_around is None and cfg.coarse_from_frames:
        _fp = _frame_coarse_seed(video_path, frames.frames, cfg.city,
                                 cache_dir=cfg.data_dir,
                                 want_vlm=cfg.coarse_from_vlm,
                                 vo_start=cfg.vo_start_sec,
                                 vo_end=cfg.vo_end_sec,
                                 ocr_engine=cfg.ocr_engine)
        if _fp is not None:
            ctx.osm_around = _fp[:3]
            ctx.frame_seed_source = str(_fp[3])
            print(f"      -> [coarse] frame-based prior "
                  f"{_fp[0]:.5f},{_fp[1]:.5f} r={_fp[2]:.0f} m ({_fp[3]})")


#: Free GPU memory (GB) below which retrieval runs on the CPU instead. MegaLoc
#: needs about 2 GB for weights plus activations for a batch of references.
VPR_MIN_FREE_VRAM_GB = 4.0


def select_retrieval_device(min_free_gb: float = VPR_MIN_FREE_VRAM_GB) -> str:
    """"cuda" when a GPU with enough free memory exists, else "cpu"."""
    try:
        import torch as _torch

        if not _torch.cuda.is_available():
            return "cpu"
        free_gb = _torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        return "cpu"
    if free_gb >= min_free_gb:
        return "cuda"
    print(f"      -> only {free_gb:.1f} GB GPU memory free (< {min_free_gb:.0f} GB); "
          f"running retrieval on the CPU (same model, slower)")
    return "cpu"


def vpr_coarse_prior(ctx: RunContext) -> None:
    """Blind VPR coarse prior (street-photo retrieval) + optional coarse-to-fine.

    Sets ``vpr_center`` / ``vpr_track`` / ``anchor_origin`` / ``vpr_source_used``
    (and may re-centre ``osm_around`` on the refined centre). Used as a re-rank
    centre and the anchor-primary placement prior; gating the OSM graph to its
    disc was refuted by experiment (664 -> 1276 m on Ulm).
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames
    estimated_length_m = ctx.estimated_length_m

    # 4a. Optional blind VPR coarse prior (KartaView + EigenPlaces): retrieve the
    # frames against GPS-tagged street photos to locate the clip INDEPENDENTLY of
    # trajectory shape — the fix for the SELECTION wall. ~53 m prior on Ulm 4K
    # (monocular_osm/kartaview_vpr.py). Used as a re-rank centre + the anchor-primary
    # placement prior; gating the OSM graph to its disc was refuted (664->1276 m).
    ctx.vpr_center = None
    ctx.vpr_track = None
    ctx.c2f_info = None      # coarse-to-fine record, attached to result later
    ctx.vpr_listing = None   # Mapillary listing tile counts per pass
    ctx.anchor_origin = None   # which channel produced vpr_center: vpr|vlm
    ctx.vpr_source_used = None  # effective reference source after fallbacks
    # VPR runs whenever enabled — it re-ranks + drives anchor-primary, so it is
    # NOT conditioned on osm_around (mega-cities pass --osm-around AND want VPR).
    if cfg.use_vpr_prior:
        _vpr_src = cfg.vpr_source
        print(f"[4a] VPR coarse prior ({_vpr_src.capitalize()} + MegaLoc)")
        try:
            import os as _os

            import osmnx as ox
            from ..kartaview_vpr import kartaview_vpr_track, _robust_center
            # Prefer an explicit --osm-around centre (robust to Nominatim
            # geocode failures / ambiguous small-town names); else geocode.
            center = ((ctx.osm_around[0], ctx.osm_around[1]) if ctx.osm_around
                      else ox.geocode(cfg.city))
            # Device: only request CUDA when it is actually available;
            # None lets the module fall back to CPU inference (slow but
            # working) instead of dying inside torch.hub and masquerading
            # as "VPR unavailable" on CPU-only machines.
            # A GPU that another process has nearly filled is worse than none:
            # the retrieval model then dies natively (the NVIDIA driver logs a
            # graphics exception, no Python traceback). Below the floor, run
            # the same model on the CPU (~0.8 s/image) and say so.
            _vpr_device = select_retrieval_device()
            _vpr_token = (_os.environ.get("MLY_TOKEN")
                          if _vpr_src in ("mapillary", "union") else None)
            if _vpr_src == "mapillary" and not _vpr_token:
                # A warm ref cache serves without the API, so only fall back
                # when there is no cache either — offline regression sweeps
                # (cached refs + cached VO) must not silently switch source.
                from ..kartaview_vpr import has_mapillary_cache
                if has_mapillary_cache(str(cfg.data_dir / _vpr_src)):
                    # Existence-only check: the fetch itself still verifies
                    # the cache SIGNATURE (centre/radius/cap) — if the query
                    # moved since the cache was written, refs come back empty
                    # and the channel degrades to "VPR unavailable" below.
                    print("      -> no MLY_TOKEN; a warm Mapillary cache "
                          "exists and will serve IF it covers this query")
                else:
                    print("      -> mapillary source needs MLY_TOKEN env var; "
                          "falling back to kartaview")
                    _vpr_src = "kartaview"
            # Never search VPR wider than the OSM graph disc: on SPARSE-coverage
            # areas (e.g. KITTI Karlsruhe) a 3 km disc dilutes the few on-route
            # refs under the ref cap and the robust-centre drifts off-route
            # (verified: KITTI r3000 prior 1.3 km off -> anchor 1155 m; r968
            # -> prior on-route -> anchor start 517->95 m). Dense areas
            # (London/comma) are unaffected. Floor 800 m to keep enough refs.
            _vpr_radius = cfg.vpr_search_radius_m
            _vpr_cap_eff = cfg.vpr_ref_cap
            if ctx.osm_around is not None:
                _vpr_radius = min(_vpr_radius, max(800.0, float(ctx.osm_around[2])))
            elif cfg.vpr_coarse_to_fine:
                # City-name-only + coarse-to-fine (the deployable mode): size
                # the COARSE pass to cover the whole municipality so a drive in
                # a peripheral district is INSIDE the search disc. A centroid +
                # 3 km default misses a drive 5 km out, forcing pass-1 to match
                # the wrong dense area (verified: Málaga 5169 m -> 11 m, Vaughan
                # 6413 m, once the disc covers the drive). Capped to bound
                # dilution; the ref cap is bumped so the bigger disc keeps
                # enough on-route refs. Coarse-to-fine then tightens. Uses the
                # OSM place polygon, NOT ground truth.
                try:
                    from ..text_anchor import city_extent_radius as _cext
                    _ext = _cext(cfg.city, cfg.data_dir / "geocode_cache.json")
                except Exception as e:
                    # A sizing hiccup must cost only the enlargement, never
                    # the whole VPR channel (the enclosing try's except would
                    # otherwise skip the fetch entirely).
                    _ext = None
                    print(f"      -> city-extent sizing failed ({e}); "
                          f"keeping the default pass-1 disc")
                if _ext is not None:
                    # The disc is centred on the GEOCODE CENTROID, but the
                    # extent is measured about the BBOX centre — cover the
                    # bbox from the centroid: dist(centroid, bbox centre) +
                    # half-diagonal (triangle inequality).
                    _dc = geodesic_distance_m(center[0], center[1],
                                              _ext[0], _ext[1])
                    _cover = float(np.clip(_dc + _ext[2],
                                           cfg.vpr_search_radius_m,
                                           _PASS1_COVER_MAX_M))
                    if _cover > _vpr_radius + 1.0:
                        _vpr_radius = _cover
                        _vpr_cap_eff = max(cfg.vpr_ref_cap, _PASS1_COVER_CAP)
                        if _vpr_token:      # dense downloads: keep it bounded
                            _vpr_cap_eff = min(_vpr_cap_eff, _PASS1_TOKEN_CAP)
                        print(f"      -> coarse-to-fine: sizing pass-1 disc to "
                              f"the city extent ({_vpr_radius:.0f} m, "
                              f"cap {_vpr_cap_eff})")
                        if _vpr_src == "mapillary" and not _vpr_token:
                            # The enlarged radius/cap changes the fetch
                            # signature: a warm (default-sized) cache will NOT
                            # serve this query, and tokenless the channel then
                            # degrades to "VPR unavailable" — say why once.
                            print("      -> note: enlarged disc has a new "
                                  "cache signature; tokenless runs need one "
                                  "MLY_TOKEN run to rewarm the pass-1 cache")
            # Real seconds between the ~80 query frames — sets the Viterbi
            # transition free radius (sequence decode kills the confident-
            # but-wrong retrievals at the source; offline A/B: median better
            # on all 5 GT clips, p90 tail collapses ~5-8x).
            _ts = np.asarray(frames.timestamps, dtype=np.float64)
            _qn = max(min(80, len(frames.frames)) - 1, 1)
            _qdt = float((_ts[-1] - _ts[0]) / _qn) if len(_ts) >= 2 else 4.0
            tr = kartaview_vpr_track(frames.frames, center,
                                     radius_m=_vpr_radius,
                                     cache_dir=str(cfg.data_dir / _vpr_src),
                                     device=_vpr_device,
                                     source=_vpr_src, token=_vpr_token,
                                     cap=_vpr_cap_eff,
                                     sequence_decode=cfg.vpr_viterbi,
                                     query_dt_s=_qdt)
            from .. import kartaview_vpr as _kv
            if _vpr_src == "mapillary" and _kv.LAST_FETCH_STATS:
                ctx.vpr_listing = {"pass1": dict(_kv.LAST_FETCH_STATS)}
            if tr is not None:
                ctx.vpr_track = tr
                v_idx, v_ll, v_sims = tr
                # Robust centre (conf-threshold + weighted geometric median +
                # spatial-MAD outlier rejection) for the re-rank channel + guard;
                # the per-frame track drives the anchor-primary placement below.
                # VPR is route-accurate (~91 m to the GT route on Ulm) — better
                # than the matcher's own pick. Re-rank, don't gate (gating its
                # disc hurt end-to-end: 664 -> 1276 m).
                ctx.vpr_center = _robust_center(v_ll, v_sims)
                ctx.anchor_origin = "vpr"
                ctx.vpr_source_used = _vpr_src
                print(f"      -> VPR prior {ctx.vpr_center[0]:.5f}, {ctx.vpr_center[1]:.5f} "
                      f"({len(v_idx)}-frame track); re-rank + anchor-primary fit")
                # [4a2] COARSE-TO-FINE refine: when pass 1 ran over a WIDE disc
                # (a coarse deployable seed), its robust centre is much tighter
                # than the seed. Re-fetch a TIGHT disc around it and re-run VPR
                # (own cache subdir so both passes stay warm), then re-centre
                # the OSM graph + downstream priors on the refined centre by
                # updating osm_around before the graph fetch below. Safe: a
                # wrong pass-1 centre (no-signal clip) only re-tightens around
                # the wrong spot — it was already lost, and this never crashes.
                # Tight radius floored by the route-length prior: the robust
                # centre sits near the route's spatial MIDDLE, so endpoints
                # lie up to ~length/2 away — a disc smaller than that clips
                # the graph below the route extent and no candidate walk can
                # span the clip (long/fast clips would silently degrade).
                _tight = max(float(cfg.vpr_c2f_radius_m),
                             0.6 * float(estimated_length_m))
                if cfg.vpr_coarse_to_fine and _vpr_radius <= 1.2 * _tight:
                    # Enabled but not fired — say why (a long route can floor
                    # _tight above the pass-1 disc, silently skipping pass 2).
                    print(f"      -> coarse-to-fine skipped: pass-1 disc "
                          f"({_vpr_radius:.0f} m) <= 1.2x tight "
                          f"({_tight:.0f} m)")
                if (cfg.vpr_coarse_to_fine
                        and _vpr_radius > 1.2 * _tight):
                    # Snap the pass-2 centre to the 4-decimal (~11 m) grid the
                    # fetch signature + graphml filename already key on — so
                    # the c2f cache/graph is stable across reruns WITHOUT
                    # throwing away pass-1 precision. The old ~500 m snap
                    # (0.005 deg), added when cold fetches drifted the centre
                    # run-to-run, degraded an already-precise pass-1 centre by
                    # up to ~400 m (London: 103 m -> 323 m, the whole 734 m
                    # headline); R1's union+sha1 store now makes the centre
                    # DETERMINISTIC across cold fetches, so the coarse snap is
                    # obsolete and the fine one keeps the cache stable anyway.
                    _g = 0.0001
                    _slat = round(float(ctx.vpr_center[0]) / _g) * _g
                    _slon = round(float(ctx.vpr_center[1]) / _g) * _g
                    print(f"      -> [4a2] coarse-to-fine: re-fetching a "
                          f"{_tight:.0f} m disc around the pass-1 centre "
                          f"(snapped {_slat:.4f}, {_slon:.4f})")
                    _p1_center, _p1_track = ctx.vpr_center, tr
                    _p1_sim = float(np.median(v_sims)) if len(v_sims) else 0.0
                    try:
                        tr2 = kartaview_vpr_track(
                            frames.frames, (_slat, _slon),
                            radius_m=_tight,
                            cache_dir=str(cfg.data_dir / f"{_vpr_src}_c2f"),
                            device=_vpr_device, source=_vpr_src,
                            token=_vpr_token, cap=cfg.vpr_ref_cap,
                            sequence_decode=cfg.vpr_viterbi, query_dt_s=_qdt)
                        if _vpr_src == "mapillary" and _kv.LAST_FETCH_STATS:
                            ctx.vpr_listing = {**(ctx.vpr_listing or {}),
                                               "pass2": dict(_kv.LAST_FETCH_STATS)}
                    except Exception as e:
                        tr2 = None
                        print(f"      -> coarse-to-fine fetch failed ({e})")
                    if tr2 is not None:
                        _p2_center = _robust_center(tr2[1], tr2[2])
                        _p2_sim = float(np.median(tr2[2])) if len(tr2[2]) else 0.0
                        # GATE the placement-prior swap on match quality: a tight
                        # disc that CLIPS the true refs (route past the radius, or
                        # a slightly-off pass-1 centre) or re-fetches a sparser
                        # cold set drifts its robust centre OUTWARD and scores a
                        # LOWER median top-1 similarity — same MegaLoc space both
                        # passes, so it is a fair, GT-free arbiter. Keep pass 2's
                        # centre/track for placement only when it does not lose
                        # match quality (London pass-1 0.10 km -> pass-2 0.32 km
                        # -> 734 m was pass 2 silently overwriting a better
                        # pass-1 prior; audit deepdive C2F-1).
                        if _p2_sim >= _p1_sim - _C2F_SIM_MARGIN:
                            ctx.vpr_track = tr2
                            v_idx, v_ll, v_sims = tr2
                            ctx.vpr_center = _p2_center
                            _accepted = "pass-2"
                        else:
                            ctx.vpr_track, ctx.vpr_center = _p1_track, _p1_center
                            _accepted = "pass-1 (pass-2 sim worse)"
                        # Centre the GRAPH on the KEPT centre (snapped to the
                        # ~11 m cache grid); +snap-tol margin. Round-5 bug C:
                        # never centre on the raw pass-1 snap when the refined
                        # centre moved.
                        _g2lat = round(float(ctx.vpr_center[0]) / _g) * _g
                        _g2lon = round(float(ctx.vpr_center[1]) / _g) * _g
                        ctx.osm_around = (_g2lat, _g2lon, _tight + _C2F_SNAP_TOL_M)
                        ctx.c2f_info = {
                            "pass1_center": [float(_p1_center[0]), float(_p1_center[1])],
                            "pass2_center": [float(_p2_center[0]), float(_p2_center[1])],
                            "tight_radius_m": _tight,
                            "kept": _accepted,
                            "pass1_sim": _p1_sim, "pass2_sim": _p2_sim,
                        }
                        print(f"      -> [4a2] kept {_accepted} prior "
                              f"(sim p1 {_p1_sim:.3f} / p2 {_p2_sim:.3f}); "
                              f"VPR {ctx.vpr_center[0]:.5f}, {ctx.vpr_center[1]:.5f}; "
                              f"graph re-centred ({_g2lat:.4f}, {_g2lon:.4f})")
                    else:
                        _hint = ("" if _vpr_token or _vpr_src != "mapillary"
                                 else " (tokenless: set MLY_TOKEN to enable "
                                      "the pass-2 refetch)")
                        print(f"      -> coarse-to-fine pass 2 unavailable; "
                              f"keeping pass 1{_hint}")
                # If coarse-to-fine is on and pass 1 located the drive but the
                # graph disc is still unset (pass 2 unavailable / skipped /
                # rejected), tighten the graph onto the pass-1 centre anyway —
                # otherwise a city-name deployable run fetches the WHOLE
                # municipality graph and dilutes the candidate pool over the
                # entire city, discarding the located centre (audit F1/H4).
                if (cfg.vpr_coarse_to_fine and ctx.osm_around is None
                        and ctx.vpr_center is not None):
                    _g = 0.0001
                    _slat = round(float(ctx.vpr_center[0]) / _g) * _g
                    _slon = round(float(ctx.vpr_center[1]) / _g) * _g
                    ctx.osm_around = (_slat, _slon,
                                      min(_vpr_radius, _tight) + _C2F_SNAP_TOL_M)
                    print(f"      -> coarse-to-fine: graph tightened onto the "
                          f"pass-1 centre ({_slat:.4f}, {_slon:.4f}, "
                          f"{ctx.osm_around[2]:.0f} m)")
            else:
                from .. import kartaview_vpr as _kv
                _why = _kv.LAST_FAILURE or "no reference track"
                print(f"      -> VPR unavailable ({_why}); keeping full-city graph")
                _record_channel(channel_status, "vpr", "inactive", reason=_why)
        except Exception as e:
            print(f"      -> VPR prior failed ({e}); keeping full-city graph")
            _record_channel(channel_status, 'vpr', "failed", error=e)


def plate_district_prior(ctx: RunContext) -> None:
    """Licence-plate registration-district anchor -> ``ctx.plate_center``.

    An independent, absolute coarse prior: read EU plate region prefixes across
    the clip, vote, geocode the modal district. A RE-RANK channel in the
    consensus (not a hard gate).
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    video_path = ctx.video_path

    # [4a'] License-plate district anchor: an independent, absolute coarse prior.
    # Reads EU plate region prefixes across the clip, votes on the registration
    # district, geocodes it. Used as a RE-RANK channel in the consensus fusion
    # (not a hard gate — gating is slow on the resulting graph and can't separate
    # the true candidate from wrong ones inside any reasonable radius). The
    # district centroid pulls the pick toward the right area and kills gross
    # km-scale errors. 0.4 km from GT on Ulm; strong where plates are readable
    # (>=720p) and the fleet is local.
    ctx.result_plate_anchor = None
    ctx.plate_center = None
    if cfg.use_plate_anchor:
        print("[4a'] License-plate district anchor (ALPR + Kfz-code vote)")
        try:
            from ..plate_anchor import plate_district_anchor
            pa = plate_district_anchor(str(video_path))
            if pa is not None:
                ctx.plate_center = (pa.lat, pa.lon)
                ctx.result_plate_anchor = {
                    "code": pa.code, "district": pa.district, "lat": pa.lat,
                    "lon": pa.lon, "votes": pa.votes, "unique": pa.total_unique,
                    "margin": round(pa.margin, 2), "tally": pa.tally,
                }
                print(f"      -> district {pa.code}={pa.district} "
                      f"({pa.votes}/{pa.total_unique} unique, x{pa.margin:.1f}) at "
                      f"{pa.lat:.4f},{pa.lon:.4f}; re-ranking candidates by proximity")
            else:
                print("      -> no confident district")
        except Exception as e:
            print(f"      -> plate anchor failed ({e})")
            _record_channel(channel_status, 'plate', "failed", error=e)


def vlm_district_prior(ctx: RunContext) -> None:
    """VLM (Gemma 4) district/landmark prior - a coverage FALLBACK.

    Only runs when VPR produced no centre (no nearby street-photo references);
    feeds the same anchor-primary path via ``vpr_center``.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames

    # [4a''] VLM (Gemma 4) district/landmark prior — a coverage FALLBACK: only
    # runs when VPR produced no centre (no nearby street-photo references). Reads
    # frames -> infers district + reads names -> geocodes -> feeds the same
    # anchor-primary path via vpr_center. Independent failure modes from VPR.
    ctx.result_vlm_anchor = None
    if cfg.use_vlm_anchor and ctx.vpr_center is None:
        print("[4a''] VLM district/landmark prior (Gemma 4)")
        try:
            from ..text_anchor import default_geocode_fn
            from ..vlm_anchor import vlm_district_anchor
            # Validated defaults (live Ulm run): trust a single legible street
            # PLATE (min_street_votes=1 recovered 'Salzstadel' at 81 m), require
            # 2 votes for districts, keep raw-text fallback OFF (an unguarded
            # 1-vote TEXT token relocated the anchor 2.2 km), 15 km city bound.
            va = vlm_district_anchor(
                frames.frames, cfg.city,
                geocode_fn=default_geocode_fn(cfg.data_dir / "geocode_cache.json"),
                n_query=6, min_votes=2, min_street_votes=1,
                use_text_fallback=False, max_km_from_city=15.0)
            if va is not None:
                ctx.vpr_center = (va.lat, va.lon)
                ctx.anchor_origin = "vlm"
                ctx.result_vlm_anchor = {"label": va.label, "lat": va.lat, "lon": va.lon,
                                         "streets": va.street_votes,
                                         "districts": va.district_votes, "texts": va.text_votes,
                                         "applied_as_prior": True}
                print(f"      -> VLM prior {va.lat:.5f}, {va.lon:.5f} ('{va.label}'); "
                      f"feeding anchor-primary")
            else:
                print("      -> VLM produced no geocodable district/name")
        except Exception as e:
            print(f"      -> VLM anchor failed ({e})")
            _record_channel(channel_status, 'vlm', "failed", error=e)
