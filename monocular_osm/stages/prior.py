"""Stage 4a: absolute coarse priors that locate the clip independently of shape.

* a GPS-free frame-based coarse seed (licence-plate district / OCR place names /
  optional VLM) that bounds the search when no disc was given;
* blind VPR (Mapillary / KartaView street photos + MegaLoc), the strongest
  shape-independent selection signal, with optional coarse-to-fine refinement;
* the licence-plate registration-district anchor;
* the VLM district/landmark prior (a coverage fallback when VPR finds nothing).
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
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
# Pass-2 is kept as the placement prior only if its median decoded-reference
# VPR similarity (raw cosine at the selected reference, not necessarily argmax)
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


def _regional_cache_directories(cfg, source):
    """Existing extra-region caches under the selected storage root."""
    external = getattr(cfg, "vpr_region_cache_dir", None)
    if external is not None:
        return (Path(external) / cfg.data_dir.name).glob(f"{source}_region_*")
    return (cfg.data_dir / f"{source}_regions").glob("*")


def _terminal_sparse_search(center, request, stats, reason):
    """Distinguish a proven completed sparse listing from an ambiguous None."""
    from ..kartaview_vpr import _fetch_signature, _sig_tag, _reference_snapshot
    source = request.get("source")
    snapshot = stats.get("snapshot") or {}
    count, lost = snapshot.get("n_refs"), stats.get("lost")
    radius = request.get("radius_m", 3000.)
    if (source != "mapillary" or stats.get("listing_stats_known") is not True
            or stats.get("listing") != "tiles" or stats.get("truncated", False) is not False
            or isinstance(count, bool) or not isinstance(count, int) or not 0 <= count < 30
            or isinstance(lost, bool) or not isinstance(lost, int) or lost != 0
            or reason != f"only {count} {source} reference images within {radius:.0f} m (need 30)"):
        return None
    signature = _fetch_signature(center, radius, request.get("cap", 1500))
    refs, metadata = [], None
    if count:
        directory = Path(request["cache_dir"])
        for path in (directory / f"mly_ref_meta_{_sig_tag(signature)}.json", directory / "mly_ref_meta.json"):
            if not path.is_file():
                continue
            blob = json.loads(path.read_text(encoding="utf-8"))
            if blob.get("signature") == signature and len(blob.get("refs") or []) == count:
                refs, metadata = blob["refs"], str(path)
                break
        if metadata is None:
            return None
    if _reference_snapshot(refs)["refs_sha256"] != snapshot.get("refs_sha256"):
        return None
    return {"status": "complete_empty" if count == 0 else "complete_sparse", "count": count,
            "source": source, "fetch_signature": signature, "snapshot": deepcopy(snapshot),
            "listing": deepcopy(stats), "reference_inventory": refs, "metadata_source": metadata,
            "reason": "completed public listing below the existing 30-reference decoder minimum"}


def _multi_region_vpr(cfg, frames, center, radius_m, cap, source, token, device,
                      query_dt_s, track_fn, *, query_cache=None):
    """Fetch independent dense regional tracks and select with temporal evidence."""
    from .. import kartaview_vpr as kv
    from ..location_prior import resolve_retrieval_regions
    from ..regional_retrieval import fill_region_corridors, merge_region_tracks

    regions = resolve_retrieval_regions(
        cfg.city, getattr(cfg, "video_title", None), getattr(cfg, "video_description", None),
        center=center, radius_m=cfg.vpr_search_radius_m,
        max_regions=getattr(cfg, "vpr_region_cap", 6))
    if not regions:
        regions = [{"name": "search seed", "center": list(center),
                    "radius_m": radius_m, "origin": "seed"}]
    # Keep the ordinary first-pass region and its cache exactly intact. Other
    # named regions use dense discs with their own full budget, never a sparse
    # outer ring that dilutes the original reference set.
    regions[0] = {**regions[0], "radius_m": radius_m}
    regions = fill_region_corridors(
        regions, max_regions=getattr(cfg, "vpr_region_cap", 6),
        radius_m=cfg.vpr_search_radius_m)
    tracks, usable_indices, records, listing = [], [], [], {}
    for index, region in enumerate(regions):
        region_cap = cap if index == 0 else cfg.vpr_ref_cap
        if index == 0:
            directory = cfg.data_dir / source
        else:
            identity = f"{region['name']}:{region['center']}"
            tag = hashlib.sha256(identity.encode()).hexdigest()[:12]
            external = getattr(cfg, "vpr_region_cache_dir", None)
            if external is not None:
                directory = Path(external) / cfg.data_dir.name / f"{source}_region_{tag}"
            else:
                directory = cfg.data_dir / f"{source}_regions" / tag
        record = {**region, "ref_cap": region_cap, "cache": str(directory)}
        print(f"      -> region {index + 1}/{len(regions)}: {region['name']}, "
              f"r={region['radius_m']:.0f} m, cap={region_cap}")
        try:
            kv.LAST_TRACK_STATS = {}
            track = track_fn(
                frames.frames, tuple(region["center"]), radius_m=region["radius_m"],
                cache_dir=str(directory), device=device, source=source, token=token,
                cap=region_cap, sequence_decode=cfg.vpr_viterbi, query_dt_s=query_dt_s,
                query_mode=getattr(cfg, "vpr_query_mode", "warp"),
                query_batch_size=getattr(cfg, "vpr_query_batch_size", 4),
                query_cache=query_cache)
            record["track_decoding"] = dict(kv.LAST_TRACK_STATS)
            if source == "mapillary" and kv.LAST_FETCH_STATS:
                listing[f"region_{index}"] = dict(kv.LAST_FETCH_STATS)
            if track is None:
                record.update(status="unavailable", reason=kv.LAST_FAILURE)
            else:
                record.update(status="ok", median_similarity=float(np.median(track[2])))
                tracks.append(track)
                usable_indices.append(index)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        records.append(record)
    merged, evidence = merge_region_tracks(tracks, frames.timestamps)
    if merged is not None:
        evidence["selected_regions"] = [usable_indices[i] for i in evidence["selected_regions"]]
        counts = [0] * len(regions)
        for index, count in zip(usable_indices, evidence["region_frame_counts"]):
            counts[index] = count
        evidence["region_frame_counts"] = counts
    diagnostics = {"experimental": True, "regions": records,
                   "n_usable_regions": len(tracks), **evidence}
    return merged, diagnostics, listing


def _city_cache_directories(cfg, source):
    """Existing city-cell reference caches under the configured storage root."""
    external = getattr(cfg, "vpr_region_cache_dir", None)
    if external is not None:
        return (Path(external) / cfg.data_dir.name).glob(f"{source}_city_*")
    return (cfg.data_dir / f"{source}_city").glob("*")


def _city_cover_vpr(cfg, frames, baseline_track, baseline_region, source, token, device,
                    query_dt_s, track_fn, *, query_cache=None, plan=None, kind="city",
                    coarse_cap=None):
    """Optional full-city hierarchy; a failed experiment retains ordinary VPR.

    ``plan`` (a callable returning ``(regions, audit)``) replaces the
    municipality cell plan, ``kind`` names the cache directories and
    ``coarse_cap`` overrides ``cfg.vpr_city_coarse_cap``; the metro-ring search
    uses all three. The defaults keep the municipality search unchanged.
    """
    from .. import kartaview_vpr as kv
    from ..city_retrieval import (CityBoundaryTypeError, CityCoverageLimitError,
                                  plan_city_regions, select_city_tracks)

    external = getattr(cfg, "vpr_region_cache_dir", None)
    shared_root = Path(external) if external is not None else cfg.data_dir.parent
    listing = {}
    baseline_backbone = kv.backbone_info()
    expected_backbone = ((baseline_backbone.get("name"), baseline_backbone.get("hub"))
                         if baseline_track is not None and baseline_backbone.get("name") else None)
    diagnostics = {"experimental": True, "confidence_calibrated": False,
                   "baseline": {**baseline_region, "track_present": baseline_track is not None,
                                "backbone": baseline_backbone,
                                "track_decoding": baseline_region.get("track_decoding", dict(kv.LAST_TRACK_STATS))},
                   "selected_city_frames": 0}
    cell_selector = getattr(cfg, "vpr_city_cell_selector", "decoded")
    diagnostics["cell_selector"] = cell_selector
    try:
        if plan is None:
            regions, plan_audit = plan_city_regions(
                cfg.city, shared_root / "_city_boundaries", radius_m=cfg.vpr_search_radius_m)
        else:
            regions, plan_audit = plan()
        diagnostics["plan"] = plan_audit
    except Exception as exc:
        diagnostics.update(status="failed", reason="city_plan_refused",
                           error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, CityCoverageLimitError):
            diagnostics["cell_limit"] = {"required_cells": exc.required_cells,
                                         "max_cells": exc.max_cells,
                                         "required_cells_is_lower_bound": exc.lower_bound}
        if isinstance(exc, CityBoundaryTypeError):
            diagnostics.update(boundary_validation=exc.validation, boundary_source=exc.source)
        print(f"      -> city coverage plan unavailable ({exc}); keeping ordinary retrieval")
        return baseline_track, diagnostics, listing

    def retrieve(region, *, cap, phase, index):
        nonlocal expected_backbone
        identity = f"{cfg.city}:{region['center']}:{region['radius_m']}"
        tag = hashlib.sha256(identity.encode()).hexdigest()[:12]
        if external is not None:
            directory = Path(external) / cfg.data_dir.name / f"{source}_{kind}_{phase}_{tag}"
        else:
            directory = cfg.data_dir / f"{source}_{kind}" / f"{phase}_{tag}"
        print(f"      -> {kind} {phase} cell {index + 1}/{len(regions)} "
              f"r={region['radius_m']:.0f} m, cap={cap}")
        kv.LAST_FETCH_STATS = {}
        kv.LAST_FAILURE = None
        kv.LAST_TRACK_STATS = {}
        record = {"cache": str(directory), "source": source, "backbone": kv.backbone_info()}
        audit = {} if cell_selector == "raw-votes" else None
        audit_options = ({"audit": audit, "query_timestamps": np.asarray(frames.timestamps, float)}
                         if audit is not None else {})
        try:
            track = track_fn(
                frames.frames, tuple(region["center"]), radius_m=region["radius_m"],
                cache_dir=str(directory), device=device, source=source, token=token,
                cap=cap, sequence_decode=cfg.vpr_viterbi, query_dt_s=query_dt_s,
                query_mode=getattr(cfg, "vpr_query_mode", "warp"),
                query_batch_size=getattr(cfg, "vpr_query_batch_size", 4), query_cache=query_cache,
                **audit_options)
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}")
            return None, record
        record["backbone"] = kv.backbone_info()
        record["track_decoding"] = dict(kv.LAST_TRACK_STATS)
        if source == "mapillary" and kv.LAST_FETCH_STATS:
            stats = dict(kv.LAST_FETCH_STATS)
            listing[f"city_{phase}_{index}"] = stats
            record["listing"] = stats
        model = record["backbone"]
        if track is not None and model.get("name"):
            fingerprint = (model.get("name"), model.get("hub"))
            if expected_backbone is not None and fingerprint != expected_backbone:
                record["error"] = "retrieval backbone changed; cosine scores are not comparable"
                return None, record
            expected_backbone = fingerprint
        if track is None:
            record["reason"] = kv.LAST_FAILURE or "no reference track"
        return (track, record, audit) if audit is not None else (track, record)

    try:
        merged, selection = select_city_tracks(
            regions, baseline_track, frames.timestamps, retrieve,
            coarse_cap=(coarse_cap if coarse_cap is not None
                        else getattr(cfg, "vpr_city_coarse_cap", 250)),
            fine_regions=getattr(cfg, "vpr_city_fine_regions", 3), fine_cap=cfg.vpr_ref_cap,
            cell_selector=cell_selector)
        diagnostics.update(selection)
        multiplier = 3 if source == "union" else 1
        diagnostics["reference_budget"].update(
            cap_is_per_source=source == "union", source_multiplier_upper_bound=multiplier,
            coarse_reference_upper_bound=selection["reference_budget"]["coarse_max"] * multiplier,
            fine_reference_upper_bound=selection["reference_budget"]["fine_max"] * multiplier)
        return merged, diagnostics, listing
    except Exception as exc:
        diagnostics.update(status="failed", reason="city_selection_failed",
                           error=f"{type(exc).__name__}: {exc}")
        print(f"      -> city retrieval failed ({exc}); keeping ordinary retrieval")
        return baseline_track, diagnostics, listing


# Metro-ring search (opt-in --vpr-metro-ring). Gate: the pre-registered
# selective rule's threshold (benchmarks/preregistered_rules.json,
# pipeline_selective.tau), reused unchanged: a track below it is the one the
# rule already treats as unreliable. Cells: hex discs of this radius, each with
# the pass-1 city disc's reference density (so 300 refs with a token).
_METRO_RING_GATE = 0.162
_METRO_CELL_M = 4000.0


def metro_ring_confidence(c2f_info, vpr_track):
    """The selective rule's confidence signal, read at runtime: the coarse-to-fine
    pass-2 median cosine when pass 2 ran, else the kept track's median cosine."""
    sim = (c2f_info or {}).get("pass2_sim")
    if sim is not None and np.isfinite(float(sim)):
        return float(sim), "vpr_coarse_to_fine.pass2_sim"
    if vpr_track is not None and len(vpr_track[2]):
        return float(np.median(vpr_track[2])), "vpr_prior.track_sim_median"
    return None, None


def metro_ring_coarse_cap(token_present: bool) -> int:
    """References per ring cell at the pass-1 city disc's density."""
    city_cap = _PASS1_TOKEN_CAP if token_present else _PASS1_COVER_CAP
    return max(1, round(city_cap * (_METRO_CELL_M / _PASS1_COVER_MAX_M) ** 2))


def _metro_ring_vpr(ctx, *, center, searched_radius_m, estimated_length_m, source, token,
                    device, query_dt_s, track_fn, query_cache):
    """When the ordinary track is weak, search the city's surroundings.

    The drive may lie outside the disc pass 1 searched (failure mode A in
    docs/failure_taxonomy_2026-11.md). Below the gate this covers the disc of
    ``cfg.vpr_metro_radius_m`` around the city's geocode, minus the searched
    disc, with the municipality search's cell hierarchy (coarse cells, densify
    the supported ones, final continuity merge in which the ordinary track
    stays a candidate). If ring evidence is selected, the centre moves there
    and a coarse-to-fine pass refines it under the ordinary pass-2 rule.
    """
    from .. import kartaview_vpr as kv
    from ..city_retrieval import metro_ring_regions
    from ..kartaview_vpr import _robust_center

    cfg = ctx.cfg
    conf, signal = metro_ring_confidence(ctx.c2f_info, ctx.vpr_track)
    record = {"experimental": True,
              "gate": {"confidence": conf, "signal": signal, "tau": _METRO_RING_GATE}}
    ctx.result["metro_ring_retrieval"] = record
    if conf is not None and conf >= _METRO_RING_GATE:
        record.update(status="not_triggered", kept="baseline")
        print(f"      -> metro ring not needed (confidence {conf:.3f} >= {_METRO_RING_GATE})")
        return
    if ctx.vpr_track is not None and len(ctx.vpr_track[0]):
        # The ordinary track in result["vpr_prior"]["track"]'s format, so the
        # same run can be scored with and without the ring (paired comparison).
        times = np.asarray(ctx.frames.timestamps, float)
        record["baseline_track"] = [
            [round(float(times[int(i)]), 2), round(float(ll[0]), 6), round(float(ll[1]), 6),
             round(float(s), 4)] for i, ll, s in zip(*ctx.vpr_track)]
    try:
        import osmnx as ox
        city_centre = tuple(map(float, ox.geocode(cfg.city)))
    except Exception as exc:
        record.update(status="failed", reason="city_geocode_failed", kept="baseline",
                      error=f"{type(exc).__name__}: {exc}")
        print(f"      -> metro ring: geocode failed ({exc}); keeping the ordinary track")
        return
    coarse_cap = metro_ring_coarse_cap(bool(token))
    outer = float(cfg.vpr_metro_radius_m)
    print(f"      -> metro ring: confidence {conf if conf is None else round(conf, 3)} < "
          f"{_METRO_RING_GATE}; searching {searched_radius_m:.0f}-{outer:.0f} m around the city "
          f"({_METRO_CELL_M:.0f} m cells, cap {coarse_cap})")

    def plan():
        return metro_ring_regions(cfg.city, city_centre, outer_radius_m=outer,
                                  searched_centre=center, searched_radius_m=searched_radius_m,
                                  cell_radius_m=_METRO_CELL_M)

    baseline_region = {"center": list(map(float, center)), "radius_m": float(searched_radius_m),
                       "source": source, "coarse_to_fine": ctx.c2f_info}
    track, diagnostics, listing = _city_cover_vpr(
        cfg, ctx.frames, ctx.vpr_track, baseline_region, source, token, device, query_dt_s,
        track_fn, query_cache=query_cache, plan=plan, kind="metro", coarse_cap=coarse_cap)
    record.update(diagnostics)
    record["coarse_cap"] = coarse_cap
    ctx.vpr_listing = {**(ctx.vpr_listing or {}), **listing} or None
    if track is None or not diagnostics.get("selected_city_frames", 0):
        record["kept"] = "baseline"
        print("      -> metro ring: no ring evidence selected; keeping the ordinary track")
        return
    ring_sim = float(np.median(track[2]))
    centre = _robust_center(track[1], track[2])
    tight = max(float(cfg.vpr_c2f_radius_m), 0.6 * float(estimated_length_m))
    grid = 0.0001
    slat = round(float(centre[0]) / grid) * grid
    slon = round(float(centre[1]) / grid) * grid
    refine = {"center": [slat, slon], "radius_m": tight, "ring_sim": ring_sim}
    kept = "ring"
    try:
        kv.LAST_TRACK_STATS = {}
        refined = track_fn(
            ctx.frames.frames, (slat, slon), radius_m=tight,
            cache_dir=str(cfg.data_dir / f"{source}_metro_c2f"), device=device, source=source,
            token=token, cap=cfg.vpr_ref_cap,
            query_mode=getattr(cfg, "vpr_query_mode", "warp"),
            query_batch_size=getattr(cfg, "vpr_query_batch_size", 4),
            query_cache=query_cache, sequence_decode=cfg.vpr_viterbi, query_dt_s=query_dt_s)
        refine["track_decoding"] = dict(kv.LAST_TRACK_STATS)
    except Exception as exc:
        refined = None
        refine["error"] = f"{type(exc).__name__}: {exc}"
    if refined is not None:
        refine["refined_sim"] = float(np.median(refined[2]))
        # The ordinary pass-2 rule: keep the refinement unless it loses match quality.
        if refine["refined_sim"] >= ring_sim - _C2F_SIM_MARGIN:
            track, centre, kept = refined, _robust_center(refined[1], refined[2]), "ring+refine"
    ctx.vpr_track, ctx.vpr_center = track, centre
    ctx.anchor_origin, ctx.vpr_source_used = "vpr", source
    ctx.channel_status.pop("vpr", None)       # Recover an empty ordinary search.
    distances = geodesic_distance_m(centre[0], centre[1], track[1][:, 0], track[1][:, 1])
    required = max(tight, float(np.max(distances))) + _C2F_SNAP_TOL_M
    radius = min(required, _PASS1_COVER_MAX_M)
    ctx.osm_around = (float(centre[0]), float(centre[1]), radius)
    record.update(kept=kept, refine=refine, selected_center=[float(centre[0]), float(centre[1])],
                  selected_sim=float(np.median(track[2])), graph_radius_m=radius,
                  graph_radius_capped=required > radius)
    print(f"      -> metro ring: kept {kept} (sim {record['selected_sim']:.3f}); "
          f"centre {centre[0]:.5f}, {centre[1]:.5f}")


def _append_retrieval_audit(ctx, audit):
    ordinal = len(ctx.retrieval_audits)
    ctx.retrieval_audits.append(audit)
    sink = getattr(ctx, "retrieval_audit_sink", None)
    if callable(sink):
        try:
            sink(ordinal, audit)
        except Exception as error:
            # Optional evidence I/O must not turn a completed retrieval into a
            # failed prior or change the ordinary candidate-selection rules.
            ctx.result.setdefault("retrieval_audit_persistence_errors", []).append({
                "ordinal": ordinal, "error": f"{type(error).__name__}: {error}"})


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
    _query_cache = {}           # descriptors live only for this video/stage
    # VPR runs whenever enabled — it re-ranks + drives anchor-primary, so it is
    # NOT conditioned on osm_around (mega-cities pass --osm-around AND want VPR).
    if cfg.use_vpr_prior:
        _vpr_src = cfg.vpr_source
        print(f"[4a] VPR coarse prior ({_vpr_src.capitalize()} + MegaLoc)")
        try:
            import os as _os

            import osmnx as ox
            from ..kartaview_vpr import kartaview_vpr_track, _robust_center
            from .. import kartaview_vpr as _kv
            if getattr(ctx, "capture_retrieval_audit", False):
                _ordinary_track = kartaview_vpr_track
                ctx.retrieval_audits = []
                def kartaview_vpr_track(*args, **kwargs):
                    # A city raw-votes caller owns this dictionary. Pass that
                    # SAME sink through the real cache/encoder call, which can
                    # clear it before populating actual consumed evidence.
                    audit = kwargs.get("audit")
                    if audit is None:
                        audit = {}
                    elif not isinstance(audit, dict):
                        raise ValueError("retrieval audit sink must be a dictionary")
                    audit.clear()
                    request = {key: value for key, value in kwargs.items()
                               if key not in {"token", "query_cache", "audit", "query_timestamps"}}
                    request["center"] = list(args[1])
                    audit.update(call_index=len(ctx.retrieval_audits), request=request)
                    if kwargs.get("source") != "union" and not kwargs.get("outer_radius_m"):
                        kwargs.update(audit=audit, query_timestamps=np.asarray(frames.timestamps, float))
                    else:
                        audit["unsupported_comparison"] = "union or outer-ring evidence"
                    try:
                        track = _ordinary_track(*args, **kwargs)
                    except Exception as error:
                        audit["provider_error"] = f"{type(error).__name__}: {error}"
                        _append_retrieval_audit(ctx, audit)
                        raise
                    if track is not None:
                        audit["returned_track"] = {"indices": np.asarray(track[0]),
                            "latlon": np.asarray(track[1]), "cosines": np.asarray(track[2])}
                    else:
                        try:
                            terminal = _terminal_sparse_search(args[1], request,
                                deepcopy(_kv.LAST_FETCH_STATS), _kv.LAST_FAILURE)
                            if terminal is not None:
                                audit["terminal_search"] = terminal
                                _kv.LAST_TRACK_STATS["terminal_search"] = deepcopy(terminal)
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            audit["terminal_proof_error"] = f"{type(error).__name__}: {error}"
                    _append_retrieval_audit(ctx, audit)
                    return track
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
                _regional_cache = (getattr(cfg, "vpr_multi_region", False)
                                   and any(has_mapillary_cache(str(p)) for p in
                                           _regional_cache_directories(cfg, _vpr_src)))
                _city_cache = (getattr(cfg, "vpr_city_cover", False)
                               and any(has_mapillary_cache(str(p)) for p in
                                       _city_cache_directories(cfg, _vpr_src)))
                if has_mapillary_cache(str(cfg.data_dir / _vpr_src)) or _regional_cache or _city_cache:
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
            _vpr_outer = _vpr_outer_cap = None
            if ctx.osm_around is not None:
                _vpr_radius = min(_vpr_radius, max(800.0, float(ctx.osm_around[2])))
                _seed_r = float(ctx.osm_around[2])
                if (cfg.vpr_seed_ring and cfg.vpr_coarse_to_fine
                        and _seed_r > _vpr_radius + 1.0):
                    # Cover the seed's own uncertainty disc, at the city-extent
                    # mode's radius cap and token ref cap (no new constants).
                    _vpr_outer = min(_seed_r, _PASS1_COVER_MAX_M)
                    _vpr_outer_cap = _PASS1_TOKEN_CAP if _vpr_token else _PASS1_COVER_CAP
                    print(f"      -> seed ring: pass 1 also searches out to "
                          f"{_vpr_outer:.0f} m (cap {_vpr_outer_cap}) around the "
                          f"{_seed_r:.0f} m seed disc")
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
            _multi_used = False
            _kv.LAST_TRACK_STATS = {}
            if getattr(cfg, "vpr_multi_region", False):
                tr, _regional, _listing = _multi_region_vpr(
                    cfg, frames, center, _vpr_radius, _vpr_cap_eff, _vpr_src,
                    _vpr_token, _vpr_device, _qdt, kartaview_vpr_track,
                    query_cache=_query_cache)
                ctx.result["multi_region_retrieval"] = _regional
                ctx.vpr_listing = _listing or None
                _multi_used = _regional["n_usable_regions"] > 1
            else:
                tr = kartaview_vpr_track(
                    frames.frames, center, radius_m=_vpr_radius,
                    cache_dir=str(cfg.data_dir / _vpr_src), device=_vpr_device,
                    source=_vpr_src, token=_vpr_token, cap=_vpr_cap_eff,
                    sequence_decode=cfg.vpr_viterbi,
                    query_mode=getattr(cfg, "vpr_query_mode", "warp"), query_dt_s=_qdt,
                    query_batch_size=getattr(cfg, "vpr_query_batch_size", 4),
                    query_cache=_query_cache,
                    outer_radius_m=_vpr_outer, outer_cap=_vpr_outer_cap)
            from .. import kartaview_vpr as _kv
            if not getattr(cfg, "vpr_multi_region", False) and _kv.LAST_TRACK_STATS:
                ctx.result["vpr_decoding"] = {"pass1": dict(_kv.LAST_TRACK_STATS)}
            if (not getattr(cfg, "vpr_multi_region", False)
                    and _vpr_src == "mapillary" and _kv.LAST_FETCH_STATS):
                ctx.vpr_listing = {"pass1": dict(_kv.LAST_FETCH_STATS)}
            if not getattr(cfg, "vpr_multi_region", False) and _kv.LAST_RING is not None:
                ctx.vpr_listing = {**(ctx.vpr_listing or {}),
                                   "pass1_ring": dict(_kv.LAST_RING)}
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
                if _multi_used:
                    # Re-fetching just the median-centred disc would discard
                    # the regional coverage this experiment is testing.
                    distances = geodesic_distance_m(ctx.vpr_center[0], ctx.vpr_center[1],
                                                    v_ll[:, 0], v_ll[:, 1])
                    _regional_radius = max(_tight, float(np.max(distances))) + _C2F_SNAP_TOL_M
                    _graph_radius = min(_regional_radius, _PASS1_COVER_MAX_M)
                    ctx.osm_around = (float(ctx.vpr_center[0]), float(ctx.vpr_center[1]),
                                      _graph_radius)
                    _regional_diagnostics = ctx.result["multi_region_retrieval"]
                    _regional_diagnostics["graph_radius_m"] = _graph_radius
                    _regional_diagnostics["graph_radius_capped"] = (
                        _regional_radius > _graph_radius)
                    print("      -> regional track kept; single-centre coarse-to-fine skipped")
                if (cfg.vpr_coarse_to_fine and not _multi_used
                        and _vpr_radius <= 1.2 * _tight):
                    # Enabled but not fired — say why (a long route can floor
                    # _tight above the pass-1 disc, silently skipping pass 2).
                    print(f"      -> coarse-to-fine skipped: pass-1 disc "
                          f"({_vpr_radius:.0f} m) <= 1.2x tight "
                          f"({_tight:.0f} m)")
                if (cfg.vpr_coarse_to_fine and not _multi_used
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
                        _kv.LAST_TRACK_STATS = {}
                        tr2 = kartaview_vpr_track(
                            frames.frames, (_slat, _slon),
                            radius_m=_tight,
                            cache_dir=str(cfg.data_dir / f"{_vpr_src}_c2f"),
                            device=_vpr_device, source=_vpr_src,
                            token=_vpr_token, cap=cfg.vpr_ref_cap,
                            query_mode=getattr(cfg, "vpr_query_mode", "warp"),
                            query_batch_size=getattr(cfg, "vpr_query_batch_size", 4),
                            query_cache=_query_cache,
                            sequence_decode=cfg.vpr_viterbi, query_dt_s=_qdt)
                        if _kv.LAST_TRACK_STATS:
                            ctx.result.setdefault("vpr_decoding", {})["pass2"] = dict(_kv.LAST_TRACK_STATS)
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
                        # LOWER median decoded-reference cosine — same MegaLoc space both
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
            # Compare city candidates with the completed ordinary baseline,
            # including pass 2, rather than suppressing its normal refinement.
            _city_admitted = (getattr(cfg, "vpr_city_cover", False)
                              and callable(getattr(ctx, "assisted_city_admission", None))
                              and ctx.assisted_city_admission())
            if getattr(cfg, "vpr_city_cover", False) and not _city_admitted:
                _baseline_pass = "pass2" if (ctx.c2f_info or {}).get("kept") == "pass-2" else "pass1"
                _baseline_decoding = (ctx.result.get("vpr_decoding") or {}).get(_baseline_pass, {})
                if getattr(cfg, "vpr_multi_region", False):
                    _baseline_decoding = {"kind": "regional", "details": "multi_region_retrieval.regions"}
                _city_track, _city, _city_listing = _city_cover_vpr(
                    cfg, frames, ctx.vpr_track,
                    {"center": list(center), "radius_m": _vpr_radius, "ref_cap": _vpr_cap_eff,
                     "cache": str(cfg.data_dir / _vpr_src), "source": _vpr_src,
                     "includes_metadata_regions": getattr(cfg, "vpr_multi_region", False),
                     "coarse_to_fine": ctx.c2f_info, "track_decoding": _baseline_decoding},
                    _vpr_src, _vpr_token, _vpr_device, _qdt, kartaview_vpr_track,
                    query_cache=_query_cache)
                ctx.result["city_cover_retrieval"] = _city
                ctx.vpr_listing = {**(ctx.vpr_listing or {}), **_city_listing} or None
                _record_channel(channel_status, "city_cover",
                                "failed" if _city.get("status") == "failed" else
                                "ok" if _city.get("status") == "ok" else "inactive",
                                reason=_city.get("reason"),
                                selected_city_frames=_city.get("selected_city_frames", 0))
                if _city_track is not None and _city.get("selected_city_frames", 0):
                    ctx.vpr_track = _city_track
                    _idx, _ll, _similarities = _city_track
                    ctx.vpr_center = _robust_center(_ll, _similarities)
                    ctx.anchor_origin, ctx.vpr_source_used = "vpr", _vpr_src
                    channel_status.pop("vpr", None)       # Recover an empty ordinary region.
                    distances = geodesic_distance_m(ctx.vpr_center[0], ctx.vpr_center[1],
                                                    _ll[:, 0], _ll[:, 1])
                    required = max(float(cfg.vpr_c2f_radius_m), 0.6 * float(estimated_length_m),
                                   float(np.max(distances))) + _C2F_SNAP_TOL_M
                    radius = min(required, _PASS1_COVER_MAX_M)
                    ctx.osm_around = (*map(float, ctx.vpr_center), radius)
                    _city.update(graph_radius_m=radius, graph_radius_capped=required > radius)
                    print("      -> city track selected against the completed ordinary baseline")
            if getattr(cfg, "vpr_metro_ring", False):
                _metro_ring_vpr(
                    ctx, center=center, searched_radius_m=max(float(_vpr_radius), float(_vpr_outer or 0.0)),
                    estimated_length_m=estimated_length_m, source=_vpr_src, token=_vpr_token,
                    device=_vpr_device, query_dt_s=_qdt, track_fn=kartaview_vpr_track,
                    query_cache=_query_cache)
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
