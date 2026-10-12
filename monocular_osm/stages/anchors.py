"""Stage 4b/4c: absolute heading from the sun and OCR scene-text anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import geodesic_distance_m
from .context import _record_channel, RunContext
from .length_prior import _length_sane_here


def estimate_sun_heading(ctx: RunContext) -> None:
    """Optional absolute heading from the sun -> ``ctx.result_sun_heading``.

    Activates only when the clip carries a usable capture time (container
    metadata or a burned-in clock); otherwise a graceful no-op.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    video_path = ctx.video_path
    osm_around = ctx.osm_around

    # 4b. Optional: absolute heading from the sun. Activates only when the clip
    # carries a usable capture time (container metadata or a burned-in clock);
    # otherwise a graceful no-op. When available it pins the matcher's free
    # rotation DOF (monocular_osm/sun_heading.py). Computed + reported here.
    if cfg.use_sun_heading:
        try:
            from ..sun_heading import estimate_heading
            cen = osm_around[:2] if osm_around else None
            if cen is None:
                import osmnx as ox
                cen = ox.geocode(cfg.city)
            sh_times = list(np.linspace(cfg.vo_start_sec,
                                        cfg.vo_end_sec or (cfg.vo_start_sec + 420.0), 30))
            sh = estimate_heading(str(video_path), cen, sh_times,
                                  ocr_engine=cfg.ocr_engine)
            ctx.result_sun_heading = sh
            if sh and sh.get("available"):
                print(f"[4b] Sun heading: {sh['median_heading']:.0f} deg via {sh['source']} "
                      f"({sh['n_used']} frames, conf {sh['confidence']:.2f})")
            else:
                print(f"[4b] Sun heading: not available "
                      f"({(sh or {}).get('reason', 'module missing')})")
        except Exception as e:
            print(f"[4b] Sun heading failed ({e})")
            _record_channel(channel_status, 'sun', "failed", error=e)
            ctx.result_sun_heading = {"available": False, "reason": str(e)}
    else:
        ctx.result_sun_heading = None


def _drop_direction_anchors(ocr_anchors, detections, ocr_source, *,
                            start_sec=0.0, use_super_res=False):
    """Drop OCR POI anchors whose sign is a *directional* one (names a place
    elsewhere) using the Gemma sign-type classifier. Each anchor is mapped
    back to its source detection (for the sign bbox), a frame is decoded from
    ``ocr_source`` at that detection's time, and the model labels the crop
    here/direction/other; 'direction' anchors are removed. Returns
    ``(kept_anchors, dropped_names)``. Any failure returns the anchors
    unchanged — this is a best-effort quality filter, never a hard dependency.

    ``use_super_res`` must match the OCR run: detection bboxes are stored in
    the (2.5x-upscaled) coordinate space OCR saw, so the decoded frame is
    upscaled the same way before cropping — otherwise every crop clamps to
    ~the whole frame and the classifier over-rejects.
    """
    if not ocr_anchors:
        return ocr_anchors, []
    try:
        import cv2

        from ..vlm_anchor import classify_sign_types
    except Exception:
        return ocr_anchors, []
    _upscale = None
    if use_super_res:
        try:
            from ..scene_text import _upscale_sharpen as _upscale
        except Exception:
            _upscale = None
    # Map each unique anchor to its most likely source detection: same text,
    # nearest in time (falls back to nearest-in-time regardless of text).
    reps, rep_for_anchor = [], {}
    for a in ocr_anchors:
        key = a.name.casefold()
        if key in rep_for_anchor:
            continue
        same = [d for d in detections
                if (getattr(d, "text", "") or "").casefold() == key]
        pool = same or detections
        if not pool:
            continue
        det = min(pool, key=lambda d: abs(getattr(d, "t_sec", 0.0) - a.t_sec))
        rep_for_anchor[key] = det
        reps.append((key, det))
    if not reps:
        return ocr_anchors, []
    # Decode one frame per rep at its detection time.
    frames_bgr, records = [], []
    cap = None
    try:
        cap = cv2.VideoCapture(str(ocr_source))
        for key, det in reps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(getattr(det, "t_sec", 0.0)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if _upscale is not None:
                # Align the frame to the (upscaled) space the bbox was read in.
                frame = _upscale(frame)
            records.append(_SignRep(
                text=getattr(det, "text", "") or "",
                bbox=getattr(det, "bbox", None),
                frame_idx=len(frames_bgr),
                key=key))
            frames_bgr.append(frame)
    except Exception:
        return ocr_anchors, []
    finally:
        if cap is not None:
            cap.release()   # release even if _upscale/read raised (audit F6)
    if not records:
        return ocr_anchors, []
    try:
        labels = classify_sign_types(frames_bgr, records)
    except Exception:
        return ocr_anchors, []
    direction = {rec.key for rec, lab in zip(records, labels) if lab == "direction"}
    if not direction:
        return ocr_anchors, []
    kept = [a for a in ocr_anchors if a.name.casefold() not in direction]
    dropped = [a.name for a in ocr_anchors if a.name.casefold() in direction]
    return kept, dropped


@dataclass
class _SignRep:
    """Lightweight detection record for :func:`classify_sign_types`."""
    text: str
    bbox: tuple | None
    frame_idx: int
    key: str


def ocr_scene_text_anchors(ctx: RunContext) -> None:
    """OCR scene text -> geocoded POI / street anchors, scale recovery, enumeration seeds.

    Computed before the match so the anchors can *seed* enumeration (add walk
    roots near each anchor), guaranteeing the anchored area is in the candidate
    pool even when drift would exclude it.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    video_path = ctx.video_path
    frames = ctx.frames
    duration_prior_m = ctx.duration_prior_m
    traj = ctx.traj
    vpr_center = ctx.vpr_center
    anchor_origin = ctx.anchor_origin
    road = ctx.road

    # 4c. Optional: OCR scene-text → geocoded POI anchors. Computed before
    # the match so the anchors can *seed* enumeration (add walk roots near
    # each anchor), guaranteeing the anchored area is in the candidate
    # pool even when drift would exclude it — the fix for the enumeration
    # failure that pure re-ranking can't address.
    ctx.ocr_anchors = []
    ctx.street_anchors = []
    ctx.anchor_xy = np.zeros((0, 2))
    ctx.anchor_seed = None
    ctx.ocr_anchor_error = None
    ctx.anchor_transform = None          # similarity VO->world (idea 2), if fit
    ctx.anchor_world_route = None        # full VO path georeferenced to world xy
    if cfg.enable_ocr_anchor:
        ocr_source = cfg.ocr_video_path or video_path
        tag = " (4K OCR source)" if cfg.ocr_video_path else ""
        print(f"[4c] OCR scene-text → anchors{tag}")
        try:
            from ..scene_text import extract_scene_text
            from ..text_anchor import (
                anchor_seed_nodes,
                anchors_to_xy,
                cluster_filter_anchors,
                default_geocode_fn,
                gazetteer_anchors,
                geocode_texts,
                match_text_to_streets,
                street_anchor_seed_nodes,
                street_anchor_xy,
            )
            # Cache key includes the source name so 720p and 4K OCR don't
            # collide.
            cache_tag = "_4k" if cfg.ocr_video_path else ""
            cache_tag += "_sr" if cfg.use_ocr_super_res else ""
            detections = extract_scene_text(
                ocr_source,
                sample_interval_sec=cfg.ocr_sample_interval_sec,
                start_sec=cfg.vo_start_sec,
                end_sec=cfg.vo_end_sec,
                min_confidence=0.3,
                super_res=cfg.use_ocr_super_res,
                cache_path=cfg.data_dir / f"scene_text_cache{cache_tag}.json",
            )
            print(f"      -> {len(detections)} text detections")
            # (a) Street-name plates → OSM graph geometry (route-relevant,
            # strongest anchor; needs legible plates → true-4K).
            ctx.street_anchors = match_text_to_streets(
                detections, road, min_confidence=cfg.ocr_min_confidence,
            )
            for s in ctx.street_anchors:
                print(f"        street: OCR {s.ocr_text!r} -> {s.name!r} "
                      f"(conf {s.confidence:.2f}, ratio {s.match_ratio:.2f}, "
                      f"{len(s.node_ids)} nodes)")
            # (b) POI/landmark names → geocoded points (works at 720p too).
            ctx.ocr_anchors = geocode_texts(
                detections, cfg.city, road,
                geocode_fn=default_geocode_fn(cfg.data_dir / "geocode_cache.json"),
                min_confidence=cfg.ocr_min_confidence,
                # Temporal stratification (geocode_texts time_buckets) is
                # available but OFF by default: on the Ulm 4K run it
                # surfaced *direction-sign* POIs (e.g. "Handwerkskammer"
                # read at t=77 s while the car reaches it at t=245 s) —
                # right place, wrong time, and spatially consistent so the
                # cluster filter can't reject them, which corrupts the
                # anchor-pin (POI-only mean 146 -> 412 m). Temporal anchor
                # coverage is instead provided by the *street-name*
                # anchors fed into the pin below, which carry a valid
                # "you are here" timestamp.
                time_buckets=0,
            )
            # Local OSM gazetteer: fuzzy-match every detection against named
            # OSM features in the graph area (offline, no Nominatim budget).
            # Merged BEFORE the VPR gate below so its far-off noise is cleaned
            # by the same 1.5 km filter. Additive + deduped by name.
            if cfg.use_osm_gazetteer:
                gaz_anchors = gazetteer_anchors(
                    detections, road,
                    cache_path=cfg.data_dir / "osm_gazetteer_cache.json",
                    existing=ctx.ocr_anchors,
                    min_confidence=cfg.ocr_min_confidence,
                )
                if gaz_anchors:
                    print(f"      -> {len(gaz_anchors)} local-gazetteer anchor(s) "
                          f"(offline, beyond Nominatim)")
                ctx.ocr_anchors = ctx.ocr_anchors + gaz_anchors
            # OCR-quality: directional/destination signs name places ELSEWHERE,
            # so they geocode FAR from the actual route (the London 'Holborn'
            # failure: a direction sign ~1.5 km off). When a route-accurate VPR
            # prior exists, reject OCR anchors > 1.5 km from it — this drops the
            # misleading direction signs while keeping local street/POI plates,
            # leaving only anchors usable in the anchor-primary fit.
            # Only a retrieval prior is route-accurate enough to veto anchors;
            # a VLM-fallback prior is a geocoded district centroid that can sit
            # kilometres from a correct on-route anchor.
            if vpr_center is not None and anchor_origin == "vpr" and ctx.ocr_anchors:
                _vlat, _vlon = vpr_center
                _kept = []
                for a in ctx.ocr_anchors:
                    _d = geodesic_distance_m(a.lat, a.lon, _vlat, _vlon)
                    if _d <= 1500.0:
                        _kept.append(a)
                    else:
                        print(f"        poi REJECTED ({_d/1000:.1f} km from VPR — "
                              f"likely a direction sign): {a.name!r}")
                ctx.ocr_anchors = _kept
            for a in ctx.ocr_anchors:
                print(f"        poi:    {a.name!r} @ ({a.lat:.5f},{a.lon:.5f}) "
                      f"conf={a.confidence:.2f}")

            # Here-vs-direction sign typing (Gemma 4): a directional sign names
            # a place ELSEWHERE and geocodes off-route (the London 'Holborn'
            # failure). Drop such anchors before they seed/score. GPU + opt-in.
            if cfg.classify_signs and ctx.ocr_anchors:
                print("      -> classifying sign types (here vs direction)")
                try:
                    ctx.ocr_anchors, _dropped = _drop_direction_anchors(
                        ctx.ocr_anchors, detections, ocr_source,
                        start_sec=cfg.vo_start_sec,
                        use_super_res=cfg.use_ocr_super_res)
                    for _nm in _dropped:
                        print(f"        poi REJECTED (direction sign): {_nm!r}")
                except Exception as e:
                    print(f"      -> sign classification failed ({e})")
                    _record_channel(channel_status, 'sign_classification', "failed", error=e)

            # --- Scale recovery / georeferencing (ideas 1 & 2) ----------
            # Use the *pre-cluster-filter* anchors here: scale needs
            # temporal+spatial SPREAD, and RANSAC rejects the noise that
            # the cluster filter would otherwise have removed. Each anchor
            # is a timed pseudo-fix (seen at t_sec, geocoded to lat/lon).
            if cfg.enable_scale_recovery and len(ctx.ocr_anchors) >= 3:
                from ..scale_recovery import (
                    apply_transform,
                    fit_similarity_ransac,
                    scaled_length,
                    vo_positions_at_times,
                )
                from ..text_anchor import anchors_to_xy as _a2xy
                t_secs = np.array([a.t_sec for a in ctx.ocr_anchors])
                world_pts = _a2xy(ctx.ocr_anchors, road.crs)
                vo_pts = vo_positions_at_times(traj.xz, frames.timestamps, t_secs)
                fit = fit_similarity_ransac(
                    vo_pts, world_pts,
                    thresh_m=cfg.scale_recovery_thresh_m,
                    min_inliers=cfg.scale_recovery_min_inliers,
                    min_world_baseline_m=cfg.scale_recovery_min_baseline_m,
                )
                recovered_len = (scaled_length(traj.xz, fit.scale)
                                 if fit is not None else None)
                # Sanity gate: a handful of noisy sign-anchors can yield a
                # self-consistent-but-wrong RANSAC triple (rms looks fine,
                # scale is garbage). Trust the anchor scale only if it
                # roughly agrees with the duration-based prior — otherwise
                # the simple prior is the safer bet (verified on Ulm: a
                # 3-anchor fit gave 974 m vs a true ~2100 m, worse than the
                # 2310 m duration prior). Require >= the inlier floor too.
                sane = (
                    fit is not None and recovered_len is not None
                    and _length_sane_here(ctx, recovered_len)
                )
                if fit is not None and sane:
                    ctx.anchor_transform = fit.transform
                    ctx.estimated_length_m = float(np.clip(recovered_len, 500.0, 12000.0))
                    ctx.anchor_world_route = apply_transform(traj.xz, fit.transform)
                    ctx.result_scale_recovery = {
                        "status": "ok",
                        "scale_m_per_vo_unit": round(fit.scale, 4),
                        "n_inliers": int(len(fit.inlier_idx)),
                        "n_anchors": int(len(ctx.ocr_anchors)),
                        "rms_m": round(fit.rms_m, 1),
                        "world_baseline_m": round(fit.world_baseline_m, 1),
                        "estimated_length_m": round(ctx.estimated_length_m, 1),
                    }
                    print(f"      -> scale recovery OK: scale={fit.scale:.3f} m/unit, "
                          f"{len(fit.inlier_idx)}/{len(ctx.ocr_anchors)} inliers, "
                          f"baseline={fit.world_baseline_m:.0f} m, rms={fit.rms_m:.0f} m "
                          f"→ route length prior {ctx.estimated_length_m:.0f} m")
                elif fit is not None and not sane:
                    ctx.result_scale_recovery = {
                        "status": "rejected_sanity",
                        "scale_m_per_vo_unit": round(fit.scale, 4),
                        "n_inliers": int(len(fit.inlier_idx)),
                        "recovered_length_m": round(recovered_len, 1),
                        "duration_prior_m": round(duration_prior_m, 1),
                        "reason": "recovered length disagrees with duration prior "
                                  ">2x; trusting the prior",
                    }
                    print(f"      -> scale recovery REJECTED (sanity): recovered "
                          f"{recovered_len:.0f} m vs duration prior "
                          f"{duration_prior_m:.0f} m (>2x off); keeping prior")
                else:
                    ctx.result_scale_recovery = {
                        "status": "declined",
                        "n_anchors": int(len(ctx.ocr_anchors)),
                        "reason": "too few well-separated inlier anchors "
                                  "(short baseline / noisy geocodes)",
                    }
                    print("      -> scale recovery declined (anchors too few/clustered "
                          "for a reliable fit); keeping default length prior")

            # Reject spatial-outlier anchors (a far-off shop sign or a
            # fuzzy-matched wrong street) — keep only the corroborated
            # cluster, so one bad anchor can't hijack the gate.
            n_poi0, n_st0 = len(ctx.ocr_anchors), len(ctx.street_anchors)
            ctx.ocr_anchors, ctx.street_anchors = cluster_filter_anchors(
                ctx.ocr_anchors, ctx.street_anchors, road,
            )
            dropped = (n_poi0 - len(ctx.ocr_anchors)) + (n_st0 - len(ctx.street_anchors))
            if dropped:
                print(f"        cluster filter: kept {len(ctx.ocr_anchors)} POI + "
                      f"{len(ctx.street_anchors)} street, dropped {dropped} outlier(s)")
            # Merge both anchor kinds for scoring + seeding.
            ctx.anchor_xy = np.vstack([
                x for x in (anchors_to_xy(ctx.ocr_anchors, road.crs),
                            street_anchor_xy(ctx.street_anchors, road))
                if len(x) > 0
            ]) if (ctx.ocr_anchors or ctx.street_anchors) else np.zeros((0, 2))
            ctx.anchor_seed = list(dict.fromkeys(
                anchor_seed_nodes(road, anchors_to_xy(ctx.ocr_anchors, road.crs))
                + street_anchor_seed_nodes(ctx.street_anchors)
            ))
            print(f"      -> {len(ctx.street_anchors)} street + {len(ctx.ocr_anchors)} POI "
                  f"anchor(s); {len(ctx.anchor_seed)} enumeration seed node(s)")
        except Exception as e:
            print(f"      -> OCR-anchor channel failed: {e}")
            _record_channel(channel_status, 'ocr_anchor', "failed", error=e)
            ctx.ocr_anchor_error = str(e)
