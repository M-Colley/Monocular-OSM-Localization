"""Stage 10a*/11: anchor-primary placement and the final position reports."""

from __future__ import annotations

import numpy as np

from ..evaluator import load_gt_waypoints
from ..hypotheses import distinct_hypotheses, hypothesis_confidence
from ..position import (
    build_position_report,
    candidate_center_latlon,
    format_position_summary,
    google_maps_url,
)
from .context import _record_channel, RunContext
from .trajectory import _match_timestamps
from .vpr_track import _VPR_FREE_RADIUS_M, _vpr_sequence_median_m, _vpr_track_geometry


# Anchor-orientation refine (_rotation_refine_about_start): accept a rotation
# only when it beats no-rotation on the full-track median by BOTH margins AND
# the rotated fit lands inside the explain gate — i.e. the route genuinely
# lies ON the track, not merely less far from it. Clean, ambiguous, or
# wrong-scale runs all fail a gate and keep the matcher's heading.
_ROT_MIN_TRACK_PTS = 8
_ROT_MIN_EXTENT_M = 100.0          # rotation unobservable on a stub route
_ROT_MARGIN_RATIO = 0.75
_ROT_MARGIN_ABS_M = 20.0
_ROT_EXPLAIN_M = 2.0 * _VPR_FREE_RADIUS_M


def _rotation_refine_about_start(traj_xy, track_frame_idx, track_xy, sims,
                                 n_frames):
    """Best rotation of ``traj_xy`` about its START to fit the per-frame VPR
    track — the orientation residual that start-pinning leaves.

    The start pin fixes position and along-track phase but inherits the
    matcher's heading, and a wrong heading swings the far end of the route
    off by up to twice its extent. With the pinned start as the trusted datum
    this is a 1-DOF search, scored by the similarity-weighted MEDIAN track
    distance (:func:`_vpr_sequence_median_m`) over the whole ~80-frame track —
    robust to the confident-but-wrong minority matches that sank the
    endpoint-based 2-point rigid fit (one noisy END estimate owned the whole
    rotation there). Coarse 2-deg full-circle grid, then a 0.25-deg local
    refine.

    Returns ``(rotated_traj, theta_deg, base_median_m, best_median_m)`` only
    when the best rotation clears every gate in the constants block above;
    ``None`` otherwise, so callers can only no-op, never regress.
    """
    traj = np.asarray(traj_xy, dtype=np.float64)
    track_xy = np.asarray(track_xy, dtype=np.float64)
    if len(traj) < 2 or len(track_xy) < _ROT_MIN_TRACK_PTS:
        return None
    rel = traj - traj[0]
    if float(np.linalg.norm(rel, axis=1).max()) < _ROT_MIN_EXTENT_M:
        return None

    def _med(theta):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rot = rel @ np.array([[c, s], [-s, c]]) + traj[0]
        return _vpr_sequence_median_m(rot, track_frame_idx, track_xy, sims,
                                      n_frames)

    base = _med(0.0)
    if base is None:
        return None
    coarse = np.radians(np.arange(-180.0, 180.0, 2.0))
    fine = coarse[int(np.argmin([_med(t) for t in coarse]))] + np.radians(
        np.arange(-2.0, 2.0 + 1e-9, 0.25))
    fmeds = [_med(t) for t in fine]
    j = int(np.argmin(fmeds))
    theta, best = float(fine[j]), float(fmeds[j])
    if (best < _ROT_MARGIN_RATIO * base
            and base - best >= _ROT_MARGIN_ABS_M
            and best <= _ROT_EXPLAIN_M):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        return (rel @ np.array([[c, s], [-s, c]]) + traj[0],
                float(np.degrees(theta)), float(base), float(best))
    return None


def _place_candidate_on_track(traj, s_xy, e_xy, vpr_xy, vi, track_xy, vs,
                              n_frames):
    """The full placement chain for ONE candidate: start-pin -> scale retry
    -> orientation refine -> elastic fusion, all against the per-frame VPR
    track. Returns ``(placed, mode, rot_deg, track_median_m, notes)`` or
    ``None`` when every pin path fails the 400 m centroid guard (the caller
    falls back to a centroid translation). ``notes`` carries the human log
    lines so the caller prints only the WINNING candidate's steps."""
    notes: list[str] = []
    placed, mode, rot_deg = None, None, None
    cand2 = None
    cand = traj + (s_xy - traj[0])                  # (1) plain start-pin
    if np.linalg.norm(cand.mean(axis=0) - vpr_xy) <= 400.0:
        placed, mode = cand, "vpr_startpin"
    else:
        # (2) the candidate's LENGTH is wrong: rescale from the VPR extent.
        vpr_ext = float(np.linalg.norm(e_xy - s_xy))
        cand_ext = float(np.linalg.norm(traj[-1] - traj[0]))
        if vpr_ext > 200.0 and cand_ext > 1.0:
            sc = float(np.clip(vpr_ext / cand_ext, 0.5, 2.0))
            cand2 = (traj - traj[0]) * sc + s_xy
            if np.linalg.norm(cand2.mean(axis=0) - vpr_xy) <= 400.0:
                placed, mode = cand2, "vpr_startpin_scaled"
    # (3) orientation refine — polish, or rescue guard-failing pins.
    tries = ([(placed, mode)] if placed is not None else
             [(t, m) for t, m in ((cand2, "vpr_startpin_scaled"),
                                  (cand, "vpr_startpin")) if t is not None])
    for t, m in tries:
        r = _rotation_refine_about_start(t, vi, track_xy, vs, n_frames)
        if r is None:
            continue                     # no rotation for this entry; try next
        if np.linalg.norm(r[0].mean(axis=0) - vpr_xy) <= 400.0:
            placed, mode, rot_deg = r[0], m + "+rot", r[1]
            notes.append(f"orientation refine: {r[1]:+.1f} deg "
                         f"(track median {r[2]:.0f} -> {r[3]:.0f} m)")
            break                        # accepted a rotation — done
        # rotation exists but its centroid failed the guard: on the polish
        # path (placed already set, one entry) the loop ends and the
        # unrotated pin is kept; on the RESCUE path (placed is None, two
        # entries) fall through to try the next pin's rotation. A bare
        # break here dropped the second rescue attempt (bug 2026-07-05).
    # (4) elastic fusion — bend out low-frequency drift.
    if placed is not None and "startpin" in mode:
        fu = _elastic_fuse_track(placed, vi, track_xy, vs, n_frames)
        if fu is not None and np.linalg.norm(
                fu[0].mean(axis=0) - vpr_xy) <= 400.0:
            placed = fu[0]
            mode += "+fuse"
            notes.append(f"elastic fusion: track median "
                         f"{fu[1]:.0f} -> {fu[2]:.0f} m")
    if placed is None:
        return None
    med = _vpr_sequence_median_m(placed, vi, track_xy, vs, n_frames)
    return placed, mode, rot_deg, med, notes


def _pick_top_k_placement(entries):
    """Select among per-candidate placement results ``[(rank, med), ...]``.

    The matcher's ranking cannot see PLACEMENT quality, and a re-rank change
    can promote a candidate that scores better but places worse (London with
    the Viterbi track: matcher mean 901 -> 221 m while the anchored headline
    regressed 42.7 -> 70 m). The fused track median — the finest instrument —
    arbitrates instead, with the usual margin discipline: a challenger must
    beat rank 0's median by >=10% AND >=10 m, else rank 0 keeps the answer
    (rank 0 always wins ties; a missing rank 0 hands it to the best
    challenger). Returns the winning rank."""
    if not entries:
        return None
    by_rank = dict(entries)
    base_med = by_rank.get(0)
    best_rank, best_med = min(
        entries, key=lambda e: (np.inf if e[1] is None else e[1], e[0]))
    if 0 not in by_rank:
        return best_rank
    if (best_rank != 0 and base_med is not None and best_med is not None
            and best_med < 0.9 * base_med and base_med - best_med >= 10.0):
        return best_rank
    return 0


# Elastic fusion (_elastic_fuse_track): bend the rigidly-placed trajectory
# toward the full VPR track. Gates mirror the rotation refine's: a clear
# margin win AND an absolute explain bound AND a bounded deformation — a
# noisy track can only no-op. The start is hard-pinned: fusion corrects
# DRIFT, it must never move the finest instrument we have.
_FUSE_MIN_TRACK_PTS = 20
_FUSE_MARGIN_RATIO = 0.85
_FUSE_MARGIN_ABS_M = 15.0
_FUSE_MAX_DEFORM_M = 300.0
_FUSE_HUBER_M = 30.0
_FUSE_SMOOTH_D1 = 0.1
_FUSE_SMOOTH_D2 = 1000.0


def _elastic_fuse_track(traj_xy, track_frame_idx, track_xy, sims, n_frames):
    """Bend the placed trajectory toward the per-frame VPR track — the
    ELASTIC step after the rigid pin/scale/rotate chain.

    Rigid placement fixes position, phase, scale and heading but cannot
    touch the low-frequency VO DRIFT baked into the shape; the ~80-frame
    track can. Solves per-vertex offsets ``d`` minimising a similarity- and
    Huber-weighted data term at the matched arc rows plus first/second-
    difference smoothness (drift is low-frequency: the curvature prior is
    stiff, rigid translations/ramps of ``d`` are free), with ``d[0] = 0``
    hard-pinned. IRLS over the Huber weights, separable x/y solves.

    Returns ``(fused_traj, rigid_median_m, fused_median_m)`` only when the
    fused track fit clears the margin gates, stays inside the explain bound
    (``_ROT_EXPLAIN_M``) and the deformation stays under
    ``_FUSE_MAX_DEFORM_M``; ``None`` otherwise (keep the rigid placement).
    """
    traj = np.asarray(traj_xy, dtype=np.float64)
    track_xy = np.asarray(track_xy, dtype=np.float64)
    n = len(traj)
    if n < 8 or len(track_xy) < _FUSE_MIN_TRACK_PTS:
        return None
    rigid_med = _vpr_sequence_median_m(traj, track_frame_idx, track_xy, sims,
                                       n_frames)
    if rigid_med is None:
        return None
    frac = np.asarray(track_frame_idx, dtype=np.float64) / max(n_frames - 1, 1)
    rows = np.clip(np.round(frac * (n - 1)).astype(int), 0, n - 1)
    w_sim = np.clip(np.asarray(sims, dtype=np.float64), 0.0, None)
    if w_sim.sum() <= 0:
        w_sim = np.ones_like(w_sim)
    # Observations farther from the RIGID placement than the deformation
    # bound can never be explained by an admissible solution — they are
    # outliers by definition, and Huber alone is not enough: its influence
    # is CONSTANT with distance, so a contiguous far-off block acts as a
    # steady lateral force on an unobserved stretch and bows it hundreds of
    # metres (observed 640 m on the synthetic outlier case). Hard-zero them.
    dist0 = np.linalg.norm(traj[rows] - track_xy, axis=1)
    w_sim = np.where(dist0 <= _FUSE_MAX_DEFORM_M, w_sim, 0.0)
    if float(w_sim.sum()) <= 0 or int((w_sim > 0).sum()) < _FUSE_MIN_TRACK_PTS:
        return None

    # Smoothness normal-matrix blocks (constant across IRLS rounds).
    d1 = np.zeros((n - 1, n))
    d1[np.arange(n - 1), np.arange(n - 1)] = -1.0
    d1[np.arange(n - 1), np.arange(1, n)] = 1.0
    d2 = np.zeros((n - 2, n))
    d2[np.arange(n - 2), np.arange(n - 2)] = 1.0
    d2[np.arange(n - 2), np.arange(1, n - 1)] = -2.0
    d2[np.arange(n - 2), np.arange(2, n)] = 1.0
    reg = _FUSE_SMOOTH_D1 * d1.T @ d1 + _FUSE_SMOOTH_D2 * d2.T @ d2

    d = np.zeros((n, 2))
    misfit = track_xy - traj[rows]         # raw data term, fixed across IRLS
    w_cap = np.full(len(rows), np.inf)
    for _ in range(5):
        # Huber weights from the current iterate, but MONOTONE: capped by the
        # previous round's weight. Without the cap, a contiguous block of
        # confident-but-wrong observations captures the solve — moving toward
        # them shrinks their residual, which RAISES their weight, which pulls
        # harder (observed: max|d| 640 -> 1842 m across rounds). With the cap
        # an observation's influence can only decay.
        dist = np.linalg.norm(traj[rows] + d[rows] - track_xy, axis=1)
        w = np.minimum(
            w_cap,
            w_sim * np.minimum(1.0, _FUSE_HUBER_M / np.maximum(dist, 1e-9)))
        w_cap = w
        A = reg.copy()
        b = np.zeros((n, 2))
        np.add.at(A, (rows, rows), w)
        np.add.at(b, rows, w[:, None] * misfit)
        A[0, :] = 0.0
        A[:, 0] = 0.0
        A[0, 0] = 1.0                      # hard pin: d[0] = 0
        b[0] = 0.0
        try:
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
    if float(np.linalg.norm(d, axis=1).max()) > _FUSE_MAX_DEFORM_M:
        return None
    fused = traj + d
    fused_med = _vpr_sequence_median_m(fused, track_frame_idx, track_xy, sims,
                                       n_frames)
    if (fused_med is not None
            and fused_med < _FUSE_MARGIN_RATIO * rigid_med
            and rigid_med - fused_med >= _FUSE_MARGIN_ABS_M
            and fused_med <= _ROT_EXPLAIN_M):
        return fused, float(rigid_med), float(fused_med)
    return None


# NOTE (2026-07-05): an along-track PHASE refine (trim the route front so the
# whole route best-explains the full VPR track) was prototyped and REVERTED as
# a measured negative. The distance-viz decomposition showed KITTI-0033's start
# error (228 m) is a phase offset on a CORRECT corridor (the route passes 3.7 m
# from the true start) with a 224 m oracle headroom. But the trim search on the
# real 0033 track improved the weighted median only ~12% (231->205 m) at an 18%
# trim vs a 15% oracle: the 0033/0009 VPR tracks are ~210 m-noisy, so the phase
# is unrecoverable from them. Controls (comma/London) correctly showed zero
# signal (best trim = 0 rows). 0009's 301 m start is a uniform CORRIDOR offset
# (the route fits its track at 60 m median), not a phase error. Conclusion: the
# start residual is retrieval-bound everywhere there is headroom; no VPR-driven
# algorithm beats the track noise. Only denser refs (MLY_TOKEN) move it.


def _arc_param_times(traj_xy, traj_t_sec, n_points: int):
    """Video time of each point of an arc-length-resampled trajectory.

    The matcher resamples the VO path to ``n_points`` equidistant in ARC
    LENGTH (visual_odometry.resample_uniform), so candidate routes carry no
    times. Map each sample's arc position back through the VO path's
    cumulative arc length to its frame time. Returns None when the inputs
    cannot support it (degenerate path, length mismatch).
    """
    xy = np.asarray(traj_xy, dtype=np.float64)
    t = np.asarray(traj_t_sec, dtype=np.float64)
    if len(xy) < 2 or len(xy) != len(t) or n_points < 1:
        return None
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if not np.isfinite(s).all() or s[-1] <= 0:
        return None
    return np.interp(np.linspace(0.0, s[-1], n_points), s, t)


def _final_position_reports(
    candidates,
    road,
    *,
    matches,
    ranking: str,
    world_route_latlon=None,
    world_route_t_sec=None,
    cand_t_sec=None,
    world_route_source: str = "ocr_anchor_pin",
    pinned_ranking: str | None = None,
    anchored_cand=None,
    anchor_origin: str | None = None,
    prior_latlon=None,
):
    """Build (headline, matcher, pinned) position reports.

    Three answers can exist, each reported under an honest label:

    * ``matcher`` — ALWAYS the raw matcher pick (``source: "matcher"``):
      the top candidate's own aligned route, nothing else applied. It is
      the "trajectory matcher alone" ablation, so it must never carry a
      route another channel moved.
    * ``pinned`` — the same candidate's route georeferenced by anchors
      (``world_route_latlon``): the scale-locked OCR anchor pin
      (``source: "ocr_anchor_pin"``) or the anchor similarity fit
      (``"anchor_georef"``). None when no such route exists. This used to
      be emitted AS ``matcher`` (``source: "matcher"``), so every
      "matcher-only" number on sign-rich clips was really matcher + pin.
    * ``headline`` — the answer ``result["position"]`` carries: the
      anchor-primary route (``"anchor_primary_vpr"|"anchor_primary_vlm"``)
      when a retrieval/VLM prior placed one, else ``pinned``, else
      ``matcher``. VPR placement outranks the OCR pin because it measured
      better where both fired (Ulm 4K: 48 m vs 160 m mean route error).
    """
    matcher_position = build_position_report(
        candidates[0], road, matches=matches, ranking=ranking,
        route_t_sec=cand_t_sec,
    )
    if matcher_position is not None:
        matcher_position["source"] = "matcher"

    pinned_position = None
    if world_route_latlon is not None:
        pinned_position = build_position_report(
            candidates[0], road, matches=matches,
            ranking=pinned_ranking or ranking,
            world_route_latlon=world_route_latlon,
            route_t_sec=world_route_t_sec,
        )
        if pinned_position is not None:
            pinned_position["source"] = world_route_source

    headline = pinned_position if pinned_position is not None else matcher_position
    if anchored_cand is not None:
        origin = anchor_origin or "vpr"
        anchored = build_position_report(
            anchored_cand,
            road,
            matches=matches,
            ranking=f"anchored({origin})",
            route_t_sec=cand_t_sec,
        )
        if anchored is not None:
            anchored["source"] = f"anchor_primary_{origin}"
            if prior_latlon is not None:
                anchored["prior_latlon"] = [
                    float(prior_latlon[0]), float(prior_latlon[1]),
                ]
            headline = anchored
    return headline, matcher_position, pinned_position


def _orienternet_refine(cfg, frames, cand, road, position, result,
                        vpr_track=None) -> None:
    """Refine the shape-matched coarse position with OrienterNet (BEV->OSM).

    Samples keyframes across the analyzed window, maps each to a coarse
    lat/lon from the shape-matched route, and runs OrienterNet sequential
    fusion. Updates ``position`` in place (keeps the coarse estimate under
    ``coarse_*``) and records the refined-vs-GT error when GT is present.

    GUARDED when a per-frame VPR track exists: the refinement must explain
    the track at least as well as the route it started from, or it is
    rejected. The integration predates the anchored placement — its prior
    used to be 500+ m off and any refinement helped; now the anchored route
    can be 15-31 m from GT and an unguarded diffuse BEV->OSM belief drags it
    off (validated 2026-07-04: London start 31 -> 190 m, comma 15 -> 160 m).
    """
    print("[10c] OrienterNet metric refinement (neural BEV->OSM)")
    try:
        from ..orienternet_localizer import refine_route
        from ..position import xy_to_latlon
        fr = frames.frames
        route_ll = xy_to_latlon(np.asarray(cand.aligned_traj_xy, dtype=np.float64), road.crs)
        if len(fr) < 2 or len(route_ll) < 2:
            print("      -> too few frames; skipping")
            return
        # Dense keyframes over the WHOLE route (the aligner chains them via
        # odometry and is anchored at the middle, so confident frames pin
        # even the loop-phase-ambiguous endpoints). Needs correct metric
        # scale — run with --use-ipm-scale --enable-loop-closure.
        n_kf = max(cfg.orienternet_keyframes, 1)
        idxs = np.linspace(0, len(fr) - 1, n_kf).round().astype(int)
        kf = [fr[i] for i in idxs]
        rfrac = (idxs / max(len(fr) - 1, 1) * (len(route_ll) - 1)).round().astype(int)
        prior_ll = route_ll[rfrac]
        # focal_px=None -> OrienterNet auto-calibrates the camera FOV.
        # cache_dir persists the raw OSM JSON per bbox, so sweep reruns and
        # tile experiments stop depending on osm.org's mood (a tile-256
        # fetch once stalled 35+ min and killed the whole experiment).
        refined = refine_route(kf, prior_ll, None, tile_m=cfg.orienternet_tile_m,
                               cache_dir=cfg.data_dir / "osm_tiles")
        if refined is None:
            print("      -> OrienterNet unavailable; keeping shape-match position")
            return
        # Project the refined route once — reused by the track gate and the GT
        # evaluation below (both need it in the graph CRS).
        from ..position import latlon_to_xy as _refine_ll2xy
        refined_xy = _refine_ll2xy(np.asarray(refined, dtype=np.float64), road.crs)
        if vpr_track is not None:
            try:
                from ..position import latlon_to_xy
                tk_xy = latlon_to_xy(np.asarray(vpr_track[1], dtype=np.float64),
                                     road.crs)
                base_med = _vpr_sequence_median_m(
                    np.asarray(cand.aligned_traj_xy, dtype=np.float64),
                    vpr_track[0], tk_xy, vpr_track[2], len(fr))
                ref_xy = refined_xy
                ref_med = _vpr_sequence_median_m(
                    ref_xy, vpr_track[0], tk_xy, vpr_track[2], len(fr))
                reject = None
                if (base_med is not None and ref_med is not None
                        and ref_med > 1.10 * base_med):
                    reject = (f"track fit {ref_med:.0f} m vs route "
                              f"{base_med:.0f} m")
                # START gate: a pinned start is the pipeline's most accurate
                # statistic (offline 15-135 m), and the ~100-200 m-median VPR
                # track cannot see a 30 m start error — on London the refined
                # route passed the track gate while dragging the pinned start
                # 159 m (GT 31 -> 190 m). No track-fit escape hatch: on comma
                # a ">=25% better track fit" excuse let the refinement move a
                # 15 m-accurate pin to a 160 m answer (self-similar highway
                # track = the NOISY statistic; it must never overrule the
                # pin). On pinned runs a big start move is simply rejected.
                _start_shift = float(np.linalg.norm(
                    ref_xy[0]
                    - np.asarray(cand.aligned_traj_xy, dtype=np.float64)[0]))
                _pinned = "startpin" in str(
                    (result.get("anchored_position") or {}).get("source", ""))
                if reject is None and _pinned and _start_shift > 100.0:
                    reject = (f"start moved {_start_shift:.0f} m off the "
                              f"pinned start")
                if reject is not None:
                    print(f"      -> OrienterNet refinement REJECTED ({reject}); "
                          f"keeping prior route")
                    result["orienternet"] = {
                        "applied": False, "reason": reject,
                        "track_fit_refined_m": (round(ref_med, 1)
                                                if ref_med is not None else None),
                        "track_fit_route_m": (round(base_med, 1)
                                              if base_med is not None else None),
                    }
                    return
            except Exception as e:
                print(f"      -> OrienterNet track gate unavailable ({e}); "
                      f"accepting refinement ungated")
        lat0, lon0 = float(refined[0][0]), float(refined[0][1])
        position["coarse_latitude"] = position["latitude"]
        position["coarse_longitude"] = position["longitude"]
        position["latitude"] = round(lat0, 6)
        position["longitude"] = round(lon0, 6)
        position["ranking"] = position.get("ranking", "") + "+orienternet"
        position["google_maps_url"] = google_maps_url(lat0, lon0)
        position["orienternet_route_latlon"] = [
            [round(float(a), 6), round(float(b), 6)] for a, b in refined]
        # The reported route must be the route the headline start belongs
        # to: keep the coarse one aside and report the refined keyframe
        # route (its per-point times are not tracked, so drop route_t_sec).
        position["coarse_route_latlon"] = position.get("route_latlon")
        position["route_latlon"] = position["orienternet_route_latlon"]
        if "route_t_sec" in position:
            position["coarse_route_t_sec"] = position.pop("route_t_sec")
        result["orienternet"] = {"applied": True, "keyframes": len(kf),
                                 "tile_m": cfg.orienternet_tile_m}
        print(f"      -> refined start: {lat0:.6f}, {lon0:.6f}")
        if cfg.ground_truth_waypoints:
            from ..evaluator import _segment_to_polyline_distance, load_gt_waypoints
            from ..position import latlon_to_xy
            wp_xy = latlon_to_xy(load_gt_waypoints(cfg.ground_truth_waypoints), road.crs)
            rt_xy = refined_xy
            d0 = float(np.linalg.norm(rt_xy[0] - wp_xy[0]))
            dm = float(np.mean([_segment_to_polyline_distance(w_, rt_xy) for w_ in wp_xy]))
            position["orienternet_gt_start_error_m"] = round(d0, 1)
            position["orienternet_gt_mean_route_error_m"] = round(dm, 1)
            # The refinement replaced the reported coordinates, so the
            # headline gt_* keys must describe THEM — leaving the anchored
            # eval there made the sweep table report the anchored error for
            # an OrienterNet position (stale-looking 30 m rows for 190 m
            # answers).
            if "gt_start_error_m" in position:
                position["gt_start_error_m"] = round(d0, 1)
            if "gt_mean_route_error_m" in position:
                position["gt_mean_route_error_m"] = round(dm, 1)
            print(f"      -> OrienterNet vs GT: start {d0:.1f} m, mean route {dm:.1f} m")
    except Exception as e:
        print(f"      -> OrienterNet refine failed ({e})")


def anchor_primary_placement(ctx: RunContext) -> None:
    """ANCHOR-PRIMARY position -> ``ctx.anchored_cand`` / ``result['anchored_position']``.

    A route-accurate absolute prior (VPR) beats any matcher candidate, so place
    the best trajectory AT the prior: start-pin, scale retry, orientation refine,
    elastic fusion, top-k arbitration; falls back to a centroid translation.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    result = ctx.result
    frames = ctx.frames
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    anchor_origin = ctx.anchor_origin
    road = ctx.road
    candidates = ctx.candidates
    t3_mesh_ref = ctx.t3_mesh_ref
    t3_samples_ref = ctx.t3_samples_ref
    t3_K_ref = ctx.t3_K_ref
    t3_skyline_ref = ctx.t3_skyline_ref

    # 10a*. ANCHOR-PRIMARY position. A route-accurate absolute prior (VPR,
    # ~141-200 m on Ulm) beats ANY matcher candidate (pick 664 m, best-in-pool
    # 521 m), and the matcher's candidate GENERATION — not its ranking — is the
    # wall (gating its disc hurts; re-ranking can't beat the prior). So place the
    # best trajectory AT the prior: translate the top candidate so its centroid
    # sits on the prior, and report THAT anchored start as the primary answer.
    # Orientation/shape come from the matcher; absolute position from the prior.
    ctx.anchored_cand = None
    if vpr_center is not None and candidates:
        # Place the top candidate's trajectory on the VPR prior. START-PIN
        # (default): the per-frame VPR track's START-region robust centre
        # estimates the true route start well (offline: 15-135 m vs a
        # centroid-only start error of 147-1084 m), so translate the trajectory
        # to put its START there — fixing the along-track PHASE the centroid
        # leaves free (comma start 1120 -> 15 m). If that drifts the centroid
        # past a sanity guard, the candidate's LENGTH is wrong, so correct the
        # scale from the VPR end-to-end extent and retry. The pinned candidate
        # then gets an ORIENTATION refine — rotate about the trusted start to
        # best explain the full per-frame track (the one residual position,
        # phase and scale leave; a 2-point rigid fit that handed the noisy END
        # estimate the rotation regressed clips, so the refine scores the
        # whole track's robust median instead and is margin-gated: it no-ops
        # unless it demonstrably locks on). Anything that fails every guard
        # falls back to the plain centroid translation.
        import dataclasses
        from ..position import latlon_to_xy as _ll2xy, xy_to_latlon as _xy2ll2
        _vpr_xy = _ll2xy(np.asarray([[vpr_center[0], vpr_center[1]]], float), road.crs)[0]
        _placed, _mode, _rot_deg, _best_rank = None, "vpr_centroid", None, 0
        _geo = (_vpr_track_geometry(vpr_track, road.crs)
                if anchor_origin == "vpr" else None)
        if _geo is not None:
            try:
                _vi, _vll, _vs, _s_xy, _e_xy = _geo
                _track_xy = _ll2xy(_vll, road.crs)
                # TOP-K placement selection: run the full pin -> scale ->
                # rotate -> fuse chain for the top 3 candidates and let the
                # fused track median arbitrate (with a >=10%+10 m challenger
                # margin — see _pick_top_k_placement for why the matcher's
                # own ranking cannot be trusted with this decision).
                _pk = {}
                for _ci, _c in enumerate(candidates[:3]):
                    _res = _place_candidate_on_track(
                        np.asarray(_c.aligned_traj_xy, dtype=np.float64),
                        _s_xy, _e_xy, _vpr_xy, _vi, _track_xy, _vs,
                        len(frames.frames))
                    if _res is not None:
                        _pk[_ci] = _res
                _win = _pick_top_k_placement(
                    [(ci, r[3]) for ci, r in _pk.items()])
                if _win is not None:
                    _best_rank = _win
                    _placed, _mode, _rot_deg, _med, _notes = _pk[_win]
                    for _n in _notes:
                        print(f"      -> {_n}")
                    if _win != 0:
                        _m0 = _pk[0][3] if 0 in _pk else None
                        print(f"      -> placement pick: candidate #{_win + 1} "
                              f"(track median {_med:.0f} m vs #1's "
                              f"{'unplaceable' if _m0 is None else f'{_m0:.0f} m'})")
            except Exception as e:
                print(f"      -> start-pin placement failed ({e}); "
                      f"falling back to centroid")
                _record_channel(channel_status, 'start_pin_placement', "failed", error=e)
                _placed = None
        if _placed is None:
            _best_rank = 0
            _traj0 = np.asarray(candidates[0].aligned_traj_xy, dtype=np.float64)
            _placed = _traj0 + (_vpr_xy - _traj0.mean(axis=0))
        ctx.anchored_cand = dataclasses.replace(candidates[_best_rank],
                                                aligned_traj_xy=_placed)
        # Optional metric placement refinement (--tile3d-refine): sharpen the
        # VPR-pinned route by aligning its rendered skyline to the video
        # (LoD-Loc-style silhouette pose optimization). No-op unless the
        # tile3d mesh/samples are available and a transform beats identity.
        if cfg.tile3d_refine and t3_mesh_ref is not None:
            try:
                from ..tile3d_match import refine_placement_skyline
                _rk, _rwh = t3_K_ref
                _placed_ref, _rinfo = refine_placement_skyline(
                    t3_mesh_ref, t3_samples_ref, _placed, _rk, _rwh,
                    skyline_fn=t3_skyline_ref)
                if _rinfo.get("applied"):
                    _placed = _placed_ref
                    ctx.anchored_cand = dataclasses.replace(
                        candidates[_best_rank], aligned_traj_xy=_placed)
                    result.setdefault("tile3d", {})["placement_refine"] = {
                        "shift_m": round(_rinfo["shift_m"], 1),
                        "rot_deg": round(_rinfo["rot_deg"], 2),
                        "agree_before": round(_rinfo["score_before"], 3),
                        "agree_after": round(_rinfo["score_after"], 3),
                    }
                    print(f"      -> tile3d placement refine: shift "
                          f"{_rinfo['shift_m']:.0f} m, rot "
                          f"{_rinfo['rot_deg']:.1f} deg (skyline agree "
                          f"{_rinfo['score_before']:.2f}->"
                          f"{_rinfo['score_after']:.2f})")
            except Exception as e:
                print(f"      -> tile3d placement refine failed ({e}); skipped")
                _record_channel(channel_status, 'tile3d_refine', "failed", error=e)
        _astart = _xy2ll2(_placed[:1], road.crs)[0]
        # The source names the channel + placement mode: a VLM district centroid
        # is far coarser than a MegaLoc VPR endpoint fit; accuracy studies need
        # to tell them apart.
        _anchor_src = "vlm_geocode" if anchor_origin == "vlm" else _mode
        result["anchored_position"] = {
            "lat": float(_astart[0]), "lon": float(_astart[1]), "source": _anchor_src,
            "prior_latlon": [float(vpr_center[0]), float(vpr_center[1])],
            "candidate_rank": int(_best_rank) + 1,
        }
        if _rot_deg is not None:
            result["anchored_position"]["rotation_deg"] = round(float(_rot_deg), 1)
        print(f"[10a*] ANCHOR-PRIMARY ({_anchor_src}): start "
              f"{_astart[0]:.5f}, {_astart[1]:.5f}")


def _pin_route_to_anchors(ctx: RunContext) -> tuple:
    """Georeference the route from OCR anchors (scale-lock pin, else the anchor similarity fit).

    Returns ``(world_route_latlon, world_route_t_sec, world_route_source, pos_ranking)``.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames
    road = ctx.road
    match_xz = ctx.match_xz
    match_ts = ctx.match_ts
    ocr_anchors = ctx.ocr_anchors
    street_anchors = ctx.street_anchors
    anchor_world_route = ctx.anchor_world_route
    locked_scale = ctx.locked_scale
    candidates = ctx.candidates
    ranking_mode = ctx.ranking_mode

    world_route_latlon = None
    world_route_t_sec = None
    world_route_source = "ocr_anchor_pin"
    pos_ranking = ranking_mode
    # Scale-lock + single-anchor pin: keep the locked metric extent
    # AND the shape-fit rotation, but pin absolute position with the
    # most confident anchor — fixes the start drift that scale-lock
    # alone leaves. Takes precedence over the idea-2 georeference.
    # Street-name anchors are first-class pin points (the temporally
    # valid "you are here" anchors that drove the Ulm gains), so the
    # pin also runs on street-name-only runs — it used to require a
    # geocodable POI, silently skipping when OCR read only street
    # plates (the common 4K case).
    if (cfg.scale_lock and locked_scale is not None
            and (ocr_anchors or street_anchors)):
        try:
            from ..position import xy_to_latlon as _xy2ll
            from ..scale_recovery import vo_positions_at_times
            from ..text_anchor import anchors_to_xy as _a2xy
            from ..trajectory_matching import anchor_pinned_route
            # Least-squares translation over the anchors, weighted by
            # confidence AND proximity to the matched route. The
            # proximity term is the key robustness fix: a precise
            # on-route anchor (the car drove past the building) sits
            # on the walk, while a coarse *direction* sign geocodes
            # 100s of m off it — and OCR confidence can't tell them
            # apart (both 1.00). Down-weighting by distance-to-walk
            # lets the on-route anchors dominate. (Naive equal/conf
            # weighting over all anchors was worse: 307 vs 146 m.)
            walk = np.asarray(candidates[0].walk_xy, dtype=np.float64)
            # POI anchors: geocoded landmark points.
            names = [a.name for a in ocr_anchors]
            a_world = list(_a2xy(ocr_anchors, road.crs))
            a_t = [a.t_sec for a in ocr_anchors]
            a_conf = [a.confidence for a in ocr_anchors]
            # Street-name anchors: temporally-valid "you are here"
            # points (the car is ON the street when its plate is read),
            # often earlier on the route than the POI signs. Each
            # street's pin point is the node where the chosen route
            # meets that street. The residual-cluster pin drops any
            # that are temporally inconsistent, so this is safe.
            for sa in street_anchors:
                if not sa.node_ids:
                    continue
                pts = np.array([[road.graph.nodes[n]["x"], road.graph.nodes[n]["y"]]
                                for n in sa.node_ids], dtype=np.float64)
                d = np.array([np.linalg.norm(walk - p, axis=1).min() for p in pts])
                a_world.append(pts[int(np.argmin(d))])
                a_t.append(sa.t_sec)
                a_conf.append(sa.confidence)
                names.append(sa.name)
            if not a_world:
                raise ValueError("no pinnable anchors (street anchors "
                                 "carry no graph nodes)")
            a_world = np.asarray(a_world, dtype=np.float64)
            # match_ts, not frames.timestamps: the staged OpenVO/DA3
            # trajectory has fewer rows than frames, and indexing it
            # with per-frame times was the silent IndexError that
            # reverted the pin (the 412->160 m win) on default runs.
            if len(match_ts) != len(match_xz):
                raise ValueError(
                    f"timestamp axis desynced from trajectory "
                    f"({len(match_ts)} ts vs {len(match_xz)} poses)")
            a_vo = vo_positions_at_times(
                match_xz, match_ts, np.asarray(a_t))
            dist_to_walk = np.array([
                np.linalg.norm(walk - w, axis=1).min() for w in a_world])
            wts = np.asarray(a_conf) / (1.0 + dist_to_walk / 50.0)
            route_xy = anchor_pinned_route(
                match_xz, candidates[0].walk_xy, locked_scale,
                a_vo, a_world, weights=wts,
            )
            world_route_latlon = _xy2ll(route_xy, road.crs)
            world_route_t_sec = np.asarray(match_ts, dtype=np.float64)
            n_poi = len(ocr_anchors)
            n_st = len(a_world) - n_poi
            pos_ranking = (ranking_mode
                           + f"+scalelock+anchor-pin({n_poi}poi+{n_st}street)")
            lead_i = int(np.argmax(wts))
            print(f"      -> route pinned (proximity-weighted LS) over "
                  f"{len(a_world)} anchor(s): {n_poi} POI + {n_st} street; "
                  f"lead={names[lead_i]!r} "
                  f"(t={a_t[lead_i]:.0f}s, d_to_walk={dist_to_walk[lead_i]:.0f} m)")
        except Exception as e:
            print(f"      -> anchor-pin failed: {e}")
            _record_channel(channel_status, 'anchor_pin', "failed", error=e)
            world_route_latlon = None
    if world_route_latlon is None and anchor_world_route is not None:
        from ..position import xy_to_latlon as _xy2ll
        try:
            world_route_latlon = _xy2ll(anchor_world_route, road.crs)
            world_route_t_sec = _match_timestamps(
                frames.timestamps, len(anchor_world_route))
            world_route_source = "anchor_georef"
            pos_ranking = ranking_mode + "+anchor-georef"
        except Exception:
            world_route_latlon = None
    return world_route_latlon, world_route_t_sec, world_route_source, pos_ranking


def final_position(ctx: RunContext) -> None:
    """Final position reports (headline / matcher / pinned), hypotheses, OrienterNet refinement.

    Output contract (see ``_final_position_reports``): ``result['position']`` is the
    HEADLINE - the anchored route when anchor-primary fired, else the anchor-pinned
    route, else the matcher pick.
    """
    cfg = ctx.cfg
    result = ctx.result
    frames = ctx.frames
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    anchor_origin = ctx.anchor_origin
    road = ctx.road
    match_xz = ctx.match_xz
    match_ts = ctx.match_ts
    candidates = ctx.candidates
    geom_pool = ctx.geom_pool
    ranking_mode = ctx.ranking_mode
    anchored_cand = ctx.anchored_cand
    waypoint_evals = ctx.waypoint_evals
    anchored_eval = ctx.anchored_eval

    if not candidates:
        return
    for i, cand in enumerate(candidates):
        latlon = candidate_center_latlon(cand, road)
        if latlon is not None:
            result["matches"][i]["center_latlon"] = [
                round(latlon[0], 6), round(latlon[1], 6),
            ]

    world_route_latlon, world_route_t_sec, world_route_source, pos_ranking = (
        _pin_route_to_anchors(ctx))
    # Output contract (see _final_position_reports): result["position"]
    # is the HEADLINE — the anchored route when anchor-primary fired
    # (VPR/VLM prior), else the anchor-pinned route, else the matcher
    # pick. result["matcher_position"] is always the raw matcher pick;
    # result["pinned_position"] carries the anchor-pinned route when one
    # was built.
    position, matcher_position, pinned_position = _final_position_reports(
        candidates,
        road,
        matches=result["matches"],
        ranking=ranking_mode,
        world_route_latlon=world_route_latlon,
        world_route_t_sec=world_route_t_sec,
        cand_t_sec=_arc_param_times(
            match_xz, match_ts, len(candidates[0].aligned_traj_xy)),
        world_route_source=world_route_source,
        pinned_ranking=pos_ranking,
        anchored_cand=anchored_cand,
        anchor_origin=anchor_origin,
        prior_latlon=vpr_center,
    )
    # GT errors (only when the caller supplied GT for in-run scoring; the
    # benchmark scores offline instead). Each report gets its OWN route's
    # numbers — the pinned route used to be scored under "matcher".
    if matcher_position is not None and waypoint_evals:
        matcher_position["gt_start_error_m"] = round(
            waypoint_evals[0].start_error_m, 1)
        matcher_position["gt_mean_route_error_m"] = round(
            waypoint_evals[0].mean_route_error_m, 1)
    if (pinned_position is not None and world_route_latlon is not None
            and cfg.ground_truth_waypoints):
        try:
            from ..evaluator import _segment_to_polyline_distance
            from ..position import latlon_to_xy
            wps = load_gt_waypoints(cfg.ground_truth_waypoints)
            wp_xy = latlon_to_xy(wps, road.crs)
            rt_xy = latlon_to_xy(world_route_latlon, road.crs)
            d0 = float(np.linalg.norm(rt_xy[0] - wp_xy[0]))
            dm = float(np.mean([
                _segment_to_polyline_distance(w, rt_xy) for w in wp_xy]))
            pinned_position["gt_start_error_m"] = round(d0, 1)
            pinned_position["gt_mean_route_error_m"] = round(dm, 1)
        except Exception as e:
            print(f"      -> pinned-route GT eval failed: {e}")
    # The anchored headline carries the anchored route's own numbers.
    if (position is not None and anchored_eval is not None
            and position is not matcher_position
            and position is not pinned_position):
        position["gt_start_error_m"] = round(anchored_eval.start_error_m, 1)
        position["gt_mean_route_error_m"] = round(
            anchored_eval.mean_route_error_m, 1)
    if matcher_position is not None:
        # Written unconditionally (not inside the headline block) so
        # the contract's "always present" holds even if the headline
        # report fails to build.
        result["matcher_position"] = matcher_position
    if pinned_position is not None:
        result["pinned_position"] = pinned_position
    if position is not None:
        # Calibrated multi-hypothesis output. Collapse the candidate
        # pool (final-ranked top_k + geometric tail) into distinct
        # location hypotheses and attach a confidence derived from
        # their spatial AGREEMENT — honest where the winner's own
        # shape score is not (see monocular_osm/hypotheses.py).
        ranked_pool = candidates + geom_pool[cfg.top_k:]
        hyps = distinct_hypotheses(ranked_pool, road, top_n=5)
        if hyps:
            spatial_conf = hypothesis_confidence(ranked_pool, hyps)
            position["spatial_confidence"] = spatial_conf
            def _hyp_route(h):
                # The representative candidate's aligned route, so an
                # evaluator can score the shortlist route-to-route, not
                # only by start point.
                try:
                    from ..position import xy_to_latlon as _hxy2ll
                    _xy = np.asarray(ranked_pool[h.candidate_index].aligned_traj_xy,
                                     dtype=np.float64)
                    return [[round(float(a), 6), round(float(b), 6)]
                            for a, b in _hxy2ll(_xy, road.crs)]
                except Exception:
                    return None

            position["hypotheses"] = [
                {
                    "rank": h.rank,
                    "latitude": round(h.lat, 6),
                    "longitude": round(h.lon, 6),
                    "street_names": h.street_names,
                    "support": h.support,
                    "google_maps_url": google_maps_url(h.lat, h.lon),
                    "route_latlon": _hyp_route(h),
                }
                for h in hyps
            ]

        # OrienterNet metric refinement (neural BEV->OSM). Refines the
        # coarse route to ~metric accuracy by fusing per-frame
        # BEV->OSM beliefs along it. The prior route is the ANCHORED
        # candidate when it exists — OrienterNet's per-keyframe tiles
        # only cover ~160 m around the prior, so it needs the best
        # prior the pipeline has (the anchored route, ~91-236 m on
        # Ulm), not the blind shape pick (~500-700 m off).
        if cfg.use_orienternet and candidates:
            _orienternet_refine(
                cfg, frames,
                anchored_cand if anchored_cand is not None else candidates[0],
                road, position, result, vpr_track=vpr_track)

        result["position"] = position
        print()
        print(format_position_summary(position))
        if (matcher_position is not None
                and position is not matcher_position):
            print(f"  Matcher pick (non-headline): "
                  f"{matcher_position['latitude']:.6f}, "
                  f"{matcher_position['longitude']:.6f}  "
                  f"({matcher_position['google_maps_url']})")
        if hyps and len(hyps) > 1:
            sc = position.get("spatial_confidence", {})
            print(f"  Spatial confidence: {sc.get('level', '?')} "
                  f"(candidates {'agree on' if sc.get('level') == 'high' else 'are spread across'} "
                  f"{len(hyps)} place(s); concentration {sc.get('concentration')}, "
                  f"spread {sc.get('spread_m')} m)")
            print("  Top location hypotheses (the true route is most likely one of these):")
            for h in hyps:
                print(f"    {h.rank}. {h.lat:.5f}, {h.lon:.5f}  "
                      f"[support {h.support}]  "
                      f"{', '.join(h.street_names) or '(unnamed)'}")
    else:
        result["position_error"] = (
            "could not convert the matched route to WGS84 "
            f"(road graph CRS: {road.crs!r})"
        )
        print(f"      -> position unavailable: {result['position_error']}")
