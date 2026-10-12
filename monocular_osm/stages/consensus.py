"""Stage 8: weighted rank-fusion consensus over the candidate channels.

Turn-sequence, OCR-anchor, plate, VPR and sun-heading signals are computed per
candidate and fused with the aerial / tile3d / shape ranks by
:mod:`monocular_osm.fusion` (the published weights, ablatable via
``cfg.fusion_weights``).
"""

from __future__ import annotations

import numpy as np

from ..fusion import (
    CandidateFeatures,
    consensus_order as fusion_consensus_order,
    effective_weights as fusion_effective_weights,
    fused_score as fusion_fused_score,
    write_feature_table,
)
from ..geo import geodesic_distance_m
from .context import RunContext
from .vpr_track import (
    _PLATE_FREE_RADIUS_M,
    _vpr_distance_penalty,
    _VPR_FREE_RADIUS_M,
    _vpr_sequence_median_m,
    _vpr_track_geometry,
)


def _mean_bearing_deg(xy: np.ndarray) -> float | None:
    """Length-weighted circular mean compass bearing (0=N, 90=E) of a
    projected polyline, or ``None`` when degenerate."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return None
    seg = np.diff(xy, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    if lengths.sum() <= 1e-9:
        return None
    # Compass convention: x is easting, y is northing -> atan2(dx, dy).
    ang = np.arctan2(seg[:, 0], seg[:, 1])
    c = float((lengths * np.cos(ang)).sum())
    s = float((lengths * np.sin(ang)).sum())
    if abs(c) < 1e-12 and abs(s) < 1e-12:
        return None
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def _heading_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference between two compass headings."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# Sun-heading penalty shape: free inside the tolerance (route curvature +
# median-vs-mean mismatch), then grows to _W_SUN at a full 180-degree
# contradiction — enough to kill mirror/rotated walks on near-ties without
# overriding the geometric channels.
_SUN_FREE_DEG = 30.0
_W_SUN = 5.0
_SUN_MIN_CONFIDENCE = 0.6


def _sun_bearing_penalty(mean_bearing_deg: float, sun_heading_deg: float) -> float:
    d = _heading_diff_deg(mean_bearing_deg, sun_heading_deg)
    return _W_SUN * max(0.0, (d - _SUN_FREE_DEG) / (180.0 - _SUN_FREE_DEG))


def _turn_channel(ctx: RunContext) -> dict[int, int]:
    """Turn-sequence channel: a drift-robust topological descriptor.

    Computed and stored for diagnostics but deliberately NOT fused (weight 0):
    a turn count/pattern isn't unique in a dense city grid. Returns the turn rank.
    """
    result = ctx.result
    match_xz = ctx.match_xz
    candidates = ctx.candidates

    # Turn-sequence channel: a drift-robust topological descriptor.
    # Cheap (signature compare over candidate polylines, no city
    # scan), so it always runs here. Lower distance = better; rank
    # ascending. See turn_matching for why it survives the VO drift
    # that degrades the dense shape/bearing comparison.
    from ..turn_matching import score_candidates_by_turns
    turn_dists, query_turns = score_candidates_by_turns(
        match_xz, [c.walk_xy for c in candidates]
    )
    turn_order = sorted(range(len(candidates)),
                        key=lambda i: (turn_dists[i], i))
    turn_rank = {idx: r + 1 for r, idx in enumerate(turn_order)}
    for i in range(len(candidates)):
        result["matches"][i]["turn_match_distance"] = (
            round(turn_dists[i], 4) if np.isfinite(turn_dists[i]) else None
        )
        result["matches"][i]["turn_match_rank"] = turn_rank[i]
    print(f"        turn-sequence channel: query has {len(query_turns)} "
          f"significant turn(s); best turn-match = candidate "
          f"#{turn_order[0] + 1}")
    return turn_rank


def _anchor_channel(ctx: RunContext) -> tuple[dict[int, int], bool]:
    """OCR-anchor channel: distance from each candidate walk to the nearest anchor.

    The one *absolute* signal; inactive (uniform rank) when there are no
    in-city anchors. Returns ``(anchor_rank, have_anchors)``.
    """
    result = ctx.result
    anchor_xy = ctx.anchor_xy
    candidates = ctx.candidates

    # OCR-anchor channel: distance from each candidate walk to the
    # nearest geocoded POI anchor. The one *absolute* signal — when
    # present it's the most trustworthy, so it earns a strong fusion
    # weight. Inactive (all-inf → uniform rank, weight 0) when the
    # OCR-anchor channel is disabled or found no in-city anchors.
    from ..text_anchor import score_candidates_by_anchors
    anchor_dists = score_candidates_by_anchors(
        [c.walk_xy for c in candidates], anchor_xy
    )
    have_anchors = len(anchor_xy) > 0 and any(
        np.isfinite(d) for d in anchor_dists
    )
    if have_anchors:
        anchor_order = sorted(range(len(candidates)),
                              key=lambda i: (anchor_dists[i], i))
        anchor_rank = {idx: r + 1 for r, idx in enumerate(anchor_order)}
        for i in range(len(candidates)):
            result["matches"][i]["anchor_distance_m"] = (
                round(anchor_dists[i], 1) if np.isfinite(anchor_dists[i])
                else None
            )
            result["matches"][i]["anchor_rank"] = anchor_rank[i]
        print(f"        OCR-anchor channel: nearest candidate to an anchor "
              f"= #{anchor_order[0] + 1} ({anchor_dists[anchor_order[0]]:.0f} m)")
    else:
        anchor_rank = {i: i + 1 for i in range(len(candidates))}
    return anchor_rank, have_anchors


def _plate_channel(ctx: RunContext) -> tuple[dict[int, int], dict[int, float]]:
    """License-plate district prior: a shape-independent absolute region cue.

    Distance-based penalty (free inside 1.5 km of the district seat). Returns
    ``(plate_rank, plate_penalty)``.
    """
    result_plate_anchor = ctx.result_plate_anchor
    plate_center = ctx.plate_center
    road = ctx.road
    candidates = ctx.candidates

    # License-plate district anchor (shape-INDEPENDENT absolute prior): rank
    # candidates by distance from their start to the voted district centroid.
    # Unlike noisy OCR shop-sign anchors this is a reliable region cue, so it
    # gets a strong weight to pull the pick toward the right district and kill
    # gross km-scale errors. No-op (rank = shape) when no district was found.
    if plate_center is not None:
        from ..position import xy_to_latlon as _xy2ll
        _plat, _plon = plate_center
        _pd = []
        for _c in candidates:
            _ll = _xy2ll(np.asarray(_c.aligned_traj_xy[:1], dtype=np.float64), road.crs)
            _la, _lo = float(_ll[0][0]), float(_ll[0][1])
            _pd.append(geodesic_distance_m(_la, _lo, _plat, _plon))
        _order = np.argsort(_pd)
        plate_rank = {int(idx): r + 1 for r, idx in enumerate(_order)}
        # DISTANCE-based penalty (not rank — rank loses magnitude). Each
        # candidate is pushed down ∝ how far its start is from the district
        # centroid (km), scaled by the vote confidence (margin, capped x2).
        # This leaves near-centroid candidates almost untouched (so an
        # already-correct pool still competes on shape) while strongly
        # demoting gross km-scale errors — the plate anchor's real job.
        # Free-radius penalty: candidates within R0 of the district seat pay
        # nothing (trust shape there, so an already-correct in-district pick
        # is preserved); beyond R0 the penalty grows ∝ extra km, strongly
        # demoting gross errors. The district centroid is seat-biased, so a
        # too-tight R0 would hurt legit non-seat drives — 1.5 km keeps the
        # common city-centre case neutral while catching km-scale mispicks.
        _conf = min(2.0, float((result_plate_anchor or {}).get("margin", 1.0)))
        _R0 = _PLATE_FREE_RADIUS_M
        plate_penalty = {i: 10.0 * _conf * max(0.0, (_pd[i] - _R0) / 1000.0)
                         for i in range(len(candidates))}
        if result_plate_anchor is not None:
            result_plate_anchor["applied"] = True
    else:
        plate_rank = {i: i + 1 for i in range(len(candidates))}
        plate_penalty = {i: 0.0 for i in range(len(candidates))}
    return plate_rank, plate_penalty


def _vpr_channel(ctx: RunContext) -> dict[int, float]:
    """VPR re-rank channel: penalise candidates by distance from the VPR prior.

    Returns ``vpr_penalty`` (candidate index -> consensus penalty).
    """
    cfg = ctx.cfg
    frames = ctx.frames
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    anchor_origin = ctx.anchor_origin
    road = ctx.road
    candidates = ctx.candidates

    # VPR re-rank channel: the VPR prior marks the clip's filmed CENTRE and is
    # route-accurate (~190 m on Ulm — finer than the matcher itself), so we
    # penalise each candidate by how far its trajectory CENTROID sits from the
    # prior, with a free radius matching the prior's own error bar (see
    # _vpr_distance_penalty). This is the strongest shape-independent
    # selection signal we have; it pulls the pick to the true area.
    # --use-vpr-sequence (experimental) scores against the per-frame VPR
    # track instead: the sequence-median distance also penalises
    # correctly-centred but flipped/rotated candidates.
    if vpr_center is not None:
        from ..position import latlon_to_xy as _vll2xy, xy_to_latlon as _vxy2ll
        _vlat, _vlon = vpr_center
        track_xy = None
        if cfg.use_vpr_sequence and vpr_track is not None:
            try:
                track_xy = _vll2xy(
                    np.asarray(vpr_track[1], dtype=np.float64), road.crs)
            except Exception as e:
                print(f"        VPR sequence penalty unavailable ({e}); "
                      f"falling back to centroid")
        vpr_penalty = {}
        for i, _c in enumerate(candidates):
            _d = None
            if track_xy is not None:
                _d = _vpr_sequence_median_m(
                    _c.aligned_traj_xy, vpr_track[0], track_xy,
                    vpr_track[2], len(frames.frames))
            if _d is None:
                _cen = np.asarray(_c.aligned_traj_xy, dtype=np.float64).mean(axis=0)
                _ll = _vxy2ll(_cen[None, :], road.crs)
                _d = geodesic_distance_m(float(_ll[0][0]), float(_ll[0][1]),
                                         _vlat, _vlon)
            vpr_penalty[i] = _vpr_distance_penalty(
                _d, _VPR_FREE_RADIUS_M if anchor_origin == "vpr"
                else _PLATE_FREE_RADIUS_M)
    else:
        vpr_penalty = {i: 0.0 for i in range(len(candidates))}
    return vpr_penalty


def _vpr_chord_diagnostic(ctx: RunContext) -> None:
    """Diagnostic: the VPR track's start->end extent vs the candidate chord."""
    result = ctx.result
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    road = ctx.road
    candidates = ctx.candidates

    # VPR chord diagnostic: the VPR track's start->end straight-line extent
    # is a scale-locked estimate of the true route's endpoint distance.
    # Candidate re-ranking by chord is a NO-OP here (every candidate
    # inherits the VO trajectory's shape+scale, so they share one chord —
    # Ulm all 534 m vs a VPR extent of 1134 m; London all 1453 m vs
    # 1022 m). So the chord mismatch is a VO-TRAJECTORY error, not a
    # selection one; we store both as a diagnostic for the VO-scale fix.
    if vpr_center is not None and vpr_track is not None and candidates:
        try:
            _g = _vpr_track_geometry(vpr_track, road.crs)
            _vpr_chord = (float(np.linalg.norm(_g[4] - _g[3]))
                          if _g is not None else 0.0)
            if _vpr_chord > 200.0:
                _t0 = np.asarray(candidates[0].aligned_traj_xy, dtype=np.float64)
                result["vpr_chord_m"] = round(_vpr_chord, 1)
                result["candidate_chord_m"] = round(
                    float(np.linalg.norm(_t0[-1] - _t0[0])), 1)
        except Exception:
            pass


def _sun_channel(ctx: RunContext) -> tuple[dict[int, float], bool]:
    """Sun-heading channel: penalise candidates whose bearing contradicts the sun.

    Returns ``(sun_penalty, sun_active)``.
    """
    result = ctx.result
    result_sun_heading = ctx.result_sun_heading
    candidates = ctx.candidates

    # Sun-heading channel: penalise candidates whose overall bearing
    # contradicts the sun-derived absolute heading. Orientation is
    # exactly what shape RMS cannot fix, so a confident sun estimate
    # kills mirror/rotated walks on near-ties. Confident estimates
    # only; a graceful no-op otherwise.
    sun_penalty = {i: 0.0 for i in range(len(candidates))}
    sun_active = False
    if (result_sun_heading is not None
            and result_sun_heading.get("available")
            and result_sun_heading.get("median_heading") is not None
            and float(result_sun_heading.get("confidence", 0.0))
            >= _SUN_MIN_CONFIDENCE):
        _sun_h = float(result_sun_heading["median_heading"])
        for i, _c in enumerate(candidates):
            _mb = _mean_bearing_deg(np.asarray(_c.aligned_traj_xy,
                                               dtype=np.float64))
            if _mb is None:
                continue
            sun_active = True
            _diff = _heading_diff_deg(_mb, _sun_h)
            result["matches"][i]["sun_heading_diff_deg"] = round(_diff, 1)
            sun_penalty[i] = _sun_bearing_penalty(_mb, _sun_h)
        if sun_active:
            print(f"        sun-heading channel: {_sun_h:.0f} deg "
                  f"(conf {float(result_sun_heading['confidence']):.2f}) "
                  f"fused as orientation penalty")
    return sun_penalty, sun_active


def _consensus(ctx: RunContext, *, fusion_eff: dict, turn_rank: dict, anchor_rank: dict,
               plate_rank: dict, plate_penalty: dict, vpr_penalty: dict,
               sun_penalty: dict, sun_active: bool, have_anchors: bool) -> None:
    """Fuse all channel signals, persist the feature table and reorder the candidates."""
    cfg = ctx.cfg
    fusion_w = ctx.fusion_w
    channel_status = ctx.channel_status
    result = ctx.result
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    plate_center = ctx.plate_center
    road = ctx.road
    aerial_rank = ctx.aerial_rank
    w_aerial = ctx.w_aerial
    tile3d_rank = ctx.tile3d_rank
    w_tile3d = ctx.w_tile3d

    def _sliding_rank(i: int) -> int:
        # Falls back to the shape rank when sliding windows did not run
        # (the default) — which is why shape's effective weight is
        # shape + sliding (see fusion.py).
        return int(result["matches"][i].get("sliding_window_rank", i + 1))

    fusion_features = [
        CandidateFeatures(
            shape_rank=i + 1,
            sliding_rank=_sliding_rank(i),
            turn_rank=int(turn_rank[i]),
            anchor_rank=int(anchor_rank[i]),
            aerial_rank=int(aerial_rank[i]),
            tile3d_rank=int(tile3d_rank[i]),
            plate_penalty=float(plate_penalty[i]),
            vpr_penalty=float(vpr_penalty[i]),
            sun_penalty=float(sun_penalty[i]),
            extra={k: result["matches"][i].get(k) for k in (
                "anchor_distance_m", "traj_coverage", "tile3d_score",
                "tile3d_skyline_err_deg", "sun_heading_diff_deg",
                "turn_match_distance", "sliding_window_score")
                if result["matches"][i].get(k) is not None}
            | {"rms_m": float(ctx.candidates[i].score),
               "bearing_corr": float(ctx.candidates[i].bearing_corr)},
        )
        for i in range(len(ctx.candidates))
    ]

    def _fused(i: int) -> float:
        return fusion_fused_score(fusion_features[i], fusion_eff)

    for i in range(len(ctx.candidates)):
        result["matches"][i]["consensus_score"] = _fused(i)

    consensus_order = fusion_consensus_order(fusion_features, fusion_eff)

    # Persist the per-candidate feature table (matcher order) so any
    # re-weighting / leave-one-channel-out ablation can be scored
    # offline against GT without re-running the pipeline.
    try:
        from ..position import xy_to_latlon as _fxy2ll
        _froutes = []
        for _c in ctx.candidates:
            _xy = np.asarray(_c.aligned_traj_xy, dtype=np.float64)
            _step = max(1, len(_xy) // 100)
            _froutes.append([[round(float(a), 6), round(float(b), 6)]
                             for a, b in _fxy2ll(_xy[::_step], road.crs)])
        write_feature_table(
            cfg.output_dir / "fusion_features.json", fusion_features,
            fusion_eff, routes_latlon=_froutes, weights=fusion_w,
            channel_status=channel_status)
    except Exception as e:
        print(f"        fusion feature table not written ({e})")
    result["fusion"] = {"weights_effective": fusion_eff,
                        "features_file": "fusion_features.json"}

    # Opt-in tile3d tie-break (experiment; --tile3d-tiebreak). The 0.4
    # rank weight is deliberately too small to let the skyline channel
    # override the shape pick, which throws away a CORRECT signal where
    # skylines are distinctive (dense high-rise): on Berlin the skyline
    # channel ranked the route-best candidate #1 but consensus still
    # took shape #1. When the channel is unambiguously discriminative,
    # promote its winner to the front — but only past a triple gate so
    # it stays inert where skylines are non-discriminative (uniform
    # mid-rise), where its top pick is a worse candidate.
    if cfg.tile3d_tiebreak and w_tile3d > 0.0 and len(ctx.candidates) >= 2:
        from ..tile3d_match import tile3d_tiebreak_winner
        _t3_scores = [result["matches"][i].get("tile3d_score")
                      for i in range(len(ctx.candidates))]
        _t3_errs = [result["matches"][i].get("tile3d_skyline_err_deg")
                    for i in range(len(ctx.candidates))]
        _t3_top = tile3d_tiebreak_winner(
            _t3_scores, _t3_errs, consensus_order[0])
        if _t3_top is not None:
            _ranked = sorted((s for s in _t3_scores if s is not None),
                             reverse=True)
            _margin = ((_ranked[0] - _ranked[1]) / _ranked[0]
                       if _ranked[0] else 0.0)
            consensus_order = [_t3_top] + [
                i for i in consensus_order if i != _t3_top]
            print(f"        tile3d tie-break: promoting candidate "
                  f"#{_t3_top + 1} (score {_t3_scores[_t3_top]:.3f}, "
                  f"margin {_margin:.0%}; skyline err "
                  f"{_t3_errs[_t3_top]:.1f} deg; shape #{_t3_top + 1}) "
                  f"over shape pick #{consensus_order[1] + 1}")
        if isinstance(result.get("tile3d"), dict):
            result["tile3d"]["tiebreak_applied"] = _t3_top is not None

    consensus_idx = consensus_order[0]
    print()
    print(f"        Consensus pick: candidate #{consensus_idx + 1} "
          f"(shape #{consensus_idx + 1}, "
          f"sliding #{_sliding_rank(consensus_idx)}, "
          f"turn #{turn_rank[consensus_idx]}, "
          f"anchor #{anchor_rank[consensus_idx]}, "
          f"plate #{plate_rank[consensus_idx]}, "
          f"aerial #{aerial_rank[consensus_idx]}, "
          f"tile3d #{tile3d_rank[consensus_idx]}, "
          f"fused={_fused(consensus_idx):.1f})")

    # Reorder candidates by consensus and update the result JSON.
    # This ensures result["matches"][0] is the consensus-best answer.
    reordered_matches = [result["matches"][i] for i in consensus_order]
    for rank, m in enumerate(reordered_matches):
        m["final_rank"] = rank + 1
    result["matches"] = reordered_matches
    # ranking_mode names the channels that actually contributed, so
    # the JSON says what ran (--no-aerial used to still claim plain
    # 'shape' while skipping every anchor channel).
    active_channels = ["shape"]
    if any("sliding_window_rank" in m for m in result["matches"]):
        active_channels.append("sliding")
    if w_aerial > 0:
        active_channels.append("aerial")
    if w_tile3d > 0:
        active_channels.append("tile3d")
    if have_anchors:
        active_channels.append("anchor")
    if plate_center is not None:
        active_channels.append("plate")
    if vpr_center is not None:
        active_channels.append("vpr-seq" if (cfg.use_vpr_sequence
                                             and vpr_track is not None)
                               else "vpr")
    if sun_active:
        active_channels.append("sun")
    if len(active_channels) > 1:
        ctx.ranking_mode = "consensus(" + "+".join(active_channels) + ")"
    else:
        ctx.ranking_mode = "shape"
    result["ranking_mode"] = ctx.ranking_mode
    print(f"        Final #1 after consensus re-rank ({ctx.ranking_mode}): "
          f"{', '.join(result['matches'][0]['street_names'][:3]) or '(unnamed)'}")
    # Mirror the reorder in candidates for downstream (GT eval).
    ctx.candidates = [ctx.candidates[i] for i in consensus_order]


def fuse_channels(ctx: RunContext) -> None:
    """Weighted rank-fusion consensus: reorders ``ctx.candidates`` and ``result['matches']``."""
    fusion_w = ctx.fusion_w
    result = ctx.result
    anchor_gated = ctx.anchor_gated
    candidates = ctx.candidates
    w_aerial = ctx.w_aerial
    w_tile3d = ctx.w_tile3d

    if not candidates:
        return
    for i in range(len(candidates)):
        result["matches"][i]["shape_rank"] = i + 1

    turn_rank = _turn_channel(ctx)

    anchor_rank, have_anchors = _anchor_channel(ctx)

    # Weighted rank-fusion consensus. Channels are weighted by how
    # well they track ground truth on GT-evaluated runs:
    #   * shape rank  — primary geometric fit (weight 1.0)
    #   * sliding-window rank — a candidate supported across many
    #     trajectory windows is hard to fool (weight 1.0). Falls back
    #     to shape rank when the sliding-window channel is disabled.
    #   * aerial (coverage) rank — useful but weaker; ORB already
    #     dropped from its score, so it is the trajectory-coverage
    #     signal only (weight 0.5).
    # Lower fused score = stronger multi-channel agreement.
    #
    # The turn-sequence rank is computed and stored above for
    # diagnostics but deliberately NOT fused (W_TURN = 0): on the Ulm
    # GT runs it didn't improve ranking and once promoted a 2.2 km-off
    # candidate, because a turn *count/pattern* isn't unique in a
    # dense city grid. Its real value is drift-robust *enumeration*
    # (the pre-filter that decides which walks to score), not
    # re-ranking an already-wrong pool — see turn_matching's
    # docstring. Kept at hand so that move is a one-line re-enable.
    #
    # OCR-anchor weight depends on how the anchor was already used:
    #   * gated enumeration (the normal case): the anchor's spatial
    #     info is *already spent* confining the pool to its vicinity,
    #     so inside that pool let shape lead — a strong anchor re-rank
    #     just chases whichever candidate (incl. a noise anchor like a
    #     shop sign) sits closest, demoting the true shape match.
    #     A small weight (0.25) only breaks near-ties. (GT-confirmed:
    #     at 7 min the gated shape #1 IS the true route.) NB raising this
    #     does NOT fix the London failure: there the 'Holborn' anchor is a
    #     DISTANT direction sign ~1.5 km from the true route, so trusting it
    #     more only chases the wrong place — an anchor-QUALITY (here-vs-
    #     direction) problem a VLM must solve, not a weight.
    #   * un-gated (anchors present but enumeration not restricted):
    #     strong weight (2.0) to pull toward the anchored area.
    #   * no anchors: 0 (clean no-op).
    # Weights live in fusion.FusionWeights (defaults = the published
    # values); effective_weights() applies the gating rule above.
    fusion_eff = fusion_effective_weights(
        fusion_w, have_anchors=have_anchors, anchor_gated=anchor_gated,
        aerial_active=w_aerial > 0, tile3d_weight=w_tile3d)
    W_ANCHOR = fusion_eff["anchor"]

    plate_rank, plate_penalty = _plate_channel(ctx)

    vpr_penalty = _vpr_channel(ctx)

    _vpr_chord_diagnostic(ctx)

    sun_penalty, sun_active = _sun_channel(ctx)

    _consensus(ctx, fusion_eff=fusion_eff, turn_rank=turn_rank,
               anchor_rank=anchor_rank, plate_rank=plate_rank,
               plate_penalty=plate_penalty, vpr_penalty=vpr_penalty,
               sun_penalty=sun_penalty, sun_active=sun_active,
               have_anchors=have_anchors)
