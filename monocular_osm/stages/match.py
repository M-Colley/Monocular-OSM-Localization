"""Stage 5: match the trajectory against the road graph."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: do not try to open a window
import matplotlib.pyplot as plt
import numpy as np

from ..osm_data import RoadGraph
from ..trajectory_matching import (
    candidate_geographic_summary,
    match_trajectory,
    MatchCandidate,
)
from .context import _record_channel, RunContext
from .vpr_track import _vpr_sequence_median_m, _vpr_track_geometry


def _plot_match(
    road: RoadGraph,
    candidates: list[MatchCandidate],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))

    # Background: every road in the city graph (light grey).
    for poly in road.polylines:
        ax.plot(poly[:, 0], poly[:, 1], color="#cccccc", linewidth=0.4, zorder=1)

    # Top candidates ranked by score, brighter = better.
    cmap = plt.get_cmap("plasma")
    for i, cand in enumerate(candidates):
        color = cmap(0.15 + 0.7 * (1 - i / max(1, len(candidates) - 1)))
        ax.plot(
            cand.walk_xy[:, 0],
            cand.walk_xy[:, 1],
            color=color,
            linewidth=2.5 if i == 0 else 1.5,
            zorder=10 - i,
            label=f"#{i+1} score={cand.score:.1f} m",
        )

    if candidates:
        best = candidates[0]
        ax.plot(
            best.aligned_traj_xy[:, 0],
            best.aligned_traj_xy[:, 1],
            "--",
            color="black",
            linewidth=1.5,
            label="VO trajectory aligned to #1",
            zorder=20,
        )
        # Tighten view around best match.
        pad = 600
        xs, ys = best.walk_xy[:, 0], best.walk_xy[:, 1]
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

    ax.set_aspect("equal")
    ax.set_title(f"Top-{len(candidates)} matches in {road.crs}")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# Size of the geometric candidate pool kept for the calibrated
# multi-hypothesis output. The matcher already scores ~500 walks in
# stage 2, so returning 50 instead of top_k is free; we collapse them
# into distinct location hypotheses (monocular_osm/hypotheses.py). The heavy
# channels (BevSplat, sliding window) and the headline pick still run on
# the top_k slice — only the hypotheses shortlist uses the wider pool.
_HYP_POOL = 50


def _vpr_alt_scale(match_xz, vpr_track, vpr_center, road, locked_scale,
                   estimated_length_m, scale_lock):
    """Alternate matching scale pinned to the VPR track's start->end extent.

    Returns ``(locked_scale, estimated_length_m, factor)`` for a second match
    pass, or ``None`` when unavailable / not meaningfully different from the
    base scale / outside a sane band. The caller matches at both scales and
    keeps whichever candidates better explain the full per-frame VPR track —
    so a noisy single endpoint can't force a bad scale (it only proposes one).
    """
    if vpr_center is None or vpr_track is None:
        return None
    try:
        from ..visual_odometry import trajectory_arc_length as _tal
        geo = _vpr_track_geometry(vpr_track, road.crs)
        if geo is None:
            return None
        vpr_chord = float(np.linalg.norm(geo[4] - geo[3]))
        mxz = np.asarray(match_xz, dtype=np.float64)
        vo_chord = float(np.linalg.norm(mxz[-1] - mxz[0]))
        arc = float(_tal(mxz)[-1])
        if vpr_chord <= 200.0 or vo_chord <= 1e-6 or arc <= 1e-6:
            return None
        tgt = vpr_chord / vo_chord                      # m per VO unit
        cur = locked_scale if locked_scale is not None else estimated_length_m / arc
        fac = tgt / cur if cur > 1e-9 else 1.0
        if 0.33 <= fac <= 3.0 and abs(fac - 1.0) > 0.15:
            return (tgt if scale_lock else None, float(tgt * arc), float(fac))
    except Exception:
        return None
    return None


def lock_alignment_scale(ctx: RunContext) -> None:
    """Scale lock: pin the alignment scale to the metric extent -> ``ctx.locked_scale``.

    So the route can't compress (the Ulm eastern-tail fix). scale = metres per VO unit.
    """
    cfg = ctx.cfg
    estimated_length_m = ctx.estimated_length_m
    match_xz = ctx.match_xz

    # 5. Match. When confident OCR anchors exist, gate enumeration to the
    # anchor vicinity (a trustworthy absolute prior beats the drift-prone
    # global shape for *where* to look) and let shape rank within it. Fall
    # back to the full-city scan if the gate yields nothing.
    # Scale lock: pin alignment scale to the metric extent so the route
    # can't compress (the eastern-tail fix). scale = metres per VO unit.
    ctx.locked_scale = None
    if cfg.scale_lock:
        from ..visual_odometry import trajectory_arc_length
        vo_arclen = float(trajectory_arc_length(match_xz)[-1])
        if vo_arclen > 1e-6:
            ctx.locked_scale = estimated_length_m / vo_arclen
            print(f"      -> scale-locked alignment: {ctx.locked_scale:.4f} m/VO-unit "
                  f"(extent {estimated_length_m:.0f} m / {vo_arclen:.1f} VO units)")


def vggt_gate(ctx: RunContext) -> None:
    """Optional VGGT gating: a drift-free feed-forward trajectory selects the AREA -> ``ctx.vggt_seed``.

    Its bearing signature matches the true street (breaking the parallel-street
    ambiguity that drifted VO can't); the loop-closed VO geometry then ranks within it.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames
    estimated_length_m = ctx.estimated_length_m
    road = ctx.road

    # VGGT gating: a drift-free feed-forward trajectory selects the AREA
    # (its bearing signature matches the true street, breaking the
    # parallel-street ambiguity that drifted VO can't), then the precise
    # loop-closed VO geometry ranks within it.
    ctx.vggt_seed = []
    ctx.result_vggt = None
    if cfg.use_vggt_gating:
        print("[5a] VGGT trajectory gating (drift-free area selection)")
        try:
            from ..vggt_trajectory import vggt_camera_trajectory, vggt_seed_nodes
            vxy = vggt_camera_trajectory(frames.frames, n_keyframes=cfg.vggt_keyframes)
            if vxy is not None:
                ctx.vggt_seed = vggt_seed_nodes(
                    vxy, road, estimated_length_m=estimated_length_m)
                ctx.result_vggt = {"poses": len(vxy), "seed_nodes": len(ctx.vggt_seed)}
                print(f"      -> VGGT: {len(vxy)} poses -> {len(ctx.vggt_seed)} seed start-node(s)")
            else:
                print("      -> VGGT unavailable (no package/GPU/weights); skipping")
        except Exception as e:
            print(f"      -> VGGT gating failed ({e}); skipping")
            _record_channel(channel_status, 'vggt', "failed", error=e)


def match_candidates(ctx: RunContext) -> None:
    """Enumerate and score candidate routes -> ``ctx.candidates`` / ``ctx.geom_pool``.

    When confident OCR anchors (or a VGGT area) exist, enumeration is gated to
    their vicinity and shape ranks within it; falls back to the full-city scan if
    the gate yields nothing. The optional two-pass VPR scale arbitration may
    replace the candidate set and the locked scale.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    frames = ctx.frames
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    road = ctx.road
    match_xz = ctx.match_xz
    ocr_anchors = ctx.ocr_anchors
    anchor_seed = ctx.anchor_seed
    vggt_seed = ctx.vggt_seed

    seed = list(dict.fromkeys((anchor_seed or []) + vggt_seed))
    gated = bool((cfg.enable_ocr_anchor and anchor_seed) or vggt_seed)
    ctx.anchor_gated = gated  # downstream reporting reuses this name
    if gated:
        why = []
        if cfg.enable_ocr_anchor and anchor_seed:
            why.append(f"{len(ocr_anchors)} OCR anchor(s)")
        if vggt_seed:
            why.append("VGGT")
        print(f"[5/5] Matching (gated to {len(seed)} seed nodes: {', '.join(why)})")
    else:
        print(f"[5/5] Matching trajectory against {road.graph.number_of_nodes()} candidate starts")
    _pool_k = max(cfg.top_k, _HYP_POOL)
    ctx.vpr_scale_applied = None

    def _run_match(loc_scale, est_len):
        c = match_trajectory(
            match_xz, road, final_top_k=_pool_k, sample_every=cfg.sample_every,
            estimated_length_m=est_len, extra_start_nodes=seed or None,
            restrict_to_start_nodes=gated, locked_scale=loc_scale,
            walks_per_node=cfg.walks_per_node)
        if gated and not c:
            print("      -> gated match found nothing; falling back to full scan")
            c = match_trajectory(
                match_xz, road, final_top_k=_pool_k, sample_every=cfg.sample_every,
                estimated_length_m=est_len, locked_scale=loc_scale,
                walks_per_node=cfg.walks_per_node)
        return c

    ctx.candidates = _run_match(ctx.locked_scale, ctx.estimated_length_m)

    # Two-pass VPR scale arbitration: every candidate inherits ONE scale, so a
    # wrong VO scale makes them all the wrong length (Ulm all 534 m vs a VPR
    # extent of 1134 m; London 1453 vs 1022). The VPR track's start->end extent
    # gives an ALTERNATE scale; match at it too and keep whichever set's
    # candidates better explain the FULL per-frame VPR track (median over ~80
    # frames -> robust to the single confident-but-wrong endpoint that sank the
    # endpoint-chord-only version on comma's self-similar highway).
    _vpr_alt = (_vpr_alt_scale(match_xz, vpr_track, vpr_center, road,
                               ctx.locked_scale, ctx.estimated_length_m, cfg.scale_lock)
                if cfg.vpr_two_pass_scale else None)
    if _vpr_alt is not None and ctx.candidates:
        _alt = _run_match(_vpr_alt[0], _vpr_alt[1])
        if _alt:
            try:
                from ..position import latlon_to_xy as _t2xy
                _tk_xy = _t2xy(np.asarray(vpr_track[1], dtype=np.float64), road.crs)

                def _setscore(cs):
                    ds = [_vpr_sequence_median_m(
                        c.aligned_traj_xy, vpr_track[0], _tk_xy, vpr_track[2],
                        len(frames.frames)) for c in cs[:5]]
                    ds = [d for d in ds if d is not None]
                    return min(ds) if ds else float("inf")

                _bs, _as = _setscore(ctx.candidates), _setscore(_alt)
                if _as < 0.9 * _bs:
                    ctx.candidates = _alt
                    ctx.locked_scale, ctx.estimated_length_m = _vpr_alt[0], _vpr_alt[1]
                    ctx.vpr_scale_applied = round(_vpr_alt[2], 3)
                    print(f"      -> VPR two-pass: corrected scale x{_vpr_alt[2]:.2f} "
                          f"wins (track fit {_as:.0f} m < base {_bs:.0f} m)")
                else:
                    print(f"      -> VPR two-pass: base scale kept (alt track "
                          f"fit {_as:.0f} m not <90% of base {_bs:.0f} m)")
            except Exception as e:
                print(f"      -> VPR two-pass compare failed ({e}); keeping base")
                _record_channel(channel_status, 'vpr_two_pass', "failed", error=e)
    # Keep the wider geometric pool for the calibrated multi-hypothesis
    # output, but run the rest of the pipeline (heavy channels, headline
    # pick, GT eval) on the top_k slice exactly as before.
    ctx.geom_pool = list(ctx.candidates)
    ctx.candidates = ctx.candidates[:cfg.top_k]


def summarise_matches(ctx: RunContext) -> None:
    """Create ``ctx.result`` with the per-candidate geographic summary."""
    cfg = ctx.cfg
    frames = ctx.frames
    estimated_length_m = ctx.estimated_length_m
    n_valid = ctx.n_valid
    road = ctx.road
    result_traj_source = ctx.result_traj_source
    ocr_anchors = ctx.ocr_anchors
    street_anchors = ctx.street_anchors
    ocr_anchor_error = ctx.ocr_anchor_error
    vpr_scale_applied = ctx.vpr_scale_applied
    candidates = ctx.candidates

    if not candidates:
        print("      -> no matches!")
        ctx.result = {"city": cfg.city, "matches": []}
    else:
        print(f"      -> top {len(candidates)} matches:")
        for i, c in enumerate(candidates):
            names = ", ".join(
                candidate_geographic_summary(c, road.graph)["street_names"][:3]
            ) or "(unnamed)"
            print(f"        #{i+1}  RMS={c.score:7.1f} m  corr={c.bearing_corr:+.3f}  {names}")
        _plot_match(road, candidates, cfg.output_dir / "match.png")

        ctx.result = {
            "city": cfg.city,
            "url": cfg.url,
            "n_frames": len(frames.frames),
            "n_valid_poses": n_valid,
            "estimated_length_m": estimated_length_m,
            "trajectory_source": result_traj_source,
            "matches": [
                candidate_geographic_summary(c, road.graph) for c in candidates
            ],
        }
        if vpr_scale_applied is not None:
            ctx.result["vpr_scale_correction"] = vpr_scale_applied
        if cfg.enable_ocr_anchor:
            from ..text_anchor import anchors_to_json, street_anchors_to_json
            ctx.result["ocr_anchors"] = anchors_to_json(ocr_anchors)
            ctx.result["ocr_street_anchors"] = street_anchors_to_json(street_anchors)
            if ocr_anchor_error:
                ctx.result["ocr_anchor_error"] = ocr_anchor_error
