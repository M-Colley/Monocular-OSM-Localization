"""Stage 5b-9: optional appearance / geometry channels over the candidates.

Each channel scores or renders the ``ctx.candidates`` and records its outcome in
``ctx.result``; the rank maps the consensus needs (aerial, tile3d) are left on the
context. A channel that fails records itself via ``_record_channel``.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..aerial_match import match_splat_against_candidates
from ..ipm import render_ipm_canvas
from ..splat import (
    build_splat_points,
    render_topdown_splat,
    render_topdown_to_file,
    save_interactive_html,
    save_ply,
)
from ..trajectory_matching import (
    candidate_geographic_summary,
    score_candidates_with_sliding_windows,
)
from ..visual_odometry import default_intrinsics
from .context import _record_channel, RunContext


class _Tile3DInactive(Exception):
    """Control-flow sentinel: the tile3d channel declined to run
    (no provider / empty mesh / coverage gap) - not an error."""


def sliding_window_channel(ctx: RunContext) -> None:
    """Optional sliding-window support: a candidate supported across many trajectory windows is hard to fool."""
    cfg = ctx.cfg
    result = ctx.result
    estimated_length_m = ctx.estimated_length_m
    road = ctx.road
    match_xz = ctx.match_xz
    candidates = ctx.candidates

    if cfg.enable_sliding_window and candidates:
        print(
            f"[5b] Sliding-window matching "
            f"(window={cfg.sliding_window_size}, step={cfg.sliding_window_step})"
        )
        sliding = score_candidates_with_sliding_windows(
            match_xz,
            road,
            candidates,
            window_size=cfg.sliding_window_size,
            step=cfg.sliding_window_step,
            window_top_k=max(cfg.top_k, 3),
            estimated_length_m=estimated_length_m,
        )
        if sliding:
            rank_order = sorted(
                range(len(sliding)),
                key=lambda i: (-sliding[i].sliding_score, sliding[i].mean_rank, i),
            )
            rank_map = {idx: rank + 1 for rank, idx in enumerate(rank_order)}
            result["sliding_window"] = {
                "window_size": cfg.sliding_window_size,
                "step": cfg.sliding_window_step,
                "n_windows": sliding[0].n_windows,
            }
            print("      -> sliding-window support:")
            for i, sw in enumerate(sliding):
                result["matches"][i]["sliding_window_support_count"] = sw.support_count
                result["matches"][i]["sliding_window_support_ratio"] = sw.support_ratio
                result["matches"][i]["sliding_window_mean_rank"] = sw.mean_rank
                result["matches"][i]["sliding_window_score"] = sw.sliding_score
                result["matches"][i]["sliding_window_rank"] = rank_map[i]
                print(
                    f"        #{i+1}  support={sw.support_count}/{sw.n_windows}  "
                    f"mean_rank={sw.mean_rank:.2f}  score={sw.sliding_score:.3f}"
                )


def splat_channel(ctx: RunContext) -> None:
    """Sparse SfM splat point cloud (for the aerial channel and visualisation) -> ``ctx.splat_img_rgb``."""
    cfg = ctx.cfg
    frames = ctx.frames
    traj = ctx.traj
    candidates = ctx.candidates

    # 6. Splat reconstruction (sparse SfM point cloud) — for the aerial
    # matching channel and for visualization.
    ctx.splat_img_rgb = None
    if cfg.enable_splat and candidates:
        h_img, w_img = frames.frames[0].shape[:2]
        K = default_intrinsics(w_img, h_img)

        print(f"[6] Building sparse splat point cloud "
              f"(<= {cfg.splat_max_pairs} frame pairs)")
        splat_pts, splat_cols = build_splat_points(
            frames.frames, traj, K, max_pairs=cfg.splat_max_pairs
        )
        print(f"      -> {len(splat_pts)} 3-D points")

        if len(splat_pts) > 0:
            ply_path = cfg.output_dir / "splat.ply"
            save_ply(splat_pts, splat_cols, ply_path)
            print(f"      -> wrote {ply_path} (open in MeshLab / CloudCompare)")

            html_path = cfg.output_dir / "splat.html"
            save_interactive_html(splat_pts, splat_cols, html_path)
            print(f"      -> wrote {html_path} (open in browser to rotate/zoom)")

            splat_img_path = cfg.output_dir / "splat_topdown.png"
            render_topdown_to_file(splat_pts, splat_cols, splat_img_path)
            print(f"      -> wrote {splat_img_path}")
            ctx.splat_img_rgb = render_topdown_splat(splat_pts, splat_cols)

            if cfg.enable_full_splat:
                from ..full_splat import render_full_splat_to_file
                hq_path = cfg.output_dir / "splat_topdown_hq.png"
                ctx.splat_img_rgb = render_full_splat_to_file(
                    splat_pts, splat_cols, hq_path,
                    scale=cfg.full_splat_scale,
                    opacity=cfg.full_splat_opacity,
                    progress=True,
                )
                print(f"      -> wrote {hq_path} (anisotropic Gaussian render)")
        else:
            print("      -> no triangulated points")


def ipm_channel(ctx: RunContext) -> None:
    """IPM road-plane BEV stitch -> ``ctx.ipm_canvas``.

    Produced *before* aerial matching so it can be the aerial input: IPM gives a
    real road-texture top-down view that shares features with the OSM raster.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    result = ctx.result
    frames = ctx.frames
    traj = ctx.traj

    # 7. IPM road-plane BEV stitch — produced *before* aerial matching so
    # we can use it as the aerial-matching input. IPM gives a real
    # road-texture top-down view that shares features (intersection
    # corners, road edges) with the OSM raster patches; the sparse splat
    # top-down has very few of those, which is why aerial matching on
    # the splat alone gave low inlier counts. With IPM available we
    # prefer it over the splat as the aerial input.
    ctx.ipm_canvas = None
    if cfg.enable_ipm:
        print(f"[7] Inverse perspective mapping (camera_height={cfg.ipm_camera_height_m}m, "
              f"pitch={cfg.ipm_pitch_deg} deg)")
        try:
            h_img, w_img = frames.frames[0].shape[:2]
            K = default_intrinsics(w_img, h_img)
            # For longer trajectories use a denser keyframe stride so the
            # stitched canvas covers the full route without large gaps.
            n_frames = len(frames.frames)
            ipm_stride = max(4, n_frames // 200)  # ~200 tiles regardless of length
            ctx.ipm_canvas = render_ipm_canvas(
                list(frames.frames), traj.xz, K,
                keyframe_stride=ipm_stride,
                camera_height_m=cfg.ipm_camera_height_m,
                pitch_deg=cfg.ipm_pitch_deg,
            )
            ipm_path = cfg.output_dir / "ipm_bev.png"
            cv2.imwrite(str(ipm_path), ctx.ipm_canvas)
            print(f"      -> wrote {ipm_path} ({ctx.ipm_canvas.shape[1]}x{ctx.ipm_canvas.shape[0]})")
            result["ipm"] = str(ipm_path.relative_to(cfg.output_dir))
        except Exception as e:
            print(f"      -> IPM failed: {e}")
            _record_channel(channel_status, 'ipm', "failed", error=e)
            result["ipm_error"] = str(e)
            ctx.ipm_canvas = None


def aerial_channel(ctx: RunContext) -> None:
    """Aerial matching channel (traj-IoU + optional ORB) -> ``ctx.aerial_rank`` / ``ctx.w_aerial``.

    ONE optional channel: the fusion itself runs whenever candidates exist, so
    ``--no-aerial`` only zeroes this channel's weight.
    """
    cfg = ctx.cfg
    fusion_w = ctx.fusion_w
    result = ctx.result
    road = ctx.road
    candidates = ctx.candidates
    splat_img_rgb = ctx.splat_img_rgb
    ipm_canvas = ctx.ipm_canvas

    # 8. Aerial matching channel + consensus fusion.
    # The aerial ORB/IoU scoring is ONE optional channel; the fusion itself
    # (turn + OCR-anchor + plate + VPR + sun penalties, consensus reorder)
    # runs whenever candidates exist. It used to sit entirely inside the
    # aerial gate, so --no-aerial silently disabled every proven
    # disambiguator (plate guard, VPR re-rank, OCR-anchor rank).
    ctx.ranking_mode = "shape"
    if not candidates:
        return
    if cfg.enable_aerial_match:
        # Primary signal: trajectory-raster IoU (uses aligned_traj_xy
        # from each MatchCandidate, no top-down image required).
        # Supplemental signal: ORB on top-down image vs OSM patch (weak
        # when the image is a photographic IPM vs a schematic OSM
        # render; retained for completeness and inter-method comparison).
        if ipm_canvas is not None:
            aerial_input = cv2.cvtColor(ipm_canvas, cv2.COLOR_BGR2RGB)
            aerial_source = "ipm_bev"
        elif splat_img_rgb is not None:
            aerial_input = splat_img_rgb
            aerial_source = "splat_topdown"
        else:
            aerial_input = None
            aerial_source = "traj_iou_only"

        print(f"[8] Aerial matching: traj-IoU"
              f"{' (input: ' + aerial_source + ')' if aerial_input is not None else ''} "
              f"({len(candidates)} candidates)")
        aerial_results = match_splat_against_candidates(
            aerial_input, road, candidates,
            output_dir=cfg.output_dir / "aerial",
        )
        result["aerial_input"] = aerial_source

        # Aerial rank by combined aerial_score (higher = better → sort descending).
        order_by_aerial = sorted(
            range(len(aerial_results)),
            key=lambda i: -aerial_results[i].aerial_score,
        )
        ctx.aerial_rank = {idx: r + 1 for r, idx in enumerate(order_by_aerial)}

        print()
        print("        ===== Method comparison (top-{:d}) =====".format(len(candidates)))
        print("        shape_rank  shape_RMS  shape_corr  | aerial_rank  coverage  traj_IoU  ORB_inliers | streets")
        print("        " + "-" * 120)
        for i, ar in enumerate(aerial_results):
            c = candidates[i]
            names = ", ".join(
                candidate_geographic_summary(c, road.graph)["street_names"][:2]
            ) or "(unnamed)"
            print(f"           #{i+1:<2}      {c.score:7.1f} m   {c.bearing_corr:+.3f}     |"
                  f"     #{ctx.aerial_rank[i]:<2}     {ar.traj_coverage:.3f}    {ar.traj_iou:.3f}       {ar.n_inliers:3d}        | {names}")

        for i, ar in enumerate(aerial_results):
            m = result["matches"][i]
            m["traj_iou"] = ar.traj_iou
            m["traj_coverage"] = ar.traj_coverage
            m["aerial_score"] = ar.aerial_score
            m["aerial_orb_matches"] = ar.n_orb_matches
            m["aerial_inliers"] = ar.n_inliers
            m["aerial_inlier_ratio"] = ar.inlier_ratio
            m["aerial_rank"] = ctx.aerial_rank[i]
        ctx.w_aerial = fusion_w.aerial
    else:
        # Neutral channel: uniform rank, zero weight — the same no-op
        # pattern the anchor channel uses when it has no data.
        ctx.aerial_rank = {i: i + 1 for i in range(len(candidates))}
        ctx.w_aerial = 0.0


def tile3d_channel(ctx: RunContext) -> None:
    """Opt-in 3D-tile skyline channel -> ``ctx.tile3d_rank`` / ``ctx.w_tile3d``.

    Render the open-data LoD2 city model at sampled poses along each candidate
    and score rendered-vs-observed skyline agreement. Follows the aerial no-op
    pattern: uniform rank + zero weight unless it produced a usable comparison.
    """
    cfg = ctx.cfg
    fusion_w = ctx.fusion_w
    channel_status = ctx.channel_status
    result = ctx.result
    frames = ctx.frames
    road = ctx.road
    match_xz = ctx.match_xz
    match_ts = ctx.match_ts
    candidates = ctx.candidates

    if not candidates:
        return
    # 3D-tile skyline channel (opt-in): render the open-data LoD2
    # city model at sampled poses along each candidate and score
    # rendered-vs-observed skyline agreement. Follows the aerial
    # no-op pattern: uniform rank + zero weight unless it produced
    # a usable comparison.
    ctx.tile3d_rank = {i: i + 1 for i in range(len(candidates))}
    ctx.w_tile3d = 0.0
    ctx.t3_mesh_ref = ctx.t3_samples_ref = ctx.t3_K_ref = ctx.t3_skyline_ref = None
    if cfg.use_tile3d:
        from ..citygml_lod2 import fetch_lod2_mesh, provider_for_latlon
        from ..position import xy_to_latlon
        from ..tile3d_match import (
            adaptive_tile3d_weight,
            arc_fractions_at_times,
            match_tiles3d_against_candidates,
            tile3d_tiebreak_winner,
            walk_coverage_fractions,
        )
        try:
            all_xy = np.vstack([c.walk_xy for c in candidates])
            center_xy = all_xy.mean(axis=0)
            # margin matches the renderer's 500 m visibility range so
            # edge poses don't look into un-fetched tiles
            t3_radius = float(
                np.linalg.norm(all_xy - center_xy[None, :], axis=1).max()
            ) + 500.0
            center_ll = xy_to_latlon(center_xy[None, :], road.crs)[0]
            # Source: an explicit CityGML provider / "osm"; "auto" uses the
            # official open LoD2 where it exists (Berlin/BW/NRW/Bavaria) and
            # falls back to the WORLDWIDE OSM LoD1 extrusion everywhere else.
            _src = cfg.tile3d_source
            _use_osm = _src == "osm" or (
                _src == "auto"
                and provider_for_latlon(float(center_ll[0]),
                                        float(center_ll[1])) is None)
            print(f"[8b] 3D-tile skyline channel: disc "
                  f"r={t3_radius:.0f} m @ {center_ll[0]:.5f},"
                  f"{center_ll[1]:.5f} "
                  f"(source={'osm' if _use_osm else _src})")
            if _use_osm:
                from ..osm_buildings3d import fetch_osm_building_mesh
                t3_mesh = fetch_osm_building_mesh(
                    float(center_ll[0]), float(center_ll[1]), t3_radius,
                    dst_crs=road.crs)
            else:
                t3_mesh = fetch_lod2_mesh(
                    float(center_ll[0]), float(center_ll[1]), t3_radius,
                    dst_crs=road.crs, provider=_src,
                    max_tiles=cfg.tile3d_max_tiles,
                )
            if t3_mesh is None or not len(t3_mesh.triangles):
                if t3_mesh is not None:
                    print("      -> LoD2 mesh empty here; "
                          "tile3d channel inactive")
                raise _Tile3DInactive
            # A capped/holey fetch must not turn missing coverage into
            # a fake skyline-disagreement penalty: if a candidate's walk
            # leaves the fetched buildings, deactivate. A coarse dilated
            # occupancy grid (walk_coverage_fractions) catches INTERIOR
            # holes (a 404 or tile-capped gap in the middle of the disc)
            # that a plain bounding box misses, while a normal street
            # between building cells still counts as covered.
            covered = walk_coverage_fractions(
                t3_mesh.building_ground, [c.walk_xy for c in candidates])
            if min(covered) < 0.9:
                n_out = sum(1 for f_ in covered if f_ < 0.9)
                print(f"      -> {n_out}/{len(candidates)} candidates "
                      f"extend beyond the fetched LoD2 coverage "
                      f"(tile cap {cfg.tile3d_max_tiles}); channel "
                      f"inactive — raise --tile3d-max-tiles to score "
                      f"them fairly")
                raise _Tile3DInactive
            h_img, w_img = frames.frames[0].shape[:2]
            K_t3 = default_intrinsics(w_img, h_img)
            # Optional learned sky segmentation (robust to night/glare)
            # in place of the Lab-smoothness skyline heuristic.
            t3_skyline_fn = None
            if cfg.tile3d_skyseg:
                try:
                    from ..sky_segmentation import (SkySegmenter,
                                                   skyline_from_frame_seg)
                    _seg = SkySegmenter()
                    t3_skyline_fn = (lambda f, wh:
                                     skyline_from_frame_seg(f, wh, _seg))
                    print("      -> tile3d: using SegFormer sky "
                          f"segmentation ({_seg.device})")
                except Exception as e:
                    print(f"      -> tile3d: sky-seg unavailable ({e}); "
                          f"using the Lab heuristic")
            n_samp = max(1, cfg.tile3d_samples)
            t3_times = np.linspace(
                float(match_ts[0]), float(match_ts[-1]), n_samp + 2,
            )[1:-1]
            t3_fracs = arc_fractions_at_times(match_xz, match_ts, t3_times)
            fts = np.asarray(frames.timestamps)
            t3_samples = [
                (float(f_), frames.frames[int(np.argmin(np.abs(fts - t_)))])
                for f_, t_ in zip(t3_fracs, t3_times)
            ]
            t3_results = match_tiles3d_against_candidates(
                t3_samples, candidates, t3_mesh, K_t3, (w_img, h_img),
                output_dir=cfg.output_dir / "tile3d",
                skyline_fn=t3_skyline_fn,
            )
            # Activate only on discriminative signal: all-zero scores
            # mean the model never matched anything anywhere — an
            # identity rank at weight 0.4 would silently re-weight
            # the shape channel.
            if max(r.tile3d_score for r in t3_results) > 0.0:
                t3_order = sorted(
                    range(len(t3_results)),
                    key=lambda i: (-t3_results[i].tile3d_score, i),
                )
                ctx.tile3d_rank = {idx: r + 1
                                   for r, idx in enumerate(t3_order)}
                for i, tr in enumerate(t3_results):
                    m = result["matches"][i]
                    m["tile3d_score"] = round(tr.tile3d_score, 4)
                    m["tile3d_skyline_err_deg"] = (
                        round(tr.skyline_err_deg, 2)
                        if tr.skyline_err_deg is not None else None
                    )
                    m["tile3d_pitch_offset_deg"] = (
                        round(tr.pitch_offset_deg, 2)
                        if tr.pitch_offset_deg is not None else None
                    )
                    m["tile3d_coverage"] = round(tr.coverage, 3)
                    m["tile3d_informative_samples"] = tr.n_informative
                    m["tile3d_rank"] = ctx.tile3d_rank[i]
                result["tile3d"] = {
                    "provider": t3_mesh.provider,
                    "n_buildings": t3_mesh.n_buildings,
                    "n_triangles": int(len(t3_mesh.triangles)),
                    "n_samples": len(t3_samples),
                }
                # Uncertainty-aware fusion (opt-in): weight the channel
                # by how discriminative its scores are, else fixed 0.4.
                if cfg.tile3d_adaptive:
                    ctx.w_tile3d = adaptive_tile3d_weight(
                        [r.tile3d_score for r in t3_results])
                    result["tile3d"]["adaptive_weight"] = round(ctx.w_tile3d, 3)
                else:
                    ctx.w_tile3d = fusion_w.tile3d
                # Stash mesh + samples so an optional post-selection
                # placement refinement (--tile3d-refine) can reuse them.
                ctx.t3_mesh_ref = t3_mesh
                ctx.t3_samples_ref = t3_samples
                ctx.t3_K_ref = (K_t3, (w_img, h_img))
                ctx.t3_skyline_ref = t3_skyline_fn
                print(f"      -> tile3d best = candidate "
                      f"#{t3_order[0] + 1} (score "
                      f"{t3_results[t3_order[0]].tile3d_score:.3f}, "
                      f"{t3_results[t3_order[0]].n_informative} "
                      f"informative samples)")
            else:
                print("      -> tile3d: no candidate matched any "
                      "skyline; channel inactive")
        except _Tile3DInactive:
            pass
        except Exception as e:  # fetch/render failure: channel no-ops
            print(f"      -> tile3d channel failed ({e}); inactive")
            _record_channel(channel_status, 'tile3d', "failed", error=e)
