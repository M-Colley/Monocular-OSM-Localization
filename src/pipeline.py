"""End-to-end pipeline: YouTube URL -> matched street(s) in target city.

The CLI in `main.py` is a thin wrapper around `run_pipeline` here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# When stdout is redirected to a file Python uses block-buffering, which
# causes progress lines to be held in a buffer for minutes. Force
# line-buffering so every print() call is visible in the log immediately.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: don't try to open a window
import matplotlib.pyplot as plt
import numpy as np

from .aerial_match import match_splat_against_candidates
from .bev_splat_match import (
    BevSplatConfig,
    score_candidates_with_bevsplat,
)
from .download import download_video
from .embedding_retrieval import score_candidates_by_embeddings
from .evaluator import best_rank_for_gt, evaluate_candidates
from .frame_extraction import extract_frames
from .ipm import render_ipm_canvas
from .osm_data import RoadGraph, fetch_city_graph
from .splat import (
    build_splat_points,
    render_topdown_splat,
    render_topdown_to_file,
    save_interactive_html,
    save_ply,
)
from .trajectory_matching import (
    MatchCandidate,
    candidate_geographic_summary,
    match_trajectory,
    score_candidates_with_sliding_windows,
)
from .visual_odometry import Trajectory, default_intrinsics, estimate_trajectory


@dataclass
class PipelineConfig:
    url: str
    city: str
    data_dir: Path
    output_dir: Path
    max_frames: int = 1500
    frame_stride: int = 6
    vo_start_sec: float = 0.0
    vo_end_sec: float | None = 300.0
    top_k: int = 5
    estimated_length_m: float = 4000.0
    skip_download: bool = False
    sample_every: int = 1
    enable_splat: bool = True
    splat_max_pairs: int = 80
    enable_aerial_match: bool = True
    enable_da3: bool = False
    use_da3_trajectory: bool = False
    da3_keyframes: int = 32
    enable_full_splat: bool = False
    full_splat_scale: float = 1.4
    full_splat_opacity: float = 0.55
    enable_train_3dgs: bool = False
    train_3dgs_iters: int = 2000
    enable_ipm: bool = False
    ipm_camera_height_m: float = 1.4
    ipm_pitch_deg: float = 6.0
    enable_sliding_window: bool = False
    sliding_window_size: int = 64
    sliding_window_step: int = 32
    embedding_sources: tuple[str, ...] = ()
    embedding_model: str = "resnet18"
    geotessera_year: int = 2024
    ground_truth_streets: tuple[str, ...] = ()
    enable_bev_splat: bool = False
    bev_splat_weights: Path | None = None
    bev_splat_repo_path: Path | None = None
    bev_splat_model_module: str = "models.models_kitti_nips"
    bev_splat_source: str = "esri"
    bev_splat_tile_size: int = 512
    bev_splat_half_extent_m: float = 60.0
    vo_workers: int | None = None  # None → auto (min(cpu_count, 12)); 1 → sequential


def _plot_trajectory(traj: Trajectory, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(traj.xz[:, 0], traj.xz[:, 1], "-", color="C0", linewidth=1.4)
    ax.scatter(traj.xz[0, 0], traj.xz[0, 1], color="green", s=60, label="start", zorder=5)
    ax.scatter(traj.xz[-1, 0], traj.xz[-1, 1], color="red", s=60, label="end", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Recovered top-down trajectory (VO, scale-free)")
    ax.set_xlabel("x (arbitrary units)")
    ax.set_ylabel("y (arbitrary units)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_xy(xy: np.ndarray, out_path: Path, title: str) -> None:
    """Minimal top-down polyline plot (used for the DA3 trajectory)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xy[:, 0], xy[:, 1], "-", color="C2", linewidth=1.4)
    ax.scatter(xy[0, 0], xy[0, 1], color="green", s=60, label="start", zorder=5)
    ax.scatter(xy[-1, 0], xy[-1, 1], color="red", s=60, label="end", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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


def run_pipeline(cfg: PipelineConfig) -> dict:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download.
    if cfg.skip_download:
        existing = list(cfg.data_dir.glob("input.*"))
        if not existing:
            raise FileNotFoundError("--skip-download but no cached video in data/")
        video_path = existing[0]
        print(f"[1/5] Using cached video: {video_path}")
    else:
        print(f"[1/5] Downloading video from {cfg.url}")
        video_path = download_video(cfg.url, cfg.data_dir)
        print(f"      -> {video_path}")

    # 2. Frames.
    print(f"[2/5] Extracting frames (stride={cfg.frame_stride}, "
          f"window={cfg.vo_start_sec}-{cfg.vo_end_sec}s)")
    frames = extract_frames(
        video_path,
        stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
        start_sec=cfg.vo_start_sec,
        end_sec=cfg.vo_end_sec,
    )
    print(f"      -> {len(frames.frames)} frames @ {frames.fps:.1f} fps")

    # 3. Visual odometry. Cache to disk: VO is the slowest CPU stage of
    # the pipeline (minutes on a 7-minute clip), and the recovered
    # trajectory is fully determined by (video, segment, stride), so a
    # re-run with the same VO parameters can short-circuit straight to
    # step 5.
    #
    # Cache key formats, newest first:
    #   v2 with max_frames:  trajectory_v2_<s>-<e>_s<stride>_f<max_frames>.npz
    #   v2 legacy:           trajectory_v2_<s>-<e>_s<stride>.npz  (max_frames was
    #                        implicit; loaded if shapes are consistent)
    # The legacy fallback exists because we have v2-legacy files on disk
    # from earlier runs (no _fN suffix); without this fallback every run
    # after the cache key change burns minutes redoing identical work.
    cache_v2_with_frames = cfg.data_dir / (
        f"trajectory_v2_{cfg.vo_start_sec:.0f}-{cfg.vo_end_sec or 'end'}"
        f"_s{cfg.frame_stride}_f{cfg.max_frames}.npz"
    )
    cache_v2_legacy = cfg.data_dir / (
        f"trajectory_v2_{cfg.vo_start_sec:.0f}-{cfg.vo_end_sec or 'end'}"
        f"_s{cfg.frame_stride}.npz"
    )
    # Older runs predate per-submission slug folders — fall back to a
    # flat-layout cache at the data-dir parent if nothing is in the slug.
    cache_v2_flat = cfg.data_dir.parent / cache_v2_legacy.name

    vo_cache = cache_v2_with_frames
    legacy_used: Path | None = None
    for candidate in (cache_v2_with_frames, cache_v2_legacy, cache_v2_flat):
        if candidate.exists():
            try:
                z_probe = np.load(candidate)
                shape_ok = z_probe["valid"].shape[0] == len(frames.frames)
            except Exception:
                continue
            if shape_ok:
                if candidate is not cache_v2_with_frames:
                    legacy_used = candidate
                vo_cache = candidate
                break

    if vo_cache.exists() and (legacy_used or vo_cache is cache_v2_with_frames):
        msg = f"[3/5] Loading cached trajectory: {vo_cache}"
        if legacy_used is not None:
            msg += "  (legacy cache key)"
        print(msg)
        z = np.load(vo_cache)
        traj = Trajectory(
            centers=z["centers"],
            xz=z["xz"],
            valid=z["valid"],
            n_inliers=z["n_inliers"].tolist(),
            rotations=z["rotations"],
            translations=z["translations"],
        )
        n_valid = int(traj.valid.sum())
        print(f"      -> {n_valid}/{len(traj.valid)} valid relative poses, "
              f"trajectory shape {traj.xz.shape}")
        # Promote the legacy cache to the canonical key so subsequent
        # runs don't pay the lookup cost again.
        if legacy_used is not None:
            cache_v2_with_frames.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_v2_with_frames,
                centers=traj.centers,
                xz=traj.xz,
                valid=traj.valid,
                n_inliers=np.asarray(traj.n_inliers),
                rotations=traj.rotations,
                translations=traj.translations,
            )
            print(f"      -> promoted legacy cache to {cache_v2_with_frames.name}")
        vo_cache = cache_v2_with_frames
    else:
        vo_cache = cache_v2_with_frames
        print(f"[3/5] Running visual odometry on {len(frames.frames)} frames "
              f"(n_workers={cfg.vo_workers or 'auto'})")
        traj = estimate_trajectory(frames.frames, n_workers=cfg.vo_workers)
        n_valid = int(traj.valid.sum())
        print(f"      -> {n_valid}/{len(traj.valid)} valid relative poses, "
              f"trajectory shape {traj.xz.shape}")
        np.savez(
            vo_cache,
            centers=traj.centers,
            xz=traj.xz,
            valid=traj.valid,
            n_inliers=np.asarray(traj.n_inliers),
            rotations=traj.rotations,
            translations=traj.translations,
        )
    _plot_trajectory(traj, cfg.output_dir / "trajectory.png")

    # 4. OSM road graph for the city.
    print(f"[4/5] Fetching OSM driving graph for {cfg.city!r}")
    cache_path = cfg.data_dir / f"{cfg.city.replace(',', '').replace(' ', '_')}.graphml"
    road = fetch_city_graph(cfg.city, cache_path=cache_path)
    print(f"      -> {road.graph.number_of_nodes()} nodes, "
          f"{road.graph.number_of_edges()} edges, CRS={road.crs}")

    # 4b. Optional: use Depth Anything 3's globally-consistent camera path
    # as the trajectory the shape matcher consumes. Monocular VO drifts
    # over long clips (the true Olgastraße route's bearing correlation was
    # only 0.16 vs 0.29–0.32 for *wrong* parallel streets — pure drift),
    # which lets several parallel streets fit equally well. DA3 is metric
    # and multi-frame-consistent, so its trajectory has far less drift.
    # We keep the VO `traj` for the splat/IPM renders (those are tied to
    # the per-frame VO poses) and only swap the matcher's input here.
    match_xz = traj.xz
    da3_rec = None
    if cfg.use_da3_trajectory:
        print("[4b] Depth Anything 3 trajectory (drift-free matcher input)")
        from .da3_reconstruction import da3_trajectory_xy, reconstruct_with_da3
        try:
            da3_rec = reconstruct_with_da3(
                frames.frames,
                n_keyframes=cfg.da3_keyframes,
                valid_mask=traj.valid,
                device="cuda",
            )
            da3_xy = da3_trajectory_xy(da3_rec)
            if len(da3_xy) >= 2 and bool(np.isfinite(da3_xy).all()):
                match_xz = da3_xy
                _plot_xy(
                    da3_xy, cfg.output_dir / "trajectory_da3.png",
                    "DA3 globally-consistent trajectory (matcher input)",
                )
                print(f"      -> using DA3 trajectory: {len(da3_xy)} keyframe poses")
                result_traj_source = "da3"
            else:
                print("      -> DA3 trajectory invalid; falling back to VO path")
                result_traj_source = "vo"
        except Exception as e:
            print(f"      -> DA3 trajectory failed ({e}); falling back to VO path")
            da3_rec = None
            result_traj_source = "vo"
    else:
        result_traj_source = "vo"

    # 5. Match.
    print(f"[5/5] Matching trajectory against {road.graph.number_of_nodes()} candidate starts")
    candidates = match_trajectory(
        match_xz,
        road,
        final_top_k=cfg.top_k,
        sample_every=cfg.sample_every,
        estimated_length_m=cfg.estimated_length_m,
    )

    if not candidates:
        print("      -> no matches!")
        result = {"city": cfg.city, "matches": []}
    else:
        print(f"      -> top {len(candidates)} matches:")
        for i, c in enumerate(candidates):
            names = ", ".join(
                candidate_geographic_summary(c, road.graph)["street_names"][:3]
            ) or "(unnamed)"
            print(f"        #{i+1}  RMS={c.score:7.1f} m  corr={c.bearing_corr:+.3f}  {names}")
        _plot_match(road, candidates, cfg.output_dir / "match.png")

        result = {
            "city": cfg.city,
            "url": cfg.url,
            "n_frames": len(frames.frames),
            "n_valid_poses": n_valid,
            "trajectory_source": result_traj_source,
            "matches": [
                candidate_geographic_summary(c, road.graph) for c in candidates
            ],
        }

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
            estimated_length_m=cfg.estimated_length_m,
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

    # 6. Splat reconstruction (sparse SfM point cloud) — for the aerial
    # matching channel and for visualization.
    splat_img_rgb = None
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
            splat_img_rgb = render_topdown_splat(splat_pts, splat_cols)

            if cfg.enable_full_splat:
                from .full_splat import render_full_splat_to_file, render_full_splat_topdown
                hq_path = cfg.output_dir / "splat_topdown_hq.png"
                render_full_splat_to_file(
                    splat_pts, splat_cols, hq_path,
                    scale=cfg.full_splat_scale,
                    opacity=cfg.full_splat_opacity,
                    progress=True,
                )
                print(f"      -> wrote {hq_path} (anisotropic Gaussian render)")
                splat_img_rgb = render_full_splat_topdown(
                    splat_pts, splat_cols,
                    scale=cfg.full_splat_scale,
                    opacity=cfg.full_splat_opacity,
                )
        else:
            print("      -> no triangulated points")

    # 7. IPM road-plane BEV stitch — produced *before* aerial matching so
    # we can use it as the aerial-matching input. IPM gives a real
    # road-texture top-down view that shares features (intersection
    # corners, road edges) with the OSM raster patches; the sparse splat
    # top-down has very few of those, which is why aerial matching on
    # the splat alone gave low inlier counts. With IPM available we
    # prefer it over the splat as the aerial input.
    ipm_canvas = None
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
            ipm_canvas = render_ipm_canvas(
                list(frames.frames), traj.xz, K,
                keyframe_stride=ipm_stride,
                camera_height_m=cfg.ipm_camera_height_m,
                pitch_deg=cfg.ipm_pitch_deg,
            )
            ipm_path = cfg.output_dir / "ipm_bev.png"
            cv2.imwrite(str(ipm_path), ipm_canvas)
            print(f"      -> wrote {ipm_path} ({ipm_canvas.shape[1]}x{ipm_canvas.shape[0]})")
            result["ipm"] = str(ipm_path.relative_to(cfg.output_dir))
        except Exception as e:
            print(f"      -> IPM failed: {e}")
            result["ipm_error"] = str(e)
            ipm_canvas = None

    # 8. Aerial matching channel.
    # Primary signal: trajectory-raster IoU (always runs — uses aligned_traj_xy
    # from each MatchCandidate, no top-down image required).
    # Supplemental signal: ORB on top-down image vs OSM patch (weak when
    # image is a photographic IPM vs a schematic OSM render; retained for
    # completeness and inter-method comparison).
    if cfg.enable_aerial_match and candidates:
        if ipm_canvas is not None:
            aerial_input = cv2.cvtColor(ipm_canvas, cv2.COLOR_BGR2RGB)
            aerial_source = "ipm_bev"
        elif splat_img_rgb is not None:
            aerial_input = splat_img_rgb
            aerial_source = "splat_topdown"
        else:
            aerial_input = None
            aerial_source = "traj_iou_only"

        print(f"[8] Aerial matching: traj-IoU + {'ORB on ' + aerial_source if aerial_input is not None else 'IoU only'} "
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
        aerial_rank = {idx: r + 1 for r, idx in enumerate(order_by_aerial)}

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
                  f"     #{aerial_rank[i]:<2}     {ar.traj_coverage:.3f}    {ar.traj_iou:.3f}       {ar.n_inliers:3d}        | {names}")

        for i, ar in enumerate(aerial_results):
            m = result["matches"][i]
            m["shape_rank"] = i + 1
            m["traj_iou"] = ar.traj_iou
            m["traj_coverage"] = ar.traj_coverage
            m["aerial_score"] = ar.aerial_score
            m["aerial_orb_matches"] = ar.n_orb_matches
            m["aerial_inliers"] = ar.n_inliers
            m["aerial_inlier_ratio"] = ar.inlier_ratio
            m["aerial_rank"] = aerial_rank[i]

        # Weighted rank-fusion consensus. Channels are weighted by how
        # well they track ground truth on GT-evaluated runs:
        #   * shape rank  — primary geometric fit (weight 1.0)
        #   * sliding-window rank — the strongest *secondary* signal: a
        #     candidate supported across many trajectory windows is hard
        #     to fool (weight 1.0). Falls back to shape rank when the
        #     sliding-window channel is disabled.
        #   * aerial (coverage) rank — useful but weaker; ORB already
        #     dropped from its score, so it is the trajectory-coverage
        #     signal only (weight 0.5).
        # Lower fused score = stronger multi-channel agreement.
        W_SHAPE, W_SLIDING, W_AERIAL = 1.0, 1.0, 0.5

        def _sliding_rank(i: int) -> int:
            return int(result["matches"][i].get("sliding_window_rank", i + 1))

        def _fused(i: int) -> float:
            return (
                W_SHAPE * (i + 1)
                + W_SLIDING * _sliding_rank(i)
                + W_AERIAL * aerial_rank[i]
            )

        for i in range(len(aerial_results)):
            result["matches"][i]["consensus_score"] = _fused(i)

        consensus_order = sorted(
            range(len(aerial_results)),
            key=lambda i: (_fused(i), i),
        )
        consensus_idx = consensus_order[0]
        print()
        print(f"        Consensus pick: candidate #{consensus_idx + 1} "
              f"(shape #{consensus_idx + 1}, "
              f"sliding #{_sliding_rank(consensus_idx)}, "
              f"aerial #{aerial_rank[consensus_idx]}, "
              f"fused={_fused(consensus_idx):.1f})")

        # Reorder candidates by consensus and update the result JSON.
        # This ensures result["matches"][0] is the consensus-best answer.
        reordered_matches = [result["matches"][i] for i in consensus_order]
        for rank, m in enumerate(reordered_matches):
            m["final_rank"] = rank + 1
        result["matches"] = reordered_matches
        # Mirror the reorder in candidates for downstream (GT eval).
        candidates = [candidates[i] for i in consensus_order]
        print(f"        Final #1 after consensus re-rank: "
              f"{', '.join(result['matches'][0]['street_names'][:3]) or '(unnamed)'}")

    if cfg.embedding_sources and candidates:
        if ipm_canvas is not None:
            embedding_query = cv2.cvtColor(ipm_canvas, cv2.COLOR_BGR2RGB)
            embedding_query_name = "ipm_bev"
        elif splat_img_rgb is not None:
            embedding_query = splat_img_rgb
            embedding_query_name = "splat_topdown"
        else:
            embedding_query = None
            embedding_query_name = None

        print(
            f"[8b] Deep embedding retrieval "
            f"({', '.join(cfg.embedding_sources)} via {cfg.embedding_model})"
        )
        if embedding_query is None:
            print("      -> skipped: no top-down query image available")
            result["embedding_retrieval_error"] = (
                "Embedding retrieval needs IPM or splat top-down imagery."
            )
        else:
            embedding_results = score_candidates_by_embeddings(
                embedding_query,
                road,
                candidates,
                output_dir=cfg.output_dir / "embeddings",
                sources=cfg.embedding_sources,
                model_name=cfg.embedding_model,
                geotessera_year=cfg.geotessera_year,
            )
            result["embedding_query"] = embedding_query_name
            result["embedding_retrieval"] = {
                "model": cfg.embedding_model,
                "sources": list(cfg.embedding_sources),
            }
            for source, source_results in embedding_results.items():
                order = sorted(
                    range(len(source_results)),
                    key=lambda i: -source_results[i].cosine_similarity,
                )
                rank_map = {idx: rank + 1 for rank, idx in enumerate(order)}
                print(f"      -> {source}:")
                for i, emb in enumerate(source_results):
                    result["matches"][i][f"{source}_embedding_score"] = emb.cosine_similarity
                    result["matches"][i][f"{source}_embedding_rank"] = rank_map[i]
                    if emb.image_path is not None:
                        result["matches"][i][f"{source}_embedding_image"] = str(
                            emb.image_path.relative_to(cfg.output_dir)
                        )
                    if emb.error:
                        result["matches"][i][f"{source}_embedding_error"] = emb.error
                    print(
                        f"        #{i+1}  sim={emb.cosine_similarity:+.3f}  "
                        f"rank=#{rank_map[i]}"
                        + (f"  error={emb.error}" if emb.error else "")
                    )

    # 8c. Optional: BevSplat cross-view localization channel.
    if cfg.enable_bev_splat and candidates:
        print(
            f"[8c] BevSplat cross-view localization "
            f"(source={cfg.bev_splat_source}, "
            f"tile={cfg.bev_splat_tile_size}px / {cfg.bev_splat_half_extent_m}m)"
        )
        h_img, w_img = frames.frames[0].shape[:2]
        K = default_intrinsics(w_img, h_img)
        # Pick a frame near the middle of the window as the BevSplat query
        # — the middle is the most "average" view of the whole route and
        # avoids start-of-clip warmup or end-of-clip drift.
        mid = len(frames.frames) // 2
        query_rgb = cv2.cvtColor(frames.frames[mid], cv2.COLOR_BGR2RGB)
        bev_results = score_candidates_with_bevsplat(
            query_rgb, K, road, candidates,
            output_dir=cfg.output_dir / "bev_splat",
            config=BevSplatConfig(
                weights_path=cfg.bev_splat_weights,
                repo_path=cfg.bev_splat_repo_path,
                model_module=cfg.bev_splat_model_module,
                satellite_source=cfg.bev_splat_source,
                satellite_size=cfg.bev_splat_tile_size,
                half_extent_m=cfg.bev_splat_half_extent_m,
                geotessera_year=cfg.geotessera_year,
            ),
        )
        if bev_results:
            n_failed = sum(1 for r in bev_results if r.error)
            n_ok = len(bev_results) - n_failed
            if n_ok > 0:
                order = sorted(
                    range(len(bev_results)),
                    key=lambda i: -bev_results[i].score,
                )
                bev_rank = {idx: r + 1 for r, idx in enumerate(order)}
            else:
                bev_rank = {i: i + 1 for i in range(len(bev_results))}
            print(f"      -> {n_ok}/{len(bev_results)} candidates scored "
                  f"({n_failed} skipped — see result.json for per-candidate errors)")
            for i, bv in enumerate(bev_results):
                m = result["matches"][i]
                m["bev_splat_score"] = bv.score
                m["bev_splat_pred_shift_u"] = bv.pred_shift_u
                m["bev_splat_pred_shift_v"] = bv.pred_shift_v
                m["bev_splat_pred_heading"] = bv.pred_heading
                m["bev_splat_rank"] = bev_rank[i]
                if bv.satellite_path is not None:
                    m["bev_splat_tile"] = str(bv.satellite_path.relative_to(cfg.output_dir))
                if bv.error:
                    m["bev_splat_error"] = bv.error
                tag = f" rank=#{bev_rank[i]}" if n_ok > 0 else ""
                err = f" error={bv.error}" if bv.error else ""
                print(f"        #{i+1}  score={bv.score:.3f}{tag}{err}")
            result["bev_splat"] = {
                "source": cfg.bev_splat_source,
                "tile_size": cfg.bev_splat_tile_size,
                "half_extent_m": cfg.bev_splat_half_extent_m,
                "n_candidates_scored": n_ok,
                "n_candidates_failed": n_failed,
            }

    # 9. Optional: DA3 dense reconstruction (proper splat replacement).
    if cfg.enable_da3:
        print(f"[9] Depth Anything 3 dense reconstruction "
              f"({cfg.da3_keyframes} keyframes)")
        from .da3_reconstruction import (
            reconstruct_with_da3,
            da3_trajectory_xy,
        )
        try:
            # Reuse the reconstruction already computed for the matcher
            # trajectory (step 4b) instead of running DA3 twice.
            if da3_rec is not None:
                rec = da3_rec
                print("      -> reusing DA3 reconstruction from step [4b]")
            else:
                rec = reconstruct_with_da3(
                    frames.frames,
                    n_keyframes=cfg.da3_keyframes,
                    valid_mask=traj.valid,
                    device="cuda",
                )
            print(f"      -> {len(rec.points_world)} dense points; "
                  f"{rec.extrinsics_w2c.shape[0]} keyframe poses")

            ply = cfg.output_dir / "splat_da3.ply"
            save_ply(rec.points_world, rec.colors_rgb, ply)
            print(f"      -> wrote {ply}")
            html = cfg.output_dir / "splat_da3.html"
            save_interactive_html(
                rec.points_world, rec.colors_rgb, html,
                title="Depth Anything 3 dense reconstruction",
            )
            print(f"      -> wrote {html}")
            top = cfg.output_dir / "splat_da3_topdown.png"
            render_topdown_to_file(rec.points_world, rec.colors_rgb, top,
                                   resolution=1024, point_radius_px=1)
            print(f"      -> wrote {top}")

            if cfg.enable_full_splat:
                from .full_splat import render_full_splat_to_file
                hq = cfg.output_dir / "splat_da3_topdown_hq.png"
                render_full_splat_to_file(
                    rec.points_world, rec.colors_rgb, hq,
                    scale=cfg.full_splat_scale,
                    opacity=cfg.full_splat_opacity,
                    progress=True,
                )
                print(f"      -> wrote {hq} (anisotropic Gaussian render)")

            if cfg.enable_train_3dgs:
                print(f"      -> training real 3DGS via gsplat ({cfg.train_3dgs_iters} iters)")
                try:
                    from .full_splat import fit_3dgs, save_trained_splat_ply
                    trained = fit_3dgs(
                        rec, frames.frames,
                        n_iters=cfg.train_3dgs_iters,
                        device="cuda",
                    )
                    out_ply = cfg.output_dir / "splat_3dgs.ply"
                    save_trained_splat_ply(trained, out_ply)
                    print(f"      -> wrote {out_ply} "
                          f"(open in SuperSplat / antimatter15 viewer; "
                          f"final loss={trained.final_loss:.4f})")
                    result["trained_3dgs"] = {
                        "n_gaussians": int(len(trained.means)),
                        "n_iters": int(trained.n_iters),
                        "final_loss": float(trained.final_loss),
                        "ply": str(out_ply.relative_to(cfg.output_dir)),
                    }
                except Exception as e:
                    print(f"      -> 3DGS training failed: {e}")
                    result["trained_3dgs_error"] = str(e)

            result["da3"] = {
                "n_points": int(len(rec.points_world)),
                "n_keyframes": int(rec.extrinsics_w2c.shape[0]),
                "outputs": [
                    str(ply.relative_to(cfg.output_dir)),
                    str(html.relative_to(cfg.output_dir)),
                    str(top.relative_to(cfg.output_dir)),
                ],
            }
        except Exception as e:
            print(f"      -> DA3 failed: {e}")
            result["da3_error"] = str(e)

    # 10. Optional: ground-truth evaluation.
    if cfg.ground_truth_streets and candidates:
        gt = list(cfg.ground_truth_streets)
        print(f"[10] Evaluating against ground-truth streets: {gt}")
        gt_results = evaluate_candidates(candidates, road, gt)
        best_dist_rank, name_rank = best_rank_for_gt(gt_results)
        print(f"      -> closest-to-GT candidate is shape #{best_dist_rank}; "
              f"first GT-named candidate is rank {name_rank}")
        for i, r in enumerate(gt_results):
            tag = f"  ON-GT({', '.join(r.matching_gt_names)})" if r.on_gt_street else ""
            print(f"        #{i+1}  d_to_GT={r.nearest_distance_m:7.1f} m{tag}")
            result["matches"][i]["gt_distance_m"] = r.nearest_distance_m
            result["matches"][i]["gt_on_street"] = r.on_gt_street
            result["matches"][i]["gt_matching_names"] = r.matching_gt_names
        result["ground_truth"] = {
            "streets": gt,
            "best_distance_rank": best_dist_rank,
            "first_named_rank": name_rank,
        }

    out_json = cfg.output_dir / "result.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"      -> wrote {out_json}")

    return result
