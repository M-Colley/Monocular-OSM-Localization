"""End-to-end pipeline: YouTube URL -> matched street(s) in target city.

The CLI in `main.py` is a thin wrapper around `run_pipeline` here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: don't try to open a window
import matplotlib.pyplot as plt
import numpy as np

from .aerial_match import match_splat_against_candidates
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

    # 3. Visual odometry. Cache to disk: VO is the slowest part of the
    # pipeline (minutes on a 5-minute clip), and the recovered trajectory
    # is fully determined by (video, segment, stride), so a re-run with
    # the same VO parameters can short-circuit straight to step 5.
    vo_cache = cfg.data_dir / (
        f"trajectory_v2_{cfg.vo_start_sec:.0f}-{cfg.vo_end_sec or 'end'}"
        f"_s{cfg.frame_stride}_f{cfg.max_frames}.npz"
    )
    if vo_cache.exists():
        print(f"[3/5] Loading cached trajectory: {vo_cache}")
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
    else:
        print(f"[3/5] Running visual odometry on {len(frames.frames)} frames")
        traj = estimate_trajectory(frames.frames)
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

    # 5. Match.
    print(f"[5/5] Matching trajectory against {road.graph.number_of_nodes()} candidate starts")
    candidates = match_trajectory(
        traj.xz,
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
            traj.xz,
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
        print("        shape_rank  shape_RMS  shape_corr  | aerial_rank  traj_IoU  ORB_inliers | streets")
        print("        " + "-" * 108)
        for i, ar in enumerate(aerial_results):
            c = candidates[i]
            names = ", ".join(
                candidate_geographic_summary(c, road.graph)["street_names"][:2]
            ) or "(unnamed)"
            print(f"           #{i+1:<2}      {c.score:7.1f} m   {c.bearing_corr:+.3f}     |"
                  f"     #{aerial_rank[i]:<2}     {ar.traj_iou:.3f}       {ar.n_inliers:3d}        | {names}")

        for i, ar in enumerate(aerial_results):
            m = result["matches"][i]
            m["shape_rank"] = i + 1
            m["traj_iou"] = ar.traj_iou
            m["aerial_score"] = ar.aerial_score
            m["aerial_orb_matches"] = ar.n_orb_matches
            m["aerial_inliers"] = ar.n_inliers
            m["aerial_inlier_ratio"] = ar.inlier_ratio
            m["aerial_rank"] = aerial_rank[i]
            m["consensus_rank_sum"] = (i + 1) + aerial_rank[i]

        # Re-rank candidates by combined score: shape rank + aerial rank
        # (lower sum = both methods agree this is the best match).
        consensus_order = sorted(
            range(len(aerial_results)),
            key=lambda i: ((i + 1) + aerial_rank[i], i),
        )
        consensus_idx = consensus_order[0]
        print()
        print(f"        Consensus pick: candidate #{consensus_idx + 1} "
              f"(shape #{consensus_idx + 1}, aerial #{aerial_rank[consensus_idx]})")

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

    # 9. Optional: DA3 dense reconstruction (proper splat replacement).
    if cfg.enable_da3:
        print(f"[9] Depth Anything 3 dense reconstruction "
              f"({cfg.da3_keyframes} keyframes)")
        from .da3_reconstruction import (
            reconstruct_with_da3,
            da3_trajectory_xy,
        )
        try:
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
