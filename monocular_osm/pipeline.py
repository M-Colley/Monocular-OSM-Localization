"""End-to-end pipeline: video (local file or YouTube URL) + city name
-> WGS84 position of the video.

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
from .evaluator import (
    best_rank_for_gt,
    best_rank_for_waypoints,
    evaluate_candidates,
    evaluate_candidates_against_waypoints,
    load_gt_waypoints,
)
from .frame_extraction import extract_frames
from .geo import geodesic_distance_m
from .fusion import (
    CandidateFeatures,
    FusionWeights,
    consensus_order as fusion_consensus_order,
    effective_weights as fusion_effective_weights,
    fused_score as fusion_fused_score,
    write_feature_table,
)
from .ipm import render_ipm_canvas
from .osm_data import RoadGraph, fetch_city_graph
from .hypotheses import distinct_hypotheses, hypothesis_confidence
from .position import (
    build_position_report,
    candidate_center_latlon,
    format_position_summary,
    google_maps_url,
)
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
from .visual_odometry import (
    VO_ALGO_VERSION,
    Trajectory,
    default_intrinsics,
    estimate_trajectory,
)


@dataclass
class PipelineConfig:
    url: str
    city: str
    data_dir: Path
    output_dir: Path
    # None = no cap; the [vo_start_sec, vo_end_sec] segment bounds the
    # frame count. A fixed cap silently truncates longer segments.
    max_frames: int | None = None
    frame_stride: int = 6
    vo_start_sec: float = 0.0
    vo_end_sec: float | None = 300.0
    # How many matcher candidates the fusion channels see. 10 matches the CLI
    # default every published run used (the config default used to be 5, so
    # API callers silently got a different pipeline).
    top_k: int = 10
    # Walk-enumeration budget per start node (trajectory_matching). With the
    # default 40, routes with >2 deliberate turns or a turn after the 8th
    # intersection cannot be generated from their true start; raising it is
    # a runtime/recall trade-off to measure (scripts/diagnose_pool_recall.py).
    walks_per_node: int = 40
    # Where cfg.osm_around came from: "user" (--osm-around), "video_title"
    # (place names in the title/description), or None. Recorded so an
    # evaluator can tell a deployable seed from a ground-truth disc.
    osm_around_source: str | None = None
    # Consensus-fusion weight overrides, e.g. {"sliding": 0.0} (see
    # monocular_osm/fusion.py: FusionWeights). None = the defaults the
    # published results used. Exposed for ablation, not tuning on the fleet.
    fusion_weights: dict | None = None
    # Route-length prior for OSM walk enumeration. None (default) derives
    # it from the analyzed duration at urban average speed — see
    # _auto_estimated_length_m. A wrong prior is costly: walks much longer
    # than the true route make Procrustes stretch the trajectory onto
    # roads the car never drove (the Ulm GT run with the old fixed 8000 m
    # default vs a true ~2.1 km route had a 878 m start error).
    estimated_length_m: float | None = None
    skip_download: bool = False
    sample_every: int = 1
    enable_splat: bool = True
    splat_max_pairs: int = 80
    enable_aerial_match: bool = True
    # 3D-tile skyline channel (opt-in): fetch the open-data LoD2 city
    # model (Berlin / Baden-Wuerttemberg CityGML, monocular_osm/citygml_lod2.py),
    # render it at sampled poses along each candidate route and score
    # rendered-vs-observed skyline agreement (monocular_osm/tile3d_match.py).
    # Open data only — Google Photorealistic 3D Tiles is ToS-prohibited
    # for analysis/offline use, see the citygml_lod2 module docstring.
    use_tile3d: bool = False
    tile3d_source: str = "auto"    # auto | berlin | bw
    tile3d_samples: int = 16
    tile3d_max_tiles: int = 80
    # Opt-in (experiment): let a CLEARLY discriminative skyline channel
    # promote its top candidate to the front of the consensus, overriding
    # the shape pick. Triple-gated (score margin + tight skyline error +
    # winner already near the top of the shape ranking) so it no-ops where
    # skylines don't discriminate (uniform mid-rise) and can only reorder
    # among already-plausible candidates. Off by default — it changes the
    # tuned consensus only when asked. See tile3d eval 2026-07-22.
    tile3d_tiebreak: bool = False
    # Uncertainty-aware fusion: scale the tile3d rank weight by how
    # discriminative the skyline scores are (self-mutes on uniform mid-rise,
    # full weight in dense high-rise) instead of a fixed 0.4.
    tile3d_adaptive: bool = False
    # Metric placement refinement: after selection, optimize a small rigid
    # transform of the reported route to best align its rendered skyline
    # with the video skyline (LoD-Loc-style; attacks the absolute-placement
    # residual an anchor leaves). No-op-safe.
    tile3d_refine: bool = False
    # Learned sky segmentation (Cityscapes SegFormer) instead of the Lab
    # smoothness heuristic for the observed skyline — robust to night /
    # glare / occlusion. Needs transformers + a one-off model download.
    tile3d_skyseg: bool = False
    enable_da3: bool = False
    use_da3_trajectory: bool = False
    use_mapanything_trajectory: bool = False
    openvo_trajectory_path: "Path | None" = None
    prefer_openvo_trajectory: bool = True   # default: auto-use a staged OpenVO path, VO fallback
    # VGGT-Long chunked-VGGT trajectory (ICRA'25): ~30% lower Procrustes RMS
    # than OpenVO on KITTI 0033 (148.6 vs 211.0 m). EXPLICIT opt-in only —
    # better shape has not historically implied better end-to-end
    # (MapAnything), so it must earn a default via the GT sweep.
    vggt_long_trajectory_path: "Path | None" = None
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
    # OCR scene-text → geocoded POI anchor channel. The only channel that
    # injects absolute geographic info from the video; seeds enumeration
    # near anchors and re-ranks by anchor proximity. Needs easyocr +
    # network geocoding (both cached after first run).
    enable_ocr_anchor: bool = False
    use_ocr_super_res: bool = False
    ocr_sample_interval_sec: float = 6.0
    ocr_min_confidence: float = 0.5
    # Local OSM gazetteer anchors: fuzzy-match OCR text against named OSM
    # features (POIs, transit stops) in the graph area — a free, offline
    # anchor source that complements the rate-limited Nominatim path. On by
    # default (purely additive; deduped by name). Recovered 2 sub-300 m
    # anchors on London where Nominatim found none. (monocular_osm/osm_gazetteer.py)
    use_osm_gazetteer: bool = True
    # Here-vs-direction sign classification (Gemma 4): reads each OCR anchor's
    # sign crop and drops *directional* signs (which name places elsewhere and
    # geocode far off-route — the London 'Holborn' failure). GPU, off by
    # default. (monocular_osm/vlm_anchor.classify_sign_types)
    classify_signs: bool = False
    # Optional separate (higher-res) video for OCR only. VO/matching stay
    # on the main video; OCR reads frames from here when set. Lets a 4K
    # source feed street-plate OCR without re-running VO at 4K.
    ocr_video_path: Path | None = None
    # Scale recovery / georeferencing from timed anchors (ideas 1 & 2).
    # Fits a similarity VO->world from anchor (time -> geocoded latlon)
    # correspondences; sets the metric length prior and georeferences the
    # reported route. Robust thresholds — declines (falls back) when the
    # anchors are too few/clustered/noisy for a reliable fit.
    # (lat, lon, radius_m) to fetch a bounded disc of the OSM graph
    # instead of the whole named place — needed for mega-cities.
    osm_around: tuple[float, float, float] | None = None
    # GPS-free frame-based coarse seed: when osm_around is None, read the
    # license-plate registration district (German clips) and/or a VLM's read
    # of the scene to bound the search near the drive — the deployable seed
    # for videos whose title names no place. Best-effort; no-op if nothing
    # resolves. (Title/description seeding is handled earlier, in main.py.)
    # Fast path = license-plate district + OCR place names (deterministic).
    coarse_from_frames: bool = True
    # Add the (slow) VLM scene-read to the frame seed. Off by default because
    # loading the VLM adds minutes; enable only where plate+OCR find nothing.
    coarse_from_vlm: bool = False
    # Blind VPR coarse prior (KartaView + EigenPlaces): a re-rank centre and the
    # anchor-primary placement prior — shape-independent fix for the SELECTION
    # wall. (Gating the OSM graph to the prior's disc was refuted by experiment:
    # 664 -> 1276 m on Ulm — so there is deliberately no gate knob.)
    use_vpr_prior: bool = True
    vpr_search_radius_m: float = 3000.0
    # Max reference photos fetched+embedded per clip. The fetch tiles the disc
    # and UNIFORM-subsamples down to this cap, so a low cap thins DENSE areas
    # (the 0009 start has 37 Mapillary images within 50 m but the cap-1500
    # cache kept 2) — raise it to densify retrieval-bound starts. Higher cap =
    # more downloads + a longer embed pass on a cold cache.
    vpr_ref_cap: int = 1500
    # VPR reference source: "kartaview" (open, tokenless) or "mapillary" (much
    # denser — needs a free MLY_TOKEN env var; validated 3-31 m to route on all
    # GT clips, incl. the ones KartaView could not cover). See mapillary VPR.
    vpr_source: str = "mapillary"
    # Coarse-to-fine VPR: after pass 1 over a WIDE (coarse-prior) disc, the
    # robust centre is far tighter than the seed (0.2-0.76 km vs a 0.5-6.6 km
    # deployable seed on the GT clips). When on, re-fetch a TIGHT disc around
    # that centre and re-run VPR — less dilution AND it re-centres the OSM
    # graph tight (shrinks the matcher pool). Fires only when pass 1 was wider
    # than 1.2x the tight radius, so GT-seeded tight runs are untouched; the
    # tight radius is floored by the route-length prior so a long route is
    # never clipped. Validated 2026-07-08: makes the fully-deployable (no-GT)
    # two-pass MATCH the GT-seeded numbers (5-clip start 113 m vs the leak's
    # 131 m) — the location leak is unnecessary. Fails only where pass 1 lands
    # in the wrong area (no signal, e.g. KITTI-0033). NOTE: distinct from
    # vpr_two_pass_scale below (a SCALE arbitration at matching time).
    vpr_coarse_to_fine: bool = True
    vpr_c2f_radius_m: float = 2000.0
    # Experimental: score candidates against the per-frame VPR track (sequence-
    # median distance at matched arc fractions) instead of the centroid-only
    # distance. Untested on GT — off by default.
    use_vpr_sequence: bool = False
    # Viterbi sequence decode of the VPR track (continuity-constrained states
    # instead of per-frame argmax). Offline: median better on all 5 GT clips,
    # p90 tail collapses ~5-8x. E2e (sweep 2026-07-04): transforms the weak
    # clips (0009 mean 80->36, 0033 pins unlock: start 246->228) at a small
    # cost on London (start 31->51) — fleet mean 109.6->90.2 m, best
    # recorded. Default on; kill-switch for A/Bs.
    vpr_viterbi: bool = True
    # Two-pass VPR scale arbitration: match at both the VO scale and a scale
    # pinned to the VPR track extent, keep whichever candidates better explain
    # the full VPR track. Fixes candidate SHAPE where the VO scale is wrong
    # (London mean 355->76 m), but the anchor placement's scale retry +
    # orientation refine reach the same headline without the second matching
    # pass (A/B 2026-07-04: identical anchored errors on London/Ulm/comma) —
    # so off by default.
    vpr_two_pass_scale: bool = False
    # License-plate registration-district anchor: read EU plate region prefixes,
    # vote, geocode the modal district -> re-rank penalty (monocular_osm/plate_anchor.py).
    # (A hard region gate was refuted; the penalty free-radius lives in
    # _PLATE_FREE_RADIUS_M below.)
    use_plate_anchor: bool = False
    # VLM (Gemma 4) district/landmark prior — a coverage fallback when VPR finds
    # no references (monocular_osm/vlm_anchor.py). Feeds the same anchor-primary path.
    use_vlm_anchor: bool = False
    # Absolute heading from the sun (activates only if the clip has a capture time).
    use_sun_heading: bool = False
    enable_scale_recovery: bool = True
    scale_recovery_thresh_m: float = 150.0
    scale_recovery_min_inliers: int = 3
    scale_recovery_min_baseline_m: float = 250.0
    # Idea 3: ground-plane optical-flow scale. Off by default — needs
    # camera calibration to be reliable (on the uncalibrated Ulm clip it
    # was wildly pitch-sensitive). When on, sets the metric length prior
    # subject to the same sanity gate.
    use_ipm_scale: bool = False
    # IPM-scale geometry. Pitch is the sensitive knob: a near-horizontal
    # dashcam (~1-2 deg) recovers scale within ~10% (KITTI 0033), while 6 deg
    # over-estimates ~35%. Default to a typical windshield-mount pitch.
    ipm_scale_pitch_deg: float = 1.5
    ipm_scale_camera_height_m: float = 1.4
    # Lock the matcher's alignment scale to (estimated_length_m / VO arc
    # length) instead of letting Procrustes choose it freely. Forces the
    # matched route to span the prescribed metric extent — stops the
    # compression that left the localized route unable to reach its far
    # end (the Ulm eastern-tail problem).
    scale_lock: bool = True
    ground_truth_streets: tuple[str, ...] = ()
    # JSON file of timestamped GPS fixes along the true route — see
    # evaluator.load_gt_waypoints for the schema and ground_truth/ for
    # examples. Enables metric error reporting per candidate.
    ground_truth_waypoints: Path | None = None
    enable_bev_splat: bool = False
    bev_splat_weights: Path | None = None
    bev_splat_repo_path: Path | None = None
    bev_splat_model_module: str = "models.models_kitti_nips"
    bev_splat_source: str = "esri"
    bev_splat_tile_size: int = 512
    bev_splat_half_extent_m: float = 60.0
    # How many top-by-geometry candidates appearance may reorder. The
    # default 5 assumes geometry already ranks the truth near the top; the
    # multi-clip benchmark shows it often does NOT (the GT-best sits at
    # geometry-rank #37 on KITTI 0033), so raise this to let appearance
    # rescue a geometrically-buried-but-correct candidate.
    bev_fusion_cap: int = 5
    # Detect a route that returns near its start and redistribute the VO
    # drift so the loop closes (monocular_osm/loop_closure.py). Pair with
    # --use-ipm-scale; closing at a wrong scale doesn't help.
    enable_loop_closure: bool = False
    # Replace the VO's unit-length steps with a per-step length measured from
    # road-plane optical flow (monocular_osm/ground_flow_scale.py). A unit step
    # is a CONSTANT-SPEED assumption imposed on the trajectory shape — the one
    # thing trajectory_matching scores — and on the overlay fleet the true step
    # lengths are worth 107.4 -> 77.6 m of shape error. Costs a second,
    # streaming pass over the video (~2 min per 4-minute clip): the estimator
    # needs the frames BETWEEN the VO's strided ones, and tracking only the
    # strided pair measurably loses most of the benefit.
    #
    # Off by default: measured on the 8-clip overlay fleet only, and the core
    # fleet includes geometry it has never seen (KITTI is 1242x375, not 16:9).
    # It abstains rather than guessing, so a clip whose road plane it cannot
    # model reproduces the unscaled trajectory exactly.
    use_ground_flow_scale: bool = False
    # OCR backend for both consumers (scene_text anchors and the
    # GPS-overlay stamp reader). See scene_text.RapidOcrReader for the
    # measurement behind the alternative.
    ocr_engine: str = "rapidocr"
    # Run VGGT (feed-forward, drift-free poses) to GATE enumeration to the
    # area its trajectory selects, then let the precise (loop-closed) VO
    # geometry pick within it (monocular_osm/vggt_trajectory.py). Needs the vggt
    # package + GPU + ~5 GB weights; degrades to a no-op if unavailable.
    use_vggt_gating: bool = False
    vggt_keyframes: int = 64
    # Refine the shape-matched coarse position with OrienterNet (neural
    # BEV->OSM matching + sequential fusion) — the metric localization head
    # (monocular_osm/orienternet_localizer.py). ~2 m on KITTI. Needs
    # third_party/OrienterNet + GPU + weights; no-op if unavailable.
    use_orienternet: bool = False
    orienternet_keyframes: int = 10
    orienternet_tile_m: float = 160.0
    vo_workers: int | None = None  # None → auto (min(cpu_count, 12)); 1 → sequential
    # When set, use this local video file directly: no download, `url` is
    # informational only (recorded in the result JSON).
    video_path: Path | None = None


# Names that moved into monocular_osm/stages/ stay importable from here.
from .stages.context import (  # noqa: F401
    _record_channel,
    RunContext,
)
from .stages.acquire import (  # noqa: F401
    _NON_VIDEO_SUFFIXES,
    _resolve_input_video,
    _VIDEO_EXT_RANK,
    pick_cached_input,
    rank_cached_inputs,
    acquire_video,
    extract_video_frames,
)
from .stages.length_prior import (  # noqa: F401
    _auto_estimated_length_m,
    _length_sane,
    _URBAN_AVG_SPEED_MPS,
    estimate_flow_scale,
    estimate_length_prior,
)
from .stages.trajectory import (  # noqa: F401
    _apply_ground_flow_scale,
    _da3_trajectory_plausible,
    _find_vo_cache,
    _match_timestamps,
    _plot_trajectory,
    _plot_xy,
    _pose_txt_xz,
    _remap_frame_pair_to_poses,
    _vo_cache_candidates,
    close_loop,
    run_visual_odometry,
    select_matcher_trajectory,
)
from .stages.prior import (  # noqa: F401
    _C2F_SIM_MARGIN,
    _C2F_SNAP_TOL_M,
    _city_extent_reach_m,
    _frame_coarse_seed,
    _OCR_SEED_MIN_PLACES,
    _PASS1_COVER_CAP,
    _PASS1_COVER_MAX_M,
    _PASS1_TOKEN_CAP,
    coarse_seed_from_frames,
    plate_district_prior,
    vlm_district_prior,
    vpr_coarse_prior,
)
from .stages.graph import (  # noqa: F401
    _fetch_road_graph,
    load_road_graph,
)
from .stages.anchors import (  # noqa: F401
    _drop_direction_anchors,
    _SignRep,
    estimate_sun_heading,
    ocr_scene_text_anchors,
)
from .stages.vpr_track import (  # noqa: F401
    _PLATE_FREE_RADIUS_M,
    _vpr_distance_penalty,
    _VPR_FREE_RADIUS_M,
    _vpr_sequence_median_m,
    _vpr_track_geometry,
)
from .stages.match import (  # noqa: F401
    _HYP_POOL,
    _plot_match,
    _vpr_alt_scale,
    lock_alignment_scale,
    match_candidates,
    summarise_matches,
    vggt_gate,
)
from .stages.channels import (  # noqa: F401
    _BEV_FUSION_CAP,
    _fuse_bev_rank,
    _Tile3DInactive,
    _W_BEV,
    aerial_channel,
    bevsplat_channel,
    da3_dense_channel,
    embedding_channel,
    ipm_channel,
    sliding_window_channel,
    splat_channel,
    tile3d_channel,
)
from .stages.consensus import (  # noqa: F401
    _anchor_channel,
    _consensus,
    _heading_diff_deg,
    _mean_bearing_deg,
    _plate_channel,
    _sun_bearing_penalty,
    _sun_channel,
    _SUN_FREE_DEG,
    _SUN_MIN_CONFIDENCE,
    _turn_channel,
    _vpr_channel,
    _vpr_chord_diagnostic,
    _W_SUN,
    fuse_channels,
)
from .stages.evaluation import (  # noqa: F401
    evaluate_ground_truth_streets,
    evaluate_waypoints,
)
from .stages.placement import (  # noqa: F401
    _arc_param_times,
    _elastic_fuse_track,
    _final_position_reports,
    _FUSE_HUBER_M,
    _FUSE_MARGIN_ABS_M,
    _FUSE_MARGIN_RATIO,
    _FUSE_MAX_DEFORM_M,
    _FUSE_MIN_TRACK_PTS,
    _FUSE_SMOOTH_D1,
    _FUSE_SMOOTH_D2,
    _orienternet_refine,
    _pick_top_k_placement,
    _pin_route_to_anchors,
    _place_candidate_on_track,
    _ROT_EXPLAIN_M,
    _ROT_MARGIN_ABS_M,
    _ROT_MARGIN_RATIO,
    _ROT_MIN_EXTENT_M,
    _ROT_MIN_TRACK_PTS,
    _rotation_refine_about_start,
    anchor_primary_placement,
    final_position,
)
from .stages.report import (  # noqa: F401
    _summarize_channels,
    finalize_result,
)




def run_pipeline(cfg: PipelineConfig) -> dict:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext.create(cfg)
    acquire_video(ctx)
    extract_video_frames(ctx)
    estimate_length_prior(ctx)
    estimate_flow_scale(ctx)
    run_visual_odometry(ctx)
    coarse_seed_from_frames(ctx)
    vpr_coarse_prior(ctx)
    plate_district_prior(ctx)
    vlm_district_prior(ctx)
    load_road_graph(ctx)
    estimate_sun_heading(ctx)
    select_matcher_trajectory(ctx)
    close_loop(ctx)
    ocr_scene_text_anchors(ctx)
    lock_alignment_scale(ctx)
    vggt_gate(ctx)
    match_candidates(ctx)
    summarise_matches(ctx)
    sliding_window_channel(ctx)
    splat_channel(ctx)
    ipm_channel(ctx)
    aerial_channel(ctx)
    tile3d_channel(ctx)
    fuse_channels(ctx)
    embedding_channel(ctx)
    bevsplat_channel(ctx)
    da3_dense_channel(ctx)
    evaluate_ground_truth_streets(ctx)
    anchor_primary_placement(ctx)
    evaluate_waypoints(ctx)
    final_position(ctx)
    finalize_result(ctx)
    return ctx.result
