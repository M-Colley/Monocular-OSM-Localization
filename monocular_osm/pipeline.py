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
    _Tile3DInactive,
    aerial_channel,
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


# Weight of the BevSplat appearance rank in the consensus fusion. On the
# Ulm GT run the channel decisively down-ranked the wrong geometric
# winner (rank 9/10) while top-scoring the best-corridor candidate, so
# it earns a strong weight — but below the geometric channels (1.0)
# until it's validated on more than one GT clip.
_W_BEV = 0.75

# Rank cap for BevSplat fusion. Appearance may only reorder candidates
# already in the top-N by geometric (base) consensus; candidates ranked
# worse than this by geometry keep their order below the reordered group.
# This is a guardrail: on the 10-min Ulm run, unconstrained fusion let a
# geometrically-implausible candidate (2.3 km from GT) win on appearance
# alone. Capping means appearance refines the geometric shortlist but
# can't promote a long-shot to #1.
_BEV_FUSION_CAP = 5

def _fuse_bev_rank(
    base_scores: list[float],
    bev_ranks: list[int],
    w_bev: float = _W_BEV,
    cap: int = _BEV_FUSION_CAP,
) -> list[int]:
    """New candidate order after folding the BevSplat appearance rank
    into the geometric (base) consensus scores (lower = better).

    Only the top-``cap`` candidates by base score are reorderable by
    appearance; the rest keep their base order appended after the
    reordered group. This prevents appearance from promoting a
    geometrically-implausible candidate to the top (the 10-min Ulm
    backfire). ``cap >= len`` reproduces unconstrained fusion. Ties are
    broken by base order, which already encodes the geometric ranking.
    """
    n = len(base_scores)
    by_base = sorted(range(n), key=lambda i: (base_scores[i], i))
    cap = max(1, min(cap, n))
    shortlist, tail = by_base[:cap], by_base[cap:]
    reordered = sorted(
        shortlist,
        key=lambda i: (base_scores[i] + w_bev * bev_ranks[i], base_scores[i], i),
    )
    return reordered + tail


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
        from .orienternet_localizer import refine_route
        from .position import xy_to_latlon
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
        from .position import latlon_to_xy as _refine_ll2xy
        refined_xy = _refine_ll2xy(np.asarray(refined, dtype=np.float64), road.crs)
        if vpr_track is not None:
            try:
                from .position import latlon_to_xy
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
            from .evaluator import _segment_to_polyline_distance, load_gt_waypoints
            from .position import latlon_to_xy
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


def _summarize_channels(status: dict, cfg: "PipelineConfig", *, produced: dict,
                        fusion: dict | None) -> dict:
    """Fill in a status for every channel that did not record a failure.

    ``produced`` maps channel -> (enabled, produced_output). Fusion rank
    channels are "ok" when their effective weight was non-zero.
    """
    for name, (enabled, made) in produced.items():
        if name in status:
            continue
        if not enabled:
            _record_channel(status, name, "disabled")
        else:
            _record_channel(status, name, "ok" if made else "inactive")
    eff = (fusion or {}).get("weights_effective") or {}
    for name, w in eff.items():
        key = f"fusion_{name}"
        if key not in status:
            _record_channel(status, key, "ok" if w else "inactive", weight=w)
    return status




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

    # --- not yet extracted: the remaining stages still run inline on locals ---
    channel_status = ctx.channel_status
    result = ctx.result
    frames = ctx.frames
    result_ipm_scale = ctx.result_ipm_scale
    traj = ctx.traj
    vo_cache = ctx.vo_cache
    frame_seed_source = ctx.frame_seed_source
    vpr_center = ctx.vpr_center
    vpr_track = ctx.vpr_track
    _c2f_info = ctx.c2f_info
    anchor_origin = ctx.anchor_origin
    vpr_source_used = ctx.vpr_source_used
    result_plate_anchor = ctx.result_plate_anchor
    plate_center = ctx.plate_center
    result_vlm_anchor = ctx.result_vlm_anchor
    road = ctx.road
    cache_path = ctx.cache_path
    result_sun_heading = ctx.result_sun_heading
    match_xz = ctx.match_xz
    match_ts = ctx.match_ts
    da3_rec = ctx.da3_rec
    result_scale_recovery = ctx.result_scale_recovery
    result_loop_closure = ctx.result_loop_closure
    ocr_anchors = ctx.ocr_anchors
    street_anchors = ctx.street_anchors
    anchor_world_route = ctx.anchor_world_route
    locked_scale = ctx.locked_scale
    result_vggt = ctx.result_vggt
    candidates = ctx.candidates
    geom_pool = ctx.geom_pool
    splat_img_rgb = ctx.splat_img_rgb
    ipm_canvas = ctx.ipm_canvas
    ranking_mode = ctx.ranking_mode
    t3_mesh_ref = ctx.t3_mesh_ref
    t3_samples_ref = ctx.t3_samples_ref
    t3_K_ref = ctx.t3_K_ref
    t3_skyline_ref = ctx.t3_skyline_ref

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

            # Fuse the appearance rank into the consensus. Only when
            # scoring succeeded broadly — with many failed tiles the
            # surviving ranks are not comparable across candidates.
            if n_ok >= max(2, int(0.8 * len(bev_results))) and len(candidates) >= 2:
                base = [
                    float(result["matches"][i].get("consensus_score", i + 1.0))
                    for i in range(len(bev_results))
                ]
                ranks = [bev_rank[i] for i in range(len(bev_results))]
                order = _fuse_bev_rank(base, ranks, cap=cfg.bev_fusion_cap)
                for new_pos, i in enumerate(order):
                    m = result["matches"][i]
                    m["consensus_score"] = base[i] + _W_BEV * ranks[i]
                    m["final_rank"] = new_pos + 1
                result["matches"] = [result["matches"][i] for i in order]
                candidates = [candidates[i] for i in order]
                ranking_mode = ranking_mode + "+bev"
                result["ranking_mode"] = ranking_mode
                result["bev_splat"]["fused"] = True
                result["bev_splat"]["weight"] = _W_BEV
                print(f"        BevSplat fused into consensus (w={_W_BEV}); new #1: "
                      f"{', '.join(result['matches'][0]['street_names'][:3]) or '(unnamed)'}")
            else:
                result["bev_splat"]["fused"] = False

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
                    _record_channel(channel_status, 'train_3dgs', "failed", error=e)
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
            _record_channel(channel_status, 'da3', "failed", error=e)
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

    # 10a*. ANCHOR-PRIMARY position. A route-accurate absolute prior (VPR,
    # ~141-200 m on Ulm) beats ANY matcher candidate (pick 664 m, best-in-pool
    # 521 m), and the matcher's candidate GENERATION — not its ranking — is the
    # wall (gating its disc hurts; re-ranking can't beat the prior). So place the
    # best trajectory AT the prior: translate the top candidate so its centroid
    # sits on the prior, and report THAT anchored start as the primary answer.
    # Orientation/shape come from the matcher; absolute position from the prior.
    anchored_cand = None
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
        from .position import latlon_to_xy as _ll2xy, xy_to_latlon as _xy2ll2
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
        anchored_cand = dataclasses.replace(candidates[_best_rank],
                                            aligned_traj_xy=_placed)
        # Optional metric placement refinement (--tile3d-refine): sharpen the
        # VPR-pinned route by aligning its rendered skyline to the video
        # (LoD-Loc-style silhouette pose optimization). No-op unless the
        # tile3d mesh/samples are available and a transform beats identity.
        if cfg.tile3d_refine and t3_mesh_ref is not None:
            try:
                from .tile3d_match import refine_placement_skyline
                _rk, _rwh = t3_K_ref
                _placed_ref, _rinfo = refine_placement_skyline(
                    t3_mesh_ref, t3_samples_ref, _placed, _rk, _rwh,
                    skyline_fn=t3_skyline_ref)
                if _rinfo.get("applied"):
                    _placed = _placed_ref
                    anchored_cand = dataclasses.replace(
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

    # 10b. Optional: metric evaluation against GPS waypoint ground truth.
    waypoint_evals = None
    anchored_eval = None
    if cfg.ground_truth_waypoints and candidates:
        print(f"[10b] Evaluating against GT waypoints: {cfg.ground_truth_waypoints}")
        try:
            waypoints = load_gt_waypoints(cfg.ground_truth_waypoints)
            waypoint_evals = evaluate_candidates_against_waypoints(
                candidates, road, waypoints
            )
        except (OSError, ValueError) as e:
            print(f"      -> waypoint evaluation failed: {e}")
            result["ground_truth_waypoints_error"] = str(e)
        if waypoint_evals:
            wp_best_rank = best_rank_for_waypoints(waypoint_evals)
            print(f"      -> best candidate by mean route error: #{wp_best_rank}")
            for i, ev in enumerate(waypoint_evals):
                m = result["matches"][i]
                m["gt_waypoint_start_error_m"] = round(ev.start_error_m, 1)
                m["gt_waypoint_mean_route_error_m"] = round(ev.mean_route_error_m, 1)
                m["gt_waypoint_max_route_error_m"] = round(ev.max_route_error_m, 1)
                print(f"        #{i+1}  start={ev.start_error_m:7.1f} m  "
                      f"mean={ev.mean_route_error_m:7.1f} m  "
                      f"max={ev.max_route_error_m:7.1f} m")
            result["ground_truth_waypoints"] = {
                "file": str(cfg.ground_truth_waypoints),
                "n_waypoints": int(len(waypoints)),
                "best_mean_error_rank": wp_best_rank,
                "final_pick_start_error_m": round(waypoint_evals[0].start_error_m, 1),
                "final_pick_mean_route_error_m": round(
                    waypoint_evals[0].mean_route_error_m, 1
                ),
            }
            if anchored_cand is not None:
                aev = evaluate_candidates_against_waypoints(
                    [anchored_cand], road, waypoints)[0]
                anchored_eval = aev
                result["anchored_position"]["gt_start_error_m"] = round(
                    aev.start_error_m, 1)
                result["anchored_position"]["gt_mean_route_error_m"] = round(
                    aev.mean_route_error_m, 1)
                print(f"      -> ANCHOR-PRIMARY (headline) vs GT: start "
                      f"{aev.start_error_m:.1f} m, mean route "
                      f"{aev.mean_route_error_m:.1f} m   (matcher pick: start "
                      f"{waypoint_evals[0].start_error_m:.1f} m, mean route "
                      f"{waypoint_evals[0].mean_route_error_m:.1f} m)")

    # 11. Final position report — the single answer to "where is this
    # video?". candidates[0] is consensus-best when the aerial channel
    # ran (the reorder above keeps candidates and result["matches"]
    # parallel), shape-best otherwise.
    if candidates:
        for i, cand in enumerate(candidates):
            latlon = candidate_center_latlon(cand, road)
            if latlon is not None:
                result["matches"][i]["center_latlon"] = [
                    round(latlon[0], 6), round(latlon[1], 6),
                ]
        # Idea 2: when the anchor similarity fit succeeded, report the
        # *georeferenced* VO path (absolute, anchor-derived) instead of
        # the free-scale shape alignment. Street names still come from
        # the matched candidate.
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
                from .position import xy_to_latlon as _xy2ll
                from .scale_recovery import vo_positions_at_times
                from .text_anchor import anchors_to_xy as _a2xy
                from .trajectory_matching import anchor_pinned_route
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
            from .position import xy_to_latlon as _xy2ll
            try:
                world_route_latlon = _xy2ll(anchor_world_route, road.crs)
                world_route_t_sec = _match_timestamps(
                    frames.timestamps, len(anchor_world_route))
                world_route_source = "anchor_georef"
                pos_ranking = ranking_mode + "+anchor-georef"
            except Exception:
                world_route_latlon = None

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
                from .evaluator import _segment_to_polyline_distance
                from .position import latlon_to_xy
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
                        from .position import xy_to_latlon as _hxy2ll
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

    if result_sun_heading is not None:
        # store a JSON-safe view (drop the per-frame numpy array)
        result["sun_heading"] = {k: v for k, v in result_sun_heading.items()
                                 if k != "headings"}

    # Diagnostics are attached UNCONDITIONALLY: failed runs (no candidates,
    # unusable CRS) are exactly the ones these are needed to debug, and
    # they used to vanish whenever the position report didn't build.
    if _c2f_info is not None:
        result["vpr_coarse_to_fine"] = _c2f_info
    if result_scale_recovery is not None:
        result["scale_recovery"] = result_scale_recovery
    if result_loop_closure is not None:
        result["loop_closure"] = result_loop_closure
    if result_vggt is not None:
        result["vggt_gating"] = result_vggt
    if result_ipm_scale is not None:
        result["ipm_scale"] = result_ipm_scale

    # Anchor-channel summaries: a run's only record of these expensive
    # computations must not be stdout.
    if result_plate_anchor is not None:
        result_plate_anchor.setdefault("applied", False)
        result["plate_anchor"] = result_plate_anchor
    elif cfg.use_plate_anchor:
        result["plate_anchor"] = {"applied": False,
                                  "status": "no_confident_district"}
    if result_vlm_anchor is not None:
        result["vlm_anchor"] = result_vlm_anchor
    elif cfg.use_vlm_anchor and vpr_center is None:
        result["vlm_anchor"] = {"applied_as_prior": False,
                                "status": "no_geocodable_prior"}
    if vpr_center is not None:
        from .kartaview_vpr import backbone_info as _vpr_backbone
        result["vpr_prior"] = {
            "lat": float(vpr_center[0]), "lon": float(vpr_center[1]),
            "origin": anchor_origin,
            # The source actually queried: "mapillary" falls back to
            # "kartaview" without a token, and results differ by kilometres.
            "source": vpr_source_used,
            "backbone": _vpr_backbone() if anchor_origin == "vpr" else None,
            "track_frames": (int(len(vpr_track[0]))
                             if vpr_track is not None else None),
        }

    result["channels"] = _summarize_channels(
        channel_status, cfg,
        produced={
            "vpr": (cfg.use_vpr_prior, vpr_center is not None),
            "plate": (cfg.use_plate_anchor, plate_center is not None),
            "vlm": (cfg.use_vlm_anchor, result_vlm_anchor is not None),
            "sun": (cfg.use_sun_heading, bool((result_sun_heading or {}).get("available"))),
            "ocr_anchor": (cfg.enable_ocr_anchor, bool(ocr_anchors or street_anchors)),
            "loop_closure": (cfg.enable_loop_closure, result_loop_closure is not None),
            "coarse_from_frames": (cfg.coarse_from_frames and cfg.osm_around is None,
                                   frame_seed_source is not None),
            "orienternet": (cfg.use_orienternet,
                            bool((result.get("orienternet") or {}).get("applied"))),
        },
        fusion=result.get("fusion"),
    )

    # Provenance: commit, full config, library versions, whether credentials
    # were set (never their values). Without it a result cannot be traced to
    # the code and settings that produced it.
    from .provenance import collect_provenance
    # The OSM snapshot matters too: OSM is edited continuously and the graph
    # is cached, so record when (and with what) this run's graph was fetched.
    _g = road.graph.graph
    result["provenance"] = collect_provenance(
        cfg, extra={"vo_algo_version": VO_ALGO_VERSION,
                    "vo_cache": str(vo_cache) if vo_cache is not None else None,
                    "osm_graph": {
                        "cache": str(cache_path),
                        "created_date": _g.get("created_date"),
                        "created_with": _g.get("created_with"),
                        "nodes": road.graph.number_of_nodes(),
                        "edges": road.graph.number_of_edges(),
                    }},
    )

    out_json = cfg.output_dir / "result.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"      -> wrote {out_json}")

    return result
