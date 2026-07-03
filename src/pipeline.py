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
from .visual_odometry import Trajectory, default_intrinsics, estimate_trajectory


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
    top_k: int = 5
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
    enable_da3: bool = False
    use_da3_trajectory: bool = False
    use_mapanything_trajectory: bool = False
    openvo_trajectory_path: "Path | None" = None
    prefer_openvo_trajectory: bool = True   # default: auto-use a staged OpenVO path, VO fallback
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
    # anchors on London where Nominatim found none. (src/osm_gazetteer.py)
    use_osm_gazetteer: bool = True
    # Here-vs-direction sign classification (Gemma 4): reads each OCR anchor's
    # sign crop and drops *directional* signs (which name places elsewhere and
    # geocode far off-route — the London 'Holborn' failure). GPU, off by
    # default. (src/vlm_anchor.classify_sign_types)
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
    # Blind VPR coarse prior (KartaView + EigenPlaces): a re-rank centre and the
    # anchor-primary placement prior — shape-independent fix for the SELECTION
    # wall. (Gating the OSM graph to the prior's disc was refuted by experiment:
    # 664 -> 1276 m on Ulm — so there is deliberately no gate knob.)
    use_vpr_prior: bool = False
    vpr_search_radius_m: float = 3000.0
    # VPR reference source: "kartaview" (open, tokenless) or "mapillary" (much
    # denser — needs a free MLY_TOKEN env var; validated 3-31 m to route on all
    # GT clips, incl. the ones KartaView could not cover). See mapillary VPR.
    vpr_source: str = "kartaview"
    # Experimental: score candidates against the per-frame VPR track (sequence-
    # median distance at matched arc fractions) instead of the centroid-only
    # distance. Untested on GT — off by default.
    use_vpr_sequence: bool = False
    # License-plate registration-district anchor: read EU plate region prefixes,
    # vote, geocode the modal district -> re-rank penalty (src/plate_anchor.py).
    # (A hard region gate was refuted; the penalty free-radius lives in
    # _PLATE_FREE_RADIUS_M below.)
    use_plate_anchor: bool = False
    # VLM (Gemma 4) district/landmark prior — a coverage fallback when VPR finds
    # no references (src/vlm_anchor.py). Feeds the same anchor-primary path.
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
    scale_lock: bool = False
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
    # drift so the loop closes (src/loop_closure.py). Pair with
    # --use-ipm-scale; closing at a wrong scale doesn't help.
    enable_loop_closure: bool = False
    # Run VGGT (feed-forward, drift-free poses) to GATE enumeration to the
    # area its trajectory selects, then let the precise (loop-closed) VO
    # geometry pick within it (src/vggt_trajectory.py). Needs the vggt
    # package + GPU + ~5 GB weights; degrades to a no-op if unavailable.
    use_vggt_gating: bool = False
    vggt_keyframes: int = 64
    # Refine the shape-matched coarse position with OrienterNet (neural
    # BEV->OSM matching + sequential fusion) — the metric localization head
    # (src/orienternet_localizer.py). ~2 m on KITTI. Needs
    # third_party/OrienterNet + GPU + weights; no-op if unavailable.
    use_orienternet: bool = False
    orienternet_keyframes: int = 10
    orienternet_tile_m: float = 160.0
    vo_workers: int | None = None  # None → auto (min(cpu_count, 12)); 1 → sequential
    # When set, use this local video file directly: no download, `url` is
    # informational only (recorded in the result JSON).
    video_path: Path | None = None


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


# Urban average driving speed including traffic-light stops. The Ulm GT
# track averaged 5.0 m/s point-to-point; the driven path is a bit longer
# than the straight segments between fixes, so 5.5 m/s is a reasonable
# city-wide prior.
_URBAN_AVG_SPEED_MPS = 5.5

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

# Size of the geometric candidate pool kept for the calibrated
# multi-hypothesis output. The matcher already scores ~500 walks in
# stage 2, so returning 50 instead of top_k is free; we collapse them
# into distinct location hypotheses (src/hypotheses.py). The heavy
# channels (BevSplat, sliding window) and the headline pick still run on
# the top_k slice — only the hypotheses shortlist uses the wider pool.
_HYP_POOL = 50


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


def _auto_estimated_length_m(duration_sec: float) -> float:
    """Route-length prior from the analyzed duration at urban speed.

    Clamped to [500, 12000]: below 500 m walk enumeration degenerates
    (every micro-walk matches anything), above 12 km the walk depth cap
    truncates walks anyway.
    """
    return float(np.clip(duration_sec * _URBAN_AVG_SPEED_MPS, 500.0, 12000.0))


def _length_sane(
    candidate_m: float,
    duration_prior_m: float,
    user_length_m: float | None = None,
) -> bool:
    """A recovered length is trustworthy if it's within 2x of the
    duration-based estimate (the robust reference) — or, when the user
    supplied an explicit ``--estimated-length-m``, within 2x of THAT.

    Without the user gate, a correct recovered length on a fast
    (highway) clip is rejected against the conservative 5.5 m/s prior
    even though the user already told us the true extent.
    """
    refs = [duration_prior_m]
    if user_length_m is not None and user_length_m > 0:
        refs.append(float(user_length_m))
    return any(0.5 * r <= candidate_m <= 2.0 * r for r in refs)


def _find_vo_cache(candidates: list, n_frames: int):
    """First VO cache file whose frame count matches, or ``None``.

    Every candidate — including the canonical key — must PASS the shape
    check; a stale canonical cache from a different stream (fps change on
    re-download) must be recomputed, not loaded. Probes are opened with a
    context manager so no npz handle stays open (an open handle blocks
    deleting/renaming the file on Windows).
    """
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with np.load(candidate) as z_probe:
                shape_ok = z_probe["valid"].shape[0] == n_frames
        except Exception:
            continue
        if shape_ok:
            return candidate
    return None


def _match_timestamps(frame_timestamps, n_rows: int) -> np.ndarray:
    """Timestamp axis aligned to the matcher trajectory's rows.

    A staged trajectory (OpenVO / DA3 / MapAnything) has FEWER rows than
    extracted frames (Ulm: 1260 poses vs 4196 frames), so indexing it
    with per-frame timestamps is out of bounds. Spread the clip's frame
    time span uniformly over the pose rows instead.
    """
    ts = np.asarray(frame_timestamps, dtype=np.float64)
    if len(ts) == 0:
        return np.zeros(n_rows, dtype=np.float64)
    if n_rows == len(ts):
        return ts
    return np.linspace(ts[0], ts[-1], n_rows)


def _remap_frame_pair_to_poses(
    pair, n_frames: int, n_poses: int
):
    """Map a loop-closure (i, j) frame-index pair into pose-index space.

    ``detect_end_to_start_loop`` indexes the extracted frames, but the
    drift redistribution acts on the matcher trajectory, which has a
    different row count whenever a staged (OpenVO/DA3/MapAnything) path
    is in use. Scale proportionally; return ``None`` when the remapped
    pair is degenerate (so the caller skips instead of warping poses).
    """
    if pair is None:
        return None
    i, j = int(pair[0]), int(pair[1])
    if n_frames == n_poses:
        return (i, j) if 0 <= i < j < n_poses else None
    if n_frames < 2 or n_poses < 2:
        return None
    scale = (n_poses - 1) / (n_frames - 1)
    ip = int(round(i * scale))
    jp = int(round(j * scale))
    if not (0 <= ip < jp < n_poses):
        return None
    return (ip, jp)


# VPR re-rank penalty: metres of slack before a candidate is penalized for
# distance to the VPR prior. The prior's own error bar is ~91-190 m on Ulm,
# so candidates within it are indistinguishable from the truth — penalizing
# them flips near-ties away from a correct shape pick.
_VPR_FREE_RADIUS_M = 150.0

# Plate-district penalty free radius (see the fusion comment for rationale).
_PLATE_FREE_RADIUS_M = 1500.0


def _vpr_distance_penalty(d_m: float) -> float:
    """Consensus penalty for a candidate ``d_m`` metres from the VPR prior:
    free inside the prior's own error bar, then 15 rank-points per km."""
    return 15.0 * max(0.0, (d_m - _VPR_FREE_RADIUS_M) / 1000.0)


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

        from .vlm_anchor import classify_sign_types
    except Exception:
        return ocr_anchors, []
    _upscale = None
    if use_super_res:
        try:
            from .scene_text import _upscale_sharpen as _upscale
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
        cap.release()
    except Exception:
        return ocr_anchors, []
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


def _vpr_sequence_median_m(
    traj_xy: np.ndarray,
    track_frame_idx,
    track_xy: np.ndarray,
    sims,
    n_frames: int,
) -> float | None:
    """Similarity-weighted median distance (m) between the candidate route
    position at each VPR-matched frame's arc fraction and that frame's
    retrieved reference location.

    Frame index -> arc fraction assumes roughly constant speed (the
    aligned trajectory is arc-uniform, the frames time-uniform); good
    enough for the median statistic this feeds. A candidate centred
    correctly but flipped/rotated has the same centroid distance yet a
    much worse sequence distance — the ambiguity class the centroid
    penalty cannot rank.
    """
    traj = np.asarray(traj_xy, dtype=np.float64)
    track_xy = np.asarray(track_xy, dtype=np.float64)
    if len(traj) < 2 or len(track_xy) == 0:
        return None
    frac = np.asarray(track_frame_idx, dtype=np.float64) / max(n_frames - 1, 1)
    rows = np.clip(np.round(frac * (len(traj) - 1)).astype(int), 0, len(traj) - 1)
    d = np.linalg.norm(traj[rows] - track_xy, axis=1)
    w = np.clip(np.asarray(sims, dtype=np.float64), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(d)
    order = np.argsort(d)
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, 0.5 * cw[-1]))
    return float(d[order][min(k, len(d) - 1)])


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


def _final_position_reports(
    candidates,
    road,
    *,
    matches,
    ranking: str,
    world_route_latlon=None,
    anchored_cand=None,
    anchor_origin: str | None = None,
    prior_latlon=None,
):
    """Build (headline, matcher) position reports per the output contract.

    ``matcher`` is ALWAYS the raw matcher-pick report (old ``position``
    schema, ``source: "matcher"``). When an anchor-primary candidate
    exists its report becomes the HEADLINE (``source:
    "anchor_primary_vpr"|"anchor_primary_vlm"`` + ``prior_latlon``) —
    the anchored answer is the project's proven accuracy win and must be
    what ``result["position"]`` carries. Falls back to the matcher
    report when the anchored one can't be built.
    """
    matcher_position = build_position_report(
        candidates[0],
        road,
        matches=matches,
        ranking=ranking,
        world_route_latlon=world_route_latlon,
    )
    if matcher_position is not None:
        matcher_position["source"] = "matcher"
    headline = matcher_position
    if anchored_cand is not None:
        origin = anchor_origin or "vpr"
        anchored = build_position_report(
            anchored_cand,
            road,
            matches=matches,
            ranking=f"anchored({origin})",
        )
        if anchored is not None:
            anchored["source"] = f"anchor_primary_{origin}"
            if prior_latlon is not None:
                anchored["prior_latlon"] = [
                    float(prior_latlon[0]), float(prior_latlon[1]),
                ]
            headline = anchored
    return headline, matcher_position


def _orienternet_refine(cfg, frames, cand, road, position, result) -> None:
    """Refine the shape-matched coarse position with OrienterNet (BEV->OSM).

    Samples keyframes across the analyzed window, maps each to a coarse
    lat/lon from the shape-matched route, and runs OrienterNet sequential
    fusion. Updates ``position`` in place (keeps the coarse estimate under
    ``coarse_*``) and records the refined-vs-GT error when GT is present.
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
        refined = refine_route(kf, prior_ll, None, tile_m=cfg.orienternet_tile_m)
        if refined is None:
            print("      -> OrienterNet unavailable; keeping shape-match position")
            return
        lat0, lon0 = float(refined[0][0]), float(refined[0][1])
        position["coarse_latitude"] = position["latitude"]
        position["coarse_longitude"] = position["longitude"]
        position["latitude"] = round(lat0, 6)
        position["longitude"] = round(lon0, 6)
        position["ranking"] = position.get("ranking", "") + "+orienternet"
        position["google_maps_url"] = google_maps_url(lat0, lon0)
        position["orienternet_route_latlon"] = [
            [round(float(a), 6), round(float(b), 6)] for a, b in refined]
        result["orienternet"] = {"keyframes": len(kf), "tile_m": cfg.orienternet_tile_m}
        print(f"      -> refined start: {lat0:.6f}, {lon0:.6f}")
        if cfg.ground_truth_waypoints:
            from .evaluator import _segment_to_polyline_distance, load_gt_waypoints
            from .position import latlon_to_xy
            wp_xy = latlon_to_xy(load_gt_waypoints(cfg.ground_truth_waypoints), road.crs)
            rt_xy = latlon_to_xy(refined, road.crs)
            d0 = float(np.linalg.norm(rt_xy[0] - wp_xy[0]))
            dm = float(np.mean([_segment_to_polyline_distance(w_, rt_xy) for w_ in wp_xy]))
            position["orienternet_gt_start_error_m"] = round(d0, 1)
            position["orienternet_gt_mean_route_error_m"] = round(dm, 1)
            print(f"      -> OrienterNet vs GT: start {d0:.1f} m, mean route {dm:.1f} m")
    except Exception as e:
        print(f"      -> OrienterNet refine failed ({e})")


def _da3_trajectory_plausible(xy: np.ndarray) -> bool:
    """Reject DA3 camera paths that look like pose-estimation failures.

    DA3 needs visually-overlapping keyframes; on sparse keyframes (long
    clips) its solver can return near-random poses. A real driven path
    almost never reverses heading by >120° between consecutive keyframe
    segments, while a failed solve zigzags constantly — on the Ulm clip
    a failed 48-keyframe solve had ~40% reversals and silently produced
    a 2.3 km localization error. Threshold 0.15 leaves room for genuine
    U-turns and noisy stationary segments.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3:
        return False
    seg = np.diff(xy, axis=0)
    norms = np.linalg.norm(seg, axis=1)
    moving = norms > 1e-9
    if moving.sum() < 2:
        return False
    d = seg[moving] / norms[moving][:, None]
    cos_between = (d[:-1] * d[1:]).sum(axis=1)
    reversal_ratio = float((cos_between < -0.5).mean())
    return reversal_ratio <= 0.15


def _resolve_input_video(cfg: PipelineConfig) -> Path:
    """Return the path of the video to analyze.

    Resolution order:
    1. ``cfg.video_path`` — a local file supplied by the user. Used as-is
       (never copied; multi-GB dashcam files shouldn't be duplicated).
    2. ``cfg.skip_download`` — a previously-downloaded ``input.*`` in
       ``cfg.data_dir``.
    3. Otherwise download ``cfg.url`` with yt-dlp.
    """
    if cfg.video_path is not None:
        video_path = Path(cfg.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"local video not found: {video_path}")
        print(f"[1/5] Using local video: {video_path}")
        return video_path
    if cfg.skip_download:
        existing = list(cfg.data_dir.glob("input.*"))
        if not existing:
            raise FileNotFoundError("--skip-download but no cached video in data/")
        # Deterministic pick when several cached inputs exist (e.g. a
        # re-download that changed container left input.webm beside
        # input.mp4): prefer a fixed extension order, then the newest.
        # Bare glob order is filesystem-dependent and can silently
        # analyze the older/other file.
        ext_rank = {".mp4": 0, ".mkv": 1, ".webm": 2}
        existing.sort(key=lambda p: (
            ext_rank.get(p.suffix.lower(), len(ext_rank)),
            -p.stat().st_mtime,
            p.name,
        ))
        if len(existing) > 1:
            print(f"      -> {len(existing)} cached inputs "
                  f"({', '.join(p.name for p in existing)}); picking {existing[0].name}")
        print(f"[1/5] Using cached video: {existing[0]}")
        return existing[0]
    print(f"[1/5] Downloading video from {cfg.url}")
    video_path = download_video(cfg.url, cfg.data_dir)
    print(f"      -> {video_path}")
    return video_path


def _fetch_road_graph(
    city: str, cache_path: Path,
    around: tuple[float, float, float] | None = None,
) -> RoadGraph:
    """Fetch the OSM graph with a user-actionable error on geocode failure.

    osmnx raises a grab-bag of exceptions (InsufficientResponseError,
    ValueError, network errors) whose messages don't tell a CLI user
    what to fix. Re-raise as a ValueError that names the city and the
    expected format, keeping the original as __cause__.
    """
    try:
        return fetch_city_graph(city, cache_path=cache_path, around=around)
    except Exception as e:
        raise ValueError(
            f"Could not fetch an OSM road graph for {city!r}. Check the "
            f"spelling and use the form 'City, Country' (e.g. 'Ulm, "
            f"Germany'). Original error: {e}"
        ) from e


def run_pipeline(cfg: PipelineConfig) -> dict:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire the input video (local file / cache / download).
    video_path = _resolve_input_video(cfg)

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

    # 2b. Route-length prior for OSM walk enumeration. The
    # duration-based estimate is the *stable sanity reference* that every
    # (noisier) scale source — IPM flow (2c), DA3 (4b), anchors (4c) — is
    # gated against. Gating each source against this fixed baseline
    # rather than against the running prior avoids one unreliable source
    # opening the gate for another (e.g. IPM flow lowering the prior so a
    # wrong anchor scale then passes).
    analyzed_sec = (
        frames.timestamps[-1] - frames.timestamps[0]
        if len(frames.timestamps) >= 2 else 0.0
    )
    duration_prior_m = _auto_estimated_length_m(analyzed_sec)
    if cfg.estimated_length_m is not None:
        estimated_length_m = float(cfg.estimated_length_m)
    else:
        estimated_length_m = duration_prior_m
        print(f"      -> estimated route length: {estimated_length_m:.0f} m "
              f"(auto: {analyzed_sec:.0f} s at urban avg speed; "
              f"override with --estimated-length-m)")

    def _length_sane_here(candidate_m: float) -> bool:
        """A recovered length is trustworthy only if it's within 2x of the
        duration-based estimate — or of an explicit user-provided
        ``--estimated-length-m`` (see module-level ``_length_sane``)."""
        return _length_sane(candidate_m, duration_prior_m,
                            user_length_m=cfg.estimated_length_m)

    # 2c. Idea 3: ground-plane optical-flow metric scale (opt-in). Sets
    # the length prior from road-feature motion, sanity-gated against the
    # duration estimate. Off by default — unreliable without real camera
    # calibration.
    if cfg.use_ipm_scale:
        print("[2c] Ground-plane optical-flow scale")
        try:
            from .speed_scale import estimate_route_length_from_flow
            h_img, w_img = frames.frames[0].shape[:2]
            K_flow = default_intrinsics(w_img, h_img)
            # Subsample to ~3 fps so the flow pass is quick.
            step = max(1, int(round(frames.fps / 3.0)) // max(1, cfg.frame_stride))
            sub = frames.frames[::step] if step > 1 else frames.frames
            flow_len, motions = estimate_route_length_from_flow(
                sub, K_flow, camera_height_m=cfg.ipm_scale_camera_height_m,
                pitch_deg=cfg.ipm_scale_pitch_deg, fps=frames.fps,
                frame_stride=cfg.frame_stride * step,
            )
            # IPM flow is a *physical measurement* (optical flow + camera
            # height), so trust it on a looser bound than the generic 2x
            # sanity gate: the duration prior uses a conservative 5.5 m/s,
            # so a faster drive legitimately measures ~2x it (KITTI 0033:
            # true 1705 m vs 876 m prior = 1.95x), which the 2x gate would
            # wrongly reject. Accept within [0.3x, 3x] of the prior.
            if flow_len > 0 and 0.3 * duration_prior_m <= flow_len <= 3.0 * duration_prior_m:
                print(f"      -> flow length {flow_len:.0f} m adopted (was "
                      f"{estimated_length_m:.0f} m)")
                estimated_length_m = float(np.clip(flow_len, 500.0, 12000.0))
                result_ipm_scale = {"status": "ok", "length_m": round(flow_len, 1)}
            else:
                print(f"      -> flow length {flow_len:.0f} m rejected (sanity vs "
                      f"prior {estimated_length_m:.0f} m); keeping prior")
                result_ipm_scale = {"status": "rejected", "length_m": round(flow_len, 1)}
        except Exception as e:
            print(f"      -> IPM-scale failed: {e}")
            result_ipm_scale = {"status": "error", "error": str(e)}
    else:
        result_ipm_scale = None

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
        f"_s{cfg.frame_stride}_f{cfg.max_frames or 'auto'}.npz"
    )
    cache_v2_legacy = cfg.data_dir / (
        f"trajectory_v2_{cfg.vo_start_sec:.0f}-{cfg.vo_end_sec or 'end'}"
        f"_s{cfg.frame_stride}.npz"
    )
    # Older runs predate per-submission slug folders — fall back to a
    # flat-layout cache at the data-dir parent if nothing is in the slug.
    cache_v2_flat = cfg.data_dir.parent / cache_v2_legacy.name
    # Any cache for this (segment, stride) with a different max_frames
    # suffix is still usable when the frame count matches (the shape
    # check below verifies) — e.g. an old `_f4200` cache after the
    # default became uncapped.
    sibling_caches = sorted(cfg.data_dir.glob(
        f"trajectory_v2_{cfg.vo_start_sec:.0f}-{cfg.vo_end_sec or 'end'}"
        f"_s{cfg.frame_stride}_f*.npz"
    ))

    # Every candidate — including the canonical key — must PASS the
    # frame-count shape check before it may be loaded. A stale canonical
    # cache (same key, different stream fps after a re-download) used to
    # slip through and misassociate poses with frames downstream.
    vo_cache = _find_vo_cache(
        [cache_v2_with_frames, cache_v2_legacy, cache_v2_flat, *sibling_caches],
        len(frames.frames),
    )
    legacy_used: Path | None = (
        vo_cache if (vo_cache is not None and vo_cache != cache_v2_with_frames)
        else None
    )

    if vo_cache is not None:
        msg = f"[3/5] Loading cached trajectory: {vo_cache}"
        if legacy_used is not None:
            msg += "  (legacy cache key)"
        print(msg)
        with np.load(vo_cache) as z:
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

    # 4a. Optional blind VPR coarse prior (KartaView + EigenPlaces): retrieve the
    # frames against GPS-tagged street photos to locate the clip INDEPENDENTLY of
    # trajectory shape — the fix for the SELECTION wall. ~53 m prior on Ulm 4K
    # (src/kartaview_vpr.py). Used as a re-rank centre + the anchor-primary
    # placement prior; gating the OSM graph to its disc was refuted (664->1276 m).
    osm_around = cfg.osm_around
    vpr_center: tuple | None = None
    vpr_track = None
    anchor_origin: str | None = None   # which channel produced vpr_center: vpr|vlm
    # VPR runs whenever enabled — it re-ranks + drives anchor-primary, so it is
    # NOT conditioned on osm_around (mega-cities pass --osm-around AND want VPR).
    if cfg.use_vpr_prior:
        _vpr_src = cfg.vpr_source
        print(f"[4a] VPR coarse prior ({_vpr_src.capitalize()} + MegaLoc)")
        try:
            import os as _os

            import osmnx as ox
            from .kartaview_vpr import kartaview_vpr_track, _robust_center
            # Prefer an explicit --osm-around centre (robust to Nominatim
            # geocode failures / ambiguous small-town names); else geocode.
            center = ((osm_around[0], osm_around[1]) if osm_around
                      else ox.geocode(cfg.city))
            # Device: only request CUDA when it is actually available;
            # None lets the module fall back to CPU inference (slow but
            # working) instead of dying inside torch.hub and masquerading
            # as "VPR unavailable" on CPU-only machines.
            try:
                import torch as _torch
                _vpr_device = "cuda" if _torch.cuda.is_available() else None
            except Exception:
                _vpr_device = None
            _vpr_token = _os.environ.get("MLY_TOKEN") if _vpr_src == "mapillary" else None
            if _vpr_src == "mapillary" and not _vpr_token:
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
            if osm_around is not None:
                _vpr_radius = min(_vpr_radius, max(800.0, float(osm_around[2])))
            tr = kartaview_vpr_track(frames.frames, center,
                                     radius_m=_vpr_radius,
                                     cache_dir=str(cfg.data_dir / _vpr_src),
                                     device=_vpr_device,
                                     source=_vpr_src, token=_vpr_token)
            if tr is not None:
                vpr_track = tr
                v_idx, v_ll, v_sims = tr
                # Robust centre (conf-threshold + weighted geometric median +
                # spatial-MAD outlier rejection) for the re-rank channel + guard;
                # the per-frame track drives the anchor-primary placement below.
                # VPR is route-accurate (~91 m to the GT route on Ulm) — better
                # than the matcher's own pick. Re-rank, don't gate (gating its
                # disc hurt end-to-end: 664 -> 1276 m).
                vpr_center = _robust_center(v_ll, v_sims)
                anchor_origin = "vpr"
                print(f"      -> VPR prior {vpr_center[0]:.5f}, {vpr_center[1]:.5f} "
                      f"({len(v_idx)}-frame track); re-rank + anchor-primary fit")
            else:
                print("      -> VPR unavailable; keeping full-city graph")
        except Exception as e:
            print(f"      -> VPR prior failed ({e}); keeping full-city graph")

    # [4a'] License-plate district anchor: an independent, absolute coarse prior.
    # Reads EU plate region prefixes across the clip, votes on the registration
    # district, geocodes it. Used as a RE-RANK channel in the consensus fusion
    # (not a hard gate — gating is slow on the resulting graph and can't separate
    # the true candidate from wrong ones inside any reasonable radius). The
    # district centroid pulls the pick toward the right area and kills gross
    # km-scale errors. 0.4 km from GT on Ulm; strong where plates are readable
    # (>=720p) and the fleet is local.
    result_plate_anchor: dict | None = None
    plate_center: tuple | None = None
    if cfg.use_plate_anchor:
        print("[4a'] License-plate district anchor (ALPR + Kfz-code vote)")
        try:
            from .plate_anchor import plate_district_anchor
            pa = plate_district_anchor(str(video_path))
            if pa is not None:
                plate_center = (pa.lat, pa.lon)
                result_plate_anchor = {
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

    # [4a''] VLM (Gemma 4) district/landmark prior — a coverage FALLBACK: only
    # runs when VPR produced no centre (no nearby street-photo references). Reads
    # frames -> infers district + reads names -> geocodes -> feeds the same
    # anchor-primary path via vpr_center. Independent failure modes from VPR.
    result_vlm_anchor: dict | None = None
    if cfg.use_vlm_anchor and vpr_center is None:
        print("[4a''] VLM district/landmark prior (Gemma 4)")
        try:
            from .text_anchor import default_geocode_fn
            from .vlm_anchor import vlm_district_anchor
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
                vpr_center = (va.lat, va.lon)
                anchor_origin = "vlm"
                result_vlm_anchor = {"label": va.label, "lat": va.lat, "lon": va.lon,
                                     "streets": va.street_votes,
                                     "districts": va.district_votes, "texts": va.text_votes,
                                     "applied_as_prior": True}
                print(f"      -> VLM prior {va.lat:.5f}, {va.lon:.5f} ('{va.label}'); "
                      f"feeding anchor-primary")
            else:
                print("      -> VLM produced no geocodable district/name")
        except Exception as e:
            print(f"      -> VLM anchor failed ({e})")

    # 4. OSM road graph for the city.
    if osm_around is not None:
        la, lo, rad = osm_around
        print(f"[4/5] Fetching OSM driving graph around ({la:.4f}, {lo:.4f}) "
              f"r={rad:.0f} m for {cfg.city!r}")
        cache_path = cfg.data_dir / (
            f"{cfg.city.replace(',', '').replace(' ', '_')}"
            f"_around_{la:.4f}_{lo:.4f}_{int(rad)}.graphml")
    else:
        print(f"[4/5] Fetching OSM driving graph for {cfg.city!r}")
        cache_path = cfg.data_dir / f"{cfg.city.replace(',', '').replace(' ', '_')}.graphml"
    road = _fetch_road_graph(cfg.city, cache_path, around=osm_around)
    print(f"      -> {road.graph.number_of_nodes()} nodes, "
          f"{road.graph.number_of_edges()} edges, CRS={road.crs}")

    # 4b. Optional: absolute heading from the sun. Activates only when the clip
    # carries a usable capture time (container metadata or a burned-in clock);
    # otherwise a graceful no-op. When available it pins the matcher's free
    # rotation DOF (src/sun_heading.py). Computed + reported here.
    if cfg.use_sun_heading:
        try:
            from .sun_heading import estimate_heading
            cen = osm_around[:2] if osm_around else None
            if cen is None:
                import osmnx as ox
                cen = ox.geocode(cfg.city)
            sh_times = list(np.linspace(cfg.vo_start_sec,
                                        cfg.vo_end_sec or (cfg.vo_start_sec + 420.0), 30))
            sh = estimate_heading(str(video_path), cen, sh_times)
            result_sun_heading = sh
            if sh and sh.get("available"):
                print(f"[4b] Sun heading: {sh['median_heading']:.0f} deg via {sh['source']} "
                      f"({sh['n_used']} frames, conf {sh['confidence']:.2f})")
            else:
                print(f"[4b] Sun heading: not available "
                      f"({(sh or {}).get('reason', 'module missing')})")
        except Exception as e:
            print(f"[4b] Sun heading failed ({e})")
            result_sun_heading = {"available": False, "reason": str(e)}
    else:
        result_sun_heading = None

    # 4b. Optional: use Depth Anything 3's globally-consistent camera path
    # as the trajectory the shape matcher consumes. Monocular VO drifts
    # over long clips (the true Olgastraße route's bearing correlation was
    # only 0.16 vs 0.29–0.32 for *wrong* parallel streets — pure drift),
    # which lets several parallel streets fit equally well. DA3 is metric
    # and multi-frame-consistent, so its trajectory has far less drift.
    # We keep the VO `traj` for the splat/IPM renders (those are tied to
    # the per-frame VO poses) and only swap the matcher's input here.
    match_xz = traj.xz
    if cfg.use_mapanything_trajectory:
        print("[4b'] MapAnything submap-stitched trajectory (matcher input)")
        from .mapanything_trajectory import mapanything_trajectory_xy
        end_s = cfg.vo_end_sec if cfg.vo_end_sec else (cfg.vo_start_sec + 420.0)
        ma = mapanything_trajectory_xy(str(video_path), cfg.vo_start_sec, end_s)
        if ma is not None and len(ma[0]) >= 2 and bool(np.isfinite(ma[0]).all()):
            match_xz = ma[0]
            _plot_xy(match_xz, cfg.output_dir / "trajectory_mapanything.png",
                     "MapAnything submap-stitched trajectory (matcher input)")
            print(f"      -> using MapAnything trajectory: {len(match_xz)} stitched poses")
        else:
            print("      -> MapAnything unavailable/invalid; keeping VO path")
    # OpenVO is the default matcher trajectory when one is staged for this clip
    # (explicit --openvo-trajectory, else <data_dir>/openvo_trajectory.txt). It is
    # intrinsic-free metric dashcam VO with lower drift than our frame-to-frame VO
    # (196 vs 241 m global-fit RMS on Ulm). Poses are KITTI 3x4 rows; the top-down
    # ground projection is (tx, tz) = cols 3,11. Falls back to VO if none present.
    ov_path = cfg.openvo_trajectory_path
    if ov_path is None and cfg.prefer_openvo_trajectory:
        cand = cfg.data_dir / "openvo_trajectory.txt"
        if cand.exists():
            ov_path = cand
    if ov_path is not None:
        print(f"[4b''] OpenVO metric trajectory (matcher input): {ov_path}")
        try:
            P = np.loadtxt(str(ov_path))
            ov = P[:, [3, 11]].astype(float)
            if len(ov) >= 2 and bool(np.isfinite(ov).all()):
                match_xz = ov
                _plot_xy(ov, cfg.output_dir / "trajectory_openvo.png",
                         "OpenVO metric trajectory (matcher input)")
                print(f"      -> using OpenVO trajectory: {len(ov)} poses")
            else:
                print("      -> OpenVO trajectory invalid; keeping VO path")
        except Exception as e:
            print(f"      -> OpenVO trajectory load failed ({e}); keeping VO path")
    da3_rec = None
    result_scale_recovery: dict | None = None   # set by DA3 (4b) or anchors (4c)
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
            da3_valid = len(da3_xy) >= 2 and bool(np.isfinite(da3_xy).all())
            if da3_valid and not _da3_trajectory_plausible(da3_xy):
                # A garbage path scores walks all over the city and is
                # strictly worse than drifty-but-coherent VO.
                print("      -> DA3 trajectory implausible (heading reverses "
                      "constantly — pose solve likely failed on sparse "
                      "keyframes); falling back to VO path")
                _plot_xy(
                    da3_xy, cfg.output_dir / "trajectory_da3_rejected.png",
                    "DA3 trajectory (REJECTED: implausible)",
                )
                da3_valid = False
            if da3_valid:
                match_xz = da3_xy
                _plot_xy(
                    da3_xy, cfg.output_dir / "trajectory_da3.png",
                    "DA3 globally-consistent trajectory (matcher input)",
                )
                print(f"      -> using DA3 trajectory: {len(da3_xy)} keyframe poses")
                result_traj_source = "da3"
                # Idea 4: DA3 is metric, so its arc length IS the route
                # length — adopt it as the prior when it sanity-checks
                # against the duration estimate.
                from .visual_odometry import trajectory_arc_length as _arclen
                da3_len = float(_arclen(da3_xy)[-1])
                if _length_sane_here(da3_len):
                    print(f"      -> DA3 metric length {da3_len:.0f} m adopted as "
                          f"route prior (was {estimated_length_m:.0f} m)")
                    estimated_length_m = float(np.clip(da3_len, 500.0, 12000.0))
                    result_scale_recovery = {
                        "status": "ok", "source": "da3",
                        "estimated_length_m": round(estimated_length_m, 1),
                    }
                else:
                    print(f"      -> DA3 metric length {da3_len:.0f} m rejected "
                          f"(sanity vs prior {estimated_length_m:.0f} m)")
            else:
                if len(da3_xy) < 2 or not bool(np.isfinite(da3_xy).all()):
                    print("      -> DA3 trajectory invalid; falling back to VO path")
                result_traj_source = "vo"
        except Exception as e:
            print(f"      -> DA3 trajectory failed ({e}); falling back to VO path")
            da3_rec = None
            result_traj_source = "vo"
    else:
        result_traj_source = "vo"

    # Timestamp axis aligned to the matcher trajectory. The staged
    # OpenVO/DA3/MapAnything paths have FEWER rows than extracted frames
    # (Ulm: 1260 poses vs 4196 frames), so anything that maps a time to
    # a match_xz row must use this axis, not frames.timestamps.
    match_ts = _match_timestamps(frames.timestamps, len(match_xz))

    # 4b2. Optional: loop-closure drift correction. On a route that returns
    # near its start, monocular VO leaves a large end-start gap (pure
    # drift) that warps the whole shape — the dominant geometric error.
    # If the last frames verifiably revisit the first, redistribute the
    # gap back along the trajectory. Pair with a real metric scale
    # (--use-ipm-scale): closing the loop at a wrong scale doesn't help.
    result_loop_closure: dict | None = None
    if cfg.enable_loop_closure:
        from .loop_closure import detect_end_to_start_loop, redistribute_drift
        from .visual_odometry import trajectory_arc_length as _arclen2
        print("[4b2] Loop-closure drift correction")
        try:
            frame_pair = detect_end_to_start_loop(frames.frames)
            # The detector indexes the extracted FRAMES; the drift fix acts
            # on match_xz POSES, which is a different (shorter) axis when a
            # staged trajectory is in use — remap proportionally.
            pair = _remap_frame_pair_to_poses(
                frame_pair, len(frames.frames), len(match_xz))
            if pair is not None:
                i, j = pair
                gap = float(np.linalg.norm(match_xz[j] - match_xz[i]))
                arclen = float(_arclen2(match_xz)[-1])
                match_xz = redistribute_drift(match_xz, i, j)
                result_loop_closure = {"closed": True, "i": int(i), "j": int(j),
                                       "frame_i": int(frame_pair[0]),
                                       "frame_j": int(frame_pair[1]),
                                       "gap_units": round(gap, 2)}
                print(f"      -> loop detected (frames {frame_pair[0]}↔"
                      f"{frame_pair[1]} -> poses {i}↔{j}); closed "
                      f"{gap:.1f}-unit gap ({100 * gap / max(arclen, 1e-6):.0f}% of arc)")
            else:
                result_loop_closure = {"closed": False}
                print("      -> no verifiable loop (end does not revisit start)")
        except Exception as e:
            print(f"      -> loop-closure failed ({e}); using VO path unchanged")

    # 4c. Optional: OCR scene-text → geocoded POI anchors. Computed before
    # the match so the anchors can *seed* enumeration (add walk roots near
    # each anchor), guaranteeing the anchored area is in the candidate
    # pool even when drift would exclude it — the fix for the enumeration
    # failure that pure re-ranking can't address.
    ocr_anchors: list = []
    street_anchors: list = []
    anchor_xy = np.zeros((0, 2))
    anchor_seed: list | None = None
    ocr_anchor_error: str | None = None
    anchor_transform = None          # similarity VO->world (idea 2), if fit
    anchor_world_route = None        # full VO path georeferenced to world xy
    if cfg.enable_ocr_anchor:
        ocr_source = cfg.ocr_video_path or video_path
        tag = " (4K OCR source)" if cfg.ocr_video_path else ""
        print(f"[4c] OCR scene-text → anchors{tag}")
        try:
            from .scene_text import extract_scene_text
            from .text_anchor import (
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
            street_anchors = match_text_to_streets(
                detections, road, min_confidence=cfg.ocr_min_confidence,
            )
            for s in street_anchors:
                print(f"        street: OCR {s.ocr_text!r} -> {s.name!r} "
                      f"(conf {s.confidence:.2f}, ratio {s.match_ratio:.2f}, "
                      f"{len(s.node_ids)} nodes)")
            # (b) POI/landmark names → geocoded points (works at 720p too).
            ocr_anchors = geocode_texts(
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
                    existing=ocr_anchors,
                    min_confidence=cfg.ocr_min_confidence,
                )
                if gaz_anchors:
                    print(f"      -> {len(gaz_anchors)} local-gazetteer anchor(s) "
                          f"(offline, beyond Nominatim)")
                ocr_anchors = ocr_anchors + gaz_anchors
            # OCR-quality: directional/destination signs name places ELSEWHERE,
            # so they geocode FAR from the actual route (the London 'Holborn'
            # failure: a direction sign ~1.5 km off). When a route-accurate VPR
            # prior exists, reject OCR anchors > 1.5 km from it — this drops the
            # misleading direction signs while keeping local street/POI plates,
            # leaving only anchors usable in the anchor-primary fit.
            if vpr_center is not None and ocr_anchors:
                import math as _om
                _vlat, _vlon = vpr_center
                _kept = []
                for a in ocr_anchors:
                    _d = _om.hypot((a.lat - _vlat) * 111320.0,
                                   (a.lon - _vlon) * 111320.0 * _om.cos(_om.radians(_vlat)))
                    if _d <= 1500.0:
                        _kept.append(a)
                    else:
                        print(f"        poi REJECTED ({_d/1000:.1f} km from VPR — "
                              f"likely a direction sign): {a.name!r}")
                ocr_anchors = _kept
            for a in ocr_anchors:
                print(f"        poi:    {a.name!r} @ ({a.lat:.5f},{a.lon:.5f}) "
                      f"conf={a.confidence:.2f}")

            # Here-vs-direction sign typing (Gemma 4): a directional sign names
            # a place ELSEWHERE and geocodes off-route (the London 'Holborn'
            # failure). Drop such anchors before they seed/score. GPU + opt-in.
            if cfg.classify_signs and ocr_anchors:
                print("      -> classifying sign types (here vs direction)")
                try:
                    ocr_anchors, _dropped = _drop_direction_anchors(
                        ocr_anchors, detections, ocr_source,
                        start_sec=cfg.vo_start_sec,
                        use_super_res=cfg.use_ocr_super_res)
                    for _nm in _dropped:
                        print(f"        poi REJECTED (direction sign): {_nm!r}")
                except Exception as e:
                    print(f"      -> sign classification failed ({e})")

            # --- Scale recovery / georeferencing (ideas 1 & 2) ----------
            # Use the *pre-cluster-filter* anchors here: scale needs
            # temporal+spatial SPREAD, and RANSAC rejects the noise that
            # the cluster filter would otherwise have removed. Each anchor
            # is a timed pseudo-fix (seen at t_sec, geocoded to lat/lon).
            if cfg.enable_scale_recovery and len(ocr_anchors) >= 3:
                from .scale_recovery import (
                    apply_transform,
                    fit_similarity_ransac,
                    scaled_length,
                    vo_positions_at_times,
                )
                from .text_anchor import anchors_to_xy as _a2xy
                t_secs = np.array([a.t_sec for a in ocr_anchors])
                world_pts = _a2xy(ocr_anchors, road.crs)
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
                    and _length_sane_here(recovered_len)
                )
                if fit is not None and sane:
                    anchor_transform = fit.transform
                    duration_prior = estimated_length_m
                    estimated_length_m = float(np.clip(recovered_len, 500.0, 12000.0))
                    anchor_world_route = apply_transform(traj.xz, fit.transform)
                    result_scale_recovery = {
                        "status": "ok",
                        "scale_m_per_vo_unit": round(fit.scale, 4),
                        "n_inliers": int(len(fit.inlier_idx)),
                        "n_anchors": int(len(ocr_anchors)),
                        "rms_m": round(fit.rms_m, 1),
                        "world_baseline_m": round(fit.world_baseline_m, 1),
                        "estimated_length_m": round(estimated_length_m, 1),
                    }
                    print(f"      -> scale recovery OK: scale={fit.scale:.3f} m/unit, "
                          f"{len(fit.inlier_idx)}/{len(ocr_anchors)} inliers, "
                          f"baseline={fit.world_baseline_m:.0f} m, rms={fit.rms_m:.0f} m "
                          f"→ route length prior {estimated_length_m:.0f} m")
                elif fit is not None and not sane:
                    result_scale_recovery = {
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
                    result_scale_recovery = {
                        "status": "declined",
                        "n_anchors": int(len(ocr_anchors)),
                        "reason": "too few well-separated inlier anchors "
                                  "(short baseline / noisy geocodes)",
                    }
                    print("      -> scale recovery declined (anchors too few/clustered "
                          "for a reliable fit); keeping default length prior")

            # Reject spatial-outlier anchors (a far-off shop sign or a
            # fuzzy-matched wrong street) — keep only the corroborated
            # cluster, so one bad anchor can't hijack the gate.
            n_poi0, n_st0 = len(ocr_anchors), len(street_anchors)
            ocr_anchors, street_anchors = cluster_filter_anchors(
                ocr_anchors, street_anchors, road,
            )
            dropped = (n_poi0 - len(ocr_anchors)) + (n_st0 - len(street_anchors))
            if dropped:
                print(f"        cluster filter: kept {len(ocr_anchors)} POI + "
                      f"{len(street_anchors)} street, dropped {dropped} outlier(s)")
            # Merge both anchor kinds for scoring + seeding.
            anchor_xy = np.vstack([
                x for x in (anchors_to_xy(ocr_anchors, road.crs),
                            street_anchor_xy(street_anchors, road))
                if len(x) > 0
            ]) if (ocr_anchors or street_anchors) else np.zeros((0, 2))
            anchor_seed = list(dict.fromkeys(
                anchor_seed_nodes(road, anchors_to_xy(ocr_anchors, road.crs))
                + street_anchor_seed_nodes(street_anchors)
            ))
            print(f"      -> {len(street_anchors)} street + {len(ocr_anchors)} POI "
                  f"anchor(s); {len(anchor_seed)} enumeration seed node(s)")
        except Exception as e:
            print(f"      -> OCR-anchor channel failed: {e}")
            ocr_anchor_error = str(e)

    # 5. Match. When confident OCR anchors exist, gate enumeration to the
    # anchor vicinity (a trustworthy absolute prior beats the drift-prone
    # global shape for *where* to look) and let shape rank within it. Fall
    # back to the full-city scan if the gate yields nothing.
    # Scale lock: pin alignment scale to the metric extent so the route
    # can't compress (the eastern-tail fix). scale = metres per VO unit.
    locked_scale = None
    if cfg.scale_lock:
        from .visual_odometry import trajectory_arc_length
        vo_arclen = float(trajectory_arc_length(match_xz)[-1])
        if vo_arclen > 1e-6:
            locked_scale = estimated_length_m / vo_arclen
            print(f"      -> scale-locked alignment: {locked_scale:.4f} m/VO-unit "
                  f"(extent {estimated_length_m:.0f} m / {vo_arclen:.1f} VO units)")

    # VGGT gating: a drift-free feed-forward trajectory selects the AREA
    # (its bearing signature matches the true street, breaking the
    # parallel-street ambiguity that drifted VO can't), then the precise
    # loop-closed VO geometry ranks within it.
    vggt_seed: list = []
    result_vggt: dict | None = None
    if cfg.use_vggt_gating:
        print("[5a] VGGT trajectory gating (drift-free area selection)")
        try:
            from .vggt_trajectory import vggt_camera_trajectory, vggt_seed_nodes
            vxy = vggt_camera_trajectory(frames.frames, n_keyframes=cfg.vggt_keyframes)
            if vxy is not None:
                vggt_seed = vggt_seed_nodes(
                    vxy, road, estimated_length_m=estimated_length_m)
                result_vggt = {"poses": len(vxy), "seed_nodes": len(vggt_seed)}
                print(f"      -> VGGT: {len(vxy)} poses -> {len(vggt_seed)} seed start-node(s)")
            else:
                print("      -> VGGT unavailable (no package/GPU/weights); skipping")
        except Exception as e:
            print(f"      -> VGGT gating failed ({e}); skipping")

    seed = list(dict.fromkeys((anchor_seed or []) + vggt_seed))
    gated = bool((cfg.enable_ocr_anchor and anchor_seed) or vggt_seed)
    anchor_gated = gated  # downstream reporting reuses this name
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
    candidates = match_trajectory(
        match_xz,
        road,
        final_top_k=_pool_k,
        sample_every=cfg.sample_every,
        estimated_length_m=estimated_length_m,
        extra_start_nodes=seed or None,
        restrict_to_start_nodes=gated,
        locked_scale=locked_scale,
    )
    if gated and not candidates:
        print("      -> gated match found nothing; falling back to full scan")
        candidates = match_trajectory(
            match_xz, road, final_top_k=_pool_k, sample_every=cfg.sample_every,
            estimated_length_m=estimated_length_m, locked_scale=locked_scale,
        )
    # Keep the wider geometric pool for the calibrated multi-hypothesis
    # output, but run the rest of the pipeline (heavy channels, headline
    # pick, GT eval) on the top_k slice exactly as before.
    geom_pool = list(candidates)
    candidates = candidates[:cfg.top_k]

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
            "estimated_length_m": estimated_length_m,
            "trajectory_source": result_traj_source,
            "matches": [
                candidate_geographic_summary(c, road.graph) for c in candidates
            ],
        }
        if cfg.enable_ocr_anchor:
            from .text_anchor import anchors_to_json, street_anchors_to_json
            result["ocr_anchors"] = anchors_to_json(ocr_anchors)
            result["ocr_street_anchors"] = street_anchors_to_json(street_anchors)
            if ocr_anchor_error:
                result["ocr_anchor_error"] = ocr_anchor_error

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
                from .full_splat import render_full_splat_to_file
                hq_path = cfg.output_dir / "splat_topdown_hq.png"
                splat_img_rgb = render_full_splat_to_file(
                    splat_pts, splat_cols, hq_path,
                    scale=cfg.full_splat_scale,
                    opacity=cfg.full_splat_opacity,
                    progress=True,
                )
                print(f"      -> wrote {hq_path} (anisotropic Gaussian render)")
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

    # 8. Aerial matching channel + consensus fusion.
    # The aerial ORB/IoU scoring is ONE optional channel; the fusion itself
    # (turn + OCR-anchor + plate + VPR + sun penalties, consensus reorder)
    # runs whenever candidates exist. It used to sit entirely inside the
    # aerial gate, so --no-aerial silently disabled every proven
    # disambiguator (plate guard, VPR re-rank, OCR-anchor rank).
    ranking_mode = "shape"
    if candidates:
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
                m["traj_iou"] = ar.traj_iou
                m["traj_coverage"] = ar.traj_coverage
                m["aerial_score"] = ar.aerial_score
                m["aerial_orb_matches"] = ar.n_orb_matches
                m["aerial_inliers"] = ar.n_inliers
                m["aerial_inlier_ratio"] = ar.inlier_ratio
                m["aerial_rank"] = aerial_rank[i]
            W_AERIAL = 0.5
        else:
            # Neutral channel: uniform rank, zero weight — the same no-op
            # pattern the anchor channel uses when it has no data.
            aerial_rank = {i: i + 1 for i in range(len(candidates))}
            W_AERIAL = 0.0

        for i in range(len(candidates)):
            result["matches"][i]["shape_rank"] = i + 1

        # Turn-sequence channel: a drift-robust topological descriptor.
        # Cheap (signature compare over candidate polylines, no city
        # scan), so it always runs here. Lower distance = better; rank
        # ascending. See turn_matching for why it survives the VO drift
        # that degrades the dense shape/bearing comparison.
        from .turn_matching import score_candidates_by_turns
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

        # OCR-anchor channel: distance from each candidate walk to the
        # nearest geocoded POI anchor. The one *absolute* signal — when
        # present it's the most trustworthy, so it earns a strong fusion
        # weight. Inactive (all-inf → uniform rank, weight 0) when the
        # OCR-anchor channel is disabled or found no in-city anchors.
        from .text_anchor import score_candidates_by_anchors
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
        W_SHAPE, W_SLIDING, W_TURN = 1.0, 1.0, 0.0
        if not have_anchors:
            W_ANCHOR = 0.0
        elif anchor_gated:
            W_ANCHOR = 0.25
        else:
            W_ANCHOR = 2.0

        # License-plate district anchor (shape-INDEPENDENT absolute prior): rank
        # candidates by distance from their start to the voted district centroid.
        # Unlike noisy OCR shop-sign anchors this is a reliable region cue, so it
        # gets a strong weight to pull the pick toward the right district and kill
        # gross km-scale errors. No-op (rank = shape) when no district was found.
        if plate_center is not None:
            import math as _math
            from .position import xy_to_latlon as _xy2ll
            _plat, _plon = plate_center
            _cl = _math.cos(_math.radians(_plat))
            _pd = []
            for _c in candidates:
                _ll = _xy2ll(np.asarray(_c.aligned_traj_xy[:1], dtype=np.float64), road.crs)
                _la, _lo = float(_ll[0][0]), float(_ll[0][1])
                _pd.append(_math.hypot((_la - _plat) * 111320.0,
                                       (_lo - _plon) * 111320.0 * _cl))
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
            import math as _vm
            from .position import latlon_to_xy as _vll2xy, xy_to_latlon as _vxy2ll
            _vlat, _vlon = vpr_center
            _vcl = _vm.cos(_vm.radians(_vlat))
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
                    _d = _vm.hypot((float(_ll[0][0]) - _vlat) * 111320.0,
                                   (float(_ll[0][1]) - _vlon) * 111320.0 * _vcl)
                vpr_penalty[i] = _vpr_distance_penalty(_d)
        else:
            vpr_penalty = {i: 0.0 for i in range(len(candidates))}

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

        def _sliding_rank(i: int) -> int:
            return int(result["matches"][i].get("sliding_window_rank", i + 1))

        def _fused(i: int) -> float:
            return (
                W_SHAPE * (i + 1)
                + W_SLIDING * _sliding_rank(i)
                + W_TURN * turn_rank[i]
                + W_ANCHOR * anchor_rank[i]
                + W_AERIAL * aerial_rank[i]
                + plate_penalty[i]
                + vpr_penalty[i]
                + sun_penalty[i]
            )

        for i in range(len(candidates)):
            result["matches"][i]["consensus_score"] = _fused(i)

        consensus_order = sorted(
            range(len(candidates)),
            key=lambda i: (_fused(i), i),
        )
        consensus_idx = consensus_order[0]
        print()
        print(f"        Consensus pick: candidate #{consensus_idx + 1} "
              f"(shape #{consensus_idx + 1}, "
              f"sliding #{_sliding_rank(consensus_idx)}, "
              f"turn #{turn_rank[consensus_idx]}, "
              f"anchor #{anchor_rank[consensus_idx]}, "
              f"plate #{plate_rank[consensus_idx]}, "
              f"aerial #{aerial_rank[consensus_idx]}, "
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
        if W_AERIAL > 0:
            active_channels.append("aerial")
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
            ranking_mode = "consensus(" + "+".join(active_channels) + ")"
        else:
            ranking_mode = "shape"
        result["ranking_mode"] = ranking_mode
        print(f"        Final #1 after consensus re-rank ({ranking_mode}): "
              f"{', '.join(result['matches'][0]['street_names'][:3]) or '(unnamed)'}")
        # Mirror the reorder in candidates for downstream (GT eval).
        candidates = [candidates[i] for i in consensus_order]

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

    # 10a*. ANCHOR-PRIMARY position. A route-accurate absolute prior (VPR,
    # ~141-200 m on Ulm) beats ANY matcher candidate (pick 664 m, best-in-pool
    # 521 m), and the matcher's candidate GENERATION — not its ranking — is the
    # wall (gating its disc hurts; re-ranking can't beat the prior). So place the
    # best trajectory AT the prior: translate the top candidate so its centroid
    # sits on the prior, and report THAT anchored start as the primary answer.
    # Orientation/shape come from the matcher; absolute position from the prior.
    anchored_cand = None
    if vpr_center is not None and candidates:
        # Place the top candidate's trajectory so its CENTROID sits on the robust
        # VPR centre, keeping the matcher's (clean) orientation. This beat fitting
        # to the per-frame VPR track (top-1 matches are too noisy: 334 vs 227 m
        # mean route error on Ulm), so the single robust centre wins.
        import dataclasses
        from .position import latlon_to_xy as _ll2xy, xy_to_latlon as _xy2ll2
        _traj = np.asarray(candidates[0].aligned_traj_xy, dtype=np.float64)
        _vpr_xy = _ll2xy(np.asarray([[vpr_center[0], vpr_center[1]]], float), road.crs)[0]
        _shift = _vpr_xy - _traj.mean(axis=0)
        anchored_cand = dataclasses.replace(candidates[0], aligned_traj_xy=_traj + _shift)
        _astart = _xy2ll2((_traj[:1] + _shift), road.crs)[0]
        # The source must name the channel that produced the prior: a VLM
        # district centroid is a far coarser prior than a MegaLoc VPR fit,
        # and per-channel accuracy studies need to tell them apart.
        _anchor_src = "vlm_geocode" if anchor_origin == "vlm" else "vpr_centroid"
        result["anchored_position"] = {
            "lat": float(_astart[0]), "lon": float(_astart[1]), "source": _anchor_src,
            "prior_latlon": [float(vpr_center[0]), float(vpr_center[1])],
        }
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
                world_route_latlon = None
        if world_route_latlon is None and anchor_world_route is not None:
            from .position import xy_to_latlon as _xy2ll
            try:
                world_route_latlon = _xy2ll(anchor_world_route, road.crs)
                pos_ranking = ranking_mode + "+anchor-georef"
            except Exception:
                world_route_latlon = None

        # Output contract: result["position"] is the HEADLINE answer —
        # the ANCHORED route when anchor-primary fired (VPR/VLM prior),
        # else the matcher pick. result["matcher_position"] always
        # carries the raw matcher-pick report (old schema). The anchored
        # answer used to live only in the anchored_position side field
        # while position/printed summary carried the matcher pick,
        # silently discarding the project's main accuracy win.
        position, matcher_position = _final_position_reports(
            candidates,
            road,
            matches=result["matches"],
            ranking=pos_ranking,
            world_route_latlon=world_route_latlon,
            anchored_cand=anchored_cand,
            anchor_origin=anchor_origin,
            prior_latlon=vpr_center,
        )
        if matcher_position is not None:
            if world_route_latlon is not None and cfg.ground_truth_waypoints:
                # GT error of the georeferenced route (the candidate-based
                # waypoint_evals don't describe this route).
                try:
                    from .evaluator import _segment_to_polyline_distance
                    from .position import latlon_to_xy
                    wps = load_gt_waypoints(cfg.ground_truth_waypoints)
                    wp_xy = latlon_to_xy(wps, road.crs)
                    rt_xy = latlon_to_xy(world_route_latlon, road.crs)
                    d0 = float(np.linalg.norm(rt_xy[0] - wp_xy[0]))
                    dm = float(np.mean([
                        _segment_to_polyline_distance(w, rt_xy) for w in wp_xy]))
                    matcher_position["gt_start_error_m"] = round(d0, 1)
                    matcher_position["gt_mean_route_error_m"] = round(dm, 1)
                except Exception as e:
                    print(f"      -> georef GT eval failed: {e}")
            elif waypoint_evals:
                matcher_position["gt_start_error_m"] = round(
                    waypoint_evals[0].start_error_m, 1
                )
                matcher_position["gt_mean_route_error_m"] = round(
                    waypoint_evals[0].mean_route_error_m, 1
                )
        # GT errors of the headline: the anchored route's own when the
        # anchor fired, else it shares the matcher pick's numbers.
        if (position is not None and position is not matcher_position
                and anchored_eval is not None):
            position["gt_start_error_m"] = round(anchored_eval.start_error_m, 1)
            position["gt_mean_route_error_m"] = round(
                anchored_eval.mean_route_error_m, 1)
        if matcher_position is not None:
            # Written unconditionally (not inside the headline block) so
            # the contract's "always present" holds even if the headline
            # report fails to build.
            result["matcher_position"] = matcher_position
        if position is not None:
            # Calibrated multi-hypothesis output. Collapse the candidate
            # pool (final-ranked top_k + geometric tail) into distinct
            # location hypotheses and attach a confidence derived from
            # their spatial AGREEMENT — honest where the winner's own
            # shape score is not (see src/hypotheses.py).
            ranked_pool = candidates + geom_pool[cfg.top_k:]
            hyps = distinct_hypotheses(ranked_pool, road, top_n=5)
            if hyps:
                spatial_conf = hypothesis_confidence(ranked_pool, hyps)
                position["spatial_confidence"] = spatial_conf
                position["hypotheses"] = [
                    {
                        "rank": h.rank,
                        "latitude": round(h.lat, 6),
                        "longitude": round(h.lon, 6),
                        "street_names": h.street_names,
                        "support": h.support,
                        "google_maps_url": google_maps_url(h.lat, h.lon),
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
                    road, position, result)

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
        result["vpr_prior"] = {
            "lat": float(vpr_center[0]), "lon": float(vpr_center[1]),
            "origin": anchor_origin,
            "track_frames": (int(len(vpr_track[0]))
                             if vpr_track is not None else None),
        }

    out_json = cfg.output_dir / "result.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"      -> wrote {out_json}")

    return result
