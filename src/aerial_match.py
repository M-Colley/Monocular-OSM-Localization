"""Aerial / top-down matching channel.

Two complementary sub-channels are combined here:

1.  **Trajectory-raster IoU** (primary, always available)
    After Procrustes alignment, both the VO trajectory and the OSM walk
    polyline live in the same metric coordinate system.  We rasterize
    both as thick lines (road-width tolerance ~12 m) and compute the
    Jaccard intersection-over-union.  High IoU means the aligned path
    actually *covers* the road — this complements the Procrustes RMS
    (which penalises large deviations) with a coverage signal that is
    robust to end-point mismatch when the walk is longer than the driven
    segment.

2.  **ORB feature matching** (supplemental, requires a top-down image)
    ORB on a rasterised OSM patch vs a rendered top-down image.  In
    practice this channel is weak when the top-down image is a
    photographic IPM stitch vs a schematic OSM render (domain gap →
    ~5 % inlier rate ≈ noise).  It is retained for completeness and to
    show the inter-method comparison; the trajectory IoU should be
    preferred for re-ranking.

    A production system would substitute the OSM schematic with real
    satellite / aerial tiles and use a cross-domain descriptor
    (e.g. SuperPoint + SuperGlue or DINOv2 features).  The matching
    machinery below is identical; only the image source changes.

Combined aerial score (normalised to [0, 1], higher = better match):

    aerial_score = 0.7 * traj_iou + 0.3 * clip(n_inliers / 20, 0, 1)

This score is used by pipeline.py to re-rank candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .osm_data import RoadGraph
from .trajectory_matching import MatchCandidate


@dataclass
class AerialMatchResult:
    candidate_index: int          # index into the input candidates list
    traj_iou: float               # Jaccard IoU of trajectory vs OSM walk rasters
    aerial_score: float           # combined score (higher = better)
    n_orb_matches: int            # raw ORB cross-check matches (0 if no image)
    n_inliers: int                # RANSAC homography inliers (0 if no image)
    inlier_ratio: float           # n_inliers / max(1, n_orb_matches)
    osm_render_path: Path | None  # where the OSM patch was written


# ---------------------------------------------------------------------------
# Trajectory-raster IoU
# ---------------------------------------------------------------------------

def _traj_iou_score(
    aligned_traj_xy: np.ndarray,
    walk_xy: np.ndarray,
    *,
    resolution: int = 512,
    road_width_m: float = 12.0,
) -> float:
    """Jaccard IoU between rasterised aligned-trajectory and OSM-walk.

    Both polylines are drawn as thick lines at the same pixel scale so
    that `road_width_m` metres correspond to the line thickness.  A
    trajectory that follows the road closely produces high overlap;
    one that drifts or belongs to a different part of the city produces
    low overlap even if the overall shapes look similar.

    Parameters
    ----------
    aligned_traj_xy:
        VO trajectory after Procrustes similarity transform into the OSM
        metric coordinate system (``MatchCandidate.aligned_traj_xy``).
    walk_xy:
        The OSM road-graph walk as a polyline in metric coordinates
        (``MatchCandidate.walk_xy``).
    resolution:
        Side length (pixels) of the raster canvas.
    road_width_m:
        Tolerance radius around each path in metres.  This acts like a
        "road width" buffer so that trajectories slightly off-centre
        still count as overlapping.
    """
    if len(aligned_traj_xy) < 2 or len(walk_xy) < 2:
        return 0.0

    all_pts = np.vstack([aligned_traj_xy, walk_xy])
    mn = all_pts.min(axis=0) - 50.0
    mx = all_pts.max(axis=0) + 50.0
    span = (mx - mn).max()
    if span < 1.0:
        return 0.0

    scale = (resolution - 1) / span
    thickness = max(2, int(road_width_m * scale))

    def _rasterise(pts: np.ndarray) -> np.ndarray:
        img = np.zeros((resolution, resolution), dtype=np.uint8)
        px = ((pts - mn) * scale).astype(np.int32)
        px[:, 1] = resolution - 1 - px[:, 1]  # flip y so north-up
        px = np.clip(px, 0, resolution - 1)
        for j in range(len(px) - 1):
            cv2.line(
                img,
                (int(px[j, 0]), int(px[j, 1])),
                (int(px[j + 1, 0]), int(px[j + 1, 1])),
                255,
                thickness,
            )
        return img

    img_traj = _rasterise(aligned_traj_xy)
    img_walk = _rasterise(walk_xy)

    inter = int(np.logical_and(img_traj > 0, img_walk > 0).sum())
    union = int(np.logical_or(img_traj > 0, img_walk > 0).sum())
    return inter / max(1, union)


# ---------------------------------------------------------------------------
# OSM patch rendering
# ---------------------------------------------------------------------------

def render_osm_patch(
    road: RoadGraph,
    center_xy: tuple[float, float],
    *,
    half_extent_m: float = 600.0,
    resolution: int = 1024,
    background: str = "white",
) -> np.ndarray:
    """Rasterise the OSM road graph in a square window centred at `center_xy`.

    Returns a grayscale uint8 image.  Roads are black on white; ORB fires
    on intersection corners and bend points.
    """
    cx, cy = center_xy
    fig, ax = plt.subplots(figsize=(6, 6), dpi=resolution / 6)
    ax.set_xlim(cx - half_extent_m, cx + half_extent_m)
    ax.set_ylim(cy - half_extent_m, cy + half_extent_m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    for poly in road.polylines:
        if poly[:, 0].max() < cx - half_extent_m or poly[:, 0].min() > cx + half_extent_m:
            continue
        if poly[:, 1].max() < cy - half_extent_m or poly[:, 1].min() > cy + half_extent_m:
            continue
        ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.6, antialiased=True)

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    if gray.shape != (resolution, resolution):
        gray = cv2.resize(gray, (resolution, resolution))
    return gray


# ---------------------------------------------------------------------------
# ORB feature matching (supplemental)
# ---------------------------------------------------------------------------

def feature_match_score(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    n_features: int = 2000,
    ransac_threshold: float = 5.0,
) -> tuple[int, int]:
    """Return (n_orb_matches, n_ransac_inliers) for two images."""
    if img_a.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY if img_a.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    if img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY if img_b.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    kp_b, des_b = orb.detectAndCompute(img_b, None)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    if len(matches) < 8:
        return len(matches), 0

    matches = sorted(matches, key=lambda m: m.distance)[:200]
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_threshold)
    if mask is None:
        return len(matches), 0
    return len(matches), int(mask.sum())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def match_splat_against_candidates(
    splat_topdown_rgb: np.ndarray | None,
    road: RoadGraph,
    candidates: list[MatchCandidate],
    *,
    output_dir: Path,
    half_extent_m: float = 600.0,
    resolution: int = 1024,
) -> list[AerialMatchResult]:
    """Score each candidate by trajectory IoU (primary) + ORB inliers (supplemental).

    Parameters
    ----------
    splat_topdown_rgb:
        Top-down image (IPM BEV, sparse splat render, etc.) used for
        ORB matching.  Pass ``None`` to skip the ORB channel entirely —
        the trajectory IoU channel always runs independently.
    road:
        Projected OSM road graph (provides polylines for OSM patch render).
    candidates:
        Trajectory-match candidates from ``match_trajectory``.  Each
        candidate carries ``aligned_traj_xy`` and ``walk_xy`` which are
        the inputs to the IoU scorer.
    output_dir:
        Directory where OSM patch PNGs are written (for inspection).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[AerialMatchResult] = []
    for i, cand in enumerate(candidates):
        # --- Trajectory-raster IoU (primary signal) ---
        iou = _traj_iou_score(cand.aligned_traj_xy, cand.walk_xy)

        # --- ORB feature matching (supplemental) ---
        n_match, n_in = 0, 0
        osm_path: Path | None = None

        if splat_topdown_rgb is not None:
            cxy = cand.walk_xy.mean(axis=0)
            osm_img = render_osm_patch(
                road,
                (float(cxy[0]), float(cxy[1])),
                half_extent_m=half_extent_m,
                resolution=resolution,
            )
            osm_path = output_dir / f"osm_candidate_{i + 1}.png"
            cv2.imwrite(str(osm_path), osm_img)
            n_match, n_in = feature_match_score(splat_topdown_rgb, osm_img)

        # Combined score: IoU dominates; ORB adds a small bonus when available
        # Clip ORB contribution: 20 inliers → 1.0, typical noise is < 10
        orb_norm = min(1.0, n_in / 20.0) if n_in > 0 else 0.0
        aerial_score = 0.7 * iou + 0.3 * orb_norm

        results.append(
            AerialMatchResult(
                candidate_index=i,
                traj_iou=iou,
                aerial_score=aerial_score,
                n_orb_matches=n_match,
                n_inliers=n_in,
                inlier_ratio=n_in / max(1, n_match),
                osm_render_path=osm_path,
            )
        )
    return results
