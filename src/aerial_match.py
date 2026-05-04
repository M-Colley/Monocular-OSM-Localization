"""Optional verification channel: feature-match the splat's top-down
render against rasterized OSM patches around each candidate location.

This is *not* the primary localization signal — that's still the
trajectory/road-graph shape matcher in `trajectory_matching.py`. Two
reasons it's a useful addition anyway:

1.  It demonstrates the conceptually-correct end of the original pipeline
    spec: take a top-down view of the recovered scene and compare it to
    a top-down view of the city.
2.  When the splat captures recognizable structure (e.g., consistent
    building footprints or road-edge transitions), feature matching can
    re-rank ties from the shape matcher.

We render the OSM road graph at each candidate location to a raster
image (matplotlib, no tile fetches → no rate limits, no API keys), run
ORB on both that image and the splat top-down render, do
cross-checked Hamming matching, and score the candidate by the number
of RANSAC homography inliers.

A real Google-Maps / Earth tile match would substitute the matplotlib
render for a satellite tile fetch. The matching machinery is identical;
only the image source changes. We avoid the tile fetch in the PoC to
keep the run hermetic and avoid TOS issues with Google. OSM tiles via
`contextily` would also drop in here.
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
    n_orb_matches: int            # raw ORB cross-check matches
    n_inliers: int                # RANSAC homography inliers
    inlier_ratio: float           # n_inliers / max(1, n_orb_matches)
    osm_render_path: Path | None  # where the OSM patch was written


def render_osm_patch(
    road: RoadGraph,
    center_xy: tuple[float, float],
    *,
    half_extent_m: float = 600.0,
    resolution: int = 1024,
    background: str = "white",
) -> np.ndarray:
    """Rasterize the OSM road graph in a square window centered at `center_xy`.

    The output is a grayscale image (uint8, single channel) intended as
    a stand-in for an aerial / streetmap tile: roads are black on white
    and ORB will fire on intersection corners and bend points. We don't
    color-code by road class; the geometry is the only signal anyway.
    """
    cx, cy = center_xy
    fig, ax = plt.subplots(figsize=(6, 6), dpi=resolution / 6)
    ax.set_xlim(cx - half_extent_m, cx + half_extent_m)
    ax.set_ylim(cy - half_extent_m, cy + half_extent_m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # Only plot polylines that intersect the window — much faster on
    # large city graphs.
    for poly in road.polylines:
        if poly[:, 0].max() < cx - half_extent_m or poly[:, 0].min() > cx + half_extent_m:
            continue
        if poly[:, 1].max() < cy - half_extent_m or poly[:, 1].min() > cy + half_extent_m:
            continue
        ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.6, antialiased=True)

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Pull the canvas as RGBA, convert to gray.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    # Normalize size — matplotlib's actual buffer size is dpi*figsize and
    # may not exactly equal `resolution`.
    if gray.shape != (resolution, resolution):
        gray = cv2.resize(gray, (resolution, resolution))
    return gray


def feature_match_score(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    n_features: int = 2000,
    ransac_threshold: float = 5.0,
) -> tuple[int, int]:
    """Return (n_orb_matches, n_ransac_inliers) for two images.

    Both images are coerced to grayscale. We do crossCheck-matched ORB
    and then run RANSAC for a homography to count inliers. The matching
    domain (synthetic top-down render vs. another raster) is the
    standard ORB use case; we don't get many features when the splat
    cloud is sparse, but the inlier count is a meaningful score.
    """
    if img_a.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.shape[2] == 3 else cv2.cvtColor(img_a, cv2.COLOR_BGRA2GRAY)
    if img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.shape[2] == 3 else cv2.cvtColor(img_b, cv2.COLOR_BGRA2GRAY)

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


def match_splat_against_candidates(
    splat_topdown_rgb: np.ndarray,
    road: RoadGraph,
    candidates: list[MatchCandidate],
    *,
    output_dir: Path,
    half_extent_m: float = 600.0,
    resolution: int = 1024,
) -> list[AerialMatchResult]:
    """For each candidate, render an OSM patch and feature-match against
    the splat top-down. Returns one result per candidate, in input order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[AerialMatchResult] = []
    for i, cand in enumerate(candidates):
        # Center the OSM patch on the centroid of the matched walk so the
        # comparison is over the same region the shape matcher chose.
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
        ratio = n_in / max(1, n_match)
        results.append(
            AerialMatchResult(
                candidate_index=i,
                n_orb_matches=n_match,
                n_inliers=n_in,
                inlier_ratio=ratio,
                osm_render_path=osm_path,
            )
        )
    return results
