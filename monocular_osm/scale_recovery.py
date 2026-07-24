"""Recover metric scale / absolute georeferencing from timed anchors.

Monocular VO produces unit-norm steps, so the recovered trajectory has
no metric scale — the downstream Procrustes match is free to shrink it,
which compresses the localized route (verified on the Ulm clip: the
prediction nails the centre but can't reach the ends). Two fixes here,
sharing one piece of machinery — correspondences between *when* an OCR
anchor was seen and *where* it geocodes:

* **Idea 1 — anchor scale lock** (:func:`estimate_anchor_scale`): the
  metric span of the anchors over the VO span of the frames where they
  were seen gives a scale factor, used to set the walk-length prior so
  enumeration isn't compressed.
* **Idea 2 — time-anchored georeferencing** (:func:`fit_similarity_ransac`
  + :func:`apply_transform`): fit a similarity transform VO→world from
  the anchor correspondences and apply it to the whole VO path, so the
  reported position comes from the absolute anchors rather than a
  free-scale shape fit.

Anchors are noisy (a sign is often read from a distance, so its
geocoded POI can sit 100s of metres from the camera), so the fit is
RANSAC with an inlier threshold and a minimum-baseline guard — with too
few well-separated inliers it returns ``None`` and the caller falls back
to the existing behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.transform import SimilarityTransform

from .visual_odometry import trajectory_arc_length


@dataclass
class AnchorFix:
    """A timed pseudo-fix: at ``t_sec`` the camera was near ``world_xy``
    (projected metres), inferred from a geocoded sign of ``confidence``."""
    t_sec: float
    world_xy: np.ndarray   # (2,) projected (x, y)
    confidence: float
    name: str


def vo_positions_at_times(
    traj_xy: np.ndarray, timestamps: np.ndarray, t_secs: np.ndarray
) -> np.ndarray:
    """VO trajectory position at each requested time (nearest frame)."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    idx = np.abs(timestamps[None, :] - np.asarray(t_secs)[:, None]).argmin(axis=1)
    return np.asarray(traj_xy, dtype=np.float64)[idx]


def estimate_anchor_scale(
    vo_pts: np.ndarray, world_pts: np.ndarray, *, min_pairs: int = 1
) -> float | None:
    """Robust metric-per-VO-unit scale from anchor correspondences.

    Uses the median over all anchor *pairs* of (world distance / VO
    distance). The median rejects the pair-wise outliers that a single
    short or noisy baseline would introduce. Returns ``None`` if no pair
    has a usable (non-degenerate) VO separation.
    """
    vo_pts = np.asarray(vo_pts, dtype=np.float64)
    world_pts = np.asarray(world_pts, dtype=np.float64)
    n = len(vo_pts)
    ratios: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            vo_d = np.linalg.norm(vo_pts[i] - vo_pts[j])
            w_d = np.linalg.norm(world_pts[i] - world_pts[j])
            if vo_d > 1e-6 and w_d > 1.0:
                ratios.append(w_d / vo_d)
    if len(ratios) < min_pairs:
        return None
    return float(np.median(ratios))


@dataclass
class AnchorTransform:
    """Result of the RANSAC similarity fit VO→world."""
    transform: SimilarityTransform
    scale: float
    inlier_idx: np.ndarray
    rms_m: float
    world_baseline_m: float   # spatial span of the inlier world points


def fit_similarity_ransac(
    vo_pts: np.ndarray,
    world_pts: np.ndarray,
    *,
    n_iter: int = 500,
    thresh_m: float = 150.0,
    min_inliers: int = 3,
    min_world_baseline_m: float = 250.0,
    rng_seed: int = 0,
) -> AnchorTransform | None:
    """RANSAC similarity transform mapping VO coords to world metres.

    Samples 2 correspondences per iteration (a similarity is exactly
    determined by two point pairs), scores inliers by reprojection
    residual, and refits on the largest inlier set. Returns ``None``
    unless at least ``min_inliers`` agree *and* they span at least
    ``min_world_baseline_m`` — a short baseline yields an unreliable
    scale, so we'd rather decline than emit a bad georeference.
    """
    vo_pts = np.asarray(vo_pts, dtype=np.float64)
    world_pts = np.asarray(world_pts, dtype=np.float64)
    n = len(vo_pts)
    if n < min_inliers:
        return None
    rng = np.random.default_rng(rng_seed)

    best_inliers: np.ndarray | None = None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    # Deterministic sweep over all pairs when few anchors; else sample.
    if len(pairs) <= n_iter:
        samples = pairs
    else:
        samples = [tuple(rng.choice(n, size=2, replace=False)) for _ in range(n_iter)]

    for i, j in samples:
        t = SimilarityTransform()
        if not t.estimate(vo_pts[[i, j]], world_pts[[i, j]]):
            continue
        resid = np.linalg.norm(t(vo_pts) - world_pts, axis=1)
        inliers = np.where(resid <= thresh_m)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers

    if best_inliers is None or len(best_inliers) < min_inliers:
        return None

    # Refit on all inliers.
    t = SimilarityTransform()
    if not t.estimate(vo_pts[best_inliers], world_pts[best_inliers]):
        return None
    resid = np.linalg.norm(t(vo_pts[best_inliers]) - world_pts[best_inliers], axis=1)
    scale = float(np.sqrt(np.linalg.det(t.params[:2, :2])))
    w = world_pts[best_inliers]
    baseline = float(np.linalg.norm(w.max(axis=0) - w.min(axis=0)))
    if baseline < min_world_baseline_m or not np.isfinite(scale) or scale <= 0:
        return None
    return AnchorTransform(
        transform=t, scale=scale, inlier_idx=best_inliers,
        rms_m=float(np.sqrt((resid ** 2).mean())), world_baseline_m=baseline,
    )


def apply_transform(traj_xy: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    """Map a VO trajectory into world (projected-metre) coordinates."""
    return transform(np.asarray(traj_xy, dtype=np.float64))


def scaled_length(traj_xy: np.ndarray, scale: float) -> float:
    """Total VO arc length times scale = estimated metric route length."""
    return float(trajectory_arc_length(traj_xy)[-1] * scale)


def da3_metric_scale(
    da3_xy: np.ndarray, vo_xy: np.ndarray, *, min_vo_len: float = 1.0
) -> float | None:
    """Metric scale (m per VO unit) from a metric DA3 trajectory (idea 4).

    DA3's reconstruction is metric, so the ratio of its trajectory arc
    length to the (unit-norm) VO arc length over the same span is the
    scale. Both paths must describe the same keyframes; the caller is
    responsible for passing comparable sequences. Returns ``None`` if the
    VO path is degenerate. (On the Ulm clip DA3's pose solve is rejected
    upstream by the plausibility guard, so this rarely fires there — it's
    the fallback for clips where DA3 *does* lock.)
    """
    da3_len = float(trajectory_arc_length(np.asarray(da3_xy, dtype=np.float64))[-1])
    vo_len = float(trajectory_arc_length(np.asarray(vo_xy, dtype=np.float64))[-1])
    if vo_len < min_vo_len or da3_len <= 0:
        return None
    return da3_len / vo_len
