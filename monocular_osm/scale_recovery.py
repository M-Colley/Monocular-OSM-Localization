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

import warnings
from dataclasses import dataclass

import numpy as np
from skimage.measure import ransac
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


def _pair_coverage_trials(n_pairs: int, miss_prob: float = 1e-9) -> int:
    """Uniform 2-subset draws needed so that *every* one of ``n_pairs``
    pairs is drawn at least once with probability >= 1 - ``miss_prob``.

    Coupon-collector union bound: P(some pair never drawn in T draws)
    <= n_pairs * exp(-T / n_pairs), so T = n_pairs * (ln n_pairs +
    ln 1/miss_prob) suffices.
    """
    n_pairs = max(int(n_pairs), 1)
    return int(np.ceil(n_pairs * (np.log(n_pairs) + np.log(1.0 / miss_prob))))


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

    Thin wrapper over :func:`skimage.measure.ransac` with
    :class:`~skimage.transform.SimilarityTransform`: 2 correspondences
    per hypothesis (a 2-D similarity is exactly determined by two point
    pairs), inliers are correspondences with reprojection residual
    ``< thresh_m``, and the model is refit on the largest consensus set
    (ties broken by the smaller sum of squared residuals). Returns
    ``None`` unless at least ``min_inliers`` agree *and* they span at
    least ``min_world_baseline_m`` — a short baseline yields an
    unreliable scale, so we'd rather decline than emit a bad
    georeference.

    Sampling budget. The previous hand-rolled loop swept every pair
    exhaustively whenever ``n*(n-1)/2 <= n_iter``. skimage samples
    pairs at random, so for that small-``n`` regime ``max_trials`` is
    raised to the coupon-collector bound (:func:`_pair_coverage_trials`,
    every pair drawn with probability >= 1 - 1e-9). skimage however
    also stops early once it is (1 - 2**-52)-confident that an
    all-inlier pair has been drawn (``stop_probability=1`` cannot
    disable this), so the result is the standard RANSAC consensus, not
    a provable maximum over all pairs. Measured against the old
    exhaustive sweep on 3000 synthetic anchor sets (n = 3..10, 10-80 m
    anchor noise, 0-50 % gross outliers, thresh 150 m): 94.7 % identical
    results, 4.5 % a different inlier set of mostly equal size (tie
    broken by lower squared-residual sum), 0.8 % accept/decline flips;
    on the 136 differing sets the relative scale error w.r.t. the true
    scale was lower, not higher (median 10.3 % vs 12.2 %, mean 17.8 %
    vs 34.8 %). With more pairs
    than ``n_iter`` the budget is ``n_iter`` random draws, as before.
    ``rng_seed`` makes the fit deterministic.
    """
    vo_pts = np.asarray(vo_pts, dtype=np.float64)
    world_pts = np.asarray(world_pts, dtype=np.float64)
    n = len(vo_pts)
    if n < max(min_inliers, 2):
        return None

    n_pairs = n * (n - 1) // 2
    max_trials = _pair_coverage_trials(n_pairs) if n_pairs <= n_iter else int(n_iter)
    with warnings.catch_warnings():
        # "No inliers found" is reported via the (None, None) return,
        # which is handled below.
        warnings.simplefilter("ignore", UserWarning)
        model, inlier_mask = ransac(
            (vo_pts, world_pts),
            SimilarityTransform,
            min_samples=2,
            residual_threshold=thresh_m,
            max_trials=max_trials,
            rng=np.random.default_rng(rng_seed),
        )
    if inlier_mask is None or not model:
        return None
    best_inliers = np.flatnonzero(inlier_mask)
    if len(best_inliers) < min_inliers:
        return None

    resid = model.residuals(vo_pts[best_inliers], world_pts[best_inliers])
    scale = float(model.scale)
    w = world_pts[best_inliers]
    baseline = float(np.linalg.norm(w.max(axis=0) - w.min(axis=0)))
    if baseline < min_world_baseline_m or not np.isfinite(scale) or scale <= 0:
        return None
    return AnchorTransform(
        transform=model, scale=scale, inlier_idx=best_inliers,
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
