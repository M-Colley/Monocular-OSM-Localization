"""Loop-closure drift correction for monocular VO trajectories.

Monocular VO accumulates drift: on a route that physically returns near
its start (a loop), the recovered trajectory does *not* close — on KITTI
drive_0033 the end-start gap is 27 % of the arc length, pure drift. That
warped shape is the dominant geometric error (road-snapping the result is
negligible; closing the loop + a correct scale drops the best-achievable
match error from 80 m to ~21 m).

This module:

* :func:`detect_end_to_start_loop` — does the drive end where it began?
  ORB-matches the last frames against the first; a geometrically-verified
  match means the camera revisited the start, giving a closure pair
  ``(i, j)`` (the two trajectory indices that are the same place).
* :func:`redistribute_drift` — force ``xz[j] == xz[i]`` by spreading the
  accumulated gap back along the trajectory in proportion to arc length
  (a first-order pose-graph closure), and shifting the post-``j`` tail by
  the same total. This removes the global drift bow without needing a
  full bundle adjustment.

It is intentionally conservative: detection requires a strong, verified
revisit, so non-loop drives get ``None`` and the trajectory is untouched.
"""

from __future__ import annotations

import numpy as np

from .visual_odometry import trajectory_arc_length


def redistribute_drift(xz: np.ndarray, i: int, j: int) -> np.ndarray:
    """Close the loop so ``xz[j]`` maps onto ``xz[i]``.

    The accumulated gap ``xz[j] - xz[i]`` is removed by subtracting a ramp
    that grows from 0 at ``i`` to the full gap at ``j`` (proportional to
    arc length — drift accumulates with distance travelled), then holding
    that full correction for every point after ``j``. Points before ``i``
    are unchanged. ``i < j`` required.
    """
    xz = np.asarray(xz, dtype=np.float64).copy()
    n = len(xz)
    if not (0 <= i < j < n):
        return xz
    gap = xz[j] - xz[i]
    arc = trajectory_arc_length(xz)
    span = arc[j] - arc[i]
    if span <= 1e-9:
        return xz
    # Ramp 0..1 over [i, j] by arc-length fraction, flat 1 after j.
    frac = np.zeros(n)
    frac[i:j + 1] = (arc[i:j + 1] - arc[i]) / span
    frac[j + 1:] = 1.0
    return xz - frac[:, None] * gap


def detect_end_to_start_loop(
    frames: list,
    *,
    head_frac: float = 0.12,
    tail_frac: float = 0.12,
    min_inliers: int = 30,
    n_anchor: int = 3,
    match_fn=None,
):
    """Detect whether the drive's end revisits its start.

    The closure pair has one frame near the very END and one near the
    START, so we *anchor on the extremes*: compare each of the last
    ``n_anchor`` frames against every frame in the first ``head_frac``,
    and each of the first ``n_anchor`` frames against every frame in the
    last ``tail_frac``. This is O(window) — far cheaper than the full
    O(window²) grid, and (unlike a sparse global subsample) it never
    misses the corner where loops actually close. Returns the
    ``(head_index, tail_index)`` of the strongest geometrically-verified
    ORB match if it clears ``min_inliers``, else ``None``.
    ``match_fn(img_a, img_b) -> n_inliers`` is injectable for testing.
    """
    n = len(frames)
    if n < 10:
        return None
    n_head = max(1, int(n * head_frac))
    n_tail = max(1, int(n * tail_frac))
    head_idx = list(range(0, n_head))
    tail_idx = list(range(n - n_tail, n))
    na = max(1, min(n_anchor, n_head, n_tail))
    pairs: set[tuple[int, int]] = set()
    for b in tail_idx[-na:]:                 # last frames vs all of the head
        pairs.update((a, b) for a in head_idx)
    for a in head_idx[:na]:                   # first frames vs all of the tail
        pairs.update((a, b) for b in tail_idx)
    if match_fn is not None:
        score = lambda a, b: match_fn(frames[a], frames[b])
    else:
        # Cache ORB keypoints/descriptors per frame: each head frame is
        # otherwise re-described once per tail anchor (and vice versa),
        # ~6x redundant detection. Identical results, ~4-6x faster.
        import cv2

        orb = cv2.ORB_create(nfeatures=1500)
        feats: dict[int, tuple] = {}

        def _feat(i: int):
            if i not in feats:
                feats[i] = _orb_describe(frames[i], orb=orb)
            return feats[i]

        score = lambda a, b: _inliers_from_features(_feat(a), _feat(b))
    best = (0, None)
    for a, b in pairs:
        if a >= b:
            continue
        nin = score(a, b)
        if nin > best[0]:
            best = (nin, (a, b))
    if best[0] >= min_inliers and best[1] is not None:
        return best[1]
    return None


def _orb_describe(img, *, n_features: int = 1500, orb=None):
    """ORB keypoints + descriptors for one frame (cacheable per frame)."""
    import cv2

    if orb is None:
        orb = cv2.ORB_create(nfeatures=n_features)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return orb.detectAndCompute(g, None)


def _inliers_from_features(feat_a, feat_b, *, ratio: float = 0.75) -> int:
    """Ratio-test matches verified by a fundamental-matrix RANSAC."""
    import cv2

    ka, da = feat_a
    kb, db = feat_b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    return int(mask.sum()) if mask is not None else 0


def _orb_inliers(img_a, img_b, *, n_features: int = 1500, ratio: float = 0.75) -> int:
    """ORB + ratio-test matches verified by a fundamental-matrix RANSAC."""
    return _inliers_from_features(
        _orb_describe(img_a, n_features=n_features),
        _orb_describe(img_b, n_features=n_features),
        ratio=ratio,
    )
