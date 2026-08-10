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


# A closure asserts that the drive returned to where it began. If the gap it
# would remove is a large fraction of the distance travelled, the trajectory
# was never a loop and applying the correction folds it roughly in half —
# an error of order the clip length (kilometres), not the tens of metres
# closure is supposed to buy. Measured on the 8 overlay clips, whose
# end-to-start gaps run 57.5-90.8% of arc length and NONE of which is a loop.
_MAX_CLOSURE_GAP_FRAC = 0.45


def redistribute_drift(xz: np.ndarray, i: int, j: int) -> np.ndarray:
    """Close the loop so ``xz[j]`` maps onto ``xz[i]``.

    The accumulated gap ``xz[j] - xz[i]`` is removed by subtracting a ramp
    that grows from 0 at ``i`` to the full gap at ``j`` (proportional to
    arc length — drift accumulates with distance travelled), then holding
    that full correction for every point after ``j``. Points before ``i``
    are unchanged. ``i < j`` required.

    A closure whose gap exceeds :data:`_MAX_CLOSURE_GAP_FRAC` of the arc it
    spans is refused unchanged: that is not drift on a loop, it is a false
    match on a drive that went somewhere else.
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
    if float(np.linalg.norm(gap)) > _MAX_CLOSURE_GAP_FRAC * float(span):
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


# A revisit is two views of the same place from different times, so its
# correspondences must MOVE. Anything below this is burned-in graphics (a
# channel watermark, a GPS/speed stamp, the dashboard) which is pixel-identical
# in any two frames of the clip and is an inlier to every fundamental matrix.
#
# Both thresholds are fractions of the image WIDTH, not pixel counts, so they
# mean the same thing on a 640-wide KITTI frame and a 1920-wide dashcam clip.
# A fixed pixel value tuned on 1080p is several percent of a small frame and
# would reject genuine closures on low-resolution datasets — which is where
# the one documented positive result lives (README: --enable-loop-closure
# takes KITTI drive_0033 from 144 m to 77 m).
_STATIC_MATCH_FRAC = 0.001
# ...and the surviving inliers must move by a real amount, or the "verified
# loop" is just the overlay agreeing with itself. Measured: the 449-inlier
# best pair on the Detroit clip ZhGb8q1kliY has a median inlier displacement
# of 0.00 px, with 90% of its inliers under 1 px — so 0.5% of width (9.6 px at
# 1080p) rejects it by an order of magnitude while staying well under the
# tens of pixels of disparity a genuine second pass produces.
_MIN_INLIER_DISPLACEMENT_FRAC = 0.005


def _keypoint_extent(ka, kb) -> float:
    """Image width, inferred from how far the keypoints spread.

    ORB corners cover most of a textured frame, so the largest x-coordinate
    is a good proxy — and it keeps the thresholds above resolution-relative
    without threading the frame size through the descriptor cache.
    """
    xs = [p.pt[0] for p in ka] + [p.pt[0] for p in kb]
    return max(max(xs) if xs else 0.0, 1.0)


def _inliers_from_features(feat_a, feat_b, *, ratio: float = 0.75) -> int:
    """Ratio-test matches verified by a fundamental-matrix RANSAC."""
    import cv2

    ka, da = feat_a
    kb, db = feat_b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    width = _keypoint_extent(ka, kb)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])

    # Drop the zero-parallax correspondences before RANSAC ever sees them.
    moving = np.linalg.norm(pb - pa, axis=1) >= _STATIC_MATCH_FRAC * width
    if int(moving.sum()) < 8:
        return 0
    pa, pb = pa[moving], pb[moving]

    try:
        F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    except cv2.error:
        # Degenerate (identical or collinear) point sets raise rather than
        # return. Swallowing this at the call site would be wrong: the caller
        # cannot distinguish "no loop" from "this pair blew up".
        return 0
    # F is None means RANSAC found no solution — and OpenCV still hands back a
    # mask, whose contents are then UNINITIALISED. Summing it returns garbage:
    # measured returns of 2195 and 1603 from a 40-point input, either of which
    # sails past min_inliers=30 and fabricates a loop out of nothing.
    if F is None or mask is None:
        return 0
    inlier = mask.ravel().astype(bool)
    if int(inlier.sum()) < 8:
        return 0
    if float(np.median(np.linalg.norm(pb[inlier] - pa[inlier], axis=1))) \
            < _MIN_INLIER_DISPLACEMENT_FRAC * width:
        return 0
    return int(inlier.sum())


def _orb_inliers(img_a, img_b, *, n_features: int = 1500, ratio: float = 0.75) -> int:
    """ORB + ratio-test matches verified by a fundamental-matrix RANSAC."""
    return _inliers_from_features(
        _orb_describe(img_a, n_features=n_features),
        _orb_describe(img_b, n_features=n_features),
        ratio=ratio,
    )
