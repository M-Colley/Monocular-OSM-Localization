"""Tests for loop-closure drift correction."""

from __future__ import annotations

import numpy as np

from monocular_osm.loop_closure import detect_end_to_start_loop, redistribute_drift


def test_redistribute_closes_full_loop() -> None:
    # A true circle, then add linear drift so the end no longer meets the
    # start. Closing the loop should make end == start again.
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.c_[np.cos(t), np.sin(t)] * 100.0
    drift = np.linspace(0, 1, 200)[:, None] * np.array([40.0, 25.0])
    drifted = circle + drift
    assert np.linalg.norm(drifted[-1] - drifted[0]) > 30.0  # open
    closed = redistribute_drift(drifted, 0, len(drifted) - 1)
    assert np.linalg.norm(closed[-1] - closed[0]) < 1e-6     # closed


def _near_loop(n_side: int = 10, lead: int = 5, tail: int = 9,
               drift: float = 0.4) -> tuple[np.ndarray, int, int]:
    """A lead-in, a square circuit that nearly closes, then a tail.

    Returns ``(xz, i, j)`` where i and j are the SAME PLACE — the entry to
    and the exit from the circuit. That distinction is the whole point: the
    correction removes ``xz[j] - xz[i]``, so it is only meaningful when those
    two indices really are a revisit. On a straight line the "gap" between
    any two points is the arc between them, and closing it collapses the span
    onto a point (see the guard test below).
    """
    steps = ([(1.0, 0.0)] * lead
             + [(1.0, 0.0)] * n_side + [(0.0, 1.0)] * n_side
             + [(-1.0, 0.0)] * n_side + [(0.0, -1.0)] * n_side
             + [(0.0, -1.0)] * tail)
    xz = np.cumsum(np.array([(0.0, 0.0)] + steps), axis=0)
    i, j = lead, lead + 4 * n_side
    # Bow the circuit outward so it does not close exactly — that residual is
    # the drift the closure is supposed to remove.
    xz[i:] += np.linspace(0.0, drift, len(xz) - i)[:, None]
    return xz, i, j


def test_redistribute_preserves_prefix_and_shifts_tail() -> None:
    xz, i, j = _near_loop()
    out = redistribute_drift(xz, i, j)
    # Points before i are untouched.
    np.testing.assert_allclose(out[:i + 1], xz[:i + 1])
    # The gap at j is fully removed: out[j] lands exactly on xz[i].
    np.testing.assert_allclose(out[j], xz[i])
    # Tail keeps the same shape (rigid shift), so step vectors are preserved.
    np.testing.assert_allclose(np.diff(out[j + 1:], axis=0),
                               np.diff(xz[j + 1:], axis=0))


def test_redistribute_refuses_a_closure_that_is_not_a_loop() -> None:
    """A straight drive's "closure" would fold the route in half.

    The gap between two points on a straight ramp IS the arc between them,
    so applying the correction collapses that whole span onto one point —
    a kilometre-scale error, not the tens of metres closure buys. Measured
    on the 8 overlay clips, whose end-to-start gaps are 57-91% of arc length
    and none of which is a loop.
    """
    straight = np.cumsum(np.ones((50, 2)), axis=0)
    np.testing.assert_allclose(redistribute_drift(straight, 10, 30), straight)


def test_redistribute_still_closes_a_realistic_drift_bow() -> None:
    # KITTI drive_0033, the one clip in the fleet that genuinely closes, has
    # an end-start gap of 27% of arc length (see this module's docstring).
    # The guard must let that through — it is the one documented positive
    # result for --enable-loop-closure (README: 144 m -> 77 m).
    xz, i, j = _near_loop(n_side=20, drift=15.0)
    gap = float(np.linalg.norm(xz[j] - xz[i]))
    arc = float(np.sum(np.linalg.norm(np.diff(xz[i:j + 1], axis=0), axis=1)))
    assert 0.2 < gap / arc < 0.35, f"fixture should sit near KITTI's 27%, got {gap/arc:.2f}"
    out = redistribute_drift(xz, i, j)
    assert not np.allclose(out, xz)                 # the closure was applied
    np.testing.assert_allclose(out[j], xz[i])


def test_redistribute_noop_on_bad_indices() -> None:
    xz = np.random.RandomState(0).randn(20, 2)
    np.testing.assert_allclose(redistribute_drift(xz, 5, 5), xz)   # i==j
    np.testing.assert_allclose(redistribute_drift(xz, 8, 3), xz)   # i>j


def test_detect_loop_with_injected_matcher() -> None:
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(50)]
    # Strong match only between the very first and very last frame.
    def matcher(a, b):
        return 99 if (a is frames[0] and b is frames[-1]) else 5
    # Identity-by-object won't work through index lookups; match on content
    # instead: tag frames by a scalar in pixel [0,0,0].
    for k, f in enumerate(frames):
        f[0, 0, 0] = k
    def matcher2(a, b):
        return 80 if (int(a[0, 0, 0]) < 6 and int(b[0, 0, 0]) > 43) else 3
    pair = detect_end_to_start_loop(frames, min_inliers=30, match_fn=matcher2)
    assert pair is not None
    i, j = pair
    assert i < 6 and j > 43


def test_detect_loop_returns_none_when_no_revisit() -> None:
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(50)]
    pair = detect_end_to_start_loop(frames, min_inliers=30, match_fn=lambda a, b: 4)
    assert pair is None


def test_detect_loop_short_clip_none() -> None:
    assert detect_end_to_start_loop([np.zeros((4, 4, 3), np.uint8)] * 5) is None


def _textured(seed: int, w: int = 160, h: int = 120) -> np.ndarray:
    """Blocky random texture — plenty of ORB corners, deterministic."""
    import cv2

    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 4, w // 4, 3)).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def test_cached_orb_path_matches_uncached_reference() -> None:
    """The per-frame descriptor cache must give identical results to
    scoring every pair from scratch with _orb_inliers."""
    from monocular_osm.loop_closure import _orb_inliers

    # Head and tail share (shifted) copies of one scene; middles differ.
    scene = _textured(0)
    frames = [np.roll(scene, 4 * k, axis=1) for k in range(2)]
    frames += [_textured(s) for s in range(1, 17)]
    frames += [np.roll(scene, 3 + 4 * k, axis=1) for k in range(2)]

    cached = detect_end_to_start_loop(frames, min_inliers=20)
    reference = detect_end_to_start_loop(frames, min_inliers=20,
                                         match_fn=_orb_inliers)
    assert cached == reference
    assert cached is not None
    i, j = cached
    assert i < 3 and j > 16


def test_default_path_describes_each_frame_once(monkeypatch) -> None:
    """The default ORB path computes keypoints/descriptors once per
    unique frame instead of once per pair."""
    import monocular_osm.loop_closure as lc

    calls: list[int] = []
    real = lc._orb_describe

    def counting(img, **kw):
        calls.append(1)
        return real(img, **kw)

    monkeypatch.setattr(lc, "_orb_describe", counting)
    frames = [_textured(s % 4) for s in range(20)]
    lc.detect_end_to_start_loop(frames, min_inliers=10 ** 6)
    # head=2, tail=2 (12 % of 20) -> 4 unique frames across all pairs.
    assert len(calls) == 4


# ---------------------------------------------------------------------------
# a "verified" loop must not be fabricated from garbage or from static pixels
# ---------------------------------------------------------------------------


def test_degenerate_input_returns_zero_not_uninitialised_memory() -> None:
    """cv2.findFundamentalMat returns F=None WITH a non-None mask whose
    contents are uninitialised. Summing that mask returns garbage — measured
    returns of 2195 and 1603 from a 40-point input, either of which sails
    past min_inliers=30 and invents a loop out of nothing."""
    import cv2

    from monocular_osm.loop_closure import _inliers_from_features

    class _KP:
        def __init__(self, x, y):
            self.pt = (float(x), float(y))

    # Identical, perfectly collinear points: RANSAC cannot find F.
    kp = [_KP(100 + i, 100) for i in range(40)]
    des = np.zeros((40, 32), dtype=np.uint8)
    for i in range(40):                      # distinct descriptors so the
        des[i, i % 32] = i + 1               # ratio test admits them
    for _ in range(4):
        n = _inliers_from_features((kp, des), (kp, des))
        assert 0 <= n <= 40, f"impossible inlier count {n} for a 40-point input"


def test_static_graphics_do_not_verify_as_a_loop() -> None:
    """Two frames that share only a burned-in overlay are not a revisit.

    Zero-disparity correspondences are inliers to EVERY fundamental matrix,
    so without a parallax gate the overlay verifies itself. Measured on the
    Detroit clip ZhGb8q1kliY: a 449-inlier "geometrically verified loop"
    whose median inlier displacement was 0.00 px.
    """
    import cv2

    from monocular_osm.loop_closure import _orb_inliers

    rng = np.random.default_rng(3)
    w, h, band = 640, 480, 90
    band_img = cv2.resize(rng.integers(0, 255, (band // 2, w // 2, 3)).astype(np.uint8),
                          (w, band), interpolation=cv2.INTER_NEAREST)
    a = _textured(11, w=w, h=h)
    b = _textured(12, w=w, h=h)      # a completely different place
    a[h - band:] = band_img          # ...sharing only the overlay
    b[h - band:] = band_img
    assert _orb_inliers(a, b) < 30, "the overlay verified itself as a revisit"


def test_a_real_revisit_still_verifies() -> None:
    # The same scene from a slightly different viewpoint: large, consistent
    # disparity. The parallax gate must not reject this.
    from monocular_osm.loop_closure import _orb_inliers

    scene = _textured(21, w=640, h=480)
    shifted = np.roll(scene, 30, axis=1)
    assert _orb_inliers(scene, shifted) >= 30
