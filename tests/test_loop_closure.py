"""Tests for loop-closure drift correction."""

from __future__ import annotations

import numpy as np

from src.loop_closure import detect_end_to_start_loop, redistribute_drift


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


def test_redistribute_preserves_prefix_and_shifts_tail() -> None:
    xz = np.cumsum(np.ones((50, 2)), axis=0)  # straight ramp
    out = redistribute_drift(xz, 10, 30)
    # Points before i are untouched.
    np.testing.assert_allclose(out[:11], xz[:11])
    # Point j is pulled onto point i's... no — onto the closure: xz[j]-gap.
    # The gap at j is fully removed, so out[j]-out[i] == 0 vector? It maps
    # j onto i only in the *correction* sense: out[j] == xz[j] - (xz[j]-xz[i]).
    np.testing.assert_allclose(out[30], xz[10])
    # Tail keeps the same shape (rigid shift), so step vectors are preserved.
    np.testing.assert_allclose(np.diff(out[31:], axis=0), np.diff(xz[31:], axis=0))


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
