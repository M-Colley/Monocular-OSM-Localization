"""Unit tests for the MapAnything submap-stitching core (no model / GPU needed)."""

from __future__ import annotations

import numpy as np

from src.mapanything_trajectory import _positions_to_xy, _umeyama, stitch_windows


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))
    R, s, t = _rand_rot(rng), 1.37, np.array([4.0, -2.0, 9.0])
    dst = s * (src @ R.T) + t
    s2, R2, t2 = _umeyama(src, dst)
    assert np.isclose(s2, s, atol=1e-6)
    assert np.allclose(R2, R, atol=1e-6)
    assert np.allclose(t2, t, atol=1e-6)


def test_stitch_recovers_trajectory_up_to_similarity():
    rng = np.random.default_rng(1)
    # a curved (well-conditioned) 3D path, like a drive with turns
    u = np.linspace(0, 4 * np.pi, 64)
    truth = np.c_[np.cos(u) * u, np.sin(u) * u, 0.2 * u]
    W, S = 16, 8
    windows = []
    for st in range(0, len(truth) - W + 1, S):
        ids = list(range(st, st + W))
        # each window lives in its own arbitrary metric frame (scale ~1 -> guard passes)
        R, s, t = _rand_rot(rng), float(rng.uniform(0.9, 1.1)), rng.normal(size=3) * 7
        windows.append((ids, s * (truth[ids] @ R.T) + t))

    G = stitch_windows(windows)
    ids = sorted(G)
    assert len(ids) == len(truth)                     # every frame placed
    rec = np.array([G[i] for i in ids])
    # recovered up to a single global similarity
    s2, R2, t2 = _umeyama(rec, truth[ids])
    res = (s2 * (rec @ R2.T) + t2) - truth[ids]
    assert np.sqrt((res ** 2).sum(1).mean()) < 1e-6


def test_stitch_collinear_overlap_no_arbitrary_roll():
    """Straight-driving overlaps are (near-)collinear: the full 3D fit's
    rotation about the driving line is noise-driven, so a turn following
    the straight stretch used to enter the global frame with an arbitrary
    twist. The rank-deficiency guard must constrain the fit to yaw-only
    and keep the turn in the ground plane."""
    rng = np.random.default_rng(7)
    straight = np.c_[np.arange(20.0), np.zeros(20), np.zeros(20)]
    turn = np.c_[np.full(6, 19.0), np.zeros(6), np.arange(1.0, 7.0)]
    truth = np.vstack([straight, turn])

    # Each window reconstructs the shared frames with its own independent
    # mm-level noise (the realistic case; with common-mode noise even a
    # degenerate fit looks exact).
    w0 = (list(range(0, 14)), truth[0:14] + rng.normal(size=(14, 3)) * 1e-3)
    # Window 1 lives in its own frame: an in-plane yaw + translation of
    # the truth. Its overlap with window 0 (ids 8..13) is collinear.
    cur = (truth[8:26] + rng.normal(size=(18, 3)) * 1e-3) \
        @ _rot_y(np.deg2rad(25.0)).T + np.array([3.0, 0.0, -5.0])
    w1 = (list(range(8, 26)), cur)

    G = stitch_windows([w0, w1])
    placed = np.array([G[i] for i in range(26)])
    # No roll twist: the turn stays in the ground plane (the old full-3D
    # fit twisted it out of plane by metres)...
    assert np.abs(placed[:, 1]).max() < 0.05
    # ... and lands where the truth says (yaw + translation recovered).
    np.testing.assert_allclose(placed, truth, atol=0.05)


def test_positions_to_xy_preserves_turn_sign_under_yaw():
    """Same arbitrary-handedness coin flip as VO's plane fit: the PCA
    projection must never MIRROR the driving plane (turn signs flip)."""
    rng = np.random.default_rng(3)
    path = np.vstack([
        np.c_[np.arange(20.0), np.zeros(20), np.zeros(20)],
        np.c_[np.full(15, 19.0), np.zeros(15), np.arange(1.0, 16.0)],
    ])
    path[:, 1] += rng.normal(size=len(path)) * 0.01

    def turn_sign(xy):
        d1 = xy[8] - xy[0]
        d2 = xy[-1] - xy[-9]
        return np.sign(d1[0] * d2[1] - d1[1] * d2[0])

    ref = turn_sign(path[:, [0, 2]])  # raw top-down drop
    for seed in range(20):
        a = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        xy = _positions_to_xy(path @ _rot_y(a).T)
        assert turn_sign(xy) == ref, f"mirrored at seed {seed}"


def test_stitch_scale_guard_rejects_blowup():
    # a degenerate window whose Umeyama scale is wild must fall back to rigid,
    # so no placed point explodes far beyond the others.
    ids0 = list(range(0, 8))
    ids1 = list(range(4, 12))
    base = np.c_[np.linspace(0, 7, 8), np.zeros(8), np.zeros(8)]
    w0 = (ids0, base)
    # window 1 overlaps frames 4..7 but with a 50x scale (degenerate) frame
    overlap = base[4:8] * 50.0
    new = np.c_[np.linspace(8, 11, 4), np.zeros(4), np.zeros(4)] * 50.0
    w1 = (ids1, np.vstack([overlap, new]))
    G = stitch_windows([w0, w1], scale_guard=(0.6, 1.7))
    placed = np.array([G[i] for i in sorted(G)])
    extent = np.linalg.norm(placed.max(0) - placed.min(0))
    assert extent < 1e3                               # no 50x blowup leaked through
