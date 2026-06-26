"""Unit tests for the MapAnything submap-stitching core (no model / GPU needed)."""

from __future__ import annotations

import numpy as np

from src.mapanything_trajectory import _umeyama, stitch_windows


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
