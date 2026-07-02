"""Chunk-alignment tests for DA3 reconstruction (fake model, no GPU).

The fake model returns ground-truth poses expressed in a random rigid
per-chunk world frame, so a correct multi-chunk stitch must recover the
global (chunk-0) poses exactly. The old code picked a shared keyframe
index OUTSIDE the previous chunk, silently fell into the identity-align
branch, and appended later chunks unaligned.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.da3_reconstruction import da3_trajectory_xy, reconstruct_with_da3


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _gt_poses(n: int) -> list[np.ndarray]:
    """World->camera (3x4) poses along an L-shaped drive, varying yaw."""
    poses = []
    for i in range(n):
        if i < n // 2:
            C = np.array([float(i) * 2.0, 0.0, 0.0])
            yaw = 0.0
        else:
            C = np.array([(n // 2 - 1) * 2.0, 0.0, float(i - n // 2 + 1) * 2.0])
            yaw = np.pi / 2
        R = _rot_y(yaw + 0.05 * i)
        t = -R @ C
        poses.append(np.hstack([R, t.reshape(3, 1)]))
    return poses


class _FakeDA3Model:
    """Stands in for DA3: emits GT poses in a random per-chunk frame.

    Frame identity is encoded in the input images' pixel values (the
    pipeline converts BGR->RGB->PIL, which preserves a uniform gray).
    """

    def __init__(self, gt_poses: list[np.ndarray]) -> None:
        self.gt = gt_poses
        self.calls = 0

    def inference(self, imgs):
        idxs = [int(np.asarray(im)[0, 0, 0]) for im in imgs]
        rng = np.random.default_rng(1000 + self.calls)
        if self.calls == 0:
            R_c, t_c = np.eye(3), np.zeros(3)  # chunk 0 = the global frame
        else:
            R_c, t_c = _rand_rot(rng), rng.normal(size=3) * 5.0
        self.calls += 1
        extr = []
        for i in idxs:
            R_i, t_i = self.gt[i][:, :3], self.gt[i][:, 3]
            # This chunk's world frame: x_chunk = R_c @ x_global + t_c.
            Rp = R_i @ R_c.T
            tp = t_i - Rp @ t_c
            extr.append(np.hstack([Rp, tp.reshape(3, 1)]))
        k = len(idxs)
        K = np.array([[10.0, 0.0, 2.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]])
        return SimpleNamespace(
            depth=np.ones((k, 4, 4), dtype=np.float32),
            conf=None,
            intrinsics=np.tile(K, (k, 1, 1)),
            extrinsics=np.asarray(extr),
            processed_images=np.zeros((k, 4, 4, 3), dtype=np.uint8),
        )


def test_multi_chunk_alignment_recovers_global_poses() -> None:
    n = 16
    gt = _gt_poses(n)
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]
    fake = _FakeDA3Model(gt)
    rec = reconstruct_with_da3(
        frames, n_keyframes=n, batch_size=8, chunk_overlap=4,
        subsample=1, model=fake, device="cpu",
    )
    # Chunks (0,8), (4,12), (8,16): three model calls, all frames posed.
    assert fake.calls == 3
    assert rec.keyframe_indices.tolist() == list(range(n))
    # Every chunk must be stitched back into the chunk-0 (global) frame;
    # the old shared-index bug appended chunks 2/3 with identity aligns.
    np.testing.assert_allclose(rec.extrinsics_w2c, np.stack(gt), atol=1e-6)
    # The projected trajectory follows the L-shaped GT drive.
    xy = da3_trajectory_xy(rec)
    gt_centers = np.array([-p[:, :3].T @ p[:, 3] for p in gt])
    np.testing.assert_allclose(xy, gt_centers[:, [0, 2]], atol=1e-6)


def test_single_chunk_unchanged() -> None:
    """n <= batch_size: one chunk, poses passed through untouched."""
    n = 6
    gt = _gt_poses(n)
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]
    fake = _FakeDA3Model(gt)
    rec = reconstruct_with_da3(
        frames, n_keyframes=n, batch_size=8, chunk_overlap=4,
        subsample=1, model=fake, device="cpu",
    )
    assert fake.calls == 1
    np.testing.assert_allclose(rec.extrinsics_w2c, np.stack(gt), atol=1e-9)
