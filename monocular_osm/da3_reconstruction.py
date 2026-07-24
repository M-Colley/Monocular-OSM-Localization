"""Dense reconstruction via Depth Anything 3.

Depth Anything 3 (https://github.com/ByteDance-Seed/Depth-Anything-3,
ByteDance, 2025) is a feed-forward model that, from a batch of images
of the same scene, jointly predicts:

  * a per-pixel depth map for each image
  * the camera intrinsics for each image
  * the camera extrinsics in a shared world frame

That's the entire SfM front-end of a 3D-Gaussian-Splatting pipeline
collapsed into one forward pass. We feed it ~30–60 keyframes from the
dashcam clip, backproject every confident pixel through the predicted
depth + pose, and stitch the result into a dense colored point cloud.

This gives a reconstruction that's *qualitatively* much closer to a
real Gaussian splat than our prior ORB-only sparse SfM. A real 3DGS
fitter (gsplat, splatfacto, etc.) would still take this DA3 output as
input and *refine* the splat by gradient descent on per-Gaussian
parameters — that's an extra training step we don't run here, but
slot in trivially.

GPU is required (CUDA). The model is `depth-anything/DA3-SMALL` from
HuggingFace; it downloads ~150 MB the first time. On an RTX 4050 it
processes ~6 frames in 1.5 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass
class DA3Reconstruction:
    """Dense per-keyframe reconstruction from Depth Anything 3."""
    points_world: np.ndarray         # M x 3
    colors_rgb: np.ndarray           # M x 3 uint8
    extrinsics_w2c: np.ndarray       # K x 3 x 4   (world → camera)
    intrinsics: np.ndarray           # K x 3 x 3   (per-frame K, in DA3's resolution)
    keyframe_indices: np.ndarray     # K  (which input frame each pose belongs to)
    processed_size: tuple[int, int]  # (H, W) DA3 internally resized to


def _torch_module():
    """Lazy import: torch is heavy and we only need it on the DA3 path."""
    import torch
    return torch


def load_da3_model(device: str = "cuda", model_id: str = "depth-anything/DA3-SMALL"):
    """Load DA3 from HuggingFace and move it to the requested device."""
    torch = _torch_module()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; pass device='cpu' or skip dense splat")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    return model


def _select_keyframes(
    n_frames: int, target: int, valid: np.ndarray | None = None
) -> np.ndarray:
    """Choose `target` keyframe indices from `n_frames` total, restricted
    to indices where `valid` is True if it's provided.

    Linear-spaced sampling is good enough — DA3 is robust to non-uniform
    temporal spacing; we mostly want broad coverage of the route.
    """
    if valid is not None:
        candidates = np.where(valid)[0]
    else:
        candidates = np.arange(n_frames)
    if len(candidates) <= target:
        return candidates
    pick = np.linspace(0, len(candidates) - 1, target).astype(int)
    return candidates[pick]


def _backproject_keyframe(
    depth: np.ndarray,                # (H, W)  float32
    conf: np.ndarray | None,          # (H, W)  float32 in [0, 1] or None
    K: np.ndarray,                    # (3, 3)
    extr_w2c: np.ndarray,             # (3, 4)  world → camera
    image_rgb: np.ndarray,            # (H, W, 3) uint8
    *,
    subsample: int,
    conf_threshold: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift one DA3 keyframe to a colored 3-D point cloud in world coords."""
    h, w = depth.shape
    K_inv = np.linalg.inv(K).astype(np.float32)
    R = extr_w2c[:, :3]
    t = extr_w2c[:, 3]

    # Pixel grid (subsampled).
    ys, xs = np.mgrid[0:h:subsample, 0:w:subsample]
    ys = ys.ravel()
    xs = xs.ravel()
    d = depth[ys, xs]
    mask = (d > 0.0) & (d < max_depth)
    if conf is not None:
        mask &= conf[ys, xs] >= conf_threshold
    if mask.sum() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    ys, xs, d = ys[mask], xs[mask], d[mask]
    # Camera-frame: (x, y, z) = depth * K_inv @ (u, v, 1)
    homog = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float32)  # 3 x N
    rays = K_inv @ homog                                                     # 3 x N
    cam_pts = rays * d[None, :]                                              # 3 x N
    # World: world = R^T @ (cam - t)
    world = R.T @ (cam_pts - t.reshape(3, 1))
    colors = image_rgb[ys, xs]
    return world.T, colors.astype(np.uint8)


def _align_chunk_to_reference(
    R_ref: np.ndarray, t_ref: np.ndarray,
    R_cur: np.ndarray, t_cur: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Align a chunk's world frame to the reference's world frame using
    one shared keyframe.

    Both `R, t` describe world→camera for the *same* underlying frame,
    in two different world frames. The 3-D similarity that maps the
    current world into the reference world is recovered as:

        T_align = T_ref^-1 * T_cur     (rigid: 4x4 SE(3))

    We treat scale as 1 because DA3 outputs metric depth in both
    chunks. Returns `(R_align (3x3), t_align (3,))` such that for any
    world point in the *current* chunk's frame: `world_ref = R_align @ world_cur + t_align`.
    """
    # camera_ref = R_ref @ X_world_ref + t_ref
    # camera_cur = R_cur @ X_world_cur + t_cur
    # Same camera, so R_ref @ X_world_ref + t_ref == R_cur @ X_world_cur + t_cur
    # → X_world_ref = R_ref.T @ (R_cur @ X_world_cur + t_cur - t_ref)
    R_align = R_ref.T @ R_cur
    t_align = R_ref.T @ (t_cur - t_ref)
    return R_align, t_align


def _align_chunk_least_squares(
    ref_poses: Sequence[np.ndarray],
    cur_poses: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid alignment over ALL shared keyframes.

    Each pair ``(ref_poses[i], cur_poses[i])`` is the same keyframe's
    world→camera (3x4) pose expressed in the reference (global) frame and
    the current chunk's frame. Aligning on a single keyframe lets one
    noisy DA3 pose corrupt the whole chunk, so instead:

    * rotation: average the per-frame relative rotations
      ``R_ref_i.T @ R_cur_i`` and project the sum back onto SO(3) via SVD
      (well-conditioned even when the shared camera centers are
      collinear, the common straight-driving case);
    * translation: least-squares over the camera centers,
      ``t_align = mean(C_ref_i - R_align @ C_cur_i)``.

    Returns ``(R_align, t_align)`` with the same meaning as
    :func:`_align_chunk_to_reference`.
    """
    M = np.zeros((3, 3))
    c_ref: list[np.ndarray] = []
    c_cur: list[np.ndarray] = []
    for ref, cur in zip(ref_poses, cur_poses):
        R_ref, t_ref = ref[:, :3], ref[:, 3]
        R_cur, t_cur = cur[:, :3], cur[:, 3]
        M += R_ref.T @ R_cur
        c_ref.append(-R_ref.T @ t_ref)   # camera center in the ref frame
        c_cur.append(-R_cur.T @ t_cur)   # ... and in the current frame
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R_align = U @ D @ Vt
    t_align = np.mean(np.asarray(c_ref) - np.asarray(c_cur) @ R_align.T, axis=0)
    return R_align, t_align


def reconstruct_with_da3(
    frames: Sequence[np.ndarray],
    *,
    n_keyframes: int = 48,
    valid_mask: np.ndarray | None = None,
    batch_size: int = 32,
    chunk_overlap: int = 4,
    subsample: int = 6,
    conf_threshold: float = 0.4,
    max_depth: float = 60.0,
    device: str = "cuda",
    model=None,
    model_id: str = "depth-anything/DA3-SMALL",
) -> DA3Reconstruction:
    """Run DA3 on a set of keyframes and stitch the per-frame depth maps
    into a single world-frame point cloud.

    `frames` are BGR uint8 arrays (the format `frame_extraction` returns).
    `valid_mask`, if given, restricts keyframe selection to frames where
    VO already considered the relative pose reliable (cuts down on
    motion-blur and tunnel frames).

    For `n_keyframes > batch_size` we run DA3 in overlapping chunks
    (`chunk_overlap` frames shared between consecutive chunks). The
    *first* chunk's world frame is the reference; subsequent chunks are
    aligned to it using the rigid transform implied by their shared
    keyframes. This is correct because DA3 is metric, so no per-chunk
    scale fit is needed — only rotation and translation.
    """
    if not frames:
        raise ValueError("no frames")

    if model is None:
        model = load_da3_model(device=device, model_id=model_id)

    keyframe_indices = _select_keyframes(len(frames), n_keyframes, valid_mask)
    if len(keyframe_indices) < 2:
        raise ValueError("need at least 2 valid keyframes for DA3")

    pil_imgs = [
        Image.fromarray(cv2.cvtColor(frames[int(i)], cv2.COLOR_BGR2RGB))
        for i in keyframe_indices
    ]

    n = len(pil_imgs)

    if n <= batch_size:
        chunks: list[tuple[int, int]] = [(0, n)]
    else:
        # Chunk with overlap so consecutive chunks share keyframes for alignment.
        step = max(1, batch_size - chunk_overlap)
        chunks = []
        i = 0
        while i < n:
            j = min(n, i + batch_size)
            chunks.append((i, j))
            if j == n:
                break
            i += step

    all_pts_world: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    final_extr: np.ndarray | None = None
    final_intr: np.ndarray | None = None
    proc_size: tuple[int, int] | None = None

    # Reference (chunk-0) world frame is the global frame. For each later
    # chunk, a similarity transform is fit by least squares over ALL keyframes
    # it shares with the already-placed frames (see the alignment below) and
    # applied to that chunk's points.
    extr_global_per_keyframe: dict[int, np.ndarray] = {}
    intr_global_per_keyframe: dict[int, np.ndarray] = {}

    for ci, (s, e) in enumerate(chunks):
        sub_imgs = pil_imgs[s:e]
        sub_keyframes = keyframe_indices[s:e]
        pred = model.inference(sub_imgs)

        depths = np.asarray(pred.depth)
        confs = np.asarray(pred.conf) if pred.conf is not None else None
        intr = np.asarray(pred.intrinsics)
        extr = np.asarray(pred.extrinsics)
        proc = np.asarray(pred.processed_images)
        h, w = depths.shape[1:]
        proc_size = (h, w)

        # Determine alignment to global frame.
        if ci == 0:
            R_align = np.eye(3)
            t_align = np.zeros(3)
        else:
            # The shared keyframes with the previous chunk are the ones in
            # [s, prev_e): the current chunk starts inside the previous one
            # by construction (step = batch_size - chunk_overlap). Collect
            # every shared keyframe whose global pose is already known and
            # align with a least-squares fit over all of them, so a single
            # noisy DA3 pose can't corrupt the whole chunk's placement.
            prev_s, prev_e = chunks[ci - 1]
            ref_poses: list[np.ndarray] = []
            cur_poses: list[np.ndarray] = []
            for shared_idx in range(s, min(prev_e, e)):
                global_kf = int(keyframe_indices[shared_idx])
                if global_kf not in extr_global_per_keyframe:
                    continue
                ref_poses.append(extr_global_per_keyframe[global_kf])
                cur_poses.append(extr[shared_idx - s])
            if not ref_poses:
                # No reliable shared keyframe → just append untransformed.
                R_align = np.eye(3)
                t_align = np.zeros(3)
            elif len(ref_poses) == 1:
                R_align, t_align = _align_chunk_to_reference(
                    ref_poses[0][:, :3], ref_poses[0][:, 3],
                    cur_poses[0][:, :3], cur_poses[0][:, 3],
                )
            else:
                R_align, t_align = _align_chunk_least_squares(ref_poses, cur_poses)

        # Backproject and transform each frame's points into the global frame.
        for k in range(len(sub_imgs)):
            pts, cols = _backproject_keyframe(
                depths[k],
                confs[k] if confs is not None else None,
                intr[k],
                extr[k],
                proc[k],
                subsample=subsample,
                conf_threshold=conf_threshold,
                max_depth=max_depth,
            )
            if len(pts) == 0:
                continue
            pts_global = pts @ R_align.T + t_align
            all_pts_world.append(pts_global)
            all_cols.append(cols)

            # Store this keyframe's pose in the GLOBAL frame for any
            # later chunk that overlaps with it.
            global_kf_idx = sub_keyframes[k]
            R_global = extr[k, :, :3] @ R_align.T  # apply alignment to pose too
            t_global = extr[k, :, 3] - R_global @ t_align
            extr_global_per_keyframe[int(global_kf_idx)] = np.hstack([R_global, t_global.reshape(3, 1)])
            intr_global_per_keyframe[int(global_kf_idx)] = intr[k]

    # Assemble final output: per-keyframe extr/intr in chronological order.
    sorted_keys = sorted(extr_global_per_keyframe.keys())
    final_extr = np.stack([extr_global_per_keyframe[k] for k in sorted_keys])
    final_intr = np.stack([intr_global_per_keyframe[k] for k in sorted_keys])

    if not all_pts_world:
        return DA3Reconstruction(
            points_world=np.zeros((0, 3)),
            colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            extrinsics_w2c=final_extr,
            intrinsics=final_intr,
            keyframe_indices=np.array(sorted_keys),
            processed_size=proc_size or (0, 0),
        )

    return DA3Reconstruction(
        points_world=np.vstack(all_pts_world),
        colors_rgb=np.vstack(all_cols).astype(np.uint8),
        extrinsics_w2c=final_extr,
        intrinsics=final_intr,
        keyframe_indices=np.array(sorted_keys),
        processed_size=proc_size or (0, 0),
    )


def da3_trajectory_xy(rec: DA3Reconstruction) -> np.ndarray:
    """Return the camera centers in DA3's world frame, projected to the
    XZ plane.

    DA3's world frame is gravity-aligned: y is up, x-z is the ground
    plane. So XZ is exactly the top-down driving path, in metric units.
    """
    centers = []
    for i in range(rec.extrinsics_w2c.shape[0]):
        R = rec.extrinsics_w2c[i, :, :3]
        t = rec.extrinsics_w2c[i, :, 3]
        C = -R.T @ t
        centers.append(C)
    centers = np.asarray(centers)
    return centers[:, [0, 2]]
