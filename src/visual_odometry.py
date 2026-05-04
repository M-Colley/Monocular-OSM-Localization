"""Monocular visual odometry: recover the camera trajectory from a sequence
of frames.

Pipeline per consecutive pair:
    ORB feature detection
        → cross-checked Hamming match
        → essential matrix with RANSAC
        → cv2.recoverPose() to get (R, t) up to scale

Per-step translations from a calibrated essential matrix have unit norm.
That gives us shape-correct, scale-free trajectories, which is exactly
what the downstream OSM matcher consumes.

The chaining convention: we track the world-to-camera pose (R_w2c, t_w2c),
update it with each relative motion, and recover the camera center in the
world as C = -R_w2c.T @ t_w2c. The camera at index 0 sits at the origin.

Top-down projection drops the camera-y axis (pointing down for a
forward-facing dashcam) and keeps (x, z), so x runs lateral to the road
at frame 0 and z runs forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class Trajectory:
    """Recovered ego trajectory.

    `centers` is an Nx3 array of camera centers in world coordinates.
    `xz` is the top-down projection (Nx2). `valid` flags which frames
    produced a usable relative pose; failed pairs hold the previous
    pose so the trajectory length matches the input.

    `rotations` and `translations` are the world-to-camera (R, t) at each
    frame, i.e. `x_camera = rotations[i] @ x_world + translations[i]`.
    Together with K they form the projection matrix `P_i = K[R_i|t_i]`
    needed for triangulating 3D points from feature matches across
    frames (the splat-building step).
    """
    centers: np.ndarray
    xz: np.ndarray
    valid: np.ndarray  # bool array, len == len(centers)
    n_inliers: list[int]
    rotations: np.ndarray       # N x 3 x 3, world-to-camera
    translations: np.ndarray    # N x 3, world-to-camera


def default_intrinsics(width: int, height: int, hfov_deg: float = 70.0) -> np.ndarray:
    """A sensible K for a YouTube dashcam clip with unknown calibration.

    HFOV ~70° is typical for windshield-mounted phones / dashcams after
    YouTube's encoding. Localization is not very sensitive to this — the
    essential matrix decomposes to the same shape under a wide range of
    fx, the matcher then strips scale anyway.
    """
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _estimate_relative_pose(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (R, t, n_inliers) for frame1 → frame2 or None on failure."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 16 or len(kp2) < 16:
        return None

    matches = matcher.match(des1, des2)
    if len(matches) < 16:
        return None

    matches = sorted(matches, key=lambda m: m.distance)[:300]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or E.shape != (3, 3):
        return None

    n_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    if n_inliers < 8:
        return None

    t = t.flatten()
    n = np.linalg.norm(t)
    if n < 1e-6:
        return None
    t = t / n  # unit step — monocular scale is unrecoverable

    return R, t, int(n_inliers)


def estimate_trajectory(
    frames: Sequence[np.ndarray],
    K: np.ndarray | None = None,
    *,
    enforce_planar: bool = True,
) -> Trajectory:
    """Compute camera centers in a world frame anchored at frame 0."""
    if len(frames) < 2:
        raise ValueError("need at least 2 frames")

    h, w = frames[0].shape[:2]
    if K is None:
        K = default_intrinsics(w, h)

    orb = cv2.ORB_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    R_w2c = np.eye(3)
    t_w2c = np.zeros(3)

    centers = [np.zeros(3)]
    valid = [True]
    inliers = [0]
    rotations = [R_w2c.copy()]
    translations = [t_w2c.copy()]

    for i in range(1, len(frames)):
        rel = _estimate_relative_pose(frames[i - 1], frames[i], K, orb, matcher)
        if rel is None:
            # Hold previous pose; downstream matcher can interpolate.
            centers.append(centers[-1].copy())
            rotations.append(rotations[-1].copy())
            translations.append(translations[-1].copy())
            valid.append(False)
            inliers.append(0)
            continue

        R_rel, t_rel, n_in = rel
        R_w2c = R_rel @ R_w2c
        t_w2c = R_rel @ t_w2c + t_rel
        c = -R_w2c.T @ t_w2c
        centers.append(c)
        rotations.append(R_w2c.copy())
        translations.append(t_w2c.copy())
        valid.append(True)
        inliers.append(n_in)

    centers_arr = np.asarray(centers)
    rotations_arr = np.asarray(rotations)
    translations_arr = np.asarray(translations)

    # Top-down: drop camera-y. For a forward-facing dashcam, y is
    # approximately the ground normal so this gives the bird's-eye path.
    xz = centers_arr[:, [0, 2]].copy()

    if enforce_planar:
        # Re-project onto a best-fit plane for the trajectory to clean up
        # accumulated pitch drift. We refit so the dominant motion lies in
        # the kept 2D plane.
        xz = _fit_plane_projection(centers_arr)

    return Trajectory(
        centers=centers_arr,
        xz=xz,
        valid=np.asarray(valid),
        n_inliers=inliers,
        rotations=rotations_arr,
        translations=translations_arr,
    )


def _fit_plane_projection(centers: np.ndarray) -> np.ndarray:
    """Project Nx3 points onto the 2D plane of maximum variance.

    Done with PCA: the trajectory of a ground vehicle should have very
    little variance along its true vertical axis. We discard the
    smallest-eigenvalue component, which on clean data is the y-axis but
    on drifted data may have leaked a bit of x or z — this is more robust
    than blindly dropping y.
    """
    if centers.shape[0] < 3:
        return centers[:, [0, 2]].copy()

    centered = centers - centers.mean(axis=0)
    # SVD: columns of Vt.T are the principal axes, descending variance.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axes_2d = Vt[:2].T  # 3x2: project onto top-2 PCs
    projected = centered @ axes_2d
    return projected


def trajectory_arc_length(xz: np.ndarray) -> np.ndarray:
    """Cumulative arc length along the trajectory."""
    if len(xz) < 2:
        return np.zeros(len(xz))
    diffs = np.diff(xz, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_uniform(xz: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample the path so consecutive samples are equidistant in arc length.

    This makes the shape comparable across paths with different spatial
    sampling — a prerequisite for the curvature-signature matcher.
    """
    if len(xz) < 2:
        raise ValueError("trajectory too short to resample")
    s = trajectory_arc_length(xz)
    total = s[-1]
    if total <= 0:
        raise ValueError("zero-length trajectory")
    targets = np.linspace(0, total, n_samples)
    out = np.empty((n_samples, 2))
    out[:, 0] = np.interp(targets, s, xz[:, 0])
    out[:, 1] = np.interp(targets, s, xz[:, 1])
    return out


def bearing_signature(xz: np.ndarray, n_samples: int = 128) -> np.ndarray:
    """Translation/rotation/scale-invariant shape descriptor.

    Resample to `n_samples` equidistant points, then return the
    *change* in heading between successive segments. The sequence of
    heading deltas (one per interior segment) captures turn pattern
    independent of where the path starts, which way it points, or how
    long it is. This is the key invariant the matcher uses against the
    OSM road graph.
    """
    pts = resample_uniform(xz, n_samples)
    diffs = np.diff(pts, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    delta = np.diff(headings)
    # Wrap to [-pi, pi].
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return delta
