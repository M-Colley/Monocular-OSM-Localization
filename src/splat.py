"""Sparse "splat" reconstruction from VO frames + poses.

A *real* 3D Gaussian Splat (3DGS) is fitted by gradient descent over
millions of anisotropic Gaussians, takes hours of CUDA compute, and
needs a pre-trained pose graph (typically from COLMAP). That isn't
something this PoC can run inline.

Instead, we build the structure-from-motion *substrate* of a splat: a
sparse, colored 3D point cloud, triangulated from ORB feature matches
across frame pairs using the world-to-camera poses VO already
recovered. Each point gets a color sampled from one of the source
frames. The result is:

  * exportable as PLY (open in MeshLab / CloudCompare / online viewers)
  * renderable as a top-down image where each point is drawn as a
    small Gaussian — visually similar to a top-down splat render

The same top-down image is what the optional aerial feature-matching
channel matches against OSM imagery in `aerial_match.py`.

Limitations vs. a real splat:
  * sparse points only (no continuous opacity field)
  * one isotropic radius per point (real splats have full 3x3 covariance)
  * no view-dependent color (each point stores a single RGB sample)

For localization purposes the top-down render is the relevant output,
and a sparse SfM cloud carries enough road-surface and building-corner
information to be visible in BEV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d

from .visual_odometry import Trajectory


def build_splat_points(
    frames: Sequence[np.ndarray],
    trajectory: Trajectory,
    K: np.ndarray,
    *,
    baseline_frames: int = 5,
    max_pairs: int = 80,
    min_baseline_m: float = 0.05,
    max_depth_m: float = 80.0,
    min_matches: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate (point, color) pairs from many frame pairs.

    For frame pair (i, i + baseline_frames) we run ORB + cross-checked
    BFMatcher, then `cv2.triangulatePoints`. Triangulated points are
    kept if they are:

      * in front of both cameras (z > 0 in camera coords)
      * closer than `max_depth_m` (filter wild outliers from low-parallax
        matches)
      * the relative motion between the two camera centers is at least
        `min_baseline_m` (insufficient baseline → triangulation is
        ill-conditioned)

    Returns `(points_world Nx3, colors_rgb Nx3 uint8)`.
    """
    if len(frames) != len(trajectory.valid):
        raise ValueError(
            f"frame count ({len(frames)}) must equal trajectory length "
            f"({len(trajectory.valid)})"
        )

    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Pick frame pairs distributed across the trajectory.
    candidates = list(range(0, len(frames) - baseline_frames))
    if len(candidates) > max_pairs:
        # Stratified sampling so we cover the full path, not just the start.
        idx = np.linspace(0, len(candidates) - 1, max_pairs).astype(int)
        candidates = [candidates[k] for k in idx]

    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for i in candidates:
        j = i + baseline_frames
        if not (trajectory.valid[i] and trajectory.valid[j]):
            continue

        # Skip pairs with too little camera motion — degenerate triangulation.
        if np.linalg.norm(trajectory.centers[j] - trajectory.centers[i]) < min_baseline_m:
            continue

        f1, f2 = frames[i], frames[j]
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY) if f1.ndim == 3 else f1
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY) if f2.ndim == 3 else f2

        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
            continue

        matches = matcher.match(des1, des2)
        if len(matches) < min_matches:
            continue

        matches = sorted(matches, key=lambda m: m.distance)[:300]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Geometric filter: keep matches consistent with an essential
        # matrix, in case the ORB matches contain outliers.
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
        if E is None or mask is None:
            continue
        keep = mask.ravel().astype(bool)
        if keep.sum() < min_matches:
            continue
        pts1 = pts1[keep]
        pts2 = pts2[keep]

        R1, t1 = trajectory.rotations[i], trajectory.translations[i]
        R2, t2 = trajectory.rotations[j], trajectory.translations[j]
        P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        w = pts4d[3]
        good = np.abs(w) > 1e-9
        if good.sum() == 0:
            continue
        pts3d = (pts4d[:3, good] / w[good]).T  # M x 3, world coords

        pts1_kept = pts1[good]

        # Cheirality: in-front-of-both-cameras AND not crazy far.
        cam1 = (R1 @ pts3d.T + t1.reshape(3, 1)).T
        cam2 = (R2 @ pts3d.T + t2.reshape(3, 1)).T
        depth_ok = (cam1[:, 2] > 0.5) & (cam1[:, 2] < max_depth_m) & \
                   (cam2[:, 2] > 0.5) & (cam2[:, 2] < max_depth_m)
        pts3d = pts3d[depth_ok]
        pts1_kept = pts1_kept[depth_ok]
        if len(pts3d) == 0:
            continue

        # Sample color from frame 1 at each kept feature.
        h, w_img = f1.shape[:2]
        ix = np.clip(np.round(pts1_kept[:, 0]).astype(int), 0, w_img - 1)
        iy = np.clip(np.round(pts1_kept[:, 1]).astype(int), 0, h - 1)
        bgr = f1[iy, ix]
        rgb = bgr[:, ::-1]  # OpenCV → RGB

        all_pts.append(pts3d)
        all_colors.append(rgb)

    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    return np.vstack(all_pts), np.vstack(all_colors).astype(np.uint8)


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Write a PLY file viewable in MeshLab / CloudCompare / online viewers.

    Delegates to Open3D, the standard library for point-cloud I/O —
    it produces a compatible binary or ASCII PLY and handles all the
    annoying details (header sizing, color packing, endianness) the
    way every PLY consumer expects.
    """
    path = Path(path)
    if len(points) != len(colors):
        raise ValueError("points and colors must be the same length")
    pcd = build_open3d_point_cloud(points, colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
    if not ok:
        raise OSError(f"open3d failed to write {path}")


def build_open3d_point_cloud(
    points: np.ndarray, colors: np.ndarray
) -> o3d.geometry.PointCloud:
    """Wrap (Nx3 points, Nx3 colors) as an Open3D PointCloud.

    Open3D wants colors in [0, 1] floats; we accept either uint8 or
    float input.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors.dtype == np.uint8:
        cols = colors.astype(np.float64) / 255.0
    else:
        cols = np.asarray(colors, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def save_interactive_html(
    points: np.ndarray,
    colors: np.ndarray,
    path: Path,
    *,
    max_points: int = 8000,
    title: str = "Sparse splat (top-down dashcam reconstruction)",
) -> None:
    """Write a self-contained HTML page with a Plotly 3-D scatter of
    the splat. The user can rotate, pan and zoom in any browser — no
    server needed, no extra software to install.

    Subsamples to `max_points` if the cloud is bigger, since Plotly's
    in-browser performance degrades on dense clouds.
    """
    import plotly.graph_objects as go  # local import: heavy module

    pts = np.asarray(points)
    cols = np.asarray(colors)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    if cols.dtype != np.uint8:
        cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)

    color_strs = [f"rgb({r},{g},{b})" for r, g, b in cols]
    fig = go.Figure(data=[
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 2], z=-pts[:, 1],   # remap so y-up looks natural
            mode="markers",
            marker=dict(size=2, color=color_strs, opacity=0.85),
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (right)",
            yaxis_title="Z (forward)",
            zaxis_title="-Y (up)",
            aspectmode="data",
            bgcolor="#111",
        ),
        paper_bgcolor="#111",
        font=dict(color="#ddd"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def render_topdown_splat(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    resolution: int = 1024,
    margin_px: int = 24,
    point_radius_px: int = 2,
    blur_sigma: float = 1.5,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Rasterize the point cloud's top-down view into an RGB image.

    Drops the camera-y axis (vertical for a forward-facing dashcam) and
    keeps (x, z) — same projection used for shape matching — but with
    color drawn from each point. Each splat is a small Gaussian (a
    filled circle followed by a Gaussian blur) so the result is a soft
    top-down render rather than a stipple of pixels.
    """
    if len(points) == 0:
        img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        img[:] = background
        return img

    # Drop the camera-vertical axis.
    xz = points[:, [0, 2]]
    xmin, ymin = xz.min(axis=0)
    xmax, ymax = xz.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    s = (resolution - 2 * margin_px) / span

    px = ((xz[:, 0] - xmin) * s + margin_px).astype(int)
    # Flip y so increasing world-z goes UP in the image.
    py = (resolution - margin_px - 1 - (xz[:, 1] - ymin) * s).astype(int)

    img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    img[:] = background

    # Splat each point. cv2.circle with thickness=-1 fills the disk.
    # Z-order doesn't really matter on sparse SfM points; later draws
    # overwrite earlier ones, which is fine.
    for x, y, c in zip(px, py, colors):
        if 0 <= x < resolution and 0 <= y < resolution:
            cv2.circle(img, (int(x), int(y)),
                       point_radius_px,
                       (float(c[0]), float(c[1]), float(c[2])),
                       thickness=-1)

    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    return np.clip(img, 0, 255).astype(np.uint8)


def render_topdown_to_file(
    points: np.ndarray,
    colors: np.ndarray,
    path: Path,
    **kwargs,
) -> None:
    img_rgb = render_topdown_splat(points, colors, **kwargs)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
