"""Inverse Perspective Mapping (IPM): warp dashcam frames into a
top-down road-plane view and stitch them along the trajectory.

For ego-driving footage the road in front of the camera lies on a
single plane (locally), and we know roughly the camera's height above
that plane and its tilt. Under that assumption every pixel in the
forward image maps to a unique point on the road plane via a
homography:

    [u, v, 1]^T  --(H)-->  [X, Z, 1]^T   (road plane in metric units)

Stitching IPMs from many frames using the recovered camera trajectory
yields a single road-plane image — the "synthetic satellite tile" that
the original spec asked for. This is the right comparison input for
the OSM-aerial channel: an IPM strip and an OSM tile are both
top-down views of the same road network and ORB will actually find
shared features (intersection corners, lane markings, road edges).

This module deliberately does *not* depend on any deep model — it's
pure geometry, runs in milliseconds per frame, and works on CPU. The
calibration parameters (camera height, pitch) are reasonable defaults
for a windshield-mounted dashcam; the user can override them.

Coordinate convention: BEV image has X (right) and Y (forward, away
from camera). The vehicle moves in +Y across consecutive frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IPMCalibration:
    """Geometry of the dashcam-to-ground projection."""
    K: np.ndarray                  # 3x3 camera intrinsics
    camera_height_m: float = 1.4   # height above road plane
    pitch_deg: float = 0.0         # downward tilt (positive = looking down)
    roll_deg: float = 0.0          # rotation about optical axis (lateral level)
    bev_width_m: float = 30.0      # horizontal extent of BEV in meters (X)
    bev_depth_m: float = 40.0      # forward extent (Y)
    near_clip_m: float = 3.0       # mask out below this many meters in front
    bev_resolution_pix_per_m: float = 8.0


def _rotation_pitch_roll(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera-to-vehicle rotation: pitch about x then roll about z.

    OpenCV camera frame: +x right, +y down, +z forward. We're rotating
    so that the road (at y = camera_height in vehicle frame) is mapped
    correctly into the camera image.
    """
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)
    Rp = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p),  np.cos(p)],
    ])
    Rr = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r),  np.cos(r), 0],
        [0,          0,         1],
    ])
    return Rr @ Rp


def compute_ipm_homography(cal: IPMCalibration) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute the homography that warps an image into the BEV.

    Returns `(H, (bev_h, bev_w))` where H takes pixels in the input
    image to pixels in a BEV image of size `bev_w x bev_h`.

    BEV pixel coordinate convention: origin at bottom-center of the
    image, +x to the right (lateral, meters * res), +y upward (forward,
    meters * res). We build H by sampling 4 ground-plane points,
    projecting each to the image with the camera model, and using the
    image↔BEV correspondences with `cv2.findHomography`.
    """
    R = _rotation_pitch_roll(cal.pitch_deg, cal.roll_deg)
    t = np.array([0.0, cal.camera_height_m, 0.0])  # camera is up by h above ground

    # Four corners of the BEV in vehicle (= ground) frame:
    #  ground points: y = 0 (road plane), x in lateral, z in forward
    half_w = cal.bev_width_m / 2.0
    near = cal.near_clip_m
    far = cal.near_clip_m + cal.bev_depth_m
    ground_pts = np.array([
        [-half_w, 0.0, near],
        [ half_w, 0.0, near],
        [ half_w, 0.0, far],
        [-half_w, 0.0, far],
    ])  # 4 x 3, in vehicle frame

    # Vehicle → camera frame: x_cam = R @ (x_veh - t).
    cam_pts = (ground_pts - t) @ R.T
    # Project: x_pix = K @ cam_pts; clip behind-camera.
    img_pts = cam_pts @ cal.K.T
    img_pts = img_pts[:, :2] / img_pts[:, 2:3]

    res = cal.bev_resolution_pix_per_m
    bev_w = int(round(cal.bev_width_m * res))
    bev_h = int(round(cal.bev_depth_m * res))
    # BEV pixel: x in [0, bev_w], y in [0, bev_h], with y=bev_h at the
    # near edge (closest to vehicle) and y=0 at the far edge.
    bev_pts = np.array([
        [0,        bev_h],   # ( -half_w, near )  → bottom-left
        [bev_w,    bev_h],   # (  half_w, near )  → bottom-right
        [bev_w,    0    ],   # (  half_w, far  )  → top-right
        [0,        0    ],   # ( -half_w, far  )  → top-left
    ], dtype=np.float32)

    H, _ = cv2.findHomography(img_pts.astype(np.float32), bev_pts)
    return H, (bev_h, bev_w)


def warp_to_bev(image: np.ndarray, H: np.ndarray, bev_size: tuple[int, int]) -> np.ndarray:
    """Apply the IPM homography. Returns a BGR image of `bev_size`."""
    bev_h, bev_w = bev_size
    return cv2.warpPerspective(
        image, H, (bev_w, bev_h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )


def stitch_bev_along_trajectory(
    frames: list[np.ndarray],
    trajectory_xz: np.ndarray,
    cal: IPMCalibration,
    *,
    keyframe_stride: int = 8,
    canvas_resolution_pix_per_m: float = 4.0,
    canvas_pad_m: float = 50.0,
    min_inlier_ratio: float = 0.0,
) -> np.ndarray:
    """Stitch IPM warps along a 2-D trajectory into one big BEV canvas.

    For every `keyframe_stride`-th frame we IPM-warp to a local BEV,
    then translate + rotate that warp onto a global canvas using the
    trajectory's heading at that frame.

    The trajectory drives where each tile lands. Since monocular VO is
    scale-free, we rescale the trajectory so that its arc-length matches
    the metric IPM tile sizes (each forward step ≈ tile depth / 4).
    """
    if len(frames) != len(trajectory_xz):
        raise ValueError(
            f"frames ({len(frames)}) and trajectory_xz ({len(trajectory_xz)}) length mismatch"
        )
    if len(frames) < 2:
        raise ValueError("need at least 2 frames")

    H_local, (tile_h, tile_w) = compute_ipm_homography(cal)
    tile_meters_per_pixel = 1.0 / cal.bev_resolution_pix_per_m

    # Rescale trajectory to metric so that tile placement is consistent.
    # Empirically: each VO step is ~1 unit; we map total trajectory length
    # to a metric estimate based on tile depth.
    diffs = np.diff(trajectory_xz, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    arc = float(seg.sum())
    if arc < 1e-6:
        raise ValueError("zero-length trajectory")
    # Heuristic: assume the trajectory covers ~ frames * 0.5 m per unit
    # if VO already had reasonable scale, but we mainly use trajectory
    # SHAPE here; the canvas auto-pads anyway.
    metric_arc_target = max(arc, 50.0)  # don't compress tiny trajectories
    scale = metric_arc_target / arc
    metric_traj = trajectory_xz * scale

    canvas_pix_per_m = canvas_resolution_pix_per_m
    xs = metric_traj[:, 0]
    ys = metric_traj[:, 1]
    pad = canvas_pad_m
    canvas_w_m = (xs.max() - xs.min()) + 2 * pad + cal.bev_width_m
    canvas_h_m = (ys.max() - ys.min()) + 2 * pad + cal.bev_depth_m
    canvas_w = int(canvas_w_m * canvas_pix_per_m)
    canvas_h = int(canvas_h_m * canvas_pix_per_m)

    # Origin in metric: (xs.min() - pad, ys.min() - pad).
    origin_x = xs.min() - pad
    origin_y = ys.min() - pad

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint16)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.uint16)

    for i in range(0, len(frames), keyframe_stride):
        bev = warp_to_bev(frames[i], H_local, (tile_h, tile_w))

        # Determine heading at this frame (from local trajectory tangent).
        i0 = max(0, i - keyframe_stride // 2)
        i1 = min(len(metric_traj) - 1, i + keyframe_stride // 2)
        if i1 == i0:
            heading = 0.0
        else:
            d = metric_traj[i1] - metric_traj[i0]
            heading = float(np.arctan2(d[1], d[0]))

        # Build the affine that places this tile on the canvas:
        # 1) tile-local: tile is (tile_w, tile_h) px; (tile_w/2, tile_h)
        #    is the camera position (bottom-center).
        # 2) we want to place that camera position at the metric trajectory
        #    point (metric_traj[i]) after rotating the tile to the heading.
        cx, cy = metric_traj[i, 0], metric_traj[i, 1]
        # Heading in canvas frame: 0 rad = pointing along +x (east).
        # The tile's "forward" axis is its +y in BEV → we rotate so that
        # tile +y aligns with heading direction.
        # Rotation by (heading - π/2) maps tile +y to canvas heading.
        theta = heading - np.pi / 2.0
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Construct the warp matrix: tile -> canvas.
        # Step A: shift tile so its camera (tile_w/2, tile_h) is at origin.
        T_origin = np.array([
            [1, 0, -tile_w / 2.0],
            [0, 1, -tile_h * 1.0],
            [0, 0, 1],
        ])
        # Step B: scale tile pixels to meters.
        S = np.array([
            [tile_meters_per_pixel, 0, 0],
            [0, tile_meters_per_pixel, 0],
            [0, 0, 1],
        ])
        # Step C: rotate by theta.
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ])
        # Step D: shift to (cx, cy) in metric, then to canvas pixels.
        T_to_canvas = np.array([
            [canvas_pix_per_m, 0, (cx - origin_x) * canvas_pix_per_m],
            [0, -canvas_pix_per_m, canvas_h - (cy - origin_y) * canvas_pix_per_m],
            [0, 0, 1],
        ])
        M = T_to_canvas @ R @ S @ T_origin
        M2x3 = M[:2, :].astype(np.float32)

        warped = cv2.warpAffine(
            bev, M2x3, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
        )
        mask = (warped.sum(axis=2) > 0).astype(np.uint16)
        canvas += warped.astype(np.uint16)
        weight += mask

    weight_safe = np.maximum(weight, 1)
    blended = (canvas / weight_safe[:, :, None]).clip(0, 255).astype(np.uint8)
    return blended


def render_ipm_canvas(
    frames: list[np.ndarray],
    trajectory_xz: np.ndarray,
    K: np.ndarray,
    *,
    keyframe_stride: int = 8,
    camera_height_m: float = 1.4,
    pitch_deg: float = 6.0,
) -> np.ndarray:
    """One-shot helper: stitch an IPM canvas with default calibration."""
    cal = IPMCalibration(
        K=K,
        camera_height_m=camera_height_m,
        pitch_deg=pitch_deg,
    )
    return stitch_bev_along_trajectory(
        frames, trajectory_xz, cal, keyframe_stride=keyframe_stride
    )
