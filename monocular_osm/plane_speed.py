"""Metric step lengths for monocular VO from the road plane and the camera height.

The VO (``visual_odometry``) recovers each step's rotation and translation
*direction* but normalises every translation to length 1, so it has no speed;
on KITTI that, not the map matching, is what keeps the road-graph particle
filter from localizing (docs/graph_filter_kitti.md: with VO headings and the
true speed it localizes drive 0033 in 29 s; with unit steps it does not).

For points on the road, two views are related by the plane-induced homography
``x2 ~ (R + t m^T) x1`` in normalised camera coordinates, where the road is
``n^T X = d`` in the first camera's frame and ``m = n / d`` (Hartley &
Zisserman, Multiple View Geometry, sec. 13.1). With the VO's ``R`` and unit
``t``, ``m`` is *linear* in the road correspondences, so it is estimated by
RANSAC (``skimage.measure.ransac``) per frame pair with no new model. The
camera then sits ``d = 1/|m|`` step-lengths above the road, so a step is
``L = h |m| |t|`` metres for a camera ``h`` metres above it — the classic
planar-road scale recovery for monocular driving odometry (Kitt, Geiger &
Lategahn, IV 2010; Song & Chandraker, CVPR 2014).

``h`` is the one assumed number, and it only sets a global scale: a 10 %
error in ``h`` is a 10 % error in every step, which the filter's per-hypothesis
scale state absorbs. The *profile* — stops, accelerations, the distance
between turns — is measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A typical windscreen dashcam is 1.2-1.7 m above the road; 1.5 m is the
#: deployable assumption (KITTI's camera is at 1.65 m, so this reads ~9 % short
#: there: an honest test of the global-scale prior, not a tuned value).
DEFAULT_CAMERA_HEIGHT_M = 1.5


@dataclass
class PlaneSpeed:
    """Per-VO-step metric lengths (``len(traj.xz) - 1`` entries)."""

    lengths_m: np.ndarray      # smoothed, gap-filled; 0 where the VO step is 0
    raw_m: np.ndarray          # per-pair estimate before smoothing; NaN = none
    inliers: np.ndarray        # RANSAC inliers per pair (0 = no estimate)
    coverage: float            # fraction of moving VO steps with a raw estimate
    rel_spread: float          # median |raw/smoothed - 1|: GT-free noise level
    camera_height_m: float

    def to_dict(self) -> dict:
        return {"coverage": self.coverage, "rel_spread": self.rel_spread,
                "camera_height_m": self.camera_height_m,
                "n_steps": int(len(self.lengths_m))}


def _skew(v: np.ndarray) -> np.ndarray:
    """Batched cross-product matrices: (N, 3) -> (N, 3, 3)."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1),
                     np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


def solve_plane(x1: np.ndarray, x2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least-squares ``m`` with ``x2 ~ (R + t m^T) x1`` (normalised homogeneous
    points, (N, 3) each): ``[x2]_x t x1^T m = -[x2]_x R x1``, two independent
    rows per point."""
    S = _skew(x2)                                   # (N, 3, 3)
    A = np.einsum("nij,j,nk->nik", S, t, x1)        # (N, 3, 3)
    b = -np.einsum("nij,jk,nk->ni", S, R, x1)       # (N, 3)
    m, *_ = np.linalg.lstsq(A.reshape(-1, 3), b.reshape(-1), rcond=None)
    return m


def transfer_error(m: np.ndarray, x1: np.ndarray, x2: np.ndarray, R: np.ndarray,
                   t: np.ndarray) -> np.ndarray:
    """Distance (normalised image units) between ``H x1`` and ``x2``."""
    y = x1 @ (R + np.outer(t, m)).T
    with np.errstate(divide="ignore", invalid="ignore"):
        y = y[:, :2] / y[:, 2:3]
        res = np.hypot(*(y - x2[:, :2] / x2[:, 2:3]).T)
    return np.where(np.isfinite(res), res, np.inf)


def _model_for(R: np.ndarray, t: np.ndarray):
    """A ``skimage.measure.ransac`` model class for one frame pair's (R, t)."""

    class _RoadPlane:
        def __init__(self, m=None):
            self.m = m

        def __bool__(self) -> bool:
            return self.m is not None and bool(np.all(np.isfinite(self.m)))

        @classmethod
        def from_estimate(cls, x1, x2):
            try:
                return cls(solve_plane(x1, x2, R, t))
            except np.linalg.LinAlgError:
                return cls(None)

        def residuals(self, x1, x2):
            return transfer_error(self.m, x1, x2, R, t)

    return _RoadPlane


def road_mask(shape: tuple[int, int], K: np.ndarray, *, below_horizon_frac: float = 0.12,
              bottom_frac: float = 0.06, side_frac: float = 0.12) -> np.ndarray:
    """uint8 mask of the image region where the road is expected: below the
    principal point by a margin, above the bonnet, away from the edges."""
    h, w = shape
    top = int(min(h - 2, K[1, 2] + below_horizon_frac * h))
    mask = np.zeros((h, w), np.uint8)
    mask[top:int(h * (1 - bottom_frac)), int(w * side_frac):int(w * (1 - side_frac))] = 255
    return mask


def foe_horizon_row(traj, K: np.ndarray, min_steps: int = 20) -> float | None:
    """Image row of the horizon, from the VO's own motion: a car moves parallel
    to the road, so its focus of expansion (the image of each step's direction
    in that frame's camera) lies on the horizon. Median over moving steps.

    The default road band assumes a level camera (horizon at the principal
    point). Dashcams are often pitched up: on an Atlanta overlay clip the road
    fills only the bottom ~15 % of the frame, so the level-camera band covered
    buildings and parked cars, and the plane fit read speed ~7x short."""
    C = np.asarray(traj.centers, float)
    R = np.asarray(traj.rotations, float)
    valid = np.asarray(traj.valid, bool)
    rows = []
    for i in range(len(C) - 1):
        if not valid[i + 1]:
            continue
        d = R[i] @ (C[i + 1] - C[i])
        if d[2] > 1e-6 * max(np.linalg.norm(d), 1e-12):
            rows.append(K[1, 1] * d[1] / d[2] + K[1, 2])
    return float(np.median(rows)) if len(rows) >= min_steps else None


def static_overlay_mask(frames_gray, *, n_sample: int = 60, std_max: float = 3.0,
                        dilate_px: int = 9) -> np.ndarray:
    """uint8 mask (255 = static) of pixels that barely change over the clip:
    burned-in timestamps, speed/GPS stamps and logos. Their features do not
    move with the road, so inside the road band they pull the plane fit
    towards infinity (speed too short). Frames are sampled across the whole
    clip, so a stop at a light does not make the scene look static."""
    import cv2

    idx = np.linspace(0, len(frames_gray) - 1, min(n_sample, len(frames_gray))).astype(int)
    stack = np.stack([np.asarray(frames_gray[i], np.float32) for i in idx])
    static = (stack.std(axis=0) < std_max).astype(np.uint8) * 255
    if dilate_px > 0:
        static = cv2.dilate(static, np.ones((dilate_px, dilate_px), np.uint8))
    return static


def pair_correspondences(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray,
                         *, max_corners: int = 400, fb_max_px: float = 1.0):
    """Shi-Tomasi corners in ``mask`` tracked img1 -> img2 by pyramidal
    Lucas-Kanade, kept when the backward track returns within ``fb_max_px``."""
    import cv2

    p1 = cv2.goodFeaturesToTrack(img1, maxCorners=max_corners, qualityLevel=0.005,
                                 minDistance=6, mask=mask, blockSize=7)
    if p1 is None or len(p1) < 8:
        return None, None
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p2, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p1, None, **lk)
    p1b, st_b, _ = cv2.calcOpticalFlowPyrLK(img2, img1, p2, None, **lk)
    fb = np.linalg.norm((p1 - p1b).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < fb_max_px)
    return p1.reshape(-1, 2)[ok], p2.reshape(-1, 2)[ok]


def _relative_pose(traj, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with X_j = R X_i + t, from the trajectory's world-to-camera poses."""
    Ri, Rj = np.asarray(traj.rotations[i], float), np.asarray(traj.rotations[j], float)
    ti, tj = np.asarray(traj.translations[i], float), np.asarray(traj.translations[j], float)
    R = Rj @ Ri.T
    return R, tj - R @ ti


def _nan_rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    half = win // 2
    out = np.full(len(x), np.nan)
    for k in range(len(x)):
        seg = x[max(0, k - half):k + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= max(2, win // 3):
            out[k] = np.median(seg)
    return out


def plane_step_lengths(
    frames_gray,
    traj,
    K: np.ndarray,
    *,
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
    max_normal_tilt_deg: float = 25.0,
    min_inliers: int = 20,
    residual_px: float = 1.0,
    smooth_steps: int = 9,
    max_step_m: float = 6.0,
    below_horizon_frac: float = 0.12,
    horizon: str = "principal",
    mask_static: bool = False,
    seed: int = 0,
) -> PlaneSpeed:
    """Metric length of every VO step from the road plane (see module doc).

    ``frames_gray[i]`` must be the image the VO used for pose ``i``. A pair
    yields an estimate when RANSAC finds ``min_inliers`` road points whose
    plane normal is within ``max_normal_tilt_deg`` of the camera's down axis
    (rejecting walls and vehicles). Raw estimates are median-smoothed over
    ``smooth_steps`` steps (speed changes slowly; single pairs are noisy) and
    gaps are filled by linear interpolation; steps the VO marks stationary or
    invalid keep length 0, as in ``graph_filter.odometry_from_trajectory``.
    """
    from skimage.measure import ransac

    n = len(traj.xz)
    Kinv = np.linalg.inv(np.asarray(K, float))
    unit = np.linalg.norm(np.diff(np.asarray(traj.xz, float), axis=0), axis=1)
    moving = np.asarray(traj.valid, bool)[1:] & (unit > 1e-6)
    raw = np.full(n - 1, np.nan)
    inl = np.zeros(n - 1, int)
    cos_tilt = np.cos(np.deg2rad(max_normal_tilt_deg))
    shape = np.asarray(frames_gray[0]).shape[:2]
    mask = road_mask(shape, K, below_horizon_frac=below_horizon_frac)
    if horizon == "foe":
        # Road band from a quarter of the way below the VO's own horizon down
        # to the bonnet margin (fixed before use; see foe_horizon_row).
        hz = foe_horizon_row(traj, K)
        if hz is not None and 0 <= hz < shape[0] - 4:
            h = shape[0]
            band = np.zeros(shape, np.uint8)
            top = int(hz + 0.25 * (h * 0.94 - hz))
            band[top:int(h * 0.94), int(shape[1] * 0.12):int(shape[1] * 0.88)] = 255
            mask = band
    elif horizon != "principal":
        raise ValueError(f"horizon must be 'principal' or 'foe', not {horizon!r}")
    if mask_static:
        mask = np.where(static_overlay_mask(frames_gray) > 0, 0, mask).astype(np.uint8)
    rng = np.random.default_rng(seed)
    for i in range(n - 1):
        if not moving[i]:
            continue
        R, t = _relative_pose(traj, i, i + 1)
        tn = float(np.linalg.norm(t))
        if tn < 1e-9:
            continue
        u1, u2 = pair_correspondences(frames_gray[i], frames_gray[i + 1], mask)
        if u1 is None or len(u1) < min_inliers:
            continue
        x1 = np.column_stack([u1, np.ones(len(u1))]) @ Kinv.T
        x2 = np.column_stack([u2, np.ones(len(u2))]) @ Kinv.T
        model, inliers = ransac((x1, x2), _model_for(R, t), min_samples=3,
                                residual_threshold=residual_px / float(K[0, 0]),
                                max_trials=200, rng=rng)
        if not model or inliers is None or inliers.sum() < min_inliers:
            continue
        m = solve_plane(x1[inliers], x2[inliers], R, t)     # refit on all inliers
        mn = float(np.linalg.norm(m))
        if not np.isfinite(mn) or mn <= 0 or m[1] / mn < cos_tilt:
            continue                       # not a road below the camera
        L = camera_height_m * mn * tn
        if 0.0 < L <= max_step_m:
            raw[i] = L
            inl[i] = int(inliers.sum())

    sm = _nan_rolling_median(raw, smooth_steps)
    idx = np.arange(n - 1)
    have = np.isfinite(sm)
    if have.any():
        sm = np.interp(idx, idx[have], sm[have])
    else:
        sm = np.zeros(n - 1)
    lengths = np.where(moving, sm, 0.0)
    est = np.isfinite(raw) & moving & (lengths > 0)
    spread = (float(np.median(np.abs(raw[est] / lengths[est] - 1.0)))
              if est.sum() >= 10 else float("nan"))
    cov = float(est.sum() / max(1, moving.sum()))
    return PlaneSpeed(lengths_m=lengths, raw_m=raw, inliers=inl, coverage=cov,
                      rel_spread=spread, camera_height_m=float(camera_height_m))
