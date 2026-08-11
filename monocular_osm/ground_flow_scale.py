"""Per-step vehicle speed from road-plane optical flow.

Monocular visual odometry normalises every relative translation to unit
length, so the recovered trajectory encodes speed as a one-bit
moving/stopped flag: our VO step magnitudes have a coefficient of
variation of 0.02-0.07 while true per-step lengths have 1.0-1.4. That is a
**constant-speed assumption** baked into the one thing
:mod:`monocular_osm.trajectory_matching` scores — the shape.

This module estimates a per-step length so the shape carries the real
speed profile. Measured on the 8-clip overlay fleet (240 s each, VO shape
similarity-fitted against each clip's own GPS track): replacing step
lengths with the *true* ones takes the mean error from 107.4 m to 77.6 m,
and this estimator recovers 61% of that on the clips it accepts.

Three properties make the problem much easier than it looks, all measured:

* **Absolute scale is worth exactly zero.** A constant 30% scale error
  scores 0.0 m — the Procrustes fit in the matcher absorbs any global
  factor. So ``camera_height_m`` being a guess costs nothing, and the
  quantity that matters is only the *relative* profile.
* **The estimate does not need to be accurate.** Corrupting the true
  lengths by 15-30% changes the score by under a metre. Freedom from
  *bias* matters; precision does not.
* **Only the horizon row matters geometrically.** The flat-plane model
  gives ``Z = f*h/(v - v_h)``, so ``f*h`` is that worthless global scale
  while ``v_h`` decides which pixels are near road and how their flow
  converts to metres. It is estimated here from the flow itself
  (:func:`calibrate_from_flow`) rather than assumed.

The estimator abstains rather than guessing: when the tracked points
disagree about how far the car moved, the flat-plane model is not
describing the footage (wet night asphalt, a lead vehicle filling the
road) and the result is faded back to the VO's own constant-speed
trajectory. On the fleet that abstention fires on 2 of 8 clips, and no
clip is made worse.

The geometry itself is :func:`monocular_osm.speed_scale.image_to_ground`,
which is exact and unit-tested; this module adds the tracking, the
self-calibration, and the per-step reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .speed_scale import image_to_ground

# Seeding band: a generous superset of "road plane between ~5 and ~30 m",
# valid for any pitch in 0-8 deg. Metric filtering happens after projection.
_ROI_TOP, _ROI_BOT = 0.55, 0.97
_ROI_LEFT, _ROI_RIGHT = 0.10, 0.90

_LK = dict(winSize=(21, 21), maxLevel=3)
_FB_TOL_PX = 2.0            # forward-backward round-trip agreement
_MAX_MPS = 45.0             # 160 km/h — beyond this the estimate is a misread

# Static-mask grid. Cells that stay still while the rest of the frame moves
# are the bonnet, dashboard reflections and the burned-in GPS/logo overlays
# these uploads carry. They are rigidly attached to the camera, so any median
# including them is dragged toward zero — the same failure that was voiding
# whole clips in visual_odometry before it was fixed.
_GRID_X, _GRID_Y = 24, 18


@dataclass(frozen=True)
class RoadTracks:
    """Per-VO-step road-surface correspondences.

    ``pts`` is (M, 4) — ``start_u, start_v, end_u, end_v`` — and
    ``offsets[j]:offsets[j+1]`` selects step ``j``'s rows. ``size`` is the
    ``(width, height)`` the tracking ran at.
    """
    pts: np.ndarray
    offsets: np.ndarray
    fps: float
    size: tuple[int, int]
    stride: int = 6

    @property
    def n_steps(self) -> int:
        return max(0, len(self.offsets) - 1)

    @property
    def step_seconds(self) -> float:
        """Wall-clock duration of one VO step."""
        return self.stride / max(self.fps, 1e-6)


@dataclass(frozen=True)
class PlaneCalibration:
    """Road-plane geometry recovered from the flow alone."""
    v_horizon: float
    pitch_deg: float
    static_mask: np.ndarray
    sharpness: float       # how well-defined the horizon minimum was, 0..1


def track_road_features(
    video_path: Path,
    *,
    stride: int,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    size: tuple[int, int] = (960, 540),
    max_corners: int = 400,
) -> RoadTracks:
    """Track road-surface features across each VO step, streaming.

    Features are seeded on the frame that starts a step and chained through
    the ``stride`` intermediate source frames to the frame that ends it,
    with a full backward chain as the consistency check.

    Chaining is not optional. At 30 fps a road point 20 m ahead moves about
    0.1 px between adjacent frames — below the LK noise floor, so the
    per-step median collapses to zero and the estimator degenerates to
    exactly the constant speed it exists to replace. Across a whole step it
    moves 4-20 px, which is real signal. Tracking the two step-boundary
    frames *directly* (which is all the pipeline's strided frame list
    holds) measurably loses most of the benefit: -3.8% versus -9.4% on the
    fleet, and it regresses a clip. Hence this reads the video itself,
    keeping only a rolling ``stride + 1`` frame buffer — retaining every
    frame of a 4-minute clip would cost tens of gigabytes.
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    w, h = size
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        first = int(start_sec * fps)
        last = int(end_sec * fps) if end_sec is not None else None
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        roi = np.zeros((h, w), np.uint8)
        roi[int(h * _ROI_TOP):int(h * _ROI_BOT),
            int(w * _ROI_LEFT):int(w * _ROI_RIGHT)] = 255

        rows_per_step: list[np.ndarray] = []
        offsets = [0]
        buf: list[np.ndarray] = []
        idx = first
        while last is None or idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2GRAY))
            idx += 1
            if len(buf) < stride + 1:
                continue
            rows = _step_correspondences(buf, roi, max_corners, (w, h))
            rows_per_step.append(rows)
            offsets.append(offsets[-1] + len(rows))
            buf = [buf[-1]]         # the step's end frame starts the next one
    finally:
        cap.release()

    pts = (np.concatenate(rows_per_step) if rows_per_step
           else np.zeros((0, 4), np.float32))
    return RoadTracks(pts=pts, offsets=np.asarray(offsets, np.int64),
                      fps=float(fps), size=(w, h), stride=stride)


def _step_correspondences(buf, roi, max_corners, size) -> np.ndarray:
    import cv2

    w, h = size
    seed = cv2.goodFeaturesToTrack(buf[0], maxCorners=max_corners,
                                   qualityLevel=0.003, minDistance=6,
                                   mask=roi, blockSize=5)
    if seed is None or len(seed) < 8:
        return np.zeros((0, 4), np.float32)
    seed = seed.reshape(-1, 2).astype(np.float32)
    fwd, keep_f = _chain(buf, seed, size)
    if not len(fwd):
        return np.zeros((0, 4), np.float32)
    back, keep_b = _chain(buf[::-1], fwd, size)
    if not len(back):
        return np.zeros((0, 4), np.float32)
    start = seed[keep_f[keep_b]]
    agrees = np.linalg.norm(back - start, axis=1) < _FB_TOL_PX
    return np.column_stack([start[agrees], fwd[keep_b][agrees]]).astype(np.float32)


def _chain(frames, points, size):
    """LK-chain `points` through `frames`; returns survivors and their indices."""
    import cv2

    w, h = size
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    cur, idx = points, np.arange(len(points))
    for a, b in zip(frames, frames[1:]):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, cur, None,
                                                  criteria=criteria, **_LK)
        ok = status.ravel() == 1
        ok &= ((nxt[:, 0] > 1) & (nxt[:, 0] < w - 2)
               & (nxt[:, 1] > 1) & (nxt[:, 1] < h - 2))
        cur, idx = nxt[ok], idx[ok]
        if not len(cur):
            break
    return cur, idx


# --- self-calibration ------------------------------------------------------


def _static_cells(tracks: RoadTracks, *, rel=0.25, frac=0.6, min_obs=15):
    """Grid cells that stay still while the rest of the frame moves."""
    w, h = tracks.size
    still = np.zeros((_GRID_Y, _GRID_X))
    seen = np.zeros((_GRID_Y, _GRID_X))
    for start, end in zip(tracks.offsets, tracks.offsets[1:]):
        if end - start < 40:
            continue
        p = tracks.pts[start:end]
        moved = np.linalg.norm(p[:, 2:] - p[:, :2], axis=1)
        reference = np.percentile(moved, 75)
        if np.percentile(moved, 90) < 4.0 or reference < 1e-3:
            continue                                  # the vehicle is stopped
        gx = np.clip((p[:, 0] / w * _GRID_X).astype(int), 0, _GRID_X - 1)
        gy = np.clip((p[:, 1] / h * _GRID_Y).astype(int), 0, _GRID_Y - 1)
        for cell in np.unique(gy * _GRID_X + gx):
            sel = (gy * _GRID_X + gx) == cell
            if sel.sum() < 4:
                continue
            j, i = divmod(int(cell), _GRID_X)
            seen[j, i] += 1
            if np.median(moved[sel]) < rel * reference:
                still[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        fired = np.where(seen >= min_obs, still / np.maximum(seen, 1), 0.0)
    return fired >= frac


def _outside_static(pts, mask, size):
    w, h = size
    gx = np.clip((pts[:, 0] / w * _GRID_X).astype(int), 0, _GRID_X - 1)
    gy = np.clip((pts[:, 1] / h * _GRID_Y).astype(int), 0, _GRID_Y - 1)
    return ~mask[gy, gx]


def _horizon_cost(sets, candidates, margin):
    cost = np.zeros(len(candidates))
    for v_start, v_end in sets:
        for i, v_h in enumerate(candidates):
            below = v_start > v_h + margin
            if below.sum() < 15:
                cost[i] += 1.0
                continue
            # Z ∝ 1/(v - v_h), so this is each point's implied travel.
            implied = 1.0 / (v_start[below] - v_h) - 1.0 / (v_end[below] - v_h)
            med = np.median(implied)
            cost[i] += (np.median(np.abs(implied - med)) / med) if med > 0 else 1.0
    return cost / max(len(sets), 1)


def calibrate_from_flow(tracks: RoadTracks, fx: float, cy: float,
                        *, max_steps: int = 200) -> PlaneCalibration:
    """Recover the horizon row — and hence pitch — from the flow itself.

    On a flat plane every in-plane point implies the SAME forward travel, so
    the horizon row is the one that makes those implied travels most
    consistent within a step. No GPS, no per-clip hand tuning.

    ``sharpness`` reports how well-defined that minimum was. Footage with no
    usable ground — camera aimed high, hills, no road texture — gives a flat
    or monotone curve, and a horizon that was never pinned down makes the
    metric conversion meaningless however tidy the residuals look.
    """
    w, h = tracks.size
    mask = _static_cells(tracks)
    keep = _outside_static(tracks.pts, mask, tracks.size)

    sets = []
    for start, end in zip(tracks.offsets, tracks.offsets[1:]):
        if end - start < 40:
            continue
        p = tracks.pts[start:end]
        v0, v1 = p[:, 1], p[:, 3]
        sel = (keep[start:end] & ((v1 - v0) > 1.5)
               & (np.abs(p[:, 0] - w / 2) < w * 0.45))
        if sel.sum() >= 25:
            sets.append((v0[sel], v1[sel]))
    if len(sets) < 10:
        return PlaneCalibration(cy, 0.0, mask, 0.0)
    if len(sets) > max_steps:
        sets = [sets[i] for i in
                np.linspace(0, len(sets) - 1, max_steps).astype(int)]

    lo, hi, margin = h * 0.37, h * 0.85, h / 21.6
    coarse = np.arange(lo, hi, h / 90.0)
    cost = _horizon_cost(sets, coarse, margin)
    best = int(np.argmin(cost))
    sharpness = float((np.mean(cost) - cost[best]) / max(np.mean(cost), 1e-6))
    if best in (0, len(cost) - 1):
        sharpness = 0.0        # the minimum is at the edge: not a real basin
    around = coarse[best]
    fine = np.arange(max(lo, around - h / 90.0), min(hi, around + h / 90.0) + 1, 1.0)
    v_h = float(fine[int(np.argmin(_horizon_cost(sets, fine, margin)))])
    return PlaneCalibration(
        v_horizon=v_h,
        pitch_deg=float(np.degrees(np.arctan((cy - v_h) / fx))),
        static_mask=mask,
        sharpness=sharpness,
    )


# --- per-step reduction ----------------------------------------------------


def _forward_motion(tracks, cal, K, *, camera_height_m, near_m, far_m, lat_m,
                    min_pts, mad_k):
    """Yaw-compensated forward travel (m) per step; nan where unusable.

    The same reduction as :func:`monocular_osm.speed_scale._pair_motion` —
    remove the median bearing change, then take the median forward closing
    distance — vectorised over all steps, plus a MAD reject so a vehicle
    occupying the road ahead cannot drag the median.
    """
    keep = _outside_static(tracks.pts, cal.static_mask, tracks.size)
    g0 = image_to_ground(tracks.pts[:, :2], K, camera_height_m=camera_height_m,
                         pitch_deg=cal.pitch_deg)
    g1 = image_to_ground(tracks.pts[:, 2:], K, camera_height_m=camera_height_m,
                         pitch_deg=cal.pitch_deg)
    usable = (np.isfinite(g0).all(1) & np.isfinite(g1).all(1)
              & (g0[:, 1] >= near_m) & (g0[:, 1] <= far_m)
              & (np.abs(g0[:, 0]) <= lat_m) & keep)
    bearing0 = np.arctan2(g0[:, 0], g0[:, 1])
    bearing1 = np.arctan2(g1[:, 0], g1[:, 1])
    dyaw = (bearing1 - bearing0 + np.pi) % (2 * np.pi) - np.pi

    n = tracks.n_steps
    motion = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    for k, (start, end) in enumerate(zip(tracks.offsets, tracks.offsets[1:])):
        sel = usable[start:end]
        if sel.sum() < min_pts:
            continue
        a0, a1 = g0[start:end][sel], g1[start:end][sel]
        yaw = float(np.median(dyaw[start:end][sel]))
        c, s = np.cos(yaw), np.sin(yaw)
        z_derotated = a1[:, 1] * c + a1[:, 0] * s
        travel = a0[:, 1] - z_derotated
        med = float(np.median(travel))
        # On a flat plane every in-plane point implies the same travel, so
        # the spread is a GPS-free confidence: tight = real road surface,
        # wide = reflections, a lead vehicle, or no texture at all.
        spread[k] = float(np.median(np.abs(travel - med))) / max(med, 1e-3)
        if mad_k:
            mad = np.median(np.abs(travel - med)) + 1e-3
            inliers = np.abs(travel - med) <= mad_k * mad
            if inliers.sum() >= min_pts:
                travel = travel[inliers]
        motion[k] = float(np.median(travel))
    return motion, spread


def step_lengths(
    tracks: RoadTracks,
    K: np.ndarray,
    *,
    step_directions_valid: np.ndarray,
    fallback_lengths: np.ndarray | None = None,
    camera_height_m: float = 1.4,
    near_m: float = 10.0,
    far_m: float = 20.0,
    lat_m: float = 8.0,
    min_pts: int = 10,
    mad_k: float = 3.0,
    accept_spread: float = 0.25,
    reject_spread: float = 0.55,
    min_sharpness: float = 0.30,
    full_sharpness: float = 0.45,
    calibration: PlaneCalibration | None = None,
) -> tuple[np.ndarray, float]:
    """Per-VO-step length in metres, plus the confidence it was given.

    Returns ``(lengths, confidence)``. ``confidence`` below 1 fades the
    result toward ``fallback_lengths`` (the VO's own step lengths), so a
    clip the plane model cannot describe reproduces the previous behaviour
    *exactly* rather than being made worse. At confidence 0 the fallback is
    returned unchanged.

    ``step_directions_valid`` marks steps where the VO found motion; the
    rest are genuine stops and get length 0.
    """
    n = len(step_directions_valid)
    fallback = (np.ones(n) if fallback_lengths is None
                else np.asarray(fallback_lengths, float))
    if tracks.n_steps == 0:
        return np.where(step_directions_valid, fallback, 0.0), 0.0

    cal = calibration or calibrate_from_flow(tracks, float(K[0, 0]), float(K[1, 2]))
    motion, spread = _forward_motion(
        tracks, cal, K, camera_height_m=camera_height_m, near_m=near_m,
        far_m=far_m, lat_m=lat_m, min_pts=min_pts, mad_k=mad_k)

    # A step cannot cover more ground than a road vehicle can travel in it.
    max_step_m = _MAX_MPS * tracks.step_seconds
    motion = np.where(np.isfinite(motion) & (motion < max_step_m), motion, np.nan)
    motion = np.clip(motion, 0.0, None)      # reversing is not a thing here

    lengths = np.full(n, np.nan)
    take = min(n, len(motion))
    lengths[:take] = motion[:take]
    missing = ~np.isfinite(lengths)
    if missing.all():
        return np.where(step_directions_valid, fallback, 0.0), 0.0
    if missing.any():
        idx = np.arange(n)
        lengths[missing] = np.interp(idx[missing], idx[~missing], lengths[~missing])
    lengths = np.where(step_directions_valid, lengths, 0.0)

    confidence = _confidence(motion, spread, cal, accept_spread, reject_spread,
                            min_sharpness, full_sharpness)
    if confidence < 1.0:
        lengths = _blend(lengths, fallback, confidence, step_directions_valid)
    return lengths, confidence


def _confidence(motion, spread, cal, accept_spread, reject_spread,
                min_sharpness, full_sharpness) -> float:
    """How much to trust the plane model on this clip, in [0, 1].

    Judged only on the steps that actually moved: a *relative* spread is
    meaningless when the denominator is a car standing still.
    """
    usable = np.isfinite(spread) & np.isfinite(motion)
    if usable.sum() > 20:
        threshold = np.percentile(motion[usable], 70)
        fast = usable & (motion >= max(threshold, 1e-3))
        typical = float(np.median(spread[fast])) if fast.sum() > 10 else 1.0
    else:
        typical = 1.0
    conf = np.clip((reject_spread - typical)
                   / max(reject_spread - accept_spread, 1e-6), 0.0, 1.0)
    conf *= np.clip((cal.sharpness - min_sharpness)
                    / max(full_sharpness - min_sharpness, 1e-6), 0.0, 1.0)
    return float(conf)


def _blend(lengths, fallback, confidence, moving) -> np.ndarray:
    """Fade toward the fallback, matching total distance so only SHAPE changes."""
    a, b = lengths[moving].sum(), fallback[moving].sum()
    if a <= 1e-6 or b <= 1e-6:
        return np.where(moving, fallback, 0.0)
    return confidence * lengths * (b / a) + (1.0 - confidence) * fallback


def rescale_steps(xz: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Rebuild a trajectory keeping each step's DIRECTION, replacing its LENGTH.

    The VO chain gives every accepted step unit length, so a world-frame
    step ``xz[i] - xz[i-1]`` already carries the direction and nothing else.
    """
    xz = np.asarray(xz, float)
    steps = np.diff(xz, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    directions = np.zeros_like(steps)
    moving = norms > 1e-9
    directions[moving] = steps[moving] / norms[moving, None]
    scaled = directions * np.asarray(lengths, float)[:, None]
    return np.vstack([xz[0], xz[0] + np.cumsum(scaled, axis=0)])
