"""3D-tile skyline channel: render an untextured LoD2 city mesh at
sampled poses along each candidate route and score agreement between
the rendered building skyline and the skyline observed in the video.

Why skylines, not features: against untextured LoD2 geometry, texture
matchers (ORB/learned features) have nothing to grip — the literature
line from SKYLINE2GPS (IROS'10) through Arth et al. (ISMAR'15) to
LoD-Loc v2 (ICCV'25) converges on building silhouettes/rooflines as
the robust cue, with LoD-Loc showing silhouette alignment against
LoD1/LoD2 can even beat textured-model baselines. This module is the
candidate-re-ranking version of that idea: no pose optimization, just
"does the city model's skyline look like the video's skyline along
this candidate route?".

Everything here is numpy + cv2 (no GL, no torch): LoD2 buildings are
few enough that projecting triangles and cv2.fillConvexPoly-ing them
into a binary mask is fast, and a skyline needs no depth buffer — the
union of projected triangles suffices (occlusion cannot change the
topmost edge).

Scoring design (shaped by an adversarial review of v1):

- ONE pitch offset per candidate (median over samples), not one per
  sample — the camera's pitch is a single physical constant, and
  refitting it per sample let wrong candidates absorb inconsistent
  offsets for free. Offsets beyond a plausible dashcam pitch are
  softly penalized.
- FIXED score denominator over all usable video samples. v1 skipped
  "both skylines flat" samples from the mean, so a wrong candidate
  with one lucky sample outscored a right candidate with eight decent
  ones; flat samples now contribute at half credit instead.
- Column-level CONTRADICTIONS are penalized: model building where the
  video sees open sky (and vice versa) — the open-field-vs-buildings
  asymmetry is the discriminative signal, so it must be scored, not
  silently dropped from the valid mask.
- The video skyline comes from a top-connected SMOOTHNESS grow, not a
  global sky-color threshold: a graded clear sky stays "open" all the
  way down instead of fabricating a flat phantom skyline where the
  gradient leaves the seed color.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "Tile3DMatchResult", "SkylineComparison", "render_building_mask",
    "skyline_from_mask", "skyline_from_frame", "compare_skylines",
    "arc_fractions_at_times", "pose_at_fraction",
    "match_tiles3d_against_candidates",
    "walk_coverage_fractions", "tile3d_tiebreak_winner",
    "adaptive_tile3d_weight", "refine_placement_skyline",
]

# Comparison constants (degrees of elevation angle unless noted).
_ERR_SCALE_DEG = 3.0        # e-folding scale of the skyline-error score
_MIN_SHARED_FRAC = 0.08     # min fraction of columns shared by both skylines
_FLAT_STD_DEG = 0.4         # both-flat => weak (half-credit) sample
_FLAT_CREDIT = 0.5          # score multiplier for both-flat samples
_EXPLAIN_SAT = 0.6          # frame-skyline fraction the model must explain
_CONTRA_WEIGHT = 0.7        # score removed at 100% column contradiction
_HORIZON_ELEV_DEG = 0.5     # at/below this elevation counts as "open"
_MODEL_CONTRA_ELEV = 2.0    # model building must clear this to contradict
_PITCH_SOFT_LIMIT = 10.0    # |candidate pitch offset| beyond this decays
_PITCH_DECAY_DEG = 6.0      # ... with this e-folding scale
_SKY_STEP_LAB = 14.0        # max row-to-row Lab change inside sky
_SKY_SEED_LAB = 45.0        # top-row similarity to the top-band seed
_DARK_L = 60.0              # top band darker than this: unusable frame


@dataclass
class Tile3DMatchResult:
    """Skyline agreement of one candidate with the LoD2 model."""
    candidate_index: int
    tile3d_score: float            # [0, 1], higher = better
    skyline_err_deg: float | None  # median per-sample skyline error
    pitch_offset_deg: float | None  # fitted per-candidate pitch offset
    coverage: float                # mean explained-skyline fraction
    n_samples_scored: int          # usable video samples (fixed denominator)
    n_informative: int             # samples with non-flat shared structure


@dataclass
class SkylineComparison:
    """Column-level comparison of one rendered vs one observed skyline."""
    delta: np.ndarray | None   # model - frame elevation over shared columns
    shared_frac: float         # shared columns / all columns
    explain_frac: float        # shared columns / frame structure columns
    contradiction_frac: float  # open-vs-building disagreement fraction
    informative: bool          # shared structure is non-flat on either side


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int],
                     dst_wh: tuple[int, int]) -> np.ndarray:
    """Rescale a pinhole K between image resolutions."""
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    K2 = K.astype(np.float64).copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    return K2


# --------------------------------------------------------------------------
# Rendering (silhouette only — no z-buffer needed for a skyline)
# --------------------------------------------------------------------------

def _clip_near(tri_cam: np.ndarray, near: float) -> list[np.ndarray]:
    """Sutherland-Hodgman clip of one camera-space triangle against the
    near plane z=near. Returns 0..1 convex polygons with 3..4 vertices."""
    poly = tri_cam
    res: list[np.ndarray] = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        a_in, b_in = a[2] > near, b[2] > near
        if a_in:
            res.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            res.append(a + t * (b - a))
    return [np.asarray(res)] if len(res) >= 3 else []


def render_building_mask(
    triangles: np.ndarray,
    cam_xy: np.ndarray,
    cam_z: float,
    heading: np.ndarray,
    K: np.ndarray,
    wh: tuple[int, int],
    *,
    max_dist_m: float = 500.0,
    near_m: float = 2.0,
) -> np.ndarray:
    """Binary building-silhouette mask (H, W) uint8 from a ground pose.

    ``heading`` is the 2D forward direction (dx, dy) in world (east,
    north) metres; the camera looks along it with zero pitch/roll and
    +z up. Culling is per-VERTEX (nearest vertex within range, any
    vertex ahead), so large ear-clipped LoD2 triangles that straddle
    the range or camera plane are kept, not silently dropped.
    """
    w, h = wh
    mask = np.zeros((h, w), dtype=np.uint8)
    if not len(triangles):
        return mask
    f2 = np.asarray(heading, dtype=np.float64)
    norm = np.linalg.norm(f2)
    f2 = f2 / norm if norm > 1e-9 else np.array([0.0, 1.0])
    right2 = np.array([f2[1], -f2[0]])

    rel = triangles[:, :, :2] - np.asarray(cam_xy, dtype=np.float64)[None, None, :]
    vdist = np.hypot(rel[:, :, 0], rel[:, :, 1])
    vahead = rel[:, :, 0] * f2[0] + rel[:, :, 1] * f2[1]
    keep = (vdist.min(axis=1) < max_dist_m) & (vahead.max(axis=1) > 0.0)
    if not np.any(keep):
        return mask
    tris = triangles[keep].astype(np.float64)

    # world -> camera (x right, y up, z forward); y flips at projection
    p0 = np.array([cam_xy[0], cam_xy[1], cam_z], dtype=np.float64)
    d = tris - p0[None, None, :]
    xc = d[:, :, 0] * right2[0] + d[:, :, 1] * right2[1]
    yc = d[:, :, 2]
    zc = d[:, :, 0] * f2[0] + d[:, :, 1] * f2[1]
    cam = np.stack([xc, yc, zc], axis=2)           # (T, 3, 3)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def _project(poly_cam: np.ndarray) -> np.ndarray:
        z = np.maximum(poly_cam[:, 2], 1e-6)
        u = fx * poly_cam[:, 0] / z + cx
        v = cy - fy * poly_cam[:, 1] / z
        pts = np.stack([u, v], axis=1)
        return np.clip(np.round(pts), -32000, 32000).astype(np.int32)

    zmin = cam[:, :, 2].min(axis=1)
    zmax = cam[:, :, 2].max(axis=1)
    safe = zmin > near_m
    # Project every fully-in-front triangle in ONE vectorized pass instead
    # of a Python-level _project call each: same rounding/clip, so the mask
    # is identical, but ~37k per-call allocations per pose collapse to one.
    cam_safe = cam[safe]
    polys: list[np.ndarray] = []
    if len(cam_safe):
        zs = np.maximum(cam_safe[:, :, 2], 1e-6)
        us = fx * cam_safe[:, :, 0] / zs + cx
        vs = cy - fy * cam_safe[:, :, 1] / zs
        pts = np.stack([us, vs], axis=2)
        polys = list(np.clip(np.round(pts), -32000, 32000).astype(np.int32))
    for tri_cam in cam[(~safe) & (zmax > near_m)]:   # straddles near plane
        for poly in _clip_near(tri_cam, near_m):
            polys.append(_project(poly))
    # One fill call per polygon: a single fillPoly call with many
    # contours uses the even-odd rule, XOR-ing overlapping faces into
    # wireframe artifacts. Every polygon here is convex (triangle or
    # near-plane-clipped triangle), so fillConvexPoly is exact.
    for poly in polys:
        cv2.fillConvexPoly(mask, poly, 255)
    return mask


def skyline_from_mask(mask: np.ndarray) -> np.ndarray:
    """Per-column topmost building row of a silhouette mask.

    Returns (W,) float rows; NaN where the column has no building.
    """
    occupied = mask > 0
    has = occupied.any(axis=0)
    top = np.argmax(occupied, axis=0).astype(np.float64)
    top[~has] = np.nan
    return top


def skyline_from_frame(frame_bgr: np.ndarray,
                       wh: tuple[int, int]) -> np.ndarray:
    """Per-column skyline row observed in a video frame.

    Sky is grown from the top edge by row-to-row Lab SMOOTHNESS (sky
    gradients are gentle; a roofline is an abrupt transition), so a
    graded clear sky stays sky all the way down instead of fabricating
    a phantom skyline where its color drifts from the zenith seed.

    Returns (W,) float values: a finite row where a structure edge
    stops the sky; +inf where the sky runs to the bottom (open to the
    horizon — usable, but shows no structure); NaN where the column is
    unusable (top not sky-like: overpass, canopy, glare) — and all-NaN
    for a dark top band (night / tunnel).
    """
    w, h = wh
    img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Seed from the BRIGHT fraction of the top band: in a dense canyon
    # buildings can occupy most of the top rows, and a plain median
    # would lock onto building color and invert the segmentation
    # (dark facades read as "sky", real sky rejected).
    band = lab[: max(1, h // 12)].reshape(-1, 3)
    l_gate = max(float(np.percentile(band[:, 0], 60)), _DARK_L)
    bright = band[band[:, 0] >= l_gate]
    if not len(bright):
        return np.full(w, np.nan)   # night / tunnel: no sky-bright pixels
    seed = np.median(bright, axis=0)
    if seed[0] < _DARK_L:
        return np.full(w, np.nan)
    smooth = np.linalg.norm(np.diff(lab, axis=0), axis=2) < _SKY_STEP_LAB
    run = 1.0 + np.cumprod(smooth, axis=0).sum(axis=0)   # sky rows from top
    rows = run.astype(np.float64)
    top_ok = np.linalg.norm(lab[0] - seed[None, :], axis=1) < _SKY_SEED_LAB
    rows[~top_ok] = np.nan
    rows[rows <= 2] = np.nan          # immediate break: occluder at top
    rows[rows >= h] = np.inf          # smooth to the bottom: open sky
    return rows


def _elev_deg(rows: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Skyline rows -> elevation angle in degrees (above optical axis).

    +inf rows (open sky) map to -inf elevation; NaN passes through.
    """
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arctan2(K[1, 2] - rows, K[1, 1]))


def compare_skylines(
    elev_model_deg: np.ndarray,
    elev_frame_deg: np.ndarray,
) -> SkylineComparison:
    """Column-level skyline comparison (see :class:`SkylineComparison`).

    Both inputs use: finite = structure elevation, -inf = open sky,
    NaN = unusable column. Structure at or below the horizon
    (elevation < 0.5 deg) is treated as open on both sides so the
    ground/horizon line never masquerades as building structure.
    """
    n = len(elev_model_deg)
    # model: structure above the horizon, or open (NaN / -inf / below
    # horizon all mean "no building visible above the horizon here")
    m_fin = np.isfinite(elev_model_deg) & (elev_model_deg >= _HORIZON_ELEV_DEG)
    m_open = ~m_fin
    # frame: NaN is UNUSABLE (occluded top / night), open is -inf or
    # below-horizon structure (ground/horizon line, not buildings)
    f_fin = np.isfinite(elev_frame_deg) & (elev_frame_deg >= _HORIZON_ELEV_DEG)
    f_open = np.isneginf(elev_frame_deg) | (
        np.isfinite(elev_frame_deg) & (elev_frame_deg < _HORIZON_ELEV_DEG))
    f_usable = f_fin | f_open

    shared = m_fin & f_fin
    shared_frac = float(shared.mean()) if n else 0.0
    f_fin_n = int(f_fin.sum())
    explain_frac = float(shared.sum()) / f_fin_n if f_fin_n else 0.0

    contra = (m_fin & f_open & (elev_model_deg > _MODEL_CONTRA_ELEV)) \
        | (m_open & f_fin)
    usable_n = int(f_usable.sum())
    contradiction_frac = (float(contra[f_usable].sum()) / usable_n
                          if usable_n else 0.0)

    if shared.sum() < max(8, _MIN_SHARED_FRAC * n):
        return SkylineComparison(None, shared_frac, explain_frac,
                                 contradiction_frac, False)
    dm = elev_model_deg[shared]
    df = elev_frame_deg[shared]
    informative = not (np.std(dm) < _FLAT_STD_DEG
                       and np.std(df) < _FLAT_STD_DEG)
    return SkylineComparison(dm - df, shared_frac, explain_frac,
                             contradiction_frac, informative)


# --------------------------------------------------------------------------
# Candidate-route pose sampling
# --------------------------------------------------------------------------

def arc_fractions_at_times(traj_xy: np.ndarray, timestamps: np.ndarray,
                           t_secs: np.ndarray) -> np.ndarray:
    """Arc-length fraction [0, 1] traveled along ``traj_xy`` at each time.

    Bridges VO time to candidate geometry: ``MatchCandidate.
    aligned_traj_xy`` is a uniform-arc-length resampling of the VO
    path, so the point at fraction f is the camera position at the
    time the VO path had covered fraction f of its length.
    """
    traj_xy = np.asarray(traj_xy, dtype=np.float64)
    seg = np.linalg.norm(np.diff(traj_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:
        return np.zeros(len(np.atleast_1d(t_secs)))
    at_t = np.interp(np.atleast_1d(t_secs), timestamps, cum)
    return np.clip(at_t / total, 0.0, 1.0)


def pose_at_fraction(aligned_xy: np.ndarray,
                     frac: float) -> tuple[np.ndarray, np.ndarray]:
    """(position, forward direction) at an arc-length fraction of a path."""
    pts = np.asarray(aligned_xy, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9 or len(pts) < 2:
        return pts[0], np.array([0.0, 1.0])
    s = float(np.clip(frac, 0.0, 1.0)) * total
    j = int(np.searchsorted(cum, s, side="right") - 1)
    j = min(max(j, 0), len(seg) - 1)
    t = (s - cum[j]) / seg[j] if seg[j] > 1e-9 else 0.0
    pos = pts[j] + t * (pts[j + 1] - pts[j])
    # heading = this segment's direction; near a shared vertex, blend
    # with the adjacent segment (<=10 m radius) to damp resample noise
    # without smearing real 90-degree turns across whole segments
    d = (pts[j + 1] - pts[j]) / seg[j] if seg[j] > 1e-9 else np.array([0.0, 1.0])
    blend = min(10.0, float(seg[j]) / 2.0)
    into, left = s - cum[j], cum[j + 1] - s
    if blend > 1e-9 and into < blend and j > 0 and seg[j - 1] > 1e-9:
        w = 0.5 * (1.0 - into / blend)
        d = (1.0 - w) * d + w * (pts[j] - pts[j - 1]) / seg[j - 1]
    elif blend > 1e-9 and left < blend and j + 1 < len(seg) and seg[j + 1] > 1e-9:
        w = 0.5 * (1.0 - left / blend)
        d = (1.0 - w) * d + w * (pts[j + 2] - pts[j + 1]) / seg[j + 1]
    norm = np.linalg.norm(d)
    fwd = d / norm if norm > 1e-9 else np.array([0.0, 1.0])
    return pos, fwd


def _write_debug_overlay(path: Path, frame_small: np.ndarray,
                         mask: np.ndarray, model_rows: np.ndarray) -> None:
    """Inspection PNG: video frame tinted red where the model has
    buildings, with the model skyline drawn as a green line."""
    dbg = frame_small.copy()
    dbg[mask > 0] = (0.55 * dbg[mask > 0]
                     + np.array([0, 0, 115.0])).astype(np.uint8)
    for u in range(dbg.shape[1]):
        r = model_rows[u]
        if np.isfinite(r):
            v = int(np.clip(r, 0, dbg.shape[0] - 1))
            dbg[max(0, v - 1): v + 1, u] = (0, 255, 0)
    cv2.imwrite(str(path), dbg)


# --------------------------------------------------------------------------
# Coverage gate + consensus tie-break (pure decision helpers)
# --------------------------------------------------------------------------

def walk_coverage_fractions(
    building_xy: np.ndarray,
    walks: list,
    *,
    cell_m: float = 250.0,
) -> list[float]:
    """Fraction of each walk's points covered by a coarse occupancy grid
    (1-cell dilated) of the fetched building footprints.

    A plain bounding box of the buildings misses INTERIOR coverage holes
    (a 404 or tile-capped gap in the middle of the disc): a walk through
    the hole is inside the box yet renders against no buildings, so it
    fabricates ``open sky`` contradictions. The occupancy grid catches
    such holes — a walk crossing a missing >=1 km tile drops well below a
    sane threshold — while the 1-cell dilation keeps a normal street
    between building cells counted as covered.
    """
    bxy = np.asarray(building_xy, dtype=np.float64)
    if not len(bxy):
        return [0.0 for _ in walks]
    bxy = bxy[:, :2]
    gij = np.floor(bxy / cell_m).astype(np.int64)
    imin = int(gij[:, 0].min())
    jmin = int(gij[:, 1].min())
    H = int(gij[:, 0].max() - imin + 1)
    W = int(gij[:, 1].max() - jmin + 1)
    occ = np.zeros((H, W), dtype=bool)
    occ[gij[:, 0] - imin, gij[:, 1] - jmin] = True
    pad = np.pad(occ, 1)
    dil = np.zeros_like(occ)
    for di in range(3):
        for dj in range(3):
            dil |= pad[di:di + H, dj:dj + W]

    out: list[float] = []
    for walk in walks:
        c = np.floor(np.asarray(walk, dtype=np.float64)[:, :2] / cell_m
                     ).astype(np.int64)
        if not len(c):
            out.append(1.0)
            continue
        ii = c[:, 0] - imin
        jj = c[:, 1] - jmin
        inb = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        hit = np.zeros(len(c), dtype=bool)
        hit[inb] = dil[ii[inb], jj[inb]]
        out.append(float(hit.mean()))
    return out


def tile3d_tiebreak_winner(
    scores: list,
    errs: list,
    consensus_top: int,
    *,
    margin: float = 0.15,
    max_err_deg: float = 6.0,
    top_n_shape: int = 5,
) -> int | None:
    """Index the tile3d channel should promote over the shape/consensus
    pick, or ``None``.

    ``scores[i]`` / ``errs[i]`` are the tile3d score and skyline error for
    candidate ``i`` in SHAPE order (so ``i`` has shape rank ``i + 1``).
    Fires only when the skyline channel is unambiguously discriminative —
    a clear score margin over the runner-up, a tight skyline error, and a
    winner already high on the shape ranking — so it stays inert where
    skylines don't discriminate (uniform mid-rise) and can only reorder
    among already-plausible candidates. Returns ``None`` if any gate fails
    or the winner is already the consensus pick.
    """
    if len(scores) < 2 or any(s is None for s in scores):
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top = order[0]
    best, second = scores[order[0]], scores[order[1]]
    if best <= 0:
        return None
    rel = (best - second) / best
    err = errs[top] if top < len(errs) else None
    if (rel >= margin and err is not None and err <= max_err_deg
            and (top + 1) <= top_n_shape and top != consensus_top):
        return top
    return None


def adaptive_tile3d_weight(scores: list, *, w_base: float = 0.4,
                           lo: float = 0.05, hi: float = 0.20) -> float:
    """Confidence-scaled fusion weight for the tile3d channel.

    The fixed 0.4 rank weight adds the SAME nudge whether the skyline is
    sharply discriminative (dense high-rise: one candidate clearly wins) or
    non-discriminative (uniform mid-rise: scores nearly flat, the ``winner``
    is noise). Scale the weight by a smoothstep of the top-vs-runner-up
    relative margin so the channel self-mutes where it cannot discriminate
    and contributes fully where it can — the uncertainty-aware fusion the
    fixed weight lacked. Returns a weight in ``[0, w_base]``.
    """
    vals = sorted((s for s in scores if s is not None), reverse=True)
    if len(vals) < 2 or vals[0] <= 0:
        return 0.0
    margin = (vals[0] - vals[1]) / vals[0]
    t = float(np.clip((margin - lo) / (hi - lo), 0.0, 1.0))
    smooth = t * t * (3.0 - 2.0 * t)     # smoothstep
    return w_base * smooth


# --------------------------------------------------------------------------
# Metric placement refinement (LoD-Loc-style skyline pose optimization)
# --------------------------------------------------------------------------

def _apply_rigid(traj_xy: np.ndarray, dx: float, dy: float, dtheta: float,
                 center: np.ndarray) -> np.ndarray:
    c, s = np.cos(dtheta), np.sin(dtheta)
    rot = np.array([[c, -s], [s, c]])
    return (traj_xy - center) @ rot.T + center + np.array([dx, dy])


def _skyline_agreement(mesh, frame_sk: list, traj: np.ndarray,
                       K_r: np.ndarray, render_wh, cam_height_m: float,
                       max_dist_m: float) -> float:
    """Mean per-sample skyline agreement of a placed route (0..1)."""
    comps: list[SkylineComparison] = []
    for frac, elev_f in frame_sk:
        pos, fwd = pose_at_fraction(traj, frac)
        cam_z = mesh.local_ground_z(pos) + cam_height_m
        local = mesh.triangles_near(pos, max_dist_m)
        m = render_building_mask(local, pos, cam_z, fwd, K_r, render_wh,
                                 max_dist_m=max_dist_m)
        comps.append(compare_skylines(_elev_deg(skyline_from_mask(m), K_r),
                                      elev_f))
    offs = [float(np.median(c.delta)) for c in comps if c.delta is not None]
    off = float(np.median(offs)) if offs else 0.0
    tot = 0.0
    for c in comps:
        if c.delta is None:
            continue
        err = float(np.median(np.abs(c.delta - off)))
        w = min(1.0, c.explain_frac / _EXPLAIN_SAT)
        tot += (float(np.exp(-err / _ERR_SCALE_DEG)) * w
                * (1.0 - _CONTRA_WEIGHT * c.contradiction_frac))
    return tot / len(frame_sk) if frame_sk else 0.0


def refine_placement_skyline(
    mesh, samples: list, aligned_traj_xy: np.ndarray,
    video_K: np.ndarray, video_wh: tuple,
    *,
    cam_height_m: float = 2.2,
    render_wh: tuple = (480, 270),
    max_dist_m: float = 500.0,
    max_shift_m: float = 120.0,
    max_rot_deg: float = 12.0,
    max_fev: int = 60,
    skyline_fn=None,
) -> tuple[np.ndarray, dict]:
    """Refine a route's absolute PLACEMENT by skyline alignment.

    Selection (which candidate) and a coarse anchor leave a residual
    absolute-position error (our ~225 m start). This searches a small rigid
    transform (dx, dy, dθ about the route centroid) that best aligns the
    rendered skyline with the observed one across the sampled frames — the
    LoD-Loc idea (silhouette pose optimization) as a placement refiner
    rather than a candidate re-ranker. Bounded and no-op-safe: if no
    transform beats the identity it returns the input unchanged.

    Returns ``(refined_traj_xy, info)`` with ``info`` carrying the applied
    shift/rotation and the before/after agreement.
    """
    sky = skyline_fn or skyline_from_frame
    K_r = scale_intrinsics(video_K, video_wh, render_wh)
    frame_sk: list = []
    for frac, frame in samples:
        rows = sky(frame, render_wh)
        if np.isfinite(rows).mean() >= _MIN_SHARED_FRAC:
            frame_sk.append((frac, _elev_deg(rows, K_r)))
    base = np.asarray(aligned_traj_xy, dtype=np.float64)
    none = {"applied": False, "shift_m": 0.0, "rot_deg": 0.0,
            "score_before": 0.0, "score_after": 0.0}
    if len(frame_sk) < 2 or len(base) < 2:
        return base, none
    center = base.mean(axis=0)
    max_rot = np.radians(max_rot_deg)

    s0 = _skyline_agreement(mesh, frame_sk, base, K_r, render_wh,
                            cam_height_m, max_dist_m)

    def cost(p: np.ndarray) -> float:
        dx, dy, dth = float(p[0]), float(p[1]), float(p[2])
        if abs(dx) > max_shift_m or abs(dy) > max_shift_m or abs(dth) > max_rot:
            return 1.0
        traj = _apply_rigid(base, dx, dy, dth, center)
        return 1.0 - _skyline_agreement(mesh, frame_sk, traj, K_r, render_wh,
                                        cam_height_m, max_dist_m)

    try:
        from scipy.optimize import minimize
        simplex = np.array([[0.0, 0.0, 0.0], [45.0, 0.0, 0.0],
                            [0.0, 45.0, 0.0], [0.0, 0.0, np.radians(5.0)]])
        res = minimize(cost, x0=np.zeros(3), method="Nelder-Mead",
                       options={"maxfev": max_fev, "xatol": 2.0,
                                "fatol": 1e-3, "initial_simplex": simplex})
        dx, dy, dth = float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception:
        return base, {**none, "score_before": s0, "score_after": s0}

    traj_ref = _apply_rigid(base, dx, dy, dth, center)
    s1 = _skyline_agreement(mesh, frame_sk, traj_ref, K_r, render_wh,
                            cam_height_m, max_dist_m)
    if not (s1 > s0 + 1e-4):            # no improvement: keep the input
        return base, {**none, "score_before": s0, "score_after": s0}
    return traj_ref, {
        "applied": True, "shift_m": float(np.hypot(dx, dy)),
        "rot_deg": float(np.degrees(dth)),
        "score_before": s0, "score_after": s1,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def match_tiles3d_against_candidates(
    samples: list[tuple[float, np.ndarray]],
    candidates: list,
    mesh,
    video_K: np.ndarray,
    video_wh: tuple[int, int],
    *,
    output_dir: Path,
    cam_height_m: float = 2.2,
    render_wh: tuple[int, int] = (480, 270),
    max_dist_m: float = 500.0,
    debug_top_n: int = 3,
    skyline_fn=None,
) -> list[Tile3DMatchResult]:
    """Score every candidate's skyline agreement with the LoD2 mesh.

    ``samples`` are ``(route_fraction, frame_bgr)`` pairs: the frame the
    video shows at the moment the camera had covered ``route_fraction``
    of the analyzed path (build fractions with
    :func:`arc_fractions_at_times`). ``mesh`` is a
    :class:`~src.citygml_lod2.Lod2Mesh` in the road CRS.

    Samples whose FRAME yields no usable skyline are dropped for all
    candidates alike; the remaining samples form a FIXED denominator.
    Per sample a candidate earns exp(-err/3 deg) scaled by how much of
    the observed skyline its model explains, discounted for open-vs-
    building column contradictions; both-flat samples earn half
    credit. One pitch offset is fitted per candidate (softly penalized
    beyond a plausible dashcam range) — see the module docstring for
    why each of these choices exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sky = skyline_fn or skyline_from_frame
    K_r = scale_intrinsics(video_K, video_wh, render_wh)

    frame_skylines: list[tuple[float, np.ndarray, np.ndarray]] = []
    for frac, frame in samples:
        rows = sky(frame, render_wh)
        usable = np.isfinite(rows) | np.isposinf(rows)
        if np.isfinite(rows).mean() >= _MIN_SHARED_FRAC and usable.any():
            small = cv2.resize(frame, render_wh, interpolation=cv2.INTER_AREA)
            frame_skylines.append((frac, _elev_deg(rows, K_r), small))
    if not frame_skylines:
        print("      -> tile3d: no sample frame yields a usable skyline "
              "(night / open sky / heavy occlusion); channel inactive")
        return [Tile3DMatchResult(i, 0.0, None, None, 0.0, 0, 0)
                for i in range(len(candidates))]
    n_usable = len(frame_skylines)

    # Pass 1: render + compare every candidate, and fit its per-candidate
    # pitch offset (the camera pitch is one physical constant, so every
    # sample shares it).
    all_comps: list[list[SkylineComparison]] = []
    offsets: list[float | None] = []
    for ci, cand in enumerate(candidates):
        comps: list[SkylineComparison] = []
        for si, (frac, elev_f, frame_small) in enumerate(frame_skylines):
            pos, fwd = pose_at_fraction(cand.aligned_traj_xy, frac)
            cam_z = mesh.local_ground_z(pos) + cam_height_m
            # Pre-filter to the local triangles via the mesh grid; the
            # renderer's per-vertex cull then trims exactly, so the mask is
            # identical to rendering the full mesh — just far cheaper.
            local_tris = mesh.triangles_near(pos, max_dist_m)
            m = render_building_mask(
                local_tris, pos, cam_z, fwd, K_r, render_wh,
                max_dist_m=max_dist_m)
            model_rows = skyline_from_mask(m)
            elev_m = _elev_deg(model_rows, K_r)
            comps.append(compare_skylines(elev_m, elev_f))
            if ci < debug_top_n and si < 4:
                _write_debug_overlay(
                    output_dir / f"cand{ci + 1}_s{si}.png",
                    frame_small, m, model_rows)
        sample_offsets = [float(np.median(c.delta)) for c in comps
                          if c.delta is not None]
        all_comps.append(comps)
        offsets.append(float(np.median(sample_offsets))
                       if sample_offsets else None)

    # Systematic pitch = the offset the WHOLE candidate pool shares. A
    # constant bias (wrong cam height / intrinsics / ground z) is not
    # evidence against any candidate — only the RESIDUAL |offset -
    # systematic| is. Penalizing the raw offset (v1) wrongly punished every
    # candidate on a clip whose camera pitch sits a fixed ~13 deg off (Ulm).
    known = [o for o in offsets if o is not None]
    systematic = float(np.median(known)) if known else 0.0

    results = []
    for ci, comps in enumerate(all_comps):
        offset = offsets[ci]
        contributions: list[float] = []
        errs: list[float] = []
        n_informative = 0
        for c in comps:
            if c.delta is None:
                contributions.append(0.0)
                continue
            err = float(np.median(np.abs(c.delta - offset)))
            errs.append(err)
            agree = float(np.exp(-err / _ERR_SCALE_DEG))
            weight = min(1.0, c.explain_frac / _EXPLAIN_SAT)
            contrib = agree * weight * (1.0 - _CONTRA_WEIGHT
                                        * c.contradiction_frac)
            if c.informative:
                n_informative += 1
            else:
                contrib *= _FLAT_CREDIT
            contributions.append(max(0.0, contrib))

        score = float(np.sum(contributions)) / n_usable
        residual = abs(offset - systematic) if offset is not None else 0.0
        if residual > _PITCH_SOFT_LIMIT:
            score *= float(np.exp(-(residual - _PITCH_SOFT_LIMIT)
                                  / _PITCH_DECAY_DEG))
        results.append(Tile3DMatchResult(
            candidate_index=ci,
            tile3d_score=score,
            skyline_err_deg=(float(np.median(errs)) if errs else None),
            pitch_offset_deg=offset,
            coverage=(float(np.mean([c.explain_frac for c in comps]))
                      if comps else 0.0),
            n_samples_scored=n_usable,
            n_informative=n_informative,
        ))
    return results
