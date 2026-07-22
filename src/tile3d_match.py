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
    polys = [_project(t) for t in cam[safe]]
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
    K_r = scale_intrinsics(video_K, video_wh, render_wh)

    frame_skylines: list[tuple[float, np.ndarray, np.ndarray]] = []
    for frac, frame in samples:
        rows = skyline_from_frame(frame, render_wh)
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

    results: list[Tile3DMatchResult] = []
    for ci, cand in enumerate(candidates):
        comps: list[SkylineComparison] = []
        for si, (frac, elev_f, frame_small) in enumerate(frame_skylines):
            pos, fwd = pose_at_fraction(cand.aligned_traj_xy, frac)
            cam_z = mesh.local_ground_z(pos) + cam_height_m
            m = render_building_mask(
                mesh.triangles, pos, cam_z, fwd, K_r, render_wh,
                max_dist_m=max_dist_m)
            model_rows = skyline_from_mask(m)
            elev_m = _elev_deg(model_rows, K_r)
            comps.append(compare_skylines(elev_m, elev_f))
            if ci < debug_top_n and si < 4:
                _write_debug_overlay(
                    output_dir / f"cand{ci + 1}_s{si}.png",
                    frame_small, m, model_rows)

        # One pitch offset per candidate: the camera's pitch is a single
        # physical constant, so every sample must share it.
        sample_offsets = [float(np.median(c.delta)) for c in comps
                          if c.delta is not None]
        offset = float(np.median(sample_offsets)) if sample_offsets else None

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
        if offset is not None and abs(offset) > _PITCH_SOFT_LIMIT:
            score *= float(np.exp(-(abs(offset) - _PITCH_SOFT_LIMIT)
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
