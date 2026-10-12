"""Geometry of the per-frame VPR track, shared by matching, fusion and placement."""

from __future__ import annotations

import numpy as np


# VPR re-rank penalty: metres of slack before a candidate is penalized for
# distance to the VPR prior. The prior's own error bar is ~91-190 m on Ulm,
# so candidates within it are indistinguishable from the truth — penalizing
# them flips near-ties away from a correct shape pick.
_VPR_FREE_RADIUS_M = 150.0

# Plate-district penalty free radius (see the fusion comment for rationale).
_PLATE_FREE_RADIUS_M = 1500.0


def _vpr_distance_penalty(d_m: float, free_radius_m: float = _VPR_FREE_RADIUS_M) -> float:
    """Consensus penalty for a candidate ``d_m`` metres from the prior: free
    inside the prior's own error bar, then 15 rank-points per km.

    The 150 m default is the error bar of a retrieval (VPR) prior. A prior
    from the VLM fallback is a geocoded district/street centroid, km-coarse,
    and is given the district-level free radius instead (see the call site).
    """
    return 15.0 * max(0.0, (d_m - free_radius_m) / 1000.0)


def _vpr_sequence_median_m(
    traj_xy: np.ndarray,
    track_frame_idx,
    track_xy: np.ndarray,
    sims,
    n_frames: int,
) -> float | None:
    """Similarity-weighted median distance (m) between the candidate route
    position at each VPR-matched frame's arc fraction and that frame's
    retrieved reference location.

    Frame index -> arc fraction assumes roughly constant speed (the
    aligned trajectory is arc-uniform, the frames time-uniform); good
    enough for the median statistic this feeds. A candidate centred
    correctly but flipped/rotated has the same centroid distance yet a
    much worse sequence distance — the ambiguity class the centroid
    penalty cannot rank.
    """
    traj = np.asarray(traj_xy, dtype=np.float64)
    track_xy = np.asarray(track_xy, dtype=np.float64)
    if len(traj) < 2 or len(track_xy) == 0:
        return None
    frac = np.asarray(track_frame_idx, dtype=np.float64) / max(n_frames - 1, 1)
    rows = np.clip(np.round(frac * (len(traj) - 1)).astype(int), 0, len(traj) - 1)
    d = np.linalg.norm(traj[rows] - track_xy, axis=1)
    w = np.clip(np.asarray(sims, dtype=np.float64), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(d)
    order = np.argsort(d)
    cw = np.cumsum(w[order])
    k = int(np.searchsorted(cw, 0.5 * cw[-1]))
    return float(d[order][min(k, len(d) - 1)])


def _vpr_track_geometry(vpr_track, crs):
    """Sorted per-frame VPR track plus projected robust endpoint centres.

    Returns ``(frame_idx, latlon, sims, start_xy, end_xy)`` with the track
    sorted by frame index and the START-/END-region robust centres
    (:func:`kartaview_vpr._robust_center` over the first/last ``len/16``
    frames — wider windows drift on fast clips: comma start 43 m at k=5 vs
    164 m at k=10) projected into ``crs``; ``None`` when there is no track.
    One home for the sort -> robust-centre -> project dance that the two-pass
    scale, the chord diagnostic and the anchor placement all need.
    """
    if vpr_track is None:
        return None
    from ..kartaview_vpr import _robust_center
    from ..position import latlon_to_xy
    vi, vll, vs = vpr_track
    vi = np.asarray(vi)
    if len(vi) == 0:
        return None
    o = np.argsort(vi)
    vll = np.asarray(vll, dtype=np.float64)[o]
    vs = np.asarray(vs, dtype=np.float64)[o]
    k = max(5, len(vll) // 16)
    s = _robust_center(vll[:k], vs[:k])
    e = _robust_center(vll[-k:], vs[-k:])
    ends = latlon_to_xy(np.asarray([s, e], dtype=np.float64), crs)
    return vi[o], vll, vs, ends[0], ends[1]
