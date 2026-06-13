"""Discrete turn-sequence matching — a drift-robust shape channel.

The existing matcher (``trajectory_matching``) scores candidates by
Procrustes RMS plus a *dense* bearing-delta correlation. Both compare
the whole geometry sample-for-sample, so they assume the query and the
candidate line up almost exactly after a similarity transform. Monocular
VO drift breaks that assumption: over a few minutes the recovered path
bends and the turns land at slightly different arc positions, so dense
comparison degrades and (as seen on the 15-minute Ulm run) the whole
candidate pool slides into the wrong part of the city together.

Turn-sequence matching is robust to exactly that failure. We reduce a
path to its handful of *significant turns* — each a (arc-fraction,
signed-angle) event — and compare two paths by aligning their turn
sequences with an edit distance that tolerates position shifts and
missing/extra turns. A left turn is a left turn regardless of
accumulated scale or rotation error, and the alignment absorbs the
position drift that sinks dense correlation.

This is a *topological* descriptor (the pattern of decisions), not a
metric one, which is why it survives drift. It does not, on its own,
separate parallel streets that share a turn pattern — that needs an
absolute anchor (see the OCR/landmark channels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .visual_odometry import resample_uniform, trajectory_arc_length


@dataclass(frozen=True)
class Turn:
    """A single significant turn along a path.

    ``arc_fraction`` is the turn's position as a fraction of total path
    length in [0, 1] (scale-invariant). ``angle_deg`` is the signed net
    heading change through the turn (positive = left/CCW, negative =
    right/CW), which is invariant to the path's absolute orientation.
    """
    arc_fraction: float
    angle_deg: float


def extract_turns(
    xy: np.ndarray,
    *,
    n_resample: int = 256,
    window_frac: float = 0.06,
    turn_threshold_deg: float = 25.0,
) -> list[Turn]:
    """Reduce a path to its significant turns.

    The path is resampled to equidistant points; the signed heading
    change is summed over a sliding window of width ``window_frac`` of
    the total length; local extrema of that windowed turn whose
    magnitude exceeds ``turn_threshold_deg`` become turn events
    (non-max-suppressed within one window so a single intersection isn't
    double-counted).

    The windowing is what rejects slow VO drift: a real intersection
    turn is sharp (its heading change is concentrated in a short arc, so
    it sums high within the window), whereas drift spreads a comparable
    total heading change over a long arc (little of it falls inside any
    one window). A pure cumulative-threshold detector would mistake
    drift for a turn.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 3 or trajectory_arc_length(xy)[-1] <= 0:
        return []

    pts = resample_uniform(xy, n_resample)
    seg = np.diff(pts, axis=0)
    headings = np.arctan2(seg[:, 1], seg[:, 0])
    # Signed turn at each interior vertex, wrapped to (-pi, pi].
    dtheta = np.diff(headings)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # len n_resample-2

    n_vert = len(dtheta)
    if n_vert == 0:
        return []
    half = max(1, int(round(window_frac * n_resample / 2)))

    # Windowed signed turn at each interior vertex.
    windowed = np.array([
        dtheta[max(0, i - half):i + half + 1].sum() for i in range(n_vert)
    ])
    windowed_deg = np.degrees(windowed)
    thr = float(turn_threshold_deg)

    # Local extrema of |windowed| above threshold, NMS within `half`.
    turns: list[Turn] = []
    order = np.argsort(-np.abs(windowed_deg))
    claimed = np.zeros(n_vert, dtype=bool)
    for i in order:
        if abs(windowed_deg[i]) < thr:
            break
        if claimed[max(0, i - half):i + half + 1].any():
            continue
        claimed[max(0, i - half):i + half + 1] = True
        # Interior vertex i sits between pts[i+1]; arc fraction of that point.
        arc_fraction = (i + 1) / (n_resample - 1)
        turns.append(Turn(arc_fraction=float(arc_fraction),
                          angle_deg=float(windowed_deg[i])))

    turns.sort(key=lambda t: t.arc_fraction)
    return turns


def turn_sequence_distance(
    seq_a: list[Turn],
    seq_b: list[Turn],
    *,
    gap_penalty: float = 1.0,
    pos_weight: float = 2.0,
    angle_scale_deg: float = 90.0,
) -> float:
    """Normalized edit distance between two turn sequences (lower = better).

    Substitution cost of matching turn ``a`` to turn ``b`` combines their
    arc-position gap (weighted by ``pos_weight``) and their signed-angle
    gap (normalized by ``angle_scale_deg``); opposite-direction turns
    cost extra because a left where a right is expected is a real
    mismatch. Unmatched turns on either side cost ``gap_penalty`` each.
    The total is normalized by the longer sequence so the score is
    comparable across candidates with different turn counts.

    Two straight paths (no turns) are identical → distance 0. A path with
    turns vs a straight one is maximally far → distance ~gap_penalty.
    """
    na, nb = len(seq_a), len(seq_b)
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return gap_penalty

    def sub_cost(a: Turn, b: Turn) -> float:
        pos = abs(a.arc_fraction - b.arc_fraction)
        ang = abs(a.angle_deg - b.angle_deg) / angle_scale_deg
        # Extra penalty when the turns go opposite directions.
        opposite = 0.5 if (a.angle_deg * b.angle_deg) < 0 else 0.0
        return pos_weight * pos + ang + opposite

    # Needleman-Wunsch alignment.
    dp = np.zeros((na + 1, nb + 1))
    dp[:, 0] = np.arange(na + 1) * gap_penalty
    dp[0, :] = np.arange(nb + 1) * gap_penalty
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            dp[i, j] = min(
                dp[i - 1, j - 1] + sub_cost(seq_a[i - 1], seq_b[j - 1]),
                dp[i - 1, j] + gap_penalty,
                dp[i, j - 1] + gap_penalty,
            )
    return float(dp[na, nb] / max(na, nb))


def score_candidates_by_turns(
    traj_xy: np.ndarray,
    candidate_polylines: list[np.ndarray],
    *,
    n_resample: int = 256,
    window_frac: float = 0.06,
    turn_threshold_deg: float = 25.0,
) -> tuple[list[float], list[Turn]]:
    """Per-candidate turn-sequence distance to the query trajectory.

    Returns ``(distances, query_turns)`` where ``distances[i]`` is the
    turn-sequence edit distance from the query to candidate ``i`` (lower
    = better) and ``query_turns`` is the query's extracted turns (handy
    for logging/diagnostics). A candidate whose polyline is too short to
    yield a signature gets ``inf``.
    """
    query_turns = extract_turns(
        traj_xy, n_resample=n_resample, window_frac=window_frac,
        turn_threshold_deg=turn_threshold_deg,
    )
    distances: list[float] = []
    for poly in candidate_polylines:
        poly = np.asarray(poly, dtype=np.float64)
        # A degenerate polyline (too few points or zero length) is
        # unmatchable — distinct from a valid but straight path, which
        # legitimately yields zero turns. Mark it inf so it can't tie
        # with genuine straights.
        if poly.ndim != 2 or len(poly) < 3 or trajectory_arc_length(poly)[-1] <= 0:
            distances.append(float("inf"))
            continue
        cand_turns = extract_turns(
            poly, n_resample=n_resample, window_frac=window_frac,
            turn_threshold_deg=turn_threshold_deg,
        )
        distances.append(turn_sequence_distance(query_turns, cand_turns))
    return distances, query_turns
