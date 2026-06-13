"""Tests for discrete turn-sequence matching.

The key properties we verify: turn extraction finds real turns and
ignores straights, the descriptor is invariant to rotation/scale/
translation, it is *robust to gradual drift* (the whole point), and the
sequence distance ranks a same-turn-pattern candidate above a
different-pattern one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.turn_matching import (
    Turn,
    extract_turns,
    score_candidates_by_turns,
    turn_sequence_distance,
)


def _straight(n: int = 200, length: float = 200.0) -> np.ndarray:
    x = np.linspace(0, length, n)
    return np.column_stack([x, np.zeros_like(x)])


def _l_left(n: int = 200, leg: float = 100.0) -> np.ndarray:
    """East for `leg`, then a 90° left turn, then north for `leg`."""
    half = n // 2
    east = np.column_stack([np.linspace(0, leg, half), np.zeros(half)])
    north = np.column_stack([np.full(half, leg), np.linspace(0, leg, half)])
    return np.vstack([east, north])


def _l_right(n: int = 200, leg: float = 100.0) -> np.ndarray:
    """East, then a 90° right turn, then south."""
    half = n // 2
    east = np.column_stack([np.linspace(0, leg, half), np.zeros(half)])
    south = np.column_stack([np.full(half, leg), np.linspace(0, -leg, half)])
    return np.vstack([east, south])


def _similarity(xy: np.ndarray, deg: float, scale: float, t: np.ndarray) -> np.ndarray:
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return scale * xy @ R.T + t


# ---------------------------------------------------------------------------
# extract_turns
# ---------------------------------------------------------------------------


def test_extract_turns_straight_has_none() -> None:
    assert extract_turns(_straight()) == []


def test_extract_turns_finds_single_left() -> None:
    turns = extract_turns(_l_left())
    assert len(turns) == 1
    assert turns[0].angle_deg > 0           # left = positive
    assert 60 < turns[0].angle_deg < 120    # ~90°
    assert 0.3 < turns[0].arc_fraction < 0.7  # mid-path


def test_extract_turns_sign_distinguishes_left_right() -> None:
    assert extract_turns(_l_left())[0].angle_deg > 0
    assert extract_turns(_l_right())[0].angle_deg < 0


def test_extract_turns_invariant_to_similarity() -> None:
    base = _l_left()
    rotated = _similarity(base, deg=53.0, scale=3.7, t=np.array([12.0, -8.0]))
    tb, tr = extract_turns(base), extract_turns(rotated)
    assert len(tb) == len(tr) == 1
    # Rotation/scale/translation must not change the turn angle or position.
    assert tr[0].angle_deg == pytest.approx(tb[0].angle_deg, abs=8.0)
    assert tr[0].arc_fraction == pytest.approx(tb[0].arc_fraction, abs=0.05)


def test_extract_turns_robust_to_gradual_drift() -> None:
    """A straight path with a slow constant curvature (VO drift) must NOT
    register as a turn, even when its total heading change is large —
    this is the property dense bearing correlation lacks."""
    n = 400
    t = np.linspace(0, 1, n)
    # Gentle arc: ~60° of heading change spread over the whole path.
    theta = np.deg2rad(60.0) * t
    step = 1.0
    xy = np.column_stack([np.cumsum(np.cos(theta)) * step,
                          np.cumsum(np.sin(theta)) * step])
    # Spread over the full length, no single window accumulates 25°.
    assert extract_turns(xy, turn_threshold_deg=25.0) == []


def test_extract_turns_two_turns() -> None:
    # East, left (north), left (west): an inverted-U with two left turns.
    leg = 100.0
    k = 100
    east = np.column_stack([np.linspace(0, leg, k), np.zeros(k)])
    north = np.column_stack([np.full(k, leg), np.linspace(0, leg, k)])
    west = np.column_stack([np.linspace(leg, 0, k), np.full(k, leg)])
    turns = extract_turns(np.vstack([east, north, west]))
    assert len(turns) == 2
    assert all(t.angle_deg > 0 for t in turns)


# ---------------------------------------------------------------------------
# turn_sequence_distance
# ---------------------------------------------------------------------------


def test_distance_zero_for_two_straights() -> None:
    assert turn_sequence_distance([], []) == 0.0


def test_distance_identical_sequences() -> None:
    seq = extract_turns(_l_left())
    assert turn_sequence_distance(seq, seq) == pytest.approx(0.0, abs=1e-9)


def test_distance_same_pattern_beats_different() -> None:
    left = extract_turns(_l_left())
    right = extract_turns(_l_right())
    straight: list[Turn] = []
    # A left-turn query is closer to another left turn than to a right
    # turn or to a straight path.
    assert turn_sequence_distance(left, left) < turn_sequence_distance(left, right)
    assert turn_sequence_distance(left, left) < turn_sequence_distance(left, straight)


def test_distance_opposite_turn_penalized() -> None:
    left = [Turn(0.5, 90.0)]
    right = [Turn(0.5, -90.0)]
    near_left = [Turn(0.5, 70.0)]
    # Same-direction (even if angle differs) must beat opposite-direction.
    assert turn_sequence_distance(left, near_left) < turn_sequence_distance(left, right)


def test_distance_tolerates_position_shift() -> None:
    """A turn at 0.45 vs 0.55 (drift shifted it) should still match far
    better than a missing turn."""
    a = [Turn(0.45, 90.0)]
    b = [Turn(0.55, 90.0)]
    shifted = turn_sequence_distance(a, b)
    missing = turn_sequence_distance(a, [])
    assert shifted < missing


# ---------------------------------------------------------------------------
# score_candidates_by_turns
# ---------------------------------------------------------------------------


def test_score_candidates_ranks_matching_shape_first() -> None:
    query = _l_left()
    candidates = [_straight(), _l_right(), _l_left()]   # index 2 matches
    dists, qturns = score_candidates_by_turns(query, candidates)
    assert len(qturns) == 1
    assert np.argmin(dists) == 2


def test_score_candidates_handles_degenerate_polyline() -> None:
    query = _l_left()
    candidates = [np.zeros((1, 2)), _l_left()]
    dists, _ = score_candidates_by_turns(query, candidates)
    assert not np.isfinite(dists[0])      # too short → inf
    assert np.isfinite(dists[1])
