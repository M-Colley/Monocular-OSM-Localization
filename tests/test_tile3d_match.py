"""Offline tests for the 3D-tile skyline matcher (no network, no GPU)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from src.citygml_lod2 import Lod2Mesh
from src.tile3d_match import (
    adaptive_tile3d_weight,
    arc_fractions_at_times,
    compare_skylines,
    match_tiles3d_against_candidates,
    pose_at_fraction,
    refine_placement_skyline,
    render_building_mask,
    scale_intrinsics,
    skyline_from_frame,
    skyline_from_mask,
    tile3d_tiebreak_winner,
    walk_coverage_fractions,
    _apply_rigid,
)


def _box_mesh(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    """Axis-aligned box as 12 triangles (N, 3, 3)."""
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                 dtype=np.float64)
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    tris = []
    for a, b, cc, d in quads:
        tris.append(c[[a, b, cc]])
        tris.append(c[[a, cc, d]])
    return np.asarray(tris)


def _k(w=480, h=270, hfov_deg=70.0) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


WH = (480, 270)


def test_render_box_ahead_lands_center_with_correct_elevation() -> None:
    # 20 m-tall box, 50 m ahead (north) of a camera at z=2
    tris = _box_mesh(-10, 10, 50, 70, 0, 20)
    K = _k()
    mask = render_building_mask(tris, np.array([0.0, 0.0]), 2.0,
                                np.array([0.0, 1.0]), K, WH)
    assert mask.any()
    rows = skyline_from_mask(mask)
    mid = rows[WH[0] // 2]
    # expected skyline elevation: atan((20-2)/50) above the horizon
    expect_v = K[1, 2] - K[1, 1] * (20.0 - 2.0) / 50.0
    assert mid == pytest.approx(expect_v, abs=2.0)


def test_render_box_behind_camera_is_empty() -> None:
    tris = _box_mesh(-10, 10, -70, -50, 0, 20)
    mask = render_building_mask(tris, np.array([0.0, 0.0]), 2.0,
                                np.array([0.0, 1.0]), _k(), WH)
    assert not mask.any()


def test_render_heading_rotation_moves_box_off_axis() -> None:
    # box due north; camera turned 30 deg east -> box shifts left of center
    tris = _box_mesh(-5, 5, 60, 70, 0, 25)
    hd = np.array([np.sin(np.radians(30.0)), np.cos(np.radians(30.0))])
    mask = render_building_mask(tris, np.array([0.0, 0.0]), 2.0, hd, _k(), WH)
    rows = skyline_from_mask(mask)
    cols = np.flatnonzero(np.isfinite(rows))
    assert len(cols) and cols.mean() < WH[0] / 2.0


def test_render_straddling_near_plane_keeps_far_part() -> None:
    # box extending from behind the camera to ahead of it: near-plane
    # clipping must keep the visible part instead of dropping the tris
    tris = _box_mesh(-4, 4, -10, 30, 0, 15)
    mask = render_building_mask(tris, np.array([0.0, 0.0]), 2.0,
                                np.array([0.0, 1.0]), _k(), WH)
    assert mask.any()


def test_skyline_from_frame_synthetic_horizon() -> None:
    w, h = WH
    frame = np.full((h * 4, w * 4, 3), 230, dtype=np.uint8)   # bright sky
    frame[h * 2:, :] = (60, 60, 70)                           # dark blocks
    rows = skyline_from_frame(frame, WH)
    valid = np.isfinite(rows)
    assert valid.mean() > 0.9
    assert np.nanmedian(rows[valid]) == pytest.approx(h / 2.0, abs=4.0)


def test_skyline_from_frame_night_is_invalid() -> None:
    frame = np.full((270, 480, 3), 12, dtype=np.uint8)
    rows = skyline_from_frame(frame, WH)
    assert np.isnan(rows).all()


def test_skyline_from_frame_gradient_sky_stays_open() -> None:
    """A graded clear sky (zenith blue -> bright horizon) with NO
    buildings must read as open sky, not fabricate a phantom skyline
    where the gradient drifts from the zenith seed color (the v1 bug
    an adversarial review demonstrated)."""
    w, h = 480, 270
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        t = r / (h - 1)
        frame[r, :] = (int(200 - 110 * t), int(150 + 85 * t),
                       int(90 + 145 * t))       # BGR blue -> near-white
    rows = skyline_from_frame(frame, WH)
    assert not np.isfinite(rows).any()          # no structure anywhere
    assert np.isposinf(rows).mean() > 0.9       # open sky, still usable


def test_compare_skylines_prefers_matching_shape() -> None:
    w = WH[0]
    x = np.linspace(0, 4 * np.pi, w)
    truth = 8.0 * np.sin(x) + 10.0
    same_offset = truth + 3.0            # pitch offset only -> ~0 error
    different = 8.0 * np.sin(x + np.pi) + 10.0
    c_same = compare_skylines(same_offset, truth)
    c_diff = compare_skylines(different, truth)
    assert c_same.informative and c_same.shared_frac > 0.9
    err_same = np.median(np.abs(c_same.delta - np.median(c_same.delta)))
    err_diff = np.median(np.abs(c_diff.delta - np.median(c_diff.delta)))
    assert err_same == pytest.approx(0.0, abs=1e-9)
    assert err_diff > 4.0


def test_compare_skylines_flat_pair_is_uninformative() -> None:
    flat = np.full(WH[0], 5.0)
    c = compare_skylines(flat, flat + 2.0)
    assert c.delta is not None and not c.informative


def test_compare_skylines_insufficient_overlap() -> None:
    a = np.full(WH[0], np.nan)
    a[:5] = 1.0
    b = np.full(WH[0], 1.0)
    c = compare_skylines(a, b)
    assert c.delta is None and not c.informative
    # all but the 5 shared columns contradict the empty model
    assert c.contradiction_frac == pytest.approx(475 / 480, abs=0.01)


def test_compare_skylines_contradiction_both_directions() -> None:
    w = WH[0]
    model = np.full(w, 12.0)
    frame = np.full(w, 12.0)
    frame[: w // 4] = -np.inf        # video: open sky, model: 12 deg
    c = compare_skylines(model, frame)
    assert c.contradiction_frac == pytest.approx(0.25, abs=0.02)
    # below-horizon structure counts as open, never as buildings
    horizonish = np.full(w, 0.2)
    c2 = compare_skylines(horizonish, np.full(w, 12.0))
    assert c2.delta is None and c2.contradiction_frac == pytest.approx(1.0)


def test_arc_fractions_at_times_constant_speed() -> None:
    traj = np.stack([np.linspace(0, 100, 11), np.zeros(11)], axis=1)
    ts = np.linspace(0.0, 10.0, 11)
    fr = arc_fractions_at_times(traj, ts, np.array([0.0, 5.0, 10.0]))
    assert fr == pytest.approx([0.0, 0.5, 1.0])


def test_pose_at_fraction_position_and_heading() -> None:
    # path: 100 m east then 100 m north
    path = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    pos, fwd = pose_at_fraction(path, 0.25)
    assert pos == pytest.approx([50.0, 0.0])
    assert fwd[0] > 0.9                       # heading east
    pos, fwd = pose_at_fraction(path, 0.75)
    assert pos == pytest.approx([100.0, 50.0])
    assert fwd[1] > 0.9                       # heading north


@dataclass
class _FakeCandidate:
    aligned_traj_xy: np.ndarray


def test_match_ranks_route_with_buildings_over_open_field(tmp_path) -> None:
    """End-to-end: a synthetic 'street canyon' of boxes along y=0; the
    camera video shows a skyline. The candidate whose route runs
    through the canyon must outscore a candidate in an empty field."""
    rng = np.random.default_rng(7)
    tris = []
    grounds = []
    for xi in range(0, 400, 25):
        h = 12.0 + float(rng.uniform(0, 14))
        for side in (-18.0, 14.0):
            tris.append(_box_mesh(xi, xi + 14, side, side + 8, 0.0, h))
            grounds.append([xi + 7, side + 4, 0.0])
    mesh = Lod2Mesh(
        triangles=np.concatenate(tris).astype(np.float32),
        building_ground=np.asarray(grounds, dtype=np.float32),
        crs="EPSG:32633", provider="berlin", n_buildings=len(grounds),
    )
    K = _k(1280, 720)

    # the "video": render the canyon route's own view, painted as an image
    good = _FakeCandidate(np.stack(
        [np.linspace(0, 380, 128), np.zeros(128)], axis=1))
    bad = _FakeCandidate(np.stack(
        [np.linspace(0, 380, 128), np.full(128, 5000.0)], axis=1))

    samples = []
    for frac in (0.2, 0.45, 0.7):
        pos, fwd = pose_at_fraction(good.aligned_traj_xy, frac)
        mask = render_building_mask(
            mesh.triangles, pos, 2.2, fwd, _k(*WH), WH)
        frame = np.full((720, 1280, 3), 235, dtype=np.uint8)
        big = cv2.resize(mask, (1280, 720),
                         interpolation=cv2.INTER_NEAREST) > 0
        frame[big] = (70, 70, 80)
        samples.append((frac, frame))

    results = match_tiles3d_against_candidates(
        samples, [good, bad], mesh, K, (1280, 720),
        output_dir=tmp_path / "t3d")
    assert results[0].tile3d_score > results[1].tile3d_score + 0.2
    assert results[0].n_samples_scored >= 2
    assert results[1].tile3d_score == pytest.approx(0.0, abs=0.05)


def test_match_no_usable_frames_returns_neutral(tmp_path) -> None:
    mesh = Lod2Mesh(
        triangles=_box_mesh(-10, 10, 50, 70, 0, 20).astype(np.float32),
        building_ground=np.array([[0, 60, 0]], dtype=np.float32),
        crs="EPSG:32633", provider="berlin", n_buildings=1,
    )
    night = np.full((720, 1280, 3), 10, dtype=np.uint8)
    cand = _FakeCandidate(np.stack(
        [np.zeros(16), np.linspace(0, 100, 16)], axis=1))
    results = match_tiles3d_against_candidates(
        [(0.5, night)], [cand], mesh, _k(1280, 720), (1280, 720),
        output_dir=tmp_path / "t3d")
    assert results[0].tile3d_score == 0.0
    assert results[0].n_samples_scored == 0


def test_scale_intrinsics() -> None:
    K = _k(1920, 1080)
    K2 = scale_intrinsics(K, (1920, 1080), (480, 270))
    assert K2[0, 0] == pytest.approx(K[0, 0] / 4.0)
    assert K2[1, 2] == pytest.approx(K[1, 2] / 4.0)


# --------------------------------------------------------------------------
# Mesh spatial grid (triangles_near)
# --------------------------------------------------------------------------

def _scattered_boxes(step: int = 40, span: int = 1000) -> np.ndarray:
    return np.concatenate([_box_mesh(x, x + 15, 40, 55, 0.0, 18.0)
                           for x in range(0, span, step)])


def test_triangles_near_render_matches_full_mesh() -> None:
    """The grid pre-filter must render a mask IDENTICAL to the full mesh
    (the renderer's own cull trims the superset exactly) while touching
    far fewer triangles — the invariant the perf optimization relies on."""
    tris = _scattered_boxes()
    mesh = Lod2Mesh(triangles=tris.astype(np.float32),
                    building_ground=np.array([[500, 50, 0]], dtype=np.float32),
                    crs="EPSG:32633", provider="berlin", n_buildings=1)
    K = _k()
    for cx in (0.0, 300.0, 700.0):
        pos = np.array([cx, 0.0])
        hd = np.array([0.0, 1.0])
        full = render_building_mask(mesh.triangles, pos, 2.0, hd, K, WH,
                                    max_dist_m=200.0)
        sub_tris = mesh.triangles_near(pos, 200.0)
        sub = render_building_mask(sub_tris, pos, 2.0, hd, K, WH,
                                   max_dist_m=200.0)
        assert np.array_equal(full, sub)
        assert len(sub_tris) < len(mesh.triangles)   # actually filtered


def test_triangles_near_is_superset_of_in_range() -> None:
    tris = _scattered_boxes()
    mesh = Lod2Mesh(triangles=tris.astype(np.float32),
                    building_ground=np.zeros((0, 3), dtype=np.float32),
                    crs="EPSG:32633", provider="berlin", n_buildings=0)
    pos = np.array([120.0, 0.0])
    md = 150.0
    near = mesh.triangles_near(pos, md)
    vdist = np.sqrt(((mesh.triangles[:, :, :2] - pos) ** 2).sum(-1)).min(1)
    exact = mesh.triangles[vdist < md]
    near_set = {t.tobytes() for t in near.astype(np.float32)}
    for t in exact.astype(np.float32):       # every in-range tri retained
        assert t.tobytes() in near_set


def test_triangles_near_empty_mesh() -> None:
    mesh = Lod2Mesh(triangles=np.zeros((0, 3, 3), dtype=np.float32),
                    building_ground=np.zeros((0, 3), dtype=np.float32),
                    crs="EPSG:32633", provider="berlin", n_buildings=0)
    assert len(mesh.triangles_near(np.array([0.0, 0.0]), 500.0)) == 0


# --------------------------------------------------------------------------
# Coverage gate (walk_coverage_fractions)
# --------------------------------------------------------------------------

def _dense_field(step: float = 30.0, span: float = 3000.0) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(0, span, step), np.arange(0, span, step))
    return np.column_stack([gx.ravel(), gy.ravel(),
                            np.zeros(gx.size)])


def test_walk_coverage_full_field_is_covered() -> None:
    bg = _dense_field()
    walk = np.column_stack([np.linspace(100, 2900, 60), np.full(60, 1500.0)])
    assert walk_coverage_fractions(bg, [walk])[0] == pytest.approx(1.0)


def test_walk_coverage_detects_missing_tile_hole() -> None:
    """A walk crossing a missing >=1 km tile (square hole) must drop below
    the 0.9 gate; a walk staying in covered area must not."""
    bg = _dense_field()
    lo, hi = 500.0, 2500.0          # 2 km square hole (a dropped BW block)
    keep = ~((bg[:, 0] >= lo) & (bg[:, 0] <= hi)
             & (bg[:, 1] >= lo) & (bg[:, 1] <= hi))
    crossing = np.column_stack([np.linspace(100, 2900, 60), np.full(60, 1500.0)])
    safe = np.column_stack([np.linspace(100, 450, 30), np.full(30, 200.0)])
    cov = walk_coverage_fractions(bg[keep], [crossing, safe])
    assert cov[0] < 0.9             # crossing the hole -> deactivate
    assert cov[1] > 0.9             # in-coverage walk -> stay active


def test_walk_coverage_edges() -> None:
    assert walk_coverage_fractions(np.zeros((0, 3)), [np.zeros((3, 2))]) == [0.0]
    bg = _dense_field(span=600.0)
    assert walk_coverage_fractions(bg, [np.zeros((0, 2))])[0] == 1.0


# --------------------------------------------------------------------------
# Consensus tie-break gate (tile3d_tiebreak_winner)
# --------------------------------------------------------------------------

# Real recorded per-candidate values (shape order) from the Berlin and Ulm
# pure-shape + tile3d runs (2026-07-22 eval).
_BERLIN_SCORES = [.1481, .1269, .1376, .1845, .097, .1375, .1207, .141, .1095, .0806]
_BERLIN_ERRS = [6.46, 6.8, 5.86, 5.28, 6.93, 6.51, 6.59, 6.11, 7.46, 9.87]
_ULM_SCORES = [.1287, .1637, .1161, .1188, .14, .0819, .1787, .1278, .1261, .1124]
_ULM_ERRS = [4.67, 5.54, 4.91, 4.67, 4.67, 6.57, 7.37, 5.07, 5.0, 5.64]


def test_tiebreak_fires_on_berlin_high_rise() -> None:
    # discriminative skyline: promote candidate #4 (index 3) over shape #1
    assert tile3d_tiebreak_winner(_BERLIN_SCORES, _BERLIN_ERRS, 0) == 3


def test_tiebreak_noops_on_ulm_midrise() -> None:
    # weak margin (8%), winner err 7.4 deg, winner shape #7 -> all gates fail
    assert tile3d_tiebreak_winner(_ULM_SCORES, _ULM_ERRS, 0) is None


def test_tiebreak_noop_when_winner_is_already_consensus() -> None:
    assert tile3d_tiebreak_winner(_BERLIN_SCORES, _BERLIN_ERRS, 3) is None


def test_tiebreak_gates_each_condition() -> None:
    base = [0.20, 0.10, 0.05]        # 50% margin, clear winner at index 0
    errs = [4.0, 5.0, 5.0]
    assert tile3d_tiebreak_winner(base, errs, 1) == 0          # fires
    # margin too small
    assert tile3d_tiebreak_winner([0.20, 0.19, 0.05], errs, 1) is None
    # winner skyline error too high
    assert tile3d_tiebreak_winner(base, [7.0, 5.0, 5.0], 1) is None
    # winner not in the top-5 shape ranks
    big = [0.05, 0.05, 0.05, 0.05, 0.05, 0.20]
    assert tile3d_tiebreak_winner(big, [4.0] * 6, 0) is None
    # missing scores / too few candidates
    assert tile3d_tiebreak_winner([0.2, None, 0.1], errs, 1) is None
    assert tile3d_tiebreak_winner([0.2], [4.0], 0) is None


# --------------------------------------------------------------------------
# Uncertainty-aware fusion weight (adaptive_tile3d_weight)
# --------------------------------------------------------------------------

def test_adaptive_weight_full_on_high_rise_muted_on_flat() -> None:
    # Berlin scores: clear 20% margin -> ~full weight; Ulm: 8% -> muted
    wb = adaptive_tile3d_weight(_BERLIN_SCORES)
    wu = adaptive_tile3d_weight(_ULM_SCORES)
    assert wb > 0.3
    assert wu < 0.1
    assert wb > wu


def test_adaptive_weight_edges() -> None:
    assert adaptive_tile3d_weight([]) == 0.0
    assert adaptive_tile3d_weight([0.1]) == 0.0
    assert adaptive_tile3d_weight([0.0, 0.0]) == 0.0
    flat = adaptive_tile3d_weight([0.10, 0.099, 0.098])   # ~1% margin
    assert flat == pytest.approx(0.0, abs=1e-6)
    peaked = adaptive_tile3d_weight([0.30, 0.10, 0.05])   # 67% margin
    assert peaked == pytest.approx(0.4, abs=1e-6)         # capped at w_base
    assert adaptive_tile3d_weight([0.30, 0.10], w_base=1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Metric placement refinement (refine_placement_skyline)
# --------------------------------------------------------------------------

def _canyon_mesh():
    rng = np.random.default_rng(7)
    tris, grounds = [], []
    for xi in range(0, 400, 25):
        h = 12.0 + float(rng.uniform(0, 14))
        for side in (-18.0, 14.0):
            tris.append(_box_mesh(xi, xi + 14, side, side + 8, 0.0, h))
            grounds.append([xi + 7, side + 4, 0.0])
    return Lod2Mesh(triangles=np.concatenate(tris).astype(np.float32),
                    building_ground=np.asarray(grounds, dtype=np.float32),
                    crs="EPSG:32633", provider="berlin", n_buildings=len(grounds))


def _canyon_video(mesh, true_traj, K_wh):
    samples = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        pos, fwd = pose_at_fraction(true_traj, frac)
        m = render_building_mask(mesh.triangles_near(pos, 500.0), pos, 2.2,
                                 fwd, K_wh, WH)
        frame = np.full((720, 1280, 3), 235, dtype=np.uint8)
        big = cv2.resize(m, (1280, 720), interpolation=cv2.INTER_NEAREST) > 0
        frame[big] = (70, 70, 80)
        samples.append((frac, frame))
    return samples


def test_refine_placement_recovers_toward_truth() -> None:
    mesh = _canyon_mesh()
    true_traj = np.stack([np.linspace(0, 380, 128), np.zeros(128)], axis=1)
    samples = _canyon_video(mesh, true_traj, _k(*WH))
    center = true_traj.mean(axis=0)
    perturbed = _apply_rigid(true_traj, 40.0, -25.0, np.radians(4.0), center)
    before = float(np.linalg.norm(perturbed - true_traj, axis=1).mean())
    refined, info = refine_placement_skyline(mesh, samples, perturbed,
                                             _k(1280, 720), (1280, 720),
                                             max_fev=80)
    after = float(np.linalg.norm(refined - true_traj, axis=1).mean())
    assert info["applied"]
    assert info["score_after"] > info["score_before"]
    assert after < before                       # moved toward the truth


def test_refine_placement_noop_when_already_aligned() -> None:
    mesh = _canyon_mesh()
    true_traj = np.stack([np.linspace(0, 380, 128), np.zeros(128)], axis=1)
    samples = _canyon_video(mesh, true_traj, _k(*WH))
    refined, info = refine_placement_skyline(mesh, samples, true_traj,
                                             _k(1280, 720), (1280, 720),
                                             max_fev=40)
    # already aligned: refinement must not push it far away
    drift = float(np.linalg.norm(refined - true_traj, axis=1).mean())
    assert drift < 12.0


def test_refine_placement_no_usable_frames_is_noop() -> None:
    mesh = _canyon_mesh()
    traj = np.stack([np.linspace(0, 380, 64), np.zeros(64)], axis=1)
    night = [(0.5, np.full((720, 1280, 3), 8, dtype=np.uint8))]
    refined, info = refine_placement_skyline(mesh, night, traj,
                                             _k(1280, 720), (1280, 720))
    assert not info["applied"]
    assert np.array_equal(refined, traj)
