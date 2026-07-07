"""Tests for pipeline input acquisition and the geocode error wrapper.

These exercise `_resolve_input_video` / `_fetch_road_graph` directly so
no network, video decoding, or OSM access is involved.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import src.pipeline as pipeline
from src.pipeline import (
    PipelineConfig,
    _auto_estimated_length_m,
    _fetch_road_graph,
    _find_vo_cache,
    _heading_diff_deg,
    _length_sane,
    _match_timestamps,
    _mean_bearing_deg,
    _remap_frame_pair_to_poses,
    _resolve_input_video,
    _sun_bearing_penalty,
    _vpr_distance_penalty,
    _vpr_sequence_median_m,
)


def _cfg(tmp_path: Path, **overrides: object) -> PipelineConfig:
    defaults: dict = dict(
        url="https://example.com/video",
        city="Ulm, Germany",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
    )
    defaults.update(overrides)
    cfg = PipelineConfig(**defaults)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _forbid_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("download_video must not be called")

    monkeypatch.setattr(pipeline, "download_video", _boom)


# ---------------------------------------------------------------------------
# _resolve_input_video
# ---------------------------------------------------------------------------


def test_local_video_used_directly_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_download(monkeypatch)
    video = tmp_path / "drive.mp4"
    video.write_bytes(b"\x00")
    cfg = _cfg(tmp_path, video_path=video)
    assert _resolve_input_video(cfg) == video


def test_local_video_wins_over_skip_download_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --video beats a stale cached input.* in data/."""
    _forbid_download(monkeypatch)
    video = tmp_path / "drive.mp4"
    video.write_bytes(b"\x00")
    cfg = _cfg(tmp_path, video_path=video, skip_download=True)
    (cfg.data_dir / "input.mp4").write_bytes(b"\x01")
    assert _resolve_input_video(cfg) == video


def test_missing_local_video_raises_with_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, video_path=tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        _resolve_input_video(cfg)


def test_skip_download_returns_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_download(monkeypatch)
    cfg = _cfg(tmp_path, skip_download=True)
    cached = cfg.data_dir / "input.mp4"
    cached.write_bytes(b"\x00")
    assert _resolve_input_video(cfg) == cached


def test_skip_download_without_cache_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, skip_download=True)
    with pytest.raises(FileNotFoundError, match="skip-download"):
        _resolve_input_video(cfg)


def test_url_branch_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []
    sentinel = tmp_path / "data" / "input.mp4"

    def fake_download(url: str, out_dir: Path) -> Path:
        calls.append((url, out_dir))
        return sentinel

    monkeypatch.setattr(pipeline, "download_video", fake_download)
    cfg = _cfg(tmp_path)
    assert _resolve_input_video(cfg) == sentinel
    assert calls == [(cfg.url, cfg.data_dir)]


def test_skip_download_prefers_mp4_over_glob_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With several cached inputs the pick must be deterministic: prefer
    .mp4, not whatever the filesystem glob returns first (input.mkv sorts
    before input.mp4 alphabetically and used to win)."""
    _forbid_download(monkeypatch)
    cfg = _cfg(tmp_path, skip_download=True)
    (cfg.data_dir / "input.mkv").write_bytes(b"\x00")
    (cfg.data_dir / "input.mp4").write_bytes(b"\x00")
    (cfg.data_dir / "input.webm").write_bytes(b"\x00")
    assert _resolve_input_video(cfg) == cfg.data_dir / "input.mp4"


# ---------------------------------------------------------------------------
# _fetch_road_graph
# ---------------------------------------------------------------------------


def test_fetch_road_graph_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        pipeline, "fetch_city_graph",
        lambda city, cache_path, around=None: sentinel,
    )
    assert _fetch_road_graph("Ulm, Germany", tmp_path / "g.graphml") is sentinel


def test_fetch_road_graph_passes_around(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}
    monkeypatch.setattr(
        pipeline, "fetch_city_graph",
        lambda city, cache_path, around=None: seen.setdefault("around", around),
    )
    _fetch_road_graph("London, UK", tmp_path / "g.graphml", around=(51.5, -0.13, 2500.0))
    assert seen["around"] == (51.5, -0.13, 2500.0)


def test_fetch_road_graph_wraps_geocode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(city: str, cache_path: Path, around=None) -> None:
        raise RuntimeError("Nominatim could not geocode 'Ulmm'")

    monkeypatch.setattr(pipeline, "fetch_city_graph", boom)
    with pytest.raises(ValueError) as exc_info:
        _fetch_road_graph("Ulmm, Germany", tmp_path / "g.graphml")

    msg = str(exc_info.value)
    assert "Ulmm, Germany" in msg
    assert "City, Country" in msg
    # The original exception must be preserved for debugging.
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# _auto_estimated_length_m
# ---------------------------------------------------------------------------


def test_auto_length_matches_urban_speed() -> None:
    # 415 s at ~5.5 m/s — calibrated against the Ulm GT track, whose
    # true 7-minute route is ~2.1-2.4 km (the old fixed 8000 m default
    # was nearly 4x too long and caused a ~880 m start error).
    length = _auto_estimated_length_m(415.0)
    assert 2000.0 <= length <= 2600.0


def test_auto_length_clamps_extremes() -> None:
    assert _auto_estimated_length_m(0.0) == 500.0       # floor
    assert _auto_estimated_length_m(10.0) == 500.0      # below floor
    assert _auto_estimated_length_m(1e6) == 12000.0     # ceiling


def test_estimated_length_default_is_auto() -> None:
    """PipelineConfig must default to None so the duration-derived prior
    kicks in; a fixed default re-introduces the length-mismatch bug."""
    cfg = PipelineConfig(
        url="u", city="c", data_dir=Path("d"), output_dir=Path("o")
    )
    assert cfg.estimated_length_m is None


# ---------------------------------------------------------------------------
# _da3_trajectory_plausible
# ---------------------------------------------------------------------------


def test_da3_plausible_accepts_smooth_drive() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    # An L-shaped drive: 20 steps east, then 20 steps north.
    path = np.array(
        [[i, 0.0] for i in range(20)] + [[19.0, j] for j in range(1, 21)]
    )
    assert _da3_trajectory_plausible(path) is True


def test_da3_plausible_rejects_zigzag_scribble() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    rng = np.random.default_rng(0)
    scribble = rng.uniform(-1, 1, size=(48, 2))  # failed pose solve
    assert _da3_trajectory_plausible(scribble) is False


def test_da3_plausible_rejects_degenerate_paths() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    assert _da3_trajectory_plausible(np.zeros((2, 2))) is False      # too short
    assert _da3_trajectory_plausible(np.zeros((10, 2))) is False     # stationary


def test_da3_plausible_tolerates_single_u_turn() -> None:
    import numpy as np

    from src.pipeline import _da3_trajectory_plausible

    # Drive 30 steps east, U-turn, 30 steps back west: exactly one
    # reversal among 59 segment pairs — a real maneuver, must pass.
    path = np.array(
        [[i, 0.0] for i in range(30)] + [[29.0 - i, 0.01] for i in range(1, 31)]
    )
    assert _da3_trajectory_plausible(path) is True


# ---------------------------------------------------------------------------
# _fuse_bev_rank
# ---------------------------------------------------------------------------


def test_fuse_bev_flips_a_near_tie() -> None:
    from src.pipeline import _fuse_bev_rank

    # Geometric scores nearly tied; BevSplat strongly prefers candidate 1.
    base = [3.5, 4.0, 9.0]
    bev_ranks = [3, 1, 2]
    order = _fuse_bev_rank(base, bev_ranks, w_bev=0.75)
    assert order[0] == 1  # 4.0 + 0.75 < 3.5 + 2.25


def test_fuse_bev_cannot_override_a_large_geometric_gap() -> None:
    from src.pipeline import _fuse_bev_rank

    base = [2.0, 12.0, 13.0]
    bev_ranks = [3, 1, 2]   # appearance prefers the geometric losers
    order = _fuse_bev_rank(base, bev_ranks, w_bev=0.75)
    assert order[0] == 0


def test_fuse_bev_ties_break_by_incoming_order() -> None:
    from src.pipeline import _fuse_bev_rank

    base = [5.0, 5.0]
    bev_ranks = [1, 1]
    assert _fuse_bev_rank(base, bev_ranks) == [0, 1]


def test_fuse_bev_cap_blocks_longshot_promotion() -> None:
    """The 10-min Ulm backfire: a geometrically-implausible candidate
    (base rank 6) that BevSplat loves (rank 1) must NOT reach #1 — the
    cap keeps it out of the reorderable shortlist."""
    from src.pipeline import _fuse_bev_rank

    # 6 candidates; index 5 is geometrically worst (base 12) but bev #1.
    base = [1.0, 2.0, 3.0, 4.0, 5.0, 12.0]
    bev_ranks = [6, 5, 4, 3, 2, 1]
    order = _fuse_bev_rank(base, bev_ranks, w_bev=0.75, cap=5)
    assert order[-1] == 5            # long-shot stays in the tail
    assert 5 not in order[:5]        # never enters the reordered shortlist


def test_fuse_bev_cap_still_reorders_within_shortlist() -> None:
    """Within the geometric top-cap, appearance still reorders."""
    from src.pipeline import _fuse_bev_rank

    base = [3.5, 4.0, 9.0, 10.0, 11.0, 12.0]
    bev_ranks = [3, 1, 2, 4, 5, 6]
    order = _fuse_bev_rank(base, bev_ranks, w_bev=0.75, cap=5)
    assert order[0] == 1             # near-tie flipped by appearance
    assert order[-1] == 5            # worst geometry stays last


def test_fuse_bev_cap_at_full_length_is_unconstrained() -> None:
    from src.pipeline import _fuse_bev_rank

    base = [2.0, 12.0, 13.0]
    bev_ranks = [3, 1, 2]
    # cap >= len reproduces plain fusion (here geometry still wins).
    assert _fuse_bev_rank(base, bev_ranks, cap=99) == _fuse_bev_rank(
        base, bev_ranks, cap=3
    )


# ---------------------------------------------------------------------------
# _length_sane — user-provided --estimated-length-m must widen the gate
# ---------------------------------------------------------------------------


def test_length_sane_against_duration_prior_only() -> None:
    assert _length_sane(2000.0, 2310.0) is True
    assert _length_sane(7900.0, 2310.0) is False       # >2x the prior


def test_length_sane_accepts_user_length() -> None:
    """Highway clip: user passed --estimated-length-m 8000; a correct
    recovered ~7900 m must not be rejected against the 2310 m duration
    prior (the old gate compared only against the duration prior)."""
    assert _length_sane(7900.0, 2310.0, user_length_m=8000.0) is True
    # Still rejects lengths far from BOTH references.
    assert _length_sane(25000.0, 2310.0, user_length_m=8000.0) is False


# ---------------------------------------------------------------------------
# _find_vo_cache — shape check must gate the canonical key too
# ---------------------------------------------------------------------------


def _write_vo_npz(path: Path, n: int) -> None:
    np.savez(
        path,
        centers=np.zeros((n, 3)),
        xz=np.zeros((n, 2)),
        valid=np.ones(n, dtype=bool),
        n_inliers=np.zeros(n),
        rotations=np.zeros((n, 3, 3)),
        translations=np.zeros((n, 3)),
    )


def test_stale_canonical_vo_cache_is_rejected(tmp_path: Path) -> None:
    """A canonical cache that FAILS the frame-count check must not be
    selected (the old load condition let it through because it existed
    under the canonical key)."""
    canonical = tmp_path / "trajectory_v2_0-420_s3_fauto.npz"
    _write_vo_npz(canonical, 4196)   # stale: from a different stream fps
    assert _find_vo_cache([canonical], n_frames=3496) is None


def test_matching_vo_cache_is_selected_and_handle_closed(tmp_path: Path) -> None:
    canonical = tmp_path / "trajectory_v2_0-420_s3_fauto.npz"
    sibling = tmp_path / "trajectory_v2_0-420_s3_f4200.npz"
    _write_vo_npz(canonical, 999)    # wrong shape
    _write_vo_npz(sibling, 100)      # matches
    picked = _find_vo_cache([canonical, sibling], n_frames=100)
    assert picked == sibling
    # Probe handles must be CLOSED: on Windows an open npz handle blocks
    # deletion of the file (the old bare np.load leaked them).
    os.remove(canonical)
    os.remove(sibling)


# ---------------------------------------------------------------------------
# _match_timestamps / _remap_frame_pair_to_poses — staged-trajectory axes
# ---------------------------------------------------------------------------


def test_match_timestamps_spans_clip_over_pose_rows() -> None:
    """OpenVO case: 4196 frames but 1260 poses — the axis must have one
    entry per POSE, spanning the clip's time range."""
    ts = list(np.linspace(0.0, 419.5, 4196))
    mts = _match_timestamps(ts, 1260)
    assert len(mts) == 1260
    assert mts[0] == pytest.approx(0.0)
    assert mts[-1] == pytest.approx(419.5)
    assert np.all(np.diff(mts) > 0)


def test_match_timestamps_identity_when_lengths_match() -> None:
    ts = [0.0, 0.2, 0.4, 0.6]
    assert np.allclose(_match_timestamps(ts, 4), ts)


def test_match_timestamps_anchor_lookup_stays_in_bounds() -> None:
    """An anchor at t=245 s used to map to frame index ~2447 in a
    1260-row OpenVO trajectory -> IndexError. With the aligned axis the
    nearest-time lookup is in bounds by construction."""
    from src.scale_recovery import vo_positions_at_times

    n_frames, n_poses = 4196, 1260
    frame_ts = list(np.linspace(0.0, 419.5, n_frames))
    match_xz = np.column_stack([np.arange(n_poses, dtype=float),
                                np.zeros(n_poses)])
    mts = _match_timestamps(frame_ts, n_poses)
    pos = vo_positions_at_times(match_xz, mts, np.array([245.0]))
    # ~245/419.5 of the way through the poses.
    assert pos[0, 0] == pytest.approx(245.0 / 419.5 * (n_poses - 1), abs=1.0)


def test_remap_frame_pair_scales_into_pose_space() -> None:
    """The Ulm OpenVO case: a loop detected on frames (10, 4190) used to
    be discarded because 4190 >= 1260 poses. It must remap
    proportionally instead."""
    pair = _remap_frame_pair_to_poses((10, 4190), n_frames=4196, n_poses=1260)
    assert pair is not None
    i, j = pair
    assert 0 <= i < j < 1260
    assert j == pytest.approx(4190 * 1259 / 4195, abs=1.0)


def test_remap_frame_pair_identity_when_lengths_equal() -> None:
    assert _remap_frame_pair_to_poses((3, 90), 100, 100) == (3, 90)


def test_remap_frame_pair_degenerate_returns_none() -> None:
    assert _remap_frame_pair_to_poses(None, 100, 50) is None
    # Collapses to the same pose index -> unusable.
    assert _remap_frame_pair_to_poses((10, 12), 4196, 5) is None
    assert _remap_frame_pair_to_poses((3, 90), 100, 1) is None


# ---------------------------------------------------------------------------
# _vpr_distance_penalty — free radius inside the prior's own error bar
# ---------------------------------------------------------------------------


def test_vpr_penalty_free_inside_prior_error_bar() -> None:
    """Candidates within ~150 m of the prior are indistinguishable from
    the truth and must pay nothing (the old 15*d_km penalized a 100 m
    candidate 1.5 rank-units — enough to flip a near-tie)."""
    assert _vpr_distance_penalty(0.0) == 0.0
    assert _vpr_distance_penalty(100.0) == 0.0
    assert _vpr_distance_penalty(150.0) == 0.0


def test_vpr_penalty_grows_beyond_free_radius() -> None:
    assert _vpr_distance_penalty(1150.0) == pytest.approx(15.0)
    assert _vpr_distance_penalty(2150.0) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# _vpr_sequence_median_m — sequence mode separates flipped candidates
# ---------------------------------------------------------------------------


def test_vpr_sequence_median_zero_for_aligned_candidate() -> None:
    n_frames = 101
    traj = np.column_stack([np.linspace(0, 100, 101), np.zeros(101)])
    track_idx = [0, 50, 100]
    track_xy = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    med = _vpr_sequence_median_m(traj, track_idx, track_xy, [1, 1, 1], n_frames)
    assert med == pytest.approx(0.0)


def test_vpr_sequence_median_penalizes_flipped_candidate() -> None:
    """A 180-deg-flipped candidate has the SAME centroid distance (0) but
    a much worse sequence distance — the ambiguity class the centroid
    penalty cannot rank."""
    n_frames = 101
    flipped = np.column_stack([np.linspace(100, 0, 101), np.zeros(101)])
    track_idx = [0, 50, 100]
    track_xy = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    med = _vpr_sequence_median_m(flipped, track_idx, track_xy, [1, 1, 1], n_frames)
    assert med == pytest.approx(100.0)


def test_vpr_sequence_median_degenerate_returns_none() -> None:
    assert _vpr_sequence_median_m(np.zeros((1, 2)), [0], np.zeros((1, 2)),
                                  [1.0], 10) is None
    assert _vpr_sequence_median_m(np.zeros((5, 2)), [], np.zeros((0, 2)),
                                  [], 10) is None


def test_use_vpr_sequence_defaults_off() -> None:
    cfg = PipelineConfig(url="u", city="c", data_dir=Path("d"), output_dir=Path("o"))
    assert cfg.use_vpr_sequence is False


# ---------------------------------------------------------------------------
# Sun-heading orientation penalty
# ---------------------------------------------------------------------------


def test_mean_bearing_compass_convention() -> None:
    east = np.array([[0.0, 0.0], [10.0, 0.0]])
    north = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert _mean_bearing_deg(east) == pytest.approx(90.0)
    assert _mean_bearing_deg(north) == pytest.approx(0.0)
    assert _mean_bearing_deg(np.zeros((3, 2))) is None   # stationary


def test_heading_diff_wraps() -> None:
    assert _heading_diff_deg(350.0, 10.0) == pytest.approx(20.0)
    assert _heading_diff_deg(90.0, 270.0) == pytest.approx(180.0)


def test_sun_penalty_free_within_tolerance_and_grows() -> None:
    assert _sun_bearing_penalty(90.0, 90.0) == 0.0
    assert _sun_bearing_penalty(90.0, 118.0) == 0.0          # inside 30 deg
    assert _sun_bearing_penalty(0.0, 180.0) == pytest.approx(5.0)   # mirror
    mid = _sun_bearing_penalty(0.0, 105.0)
    assert 0.0 < mid < 5.0


# ---------------------------------------------------------------------------
# Refuted gating knobs are gone
# ---------------------------------------------------------------------------


def test_refuted_gate_config_fields_removed() -> None:
    """VPR/plate OSM gating was refuted by experiment; the dead config
    knobs must not silently parse."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(PipelineConfig)}
    assert "vpr_gate" not in names
    assert "vpr_gate_radius_m" not in names
    assert "plate_gate_radius_m" not in names


# ---------------------------------------------------------------------------
# Here-vs-direction sign classification wiring (--classify-signs)
# ---------------------------------------------------------------------------


def _fake_capture(monkeypatch):
    import cv2

    class FakeCap:
        def __init__(self, *a):
            pass

        def set(self, *a):
            return True

        def read(self):
            return True, np.zeros((24, 24, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)


def test_drop_direction_anchors_removes_direction_signs(monkeypatch) -> None:
    """A directional sign (Holborn) is dropped; a here-sign (Russell Square)
    survives — the London 'Holborn' failure fix, wired at the pipeline level."""
    import src.vlm_anchor as vlm
    from src.scene_text import SceneText
    from src.text_anchor import PoiAnchor

    anchors = [
        PoiAnchor(name="Holborn", lat=51.517, lon=-0.120, confidence=0.9, t_sec=10.0),
        PoiAnchor(name="Russell Square", lat=51.523, lon=-0.127, confidence=0.9, t_sec=20.0),
    ]
    dets = [
        SceneText("Holborn", 0.9, 10.0, (1, 1, 9, 9)),
        SceneText("Russell Square", 0.9, 20.0, (1, 1, 9, 9)),
    ]
    _fake_capture(monkeypatch)
    monkeypatch.setattr(vlm, "classify_sign_types",
                        lambda frames, recs: ["direction" if r.text == "Holborn"
                                              else "here" for r in recs])
    kept, dropped = pipeline._drop_direction_anchors(anchors, dets, "x.mp4")
    assert dropped == ["Holborn"]
    assert [a.name for a in kept] == ["Russell Square"]


def test_drop_direction_anchors_noop_when_all_here(monkeypatch) -> None:
    """When nothing classifies as 'direction', the anchor list is unchanged."""
    import src.vlm_anchor as vlm
    from src.scene_text import SceneText
    from src.text_anchor import PoiAnchor

    anchors = [PoiAnchor(name="Sedelhoefe", lat=48.4, lon=9.99, confidence=0.9, t_sec=5.0)]
    dets = [SceneText("Sedelhoefe", 0.9, 5.0, (0, 0, 8, 8))]
    _fake_capture(monkeypatch)
    monkeypatch.setattr(vlm, "classify_sign_types", lambda frames, recs: ["here"])
    kept, dropped = pipeline._drop_direction_anchors(anchors, dets, "x.mp4")
    assert dropped == []
    assert kept is anchors


def test_drop_direction_anchors_empty_input() -> None:
    kept, dropped = pipeline._drop_direction_anchors([], [], "x.mp4")
    assert kept == [] and dropped == []


# ---------------------------------------------------------------------------
# _vpr_track_geometry — shared sort -> robust-centre -> project helper
# ---------------------------------------------------------------------------


_UTM32N = "EPSG:32632"  # Ulm sits in UTM zone 32N


def _cluster_track(n_side: int = 16, jitter_deg: float = 0.00005):
    """Synthetic VPR track: a start cluster near (48.40, 9.98) and an end
    cluster near (48.40, 10.00), returned in SHUFFLED frame order."""
    rng = np.random.default_rng(7)
    start = np.array([48.40, 9.98])
    end = np.array([48.40, 10.00])
    ll = np.vstack([
        start + rng.normal(0, jitter_deg, (n_side, 2)),
        end + rng.normal(0, jitter_deg, (n_side, 2)),
    ])
    idx = np.arange(2 * n_side)
    sims = np.linspace(0.5, 0.9, 2 * n_side)
    perm = rng.permutation(2 * n_side)
    return (idx[perm], ll[perm], sims[perm]), start, end


def test_vpr_track_geometry_sorts_and_projects() -> None:
    track, start, end = _cluster_track()
    geo = pipeline._vpr_track_geometry(track, _UTM32N)
    assert geo is not None
    vi, vll, vs, s_xy, e_xy = geo
    # sorted by frame index, arrays kept parallel
    assert list(vi) == sorted(vi)
    assert vll.shape == (32, 2) and vs.shape == (32,)
    from src.position import latlon_to_xy
    true_ends = latlon_to_xy(np.vstack([start, end]), _UTM32N)
    # robust centres land on the cluster centres (jitter is ~5 m)
    assert np.linalg.norm(s_xy - true_ends[0]) < 30.0
    assert np.linalg.norm(e_xy - true_ends[1]) < 30.0
    # chord ~= 0.02 deg of longitude at 48.4N ~= 1.48 km
    chord = np.linalg.norm(e_xy - s_xy)
    assert 1300.0 < chord < 1650.0


def test_vpr_track_geometry_none_on_missing_or_empty() -> None:
    assert pipeline._vpr_track_geometry(None, _UTM32N) is None
    empty = (np.array([]), np.zeros((0, 2)), np.array([]))
    assert pipeline._vpr_track_geometry(empty, _UTM32N) is None


# ---------------------------------------------------------------------------
# _rotation_refine_about_start — orientation residual on the pinned candidate
# ---------------------------------------------------------------------------


def _l_route(n: int = 200) -> np.ndarray:
    """L-shaped ~1.5 km route in local metres, start at the origin."""
    a = np.linspace([0.0, 0.0], [1000.0, 0.0], n // 2)
    b = np.linspace([1000.0, 0.0], [1000.0, 500.0], n - n // 2)
    return np.vstack([a, b])


def _track_for(traj: np.ndarray, n_frames: int = 300, n_track: int = 40,
               noise_m: float = 20.0, seed: int = 3):
    """Per-frame 'VPR' observations of ``traj`` at uniform frame fractions."""
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, n_frames - 1, n_track).astype(int)
    rows = np.round(idx / (n_frames - 1) * (len(traj) - 1)).astype(int)
    track = traj[rows] + rng.normal(0, noise_m, (n_track, 2))
    return idx, track, np.ones(n_track)


def _rot_about_start(traj: np.ndarray, deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]])
    return (R @ (traj - traj[0]).T).T + traj[0]


def test_rotation_refine_recovers_true_heading() -> None:
    """A candidate rotated 60 deg off the track's heading is rotated back:
    the far end returns to truth and both margin gates pass."""
    truth = _l_route()
    idx, track, sims = _track_for(truth)
    cand = _rot_about_start(truth, 60.0)
    out = pipeline._rotation_refine_about_start(cand, idx, track, sims, 300)
    assert out is not None
    rot, deg, base, best = out
    assert best < 0.75 * base and base - best >= 20.0
    # recovered angle cancels the applied one (circular diff)
    assert abs(((deg + 60.0) + 180.0) % 360.0 - 180.0) < 1.5
    assert np.linalg.norm(rot[-1] - truth[-1]) < 60.0
    # the pivot never moves: start stays pinned
    assert np.allclose(rot[0], cand[0], atol=1e-9)


def test_rotation_refine_robust_to_confident_outliers() -> None:
    """25% confident-but-wrong track points (the failure mode that sank the
    endpoint-based 2-point fit) do not break the median objective."""
    truth = _l_route()
    idx, track, sims = _track_for(truth)
    n_bad = len(track) // 4
    track[:n_bad] = track[:n_bad] + np.array([2500.0, -1800.0])
    sims[:n_bad] = 1.2                      # outliers claim top confidence
    cand = _rot_about_start(truth, -75.0)
    out = pipeline._rotation_refine_about_start(cand, idx, track, sims, 300)
    assert out is not None
    rot, deg, base, best = out
    assert abs(((deg - 75.0) + 180.0) % 360.0 - 180.0) < 3.0
    assert np.linalg.norm(rot[-1] - truth[-1]) < 100.0


def test_rotation_refine_noop_when_already_aligned() -> None:
    """A correctly-oriented candidate gains nothing worth the margin gates ->
    None (keep the matcher's heading; never chase track noise)."""
    truth = _l_route()
    idx, track, sims = _track_for(truth)
    assert pipeline._rotation_refine_about_start(truth, idx, track, sims, 300) is None


def test_rotation_refine_rejects_unexplainable_track() -> None:
    """A track the route cannot lie on at ANY rotation (translated 1.2 km off
    the pinned start) fails the explain gate even if some rotation helps."""
    truth = _l_route()
    idx, track, sims = _track_for(truth)
    track = track + np.array([1200.0, 900.0])
    out = pipeline._rotation_refine_about_start(truth, idx, track, sims, 300)
    assert out is None


# ---------------------------------------------------------------------------
# _pick_top_k_placement — the fused track median arbitrates candidate choice
# ---------------------------------------------------------------------------


def test_top_k_placement_challenger_needs_the_margin() -> None:
    # 28 vs 30 m: inside the 10%+10 m margin -> rank 0 keeps the answer
    assert pipeline._pick_top_k_placement([(0, 30.0), (1, 28.0)]) == 0
    # 60 vs 200 m: decisive -> the challenger wins
    assert pipeline._pick_top_k_placement([(0, 200.0), (2, 60.0)]) == 2


def test_top_k_placement_rank0_wins_ties_and_none_challengers() -> None:
    assert pipeline._pick_top_k_placement([(0, 50.0), (1, 50.0)]) == 0
    assert pipeline._pick_top_k_placement([(0, 50.0), (1, None)]) == 0
    assert pipeline._pick_top_k_placement([]) is None


def test_top_k_placement_unplaceable_rank0_hands_off() -> None:
    """When rank 0 failed every pin guard, the best challenger takes it."""
    assert pipeline._pick_top_k_placement([(1, 90.0), (2, 40.0)]) == 2


def test_place_candidate_on_track_full_chain() -> None:
    """A drifted candidate runs the whole pin->rotate->fuse chain and comes
    back with a finite track median + the winner's log notes."""
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=40, noise_m=15.0)
    drifted = _rot_about_start(_drifted(truth, 120.0), 60.0)
    s_xy, e_xy = truth[0], truth[-1]
    vpr_xy = truth.mean(axis=0)
    res = pipeline._place_candidate_on_track(
        drifted, s_xy, e_xy, vpr_xy, idx, track, sims, 300)
    assert res is not None
    placed, mode, rot_deg, med, notes = res
    assert "startpin" in mode and "+rot" in mode and "+fuse" in mode
    assert med is not None and med < 60.0
    assert np.allclose(placed[0], s_xy, atol=1e-6)   # pin survives the chain
    assert len(notes) == 2


def test_place_candidate_rescue_tries_second_pin_after_first_fails_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (bug 2026-07-05): when BOTH pins fail the 400 m centroid
    guard, the rescue must try EACH pin's rotation. A bare `break` abandoned
    the second attempt whenever the first pin's rotation EXISTED but missed
    the guard (as opposed to returning None, which `continue` already handled).

    We drive the control flow directly by stubbing the rotation: the scaled
    pin (first `tries` entry) gets a valid rotation whose centroid still fails
    the guard; the plain pin (second entry) gets one that lands on the prior.
    Both pins must fail the INITIAL guard first so the rescue path (two
    entries) is taken: the candidate runs east while the prior is north, and a
    bad far-east end estimate over-scales the scale-retry pin to 2 km."""
    n = 10
    cand = np.column_stack([np.linspace(0, 1000.0, n), np.zeros(n)])    # 1 km E
    s_xy = np.array([0.0, 0.0])
    e_xy_bad = np.array([5000.0, 0.0])     # -> scale retry factor clips to 2
    vpr_xy = np.array([0.0, 500.0])        # prior north of the candidate
    idx = np.arange(n)
    track = np.column_stack([np.zeros(n), np.linspace(0, 1000.0, n)])
    sims = np.ones(n)

    seen_extents: list[int] = []

    def fake_rot(t, vi, track_xy, vs, n_frames):
        ext = round(float(np.linalg.norm(t[-1] - t[0])))
        seen_extents.append(ext)
        if ext > 1500:                     # the scaled 2 km pin: guard-failing
            return (t + np.array([0.0, 3000.0]), 90.0, 500.0, 100.0)
        placed = np.column_stack([np.zeros(len(t)),                     # on prior
                                  np.linspace(0.0, 1000.0, len(t))])
        return (placed, 90.0, 500.0, 100.0)

    monkeypatch.setattr(pipeline, "_rotation_refine_about_start", fake_rot)
    monkeypatch.setattr(pipeline, "_elastic_fuse_track", lambda *a, **k: None)
    res = pipeline._place_candidate_on_track(
        cand, s_xy, e_xy_bad, vpr_xy, idx, track, sims, n)
    assert res is not None                          # plain-pin rotation rescued
    assert len(seen_extents) == 2                   # BOTH pins tried (bug: 1)
    assert seen_extents[0] > 1500 and seen_extents[1] < 1500  # scaled then plain
    placed, mode, rot_deg, med, notes = res
    assert mode == "vpr_startpin+rot"
    assert np.linalg.norm(placed.mean(axis=0) - vpr_xy) <= 400.0


# ---------------------------------------------------------------------------
# _pose_txt_xz — KITTI 3x4 and VGGT-Long 4x4 pose files -> top-down path
# ---------------------------------------------------------------------------


def test_pose_txt_xz_reads_both_layouts(tmp_path: Path) -> None:
    tx, tz = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
    kitti = tmp_path / "openvo_trajectory.txt"
    kitti.write_text("\n".join(
        " ".join(str(v) for v in [1, 0, 0, tx[i], 0, 1, 0, 0.5, 0, 0, 1, tz[i]])
        for i in range(3)), encoding="utf-8")
    c2w = tmp_path / "camera_poses.txt"
    c2w.write_text("\n".join(
        " ".join(str(v) for v in [1, 0, 0, tx[i], 0, 1, 0, 0.5,
                                  0, 0, 1, tz[i], 0, 0, 0, 1])
        for i in range(3)), encoding="utf-8")
    for f in (kitti, c2w):
        xz = pipeline._pose_txt_xz(f)
        assert xz is not None and xz.shape == (3, 2)
        assert np.allclose(xz[:, 0], tx) and np.allclose(xz[:, 1], tz)


def test_pose_txt_xz_rejects_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 3\n4 5 6\n", encoding="utf-8")      # wrong width
    assert pipeline._pose_txt_xz(bad) is None
    nan = tmp_path / "nan.txt"
    nan.write_text(" ".join(["nan"] * 16) + "\n" + " ".join(["1"] * 16),
                   encoding="utf-8")
    assert pipeline._pose_txt_xz(nan) is None
    assert pipeline._pose_txt_xz(tmp_path / "missing.txt") is None


# ---------------------------------------------------------------------------
# _elastic_fuse_track — bend out low-frequency VO drift toward the VPR track
# ---------------------------------------------------------------------------


def _drifted(truth: np.ndarray, amp_m: float) -> np.ndarray:
    """Truth plus a smooth quadratic drift growing to ``amp_m`` at the end —
    the low-frequency VO error no rigid transform can express."""
    s = np.linspace(0.0, 1.0, len(truth))[:, None]
    return truth + np.array([[0.0, 1.0]]) * amp_m * s ** 2


def test_elastic_fuse_recovers_smooth_drift() -> None:
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=40, noise_m=15.0)
    drifted = _drifted(truth, 120.0)
    out = pipeline._elastic_fuse_track(drifted, idx, track, sims, 300)
    assert out is not None
    fused, rigid_med, fused_med = out
    assert fused_med < rigid_med
    # the drifted end sits 120 m off; fusion pulls it most of the way back
    assert np.linalg.norm(drifted[-1] - truth[-1]) > 100.0
    assert np.linalg.norm(fused[-1] - truth[-1]) < 45.0
    # the pinned start never moves
    assert np.allclose(fused[0], drifted[0], atol=1e-9)


def test_elastic_fuse_noop_when_already_on_track() -> None:
    """No drift to fix -> the margin gates reject chasing track noise."""
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=40, noise_m=15.0)
    assert pipeline._elastic_fuse_track(truth, idx, track, sims, 300) is None


def test_elastic_fuse_robust_to_confident_outliers() -> None:
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=40, noise_m=15.0)
    n_bad = len(track) // 4
    track[10:10 + n_bad] += np.array([1800.0, -2200.0])
    sims[10:10 + n_bad] = 1.5
    drifted = _drifted(truth, 120.0)
    out = pipeline._elastic_fuse_track(drifted, idx, track, sims, 300)
    assert out is not None
    assert np.linalg.norm(out[0][-1] - truth[-1]) < 60.0


def test_elastic_fuse_rejects_oversized_deformation() -> None:
    """A track offset 500 m from the route needs a >300 m bend -> None
    (that is a wrong-hypothesis signal, not drift)."""
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=40, noise_m=10.0)
    track = track + np.array([500.0, 0.0])
    assert pipeline._elastic_fuse_track(truth, idx, track, sims, 300) is None


def test_elastic_fuse_degenerate_inputs() -> None:
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=10)   # < min track pts
    assert pipeline._elastic_fuse_track(truth, idx, track, sims, 300) is None


# ---------------------------------------------------------------------------
# _orienternet_refine — track-gated acceptance of the neural refinement
# ---------------------------------------------------------------------------


def _onet_setup(tmp_path: Path, refined_shift_m: float, track_noise_m: float):
    """Build the fixtures for an _orienternet_refine call: a 1 km route near
    Ulm in EPSG:32632, a VPR track observing it, and a fake refine_route that
    returns the route shifted east by ``refined_shift_m``."""
    from types import SimpleNamespace

    from src.position import xy_to_latlon

    rng = np.random.default_rng(11)
    xy = np.column_stack([np.linspace(570000.0, 571000.0, 60),
                          np.full(60, 5360000.0)])
    n_frames = 300
    t_idx = np.linspace(0, n_frames - 1, 40).astype(int)
    rows = np.round(t_idx / (n_frames - 1) * (len(xy) - 1)).astype(int)
    track_ll = xy_to_latlon(
        xy[rows] + rng.normal(0, track_noise_m, (len(rows), 2)), _UTM32N)
    vpr_track = (t_idx, track_ll, np.ones(len(t_idx)))
    refined_ll = xy_to_latlon(xy + np.array([refined_shift_m, 0.0]), _UTM32N)

    cfg = _cfg(tmp_path)
    frames = SimpleNamespace(frames=list(range(n_frames)))
    cand = SimpleNamespace(aligned_traj_xy=xy)
    road = SimpleNamespace(crs=_UTM32N)
    position = {"latitude": 1.0, "longitude": 2.0}
    result: dict = {}
    return cfg, frames, cand, road, position, result, vpr_track, refined_ll


def test_orienternet_gate_rejects_off_track_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refinement that explains the VPR track worse than the route it
    started from must NOT overwrite the position (London: 31 -> 190 m)."""
    import src.orienternet_localizer as onl

    cfg, frames, cand, road, position, result, vpr_track, refined_ll = \
        _onet_setup(tmp_path, refined_shift_m=500.0, track_noise_m=5.0)
    monkeypatch.setattr(onl, "refine_route", lambda *a, **k: refined_ll)
    pipeline._orienternet_refine(cfg, frames, cand, road, position, result,
                                 vpr_track=vpr_track)
    assert position == {"latitude": 1.0, "longitude": 2.0}   # untouched
    assert result["orienternet"]["applied"] is False
    assert result["orienternet"]["track_fit_refined_m"] > \
        result["orienternet"]["track_fit_route_m"]


def test_orienternet_start_gate_rejects_dragged_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The London failure shape: the refined route's BODY still fits the track
    (track gate passes) but the START is dragged far off the pinned start ->
    rejected on start-pinned runs."""
    import src.orienternet_localizer as onl

    from src.position import xy_to_latlon

    cfg, frames, cand, road, position, result, vpr_track, _ = \
        _onet_setup(tmp_path, refined_shift_m=0.0, track_noise_m=40.0)
    result["anchored_position"] = {"source": "vpr_startpin+rot"}
    warped = np.asarray(cand.aligned_traj_xy, dtype=np.float64).copy()
    warped[:4] += np.array([300.0, 0.0])   # drag only the very start: the
    # body still fits the ~40 m-noise track (median barely moves, track gate
    # passes) — exactly how London slipped through on the track gate alone
    refined_ll = xy_to_latlon(warped, _UTM32N)
    monkeypatch.setattr(onl, "refine_route", lambda *a, **k: refined_ll)
    pipeline._orienternet_refine(cfg, frames, cand, road, position, result,
                                 vpr_track=vpr_track)
    assert position == {"latitude": 1.0, "longitude": 2.0}   # untouched
    assert result["orienternet"]["applied"] is False
    assert "pinned start" in result["orienternet"]["reason"]


def test_orienternet_gate_accepts_on_track_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refinement that fits the track as well as the prior route is applied
    (position overwritten, coarse kept)."""
    import src.orienternet_localizer as onl

    cfg, frames, cand, road, position, result, vpr_track, refined_ll = \
        _onet_setup(tmp_path, refined_shift_m=3.0, track_noise_m=30.0)
    monkeypatch.setattr(onl, "refine_route", lambda *a, **k: refined_ll)
    pipeline._orienternet_refine(cfg, frames, cand, road, position, result,
                                 vpr_track=vpr_track)
    assert result["orienternet"]["applied"] is True
    assert position["coarse_latitude"] == 1.0
    assert position["latitude"] == pytest.approx(refined_ll[0][0], abs=1e-4)


def test_rotation_refine_degenerate_inputs() -> None:
    """Too few track points or a stub route -> None (unobservable)."""
    truth = _l_route()
    idx, track, sims = _track_for(truth, n_track=5)
    assert pipeline._rotation_refine_about_start(truth, idx, track, sims, 300) is None
    stub = np.array([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]])
    idx2, track2, sims2 = _track_for(stub, n_track=12)
    assert pipeline._rotation_refine_about_start(stub, idx2, track2, sims2, 300) is None
