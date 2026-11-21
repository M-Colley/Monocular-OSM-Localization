"""Frontier extension of the coarse-to-fine disc (--vpr-c2f-extend)."""
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm.geo import LocalFrame, geodesic_distance_m
from monocular_osm.stages import prior

CENTRE = (50.0, 8.0)
FRAME = LocalFrame(*CENTRE)
R = 2000.0


def east(metres):
    return FRAME.to_latlon(float(metres), 0.0)


def test_rim_runs_count_only_the_ends():
    pts = [east(1900), east(1700), east(100), east(0), east(1600), east(1800), east(1950)]
    assert prior.frontier_end_runs(pts, CENTRE, R) == (2, 3)
    assert prior.frontier_end_runs([east(0)] * 4, CENTRE, R) == (0, 0)
    assert prior.frontier_end_runs([east(1900)] * 4, CENTRE, R) == (4, 4)
    assert prior.frontier_end_runs([], CENTRE, R) == (0, 0)


def test_next_disc_lies_beyond_the_frontier_run():
    new = prior.extension_centre([east(1800), east(1950)], CENTRE, R)
    x, y = FRAME.to_xy(*new)
    assert x == pytest.approx(1.6 * R, abs=1.0) and y == pytest.approx(0.0, abs=1.0)
    assert prior.extension_centre([CENTRE], CENTRE, R) is None
    assert prior.extension_centre([], CENTRE, R) is None


def _route(n):
    """A drive due east from the disc centre to 7 km: the last frames leave the disc."""
    return [east(x) for x in np.linspace(0.0, 7000.0, n)]


def _run(tmp_path, base_ll, *, max_discs=6, kept="pass-2", world=None):
    n = len(base_ll)
    world = _route(n) if world is None else world
    calls = []

    def track_fn(frames, center, radius_m, **kwargs):
        # Each disc sees the true positions inside it (cosine 0.3); frames whose
        # true position lies outside snap to the disc's nearest rim point (0.1).
        calls.append(tuple(center))
        out, sims = [], []
        for p in world:
            d = geodesic_distance_m(*center, *p)
            if d <= radius_m:
                out.append(p), sims.append(0.30)
            else:
                f = LocalFrame(*center)
                x, y = f.to_xy(*p)
                out.append(f.to_latlon(x * radius_m / d, y * radius_m / d)), sims.append(0.10)
        return np.arange(n), np.asarray(out), np.asarray(sims)

    base = (np.arange(n), np.asarray(base_ll), np.where(
        geodesic_distance_m(*CENTRE, np.asarray(world)[:, 0], np.asarray(world)[:, 1]) <= R,
        0.30, 0.10))
    cfg = SimpleNamespace(data_dir=tmp_path, vpr_ref_cap=1500, vpr_viterbi=True,
                          vpr_query_mode="warp", vpr_query_batch_size=4)
    ctx = SimpleNamespace(
        cfg=cfg, vpr_track=base, vpr_center=CENTRE, result={}, osm_around=(*CENTRE, R + 50),
        c2f_info={"kept": kept, "pass1_center": list(CENTRE), "pass2_disc_center": list(CENTRE),
                  "tight_radius_m": R},
        frames=SimpleNamespace(frames=[None] * n, timestamps=np.arange(n) * 5.0))
    prior._c2f_frontier_extend(ctx, source="mapillary", token="set", device="cpu",
                               query_dt_s=5.0, track_fn=track_fn, query_cache={},
                               max_discs=max_discs)
    return ctx, calls


def _snapped(n):
    """The ordinary pass-2 track for _route(n): frames past the rim sit on it."""
    return [p if geodesic_distance_m(*CENTRE, *p) <= R else east(R) for p in _route(n)]


def test_a_drive_past_the_rim_is_followed_disc_by_disc(tmp_path):
    n = 40
    ctx, calls = _run(tmp_path, _snapped(n))
    record = ctx.result["c2f_extension"]
    assert record["kept"] == "extended"
    assert record["gate"]["trail"] >= record["gate"]["min_run"] and record["gate"]["lead"] == 0
    assert len(calls) == 2                       # 3.2 km and 6.4 km east cover the 7 km drive
    err = geodesic_distance_m(ctx.vpr_track[1][:, 0], ctx.vpr_track[1][:, 1],
                              np.asarray(_route(n))[:, 0], np.asarray(_route(n))[:, 1])
    assert np.max(err) < 1.0
    assert len(record["baseline_track"]) == n
    assert ctx.osm_around[2] > R


def test_the_extension_budget_is_respected(tmp_path):
    ctx, calls = _run(tmp_path, _snapped(40), max_discs=1)
    assert len(calls) == 1
    assert len(ctx.result["c2f_extension"]["discs"]) == 2


def test_a_drive_inside_the_disc_is_left_alone(tmp_path):
    world = [east(x) for x in np.linspace(-800.0, 800.0, 40)]
    ctx, calls = _run(tmp_path, world, world=world)
    assert calls == []
    assert ctx.result["c2f_extension"]["status"] == "not_triggered"
    assert ctx.osm_around == (*CENTRE, R + 50)


def test_nothing_happens_when_pass_two_was_not_kept(tmp_path):
    ctx, calls = _run(tmp_path, _snapped(40), kept="pass-1 (pass-2 sim worse)")
    assert calls == [] and ctx.result["c2f_extension"]["status"] == "not_applicable"
