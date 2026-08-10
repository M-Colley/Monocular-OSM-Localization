"""Tests for the auto-extracted-GT auditor (scripts/check_overlay_gt.py).

The overlay fleet's ground truth is OCR'd, not labelled, so this auditor is
the thing standing between a misread digit and a benchmark number measured
against a lie. Its false-negative behaviour matters as much as its
detections: it must not flag a normal city drive.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_overlay_gt_under_test", ROOT / "scripts" / "check_overlay_gt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


def _write(tmp_path: Path, waypoints: list[dict], **extra) -> Path:
    p = tmp_path / "overlay_test.json"
    p.write_text(json.dumps({"city": "Chicago, Illinois, USA",
                             "video_id": "test", "waypoints": waypoints, **extra}),
                 encoding="utf-8")
    return p


def _straight_drive(n: int, step_deg: float, dt: float) -> list[dict]:
    """`n` waypoints heading due north, `step_deg` of latitude every `dt` s.

    1e-3 degrees of latitude is ~111 m, so step_deg=2.5e-3 over dt=30 is
    ~33 km/h — ordinary city traffic.
    """
    return [{"t_sec": i * dt, "lat": 41.88 + i * step_deg, "lon": -87.63}
            for i in range(n)]


# ---------------------------------------------------------------------------
# it must PASS a normal drive
# ---------------------------------------------------------------------------


def test_accepts_a_plausible_city_drive(tmp_path: Path) -> None:
    # ~28 m/s of latitude change per 30 s leg -> ~33 km/h, ordinary traffic.
    wps = _straight_drive(n=12, step_deg=2.5e-3, dt=30.0)
    ok, problems, stats = M.check(_write(tmp_path, wps))
    assert ok, problems
    assert 20.0 < stats["mean_kmh"] < 50.0
    assert stats["waypoints"] == 12


def test_accepts_a_stop_at_a_light(tmp_path: Path) -> None:
    # Three identical positions in the middle: stopped, not broken.
    wps = _straight_drive(n=6, step_deg=2.5e-3, dt=30.0)
    for i in (2, 3):
        wps[i] = {**wps[i], "lat": wps[1]["lat"], "lon": wps[1]["lon"]}
    ok, problems, _ = M.check(_write(tmp_path, wps))
    assert ok, problems


# ---------------------------------------------------------------------------
# it must CATCH the failure modes OCR actually produces
# ---------------------------------------------------------------------------


def test_flags_a_teleport(tmp_path: Path) -> None:
    # One digit misread in the latitude: 41.9 -> 42.9, ~111 km away.
    wps = _straight_drive(n=6, step_deg=2.5e-3, dt=30.0)
    wps[3] = {**wps[3], "lat": wps[3]["lat"] + 1.0}
    ok, problems, _ = M.check(_write(tmp_path, wps))
    assert not ok
    assert any("km/h" in p for p in problems)


def test_flags_an_out_and_back_detour(tmp_path: Path) -> None:
    # A misread that survives a speed check because the legs are long enough:
    # the middle waypoint sits 400 m off the line and comes straight back.
    wps = [
        {"t_sec": 0, "lat": 41.8800, "lon": -87.6300},
        {"t_sec": 60, "lat": 41.8840, "lon": -87.6300},   # 445 m north
        {"t_sec": 120, "lat": 41.8801, "lon": -87.6300},  # back to the start
        {"t_sec": 180, "lat": 41.8802, "lon": -87.6300},
    ]
    ok, problems, _ = M.check(_write(tmp_path, wps))
    assert not ok
    assert any("detour" in p for p in problems)


def test_flags_non_increasing_time(tmp_path: Path) -> None:
    wps = _straight_drive(n=5, step_deg=2.5e-3, dt=30.0)
    wps[2] = {**wps[2], "t_sec": wps[1]["t_sec"]}
    ok, problems, _ = M.check(_write(tmp_path, wps))
    assert not ok
    assert any("increasing" in p for p in problems)


def test_flags_a_track_that_never_moves(tmp_path: Path) -> None:
    # Every fix identical: the stamp was not actually read, or the car is parked
    # for the whole window. Either way it is useless as route ground truth.
    wps = [{"t_sec": i * 30.0, "lat": 41.88, "lon": -87.63} for i in range(8)]
    ok, problems, _ = M.check(_write(tmp_path, wps))
    assert not ok
    assert any("barely moves" in p for p in problems)


def test_rejects_too_few_waypoints(tmp_path: Path) -> None:
    ok, problems, _ = M.check(_write(tmp_path, [{"t_sec": 0, "lat": 41.0, "lon": -87.0}]))
    assert not ok
    assert "waypoints" in problems[0]


# ---------------------------------------------------------------------------
# precision reporting — the stamp's own resolution bounds any error number
# ---------------------------------------------------------------------------


def test_reports_the_coordinate_precision_present(tmp_path: Path) -> None:
    # A 1-arcsecond DMS overlay lands on a ~31 m grid and shows up as 4 dp.
    coarse = [{"t_sec": i * 30.0, "lat": round(42.3831 + i * 2.5e-3, 4),
               "lon": -83.1764} for i in range(8)]
    _ok, _p, stats = M.check(_write(tmp_path, coarse))
    assert stats["decimals"] == 4

    fine = [{"t_sec": i * 30.0, "lat": 33.773609 + i * 2.5e-3, "lon": -84.371269}
            for i in range(8)]
    _ok, _p, stats = M.check(_write(tmp_path, fine))
    assert stats["decimals"] == 6
