"""Tests for the GT sweep's freshness/reporting logic (scripts/run_all_gt.py)."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_all_gt_under_test", ROOT / "scripts" / "run_all_gt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


# ---------------------------------------------------------------------------
# --blind: the GT-leaking --osm-around discs must be droppable
# ---------------------------------------------------------------------------


def test_strip_osm_around_removes_flag_and_value() -> None:
    args = ["--video", "x.mp4", "--osm-around", "49.0,8.4,700",
            "--vo-segment", "0:47", "--no-splat"]
    assert M._strip_osm_around(args) == [
        "--video", "x.mp4", "--vo-segment", "0:47", "--no-splat"]


def test_strip_osm_around_noop_without_flag() -> None:
    args = ["--video", "x.mp4", "--city", "Ulm, Germany"]
    assert M._strip_osm_around(args) == args


def test_blind_mode_keeps_mega_city_disc() -> None:
    # London's point+radius fetch is an infra necessity, not a GT leak.
    assert "London (Bloomsbury)" in M.MEGA_CITY_CLIPS


# ---------------------------------------------------------------------------
# freshness: a failed run must never report the previous run's result.json
# ---------------------------------------------------------------------------


def test_load_fresh_result_rejects_nonzero_rc(tmp_path: Path) -> None:
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"position": {"gt_mean_route_error_m": 143.0}}))
    assert M._load_fresh_result(res, rc=1, run_start=0.0) is None


def test_load_fresh_result_rejects_stale_file(tmp_path: Path) -> None:
    # File written BEFORE the run started == leftover from an earlier run.
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"position": {}}))
    old = time.time() - 3600
    os.utime(res, (old, old))
    assert M._load_fresh_result(res, rc=0, run_start=time.time() - 60) is None


def test_load_fresh_result_accepts_fresh_success(tmp_path: Path) -> None:
    run_start = time.time() - 5
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"position": {"gt_mean_route_error_m": 143.0}}))
    out = M._load_fresh_result(res, rc=0, run_start=run_start)
    assert out is not None
    assert out["position"]["gt_mean_route_error_m"] == 143.0


def test_load_fresh_result_missing_file(tmp_path: Path) -> None:
    assert M._load_fresh_result(tmp_path / "result.json", 0, 0.0) is None


def test_stash_previous_result_moves_file_aside(tmp_path: Path) -> None:
    res = tmp_path / "result.json"
    res.write_text("{}")
    M._stash_previous_result(res)
    assert not res.exists()
    assert (tmp_path / "result.prev.json").exists()
    M._stash_previous_result(res)  # no file: must be a no-op


# ---------------------------------------------------------------------------
# row extraction: headline (position) AND matcher pick both visible
# ---------------------------------------------------------------------------


def test_result_row_reports_headline_and_matcher() -> None:
    result = {
        "position": {
            "source": "anchor_primary_vpr",
            "gt_mean_route_error_m": 236.0,
            "gt_start_error_m": 160.0,
            "street_names": ["Olgastraße"],
            "spatial_confidence": {"level": "medium", "spread_m": 250.0},
            "hypotheses": [{}, {}],
        },
        "matcher_position": {
            "gt_mean_route_error_m": 491.0,
            "gt_start_error_m": 520.0,
        },
    }
    row = M._result_row("Ulm", 0, result)
    assert row["source"] == "anchor_primary_vpr"   # headline is the anchored answer
    assert row["gt_mean"] == 236.0                 # headline error in the main column
    assert row["m_gt_mean"] == 491.0               # matcher pick visible alongside
    assert row["m_gt_start"] == 520.0


def test_result_row_falls_back_on_old_schema() -> None:
    # Pre-contract result.json: no matcher_position, no source.
    result = {"position": {"gt_mean_route_error_m": 143.0,
                           "gt_start_error_m": 95.0}}
    row = M._result_row("KITTI", 0, result)
    assert row["source"] == "matcher"
    assert row["gt_mean"] == 143.0
    assert row["m_gt_mean"] == 143.0


def test_result_row_none_result_keeps_rc() -> None:
    row = M._result_row("clip", 2, None)
    assert row == {"name": "clip", "rc": 2}
