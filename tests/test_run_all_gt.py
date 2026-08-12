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


# ---------------------------------------------------------------------------
# the overlay fleet: read from build_overlay_fleet.py's index, never by hand
# ---------------------------------------------------------------------------


def _write_index(tmp_path: Path, clips: list[dict]) -> Path:
    idx = tmp_path / "overlay_fleet.json"
    idx.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return idx


def _clip(video: str, **over) -> dict:
    base = {
        "video_id": "vCSEG6KaFng", "name": "Chicago (vCSEG6KaFng)",
        "slug": "vcseg6kafng-chicago", "title": "Chicago", "city": "Chicago, Illinois, USA",
        "video": video, "ground_truth": "ground_truth/overlay_vCSEG6KaFng.json",
        "osm_around": "41.889,-87.629,1400", "vo_segment": "0:600",
    }
    base.update(over)
    return base


def test_load_overlay_clips_missing_index_is_not_an_error(tmp_path: Path) -> None:
    assert M.load_overlay_clips(tmp_path / "nope.json") == []


def test_load_overlay_clips_builds_the_sweep_args(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input_0-600.mp4").write_bytes(b"")
    idx = _write_index(tmp_path, [_clip("data/input_0-600.mp4")])

    clips = M.load_overlay_clips(idx, root=tmp_path)
    assert len(clips) == 1
    name, slug, args = clips[0]
    assert name == "Chicago (vCSEG6KaFng)"
    assert slug == "vcseg6kafng-chicago"
    assert args[args.index("--ground-truth-waypoints") + 1] == \
        "ground_truth/overlay_vCSEG6KaFng.json"
    assert args[args.index("--osm-around") + 1] == "41.889,-87.629,1400"
    assert args[args.index("--vo-segment") + 1] == "0:600"


def test_load_overlay_clips_skips_a_clip_whose_video_is_gone(tmp_path: Path) -> None:
    # The GT JSON is committed but the multi-hundred-MB download is not, so a
    # fresh checkout must skip the clip rather than launch a run that cannot
    # possibly work.
    idx = _write_index(tmp_path, [_clip("data/never_downloaded.mp4")])
    assert M.load_overlay_clips(idx, root=tmp_path) == []


def test_index_merge_keeps_clips_that_were_not_rebuilt(tmp_path: Path) -> None:
    """`--only <clip>` must not drop the rest of the fleet from the index.

    The builder writes the index from what IT built, so a single-clip run
    would leave an index of one — and the sweep would then evaluate one clip
    and report it as the whole fleet, which looks like a result rather than
    like a mistake.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_overlay_fleet_under_test", ROOT / "scripts" / "build_overlay_fleet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    idx = tmp_path / "overlay_fleet.json"
    idx.write_text(json.dumps({"clips": [
        {"video_id": "keepme", "name": "Old (keepme)", "slug": "s", "title": "t",
         "city": "C", "video": "data/a.mp4", "ground_truth": "ground_truth/a.json",
         "osm_around": "1,2,3", "vo_segment": "0:600", "n_dropped_to_cuts": 0,
         "n_fixes": 10, "span_km": 1.0, "note": ""}]}), encoding="utf-8")

    clip = mod.BY_ID["vCSEG6KaFng"]
    mod._write_index([{
        "clip": clip, "slug": "new-slug", "title": "T", "city": "Chicago, IL, USA",
        "gt_path": "ground_truth/overlay_vCSEG6KaFng.json",
        "video_path": "data/new.mp4", "osm_around": "4,5,6",
        "vo_segment": "5:595", "n_fixes": 119, "span_km": 1.46, "n_dropped": 0,
    }], idx)

    ids = [c["video_id"] for c in json.loads(idx.read_text(encoding="utf-8"))["clips"]]
    assert "keepme" in ids, "a clip that was not rebuilt was dropped from the index"
    assert "vCSEG6KaFng" in ids


def test_overlay_clips_keep_the_gt_disc_strippable() -> None:
    # --blind must be able to drop the overlay fleet's discs too: they are
    # derived from the OCR'd track, which is ground truth.
    args = ["--video", "x.mp4", "--osm-around", "41.889,-87.629,1400",
            "--vo-segment", "0:600", "--scale-lock"]
    assert "--osm-around" not in M._strip_osm_around(args)
