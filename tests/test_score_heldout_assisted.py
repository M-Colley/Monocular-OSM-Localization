"""scripts/score_heldout_assisted.py on a synthetic one-clip run."""
import json

import pytest

pytest.importorskip("statsmodels")
from benchmarks.build_manifest import gt_waypoints_digest  # noqa: E402
from scripts import score_heldout_assisted as S  # noqa: E402

CITY = "Trondheim, Norway"  # in benchmarks/geocode_snapshot.json


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf8")


def _route(lat0, n=10, step=0.001):
    return [[lat0 + k * step, 10.40] for k in range(n)]


def _clip(tmp_path, answer_lat, baseline_lat, fine_lat=None):
    gt = {"waypoints": [{"t_sec": float(k), "lat": lat, "lon": lon}
                        for k, (lat, lon) in enumerate(_route(63.43))]}
    _write(tmp_path / "gt.json", gt)
    entry = {"id": "overlay_X", "name": "Trondheim (X)", "city": CITY,
             "gt": str(tmp_path / "gt.json"), "gt_waypoints_sha256": gt_waypoints_digest(gt)}
    rep = tmp_path / "run" / "overlay_X" / "rep0"
    _write(rep / "status.json", {"rc": 0, "timed_out": False, "result_found": True,
                                 "city": CITY, "vo_segment": "0:9", "wall_s": 1.0})
    out = rep / "pipeline_out" / "slug"
    answer = {"route_latlon": _route(answer_lat)}
    _write(out / "result.json", {
        "assisted_position": answer, "answer": answer,
        "assisted_audit": {"decision_trace": [{"selected_candidate": "baseline",
                                               "selection_reason": "no_admissible_assistance"}]}})
    child = out / "children" / "c" / "baseline"
    _write(child / "result.json", {"answer": {"route_latlon": _route(baseline_lat)}})
    log = "" if fine_lat is None else (
        f"re-fetching a 2000 m disc around the pass-1 centre (snapped {fine_lat}, 10.40)")
    (child / "worker.log").write_text(log, encoding="utf8")
    _write(out / "assisted_runtime.json", {"started_monotonic_s": 0.0, "events": [
        {"kind": "baseline", "status": "process_complete", "monotonic_s": 5.0}]})
    return entry, tmp_path / "run"


def test_scores_answer_baseline_and_centre_with_mode(tmp_path):
    entry, run = _clip(tmp_path, answer_lat=63.43, baseline_lat=63.43, fine_lat=63.4345)
    rec = S.score_clip(entry, run)
    assert rec["assisted_m"] == pytest.approx(0.0, abs=1e-6)
    assert rec["baseline_track_m"] == rec["assisted_m"]
    assert rec["assisted_within"] and rec["baseline_within"] and rec["beats_centre"]
    assert rec["mode"] == "ok" and rec["fine_share"] == 1.0
    assert rec["stages"] == {"baseline": {"status": "process_complete", "end_s": 5.0}}


def test_far_answer_is_a_miss_and_pass2_elsewhere_is_mode_b(tmp_path):
    entry, run = _clip(tmp_path, answer_lat=63.46, baseline_lat=63.43, fine_lat=63.50)
    rec = S.score_clip(entry, run)
    assert rec["assisted_m"] > 2000 and not rec["assisted_within"] and rec["baseline_within"]
    assert rec["mode"] == "B"


def test_refuses_a_ground_truth_that_differs_from_the_manifest(tmp_path):
    entry, run = _clip(tmp_path, answer_lat=63.43, baseline_lat=63.43)
    entry["gt_waypoints_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="ground truth differs"):
        S.score_clip(entry, run)


def test_wilson_matches_the_recorded_intervals():
    assert [round(100 * x) for x in S.wilson(1, 6)] == [3, 56]
    assert [round(100 * x) for x in S.wilson(14, 15)] == [70, 99]
