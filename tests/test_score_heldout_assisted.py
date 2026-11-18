"""scripts/score_heldout_assisted.py on a synthetic one-clip run."""
import json

import pytest

pytest.importorskip("statsmodels")
from scripts import score_heldout_assisted as S  # noqa: E402

gt_waypoints_digest = S.gt_waypoints_digest

CITY = "Trondheim, Norway"  # in benchmarks/geocode_snapshot.json
COMMIT = "409a9595f22b8d726b71a3ab6f848559773fc9a3"


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
    provenance = {"git": {"commit": COMMIT, "dirty": False}}
    _write(out / "result.json", {
        "assisted_position": answer, "answer": answer, "provenance": provenance,
        "assisted_audit": {"decision_trace": [{"selected_candidate": "baseline",
                                               "selection_reason": "no_admissible_assistance"}]}})
    child = out / "children" / "c" / "baseline"
    _write(child / "result.json", {"answer": {"route_latlon": _route(baseline_lat)},
                                   "provenance": provenance})
    log = "" if fine_lat is None else (
        f"re-fetching a 2000 m disc around the pass-1 centre (snapped {fine_lat}, 10.40)")
    (child / "worker.log").write_text(log, encoding="utf8")
    _write(out / "assisted_runtime.json", {"started_monotonic_s": 0.0, "events": [
        {"kind": "baseline", "status": "process_complete", "monotonic_s": 5.0}]})
    return entry, tmp_path / "run"


def test_scores_answer_baseline_and_centre_with_mode(tmp_path):
    entry, run = _clip(tmp_path, answer_lat=63.43, baseline_lat=63.43, fine_lat=63.4345)
    rec = S.score_clip(entry, run, commit=COMMIT)
    assert rec["assisted_m"] == pytest.approx(0.0, abs=1e-6)
    assert rec["baseline_track_m"] == rec["assisted_m"]
    assert rec["assisted_within"] and rec["baseline_within"] and rec["beats_centre"]
    assert rec["mode"] == "ok" and rec["fine_share"] == 1.0
    assert rec["stages"] == {"baseline": {"status": "process_complete", "end_s": 5.0}}


def test_far_answer_is_a_miss_and_pass2_elsewhere_is_mode_b(tmp_path):
    entry, run = _clip(tmp_path, answer_lat=63.46, baseline_lat=63.43, fine_lat=63.50)
    rec = S.score_clip(entry, run, commit=COMMIT)
    assert rec["assisted_m"] > 2000 and not rec["assisted_within"] and rec["baseline_within"]
    assert rec["mode"] == "B"


def test_refuses_a_ground_truth_that_differs_from_the_manifest(tmp_path):
    entry, run = _clip(tmp_path, answer_lat=63.43, baseline_lat=63.43)
    entry["gt_waypoints_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="ground truth differs"):
        S.score_clip(entry, run, commit=COMMIT)


def test_uses_the_registered_manifest_after_a_new_split_is_committed():
    rule = S._json(S.ROOT / "benchmarks" / "preregistered_rules.json")["rules"]["test4_2026_11"]

    manifest = S._frozen_manifest(rule["dataset_freeze"], None)

    assert {c["id"] for c in manifest["clips"] if c["split"] == "test4"} == {
        f"overlay_{vid}" for vid in rule["clips"]
    }
    assert not any(c["split"] == "test5" for c in manifest["clips"])


def test_refuses_an_explicit_manifest_that_differs_from_the_registered_one(tmp_path):
    rule = S._json(S.ROOT / "benchmarks" / "preregistered_rules.json")["rules"]["test4_2026_11"]
    manifest = tmp_path / "manifest.json"
    _write(manifest, {"clips": []})

    with pytest.raises(SystemExit, match="is not the frozen one"):
        S._frozen_manifest(rule["dataset_freeze"], manifest)


@pytest.mark.parametrize(("source", "git"), [
    ("parent", {"commit": "79517c12dc9e410edeec53d092bd3d929bfa817c", "dirty": False}),
    ("parent", {}),
    ("child", {"commit": COMMIT, "dirty": True}),
    ("child", {"commit": "79517c12dc9e410edeec53d092bd3d929bfa817c", "dirty": False}),
])
def test_refuses_results_without_the_registered_clean_commit(tmp_path, source, git):
    entry, run = _clip(tmp_path, answer_lat=63.43, baseline_lat=63.43)
    out = run / "overlay_X" / "rep0" / "pipeline_out" / "slug"
    result_path = (out if source == "parent" else out / "children" / "c" / "baseline") / "result.json"
    result = S._json(result_path)
    result["provenance"]["git"] = git
    _write(result_path, result)

    with pytest.raises(SystemExit, match="expected clean result from " + COMMIT):
        S.score_clip(entry, run, commit=COMMIT)


def test_wilson_matches_the_recorded_intervals():
    assert [round(100 * x) for x in S.wilson(1, 6)] == [3, 56]
    assert [round(100 * x) for x in S.wilson(14, 15)] == [70, 99]
