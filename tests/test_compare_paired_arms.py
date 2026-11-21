import json

import pytest

from scripts import compare_paired_arms as paired


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _arm(root, arm, error, *, clip="test_clip", completed=True, legacy=False, rate=None):
    run = root / f"{arm}_test5"
    method = {"symmetric_mean_m": {"median": round(error, 1) if error != "inf" else error}}
    if rate is not None:
        method["success_rate"] = {"symmetric_mean_m": {"200": rate}}
    summary = {"clips": {clip: {"name": clip, "n_reps": 1, "n_completed": int(completed),
                                "methods": {"answer": method}}}}
    if not legacy:
        summary["reps"] = [{"clip": clip, "rep": 0, "completed": completed,
                            "scores": {"answer": {"symmetric_mean_m": error}}}]
    _write(run / "summary.json", summary)
    if completed:
        _write(run / clip / "rep0" / "status.json",
               {"rc": int(error == "inf"), "result_found": error != "inf", "wall_s": 3})
        if error != "inf":
            _write(run / clip / "rep0" / "result.json", {})


def _compare(root):
    return paired.compare(root, "ordinary", "dense", ["test5"])


def test_comparison_uses_unrounded_repeat_scores_for_thresholds(tmp_path):
    _arm(tmp_path, "ordinary", 200.04)
    _arm(tmp_path, "dense", 199.99)

    out = _compare(tmp_path)

    assert out["clips"][0]["ordinary_m"] == out["clips"][0]["new_m"] == 200.0
    assert out["summary"]["gains"] == ["test_clip"]
    assert out["summary"]["within_200_ordinary"] == 0
    assert out["summary"]["within_200_new"] == 1


def test_exact_modern_median_at_threshold_counts_as_within(tmp_path):
    _arm(tmp_path, "ordinary", 199.0)
    _arm(tmp_path, "dense", 201.0)
    path = tmp_path / "ordinary_test5" / "summary.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["clips"]["test_clip"].update(n_reps=2, n_completed=2)
    snapshot["reps"][0]["scores"]["answer"]["success"] = {"symmetric_mean_m": {"200": True}}
    snapshot["reps"].append({"clip": "test_clip", "rep": 1, "completed": True, "scores": {
        "answer": {"symmetric_mean_m": 201.0, "success": {"symmetric_mean_m": {"200": False}}},
    }})
    _write(path, snapshot)
    assert _compare(tmp_path)["summary"]["n"] == 0

    snapshot["repeat_score_precision"] = "unrounded"
    _write(path, snapshot)

    assert _compare(tmp_path)["summary"]["losses"] == ["test_clip"]


def test_archive_uses_success_rate_when_only_the_rounded_boundary_is_saved(tmp_path):
    _arm(tmp_path, "ordinary", 200.0, legacy=True, rate=0.0)
    _arm(tmp_path, "dense", 195.0, legacy=True)

    assert _compare(tmp_path)["summary"]["gains"] == ["test_clip"]


def test_comparison_reports_incomplete_and_ambiguous_pairs(tmp_path):
    _arm(tmp_path, "ordinary", 199.0, completed=False)
    _arm(tmp_path, "dense", 190.0)
    out = _compare(tmp_path)
    assert out["summary"]["n"] == 0
    assert out["unpaired"][0]["clip"] == "test_clip"

    _arm(tmp_path, "ordinary", 200.0, legacy=True)
    out = _compare(tmp_path)
    assert out["summary"]["n"] == 0
    assert "ambiguous" in out["unpaired"][0]["reason"]


def test_partial_repeat_log_cannot_decide_the_pair(tmp_path):
    _arm(tmp_path, "ordinary", 180.0)
    _arm(tmp_path, "dense", 190.0)
    path = tmp_path / "ordinary_test5" / "summary.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["clips"]["test_clip"].update(n_reps=2, n_completed=2)
    _write(path, snapshot)

    assert _compare(tmp_path)["summary"]["n"] == 0


def test_completed_failure_without_result_still_counts_as_a_scored_miss(tmp_path):
    _arm(tmp_path, "ordinary", "inf")
    _arm(tmp_path, "dense", 190.0)

    out = _compare(tmp_path)

    assert out["summary"]["n"] == 1
    assert out["summary"]["gains"] == ["test_clip"]
    assert out["clips"][0]["ordinary_m"] == "inf"
    assert out["clips"][0]["pass1_to_route_m"] == [None, None]
    assert out["clips"][0]["pass1_refs"] == [None, None]


@pytest.mark.parametrize("error", [0.0, "inf"])
def test_equal_answers_do_not_vote_in_the_sign_test(tmp_path, error):
    _arm(tmp_path, "ordinary", error)
    _arm(tmp_path, "dense", error)

    result = _compare(tmp_path)["summary"]

    assert result["tie"] == 1
    assert result["better"] == result["worse"] == 0
    assert result["sign_test_one_sided_p"] is None
