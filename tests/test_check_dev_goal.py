"""The development target must never pass by dropping failures or clips."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_dev_goal_tested", ROOT / "scripts/check_dev_goal.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def _valid(count: int = 15, repeats: int = 1):
    manifest = {"clips": []}
    summary = {
        "clips": {}, "reps": [],
        "provenance": {
            "consistent": True, "problems": [], "leaks": {}, "dirty": ["False"],
            "commits": ["commit-A"], "package_trees": ["pipeline-tree"],
            "method_config_fingerprints": ["method-config"],
            "environment_fingerprints": ["libraries"], "vpr_model_fingerprints": ["megaloc"],
        },
        "run_config": {"execution_plan": {}, "config": {"pipeline_args": []}},
        "execution": {"plan_recorded": True, "complete": True, "incomplete_reps": [],
                      "expected_reps": count * repeats, "n_completed": count * repeats},
    }
    for i in range(count):
        cid = f"clip{i:02}"
        manifest["clips"].append({"id": cid, "split": "dev", "available_locally": True,
                                  "city": "Registered city", "vo_segment": "0:300",
                                  "gt_waypoints_sha256": f"gt-{i}"})
        summary["clips"][cid] = {
            "split": "dev", "city": "Registered city", "vo_segments": ["0:300"],
            "gt_waypoints_sha256": f"gt-{i}", "n_reps": repeats,
            "n_ok": repeats, "n_completed": repeats,
            "methods": {"answer": {"n_reps": repeats, "n_answered": repeats,
                                    "symmetric_mean_m": {"median": 100.0}}},
        }
        summary["run_config"]["execution_plan"][cid] = {"reps": list(range(repeats))}
        for k in range(repeats):
            summary["reps"].append({
                "clip": cid, "rep": k, "rc": 0, "ok": True, "completed": True,
                "result_found": True, "commit": "commit-A",
                "scores": {"answer": {
                    "symmetric_mean_m": 100.0, "answered": True, "failed": False,
                    "oracle": False, "source": "result.answer",
                    "meta": {"mode": "track", "method": "vpr_track_only"},
                }},
            })
    # Unavailable dev clips and held-out clips must not enter the dev target.
    manifest["clips"] += [
        {"id": "unavailable", "split": "dev", "available_locally": False},
        {"id": "held-out", "split": "test", "available_locally": True},
    ]
    return summary, manifest


def test_complete_registered_development_target_passes():
    summary, manifest = _valid()
    report = M.check_goal(summary, manifest)
    assert report["goal_met"] and report["n_passing"] == 15
    assert not report["reasons"] and not report["missing_ids"]


def test_missing_clip_cannot_change_the_denominator():
    summary, manifest = _valid()
    summary["clips"].pop("clip14")
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["missing_ids"] == ["clip14"]


@pytest.mark.parametrize("error", ["inf", float("inf"), float("nan"), None])
def test_nonfinite_or_missing_predictions_are_failures(error):
    summary, manifest = _valid()
    summary["clips"]["clip00"]["methods"]["answer"]["symmetric_mean_m"]["median"] = error
    summary["reps"][0]["scores"]["answer"]["symmetric_mean_m"] = error
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["n_passing"] == 14
    json.dumps(report, allow_nan=False)           # CLI output remains valid JSON


def test_failed_repeat_cannot_hide_behind_a_good_median():
    summary, manifest = _valid(repeats=3)
    summary["reps"][0].update(rc=1, ok=False, result_found=False)
    summary["reps"][0]["scores"]["answer"].update(answered=False, failed=True,
                                                       symmetric_mean_m="inf")
    report = M.check_goal(summary, manifest, min_repeats=3)
    assert not report["goal_met"]
    assert any("failed, unanswered" in r for r in report["clips"][0]["reasons"])


def test_repeats_are_configurable_and_threshold_uses_actual_scores():
    summary, manifest = _valid(repeats=2)
    assert M.check_goal(summary, manifest, min_repeats=2)["goal_met"]
    assert not M.check_goal(summary, manifest, min_repeats=3)["goal_met"]
    assert not M.check_goal(summary, manifest, threshold_m=99.9)["goal_met"]
    assert M.check_goal(summary, manifest, threshold_m=100)["goal_met"]
    # Editing only a summary median cannot satisfy the target.
    summary["clips"]["clip00"]["methods"]["answer"]["symmetric_mean_m"]["median"] = 1.0
    assert not M.check_goal(summary, manifest)["goal_met"]


@pytest.mark.parametrize("change", ["plan", "complete", "provenance", "dirty", "code",
                                    "libraries", "config", "commit", "window", "gt", "mode"])
def test_incomplete_or_inconsistent_results_cannot_pass(change):
    summary, manifest = _valid()
    if change == "plan":
        summary["run_config"]["execution_plan"].pop("clip00")
    elif change == "complete":
        summary["execution"]["complete"] = False
    elif change == "provenance":
        summary["provenance"]["consistent"] = False
    elif change == "dirty":
        summary["provenance"]["dirty"] = ["True"]
    elif change == "code":
        summary["provenance"]["package_trees"].append("other-code")
    elif change == "libraries":
        summary["provenance"]["environment_fingerprints"].append("other-libraries")
    elif change == "config":
        summary["provenance"]["method_config_fingerprints"].append("other-settings")
    elif change == "commit":
        summary["reps"][0]["commit"] = "different-commit"
    elif change == "window":
        summary["clips"]["clip00"]["vo_segments"] = ["0:30"]
    elif change == "gt":
        summary["clips"]["clip00"]["gt_waypoints_sha256"] = "changed-coordinates"
    else:
        summary["reps"][0]["scores"]["answer"]["meta"]["mode"] = "fit"
    assert not M.check_goal(summary, manifest)["goal_met"]


@pytest.mark.parametrize("leak", ["oracle", "audit", "argv", "cfg", "source", "chosen"])
def test_known_oracle_or_ground_truth_leakage_is_rejected(leak):
    summary, manifest = _valid()
    if leak == "oracle":
        summary["reps"][0]["scores"]["answer"]["oracle"] = True
    elif leak == "audit":
        summary["provenance"]["leaks"] = {"clip00": ["ground truth reached the pipeline"]}
    elif leak == "argv":
        summary["run_config"]["config"]["pipeline_args"] = ["--osm", "48.4,9.9,800"]
    elif leak == "source":
        summary["reps"][0]["scores"]["answer"]["source"] = "ground_truth.waypoints"
    elif leak == "chosen":
        summary["reps"][0]["scores"]["answer"]["meta"]["method"] = "shortlist_oracle@5"
    else:
        summary["reps"][0]["provenance"] = {"config": {"ground_truth_waypoints": "gps.json"}}
    assert not M.check_goal(summary, manifest)["goal_met"]


def test_selected_method_must_be_current_and_deployable():
    summary, manifest = _valid()
    summary["reps"][0]["scores"]["answer"]["source"] = "result.position"
    assert not M.check_goal(summary, manifest)["goal_met"]
    assert not M.check_goal(summary, manifest, method="shortlist_oracle@5")["goal_met"]


def test_provisional_in_progress_snapshot_cannot_meet_the_goal():
    summary, manifest = _valid()
    summary["review"] = {"status": "in_progress_snapshot"}
    assert not M.check_goal(summary, manifest)["goal_met"]


def test_unexpected_clip_and_duplicate_repeat_are_rejected():
    summary, manifest = _valid()
    summary["clips"]["held-out"] = copy.deepcopy(summary["clips"]["clip00"])
    summary["reps"].append(copy.deepcopy(summary["reps"][0]))
    assert not M.check_goal(summary, manifest)["goal_met"]


def test_nondefault_expected_count_requires_all_manifest_clips():
    summary, manifest = _valid(count=3)
    assert not M.check_goal(summary, manifest)["goal_met"]
    assert M.check_goal(summary, manifest, expected_count=3)["goal_met"]


def test_cli_emits_json_and_nonzero_until_the_goal_is_met(tmp_path, capsys):
    summary, manifest = _valid()
    sp, mp = tmp_path / "summary.json", tmp_path / "manifest.json"
    sp.write_text(json.dumps(summary), encoding="utf-8")
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    args = [str(sp), "--manifest", str(mp)]
    assert M.main(args) == 0
    assert json.loads(capsys.readouterr().out)["goal_met"] is True
    summary["clips"].pop("clip00")
    sp.write_text(json.dumps(summary), encoding="utf-8")
    assert M.main(args) == 1
    assert json.loads(capsys.readouterr().out)["goal_met"] is False
    sp.write_text("invalid json", encoding="utf-8")
    assert M.main(args) == 1
    assert "cannot audit inputs" in json.loads(capsys.readouterr().out)["reasons"][0]


def test_current_aggregator_complete_answer_summary_can_pass(tmp_path, monkeypatch):
    from tests.test_benchmark import M as benchmark, _fake_result, _setup_root, _write_rep

    root, manifest = _setup_root(tmp_path)
    run_dir = tmp_path / "runs" / "complete"
    plan = {}
    monkeypatch.setattr(benchmark, "package_tree", lambda commit, root=None: "known-tree")
    for entry, shift in zip(manifest["clips"], (0.0, 5000.0, 10000.0)):
        cid = entry["id"]
        result = _fake_result(shift, 20.0)
        result["provenance"]["config"]["answer"] = "track"
        result["answer"] = {"mode": "track", "method": "vpr_track_only",
                            "route_latlon": result["position"]["route_latlon"]}
        _write_rep(run_dir, cid, 0, result)
        status_path = run_dir / cid / "rep0/status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["vo_segment"] = entry["vo_segment"]
        status_path.write_text(json.dumps(status), encoding="utf-8")
        plan[cid] = {"city": entry["city"], "vo_segment": entry["vo_segment"], "reps": [0]}
    (run_dir / "run_config.json").write_text(
        json.dumps({"execution_plan": plan, "config": {"pipeline_args": []}}), encoding="utf-8")
    summary = benchmark.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    report = M.check_goal(summary, manifest, expected_count=3)
    assert report["goal_met"], report
