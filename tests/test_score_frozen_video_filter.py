import json

import pytest

from scripts.probe_strict_video_filter import sha256
from scripts.score_frozen_video_filter import aggregate_scores, verify_complete_batch


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _batch(tmp_path, changed_config=False):
    base, batch = tmp_path / "base", tmp_path / "batch"
    base.mkdir()
    batch.mkdir()
    (base / "prediction.json").write_text("prediction, no labels")
    _json(base / "frozen_prediction.json", {"artifacts": {"prediction.json": sha256(base / "prediction.json")}})
    base_hash = sha256(base / "frozen_prediction.json")
    fixed = {"seed": 0, "n_particles": 600_000, "path_budget_mb": 256}
    _json(batch / "batch_config.json", {"base": str(base), "seeds_predeclared": [1], "fixed_filter_config": fixed})
    seed = batch / "seed_1"
    seed.mkdir()
    _json(seed / "config.json", {"filter_config": {**fixed, "seed": 1, **({"n_particles": 20} if changed_config else {})}})
    (seed / "prediction.json").write_text("seed one prediction")
    _json(seed / "frozen_prediction.json", {"seed": 1, "base_frozen_manifest_sha256": base_hash,
          "artifacts": {name: sha256(seed / name) for name in ("config.json", "prediction.json")}})
    _json(batch / "frozen_predictions.json", {"all_predictions_frozen_before_scoring": True,
          "seeds_predeclared": [1], "base_frozen_manifest_sha256": base_hash,
          "artifacts": {"batch_config.json": sha256(batch / "batch_config.json")},
          "seeds": {"1": {"directory": "seed_1", "manifest_sha256": sha256(seed / "frozen_prediction.json")}},
          "replay_source_sha256": {}})
    return batch


def test_score_requires_complete_frozen_seed_set(tmp_path):
    batch = _batch(tmp_path)
    _, _, runs = verify_complete_batch(batch)
    assert [seed for seed, _ in runs] == [0, 1]
    path = batch / "frozen_predictions.json"
    manifest = json.loads(path.read_text())
    manifest["seeds_predeclared"] = [1, 2]
    _json(path, manifest)
    with pytest.raises(ValueError, match="complete predeclared seed set"):
        verify_complete_batch(batch)


def test_scoring_rejects_replay_that_changed_more_than_seed(tmp_path):
    batch = _batch(tmp_path, changed_config=True)
    with pytest.raises(ValueError, match="changed more than the seed"):
        verify_complete_batch(batch)


def test_failures_remain_in_whole_run_aggregates():
    def row(seed, error, latency):
        metrics = {key: error for key in ("mean_err_m", "median_err_m", "final_err_m", "smoothed_mean_err_m",
                                          "frac_steps_lt10m", "frac_steps_lt25m")}
        metrics["time_to_localize_s"] = latency
        route = {key: error for key in ("gt_to_pred_mean_m", "symmetric_mean_m", "start_error_m", "end_error_m", "hausdorff_m")}
        return {"seed": seed, "per_frame_metrics": metrics, "smoothed_waypoint_route_metrics": route}
    aggregate = aggregate_scores([row(0, 5, 78.5), row(1, 9000, None)])
    assert aggregate["n_seeds"] == 2
    assert aggregate["localized_seeds"] == 1
    assert aggregate["seeds_included"] == [0, 1]
    assert aggregate["per_frame_metrics"]["mean_err_m"]["mean"] == 4502.5
    assert aggregate["time_to_localize_s_among_localized"]["mean"] == 78.5
