"""Fresh pilot guards: no real models, network, labels or reference acquisitions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pilot_live_poi_alignment.py"
spec = importlib.util.spec_from_file_location("live_poi_pilot_test", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def _evidence(identifier="mapillary:a"):
    times = np.arange(80, dtype=float)
    return {"query_indices": np.arange(80), "query_times": times, "ref_latlon": np.array([[40., 8.]]),
            "similarities": np.zeros((80, 1), np.float32), "ref_ids": np.array([identifier]),
            "provenance": {"backbone": {"name": "model", "hub": "pinned"},
                           "preprocessing": {"version": 1}, "query_mode": "warp"}}


def test_memory_preflight_refuses_before_model_or_reference_work(monkeypatch):
    monkeypatch.setattr(pilot, "memory_status", lambda: {"available_commit_bytes": 7 * 1024**3})
    with pytest.raises(RuntimeError, match="8 GiB"):
        pilot.require_memory()


def test_regional_pool_deduplicates_ids_and_retains_every_distinct_reference():
    pool, first = {}, _evidence()
    pilot.merge_evidence(pool, first, np.arange(80))
    pilot.merge_evidence(pool, first, np.arange(80))
    pilot.merge_evidence(pool, _evidence("mapillary:b"), np.arange(80))
    assert list(pool["columns"]) == ["mapillary:a", "mapillary:b"]


@pytest.mark.parametrize("change", ["mode", "times", "coordinates", "scores"])
def test_conflicting_region_is_rejected_before_union_mutation(change):
    pool, first, second = {}, _evidence(), _evidence()
    pilot.merge_evidence(pool, first, np.arange(80))
    if change == "mode":
        second["provenance"]["query_mode"] = "crops"
    elif change == "times":
        second["query_times"][3] += .1
    elif change == "coordinates":
        second["ref_latlon"][0, 0] += .1
    else:
        second["similarities"][0, 0] += .1
    with pytest.raises(ValueError):
        pilot.merge_evidence(pool, second, np.arange(80))
    assert len(pool["columns"]) == 1
    np.testing.assert_array_equal(pool["columns"]["mapillary:a"][1], np.zeros(80))


def test_request_diagnostics_never_persist_token_values(monkeypatch):
    monkeypatch.setenv("MLY_TOKEN", "synthetic-secret-only")
    sanitized = pilot.clean_strings({"error": "URL?access_token=synthetic-secret-only"})
    assert "synthetic-secret-only" not in str(sanitized)
    assert "[REDACTED]" in sanitized["error"]


def _freeze(out):
    pilot.write(out / "experiment.json", {"settings": {"appearance_variants": ["raw", "reference_mean"]}})
    pilot.write(out / "plans.json", [{"hypothesis_id": "hypothesis_0"}])
    pilot.write(out / "regions.json", [])
    for variant in ("raw", "reference_mean"):
        pilot.write(out / f"{variant}.json", {"status": "inactive"})
    pilot.write(out / "predictions_frozen.json", {
        "experiment_sha256": pilot.sha256(out / "experiment.json"),
        "predictions": {v: pilot.sha256(out / f"{v}.json") for v in ("raw", "reference_mean")},
        "artifacts": {name: pilot.sha256(out / name) for name in ("plans.json", "regions.json")}})


@pytest.mark.parametrize("target", ["raw.json", "reference_mean.json", "plans.json", "regions.json"])
def test_score_validates_both_predictions_and_input_plans_before_labels(tmp_path, monkeypatch, target):
    _freeze(tmp_path)
    pilot.write(tmp_path / target, {"changed": True})
    original = pilot.read
    def guarded(path):
        if Path(path).name == "labels.json":
            raise AssertionError("labels opened before immutable inputs were validated")
        return original(path)
    monkeypatch.setattr(pilot, "read", guarded)
    with pytest.raises(ValueError, match="changed"):
        pilot.score(SimpleNamespace(out=tmp_path, gt=tmp_path / "labels.json"))


def test_missing_variant_cannot_be_scored(tmp_path):
    _freeze(tmp_path)
    frozen = pilot.read(tmp_path / "predictions_frozen.json")
    frozen["predictions"].pop("reference_mean")
    pilot.write(tmp_path / "predictions_frozen.json", frozen)
    with pytest.raises(ValueError, match="both registered predictions"):
        pilot.load_frozen(tmp_path)


def test_ram_guard_uses_peak_resident_bytes_and_retains_every_failed_region(tmp_path, monkeypatch):
    out, data = tmp_path / "out", tmp_path / "data"
    out.mkdir()
    memory = {"available_commit_bytes": 20 * 1024**3, "private_bytes": 4 * 1024**3,
              "peak_private_bytes": 5 * 1024**3, "working_set_bytes": 900_000_000,
              "peak_working_set_bytes": 1_000_000_000}
    monkeypatch.setattr(pilot, "memory_status", lambda: memory)
    fake = SimpleNamespace(kartaview_vpr_evidence=lambda *args, **kwargs: None, LAST_FAILURE="no references")
    plans = [{"hypothesis_id": f"hypothesis_{i}", "center": [40., 8.], "radius_m": 1000., "cap": 500}
             for i in range(5)]
    pool, records = pilot.acquire_regions(SimpleNamespace(out=out, data=data), fake, [None] * 80, np.arange(80), plans)
    assert pool == {} and len(records) == 5
    assert all(r["status"] == "failed" for r in records)
