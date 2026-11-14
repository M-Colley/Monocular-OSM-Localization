"""Reference planning is based only on input identities, map positions and budgets."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pilot_poi_reference_neighborhoods.py"
spec = importlib.util.spec_from_file_location("poi_neighborhood_pilot_test", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_every_branch_receives_same_cap_and_only_radius_eligible_references():
    ids = [f"ref{i}" for i in range(10)]
    xy = np.c_[np.r_[np.arange(5), np.arange(5) + 100], np.zeros(10)]
    plan = pilot.plan_neighborhoods(ids, xy, [[0, 0], [100, 0]], 10., 3)
    assert [b["selected_count"] for b in plan["branches"]] == [3, 3]
    assert [b["eligible_count"] for b in plan["branches"]] == [5, 5]
    assert plan["selected_count"] == 6
    assert set(plan["branches"][0]["selected_indices"]) <= set(range(5))
    assert set(plan["branches"][1]["selected_indices"]) <= set(range(5, 10))


def test_hash_sampling_is_invariant_to_reference_order_and_deduplicates_shared_pool():
    ids = np.array([f"ref{i}" for i in range(10)])
    xy = np.c_[np.arange(10), np.zeros(10)]
    first = pilot.plan_neighborhoods(ids, xy, [[4, 0], [4, 0]], 20., 4)
    order = np.array([3, 6, 1, 8, 2, 9, 0, 7, 4, 5])
    second = pilot.plan_neighborhoods(ids[order], xy[order], [[4, 0], [4, 0]], 20., 4)
    assert set(first["selected_ids"]) == set(second["selected_ids"])
    assert first["branches"][0]["selected_ids"] == first["branches"][1]["selected_ids"]
    assert first["selected_count"] == 4


@pytest.mark.parametrize("radius,cap", [(np.nan, 1), (-1., 1), (1., 1.5), (1., 0)])
def test_invalid_budgets_are_rejected(radius, cap):
    with pytest.raises(ValueError):
        pilot.plan_neighborhoods(["a"], [[0, 0]], [[0, 0]], radius, cap)


def _freeze(out):
    pilot.write(out / "experiment.json", {"runs": ["tiny__raw"]})
    plan_path = out / "tiny_references.json"
    pilot.write(plan_path, {"selected_ids": ["a"]})
    prediction = {"provenance": {"caller_sources": {"reference_plan_sha256": pilot.sha256(plan_path)}}}
    pilot.write(out / "tiny__raw.json", prediction)
    pilot.write(out / "predictions_frozen.json", {
        "experiment_sha256": pilot.sha256(out / "experiment.json"),
        "predictions": {"tiny__raw": pilot.sha256(out / "tiny__raw.json")}})


@pytest.mark.parametrize("target", ["tiny__raw.json", "tiny_references.json", "experiment.json"])
def test_score_refuses_mutation_before_opening_ground_truth(tmp_path, monkeypatch, target):
    _freeze(tmp_path)
    pilot.write(tmp_path / target, {"changed": True})
    original = pilot.read
    def guarded(path):
        if Path(path).name == "labels.json":
            raise AssertionError("labels opened before freeze validation")
        return original(path)
    monkeypatch.setattr(pilot, "read", guarded)
    with pytest.raises(ValueError, match="changed"):
        pilot.score(SimpleNamespace(out=tmp_path, gt=tmp_path / "labels.json"))


def test_all_registered_predictions_are_required_before_scoring(tmp_path):
    _freeze(tmp_path)
    frozen = pilot.read(tmp_path / "predictions_frozen.json")
    frozen["predictions"] = {}
    pilot.write(tmp_path / "predictions_frozen.json", frozen)
    with pytest.raises(ValueError, match="every registered prediction"):
        pilot.load_frozen_predictions(tmp_path)


@pytest.mark.parametrize("mutate", [False, True])
def test_resumed_inference_validates_freeze_without_rewriting_or_reblessing(tmp_path, monkeypatch, mutate):
    _freeze(tmp_path)
    if mutate:
        path = tmp_path / "tiny__raw.json"
        edited = pilot.read(path)
        edited["selected"] = {"route_latlon": [[1, 2]]}
        pilot.write(path, edited)
    def forbidden(*args, **kwargs):
        raise AssertionError("resumed inference rewrote a frozen file")
    monkeypatch.setattr(pilot, "write", forbidden)
    args = SimpleNamespace(out=tmp_path)
    registration = pilot.read(tmp_path / "experiment.json")
    if mutate:
        with pytest.raises(ValueError, match="prediction changed"):
            pilot.run_plans(args, {}, registration, None, {})
    else:
        pilot.run_plans(args, {}, registration, None, {})


def test_joint_artifact_is_verified_before_its_arrays_or_metadata_are_used(tmp_path):
    original, baseline, joint = [tmp_path / name for name in ("original", "baseline", "joint")]
    (original / "frozen_sources").mkdir(parents=True)
    baseline.mkdir()
    joint.mkdir()
    (original / "frozen_sources/pilot_joint_alignment.py").write_text("# unused geometry stub\n")
    pilot.write(baseline / "experiment.json", {"core_sha256": pilot.sha256(pilot.core.__file__)})
    pilot.write(original / "experiment.json", {"source_joint": {"raw": {"sha256": "0" * 64}}})
    (joint / "raw.npz").write_bytes(b"not an npz: hash refusal must happen first")
    with pytest.raises(ValueError, match="joint raw artifact changed"):
        pilot.load_inputs(SimpleNamespace(original=original, baseline=baseline, joint=joint))
