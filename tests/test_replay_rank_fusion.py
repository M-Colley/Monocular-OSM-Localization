"""Synthetic frozen-input/provenance runner checks; no real galleries/labels."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import replay_rank_fusion as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "SHAPE", (3, 4))
    monkeypatch.setattr(r, "source_identity", lambda *args: {})
    monkeypatch.setattr(r.utility, "check_headroom", lambda: {"available_physical_bytes": 20*1024**3,
                                                          "available_commit_bytes": 20*1024**3})
    root = tmp_path/"old"
    output = root/"inference"
    output.mkdir(parents=True)
    arrays = {"ref_ids": np.array(["z", "a", "b", "c"]),
        "ref_latlon": np.array([[50., 0.], [50., .001], [50., .002], [50., .003]]),
        "query_times": np.array([0., 3., 8.]), "query_indices": np.arange(3), "frame_ordinals": np.array([10, 100, 250])}
    for i, key in enumerate(r.MATRIX_KEYS.values()):
        arrays[key] = np.roll(np.array([[.1, .1, .3, .4], [.2, .5, .3, .1], [.8, .2, .4, .3]], np.float32), i, axis=1)
    np.savez(output/"full_evidence.npz", **arrays)
    identity = {k: r.heading_core._array_sha(arrays[k]) for k in r.SHARED}
    identity["matrices"] = {m: r.heading_core._array_sha(arrays[k]) for m, k in r.MATRIX_KEYS.items()}
    identity.update(video_sha256="v"*64, native_pixels_sha256="p"*64)
    write(output/"runtime.json", {"source_commit": "old_pin", "input_freeze_sha256": "a"*64, "input_identity": identity})
    controls = []
    for i, name in enumerate(r.OLD_OUTPUTS):
        filename = f"control_{i}.json"
        write(output/filename, {"status": "complete", "original_control": i})
        controls.append({"name": name, "file": filename, "sha256": r.sha(output/filename)})
    write(output/"comparison_index.json", {"profile_sha256": r.SOURCE_PROFILE, "outputs": controls})
    master = output/"frozen_predictions.json"
    write(master, {"status": "complete", "scoring_has_not_run": True, "profile_sha256": r.SOURCE_PROFILE,
        "source_commit": "old_pin", "input_freeze_sha256": "a"*64, "selected_variant": None,
        "new_outputs": 8, "retained_controls": 16, "all_requested_outputs": r.OLD_OUTPUTS,
        "artifacts": {p.name: r.sha(p) for p in output.iterdir()}})
    upstream = root/"upstream_pixel.dat"
    upstream.write_bytes(b"protected canonical pixel snapshot")
    protected = {str(p): r.sha(p) for p in [*output.iterdir(), upstream]}
    audit = tmp_path/"audit"/"audit.json"
    write(audit, {"status": "PASS", "input_only": False, "family": str(root), "source_commit": "old_pin",
        "ground_truth_or_official_scores_opened": False, "new_neural_forwards": False, "model_constructors": False,
        "auditor_sha256": "c"*64, "dependency_sha256": "d"*64,
        "freeze_sha256": r.sha(master), "input_freeze_sha256": "a"*64,
        "protected_sha256": protected, "protected_files": len(protected)})
    spec = tmp_path/"spec.json"
    write(spec, {"source": {"freeze": str(master), "sha256": r.sha(master)},
                 "audit": {"path": str(audit), "sha256": r.sha(audit)}})
    return spec, output, upstream, arrays


def test_import_is_model_free():
    code = "import sys; from scripts import replay_rank_fusion; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=r.ROOT, check=True)


def test_profile_registers_all16_new24_unchanged_and_no_winner():
    assert len(r.NEW_OUTPUTS) == 16 and len(set(r.NEW_OUTPUTS)) == 16
    assert len(r.OLD_OUTPUTS) == 24 and len(set(r.ALL_OUTPUTS)) == 40
    assert r.PROFILE["percentile_dtype"] == "float64"
    assert r.PROFILE["selected_variant"] is r.PROFILE["confidence"] is None
    assert r.PROFILE["candidate_pruning"] is r.PROFILE["new_forwards"] is False
    assert r.PROFILE["heading_profile_sha256"] == r.heading_core.PROFILE_SHA256
    assert r.PROFILE["simple_profile_sha256"] == r.full.PROFILE_SHA256


def test_windows_frozen_artifact_spelling_matches_posix_index_without_aliases():
    digest = "a"*64
    assert r.canonical_artifacts({"controls\\heading_prediction.json": digest}) == {
        "controls/heading_prediction.json": digest}
    for entries in ({"controls\\a.json": digest, "controls/a.json": digest},
                    {"controls/a.json": digest, "Controls/A.json": digest}, {"../escape.json": digest}):
        with pytest.raises(ValueError): r.canonical_artifacts(entries)


def test_source_read_validates_complete_native_and_protected_identity_without_ranking(tmp_path, monkeypatch):
    spec, output, _, _ = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(r.core, "calibrate_evidence", lambda *_: pytest.fail("prepare must not rank"))
    monkeypatch.setattr(r.core, "decode_erased_tracks", lambda *_: pytest.fail("prepare must not decode"))
    bundle = r.verified_source(r.specification(spec))
    assert len(bundle["controls"]) == 24 and bundle["source"] == output/"frozen_predictions.json"
    assert set(bundle["identity"]["matrices"]) == set(r.core.MODELS)


@pytest.mark.parametrize("change", ["audit_status", "audit_master", "audit_count", "protected_bytes", "missing_control"])
def test_unaccepted_or_modified_source_declines(tmp_path, monkeypatch, change):
    spec, output, upstream, _ = source_fixture(tmp_path, monkeypatch)
    declared = r.specification(spec)
    audit_path = Path(declared["audit"]["path"])
    audit = r.load(audit_path)
    if change == "audit_status": audit["status"] = "INPUT_ONLY"
    if change == "audit_master": audit["freeze_sha256"] = "b"*64
    if change == "audit_count": audit["protected_files"] -= 1
    if change == "protected_bytes": upstream.write_bytes(b"modified")
    if change == "missing_control": (output/"control_12.json").unlink()
    if change.startswith("audit_"):
        write(audit_path, audit)
        declared["audit"]["sha256"] = r.sha(audit_path)
    with pytest.raises((ValueError, FileNotFoundError)): r.verified_source(declared)


@pytest.mark.parametrize("bad", ["offline_scores.json", "run_official_score.json", "Ground_Truth/input.json"])
def test_label_and_saved_score_names_decline_before_hash(tmp_path, monkeypatch, bad):
    path = tmp_path/"spec.json"
    write(path, {"source": {"freeze": str(tmp_path/bad), "sha256": "a"*64},
                 "audit": {"path": "audit.json", "sha256": "b"*64}})
    with pytest.raises(ValueError, match="scores|labels"): r.specification(path)


@pytest.mark.parametrize("bad", ["dtype", "shape", "NaN", "ordinals", "queries", "extra"])
def test_evidence_header_and_grid_fail_closed(tmp_path, monkeypatch, bad):
    _, output, _, arrays = source_fixture(tmp_path, monkeypatch)
    path = tmp_path/"bad.npz"
    if bad == "dtype": arrays["megaloc_raw_cosines"] = arrays["megaloc_raw_cosines"].astype(np.float64)
    if bad == "shape": arrays["image_raw_cosines"] = arrays["image_raw_cosines"][:, :-1]
    if bad == "NaN": arrays["selavpr_raw_cosines"][0, 0] = np.nan
    if bad == "ordinals": arrays["frame_ordinals"][1] = arrays["frame_ordinals"][0]
    if bad == "queries": arrays["query_indices"] = np.array([1, 2, 3])
    if bad == "extra": arrays["unknown"] = np.ones(2)
    np.savez(path, **arrays)
    with pytest.raises(ValueError): r.read_evidence(path)


def test_small_complete_prepare_infer_freezes40_and_preserves_controls(tmp_path, monkeypatch):
    spec, _, _, _ = source_fixture(tmp_path, monkeypatch)
    family = tmp_path/"new"
    registered = r.prepare(spec, family, "pin")
    digest = registered["input_freeze_sha256"]
    assert not (family/"predictions").exists()
    result = r.infer(family, "pin", digest)
    assert result["outputs"] == 40
    frozen = r.utility.freeze(family/"predictions/frozen_predictions.json", result["freeze_sha256"])
    assert frozen["all_requested_outputs"] == r.ALL_OUTPUTS
    index = r.load(family/"predictions/comparison_index.json")["outputs"]
    assert len(index) == 40 and all(r.sha(family/"predictions"/row["file"]) == row["sha256"] for row in index)
    for row in index[16:]: assert Path(row["source"]).read_bytes() == (family/"predictions"/row["file"]).read_bytes()
    with np.load(family/"predictions/calibrated_evidence.npz", allow_pickle=False) as z:
        assert z["equal3"].dtype == np.float64
        np.testing.assert_array_equal(z["equal3"], (z["rank_megaloc"]+z["rank_image"]+z["rank_selavpr"])/3)
    with pytest.raises(FileExistsError): r.infer(family, "pin", digest)


def test_final_expected_input_sha_prevents_in_run_plan_reblessing(tmp_path, monkeypatch):
    spec, _, _, _ = source_fixture(tmp_path, monkeypatch)
    family = tmp_path/"new"
    digest = r.prepare(spec, family, "pin")["input_freeze_sha256"]
    original = r.decode_all
    def mutate(*args):
        result = original(*args)
        plan = r.load(family/"plan.json")
        plan["tampered"] = True
        write(family/"plan.json", plan)
        frozen = r.load(family/"inputs_frozen.json")
        frozen["artifacts"]["plan.json"] = r.sha(family/"plan.json")
        write(family/"inputs_frozen.json", frozen)
        return result
    monkeypatch.setattr(r, "decode_all", mutate)
    with pytest.raises(ValueError, match="freeze changed"): r.infer(family, "pin", digest)
    assert not (family/"predictions/frozen_predictions.json").exists()
    assert (family/"predictions/failure.json").exists()


def test_decode_failure_keeps_partial_without_master(tmp_path, monkeypatch):
    spec, _, _, _ = source_fixture(tmp_path, monkeypatch)
    family = tmp_path/"new"
    digest = r.prepare(spec, family, "pin")["input_freeze_sha256"]
    def fail(*args): raise RuntimeError("synthetic resource refusal")
    monkeypatch.setattr(r, "decode_all", fail)
    with pytest.raises(RuntimeError, match="resource"): r.infer(family, "pin", digest)
    assert not (family/"predictions/frozen_predictions.json").exists()
    assert (family/"predictions/calibrated_evidence.npz").exists()


def test_runtime_gate_checks_wall_and_both_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(r.time, "monotonic", lambda: 20000)
    with pytest.raises(ValueError, match="wall"): r.resource_gate(0)
    monkeypatch.setattr(r.time, "monotonic", lambda: 1)
    def decline(): raise ValueError("physical and actual commit unavailable")
    monkeypatch.setattr(r.utility, "check_headroom", decline)
    with pytest.raises(ValueError, match="actual commit"): r.resource_gate(0)


@pytest.mark.parametrize("flag", ["ground_truth_or_official_scores_opened", "new_neural_forwards", "model_constructors"])
def test_audit_must_explicitly_exclude_labels_models_and_forwards(tmp_path, monkeypatch, flag):
    spec, _, _, _ = source_fixture(tmp_path, monkeypatch)
    declared = r.specification(spec)
    path = Path(declared["audit"]["path"])
    audit = r.load(path)
    audit[flag] = True
    write(path, audit)
    declared["audit"]["sha256"] = r.sha(path)
    with pytest.raises(ValueError, match="input-only scope"): r.verified_source(declared)


def test_actual_native_thread_limits_are_two():
    with r.numerical_threads() as pools:
        assert len(pools) >= 2 and all(row["num_threads"] == 2 for row in pools)


def test_no_existing_family_overwrite(tmp_path, monkeypatch):
    spec, _, _, _ = source_fixture(tmp_path, monkeypatch)
    family = tmp_path/"exists"
    family.mkdir()
    with pytest.raises(ValueError, match="isolated"): r.prepare(spec, family, "pin")
