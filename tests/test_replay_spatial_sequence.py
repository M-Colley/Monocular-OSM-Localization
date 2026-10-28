from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import zipfile

import numpy as np
import pytest

from scripts import replay_spatial_sequence as runner


def freeze(root, name):
    runner.write(root/name, {"status": "complete", "scoring_has_not_run": True,
        "artifacts": {str(path.relative_to(root)): runner.sha(path) for path in root.rglob("*") if path.is_file()}})
    return runner.sha(root/name)


@pytest.fixture
def family(tmp_path):
    root = tmp_path/"original"; evidence = root/"inference"; inputs = root/"inputs"
    evidence.mkdir(parents=True); inputs.mkdir()
    ids = np.array(["mapillary/1", "mapillary/2", "mapillary/3"])
    refs = np.array([[48., 8.], [48., 8.001], [48.001, 8.]])
    times, ordinals = np.array([0.2, 1.2, 2.2]), np.array([6, 36, 66])
    full = {"raw_cosines": np.array([[.1, .2, .3], [.3, .2, .1], [.2, .3, .1]], dtype=np.float32),
        "query_times": times, "query_indices": np.arange(3), "frame_ordinals": ordinals,
        "ref_ids": ids, "ref_latlon": refs}
    runner.write(inputs/"replay_plan.json", {"union": {"ref_ids": ids.tolist(), "count": 3,
        "ref_latlon_sha256": runner.array_sha(refs)}, "query": {"video": {"sha256": "b"*64},
        "registered_window": [0., 3.], "planned_frame_ordinals": ordinals.tolist(), "stride": 30, "fps": 30.}})
    input_digest = freeze(root, "inputs_frozen.json")
    np.savez(evidence/"full_evidence.npz", **full)
    runner.write(evidence/"runtime_and_lineage.json", {"labels_opened": False, "scoring_called": False,
        "input_freeze_sha256": input_digest, "actual_query_times_sha256": runner.array_sha(times),
        "per_query_native_pixel_sha256": ["c"*64, "d"*64, "e"*64]})
    digest = freeze(evidence, "frozen_predictions.json")
    return evidence, digest, full


def refreeze(evidence):
    path = evidence/"frozen_predictions.json"
    path.unlink()
    return freeze(evidence, path.name)


def test_full_evidence_identity_and_no_input_mutation(family):
    evidence, digest, full = family
    before = {path: runner.sha(path) for path in evidence.parent.rglob("*") if path.is_file()}
    loaded, identity = runner.load_frozen_evidence(evidence, digest)
    for name, value in full.items():
        np.testing.assert_array_equal(loaded[name], value)
    assert identity["reference_count"] == 3 and identity["query_count"] == 3
    assert identity["matrix_file_sha256"] == runner.sha(evidence/"full_evidence.npz")
    assert before == {path: runner.sha(path) for path in before}


@pytest.mark.parametrize("change", ["time", "ordinal", "union", "pixels", "labels"])
def test_reblessed_different_identity_is_refused(family, change):
    evidence, _, full = family
    if change == "time":
        full["query_times"][1] += .1
    if change == "ordinal":
        full["frame_ordinals"][1] += 1
    if change == "union":
        full["ref_latlon"][0, 0] += .001
    if change in {"time", "ordinal", "union"}:
        np.savez(evidence/"full_evidence.npz", **full)
    else:
        runtime = runner.read(evidence/"runtime_and_lineage.json")
        runtime["per_query_native_pixel_sha256" if change == "pixels" else "labels_opened"] = [] if change == "pixels" else True
        runner.write(evidence/"runtime_and_lineage.json", runtime)
    digest = refreeze(evidence)
    with pytest.raises(ValueError):
        runner.load_frozen_evidence(evidence, digest)


def test_unfrozen_input_change_fails_hash_before_load(family, monkeypatch):
    evidence, digest, _ = family
    (evidence/"runtime_and_lineage.json").write_text("{}")
    monkeypatch.setattr(runner, "bounded_evidence", lambda _: pytest.fail("loaded changed input"))
    with pytest.raises(ValueError, match="changed"):
        runner.load_frozen_evidence(evidence, digest)


def test_omitted_matrix_is_not_implicitly_blessed_by_manifest(family):
    evidence, _, _ = family
    record = runner.read(evidence/"frozen_predictions.json")
    del record["artifacts"]["full_evidence.npz"]
    runner.write(evidence/"frozen_predictions.json", record)
    with pytest.raises(ValueError, match="not covered"):
        runner.load_frozen_evidence(evidence, runner.sha(evidence/"frozen_predictions.json"))


def test_npz_rejects_object_and_duplicate_members_before_numpy_load(family, monkeypatch, tmp_path):
    _, _, full = family
    path = tmp_path/"bad.npz"
    full["raw_cosines"] = np.array([[object()]], dtype=object)
    np.savez(path, **full)
    monkeypatch.setattr(runner.np, "load", lambda *a, **k: pytest.fail("loaded object data"))
    with pytest.raises(ValueError, match="typed budget"):
        runner.bounded_evidence(path)
    buffer = tmp_path/"array.npy"
    with buffer.open("wb") as stream:
        np.save(stream, np.array([1.]))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("raw_cosines.npy", buffer.read_bytes())
        with pytest.warns(UserWarning):
            archive.writestr("raw_cosines.npy", buffer.read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        runner.bounded_evidence(path)


def test_new_output_refuses_existing_input_and_source_locations(family, tmp_path):
    evidence, _, _ = family
    for path in (evidence, evidence/"child", evidence.parent/"another", runner.ROOT/"new-output", tmp_path):
        with pytest.raises(ValueError, match="isolated"):
            runner.new_output(path, evidence)
    assert runner.new_output(tmp_path/"new-family", evidence) == tmp_path/"new-family"


def test_replay_freezes_both_posteriors_routes_sources_and_config_before_any_score(family, tmp_path, monkeypatch):
    evidence, digest, full = family
    monkeypatch.setattr(runner, "source_hashes", lambda *a: {})
    monkeypatch.setattr(runner, "windows_memory", lambda: {"available_physical_gib": 10., "available_commit_gib": 10.})
    output = tmp_path/"new-family"
    args = Namespace(expected_commit="f"*40, evidence_family=evidence, evidence_freeze_sha256=digest, output=output)
    runner.replay(args)
    freeze_record = runner.read(output/"frozen_predictions.json")
    assert set(freeze_record["variants"]) == {"raw", "reference_mean"}
    assert freeze_record["selected_variant"] is None and freeze_record["scoring_has_not_run"] is True
    for relative, expected in freeze_record["artifacts"].items():
        assert runner.sha(output/relative) == expected
    registration = runner.read(output/"spatial_registration.json")
    assert registration["profile_sha256"] == runner.json_sha(runner.SpatialSequenceConfig().record())
    assert registration["models_or_network_or_vo"] is False and registration["ground_truth_used"] is False
    with np.load(output/"spatial_posteriors.npz", allow_pickle=False) as arrays:
        for variant in ("raw", "reference_mean"):
            prediction = runner.read(output/f"prediction_{variant}.json")
            cube = arrays[variant+"_posterior"]
            assert len(cube) == len(full["query_times"])
            assert prediction["posterior_sha256"] == runner.array_sha(cube)
            assert prediction["route_timestamps"] == full["query_times"].tolist()
            assert prediction["confidence"] is None
    before = runner.sha(output/"frozen_predictions.json")
    with pytest.raises(ValueError, match="no resume"):
        runner.replay(args)
    assert before == runner.sha(output/"frozen_predictions.json")


def test_inputs_changed_during_inference_prevent_freeze(family, tmp_path, monkeypatch):
    evidence, digest, _ = family
    monkeypatch.setattr(runner, "source_hashes", lambda *a: {})
    monkeypatch.setattr(runner, "windows_memory", lambda: {"available_physical_gib": 10., "available_commit_gib": 10.})
    decode = runner.decode_spatial_tracks
    def mutate(*args, **kwargs):
        result = decode(*args, **kwargs)
        (evidence/"runtime_and_lineage.json").write_text("{}")
        return result
    monkeypatch.setattr(runner, "decode_spatial_tracks", mutate)
    output = tmp_path/"new-family"
    with pytest.raises(ValueError, match="changed"):
        runner.replay(Namespace(expected_commit="f"*40, evidence_family=evidence, evidence_freeze_sha256=digest, output=output))
    assert not (output/"frozen_predictions.json").exists()
    assert runner.read(output/"failed.json")["all_predictions_frozen"] is False


def test_memory_guard_precedes_evidence_or_decoder(family, tmp_path, monkeypatch):
    evidence, digest, _ = family
    monkeypatch.setattr(runner, "source_hashes", lambda *a: {})
    monkeypatch.setattr(runner, "windows_memory", lambda: {"available_physical_gib": 7.99, "available_commit_gib": 12.})
    monkeypatch.setattr(runner, "load_frozen_evidence", lambda *a: pytest.fail("loaded before memory guard"))
    with pytest.raises(ValueError, match="8GiB"):
        runner.replay(Namespace(expected_commit="f"*40, evidence_family=evidence, evidence_freeze_sha256=digest, output=tmp_path/"out"))


def test_executed_source_and_imported_modules_are_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="executed"):
        runner.source_hashes("f"*40)


def test_final_source_digest_mismatch_is_refused(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda args, **kwargs: "f"*40 if "rev-parse" in args else "")
    with pytest.raises(ValueError, match="source changed"):
        runner.source_hashes("f"*40, {})
    outside = SimpleNamespace(__file__="E:/outside/fake.py")
    monkeypatch.setitem(sys.modules, "monocular_osm._test_escape", outside)
    with pytest.raises(ValueError, match="escaped"):
        runner.source_hashes("f"*40)
