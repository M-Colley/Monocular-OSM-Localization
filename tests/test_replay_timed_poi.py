"""Model-free cached input/freeze tests; no actual experiment data."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import replay_timed_poi as runner
from monocular_osm.poi_clues import build_poi_hypotheses


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def input_case(tmp_path):
    cues, union = tmp_path / "cues", tmp_path / "union"
    cues.mkdir();union.mkdir()
    video = "a"*64
    K = np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    native = np.arange(160, dtype=np.float64)/10
    indices = np.linspace(0, 159, 80).astype(np.int64);times = native[indices]
    np.savez(cues / "cues_vo.npz", K=K, timestamps=native)
    pois = [{"id": f"node/{i*2+j}", "lat": lat+j*.00005, "lon": -80., "tags": {"name": name, "shop": "supermarket"}}
            for i, lat in enumerate((40., 40.01)) for j, name in enumerate(("AlphaMarket", "BetaCafe"))]
    inventory = {"city": "Synthetic", "scope": "named_city_bbox", "pois": pois}
    detections = [{"text": name, "confidence": .99, "t_sec": .5, "bbox": box} for name, box in
                  (("AlphaMarket", [60., 20., 80., 30.]), ("BetaCafe", [20., 20., 40., 30.]))]
    clues = build_poi_hypotheses(detections, inventory)
    provenance = {"source": "same_video_native_frames", "video_sha256": video, "native_bbox_grid": True,
        "position_stamps_rejected": True, "position_stamp_filter": "scene_text.is_position_stamp",
        "sample_count": 24, "min_len": 2, "min_confidence": .3, "super_res": False, "use_gpu": False,
        "frame_shape_hw": [80, 100], "window": [0., 15.9], "K_sha256": runner.core.base._array_sha(K),
        "detections_sha256": clues["detections_sha256"]}
    write(cues / "poi_ocr.json", {"provenance": provenance, "detections": detections})
    write(cues / "poi_inventory.json", inventory)
    write(cues / "cues.json", {"video_sha256": video, "clues": clues, "auto_quality": {"status": "complete",
        "n_frames": len(native), "timestamps_sha256": runner.core.base._array_sha(native), "video_sha256": video, "window": [0., 16.]}})
    plan = runner.producer.plan_image_union(clues, provenance)
    write(union / "image_union_plan.json", plan)
    arrays = {"ref_ids": np.array(["mapillary:a", "mapillary:b"]), "ref_latlon": np.array([[40., -80.], [40.01, -80.]]),
        "query_times": times, "query_indices": indices, "similarities": np.zeros((80, 2), np.float32)}
    np.savez(union / "image_reference_union.npz", **arrays)
    hashes = {"query_indices_sha256": runner.producer._array_sha(indices), "query_times_sha256": runner.producer._array_sha(times),
        "ref_ids_sha256": runner.producer._json_sha(arrays["ref_ids"].tolist()),
        "ref_latlon_sha256": runner.producer._array_sha(arrays["ref_latlon"]),
        "similarities_sha256": runner.producer._array_sha(arrays["similarities"])}
    identity = {"query_video_sha256": video, "query_mode": "crops", "views": 3,
        "aggregation": runner.producer.PROFILE["crop_aggregation"],
        "query_indices_sha256": hashes["query_indices_sha256"], "query_times_sha256": hashes["query_times_sha256"],
        "query_pixels_sha256": "b"*64, "query_descriptors_sha256": "c"*64, "actual_model_state_sha256": "d"*64}
    proof = {"status": "complete", "route_valid": True, "failed_regions": [], "successful_branch_count": 2,
        "profile": dict(runner.producer.PROFILE), "reference_union": {"hashes": hashes, "ref_ids": arrays["ref_ids"].tolist()},
        "provenance": {"common_query_encoder_identity": {"backbone": {"name": "megaloc", "device": "cpu",
            "hub": "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db"}, "query_identity": identity}}}
    write(union / "image_union_prediction.json", proof)
    paths = {role: {name: directory / name for name in runner.MEMBERS[role]}
             for role, directory in (("cues", cues), ("union", union))}
    return paths, video


def test_import_and_cli_have_no_neural_runtime():
    code = "import sys; from scripts import replay_timed_poi; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=Path(runner.__file__).parents[1], check=True)


def test_typed_native_and_full_matrix_identity_builds_all_eight_outputs(tmp_path):
    paths, video = input_case(tmp_path)
    arrays, plan, identity = runner.typed_inputs(paths, video)
    assert arrays["similarities"].shape == (80, 2)
    assert len(plan["events"]) == 2 and len(plan["branches"]) == 2
    assert len(runner.requested_outputs(plan)) == 8
    assert identity["registered_window"] == [0., 16.]
    assert plan["events"][0]["row"] == 3


@pytest.mark.parametrize("change", ["grid", "video", "model", "hash", "partial"])
def test_mismatched_grid_video_encoder_matrix_or_partial_listing_refused(tmp_path, change):
    paths, video = input_case(tmp_path)
    path = paths["union"]["image_union_prediction.json"]
    proof = json.loads(path.read_text())
    common = proof["provenance"]["common_query_encoder_identity"]
    if change == "grid": common["query_identity"]["query_indices_sha256"] = "e"*64
    if change == "video": common["query_identity"]["query_video_sha256"] = "e"*64
    if change == "model": common["backbone"]["name"] = "another"
    if change == "hash": proof["reference_union"]["hashes"]["similarities_sha256"] = "e"*64
    if change == "partial": proof["successful_branch_count"] = 1
    write(path, proof)
    with pytest.raises(ValueError): runner.typed_inputs(paths, video)


@pytest.mark.parametrize("linked", [True, False])
def test_parent_must_bind_exact_child_freezes(tmp_path, monkeypatch, linked):
    parent = tmp_path / "parent"
    paths = {"parent": parent / "frozen.json", "cues": parent / "children/run/cues/child_frozen.json",
             "union": parent / "children/run/union/child_frozen.json"}
    for path in paths.values(): write(path, {})
    spec = {role: {"freeze": str(path), "freeze_sha256": (str(i)*64)} for i, (role, path) in enumerate(paths.items(), 1)}
    manifests = {"parent": {"scoring_has_not_run": True, "artifacts": {
        str(paths[role].relative_to(parent)): spec[role]["freeze_sha256"] if linked else "f"*64 for role in ("cues", "union")}}}
    for role in ("cues", "union"):
        manifests[role] = {"scoring_has_not_run": True, "video_sha256": "a"*64,
                           "artifacts": {name: "b"*64 for name in runner.MEMBERS[role]}}
    monkeypatch.setattr(runner.frozen, "freeze", lambda path, expected: manifests[next(role for role, p in paths.items() if p == path)])
    if linked:
        records, found, video = runner.original_inputs(spec)
        assert video == "a"*64 and set(found) == {"cues", "union"}
    else:
        with pytest.raises(ValueError, match="exact child"):
            runner.original_inputs(spec)


def _mock_infer(tmp_path, monkeypatch, *, mutate=False, incomplete=False):
    family = tmp_path / "family";family.mkdir()
    (family / "inputs_frozen.json").write_text("original")
    expected = runner.sha(family / "inputs_frozen.json")
    plan = {"branches": [{"id": "one"}, {"id": "two"}]}
    wanted = runner.requested_outputs(plan)
    registration = {"all_requested_outputs": wanted, "source_sha256": {}, "input_identity": {}}
    arrays = {"similarities": np.zeros((80, 2), np.float32), "ref_ids": np.array(["a", "b"]),
        "ref_latlon": np.zeros((2, 2)), "query_times": np.arange(80.)}
    calls = []
    def inputs(root, exact, commit):
        calls.append(exact)
        runner.require(runner.sha(root / "inputs_frozen.json") == exact, "expected freeze changed")
        return arrays, plan, registration
    def decode(*args):
        outputs = {name: {"status": "complete", "synthetic": i} for i, name in enumerate(wanted)}
        if incomplete: del outputs[wanted[-1]]
        if mutate: (family / "inputs_frozen.json").write_text("reblessed")
        return {"profile_sha256": runner.core.PROFILE_SHA256, "outputs": outputs,
                "confidence": None, "selected_variant": None}
    monkeypatch.setattr(runner, "prepared_inputs", inputs)
    monkeypatch.setattr(runner.core, "decode_timed_tracks", decode)
    monkeypatch.setattr(runner.frozen, "check_headroom", lambda: {})
    monkeypatch.setattr(runner, "sys", SimpleNamespace(modules={}))
    return family, expected, calls


def test_mocked_cached_decode_freezes_every_eight_output(tmp_path, monkeypatch):
    family, expected, calls = _mock_infer(tmp_path, monkeypatch)
    result = runner.infer(family, "pin", expected)
    assert result["outputs"] == 8
    master = json.loads((family / "predictions/frozen_predictions.json").read_text())
    assert len(master["all_requested_outputs"]) == 8 and master["confidence"] is None
    assert calls == [expected, expected]


@pytest.mark.parametrize("failure", ["mutation", "incomplete"])
def test_reblessing_or_missing_output_cannot_freeze_master(tmp_path, monkeypatch, failure):
    family, expected, _ = _mock_infer(tmp_path, monkeypatch,
        mutate=failure == "mutation", incomplete=failure == "incomplete")
    with pytest.raises(ValueError): runner.infer(family, "pin", expected)
    assert not (family / "predictions/frozen_predictions.json").exists()
    assert (family / "predictions/failure.json").exists()


def test_spec_and_cli_reject_saved_scores_and_label_arguments(tmp_path, monkeypatch):
    spec = {role: {"freeze": str(tmp_path / role / "frozen.json"), "freeze_sha256": "a"*64}
            for role in ("cues", "union", "parent")}
    path = tmp_path / "spec.json";write(path, spec)
    assert runner.specification(path) == spec
    spec["cues"]["freeze"] = str(tmp_path / "offline_scores.json");write(path, spec)
    with pytest.raises(ValueError, match="scores"): runner.specification(path)
    monkeypatch.setattr(runner.frozen, "offline", lambda: nullcontext())
    with pytest.raises(SystemExit):
        runner.main(["infer", "--family", str(tmp_path), "--expected-commit", "pin",
                     "--input-freeze-sha256", "a"*64, "--ground-truth", "labels"])


def test_self_consistent_partial_query_grid_is_still_refused(tmp_path):
    paths, video = input_case(tmp_path)
    path = paths["union"]["image_reference_union.npz"]
    with np.load(path) as saved: arrays = {name: saved[name] for name in saved.files}
    arrays["query_indices"][-1] = 158
    arrays["query_times"][-1] = 15.8
    np.savez(path, **arrays)
    proof_path = paths["union"]["image_union_prediction.json"]
    proof = json.loads(proof_path.read_text())
    for field in ("query_indices", "query_times"):
        digest = runner.producer._array_sha(arrays[field])
        proof["reference_union"]["hashes"][field+"_sha256"] = digest
        proof["provenance"]["common_query_encoder_identity"]["query_identity"][field+"_sha256"] = digest
    write(proof_path, proof)
    with pytest.raises(ValueError, match="native camera grid"):
        runner.typed_inputs(paths, video)
