"""Synthetic/model-free full Sela comparison runner boundaries."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import replay_selavpr_gallery as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def test_import_does_not_import_Torch_or_official_model():
    code = ("import sys; from scripts import replay_selavpr_gallery; assert 'torch' not in sys.modules; "
            "assert not any(n=='model' or n.startswith('model.') for n in sys.modules)")
    subprocess.run([sys.executable, "-c", code], cwd=Path(runner.__file__).parents[1], check=True)


def test_profile_fixes_all_eight_new_sixteen_old_and_no_selection():
    assert len(runner.NEW_OUTPUTS) == 8 and len(set(runner.NEW_OUTPUTS)) == 8
    assert len(runner.CONTROL_FILES) == 16
    assert runner.PROFILE["comparison_count"] == 24
    assert runner.PROFILE["selected_variant"] is runner.PROFILE["confidence"] is None
    assert runner.PROFILE["fresh_forward_count"] == 9735 + 240
    assert runner.PROFILE["topK"] is runner.PROFILE["labels_or_scores"] is False
    assert runner.PROFILE["runtime_wall_budget_s"] == 10800
    assert runner.PROFILE["heading_profile_sha256"] == runner.heading_core.PROFILE_SHA256


def test_exact_spec_roles_and_score_guard(tmp_path):
    spec = {role: {"freeze": str(tmp_path / role / "frozen.json"), "freeze_sha256": "a" * 64}
            for role in runner.ROLES}
    spec["assets"] = str(tmp_path / "assets")
    path = tmp_path / "spec.json"
    write_json(path, spec)
    assert runner.specification(path) == spec
    spec["gallery"]["freeze"] = str(tmp_path / "offline_scores.json")
    write_json(path, spec)
    with pytest.raises(ValueError, match="scores"):
        runner.specification(path)
    del spec["gallery"]
    write_json(path, spec)
    with pytest.raises(ValueError, match="roles"):
        runner.specification(path)


def _smoke(tmp_path, monkeypatch):
    root = tmp_path / "smoke"
    output = root / "inference"
    output.mkdir(parents=True)
    descriptors = np.zeros((44, 2048), np.float32)
    descriptors[:, 0] = 1
    np.save(output / "descriptors.npy", descriptors)
    actual = {"forward_calls": 44, "cpu_threads": 2, "interop_threads": 1, "device": "cpu",
        "strict_state_load": True, "xformers_available": {"a": False}, "torch": "version", "torchvision": "vision"}
    runtime = {"runtime": actual, "state": {"elements": 97909629, "sha256": "a" * 64}, "constructor_s": 1,
        "per_image": [{"seconds": 1, "descriptor_norm": 1,
            "preprocessing": {"canonical_bitwise_equal": True, "input_tensor_sha256": "b" * 64}} for _ in range(44)]}
    registration = {"profile": runner.smoke.PROFILE, "profile_sha256": runner.smoke.PROFILE_SHA256,
        "source_commit": "pin", "assets": {}, "reference_columns": list(range(32))}
    write_json(root / "registration.json", registration)
    write_json(output / "runtime.json", runtime)
    frozen = {"status": "complete", "forward_count": 44, "descriptors_shape": [44, 2048],
        "no_accuracy_result": True, "labels_opened": False, "route_decoding": False,
        "profile_sha256": runner.smoke.PROFILE_SHA256, "input_freeze_sha256": "c" * 64, "source_commit": "pin"}
    path = output / "runtime_frozen.json"
    write_json(path, frozen)
    calls = []
    monkeypatch.setattr(runner.heading, "freeze", lambda p, expected: calls.append(expected) or frozen)
    return root, path, runtime, calls


def test_smoke_arithmetic_and_exact_input_provenance_checked(tmp_path, monkeypatch):
    _, path, _, calls = _smoke(tmp_path, monkeypatch)
    result = runner.smoke_proof(path, "d" * 64)
    assert result["all44_norms_and_preprocessing_verified"] is True
    assert calls == ["d" * 64, "c" * 64]


@pytest.mark.parametrize("change", ["count", "normalization", "preprocessing", "CPU"])
def test_incomplete_or_bad_smoke_declines_without_model(tmp_path, monkeypatch, change):
    root, path, runtime, _ = _smoke(tmp_path, monkeypatch)
    if change == "count": runtime["runtime"]["forward_calls"] = 43
    if change == "CPU": runtime["runtime"]["device"] = "cuda"
    if change == "preprocessing": runtime["per_image"][3]["preprocessing"]["canonical_bitwise_equal"] = False
    if change == "normalization":
        values = np.load(path.parent / "descriptors.npy")
        values[0] *= 2
        np.save(path.parent / "descriptors.npy", values)
    write_json(root / "inference/runtime.json", runtime)
    with pytest.raises(ValueError):
        runner.smoke_proof(path, "d" * 64)


def test_complete_view_cosines_preserve_all_rows_columns_and_frame_major_order():
    queries = np.zeros((240, 2048), np.float32)
    queries[np.arange(240), np.arange(240) % 11] = 1
    refs = np.zeros((257, 2048), np.float32)
    refs[np.arange(257), np.arange(257) % 11] = 1
    actual = runner.cosine_views(queries, refs)
    assert actual.shape == (80, 3, 257)
    np.testing.assert_array_equal(actual.reshape(240, 257), queries @ refs.T)


def test_control_encoder_freezes_must_match_across_both_objectives(tmp_path):
    for role in ("image", "megaloc", "metadata", "motion"):
        write_json(tmp_path / "inputs" / role / "original_freeze.json", {"role": role})
    image = runner.sha(tmp_path / "inputs/image/original_freeze.json")
    mega = runner.sha(tmp_path / "inputs/megaloc/original_freeze.json")
    actual = runner.control_linkage(tmp_path, {"parent_prediction_freeze_sha256": mega},
                                    {"simple": {"freeze_sha256": image}})
    assert set(actual) == {"metadata", "motion"}
    with pytest.raises(ValueError, match="same frozen encoder"):
        runner.control_linkage(tmp_path, {"parent_prediction_freeze_sha256": mega},
                               {"simple": {"freeze_sha256": "a" * 64}})


@pytest.mark.parametrize("failure", ["dtype", "norm", "shape"])
def test_invalid_descriptor_grids_fail(failure):
    q = np.zeros((240, 2048), np.float32);q[:, 0] = 1
    r = np.zeros((2, 2048), np.float32);r[:, 0] = 1
    if failure == "dtype": q = q.astype(np.float64)
    if failure == "norm": r[0] *= 2
    if failure == "shape": q = q[:-1]
    with pytest.raises(ValueError):
        runner.cosine_views(q, r)


def test_both_decoders_create_all_eight_routes_on_small_synthetic_case(tmp_path):
    shared = {"ref_ids": np.array(["mapillary:a", "mapillary:b", "mapillary:c"]),
        "ref_latlon": np.array([[40., -80.], [40.001, -80.], [40.002, -80.]]),
        "query_times": np.array([0., 7., 14.])}
    bundle = {"shared": shared, "cameras": ["perspective", "spherical", None],
        "angles": np.array([0., np.nan, np.nan]), "offsets": np.array([-17., 0., 17.]),
        "yaw": {"vo_yaw_delta_deg": np.array([0., np.nan]), "vo_quality": np.array([1., 0.])}}
    views = np.random.default_rng(0).uniform(.1, .5, (3, 3, 3)).astype(np.float32)
    rows = runner.write_new_routes(tmp_path, views, bundle)
    assert [row["name"] for row in rows] == runner.NEW_OUTPUTS
    assert all(runner.sha(tmp_path / row["file"]) == row["sha256"] for row in rows)
    assert len(list(tmp_path.glob("prediction_selavpr_*.json"))) == 8


def test_reference_encoder_visits_every_canonical_pixel_and_closes_staging(tmp_path, monkeypatch):
    root, output = tmp_path / "gallery", tmp_path / "inference"
    root.mkdir();output.mkdir()
    raw = np.stack([np.full((512, 512, 3), i, np.uint8) for i in (3, 7)])
    ids, xy = np.array(["11", "12"]), np.array([[40., -80.], [40.001, -80.]])
    np.savez(root / "raw.npz", raw=raw, ids=ids, xy=xy)
    source = {"source": "original", "copy": "raw.npz", "sha256": runner.sha(root / "raw.npz"),
              "raw_shape": list(raw.shape)}
    rows = [{"source": "original", "raw_row": i, "column": 1-i, "id": "mapillary:" + str(ids[i]),
        "latlon": xy[i].tolist(), "pixels_sha256": hashlib.sha256(raw[i].tobytes()).hexdigest()} for i in range(2)]
    plan = {"raw_sources": [source], "canonical_rows": rows}
    visited = []
    def encode_one(model, pixels, record, started, records):
        visited.append(record["id"])
        value = np.zeros(2048, np.float32);value[int(pixels[0, 0, 0])] = 1
        return value
    monkeypatch.setattr(runner, "encode_one", encode_one)
    monkeypatch.setattr(runner, "resource_gate", lambda *a: {})
    runner.encode_references(root, plan, object(), output / "refs.npy", 0, [])
    result = np.load(output / "refs.npy")
    assert visited == ["mapillary:11", "mapillary:12"]
    assert result[0, 7] == result[1, 3] == 1
    assert not (output / "raw_staging").exists()


def test_all80_queries_each_three_native_crops_and_cleanup(tmp_path, monkeypatch):
    events, rows = [], []
    frames = SimpleNamespace(frames=[np.full((2, 3, 3), i, np.uint8) for i in range(80)],
                             close=lambda: events.append("closed"))
    monkeypatch.setattr(runner.gallery, "native_queries", lambda *a: frames)
    def encode_one(model, pixels, record, started, records):
        assert pixels.shape == (2, 2, 3)
        records.append(record)
        result = np.zeros(2048, np.float32);result[record["view"]] = 1
        return result
    monkeypatch.setattr(runner, "encode_one", encode_one)
    result = runner.encode_queries(tmp_path, {"query": {"planned_frame_ordinals": list(range(80))},
        "per_query_native_pixel_sha256": ["sha"] * 80}, np.arange(80.), object(), tmp_path / "queries.npy", 0, rows)
    assert result.shape == (240, 2048) and len(rows) == 240
    assert [(r["position"], r["view"]) for r in rows] == [(i, v) for i in range(80) for v in range(3)]
    assert events == ["closed"] and not (tmp_path / "frame_tmp").exists()


def _comparison(tmp_path):
    family = tmp_path / "family";target = family / "inference"
    target.mkdir(parents=True)
    new, controls = [], []
    for i, name in enumerate(runner.NEW_OUTPUTS):
        filename = f"new_{i}.json"
        write_json(target / filename, {"value": i})
        new.append({"name": name, "file": filename, "sha256": runner.sha(target / filename), "new": True})
    for i, name in enumerate(runner.CONTROL_FILES):
        filename = f"controls/old_{i}.json"
        write_json(target / filename, {"value": i})
        controls.append({"name": name, "copy": filename, "sha256": runner.sha(target / filename), "source": "original"})
    (family / "inputs_frozen.json").write_text("original")
    return family, target, new, {"controls": controls, "identity": {}}, runner.sha(family / "inputs_frozen.json")


@pytest.mark.parametrize("mutate", [False, True])
def test_master_requires_all24_and_preserves_entry_freeze_SHA(tmp_path, monkeypatch, mutate):
    family, target, new, bundle, expected = _comparison(tmp_path)
    calls = []
    def verify_prepared(root, exact, commit):
        calls.append(exact)
        if mutate:
            (root / "inputs_frozen.json").write_text("reblessed")
        runner.require(runner.sha(root / "inputs_frozen.json") == exact, "expected freeze changed")
    monkeypatch.setattr(runner, "verify_prepared", verify_prepared)
    monkeypatch.setattr(runner, "resource_gate", lambda *a: {})
    args = (family, target, {}, bundle, "pin", expected, 0, {}, {}, 0, {}, {}, [], np.zeros((1, 3, 1)), {}, new)
    if mutate:
        with pytest.raises(ValueError, match="changed"):
            runner._finalize(*args)
        assert not (target / "frozen_predictions.json").exists()
    else:
        runner._finalize(*args)
        master = json.loads((target / "frozen_predictions.json").read_text())
        assert master["new_outputs"] == 8 and master["retained_controls"] == 16
        assert len(master["all_requested_outputs"]) == 24
    assert calls == [expected]


def _mock_inference(tmp_path, monkeypatch, *, wrong_state=False, mutate=False):
    family = tmp_path / "family";family.mkdir()
    (family / "inputs_frozen.json").write_text("frozen")
    expected = runner.sha(family / "inputs_frozen.json")
    controls = []
    for i, name in enumerate(runner.CONTROL_FILES):
        relative = f"controls/old{i}.json"
        write_json(family / "inputs" / relative, {"old": i})
        controls.append({"name": name, "copy": relative, "source": "old_freeze",
                         "sha256": runner.sha(family / "inputs" / relative)})
    events, plan, state = [], {"declared": {}}, {"sha256": "accepted"}
    bundle = {"controls": controls, "assets": {"root": "assets"}, "gallery_root": family, "plan": {},
        "smoke": {"state": state, "torch": "torch", "torchvision": "vision"}, "identity": {},
        "shared": {"ref_ids": np.array(["a", "b"]), "ref_latlon": np.zeros((2, 2)),
                   "query_times": np.arange(80.), "query_indices": np.arange(80), "frame_ordinals": np.arange(80)},
        "matrices": {"megaloc": np.zeros((80, 2), np.float32), "image": np.zeros((80, 2), np.float32)}}
    def verify_prepared(root, exact, commit):
        events.append(("verified", exact))
        runner.require(runner.sha(root / "inputs_frozen.json") == exact, "expected freeze changed")
        return plan, bundle
    class Model:
        def __init__(self, root):
            events.append("constructor")
            self.state = {"sha256": "wrong"} if wrong_state else state
            self.runtime = {"torch": "torch", "torchvision": "vision", "forward_calls": 9975}
        def verify_unchanged(self): events.append("verified_model")
    def references(root, plan, model, path, started, rows):
        events.append("encoded_refs")
        values = np.zeros((2, 2048), np.float32);values[:, 0] = 1
        np.save(path, values)
    def queries(root, plan, times, model, path, started, rows):
        values = np.zeros((240, 2048), np.float32);values[:, 0] = 1
        np.save(path, values)
        rows.extend([{}] * 9975)
        return values
    def routes(output, views, bundle):
        rows = []
        for i, name in enumerate(runner.NEW_OUTPUTS):
            path = output / f"new{i}.json"
            write_json(path, {"synthetic": i})
            rows.append({"name": name, "file": path.name, "sha256": runner.sha(path), "new": True})
        if mutate:
            (family / "inputs_frozen.json").write_text("reblessed")
        return rows
    monkeypatch.setattr(runner, "verify_prepared", verify_prepared)
    monkeypatch.setattr(runner, "resource_gate", lambda *a: {})
    monkeypatch.setattr(runner.backend, "ReleasedSelaBase", Model)
    monkeypatch.setattr(runner, "encode_references", references)
    monkeypatch.setattr(runner, "encode_queries", queries)
    monkeypatch.setattr(runner, "verify_smoke_replay", lambda *a: {})
    monkeypatch.setattr(runner, "write_new_routes", routes)
    return family, expected, events


def test_mocked_end_to_end_keeps_all24_controls_byte_identical(tmp_path, monkeypatch):
    family, expected, events = _mock_inference(tmp_path, monkeypatch)
    result = runner.infer(family, "pin", expected)
    assert result["new_outputs"] == 8 and result["comparison_outputs"] == 24
    index = json.loads((family / "inference/comparison_index.json").read_text())
    assert len(index["outputs"]) == 24 and sum(row["new"] for row in index["outputs"]) == 8
    for row in index["outputs"][8:]:
        assert (family / "inference" / row["file"]).read_bytes() == (family / "inputs" / row["file"]).read_bytes()
    assert events[0] == events[-1] == ("verified", expected)


def test_model_state_mismatch_declines_before_any_new_encoding(tmp_path, monkeypatch):
    family, expected, events = _mock_inference(tmp_path, monkeypatch, wrong_state=True)
    with pytest.raises(ValueError, match="accepted smoke"):
        runner.infer(family, "pin", expected)
    assert "encoded_refs" not in events
    assert (family / "inference/failure.json").exists()
    assert not (family / "inference/frozen_predictions.json").exists()


def test_in_run_manifest_reblessing_declines_before_comparison_master(tmp_path, monkeypatch):
    family, expected, _ = _mock_inference(tmp_path, monkeypatch, mutate=True)
    with pytest.raises(ValueError, match="freeze changed"):
        runner.infer(family, "pin", expected)
    assert not (family / "inference/frozen_predictions.json").exists()


def test_cli_has_no_score_or_label_argument(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner.heading, "offline", lambda: nullcontext())
    monkeypatch.setattr(runner, "prepare", lambda **kwargs: calls.append(kwargs) or {"status": "fake"})
    runner.main(["prepare", "--spec", str(tmp_path / "spec"), "--family", str(tmp_path / "family"),
                 "--expected-commit", "pin"])
    assert calls[0]["spec_path"] == tmp_path / "spec"
    with pytest.raises(SystemExit):
        runner.main(["infer", "--family", str(tmp_path), "--expected-commit", "pin",
                     "--input-freeze-sha256", "a" * 64, "--ground-truth", "labels"])
