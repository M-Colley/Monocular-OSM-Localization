"""Isolated input copying, sparse evidence freezing and bounded failures."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import pilot_sparse_city_replay as runner
from scripts import plan_sparse_city_replay as planner
from tests.test_plan_sparse_city_replay import setup as archived_case


def prepared_case(tmp_path, monkeypatch):
    result, archive, video, plan_dir, registration = archived_case(tmp_path, monkeypatch)
    owned_video = tmp_path / "original_video/video.mp4"
    owned_video.parent.mkdir(); video.rename(owned_video); video = owned_video
    for path in archive.rglob("ref_emb*.npz"):
        with np.load(path) as data: fp = str(data["fingerprint"])
        np.savez(path, fingerprint=np.array(fp), emb=np.eye(4, dtype=np.float32)[:3])
    query = {"video": {"path": str(video), "sha256": planner.sha(video)}, "query_mode": "crops",
        "stride": 1, "fps": 10., "dense_count": 80, "sample_count": 80, "native_shape_hw": [2, 3],
        "registered_window": [0., 8.], "query_dense_indices": list(range(80)), "planned_frame_ordinals": list(range(80))}
    monkeypatch.setattr(planner, "video_grid", lambda *args: deepcopy(query))
    planner.plan(result, archive, video, plan_dir)
    models = tmp_path / "original_models"
    asset = models / "torch/hub/model.bin"; asset.parent.mkdir(parents=True); asset.write_bytes(b"synthetic cached model")
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"megaloc": {"source": str(models), "files": [
        {"relative": "torch/hub/model.bin", "sha256": planner.sha(asset), "bytes": asset.stat().st_size}]}}))
    family = tmp_path / "isolated_family"
    lineage = runner.prepare(plan_dir / "replay_plan.json", manifest, family)
    return family, lineage


def test_prepare_copies_only_verified_inputs_and_preserves_originals(tmp_path, monkeypatch):
    family, lineage = prepared_case(tmp_path, monkeypatch)
    assert len(lineage["pools"]) == 4 and lineage["models_have_not_run"]
    runner.verify_freeze(family / "inputs_frozen.json")
    runner.verify_originals(lineage["copied"])
    assert not list(family.rglob("ref_img_store.npz")) and not (family / "inference").exists()
    assert all(row["source"] != row["path"] for row in lineage["copied"])


def test_streamed_union_uses_all_pools_and_stable_deduplicated_ids(tmp_path, monkeypatch):
    family, lineage = prepared_case(tmp_path, monkeypatch)
    plan = runner.load(family / "inputs/replay_plan.json")
    union, ids, coords, rows = runner.load_union(plan, lineage)
    np.testing.assert_array_equal(union, np.eye(4, dtype=np.float32)[:3])
    assert ids.tolist() == plan["union"]["ref_ids"] and len(rows) == 4
    assert [r["added"] for r in rows] == [3, 0, 0, 0]
    assert runner._array_sha(coords) == plan["union"]["ref_latlon_sha256"]


def test_existing_family_is_never_reblessed(tmp_path, monkeypatch):
    family, lineage = prepared_case(tmp_path, monkeypatch)
    original = (family / "inputs_frozen.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        runner.prepare(family / "inputs/replay_plan.json", family / "inputs/model_asset_manifest.json", family)
    assert (family / "inputs_frozen.json").read_bytes() == original


def test_commit_headroom_is_verified_before_any_heavy_stage(monkeypatch):
    memory = {"available_physical_bytes": runner.MIN_COMMIT, "available_commit_bytes": runner.MIN_COMMIT - 1}
    monkeypatch.setattr(runner, "memory_snapshot", lambda: memory.copy())
    with pytest.raises(ValueError, match="8GiB"): runner.check_headroom()
    memory["available_commit_bytes"] = runner.MIN_COMMIT
    assert runner.check_headroom() == memory


@pytest.mark.parametrize("key", ["available_physical_bytes", "available_commit_bytes"])
@pytest.mark.parametrize("value", [None, -1, float("inf"), runner.MIN_COMMIT - 1])
def test_either_unknown_or_low_budget_refuses_heavy_work(monkeypatch, key, value):
    memory = {"available_physical_bytes": runner.MIN_COMMIT, "available_commit_bytes": runner.MIN_COMMIT}
    memory[key] = value
    monkeypatch.setattr(runner, "memory_snapshot", lambda: memory)
    with pytest.raises(ValueError, match="physical AND actual commit"): runner.check_headroom()


@pytest.mark.parametrize("fault", ["duplicate_vector", "norm", "nan", "row_order", "missing_id", "pool_order", "missing_pool", "union_budget"])
def test_inconsistent_or_partial_union_is_never_silently_selected(tmp_path, monkeypatch, fault):
    family, lineage = prepared_case(tmp_path, monkeypatch)
    plan = runner.load(family / "inputs/replay_plan.json")
    if fault in {"duplicate_vector", "norm", "nan"}:
        path = Path(lineage["pools"][1]["embeddings"])
        with np.load(path) as z: vector, fp = z["emb"], str(z["fingerprint"])
        if fault == "duplicate_vector": vector[0] = vector[1]
        elif fault == "norm": vector[0] *= 2
        else: vector[0, 0] = np.nan
        np.savez(path, emb=vector, fingerprint=np.array(fp))
        plan["pools"][1]["embeddings"].update(sha256=runner.sha(path), header=runner.header(path, "emb"))
    elif fault == "row_order":
        p = plan["pools"][0]
        p["ref_ids"] = p["ref_ids"][::-1]; p["ref_latlon"] = p["ref_latlon"][::-1]
        p["ref_ids_sha256"] = runner._json_sha(p["ref_ids"])
        p["ref_latlon_sha256"] = runner._array_sha(np.asarray(p["ref_latlon"], float))
    elif fault == "missing_id": plan["pools"][0]["ref_ids"].pop()
    elif fault == "pool_order": lineage["pools"][0]["index"] = 1
    elif fault == "missing_pool": lineage["pools"].pop()
    else: monkeypatch.setattr(runner, "MAX_UNION_BYTES", 1)
    with pytest.raises(ValueError): runner.load_union(plan, lineage)


def test_changed_prepared_input_fails_the_original_freeze(tmp_path, monkeypatch):
    family, lineage = prepared_case(tmp_path, monkeypatch)
    Path(lineage["pools"][0]["embeddings"]).write_bytes(b"changed prepared descriptor")
    with pytest.raises(ValueError): runner.verify_freeze(family / "inputs_frozen.json")


def test_frozen_mock_inference_retains_complete_matrix_and_both_outputs(tmp_path, monkeypatch):
    import monocular_osm.frame_extraction as extraction
    family, lineage = prepared_case(tmp_path, monkeypatch)
    closed = []
    frames = SimpleNamespace(frames=[np.full((2, 3, 3), i, np.uint8) for i in range(80)],
        timestamps=np.arange(80.) / 10, fps=10., sampling={"dense_count": 80, "dense_indices": list(range(80))},
        close=lambda: closed.append(True))
    monkeypatch.setattr(runner, "source_identity", lambda value: value)
    monkeypatch.setattr(runner, "check_headroom", lambda: runner.MIN_COMMIT)
    monkeypatch.setattr(extraction, "extract_frames", lambda *a, **k: frames)
    monkeypatch.setattr(runner, "verify_query_pixels", lambda *a: ["a" * 64] * 80)
    scores = np.full((80, 3), .1, np.float32)
    scores[np.arange(80), np.arange(80) % 3] = .8
    monkeypatch.setattr(runner, "forward_queries", lambda *a: (scores, np.zeros((240, 4), np.float32), {"mock": True}))
    result = runner.infer(family, "mocked-clean-source")
    assert result["status"] == "complete" and closed
    freeze = runner.verify_freeze(family / "inference/frozen_predictions.json")
    assert freeze["variants"] == ["raw", "reference_mean"] and freeze["selected_variant"] is None
    with np.load(family / "inference/full_evidence.npz") as z:
        np.testing.assert_array_equal(z["raw_cosines"], scores)
        assert z["ref_ids"].shape == (3,) and z["frame_ordinals"].tolist() == list(range(80))
    assert all((family / "inference" / f"prediction_{name}.json").exists() for name in freeze["variants"])
    with pytest.raises(ValueError, match="overwrite"): runner.infer(family, "mocked-clean-source")


@pytest.mark.parametrize("fault", [None, "pixels", "pts", "ordinal", "seek"])
def test_every_native_query_ordinal_pixels_and_pts_are_independently_verified(monkeypatch, fault):
    import cv2
    ordinals = [2, 5, 9]
    images = [np.full((2, 3, 3), i, np.uint8) for i in ordinals]
    frames = SimpleNamespace(frames=images, timestamps=np.asarray(ordinals, float) / 10)
    released = []
    class Capture:
        current = 0
        def isOpened(self): return True
        def set(self, key, value): self.current = value; return fault != "seek"
        def grab(self): self.current += 1; return True
        def read(self):
            image = np.full((2, 3, 3), self.current, np.uint8); self.current += 1
            return True, image + (1 if fault == "pixels" else 0)
        def get(self, key):
            if key == cv2.CAP_PROP_POS_FRAMES: return self.current + (1 if fault == "ordinal" else 0)
            return (self.current - 1) * 100 + (1 if fault == "pts" else 0)
        def release(self): released.append(True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    if fault:
        with pytest.raises(ValueError): runner.verify_query_pixels(frames, "mock.mp4", {"planned_frame_ordinals": ordinals, "fps": 10.})
    else:
        assert runner.verify_query_pixels(frames, "mock.mp4", {"planned_frame_ordinals": ordinals, "fps": 10.}) == [runner._array_sha(im) for im in images]
    assert released


def test_offline_guard_refuses_connections_and_restores_socket_apis():
    import socket
    original = socket.socket.connect
    with runner.offline(), socket.socket() as sock:
        with pytest.raises(RuntimeError, match="offline"): sock.connect(("127.0.0.1", 1))
    assert socket.socket.connect is original


def test_infer_resource_refusal_precedes_union_frames_and_model(tmp_path, monkeypatch):
    family, lineage = prepared_case(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "source_identity", lambda value: value)
    monkeypatch.setattr(runner, "check_headroom", lambda: (_ for _ in ()).throw(ValueError("8GiB unavailable")))
    monkeypatch.setattr(runner, "load_union", lambda *a: pytest.fail("must not allocate"))
    monkeypatch.setattr(runner, "forward_queries", lambda *a: pytest.fail("must not load model"))
    with pytest.raises(ValueError, match="8GiB"): runner.infer(family, "mock-source")
    assert not (family / "inference").exists()


def test_query_adapter_uses_one_fresh_three_crop_cache_and_actual_cpu_bounds(tmp_path, monkeypatch):
    import sys
    from monocular_osm import kartaview_vpr as kv
    state, calls = {"intra": 2, "inter": 8}, []
    fake_torch = SimpleNamespace(__version__="mock", get_num_threads=lambda: state["intra"],
        set_num_threads=lambda v: state.update(intra=v), get_num_interop_threads=lambda: state["inter"],
        set_num_interop_threads=lambda v: state.update(inter=v), from_numpy=lambda a: a,
        hub=SimpleNamespace(set_dir=lambda p: calls.append(("hub", p))))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "configure_environment", lambda: None)
    monkeypatch.setattr(runner, "check_headroom", lambda: runner.MIN_COMMIT)
    for key in ("TORCH_HOME", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(key, "old")
    monkeypatch.setattr(kv, "_resolve_backbone", lambda *a: "megaloc")
    monkeypatch.setattr(kv, "backbone_info", lambda: {"name": "megaloc", "hub": kv.HUB_MEGALOC, "device": "cpu"})
    monkeypatch.setattr(kv, "_audit_model_state", lambda: {"sha256": "a" * 64})
    def query(frames, indices, refs, device, **kwargs):
        assert kwargs["mode"] == "crops" and kwargs["batch_size"] == 4 and device == "cpu"
        assert indices.tolist() == list(range(80)) and state == {"intra": 4, "inter": 1}
        kwargs["query_cache"]["only"] = SimpleNamespace(numpy=lambda: np.ones((240, 4), np.float32))
        calls.append(("query", len(refs)))
        return np.full((80, len(refs)), .2, np.float32)
    monkeypatch.setattr(kv, "_query_similarities", query)
    scores, descriptors, proof = runner.forward_queries(SimpleNamespace(frames=[None] * 80), np.eye(4, dtype=np.float32)[:3], tmp_path)
    assert scores.shape == (80, 3) and descriptors.shape == (240, 4)
    assert [row for row in calls if row[0] == "query"] == [("query", 3)]
    assert proof["cpu_runtime"]["intra_op_threads"] == 4 and proof["cpu_runtime"]["inter_op_threads"] == 1
    assert state["intra"] == 2 and not proof["network_allowed"]


@pytest.mark.parametrize("dirty", [False, True])
def test_source_pin_mismatch_or_dirty_tree_refuses_inference_before_models(monkeypatch, dirty):
    def git(command, **kwargs):
        return SimpleNamespace(stdout=("expected" if dirty else "different") if command[1] == "rev-parse" else " M file.py")
    monkeypatch.setattr(runner.subprocess, "run", git)
    with pytest.raises(ValueError, match="clean committed source"): runner.source_identity("expected")
