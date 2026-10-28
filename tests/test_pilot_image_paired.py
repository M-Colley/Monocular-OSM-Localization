"""Bounded mocked paired experiment; no model constructors/downloads/labels."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import pilot_image_paired as p


NETWORK = '''import torch
from transformers import ViTModel
from aggregators.netvlad import NetVLAD
from aggregators.salad import SALAD
from aggregators.boq import BoQ
class VPRmodel:
    def forward(self, x): return x + 1
def get_backbone(args): return args
def get_aggregator(args): return None
'''


def setup(tmp_path, monkeypatch):
    parent = tmp_path / "original"; (parent / "references").mkdir(parents=True)
    official = tmp_path / "official"; (official / "backbone").mkdir(parents=True)
    (official / "network.py").write_text(NETWORK)
    (official / "backbone/vision_transformer.py").write_text("# copied source\n")
    checkpoint = tmp_path / "weights/model.pth"; checkpoint.parent.mkdir(); checkpoint.write_bytes(b"mock weights")
    monkeypatch.setattr(p, "CHECKPOINT_SHA", p.sha(checkpoint)); monkeypatch.setattr(p, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    receipt = checkpoint.parent / "receipt.json"
    p.write(receipt, {"actual_sha256": p.CHECKPOINT_SHA, "source_pin": p.OFFICIAL_PIN, "official_asset_digest": "sha256:" + p.CHECKPOINT_SHA})
    monkeypatch.setattr(p, "git", lambda root, *args: p.OFFICIAL_PIN if args[0] == "rev-parse" else
        "network.py\nbackbone/vision_transformer.py" if args[0] == "ls-files" else "")
    video = parent / "video.mp4"; video.write_bytes(b"encoded native video fixture")
    query = {"timestamps": np.arange(80, dtype=float).tolist(), "fps": 1.,
        "sampling": {"sample_count": 80, "mode": "linspace", "dense_indices": list(range(80))},
        "native_frame_shape": [2, 4, 3], "pixels_sha256": "synthetic"}
    experiment = {"profile": {"model": "megaloc", "query_mode": "crops"}, "runtime": {"query_stride": 1},
        "video": {"path": str(video), "sha256": p.sha(video)}, "window": [0., 80.]}
    p.write(parent / "experiment.json", experiment); p.write(parent / "query_input.json", query); p.write(parent / "plan.json", {})
    xy = np.array([[42., -83.], [42.001, -83.001]])
    np.savez(parent / "references/ref_img_store.npz", ids=np.array(["b", "a"]), xy=xy[::-1],
        raw=np.stack([np.ones((512, 512, 3), np.uint8), np.zeros((512, 512, 3), np.uint8)]))
    p.write(parent / "reference_snapshot.json", [{"relative": "references/ref_img_store.npz", "sha256": p.sha(parent / "references/ref_img_store.npz")}])
    np.savez(parent / "reference_evidence.npz", query_indices=np.arange(80), query_times=np.arange(80, dtype=float),
        ref_ids=np.array(["mapillary:a", "mapillary:b"]), ref_latlon=xy, similarities=np.full((80, 2), .2, np.float32),
        metadata=np.array(json.dumps({"query_mode": "crops", "preprocessing": {"reference_color": "RGB"}, "reference_encoding_identity": "unknown"})))
    p.write(parent / "frozen_prediction.json", {"ground_truth_opened": False, "prediction_available": True,
        "artifacts": {name: p.sha(parent / name) for name in p.PARENT_FILES}})
    family = tmp_path / "isolated"
    return parent, official, checkpoint, receipt, family


def prepared(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch); p.prepare(*args)
    return args[-1]


def test_prepare_is_copy_only_full_pool_mmap_and_originals_unchanged(tmp_path, monkeypatch):
    family = prepared(tmp_path, monkeypatch)
    p.verify_freeze(family / "inputs_frozen.json"); p.verify_originals(p.load(family / "input_lineage.json")["copied"])
    raw = np.load(family / "inputs/raw.npy", mmap_mode="r")
    assert isinstance(raw, np.memmap) and raw.shape == (2, 512, 512, 3)
    values, order, _ = p.paired_inputs(family)
    assert order.tolist() == [1, 0] and values["similarities"].shape == (80, 2)
    assert not (family / "inference").exists() and not list(family.rglob("scores.json"))


@pytest.mark.parametrize("change", ["video", "evidence", "checkpoint", "official_dirty", "parent_labels"])
def test_prepare_refuses_stale_or_untrusted_sources_before_copy(tmp_path, monkeypatch, change):
    args = setup(tmp_path, monkeypatch); parent, official, checkpoint, receipt, family = args
    if change == "video": (parent / "video.mp4").write_bytes(b"changed")
    if change == "evidence": (parent / "reference_evidence.npz").write_bytes(b"changed")
    if change == "checkpoint": checkpoint.write_bytes(b"changed")
    if change == "official_dirty": monkeypatch.setattr(p, "git", lambda root, *args: p.OFFICIAL_PIN if args[0] == "rev-parse" else "dirty")
    if change == "parent_labels":
        freeze = p.load(parent / "frozen_prediction.json"); freeze["ground_truth_opened"] = True; p.write(parent / "frozen_prediction.json", freeze)
    with pytest.raises(ValueError): p.prepare(*args)
    assert not family.exists()


def test_no_reblessing_existing_family(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch); p.prepare(*args); before = (args[-1] / "inputs_frozen.json").read_bytes()
    with pytest.raises(ValueError): p.prepare(*args)
    assert (args[-1] / "inputs_frozen.json").read_bytes() == before


def test_raw_member_budget_checked_before_decompression(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "MAX_RAW_BYTES", 10)
    archive = tmp_path / "pixels.npz"; np.savez(archive, raw=np.zeros((1, 512, 512, 3), np.uint8))
    with pytest.raises(ValueError, match="budget"): p.extract_raw(archive, tmp_path / "raw.npy")
    assert not (tmp_path / "raw.npy").exists()


@pytest.mark.parametrize("change", ["coordinate", "duplicate_id", "omitted_column", "nan", "query_grid", "color"])
def test_paired_inventory_rejects_incomplete_or_misaligned_inputs(tmp_path, monkeypatch, change):
    family = prepared(tmp_path, monkeypatch)
    path = family / "inputs/reference_evidence.npz"
    with np.load(path) as z: data = dict(z)
    if change == "coordinate": data["ref_latlon"][0, 0] += .1
    if change == "duplicate_id": data["ref_ids"][1] = data["ref_ids"][0]
    if change == "omitted_column": data["similarities"] = data["similarities"][:, :1]
    if change == "nan": data["similarities"][0, 0] = np.nan
    if change == "query_grid": data["query_indices"][0] = 1
    if change == "color": data["metadata"] = np.array(json.dumps({"query_mode": "crops", "preprocessing": {"reference_color": "BGR"}}))
    np.savez(path, **data)
    with pytest.raises(ValueError): p.paired_inputs(family)


def test_ast_removal_preserves_official_forward_and_backbone_exactly():
    tree, proof = p.filtered_network(NETWORK)
    assert not any(isinstance(n, ast.ImportFrom) for n in tree.body)
    original = ast.parse(NETWORK)
    for name in ("VPRmodel", "get_backbone", "get_aggregator"):
        a = next(n for n in original.body if getattr(n, "name", None) == name)
        b = next(n for n in tree.body if getattr(n, "name", None) == name)
        assert ast.dump(a) == ast.dump(b) and name in proof["official_ast_sha256"]


@pytest.mark.parametrize("text", [NETWORK.replace("import ViTModel", "import ViTModel as Alias"),
    NETWORK.replace("from aggregators.boq import BoQ", ""), NETWORK.replace("def get_backbone", "def wrong_backbone")])
def test_ast_adapter_refuses_unreviewed_import_shape_or_missing_core(text):
    with pytest.raises(ValueError): p.filtered_network(text)


@pytest.mark.parametrize("state", [{"module.a": 1, "b": 2}, {"module.module.a": 1}, {}, {1: 1}])
def test_module_prefix_refuses_ambiguous_or_invalid_state(state):
    with pytest.raises(ValueError): p.strip_module_prefix(state)


def test_module_prefix_only_removes_one_uniform_prefix():
    assert p.strip_module_prefix({"module.a": 1, "module.b": 2}) == {"a": 1, "b": 2}
    assert p.strip_module_prefix({"a": 1}) == {"a": 1}


def test_checkpoint_is_weights_only_with_narrow_numpy_allowlist():
    calls = []
    @contextmanager
    def safe(values): calls.append(values); yield
    def read(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return {"model_state_dict": {"module.a": 1}}
    torch = SimpleNamespace(load=read, serialization=SimpleNamespace(safe_globals=safe,
        get_unsafe_globals_in_checkpoint=lambda path: ["numpy.core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype"]))
    state, unsafe = p.checkpoint_state(torch, "fixture")
    assert state == {"a": 1} and len(calls) == 1 and "numpy.dtype" in unsafe
    torch.serialization.get_unsafe_globals_in_checkpoint = lambda path: ["unreviewed.executable"]
    torch.load = lambda *a, **k: pytest.fail("must refuse before loading")
    with pytest.raises(ValueError, match="unaudited"): p.checkpoint_state(torch, "fixture")


def test_both_bgr_reference_and_three_query_views_convert_to_rgb():
    frame = np.arange(2 * 6 * 3, dtype=np.uint8).reshape(2, 6, 3)
    query = p.Views([frame], query=True)
    assert len(query) == 3
    for i, offset in enumerate((0, 2, 4)): np.testing.assert_array_equal(query[i], frame[:, offset:offset+2, ::-1])
    refs = p.Views([frame, frame + 1], order=np.array([1, 0]))
    np.testing.assert_array_equal(refs[0], (frame + 1)[:, :, ::-1])


def test_actual_reference_writer_bgr_store_matches_query_rgb_channels(tmp_path, monkeypatch):
    """Metadata reference_color describes _prep output; persisted raw stays BGR."""
    import cv2
    import requests
    from monocular_osm import kartaview_vpr as kv
    bgr = np.empty((16, 16, 3), np.uint8)
    bgr[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] + 3
    bgr[:, :, 1] = 83
    bgr[:, :, 2] = np.arange(16, dtype=np.uint8)[:, None] + 211
    success, encoded = cv2.imencode(".png", bgr)
    assert success
    monkeypatch.setattr(requests, "Session", lambda: SimpleNamespace(get=lambda *a, **k:
        SimpleNamespace(status_code=200, content=encoded.tobytes())))
    refs = [{"id": "asymmetric", "lat": 42., "lon": -83., "url": "https://fixture.invalid/ref.png"}]
    raw, xy, fingerprint = kv._load_ref_images(refs, str(tmp_path))
    expected_bgr = cv2.resize(bgr, (512, 512))
    np.testing.assert_array_equal(raw[0], expected_bgr)
    with np.load(tmp_path / "ref_img_store.npz", allow_pickle=False) as saved:
        persisted_bgr = saved["raw"]
    np.testing.assert_array_equal(persisted_bgr[0], expected_bgr)
    rgb = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
    assert not np.array_equal(rgb, expected_bgr)
    np.testing.assert_array_equal(p.Views(persisted_bgr)[0], rgb)
    for view in p.Views([expected_bgr], query=True): np.testing.assert_array_equal(view, rgb)
    assert p.PROFILE["image"]["cached_reference_storage_color"] == "BGR"
    assert p.PROFILE["image"]["network_input_color"] == "RGB"


def test_canonical_preprocessing_order_and_explicit_official_defaults(monkeypatch):
    calls = []
    def tensor(image): calls.append(("tensor", image.mode)); return "tensor"
    def normalize(value, mean, std): calls.append(("normalize", value, mean, std)); return "normalized"
    def resize(value, size, **kwargs): calls.append(("resize", value, size, kwargs)); return "resized"
    fake = SimpleNamespace(to_tensor=tensor, normalize=normalize, resize=resize)
    import sys
    transforms = SimpleNamespace(functional=fake, InterpolationMode=SimpleNamespace(BILINEAR="bilinear"))
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace(transforms=transforms))
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional", fake)
    assert p.canonical_tensor(np.zeros((2, 3, 3), np.uint8)) == "resized"
    assert [x[0] for x in calls] == ["tensor", "normalize", "resize"]
    assert calls[-1] == ("resize", "normalized", [322, 322], {"interpolation": "bilinear", "antialias": True})


def test_encode_bounds_batches_and_refuses_corrupt_descriptor(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(p, "DIMENSION", 3)
    torch = SimpleNamespace(inference_mode=nullcontext, stack=np.stack)
    batches = []
    class Output:
        def __init__(self, a): self.a = a
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a
    def model(a):
        batches.append(len(a)); return Output(np.tile(np.array([[1, 0, 0]], np.float32), (len(a), 1)))
    result = p.encode(list(range(10)), model, torch, transform=lambda x: np.array([x]))
    assert batches == [4, 4, 2] and result.shape == (10, 3)
    with pytest.raises(ValueError):
        p.encode([1], lambda a: Output(np.zeros((len(a), 3), np.float32)), torch, transform=np.array)


def test_both_models_and_normalizations_use_identical_pinned_decoder():
    old = np.tile(np.array([[.2, .1]], np.float32), (80, 1)); new = old[:, ::-1].copy()
    ids, xy, times = np.array(["b", "a"]), np.array([[42., -83.], [42.001, -83.001]]), np.arange(80, dtype=float)
    got = p.paired_predictions(old, new, ids, xy, times)
    assert list(got["outputs"]) == ["megaloc_raw", "megaloc_reference_mean", "image_raw", "image_reference_mean"]
    assert got["selected_variant"] is None and got["profile_sha256"] == p.PROFILE_HASH
    for model, matrix in (("megaloc", old), ("image", new)):
        expected = p.decode_sparse_tracks(matrix, ids, xy, times)
        for variant in ("raw", "reference_mean"): assert got["outputs"][model + "_" + variant] == expected["outputs"][variant]
    with pytest.raises(ValueError): p.paired_predictions(old[:, :1], new, ids, xy, times)


def mocked_infer(tmp_path, monkeypatch, *, corrupt=False):
    family = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(p, "source_identity", lambda expected: expected)
    monkeypatch.setattr(p, "check_headroom", lambda: {"available_physical_bytes": 8*1024**3, "available_commit_bytes": 8*1024**3})
    closed = []
    frames = SimpleNamespace(frames=[np.zeros((2, 4, 3), np.uint8)] * 80, timestamps=np.arange(80, dtype=float), close=lambda: closed.append(True))
    monkeypatch.setattr(p, "native_queries", lambda f, plan: (frames, {"frame_ordinals": list(range(80)), "per_query_native_pixel_sha256": ["mock"]*80}))
    def forward(f, raw, order, fr):
        assert isinstance(raw, np.memmap) and order.tolist() == [1, 0]
        if corrupt: (f / "inputs/video.mp4").write_bytes(b"changed during inference")
        return np.full((80, 2), .3, np.float32), np.zeros((2, 3), np.float32), np.zeros((240, 3), np.float32), {"model": "mock"}
    monkeypatch.setattr(p, "forward_pair", forward)
    return family, closed


def test_mocked_inference_freezes_all_four_full_predictions_and_evidence(tmp_path, monkeypatch):
    family, closed = mocked_infer(tmp_path, monkeypatch)
    result = p.infer(family, "clean-pin")
    assert result["status"] == "complete" and closed == [True]
    freeze = p.verify_freeze(family / "inference/frozen_predictions.json")
    assert len(freeze["variants"]) == 4 and freeze["selected_variant"] is None
    with np.load(family / "inference/full_evidence.npz") as z:
        assert z["image_raw_cosines"].shape == z["megaloc_raw_cosines"].shape == (80, 2)
    assert len(list((family / "inference").glob("prediction_*.json"))) == 4
    with pytest.raises(ValueError, match="overwrite"): p.infer(family, "clean-pin")


def test_post_forward_tamper_preserves_failure_but_no_success_freeze(tmp_path, monkeypatch):
    family, closed = mocked_infer(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="changed"): p.infer(family, "clean-pin")
    assert closed == [True] and (family / "inference/failure.json").exists()
    assert not (family / "inference/frozen_predictions.json").exists()


def test_memory_refusal_happens_before_native_extraction_or_forward(tmp_path, monkeypatch):
    family = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(p, "source_identity", lambda expected: expected)
    monkeypatch.setattr(p, "check_headroom", lambda: p.require(False, "low actual memory"))
    monkeypatch.setattr(p, "native_queries", lambda *args: pytest.fail("must not extract"))
    monkeypatch.setattr(p, "forward_pair", lambda *args: pytest.fail("must not construct"))
    with pytest.raises(ValueError, match="low actual memory"): p.infer(family, "clean-pin")
    assert (family / "inference/failure.json").exists()


@pytest.mark.parametrize("change", [None, "pixel", "timestamp", "sampling", "shape"])
def test_native_grid_and_pixels_bound_to_original_control(tmp_path, monkeypatch, change):
    from monocular_osm import frame_extraction
    import hashlib
    images = [np.zeros((2, 4, 3), np.uint8) + i for i in range(80)]
    sampling = {"mode": "linspace", "dense_indices": list(range(80)), "sample_count": 80}
    times = np.arange(80) * .31 + 2.
    digest = hashlib.sha256()
    for image in images:
        digest.update(str((image.shape, image.dtype.str)).encode()); digest.update(memoryview(image).cast("B"))
    plan = {"window": [2., 27.], "stride": 3, "query": {"sampling": deepcopy(sampling), "fps": 10.,
        "timestamps": times.tolist(), "native_frame_shape": [2, 4, 3], "pixels_sha256": digest.hexdigest()}}
    closed = []
    frames = SimpleNamespace(frames=images, fps=10., sampling=sampling, timestamps=times.copy(), close=lambda: closed.append(True))
    if change == "pixel": images[0][0, 0, 0] += 1
    if change == "timestamp": frames.timestamps[3] += .01
    if change == "sampling": frames.sampling["dense_indices"][0] = 1
    if change == "shape": images[0] = np.zeros((2, 3, 3), np.uint8)
    monkeypatch.setattr(frame_extraction, "extract_frames", lambda *args, **kwargs: frames)
    planned = []
    monkeypatch.setattr(p, "verify_query_pixels", lambda frames, video, query: planned.extend(query["planned_frame_ordinals"]) or ["verified"]*80)
    if change:
        with pytest.raises(ValueError): p.native_queries(tmp_path, plan)
        assert closed == [True] and not planned
    else:
        _, proof = p.native_queries(tmp_path, plan)
        assert planned == [20 + 3*i for i in range(80)]
        assert proof["query_pixels_sha256"] == digest.hexdigest()
        assert proof["query_times_sha256"] == p._array_sha(times)
        assert not closed


def test_original_input_mutation_refuses_success_freeze(tmp_path, monkeypatch):
    family, closed = mocked_infer(tmp_path, monkeypatch)
    forward = p.forward_pair
    def mutate(*args):
        result = forward(*args)
        original = p.load(family / "input_lineage.json")["copied"][0]["source"]
        Path(original).write_bytes(b"original changed while running")
        return result
    monkeypatch.setattr(p, "forward_pair", mutate)
    with pytest.raises(ValueError, match="original"): p.infer(family, "clean-pin")
    assert closed == [True] and not (family / "inference/frozen_predictions.json").exists()


def test_profile_cannot_be_reblessed_with_different_decoder(tmp_path, monkeypatch):
    family = prepared(tmp_path, monkeypatch)
    plan = p.load(family / "plan.json"); plan["profile"]["decoder_profile_sha256"] = "changed"; p.write(family / "plan.json", plan)
    freeze = p.load(family / "inputs_frozen.json"); freeze["artifacts"]["plan.json"] = p.sha(family / "plan.json"); p.write(family / "inputs_frozen.json", freeze)
    monkeypatch.setattr(p, "source_identity", lambda expected: expected)
    with pytest.raises(ValueError, match="profile changed"): p.infer(family, "clean-pin")
    assert not (family / "inference").exists()
