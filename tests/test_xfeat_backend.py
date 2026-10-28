from pathlib import Path
import hashlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
import xfeat_backend as x


def test_bgr_is_converted_to_contiguous_rgb_without_normalizing_or_mutating_storage():
    bgr = np.array([[[1, 2, 255], [4, 5, 6]]], np.uint8)
    before = bgr.copy()
    rgb = x.rgb_view(bgr)
    assert rgb.tolist() == [[[255, 2, 1], [6, 5, 4]]]
    assert rgb.flags.c_contiguous and rgb.dtype == np.uint8
    assert np.array_equal(bgr, before) and not np.shares_memory(rgb, bgr)


@pytest.mark.parametrize("pixels", [np.zeros((2, 2)), np.zeros((2, 2, 4), np.uint8),
    np.zeros((2, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8)])
def test_rgb_rejects_unsupported_native_pixels(pixels):
    with pytest.raises(ValueError):
        x.rgb_view(pixels)


@pytest.fixture
def asset_root(tmp_path, monkeypatch):
    root = tmp_path/"approved"
    (root/"weights").mkdir(parents=True)
    (root/"modules").mkdir()
    body = b"synthetic weight identity; never deserialized"
    (root/"weights/xfeat.pt").write_bytes(body)
    (root/"modules/xfeat.py").write_text("# no inference\n", encoding="utf8")
    monkeypatch.setattr(x, "WEIGHT_BYTES", len(body))
    monkeypatch.setattr(x, "WEIGHT_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(x, "WEIGHT_GIT_BLOB", hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest())
    monkeypatch.setattr(x, "CODE_BLOBS", {"modules/xfeat.py": "approved-code-oid"})
    def git(command, **kwargs):
        if "rev-parse" in command:
            return x.COMMIT+"\n"
        if "status" in command:
            return ""
        if "hash-object" in command:
            return "approved-code-oid\n"
        raise AssertionError(command)
    monkeypatch.setattr(x.subprocess, "check_output", git)
    return root


def test_asset_inspection_hashes_without_importing_torch_or_deserializing(asset_root, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "torch"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    report = x.inspect_assets(asset_root)
    assert report["weight"]["sha256"] == x.WEIGHT_SHA256
    assert report["weight"]["sha256_is_publisher_manifest"] is False
    assert report["license"] == "Apache-2.0"


@pytest.mark.parametrize("kind", ["wrong_commit", "dirty", "wrong_code", "wrong_weight", "oversize"])
def test_asset_inspection_refuses_changed_source_or_checkpoint(asset_root, monkeypatch, kind):
    old = x.subprocess.check_output
    def git(command, **kwargs):
        if kind == "wrong_commit" and "rev-parse" in command: return "wrong"
        if kind == "dirty" and "status" in command: return " M modules/xfeat.py"
        if kind == "wrong_code" and "hash-object" in command: return "wrong"
        return old(command, **kwargs)
    monkeypatch.setattr(x.subprocess, "check_output", git)
    if kind == "wrong_weight":
        p = asset_root/"weights/xfeat.pt"
        p.write_bytes(b"z"*p.stat().st_size)
    if kind == "oversize":
        with (asset_root/"weights/xfeat.pt").open("ab") as stream: stream.write(b"x")
    with pytest.raises(ValueError):
        x.inspect_assets(asset_root)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.value
    def __len__(self): return len(self.value)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim


def fake_backend():
    backend = object.__new__(x.ReleasedXFeat)
    calls = {}
    points = np.array([[1, 2], [3, 4]], np.float32)
    def parse(rgb): calls["rgb"] = rgb.copy(); return "official-parsed-tensor"
    def detect(tensor, **kwargs):
        calls["detect"] = tensor, kwargs
        return [{"keypoints": Tensor(points), "descriptors": Tensor(np.ones((2, 64), np.float32))}]
    def match(a, b, **kwargs):
        calls["match"] = kwargs
        return Tensor([1, 0]), Tensor([0, 1])
    backend.model = SimpleNamespace(parse_input=parse, detectAndCompute=detect, match=match)
    backend.torch = SimpleNamespace(isfinite=lambda tensor: np.isfinite(tensor.value))
    backend.runtime = {"feature_forward_calls": 0, "pair_match_requests": 0, "actual_mnn_calls": 0}
    return backend, calls


def test_features_use_official_parse_and_fixed_sparse_parameters():
    backend, calls = fake_backend()
    points, descriptors = backend.features(np.array([[[1, 2, 3]]], np.uint8))
    assert calls["rgb"].tolist() == [[[3, 2, 1]]]
    assert calls["detect"] == ("official-parsed-tensor", {"top_k": 2500, "detection_threshold": .05})
    assert points.shape == (2, 2) and descriptors.shape == (2, 64)


def test_correspondences_use_released_mnn_minus_one_without_orb_ratio():
    backend, calls = fake_backend()
    features = backend.features(np.zeros((2, 2, 3), np.uint8))
    pa, pb = backend.correspondences(features, features)
    assert calls["match"] == {"min_cossim": -1}
    assert pa.tolist() == [[3, 4], [1, 2]] and pb.tolist() == [[1, 2], [3, 4]]


def test_empty_features_do_not_call_released_matcher():
    backend, calls = fake_backend()
    empty = np.empty((0, 2), np.float32), Tensor(np.empty((0, 64), np.float32))
    a, b = backend.correspondences(empty, empty)
    assert a.shape == b.shape == (0, 2) and "match" not in calls


@pytest.mark.parametrize("kind", ["point_nan", "descriptor_nan", "wrong_width", "too_many"])
def test_invalid_sparse_outputs_fail_closed(kind):
    backend, _ = fake_backend()
    points = np.ones((2501 if kind == "too_many" else 2, 2), np.float32)
    descriptors = np.ones((len(points), 63 if kind == "wrong_width" else 64), np.float32)
    if kind == "point_nan": points[0, 0] = np.nan
    if kind == "descriptor_nan": descriptors[0, 0] = np.nan
    backend.model.detectAndCompute = lambda *args, **kwargs: [{"keypoints": Tensor(points), "descriptors": Tensor(descriptors)}]
    with pytest.raises(ValueError): backend.features(np.zeros((2, 2, 3), np.uint8))


@pytest.mark.parametrize("value", [None, "0", "1", "0,1"])
def test_cpu_constructor_rejects_visible_devices_before_torch_import(monkeypatch, value):
    monkeypatch.setattr(x, "inspect_assets", lambda root: {})
    if value is None: monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else: monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with pytest.raises(ValueError): x.ReleasedXFeat("unused")


def test_state_hash_changes_with_actual_numerics():
    value = Tensor(np.array([1, 2], np.float32))
    model = SimpleNamespace(state_dict=lambda: {"a": value})
    first = x._state_identity(model)
    value.value[1] = 3
    second = x._state_identity(model)
    assert first["sha256"] != second["sha256"] and first["element_count"] == 2


def test_constructor_uses_cpu_weights_only_and_dict_without_hub_or_lighterglue(tmp_path, monkeypatch):
    root = tmp_path/"upstream"
    root.mkdir()
    asset = {"root": str(root), "weight": {"path": str(root/"approved.pt")}}
    monkeypatch.setattr(x, "inspect_assets", lambda _: asset)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    calls = {}
    state = {"weight": Tensor(np.array([1, 2], np.float32))}
    def load(path, **kwargs): calls["load"] = path, kwargs; return state
    threads = {"num": 99, "interop": 99}
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
        set_num_threads=lambda n: threads.update(num=n), set_num_interop_threads=lambda n: threads.update(interop=n),
        get_num_threads=lambda: threads["num"], get_num_interop_threads=lambda: threads["interop"],
        load=load, __version__="synthetic-test")
    monkeypatch.setitem(sys.modules, "torch", torch)
    names = ("modules", "modules.xfeat", "modules.model", "modules.interpolator")
    for name in names: monkeypatch.delitem(sys.modules, name, raising=False)
    model = SimpleNamespace(dev=SimpleNamespace(type="cpu"), lighterglue=None,
        parameters=lambda: [], state_dict=lambda: state)
    model.eval = lambda: model
    def construct(**kwargs): calls["constructor"] = kwargs; return model
    def import_module(name):
        assert name == "modules.xfeat"
        for module_name in names:
            monkeypatch.setitem(sys.modules, module_name,
                SimpleNamespace(__file__=str(root/(module_name.replace(".", "/")+".py"))))
        return SimpleNamespace(XFeat=construct)
    monkeypatch.setattr(x.importlib, "import_module", import_module)
    instance = x.ReleasedXFeat(root)
    assert calls["load"][1] == {"map_location": "cpu", "weights_only": True}
    assert calls["constructor"] == {"weights": state, "top_k": 2500, "detection_threshold": .05}
    assert instance.runtime["cpu_threads"] == 2 and instance.runtime["interop_threads"] == 1
    assert instance.runtime["lighterglue_constructed"] is False
    assert instance.runtime["feature_forward_calls"] == instance.runtime["actual_mnn_calls"] == 0


def test_state_identity_handles_scalar_batchnorm_buffers():
    model = SimpleNamespace(state_dict=lambda: {"num_batches_tracked": Tensor(np.array(0, np.int64))})
    assert x._state_identity(model)["element_count"] == 1
