"""Static/synthetic SelaVPR++ adapter tests: no real model or Torch import."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest

from scripts import selavpr_backend as backend


def test_module_has_no_torch_or_upstream_import():
    path = Path(backend.__file__).resolve()
    code = ("import sys; from scripts import selavpr_backend; "
            "assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules; "
            "assert not any(n == 'model' or n.startswith('model.') for n in sys.modules)")
    subprocess.run([sys.executable, "-c", code], cwd=path.parents[1], check=True)


def test_profile_is_fixed_and_hashed():
    assert backend.PROFILE_SHA256 == hashlib.sha256(json.dumps(backend.PROFILE,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert backend.PROFILE["dimensions"] == 2048
    assert backend.PROFILE["resume"] is True
    assert backend.PROFILE["hashing"] is backend.PROFILE["rerank"] is False
    assert backend.PROFILE["batch_size"] == 1
    assert backend.PROFILE["resize_hw"] == [322, 322]
    assert backend.PROFILE["published_github_asset_digest"] == "sha256:" + backend.WEIGHT_SHA256


def test_unused_faiss_removal_preserves_all_other_nodes():
    text = "import math\nimport faiss\nx=3\nclass GeM: pass\nclass Flatten: pass\ndef f(): return x\n"
    tree, record = backend.filtered_aggregation(text)
    expected = ast.parse(text)
    del expected.body[1]
    assert ast.dump(tree) == ast.dump(expected)
    assert record["removed_imports"] == ["faiss"]
    assert set(record["unchanged_definitions_sha256"]) == {"GeM", "Flatten", "f"}


@pytest.mark.parametrize("text", [
    "class GeM: pass\nclass Flatten: pass", "import faiss, math\nclass GeM: pass\nclass Flatten: pass",
    "import faiss as f\nclass GeM: pass\nclass Flatten: pass",
    "import faiss\nx=faiss.IndexFlatL2(2)\nclass GeM: pass\nclass Flatten: pass",
    "import faiss\nimport faiss\nclass GeM: pass\nclass Flatten: pass",
    "import faiss\nclass Something: pass",
])
def test_optional_import_filter_refuses_source_drift(text):
    with pytest.raises(ValueError):
        backend.filtered_aggregation(text)


def test_asymmetric_bgr_to_rgb_is_contiguous_and_does_not_mutate():
    bgr = np.array([[[3, 57, 201], [11, 17, 23]]], dtype=np.uint8)
    old = bgr.copy()
    rgb = backend.rgb_pixels(bgr)
    np.testing.assert_array_equal(rgb, [[[201, 57, 3], [23, 17, 11]]])
    np.testing.assert_array_equal(bgr, old)
    assert rgb.flags.c_contiguous and not np.shares_memory(rgb, bgr)


@pytest.mark.parametrize("pixels", [np.zeros((0, 3, 3), np.uint8), np.zeros((2, 3), np.uint8),
    np.zeros((2, 3, 4), np.uint8), np.zeros((2, 3, 3), np.float32)])
def test_bad_pixels_fail(pixels):
    with pytest.raises(ValueError):
        backend.rgb_pixels(pixels)


@pytest.mark.parametrize("state,expected", [({"module.a": 3, "module.b": 4}, {"a": 3, "b": 4}),
    ({"a": 3}, {"a": 3})])
def test_strict_prefix_removal(state, expected):
    assert backend.strip_module_prefix(state) == expected


@pytest.mark.parametrize("state", [{}, {"module.a": 1, "b": 2}, {"module.module.a": 1},
    {"module.": 1}, {1: 2}, []])
def test_ambiguous_prefixes_fail(state):
    with pytest.raises(ValueError):
        backend.strip_module_prefix(state)


def test_namespace_and_child_origin_validation(tmp_path):
    (tmp_path / "model").mkdir()
    child = tmp_path / "model/network.py"
    child.write_text("", encoding="utf8")
    modules = {"model": SimpleNamespace(__path__=[str(child.parent)]),
               "model.network": SimpleNamespace(__file__=str(child))}
    assert set(backend.source_origins(tmp_path, modules)) == set(modules)
    modules["model.foreign"] = SimpleNamespace(__file__=__file__)
    with pytest.raises(ValueError, match="foreign"):
        backend.source_origins(tmp_path, modules)


def test_unknown_namespace_origin_fails(tmp_path):
    with pytest.raises(ValueError, match="foreign"):
        backend.source_origins(tmp_path, {"model": SimpleNamespace()})


def test_offline_import_context_restores_state():
    import socket
    old = socket.socket.connect, socket.create_connection, sys.dont_write_bytecode
    with pytest.raises(RuntimeError, match="refuses network"):
        with backend._offline_imports():
            socket.create_connection(("example.org", 80))
    assert (socket.socket.connect, socket.create_connection, sys.dont_write_bytecode) == old
    assert not any(isinstance(f, backend._DisableXformers) for f in sys.meta_path)


def test_official_fallback_blocker_and_preloaded_module_refusal(monkeypatch):
    finder = backend._DisableXformers()
    with pytest.raises(ModuleNotFoundError):
        finder.find_spec("xformers.ops")
    assert finder.find_spec("not_xformers") is None
    monkeypatch.setitem(sys.modules, "xformers", SimpleNamespace())
    with pytest.raises(ValueError, match="preloaded"):
        with backend._offline_imports():
            pass


def test_asset_verifier_refuses_wrong_pin_before_checkpoint_read(monkeypatch, tmp_path):
    (tmp_path / "source").mkdir()
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "wrong-pin\n")
    with pytest.raises(ValueError, match="commit"):
        backend.inspect_assets(tmp_path)


def test_sha_streams_file(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(b"test" * 300000)
    assert backend.sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_loader_uses_only_scoped_safe_globals_and_cpu_mmap():
    events = []
    @contextmanager
    def safe_globals(values):
        events.append(("enter", values))
        yield
        events.append(("exit",))
    def load(path, **kwargs):
        events.append(("load", path, kwargs))
        return {"model_state_dict": {"module.a": "tensor"}}
    torch = SimpleNamespace(serialization=SimpleNamespace(
        get_unsafe_globals_in_checkpoint=lambda path: ["numpy.core.multiarray._reconstruct", "numpy.dtype"],
        safe_globals=safe_globals), load=load)
    state, unsafe = backend.checkpoint_state(torch, "weight")
    assert state == {"a": "tensor"}
    assert unsafe == ["numpy.core.multiarray._reconstruct", "numpy.dtype"]
    assert events[1] == ("load", "weight", {"map_location": "cpu", "weights_only": True, "mmap": True})
    assert events[0][0] == "enter" and events[2][0] == "exit"


def test_checkpoint_loader_never_falls_back_for_unknown_globals():
    torch = SimpleNamespace(serialization=SimpleNamespace(
        get_unsafe_globals_in_checkpoint=lambda path: ["arbitrary.module.callable"]),
        load=lambda *a, **k: pytest.fail("unrestricted deserialization attempted"))
    with pytest.raises(ValueError, match="unaudited"):
        backend.checkpoint_state(torch, "weight")
