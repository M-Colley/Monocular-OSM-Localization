"""Inspected, lazy CPU adapter for the official standard SelaVPR++ base release.

Importing this module does not import Torch, deserialize tensors, or construct a
model. The constructor is reserved for a separately reviewed, frozen experiment.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

SOURCE_COMMIT = "56bd921cbd3d53e9c5f91d0aafff147f95fb362a"
WEIGHT_BYTES = 482674957
WEIGHT_SHA256 = "b048490dbd1c27dee67fce6faaec7bec267d19a044c85877af94b8588e596a62"
ASSET_URL = ("https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/"
             "SelaVPR%2B%2B/SelaVPRplusplus_base.pth")
PROFILE = {
    "version": "official-selavprplusplus-base-cpu-v1", "experimental": True,
    "validation_status": "posthoc_dev_unvalidated_no_accuracy_claim",
    "source_commit": SOURCE_COMMIT, "weight_sha256": WEIGHT_SHA256,
    "published_github_asset_digest": "sha256:" + WEIGHT_SHA256,
    "author_checksum_manifest_verified": False,
    "backbone": "dinov2-base", "aggregation": "gem", "hashing": False,
    "rerank": False, "resume": True, "foundation_model_path": None,
    "dimensions": 2048, "resize_hw": [322, 322],
    "storage_color": "BGR", "neural_input_color": "RGB",
    "preprocessing": "RGB_uint8_to_float32_CHW_div255_ImageNet_normalize_then_tensor_resize",
    "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "interpolation": "bilinear", "antialias": True,
    "precision": "float32", "device": "cpu", "cpu_threads": 2,
    "interop_threads": 1, "batch_size": 1,
    "checkpoint_load": "weights_only_CPU_mmap_scoped_NumPy_metadata_allowlist",
    "xformers": "disabled_using_official_ImportError_fallback",
    "faiss": "only_unused_top_level_import_removed_by_AST",
    "license": "repository_MIT_with_retained_upstream_DINO_source_notices",
}
PROFILE_SHA256 = hashlib.sha256(json.dumps(PROFILE, sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def inspect_assets(root):
    """Verify acquired source/weight/schema without importing official code."""
    root = Path(root).resolve(strict=True)
    source = (root / "source").resolve(strict=True)
    require(source.is_relative_to(root), "source escaped asset root")
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(source), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == SOURCE_COMMIT, "official source commit differs")
    require(not git("status", "--porcelain", "--untracked-files=all"), "official source must be clean")
    plan, receipt, schema, release = [_json(root / name) for name in (
        "acquisition_plan.json", "checkpoint_receipt.json", "checkpoint_schema.json", "release_asset.json")]
    require(plan.get("source_commit") == receipt.get("source_commit") == SOURCE_COMMIT,
            "acquisition source identity differs")
    actual_source = {}
    for name in git("ls-files").splitlines():
        path = (source / name).resolve(strict=True)
        require(path.is_relative_to(source) and path.is_file(), "source member escaped checkout")
        actual_source[name] = sha(path)
    require(actual_source == plan.get("source_sha256"), "frozen official source inventory differs")
    weight = (root / "SelaVPRplusplus_base.pth").resolve(strict=True)
    require(weight.is_relative_to(root) and weight.stat().st_size == WEIGHT_BYTES
            and sha(weight) == WEIGHT_SHA256, "bounded official weight identity differs")
    require(receipt.get("status") == "complete" and receipt.get("actual_sha256") == WEIGHT_SHA256
            and receipt.get("actual_bytes") == WEIGHT_BYTES and receipt.get("official_asset_url") == ASSET_URL
            and receipt.get("published_release_digest") == "sha256:" + WEIGHT_SHA256,
            "checkpoint receipt differs from published GitHub asset")
    require(release.get("asset_url") == ASSET_URL and release.get("asset_bytes") == WEIGHT_BYTES
            and release.get("published_release_digest") == "sha256:" + WEIGHT_SHA256
            and sha(root / "release_metadata.body") == release.get("release_body_sha256"),
            "official release metadata identity differs")
    require(schema.get("status") == "complete" and schema.get("actual_sha256") == WEIGHT_SHA256
            and schema.get("Torch_imported") is False and schema.get("model_constructed") is False
            and schema.get("storage_payloads_read") is False
            and 0 < schema.get("tensor_count", 0) <= 10000,
            "symbolic checkpoint inspection is incomplete")
    return {"root": str(root), "source": str(source), "source_commit": SOURCE_COMMIT,
        "source_sha256": actual_source, "weight_path": str(weight),
        "weight_sha256": WEIGHT_SHA256, "weight_bytes": WEIGHT_BYTES,
        "receipt_sha256": sha(root / "checkpoint_receipt.json"),
        "schema_sha256": sha(root / "checkpoint_schema.json"),
        "schema_tensor_count": schema["tensor_count"], "schema_elements": schema["elements"]}


def filtered_aggregation(text):
    """Remove exactly the unused FAISS import; retain every other AST node."""
    original = ast.parse(text)
    require(not any(isinstance(n, ast.Name) and n.id == "faiss" for n in ast.walk(original)),
            "FAISS is used by this source revision")
    removed, kept = [], []
    for node in original.body:
        if isinstance(node, ast.Import) and any(a.name == "faiss" for a in node.names):
            require(len(node.names) == 1 and node.names[0].asname is None,
                    "optional FAISS import changed")
            removed.append("faiss")
        else:
            kept.append(node)
    require(removed == ["faiss"], "expected exactly one unused FAISS import")
    tree = ast.Module(body=kept, type_ignores=original.type_ignores)
    hashes = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    require("GeM" in hashes and "Flatten" in hashes, "official aggregation AST incomplete")
    return tree, {"removed_imports": removed, "unchanged_definitions_sha256": hashes,
        "filtered_ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()}


def rgb_pixels(bgr):
    pixels = np.asarray(bgr)
    require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
            and 0 < pixels.shape[0] <= 16384 and 0 < pixels.shape[1] <= 16384,
            "bounded nonempty uint8 BGR image required")
    return np.ascontiguousarray(pixels[..., ::-1])


def strip_module_prefix(state):
    require(isinstance(state, dict) and 0 < len(state) <= 10000
            and all(isinstance(k, str) and k for k in state), "invalid checkpoint state")
    flags = [key.startswith("module.") for key in state]
    require(all(flags) or not any(flags), "mixed checkpoint module prefixes")
    result = {key[7:] if all(flags) else key: value for key, value in state.items()}
    require(all(key and not key.startswith("module.") for key in result)
            and len(result) == len(state), "ambiguous checkpoint module prefixes")
    return result


def checkpoint_state(torch, path):
    """Training checkpoint metadata needs four explicit NumPy safe globals."""
    multi = np._core.multiarray
    allowed = {"numpy.core.multiarray._reconstruct": multi._reconstruct,
        "numpy.core.multiarray.scalar": multi.scalar,
        "numpy.ndarray": np.ndarray, "numpy.dtype": np.dtype}
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    require(set(unsafe) <= set(allowed), "checkpoint contains unaudited pickle globals")
    safe = [(allowed[name], name) for name in unsafe]
    safe += [type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    require(isinstance(checkpoint, dict) and "model_state_dict" in checkpoint,
            "released checkpoint lacks model_state_dict")
    return strip_module_prefix(checkpoint["model_state_dict"]), unsafe


def source_origins(source, modules=None):
    """Namespace-package search paths and every loaded child must stay pinned."""
    source = Path(source).resolve(strict=True)
    result = {}
    for name, module in list((sys.modules if modules is None else modules).items()):
        if name != "model" and not name.startswith("model."):
            continue
        filename = getattr(module, "__file__", None)
        paths = [filename] if filename else list(getattr(module, "__path__", []))
        require(paths and all(Path(p).resolve(strict=True).is_relative_to(source) for p in paths),
                "foreign model package would shadow the pinned SelaVPR++ source")
        result[name] = [str(Path(p).resolve()) for p in paths]
    return result


class _DisableXformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xformers" or fullname.startswith("xformers."):
            raise ModuleNotFoundError("fixed CPU profile uses the official xFormers fallback")
        return None


@contextmanager
def _offline_imports():
    require(not any(name == "xformers" or name.startswith("xformers.") for name in sys.modules),
            "fresh process without preloaded xFormers required")
    def blocked(*args, **kwargs):
        raise RuntimeError("SelaVPR++ adapter refuses network access")
    old = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    finder, previous_bytecode = _DisableXformers(), sys.dont_write_bytecode
    sys.meta_path.insert(0, finder)
    sys.dont_write_bytecode = True
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = old
        sys.dont_write_bytecode = previous_bytecode
        sys.meta_path.remove(finder)


def _state_digest(model):
    digest, count = hashlib.sha256(), 0
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((name, array.shape, array.dtype.str)).encode())
        raw = memoryview(array).cast("B")
        for begin in range(0, len(raw), 8 * 1024**2):
            digest.update(raw[begin:begin + 8 * 1024**2])
        count += array.size
    return {"sha256": digest.hexdigest(), "elements": int(count)}


class ReleasedSelaBase:
    """Lazy constructor; caller must enforce the external resource/freeze gates.

    No construction or forward has been validated by the static adapter tests.
    The fixed standard float descriptor uses neither binary hashing nor rerank.
    """
    def __init__(self, root):
        self.assets = inspect_assets(root)
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
                "set CUDA_VISIBLE_DEVICES=-1 before starting the fresh CPU child")
        source = Path(self.assets["source"])
        require(not source_origins(source), "fresh process without a preloaded model package required")
        with _offline_imports():
            import torch
            import torchvision
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as transforms
            require(not torch.cuda.is_available(), "CPU adapter refuses an available accelerator")
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            sys.path.insert(0, str(source))
            try:
                importlib.import_module("model")
                path = source / "model/aggregation.py"
                tree, audit = filtered_aggregation(path.read_text(encoding="utf8"))
                spec = importlib.util.spec_from_file_location("model.aggregation", path)
                aggregation = importlib.util.module_from_spec(spec)
                sys.modules["model.aggregation"] = aggregation
                sys.modules["model"].aggregation = aggregation
                exec(compile(tree, str(path), "exec"), aggregation.__dict__)
                upstream = importlib.import_module("model.network")
            finally:
                sys.path.remove(str(source))
            origins = source_origins(source)
            fallbacks = {name: sys.modules[name].XFORMERS_AVAILABLE for name in (
                "model.adapter", "model.dinov2.attention", "model.dinov2.block", "model.dinov2.swiglu_ffn")}
            require(not any(fallbacks.values()), "official CPU attention fallback was not selected")
            state, unsafe = checkpoint_state(torch, self.assets["weight_path"])
            require(all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in state.values()),
                    "checkpoint state contains non-CPU/non-tensor members")
            args = SimpleNamespace(backbone="dinov2-base", aggregation="gem", hashing=False,
                                   rerank=False, resume=True, foundation_model_path=None)
            self.model = upstream.GeoLocalizationNet(args).cpu()
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
        self.torch, self.transforms, self.interpolation = torch, transforms, InterpolationMode.BILINEAR
        self.mean = torch.tensor(PROFILE["mean"], dtype=torch.float32).reshape(3, 1, 1)
        self.std = torch.tensor(PROFILE["std"], dtype=torch.float32).reshape(3, 1, 1)
        self.state = _state_digest(self.model)
        require(len(self.model.state_dict()) == self.assets["schema_tensor_count"]
                and self.state["elements"] == self.assets["schema_elements"],
                "loaded model state disagrees with the symbolically inspected tensor schema")
        self.runtime = {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "device": "cpu", "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "weight_load": "weights_only_true_cpu_mmap",
            "strict_state_load": True, "upstream_origins": origins,
            "scoped_safe_globals": unsafe,
            "xformers_available": fallbacks, "aggregation_ast": audit, "forward_calls": 0}
        require(inspect_assets(root) == self.assets, "source/assets changed during construction")

    def input_tensor(self, bgr):
        rgb = rgb_pixels(bgr)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self.torch.float32).div(255)
        tensor = (tensor - self.mean) / self.std
        return self.transforms.resize(tensor, PROFILE["resize_hw"],
            interpolation=self.interpolation, antialias=True).unsqueeze(0)

    def encode(self, bgr):
        with self.torch.inference_mode():
            result = self.model(self.input_tensor(bgr))
        output = result.detach().cpu().numpy()
        require(output.shape == (1, 2048) and output.dtype == np.float32
                and np.isfinite(output).all() and abs(float(np.linalg.norm(output[0])) - 1) <= 1e-4,
                "official encoder returned an invalid normalized float descriptor")
        self.runtime["forward_calls"] += 1
        return output[0]

    def verify_unchanged(self):
        require(inspect_assets(self.assets["root"]) == self.assets
                and _state_digest(self.model) == self.state
                and source_origins(self.assets["source"]) == self.runtime["upstream_origins"],
                "SelaVPR++ source/assets/imports/numerical state changed")
