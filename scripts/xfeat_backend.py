"""Pinned official XFeat sparse extractor/MNN adapter for offline diagnostics."""
from __future__ import annotations
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
WEIGHT_BYTES = 6247949
WEIGHT_GIT_BLOB = "9da942e9c7406b8add3746bf98d4b1840233cd92"
WEIGHT_SHA256 = "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
CODE_BLOBS = {
    "modules/__init__.py": "a730bea8a7928ea490d96982e77cab7c6abccebf",
    "modules/xfeat.py": "ce6c512d10d1e060e2264ba71725f81a3d31f0d4",
    "modules/model.py": "57539fd35f881348b6bf9887738c7b56a6cbea4f",
    "modules/interpolator.py": "699c1b88ffff56ccf364480282adeca8bc732a2a",
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
}
LOCAL_PROFILE = {"model": "official_XFeat_sparse", "repository_commit": COMMIT,
    "weight_sha256": WEIGHT_SHA256, "weight_git_blob_sha1": WEIGHT_GIT_BLOB,
    "weight_sha256_is_publisher_manifest": False, "top_k": 2500, "detection_threshold": .05,
    "matcher": "official_XFeat.match_reciprocal_cosine", "min_cossim": -1,
    "input": "contiguous_RGB_uint8_then_official_parse_input_BCHW_float_div255",
    "network_normalization": "official_channel_mean_then_InstanceNorm",
    "lighterglue": False, "semidense": False, "device": "cpu", "cpu_threads": 2,
    "interop_threads": 1, "batch_size": 1, "license": "Apache-2.0"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_assets(root):
    """No imports of upstream code, Torch, weight deserialization or constructors."""
    root = Path(root).resolve(strict=True)
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("XFeat checkout is not the approved official pin")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("XFeat checkout must be clean")
    code = {}
    for relative, expected in CODE_BLOBS.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("XFeat source escaped its approved root")
        actual = subprocess.check_output(["git", "-C", str(root), "hash-object", str(path)], text=True).strip()
        if actual != expected:
            raise ValueError(f"XFeat official source blob differs: {relative}")
        code[relative] = {"sha256": sha(path), "git_blob_sha1": actual}
    weight = (root / "weights/xfeat.pt").resolve(strict=True)
    if not weight.is_relative_to(root) or weight.stat().st_size != WEIGHT_BYTES:
        raise ValueError("XFeat weight path/size differs from approved bounded asset")
    body = weight.read_bytes()
    if (len(body) != WEIGHT_BYTES or hashlib.sha256(body).hexdigest() != WEIGHT_SHA256
            or hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest() != WEIGHT_GIT_BLOB):
        raise ValueError("XFeat weight size/Git-object/SHA256 identity mismatch")
    return {"root": str(root), "repository": "https://github.com/verlab/accelerated_features",
            "commit": COMMIT, "code": code, "weight": {"path": str(weight),
                "bytes": WEIGHT_BYTES, "sha256": WEIGHT_SHA256, "official_git_blob_sha1": WEIGHT_GIT_BLOB,
                "sha256_is_publisher_manifest": False}, "license": "Apache-2.0"}


def rgb_view(bgr):
    pixels = np.asarray(bgr)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8 or not all(pixels.shape):
        raise ValueError("nonempty native uint8 BGR image required")
    return np.ascontiguousarray(pixels[..., ::-1])


def _state_identity(model):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((name, array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
        count += int(array.size)
    return {"sha256": digest.hexdigest(), "element_count": count}


class ReleasedXFeat:
    """Fresh CPU process only; uses the released sparse extraction/MNN functions."""
    def __init__(self, root):
        self.assets = inspect_assets(root)
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"-1", ""}:
            raise ValueError("disable CUDA before starting the XFeat CPU process")
        import torch
        if torch.cuda.is_available():
            raise RuntimeError("XFeat diagnostic refuses an available accelerator")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        self.torch = torch
        root = Path(self.assets["root"])
        for name in ("modules", "modules.xfeat", "modules.model", "modules.interpolator"):
            module = sys.modules.get(name)
            if module is not None and (not getattr(module, "__file__", None)
                    or not Path(module.__file__).resolve().is_relative_to(root)):
                raise RuntimeError("foreign modules package would shadow the pinned XFeat source")
        sys.path.insert(0, str(root))
        upstream = importlib.import_module("modules.xfeat")
        for name in ("modules", "modules.xfeat", "modules.model", "modules.interpolator"):
            if not Path(sys.modules[name].__file__).resolve().is_relative_to(root):
                raise RuntimeError("XFeat imported source outside the approved checkout")
        state = torch.load(self.assets["weight"]["path"], map_location="cpu", weights_only=True)
        # A dict avoids upstream's filename torch.load and its hub URL loader.
        self.model = upstream.XFeat(weights=state, top_k=2500, detection_threshold=.05).eval()
        if self.model.dev.type != "cpu" or any(parameter.device.type != "cpu" for parameter in self.model.parameters()):
            raise RuntimeError("released XFeat did not stay on CPU")
        self.state = _state_identity(self.model)
        self.runtime = {"torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "device": str(self.model.dev),
            "weight_load": "torch.load_weights_only_true_cpu_then_official_dict_constructor",
            "lighterglue_constructed": self.model.lighterglue is not None,
            "feature_forward_calls": 0, "pair_match_requests": 0, "actual_mnn_calls": 0}
        if self.runtime["lighterglue_constructed"]:
            raise RuntimeError("LighterGlue is outside this preregistered experiment")

    def features(self, bgr):
        tensor = self.model.parse_input(rgb_view(bgr))
        result = self.model.detectAndCompute(tensor, top_k=2500, detection_threshold=.05)[0]
        points = result["keypoints"].detach().cpu().numpy()
        descriptors = result["descriptors"].detach().cpu()
        if (points.shape != (len(descriptors), 2) or len(points) > 2500
                or descriptors.ndim != 2 or descriptors.shape[1] != 64
                or not np.isfinite(points).all() or not self.torch.isfinite(descriptors).all()):
            raise ValueError("released XFeat returned invalid sparse features")
        self.runtime["feature_forward_calls"] += 1
        return points, descriptors

    def correspondences(self, first, second):
        self.runtime["pair_match_requests"] += 1
        points_a, desc_a = first
        points_b, desc_b = second
        if not len(desc_a) or not len(desc_b):
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        ia, ib = self.model.match(desc_a, desc_b, min_cossim=-1)
        self.runtime["actual_mnn_calls"] += 1
        ia, ib = ia.detach().cpu().numpy(), ib.detach().cpu().numpy()
        return points_a[ia], points_b[ib]

    def verify_unchanged(self):
        if inspect_assets(self.assets["root"]) != self.assets or _state_identity(self.model) != self.state:
            raise ValueError("XFeat source, weight or loaded numerical state changed during execution")
