"""Lazy genuine public DVLT runtime; no model imports at module import time.

Finite IO must admit new study, closed DATA and masks before constructing it.
The raw/loaded state witnesses below retain the reviewed S7 implementation.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tomllib

COMMIT='f59bb18259ab446a7585c4f47731a77cfdefa426'
CHECKPOINT_SHA='87aac651e77e381eea02c40c14985d196edfb0d4a1c2d4197acb50ef18776eb5'
CHECKPOINT_BYTES=468367852
CONFIG_SHA='9a2766604ec6ab62b1d9bf783c289b74128703dec89f6b34e5d58d2f36989adc'
STATE_ELEMENTS=117081731
MODEL_PROFILE={'schema':'dvlt-native-public-model-profile-v1','official_commit':COMMIT,
    'stored_F32_elements':STATE_ELEMENTS,'inference_steps':12,'longest_side':504,
    'patch_size':14,'seed':0,'CPU_threads':2,'interop_threads':1,
    'attention':'official auto SDPA','mixed_precision':'no','native_K_is_model_input':False,
    'public_outputs_preserved':True,'model_internal_consistency_is_independent_geometry':False}
PROFILE_SHA256=hashlib.sha256(json.dumps(MODEL_PROFILE,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def source_check(root,expected):
    if git(root,'rev-parse','HEAD')!=COMMIT or git(root,'status','--porcelain'):
        raise ValueError('clean exact official checkout required')
    files=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
    wanted={p for p in files if Path(p).suffix in {'.py','.yaml','.yml','.toml'}}
    if set(expected)!=wanted or len(wanted)>1000:raise ValueError('ALL tracked source/config roles required')
    if sum((root/p).stat().st_size for p in wanted)>16<<20:raise ValueError('source byte cap')
    for p in wanted:
        if sha(root/p)!=expected[p]:raise ValueError('source mismatch: '+p)

def state_digest(model):
    # One actual CPU state tensor at a time; never a second full state copy.
    h=hashlib.sha256();count=0
    for name,tensor in sorted(model.state_dict().items()):
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,list(a.shape),a.dtype.str],separators=(',',':')).encode())
        h.update(memoryview(a).cast('B'));count+=a.size
    return h.hexdigest(),int(count)

def buffer_digest(model):
    # Include nonpersistent normalization buffers absent from checkpoint/state_dict.
    h=hashlib.sha256();count=0
    for name,tensor in sorted(model.named_buffers()):
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,list(a.shape),a.dtype.str],separators=(',',':')).encode())
        h.update(memoryview(a).cast('B'));count+=a.size
    return h.hexdigest(),int(count)

def checkpoint_state_witness(path,model):
    """Independent SDK/raw-to-loaded binding, not just self-consistency.

    The ONLY name transformation is Module.load_pretrained's documented all-
    keys ``module.`` prefix removal. DVLT inherits the identity subclass hook.
    No filters, fallback keys, shape coercions or randomly initialized entries.
    """
    import numpy as np
    from safetensors import safe_open
    target=model.state_dict();rows=[];count=0
    def digest(a):
        h=hashlib.sha256(json.dumps([list(a.shape),a.dtype.str],separators=(',',':')).encode())
        h.update(memoryview(a).cast('B'));return h.hexdigest()
    with safe_open(str(path),framework='pt',device='cpu') as f:
        keys=list(f.keys());strip=bool(keys) and all(k.startswith('module.') for k in keys)
        mapping={raw:raw[7:] if strip else raw for raw in keys}
        if (len(keys)>10000 or len(set(mapping.values()))!=len(keys)
            or set(mapping.values())!=set(target)):raise ValueError('exact complete raw/loaded key bijection required')
        for raw,key in sorted(mapping.items()):
            shape=tuple(f.get_slice(raw).get_shape());loaded=target[key].detach()
            if loaded.device.type!='cpu' or tuple(loaded.shape)!=shape:
                raise ValueError('raw header/actual loaded shape or CPU mismatch')
            raw_tensor=f.get_tensor(raw);a=raw_tensor.contiguous().numpy();b=loaded.contiguous().numpy()
            if (a.dtype!=np.float32 or b.dtype!=np.float32 or a.shape!=b.shape
                or not np.isfinite(a).all()):raise ValueError('finite actual F32 raw and loaded tensors required')
            raw_sha,loaded_sha=digest(a),digest(b)
            if raw_sha!=loaded_sha:raise ValueError('checkpoint and actual loaded tensor differ: '+key)
            count+=a.size
            rows.append({'raw_key':raw,'loaded_key':key,'shape':list(shape),'dtype':a.dtype.str,
                'raw_typed_sha256':raw_sha,'loaded_typed_sha256':loaded_sha})
            del raw_tensor,a,b
    if count!=STATE_ELEMENTS:raise ValueError('whole checkpoint element count differs')
    return {'schema':'independent-dvlt-checkpoint-loaded-state-v1',
        'name_transform':'strip module. from ALL keys' if strip else 'identity',
        'keys':len(rows),'F32_elements':int(count),'all_actual_tensors_equal':True,'tensors':rows}

def require_cpu_eval(model):
    if any(m.training for m in model.modules()):raise ValueError('all genuine model modules must be eval')
    if any(t.device.type!='cpu' for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('every parameter and buffer must remain on CPU')


def preflight(control, ledger):
    """No model import: exact source, environment and release bytes first."""
    from monocular_osm.dvlt_native_protocol import require, safe, same_ref
    require(set(control) == {"schema", "source_root", "official_source_hashes", "checkpoint_ref",
            "publisher_config_ref", "environment_ref", "profile_sha256"}
            and control["schema"] == "root-frozen-dvlt-native-model-control-v1"
            and control["profile_sha256"] == PROFILE_SHA256, "fixed genuine model control required")
    root = safe(control["source_root"])
    source_check(root, control["official_source_hashes"])
    for name, expected in control["official_source_hashes"].items():
        ledger.add({"path": str(root/name), "sha256": expected})
    weight = safe(control["checkpoint_ref"]["path"])
    require(weight.suffix == ".safetensors" and weight.stat().st_size == CHECKPOINT_BYTES
            and control["checkpoint_ref"]["sha256"] == CHECKPOINT_SHA
            and control["publisher_config_ref"]["sha256"] == CONFIG_SHA, "immutable published release required")
    ledger.add(control["checkpoint_ref"])
    architecture = ledger.json(control["publisher_config_ref"])
    require(architecture["inference_steps"] == 12 and architecture["img_size"] == 504
            and architecture["load_patch_embed_weights"] is False, "published K12 architecture required")
    environment = ledger.json(control["environment_ref"])
    package = tomllib.loads((root/"pyproject.toml").read_text(encoding="utf-8"))
    versions = validate_environment(environment, package["project"]["dependencies"])
    return root, weight, architecture, versions


def validate_environment(environment, core_dependencies):
    """Original frozen full inventory, live identity and official subset; no model IO."""
    from monocular_osm.dvlt_native_protocol import require, same_path
    require(environment.get("schema") == "root-isolated-dvlt-exact-core-environment-v1"
            and environment.get("python") == sys.version
            and environment.get("all_official_core_exact_pins_satisfied") is True
            and environment.get("shared_project_venv_mutated") is False,
            "original exact isolated DVLT environment metadata required")
    for name, actual in (("executable", sys.executable), ("prefix", sys.prefix), ("base_prefix", sys.base_prefix)):
        require(same_path(environment.get(name), actual), "exact live Python environment identity differs")
    # This historical constructor-time flag is retained, never NN permission.
    require(type(environment.get("model_imported_or_forwarded")) is bool, "historical environment scope flag required")
    expected_full = environment.get("versions")
    require(isinstance(expected_full, dict) and 0 < len(expected_full) <= 1000
            and all(isinstance(k, str) and isinstance(v, str) for k, v in expected_full.items()),
            "full frozen distribution map required")
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    wanted_full = {normalize(k): v for k, v in expected_full.items()}
    actual_full = {}
    for distribution in importlib.metadata.distributions():
        name = normalize(distribution.metadata["Name"])
        require(name not in actual_full, "ambiguous duplicate installed distributions")
        actual_full[name] = distribution.version
    require(len(wanted_full) == len(expected_full) and actual_full == wanted_full,
            "live full frozen distribution inventory differs")
    for dep in core_dependencies:
        name, wanted = dep.split("==")
        actual = importlib.metadata.version(name)
        require(actual.split("+")[0] == wanted, "pinned official dependency required: "+dep)
    return dict(expected_full)


class PublicModel:
    """Exactly one CPU/eval model per actor, one seven-view forward per call."""
    def __init__(self, control, ledger, *, check=lambda: None):
        from monocular_osm.dvlt_native_protocol import require
        root, weight, architecture, versions = preflight(control, ledger)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "2"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
        sys.dont_write_bytecode = True
        require(not any(name == "dvlt" or name.startswith("dvlt.") for name in sys.modules),
                "fresh official DVLT import required; no cached model code")
        sys.path.insert(0, str(root/"src"))
        import numpy as np
        import torch
        from accelerate import Accelerator
        from dvlt.model.dvlt.model import DVLT
        from dvlt.model_components import set_attn_backend
        from dvlt.common.constants import DataField, PredictionField
        from dvlt.util.preprocess import preprocess_images
        for name, module in list(sys.modules.items()):
            if name == "dvlt" or name.startswith("dvlt."):
                path = getattr(module, "__file__", None)
                require(path is None or Path(path).resolve().is_relative_to(root/"src"), "foreign DVLT source origin")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        set_attn_backend("auto")
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        require(accelerator.device.type == "cpu", "CPU-only genuine accelerator required")
        check()
        model = DVLT(**architecture)
        cwd = Path.cwd()
        try:
            os.chdir(weight.parent)
            require(Path(weight.name).resolve() == weight, "documented relative file loader identity")
            model.load_pretrained(weight.name, strict=True, remap=None, filter=None)
        finally:
            os.chdir(cwd)
        model.setup_test(accelerator)
        require_cpu_eval(model.model)
        binding = checkpoint_state_witness(weight, model.model)
        self.before, self.elements = state_digest(model.model)
        self.buffers_before = buffer_digest(model.model)
        require(self.elements == STATE_ELEMENTS and all(t.dtype == torch.float32 for t in model.model.state_dict().values()),
                "complete actual F32 loaded state required")
        self.model, self.accelerator, self.torch = model, accelerator, torch
        self.preprocess, self.DataField, self.PredictionField = preprocess_images, DataField, PredictionField
        self.binding, self.versions, self.calls = binding, versions, 0
        check()

    def infer(self, masked_bgr, *, check=lambda: None):
        check()
        import numpy as np
        from PIL import Image
        from monocular_osm.dvlt_native_protocol import require, typed
        require(self.calls == 0, "one immutable seven-view inference per actor")
        require(masked_bgr.dtype == np.uint8 and masked_bgr.ndim == 4 and masked_bgr.shape[0] == 7
                and masked_bgr.shape[-1] == 3, "seven sanitized native BGR frames required")
        rgb = [np.ascontiguousarray(v[..., ::-1]) for v in masked_bgr]
        rgb_hashes = [typed(v) for v in rgb]
        frames = [Image.fromarray(v, "RGB") for v in rgb]
        del rgb
        batch = self.preprocess(frames, img_size=504, patch_size=14, device=self.accelerator.device)
        del frames  # Full-native RGB/PIL copies must not overlap neural activations.
        images = batch[self.DataField.IMAGES]
        require(images.dtype == self.torch.float32 and images.shape[:3] == (1, 7, 3)
                and max(images.shape[-2:]) <= 504 and all(n % 14 == 0 for n in images.shape[-2:]),
                "official native-aspect preprocessing shape differs")
        self.last_input_arrays = {
            "input_images": np.ascontiguousarray(images.detach().cpu().numpy()),
            "gradio_valid_pixels": np.ascontiguousarray(batch["gradio_valid_pixels"].detach().cpu().numpy())}
        before_inputs = {k: typed(v) for k, v in self.last_input_arrays.items()}
        check()
        self.calls += 1
        with self.torch.no_grad(), self.accelerator.autocast():
            prediction = self.model.predict(batch, self.accelerator)
        self.forward_completed = True
        fields = self.PredictionField
        mapping = {fields.DEPTHS: "depths", fields.DEPTHS_CONF: "depths_conf", fields.WORLD_POINTS: "world_points",
                   fields.WORLD_POINTS_DIRECT: "world_points_direct", fields.WORLD_POINTS_DIRECT_CONF: "world_points_direct_conf"}
        require(set(prediction) == set(mapping) | {fields.CAMERAS}, "all and only genuine public prediction fields required")
        cameras = prediction[fields.CAMERAS]
        require(len(cameras) == 1, "single seven-view batch required")
        arrays = {name: np.ascontiguousarray(prediction[key].detach().cpu().numpy()) for key, name in mapping.items()}
        arrays.update(camera_to_worlds=np.ascontiguousarray(cameras[0].camera_to_worlds.detach().cpu().numpy()),
                      intrinsics=np.ascontiguousarray(cameras[0].get_intrinsics_matrices().detach().cpu().numpy()),
                      **self.last_input_arrays)
        require(all(typed(v) == before_inputs[k] for k, v in self.last_input_arrays.items()), "model changed actual input tensors")
        require(arrays["camera_to_worlds"].shape == (7, 3, 4) and arrays["intrinsics"].shape == (7, 3, 3),
                "official per-view cameras differ")
        require(all(v.dtype == np.float32 for k, v in arrays.items() if k != "gradio_valid_pixels")
                and arrays["gradio_valid_pixels"].dtype == np.bool_, "actual F32 public outputs required")
        self.last_prediction_arrays = arrays
        check()
        return arrays, {"profile_sha256": PROFILE_SHA256, "RGB_conversion": "native BGR channel reversal only",
                        "native_RGB_typed_sha256": rgb_hashes, "forward_count": self.calls,
                        "native_K_supplied_to_model": False, "geometry_quality_admitted": False,
                        "full_frozen_distribution_count": len(self.versions),
                        "live_full_environment_and_official_core_exact": True,
                        "python_binary_hash_claimed": False,
                        "historical_environment_no_forward_flag_is_not_current_authority": True}

    def verify(self, *, check=lambda: None):
        from monocular_osm.dvlt_native_protocol import require
        check()
        require_cpu_eval(self.model.model)
        require(state_digest(self.model.model) == (self.before, self.elements)
                and buffer_digest(self.model.model) == self.buffers_before, "actual state/buffers changed")
        return {"state_sha256": self.before, "F32_elements": self.elements,
                "buffer_sha256": self.buffers_before[0], "buffer_elements": self.buffers_before[1],
                "all_state_and_buffers_unchanged": True, "raw_loaded_state_binding": self.binding}
