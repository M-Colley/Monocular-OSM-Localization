"""Input-only paired ImAge/MegaLoc replay; freeze every variant before scoring."""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_key] = "4"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.pilot_sparse_city_replay import (configure_environment, copy_verified, verify_originals,
    write, load, relative, require, sha, offline, check_headroom, source_identity, verify_query_pixels, verify_freeze)
if __name__ == "__main__": configure_environment()
import numpy as np
from monocular_osm.sparse_sequence import PROFILE_SHA256, decode_sparse_tracks, _array_sha, _json_sha

OFFICIAL_PIN = "5d5e2d8cad11fe7ba699fdbd1f6b0a3c69dade18"
CHECKPOINT_SHA = "ffafd68a5dda19bf45ed211339f754cd3967448ea2fbabe998a5141778d927f8"
CHECKPOINT_BYTES = 573388890
DIMENSION, BATCH = 6144, 4
MAX_RAW_BYTES = 2 * 1024**3
PROFILE = {"version": "image-paired-sparse-v1", "experimental": True,
    "validation_status": "posthoc_dev_unvalidated", "selected_variant": None,
    "models": ["frozen_megaloc", "ImAge_GSV"], "variants": ["raw", "reference_mean"],
    "decoder_profile_sha256": PROFILE_SHA256, "query_mode": "three_native_crops_max_cosine",
    "image": {"official_pin": OFFICIAL_PIN, "checkpoint_sha256": CHECKPOINT_SHA,
        "backbone": "dinov2", "aggregator": None, "resume": True, "freeze_te": 8,
        "num_learnable_aggregation_tokens": 8, "dimensions": DIMENSION,
        "cached_reference_storage_color": "BGR", "native_query_storage_color": "BGR",
        "network_input_color": "RGB", "color_conversion": "BGR_to_RGB_before_tensor_preprocessing",
        "transform": "RGB_to_float32_tensor_ImageNet_normalize_then_bilinear_resize",
        "resize_hw": [322, 322], "antialias": True,
        "mean": [.485, .456, .406], "std": [.229, .224, .225]},
    "cpu_threads": 4, "inter_op_threads": 1, "batch_size": BATCH,
    "limitations": ["paired_method_includes_preprocessing512_to322",
        "legacy_MegaLoc_reference_encoding_revision_unverified", "no_default_policy_adoption"]}
PROFILE_HASH = _json_sha(PROFILE)
PARENT_FILES = ("experiment.json", "query_input.json", "reference_snapshot.json", "reference_evidence.npz", "plan.json")


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def official_files(root):
    require(git(root, "rev-parse", "HEAD") == OFFICIAL_PIN and not git(root, "status", "--porcelain", "--untracked-files=all"),
            "official ImAge source must be exact clean registered pin")
    names = [name for name in git(root, "ls-files").splitlines() if name.endswith(".py") or name.endswith("LICENSE")]
    require("network.py" in names and "backbone/vision_transformer.py" in names and len(names) <= 128,
            "official source inventory incomplete or unbounded")
    return [{"relative": name, "sha256": sha(root / relative(name))} for name in names]


def parent_inputs(parent):
    frozen = load(parent / "frozen_prediction.json")
    require(frozen.get("ground_truth_opened") is False and frozen.get("prediction_available") is True,
            "parent prediction was not input-frozen before labels")
    for name in PARENT_FILES:
        require(name in frozen["artifacts"] and sha(parent / name) == frozen["artifacts"][name], "parent frozen evidence changed")
    experiment, query = load(parent / "experiment.json"), load(parent / "query_input.json")
    require(experiment["profile"]["model"] == "megaloc" and experiment["profile"]["query_mode"] == "crops"
            and experiment["runtime"]["query_stride"] > 0, "parent is not registered crop retrieval control")
    require(query["sampling"]["sample_count"] == len(query["timestamps"]) == len(query["sampling"]["dense_indices"]) == 80
            and query["sampling"]["mode"] == "linspace" and query["native_frame_shape"][2] == 3, "complete native80-query grid required")
    require(sha(experiment["video"]["path"]) == experiment["video"]["sha256"], "parent video changed")
    refs = {str(relative(item["relative"])): item for item in load(parent / "reference_snapshot.json")}
    raw = refs.get(str(Path("references/ref_img_store.npz")))
    require(raw is not None and sha(parent / "references/ref_img_store.npz") == raw["sha256"], "cached BGR pixel store changed")
    return experiment, query, raw


def extract_raw(archive, target):
    """Stream one NPY member to owned disk; never materialize the raw pool."""
    require(not target.exists(), "refuse to overwrite extracted pixels")
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.namelist().count("raw.npy") == 1, "duplicate/missing raw member")
        info = zipped.getinfo("raw.npy")
        require(0 < info.file_size <= MAX_RAW_BYTES, "raw pixel member exceeds budget")
        with zipped.open(info) as source, target.open("xb") as dest:
            shutil.copyfileobj(source, dest, length=8 * 1024**2)
    raw = np.load(target, mmap_mode="r", allow_pickle=False)
    require(raw.dtype == np.uint8 and raw.ndim == 4 and raw.shape[1:] == (512, 512, 3)
            and 0 < len(raw) <= 1500 and raw.flags.c_contiguous, "unsupported cached BGR pixel grid")
    return {"shape": list(raw.shape), "dtype": raw.dtype.str, "sha256": sha(target), "storage_color": "BGR"}


def prepare(parent, official, checkpoint, receipt, family):
    parent, official, checkpoint, receipt = [Path(p).resolve(strict=True) for p in (parent, official, checkpoint, receipt)]
    family = Path(family).resolve()
    require(not family.exists() and not any(family.is_relative_to(p) for p in (parent, official, checkpoint.parent)),
            "new isolated family required outside original inputs")
    experiment, query, raw = parent_inputs(parent)
    code, received = official_files(official), load(receipt)
    require(sha(checkpoint) == CHECKPOINT_SHA and checkpoint.stat().st_size == CHECKPOINT_BYTES
            and received["actual_sha256"] == CHECKPOINT_SHA and received["source_pin"] == OFFICIAL_PIN
            and received["official_asset_digest"] == "sha256:" + CHECKPOINT_SHA,
            "official checkpoint/receipt differs from registered asset")
    family.mkdir(parents=True)
    records = []
    try:
        return prepare_copies(parent, official, checkpoint, receipt, family, experiment, query, raw, code, records)
    except Exception as error:
        write(family / "preparation_failed.json", {"status": "failed", "models_have_not_run": True,
            "scoring_has_not_run": True, "error": f"{type(error).__name__}: {error}", "copied": records})
        raise


def prepare_copies(parent, official, checkpoint, receipt, family, experiment, query, raw, code, records):
    files = {name: (parent / name, sha(parent / name)) for name in PARENT_FILES}
    files.update({"parent_frozen_prediction.json": (parent / "frozen_prediction.json", sha(parent / "frozen_prediction.json")),
        "video.mp4": (Path(experiment["video"]["path"]), experiment["video"]["sha256"]),
        "ref_img_store.npz": (parent / "references/ref_img_store.npz", raw["sha256"]),
        "ImAge_GSV.pth": (checkpoint, CHECKPOINT_SHA), "checkpoint_receipt.json": (receipt, sha(receipt))})
    for name, (source, expected) in files.items(): copy_verified(source, family / "inputs" / name, expected, records)
    for item in code:
        name = relative(item["relative"])
        copy_verified(official / name, family / "inputs/official" / name, item["sha256"], records)
    raw_record = extract_raw(family / "inputs/ref_img_store.npz", family / "inputs/raw.npy")
    paired_inputs(family)
    verify_originals(records); official_files(official)
    lineage = {"copied": records, "official_source": str(official), "official_files": code,
        "parent": str(parent), "parent_freeze_sha256": sha(parent / "frozen_prediction.json"),
        "raw_extracted": raw_record, "models_have_not_run": True, "scoring_has_not_run": True}
    write(family / "input_lineage.json", lineage)
    write(family / "plan.json", {"profile": PROFILE, "profile_sha256": PROFILE_HASH,
        "parent_video": experiment["video"], "window": experiment["window"], "query": query,
        "stride": experiment["runtime"]["query_stride"], "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    artifacts = {str(Path(row["path"]).relative_to(family)): row["sha256"] for row in records}
    artifacts.update({name: sha(family / name) for name in ("inputs/raw.npy", "input_lineage.json", "plan.json")})
    write(family / "inputs_frozen.json", {"scoring_has_not_run": True, "models_have_not_run": True, "artifacts": artifacts})
    return {"status": "prepared_only", "files": len(records), "profile_sha256": PROFILE_HASH}


def paired_inputs(family):
    """Bind each embedded column to cached BGR pixels, not metadata's RGB neural input."""
    with np.load(family / "inputs/reference_evidence.npz", allow_pickle=False) as z:
        values = {name: z[name] for name in ("query_indices", "query_times", "ref_ids", "ref_latlon", "similarities")}
        metadata = json.loads(str(z["metadata"]))
    ids, xy, scores = values["ref_ids"], values["ref_latlon"], values["similarities"]
    require(0 < len(ids) <= 1500 and len(set(ids.tolist())) == len(ids) and scores.shape == (80, len(ids))
            and xy.shape == (len(ids), 2) and np.isfinite(scores).all() and np.isfinite(xy).all()
            and scores.dtype == np.float32 and np.max(np.abs(scores)) <= 1.00001,
            "full frozen reference matrix incomplete or invalid")
    require(np.array_equal(values["query_indices"], np.arange(80)) and metadata["query_mode"] == "crops"
            and metadata["preprocessing"]["reference_color"] == "RGB", "parent query/color identity changed")
    with np.load(family / "inputs/ref_img_store.npz", allow_pickle=False) as z:
        raw_ids, raw_xy = ["mapillary:" + str(v) for v in z["ids"]], z["xy"]
    require(len(raw_ids) == len(ids) == len(set(raw_ids)) and set(raw_ids) == set(ids.tolist()), "pixel pool and matrix IDs differ")
    columns = {key: i for i, key in enumerate(raw_ids)}
    order = np.array([columns[key] for key in ids], dtype=np.int64)
    require(np.array_equal(raw_xy[order], xy), "public reference coordinates differ across pixels/matrix")
    return values, order, metadata


def filtered_network(text):
    """Remove only four unused optional imports; retain every executable AST."""
    tree = ast.parse(text)
    remove = {"transformers": "ViTModel", "aggregators.netvlad": "NetVLAD",
              "aggregators.salad": "SALAD", "aggregators.boq": "BoQ"}
    removed, body = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in remove:
            require(len(node.names) == 1 and node.names[0].name == remove[node.module]
                    and node.names[0].asname is None, "optional import differs from audited official source")
            removed.append(node.module)
        else: body.append(node)
    require(set(removed) == set(remove) and len(removed) == 4, "official optional imports changed")
    kept = ast.Module(body=body, type_ignores=tree.type_ignores)
    required = ("VPRmodel", "get_backbone", "get_aggregator")
    hashes = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
              for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in required}
    require(set(hashes) == set(required), "official network AST incomplete")
    return kept, {"removed_optional_imports": removed, "official_ast_sha256": hashes,
                  "filtered_module_ast_sha256": hashlib.sha256(ast.dump(kept, include_attributes=False).encode()).hexdigest()}


def strip_module_prefix(state):
    require(isinstance(state, dict) and bool(state) and all(isinstance(k, str) for k in state), "invalid checkpoint state")
    flags = [key.startswith("module.") for key in state]
    require(all(flags) or not any(flags), "mixed module prefixes forbidden")
    result = {key[7:] if all(flags) else key: value for key, value in state.items()}
    require(len(result) == len(state) and all(not key.startswith("module.") for key in result), "ambiguous checkpoint prefixes")
    return result


def checkpoint_state(torch, path):
    """Never fall back to unrestricted pickle loading."""
    multi = np._core.multiarray
    allowed = {"numpy.core.multiarray._reconstruct": multi._reconstruct,
        "numpy._core.multiarray._reconstruct": multi._reconstruct,
        "numpy.core.multiarray.scalar": multi.scalar, "numpy._core.multiarray.scalar": multi.scalar,
        "numpy.ndarray": np.ndarray, "numpy.dtype": np.dtype}
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    require(set(unsafe) <= set(allowed), "checkpoint contains unaudited pickle globals")
    safe = [(allowed[name], name) for name in unsafe] + [type(np.dtype("float64")), type(np.dtype("float32")), type(np.dtype("int64"))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict) and "model_state_dict" in checkpoint, "checkpoint lacks model_state_dict")
    return strip_module_prefix(checkpoint["model_state_dict"]), unsafe


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


def verify_backbone_imports(official, modules=None):
    """A namespace package has search paths rather than a __file__."""
    official = Path(official).resolve(strict=True)
    origins = {}
    for name, module in list((sys.modules if modules is None else modules).items()):
        if name != "backbone" and not name.startswith("backbone."): continue
        filename = getattr(module, "__file__", None)
        if filename is not None:
            path = Path(filename).resolve(strict=True)
            require(path.is_relative_to(official), "official backbone import escaped copied source")
            origins[name] = {"file": str(path)}
        else:
            paths = [Path(path).resolve(strict=True) for path in getattr(module, "__path__", [])]
            require(name == "backbone" and paths == [official / "backbone"],
                    "namespace backbone has missing, mixed or escaped search paths")
            origins[name] = {"namespace_paths": [str(path) for path in paths]}
    require("backbone" in origins, "official backbone import is missing")
    return origins


def load_model(inputs, torch):
    official = inputs / "official"
    require(sha(inputs / "ImAge_GSV.pth") == CHECKPOINT_SHA, "prepared official checkpoint changed")
    require(not any(name == "backbone" or name.startswith("backbone.") for name in sys.modules), "refuse preloaded ambiguous backbone module")
    tree, audit = filtered_network((official / "network.py").read_text(encoding="utf8"))
    namespace = {"__file__": str(official / "network.py"), "__name__": "frozen_official_image_network"}
    sys.path.insert(0, str(official))
    try:
        exec(compile(tree, namespace["__file__"], "exec"), namespace)
        args = SimpleNamespace(backbone="dinov2", aggregator=None, resume=True, freeze_te=8, num_learnable_aggregation_tokens=8)
        model = namespace["VPRmodel"](args)
        state, unsafe = checkpoint_state(torch, inputs / "ImAge_GSV.pth")
        model.load_state_dict(state, strict=True)
        del state
        model = model.to(device="cpu", dtype=torch.float32).eval()
        audit["loaded_backbone_origins"] = verify_backbone_imports(official)
        require(sys.modules["backbone.dinov2.attention"].XFORMERS_AVAILABLE is False, "CPU experiment forbids xformers")
        require(sys.modules["backbone.dinov2.block"].XFORMERS_AVAILABLE is False, "CPU experiment forbids xformers blocks")
        audit.update(unsafe_globals_allowlisted=unsafe, state_dict_strict=True, weights_only=True)
        return model, audit
    finally: sys.path.remove(str(official))


def canonical_tensor(rgb):
    """Actual official datasets_ws order, including current bilinear defaults."""
    import torchvision.transforms.functional as tf
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    tensor = tf.to_tensor(Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB"))
    tensor = tf.normalize(tensor, PROFILE["image"]["mean"], PROFILE["image"]["std"])
    return tf.resize(tensor, [322, 322], interpolation=InterpolationMode.BILINEAR, antialias=True)


def encode(images, model, torch, transform=canonical_tensor):
    result = np.empty((len(images), DIMENSION), np.float32)
    with torch.inference_mode():
        for start in range(0, len(images), BATCH):
            tensor = torch.stack([transform(images[i]) for i in range(start, min(start + BATCH, len(images)))])
            output = model(tensor).detach().cpu().numpy()
            require(output.shape == (len(tensor), DIMENSION) and output.dtype == np.float32 and np.isfinite(output).all()
                    and np.max(np.abs(np.linalg.norm(output, axis=1) - 1)) <= 1e-4, "invalid actual ImAge descriptor")
            result[start:start + len(output)] = output
    return result


class Views:
    def __init__(self, images, *, query=False, order=None):
        self.images, self.query, self.order = images, query, order
    def __len__(self): return len(self.images) * (3 if self.query else 1)
    def __getitem__(self, index):
        if not self.query: return self.images[index if self.order is None else self.order[index]][:, :, ::-1]
        from monocular_osm.kartaview_vpr import _square_query_crop
        return _square_query_crop(self.images[index // 3], index % 3)[:, :, ::-1]


def native_queries(family, plan):
    from monocular_osm.frame_extraction import extract_frames
    query = plan["query"]
    frames = extract_frames(family / "inputs/video.mp4", stride=plan["stride"], max_frames=None,
        start_sec=plan["window"][0], end_sec=plan["window"][1], memory_mb=128., max_width=None, sample_count=80)
    try:
        require(frames.sampling == query["sampling"] and frames.fps == query["fps"]
                and np.array_equal(frames.timestamps, query["timestamps"])
                and all(list(p.shape) == query["native_frame_shape"] for p in frames.frames), "native query grid differs from frozen control")
        pixel = hashlib.sha256()
        for p in frames.frames:
            p = np.ascontiguousarray(p)
            pixel.update(str((p.shape, p.dtype.str)).encode()); pixel.update(memoryview(p).cast("B"))
        require(pixel.hexdigest() == query["pixels_sha256"], "native query pixels differ from frozen control")
        ordinals = [int(plan["window"][0] * query["fps"]) + i * plan["stride"] for i in query["sampling"]["dense_indices"]]
        per_frame = verify_query_pixels(frames, family / "inputs/video.mp4", {"fps": query["fps"], "planned_frame_ordinals": ordinals})
        return frames, {"frame_ordinals": ordinals, "per_query_native_pixel_sha256": per_frame,
            "query_pixels_sha256": pixel.hexdigest(), "query_times_sha256": _array_sha(np.asarray(frames.timestamps, np.float64))}
    except Exception:
        frames.close(); raise


def forward_pair(family, raw, order, frames):
    configure_environment()
    os.environ["XFORMERS_DISABLED"] = "1"
    import inspect
    import cv2
    import torch
    import torchvision.transforms.functional as tf
    from monocular_osm.stages.poi_runtime import torch_cpu_threads
    from monocular_osm.provenance import library_versions
    require(inspect.signature(tf.resize).parameters["antialias"].default is True,
            "installed torchvision resize default differs from canonical official transform")
    torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    with offline(), torch_cpu_threads(4) as cpu:
        before = check_headroom()
        model, adapter = load_model(family / "inputs", torch)
        after = check_headroom()
        initial_state = state_digest(model)
        refs = encode(Views(raw, order=order), model, torch)
        queries = encode(Views(frames.frames, query=True), model, torch)
        require(state_digest(model) == initial_state and not torch.cuda.is_initialized(), "actual model changed or GPU initialized")
        # Actual float32 CPU dot products, under the same explicit four-thread scope.
        full = (torch.from_numpy(queries) @ torch.from_numpy(refs).T).numpy().reshape(80, 3, len(refs)).max(axis=1)
        full = np.ascontiguousarray(full, dtype=np.float32)
    runtime = {"cpu_runtime": cpu, "memory_before_model": before, "memory_after_model": after,
        "actual_model_state_sha256": initial_state, "adapter": adapter, "canonical_preprocessing": PROFILE["image"],
        "cached_reference_storage_color": "BGR", "native_query_storage_color": "BGR", "network_input_color": "RGB",
        "network_allowed": False, "xformers_enabled": False, "gpu_initialized": False,
        "libraries": library_versions(("numpy", "opencv-python", "torch", "torchvision", "pillow")),
        "reference_descriptors_sha256": _array_sha(refs), "query_descriptors_sha256": _array_sha(queries)}
    return full, refs, queries, runtime


def paired_predictions(old, new, ids, coordinates, times):
    require(old.shape == new.shape and old.shape == (len(times), len(ids)), "paired matrix grids differ")
    result = {"experimental": True, "selected_variant": None, "profile_sha256": PROFILE_HASH, "outputs": {}}
    for name, scores in (("megaloc", old), ("image", new)):
        decoded = decode_sparse_tracks(np.ascontiguousarray(scores), ids, coordinates, times)
        for variant in ("raw", "reference_mean"):
            result["outputs"][name + "_" + variant] = decoded["outputs"][variant]
    return result


@offline()
def infer(family, expected_commit):
    family = Path(family).resolve(strict=True)
    source_identity(expected_commit)
    verify_freeze(family / "inputs_frozen.json")
    plan, lineage = load(family / "plan.json"), load(family / "input_lineage.json")
    require(plan["profile"] == PROFILE and plan["profile_sha256"] == PROFILE_HASH, "paired profile changed")
    output = family / "inference"
    require(not output.exists(), "refuse to rerun/overwrite paired inference")
    output.mkdir()
    frames, previous_temp, started = None, tempfile.tempdir, time.monotonic()
    try:
        before = check_headroom()
        from monocular_osm.provenance import snapshot_sources
        sources = snapshot_sources([Path(__file__), ROOT / "scripts/pilot_sparse_city_replay.py",
            ROOT / "scripts/plan_sparse_city_replay.py", *sorted((ROOT / "monocular_osm").rglob("*.py"))], output)
        values, order, metadata = paired_inputs(family)
        require(np.array_equal(values["query_times"], plan["query"]["timestamps"]), "control matrix and native query times differ")
        raw = np.load(family / "inputs/raw.npy", mmap_mode="r", allow_pickle=False)
        require(list(raw.shape) == lineage["raw_extracted"]["shape"] and len(raw) == len(order)
                and raw.dtype == np.uint8 and raw.flags.c_contiguous, "prepared mmap pixel grid changed")
        temp = family / "frame_tmp"; temp.mkdir(); tempfile.tempdir = str(temp)
        frames, pixels = native_queries(family, plan)
        new, refs, queries, runtime = forward_pair(family, raw, order, frames)
        runtime["memory_before_inputs"] = before
        times = np.asarray(frames.timestamps, np.float64)
        predictions = paired_predictions(values["similarities"], new, values["ref_ids"], values["ref_latlon"], times)
        freeze_outputs(output, family, expected_commit, sources, lineage, values, new, refs, queries, times,
                       runtime, pixels, metadata, predictions, started)
        return {"status": "complete", "freeze_sha256": sha(output / "frozen_predictions.json"), "variants": list(predictions["outputs"])}
    except Exception as error:
        write(output / "failure.json", {"status": "failed", "scoring_has_not_run": True,
            "error": f"{type(error).__name__}: {error}", "elapsed_s": time.monotonic() - started})
        raise
    finally:
        if frames is not None: frames.close()
        tempfile.tempdir = previous_temp


def freeze_outputs(output, family, expected_commit, sources, lineage, values, new, refs, queries, times,
                   runtime, pixels, metadata, predictions, started):
    np.savez(output / "full_evidence.npz", megaloc_raw_cosines=values["similarities"], image_raw_cosines=new,
        ref_ids=values["ref_ids"], ref_latlon=values["ref_latlon"], query_times=times,
        query_indices=values["query_indices"], frame_ordinals=np.asarray(pixels["frame_ordinals"], np.int64))
    np.savez(output / "image_descriptors.npz", reference_descriptors=refs, query_descriptors=queries,
        ref_ids=values["ref_ids"], query_times=times)
    write(output / "predictions.json", predictions)
    for name, prediction in predictions["outputs"].items(): write(output / ("prediction_" + name + ".json"), prediction)
    write(output / "runtime_and_lineage.json", {"source_commit": expected_commit, "source_sha256": sources,
        "input_freeze_sha256": sha(family / "inputs_frozen.json"), "input_lineage_sha256": sha(family / "input_lineage.json"),
        "native_query": pixels, "runtime": runtime, "legacy_control_provenance": metadata,
        "raw_control_sha256": _array_sha(values["similarities"]), "raw_image_sha256": _array_sha(new),
        "labels_opened": False, "scoring_called": False, "selected_variant": None,
        "elapsed_s": time.monotonic() - started})
    source_identity(expected_commit); verify_freeze(family / "inputs_frozen.json"); verify_originals(lineage["copied"])
    official_files(Path(lineage["official_source"]))
    artifacts = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}
    write(output / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
        "source_commit": expected_commit, "profile_sha256": PROFILE_HASH, "selected_variant": None,
        "variants": list(predictions["outputs"]), "artifacts": artifacts,
        "input_freeze_sha256": sha(family / "inputs_frozen.json"), "original_copied_inputs_unchanged": True})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    for name in ("parent", "official", "checkpoint", "receipt", "family"): setup.add_argument("--" + name, type=Path, required=True)
    run = sub.add_parser("infer"); run.add_argument("--family", type=Path, required=True); run.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    print(json.dumps(prepare(**args) if command == "prepare" else infer(**args)))


if __name__ == "__main__": main()
