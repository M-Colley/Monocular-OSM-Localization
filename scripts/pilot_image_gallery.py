"""Bounded cached-gallery ImAge comparison; no acquisition or label interface.

Prepare copies of every registered input. Infer requires a clean explicit source
pin and freezes all four fixed full-support decoder outputs before evaluation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import inspect
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile

if __name__ == "__main__":
    for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_name] = "4"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import pilot_image_paired as image
from scripts import pilot_sparse_city_replay as sparse
from monocular_osm import full_sequence as full
from monocular_osm.sparse_sequence import _array_sha, _json_sha
import numpy as np

PROFILE = {"version": "image-full-gallery-v1", "experimental": True,
    "validation_status": "posthoc_dev_method_diagnostic", "selected_variant": None,
    "models": ["frozen_megaloc", "ImAge_GSV"], "variants": ["raw", "reference_mean"],
    "decoder_profile_sha256": full.PROFILE_SHA256, "image": image.PROFILE["image"],
    "query_mode": "three_native_crops_max_cosine", "query_count": 80,
    "canonical_reference": "first frozen-plan pool occurrence; lexicographic union columns",
    "reference_storage_color": "BGR", "network_input_color": "RGB",
    "batch_size": 4, "cpu_threads": 4, "inter_op_threads": 1,
    "max_references": 10000, "raw_pool_max_rows": 1500, "raw_pool_max_bytes": image.MAX_RAW_BYTES,
    "reference_descriptor_storage": "float32 C-order NPY mmap; one output row per canonical ID",
    "similarity_reference_block": 256, "network": False, "new_reference_images": False,
    "limitations": ["legacy_MegaLoc_pixel_to_embedding_revision_and_preprocessing_UNVERIFIED",
        "paired_method_includes_preprocessing512_to322", "no_causal_encoder_only_claim",
        "no_default_policy_adoption"]}
PROFILE_SHA256 = _json_sha(PROFILE)
require, write = sparse.require, sparse.write


def allowed_path(path):
    """Reject lexical names and resolved aliases before reading any data bytes."""
    path = Path(path)
    for value in (path, path.resolve()):
        text = str(value).replace("\\", "/").casefold()
        saved_score = value.name.casefold().endswith((".json", ".jsonl")) and re.search(
            r"(?:^|[_.-])(?:scores?|scoring)(?:[_.-]|$)", value.name.casefold())
        require("ground_truth" not in text and not saved_score,
                "ground_truth and saved score files are outside gallery inputs")
    return path


def sha(path):
    return sparse.sha(allowed_path(path))


def load(path):
    return sparse.load(allowed_path(path))


def verify_freeze(path):
    path = allowed_path(path).resolve(strict=True)
    value = load(path)
    require(value.get("scoring_has_not_run") is True, "input was not frozen before evaluation")
    targets = []
    for name, expected in value["artifacts"].items():
        allowed_path(name)
        target = allowed_path(path.parent / sparse.relative(name)).resolve(strict=True)
        require(target.is_relative_to(path.parent), "input artifact escaped frozen root")
        targets.append((target, expected))
    for target, expected in targets:
        require(sha(target) == expected, "input freeze changed")
    return value


def copy_verified(source, target, expected, records):
    sparse.copy_verified(allowed_path(source), allowed_path(target), expected, records)


def verify_originals(records):
    require(all(sha(item["source"]) == item["sha256"] for item in records), "original copied input changed")


def key(path):
    return os.path.normcase(str(Path(path).resolve()))


def frozen_inventory(path):
    path = allowed_path(path).resolve(strict=True)
    frozen = load(path)
    require(frozen.get("ground_truth_opened") is False, "pixel inventory was not input-only")
    for name, digest in frozen["artifacts"].items():
        allowed_path(name)
        target = allowed_path(path.parent / sparse.relative(name)).resolve(strict=True)
        require(target.is_relative_to(path.parent) and sha(target) == digest, "pixel inventory artifact changed")
    require({"pixel_identity.json", "feasibility.json", "inventory_plan.json"} <= set(frozen["artifacts"]),
            "complete pixel inventory roles required")
    ledger, summary = load(path.parent / "pixel_identity.json"), load(path.parent / "feasibility.json")
    require(summary.get("ground_truth_opened") is False and summary.get("model_constructors") == 0
            and summary.get("forwards") == 0 and summary.get("downloads") == 0,
            "pixel audit used unsupported inputs")
    require(not summary["missing_ids"] and not summary["metadata_failures"] and not summary["pixel_stream_failures"],
            "pixel inventory is incomplete")
    return frozen, ledger, summary


def parent_inputs(parent):
    parent = Path(parent).resolve(strict=True)
    verify_freeze(parent / "inputs_frozen.json")
    frozen = verify_freeze(parent / "inference/frozen_predictions.json")
    require(frozen.get("status") == "complete" and frozen.get("selected_variant") is None,
            "registered parent prediction is incomplete or selected")
    plan = load(parent / "inputs/replay_plan.json")
    runtime = load(parent / "inference/runtime_and_lineage.json")
    with np.load(parent / "inference/full_evidence.npz", allow_pickle=False) as archive:
        values = {name: archive[name] for name in ("ref_ids", "ref_latlon", "query_times", "query_indices", "frame_ordinals", "raw_cosines")}
    ids, coordinates, times, scores = [values[name] for name in ("ref_ids", "ref_latlon", "query_times", "raw_cosines")]
    require(ids.tolist() == plan["union"]["ref_ids"] == sorted(set(ids.tolist()))
            and len(ids) == plan["union"]["count"] <= PROFILE["max_references"]
            and _json_sha(ids.tolist()) == plan["union"]["ref_ids_sha256"], "frozen union IDs changed")
    require(_array_sha(coordinates) == plan["union"]["ref_latlon_sha256"]
            and scores.dtype == np.float32 and scores.shape == (80, len(ids))
            and np.isfinite(scores).all() and np.max(np.abs(scores)) <= 1.00001,
            "complete finite frozen control matrix required")
    full._inputs(scores, ids, coordinates, times)
    query = plan["query"]
    require(query["sample_count"] == 80 and query["query_mode"] == "crops" and query["views_per_query"] == 3
            and query["native_pixels"] is True and np.array_equal(values["query_indices"], np.arange(80))
            and np.array_equal(values["frame_ordinals"], query["planned_frame_ordinals"])
            and _array_sha(times) == runtime["actual_query_times_sha256"]
            and len(runtime["per_query_native_pixel_sha256"]) == 80,
            "native crop query identity is incomplete")
    return plan, runtime, values


def canonical_rows(plan, ledger):
    """Validate every ID and duplicate, preserving the original first-pool rule."""
    ids = plan["union"]["ref_ids"]
    require(ledger.get("schema") == "reference-raw-pixel-inventory-v1"
            and ledger.get("storage_color") == "BGR_uint8_OpenCV"
            and ledger["ids_in_frozen_union_order"] == ids and set(ledger["per_id"]) == set(ids)
            and not ledger["missing_ids"], "pixel ledger does not cover exactly the frozen union")
    stores = {}
    for store in ledger["stores"]:
        path = key(store["path"])
        require(path not in stores and store["status"] == "available"
                and len(store["raw_shape"]) == 4 and store["raw_shape"][1:] == [512, 512, 3]
                and 0 < store["raw_shape"][0] <= PROFILE["raw_pool_max_rows"]
                and 0 < store["bytes"] <= PROFILE["raw_pool_max_bytes"], "invalid or duplicate raw-store inventory")
        stores[path] = store
    require(0 < len(stores) <= 64, "raw store count exceeds bound")
    first = {}
    for index, pool in enumerate(plan["pools"]):
        require(len(pool["ref_ids"]) == len(pool["ref_latlon"]) == pool["served_count"], "pool row identity incomplete")
        for identifier, xy in zip(pool["ref_ids"], pool["ref_latlon"]):
            if identifier in first:
                require(first[identifier][1] == xy, "duplicate ID has conflicting public positions")
            else:
                first[identifier] = (index, xy, key(pool["reference_store"]["path"]))
    require(set(first) == set(ids), "frozen pool list does not cover union")
    paths = [key(pool["reference_store"]["path"]) for pool in plan["pools"]]
    require(set(paths) == set(stores), "raw store inventory differs from registered pool family")
    for path, store in stores.items():
        require(store["pool_indices"] == [i for i, value in enumerate(paths) if value == path],
                "raw store pool order differs from frozen plan")
    rows, conflicts = [], []
    for column, identifier in enumerate(ids):
        item, (index, xy, path) = ledger["per_id"][identifier], first[identifier]
        chosen = item["canonical"]
        require(item["canonical_pool_index"] == index and key(chosen["path"]) == path
                and chosen["latlon"] == xy and chosen in item["occurrences"], "canonical pixels are not from first registered pool")
        require(path in stores and index in stores[path]["pool_indices"], "canonical pool missing from raw store")
        digests = set()
        for occurrence in item["occurrences"]:
            source = stores.get(key(occurrence["path"]))
            require(source is not None and occurrence["latlon"] == xy
                    and type(occurrence["raw_row"]) is int and 0 <= occurrence["raw_row"] < source["raw_shape"][0]
                    and len(occurrence["pixels_sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in occurrence["pixels_sha256"]), "invalid pixel occurrence")
            digests.add(occurrence["pixels_sha256"])
        if len(digests) > 1:
            conflicts.append(identifier)
        rows.append({"column": column, "id": identifier, "pool_index": index, "source": path,
                     "raw_row": chosen["raw_row"], "pixels_sha256": chosen["pixels_sha256"], "latlon": xy})
    require(conflicts == ledger["conflicting_pixel_ids"], "duplicate pixel conflicts were hidden or changed")
    return rows, stores


def prepare(parent, inventory, official, checkpoint, receipt, family):
    parent, inventory, official, checkpoint, receipt = [allowed_path(p).resolve(strict=True) for p in
        (parent, inventory, official, checkpoint, receipt)]
    family = Path(family).resolve()
    require(not family.exists() and not any(family.is_relative_to(p) for p in
            (parent, inventory.parent, official, checkpoint.parent)), "new isolated output family required")
    plan, runtime, control = parent_inputs(parent)
    frozen, ledger, summary = frozen_inventory(inventory)
    require(frozen["source_plan_sha256"] == sha(parent / "inputs/replay_plan.json")
            and summary["source_plan_sha256"] == frozen["source_plan_sha256"], "pixel audit uses another frozen plan")
    rows, stores = canonical_rows(plan, ledger)
    require(not any(family.is_relative_to(Path(path).parent) for path in stores),
            "output family must be isolated from original raw stores")
    require(summary["available_canonical_ids"] == len(rows), "pixel summary count differs")
    code, received = image.official_files(official), load(receipt)
    require(sha(checkpoint) == image.CHECKPOINT_SHA and checkpoint.stat().st_size == image.CHECKPOINT_BYTES
            and received["actual_sha256"] == image.CHECKPOINT_SHA and received["source_pin"] == image.OFFICIAL_PIN
            and received["official_asset_digest"] == "sha256:" + image.CHECKPOINT_SHA,
            "registered official checkpoint and receipt required")
    family.mkdir(parents=True)
    copied, raw_sources = [], []
    try:
        files = {"replay_plan.json": parent / "inputs/replay_plan.json",
            "parent_runtime.json": parent / "inference/runtime_and_lineage.json",
            "parent_frozen_predictions.json": parent / "inference/frozen_predictions.json",
            "parent_inputs_frozen.json": parent / "inputs_frozen.json",
            "full_evidence.npz": parent / "inference/full_evidence.npz",
            "video.mp4": parent / "inputs/video.mp4", "inventory/inventory_frozen.json": inventory,
            "checkpoint_receipt.json": receipt}
        for name in frozen["artifacts"]:
            files["inventory/" + name] = inventory.parent / sparse.relative(name)
        for name, path in files.items():
            copy_verified(path, family / "inputs" / sparse.relative(name), sha(path), copied)
        require(sha(family / "inputs/video.mp4") == plan["query"]["video"]["sha256"], "source video changed")
        copy_verified(checkpoint, family / "inputs/ImAge_GSV.pth", image.CHECKPOINT_SHA, copied)
        for item in code:
            name = sparse.relative(item["relative"])
            copy_verified(official / name, family / "inputs/official" / name, item["sha256"], copied)
        for index, (path, store) in enumerate(stores.items()):
            name = f"inputs/raw/pool_{index:02d}.npz"
            copy_verified(path, family / name, store["sha256"], copied)
            raw_sources.append({"source": path, "copy": name, "sha256": store["sha256"],
                                "raw_shape": store["raw_shape"], "pool_indices": store["pool_indices"]})
        verify_originals(copied)
        image.official_files(official)
        write(family / "plan.json", {"profile": PROFILE, "profile_sha256": PROFILE_SHA256,
            "query": plan["query"], "per_query_native_pixel_sha256": runtime["per_query_native_pixel_sha256"],
            "query_times_sha256": runtime["actual_query_times_sha256"], "canonical_rows": rows,
            "raw_sources": raw_sources, "duplicate_pixel_conflict_ids": ledger["conflicting_pixel_ids"],
            "control": {"ref_ids_sha256": _json_sha(control["ref_ids"].tolist()),
                "ref_latlon_sha256": _array_sha(control["ref_latlon"]), "raw_cosines_sha256": _array_sha(control["raw_cosines"])},
            "parent": str(parent), "parent_prediction_freeze_sha256": sha(parent / "inference/frozen_predictions.json"),
            "pixel_inventory_freeze_sha256": sha(inventory), "selected_variant": None})
        write(family / "input_lineage.json", {"copied": copied, "official_source": str(official),
            "official_files": code, "models_have_not_run": True, "scoring_has_not_run": True})
        artifacts = {str(Path(row["path"]).relative_to(family)): row["sha256"] for row in copied}
        artifacts.update({name: sha(family / name) for name in ("plan.json", "input_lineage.json")})
        write(family / "inputs_frozen.json", {"scoring_has_not_run": True, "models_have_not_run": True,
            "artifacts": artifacts})
        return {"status": "prepared_only", "reference_count": len(rows), "raw_pools": len(stores),
                "profile_sha256": PROFILE_SHA256}
    except Exception as error:
        write(family / "preparation_failed.json", {"status": "failed", "models_have_not_run": True,
            "scoring_has_not_run": True, "error": f"{type(error).__name__}: {error}", "copied": copied})
        raise


def close_map(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def raw_header(archive, expected_shape):
    """Reject unsupported grids before the old adapter can retain a failed mmap."""
    with zipfile.ZipFile(allowed_path(archive)) as zipped:
        require(zipped.namelist().count("raw.npy") == 1, "raw member is missing or duplicated")
        info = zipped.getinfo("raw.npy")
        require(0 < info.file_size <= image.MAX_RAW_BYTES, "raw member exceeds bounded extraction budget")
        with zipped.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "unsupported raw NPY version")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
            require(list(shape) == expected_shape and dtype == np.uint8 and not fortran
                    and len(shape) == 4 and shape[1:] == (512, 512, 3)
                    and 0 < shape[0] <= PROFILE["raw_pool_max_rows"]
                    and info.file_size == stream.tell() + int(np.prod(shape)) * dtype.itemsize,
                    "unsupported or inconsistent raw header before mmap")


def minimum_memory(records):
    return {name: min(row[name] for row in records if name in row) for name in
        ("available_physical_bytes", "available_commit_bytes") if any(name in row for row in records)}


class RGBRows:
    def __init__(self, raw, rows):
        self.raw, self.rows = raw, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.raw[self.rows[index]][:, :, ::-1]


def encode_gallery(family, plan, model, torch, output, *, encoder=None):
    """One read-only raw map and at most four reference descriptors in RAM."""
    encoder = image.encode if encoder is None else encoder
    rows, sources = plan["canonical_rows"], plan["raw_sources"]
    require(0 < len(rows) <= PROFILE["max_references"]
            and [row["column"] for row in rows] == list(range(len(rows))), "incomplete canonical column order")
    require(not output.exists(), "refuse overwriting reference descriptors")
    descriptors = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                            shape=(len(rows), image.DIMENSION))
    seen, stats = np.zeros(len(rows), bool), []
    staging = family / "raw_staging"
    staging.mkdir(exist_ok=False)
    try:
        for number, source in enumerate(sources):
            memory = sparse.check_headroom()
            archive = (family / sparse.relative(source["copy"])).resolve(strict=True)
            require(archive.is_relative_to(family) and sha(archive) == source["sha256"], "copied raw pool changed")
            selected = [row for row in rows if row["source"] == source["source"]]
            require(all(not seen[row["column"]] for row in selected), "canonical descriptor repeated")
            target, raw = staging / f"pool_{number:02d}.npy", None
            checks = [memory]
            try:
                raw_header(archive, source["raw_shape"])
                with np.load(archive, allow_pickle=False) as saved:
                    ids, xy = saved["ids"], saved["xy"]
                require(len(ids) == source["raw_shape"][0] and xy.shape == (len(ids), 2)
                        and len(set(ids.astype(str).tolist())) == len(ids), "invalid raw metadata row inventory")
                for row in selected:
                    index = row["raw_row"]
                    require("mapillary:" + str(ids[index]) == row["id"] and xy[index].tolist() == row["latlon"],
                            "canonical raw row differs from registered public ID/position")
                extracted = image.extract_raw(archive, target)
                require(extracted["shape"] == source["raw_shape"], "raw pool shape differs from frozen inventory")
                raw = np.load(target, mmap_mode="r", allow_pickle=False)
                for start in range(0, len(selected), image.BATCH):
                    checks.append(sparse.check_headroom())
                    batch = selected[start:start + image.BATCH]
                    for row in batch:
                        frame = raw[row["raw_row"]]
                        require(hashlib.sha256(memoryview(frame).cast("B")).hexdigest() == row["pixels_sha256"],
                                "canonical raw pixels differ from frozen per-ID ledger")
                    values = encoder(RGBRows(raw, [row["raw_row"] for row in batch]), model, torch)
                    require(values.shape == (len(batch), image.DIMENSION) and values.dtype == np.float32
                            and np.isfinite(values).all()
                            and np.max(np.abs(np.linalg.norm(values, axis=1)-1)) <= 1e-4,
                            "invalid actual ImAge gallery descriptors")
                    columns = [row["column"] for row in batch]
                    descriptors[columns] = values
                    seen[columns] = True
                descriptors.flush()
                stats.append({"pool": number, "source": source, "canonical_rows_encoded": len(selected),
                    "memory_before_pool": memory, "memory_minimum": minimum_memory(checks), "extracted": extracted})
                print(f"ImAge gallery pool {number+1}/{len(sources)}: {len(selected)} canonical refs", flush=True)
            finally:
                if raw is not None:
                    close_map(raw)
                if target.exists():
                    target.unlink()
        require(seen.all(), "one or more registered reference IDs were not encoded")
        descriptors.flush()
        return stats, _array_sha(descriptors)
    finally:
        close_map(descriptors)


def encode_queries(frames, model, torch):
    views = image.Views(frames.frames, query=True)
    require(len(views) == 240, "complete three-view native query grid required")
    descriptors = np.empty((len(views), image.DIMENSION), np.float32)
    checks = []
    for start in range(0, len(views), image.BATCH):
        checks.append(sparse.check_headroom())
        images = [views[index] for index in range(start, min(start+image.BATCH, len(views)))]
        descriptors[start:start+len(images)] = image.encode(images, model, torch)
    return descriptors, minimum_memory(checks)


def similarities(queries, references, torch):
    require(queries.shape == (240, image.DIMENSION) and queries.dtype == np.float32
            and references.ndim == 2 and references.shape[1] == image.DIMENSION
            and references.dtype == np.float32 and 0 < len(references) <= PROFILE["max_references"],
            "incomplete native views or gallery descriptors")
    result = np.empty((80, len(references)), np.float32)
    with torch.inference_mode():
        query_tensor = torch.from_numpy(queries)
        block = PROFILE["similarity_reference_block"]
        for start in range(0, len(references), block):
            # Small explicit copy avoids passing a read-only mmap to a mutable Tensor.
            ref = np.array(references[start:start+block], dtype=np.float32, copy=True)
            product = (query_tensor @ torch.from_numpy(ref).T).numpy()
            result[:, start:start+len(ref)] = product.reshape(80, 3, len(ref)).max(axis=1)
    require(np.isfinite(result).all() and np.max(np.abs(result)) <= 1.00001, "invalid actual cosine matrix")
    return result


def control_inputs(family, plan):
    with np.load(allowed_path(family / "inputs/full_evidence.npz"), allow_pickle=False) as archive:
        values = {name: archive[name] for name in ("raw_cosines", "ref_ids", "ref_latlon", "query_times", "frame_ordinals", "query_indices")}
    require(_json_sha(values["ref_ids"].tolist()) == plan["control"]["ref_ids_sha256"]
            and _array_sha(values["ref_latlon"]) == plan["control"]["ref_latlon_sha256"]
            and _array_sha(values["raw_cosines"]) == plan["control"]["raw_cosines_sha256"]
            and _array_sha(values["query_times"]) == plan["query_times_sha256"]
            and np.array_equal(values["query_indices"], np.arange(80))
            and np.array_equal(values["frame_ordinals"], plan["query"]["planned_frame_ordinals"]),
            "frozen control matrix or query grid changed")
    require([row["id"] for row in plan["canonical_rows"]] == values["ref_ids"].tolist(),
            "canonical pixel columns differ from frozen control")
    full._inputs(values["raw_cosines"], values["ref_ids"], values["ref_latlon"], values["query_times"])
    return values


def native_queries(family, plan, expected_times):
    from monocular_osm.frame_extraction import extract_frames
    query = plan["query"]
    video = allowed_path(family / "inputs/video.mp4")
    frames = extract_frames(video, stride=query["stride"], max_frames=None,
        start_sec=query["registered_window"][0], end_sec=query["registered_window"][1],
        memory_mb=128., max_width=None, sample_count=80)
    try:
        require(len(frames.frames) == 80 and frames.sampling["dense_count"] == query["dense_count"]
                and frames.sampling["dense_indices"] == query["query_dense_indices"]
                and float(frames.fps) == query["fps"]
                and all(list(p.shape) == query["native_shape_hw"] + [3] for p in frames.frames)
                and np.array_equal(np.asarray(frames.timestamps, np.float64), expected_times),
                "native frames or actual timestamps differ from frozen query grid")
        pixels = sparse.verify_query_pixels(frames, family / "inputs/video.mp4", query)
        require(pixels == plan["per_query_native_pixel_sha256"], "native query pixels differ from frozen control")
        return frames
    except Exception:
        frames.close()
        raise


def decode_outputs(output, image_scores, control):
    require(image_scores.shape == control["raw_cosines"].shape and image_scores.dtype == np.float32,
            "paired score matrices do not share every row and column")
    result = {}
    for model, matrix in (("frozen_megaloc", control["raw_cosines"]), ("ImAge_GSV", image_scores)):
        decoded = full.decode_full_tracks(matrix, control["ref_ids"], control["ref_latlon"], control["query_times"])
        require(decoded["selected_variant"] is None and decoded["profile_sha256"] == full.PROFILE_SHA256,
                "full-support decoder contract changed")
        write(output / f"predictions_{model}.json", decoded)
        for variant in PROFILE["variants"]:
            write(output / f"prediction_{model}_{variant}.json", decoded["outputs"][variant])
        result[model] = {"inputs": decoded["inputs"], "allocation": decoded["allocation"]}
    return result


def source_binding(expected_commit):
    require(Path(__file__).resolve() == ROOT / "scripts/pilot_image_gallery.py"
            and image.ROOT == sparse.ROOT == ROOT
            and Path(image.__file__).resolve() == ROOT / "scripts/pilot_image_paired.py"
            and Path(sparse.__file__).resolve() == ROOT / "scripts/pilot_sparse_city_replay.py"
            and Path(full.__file__).resolve() == ROOT / "monocular_osm/full_sequence.py",
            "executed driver or imported helpers belong to another source tree")
    sparse.source_identity(expected_commit)


@sparse.offline()
def infer(family, expected_commit, expected_profile):
    require(expected_profile == PROFILE_SHA256, "preregistered gallery profile differs")
    source_binding(expected_commit)
    sparse.configure_environment()
    os.environ["XFORMERS_DISABLED"] = "1"
    family = Path(family).resolve(strict=True)
    verify_freeze(family / "inputs_frozen.json")
    plan, lineage = load(family / "plan.json"), load(family / "input_lineage.json")
    require(plan["profile"] == PROFILE and plan["profile_sha256"] == PROFILE_SHA256, "prepared profile changed")
    _, ledger, _ = frozen_inventory(family / "inputs/inventory/inventory_frozen.json")
    registered = load(family / "inputs/replay_plan.json")
    rows, stores = canonical_rows(registered, ledger)
    require(rows == plan["canonical_rows"] and len(stores) == len(plan["raw_sources"]), "prepared canonical mapping changed")
    control = control_inputs(family, plan)
    before = sparse.check_headroom()
    output = family / "inference"
    require(not output.exists(), "refuse rerunning a completed or partial inference family")
    output.mkdir()
    frames, previous_temp, started = None, tempfile.tempdir, time.monotonic()
    try:
        from monocular_osm.provenance import snapshot_sources
        paths = [ROOT / "main.py", Path(__file__), ROOT / "scripts/pilot_image_paired.py",
            ROOT / "scripts/pilot_sparse_city_replay.py", ROOT / "scripts/plan_sparse_city_replay.py",
            *sorted((ROOT / "monocular_osm").rglob("*.py"))]
        sources = snapshot_sources(paths, output)
        temp = family / "frame_tmp"
        temp.mkdir()
        tempfile.tempdir = str(temp)
        frames = native_queries(family, plan, control["query_times"])
        import cv2
        import torch
        import torchvision.transforms.functional as tf
        from monocular_osm.stages.poi_runtime import torch_cpu_threads
        from monocular_osm.provenance import library_versions
        require(inspect.signature(tf.resize).parameters["antialias"].default is True,
                "installed transform differs from canonical official preprocessing")
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        with torch_cpu_threads(4) as cpu:
            before_model = sparse.check_headroom()
            model, adapter = image.load_model(family / "inputs", torch)
            after_model = sparse.check_headroom()
            initial_state = image.state_digest(model)
            pools, descriptor_digest = encode_gallery(family, plan, model, torch, output / "reference_descriptors.npy")
            queries, query_memory_minimum = encode_queries(frames, model, torch)
            references = np.load(output / "reference_descriptors.npy", mmap_mode="r", allow_pickle=False)
            try:
                scores = similarities(queries, references, torch)
            finally:
                close_map(references)
            require(image.state_digest(model) == initial_state and not torch.cuda.is_initialized(),
                    "actual model state changed or GPU initialized")
            del model
            gc.collect()
        np.save(output / "query_descriptors.npy", queries, allow_pickle=False)
        np.savez(output / "full_evidence.npz", image_raw_cosines=scores, megaloc_raw_cosines=control["raw_cosines"],
            ref_ids=control["ref_ids"], ref_latlon=control["ref_latlon"], query_times=control["query_times"],
            query_indices=control["query_indices"], frame_ordinals=control["frame_ordinals"])
        before_decode = sparse.check_headroom()
        decoded = decode_outputs(output, scores, control)
        source_binding(expected_commit)
        require(all(sha(ROOT / path) == digest for path, digest in sources.items()), "loaded source bytes changed")
        verify_freeze(family / "inputs_frozen.json")
        verify_originals(lineage["copied"])
        image.official_files(Path(lineage["official_source"]))
        require(sha(Path(plan["parent"]) / "inference/frozen_predictions.json") == plan["parent_prediction_freeze_sha256"],
                "original prediction freeze changed")
        import psutil
        memory = psutil.Process().memory_info()
        memory_end = sparse.check_headroom()
        write(output / "runtime.json", {"source_commit": expected_commit, "source_sha256": sources,
            "profile_sha256": PROFILE_SHA256, "input_freeze_sha256": sha(family / "inputs_frozen.json"),
            "actual_model_state_sha256": initial_state, "adapter": adapter, "cpu_runtime": cpu,
            "memory_before": before, "memory_before_model": before_model, "memory_after_model": after_model,
            "memory_query_minimum": query_memory_minimum, "memory_before_decode": before_decode, "memory_end": memory_end,
            "memory_minimum": minimum_memory([before, before_model, after_model, query_memory_minimum,
                before_decode, memory_end, *[row["memory_minimum"] for row in pools]]),
            "reference_descriptors_sha256": descriptor_digest, "query_descriptors_sha256": _array_sha(queries),
            "canonical_preprocessing": image.PROFILE["image"], "pool_encoding": pools,
            "per_query_native_pixel_sha256": plan["per_query_native_pixel_sha256"],
            "actual_query_times_sha256": _array_sha(control["query_times"]), "decode": decoded,
            "libraries": library_versions(("numpy", "opencv-python", "torch", "torchvision", "pillow")),
            "elapsed_s": time.monotonic()-started, "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
            "peak_private_bytes": getattr(memory, "peak_pagefile", memory.vms), "network_allowed": False,
            "gpu_initialized": False, "ground_truth_opened": False, "scoring_called": False,
            "selected_variant": None, "legacy_reference_encoding_identity": "UNVERIFIED",
            "duplicate_pixel_conflict_ids": plan["duplicate_pixel_conflict_ids"]})
        write(output / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
            "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256, "selected_variant": None,
            "variants": [f"{model}/{variant}" for model in PROFILE["models"] for variant in PROFILE["variants"]],
            "artifacts": {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()},
            "input_freeze_sha256": sha(family / "inputs_frozen.json"), "original_inputs_unchanged": True})
        return {"status": "complete", "reference_count": len(plan["canonical_rows"]),
                "freeze_sha256": sha(output / "frozen_predictions.json"), "selected_variant": None}
    except Exception as error:
        write(output / "failure.json", {"status": "failed", "scoring_has_not_run": True,
            "error": f"{type(error).__name__}: {error}", "elapsed_s": time.monotonic()-started})
        raise
    finally:
        if frames is not None:
            frames.close()
        tempfile.tempdir = previous_temp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ready = commands.add_parser("prepare")
    for name in ("parent", "inventory", "official", "checkpoint", "receipt", "family"):
        ready.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("infer")
    run.add_argument("--family", type=Path, required=True)
    run.add_argument("--expected-commit", required=True)
    run.add_argument("--expected-profile", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.parent, args.inventory, args.official, args.checkpoint, args.receipt, args.family)
    else:
        result = infer(args.family, args.expected_commit, args.expected_profile)
    print(__import__("json").dumps(result, indent=2))


if __name__ == "__main__":
    main()
