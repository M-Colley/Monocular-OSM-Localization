"""Isolated cached-reference replay; no label/evaluation interface.

Prepare copies only. Infer requires an exact clean source pin and8GiB free
commit; it freezes the full fresh matrix and BOTH declared decoder outputs.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

def configure_environment():
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      PYTHONDONTWRITEBYTECODE="1", PROJ_NETWORK="OFF")
    sys.dont_write_bytecode = True


if __name__ == "__main__": configure_environment()
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from monocular_osm.sparse_sequence import PROFILE_SHA256, decode_sparse_tracks, _array_sha, _json_sha
from scripts.plan_sparse_city_replay import header, require, sha

MIN_COMMIT = 8 * 1024**3
MAX_UNION_BYTES, MAX_POOL_BYTES = 512 * 1024**2, 128 * 1024**2


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def relative(value):
    path = Path(str(value).replace("\\", "/"))
    require(not path.is_absolute() and not path.drive and ".." not in path.parts, "escaping relative input path")
    return path


def load(path):
    require(Path(path).stat().st_size <= 64 * 1024**2, "JSON input exceeds the pilot budget")
    return json.loads(Path(path).read_text(encoding="utf8"))


def verify_freeze(path):
    path = Path(path).resolve(strict=True)
    value = load(path)
    require(value.get("scoring_has_not_run") is True, "input was not frozen before evaluation")
    for name, expected in value["artifacts"].items():
        target = (path.parent / relative(name)).resolve(strict=True)
        require(target.is_relative_to(path.parent) and sha(target) == expected, "input freeze changed or escaped its root")
    return value


def copy_verified(source, target, expected, records):
    source = Path(source).resolve(strict=True)
    require(source.is_file() and sha(source) == expected, "source input differs from its registered digest")
    require(not target.exists(), "refuse to overwrite a prepared input")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha(target) == expected, "copy changed a registered input")
    records.append({"source": str(source), "path": str(target.resolve()), "sha256": expected,
                    "bytes": target.stat().st_size})


def verify_originals(records):
    require(all(sha(item["source"]) == item["sha256"] for item in records), "an original copied input changed")


def memory_snapshot():
    import ctypes
    from monocular_osm.assisted_worker import _commit_headroom
    require(os.name == "nt", "actual commit verification currently requires Windows")
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total_phys", "available_phys", "total_pagefile", "available_pagefile",
                                                  "total_virtual", "available_virtual", "extended")]
    status = Memory(); status.length = ctypes.sizeof(status)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)), "cannot verify physical memory headroom")
    return {"available_physical_bytes": int(status.available_phys), "available_commit_bytes": _commit_headroom(),
            "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def check_headroom():
    available = memory_snapshot()
    require(all(isinstance(available[key], int) and available[key] >= MIN_COMMIT for key in
                ("available_physical_bytes", "available_commit_bytes")), "pilot requires verified8GiB physical AND actual commit before heavy allocation")
    return available


def source_identity(expected):
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    require(git("rev-parse", "HEAD") == expected and not git("status", "--porcelain", "--untracked-files=all"),
            "inference requires exact clean committed source, including the new runner/helper")
    return expected


def prepare(plan_path, assets_path, family, *, model_source=None):
    plan_path, assets_path, family = Path(plan_path).resolve(strict=True), Path(assets_path).resolve(strict=True), Path(family).resolve()
    require(not family.exists(), "refuse to overwrite an existing pilot family")
    frozen = verify_freeze(plan_path.parent / "plan_frozen.json")
    plan = load(plan_path)
    require(sha(plan_path) == frozen["plan_sha256"] and plan["decoder_profile_sha256"] == PROFILE_SHA256
            and plan["execution_plan"]["selected_variant"] is None, "registered sparse plan/profile differs")
    assets = load(assets_path)["megaloc"]
    asset_root = Path(model_source or assets["source"]).resolve(strict=True)
    forbidden = [Path(plan["query"]["video"]["path"]).parent, asset_root, *[Path(row["archive"]) for row in plan["pools"]]]
    require(not any(family.is_relative_to(root.resolve()) for root in forbidden), "pilot output must be outside original cache roots")
    family.mkdir(parents=True)
    records = []
    try:
        inputs = family / "inputs"
        copy_verified(plan_path, inputs / "replay_plan.json", frozen["plan_sha256"], records)
        copy_verified(assets_path, inputs / "model_asset_manifest.json", sha(assets_path), records)
        copy_verified(plan["query"]["video"]["path"], inputs / "video.mp4", plan["query"]["video"]["sha256"], records)
        require(0 < len(assets["files"]) <= 64, "model asset family is empty or unbounded")
        for item in assets["files"]:
            name = relative(item["relative"])
            source = (asset_root / name).resolve(strict=True)
            require(source.is_relative_to(asset_root), "model asset escaped its declared source root")
            copy_verified(source, inputs / "models" / name, item["sha256"], records)
        pools = []
        for index, pool in enumerate(plan["pools"]):
            target = inputs / "pools" / f"{index:04d}"
            copy_verified(pool["embeddings"]["path"], target / "embeddings.npz", pool["embeddings"]["sha256"], records)
            copy_verified(pool["metadata"]["path"], target / "metadata.json", pool["metadata"]["sha256"], records)
            pools.append({"index": index, "embeddings": str((target / "embeddings.npz").resolve()),
                          "metadata": str((target / "metadata.json").resolve())})
        verify_originals(records)
        lineage = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "scoring_has_not_run": True,
            "models_have_not_run": True, "plan_sha256": frozen["plan_sha256"], "model_asset_manifest_sha256": sha(assets_path),
            "model_asset_source": str(asset_root), "copied": records, "pools": pools,
            "policy": "all registered calls; copies only; no original-cache writes; no raw pixel-store reads"}
        write(family / "input_lineage.json", lineage)
        artifacts = {str(Path(row["path"]).relative_to(family)): row["sha256"] for row in records}
        artifacts["input_lineage.json"] = sha(family / "input_lineage.json")
        write(family / "inputs_frozen.json", {"scoring_has_not_run": True, "models_have_not_run": True,
              "artifacts": artifacts, "plan_sha256": frozen["plan_sha256"]})
        return lineage
    except Exception as error:
        write(family / "preparation_failed.json", {"status": "failed", "models_have_not_run": True,
              "scoring_has_not_run": True, "error": f"{type(error).__name__}: {error}", "copied": records})
        raise


def load_union(plan, lineage):
    ids = plan["union"]["ref_ids"]
    require(ids == sorted(set(ids)) and _json_sha(ids) == plan["union"]["ref_ids_sha256"]
            and len(ids) == plan["union"]["count"] <= 100000, "registered union IDs are incomplete")
    require(len(lineage["pools"]) == len(plan["pools"]) == plan["call_count"], "a registered pool is missing")
    dimension = plan["pools"][0]["embeddings"]["header"]["shape"][1]
    require(len(ids) * dimension * 4 <= MAX_UNION_BYTES, "complete union exceeds512MiB; no pruning allowed")
    columns = {key: index for index, key in enumerate(ids)}
    union = np.empty((len(ids), dimension), np.float32)
    coordinates, seen, records = np.empty((len(ids), 2), np.float64), np.zeros(len(ids), bool), []
    for index, (pool, prepared) in enumerate(zip(plan["pools"], lineage["pools"])):
        require(prepared["index"] == index, "prepared pool order changed")
        path, declared = Path(prepared["embeddings"]), pool["embeddings"]
        require(sha(path) == declared["sha256"], "prepared embedding pool changed")
        shape = header(path, "emb")
        require(shape == declared["header"] and shape["shape"] == [pool["served_count"], dimension]
                and pool["served_count"] * dimension * 4 <= MAX_POOL_BYTES, "pool header exceeds its frozen bound")
        with np.load(path, allow_pickle=False) as archive:
            require(str(archive["fingerprint"]) == pool["ordered_servable_fingerprint"], "pool fingerprint changed")
            embedding = archive["emb"]
        require(len(pool["ref_ids"]) == len(pool["ref_latlon"]) == len(embedding)
                and _json_sha(pool["ref_ids"]) == pool["ref_ids_sha256"]
                and _array_sha(np.asarray(pool["ref_latlon"], float)) == pool["ref_latlon_sha256"], "pool ID/coordinate inventory is incomplete")
        metadata = load(prepared["metadata"])
        wanted = set(pool["ref_ids"])
        served = [r for r in metadata["refs"] if "mapillary:" + str(r["id"]) in wanted]
        fp = hashlib.sha1(json.dumps([[r["id"], r["lat"], r["lon"]] for r in served]).encode()).hexdigest()
        require(["mapillary:" + str(r["id"]) for r in served] == pool["ref_ids"]
                and [[r["lat"], r["lon"]] for r in served] == pool["ref_latlon"]
                and fp == pool["ordered_servable_fingerprint"], "embedded row IDs differ from their original metadata fingerprint")
        duplicate_max, norm_deviation, added = 0., 0., 0
        for identifier, xy, vector in zip(pool["ref_ids"], pool["ref_latlon"], embedding):
            require(identifier in columns and np.isfinite(vector).all(), "unknown ID or nonfinite cached descriptor")
            norm_deviation = max(norm_deviation, abs(float(np.linalg.norm(vector)) - 1.))
            require(norm_deviation <= 1e-4, "cached descriptor is not normalized; no implicit re-encoding")
            column = columns[identifier]
            if seen[column]:
                require(np.array_equal(coordinates[column], xy), "duplicate ID has conflicting positions")
                delta = float(np.max(np.abs(union[column] - vector)))
                duplicate_max = max(duplicate_max, delta)
                require(delta <= 1e-6, "duplicate ID has conflicting descriptors; refuse whole replay")
            else:
                union[column], coordinates[column], seen[column] = vector, xy, True
                added += 1
        records.append({"index": index, "served_count": len(embedding), "added": added,
                        "duplicate_max_abs": duplicate_max, "max_norm_deviation": norm_deviation})
        del embedding
    require(seen.all() and _array_sha(coordinates) == plan["union"]["ref_latlon_sha256"], "union omits or changes registered positions")
    return union, np.asarray(ids), coordinates, records


def verify_query_pixels(frames, video, query):
    """Independently check every frozen native ordinal and actual PTS."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        require(capture.isOpened(), "cannot independently verify query video")
        first = query["planned_frame_ordinals"][0]
        require(capture.set(cv2.CAP_PROP_POS_FRAMES, first), "video backend cannot seek to the registered first ordinal")
        current, hashes = first, []
        for row, target in enumerate(query["planned_frame_ordinals"]):
            while current < target:
                require(capture.grab(), "query ordinal exceeds decoded video")
                current += 1
            ok, pixels = capture.read(); current += 1
            require(ok and np.array_equal(pixels, frames.frames[row]), "fresh pixels differ from their registered native ordinal")
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            require(np.isfinite(position) and abs(position - target - 1) <= .5, "backend decoded a different native frame ordinal")
            milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
            actual = milliseconds / 1000. if milliseconds is not None and milliseconds > 0 else target / query["fps"]
            require(actual == frames.timestamps[row], "query PTS differs from its independently decoded ordinal")
            hashes.append(_array_sha(pixels))
        return hashes
    finally: capture.release()


@contextmanager
def offline():
    previous, previous_ex = socket.socket.connect, socket.socket.connect_ex
    def refuse(*args, **kwargs): raise RuntimeError("sparse replay is offline; downloads/network are forbidden")
    socket.socket.connect = socket.socket.connect_ex = refuse
    try: yield
    finally: socket.socket.connect, socket.socket.connect_ex = previous, previous_ex


def initialize_query_normalization(kv, torch):
    """Initialize canonical preprocessing used by normal VPR entry points."""
    kv._MEAN = torch.tensor([.485, .456, .406], dtype=torch.float32).reshape(3, 1, 1)
    kv._STD = torch.tensor([.229, .224, .225], dtype=torch.float32).reshape(3, 1, 1)


def forward_queries(frames, union, models):
    configure_environment()
    os.environ["TORCH_HOME"], os.environ["HF_HOME"] = str(models / "torch"), str(models / "hf")
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ[key] = str(models / "hf/hub")
    import cv2
    import torch
    from monocular_osm import kartaview_vpr as kv
    from monocular_osm.stages.poi_runtime import torch_cpu_threads
    from monocular_osm.provenance import library_versions
    torch.set_num_interop_threads(1)
    torch.hub.set_dir(str(models / "torch/hub"))
    cv2.setNumThreads(1)
    query_cache, audit = {}, {}
    with offline(), torch_cpu_threads(4) as runtime:
        initialize_query_normalization(kv, torch)
        before_model_memory = check_headroom()
        kv._resolve_backbone("megaloc", "cpu")
        require(kv.backbone_info() == {"name": "megaloc", "hub": kv.HUB_MEGALOC, "device": "cpu"}, "registered model unavailable; backbone fallback forbidden")
        before = kv._audit_model_state()
        after_model_memory = check_headroom()
        indices = np.arange(len(frames.frames), dtype=np.int64)
        audit["query_indices"] = indices
        scores = kv._query_similarities(frames.frames, indices, torch.from_numpy(union), "cpu", mode="crops",
            batch_size=4, query_cache=query_cache, model_name="megaloc", audit=audit)
        require(len(query_cache) == 1 and not audit.get("issues"), "query descriptor cache/audit is incomplete")
        descriptor = next(iter(query_cache.values())).numpy().copy()
        after = kv._audit_model_state()
        require(before == after, "model state changed during fresh query inference")
        proof = {"cpu_runtime": runtime, "opencv_threads": cv2.getNumThreads(), "backbone": kv.backbone_info(),
            "memory_before_model": before_model_memory, "memory_after_model": after_model_memory,
            "actual_model_state": after, "query_audit": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in audit.items()},
            "query_cache_entries": len(query_cache), "reference_encoding_identity": "unverified legacy revision/preprocessing; cached vectors only", "network_allowed": False}
        proof["libraries"] = library_versions(("numpy", "opencv-python", "torch", "torchvision", "timm", "huggingface-hub", "safetensors"))
        proof["query_preprocessing"] = {"mode": "crops", "views": 3, "crop": "three evenly placed native square views along long axis",
            "color": "BGR to RGB", "size": [512, 512], "resize": "opencv INTER_LINEAR", "scale": "divide by255",
            "mean": [.485, .456, .406], "std": [.229, .224, .225], "descriptor_normalization": "L2",
            "aggregation": "maximum cosine over the three crop descriptors for each reference"}
        return np.ascontiguousarray(scores, dtype=np.float32), descriptor, proof


@offline()
def infer(family, expected_commit):
    family = Path(family).resolve(strict=True)
    source_identity(expected_commit)
    verify_freeze(family / "inputs_frozen.json")
    plan, lineage = load(family / "inputs/replay_plan.json"), load(family / "input_lineage.json")
    require(plan["decoder_profile_sha256"] == PROFILE_SHA256, "decoder differs from its frozen plan")
    before_union_memory = check_headroom()
    output = family / "inference"
    require(not output.exists(), "refuse to rerun or overwrite an inference family")
    output.mkdir()
    frames, previous_temp, started = None, tempfile.tempdir, time.monotonic()
    try:
        from monocular_osm.provenance import snapshot_sources
        sources = snapshot_sources([ROOT / "main.py", Path(__file__), ROOT / "scripts/plan_sparse_city_replay.py",
                                    *sorted((ROOT / "monocular_osm").rglob("*.py"))], output)
        union, ids, coordinates, pools = load_union(plan, lineage)
        after_union_memory = check_headroom()
        import cv2
        from monocular_osm.frame_extraction import extract_frames
        cv2.setNumThreads(1)
        temp = family / "frame_tmp"; temp.mkdir(); tempfile.tempdir = str(temp)
        query = plan["query"]
        frames = extract_frames(family / "inputs/video.mp4", stride=query["stride"], max_frames=None,
            start_sec=query["registered_window"][0], end_sec=query["registered_window"][1],
            memory_mb=128., max_width=None, sample_count=80)
        require(len(frames.frames) == 80 and frames.sampling["dense_count"] == query["dense_count"]
                and frames.sampling["dense_indices"] == query["query_dense_indices"]
                and float(frames.fps) == query["fps"]
                and all(list(image.shape[:2]) == query["native_shape_hw"] for image in frames.frames), "fresh extraction changed its registered native query grid")
        pixel_hashes = verify_query_pixels(frames, family / "inputs/video.mp4", query)
        scores, descriptors, runtime = forward_queries(frames, union, family / "inputs/models")
        runtime.update(memory_before_union=before_union_memory, memory_after_union=after_union_memory)
        times = np.asarray(frames.timestamps, dtype=np.float64)
        require(scores.shape == (80, len(ids)) and descriptors.shape == (240, union.shape[1]), "fresh crop matrix/descriptor grid is incomplete")
        freeze_predictions(output, family, expected_commit, sources, plan, lineage, union, ids, coordinates,
                           pools, pixel_hashes, scores, descriptors, times, runtime, started)
        return {"status": "complete", "freeze_sha256": sha(output / "frozen_predictions.json"), "reference_count": len(ids)}
    except Exception as error:
        write(output / "failure.json", {"status": "failed", "scoring_has_not_run": True,
              "error": f"{type(error).__name__}: {error}", "elapsed_s": time.monotonic() - started})
        raise
    finally:
        if frames is not None: frames.close()
        tempfile.tempdir = previous_temp


def freeze_predictions(output, family, expected_commit, sources, plan, lineage, union, ids, coordinates,
                       pools, pixel_hashes, scores, descriptors, times, runtime, started):
    np.savez(output / "full_evidence.npz", raw_cosines=scores, query_times=times,
        query_indices=np.arange(80, dtype=np.int64), frame_ordinals=np.asarray(plan["query"]["planned_frame_ordinals"], np.int64),
        ref_ids=ids, ref_latlon=coordinates)
    np.savez(output / "query_descriptors.npz", descriptors=descriptors)
    np.savez(output / "reference_union.npz", embeddings=union, ref_ids=ids, ref_latlon=coordinates)
    predictions = decode_sparse_tracks(scores, ids, coordinates, times)
    for variant in ("raw", "reference_mean"):
        write(output / ("prediction_" + variant + ".json"), predictions["outputs"][variant])
    write(output / "predictions.json", predictions)
    write(output / "runtime_and_lineage.json", {"source_commit": expected_commit, "source_sha256": sources,
        "input_freeze_sha256": sha(family / "inputs_frozen.json"), "input_lineage_sha256": sha(family / "input_lineage.json"),
        "per_query_native_pixel_sha256": pixel_hashes, "actual_query_times_sha256": _array_sha(times),
        "reference_embeddings_sha256": _array_sha(union), "pool_replay": pools, "runtime": runtime,
        "elapsed_s": time.monotonic() - started, "labels_opened": False, "scoring_called": False, "selected_variant": None})
    source_identity(expected_commit)
    verify_freeze(family / "inputs_frozen.json")
    verify_originals(lineage["copied"])
    artifacts = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}
    write(output / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
        "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256, "selected_variant": None,
        "variants": ["raw", "reference_mean"], "artifacts": artifacts,
        "input_freeze_sha256": sha(family / "inputs_frozen.json"), "original_copied_inputs_unchanged": True})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare")
    setup.add_argument("--plan", type=Path, required=True)
    setup.add_argument("--assets", type=Path, required=True)
    setup.add_argument("--family", type=Path, required=True)
    setup.add_argument("--model-source", type=Path)
    run = sub.add_parser("infer")
    run.add_argument("--family", type=Path, required=True)
    run.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    result = prepare(args.plan, args.assets, args.family, model_source=args.model_source) if args.command == "prepare" else infer(args.family, args.expected_commit)
    print(json.dumps(result if args.command == "infer" else {"status": "prepared_only", "files": len(result["copied"])}))


if __name__ == "__main__": main()
