"""Fixed-profile native-OCR POI image-only pilot; score in a separate process.

No VO input or old prediction/clue pool is accepted. One literal branch must
be produced from freshly sampled native OCR and a public city inventory.
All source/assets/inputs and the final prediction freeze before labels open.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import quote, quote_plus
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  TRANSFORMERS_OFFLINE="1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import numpy as np

RUNTIME = {"cpu_threads": 4, "batch_size": 2, "frame_memory_mb": 128,
           "ocr_threads": [2, 1], "minimum_free_commit_bytes": 8 * 1024**3,
           "query_stride": 1, "ocr_min_len": 2, "ocr_min_confidence": .3}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024**2): result.update(block)
    return result.hexdigest()


def safe(value):
    result = json.dumps(value, allow_nan=False)
    token = os.environ.get("MLY_TOKEN")
    if token:
        for spelling in (token, quote(token, safe=""), quote_plus(token)):
            result = result.replace(spelling, "[REDACTED]")
    return json.loads(result)


def available_commit():
    class Performance(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            (name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
    record = Performance()
    record.cb = ctypes.sizeof(record)
    if not ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo(ctypes.byref(record), record.cb):
        raise OSError("cannot measure actual Windows commit headroom")
    return (record.CommitLimit - record.CommitTotal) * record.PageSize


def require_memory():
    amount = available_commit()
    if amount < RUNTIME["minimum_free_commit_bytes"]:
        raise RuntimeError("pilot requires at least 8 GiB free actual Windows commit before launch")
    return amount


def pinned_source(args):
    source = args.source_root.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if revision != args.source_commit or dirty:
        raise ValueError("source checkout must be completely clean at the declared pin")
    sys.path.insert(0, str(source))
    from monocular_osm import poi_image_track as image
    if not Path(image.__file__).resolve().is_relative_to(source):
        raise ValueError("image decoder import escaped pinned source")
    return source, image


def copy_assets(args):
    plan, copied = read(args.assets), []
    for family, destination in (("megaloc", args.run / "models"),
                                ("rapidocr", args.run / "runtime_packages/rapidocr")):
        root = Path(plan[family]["source"]).resolve(strict=True)
        for record in plan[family]["files"]:
            original = (root / record["relative"]).resolve(strict=True)
            target = (destination / record["relative"]).resolve()
            if not original.is_relative_to(root) or not target.is_relative_to(args.run.resolve()):
                raise ValueError("asset copy escaped its declared source or isolated run")
            if sha(original) != record["sha256"] or original.stat().st_size != record["bytes"]:
                raise ValueError("verified public model asset changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if sha(target) != record["sha256"]:
                raise ValueError("model asset copy changed")
            copied.append({**record, "family": family, "source_relative": record["relative"],
                           "relative": str(target.relative_to(args.run))})
    sys.path.insert(0, str(args.run / "runtime_packages"))
    os.environ.update(TORCH_HOME=str(args.run / "models/torch"), HF_HOME=str(args.run / "models/hf"),
                      HUGGINGFACE_HUB_CACHE=str(args.run / "models/hf/hub"))
    return copied


@contextmanager
def monitor(run):
    import psutil
    process, stop = psutil.Process(), threading.Event()
    peak = {"rss_bytes": 0, "private_bytes": 0, "native_peak_rss_sum_bytes": 0,
            "native_peak_private_sum_bytes": 0}
    started = time.perf_counter()
    def watch():
        with (run / "resources.jsonl").open("w", encoding="utf-8") as handle:
            while not stop.is_set():
                values = {key: 0 for key in peak}
                for child in [process, *process.children(recursive=True)]:
                    try: info = child.memory_info()
                    except (psutil.NoSuchProcess, psutil.AccessDenied): continue
                    values["rss_bytes"] += info.rss
                    values["private_bytes"] += getattr(info, "private", info.vms)
                    values["native_peak_rss_sum_bytes"] += getattr(info, "peak_wset", info.rss)
                    values["native_peak_private_sum_bytes"] += getattr(info, "peak_pagefile", info.vms)
                for key, value in values.items(): peak[key] = max(peak[key], value)
                handle.write(json.dumps({"elapsed_s": time.perf_counter()-started,
                    "available_commit_bytes": available_commit(), **values}) + "\n")
                handle.flush()
                stop.wait(1.)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try: yield peak
    finally:
        stop.set()
        thread.join()
        write(run / "resources.json", {"peak": peak, "elapsed_s": time.perf_counter()-started})


def native_ocr(args, image, video, video_sha, shape):
    from monocular_osm import scene_text
    from monocular_osm.stages.poi_runtime import pinned_reader, reader_spec, validate_reader_record
    sampled = []
    def frames(path, start, end, interval):
        for timestamp, pixels in scene_text._sample_frames(path, start, end, interval,
                                                           sample_count=image.PROFILE["ocr_samples"]):
            if pixels.shape[:2] != shape:
                raise ValueError("OCR pixels must remain on the actual native query image grid")
            contiguous = np.ascontiguousarray(pixels)
            digest = hashlib.sha256(str((pixels.shape, pixels.dtype.str)).encode())
            digest.update(memoryview(contiguous).cast("B"))
            sampled.append({"t_sec": timestamp, "shape": list(pixels.shape), "pixels_sha256": digest.hexdigest()})
            yield timestamp, pixels
    spec = reader_spec("rapidocr", model_dir=args.run / "runtime_packages/rapidocr/models")
    with pinned_reader(spec) as (reader, runtime):
        detections = scene_text.extract_scene_text(video, start_sec=args.start, end_sec=args.end,
            sample_count=image.PROFILE["ocr_samples"], min_len=RUNTIME["ocr_min_len"],
            min_confidence=RUNTIME["ocr_min_confidence"], use_gpu=False, super_res=False,
            engine="rapidocr", frame_reader=frames, ocr_reader=reader, cache_path=args.run / "ocr_cache.json")
    validate_reader_record(runtime)
    if runtime.get("read_calls") != len(sampled) or len(sampled) != image.PROFILE["ocr_samples"]:
        raise ValueError("real native OCR calls are unproven")
    detections = [asdict(d) for d in detections]
    provenance = {"source": "same_video_native_frames", "native_bbox_grid": True,
        "video_sha256": video_sha, "sample_count": len(sampled), "frame_shape_hw": list(shape),
        "window": [args.start, args.end], "super_res": False, "use_gpu": False,
        "position_stamps_rejected": True, "reader": runtime, "sampled_frames": sampled}
    write(args.run / "native_ocr.json", {"provenance": provenance, "detections": detections})
    return detections, provenance


def infer(args):
    preflight = require_memory()
    args.run = args.run.resolve()
    if args.run.drive.upper() != "E:" or args.run.is_relative_to(args.source_root.resolve()):
        raise ValueError("fresh run must be separate from source and on the isolated E drive")
    if (not np.isfinite(args.start) or args.start < 0
            or (args.end is not None and (not np.isfinite(args.end) or args.end <= args.start))):
        raise ValueError("registered video window is invalid")
    source, image = pinned_source(args)
    args.run.mkdir(parents=True, exist_ok=False)
    with monitor(args.run):
        try:
            run_inference(args, source, image, preflight)
        except BaseException as exc:
            write(args.run / "failed.json", {"type": type(exc).__name__, "error": safe(str(exc)),
                                             "ground_truth_opened": False})
            raise RuntimeError(safe(str(exc))) from None
    freeze_outputs(args.run)  # Resource files are closed before their hashes freeze.


def run_inference(args, source, image, preflight):
    import cv2
    from monocular_osm.frame_extraction import extract_frames
    from monocular_osm.poi_clues import build_poi_hypotheses
    assets = copy_assets(args)
    (args.run / "inputs").mkdir()
    video, inventory_path = args.run / "inputs/video.mp4", args.run / "inputs/inventory.json"
    for original, target in ((args.video, video), (args.inventory, inventory_path)):
        expected = sha(original)
        shutil.copyfile(original, target)
        if sha(target) != expected: raise ValueError("public input copy changed")
    inventory = read(inventory_path)
    if inventory.get("city") != args.city or inventory.get("scope") != "named_city_bbox":
        raise ValueError("inventory must belong to the supplied named city")
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    try: fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.
    finally: cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("actual encoded video FPS is required; no assumed timebase")
    registration = {"experimental": True, "validation_status": "posthoc_dev_unvalidated",
        "source_commit": args.source_commit, "source_root": str(source), "runner_sha256": sha(__file__),
        "profile": dict(image.PROFILE), "profile_sha256": image.PROFILE_SHA256, "runtime": RUNTIME,
        "video": {"path": str(video), "sha256": sha(video)}, "city": args.city,
        "inventory": {"path": str(inventory_path), "sha256": sha(inventory_path)},
        "window": [args.start, args.end], "model_assets": assets, "asset_plan_sha256": sha(args.assets),
        "input_policy": {"vo": False, "old_predictions": False, "old_reference_pool": False,
                         "native_ocr": True, "public_reference_positions": True, "ground_truth": False},
        "preflight_available_commit_bytes": preflight}
    registration["dependencies"] = {}
    for package in ("numpy", "opencv-python", "pyproj", "rapidocr", "onnxruntime", "torch", "huggingface_hub", "safetensors"):
        try: registration["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: registration["dependencies"][package] = None
    snapshot = args.run / "source_snapshot"
    snapshot.mkdir()
    shutil.copyfile(__file__, snapshot / Path(__file__).name)
    names = ["poi_image_track.py", "poi_clues.py", "scene_text.py", "frame_extraction.py",
             "kartaview_vpr.py", "geo.py", "metrics.py", "stages/poi_runtime.py"]
    registration["source_files"] = {}
    for name in names:
        original = source / "monocular_osm" / name
        target = snapshot / "monocular_osm" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        registration["source_files"][name] = sha(target)
    write(args.run / "experiment.json", registration)  # Before OCR/model inference.
    compute_queries(args, image, registration, inventory, extract_frames, build_poi_hypotheses)


def compute_queries(args, image, registration, inventory, extract, build_clues):
    import gc
    import tempfile
    temporary = args.run / "frame_tmp"
    temporary.mkdir()
    tempfile.tempdir = str(temporary)
    frames = extract(Path(registration["video"]["path"]), stride=RUNTIME["query_stride"], max_frames=None,
        sample_count=image.PROFILE["max_queries"], start_sec=args.start, end_sec=args.end,
        memory_mb=RUNTIME["frame_memory_mb"])
    try:
        if len(frames.frames) < 2:
            raise ValueError("image-only decoding needs at least two real native frames")
        digest = hashlib.sha256()
        for pixels in frames.frames:
            pixels = np.ascontiguousarray(pixels)
            digest.update(str((pixels.shape, pixels.dtype.str)).encode())
            digest.update(memoryview(pixels).cast("B"))
        times = np.asarray(frames.timestamps, float)
        write(args.run / "query_input.json", {"timestamps": times.tolist(), "fps": frames.fps,
            "sampling": frames.sampling, "pixels_sha256": digest.hexdigest(),
            "native_frame_shape": list(frames.frames[0].shape)})
        detections, native = native_ocr(args, image, Path(registration["video"]["path"]),
                                      registration["video"]["sha256"], frames.frames[0].shape[:2])
        clues = build_clues(detections, inventory)
        write(args.run / "clues.json", clues)
        if len(clues["candidate_clusters"]) != 1:
            write(args.run / "prediction.json", {"experimental": True, "status": "unavailable",
                "reason": "exactly_one_native_literal_branch_required", "uses_rigid_vo": False,
                "branch_count": len(clues["candidate_clusters"]), "profile_sha256": image.PROFILE_SHA256})
            return
        plan = image.plan_single_branch(clues, native)
        write(args.run / "plan.json", plan)  # Fixed before acquiring any references.
        gc.collect()  # Release the OCR session before loading the retrieval model.
        acquire_and_decode(args, image, registration, frames.frames, times, plan)
    finally:
        frames.close()


def acquire_and_decode(args, image, registration, frames, times, plan):
    from monocular_osm import kartaview_vpr as kv
    from monocular_osm.stages.poi_runtime import torch_cpu_threads
    import torch
    require_memory()
    repository = kv.HUB_MEGALOC.replace("/", "_").replace(":", "_")
    if not (args.run / "models/torch/hub" / repository).is_dir():
        raise ValueError("required pinned model repository is not among verified copied assets")
    def no_model_download(*args, **kwargs):
        raise RuntimeError("model downloads disabled; cached verified assets required")
    torch.hub.download_url_to_file = no_model_download
    torch.set_num_interop_threads(1)
    query_cache = {}
    with torch_cpu_threads(RUNTIME["cpu_threads"]) as runtime:
        evidence = kv.kartaview_vpr_evidence(frames, times, tuple(plan["center"]),
            radius_m=image.PROFILE["radius_m"], cap=image.PROFILE["cap"], n_query=image.PROFILE["max_queries"],
            model_name=image.PROFILE["model"], source="mapillary", device="cpu", query_mode="crops",
            cache_dir=str(args.run / "references"), query_batch_size=RUNTIME["batch_size"], query_cache=query_cache)
    write(args.run / "retrieval_runtime.json", {"runtime": runtime, "query_cache_entries": len(query_cache)})
    if evidence is None:
        write(args.run / "prediction.json", {"experimental": True, "status": "unavailable",
            "uses_rigid_vo": False, "reason": safe(kv.LAST_FAILURE or "no reference evidence"),
            "profile_sha256": image.PROFILE_SHA256})
        return
    validate_query_identity(evidence, times)
    evidence["provenance"] = safe(evidence["provenance"])
    np.savez_compressed(args.run / "reference_evidence.npz", **{key: evidence[key] for key in (
        "query_indices", "query_times", "ref_latlon", "ref_ids", "similarities")},
        metadata=np.array(json.dumps(evidence["provenance"])))
    prediction = image.decode_single_branch_image_track(evidence, plan,
        query_video_sha256=registration["video"]["sha256"])
    write(args.run / "prediction.json", prediction)
    print(f"image-only prediction saved: {prediction['point_count']}queries, "
          f"{len(evidence['ref_ids'])}references; no labels opened", flush=True)


def validate_query_identity(evidence, timestamps):
    indices = np.asarray(evidence["query_indices"])
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != len(timestamps) or not np.array_equal(indices, np.arange(len(timestamps)))
            or not np.array_equal(np.asarray(evidence["query_times"]), np.asarray(timestamps)[indices])):
        raise ValueError("retrieval query grid must exactly match every freshly sampled input frame/time")


def freeze_outputs(run):
    if (run / "frozen_prediction.json").exists():
        raise ValueError("an existing prediction freeze cannot be rewritten")
    references = [{"relative": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha(path)}
                  for path in sorted((run / "references").rglob("*")) if path.is_file()]
    write(run / "reference_snapshot.json", references)
    artifacts = {path.name: sha(path) for path in run.iterdir() if path.is_file()
                 and path.name not in {"frozen_prediction.json", "scores.json"}}
    write(run / "frozen_prediction.json", {"ground_truth_opened": False,
        "prediction_available": read(run / "prediction.json").get("status") == "complete",
        "artifacts": artifacts})


def verify_frozen(run):
    frozen = read(run / "frozen_prediction.json")
    required = {"experiment.json", "prediction.json", "query_input.json", "clues.json",
                "native_ocr.json", "resources.json", "reference_snapshot.json"}
    if (frozen.get("ground_truth_opened") is not False
            or not required.issubset(frozen.get("artifacts", {}))):
        raise ValueError("inference must be completely frozen before scoring")
    for relative, expected in frozen["artifacts"].items():
        path = (run / relative).resolve(strict=True)
        if not path.is_relative_to(run.resolve()) or sha(path) != expected:
            raise ValueError("frozen inference artifact changed")
    experiment = read(run / "experiment.json")
    for key in ("video", "inventory"):
        if sha(experiment[key]["path"]) != experiment[key]["sha256"]:
            raise ValueError("frozen input changed")
    for name, expected in experiment["source_files"].items():
        if sha(run / "source_snapshot/monocular_osm" / name) != expected:
            raise ValueError("frozen source changed")
    if (sha(run / "source_snapshot" / Path(__file__).name) != experiment["runner_sha256"]
            or sha(__file__) != experiment["runner_sha256"]):
        raise ValueError("frozen runner source changed")
    for record in [*experiment["model_assets"], *read(run / "reference_snapshot.json")]:
        path = (run / record["relative"]).resolve(strict=True)
        if not path.is_relative_to(run.resolve()) or sha(path) != record["sha256"]:
            raise ValueError("frozen public asset/reference cache changed")
    return experiment, read(run / "prediction.json")


def scoring_truth(waypoints, window):
    """Offline geodesic boundary interpolation; never extrapolate labels."""
    from monocular_osm.geo import WGS84
    times = np.array([point["t_sec"] for point in waypoints], float)
    coords = np.array([[point["lat"], point["lon"]] for point in waypoints], float)
    if (len(times) < 2 or coords.shape != (len(times), 2) or not np.isfinite(times).all()
            or not np.isfinite(coords).all() or np.any(np.diff(times) <= 0)
            or np.any(np.abs(coords[:, 0]) > 90) or np.any(np.abs(coords[:, 1]) > 180)):
        raise ValueError("scoring needs finite ordered WGS84 waypoints")
    start, end = window
    end = times[-1] if end is None else end
    if not times[0] <= start < end <= times[-1]:
        raise ValueError("registered analysis window must be covered by labels")
    def at(timestamp):
        index = min(np.searchsorted(times, timestamp, side="right") - 1, len(times) - 2)
        a, b = coords[index:index + 2]
        bearing, _, distance = WGS84.inv(a[1], a[0], b[1], b[0])
        fraction = (timestamp - times[index]) / (times[index + 1] - times[index])
        lon, lat, _ = WGS84.fwd(a[1], a[0], bearing, distance * fraction)
        return [lat, lon]
    target_times = np.r_[start, times[(times > start) & (times < end)], end]
    return target_times, np.array([at(timestamp) for timestamp in target_times]), at


def score(args):
    experiment, prediction = verify_frozen(args.run)  # Every input verified before labels.
    target = args.run / "scores.json"
    if target.exists():
        raise ValueError("offline scores already exist; frozen runs are not overwritten")
    sys.path.insert(0, experiment["source_root"])
    from monocular_osm.metrics import local_transformer, project, score_route
    truth = read(args.gt)["waypoints"]  # First label read in this process.
    times, route, at = scoring_truth(truth, experiment["window"])
    result = {"status": prediction["status"], "analysis_window": [float(times[0]), float(times[-1])]}
    if prediction["status"] == "complete":
        query = np.asarray(prediction["route_timestamps"], float)
        if query[0] < times[0] or query[-1] > times[-1]:
            raise ValueError("prediction timestamps extend outside the covered scoring window")
        predicted = np.asarray(prediction["route_latlon"], float)
        expected = np.array([at(timestamp) for timestamp in query])
        transformer = local_transformer(*route.mean(axis=0))
        error = np.linalg.norm(project(predicted, transformer) - project(expected, transformer), axis=1)
        result.update(score_route(predicted, route).to_dict(), mean_time_error_m=float(error.mean()))
    write(target, {"result": result, "ground_truth_sha256": sha(args.gt),
        "prediction_freeze_sha256": sha(args.run / "frozen_prediction.json"),
        "caveat": "Post-hoc DEV prototype. Sparse label chords bound scoring accuracy; no generalization claim."})
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inf = commands.add_parser("infer", help="Fresh native OCR and isolated live public-reference retrieval")
    inf.add_argument("--source-root", type=Path, required=True)
    inf.add_argument("--source-commit", required=True)
    inf.add_argument("--run", type=Path, required=True)
    inf.add_argument("--video", type=Path, required=True)
    inf.add_argument("--city", required=True)
    inf.add_argument("--inventory", type=Path, required=True)
    inf.add_argument("--assets", type=Path, required=True)
    inf.add_argument("--start", type=float, default=0.)
    inf.add_argument("--end", type=float)
    scr = commands.add_parser("score", help="Verify frozen inference before opening offline labels")
    scr.add_argument("--run", type=Path, required=True)
    scr.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args(argv)
    (infer if args.command == "infer" else score)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
