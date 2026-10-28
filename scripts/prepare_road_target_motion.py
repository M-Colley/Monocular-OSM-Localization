"""Freeze fresh native AUTO-grid motion for experimental target alignment.

Registered video/window come from the frozen uniform OCR input plan, never
from target coordinates, a previous route, or labels. No OCR/model/network
work occurs. The old unsigned VO cache is deliberately not loaded.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".part.json")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_family(freeze_path, expected):
    freeze_path = Path(freeze_path).resolve(strict=True)
    if sha(freeze_path) != expected:
        raise ValueError("frozen input manifest changed")
    root = freeze_path.parent
    manifest = read(freeze_path)
    for relative, digest in manifest["artifacts"].items():
        path = (root/relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("frozen public-map/observation artifact changed")
    return manifest


def verified_inputs(pbf_family, pbf_sha, input_plan, plan_sha, video):
    verify_family(Path(pbf_family)/"frozen_pbf_probe.json", pbf_sha)
    if sha(input_plan) != plan_sha:
        raise ValueError("registered input-only OCR plan changed")
    acquisition = read(Path(pbf_family)/"acquisition_plan.json")
    ocr = read(Path(pbf_family)/"original_ocr.json")
    video_sha = sha(video)
    rows = [row for row in read(input_plan)["clips"] if row.get("video", {}).get("sha256") == video_sha]
    if (len(rows) != 1 or rows[0]["city"] != acquisition["public_city"]
            or rows[0]["window"] != ocr["window"] or Path(video).stat().st_size != rows[0]["video"]["bytes"]
            or sha(Path(pbf_family)/"original_ocr.json") != acquisition["input_manifest"]["ocr"]["sha256"]):
        raise ValueError("same-video native OCR/city/window identity is required")
    return {"video_sha256": video_sha, "video_bytes": Path(video).stat().st_size,
            "city": rows[0]["city"], "window": rows[0]["window"], "input_plan_sha256": plan_sha,
            "pbf_freeze_sha256": pbf_sha, "ocr_sha256": sha(Path(pbf_family)/"original_ocr.json"),
            "all_targets_sha256": sha(Path(pbf_family)/"all_junction_targets.json")}


def windows_memory():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_commit", "avail_commit", "total_virtual", "avail_virtual", "avail_extended")]]
    value = Memory(); value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("cannot measure actual Windows RAM and commit availability")
    return {"available_physical_gib": value.avail_phys/2**30, "available_commit_gib": value.avail_commit/2**30}


def require_headroom(memory):
    values = [memory["available_physical_gib"], memory["available_commit_gib"]]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values) or min(values) < 8.:
        raise MemoryError("fresh motion requires both physical and actual free commit >=8GiB")


def verify_source(source, commit, hashes=None):
    source = Path(source).resolve(strict=True)
    executed = Path(__file__).resolve(strict=True)
    declared = (source/"scripts/prepare_road_target_motion.py").resolve(strict=True)
    if executed != declared or sha(executed) != sha(declared):
        raise ValueError("executed motion runner is not the declared pinned script")
    if (subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != commit
            or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()):
        raise ValueError("fresh motion requires the exact clean declared source pin")
    for relative, expected in (hashes or {}).items():
        if sha(source/relative) != expected:
            raise ValueError("motion source changed during estimation")
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name == "monocular_osm" or name.startswith("monocular_osm."):
            if path is None or not Path(path).resolve().is_relative_to(source):
                raise ValueError("loaded motion module escaped the pinned source")


def verify_native_grid(frames, video, *, fps, start, end, stride, total_frames):
    """Bind actual native ordinals independently; PTS*FPS is invalid for VFR."""
    import cv2
    import numpy as np
    from monocular_osm.road_target_alignment import _array_sha
    first = int(start*fps)
    stop = min(total_frames, int(end*fps))
    ordinals = np.arange(first, stop, stride, dtype=np.int64)
    if len(ordinals) != len(frames.frames):
        raise ValueError("native AUTO frame count differs from registered ordinal grid")
    capture = cv2.VideoCapture(str(video)); pixel_hashes = []
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError("native decoder cannot seek to the registered first ordinal")
        current = first
        for index, target in enumerate(ordinals):
            while current < target:
                if not capture.grab():
                    raise ValueError("native ordinal exceeds independently decoded video")
                current += 1
            ok, pixels = capture.read(); current += 1
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not ok or not np.isfinite(position) or abs(position-int(target)-1) > .5:
                raise ValueError("native decoder returned a different ordinal")
            milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
            actual_time = milliseconds/1000. if milliseconds is not None and milliseconds > 0 else int(target)/fps
            if actual_time != frames.timestamps[index] or not np.array_equal(pixels, frames.frames[index]):
                raise ValueError("native ordinal pixels/actual PTS differ from motion inputs")
            pixel_hashes.append(_array_sha(pixels))
    finally:
        capture.release()
    return ordinals, np.asarray(pixel_hashes)


def prepare(args):
    source = args.source_root.resolve(strict=True)
    verify_source(source, args.source_commit)
    sys.path.insert(0, str(source))
    import cv2
    import numpy as np
    import psutil
    from monocular_osm.cli import _auto_frame_stride
    from monocular_osm.frame_extraction import extract_frames
    from monocular_osm.visual_odometry import default_intrinsics, estimate_trajectory, VO_ALGO_VERSION
    from monocular_osm.stages.trajectory import _vo_input_signature, _save_vo_cache
    from monocular_osm.road_target_alignment import RoadTargetConfig, _array_sha
    if not Path(sys.modules["monocular_osm.road_target_alignment"].__file__).resolve().is_relative_to(source):
        raise ValueError("motion dependencies escaped pinned source")
    inputs = verified_inputs(args.pbf_family, args.pbf_freeze_sha256, args.input_plan,
                             args.input_plan_sha256, args.video)
    memory = windows_memory(); require_headroom(memory)
    output = args.output.resolve()
    if output.drive.upper() != "E:" or output.is_relative_to(source):
        raise ValueError("use a new isolated E motion family")
    output.mkdir(parents=True, exist_ok=False)
    temporary = output/"frame_tmp"; temporary.mkdir()
    tempfile.tempdir = str(temporary)
    before_threads = cv2.getNumThreads(); cv2.setNumThreads(1); cv2.setRNGSeed(0)
    cap = cv2.VideoCapture(str(args.video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    start, end = inputs["window"]
    if not np.isfinite(fps) or fps <= 0 or count <= 0 or min(width, height) <= 0 or end is None or not 0 <= start < end:
        raise ValueError("registered native finite video/window required")
    stride = _auto_frame_stride(end-start, fps)
    cfg = SimpleNamespace(vo_start_sec=start, vo_end_sec=end, frame_stride=stride, max_frames=None,
                          frame_max_width=None, frame_memory_mb=128., vo_workers=2)
    source_files = ["scripts/prepare_road_target_motion.py", *[str(path.relative_to(source))
                    for path in sorted((source/"monocular_osm").rglob("*.py"))]]
    hashes = {}
    for relative in source_files:
        target = output/"source_snapshot"/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, target); hashes[relative] = sha(target)
    registration = {"experimental": True, "ground_truth_used": False, "source_commit": args.source_commit,
        "sources": hashes, "inputs": inputs, "video_path": str(args.video.resolve()), "fps": fps,
        "container_frame_count": count, "native_shape_hw": [height, width], "stride": stride,
        "auto_rule": "unmodified CLI _auto_frame_stride(duration, actual_fps)", "frame_memory_mb": 128.,
        "max_frames": None, "max_width": None, "vo_workers": 2, "opencv_threads": 1, "seed": 0,
        "intrinsics": "default_intrinsics(native width,height,hfov70)", "VO_ALGO_VERSION": VO_ALGO_VERSION,
        "legacy_cache_used": False, "network_used": False, "new_ocr_models_or_references": False,
        "search_profile_predeclared": RoadTargetConfig().__dict__, "resource_preflight": memory}
    write(output/"motion_registration.json", registration)
    started = time.monotonic(); process = psutil.Process(); frames = None
    try:
        frames = extract_frames(args.video, stride=stride, max_frames=None, start_sec=start, end_sec=end,
                                memory_mb=128., max_width=None)
        if frames.frames[0].shape[:2] != (height, width) or len(frames.frames) > RoadTargetConfig().max_motion_poses:
            raise ValueError("native decoded geometry/frame count differs from bounded registration")
        K = default_intrinsics(width, height)
        signature = _vo_input_signature(cfg, args.video, frames)
        write(output/"pixel_input_signature.json", json.loads(signature))
        ordinals, pixel_hashes = verify_native_grid(frames, args.video, fps=fps, start=start, end=end,
                                                    stride=stride, total_frames=count)
        require_headroom(windows_memory())
        trajectory = estimate_trajectory(frames.frames, K=K, n_workers=2)
        times = np.asarray(frames.timestamps, dtype=np.float64)
        _save_vo_cache(output/"signed_vo_cache.npz", trajectory, signature)
        np.savez_compressed(output/"native_motion.npz", centers=trajectory.centers, xz=trajectory.xz,
            valid=trajectory.valid, n_inliers=np.asarray(trajectory.n_inliers), rotations=trajectory.rotations,
            translations=trajectory.translations, timestamps=times, frame_ordinals=ordinals,
            native_frame_pixels_sha256=pixel_hashes, K=K)
        if sha(args.video) != inputs["video_sha256"]:
            raise ValueError("same-video pixels changed during native motion estimation")
        memory = process.memory_info()
        summary = {"status": "complete", "n_frames": len(times), "valid_fraction": float(trajectory.valid.mean()),
            "n_valid": int(trajectory.valid.sum()), "timestamps_sha256": _array_sha(times),
            "frame_ordinals_sha256": _array_sha(ordinals), "native_frame_pixels_sha256": _array_sha(pixel_hashes),
            "motion_npz_sha256": sha(output/"native_motion.npz"), "signed_cache_sha256": sha(output/"signed_vo_cache.npz"),
            "native_first_last_times": [float(times[0]), float(times[-1])], "elapsed_s": time.monotonic()-started,
            "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss), "peak_private_bytes": getattr(memory, "peak_pagefile", memory.vms),
            "torch_imported": "torch" in sys.modules, "ground_truth_used": False, "no_predictions_yet": True}
        write(output/"motion_summary.json", summary)
    except Exception as error:
        write(output/"failed.json", {"status": "unknown", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if frames is not None:
            frames.frames.close()
        cv2.setNumThreads(before_threads)
    verify_source(source, args.source_commit, hashes)
    verified_inputs(args.pbf_family, args.pbf_freeze_sha256, args.input_plan, args.input_plan_sha256, args.video)
    write(output/"motion_frozen.json", {"ground_truth_used": False, "no_predictions_yet": True,
        "artifacts": {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file() and path.name != "motion_frozen.json"}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--source-commit", required=True)
    parser.add_argument("--pbf-family", type=Path, required=True); parser.add_argument("--pbf-freeze-sha256", required=True)
    parser.add_argument("--input-plan", type=Path, required=True); parser.add_argument("--input-plan-sha256", required=True)
    parser.add_argument("--video", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
