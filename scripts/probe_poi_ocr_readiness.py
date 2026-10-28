"""Offline CPU readiness check for the experimental POI profile's native OCR.

Runs normal dense acquisition and the unmodified profile OCR helper only.
There is no VO estimation, place lookup, reference/model retrieval or scoring.
The output directory must be new; model assets must already be cached.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@contextmanager
def offline_network():
    previous_connect, previous_create = socket.socket.connect, socket.create_connection
    def refuse(*args, **kwargs):
        raise RuntimeError("Network is disabled for the OCR readiness probe")
    socket.socket.connect, socket.create_connection = refuse, refuse
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = previous_connect, previous_create


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.start < args.end:
        parser.error("require 0 <= start < end")
    video = args.video.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import cv2
    import numpy as np
    import psutil
    from monocular_osm import scene_text
    from monocular_osm.pipeline import PipelineConfig
    from monocular_osm.poi_localization import _array_sha, _json_sha
    from monocular_osm.provenance import config_snapshot, file_sha256, snapshot_sources
    from monocular_osm.stages.acquire import extract_video_frames
    from monocular_osm.stages.context import RunContext
    from monocular_osm.stages.poi_only import _ocr
    from monocular_osm.visual_odometry import default_intrinsics

    process = psutil.Process()
    limit = int(1.5 * 1024 ** 3)
    peak = {"rss_bytes": 0, "private_bytes": 0, "samples": 0}
    stopped = threading.Event()
    def observe():
        memory = process.memory_info()
        peak["rss_bytes"] = max(peak["rss_bytes"], memory.rss, getattr(memory, "peak_wset", 0))
        peak["private_bytes"] = max(peak["private_bytes"], getattr(memory, "private", 0))
        peak["samples"] += 1
        return memory.rss
    def bounded():
        if observe() >= limit:
            raise MemoryError("OCR readiness resident-memory limit reached")
    def monitor():
        while not stopped.wait(.2):
            observe()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cfg = PipelineConfig(url="", video_path=video, city=args.city, answer="poi",
        data_dir=output / "cache", output_dir=output, vo_start_sec=args.start, vo_end_sec=args.end,
        frame_memory_mb=64, frame_stride=None, max_frames=None)
    cfg.data_dir.mkdir()
    ctx = RunContext.create(cfg)
    ctx.video_path = video
    sources = [Path(__file__).resolve(), *[ROOT / "monocular_osm" / name for name in (
        "pipeline.py", "cli.py", "frame_extraction.py", "scene_text.py", "visual_odometry.py",
        "provenance.py", "stages/acquire.py", "stages/poi_only.py", "stages/poi_runtime.py")]]
    source_hashes = snapshot_sources(sources, output)
    samples, dense = [], []
    original_sample = scene_text._sample_frames
    def observed_samples(*args, **kwargs):
        for timestamp, image in original_sample(*args, **kwargs):
            bounded()
            samples.append({"t_sec": float(timestamp), "shape": list(image.shape), "dtype": image.dtype.str,
                            "pixel_sha256": _array_sha(image)})
            print(f"OCR native frame {len(samples)}/24 at {timestamp:.3f}s", flush=True)
            yield timestamp, image
    began = time.perf_counter()
    record = {"purpose": "CPU/native OCR readiness only; no geographic accuracy evaluation",
              "network_disabled": True, "inference_inputs": ["encoded_video"],
              "not_run": ["vo_estimation", "maps", "inventory", "reference_retrieval", "megaloc", "scoring"],
              "video": {"path": str(video), "sha256": file_sha256(video)},
              "registered_window": [args.start, args.end], "source_sha256": source_hashes,
              "free_memory_before_bytes": psutil.virtual_memory().available}
    try:
        with offline_network():
            bounded()
            extract_video_frames(ctx)
            for index, (timestamp, image) in enumerate(zip(ctx.frames.timestamps, ctx.frames.frames)):
                bounded()
                dense.append({"index": index, "t_sec": float(timestamp), "shape": list(image.shape),
                              "pixel_sha256": _array_sha(image)})
            shape = ctx.frames.frames[0].shape[:2]
            K = default_intrinsics(shape[1], shape[0])
            scene_text._sample_frames = observed_samples
            detections, metadata = _ocr(ctx, K, record["video"]["sha256"], shape)
            bounded()
        if len(samples) != 24 or metadata["reader"]["read_calls"] != 24:
            raise RuntimeError("normal OCR profile did not actually process all 24 fixed frames")
        if "torch" in sys.modules:
            raise RuntimeError("Unexpected Torch import during lightweight OCR readiness")
        if peak["rss_bytes"] >= limit:
            raise MemoryError("OCR readiness exceeded its resident-memory limit")
        record.update(status="ok", config=config_snapshot(cfg), sampling=ctx.frames.sampling,
            dense_frames=dense, dense_pixels_sha256=_json_sha(dense), ocr_frames=samples,
            ocr_pixels_sha256=_json_sha(samples), K_sha256=_array_sha(K), ocr=metadata,
            detections=detections, detections_sha256=_json_sha(detections))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                      dense_frames=dense, ocr_frames=samples)
        raise
    finally:
        scene_text._sample_frames = original_sample
        if ctx.frames is not None:
            ctx.frames.frames.close()
        cv2.setNumThreads(previous_threads)
        stopped.set()
        monitor_thread.join()
        observe()
        record.update(elapsed_sec=time.perf_counter() - began, memory=peak,
            frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), scoring_has_not_run=True)
        path = output / "readiness.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        frozen = {"status": record["status"], "scoring_has_not_run": True,
                  "artifacts": {str(p.relative_to(output)): file_sha256(p)
                    for p in sorted(output.rglob("*")) if p.is_file()}}
        manifest = output / "frozen_readiness.json"
        manifest.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print(json.dumps({"status": record["status"], "output": str(output), "memory": peak,
                          "frozen_sha256": file_sha256(manifest)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
