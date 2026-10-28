"""Fresh, CPU-only road-graph filtering from an MP4 and a named-city map.

This inference entry point has no ground-truth, calibration, raw-image, sensor
timestamp, initial-position or retrieval arguments. Video presentation times
(or the container FPS fallback used by extract_frames) are the only timing
input. Run any scoring in a separate process, after frozen_prediction.json
has been written. Existing VO/speed caches are deliberately never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class LockedFrames:
    """Serialize the tiny grayscale LRU while VO's workers read frame pairs."""

    def __init__(self, frames):
        self.frames, self.lock = frames, threading.RLock()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        with self.lock:
            return self.frames[index]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    # Non-finite diagnostics are explicitly null, never invalid JSON literals.
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def preserve_sources(paths, output, root=ROOT):
    """Keep the actual hashed source bytes so later replay needs no live checkout."""
    hashes = {}
    for path in paths:
        relative = str(Path(path).resolve().relative_to(Path(root).resolve()))
        data = Path(path).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        snapshot = Path(output) / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
    write_json(Path(output) / "source_snapshot.json", {
        "purpose": "Source bytes preserved before inference and separate scoring.",
        "files": {relative: {"sha256": value} for relative, value in hashes.items()}})
    return hashes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--city", required=True, help="Named-city map provenance, never a position fix")
    p.add_argument("--graph", type=Path, required=True, help="Existing whole-city driving GraphML")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("--camera-height", type=float, default=1.5)
    p.add_argument("--particles", type=int, default=600_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, choices=(2, 3, 4), default=2)
    p.add_argument("--frame-memory-mb", type=float, default=128)
    p.add_argument("--path-budget-mb", type=float, default=256,
                   help="Genealogy storage only; thinning does not change filter modes")
    p.add_argument("--output", type=Path, required=True)
    return p


def video_odometry(traj, timestamps, lengths):
    """Use the decoded video's timebase, including nonuniform presentation times."""
    import numpy as np
    from monocular_osm.graph_filter import odometry_from_trajectory, aggregate_odometry

    times = np.asarray(timestamps, float)
    dt = np.diff(times)
    if len(times) != len(traj.xz) or not len(dt) or not np.all(np.isfinite(times)) or np.any(dt <= 0):
        raise ValueError("video timestamps must be finite, strictly increasing and match the VO")
    median_dt = float(np.median(dt))
    odom = odometry_from_trajectory(traj, median_dt, heading_source="rotation", step_lengths_m=lengths)
    odom.dt = dt
    factor = max(1, int(round(0.5 / median_dt)))
    return aggregate_odometry(odom, factor), factor


def metric_config(particles, seed, spread, path_budget_mb):
    import numpy as np
    from monocular_osm.graph_filter import FilterConfig

    dev = spread if np.isfinite(spread) else 0.15
    return FilterConfig(n_particles=particles, seed=seed, metric_logk_sigma=0.25,
                        metric_speed_dev_sigma=float(np.clip(dev, 0.05, 0.3)),
                        path_budget_mb=path_budget_mb)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        raise ValueError("invalid video window")
    if min(args.particles, args.frame_memory_mb, args.path_budget_mb, args.camera_height) <= 0:
        raise ValueError("particle count, memory budgets and assumed camera height must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import cv2
    import numpy as np
    import osmnx as ox
    import psutil
    from dataclasses import asdict
    from monocular_osm.frame_extraction import GrayscaleFrames, extract_frames
    from monocular_osm.graph_filter import RoadNetwork, run_filter
    from monocular_osm.plane_speed import plane_step_lengths
    from monocular_osm.visual_odometry import VO_ALGO_VERSION, default_intrinsics, estimate_trajectory

    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    # This experiment must fail instead of fetching missing input or models.
    def no_network(*_a, **_kw):
        raise RuntimeError("strict video inference forbids network access")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network

    video, graph, out = args.video.resolve(strict=True), args.graph.resolve(strict=True), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to reuse a nonempty experiment directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    resources = {"physical_available_gb": psutil.virtual_memory().available / 2**30,
                 "process_rss_gb": process.memory_info().rss / 2**30}
    if resources["physical_available_gb"] < 2.25:
        raise MemoryError(f"need at least 2.25 GB available before starting: {resources}")
    probe = cv2.VideoCapture(str(video))
    try:
        if not probe.isOpened():
            raise RuntimeError("video cannot be opened")
        fps, n_header = probe.get(cv2.CAP_PROP_FPS), probe.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("container has no usable FPS; refusing an assumed timebase")
    finally:
        probe.release()

    source_paths = [Path(__file__).resolve(), *[ROOT / "monocular_osm" / name for name in
                   ("visual_odometry.py", "plane_speed.py", "graph_filter.py", "frame_extraction.py")]]
    source_hashes = preserve_sources(source_paths, out)
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "city": args.city,
              "video": str(video), "graph": str(graph), "video_sha256": sha256(video),
              "graph_sha256": sha256(graph), "window": {"start_sec": args.start, "end_sec": args.end},
              "container_fps": fps, "container_frame_count": int(n_header),
              "source_sha256": source_hashes,
              "dependencies": {name: importlib.metadata.version(name) for name in
                               ("numpy", "opencv-python", "scipy", "scikit-image", "osmnx")},
              "vo": {"fresh": True, "version": VO_ALGO_VERSION, "stride": 1, "max_frames": None,
                     "hfov_deg": 70.0, "enforce_planar": True, "workers": args.threads},
              "road_plane": {"camera_height_m": args.camera_height, "horizon": "principal",
                             "mask_static": False, "seed": args.seed},
              "particles": args.particles, "seed": args.seed, "cpu_threads": args.threads,
              "frame_memory_mb": args.frame_memory_mb, "path_budget_mb": args.path_budget_mb,
              "heading_source": "rotation", "target_filter_step_s": 0.5,
              "input_policy": {"calibration": False, "raw_images": False, "sensor_timestamps": False,
                               "gps_or_oxts": False, "retrieval": False, "initial_position": False,
                               "ground_truth": False, "network": False,
                               "timing": "decoded MP4 PTS; encoded container FPS fallback only"},
              "initial_resources": resources}
    write_json(out / "inference_config.json", config)

    phase, stop, telemetry = ["extract"], threading.Event(), []
    started = time.perf_counter()
    def monitor():
        while not stop.is_set():
            info = process.memory_info()
            row = {"elapsed_s": round(time.perf_counter() - started, 1), "phase": phase[0],
                   "rss_mb": round(info.rss / 2**20, 1),
                   "private_mb": round(getattr(info, "private", info.vms) / 2**20, 1),
                   "available_mb": round(psutil.virtual_memory().available / 2**20, 1)}
            telemetry.append(row)
            if len(telemetry) == 1 or len(telemetry) % 15 == 0:
                print(json.dumps({"progress": row}), flush=True)
            stop.wait(2)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    frames = None
    try:
        frames = extract_frames(video, stride=1, max_frames=None, start_sec=args.start, end_sec=args.end,
                                memory_mb=args.frame_memory_mb)
        timestamps = np.asarray(frames.timestamps, float)
        h, w = frames.frames[0].shape[:2]
        K = default_intrinsics(w, h)
        print(json.dumps({"frames": len(frames.frames), "shape_hw": [h, w], "fps": frames.fps,
                          "first_video_t": float(timestamps[0]), "last_video_t": float(timestamps[-1]),
                          "disk_backed": frames.frames.spilled, "K": K.tolist()}), flush=True)
        phase[0] = "fresh_video_odometry"
        # Lazy view: two grayscale images plus the bounded BGR cache.
        gray = LockedFrames(GrayscaleFrames(frames.frames))
        traj = estimate_trajectory(gray, K, n_workers=args.threads)
        np.savez_compressed(out / "fresh_trajectory.npz", centers=traj.centers, xz=traj.xz,
                            valid=traj.valid, rotations=traj.rotations, translations=traj.translations,
                            n_inliers=np.asarray(traj.n_inliers), timestamps=timestamps, K=K)
        phase[0] = "road_plane_speed"
        speed = plane_step_lengths(gray, traj, K, camera_height_m=args.camera_height, seed=args.seed)
        if speed.coverage < 0.2:
            raise RuntimeError(f"road plane has insufficient coverage ({speed.coverage:.3f}); no hidden unit fallback")
        np.savez_compressed(out / "plane_speed.npz", lengths=speed.lengths_m, raw=speed.raw_m,
                            inliers=speed.inliers, coverage=speed.coverage, rel_spread=speed.rel_spread)
        odom, factor = video_odometry(traj, timestamps, speed.lengths_m)
        fcfg = metric_config(args.particles, args.seed, speed.rel_spread, args.path_budget_mb)
        write_json(out / "measured_config.json", {"K": K.tolist(), "shape_hw": [h, w],
                   "n_frames": len(timestamps), "video_first_s": float(timestamps[0]),
                   "video_last_s": float(timestamps[-1]), "road_plane": speed.to_dict(),
                   "valid_vo_pairs": int(np.count_nonzero(traj.valid[1:])),
                   "aggregation_factor": factor, "filter_config": asdict(fcfg)})
        np.savez_compressed(out / "odometry.npz", step=odom.step, heading=odom.heading, dt=odom.dt,
                            heading_var=odom.heading_var, frame_index=odom.frame_index)
        frames.close()
        frames = None
        del gray
        phase[0] = "load_cached_city_graph"
        graph_data = ox.load_graphml(graph)
        net = RoadNetwork.from_graph(graph_data)
        del graph_data
        print(json.dumps({"road_plane": speed.to_dict(), "filter_steps": len(odom),
                          "map": net.stats(), "filter_config": asdict(fcfg)}), flush=True)
        phase[0] = "particle_filter"
        # No position observations, heading observations, initial disc or diagnostic GT.
        result = run_filter(net, odom, fcfg)
        phase[0] = "freeze_prediction"
        prediction = result.to_json()
        prediction["video_times_s"] = timestamps[result.frame_index].tolist()
        prediction["input_policy"] = config["input_policy"]
        prediction["city"] = args.city
        prediction["confidence_calibrated"] = False
        write_json(out / "prediction.json", prediction)
        np.savez_compressed(out / "prediction.npz", times=result.times, frame_index=result.frame_index,
                            video_times_s=timestamps[result.frame_index], mode_latlon=result.mode_latlon,
                            smoothed_latlon=result.smoothed_latlon, final_xy=result.final_xy,
                            final_w=result.final_w, confidence=result.confidence, n_modes=result.n_modes)
        stop.set()
        watcher.join()
        write_json(out / "resources.json", {"samples": telemetry,
                   "peak_sampled_rss_mb": max(x["rss_mb"] for x in telemetry),
                   "peak_sampled_private_mb": max(x["private_mb"] for x in telemetry),
                   "elapsed_s": time.perf_counter() - started})
        artifacts = sorted(p for p in out.iterdir() if p.is_file() and p.suffix in (".json", ".npz"))
        manifest = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                    "scoring_has_not_run": True, "video_sha256": config["video_sha256"],
                    "graph_sha256": config["graph_sha256"], "source_sha256": config["source_sha256"],
                    "artifacts": {p.name: sha256(p) for p in artifacts}}
        write_json(out / "frozen_prediction.json", manifest)
        print(json.dumps({"frozen_prediction": str(out / "frozen_prediction.json"),
                          "final_mode_latlon": result.mode_latlon[-1].tolist(),
                          "final_confidence": float(result.confidence[-1]),
                          "elapsed_s": time.perf_counter() - started}), flush=True)
        return 0
    except BaseException as error:
        write_json(out / "failed.json", {"phase": phase[0], "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if frames is not None:
            frames.close()
        stop.set()
        watcher.join()


if __name__ == "__main__":
    raise SystemExit(main())
