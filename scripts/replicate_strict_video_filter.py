"""Replicate frozen MP4-derived road-graph odometry across a fixed seed set.

This entry point verifies the original freeze, video, map and source snapshots,
then changes only the particle-filter seed. All predictions are frozen as one
batch before a separate process may score them. It has no label arguments and
never reads an existing offline score. No image decoding or VO is repeated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_strict_video_filter import sha256, write_json


def verify_frozen_inputs(base, source_root=ROOT):
    """Verify every frozen artifact and the exact PF source used for replay."""
    base = Path(base).resolve(strict=True)
    manifest_path = base / "frozen_prediction.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base) or sha256(path) != expected:
            raise ValueError(f"frozen artifact hash mismatch: {name}")
    for relative, expected in manifest["source_sha256"].items():
        snapshot = (base / "source_snapshot" / relative).resolve(strict=True)
        if not snapshot.is_relative_to(base / "source_snapshot") or sha256(snapshot) != expected:
            raise ValueError(f"frozen source snapshot hash mismatch: {relative}")
        if Path(relative).name == "graph_filter.py":
            if sha256(Path(source_root) / relative) != expected:
                raise ValueError("current particle-filter source differs from the frozen run")
    config = json.loads((base / "inference_config.json").read_text(encoding="utf-8"))
    for key in ("video", "graph"):
        if sha256(Path(config[key])) != config[f"{key}_sha256"]:
            raise ValueError(f"original {key} hash mismatch")
    measured = json.loads((base / "measured_config.json").read_text(encoding="utf-8"))
    if measured["filter_config"].get("init_disc") is not None:
        raise ValueError("strict replay requires a city-wide initial prior")
    policy = config["input_policy"]
    forbidden = ("calibration", "raw_images", "sensor_timestamps", "gps_or_oxts",
                 "retrieval", "initial_position", "ground_truth", "network")
    if any(policy.get(key) is not False for key in forbidden):
        raise ValueError("base inference does not satisfy the strict video-only policy")
    return manifest, config, measured


def fixed_seed_config(frozen, seed):
    """Clone every experimental setting, changing the seed alone."""
    from monocular_osm.graph_filter import FilterConfig

    values = dict(frozen)
    values["seed"] = int(seed)
    return FilterConfig(**values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="Original frozen MP4 inference directory")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--threads", type=int, choices=(2, 3, 4), default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import cv2
    import numpy as np
    import osmnx as ox
    import psutil
    from monocular_osm.graph_filter import Odometry, RoadNetwork, run_filter

    cv2.setNumThreads(1)
    def no_network(*_args, **_kwargs):
        raise RuntimeError("strict replay forbids network access")
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    base, out = args.base.resolve(strict=True), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to reuse a nonempty batch directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    phase = ["verify_frozen_inputs"]
    started = time.perf_counter()
    manifest, original, measured = verify_frozen_inputs(base)
    for name, expected in original["dependencies"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"dependency version changed: {name}")
    process = psutil.Process()
    available = psutil.virtual_memory().available
    if available < 2.25 * 2**30:
        raise MemoryError(f"need 2.25 GB available before replay; available={available / 2**30:.2f} GB")
    replay_source = {"scripts/replicate_strict_video_filter.py": sha256(Path(__file__)),
                     "scripts/probe_strict_video_filter.py": sha256(ROOT / "scripts/probe_strict_video_filter.py"),
                     "monocular_osm/graph_filter.py": sha256(ROOT / "monocular_osm/graph_filter.py")}
    for relative, expected in replay_source.items():
        dest = out / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        assert sha256(dest) == expected
    batch_config = {"created_utc": datetime.now(timezone.utc).isoformat(), "base": str(base),
                    "base_frozen_manifest_sha256": sha256(base / "frozen_prediction.json"),
                    "base_artifact_sha256": manifest["artifacts"],
                    "video_sha256": original["video_sha256"], "graph_sha256": original["graph_sha256"],
                    "base_source_sha256": manifest["source_sha256"], "replay_source_sha256": replay_source,
                    "seeds_predeclared": args.seeds, "cpu_threads": args.threads,
                    "input_policy": original["input_policy"], "fixed_filter_config": measured["filter_config"],
                    "initial_available_mb": available / 2**20,
                    "selection_rule": "Every predeclared seed is retained; all predictions freeze before scoring."}
    write_json(out / "batch_config.json", batch_config)
    z = np.load(base / "odometry.npz", allow_pickle=False)
    odom = Odometry(step=z["step"], heading=z["heading"], dt=z["dt"],
                    heading_var=z["heading_var"], frame_index=z["frame_index"], metric=True)
    with np.load(base / "fresh_trajectory.npz", allow_pickle=False) as trajectory:
        timestamps = trajectory["timestamps"].copy()
    phase[0] = "load_verified_city_map"
    graph = ox.load_graphml(original["graph"])
    net = RoadNetwork.from_graph(graph)
    del graph
    telemetry, stopped = [], threading.Event()
    def monitor():
        while not stopped.is_set():
            info = process.memory_info()
            row = {"elapsed_s": round(time.perf_counter() - started, 1), "phase": phase[0],
                   "rss_mb": round(info.rss / 2**20, 1),
                   "private_mb": round(getattr(info, "private", info.vms) / 2**20, 1),
                   "peak_working_set_mb": round(getattr(info, "peak_wset", info.rss) / 2**20, 1),
                   "peak_private_commit_mb": round(getattr(info, "peak_pagefile", info.vms) / 2**20, 1),
                   "available_mb": round(psutil.virtual_memory().available / 2**20, 1)}
            telemetry.append(row)
            if len(telemetry) == 1 or len(telemetry) % 15 == 0:
                print(json.dumps({"progress": row}), flush=True)
            stopped.wait(2)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    frozen_seeds = {}
    try:
        for seed in args.seeds:
            phase[0] = f"seed_{seed}"
            seed_out = out / f"seed_{seed}"
            seed_out.mkdir()
            cfg = fixed_seed_config(measured["filter_config"], seed)
            write_json(seed_out / "config.json", {"filter_config": asdict(cfg),
                       "base_frozen_manifest_sha256": batch_config["base_frozen_manifest_sha256"],
                       "video_sha256": original["video_sha256"], "graph_sha256": original["graph_sha256"],
                       "input_policy": original["input_policy"], "replay_source_sha256": replay_source})
            # No observations, initial disc, diagnostic GPS or new VO computation.
            result = run_filter(net, odom, cfg)
            prediction = result.to_json()
            prediction["video_times_s"] = timestamps[result.frame_index].tolist()
            prediction["input_policy"] = original["input_policy"]
            prediction["city"] = original["city"]
            prediction["confidence_calibrated"] = False
            write_json(seed_out / "prediction.json", prediction)
            np.savez_compressed(seed_out / "prediction.npz", times=result.times, frame_index=result.frame_index,
                                video_times_s=timestamps[result.frame_index], mode_latlon=result.mode_latlon,
                                smoothed_latlon=result.smoothed_latlon, final_xy=result.final_xy,
                                final_w=result.final_w, confidence=result.confidence, n_modes=result.n_modes)
            files = {path.name: sha256(path) for path in sorted(seed_out.iterdir()) if path.is_file()}
            seed_manifest = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                             "scoring_has_not_run": True, "artifacts": files,
                             "base_frozen_manifest_sha256": batch_config["base_frozen_manifest_sha256"],
                             "replay_source_sha256": replay_source}
            write_json(seed_out / "frozen_prediction.json", seed_manifest)
            frozen_seeds[str(seed)] = {"directory": seed_out.name,
                                       "manifest_sha256": sha256(seed_out / "frozen_prediction.json")}
            print(json.dumps({"seed_frozen": seed, "runtime_s": result.runtime_s,
                              "final_mode_latlon": result.mode_latlon[-1].tolist(),
                              "confidence": float(result.confidence[-1])}), flush=True)
            del result, prediction
            gc.collect()
        phase[0] = "freeze_complete_seed_batch"
        stopped.set()
        watcher.join()
        write_json(out / "resources.json", {"samples": telemetry,
                   "peak_working_set_mb": max(row["peak_working_set_mb"] for row in telemetry),
                   "peak_private_commit_mb": max(row["peak_private_commit_mb"] for row in telemetry),
                   "elapsed_s": time.perf_counter() - started})
        frozen_batch = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                        "all_predictions_frozen_before_scoring": True, "seeds_predeclared": args.seeds,
                        "base_frozen_manifest_sha256": batch_config["base_frozen_manifest_sha256"],
                        "artifacts": {name: sha256(out / name) for name in ("batch_config.json", "resources.json")},
                        "seeds": frozen_seeds, "replay_source_sha256": replay_source}
        write_json(out / "frozen_predictions.json", frozen_batch)
        print(json.dumps({"batch_frozen": str(out / "frozen_predictions.json"),
                          "batch_manifest_sha256": sha256(out / "frozen_predictions.json"),
                          "elapsed_s": time.perf_counter() - started}), flush=True)
    except BaseException as error:
        write_json(out / "failed.json", {"phase": phase[0], "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        stopped.set()
        watcher.join()
        z.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
