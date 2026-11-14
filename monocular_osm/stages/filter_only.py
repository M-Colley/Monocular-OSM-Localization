"""Opt-in independent video + named-city particle-filter execution."""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import cv2
import numpy as np

from ..provenance import config_snapshot, file_sha256, snapshot_sources
from ..visual_odometry import default_intrinsics
from .context import _record_channel
from .report import _write_result_json


def _freeze(ctx):
    out = ctx.cfg.output_dir
    names = ("filter_inputs.json", "result.json", "answer.png", "filter_prediction.json", "filter_prediction.npz")
    _write_result_json(out / "frozen_prediction.json", {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scoring_has_not_run": True, "prediction_available": "answer" in ctx.result,
        "artifacts": {name: file_sha256(out / name) for name in names if (out / name).is_file()},
        "inputs": ctx.result.get("filter_inputs"),
    })


def run_filter_only(ctx, *, extract, run_vo, run_filter, finalize):
    """Skip retrieval/matching and freeze the selected filter route before scoring."""
    cfg = ctx.cfg
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    for name in ("frozen_prediction.json", "filter_prediction.json", "filter_prediction.npz"):
        (cfg.output_dir / name).unlink(missing_ok=True)
    try:
        extract(ctx)
        h, w = ctx.frames.frames[0].shape[:2]
        root = Path(__file__).resolve().parents[2]
        sources = [root / "main.py", *[root / "monocular_osm" / relative for relative in (
            "cli.py", "pipeline.py", "visual_odometry.py", "frame_extraction.py",
            "graph_filter.py", "plane_speed.py", "osm_data.py", "provenance.py",
            "position.py", "baselines.py", "stages/context.py", "stages/acquire.py",
            "stages/trajectory.py", "stages/graph_filter.py", "stages/filter_only.py", "stages/report.py")]]
        inputs = {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input_policy": {key: False for key in ("calibration", "raw_images", "sensor_timestamps",
                              "gps_or_oxts", "ground_truth", "retrieval", "initial_position")},
            "video": {"path": str(Path(ctx.video_path).resolve()), "sha256": file_sha256(ctx.video_path)},
            "timing": {"source": "video_pts_or_encoded_fps", "fps": float(ctx.frames.fps),
                       "first_s": float(ctx.frames.timestamps[0]), "last_s": float(ctx.frames.timestamps[-1]),
                       "timestamps_sha256": hashlib.sha256(np.asarray(ctx.frames.timestamps, dtype="<f8").tobytes()).hexdigest()},
            "extraction": {"stride": cfg.frame_stride, "n_frames": len(ctx.frames.frames),
                           "max_frames": cfg.max_frames, "frame_memory_mb": cfg.frame_memory_mb,
                           "frame_max_width": cfg.frame_max_width},
            "camera": {"intrinsics": "generic_70_degree_hfov", "K": default_intrinsics(w, h).tolist(),
                       "height_m": cfg.camera_height_m, "vo_rng_seed": 0},
            "source_sha256": snapshot_sources(sources, cfg.output_dir),
        }
        ctx.result["filter_inputs"] = inputs
        _write_result_json(cfg.output_dir / "filter_inputs.json", {**inputs, "config": config_snapshot(cfg)})
        for channel in ("vpr", "plate", "vlm", "sun", "ocr_anchor", "loop_closure",
                        "coarse_from_frames", "orienternet", "geometry"):
            _record_channel(ctx.channel_status, channel, "disabled", reason="filter_only_execution")
        ctx.result["execution"] = {"stage": "filter", "geometry": "skipped", "retrieval": "skipped"}
        run_vo(ctx)
        ctx.result["n_frames"] = len(ctx.frames.frames)
        ctx.result["n_valid_poses"] = ctx.n_valid
        run_filter(ctx)
        if "filter_position" not in ctx.result:
            raise RuntimeError("filter-only execution produced no route; see result.json")
        ctx.result["execution"]["stage"] = "complete"
        finalize(ctx)
        if "answer" not in ctx.result:
            raise RuntimeError("filter-only execution produced no selected answer")
        _freeze(ctx)
        return ctx.result
    except Exception as error:
        ctx.result["execution"] = {"stage": "failed", "geometry": "skipped", "retrieval": "skipped",
                                   "error": f"{type(error).__name__}: {error}"}
        _record_channel(ctx.channel_status, "graph_filter", "failed", error=error)
        finalize(ctx)
        _freeze(ctx)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
