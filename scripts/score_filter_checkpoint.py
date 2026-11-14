"""Separately score a frozen normal-pipeline filter answer against frame labels.

Labels are ordered text files beginning latitude longitude, one per original
rendered video frame. Capture/sensor timestamp files are never opened. This
program verifies the complete prediction/input/source freeze before labels.
It accepts proven full-video constant-FPS ordinal mappings only. Registered
subwindows remain supported by the official benchmark's time-aware scorer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monocular_osm.provenance import file_sha256
from scripts.score_frozen_video_filter import full_video_frame_indices


def verify_checkpoint(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads((directory / "frozen_prediction.json").read_text(encoding="utf-8"))
    if manifest.get("prediction_available") is not True or manifest.get("scoring_has_not_run") is not True:
        raise ValueError("checkpoint does not contain an available frozen prediction")
    for relative, expected in manifest["artifacts"].items():
        path = (directory / relative).resolve(strict=True)
        if not path.is_relative_to(directory) or file_sha256(path) != expected:
            raise ValueError(f"prediction artifact changed: {relative}")
    inputs = manifest["inputs"]
    for relative, expected in inputs["source_sha256"].items():
        path = (directory / "source_snapshot" / relative).resolve(strict=True)
        if not path.is_relative_to(directory / "source_snapshot") or file_sha256(path) != expected:
            raise ValueError(f"source snapshot changed: {relative}")
    if file_sha256(inputs["video"]["path"]) != inputs["video"]["sha256"]:
        raise ValueError("video changed")
    if file_sha256(inputs["map"]["cache_path"]) != inputs["map"]["sha256"]:
        raise ValueError("city map changed")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--waypoints", type=Path, required=True)
    args = parser.parse_args(argv)
    import cv2
    import numpy as np
    from monocular_osm.baselines import answer
    from monocular_osm.graph_filter import evaluate_against_gt
    from monocular_osm.metrics import score_route

    directory = args.checkpoint.resolve(strict=True)
    manifest = verify_checkpoint(directory)
    inputs = manifest["inputs"]
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    config = result["provenance"]["config"]
    cap = cv2.VideoCapture(inputs["video"]["path"])
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    extraction, timing = inputs["extraction"], inputs["timing"]
    if (extraction["stride"] != config["frame_stride"] or extraction["max_frames"] != config["max_frames"]
            or not np.isfinite(timing["fps"]) or not np.isclose(timing["fps"], fps, rtol=0, atol=1e-8)):
        raise ValueError("frozen full-video extraction/FPS differs from the video")
    with np.load(directory / "filter_prediction.npz", allow_pickle=False) as prediction:
        indices = full_video_frame_indices(
            start_sec=config["vo_start_sec"], end_sec=config["vo_end_sec"], stride=extraction["stride"],
            max_frames=extraction["max_frames"], fps=fps, frame_count=count,
            extracted_count=extraction["n_frames"], first_s=timing["first_s"], last_s=timing["last_s"],
            indices=prediction["frame_index"], video_times=prediction["video_times_s"])
    # This is the first access to the separately supplied labels.
    files = sorted(args.labels_dir.resolve(strict=True).glob("*.txt"))
    if len(files) != count:
        raise ValueError("label/video frame counts differ; one-to-one rendered-frame alignment is unproven")
    metadata = json.loads((directory / "filter_prediction.json").read_text(encoding="utf-8"))
    with np.load(directory / "filter_prediction.npz", allow_pickle=False) as z:
        labels, digest = [], hashlib.sha256()
        for index in indices:
            path = files[int(index)]
            data = path.read_bytes()
            digest.update(path.name.encode("utf-8"))
            digest.update(data)
            fields = data.decode("utf-8").split()
            labels.append([float(fields[0]), float(fields[1])])
        filtered = SimpleNamespace(times=z["times"], mode_latlon=z["mode_latlon"], smoothed_latlon=z["smoothed_latlon"],
                   confidence=z["confidence"], n_modes=z["n_modes"], final_xy=z["final_xy"], final_w=z["final_w"],
                   map_stats=metadata["map_stats"], gt_mass=None)
        metrics = evaluate_against_gt(filtered, np.asarray(labels))
    waypoints = json.loads(args.waypoints.read_text(encoding="utf-8"))
    gt_route = [[point["lat"], point["lon"]] for point in waypoints["waypoints"]]
    selected = answer(result)
    if selected is None or selected.meta.get("mode") != "filter":
        raise ValueError("checkpoint has no selected filter answer")
    route_metrics = score_route(selected.route_latlon, gt_route).to_dict()
    score = {"scored_utc": datetime.now(timezone.utc).isoformat(), "separate_post_freeze_scoring": True,
             "frozen_manifest_sha256": file_sha256(directory / "frozen_prediction.json"),
             "scoring_script_sha256": file_sha256(Path(__file__)),
             "label_alignment": "Proven full-video native ordinal coverage, zero start and frozen stride; no sensor timestamps.",
             "labels": {"directory": str(args.labels_dir.resolve()), "selected_label_bytes_sha256": digest.hexdigest(),
                        "waypoints_sha256": file_sha256(args.waypoints)},
             "per_frame_metrics": metrics, "selected_answer_source": selected.source,
             "selected_answer_meta": selected.meta, "selected_answer_route_metrics": route_metrics}
    (directory / "offline_score.json").write_text(json.dumps(score, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({**score, "per_frame_metrics": {key: value for key, value in metrics.items() if key != "errors_m"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
