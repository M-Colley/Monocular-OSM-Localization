"""Score a complete frozen seed batch in a separate, label-reading process.

Ordered label text files must begin with latitude and longitude and correspond
one-to-one to the rendered video frames. Sensor timestamps are never read.
Every predeclared seed, including the original run, is reported and aggregated.
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
from scripts.probe_strict_video_filter import sha256, write_json


def verify_artifacts(directory, manifest):
    directory = Path(directory).resolve()
    for relative, expected in manifest["artifacts"].items():
        path = (directory / relative).resolve(strict=True)
        if not path.is_relative_to(directory) or sha256(path) != expected:
            raise ValueError(f"frozen artifact hash mismatch: {path}")


def verify_complete_batch(batch):
    """Finish all input verification before the caller opens any labels."""
    batch = Path(batch).resolve(strict=True)
    freeze = json.loads((batch / "frozen_predictions.json").read_text(encoding="utf-8"))
    if freeze.get("all_predictions_frozen_before_scoring") is not True:
        raise ValueError("batch has not frozen all predictions")
    verify_artifacts(batch, freeze)
    config = json.loads((batch / "batch_config.json").read_text(encoding="utf-8"))
    expected_seeds = list(config["seeds_predeclared"])
    if expected_seeds != freeze["seeds_predeclared"] or set(map(str, expected_seeds)) != set(freeze["seeds"]):
        raise ValueError("frozen predictions do not match the complete predeclared seed set")
    base = Path(config["base"])
    base_manifest_path = base / "frozen_prediction.json"
    if sha256(base_manifest_path) != freeze["base_frozen_manifest_sha256"]:
        raise ValueError("base frozen manifest changed")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    verify_artifacts(base, base_manifest)
    original_seed = int(config["fixed_filter_config"]["seed"])
    if original_seed in expected_seeds:
        raise ValueError("replication seed set repeats the original seed")
    runs = [(original_seed, base)]
    for seed in expected_seeds:
        entry = freeze["seeds"][str(seed)]
        directory = (batch / entry["directory"]).resolve(strict=True)
        if not directory.is_relative_to(batch):
            raise ValueError("seed directory escapes the frozen batch")
        manifest_path = directory / "frozen_prediction.json"
        if sha256(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"seed {seed} frozen manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["seed"] != seed or manifest["base_frozen_manifest_sha256"] != freeze["base_frozen_manifest_sha256"]:
            raise ValueError(f"seed {seed} provenance mismatch")
        verify_artifacts(directory, manifest)
        seed_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if seed_config["filter_config"] != {**config["fixed_filter_config"], "seed": seed}:
            raise ValueError(f"seed {seed} changed more than the seed")
        runs.append((seed, directory))
    for relative, expected in freeze["replay_source_sha256"].items():
        path = batch / "source_snapshot" / relative
        if sha256(path) != expected:
            raise ValueError(f"replay source snapshot changed: {relative}")
    return freeze, config, runs


def aggregate_scores(rows):
    import numpy as np

    def summary(values):
        a = np.asarray(values, float)
        return {"mean": float(a.mean()), "std_population": float(a.std()),
                "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}
    localized = [row for row in rows if row["per_frame_metrics"]["time_to_localize_s"] is not None]
    out = {"n_seeds": len(rows), "localized_seeds": len(localized),
           "seeds_included": [row["seed"] for row in rows],
           "selection_rule": "All predeclared seeds; failures remain in every whole-run metric."}
    out["per_frame_metrics"] = {key: summary([row["per_frame_metrics"][key] for row in rows]) for key in
        ("mean_err_m", "median_err_m", "final_err_m", "smoothed_mean_err_m", "frac_steps_lt10m", "frac_steps_lt25m")}
    out["smoothed_waypoint_route_metrics"] = {key: summary([row["smoothed_waypoint_route_metrics"][key] for row in rows])
        for key in ("gt_to_pred_mean_m", "symmetric_mean_m", "start_error_m", "end_error_m", "hausdorff_m")}
    out["time_to_localize_s_among_localized"] = (summary([row["per_frame_metrics"]["time_to_localize_s"] for row in localized])
                                                     if localized else None)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, required=True, help="Ordered per-frame text files beginning lat lon")
    p.add_argument("--waypoints", type=Path, required=True, help="Committed waypoint JSON, for route metrics")
    args = p.parse_args(argv)
    import numpy as np
    from monocular_osm.graph_filter import evaluate_against_gt
    from monocular_osm.metrics import score_route

    batch = args.batch.resolve(strict=True)
    freeze, config, runs = verify_complete_batch(batch)
    # Verification above must precede this first label-directory access.
    labels_dir = args.labels_dir.resolve(strict=True)
    label_files = sorted(labels_dir.glob("*.txt"))
    first_indices = None
    for seed, directory in runs:
        with np.load(directory / "prediction.npz", allow_pickle=False) as prediction:
            indices = prediction["frame_index"].astype(int)
        if first_indices is None:
            first_indices = indices
        elif not np.array_equal(first_indices, indices):
            raise ValueError("seeds use different video-frame indices")
    original = json.loads((Path(config["base"]) / "inference_config.json").read_text(encoding="utf-8"))
    if len(label_files) != original["container_frame_count"]:
        raise ValueError("label count and original video-frame count differ; frame alignment is unproven")
    gt, label_digest = [], hashlib.sha256()
    for index in first_indices:
        path = label_files[int(index)]
        data = path.read_bytes()
        label_digest.update(path.name.encode("utf-8"))
        label_digest.update(data)
        fields = data.decode("utf-8").split()
        gt.append([float(fields[0]), float(fields[1])])
    gt = np.asarray(gt)
    waypoints_path = args.waypoints.resolve(strict=True)
    waypoints = json.loads(waypoints_path.read_text(encoding="utf-8"))
    route_gt = np.asarray([[point["lat"], point["lon"]] for point in waypoints["waypoints"]])
    rows = []
    for seed, directory in runs:
        metadata = json.loads((directory / "prediction.json").read_text(encoding="utf-8"))
        with np.load(directory / "prediction.npz", allow_pickle=False) as z:
            result = SimpleNamespace(times=z["times"], mode_latlon=z["mode_latlon"],
                     smoothed_latlon=z["smoothed_latlon"], confidence=z["confidence"], n_modes=z["n_modes"],
                     final_xy=z["final_xy"], final_w=z["final_w"], map_stats=metadata["map_stats"], gt_mass=None)
            metrics = evaluate_against_gt(result, gt)
            errors = metrics.pop("errors_m")
            route = score_route(result.smoothed_latlon, route_gt).to_dict()
        row = {"seed": seed, "prediction_sha256": sha256(directory / "prediction.json"),
               "pf_runtime_s": metadata["runtime_s"], "per_frame_metrics": metrics,
               "smoothed_waypoint_route_metrics": route, "errors_m": errors}
        rows.append(row)
        print(json.dumps({"seed": seed, "per_frame_metrics": metrics,
                          "route_symmetric_mean_m": route["symmetric_mean_m"]}), flush=True)
    score = {"scored_utc": datetime.now(timezone.utc).isoformat(), "separate_post_batch_freeze_scoring": True,
             "batch_frozen_manifest_sha256": sha256(batch / "frozen_predictions.json"),
             "scoring_script_sha256": sha256(Path(__file__)),
             "label_alignment": "Ordered synced frame index; latency uses video PTS; no sensor timestamp files read.",
             "labels": {"directory": str(labels_dir), "n_files": len(label_files),
                        "selected_label_bytes_sha256": label_digest.hexdigest(),
                        "waypoints": str(waypoints_path), "waypoints_sha256": sha256(waypoints_path)},
             "all_seeds": rows, "aggregate_all_seeds": aggregate_scores(rows),
             "aggregate_replication_seeds": aggregate_scores(rows[1:])}
    write_json(batch / "offline_score_all_seeds.json", score)
    print(json.dumps({"aggregate_all_seeds": score["aggregate_all_seeds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
