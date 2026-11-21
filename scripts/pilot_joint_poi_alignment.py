"""Post-hoc dev extension: timed literal POI cues plus full joint VO search.

All frozen branch hypotheses are scored by the same soft visibility rule.
Inference reads the FULL saved coarse objective cube, not its wrong fine
shortlist. Every hypothesis receives the same local-refinement budget.
No new models, image downloads, calibration files or ground truth enter infer.

    python scripts/pilot_joint_poi_alignment.py infer
    python scripts/pilot_joint_poi_alignment.py score
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_joint_alignment as joint


SETTINGS = {
    "visibility_modes": ["distance", "distance_bearing"],
    "visibility_m": 100.0, "distance_sigma_m": 100.0, "distance_penalty_cap": 8.0,
    "bearing_free_deg": 20.0, "bearing_sigma_deg": 45.0, "bearing_penalty_cap": 4.0,
    "heading_window_s": 1.5, "poi_weight": 1.0,
    "basins_per_hypothesis": 16, "seed_top_scores": 16384,
    "missing_observation_penalty": 8.0,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def obs_key(observation):
    box = observation.get("bbox")
    return (float(observation["t_sec"]), str(observation["text"]).casefold(),
            tuple(float(v) for v in box) if box is not None else ())


def load_hypotheses(manifest_path, clip, transformer):
    manifest = read_json(manifest_path)
    if manifest.get("ground_truth_used") is not False:
        raise ValueError("POI manifest must declare ground_truth_used=false")
    entry = next(e for e in manifest["artifacts"] if e["id"] == clip)
    artifact = Path(entry["path"])
    if joint.sha256(artifact) != entry["sha256"]:
        raise ValueError("frozen POI artifact hash mismatch")
    record = read_json(artifact)
    if record.get("selected_answer") is not None:
        raise ValueError("POI hypotheses must not contain a preselected answer")
    if len(record["candidate_clusters"]) != entry["n_candidate_clusters"]:
        raise ValueError("frozen hypothesis count mismatch")
    clues_path = Path(record["clue_source"])
    if joint.sha256(clues_path) != record["clue_sha256"]:
        raise ValueError("source OCR clue hash mismatch")
    global_observations, branch_observations = {}, []
    hypotheses = []
    for index, cluster in enumerate(record["candidate_clusters"]):
        locations = {}
        for poi in cluster["pois"]:
            xy = transformer.transform(float(poi["lon"]), float(poi["lat"]))
            if not np.isfinite(xy).all():
                raise ValueError("nonfinite frozen POI coordinates")
            locations[poi["id"]] = np.asarray(xy)
        observations = {}
        for edge in cluster["conjunctions"]:
            for witness in edge["witnesses"]:
                for observation in witness["observations"]:
                    key = obs_key(observation)
                    global_observations.setdefault(key, observation)
                    observations.setdefault(key, {})[observation["poi_id"]] = locations[observation["poi_id"]]
        branch_observations.append(observations)
        hypotheses.append({"index": index, "poi_ids": cluster["poi_ids"],
                           "observation_count": len(observations), "pois": cluster["pois"]})
    if not hypotheses:
        raise ValueError("no frozen POI hypotheses")
    keys = sorted(global_observations)
    observations = [global_observations[key] for key in keys]
    allowed = [[np.asarray(list(branch.get(key, {}).values())).reshape(-1, 2) for key in keys]
               for branch in branch_observations]
    provenance = {"manifest": str(manifest_path.resolve()), "manifest_sha256": joint.sha256(manifest_path),
                  "artifact": str(artifact.resolve()), "artifact_sha256": entry["sha256"],
                  "clue_sha256": record["clue_sha256"],
                  "osm_snapshot_sha256": record["osm_snapshot_sha256"]}
    return observations, allowed, hypotheses, provenance


def observation_geometry(vo_path, metadata, observations):
    if joint.sha256(vo_path) != metadata["vo"]["cache_sha256"]:
        raise ValueError("saved joint VO artifact changed")
    with np.load(vo_path, allow_pickle=False) as z:
        identity = json.loads(str(z["input_identity"]))
        xy, times = z["xz"], z["timestamps"]
    arc = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    when = np.array([float(o["t_sec"]) for o in observations])
    if np.any(when < times[0]) or np.any(when > times[-1]):
        raise ValueError("OCR observations fall outside the saved VO timeline")

    def at(t):
        return np.c_[np.interp(t, times, xy[:, 0]), np.interp(t, times, xy[:, 1])]

    unit_positions = (at(when) - xy[0]) / arc
    half = SETTINGS["heading_window_s"] / 2
    tangent = at(np.minimum(when + half, times[-1])) - at(np.maximum(when - half, times[0]))
    headings = np.arctan2(tangent[:, 1], tangent[:, 0])
    K = np.asarray(identity["K"])
    rays, bearing_available = [], []
    for i, observation in enumerate(observations):
        box = observation.get("bbox")
        available = box is not None and len(box) == 4 and np.linalg.norm(tangent[i]) > 1e-6
        rays.append(float(np.arctan(((box[0] + box[2]) / 2 - K[0, 2]) / K[0, 0])) if available else 0.)
        bearing_available.append(bool(available))
    weights = np.array([float(o["confidence"]) for o in observations])
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("OCR confidence weights must be finite and positive")
    return {"unit_positions": unit_positions, "headings": headings, "rays": np.asarray(rays),
            "bearing_available": np.asarray(bearing_available), "weights": weights / weights.sum(),
            "K": K, "identity": identity}


def poi_losses(parameters, geometry, allowed):
    """Return [visibility mode,hypothesis,candidate] soft losses, without labels."""
    camera = joint.transformed_routes(parameters, geometry["unit_positions"])
    yaw = np.radians(parameters[:, 2])
    losses = np.zeros((2, len(allowed), len(parameters)), dtype=float)
    for hypothesis, choices in enumerate(allowed):
        for obs, pois in enumerate(choices):
            if not len(pois):
                losses[:, hypothesis] += geometry["weights"][obs] * SETTINGS["missing_observation_penalty"]
                continue
            delta = pois[None, :, :] - camera[:, obs, None, :]
            distance = np.linalg.norm(delta, axis=2)
            excess = np.maximum(distance - SETTINGS["visibility_m"], 0)
            radial = np.minimum(0.5 * (excess / SETTINGS["distance_sigma_m"]) ** 2,
                                SETTINGS["distance_penalty_cap"])
            angular = np.zeros_like(radial)
            if geometry["bearing_available"][obs]:
                # In the VO x/z plane, image-right is clockwise from camera
                # forward, so a positive image bearing subtracts from XY yaw.
                expected = yaw + geometry["headings"][obs] - geometry["rays"][obs]
                bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0])
                error = np.abs((bearing - expected[:, None] + np.pi) % (2 * np.pi) - np.pi)
                excess_angle = np.maximum(error - np.radians(SETTINGS["bearing_free_deg"]), 0)
                angular = np.minimum(0.5 * (excess_angle / np.radians(SETTINGS["bearing_sigma_deg"])) ** 2,
                                     SETTINGS["bearing_penalty_cap"])
            losses[0, hypothesis] += geometry["weights"][obs] * radial.min(axis=1)
            losses[1, hypothesis] += geometry["weights"][obs] * (radial + angular).min(axis=1)
    return losses


def coarse_losses(starts, yaws, lengths, geometry, allowed):
    result = np.empty((2, len(allowed), len(lengths), len(yaws), len(starts)), dtype=np.float32)
    batch = joint.SETTINGS["transform_batch"]
    for i, length in enumerate(lengths):
        for j, yaw in enumerate(yaws):
            for k in range(0, len(starts), batch):
                positions = starts[k:k + batch]
                parameters = np.c_[positions, np.full(len(positions), yaw), np.full(len(positions), length)]
                result[:, :, i, j, k:k + len(positions)] = poi_losses(parameters, geometry, allowed)
        print(f"all POI hypotheses: coarse length {i + 1}/{len(lengths)}", flush=True)
    return result


def seeds_for_hypothesis(scores, starts, yaws, lengths):
    flat = scores.ravel()
    keep = min(SETTINGS["seed_top_scores"], len(flat))
    selected = np.argpartition(-flat, keep - 1)[:keep]
    selected = selected[np.argsort(-flat[selected], kind="stable")]
    seeds, seen = [], set()
    for index in selected:
        i, j, k = np.unravel_index(index, scores.shape)
        p = np.array([*starts[k], yaws[j], lengths[i]])
        key = (round(p[0] / joint.SETTINGS["basin_position_bin_m"]),
               round(p[1] / joint.SETTINGS["basin_position_bin_m"]),
               round(p[2] / joint.SETTINGS["basin_yaw_bin_deg"]),
               round(np.log(p[3]) / joint.SETTINGS["basin_log_length_bin"]))
        if key not in seen:
            seen.add(key)
            seeds.append(p)
            if len(seeds) == SETTINGS["basins_per_hypothesis"]:
                break
    return seeds


def refinement_parameters(seeds_by_hypothesis, lengths):
    parameters, owners = [], []
    for hypothesis, seeds in enumerate(seeds_by_hypothesis):
        for seed in seeds:
            for factor in joint.SETTINGS["refine_length_multipliers"]:
                length = seed[3] * factor
                if not lengths[0] <= length <= lengths[-1]:
                    continue
                for da in joint.SETTINGS["refine_yaw_offsets_deg"]:
                    for dx in joint.SETTINGS["refine_xy_offsets_m"]:
                        for dy in joint.SETTINGS["refine_xy_offsets_m"]:
                            parameters.append([seed[0] + dx, seed[1] + dy, (seed[2] + da) % 360, length])
                            owners.append(hypothesis)
    return np.asarray(parameters), np.asarray(owners)


def infer(args):
    from pyproj import Transformer

    args.out.mkdir(parents=True, exist_ok=True)
    sources = {variant: args.joint / f"{variant}.npz" for variant in joint.SETTINGS["variants"]}
    with np.load(sources["raw"], allow_pickle=False) as z:
        source_meta = json.loads(str(z["metadata"]))
        starts, yaws, lengths = z["starts"], z["yaws"], z["lengths"]
        unit_shape, query_times = z["unit_shape"], z["query_times"]
    if joint.sha256(joint.__file__) != source_meta["configuration"]["script_sha256"]:
        raise ValueError("original joint-inference script changed; do not silently change the base objective")
    tf = Transformer.from_crs("EPSG:4326", source_meta["crs"], always_xy=True)
    inverse = Transformer.from_crs(source_meta["crs"], "EPSG:4326", always_xy=True)
    observations, allowed, hypotheses, poi_provenance = load_hypotheses(args.manifest, args.clip, tf)
    geometry = observation_geometry(args.joint / "verified_video_vo.npz", source_meta, observations)
    config = {"status": "post-hoc development experiment; earlier GT results have been seen",
              "settings": SETTINGS, "base_joint_settings": joint.SETTINGS,
              "ground_truth_used_during_inference": False, "default_changed": False,
              "hypothesis_selection": "all branches, equal refinement budget, same global maximum rule",
              "clip": args.clip, "source_joint": {k: {"path": str(v.resolve()), "sha256": joint.sha256(v)}
                                                    for k, v in sources.items()},
              "poi_provenance": poi_provenance, "script_sha256": joint.sha256(__file__)}
    registration = args.out / "experiment.json"
    if registration.exists() and read_json(registration) != config:
        raise ValueError("output belongs to another frozen configuration; choose a new --out")
    joint.write_json(registration, config)
    joint.write_json(args.out / "hypotheses.json", {"hypotheses": hypotheses, "observations": observations,
                     "K": geometry["K"].tolist(), "bearing_available": geometry["bearing_available"].tolist(),
                     "camera_horizontal_rays_deg": np.degrees(geometry["rays"]).tolist()})
    losses = coarse_losses(starts, yaws, lengths, geometry, allowed)
    np.savez_compressed(args.out / "all_branch_coarse_losses.npz", losses=losses, starts=starts,
                        yaws=yaws, lengths=lengths, metadata=np.array(json.dumps(config)))

    matrix_path = Path(source_meta["inputs"]["similarity_matrix"]["path"])
    graph_path = Path(source_meta["inputs"]["city_graph_B"]["path"])
    for name, path in (("similarity_matrix", matrix_path), ("city_graph_B", graph_path)):
        if joint.sha256(path) != source_meta["inputs"][name]["sha256"]:
            raise ValueError(f"base joint input changed: {name}")
    with np.load(matrix_path, allow_pickle=False) as z:
        raw_scores, refs = z["similarities"], z["ref_latlon"]
        if json.loads(str(z["metadata"]))["clip"] != args.clip:
            raise ValueError("POI clip and saved appearance queries disagree")
    roads, _crs, _stats = joint.city_roads(graph_path)
    reference_xy = np.column_stack(tf.transform(refs[:, 1], refs[:, 0]))
    low, shape, road_distance, rebuilt_starts, _n = joint.road_and_start_grids(roads, reference_xy, lengths[-1])
    if not np.array_equal(starts, rebuilt_starts):
        raise ValueError("rebuilt fine-scoring geometry differs from saved full coarse search")
    duration = float(geometry["identity"]["timestamps"][-1] - geometry["identity"]["timestamps"][0])
    batch = joint.SETTINGS["transform_batch"]
    for variant, source in sources.items():
        with np.load(source, allow_pickle=False) as z:
            base_cube = z["coarse_scores"]
        evidence = raw_scores if variant == "raw" else raw_scores - raw_scores.mean(axis=0, keepdims=True)
        field = joint.appearance_field(evidence, reference_xy, low, shape)
        for mode_index, mode in enumerate(SETTINGS["visibility_modes"]):
            started = time.perf_counter()
            all_hypothesis_scores = base_cube[None] - SETTINGS["poi_weight"] * losses[mode_index]
            coarse_hypothesis = all_hypothesis_scores.argmax(axis=0).astype(np.uint8)
            combined_cube = all_hypothesis_scores.max(axis=0)
            seeds = [seeds_for_hypothesis(scores, starts, yaws, lengths) for scores in all_hypothesis_scores]
            parameters, owners = refinement_parameters(seeds, lengths)
            rows = []
            for k in range(0, len(parameters), batch):
                params = parameters[k:k + batch]
                base = joint.score_parameters(params, unit_shape, field, road_distance, low, duration)
                penalty = poi_losses(params, geometry, allowed)[mode_index]
                chosen_hypothesis = penalty.argmin(axis=0)
                best_loss = penalty[chosen_hypothesis, np.arange(len(params))]
                total = base[:, 4] - SETTINGS["poi_weight"] * best_loss
                rows.append(np.c_[base, best_loss, chosen_hypothesis, total, owners[k:k + len(params)]])
            fine = np.concatenate(rows)
            fine = fine[np.argsort(-fine[:, 10], kind="stable")]
            selected_xy = joint.transformed_routes(fine[:1], unit_shape)[0]
            lon, lat = inverse.transform(selected_xy[:, 0], selected_xy[:, 1])
            label = f"{variant}__{mode}"
            metadata = {"variant": label, "ground_truth_used": False, "configuration": config,
                        "window": source_meta["window"], "crs": source_meta["crs"],
                        "query_times": query_times.tolist(), "coarse_transform_count": int(base_cube.size),
                        "hypothesis_count": len(allowed), "seeds_per_hypothesis": [len(s) for s in seeds],
                        "fine_transform_count": len(fine), "selected_hypothesis": int(fine[0, 9]),
                        "selected_parameters": fine[0].tolist(), "elapsed_s": time.perf_counter() - started,
                        "fine_columns": joint.PARAMETERS + ["poi_loss", "hypothesis", "combined_objective", "seed_hypothesis"]}
            output = args.out / f"{label}.npz"
            temp = output.with_suffix(".tmp.npz")
            np.savez_compressed(temp, route_latlon=np.c_[lat, lon], query_times=query_times, unit_shape=unit_shape,
                                starts=starts, yaws=yaws, lengths=lengths, coarse_scores=combined_cube,
                                coarse_hypothesis=coarse_hypothesis, fine_candidates=fine,
                                metadata=np.array(json.dumps(metadata)))
            temp.replace(output)
            print(f"saved {label}: all {base_cube.size} coarse poses, {len(allowed)} branches, "
                  f"{len(fine)} refined poses; selected branch {metadata['selected_hypothesis']}", flush=True)


def score(args):
    from pyproj import Transformer
    from monocular_osm.metrics import score_route

    truth = read_json(args.gt)["waypoints"]
    results = {}
    for path in sorted(args.out.glob("*__*.npz")):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            route, times, unit, fine = z["route_latlon"], z["query_times"], z["unit_shape"], z["fine_candidates"]
        fixes = [p for p in truth if meta["window"][0] <= p["t_sec"] <= meta["window"][1]]
        gt_ll = np.array([[p["lat"], p["lon"]] for p in fixes])
        gt_t = np.array([p["t_sec"] for p in fixes])
        q_ll = np.c_[np.interp(times, gt_t, gt_ll[:, 0]), np.interp(times, gt_t, gt_ll[:, 1])]
        tf = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
        gt_xy = np.column_stack(tf.transform(q_ll[:, 1], q_ll[:, 0]))
        pred_xy = np.column_stack(tf.transform(route[:, 1], route[:, 0]))
        metrics = score_route(route, gt_ll).to_dict()
        best = (float("inf"), None)
        for k in range(0, len(fine), joint.SETTINGS["transform_batch"]):
            poses = joint.transformed_routes(fine[k:k + joint.SETTINGS["transform_batch"]], unit)
            error = np.linalg.norm(poses - gt_xy, axis=2).mean(axis=1)
            pick = int(error.argmin())
            if error[pick] < best[0]:
                best = float(error[pick]), k + pick
        results[meta["variant"]] = {
            "symmetric_mean_m": metrics["symmetric_mean_m"], "start_error_m": metrics["start_error_m"],
            "mean_time_error_m": float(np.linalg.norm(pred_xy - gt_xy, axis=1).mean()),
            "selected_hypothesis": meta["selected_hypothesis"], "selected_parameters": meta["selected_parameters"],
            "fine_shortlist_oracle": {"deployable": False, "selection": "GT mean time error",
                                       "mean_time_error_m": best[0], "objective_rank": best[1] + 1,
                                       "hypothesis": int(fine[best[1], 9])},
            "hypotheses_tested": meta["hypothesis_count"], "fine_transform_count": len(fine),
        }
    joint.write_json(args.out / "scores.json", results)
    print(json.dumps(results, indent=2))


def check():
    # Camera at origin looking +x. Image-right is -y in the right-handed XY
    # map, so a positive horizontal image ray must match a negative yaw.
    geometry = {"unit_positions": np.zeros((1, 2)), "headings": np.array([0.]),
                "rays": np.radians([30.]), "bearing_available": np.array([True]), "weights": np.ones(1)}
    allowed = [[np.array([[50 * np.cos(-np.pi / 6), 50 * np.sin(-np.pi / 6)]])],
               [np.array([[1000., 0.]])]]
    losses = poi_losses(np.array([[0., 0., 0., 100.]]), geometry, allowed)
    np.testing.assert_allclose(losses[:, 0], 0, atol=1e-10)
    assert np.all(losses[:, 1] > 0)
    print("synthetic timed visibility and horizontal-bearing checks passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("infer")
    inf.add_argument("--joint", type=Path, default=ROOT / "output/joint_alignment_pilot_v3")
    inf.add_argument("--manifest", type=Path, default=ROOT / "output/poi_clue_probe/frozen_candidates.json")
    inf.add_argument("--clip", default="kitti_0009")
    inf.add_argument("--out", type=Path, default=ROOT / "output/joint_poi_alignment_pilot")
    scr = sub.add_parser("score")
    scr.add_argument("--out", type=Path, default=ROOT / "output/joint_poi_alignment_pilot")
    scr.add_argument("--gt", type=Path, default=ROOT / "ground_truth/kitti_drive_0009.json")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    {"infer": infer, "score": score, "check": lambda _: check()}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
