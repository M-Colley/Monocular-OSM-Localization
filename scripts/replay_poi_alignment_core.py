"""Post-hoc core replay on frozen dense inputs; labels opened only by score.

The 8610-reference pool is explicit. This does not establish sparse1500 support.
No new model forwards or downloads; original pilot artifacts remain unchanged.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["PROJ_NETWORK"] = "OFF"
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monocular_osm import poi_alignment as core


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def infer(args):
    from pyproj import Transformer
    frozen = args.original / "frozen_sources/pilot_joint_alignment.py"
    spec = importlib.util.spec_from_file_location("frozen_joint_geometry", frozen)
    joint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joint)
    original = read(args.original / "experiment.json")
    with np.load(args.joint / "raw.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        starts, lengths, query_times = z["starts"], z["lengths"], z["query_times"]
    if joint.sha256(frozen) != meta["configuration"]["script_sha256"]:
        raise ValueError("frozen geometry source changed")
    poi_source = original["poi_provenance"]
    artifact = Path(poi_source["artifact"])
    if joint.sha256(artifact) != poi_source["artifact_sha256"]:
        raise ValueError("frozen hypotheses changed")
    clusters = read(artifact)
    if clusters["selected_answer"] is not None:
        raise ValueError("hypotheses contain a selected answer")
    matrix = Path(meta["inputs"]["similarity_matrix"]["path"])
    graph = Path(meta["inputs"]["city_graph_B"]["path"])
    for name, path in (("similarity_matrix", matrix), ("city_graph_B", graph)):
        if joint.sha256(path) != meta["inputs"][name]["sha256"]:
            raise ValueError(f"frozen source changed: {name}")
    vo_path = args.joint / "verified_video_vo.npz"
    if joint.sha256(vo_path) != meta["vo"]["cache_sha256"]:
        raise ValueError("pixel-signed VO changed")
    with np.load(vo_path, allow_pickle=False) as z:
        vo, times = z["xz"], z["timestamps"]
        identity = json.loads(str(z["input_identity"]))
    if identity["intrinsics_source"] != "video dimensions and default70degree HFOV":
        raise ValueError("strict replay requires video-derived default intrinsics")
    with np.load(matrix, allow_pickle=False) as z:
        similarities, refs = z["similarities"], z["ref_latlon"]
        if not np.array_equal(query_times, z["query_times"]):
            raise ValueError("query identity differs from frozen pilot")
    tf = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    ref_xy = np.column_stack(tf.transform(refs[:, 1], refs[:, 0]))
    roads, _, _ = joint.city_roads(graph)
    low, shape, distance, rebuilt, _ = joint.road_and_start_grids(roads, ref_xy, lengths[-1])
    if not np.array_equal(starts, rebuilt):
        raise ValueError("road start reconstruction differs")
    config = core.POIAlignmentConfig()
    for settings in (joint.SETTINGS, original["settings"]):
        for key in settings.keys() & config.to_dict().keys():
            if np.asarray(config.to_dict()[key]).tolist() != np.asarray(settings[key]).tolist():
                raise ValueError(f"configuration differs: {key}")
    registration = {
        "status": "post-hoc development replay; earlier scores seen, fixed pilot settings",
        "ground_truth_used_during_inference": False, "defaults_changed": False,
        "variants": ["raw", "reference_mean"], "config": config.to_dict(),
        "core_sha256": joint.sha256(core.__file__), "runner_sha256": joint.sha256(__file__),
        "geometry_sha256": joint.sha256(frozen), "vo_sha256": joint.sha256(vo_path),
        "matrix_sha256": joint.sha256(matrix), "graph_sha256": joint.sha256(graph),
        "poi_sha256": joint.sha256(artifact), "reference_count": len(refs),
        "window": meta["window"], "crs": meta["crs"],
        "reference_pool_caveat": "full dense pool; sparse1500 support is untested"}
    run_variants(args, joint, registration, config, ref_xy, similarities, query_times,
                 low, shape, distance, starts, vo, times, identity, clusters)


def run_variants(args, joint, registration, config, ref_xy, similarities, query_times,
                 low, shape, distance, starts, vo, times, identity, clusters):
    args.out.mkdir(parents=True, exist_ok=True)
    record = args.out / "experiment.json"
    if record.exists() and read(record) != json.loads(json.dumps(registration)):
        raise ValueError("output belongs to another configuration; choose a new directory")
    joint.write_json(record, registration)
    road = core.RasterField(distance, tuple(low), config.road_cell_m, 200.,
                            provenance={"graph_sha256": registration["graph_sha256"]})
    for variant in registration["variants"]:
        path = args.out / f"{variant}.json"
        if path.exists():
            print(f"reusing frozen {variant}", flush=True)
            continue
        started = time.perf_counter()
        field = core.build_appearance_field(
            ref_xy, similarities, query_times, origin_xy=low, shape=shape,
            reference_mean=variant == "reference_mean", config=config,
            provenance={"matrix_sha256": registration["matrix_sha256"], "variant": variant})
        result = core.align_candidate_clusters(
            vo, times, identity["K"], clusters, map_crs=registration["crs"],
            appearance_field=field, road_distance_field=road,
            start_samples=starts, query_times=query_times, config=config, provenance=registration)
        result["elapsed_s"] = time.perf_counter() - started
        joint.write_json(path, result)
        print(f"saved {variant}: {result['selected']['hypothesis_id']} "
              f"objective={result['selected']['objective']:.12f}, {result['elapsed_s']:.1f}s", flush=True)
    # Every prediction exists before comparing to old predictions or opening labels.
    comparison = {}
    for variant in registration["variants"]:
        result = read(args.out / f"{variant}.json")
        with np.load(args.original / f"{variant}__distance_bearing.npz", allow_pickle=False) as z:
            old_record = json.loads(str(z["metadata"]))
        old = old_record["selected_parameters"]
        selected = result["selected"]
        parameters = [*selected["start_xy"], selected["yaw_deg"], selected["length_m"]]
        comparison[variant] = {
            "original_hypothesis": old_record["selected_hypothesis"],
            "core_hypothesis": selected["hypothesis_id"],
            "parameter_delta": (np.asarray(parameters) - old[:4]).tolist(),
            "objective_delta": selected["objective"] - old[10],
            "appearance_delta": selected["appearance"] - old[5],
            "poi_loss_delta": selected["poi_loss"] - old[8],
            "fine_pose_count_original": old_record["fine_transform_count"],
            "fine_pose_count_core": result["search"]["fine_pose_count"],
            "output_route_samples": len(selected["route_timestamps"]),
            "original_route_samples": len(old_record["query_times"])}
    joint.write_json(args.out / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


def score(args):
    from pyproj import Transformer
    from monocular_osm.metrics import score_route
    registration = read(args.out / "experiment.json")
    predictions = {v: read(args.out / f"{v}.json") for v in registration["variants"]}
    truth = read(args.gt)["waypoints"]
    fixes = [p for p in truth if registration["window"][0] <= p["t_sec"] <= registration["window"][1]]
    ll = np.array([[p["lat"], p["lon"]] for p in fixes])
    times = np.array([p["t_sec"] for p in fixes])
    tf = Transformer.from_crs("EPSG:4326", registration["crs"], always_xy=True)
    results = {}
    for variant, result in predictions.items():
        selected = result["selected"]
        route = np.asarray(selected["route_latlon"])
        query = np.asarray(selected["route_timestamps"])
        expected = np.c_[np.interp(query, times, ll[:, 0]), np.interp(query, times, ll[:, 1])]
        actual_xy = np.column_stack(tf.transform(route[:, 1], route[:, 0]))
        expected_xy = np.column_stack(tf.transform(expected[:, 1], expected[:, 0]))
        results[variant] = {**score_route(route, ll).to_dict(),
                            "mean_time_error_m": float(np.linalg.norm(actual_xy - expected_xy, axis=1).mean()),
                            "selected_hypothesis": selected["hypothesis_id"],
                            "output_route_samples": len(query)}
    (args.out / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inf = commands.add_parser("infer")
    inf.add_argument("--joint", type=Path, default=ROOT / "output/joint_alignment_pilot_v3")
    inf.add_argument("--original", type=Path, default=ROOT / "output/joint_poi_alignment_pilot")
    inf.add_argument("--out", type=Path, default=ROOT / "output/poi_alignment_core_replay")
    scr = commands.add_parser("score")
    scr.add_argument("--out", type=Path, default=ROOT / "output/poi_alignment_core_replay")
    scr.add_argument("--gt", type=Path, default=ROOT / "ground_truth/kitti_drive_0009.json")
    args = parser.parse_args()
    (infer if args.command == "infer" else score)(args)
