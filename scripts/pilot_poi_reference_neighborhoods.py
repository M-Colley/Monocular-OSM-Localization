"""Fixed POI budgets from saved dense references; labels open only in score.

Post-hoc dev experiment. Dense-origin ID-hash sampling does not prove live
imagery fetch returns the same pool. No new forwards/downloads/default changes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
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

PLANS = (
    {"id": "all8610", "radius_m": None, "cap_per_branch": None},
    {"id": "radius1000_cap500", "radius_m": 1000., "cap_per_branch": 500},
    {"id": "radius1500_cap750", "radius_m": 1500., "cap_per_branch": 750},
)
VARIANTS = ("raw", "reference_mean")
HASH_SALT = "poi-neighborhood-reference-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def plan_neighborhoods(reference_ids, reference_xy, branch_centers, radius_m, cap):
    """Apply the same score-free ID-hash ordering and cap to every branch."""
    ids = [str(v) for v in reference_ids]
    xy, centers = np.asarray(reference_xy, float), np.asarray(branch_centers, float)
    if (xy.shape != (len(ids), 2) or centers.ndim != 2 or centers.shape[1] != 2
            or not len(centers) or len(set(ids)) != len(ids)
            or not np.isfinite(xy).all() or not np.isfinite(centers).all()
            or not np.isfinite(radius_m) or radius_m <= 0
            or isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("reference identities, branch centers and budgets must be valid")
    hashes = [hashlib.sha256((HASH_SALT + "\0" + identifier).encode()).hexdigest() for identifier in ids]
    union, branches = set(), []
    for index, center in enumerate(centers):
        eligible = np.flatnonzero(np.linalg.norm(xy - center, axis=1) <= radius_m).tolist()
        selected = sorted(eligible, key=lambda i: (hashes[i], ids[i]))[:cap]
        union.update(selected)
        branches.append({"hypothesis_id": f"hypothesis_{index}", "center_xy": center.tolist(),
                         "eligible_count": len(eligible), "selected_count": len(selected),
                         "selected_indices": selected, "selected_ids": [ids[i] for i in selected]})
    selected = sorted(union)
    return {"radius_m": radius_m, "cap_per_branch": cap, "hash_salt": HASH_SALT,
            "selection_rule": "centroid radius, ID SHA256 ascending, cap per branch, union",
            "branches": branches, "selected_indices": selected, "selected_ids": [ids[i] for i in selected],
            "selected_count": len(selected)}


def load_inputs(args):
    from pyproj import Transformer
    frozen = args.original / "frozen_sources/pilot_joint_alignment.py"
    spec = importlib.util.spec_from_file_location("frozen_neighborhood_geometry", frozen)
    joint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(joint)
    baseline = read(args.baseline / "experiment.json")
    original = read(args.original / "experiment.json")
    if sha256(core.__file__) != baseline["core_sha256"]:
        snapshot = args.baseline / "frozen_sources/poi_alignment.py"
        previous_tree = ast.dump(ast.parse(snapshot.read_text(encoding="utf-8")))
        current_tree = ast.dump(ast.parse(Path(core.__file__).read_text(encoding="utf-8")))
        if sha256(snapshot) != baseline["core_sha256"] or previous_tree != current_tree:
            raise ValueError("executable core differs from the frozen full-pool replay")
    joint_raw = args.joint / "raw.npz"
    if sha256(joint_raw) != original["source_joint"]["raw"]["sha256"]:
        raise ValueError("frozen joint raw artifact changed")
    with np.load(joint_raw, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        starts, lengths, query_times = z["starts"], z["lengths"], z["query_times"]
    if meta["crs"] != baseline["crs"] or meta["window"] != baseline["window"]:
        raise ValueError("frozen CRS or scoring window differs from baseline")
    if sha256(frozen) != baseline["geometry_sha256"]:
        raise ValueError("frozen geometry source changed")
    paths = {"matrix": Path(meta["inputs"]["similarity_matrix"]["path"]),
             "graph": Path(meta["inputs"]["city_graph_B"]["path"]),
             "vo": args.joint / "verified_video_vo.npz",
             "poi": Path(original["poi_provenance"]["artifact"])}
    for key, path in paths.items():
        if sha256(path) != baseline[f"{key}_sha256"]:
            raise ValueError(f"frozen {key} source changed")
    clusters = read(paths["poi"])
    if clusters["selected_answer"] is not None:
        raise ValueError("frozen clusters contain a selected answer")
    with np.load(paths["vo"], allow_pickle=False) as z:
        vo, times = z["xz"], z["timestamps"]
        identity = json.loads(str(z["input_identity"]))
    if identity["intrinsics_source"] != "video dimensions and default70degree HFOV":
        raise ValueError("strict input requires video-derived intrinsics")
    with np.load(paths["matrix"], allow_pickle=False) as z:
        similarities, refs = z["similarities"], z["ref_latlon"]
        matrix_meta = json.loads(str(z["metadata"]))
        if not np.array_equal(query_times, z["query_times"]):
            raise ValueError("query row identities changed")
    archive = Path(matrix_meta["references"]["embedding_file"])
    if sha256(archive) != matrix_meta["references"]["sha256"]:
        raise ValueError("dense ID archive changed")
    with np.load(archive, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        if not np.array_equal(refs, z["latlon"]):
            raise ValueError("reference archive and matrix column identities differ")
    if len(ids) != baseline["reference_count"]:
        raise ValueError("full reference count differs from baseline")
    paths["joint_raw"] = joint_raw
    tf = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    xy = np.column_stack(tf.transform(refs[:, 1], refs[:, 0]))
    centers = []
    for cluster in clusters["candidate_clusters"]:
        pois = np.array([[p["lon"], p["lat"]] for p in cluster["pois"]])
        centers.append(np.column_stack(tf.transform(pois[:, 0], pois[:, 1])).mean(axis=0))
    roads, _, _ = joint.city_roads(paths["graph"])
    low, shape, distance, rebuilt, _ = joint.road_and_start_grids(roads, xy, lengths[-1])
    if not np.array_equal(starts, rebuilt):
        raise ValueError("whole-city start reconstruction differs")
    return dict(joint=joint, baseline=baseline, paths=paths, meta=meta, clusters=clusters,
                vo=vo, times=times, identity=identity, similarities=similarities,
                refs=refs, ids=ids, xy=xy, centers=np.asarray(centers), query_times=query_times,
                low=low, shape=shape, distance=distance, starts=starts, lengths=lengths, archive=archive)


def infer(args):
    data = load_inputs(args)
    config = core.POIAlignmentConfig()
    if json.loads(json.dumps(config.to_dict())) != data["baseline"]["config"]:
        raise ValueError("core defaults differ from frozen full-pool control")
    planned = {}
    for settings in PLANS:
        if settings["radius_m"] is None:
            plan = {"selected_indices": list(range(len(data["ids"]))),
                    "selected_ids": data["ids"].tolist(), "selected_count": len(data["ids"]),
                    "branches": [], "selection_rule": "all references"}
        else:
            plan = plan_neighborhoods(data["ids"], data["xy"], data["centers"],
                                      settings["radius_m"], settings["cap_per_branch"])
        planned[settings["id"]] = {**settings, **plan}
    registration = {
        "status": "post-hoc dev; fixed settings registered before new scoring",
        "ground_truth_used_during_inference": False, "defaults_changed": False,
        "plans": list(PLANS), "appearance_variants": list(VARIANTS),
        "runs": [f"{plan['id']}__{variant}" for plan in PLANS for variant in VARIANTS],
        "core_config": config.to_dict(), "core_sha256": sha256(core.__file__),
        "runner_sha256": sha256(__file__), "baseline_sha256": sha256(args.baseline / "experiment.json"),
        "source_hashes": {key: sha256(path) for key, path in data["paths"].items()},
        "dense_id_archive_sha256": sha256(data["archive"]),
        "start_samples_sha256": array_digest(data["starts"]),
        "query_times_sha256": array_digest(data["query_times"]),
        "lengths_sha256": array_digest(data["lengths"]),
        "pool_size": len(data["ids"]), "hypothesis_count": len(data["centers"]),
        "centers_rule": "project every branch POI, arithmetic centroid",
        "hash_salt": HASH_SALT, "crs": data["meta"]["crs"], "window": data["meta"]["window"],
        "start_policy": "same whole-city road starts for every plan and hypothesis",
        "caveat": "dense-origin saved samples do not prove a live fetch returns that pool"}
    run_plans(args, data, registration, config, planned)


def run_plans(args, data, registration, config, planned):
    args.out.mkdir(parents=True, exist_ok=True)
    record = args.out / "experiment.json"
    if record.exists() and read(record) != json.loads(json.dumps(registration)):
        raise ValueError("output belongs to another registered configuration")
    if (args.out / "predictions_frozen.json").exists():
        load_frozen_predictions(args.out)
        print("validated complete frozen predictions; no files rewritten", flush=True)
        return
    write(record, registration)
    # Freeze ALL reference IDs/configurations before the first alignment.
    for name, plan in planned.items():
        path = args.out / f"{name}_references.json"
        if path.exists() and read(path) != plan:
            raise ValueError("frozen reference selection changed")
        write(path, plan)
        print(f"planned {name}: union={plan['selected_count']}; branches="
              f"{[p['selected_count'] for p in plan['branches']]}", flush=True)
    road = core.RasterField(data["distance"], tuple(data["low"]), config.road_cell_m, 200.,
                            provenance={"graph_sha256": registration["source_hashes"]["graph"]})
    for name, plan in planned.items():
        indices = np.asarray(plan["selected_indices"], dtype=int)
        for variant in VARIANTS:
            label = f"{name}__{variant}"
            path = args.out / f"{label}.json"
            source = {**registration, "run": label,
                      "reference_plan_sha256": sha256(args.out / f"{name}_references.json")}
            if path.exists():
                if read(path)["provenance"]["caller_sources"] != json.loads(json.dumps(source)):
                    raise ValueError("existing prediction has incompatible provenance")
                continue
            started = time.perf_counter()
            field = core.build_appearance_field(
                data["xy"][indices], np.ascontiguousarray(data["similarities"][:, indices]), data["query_times"],
                origin_xy=data["low"], shape=data["shape"], config=config,
                reference_mean=variant == "reference_mean", provenance=source)
            result = core.align_candidate_clusters(
                data["vo"], data["times"], data["identity"]["K"], data["clusters"], map_crs=registration["crs"],
                appearance_field=field, road_distance_field=road, start_samples=data["starts"],
                query_times=data["query_times"], config=config, provenance=source)
            result["elapsed_s"] = time.perf_counter() - started
            write(path, result)
            print(f"saved {label}: {result['selected']['hypothesis_id']}, "
                  f"objective={result['selected']['objective']:.9f}, {result['elapsed_s']:.1f}s", flush=True)
    write(args.out / "predictions_frozen.json", {
        "experiment_sha256": sha256(record), "ground_truth_opened": False,
        "predictions": {label: sha256(args.out / f"{label}.json") for label in registration["runs"]}})


def load_frozen_predictions(out):
    registration = read(out / "experiment.json")
    frozen = read(out / "predictions_frozen.json")
    if frozen["experiment_sha256"] != sha256(out / "experiment.json"):
        raise ValueError("registered configuration changed after freezing predictions")
    if set(frozen["predictions"]) != set(registration["runs"]):
        raise ValueError("not every registered prediction has frozen")
    predictions = {}
    for label in registration["runs"]:
        path = out / f"{label}.json"
        if sha256(path) != frozen["predictions"][label]:
            raise ValueError("frozen prediction changed")
        predictions[label] = read(path)
        name = label.split("__")[0]
        plan_path = out / f"{name}_references.json"
        if predictions[label]["provenance"]["caller_sources"]["reference_plan_sha256"] != sha256(plan_path):
            raise ValueError("reference plan changed after prediction")
    return registration, predictions


def score(args):
    from pyproj import Transformer
    from monocular_osm.metrics import score_route
    registration, predictions = load_frozen_predictions(args.out)
    # Labels open only after every registered prediction passed its hash check.
    truth = read(args.gt)["waypoints"]
    fixes = [p for p in truth if registration["window"][0] <= p["t_sec"] <= registration["window"][1]]
    ll = np.array([[p["lat"], p["lon"]] for p in fixes])
    times = np.array([p["t_sec"] for p in fixes])
    tf = Transformer.from_crs("EPSG:4326", registration["crs"], always_xy=True)
    results = {}
    for label, prediction in predictions.items():
        name = label.split("__")[0]
        plan = read(args.out / f"{name}_references.json")
        selected = prediction["selected"]
        route, query = np.asarray(selected["route_latlon"]), np.asarray(selected["route_timestamps"])
        expected = np.c_[np.interp(query, times, ll[:, 0]), np.interp(query, times, ll[:, 1])]
        actual_xy = np.column_stack(tf.transform(route[:, 1], route[:, 0]))
        expected_xy = np.column_stack(tf.transform(expected[:, 1], expected[:, 0]))
        results[label] = {
            **score_route(route, ll).to_dict(), "reference_count": plan["selected_count"],
            "per_branch_selected_counts": [p["selected_count"] for p in plan["branches"]],
            "mean_time_error_m": float(np.linalg.norm(actual_xy - expected_xy, axis=1).mean()),
            "selected_hypothesis": selected["hypothesis_id"],
            "transform": [*selected["start_xy"], selected["yaw_deg"], selected["length_m"]],
            "hypotheses_tested": prediction["search"]["hypothesis_count"],
            "coarse_pose_count": prediction["search"]["coarse_pose_count"]}
    write(args.out / "scores.json", {"ground_truth_sha256": sha256(args.gt),
                                     "prediction_freeze_sha256": sha256(args.out / "predictions_frozen.json"),
                                     "results": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inf = commands.add_parser("infer")
    inf.add_argument("--joint", type=Path, default=ROOT / "output/joint_alignment_pilot_v3")
    inf.add_argument("--original", type=Path, default=ROOT / "output/joint_poi_alignment_pilot")
    inf.add_argument("--baseline", type=Path, default=ROOT / "output/poi_alignment_core_replay_v2")
    inf.add_argument("--out", type=Path, default=ROOT / "output/poi_reference_neighborhood_pilot")
    scr = commands.add_parser("score")
    scr.add_argument("--out", type=Path, default=ROOT / "output/poi_reference_neighborhood_pilot")
    scr.add_argument("--gt", type=Path, default=ROOT / "ground_truth/kitti_drive_0009.json")
    args = parser.parse_args()
    (infer if args.command == "infer" else score)(args)
