"""Bounded, input-only joint VO/appearance/road alignment experiment.

The primary inputs are video, saved dense-reference query similarities, and
the cached full-city Karlsruhe graph B. Default 70-degree video intrinsics
are used; no KITTI calibration, OXTS or GPS enters inference. Legacy unsigned
VO caches are not used. All candidates/predictions are saved before the
separate offline score command opens ground truth.

    python scripts/pilot_joint_alignment.py infer --threads 4
    python scripts/pilot_joint_alignment.py score
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "4"
os.environ["PROJ_NETWORK"] = "OFF"

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fixed before this experiment's scores. These are broad physical/noise
# assumptions, not parameters selected by ground-truth performance.
SETTINGS = {
    "variants": ["raw", "reference_mean"],
    "field_cell_m": 100.0, "appearance_sigma_m": 150.0,
    "row_std_floor": 0.005, "z_clip": 6.0, "empty_appearance": -6.0,
    "road_cell_m": 25.0, "road_sigma_m": 50.0, "road_weight": 0.25,
    "road_distance_cap_sigmas": 4.0,
    "average_speed_range_mps": [2.0, 25.0], "length_count": 9,
    "duration_prior_speed_mps": 5.5, "length_log_sigma": 0.7, "length_prior_weight": 0.10,
    "start_spacing_m": 100.0, "max_starts": 15000,
    "yaw_step_deg": 15.0, "keep_per_yaw_length": 16,
    "refine_basins": 64, "basin_position_bin_m": 500.0,
    "basin_yaw_bin_deg": 30.0, "basin_log_length_bin": 0.35,
    "refine_xy_offsets_m": [-150, -100, -50, 0, 50, 100, 150],
    "refine_yaw_offsets_deg": [-10, -5, 0, 5, 10],
    "refine_length_multipliers": [0.8, 0.9, 1.0, 1.1, 1.2],
    "transform_batch": 256, "frame_memory_mb": 128,
    "vo_max_frames": 256, "vo_default_hfov_deg": 70.0,
}
PARAMETERS = ["start_x_m", "start_y_m", "yaw_deg", "length_m", "objective",
              "appearance", "road_penalty", "length_penalty"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 ** 2):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def verified_vo(args, window, source_hash):
    """Recompute once, then reuse only an exact decoded-pixel signed cache."""
    import cv2
    from monocular_osm.frame_extraction import extract_frames
    from monocular_osm.visual_odometry import VO_ALGO_VERSION, default_intrinsics, estimate_trajectory

    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    frames = extract_frames(args.video, stride=args.frame_stride, max_frames=SETTINGS["vo_max_frames"],
                            start_sec=window[0], end_sec=window[1],
                            memory_mb=SETTINGS["frame_memory_mb"])
    try:
        height, width = frames.frames[0].shape[:2]
        K = default_intrinsics(width, height, hfov_deg=SETTINGS["vo_default_hfov_deg"])
        pixels = hashlib.sha256()
        for image in frames.frames:
            image = np.ascontiguousarray(image)
            pixels.update(str((image.shape, image.dtype.str)).encode("ascii"))
            pixels.update(memoryview(image).cast("B"))
        signature = {
            "video_sha256": source_hash, "decoded_pixels_sha256": pixels.hexdigest(),
            "timestamps": frames.timestamps, "fps": frames.fps, "frame_stride": args.frame_stride,
            "K": K.tolist(), "intrinsics_source": "video dimensions and default70degree HFOV",
            "vo_algorithm_version": VO_ALGO_VERSION,
            "vo_code_sha256": sha256(ROOT / "monocular_osm/visual_odometry.py"),
            "extractor_sha256": sha256(ROOT / "monocular_osm/frame_extraction.py"),
            "workers": args.threads, "seed": 0,
        }
        identity = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        cache = args.out / "verified_video_vo.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                if str(z["input_identity"]) != identity:
                    raise ValueError("saved pilot VO no longer matches every decoded pixel/input; use a new --out")
                xy, valid = z["xz"], z["valid"]
            print("reusing exact pixel-verified pilot VO", flush=True)
        else:
            print(f"estimating default-intrinsics VO on {len(frames.frames)} decoded frames", flush=True)
            trajectory = estimate_trajectory(frames.frames, K=K, n_workers=args.threads)
            xy, valid = trajectory.xz, trajectory.valid
            np.savez_compressed(cache, xz=xy, valid=valid, timestamps=frames.timestamps,
                                input_identity=np.array(identity))
        return xy, np.asarray(frames.timestamps), {**signature, "valid_fraction": float(valid.mean()),
                                                    "cache_sha256": sha256(cache)}
    finally:
        frames.close()


def city_roads(path):
    """Read cached graph B directly; never fetch a graph or use a recording box."""
    import networkx as nx
    from pyproj import CRS
    from shapely import wkt

    graph = nx.read_graphml(path, node_type=int, force_multigraph=True)
    crs = CRS.from_user_input(graph.graph["crs"])
    if not crs.is_projected or not np.isclose(crs.axis_info[0].unit_conversion_factor, 1.0):
        raise ValueError("cached city graph must have a projected metric CRS")
    samples = []
    for a, b, data in graph.edges(data=True):
        geometry = data.get("geometry")
        points = (np.asarray(wkt.loads(geometry).coords, dtype=float)[:, :2] if geometry else
                  np.array([[float(graph.nodes[a]["x"]), float(graph.nodes[a]["y"])],
                            [float(graph.nodes[b]["x"]), float(graph.nodes[b]["y"])]]))
        for start, end in zip(points, points[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end - start) / (SETTINGS["road_cell_m"] / 2))))
            samples.append(np.linspace(start, end, count + 1))
    if not samples:
        raise ValueError("city graph contains no road geometry")
    stats = {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
             "crs": crs.to_string(), "source": "cached full Karlsruhe city graph B"}
    return np.concatenate(samples), crs, stats


def road_and_start_grids(road_points, reference_xy, max_length):
    from scipy.ndimage import distance_transform_edt

    cell = SETTINGS["field_cell_m"]
    margin = max_length + 3 * SETTINGS["appearance_sigma_m"]
    all_low = np.minimum(road_points.min(axis=0), reference_xy.min(axis=0))
    all_high = np.maximum(road_points.max(axis=0), reference_xy.max(axis=0))
    low = np.floor((all_low - margin) / cell) * cell
    high = np.ceil((all_high + margin) / cell) * cell
    shape = np.ceil((high - low) / cell).astype(int)[::-1] + 1
    road_cell = SETTINGS["road_cell_m"]
    road_shape = np.ceil((high - low) / road_cell).astype(int)[::-1] + 1
    mask = np.zeros(tuple(road_shape), dtype=bool)
    indices = np.rint((road_points - low) / road_cell).astype(int)
    mask[indices[:, 1], indices[:, 0]] = True
    road_distance = (distance_transform_edt(~mask) * road_cell).astype(np.float32)

    spacing = SETTINGS["start_spacing_m"]
    bins = np.floor((road_points - low) / spacing).astype(int)
    flat = bins[:, 1] * (int(np.ceil((high[0] - low[0]) / spacing)) + 1) + bins[:, 0]
    centers = low + (bins + 0.5) * spacing
    distance = np.sum((road_points - centers) ** 2, axis=1)
    order = np.lexsort((np.arange(len(flat)), distance, flat))
    first = np.r_[True, flat[order[1:]] != flat[order[:-1]]]
    starts = road_points[order[first]]
    uncapped = len(starts)
    if uncapped > SETTINGS["max_starts"]:
        starts = starts[np.linspace(0, uncapped - 1, SETTINGS["max_starts"]).astype(int)]
    return low, tuple(int(value) for value in shape), road_distance, starts, uncapped


def appearance_field(scores, reference_xy, low, shape):
    from scipy.ndimage import gaussian_filter

    cell = SETTINGS["field_cell_m"]
    indices = np.rint((reference_xy - low) / cell).astype(int)
    counts = np.zeros(shape, dtype=np.float32)
    np.add.at(counts, (indices[:, 1], indices[:, 0]), 1)
    sigma = SETTINGS["appearance_sigma_m"] / cell
    density = gaussian_filter(counts, sigma=sigma, mode="constant", truncate=3)
    supported = density > 1e-6
    z = (scores - scores.mean(axis=1, keepdims=True)) / np.maximum(
        scores.std(axis=1, keepdims=True), SETTINGS["row_std_floor"])
    z = np.clip(z, -SETTINGS["z_clip"], SETTINGS["z_clip"])
    field = np.full((len(scores), *shape), SETTINGS["empty_appearance"], dtype=np.float32)
    for q in range(len(scores)):
        weight = np.zeros(shape, dtype=np.float32)
        np.add.at(weight, (indices[:, 1], indices[:, 0]), np.exp(z[q]))
        weight = gaussian_filter(weight, sigma=sigma, mode="constant", truncate=3)
        field[q, supported] = np.log(np.maximum(weight[supported] / density[supported], 1e-8))
    return field


def sample_grid(grid, xy, low, cell, outside):
    """Bilinear samples for [candidate,query,xy]; temporal rows never mix."""
    col, row = (xy[..., 0] - low[0]) / cell, (xy[..., 1] - low[1]) / cell
    height, width = grid.shape[-2:]
    inside = (col >= 0) & (row >= 0) & (col < width - 1) & (row < height - 1)
    x = np.floor(np.clip(col, 0, width - 2)).astype(int)
    y = np.floor(np.clip(row, 0, height - 2)).astype(int)
    fx, fy = np.clip(col - x, 0, 1), np.clip(row - y, 0, 1)
    if grid.ndim == 3:
        q = np.arange(xy.shape[1])[None, :]
        a, b, c, d = grid[q, y, x], grid[q, y, x + 1], grid[q, y + 1, x], grid[q, y + 1, x + 1]
    else:
        a, b, c, d = grid[y, x], grid[y, x + 1], grid[y + 1, x], grid[y + 1, x + 1]
    value = (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * c + fx * d)
    return np.where(inside, value, outside)


def transformed_routes(parameters, unit_shape):
    angle = np.radians(parameters[:, 2])[:, None]
    cos, sin = np.cos(angle), np.sin(angle)
    x = cos * unit_shape[None, :, 0] - sin * unit_shape[None, :, 1]
    y = sin * unit_shape[None, :, 0] + cos * unit_shape[None, :, 1]
    return np.stack([x, y], axis=2) * parameters[:, None, 3:4] + parameters[:, None, :2]


def score_parameters(parameters, unit_shape, field, road_distance, low, duration):
    xy = transformed_routes(parameters, unit_shape)
    app = sample_grid(field, xy, low, SETTINGS["field_cell_m"], SETTINGS["empty_appearance"]).mean(axis=1)
    distances = sample_grid(road_distance, xy, low, SETTINGS["road_cell_m"],
                            SETTINGS["road_sigma_m"] * SETTINGS["road_distance_cap_sigmas"])
    road_penalty = SETTINGS["road_weight"] * np.minimum(
        distances / SETTINGS["road_sigma_m"], SETTINGS["road_distance_cap_sigmas"]) ** 2
    road_penalty = road_penalty.mean(axis=1)
    length_penalty = SETTINGS["length_prior_weight"] * (
        np.log(parameters[:, 3] / (duration * SETTINGS["duration_prior_speed_mps"]))
        / SETTINGS["length_log_sigma"]) ** 2
    total = app - road_penalty - length_penalty
    return np.c_[parameters, total, app, road_penalty, length_penalty]


def search(starts, unit_shape, lengths, yaws, field, road_distance, low, duration, variant):
    cube = np.empty((len(lengths), len(yaws), len(starts)), dtype=np.float32)
    retained = []
    batch = SETTINGS["transform_batch"]
    for i, length in enumerate(lengths):
        for j, yaw in enumerate(yaws):
            params = np.c_[starts, np.full(len(starts), yaw), np.full(len(starts), length)]
            evaluated = np.concatenate([
                score_parameters(params[k:k + batch], unit_shape, field, road_distance, low, duration)
                for k in range(0, len(params), batch)])
            cube[i, j] = evaluated[:, 4]
            retained.append(evaluated[np.argsort(-evaluated[:, 4], kind="stable")[:SETTINGS["keep_per_yaw_length"]]])
        print(f"{variant}: coarse length {i + 1}/{len(lengths)}, best objective {cube[:i + 1].max():.4f}", flush=True)
    coarse = np.concatenate(retained)
    order = np.argsort(-coarse[:, 4], kind="stable")
    seeds, seen = [], set()
    for index in order:
        candidate = coarse[index]
        key = (round(candidate[0] / SETTINGS["basin_position_bin_m"]),
               round(candidate[1] / SETTINGS["basin_position_bin_m"]),
               round(candidate[2] / SETTINGS["basin_yaw_bin_deg"]),
               round(np.log(candidate[3]) / SETTINGS["basin_log_length_bin"]))
        if key not in seen:
            seen.add(key)
            seeds.append(candidate)
            if len(seeds) == SETTINGS["refine_basins"]:
                break
    fine_parameters = []
    for seed in seeds:
        for multiplier in SETTINGS["refine_length_multipliers"]:
            length = seed[3] * multiplier
            if not lengths[0] <= length <= lengths[-1]:
                continue
            for angle_offset in SETTINGS["refine_yaw_offsets_deg"]:
                for dx in SETTINGS["refine_xy_offsets_m"]:
                    for dy in SETTINGS["refine_xy_offsets_m"]:
                        fine_parameters.append([seed[0] + dx, seed[1] + dy,
                                                (seed[2] + angle_offset) % 360, length])
    fine_parameters = np.asarray(fine_parameters)
    fine = np.concatenate([
        score_parameters(fine_parameters[k:k + batch], unit_shape, field, road_distance, low, duration)
        for k in range(0, len(fine_parameters), batch)])
    fine = fine[np.argsort(-fine[:, 4], kind="stable")]
    return cube, coarse, fine


def infer(args):
    from pyproj import Transformer

    args.out.mkdir(parents=True, exist_ok=True)
    inputs = {"video": args.video, "similarity_matrix": args.matrix, "city_graph_B": args.graph}
    hashes = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in inputs.items()}
    configuration = {"settings": SETTINGS, "inputs": hashes, "frame_stride": args.frame_stride,
                     "threads": args.threads, "script_sha256": sha256(__file__),
                     "phase": "inference", "ground_truth_read": False,
                     "disclosure": "development pilot after earlier KITTI results; no GT-dependent parameter selection",
                     "intrinsics": "default70degree HFOV from video dimensions; no KITTI calibration/OXTS"}
    registration = args.out / "experiment.json"
    if registration.exists() and json.loads(registration.read_text()) != configuration:
        raise ValueError("output belongs to a different frozen configuration; use a new --out")
    write_json(registration, configuration)
    with np.load(args.matrix, allow_pickle=False) as z:
        scores, refs, query_times = z["similarities"], z["ref_latlon"], z["query_times"]
        source_metadata = json.loads(str(z["metadata"]))
    if source_metadata.get("clip") != "kitti_0009" or source_metadata.get("gt_used") is not False:
        raise ValueError("pilot expects saved GT-free KITTI0009 dense-pool queries")
    window = source_metadata["query_window"]
    vo, times, vo_metadata = verified_vo(args, window, hashes["video"]["sha256"])
    if times[-1] < query_times[-1]:
        raise ValueError("bounded VO did not cover every saved query time")
    arc = float(np.linalg.norm(np.diff(vo, axis=0), axis=1).sum())
    if arc <= 1e-6:
        raise ValueError("VO has no usable route shape")
    query_vo = np.c_[np.interp(query_times, times, vo[:, 0]), np.interp(query_times, times, vo[:, 1])]
    unit_shape = (query_vo - vo[0]) / arc
    duration = float(times[-1] - times[0])
    lengths = np.geomspace(duration * SETTINGS["average_speed_range_mps"][0],
                           duration * SETTINGS["average_speed_range_mps"][1], SETTINGS["length_count"])
    yaws = np.arange(0, 360, SETTINGS["yaw_step_deg"])
    roads, crs, graph_stats = city_roads(args.graph)
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    reference_xy = np.column_stack(tf.transform(refs[:, 1], refs[:, 0]))
    low, shape, road_distance, starts, uncapped = road_and_start_grids(roads, reference_xy, lengths[-1])
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    print(f"city grid {shape}, {len(starts)}/{uncapped} road starts, {len(query_times)} queries", flush=True)
    metadata = {"inputs": hashes, "configuration": configuration, "vo": vo_metadata,
                "graph": graph_stats, "window": window, "crs": crs.to_string(),
                "field_origin_xy": low.tolist(), "field_shape": list(shape),
                "start_count": len(starts), "uncapped_start_count": uncapped,
                "coarse_transform_count": len(starts) * len(yaws) * len(lengths),
                "parameter_columns": PARAMETERS,
                "note": "saved VPR source predates pixel hashes; same video/session and query times, current video hash recorded"}
    write_json(args.out / "inputs_derived.json", metadata)
    for variant in SETTINGS["variants"]:
        output = args.out / f"{variant}.npz"
        if output.exists():
            print(f"reusing frozen {variant} output", flush=True)
            continue
        started = time.perf_counter()
        evidence = scores if variant == "raw" else scores - scores.mean(axis=0, keepdims=True)
        field = appearance_field(evidence, reference_xy, low, shape)
        cube, coarse, fine = search(starts, unit_shape, lengths, yaws, field,
                                    road_distance, low, duration, variant)
        selected_xy = transformed_routes(fine[:1], unit_shape)[0]
        lon, lat = inverse.transform(selected_xy[:, 0], selected_xy[:, 1])
        route = np.column_stack([lat, lon])
        result_metadata = {**metadata, "variant": variant, "gt_used": False,
                           "elapsed_s": time.perf_counter() - started,
                           "fine_transform_count": len(fine), "selected_parameters": fine[0].tolist()}
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, route_latlon=route, query_times=query_times, unit_shape=unit_shape,
                            starts=starts, yaws=yaws, lengths=lengths, coarse_scores=cube,
                            coarse_shortlist=coarse, fine_candidates=fine,
                            metadata=np.array(json.dumps(result_metadata)))
        temporary.replace(output)
        print(f"saved {variant}: {len(fine)} fine transforms, {result_metadata['elapsed_s']:.1f}s", flush=True)


def score(args):
    """Offline only: selected predictions and explicitly labelled GT oracles."""
    from pyproj import Transformer
    from monocular_osm.metrics import score_route

    ground_truth = json.loads(args.gt.read_text(encoding="utf-8"))["waypoints"]
    results = {}
    for source in sorted(args.out.glob("*.npz")):
        if source.name == "verified_video_vo.npz":
            continue
        with np.load(source, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            route, times, unit = z["route_latlon"], z["query_times"], z["unit_shape"]
            fine = z["fine_candidates"]
        fixes = [p for p in ground_truth if meta["window"][0] <= p["t_sec"] <= meta["window"][1]]
        gt_ll = np.array([[p["lat"], p["lon"]] for p in fixes])
        gt_times = np.array([p["t_sec"] for p in fixes])
        q_ll = np.c_[np.interp(times, gt_times, gt_ll[:, 0]), np.interp(times, gt_times, gt_ll[:, 1])]
        tf = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
        gt_xy = np.column_stack(tf.transform(q_ll[:, 1], q_ll[:, 0]))
        pred_xy = np.column_stack(tf.transform(route[:, 1], route[:, 0]))
        headline = score_route(route, gt_ll).to_dict()
        best_oracle = (float("inf"), None)
        for k in range(0, len(fine), SETTINGS["transform_batch"]):
            candidates = transformed_routes(fine[k:k + SETTINGS["transform_batch"]], unit)
            error = np.linalg.norm(candidates - gt_xy, axis=2).mean(axis=1)
            pick = int(error.argmin())
            if error[pick] < best_oracle[0]:
                best_oracle = (float(error[pick]), k + pick)
        # Continuous similarity fit is an oracle shape diagnostic, not an answer.
        x, y = unit - unit.mean(axis=0), gt_xy - gt_xy.mean(axis=0)
        u, singular, vt = np.linalg.svd(x.T @ y)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        scale = float(np.sum((x @ rotation) * y) / np.sum(x ** 2))
        oracle_xy = scale * x @ rotation + gt_xy.mean(axis=0)
        inverse = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
        lon, lat = inverse.transform(oracle_xy[:, 0], oracle_xy[:, 1])
        results[meta["variant"]] = {
            "symmetric_mean_m": headline["symmetric_mean_m"], "start_error_m": headline["start_error_m"],
            "mean_time_aligned_error_m": float(np.linalg.norm(pred_xy - gt_xy, axis=1).mean()),
            "selected_parameters": meta["selected_parameters"], "elapsed_s": meta["elapsed_s"],
            "coarse_transform_count": meta["coarse_transform_count"], "fine_transform_count": len(fine),
            "fine_shortlist_oracle": {"deployable": False, "selection": "GT mean time-aligned error",
                                       "mean_time_error_m": best_oracle[0], "objective_rank": best_oracle[1] + 1},
            "continuous_shape_oracle": {"deployable": False, "source": "GT similarity alignment",
                                         "length_m": scale, "mean_time_error_m": float(
                                             np.linalg.norm(oracle_xy - gt_xy, axis=1).mean()),
                                         "symmetric_mean_m": score_route(np.c_[lat, lon], gt_ll).symmetric_mean_m},
        }
    write_json(args.out / "scores.json", results)
    print(json.dumps(results, indent=2))


def check():
    # Independent simple fields verify spatial/temporal indexing and transform
    # conventions before inference, without opening project data or GT.
    shape = np.array([[0., 0.], [1., 0.]])
    params = np.array([[10., 20., 90., 2.]])
    np.testing.assert_allclose(transformed_routes(params, shape)[0], [[10, 20], [10, 22]], atol=1e-8)
    field = np.stack([np.full((3, 3), 1.), np.full((3, 3), 7.)])
    samples = sample_grid(field, np.array([[[0.5, 0.5], [1.5, 1.5]]]), np.zeros(2), 1, -9)
    np.testing.assert_array_equal(samples, [[1., 7.]])
    print("synthetic transform and spatial/temporal interpolation checks passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inf = sub.add_parser("infer")
    inf.add_argument("--out", type=Path, default=ROOT / "output/joint_alignment_pilot")
    inf.add_argument("--video", type=Path, default=ROOT / "data/kitti/drive_0009.mp4")
    inf.add_argument("--matrix", type=Path, default=ROOT / "output/query_preprocessing_pilot/kitti_0009_warp.npz")
    inf.add_argument("--graph", type=Path, default=ROOT / "output/graph_filter/graph_B_karlsruhe_city.graphml")
    inf.add_argument("--threads", type=int, choices=[2, 3, 4], default=4)
    inf.add_argument("--frame-stride", type=int, default=3)
    scr = sub.add_parser("score")
    scr.add_argument("--out", type=Path, default=ROOT / "output/joint_alignment_pilot")
    scr.add_argument("--gt", type=Path, default=ROOT / "ground_truth/kitti_drive_0009.json")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    {"infer": infer, "score": score, "check": lambda _: check()}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
