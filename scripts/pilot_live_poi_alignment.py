"""Fresh POI neighborhoods; both predictions freeze before separate GT scoring.

Post-hoc dev study. Detached source, isolated reference/model caches, fresh
video queries; only already cached model weights may be copied. No dense-pilot
references or candidate winners are loaded. Regional failures remain explicit.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["PROJ_NETWORK"] = "OFF"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "8aa9f05f80bd1e432546cc2575199117371d6612"
SETTINGS = {"radius_m": 1000., "cap_per_branch": 500, "n_query": 80,
            "query_mode": "warp", "query_batch_size": 2, "model": "megaloc",
            "source": "mapillary", "device": "cpu", "threads": 4,
            "frame_stride": 3, "max_frames": 256, "frame_memory_mb": 128,
            "minimum_free_commit_gib": 8., "ram_target_bytes": 3_000_000_000,
            "appearance_variants": ["raw", "reference_mean"]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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


def memory_status():
    """Windows system commit headroom and process peak private commit."""
    class Performance(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            (name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
             "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
             "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    status, process = Performance(), Counters()
    status.cb, process.cb = ctypes.sizeof(status), ctypes.sizeof(process)
    psapi, kernel = ctypes.WinDLL("psapi", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetPerformanceInfo(ctypes.byref(status), ctypes.sizeof(status)):
        raise OSError("cannot measure system commit")
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.sizeof(process)):
        raise OSError("cannot measure pilot memory")
    return {"available_commit_bytes": (status.CommitLimit - status.CommitTotal) * status.PageSize,
            "private_bytes": process.PrivateUsage, "peak_private_bytes": process.PeakPagefileUsage,
            "working_set_bytes": process.WorkingSetSize, "peak_working_set_bytes": process.PeakWorkingSetSize}


def require_memory():
    status = memory_status()
    if status["available_commit_bytes"] < SETTINGS["minimum_free_commit_gib"] * 1024**3:
        raise RuntimeError("fresh pilot requires at least 8 GiB free system commit")
    return status


def pinned_modules(source):
    source = source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != SOURCE_COMMIT or dirty.strip():
        raise ValueError("source checkout must be clean at the declared commit")
    sys.path.insert(0, str(source))
    from monocular_osm import kartaview_vpr as kv, poi_alignment as core
    from monocular_osm.frame_extraction import extract_frames
    from monocular_osm.visual_odometry import default_intrinsics, VO_ALGO_VERSION
    for module in (kv, core):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("import escaped pinned source checkout")
    spec = importlib.util.spec_from_file_location("pinned_live_city_geometry", source / "scripts/pilot_joint_alignment.py")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    return kv, core, extract_frames, default_intrinsics, VO_ALGO_VERSION, geometry


def isolated_models(args, kv):
    """Copy only the pinned MegaLoc repository and its existing HF model cache."""
    destination = args.data / "models"
    repo_name = kv.HUB_MEGALOC.replace("/", "_").replace(":", "_")
    sources = [(args.cached_torch_hub / repo_name, destination / "torch/hub" / repo_name),
               (args.cached_hf_hub / "models--gberton--MegaLoc", destination / "hf/hub/models--gberton--MegaLoc")]
    records = []
    for source, target in sources:
        if not source.is_dir():
            raise FileNotFoundError("required cached MegaLoc assets unavailable; downloads disabled")
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(source)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = sha256(path)
            if output.exists():
                if sha256(output) != digest:
                    raise ValueError("isolated model asset changed")
            else:
                shutil.copyfile(path, output)
            records.append({"relative": str(output.relative_to(destination)),
                            "bytes": path.stat().st_size, "sha256": digest})
    os.environ["HF_HOME"] = str(destination / "hf")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(destination / "hf/hub")
    import torch
    torch.set_num_threads(SETTINGS["threads"])
    torch.set_num_interop_threads(1)
    torch.hub.set_dir(str(destination / "torch/hub"))
    def no_model_download(*args, **kwargs):
        raise RuntimeError("model download forbidden: use the isolated copied assets")
    torch.hub.download_url_to_file = no_model_download
    return records, {"torch": torch.__version__, "numpy": np.__version__}


def verified_inputs(args, extract_frames, default_intrinsics, vo_version):
    """Reuse motion only after decoding and signing the complete actual grid."""
    meta = read(args.inputs / "inputs_derived.json")
    video = Path(meta["inputs"]["video"]["path"])
    vo_path = args.inputs / "verified_video_vo.npz"
    if sha256(video) != meta["inputs"]["video"]["sha256"] or sha256(vo_path) != meta["vo"]["cache_sha256"]:
        raise ValueError("frozen video or VO bytes changed")
    with np.load(vo_path, allow_pickle=False) as data:
        vo, times = data["xz"], data["timestamps"]
        identity = json.loads(str(data["input_identity"]))
    frames = extract_frames(video, stride=SETTINGS["frame_stride"], max_frames=SETTINGS["max_frames"],
                            start_sec=meta["window"][0], end_sec=meta["window"][1],
                            memory_mb=SETTINGS["frame_memory_mb"])
    try:
        pixels = hashlib.sha256()
        for frame in frames.frames:
            frame = np.ascontiguousarray(frame)
            pixels.update(str((frame.shape, frame.dtype.str)).encode("ascii"))
            pixels.update(memoryview(frame).cast("B"))
        height, width = frames.frames[0].shape[:2]
        K = default_intrinsics(width, height)
        if (pixels.hexdigest() != identity["decoded_pixels_sha256"]
                or not np.array_equal(frames.timestamps, times)
                or not np.array_equal(K, identity["K"])
                or vo_version != identity["vo_algorithm_version"]
                or sha256(args.source_root / "monocular_osm/visual_odometry.py") != identity["vo_code_sha256"]):
            raise ValueError("VO reuse refused: decoded pixels/times/K/VO algorithm must match")
        return frames, vo, times, K, meta, identity
    except BaseException:
        frames.close()
        raise


def frozen_clues(args, meta):
    from pyproj import Transformer
    manifest = read(args.manifest)
    entry = next(item for item in manifest["artifacts"] if Path(item["path"]).resolve() == args.clusters.resolve())
    if manifest["ground_truth_used"] is not False or sha256(args.clusters) != entry["sha256"]:
        raise ValueError("frozen clue manifest changed")
    clues = read(args.clusters)
    ocr, inventory = Path(clues["clue_source"]), read(args.inventory)
    content = hashlib.sha256(json.dumps(inventory["pois"], sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()
    if (clues["selected_answer"] is not None or sha256(ocr) != clues["clue_sha256"]
            or content != clues["osm_snapshot_sha256"] or inventory["city"] != clues["city"]):
        raise ValueError("literal OCR/OSM source identity changed")
    project = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    inverse = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    plans = []
    for index, branch in enumerate(clues["candidate_clusters"]):
        xy = np.array([project.transform(p["lon"], p["lat"]) for p in branch["pois"]])
        lon, lat = inverse.transform(*xy.mean(axis=0))
        plans.append({"hypothesis_id": f"hypothesis_{index}", "center": [float(lat), float(lon)],
                      "radius_m": SETTINGS["radius_m"], "cap": SETTINGS["cap_per_branch"],
                      "poi_ids": branch["poi_ids"]})
    return clues, plans, ocr


def clean_strings(value):
    """API diagnostics may include request URLs; never persist token values."""
    serialized = json.dumps(value, allow_nan=False)
    token = os.environ.get("MLY_TOKEN")
    if token:
        serialized = serialized.replace(token, "[REDACTED]")
    return json.loads(serialized)


def merge_evidence(pool, evidence, timestamps):
    indices = np.asarray(evidence["query_indices"])
    times = np.asarray(evidence["query_times"])
    refs, scores = np.asarray(evidence["ref_latlon"]), np.asarray(evidence["similarities"])
    ids = evidence["ref_ids"].astype(str).tolist()
    provenance = evidence["provenance"]
    identity = {key: provenance[key] for key in ("backbone", "preprocessing", "query_mode")}
    if (not np.array_equal(times, np.asarray(timestamps)[indices]) or len(indices) != SETTINGS["n_query"]
            or refs.shape != (len(ids), 2) or scores.shape != (len(times), len(ids))
            or len(set(ids)) != len(ids) or len(ids) > SETTINGS["cap_per_branch"]
            or not np.isfinite(scores).all() or not np.isfinite(refs).all()):
        raise ValueError("fresh evidence query/reference identities or matrix are invalid")
    if pool and (identity != pool["identity"] or not np.array_equal(indices, pool["indices"])):
        raise ValueError("regional query grid/model/preprocessing differs")
    columns = pool.setdefault("columns", {})
    for index, identifier in enumerate(ids):
        if identifier in columns:
            old_xy, old_scores = columns[identifier]
            if not np.array_equal(refs[index], old_xy) or not np.allclose(scores[:, index], old_scores, rtol=0, atol=1e-6):
                raise ValueError("duplicate reference has conflicting coordinates/scores")
    for index, identifier in enumerate(ids):
        columns.setdefault(identifier, (refs[index].copy(), scores[:, index].copy()))
    pool.update(identity=identity, indices=indices.copy(), times=times.copy())


def acquire_regions(args, kv, frames, times, plans):
    pool, records, query_cache = {}, [], {}
    region_root = args.data / "references"
    region_root.mkdir(parents=True)
    (args.out / "regions").mkdir()
    for plan in plans:
        name = plan["hypothesis_id"]
        cache = region_root / name
        record = {**plan, "status": "failed", "cache": str(cache),
                  "query_cache_entries_before": len(query_cache)}
        try:
            record["memory_before"] = require_memory()
            started = time.perf_counter()
            evidence = kv.kartaview_vpr_evidence(
                frames, times, tuple(plan["center"]), radius_m=plan["radius_m"], cap=plan["cap"],
                cache_dir=str(cache), source=SETTINGS["source"], model_name=SETTINGS["model"],
                device="cpu", n_query=SETTINGS["n_query"], query_mode="warp",
                query_batch_size=SETTINGS["query_batch_size"], query_cache=query_cache)
            record["elapsed_s"] = time.perf_counter() - started
            if evidence is None:
                raise RuntimeError(kv.LAST_FAILURE or "reference evidence unavailable")
            if evidence["provenance"]["backbone"]["name"] != SETTINGS["model"]:
                raise ValueError("requested MegaLoc was unavailable; no backbone substitution in this pilot")
            # Save the full acquired matrix/inventory before trying to merge it.
            artifact = args.out / "regions" / f"{name}.npz"
            np.savez_compressed(artifact, **{key: evidence[key] for key in (
                "query_indices", "query_times", "similarities", "ref_latlon", "ref_ids")},
                metadata=np.array(json.dumps(clean_strings(evidence["provenance"]))))
            merge_evidence(pool, evidence, times)
            record.update(status="ok", references=len(evidence["ref_ids"]), artifact_sha256=sha256(artifact),
                          provenance=clean_strings(evidence["provenance"]))
        except Exception as exc:
            record["error"] = clean_strings(f"{type(exc).__name__}: {exc}")
        record["memory_after"] = memory_status()
        record["query_cache_entries_after"] = len(query_cache)
        records.append(record)
        write(args.out / "regions.json", clean_strings(records))
        print(f"region {name}: {record['status']}, refs={record.get('references', 0)}, "
              f"cache_entries={len(query_cache)}, resident_peak_GiB="
              f"{record['memory_after']['peak_working_set_bytes']/1024**3:.2f}", flush=True)
        if record["memory_after"]["peak_working_set_bytes"] > SETTINGS["ram_target_bytes"]:
            raise RuntimeError("pilot exceeded its 3 GB resident-RAM target; partial artifacts preserved")
    return pool, records


def snapshot_inputs(args, meta, ocr):
    sources = {"video": Path(meta["inputs"]["video"]["path"]),
               "graph": Path(meta["inputs"]["city_graph_B"]["path"]),
               "vo": args.inputs / "verified_video_vo.npz", "vo_metadata": args.inputs / "inputs_derived.json",
               "clue_manifest": args.manifest, "clusters": args.clusters,
               "ocr": ocr, "inventory": args.inventory}
    if sha256(sources["graph"]) != meta["inputs"]["city_graph_B"]["sha256"]:
        raise ValueError("full-city graph B changed")
    target = args.data / "input_snapshots"
    target.mkdir(parents=True)
    result = {}
    for role, path in sources.items():
        output = target / f"{role}{path.suffix}"
        shutil.copyfile(path, output)
        result[role] = {"path": str(output.resolve()), "sha256": sha256(output)}
    write(args.out / "inputs.json", result)
    return result


def infer(args):
    preflight = require_memory()
    if args.data.resolve().drive.upper() != "E:":
        raise ValueError("pilot data/model/reference caches must use the isolated E drive")
    if args.out.exists():
        if (args.out / "predictions_frozen.json").exists():
            load_frozen(args.out)
            print("validated existing immutable pilot; no files changed")
            return
        raise ValueError("output already exists; preserve partial run and choose new output/data")
    if (args.data / "references").exists():
        raise ValueError("fresh reference pilot refuses any existing regional cache")
    kv, core, extract, intrinsics, vo_version, geometry = pinned_modules(args.source_root)
    args.out.mkdir(parents=True)
    args.data.mkdir(parents=True, exist_ok=True)
    temporary = args.data / "frame_tmp"
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)
    model_assets, versions = isolated_models(args, kv)
    import cv2
    import torch
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    torch.manual_seed(0)
    frames, vo, times, K, meta, identity = verified_inputs(args, extract, intrinsics, vo_version)
    try:
        clues, plans, ocr = frozen_clues(args, meta)
        sources = snapshot_inputs(args, meta, ocr)
        indices = np.linspace(0, len(times) - 1, SETTINGS["n_query"]).astype(int)
        config = core.POIAlignmentConfig()
        registration = {
            "status": "post-hoc dev; fixed live acquisition settings before new scoring",
            "ground_truth_used_during_inference": False, "defaults_changed": False,
            "source_commit": SOURCE_COMMIT, "runner_sha256": sha256(__file__),
            "source_files": {name: sha256(args.source_root / name) for name in (
                "monocular_osm/kartaview_vpr.py", "monocular_osm/poi_alignment.py",
                "monocular_osm/frame_extraction.py", "monocular_osm/visual_odometry.py",
                "scripts/pilot_joint_alignment.py")},
            "settings": SETTINGS, "alignment_config": config.to_dict(), "versions": versions,
            "data_root": str(args.data.resolve()), "source_root": str(args.source_root.resolve()),
            "sources": sources, "model_assets": model_assets, "region_plans": plans,
            "preflight_memory": preflight, "intrinsics": K.tolist(),
            "frame_count": len(times), "query_indices": indices.tolist(), "query_times": times[indices].tolist(),
            "decoded_pixels_sha256": identity["decoded_pixels_sha256"],
            "vo_reuse_rule": "video bytes, every decoded pixel, full timestamps, K and VO code verified",
            "window": meta["window"], "crs": meta["crs"],
            "difference_from_saved_pilot": "fresh normal dense frame grid/80queries and live Mapillary pool; no exact-score expectation"}
        write(args.out / "experiment.json", registration)
        write(args.out / "plans.json", plans)
        source_dir = args.data / "source_snapshots"
        source_dir.mkdir()
        shutil.copyfile(__file__, source_dir / "pilot_live_poi_alignment.py")
        for name in registration["source_files"]:
            target = source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source_root / name, target)
        print(f"registered ALL{len(plans)} branches, {len(times)} video frames, {len(indices)} fresh queries", flush=True)
        pool, records = acquire_regions(args, kv, frames.frames, times, plans)
        predict(args, core, geometry, registration, clues, vo, times, K, config, pool, records)
    finally:
        frames.close()


def predict(args, core, geometry, registration, clues, vo, times, K, config, pool, records):
    failed = [r["hypothesis_id"] for r in records if r["status"] != "ok"]
    if pool:
        from pyproj import Transformer
        ids = list(pool["columns"])
        refs = np.array([pool["columns"][identifier][0] for identifier in ids])
        similarities = np.ascontiguousarray(np.column_stack([
            pool["columns"][identifier][1] for identifier in ids]), dtype=np.float32)
        np.savez_compressed(args.out / "union_evidence.npz", ref_ids=np.array(ids), ref_latlon=refs,
                            similarities=similarities, query_indices=pool["indices"], query_times=pool["times"],
                            metadata=np.array(json.dumps(pool["identity"])))
        project = Transformer.from_crs("EPSG:4326", registration["crs"], always_xy=True)
        xy = np.column_stack(project.transform(refs[:, 1], refs[:, 0]))
        roads, crs, stats = geometry.city_roads(registration["sources"]["graph"]["path"])
        low, shape, distance, starts, uncapped = geometry.road_and_start_grids(
            roads, xy, float(times[-1] - times[0]) * config.average_speed_range_mps[1])
        grid = {"origin_xy": low.tolist(), "shape": list(shape), "road_shape": list(distance.shape),
                "start_count": len(starts), "uncapped_start_count": uncapped,
                "start_samples_sha256": array_digest(starts), "graph": stats,
                "reference_count": len(ids), "query_count": len(pool["times"])}
        write(args.out / "grid.json", grid)
        road = core.RasterField(distance, tuple(low), config.road_cell_m, 200.,
                                provenance=registration["sources"]["graph"])
    else:
        ids = []
    for variant in SETTINGS["appearance_variants"]:
        output = {"status": "inactive" if failed else "ok", "failed_regions": failed,
                  "reference_count": len(ids), "ground_truth_used_during_inference": False}
        if pool:
            field = core.build_appearance_field(xy, similarities, pool["times"], origin_xy=low, shape=shape,
                reference_mean=variant == "reference_mean", config=config,
                provenance={"union_evidence_sha256": sha256(args.out / "union_evidence.npz"), "variant": variant})
            output["alignment"] = core.align_candidate_clusters(
                vo, times, K, clues, map_crs=crs, appearance_field=field, road_distance_field=road,
                start_samples=starts, query_times=pool["times"], config=config,
                provenance={"experiment_sha256": sha256(args.out / "experiment.json"), "variant": variant,
                            "grid": grid, "all_regions_complete": not failed})
        else:
            output["reason"] = "no_compatible_reference_evidence"
        write(args.out / f"{variant}.json", output)
        print(f"saved {variant}: status={output['status']}, union={len(ids)}, branch="
              f"{output.get('alignment', {}).get('selected', {}).get('hypothesis_id')}", flush=True)
    freeze_outputs(args)


def freeze_outputs(args):
    caches = [{"relative": str(path.relative_to(args.data)), "bytes": path.stat().st_size, "sha256": sha256(path)}
              for path in sorted((args.data / "references").rglob("*")) if path.is_file()]
    write(args.out / "reference_cache_snapshot.json", caches)
    write(args.out / "memory.json", memory_status())
    freeze = {"experiment_sha256": sha256(args.out / "experiment.json"),
              "plans_sha256": sha256(args.out / "plans.json"), "ground_truth_opened": False,
              "cache_snapshot_sha256": sha256(args.out / "reference_cache_snapshot.json"),
              "predictions": {v: sha256(args.out / f"{v}.json") for v in SETTINGS["appearance_variants"]},
              "artifacts": {p.name: sha256(p) for p in args.out.iterdir()
                            if p.is_file() and p.name not in {"raw.json", "reference_mean.json"}}}
    write(args.out / "predictions_frozen.json", freeze)


def load_frozen(out):
    freeze = read(out / "predictions_frozen.json")
    registration = read(out / "experiment.json")
    if freeze["experiment_sha256"] != sha256(out / "experiment.json"):
        raise ValueError("frozen experiment changed")
    if set(freeze["predictions"]) != set(registration["settings"]["appearance_variants"]):
        raise ValueError("both registered predictions must freeze before scoring")
    for name, digest in {**freeze["artifacts"], **{f"{k}.json": v for k, v in freeze["predictions"].items()}}.items():
        if sha256(out / name) != digest:
            raise ValueError(f"frozen artifact changed: {name}")
    for record in read(out / "regions.json"):
        if "artifact_sha256" in record:
            if sha256(out / "regions" / f"{record['hypothesis_id']}.npz") != record["artifact_sha256"]:
                raise ValueError("frozen regional evidence changed")
    return registration, {v: read(out / f"{v}.json") for v in freeze["predictions"]}


def score(args):
    registration, predictions = load_frozen(args.out)
    sys.path.insert(0, registration["source_root"])
    from monocular_osm.metrics import score_route
    from pyproj import Transformer
    # Both variants and every saved region are validated before labels open.
    truth = read(args.gt)["waypoints"]
    points = [p for p in truth if registration["window"][0] <= p["t_sec"] <= registration["window"][1]]
    gt_ll = np.array([[p["lat"], p["lon"]] for p in points])
    gt_t = np.array([p["t_sec"] for p in points])
    project = Transformer.from_crs("EPSG:4326", registration["crs"], always_xy=True)
    scores = {}
    for variant, prediction in predictions.items():
        item = {"status": prediction["status"], "failed_regions": prediction["failed_regions"],
                "reference_count": prediction["reference_count"]}
        if "alignment" in prediction:
            selected = prediction["alignment"]["selected"]
            route = np.asarray(selected["route_latlon"])
            query = np.asarray(selected["route_timestamps"])
            expected = np.c_[np.interp(query, gt_t, gt_ll[:, 0]), np.interp(query, gt_t, gt_ll[:, 1])]
            actual_xy = np.column_stack(project.transform(route[:, 1], route[:, 0]))
            expected_xy = np.column_stack(project.transform(expected[:, 1], expected[:, 0]))
            item.update(score_route(route, gt_ll).to_dict(),
                        mean_time_error_m=float(np.linalg.norm(actual_xy - expected_xy, axis=1).mean()),
                        selected_hypothesis=selected["hypothesis_id"],
                        transform=[*selected["start_xy"], selected["yaw_deg"], selected["length_m"]])
        scores[variant] = item
    write(args.out / "scores.json", {"results": scores, "ground_truth_sha256": sha256(args.gt),
                                     "prediction_freeze_sha256": sha256(args.out / "predictions_frozen.json")})
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inf = commands.add_parser("infer")
    inf.add_argument("--source-root", type=Path, default=Path("E:/monocular-osm-runs/poi-live-8aa9f05"))
    inf.add_argument("--data", type=Path, default=Path("E:/monocular-osm-live-poi-pilot/run2"))
    inf.add_argument("--out", type=Path, default=WORKSPACE / "output/live_poi_alignment_pilot_v2")
    inf.add_argument("--inputs", type=Path, default=WORKSPACE / "output/joint_alignment_pilot_v3")
    inf.add_argument("--manifest", type=Path, default=WORKSPACE / "output/poi_clue_probe/frozen_candidates.json")
    inf.add_argument("--clusters", type=Path, default=WORKSPACE / "output/poi_clue_probe/kitti_0009.json")
    inf.add_argument("--inventory", type=Path, default=WORKSPACE / "output/poi_clue_probe/osm/karlsruhe-germany.json")
    inf.add_argument("--cached-torch-hub", type=Path, default=Path.home() / ".cache/torch/hub")
    inf.add_argument("--cached-hf-hub", type=Path, default=Path.home() / ".cache/huggingface/hub")
    scr = commands.add_parser("score")
    scr.add_argument("--out", type=Path, default=WORKSPACE / "output/live_poi_alignment_pilot_v2")
    scr.add_argument("--gt", type=Path, default=WORKSPACE / "ground_truth/kitti_drive_0009.json")
    args = parser.parse_args()
    (infer if args.command == "infer" else score)(args)
