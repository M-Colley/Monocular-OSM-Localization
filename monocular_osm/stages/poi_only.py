"""Explicit experimental POI component; acquisition and selection stay auditable."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, quote_plus

import cv2
import numpy as np

from ..poi_localization import POILocalizationConfig, _array_sha, _json_sha, localize_with_poi
from ..provenance import config_snapshot, file_sha256, snapshot_sources
from ..visual_odometry import VO_ALGO_VERSION, default_intrinsics
from .context import _record_channel
from .report import _write_result_json
from .poi_runtime import pinned_reader, reader_spec, torch_cpu_threads, validate_reader_record

OCR_SAMPLES = 24
QUERY_SAMPLES = 80
ROAD_SPACING_M = 12.5
MAX_ROAD_SAMPLES = 2_000_000


def _safe(value):
    """Never persist credential-bearing API diagnostics."""
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        token = os.environ.get("MLY_TOKEN")
        if token:
            for secret in {token, quote(token, safe=""), quote_plus(token, safe="")}:
                value = value.replace(secret, "[REDACTED]")
        return value
    return value


def _whole_city_map(ctx):
    from ..graph_filter import fetch_drive_graph
    from ..osm_data import _build_polyline_view
    import osmnx as ox

    explicit = ctx.cfg.poi_map_path
    if explicit is not None:
        path = Path(explicit).resolve(strict=True)
        graph, source = ox.load_graphml(path), "explicit_cache"
    else:
        slug = ctx.cfg.city.replace(",", "").replace(" ", "_")
        path = (ctx.cfg.data_dir / f"{slug}_poi_city.graphml").resolve()
        before = file_sha256(path) if path.is_file() else None
        graph = fetch_drive_graph(place=ctx.cfg.city, cache_path=path)
        source = "city_cache" if before is not None and before == file_sha256(path) else "named_city_fetch"
    if not ox.projection.is_projected(graph.graph["crs"]):
        graph = ox.project_graph(graph)
    road = _build_polyline_view(graph)
    metadata = {"scope": "whole_city", "city": ctx.cfg.city, "query": ctx.cfg.city,
                "cache_path": str(path), "configured_path": str(explicit) if explicit is not None else None,
                "sha256": file_sha256(path), "source": source, "around": None, "bbox": None}
    return road, metadata


def _road_samples(road):
    """Fixed full-city sampling, bounded before allocating a point cloud."""
    pieces, count = [], 0
    for polyline in road.polylines:
        xy = np.asarray(polyline, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError("city road geometry is not finite metric XY")
        steps = np.maximum(1, np.ceil(np.linalg.norm(np.diff(xy, axis=0), axis=1) / ROAD_SPACING_M).astype(int))
        count += int(steps.sum()) + len(steps)
        if count > MAX_ROAD_SAMPLES:
            raise ValueError("whole-city road-sample budget exceeded; no spatial crop was applied")
        pieces.append((xy, steps))
    if not pieces:
        raise ValueError("whole-city road map has no geometry")
    samples = np.empty((count, 2), float)
    cursor = 0
    for xy, steps in pieces:
        for start, end, n in zip(xy[:-1], xy[1:], steps):
            samples[cursor:cursor + n + 1] = np.linspace(start, end, int(n) + 1)
            cursor += int(n) + 1
    return samples


def _inventory(ctx):
    from ..poi_clues import fetch_city_retail

    directory = (Path(ctx.cfg.poi_inventory_dir) if ctx.cfg.poi_inventory_dir is not None
                 else ctx.cfg.data_dir.parent / "poi_inventory").resolve()
    inventory = fetch_city_retail(ctx.cfg.city, directory)
    if (inventory.get("city") != ctx.cfg.city or inventory.get("scope") != "named_city_bbox"
            or inventory.get("snapshot_sha256") != _json_sha(inventory.get("pois"))
            or not isinstance(inventory.get("response_sha256"), str)
            or re.fullmatch(r"[a-fA-F0-9]{64}", inventory["response_sha256"]) is None):
        raise ValueError("retail inventory lacks a valid fingerprinted named-city source")
    path = ctx.cfg.output_dir / "poi_inventory.json"
    _write_result_json(path, inventory)
    metadata = {key: inventory.get(key) for key in
                ("city", "scope", "snapshot_sha256", "response_sha256", "query", "bbox_wsen")}
    metadata.update(path=str(path.resolve()), file_sha256=file_sha256(path), cache_dir=str(directory))
    return inventory, metadata


def _ocr(ctx, K, video_sha, shape):
    from .. import scene_text
    from ..provenance import library_versions

    first, last = map(float, (ctx.frames.timestamps[0], ctx.frames.timestamps[-1]))
    # End-exclusive native sampling stays inside the actual VO horizon. This
    # bound is derived from decoded video, never a clue-dependent time window.
    runtime = library_versions(("rapidocr", "onnxruntime", "opencv-python", "numpy")
                               if ctx.cfg.ocr_engine == "rapidocr" else ("easyocr", "torch", "opencv-python", "numpy"))
    runtime.update(cv2=cv2.__version__, numpy=np.__version__)
    spec = reader_spec(ctx.cfg.ocr_engine)
    identity = {"video": video_sha, "window": [first, last], "shape": list(shape), "runtime": runtime,
                "samples": OCR_SAMPLES, "min_len": 2, "engine": ctx.cfg.ocr_engine,
                "scene_text_source": file_sha256(Path(scene_text.__file__)), "reader_spec_sha256": spec["spec_sha256"]}
    cache = ctx.cfg.data_dir / f"poi_scene_text_{_json_sha(identity)[:16]}.json"
    sidecar = cache.with_suffix(".reader.json")
    previous = None
    if cache.is_file():
        try:
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if (saved["cache_sha256"] == file_sha256(cache)
                    and saved["reader"]["spec_sha256"] == spec["spec_sha256"]):
                validate_reader_record(saved["reader"])
                previous = saved["reader"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if previous is None:
            # This is our content-addressed POI cache, never a legacy OCR cache.
            cache.unlink()
    def native_frames(path, start, end, interval):
        for timestamp, image in scene_text._sample_frames(path, start, end, interval, sample_count=OCR_SAMPLES):
            if image.shape[:2] != tuple(shape) or not first <= timestamp <= last:
                raise ValueError("native OCR pixels/times differ from the supplied VO camera grid")
            yield timestamp, image
    with pinned_reader(spec) as (reader, actual):
        raw = scene_text.extract_scene_text(ctx.video_path, sample_count=OCR_SAMPLES, min_len=2,
                min_confidence=.3, start_sec=first, end_sec=last, use_gpu=False, super_res=False,
                engine=ctx.cfg.ocr_engine, cache_path=cache, frame_reader=native_frames, ocr_reader=reader)
    if not actual.get("read_calls"):
        if previous is None:
            raise ValueError("OCR cache/runtime has no verified CPU provider identity")
        actual = {**previous, "cache_hit": True}
    validate_reader_record(actual)
    _write_result_json(sidecar, {"cache_sha256": file_sha256(cache), "reader": actual})
    detections = [asdict(item) if not isinstance(item, dict) else dict(item) for item in raw]
    detections = [item for item in detections if not scene_text.is_position_stamp(item["text"])]
    if any(not first <= float(item["t_sec"]) <= last for item in detections):
        raise ValueError("cached OCR timestamps fall outside the actual VO horizon")
    metadata = {"source": "same_video_native_frames", "video_sha256": video_sha,
                "engine": ctx.cfg.ocr_engine, "cache_path": str(cache.resolve()), "cache_sha256": file_sha256(cache),
                "sample_count": OCR_SAMPLES, "min_len": 2, "min_confidence": .3,
                "super_res": False, "use_gpu": False, "native_bbox_grid": True,
                "frame_shape_hw": list(shape), "window": [first, last],
                "position_stamp_filter": "scene_text.is_position_stamp", "position_stamps_rejected": True,
                "detections_sha256": _json_sha(detections), "K_sha256": _array_sha(K), "runtime": runtime,
                "reader": actual, "reader_sidecar_path": str(sidecar.resolve()),
                "reader_sidecar_sha256": file_sha256(sidecar), "model_weight_identity": "explicit_cached_sha256"}
    _write_result_json(ctx.cfg.output_dir / "poi_ocr.json", {"provenance": metadata, "detections": detections})
    return detections, metadata


def _retrieval(ctx, inputs, *, audit_query=False):
    """Own one stage-local descriptor cache; preserve every callback's evidence."""
    from .. import kartaview_vpr as kv

    source = ctx.cfg.vpr_source
    api_sha = file_sha256(Path(kv.__file__))
    def retrieve(frames, timestamps, center, *, radius_m, cap, n_query, query_cache):
        identity = {"center": list(center), "radius_m": radius_m, "cap": cap,
                    "inventory": inputs["inventory"]["snapshot_sha256"], "source": source,
                    "requested_model": "megaloc", "query_mode": ctx.cfg.vpr_query_mode,
                    "cpu_threads": 4, "evidence_api": api_sha}
        tag = f"poi_{_json_sha(identity)[:16]}"
        external = ctx.cfg.vpr_region_cache_dir
        directory = (Path(external) / ctx.cfg.data_dir.name / f"{source}_region_{tag}" if external is not None
                     else ctx.cfg.data_dir / f"{source}_regions" / tag)
        audit = {} if audit_query else None
        audited = ({"audit": audit, "query_video_sha256": inputs["video"]["sha256"]}
                   if audit_query else {})
        with torch_cpu_threads(4) as cpu_runtime:
            raw = kv.kartaview_vpr_evidence(frames, timestamps, center, radius_m=radius_m,
                cache_dir=str(directory), n_query=n_query, source=source, cap=cap, device="cpu",
                token=os.environ.get("MLY_TOKEN"), query_mode=ctx.cfg.vpr_query_mode,
                query_batch_size=ctx.cfg.vpr_query_batch_size, query_cache=query_cache, **audited)
        if raw is None:
            raise RuntimeError(_safe(kv.LAST_FAILURE or "no neighborhood evidence"))
        raw = dict(raw)
        provenance = deepcopy(raw.get("provenance") or {})
        provenance["cpu_runtime"] = cpu_runtime
        scores = np.ascontiguousarray(raw["similarities"], dtype=np.float32)
        coordinates = np.asarray(raw["ref_latlon"], float)
        identifiers = [str(value) for value in raw["ref_ids"]]
        provenance["evidence_identity"] = {
            "query_indices": np.asarray(raw["query_indices"]).tolist(),
            "query_times": np.asarray(raw["query_times"], float).tolist(), "ref_ids": identifiers,
            "similarities_sha256": _array_sha(scores), "ref_latlon_sha256": _array_sha(coordinates)}
        number = len(inputs["evidence_artifacts"])
        path = ctx.cfg.output_dir / "poi_evidence" / f"{number:04d}_{tag}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, query_indices=raw["query_indices"], query_times=raw["query_times"],
                            similarities=scores, ref_latlon=coordinates, ref_ids=np.asarray(identifiers))
        record = {"path": str(path.relative_to(ctx.cfg.output_dir)), "sha256": file_sha256(path)}
        inputs["evidence_artifacts"].append(record)
        provenance["matrix_artifact"] = record
        raw["provenance"] = _safe(provenance)
        return raw
    return retrieve


def _audit(result):
    audit = {name: deepcopy(result.get(name)) for name in
             ("status", "failed_regions", "region_plans", "provenance", "clues", "regions", "reference_union", "grid")}
    aligned = result.get("alignment") or {}
    audit["alignment"] = {"hypothesis_ids": [str(item.get("id", item.get("hypothesis_id")))
                           for item in aligned.get("hypotheses", [])],
        "selected_hypothesis_id": (aligned.get("selected") or {}).get("hypothesis_id"),
        "search": aligned.get("search"), "provenance": aligned.get("provenance")}
    return _safe(audit)


def _freeze(ctx):
    paths = [ctx.cfg.output_dir / name for name in
             ("poi_inputs.json", "poi_prediction.json", "poi_ocr.json", "poi_inventory.json", "poi_vo.npz",
              "poi_vo_cache.npz", "result.json", "answer.png")]
    inputs = ctx.result.get("poi_inputs") or {}
    paths += [ctx.cfg.output_dir / record["path"] for record in inputs.get("evidence_artifacts", [])]
    _write_result_json(ctx.cfg.output_dir / "frozen_prediction.json", {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scoring_has_not_run": True, "experimental": True, "prediction_available": "answer" in ctx.result,
        "inputs": _safe(inputs), "artifacts": {str(path.relative_to(ctx.cfg.output_dir)): file_sha256(path)
                                                for path in paths if path.is_file()}})


def _write_inputs(ctx):
    _write_result_json(ctx.cfg.output_dir / "poi_inputs.json",
                       {**_safe(ctx.result["poi_inputs"]), "config": config_snapshot(ctx.cfg)})


def _begin_inputs(ctx, K, shape):
    root = Path(__file__).resolve().parents[2]
    sources = [root / "main.py", *[root / "monocular_osm" / relative for relative in (
        "cli.py", "pipeline.py", "visual_odometry.py", "frame_extraction.py", "osm_data.py",
        "graph_filter.py", "provenance.py", "position.py", "baselines.py", "scene_text.py",
        "kartaview_vpr.py", "poi_alignment.py", "poi_clues.py", "poi_localization.py", "geo.py",
        "net.py", "osm_gazetteer.py", "stages/context.py", "stages/acquire.py",
        "stages/trajectory.py", "stages/poi_only.py", "stages/poi_runtime.py", "stages/report.py")]]
    times = np.asarray(ctx.frames.timestamps, float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("POI component requires finite increasing video presentation times")
    inputs = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experimental": True,
        "input_policy": {**{name: False for name in ("calibration", "raw_images", "sensor_timestamps",
            "gps_or_oxts", "ground_truth", "manual_region_coordinates", "external_position_prior",
            "external_trajectory", "external_ocr_video")},
            **{name: True for name in ("video_ocr", "public_poi_inventory", "public_reference_positions",
                                      "position_stamps_rejected", "retrieval")}},
        "video": {"path": str(Path(ctx.video_path).resolve()), "sha256": file_sha256(ctx.video_path)},
        "timing": {"source": "video_pts_or_encoded_fps", "fps": float(ctx.frames.fps),
                   "first_s": float(times[0]), "last_s": float(times[-1]), "timestamps_sha256": _array_sha(times)},
        "extraction": {"stride": ctx.cfg.frame_stride, "n_frames": len(times), "max_frames": None,
                       "frame_max_width": None, "native_pixels": True,
                       "registered_window": [ctx.cfg.vo_start_sec, ctx.cfg.vo_end_sec]},
        "camera": {"intrinsics": "generic_70_degree_hfov", "K": K.tolist(),
                   "shape_hw": list(shape), "vo_rng_seed": 0},
        "source_sha256": snapshot_sources(sources, ctx.cfg.output_dir), "evidence_artifacts": [],
        "retrieval": {"requested_model": "megaloc", "device": "cpu", "n_query": QUERY_SAMPLES,
                      "cpu_threads": 4,
                      "query_mode": ctx.cfg.vpr_query_mode, "batch_size": ctx.cfg.vpr_query_batch_size,
                      "appearance": "raw", "poi_loss": "distance_and_bearing"}}
    ctx.result["poi_inputs"] = inputs
    _write_inputs(ctx)
    return inputs


def _vo_identity(ctx, K):
    import shutil
    import monocular_osm.visual_odometry as vo

    path = Path(ctx.vo_cache).resolve(strict=True)
    with np.load(path, allow_pickle=False) as cache:
        signature = str(cache["cache_signature"].item())
        identity = json.loads(signature)
    if not isinstance(identity, dict):
        raise ValueError("ordinary VO cache has no signed video identity")
    shutil.copyfile(path, ctx.cfg.output_dir / "poi_vo_cache.npz")
    times = np.asarray(ctx.frames.timestamps, float)
    np.savez_compressed(ctx.cfg.output_dir / "poi_vo.npz", xz=ctx.traj.xz, timestamps=times,
                        K=K, valid=ctx.traj.valid, input_signature=signature)
    return {"source": "same_video_generic_vo", "algorithm_version": VO_ALGO_VERSION,
            "cache_path": str(path), "cache_sha256": file_sha256(path),
            "input_signature_sha256": hashlib.sha256(signature.encode()).hexdigest(),
            "trajectory_sha256": _array_sha(ctx.traj.xz), "timestamps_sha256": _array_sha(times),
            "K_sha256": _array_sha(K), "code_sha256": file_sha256(Path(vo.__file__)), "input_identity": identity}


def _complete_result(result):
    if result.get("status") != "ok" or result.get("failed_regions") != []:
        return False
    branches = (result.get("clues") or {}).get("candidate_clusters") or []
    identifiers = [str(branch.get("id", f"hypothesis_{i}")) for i, branch in enumerate(branches)]
    plans, regions = result.get("region_plans") or [], result.get("regions") or []
    aligned = result.get("alignment") or {}
    aligned_ids = [str(item.get("id", item.get("hypothesis_id"))) for item in aligned.get("hypotheses", [])]
    if (not identifiers or identifiers != [item["hypothesis_id"] for item in plans]
            or identifiers != [item["hypothesis_id"] for item in regions] or identifiers != aligned_ids
            or any(item.get("status") != "ok" for item in regions)
            or (aligned.get("selected") or {}).get("hypothesis_id") not in identifiers):
        raise ValueError("POI helper did not preserve every literal branch through complete retrieval/alignment")
    return True


def run_poi_only(ctx, *, extract, run_vo, finalize):
    """Experimental component; unanswered cases never fall back to another mode."""
    cfg = ctx.cfg
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    for name in ("frozen_prediction.json", "poi_prediction.json", "poi_inputs.json", "poi_ocr.json",
                 "poi_inventory.json", "poi_vo.npz", "poi_vo_cache.npz"):
        (cfg.output_dir / name).unlink(missing_ok=True)
    ctx.result["execution"] = {"stage": "poi", "geometry": "skipped", "retrieval": "poi_neighborhoods"}
    for channel in ("vpr", "plate", "vlm", "sun", "ocr_anchor", "loop_closure", "coarse_from_frames",
                    "orienternet", "geometry", "graph_filter"):
        _record_channel(ctx.channel_status, channel, "disabled", reason="poi_only_execution")
    try:
        extract(ctx)
        shape = ctx.frames.frames[0].shape[:2]
        K = default_intrinsics(shape[1], shape[0])
        inputs = _begin_inputs(ctx, K, shape)
        run_vo(ctx)
        ctx.result.update(n_frames=len(ctx.frames.frames), n_valid_poses=ctx.n_valid)
        inputs["vo"] = _vo_identity(ctx, K)
        _write_inputs(ctx)
        ctx.road, inputs["map"] = _whole_city_map(ctx)
        ctx.cache_path = Path(inputs["map"]["cache_path"])
        roads = _road_samples(ctx.road)
        inputs["road_samples"] = {"spacing_m": ROAD_SPACING_M, "count": len(roads), "sha256": _array_sha(roads),
                                  "sampling_policy": "per-segment linspace including both endpoints"}
        inventory, inputs["inventory"] = _inventory(ctx)
        detections, inputs["ocr"] = _ocr(ctx, K, inputs["video"]["sha256"], shape)
        _write_inputs(ctx)
        query_cache = {}
        try:
            prediction = localize_with_poi(ctx.frames.frames, ctx.traj.xz, np.asarray(ctx.frames.timestamps, float),
                K, cfg.city, roads, ctx.road.crs, detections, inventory, _retrieval(ctx, inputs),
                config=POILocalizationConfig(radius_m=cfg.poi_radius_m, cap_per_branch=cfg.poi_ref_cap,
                                             n_query=QUERY_SAMPLES),
                query_cache=query_cache, map_provenance=inputs["map"])
        finally:
            query_cache.clear()
        prediction = _safe(prediction)
        ctx.result["poi_localization"] = prediction
        ctx.result["poi_localization_audit"] = _audit(prediction)
        _write_result_json(cfg.output_dir / "poi_prediction.json", prediction)
        _write_inputs(ctx)
        complete = _complete_result(prediction)
        if complete:
            position = deepcopy(prediction["diagnostic_position"])
            position["route_t_sec"] = position.pop("route_timestamps")
            position.update(source="experimental_poi_alignment", experimental=True, confidence_calibrated=False)
            route_times = np.asarray(position["route_t_sec"], float)
            if (route_times.ndim != 1 or len(route_times) != len(position["route_latlon"])
                    or len(route_times) < 2 or not np.isfinite(route_times).all() or np.any(np.diff(route_times) <= 0)
                    or route_times[0] < ctx.frames.timestamps[0] or route_times[-1] > ctx.frames.timestamps[-1]):
                raise ValueError("POI route times do not identify the supplied video horizon")
            ctx.result["poi_position"] = position
        _record_channel(ctx.channel_status, "poi", "ok" if complete else "inactive", reason=prediction.get("reason"))
        backbones = [item.get("provenance", {}).get("backbone") for item in prediction.get("regions", [])
                     if item.get("status") == "ok"]
        _record_channel(ctx.channel_status, "poi_retrieval", "ok" if complete else "inactive",
                        reason="poi_neighborhood_evidence", source=cfg.vpr_source,
                        backbones=backbones, failed_regions=prediction.get("failed_regions", []))
        ctx.result["execution"].update(stage="complete", answer_available=complete)
        finalize(ctx)
        _freeze(ctx)
        return ctx.result
    except Exception as error:
        ctx.result.pop("poi_position", None)
        ctx.result["execution"].update(stage="failed", answer_available=False,
                                       error=f"{type(error).__name__}: {_safe(str(error))}")
        _record_channel(ctx.channel_status, "poi", "failed", error=RuntimeError(_safe(str(error))))
        finalize(ctx)
        _freeze(ctx)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
