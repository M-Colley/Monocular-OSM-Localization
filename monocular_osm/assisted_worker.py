"""Isolated CPU child execution for the experimental assisted profile."""
from __future__ import annotations
import argparse
import ctypes
from copy import deepcopy
from dataclasses import fields, replace
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import numpy as np
from .provenance import config_snapshot, file_sha256, snapshot_sources
from .stages.poi_only import _safe
from .stages.report import _write_result_json

KINDS = {"baseline", "cues", "poi_joint", "poi_image_track", "poi_image_union", "metric_pf", "expanded"}
PATH_FIELDS = {"data_dir", "output_dir", "video_path", "poi_map_path", "poi_inventory_dir",
               "graph_filter_map_path", "ground_truth_waypoints", "openvo_trajectory_path",
               "vggt_long_trajectory_path", "ocr_video_path", "vpr_region_cache_dir"}


def _array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _from_snapshot(snapshot):
    from .pipeline import PipelineConfig
    known = {f.name for f in fields(PipelineConfig)}
    values = {name: value for name, value in snapshot.items() if name in known}
    for name in PATH_FIELDS:
        if values.get(name) is not None:
            values[name] = Path(values[name])
    return PipelineConfig(**values)


def child_config(cfg, kind, video_path):
    """Operational isolation changes no ordinary baseline inference setting."""
    if kind not in KINDS:
        raise ValueError(f"unknown assisted child {kind}")
    output = cfg.output_dir / "children" / kind
    data = cfg.data_dir if kind == "baseline" else cfg.data_dir / "assisted" / (
        "native" if kind in {"cues", "poi_joint", "poi_image_track", "poi_image_union"} else kind) / cfg.data_dir.name
    changes = dict(answer="track", retrieval_only=False, output_dir=output, data_dir=data,
                   video_path=Path(video_path), skip_download=False,
                   ground_truth_waypoints=None, ground_truth_streets=())
    if kind in {"baseline", "expanded"}:
        changes["retrieval_only"] = True
        if kind == "expanded":
            changes.update(vpr_multi_region=True, vpr_city_cover=True)
    else:
        changes.update(answer="poi", max_frames=None, frame_max_width=None, frame_memory_mb=128.,
            vo_workers=2, osm_around=None, osm_around_source=None, estimated_length_m=None,
            graph_filter=False, graph_filter_vpr=False, graph_filter_map_path=None,
            use_ipm_scale=False, use_ground_flow_scale=False)
        if kind in {"poi_image_track", "poi_image_union"}:
            changes.update(vpr_query_mode="crops", poi_radius_m=3000., poi_ref_cap=1500)
        if kind == "metric_pf":
            changes.update(answer="filter", frame_stride=None, graph_filter=True,
                graph_filter_particles=600_000, graph_filter_seed=0, graph_filter_target_hz=10.,
                graph_filter_path_budget_mb=256., graph_filter_speed="plane", camera_height_m=1.5)
    return replace(cfg, **changes)


def _commit_headroom():
    if os.name != "nt":
        return None
    class Performance(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            (name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
    record = Performance()
    record.cb = ctypes.sizeof(record)
    if not ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo(ctypes.byref(record), record.cb):
        raise OSError("cannot verify Windows commit headroom")
    return (record.CommitLimit - record.CommitTotal) * record.PageSize


def _stop_owned_child(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        import signal
        os.killpg(child.pid, signal.SIGTERM)
    child.wait()


def run_child(kind, parent):
    headroom = _commit_headroom()
    if headroom is not None and headroom < 8 * 1024**3:
        raise RuntimeError("assisted child unavailable: less than 8 GiB actual Windows commit headroom")
    cfg = child_config(parent.cfg, kind, parent.video_path)
    cfg.output_dir = parent.assisted_child_root / kind
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    request = {"kind": kind, "config": config_snapshot(cfg),
        "video_sha256": parent.result["assisted_inputs"]["video"]["sha256"],
        "source_sha256": parent.result["assisted_inputs"]["source_sha256"],
        "parent_output": str(parent.cfg.output_dir.resolve()), "launch_commit_headroom_bytes": headroom,
        "policy_evidence": deepcopy(parent.assisted_evidence),
        "cues_artifact": str((parent.assisted_child_root / "cues/candidate.json").resolve())}
    execution_id = parent.result["assisted_inputs"].get("execution_id", parent.assisted_child_root.name)
    request["execution_id"] = execution_id
    if kind == "expanded":
        from .public_reference_cache import prepare_reuse
        request["public_reference_reuse"] = prepare_reuse(
            getattr(parent, "assisted_worker_outcomes", {}).get("baseline"), cfg,
            execution_id=execution_id, video_sha256=request["video_sha256"],
            source_sha256=request["source_sha256"], child_root=parent.assisted_child_root)
    path = cfg.output_dir / "request.json"
    _write_result_json(path, _safe(request))
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "4"
    env.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONDONTWRITEBYTECODE="1", PROJ_NETWORK="OFF")
    source = Path(__file__).resolve().parents[1]
    paths = [str(source), *env.get("PYTHONPATH", "").split(os.pathsep)]
    env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
    print(f"[assisted] Running {kind.replace('_', ' ')}", flush=True)
    with (cfg.output_dir / "worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "monocular_osm.assisted_worker", "--request", str(path)],
                                  env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            returncode = child.wait()
        except BaseException:
            _stop_owned_child(child)
            raise
    log_path = cfg.output_dir / "worker.log"
    log_path.write_text(_safe(log_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    response = cfg.output_dir / "response.json"
    if not response.is_file():
        raise RuntimeError(f"{kind} worker exited {returncode} without a frozen response; see {cfg.output_dir / 'worker.log'}")
    value = json.loads(response.read_text(encoding="utf-8"))
    value["worker_returncode"] = returncode
    if returncode != 0:
        value["reported_status"] = value.get("status")
        value["status"] = "failed"
        value["transport_error"] = f"worker exited with code {returncode}; frozen diagnostics retained"
    outcomes = getattr(parent, "assisted_worker_outcomes", {})
    outcomes[kind] = {"kind": kind, "status": value.get("status"), "worker_returncode": returncode,
        "execution_id": execution_id, "video_sha256": request["video_sha256"],
        "source_sha256": deepcopy(request["source_sha256"]), "output_dir": str(cfg.output_dir.resolve()),
        "request_sha256": file_sha256(path), "frozen_artifact": deepcopy(value.get("frozen_artifact"))}
    parent.assisted_worker_outcomes = outcomes
    return value


def _sources(output):
    root = Path(__file__).resolve().parents[1]
    return snapshot_sources([root / "main.py", *sorted((root / "monocular_osm").rglob("*.py"))], output)


def _cues(cfg, video):
    import cv2
    from .assisted_policy import policy_config
    from .poi_clues import build_poi_hypotheses
    from .stages.context import RunContext
    from .stages.acquire import extract_video_frames
    from .stages.trajectory import run_visual_odometry
    from .stages.poi_only import _ocr, _inventory
    from .visual_odometry import default_intrinsics
    ctx = RunContext.create(cfg)
    ctx.video_path = video
    before = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    try:
        extract_video_frames(ctx)
        run_visual_odometry(ctx)
        times = np.asarray(ctx.frames.timestamps, float)
        shape = ctx.frames.frames[0].shape[:2]
        K = default_intrinsics(shape[1], shape[0])
        video_sha = file_sha256(video)
        path = cfg.output_dir / "cues_vo.npz"
        np.savez_compressed(path, xz=ctx.traj.xz, valid=ctx.traj.valid, timestamps=times, K=K)
        cache_source = Path(ctx.vo_cache).resolve(strict=True)
        cache_sha = file_sha256(cache_source)
        frozen_cache = cfg.output_dir / "cues_vo_cache.npz"
        shutil.copyfile(cache_source, frozen_cache)
        if file_sha256(frozen_cache) != cache_sha or file_sha256(cache_source) != cache_sha:
            raise ValueError("signed native VO cache changed while freezing cue inputs")
        with np.load(frozen_cache, allow_pickle=False) as saved:
            if not np.array_equal(saved["xz"], ctx.traj.xz) or not np.array_equal(saved["valid"], ctx.traj.valid):
                raise ValueError("frozen signed VO cache does not identify the actual cue trajectory")
        auto = {"status": "complete", "valid_fraction": float(np.mean(ctx.traj.valid)),
            "n_valid": int(np.sum(ctx.traj.valid)), "n_frames": len(times), "stride": cfg.frame_stride,
            "fps": float(ctx.frames.fps), "duration_s": float(times[-1]-times[0]),
            "timestamps_sha256": _array_sha(times), "video_sha256": video_sha,
            "window": [cfg.vo_start_sec, cfg.vo_end_sec],
            "vo_artifact": {"path": str(path.resolve()), "sha256": file_sha256(path)},
            "signed_vo_cache": {"path": str(frozen_cache.resolve()), "sha256": cache_sha},
            "warm_vo_cache_path": str(cache_source)}
        cues = {"status": "complete", "duration_s": auto["duration_s"], "vo": auto,
                "ocr": {"status": "not_run"}, "plane": {"status": "not_run", "measured": False}}
        result = {"status": "complete", "cues": cues, "auto_quality": auto,
                  "video_sha256": video_sha, "window": auto["window"]}
        try:
            detections, ocr = _ocr(ctx, K, video_sha, shape)
            result["ocr_provenance"] = ocr
            inventory, inventory_meta = _inventory(ctx)
            clues = build_poi_hypotheses(detections, inventory)
            result.update(ocr_provenance=ocr, inventory=inventory_meta, clues=clues)
            cues["ocr"] = {"status": "complete", "clue_branch_count": len(clues["candidate_clusters"]),
                           "detections_sha256": ocr["detections_sha256"], "inventory_sha256": inventory_meta["snapshot_sha256"]}
        except Exception as error:
            cues["ocr"] = {"status": "failed", "error": f"{type(error).__name__}: {_safe(str(error))}"}
        thresholds = policy_config()
        if (auto["duration_s"] >= thresholds["pf_duration_min_s"]
                and auto["valid_fraction"] >= thresholds["pf_vo_valid_min"]):
            ctx.frames.close()
            ctx.frames = None
            from .assisted_motion import prepare_metric_inputs
            prepared = prepare_metric_inputs(cfg, video, cfg.output_dir / "metric_preflight",
                                              cfg.data_dir / "metric_preflight", auto_quality=auto)
            result["metric_preflight"] = prepared
            cues["plane"] = deepcopy(prepared["summary"]["plane"])
            cues["plane"]["native_quality"] = deepcopy(prepared["summary"]["native_quality"])
            cues["plane"]["native_eligible"] = prepared["summary"]["eligible"]
        _write_result_json(cfg.output_dir / "cues.json", _safe(result))
        return result
    finally:
        if ctx.frames is not None:
            ctx.frames.close()
        cv2.setNumThreads(before)


def _valid_terminal(terminal, request):
    from .kartaview_vpr import _fetch_signature, _reference_snapshot
    try:
        count = terminal["count"]
        listing, snapshot = terminal["listing"], terminal["snapshot"]
        refs = terminal["reference_inventory"]
        signature = _fetch_signature(request["center"], request["radius_m"],
                                     request.get("ref_cap", request.get("cap")))
        return (terminal["status"] in {"complete_empty", "complete_sparse"}
            and not isinstance(count, bool) and isinstance(count, int) and 0 <= count < 30
            and terminal["source"] == "mapillary" and terminal["fetch_signature"] == signature
            and len(refs) == count and snapshot["n_refs"] == count
            and listing.get("listing_stats_known") is True and type(listing.get("lost")) is int
            and listing.get("listing") == "tiles" and listing.get("truncated", False) is False
            and listing["lost"] == 0 and listing["snapshot"] == snapshot
            and _reference_snapshot(refs)["refs_sha256"] == snapshot["refs_sha256"])
    except (KeyError, TypeError, ValueError):
        return False


def _completed_region(record):
    if record.get("status") == "ok" and not record.get("error"):
        return True
    terminal = (record.get("track_decoding") or {}).get("terminal_search") or {}
    return record.get("status") == "unavailable" and not record.get("error") and _valid_terminal(terminal, record)


def _expansion_complete(result, audits=None):
    multi = result.get("multi_region_retrieval") or {}
    regions = multi.get("regions") or []
    if not regions or any(not _completed_region(r) for r in regions):
        return False
    city = result.get("city_cover_retrieval")
    if city is not None:
        if city.get("status") not in {"ok", "unavailable"} or not city.get("coarse_regions"):
            return False
        if any(not _completed_region(r) for name in ("coarse_regions", "fine_regions") for r in city.get(name, [])):
            return False
    if audits is not None and any("returned_track" not in a and not _valid_terminal(
            a.get("terminal_search") or {}, a.get("request") or {}) for a in audits):
        return False
    return True


def _metadata_admission(ctx, request):
    from .assisted_profiles import resolve_policy
    decide_assisted_policy = resolve_policy(getattr(ctx.cfg, "assisted_profile", "v1")).decide_assisted_policy
    from .stages.assisted import summary
    directory = ctx.cfg.output_dir / "metadata_admission"
    directory.mkdir(exist_ok=True)
    try:
        track = ctx.vpr_track
        candidate = {"status": "complete" if track is not None and _expansion_complete(
                         ctx.result, getattr(ctx, "retrieval_audits", [])) else "partial",
                     "route_latlon": np.asarray(track[1]).tolist() if track is not None else [],
                     "comparison": _comparison(ctx, directory)}
        evidence = {**deepcopy(request.get("policy_evidence") or {}), "expanded": summary("expanded", candidate)}
        decision = decide_assisted_policy(evidence)
        admitted = decision.get("selected_candidate") == "expanded"
        record = {"ordered_subplans": ["metadata_and_corridors", "municipality_only_if_not_admitted"],
                  "evidence": evidence, "decision": decision, "municipality_skipped": admitted,
                  "comparison": candidate["comparison"]}
    except Exception as error:
        admitted = False
        record = {"municipality_skipped": False, "error": f"{type(error).__name__}: {_safe(str(error))}"}
    _write_result_json(directory / "admission.json", _safe(record))
    ctx.result["assisted_expansion_gate"] = record
    return admitted


def _pipeline_child(cfg, kind, request=None):
    from . import pipeline, baselines
    ctx = pipeline.RunContext.create(cfg)
    ctx.capture_retrieval_audit = kind in {"baseline", "expanded"}
    ctx.require_retrieval_track = kind == "expanded"
    if kind == "expanded":
        ctx.assisted_city_admission = lambda: _metadata_admission(ctx, request or {})
    try:
        if kind in {"baseline", "expanded"}:
            from .public_reference_cache import track_scope, seal_pool_audits
            with track_scope(cfg, kind, (request or {}).get("public_reference_reuse"), expected=request) as calls:
                result = pipeline._run_pipeline_stages(ctx)
            audits = getattr(ctx, "retrieval_audits", [])
            if len(calls) == len(audits):
                for audit, call in zip(audits, calls):
                    audit["public_cache_request"] = call["request"]
                    audit["public_cache_reuse"] = call["reuse"]
            if kind == "baseline":
                seal_pool_audits(audits, cfg)
        else:
            result = pipeline._run_pipeline_stages(ctx)
        prediction = baselines.answer(result)
        candidate = {"status": "complete" if prediction is not None else "unavailable"}
        if prediction is not None:
            candidate["route_latlon"] = deepcopy(prediction.route_latlon)
            times = (result.get("answer") or {}).get("route_t_sec")
            if times is None and ctx.vpr_track is not None and len(ctx.vpr_track[0]) == len(prediction.route_latlon):
                times = np.asarray(ctx.frames.timestamps, float)[np.asarray(ctx.vpr_track[0], int)].tolist()
            if times is not None:
                candidate["route_timestamps"] = times
        if kind in {"baseline", "expanded"}:
            try:
                candidate["comparison"] = _comparison(ctx, cfg.output_dir)
            except Exception as error:
                candidate["comparison"] = {"unavailable_reason": f"audit extraction failed: {type(error).__name__}: {_safe(str(error))}"}
            candidate["retrieval_selection"] = {name: deepcopy(result.get(name)) for name in
                ("multi_region_retrieval", "city_cover_retrieval", "vpr_coarse_to_fine",
                 "vpr_decoding", "vpr_prior", "vpr_listing", "assisted_expansion_gate")}
            if request and request.get("public_reference_reuse"):
                candidate["public_reference_cache_reuse"] = deepcopy(request["public_reference_reuse"])
            if kind == "expanded" and not _expansion_complete(result, getattr(ctx, "retrieval_audits", [])):
                candidate["status"] = "partial"
        if kind == "poi_joint":
            localized = result.get("poi_localization") or {}
            regions = localized.get("regions") or []
            candidate.update(retained_branch_count=len(localized.get("region_plans") or []),
                             successful_branch_count=sum(r.get("status") == "ok" for r in regions))
            if localized.get("failed_regions") or any(r.get("status") != "ok" for r in regions):
                candidate["status"] = "partial"
            candidate["poi_audit"] = result.get("provenance", {}).get("poi_localization_audit")
            candidate["poi_inputs"] = result.get("provenance", {}).get("poi_inputs")
        return candidate
    finally:
        if ctx.frames is not None:
            ctx.frames.close()


def _load_cues(request):
    path = Path(request["cues_artifact"]).resolve(strict=True)
    parent = Path(request["parent_output"]).resolve(strict=True)
    if not path.is_relative_to(parent):
        raise ValueError("cue artifact escaped its parent execution")
    cue = json.loads(path.read_text(encoding="utf-8"))
    freeze = json.loads((path.parent / "child_frozen.json").read_text(encoding="utf-8"))
    if freeze["artifacts"].get(path.name) != file_sha256(path):
        raise ValueError("cue candidate changed after its child freeze")
    if cue.get("video_sha256") != request["video_sha256"]:
        raise ValueError("cue artifacts identify a different encoded video")
    return cue


def _image_child(cfg, video, request):
    from .poi_image_track import plan_single_branch, decode_single_branch_image_track
    from .stages.context import RunContext
    from .stages.acquire import extract_video_frames
    from .stages.poi_only import _retrieval
    cue = _load_cues(request)
    plan = plan_single_branch(cue["clues"], cue["ocr_provenance"])
    _write_result_json(cfg.output_dir / "image_plan.json", _safe(plan))
    ctx = RunContext.create(cfg)
    ctx.video_path = video
    query_cache = {}
    try:
        extract_video_frames(ctx)
        if list(ctx.frames.frames[0].shape[:2]) != cue["ocr_provenance"]["frame_shape_hw"]:
            raise ValueError("image query pixels and native OCR boxes use different grids")
        auto = cue["auto_quality"]
        window = [cfg.vo_start_sec, cfg.vo_end_sec]
        if (cue.get("window") != window or auto.get("window") != window
                or auto.get("stride") != cfg.frame_stride or auto.get("fps") != float(ctx.frames.fps)
                or auto.get("n_frames") != len(ctx.frames.frames)
                or auto.get("timestamps_sha256") != _array_sha(np.asarray(ctx.frames.timestamps, float))):
            raise ValueError("image acquisition differs from its frozen native cue grid/window")
        inputs = {"inventory": cue["inventory"], "evidence_artifacts": []}
        evidence = _retrieval(ctx, inputs)(ctx.frames.frames, ctx.frames.timestamps, tuple(plan["center"]),
                     radius_m=3000., cap=1500, n_query=80, query_cache=query_cache)
        indices = np.asarray(evidence["query_indices"], int)
        if not np.array_equal(np.asarray(evidence["query_times"]), np.asarray(ctx.frames.timestamps)[indices]):
            raise ValueError("image evidence query rows differ from actual native video times")
        candidate = decode_single_branch_image_track(evidence, plan, query_video_sha256=request["video_sha256"])
        candidate.update(image_plan=plan, evidence_artifacts=inputs["evidence_artifacts"],
                         acquisition={"stride": cfg.frame_stride, "n_frames": len(ctx.frames.frames),
                                      "timestamps_sha256": _array_sha(np.asarray(ctx.frames.timestamps, float))})
        _write_result_json(cfg.output_dir / "image_prediction.json", _safe(candidate))
        return candidate
    finally:
        query_cache.clear()
        if ctx.frames is not None:
            ctx.frames.close()


def _metric_child(cfg, video, request):
    from types import SimpleNamespace
    from .assisted_motion import load_metric_inputs
    from .stages.context import RunContext, _record_channel
    from .stages.graph_filter import graph_filter_channel
    from .stages.report import finalize_result
    from .stages.filter_only import _freeze
    cue = _load_cues(request)
    prepared = cue.get("metric_preflight") or {}
    if (prepared.get("summary") or {}).get("eligible") is not True:
        return {"status": "unavailable", "error": "native VO/plane preflight failed fixed admission gates",
                "metric_preflight": prepared}
    trajectory, speed, times, K = load_metric_inputs(prepared, video,
                                                    window=[cfg.vo_start_sec, cfg.vo_end_sec])
    from .assisted_policy import policy_config
    thresholds = policy_config()
    if (float(np.mean(trajectory.valid)) < thresholds["pf_vo_valid_min"]
            or not np.isfinite(speed.coverage) or speed.coverage < thresholds["pf_plane_coverage_min"]
            or not np.isfinite(speed.rel_spread) or speed.rel_spread > thresholds["pf_plane_rel_spread_max"]):
        return {"status": "unavailable", "metric_preflight": prepared,
                "error": "actual loaded native VO/plane arrays fail fixed PF admission"}
    quality = prepared["summary"]["native_quality"]
    cfg.frame_stride = int(quality["stride"])
    ctx = RunContext.create(cfg)
    ctx.video_path, ctx.traj, ctx.n_valid = video, trajectory, int(np.sum(trajectory.valid))
    ctx.frames = SimpleNamespace(timestamps=times, fps=quality["fps"], frames=None,
                                 sampling={"mode": "verified_native_metric_preflight"})
    ctx.validated_metric_speed, ctx.require_metric_filter = speed, True
    inputs = {"input_policy": {name: False for name in ("calibration", "raw_images", "sensor_timestamps",
        "gps_or_oxts", "ground_truth", "retrieval", "initial_position")},
        "video": {"path": str(video.resolve()), "sha256": file_sha256(video)},
        "timing": {"source": "video_pts_or_encoded_fps", "fps": quality["fps"],
                   "first_s": float(times[0]), "last_s": float(times[-1]),
                   "timestamps_sha256": hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest()},
        "extraction": {"stride": cfg.frame_stride, "n_frames": len(times), "max_frames": None,
                       "frame_memory_mb": 128., "frame_max_width": None},
        "camera": {"intrinsics": "generic_70_degree_hfov", "K": K.tolist(),
                   "height_m": 1.5, "vo_rng_seed": 0},
        "metric_preflight": prepared, "source_sha256": _sources(cfg.output_dir)}
    ctx.result.update(filter_inputs=inputs, n_frames=len(times), n_valid_poses=ctx.n_valid,
                      execution={"stage": "filter", "geometry": "skipped", "retrieval": "skipped"})
    _write_result_json(cfg.output_dir / "filter_inputs.json", {**inputs, "config": config_snapshot(cfg)})
    for channel in ("vpr", "plate", "vlm", "sun", "ocr_anchor", "loop_closure", "coarse_from_frames",
                    "orienternet", "geometry"):
        _record_channel(ctx.channel_status, channel, "disabled", reason="metric_only_assisted_child")
    graph_filter_channel(ctx)
    if "filter_position" not in ctx.result or not ctx.result.get("graph_filter", {}).get("metric_odometry"):
        ctx.result["execution"].update(stage="failed", error="metric-only filter produced no admissible route")
        finalize_result(ctx)
        _freeze(ctx)
        return {"status": "unavailable", "metric_preflight": prepared}
    ctx.result["execution"]["stage"] = "complete"
    finalize_result(ctx)
    _freeze(ctx)
    answer = ctx.result["answer"]
    return {"status": "complete", "route_latlon": answer["route_latlon"],
        "route_timestamps": answer["route_t_sec"], "metric_only": ctx.result["graph_filter"]["metric_odometry"],
        "uses_user_spatial_prior": False,
        "position_observation_count": ctx.result["graph_filter"]["n_position_observations"],
        "metric_preflight": prepared, "filter_inputs": ctx.result["provenance"]["graph_filter_inputs"]}


def _finish(cfg, request, candidate):
    root = Path(__file__).resolve().parents[1]
    if any(file_sha256(root / name) != expected for name, expected in request["source_sha256"].items()):
        raise ValueError("assisted source changed during child execution")
    candidate = _safe(candidate)
    candidate.update(kind=request["kind"], video_sha256=request["video_sha256"],
        window=[cfg.vo_start_sec, cfg.vo_end_sec], config=config_snapshot(cfg),
        source_sha256=_sources(cfg.output_dir))
    if file_sha256(cfg.video_path) != request["video_sha256"]:
        raise ValueError("encoded video changed during assisted child execution")
    result_path = cfg.output_dir / "result.json"
    if not result_path.is_file():
        _write_result_json(result_path, {"experimental_component": request["kind"], "candidate": candidate})
    candidate_path = cfg.output_dir / "candidate.json"
    _write_result_json(candidate_path, candidate)
    paths = [path for path in cfg.output_dir.rglob("*") if path.is_file()
             and path.name not in {"worker.log", "child_frozen.json", "response.json"}]
    frozen_path = cfg.output_dir / "child_frozen.json"
    _write_result_json(frozen_path, {"scoring_has_not_run": True, "kind": request["kind"],
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": candidate["source_sha256"], "video_sha256": request["video_sha256"],
        "artifacts": {str(path.relative_to(cfg.output_dir)): file_sha256(path) for path in paths}})
    response = {**candidate, "candidate_artifact": {"path": str(candidate_path.resolve()), "sha256": file_sha256(candidate_path)},
                "result_artifact": {"path": str(result_path.resolve()), "sha256": file_sha256(result_path)},
                "frozen_artifact": {"path": str(frozen_path.resolve()), "sha256": file_sha256(frozen_path)}}
    _write_result_json(cfg.output_dir / "response.json", response)
    return response


def _visual_child(cfg, video, kind, request):
    import cv2
    import torch
    from .stages.poi_runtime import torch_cpu_threads
    previous = torch.hub.download_url_to_file
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    def refuse_download(*args, **kwargs):
        raise RuntimeError("assisted component requires cached model assets; downloads are disabled")
    torch.hub.download_url_to_file = refuse_download
    try:
        with torch_cpu_threads(4) as runtime:
            if kind == "poi_image_union":
                from .assisted_image_union import run_image_union_child
                candidate = run_image_union_child(cfg, video, request)
            else:
                candidate = (_image_child(cfg, video, request) if kind == "poi_image_track"
                             else _pipeline_child(cfg, kind, request))
            candidate["cpu_runtime"] = runtime
            candidate["cpu_runtime"]["opencv_num_threads"] = cv2.getNumThreads()
            candidate["model_downloads_disabled"] = True
            from . import kartaview_vpr as kv
            if kv._MODEL is not None:
                try:
                    candidate["actual_model_state"] = kv._audit_model_state()
                    candidate["actual_backbone"] = kv.backbone_info()
                except Exception as error:
                    candidate["model_state_unavailable"] = f"{type(error).__name__}: {_safe(str(error))}"
            return candidate
    finally:
        torch.hub.download_url_to_file = previous
        cv2.setNumThreads(previous_cv_threads)


def execute_request(request):
    cfg = _from_snapshot(request["config"])
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for name, expected in request["source_sha256"].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError("assisted source changed between parent and child")
    video = Path(cfg.video_path).resolve(strict=True)
    if file_sha256(video) != request["video_sha256"]:
        raise ValueError("assisted child received different video bytes")
    kind = request["kind"]
    if kind not in KINDS:
        raise ValueError("unrecognized assisted child")
    try:
        if kind == "cues":
            candidate = _cues(cfg, video)
        elif kind == "metric_pf":
            candidate = _metric_child(cfg, video, request)
        else:
            candidate = _visual_child(cfg, video, kind, request)
    except Exception as error:
        candidate = {"status": "failed", "error": f"{type(error).__name__}: {_safe(str(error))}"}
    return _finish(cfg, request, candidate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--request", type=Path)
    inputs.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args(argv)
    if args.check_runtime:
        missing = [name for name in ("_comparison", "_save_audits", "_metadata_admission", "execute_request")
                   if not callable(globals().get(name))]
        if missing:
            raise RuntimeError(f"assisted worker entered before its helpers were defined: {missing}")
        print(json.dumps({"helpers_ready": True, "models_run": False, "source": str(Path(__file__).resolve())}))
        return
    execute_request(json.loads(args.request.read_text(encoding="utf-8")))


def _save_audits(audits, output):
    directory = output / "retrieval_audit"
    directory.mkdir(exist_ok=True)
    artifacts = []
    for index, audit in enumerate(audits):
        arrays = {key: value for key, value in audit.items() if isinstance(value, np.ndarray)}
        arrays.update({"returned_" + key: value for key, value in audit.get("returned_track", {}).items()})
        path = directory / f"{index:04d}.npz"
        np.savez_compressed(path, **arrays)
        metadata = {key: value for key, value in audit.items()
                    if key not in arrays and key != "returned_track"}
        meta = path.with_suffix(".json")
        metadata["array_artifact"] = {"path": str(path.resolve()), "sha256": file_sha256(path),
            "arrays": {key: {"sha256": _array_sha(value), "shape": list(np.shape(value)),
                              "dtype": np.asarray(value).dtype.str} for key, value in arrays.items()}}
        _write_result_json(meta, _safe(metadata))
        artifacts.extend({"path": str(p.resolve()), "sha256": file_sha256(p)} for p in (path, meta))
    return artifacts


def _valid_audit(audit, timestamps=None):
    from .stages.assisted import digest
    try:
        if audit.get("status") != "complete":
            return False
        indices, times = np.asarray(audit["query_indices"]), np.asarray(audit["query_times"], float)
        references = np.asarray(audit["ref_latlon"], float)
        matrix, chosen = np.asarray(audit["raw_similarities"]), np.asarray(audit["decoded_ref_indices"])
        cosines = np.asarray(audit["decoded_cosines"], dtype=np.float64)
        identifiers = np.asarray(audit["ref_ids"]).astype(str)
        if (indices.dtype.kind not in "iu" or indices.ndim != 1 or len(indices) < 2
                or indices[0] < 0 or np.any(np.diff(indices) <= 0) or times.shape != indices.shape
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or references.ndim != 2 or references.shape[1] != 2
                or not np.isfinite(references).all() or len(set(identifiers)) != len(references)
                or identifiers.shape != (len(references),) or matrix.shape != (len(indices), len(references))
                or not np.isfinite(matrix).all() or np.any(np.abs(matrix) > 1.0001)
                or chosen.dtype.kind not in "iu" or chosen.shape != indices.shape
                or np.any(chosen < 0) or np.any(chosen >= len(references))
                or cosines.shape != indices.shape):
            return False
        actual_cosines = np.asarray(matrix[np.arange(len(indices)), chosen], dtype=np.float64)
        returned = audit["returned_track"]
        if (not np.array_equal(cosines, actual_cosines)
                or not np.array_equal(returned["indices"], indices)
                or not np.array_equal(returned["latlon"], references[chosen])
                or not np.array_equal(np.asarray(returned["cosines"], dtype=np.float64), cosines)):
            return False
        if timestamps is not None and not np.array_equal(times, np.asarray(timestamps)[indices]):
            return False
        identity = {key: audit[key] for key in ("query_pixels_sha256", "query_descriptors_sha256",
            "actual_model_state_sha256", "backbone", "preprocessing", "query_mode", "score_kind", "decoder")}
        identity.update(query_indices_sha256=_array_sha(indices), query_times_sha256=_array_sha(times))
        return (identity == audit["comparison_identity"]
                and digest(identity) == audit["comparison_identity_sha256"])
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _comparison(ctx, output):
    audits = getattr(ctx, "retrieval_audits", [])
    artifacts = _save_audits(audits, output)
    if ctx.vpr_track is None:
        return {"unavailable_reason": "no raw retrieval track", "audit_artifacts": artifacts}
    indices, coordinates, cosines = map(np.asarray, ctx.vpr_track)
    valid_audits = [(number, audit) for number, audit in enumerate(audits)
                    if _valid_audit(audit, getattr(ctx.frames, "timestamps", None))]
    rows, used = [], []
    for index, coordinate, cosine in zip(indices, coordinates, cosines):
        match = None
        for number, audit in valid_audits:
            track = audit["returned_track"]
            for row in np.flatnonzero(np.asarray(track["indices"]) == index):
                if (np.array_equal(track["latlon"][row], coordinate)
                        and float(track["cosines"][row]) == float(cosine)):
                    match = (number, int(row))
                    break
            if match is not None:
                break
        if match is None:
            return {"unavailable_reason": "selected track has no complete exact decoded-reference audit",
                    "audit_artifacts": artifacts}
        rows.append(match)
        used.append(audits[match[0]])
    if not used or len({a["comparison_identity_sha256"] for a in used}) != 1:
        return {"unavailable_reason": "selected regional rows have incompatible actual query/model/decoder identities",
                "audit_artifacts": artifacts}
    first = used[0]
    decoder = first["decoder"]
    model = {**first["backbone"], "weights_sha256": first["actual_model_state_sha256"]}
    query = {"indices": np.asarray(first["query_indices"]).tolist(),
             "times": np.asarray(first["query_times"]).tolist(), "mode": first["query_mode"],
             "preprocessing": first["preprocessing"], "pixels_sha256": first["query_pixels_sha256"],
             "descriptors_sha256": first["query_descriptors_sha256"]}
    identifiers = [str(a["ref_ids"][int(a["decoded_ref_indices"][row])]) for a, (_, row) in zip(used, rows)]
    return {"statistic": first["score_kind"], "model": model, "query": query,
        "inner_decoder": {"actual": decoder["actual"], "fallback": bool(decoder.get("fallback_reason")),
                          "settings": {key: value for key, value in decoder.items()
                                       if key not in {"actual", "fallback_reason"}}},
        "audited_identity": first["comparison_identity"],
        "audited_identity_sha256": first["comparison_identity_sha256"],
        "decoded_reference_cosines": cosines.tolist(), "decoded_reference_latlon": coordinates.tolist(),
        "decoded_reference_ids": identifiers, "selected_audit_rows": rows,
        "audit_artifacts": artifacts,
        "reference_encoding_identity": first.get("reference_encoding_identity"),
        "source_sampling": deepcopy(getattr(ctx.frames, "sampling", None))}


if __name__ == "__main__":
    main()
