"""Audit a benchmark summary against the full registered development target.

This reads JSON only; it never runs a model or changes benchmark data. Missing
clips, failed repeats and unknown provenance cannot satisfy the goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_FLAGS = ("--osm-around", "--ground-truth", "--ground-truth-waypoints",
                    "--estimated-length-m")
# The committed timed-poi-similarity-v1 defaults; POI answers must retain the
# fixed raw-appearance/distance/bearing experiment, without tuned score weights.
_POI_CORE_CONFIG_SHA256 = "940e6e8c320e2b9183b4c045aa8065128dabf27a19c5ede9f0eac2e5e1a4e4e4"


def _json_sha(value, *, ascii=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


def _float64_array_sha(values, shape):
    digest = hashlib.sha256(str((shape, "<f8")).encode())
    digest.update(struct.pack("<" + "d" * len(values), *values))
    return digest.hexdigest()


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_error(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _known_leak(args) -> bool:
    for arg in args or []:
        text = str(arg)
        flag = text.split("=", 1)[0]
        if (flag.startswith("--") and len(flag) > 2
                and any(f.startswith(flag) for f in _FORBIDDEN_FLAGS)):
            return True
        if ("ground_truth" in text.replace("\\", "/").lower()
                or re.search(r"-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+", text)):
            return True
    return False


def _absolute_path(value) -> bool:
    return (isinstance(value, str) and bool(value)
            and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()))


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _paths_match(left, right):
    if PureWindowsPath(left).drive or PureWindowsPath(right).drive:
        return PureWindowsPath(left) == PureWindowsPath(right)
    return PurePosixPath(left) == PurePosixPath(right)


def _filter_answer_issues(rep, score, city):
    """Audit video-only filtering; explicit city-map scope is a trusted declaration.

    A file digest binds the actual database, not its acquisition history. The
    caller must supply a whole-city map, rather than a clip/GT-centred crop.
    """
    problems = []
    provenance = rep.get("provenance") or {}
    cfg = provenance.get("config") or {}
    if ((score.get("meta") or {}).get("method") != "graph_filter_only"
            or cfg.get("answer") != "filter" or cfg.get("graph_filter") is not True
            or cfg.get("graph_filter_vpr") is not False):
        problems.append("filter answer lacks its enabled, retrieval-independent method configuration")
    if (cfg.get("city") != city or cfg.get("osm_around") is not None
            or cfg.get("estimated_length_m") is not None):
        problems.append("filter configuration violates the registered city-only inference contract")
    inputs = provenance.get("graph_filter_inputs") or {}
    policy = inputs.get("input_policy") or {}
    forbidden = ("calibration", "raw_images", "sensor_timestamps", "gps_or_oxts",
                 "ground_truth", "retrieval", "initial_position")
    if any(policy.get(name) is not False for name in forbidden):
        problems.append("strict filter input-policy audit is missing or reports external inputs")
    video = inputs.get("video") or {}
    if (not _absolute_path(video.get("path")) or not _sha256(video.get("sha256"))
            or _known_leak([video.get("path", "")])):
        problems.append("filter video provenance lacks a clean absolute path and content digest")
    if (inputs.get("timing") or {}).get("source") != "video_pts_or_encoded_fps":
        problems.append("filter timing does not come from the encoded video")
    graph = inputs.get("map") or {}
    if (graph.get("scope") != "whole_city" or graph.get("city") != city or graph.get("query") != city
            or "around" not in graph or graph["around"] is not None
            or "bbox" not in graph or graph["bbox"] is not None
            or graph.get("source") not in {"explicit_cache", "city_cache", "named_city_fetch"}
            or not _absolute_path(graph.get("cache_path")) or not _sha256(graph.get("sha256"))
            or _known_leak([graph.get("cache_path", "")])):
        problems.append("filter map lacks a fingerprinted whole-city database contract")
    configured = cfg.get("graph_filter_map_path")
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            problems.append("explicit filter map override is not a valid recorded path")
        if graph.get("source") != "explicit_cache":
            problems.append("explicit filter map override disagrees with recorded map provenance")
        if _absolute_path(configured):
            if not _paths_match(configured, str(graph.get("cache_path", ""))):
                problems.append("explicit filter map path differs from the map actually used")
        elif graph.get("configured_path") != configured:
            problems.append("relative filter map override lacks its exact configured-path provenance")
    elif graph.get("source") == "explicit_cache":
        problems.append("explicit filter map use is absent from the effective method configuration")
    return problems


def _poi_ocr_runtime_identity(ocr):
    """Semantic OCR identity survives copies of identical assets/environments."""
    try:
        reader = ocr["reader"]
        parameters = {key: value for key, value in reader["parameters"].items()
                      if key not in {"Det.model_path", "Rec.model_path", "Cls.model_path"}}
        assets = {task: {"name": PureWindowsPath(asset["path"]).name,
                         "bytes": asset["bytes"], "sha256": asset["sha256"]}
                  for task, asset in reader["model_assets"].items()}
        return _json_sha({"backend": reader["backend"], "parameters": parameters,
                          "model_assets": assets, "downloads_disabled": reader["downloads_disabled"],
                          "actual_sessions": reader["actual_sessions"], "actual_batches": reader["actual_batches"],
                          "runtime": ocr.get("runtime") or {}})
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _poi_reader_issues(ocr):
    problems = []
    reader = ocr.get("reader") or {}
    assets, parameters = reader.get("model_assets") or {}, reader.get("parameters") or {}
    tasks = {"Det", "Rec", "Cls"}
    if (ocr.get("engine") != "rapidocr" or ocr.get("model_weight_identity") != "explicit_cached_sha256"
            or not _absolute_path(ocr.get("reader_sidecar_path")) or not _sha256(ocr.get("reader_sidecar_sha256"))
            or reader.get("backend") != "rapidocr_onnxruntime" or reader.get("downloads_disabled") is not True
            or not isinstance(reader.get("cache_hit"), bool) or reader.get("read_calls") != 24
            or not _positive_int(reader.get("read_calls"))):
        problems.append("POI OCR lacks mandatory cached-asset/no-download/24-frame runtime proof")
    if (set(assets) != tasks or any(not isinstance(asset, dict)
            or not _absolute_path(asset.get("path")) or not _positive_int(asset.get("bytes"))
            or not _sha256(asset.get("sha256"))
            or not isinstance(parameters.get(f"{task}.model_path"), str)
            or not _paths_match(parameters[f"{task}.model_path"], asset["path"])
            for task, asset in assets.items())):
        problems.append("POI OCR Det/Rec/Cls asset hashes or explicit model paths are incomplete")
    fixed = {"Global.log_level": "error", "Global.text_score": .3,
             "Global.use_det": True, "Global.use_rec": True, "Global.use_cls": True,
             "EngineConfig.onnxruntime.intra_op_num_threads": 2,
             "EngineConfig.onnxruntime.inter_op_num_threads": 1,
             "EngineConfig.onnxruntime.use_cuda": False, "EngineConfig.onnxruntime.use_dml": False,
             "EngineConfig.onnxruntime.use_cann": False, "EngineConfig.onnxruntime.use_coreml": False,
             "Rec.rec_batch_num": 4, "Cls.cls_batch_num": 4}
    if any(parameters.get(key) != value or type(parameters.get(key)) is not type(value) for key, value in fixed.items()):
        problems.append("POI OCR declared CPU/2/1-thread/batch-4 parameters differ from the fixed profile")
    spec = {key: reader.get(key) for key in ("backend", "parameters", "model_assets", "downloads_disabled")}
    if not _sha256(reader.get("spec_sha256")) or reader["spec_sha256"] != _json_sha(spec):
        problems.append("POI OCR reader specification hash is missing or stale")
    sessions = reader.get("actual_sessions") or {}
    if (set(sessions) != {"text_det", "text_rec", "text_cls"}
            or any(session.get("providers") != ["CPUExecutionProvider"]
                   or session.get("intra_op_num_threads") != 2 or session.get("inter_op_num_threads") != 1
                   or not _positive_int(session.get("intra_op_num_threads"))
                   or not _positive_int(session.get("inter_op_num_threads")) for session in sessions.values())):
        problems.append("POI OCR actual Det/Rec/Cls providers/threads do not prove CPU 2/1 execution")
    if reader.get("actual_batches") != {"rec_batch_num": 4, "cls_batch_num": 4}:
        problems.append("POI OCR actual recognition/classification batches do not prove the bound of four")
    return problems


def _poi_cpu_runtime_identity(runtime):
    try:
        return _json_sha({key: runtime[key] for key in
                          ("device", "intra_op_threads", "inter_op_threads", "torch_version")})
    except (KeyError, TypeError, ValueError):
        return None


def _poi_input_issues(inputs, cfg, city, segment):
    problems = []
    policy = inputs.get("input_policy") or {}
    forbidden = ("calibration", "raw_images", "sensor_timestamps", "gps_or_oxts", "ground_truth",
                 "manual_region_coordinates", "external_position_prior", "external_trajectory", "external_ocr_video")
    required = ("video_ocr", "public_poi_inventory", "public_reference_positions", "position_stamps_rejected", "retrieval")
    if any(policy.get(key) is not False for key in forbidden) or any(policy.get(key) is not True for key in required):
        problems.append("POI input policy is missing or permits non-video/manual/GT inputs")
    video = inputs.get("video") or {}
    if (not _absolute_path(video.get("path")) or not _sha256(video.get("sha256"))
            or _known_leak([video.get("path", "")])):
        problems.append("POI query-video identity is missing or invalid")
    timing = inputs.get("timing") or {}
    if (timing.get("source") != "video_pts_or_encoded_fps" or not _sha256(timing.get("timestamps_sha256"))
            or _finite_error(timing.get("fps")) in (None, 0)
            or _finite_error(timing.get("first_s")) is None or _finite_error(timing.get("last_s")) is None
            or timing.get("last_s", 0) <= timing.get("first_s", 0)):
        problems.append("POI timing lacks an encoded-video grid identity")
    extraction = inputs.get("extraction") or {}
    expected_window = [float(value) if value else None for value in segment.split(":")]
    if (extraction.get("native_pixels") is not True or extraction.get("frame_max_width") is not None
            or extraction.get("max_frames") is not None or not _positive_int(extraction.get("n_frames"))
            or not _positive_int(extraction.get("stride")) or extraction.get("registered_window") != expected_window
            or cfg.get("max_frames") is not None or cfg.get("frame_max_width") is not None):
        problems.append("POI extraction does not preserve the full registered native-pixel window")
    camera = inputs.get("camera") or {}
    shape, K = camera.get("shape_hw") or [], camera.get("K") or []
    if (camera.get("intrinsics") != "generic_70_degree_hfov" or camera.get("vo_rng_seed") != 0
            or len(shape) != 2 or not all(_positive_int(v) for v in shape)
            or len(K) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in K)):
        problems.append("POI camera lacks native-image generic intrinsics")
    else:
        height, width = shape
        focal = width / 2 / math.tan(math.radians(35))
        expected = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               or not math.isclose(v, e, rel_tol=1e-10, abs_tol=1e-10)
               for row, reference in zip(K, expected) for v, e in zip(row, reference)):
            problems.append("POI camera matrix differs from generic video-derived intrinsics")
    vo = inputs.get("vo") or {}
    if (vo.get("source") != "same_video_generic_vo" or not _positive_int(vo.get("algorithm_version"))
            or not _absolute_path(vo.get("cache_path"))
            or any(not _sha256(vo.get(key)) for key in ("cache_sha256", "input_signature_sha256", "trajectory_sha256",
                                                       "timestamps_sha256", "K_sha256", "code_sha256"))):
        problems.append("POI VO lacks signed video-derived trajectory identities")
    if timing.get("timestamps_sha256") != vo.get("timestamps_sha256"):
        problems.append("POI encoded-video and VO timestamp identities disagree")
    if len(K) == 3 and all(isinstance(row, list) and len(row) == 3 for row in K):
        if _float64_array_sha([float(v) for row in K for v in row], (3, 3)) != vo.get("K_sha256"):
            problems.append("POI camera/VO intrinsics identities disagree")
    ocr = inputs.get("ocr") or {}
    problems.extend(_poi_reader_issues(ocr))
    if (ocr.get("source") != "same_video_native_frames" or ocr.get("video_sha256") != video.get("sha256")
            or not ocr.get("engine") or ocr.get("engine") != cfg.get("ocr_engine")
            or not _absolute_path(ocr.get("cache_path"))
            or any(not _sha256(ocr.get(key)) for key in ("cache_sha256", "detections_sha256"))
            or ocr.get("sample_count") != 24 or ocr.get("min_len") != 2 or ocr.get("min_confidence") != .3
            or ocr.get("super_res") is not False or ocr.get("use_gpu") is not False
            or ocr.get("native_bbox_grid") is not True or ocr.get("frame_shape_hw") != shape
            or ocr.get("position_stamp_filter") != "scene_text.is_position_stamp"
            or ocr.get("position_stamps_rejected") is not True
            or ocr.get("K_sha256") != vo.get("K_sha256")
            or ocr.get("window") != [timing.get("first_s"), timing.get("last_s")]):
        problems.append("POI OCR is not the fixed, stamp-filtered native query-video evidence")
    inventory = inputs.get("inventory") or {}
    bbox = inventory.get("bbox_wsen") or []
    if (inventory.get("city") != city or inventory.get("scope") != "named_city_bbox"
            or not _absolute_path(inventory.get("path")) or not _absolute_path(inventory.get("cache_dir"))
            or any(not _sha256(inventory.get(key)) for key in ("file_sha256", "snapshot_sha256", "response_sha256"))
            or len(bbox) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)):
        problems.append("POI inventory lacks a fingerprinted named-city database contract")
    elif (not isinstance(inventory.get("query"), str)
          or f"({bbox[1]:.7f},{bbox[0]:.7f},{bbox[3]:.7f},{bbox[2]:.7f})" not in inventory["query"]
          or 'nwr["shop"]' not in inventory["query"] or "out center tags" not in inventory["query"]):
        problems.append("POI public inventory query does not match its declared city-bbox snapshot")
    configured_inventory = cfg.get("poi_inventory_dir")
    if _absolute_path(configured_inventory) and not _paths_match(configured_inventory, str(inventory.get("cache_dir", ""))):
        problems.append("explicit POI inventory directory differs from the database actually used")
    graph = inputs.get("map") or {}
    if (graph.get("scope") != "whole_city" or graph.get("city") != city or graph.get("query") != city
            or "around" not in graph or graph["around"] is not None or "bbox" not in graph or graph["bbox"] is not None
            or graph.get("source") not in {"explicit_cache", "city_cache", "named_city_fetch"}
            or not _absolute_path(graph.get("cache_path")) or not _sha256(graph.get("sha256"))
            or _known_leak([graph.get("cache_path", "")])):
        problems.append("POI map lacks a fingerprinted whole-city database contract")
    configured = cfg.get("poi_map_path")
    if configured is not None:
        if graph.get("source") != "explicit_cache" or not isinstance(configured, str) or not configured:
            problems.append("explicit POI map override disagrees with recorded map provenance")
        elif (_absolute_path(configured) and not _paths_match(configured, str(graph.get("cache_path", "")))) or (
                not _absolute_path(configured) and graph.get("configured_path") != configured):
            problems.append("explicit POI map path differs from the database actually used")
    elif graph.get("source") == "explicit_cache":
        problems.append("explicit POI map use is absent from effective method configuration")
    sources = inputs.get("source_sha256") or {}
    if not sources or any(not _sha256(value) for value in sources.values()):
        problems.append("POI implementation source identities are missing")
    retrieval = inputs.get("retrieval") or {}
    if (retrieval.get("requested_model") != "megaloc" or retrieval.get("device") != "cpu"
            or retrieval.get("cpu_threads") != 4 or not _positive_int(retrieval.get("cpu_threads"))
            or retrieval.get("n_query") != 80 or retrieval.get("query_mode") != cfg.get("vpr_query_mode")
            or retrieval.get("appearance") != "raw" or retrieval.get("poi_loss") != "distance_and_bearing"):
        problems.append("POI retrieval does not declare its fixed raw appearance/query contract")
    return problems


def _poi_evidence_issues(audit, inputs, cfg):
    problems = []
    clues, provenance = audit.get("clues") or {}, audit.get("provenance") or {}
    plans, regions = audit.get("region_plans") or [], audit.get("regions") or []
    clusters = clues.get("candidate_clusters") or []
    ids = [str(branch.get("id", f"hypothesis_{i}")) for i, branch in enumerate(clusters)]
    if (audit.get("status") != "ok" or "failed_regions" not in audit or audit["failed_regions"]
            or not ids or len(set(ids)) != len(ids) or len(plans) != len(ids) or len(regions) != len(ids)
            or clues.get("selected_answer") is not None):
        problems.append("POI localization is partial, failed, or missing complete unselected clue branches")
    if ([p.get("hypothesis_id") for p in plans] != ids or [r.get("hypothesis_id") for r in regions] != ids
            or provenance.get("region_plans_sha256") != _json_sha(plans)):
        problems.append("POI region plans do not retain exactly every clue branch")
    config, fixed = provenance.get("config") or {}, provenance.get("fixed_alignment_config") or {}
    if (_json_sha(fixed) != _POI_CORE_CONFIG_SHA256 or config.get("radius_m") != cfg.get("poi_radius_m")
            or config.get("cap_per_branch") != cfg.get("poi_ref_cap") or config.get("n_query") != 80
            or provenance.get("query_cache_shared") is not True):
        problems.append("POI retrieval budgets/fixed core/shared-query contract disagrees with configuration")
    if ((clues.get("inventory") or {}).get("snapshot_sha256") != (inputs.get("inventory") or {}).get("snapshot_sha256")
            or clues.get("detections_sha256") != (inputs.get("ocr") or {}).get("detections_sha256")):
        problems.append("POI clue inventory/OCR identities disagree with video-derived inputs")
    union = audit.get("reference_union") or {}
    indices, times = union.get("query_indices") or [], union.get("query_times") or []
    extraction, timing = inputs.get("extraction") or {}, inputs.get("timing") or {}
    if (not 2 <= len(indices) <= 80 or len(times) != len(indices)
            or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= extraction.get("n_frames", 0) for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))
            or any(_finite_error(t) is None or not timing.get("first_s", 0) <= t <= timing.get("last_s", -1) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        problems.append("POI common query grid is missing or outside the supplied video timeline")
    expected_union, seen, identities, cpu_identities = [], set(), set(), set()
    artifacts = inputs.get("evidence_artifacts") or []
    if (len(artifacts) != len(regions) or len({r.get("path") for r in artifacts}) != len(artifacts)
            or any(not r.get("path") or not _sha256(r.get("sha256")) for r in artifacts)):
        problems.append("POI complete full-matrix artifact identities are missing")
    for index, region in enumerate(regions):
        plan = plans[index] if index < len(plans) else {}
        center = plan.get("center") or []
        if (plan.get("origin") != "literal_poi_branch_centroid" or len(center) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in center)
                or abs(center[0]) > 90 or abs(center[1]) > 180
                or plan.get("radius_m") != cfg.get("poi_radius_m") or plan.get("cap") != cfg.get("poi_ref_cap")
                or index >= len(clusters) or plan.get("poi_ids") != clusters[index].get("poi_ids")
                or any(region.get(key) != plan.get(key) for key in plan) or region.get("status") != "ok"):
            problems.append("POI regions are failed, manually selected, or have unequal branch budgets")
        rp = region.get("provenance") or {}
        cpu = rp.get("cpu_runtime") or {}
        if (cpu.get("device") != "cpu" or cpu.get("intra_op_threads") != 4
                or not _positive_int(cpu.get("intra_op_threads")) or not _positive_int(cpu.get("inter_op_threads"))
                or not isinstance(cpu.get("torch_version"), str) or not cpu["torch_version"]):
            problems.append("POI retrieval lacks actual CPU/four-thread/Torch runtime proof")
        cpu_identity = _poi_cpu_runtime_identity(cpu)
        if cpu_identity is not None:
            cpu_identities.add(cpu_identity)
        evidence = rp.get("evidence_identity") or {}
        if index >= len(artifacts) or rp.get("matrix_artifact") != artifacts[index]:
            problems.append("POI region matrix artifact differs from acquired evidence")
        ref_ids = evidence.get("ref_ids") or []
        if (not ref_ids or len(set(ref_ids)) != len(ref_ids)
                or region.get("reference_count") != len(ref_ids) or len(ref_ids) > cfg.get("poi_ref_cap", 0)
                or region.get("ref_ids_sha256") != _json_sha(ref_ids)
                or evidence.get("query_indices") != indices or evidence.get("query_times") != times
                or region.get("query_count") != len(times)
                or not _sha256(evidence.get("ref_latlon_sha256"))
                or not _sha256(evidence.get("similarities_sha256"))
                or region.get("similarities_sha256") != evidence.get("similarities_sha256")):
            problems.append("POI region matrices/reference IDs do not share a complete fingerprinted query grid")
        source = rp.get("source")
        if (source not in {"mapillary", "kartaview", "panoramax"}
                or any(not isinstance(identifier, str) or not identifier.startswith(f"{source}:") for identifier in ref_ids)
                or not rp.get("reference_encoding_identity")
                or (rp.get("reference_inventory") or {}).get("n_refs") != len(ref_ids)
                or not _sha256((rp.get("reference_inventory") or {}).get("refs_sha256"))):
            problems.append("POI references lack public street-photo source/encoding/inventory provenance")
        backbone, preprocessing = rp.get("backbone") or {}, rp.get("preprocessing") or {}
        if (not backbone.get("name") or not backbone.get("hub") or not backbone.get("device")
                or backbone.get("device") != (inputs.get("retrieval") or {}).get("device")
                or rp.get("query_mode") != cfg.get("vpr_query_mode")
                or any(key not in preprocessing for key in ("version", "reference_color", "reference_size",
                    "reference_resize", "scale", "mean", "std", "descriptor_normalization", "dimensions"))
                or not _positive_int(preprocessing.get("dimensions"))):
            problems.append("POI retrieval backbone/preprocessing identity is missing")
        identity = _json_sha({"backbone": backbone, "preprocessing": preprocessing, "query_mode": rp.get("query_mode")})
        identities.add(identity)
        if identity != region.get("identity_sha256"):
            problems.append("POI declared retrieval identity differs from its actual region identity")
        new = [identifier for identifier in ref_ids if identifier not in seen]
        if (region.get("new_unique_reference_count") != len(new)
                or region.get("duplicate_reference_count") != len(ref_ids) - len(new)):
            problems.append("POI duplicate-reference accounting is inconsistent")
        seen.update(ref_ids)
        expected_union.extend(new)
    if (len(identities) != 1 or union.get("identity_sha256") not in identities
            or union.get("ids") != expected_union or union.get("count") != len(expected_union)
            or union.get("ids_sha256") != _json_sha(expected_union)
            or any(not _sha256(union.get(key)) for key in ("ref_latlon_sha256", "similarities_sha256"))):
        problems.append("POI reference union/model identity is incomplete or inconsistent")
    if len(cpu_identities) != 1:
        problems.append("POI regions used inconsistent or unknown CPU retrieval runtimes")
    alignment = audit.get("alignment") or {}
    ap, appearance = alignment.get("provenance") or {}, (alignment.get("provenance") or {}).get("appearance") or {}
    field = appearance.get("sources") or {}
    if (alignment.get("hypothesis_ids") != ids or alignment.get("selected_hypothesis_id") not in ids
            or (alignment.get("search") or {}).get("hypothesis_count") != len(ids)
            or ap.get("algorithm") != "timed-poi-similarity-v1" or _json_sha(ap.get("config")) != _POI_CORE_CONFIG_SHA256
            or ap.get("candidate_clusters_sha256") != _json_sha(clusters, ascii=True)
            or ap.get("literal_clues_sha256") != _json_sha({k: v for k, v in clues.items() if k != "candidate_clusters"}, ascii=True)):
        problems.append("POI alignment does not retain the complete fixed-core hypothesis set")
    vo = inputs.get("vo") or {}
    if (ap.get("vo_sha256") != vo.get("trajectory_sha256") or ap.get("times_sha256") != vo.get("timestamps_sha256")
            or ap.get("intrinsics_sha256") != vo.get("K_sha256")
            or any(not _sha256(ap.get(key)) for key in ("observations_sha256", "pois_sha256", "hypotheses_sha256", "query_times_sha256"))):
        problems.append("POI alignment motion/camera/clue identities disagree with acquired inputs")
    if ap.get("query_times_sha256") != _float64_array_sha([float(t) for t in times], (len(times),)):
        problems.append("POI core query-time identity differs from the shared retrieval rows")
    grid, road = audit.get("grid") or {}, ap.get("road_distance") or {}
    samples = inputs.get("road_samples") or {}
    if (field.get("reference_mean") is not False or field.get("reference_count") != union.get("count")
            or field.get("similarities_sha256") != union.get("similarities_sha256")
            or _json_sha(field.get("config")) != _POI_CORE_CONFIG_SHA256
            or appearance.get("query_times_sha256") != ap.get("query_times_sha256")
            or any(not _sha256(value) for value in (appearance.get("values_sha256"), road.get("values_sha256"),
                                                   provenance.get("road_samples_sha256"), grid.get("start_samples_sha256")))
            or (road.get("sources") or {}).get("road_samples_sha256") != provenance.get("road_samples_sha256")
            or samples.get("sha256") != provenance.get("road_samples_sha256")
            or samples.get("spacing_m") != 12.5 or not _positive_int(samples.get("count"))
            or grid.get("start_policy") != "nearest road sample per fixed 100m bin; uniform cap over all-city bins"
            or ap.get("start_samples_sha256") != grid.get("start_samples_sha256")):
        problems.append("POI field/whole-city road-start identities do not match fixed raw alignment")
    search, starts = alignment.get("search") or {}, grid.get("start_count")
    coarse = search.get("coarse_pose_count")
    if (not _positive_int(starts) or starts > 15000 or not _positive_int(coarse)
            or coarse != 9 * 24 * starts or coarse > 4_000_000
            or search.get("coarse_hypothesis_scores") != coarse * len(ids)
            or coarse * len(ids) > 20_000_000 or not _positive_int(search.get("fine_pose_count"))
            or search.get("fine_pose_count", 0) > 250_000 or search.get("query_count") != len(times)
            or search.get("start_mode") != "provided"):
        problems.append("POI fixed whole-city search counts are incomplete or inconsistent")
    return problems


def _poi_answer_issues(rep, score, city, segment):
    try:
        provenance = rep.get("provenance") or {}
        cfg = provenance.get("config") or {}
        problems = []
        if ((score.get("meta") or {}).get("method") != "poi_only" or cfg.get("answer") != "poi"
                or cfg.get("city") != city or cfg.get("osm_around") is not None
                or cfg.get("estimated_length_m") is not None or _finite_error(cfg.get("poi_radius_m")) in (None, 0)
                or not _positive_int(cfg.get("poi_ref_cap"))
                or any(cfg.get(key) is not None for key in ("openvo_trajectory_path", "vggt_long_trajectory_path", "ocr_video_path"))):
            problems.append("POI selected method/configuration violates video/city-only inference")
        inputs, audit = provenance.get("poi_inputs") or {}, provenance.get("poi_localization_audit") or {}
        problems.extend(_poi_input_issues(inputs, cfg, city, segment))
        problems.extend(_poi_evidence_issues(audit, inputs, cfg))
        return problems
    except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
        return [f"POI audit is missing or malformed: {exc}"]


def check_goal(summary: dict, manifest: dict, *, threshold_m: float = 200.0,
               expected_count: int = 15, method: str = "answer",
               min_repeats: int = 1) -> dict:
    """Require every available split=dev clip, with failures preserved.

    A clip's target is its median symmetric error over repeats. Every repeat
    must nevertheless have a real, finite answer; a median cannot hide a
    failed invocation. Recompute medians from the saved per-repeat scores.
    """
    reasons = []
    if summary.get("provisional") or (summary.get("review") or {}).get("status") == "in_progress_snapshot":
        reasons.append("summary is an in-progress snapshot, not a completed result")
    if not math.isfinite(threshold_m) or threshold_m < 0 or expected_count < 1 or min_repeats < 1:
        raise ValueError("threshold must be finite/nonnegative; counts must be positive")
    if any(word in method.lower() for word in ("oracle", "ground_truth")):
        reasons.append("the selected method uses ground truth or an oracle")
    expected = {c["id"]: c for c in manifest.get("clips", [])
                if c.get("split") == "dev" and c.get("available_locally") is True}
    if len(expected) != expected_count:
        reasons.append(f"manifest has {len(expected)} eligible dev clips, expected {expected_count}")
    clips = summary.get("clips") or {}
    missing, unexpected = sorted(set(expected) - set(clips)), sorted(set(clips) - set(expected))
    if missing:
        reasons.append("expected clips are missing: " + ", ".join(missing))
    if unexpected:
        reasons.append("summary contains clips outside the target: " + ", ".join(unexpected))

    prov = summary.get("provenance") or {}
    if prov.get("consistent") is not True or prov.get("problems"):
        reasons.append("benchmark provenance is inconsistent or unknown")
    if "leaks" not in prov or prov["leaks"]:
        reasons.append("ground-truth leak audit is missing or reports leakage")
    if prov.get("dirty") != ["False"]:
        reasons.append("prediction code is dirty or its clean state is unknown")
    commits = prov.get("commits") or []
    if not commits or any(c in (None, "None") for c in commits):
        reasons.append("prediction commits are missing or unknown")
    trees = prov.get("package_trees") or []
    if len(trees) != 1 or str(trees[0]).startswith("unknown@"):
        reasons.append("all target clips must use one known pipeline code tree")
    for field in ("method_config_fingerprints", "environment_fingerprints"):
        if len(prov.get(field) or []) != 1:
            reasons.append(f"one consistent {field} value is required")
    if len(prov.get("vpr_model_fingerprints") or []) > 1:
        reasons.append("target clips used different VPR models")

    execution = summary.get("execution") or {}
    if (execution.get("plan_recorded") is not True or execution.get("complete") is not True
            or execution.get("incomplete_reps")):
        reasons.append("a recorded, complete execution plan is required")
    metadata = summary.get("run_config") or {}
    plan = metadata.get("execution_plan") or {}
    if set(plan) != set(expected):
        reasons.append("execution plan does not cover exactly the expected dev clips")
    if _known_leak((metadata.get("config") or {}).get("pipeline_args")):
        reasons.append("run configuration contains known ground-truth-derived inputs")

    reps = summary.get("reps") or []
    if execution.get("expected_reps") != len(reps) or execution.get("n_completed") != len(reps):
        reasons.append("execution counts disagree with the recorded repeats")
    by_clip = {cid: [] for cid in expected}
    for rep in reps:
        if rep.get("clip") not in by_clip:
            reasons.append("a repeat belongs to a clip outside the development target")
            continue
        by_clip[rep["clip"]].append(rep)
        if rep.get("commit") not in commits:
            reasons.append(f"{rep['clip']}: repeat commit disagrees with provenance")
        rep_prov = rep.get("provenance") or {}
        cfg = rep_prov.get("config") or {}
        if (_known_leak(rep_prov.get("argv")) or cfg.get("ground_truth_waypoints")
                or cfg.get("ground_truth_streets") or rep_prov.get("leaks")
                or (cfg.get("osm_around") and cfg.get("osm_around_source") != "video_title")):
            reasons.append(f"{rep['clip']}: repeat provenance contains known ground-truth leakage")

    rows, answer_modes, poi_identities = [], set(), set()
    poi_ocr_identities, poi_cpu_identities = set(), set()
    for cid, entry in expected.items():
        errors = []
        issues = []
        clip = clips.get(cid) or {}
        records = by_clip[cid]
        if clip.get("split") != "dev":
            issues.append("development split is missing or incorrect")
        if clip.get("city") not in {entry["city"], *entry.get("city_alternatives", [])}:
            issues.append("city differs from the registered inputs")
        if clip.get("vo_segments") != [entry["vo_segment"]]:
            issues.append("analysis window differs from the registered inputs")
        if (not entry.get("gt_waypoints_sha256")
                or clip.get("gt_waypoints_sha256") != entry["gt_waypoints_sha256"]):
            issues.append("ground-truth fingerprint is stale or missing")
        method_summary = (clip.get("methods") or {}).get(method) or {}
        median = _finite_error((method_summary.get("symmetric_mean_m") or {}).get("median"))
        if median is None:
            issues.append("selected method has no finite symmetric error")
        if len(records) < min_repeats:
            issues.append(f"fewer than {min_repeats} recorded repeats")
        for count in (clip.get("n_reps"), clip.get("n_ok"), clip.get("n_completed"),
                      method_summary.get("n_reps"), method_summary.get("n_answered")):
            if count != len(records):
                issues.append("repeat/answer counts are incomplete or inconsistent")
                break
        indices = [r.get("rep") for r in records]
        planned = (plan.get(cid) or {}).get("reps") or []
        if len(set(indices)) != len(indices) or set(indices) != set(planned):
            issues.append("recorded repeats do not match the execution plan")
        for rep in records:
            score = (rep.get("scores") or {}).get(method) or {}
            error = _finite_error(score.get("symmetric_mean_m"))
            if (rep.get("ok") is not True or rep.get("completed") is not True
                    or rep.get("result_found") is not True or rep.get("rc") != 0
                    or score.get("answered") is not True or score.get("failed") is not False
                    or error is None):
                issues.append(f"rep{rep.get('rep')}: failed, unanswered or nonfinite")
            else:
                errors.append(error)
            source = str(score.get("source") or "").lower()
            chosen = str((score.get("meta") or {}).get("method") or "").lower()
            if (score.get("oracle") or "oracle" in source or "ground_truth" in source
                    or "oracle" in chosen or "ground_truth" in chosen):
                issues.append(f"rep{rep.get('rep')}: selected score is an oracle/GT result")
            if method == "answer":
                mode = (score.get("meta") or {}).get("mode")
                configured_answer = ((rep.get("provenance") or {}).get("config") or {}).get("answer")
                if source != "result.answer" or mode not in {"fit", "track", "checked", "filter", "poi"}:
                    issues.append(f"rep{rep.get('rep')}: selected answer is stale or missing")
                else:
                    answer_modes.add(mode)
                filter_answer = mode == "filter" or configured_answer == "filter" or chosen == "graph_filter_only"
                poi_answer = mode == "poi" or configured_answer == "poi" or chosen == "poi_only"
                if filter_answer:
                    if mode != "filter":
                        issues.append(f"rep{rep.get('rep')}: filter answer mode disagrees with method/configuration")
                    issues.extend(f"rep{rep.get('rep')}: {problem}"
                                  for problem in _filter_answer_issues(rep, score, clip.get("city")))
                if poi_answer:
                    if mode != "poi":
                        issues.append(f"rep{rep.get('rep')}: POI answer mode disagrees with method/configuration")
                    issues.extend(f"rep{rep.get('rep')}: {problem}" for problem in
                                  _poi_answer_issues(rep, score, clip.get("city"), entry["vo_segment"]))
                    identity = (((rep.get("provenance") or {}).get("poi_localization_audit") or {}).get(
                        "reference_union") or {}).get("identity_sha256")
                    if _sha256(identity):
                        poi_identities.add(identity)
                    rep_prov = rep.get("provenance") or {}
                    ocr_identity = _poi_ocr_runtime_identity(((rep_prov.get("poi_inputs") or {}).get("ocr") or {}))
                    if ocr_identity is not None:
                        poi_ocr_identities.add(ocr_identity)
                    for region in (rep_prov.get("poi_localization_audit") or {}).get("regions") or []:
                        if not isinstance(region, dict):
                            continue
                        runtime_identity = _poi_cpu_runtime_identity((region.get("provenance") or {}).get("cpu_runtime") or {})
                        if runtime_identity is not None:
                            poi_cpu_identities.add(runtime_identity)
        filter_inputs = [(rep.get("provenance") or {}).get("graph_filter_inputs") or {}
                         for rep in records if (((rep.get("scores") or {}).get(method) or {}).get(
                             "meta") or {}).get("mode") == "filter"]
        if len({((data.get("video") or {}).get("sha256"),
                 (data.get("map") or {}).get("sha256")) for data in filter_inputs}) > 1:
            issues.append("filter video/map content changes between repeats")
        poi_inputs = [(rep.get("provenance") or {}).get("poi_inputs") or {} for rep in records
                      if (((rep.get("scores") or {}).get(method) or {}).get("meta") or {}).get("mode") == "poi"]
        stable_poi_inputs = {
            ((data.get("video") or {}).get("sha256"), (data.get("map") or {}).get("sha256"),
             (data.get("inventory") or {}).get("snapshot_sha256"), (data.get("ocr") or {}).get("detections_sha256"),
             (data.get("vo") or {}).get("trajectory_sha256"), (data.get("vo") or {}).get("timestamps_sha256"),
             (data.get("vo") or {}).get("K_sha256")) for data in poi_inputs}
        if len(stable_poi_inputs) > 1:
            issues.append("POI video/map/inventory/OCR/VO identities change between repeats")
        measured = statistics.median(errors) if errors else None
        if measured is not None and median is not None and abs(round(measured, 1) - median) > 0.051:
            issues.append("reported median does not match the recorded repeat scores")
        if median is not None and median > threshold_m:
            issues.append(f"symmetric error exceeds {threshold_m:g} m")
        rows.append({"clip": cid, "n_repeats": len(records), "symmetric_mean_m": median,
                     "passes": not issues, "reasons": list(dict.fromkeys(issues))})
    if len(answer_modes) > 1:
        reasons.append("the selected answer mode differs between clips/repeats")
    if len(poi_identities) > 1:
        reasons.append("POI answers used different query/backbone/preprocessing identities")
    if len(poi_ocr_identities) > 1:
        reasons.append("POI answers used different semantic OCR model/config/runtime identities")
    if len(poi_cpu_identities) > 1:
        reasons.append("POI answers used different CPU retrieval runtime identities")
    passed = sum(row["passes"] for row in rows)
    return {
        "goal_met": not reasons and passed == expected_count,
        "method": method, "threshold_m": threshold_m, "expected_count": expected_count,
        "min_repeats": min_repeats, "n_passing": passed, "n_expected": len(expected),
        "missing_ids": missing, "unexpected_ids": unexpected,
        "reasons": list(dict.fromkeys(reasons)), "clips": rows,
        "input_contract_notes": (["Explicit maps and inventories are trusted named-city database inputs; "
                                  "content hashes bind the file, not its acquisition history."]
                                 if answer_modes & {"filter", "poi"} else []) + (
            ["POI query-video GPS/manual priors are forbidden; public POI and street-photo positions are permitted. "
             "Reference-cache encoding uncertainty remains recorded in every region's provenance."]
            if "poi" in answer_modes else []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="benchmark aggregate summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifest.json")
    parser.add_argument("--threshold-m", type=float, default=200.0)
    parser.add_argument("--expected-count", type=int, default=15)
    parser.add_argument("--method", default="answer")
    parser.add_argument("--min-repeats", "--min-reps", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = check_goal(summary, manifest, threshold_m=args.threshold_m,
                            expected_count=args.expected_count, method=args.method,
                            min_repeats=args.min_repeats)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = {"goal_met": False, "reasons": [f"cannot audit inputs: {error}"]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["goal_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
