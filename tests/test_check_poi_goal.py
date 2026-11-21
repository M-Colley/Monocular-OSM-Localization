"""POI primary answers require complete, input-derived all-branch evidence."""

from copy import deepcopy
import hashlib
import json
import math

import pytest

from monocular_osm.poi_alignment import POIAlignmentConfig
from tests.test_check_dev_goal import M, _valid


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _reader_proof(directory="/models"):
    names = {"Det": "PP-OCRv6_det_small.onnx", "Rec": "PP-OCRv6_rec_small.onnx",
             "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx"}
    assets = {task: {"path": f"{directory}/{name}", "bytes": 1000 + len(name), "sha256": _hash(name)}
              for task, name in names.items()}
    params = {f"{task}.model_path": asset["path"] for task, asset in assets.items()}
    params.update({"Global.log_level": "error", "Global.text_score": .3,
        "Global.use_det": True, "Global.use_rec": True, "Global.use_cls": True,
        "EngineConfig.onnxruntime.intra_op_num_threads": 2, "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False, "EngineConfig.onnxruntime.use_dml": False,
        "EngineConfig.onnxruntime.use_cann": False, "EngineConfig.onnxruntime.use_coreml": False,
        "Rec.rec_batch_num": 4, "Cls.cls_batch_num": 4})
    reader = {"backend": "rapidocr_onnxruntime", "parameters": params, "model_assets": assets, "downloads_disabled": True}
    reader["spec_sha256"] = M._json_sha(reader)
    reader.update(actual_sessions={name: {"providers": ["CPUExecutionProvider"], "intra_op_num_threads": 2,
                                          "inter_op_num_threads": 1} for name in ("text_det", "text_rec", "text_cls")},
                  actual_batches={"rec_batch_num": 4, "cls_batch_num": 4}, read_calls=24, cache_hit=False)
    return reader


def _poi_contract(city):
    cfg = {"answer": "poi", "city": city, "poi_radius_m": 1000., "poi_ref_cap": 500,
           "poi_map_path": None, "poi_inventory_dir": None, "ocr_engine": "rapidocr",
           "osm_around": None, "estimated_length_m": None, "max_frames": None, "frame_max_width": None,
           "vpr_source": "mapillary", "vpr_query_mode": "warp"}
    fixed = json.loads(json.dumps(POIAlignmentConfig().to_dict()))
    focal = 320 / math.tan(math.radians(35))
    K = [[focal, 0., 320.], [0., focal, 240.], [0., 0., 1.]]
    K_sha = M._float64_array_sha([float(v) for row in K for v in row], (3, 3))
    policy = {key: False for key in ("calibration", "raw_images", "sensor_timestamps", "gps_or_oxts",
        "ground_truth", "manual_region_coordinates", "external_position_prior", "external_trajectory", "external_ocr_video")}
    policy.update({key: True for key in ("video_ocr", "public_poi_inventory", "public_reference_positions",
                                       "position_stamps_rejected", "retrieval")})
    inputs = {"input_policy": policy, "video": {"path": "/video/input.mp4", "sha256": _hash("video")},
        "timing": {"source": "video_pts_or_encoded_fps", "fps": 10., "first_s": 0., "last_s": 300.,
                   "timestamps_sha256": _hash("timestamps_sha256")},
        "extraction": {"stride": 1, "n_frames": 3001, "max_frames": None, "frame_max_width": None,
                       "native_pixels": True, "registered_window": [0., 300.]},
        "camera": {"intrinsics": "generic_70_degree_hfov", "K": K, "shape_hw": [480, 640], "vo_rng_seed": 0},
        "vo": {"source": "same_video_generic_vo", "algorithm_version": 10, "cache_path": "/cache/video_vo.npz",
               **{key: _hash(key) for key in ("cache_sha256", "input_signature_sha256", "trajectory_sha256",
                                             "timestamps_sha256", "code_sha256")}, "K_sha256": K_sha},
        "ocr": {"source": "same_video_native_frames", "video_sha256": _hash("video"), "engine": "rapidocr",
                "cache_path": "/cache/video_ocr.json", "cache_sha256": _hash("ocrfile"), "sample_count": 24,
                "min_len": 2, "min_confidence": .3, "super_res": False, "use_gpu": False,
                "native_bbox_grid": True, "frame_shape_hw": [480, 640], "window": [0., 300.],
                "position_stamp_filter": "scene_text.is_position_stamp", "position_stamps_rejected": True,
                "detections_sha256": _hash("detections"), "K_sha256": K_sha,
                "reader": _reader_proof(), "reader_sidecar_path": "/cache/video_ocr.reader.json",
                "reader_sidecar_sha256": _hash("sidecar"), "model_weight_identity": "explicit_cached_sha256"},
        "inventory": {"city": city, "query": '[out:json];nwr["shop"](48.0000000,8.0000000,49.0000000,9.0000000);out center tags;',
                      "scope": "named_city_bbox", "path": "/output/inventory.json",
                      "cache_dir": "/cache/inventory", "file_sha256": _hash("inventoryfile"),
                      "snapshot_sha256": _hash("inventory"), "response_sha256": _hash("publicresponse"),
                      "bbox_wsen": [8., 48., 9., 49.]},
        "map": {"scope": "whole_city", "city": city, "query": city, "cache_path": "/cache/city.graphml",
                "sha256": _hash("wholecitymap"), "source": "city_cache", "around": None, "bbox": None},
        "source_sha256": {"poi_only.py": _hash("code")}, "evidence_artifacts": [],
        "road_samples": {"spacing_m": 12.5, "count": 1000, "sha256": _hash("roads")},
        "retrieval": {"requested_model": "megaloc", "device": "cpu", "cpu_threads": 4, "n_query": 80,
                      "query_mode": "warp", "appearance": "raw", "poi_loss": "distance_and_bearing"}}
    clusters = [{"poi_ids": [f"a{i}", f"b{i}"], "pois": [{"id": f"a{i}", "lat": 48., "lon": 8. + i}]} for i in range(2)]
    clues = {"selected_answer": None, "inventory": {"snapshot_sha256": _hash("inventory")},
             "detections_sha256": _hash("detections"), "candidate_clusters": clusters}
    plans, regions = [], []
    common = {"backbone": {"name": "megaloc", "hub": "repo@fixed", "device": "cpu"},
              "preprocessing": {"version": 1, "reference_color": "RGB", "reference_size": [512, 512],
                  "reference_resize": "opencv INTER_LINEAR", "scale": "divide by255", "mean": [.485, .456, .406],
                  "std": [.229, .224, .225], "descriptor_normalization": "L2", "dimensions": 8448}, "query_mode": "warp"}
    identity = M._json_sha(common)
    for i in range(2):
        plan = {"hypothesis_id": f"hypothesis_{i}", "origin": "literal_poi_branch_centroid",
                "center": [48., 8. + i], "poi_ids": clusters[i]["poi_ids"], "radius_m": 1000., "cap": 500}
        ids = ["mapillary:shared", f"mapillary:{i}"]
        artifact = {"path": f"poi_evidence/{i}.npz", "sha256": _hash(f"artifact{i}")}
        inputs["evidence_artifacts"].append(artifact)
        rp = {**deepcopy(common), "source": "mapillary", "reference_encoding_identity": "unverified legacy encoding",
              "cpu_runtime": {"device": "cpu", "intra_op_threads": 4, "inter_op_threads": 2, "torch_version": "test"},
              "reference_inventory": {"n_refs": 2, "refs_sha256": _hash(f"snapshot{i}")},
              "matrix_artifact": artifact, "evidence_identity": {
                  "query_indices": [0, 1500, 3000], "query_times": [0., 150., 300.], "ref_ids": ids,
                  "similarities_sha256": _hash(f"matrix{i}"), "ref_latlon_sha256": _hash(f"coords{i}")}}
        plans.append(plan)
        regions.append({**deepcopy(plan), "status": "ok", "reference_count": 2, "query_count": 3,
                        "ref_ids_sha256": M._json_sha(ids), "similarities_sha256": _hash(f"matrix{i}"),
                        "identity_sha256": identity, "new_unique_reference_count": 2 if i == 0 else 1,
                        "duplicate_reference_count": i, "provenance": rp})
    union_ids = ["mapillary:shared", "mapillary:0", "mapillary:1"]
    union = {"ids": union_ids, "count": 3, "ids_sha256": M._json_sha(union_ids),
             "ref_latlon_sha256": _hash("unioncoords"), "similarities_sha256": _hash("unionmatrix"),
             "query_indices": [0, 1500, 3000], "query_times": [0., 150., 300.], "identity_sha256": identity}
    ap = {"algorithm": "timed-poi-similarity-v1", "config": fixed,
          "vo_sha256": inputs["vo"]["trajectory_sha256"], "times_sha256": inputs["vo"]["timestamps_sha256"],
          "intrinsics_sha256": K_sha, **{key: _hash(key) for key in ("observations_sha256", "pois_sha256", "hypotheses_sha256", "query_times_sha256")},
          "candidate_clusters_sha256": M._json_sha(clusters, ascii=True),
          "literal_clues_sha256": M._json_sha({k: v for k, v in clues.items() if k != "candidate_clusters"}, ascii=True),
          "start_samples_sha256": _hash("starts"),
          "appearance": {"values_sha256": _hash("appearance"), "query_times_sha256": _hash("query_times_sha256"),
                         "sources": {"reference_mean": False, "reference_count": 3,
                                     "similarities_sha256": _hash("unionmatrix"), "config": fixed}},
          "road_distance": {"values_sha256": _hash("roadfield"), "sources": {"road_samples_sha256": _hash("roads")}}}
    query_sha = M._float64_array_sha([0., 150., 300.], (3,))
    ap["query_times_sha256"] = query_sha
    ap["appearance"]["query_times_sha256"] = query_sha
    audit = {"status": "ok", "failed_regions": [], "region_plans": plans, "regions": regions, "clues": clues,
             "reference_union": union, "provenance": {"config": {"radius_m": 1000., "cap_per_branch": 500, "n_query": 80},
                "fixed_alignment_config": fixed, "query_cache_shared": True, "region_plans_sha256": M._json_sha(plans),
                "road_samples_sha256": _hash("roads")},
             "alignment": {"hypothesis_ids": ["hypothesis_0", "hypothesis_1"], "selected_hypothesis_id": "hypothesis_1",
                           "search": {"hypothesis_count": 2, "coarse_pose_count": 2160, "coarse_hypothesis_scores": 4320,
                                      "fine_pose_count": 100, "query_count": 3, "start_mode": "provided"}, "provenance": ap},
             "grid": {"start_policy": "nearest road sample per fixed 100m bin; uniform cap over all-city bins",
                      "start_samples_sha256": _hash("starts"), "start_count": 10}}
    return cfg, inputs, audit


def _valid_poi(repeats=1):
    summary, manifest = _valid(repeats=repeats)
    for rep in summary["reps"]:
        cfg, inputs, audit = _poi_contract("Registered city")
        rep["provenance"] = {"config": cfg, "poi_inputs": inputs, "poi_localization_audit": audit}
        rep["scores"]["answer"]["meta"] = {"mode": "poi", "method": "poi_only"}
    return summary, manifest


def test_complete_poi_answers_keep_full_target_and_allow_public_reference_positions():
    summary, manifest = _valid_poi(repeats=2)
    report = M.check_goal(summary, manifest, min_repeats=2)
    assert report["goal_met"], json.dumps(report, indent=2)
    assert report["n_passing"] == 15 and "street-photo positions are permitted" in report["input_contract_notes"][1]


def _rehash_reader(reader):
    reader["spec_sha256"] = M._json_sha({key: reader[key] for key in
        ("backend", "parameters", "model_assets", "downloads_disabled")})


@pytest.mark.parametrize("change", ["missing_reader", "missing_asset", "asset_hash", "asset_size", "model_path",
    "spec_hash", "downloads", "missing_session", "gpu_provider", "actual_threads", "declared_threads",
    "actual_batches", "declared_batches", "missing_calls", "cache_flag", "sidecar_hash", "weight_identity"])
def test_poi_requires_actual_ocr_assets_sessions_and_bounded_cpu_runtime(change):
    summary, manifest = _valid_poi()
    ocr = summary["reps"][0]["provenance"]["poi_inputs"]["ocr"]
    reader = ocr["reader"]
    if change == "missing_reader": ocr.pop("reader")
    elif change == "missing_asset": reader["model_assets"].pop("Rec")
    elif change == "asset_hash": reader["model_assets"]["Det"]["sha256"] = "unknown"
    elif change == "asset_size": reader["model_assets"]["Cls"]["bytes"] = 0
    elif change == "model_path": reader["parameters"]["Rec.model_path"] = "/other/model.onnx"
    elif change == "spec_hash": reader["spec_sha256"] = _hash("stale spec")
    elif change == "downloads": reader["downloads_disabled"] = False
    elif change == "missing_session": reader["actual_sessions"].pop("text_rec")
    elif change == "gpu_provider": reader["actual_sessions"]["text_det"]["providers"] = ["CUDAExecutionProvider"]
    elif change == "actual_threads": reader["actual_sessions"]["text_rec"]["intra_op_num_threads"] = 8
    elif change == "declared_threads": reader["parameters"]["EngineConfig.onnxruntime.inter_op_num_threads"] = 8
    elif change == "actual_batches": reader["actual_batches"]["rec_batch_num"] = 16
    elif change == "declared_batches": reader["parameters"]["Cls.cls_batch_num"] = 16
    elif change == "missing_calls": reader.pop("read_calls")
    elif change == "cache_flag": reader["cache_hit"] = "unknown"
    elif change == "sidecar_hash": ocr.pop("reader_sidecar_sha256")
    elif change == "weight_identity": ocr.pop("model_weight_identity")
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["n_passing"] == 14


@pytest.mark.parametrize("change", ["missing", "device", "intra", "inter", "version", "declared"])
def test_poi_requires_measured_cpu_four_thread_retrieval_runtime(change):
    summary, manifest = _valid_poi()
    prov = summary["reps"][0]["provenance"]
    region = prov["poi_localization_audit"]["regions"][0]["provenance"]
    runtime = region["cpu_runtime"]
    if change == "missing": region.pop("cpu_runtime")
    elif change == "device": runtime["device"] = "cuda"
    elif change == "intra": runtime["intra_op_threads"] = 16
    elif change == "inter": runtime["inter_op_threads"] = 0
    elif change == "version": runtime.pop("torch_version")
    elif change == "declared": prov["poi_inputs"]["retrieval"].pop("cpu_threads")
    assert not M.check_goal(summary, manifest)["goal_met"]


def test_identical_ocr_asset_copies_and_warm_hits_are_semantically_comparable():
    summary, manifest = _valid_poi()
    specs = set()
    for index, rep in enumerate(summary["reps"]):
        ocr = rep["provenance"]["poi_inputs"]["ocr"]
        ocr["reader"] = _reader_proof(f"E:/copied-environment-{index}/models")
        ocr["reader"]["cache_hit"] = bool(index % 2)
        specs.add(ocr["reader"]["spec_sha256"])
    assert len(specs) == 15  # Full specs honestly include the different locations.
    report = M.check_goal(summary, manifest)
    assert report["goal_met"], report


def test_stale_ocr_copy_paths_fail_unless_bound_to_the_current_spec():
    summary, manifest = _valid_poi()
    reader = summary["reps"][0]["provenance"]["poi_inputs"]["ocr"]["reader"]
    for task, asset in reader["model_assets"].items():
        asset["path"] = f"E:/new-copy/models/{asset['path'].split('/')[-1]}"
        reader["parameters"][f"{task}.model_path"] = asset["path"]
    assert not M.check_goal(summary, manifest)["goal_met"]
    _rehash_reader(reader)
    assert M.check_goal(summary, manifest)["goal_met"]


def test_ocr_model_weight_changes_fail_even_with_an_internally_valid_new_spec():
    summary, manifest = _valid_poi()
    reader = summary["reps"][0]["provenance"]["poi_inputs"]["ocr"]["reader"]
    reader["model_assets"]["Rec"]["sha256"] = _hash("different-recognition-weights")
    _rehash_reader(reader)
    report = M.check_goal(summary, manifest)
    assert report["n_passing"] == 15 and not report["goal_met"]
    assert "semantic OCR" in " ".join(report["reasons"])


def test_cpu_runtime_must_match_across_regions_and_target_clips():
    summary, manifest = _valid_poi()
    regions = summary["reps"][0]["provenance"]["poi_localization_audit"]["regions"]
    regions[0]["provenance"]["cpu_runtime"]["inter_op_threads"] = 1
    assert not M.check_goal(summary, manifest)["goal_met"]
    regions[1]["provenance"]["cpu_runtime"]["inter_op_threads"] = 1
    report = M.check_goal(summary, manifest)
    assert report["n_passing"] == 15 and not report["goal_met"]
    assert "CPU retrieval runtime identities" in " ".join(report["reasons"])


@pytest.mark.parametrize("field", ["calibration", "raw_images", "sensor_timestamps", "gps_or_oxts", "ground_truth",
    "manual_region_coordinates", "external_position_prior", "external_trajectory", "external_ocr_video",
    "video_ocr", "public_poi_inventory", "public_reference_positions", "position_stamps_rejected", "retrieval"])
def test_poi_rejects_unknown_input_policy_and_query_side_external_priors(field):
    summary, manifest = _valid_poi()
    policy = summary["reps"][0]["provenance"]["poi_inputs"]["input_policy"]
    policy[field] = not policy[field]
    assert not M.check_goal(summary, manifest)["goal_met"]
    policy.pop(field)
    assert not M.check_goal(summary, manifest)["goal_met"]


@pytest.mark.parametrize("change", ["partial", "failed_region", "failed_list", "missing_audit", "omit_branch",
    "cap", "manual_origin", "plan_hash", "matrix_hash", "coordinate_hash", "query_grid", "duplicate_count",
    "reference_union", "model", "encoding_unknown_missing", "public_id", "matrix_artifact", "selected_branch",
    "core_config", "reference_mean", "core_vo", "core_ocr", "core_inventory", "core_starts", "core_queries", "search_counts"])
def test_poi_rejects_partial_branches_incomparable_evidence_and_stale_alignment(change):
    summary, manifest = _valid_poi()
    prov = summary["reps"][0]["provenance"]
    audit = prov["poi_localization_audit"]
    region = audit["regions"][1]
    rp = region["provenance"]
    ap = audit["alignment"]["provenance"]
    if change == "partial": audit["status"] = "inactive"
    elif change == "failed_region": region["status"] = "failed"
    elif change == "failed_list": audit["failed_regions"] = ["hypothesis_1"]
    elif change == "missing_audit": prov.pop("poi_localization_audit")
    elif change == "omit_branch": audit["regions"].pop()
    elif change == "cap": region["cap"] = 750
    elif change == "manual_origin": audit["region_plans"][1]["origin"] = "manual_disc"
    elif change == "plan_hash": audit["provenance"]["region_plans_sha256"] = _hash("other")
    elif change == "matrix_hash": rp["evidence_identity"]["similarities_sha256"] = _hash("other")
    elif change == "coordinate_hash": rp["evidence_identity"].pop("ref_latlon_sha256")
    elif change == "query_grid": rp["evidence_identity"]["query_times"][1] = 155.
    elif change == "duplicate_count": region["duplicate_reference_count"] = 0
    elif change == "reference_union": audit["reference_union"]["ids"] = ["mapillary:shared"]
    elif change == "model": rp["backbone"]["hub"] = "other"
    elif change == "encoding_unknown_missing": rp.pop("reference_encoding_identity")
    elif change == "public_id": rp["evidence_identity"]["ref_ids"][0] = "query_gps:0"
    elif change == "matrix_artifact": prov["poi_inputs"]["evidence_artifacts"] = []
    elif change == "selected_branch": audit["alignment"]["selected_hypothesis_id"] = "omitted_branch"
    elif change == "core_config": ap["config"]["poi_weight"] = 10
    elif change == "reference_mean": ap["appearance"]["sources"]["reference_mean"] = True
    elif change == "core_vo": ap["vo_sha256"] = _hash("other")
    elif change == "core_ocr": prov["poi_inputs"]["ocr"]["detections_sha256"] = _hash("other")
    elif change == "core_inventory": prov["poi_inputs"]["inventory"]["snapshot_sha256"] = _hash("other")
    elif change == "core_starts": audit["grid"]["start_policy"] = "POI local starts"
    elif change == "core_queries": ap["query_times_sha256"] = _hash("different query rows")
    elif change == "search_counts": audit["alignment"]["search"]["coarse_pose_count"] = 1
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["n_passing"] == 14


@pytest.mark.parametrize("change", ["mode", "method", "city", "disc", "length", "external_vo", "external_ocr",
    "truncated", "resize", "camera", "vo_signature", "ocr_source", "ocr_stamps", "ocr_native", "ocr_window",
    "inventory_scope", "inventory_file", "inventory_query", "map_scope", "map_crop", "map_hash", "sources",
    "retrieval_profile", "model_device", "road_samples", "ocr_K"])
def test_poi_rejects_invalid_acquisition_and_configuration(change):
    summary, manifest = _valid_poi()
    rep = summary["reps"][0]
    cfg, inputs = rep["provenance"]["config"], rep["provenance"]["poi_inputs"]
    if change == "mode": cfg["answer"] = "track"
    elif change == "method": rep["scores"]["answer"]["meta"]["method"] = "vpr_track_only"
    elif change == "city": cfg["city"] = "Other city"
    elif change == "disc": cfg.update(osm_around=[48.4, 9.9, 800], osm_around_source="video_title")
    elif change == "length": cfg["estimated_length_m"] = 1000
    elif change == "external_vo": cfg["openvo_trajectory_path"] = "external.npz"
    elif change == "external_ocr": cfg["ocr_video_path"] = "another.mp4"
    elif change == "truncated": inputs["extraction"]["max_frames"] = 50
    elif change == "resize": inputs["extraction"]["frame_max_width"] = 640
    elif change == "camera": inputs["camera"]["K"][0][0] *= 2
    elif change == "vo_signature": inputs["vo"].pop("input_signature_sha256")
    elif change == "ocr_source": inputs["ocr"]["video_sha256"] = _hash("another video")
    elif change == "ocr_stamps": inputs["ocr"]["position_stamps_rejected"] = False
    elif change == "ocr_native": inputs["ocr"]["native_bbox_grid"] = False
    elif change == "ocr_window": inputs["ocr"]["window"] = [100., 200.]
    elif change == "inventory_scope": inputs["inventory"]["scope"] = "clip_bbox"
    elif change == "inventory_file": inputs["inventory"].pop("file_sha256")
    elif change == "inventory_query": inputs["inventory"]["query"] = "manually narrowed store lookup"
    elif change == "map_scope": inputs["map"]["scope"] = "clip_bbox"
    elif change == "map_crop": inputs["map"]["around"] = [48., 8., 1000]
    elif change == "map_hash": inputs["map"].pop("sha256")
    elif change == "sources": inputs.pop("source_sha256")
    elif change == "retrieval_profile": inputs["retrieval"]["appearance"] = "reference_mean"
    elif change == "model_device": inputs["retrieval"]["device"] = "cuda"
    elif change == "road_samples": inputs["road_samples"]["sha256"] = _hash("other roads")
    elif change == "ocr_K": inputs["ocr"]["K_sha256"] = _hash("other K")
    assert not M.check_goal(summary, manifest)["goal_met"]


@pytest.mark.parametrize("change", ["missing_clip", "failed_rep", "oracle", "dirty", "incomplete_plan", "mixed_config", "mixed_mode"])
def test_poi_mode_cannot_relax_existing_fifteen_clip_and_provenance_gates(change):
    summary, manifest = _valid_poi()
    if change == "missing_clip": summary["clips"].pop("clip14")
    elif change == "failed_rep": summary["reps"][0]["rc"] = 1
    elif change == "oracle": summary["reps"][0]["scores"]["answer"]["oracle"] = True
    elif change == "dirty": summary["provenance"]["dirty"] = ["True"]
    elif change == "incomplete_plan": summary["execution"]["complete"] = False
    elif change == "mixed_config": summary["provenance"]["method_config_fingerprints"].append("other")
    elif change == "mixed_mode": summary["reps"][0]["scores"]["answer"]["meta"]["mode"] = "track"
    assert not M.check_goal(summary, manifest)["goal_met"]


def test_stale_legacy_mode_cannot_bypass_poi_audit_for_all_clips():
    summary, manifest = _valid_poi()
    for rep in summary["reps"]:
        rep["scores"]["answer"]["meta"]["mode"] = "track"
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["n_passing"] == 0
    assert "mode disagrees" in " ".join(report["clips"][0]["reasons"])


def test_stale_legacy_mode_cannot_bypass_existing_filter_audit():
    from tests.test_check_dev_goal import _valid_filter

    summary, manifest = _valid_filter()
    for rep in summary["reps"]:
        rep["scores"]["answer"]["meta"]["mode"] = "track"
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["n_passing"] == 0


def test_poi_rejects_model_changes_between_clips_even_when_each_clip_is_coherent():
    summary, manifest = _valid_poi()
    audit = summary["reps"][0]["provenance"]["poi_localization_audit"]
    for region in audit["regions"]:
        rp = region["provenance"]
        rp["backbone"]["hub"] = "different@fixed"
        region["identity_sha256"] = M._json_sha({key: rp[key] for key in ("backbone", "preprocessing", "query_mode")})
    audit["reference_union"]["identity_sha256"] = audit["regions"][0]["identity_sha256"]
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"] and report["n_passing"] == 15
    assert "different query/backbone/preprocessing" in " ".join(report["reasons"])


def test_poi_rejects_changed_database_inputs_between_repeats():
    summary, manifest = _valid_poi(repeats=2)
    summary["reps"][1]["provenance"]["poi_inputs"]["map"]["sha256"] = _hash("new-map-epoch")
    report = M.check_goal(summary, manifest)
    assert not report["goal_met"]
    assert "identities change" in " ".join(report["clips"][0]["reasons"])


def test_explicit_poi_databases_remain_visible_and_bound_to_actual_files():
    summary, manifest = _valid_poi()
    for rep in summary["reps"]:
        cfg, inputs = rep["provenance"]["config"], rep["provenance"]["poi_inputs"]
        cfg["poi_map_path"] = inputs["map"]["cache_path"]
        cfg["poi_inventory_dir"] = inputs["inventory"]["cache_dir"]
        inputs["map"]["source"] = "explicit_cache"
    assert M.check_goal(summary, manifest)["goal_met"]
    summary["reps"][0]["provenance"]["config"]["poi_map_path"] = "/maps/other.graphml"
    assert not M.check_goal(summary, manifest)["goal_met"]


def test_actual_aggregate_preserves_poi_audit_and_reports_selected_primary(monkeypatch, tmp_path):
    from tests.test_benchmark import M as benchmark, _fake_result, _setup_root, _write_rep

    root, manifest = _setup_root(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(benchmark, "package_tree", lambda *a, **k: "known-tree")
    plan = {}
    for entry, shift in zip(manifest["clips"], (0., 5000., 10000.)):
        result = _fake_result(shift, 20.)
        route = result.pop("position")["route_latlon"]
        result.pop("matcher_position")
        result.pop("vpr_prior")
        cfg, inputs, audit = _poi_contract(entry["city"])
        cfg["use_vpr_prior"] = True  # Ordinary option remains present but inactive in this mode.
        result["provenance"]["config"].update(cfg)
        result["provenance"].update(poi_inputs=inputs, poi_localization_audit=audit)
        result.update(n_frames=3001, n_valid_poses=2900,
            poi_position={"source": "experimental_poi_alignment", "route_latlon": route},
            answer={"mode": "poi", "method": "poi_only", "source": "result.poi_position", "route_latlon": route},
            channels={"vpr": {"status": "disabled", "reason": "poi_only_execution"},
                      "poi_retrieval": {"status": "ok"}, "poi": {"status": "ok"}})
        _write_rep(run, entry["id"], 0, result)
        status_path = run / entry["id"] / "rep0/status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["vo_segment"] = entry["vo_segment"]
        status_path.write_text(json.dumps(status), encoding="utf-8")
        plan[entry["id"]] = {"reps": [0]}
    (run / "run_config.json").write_text(json.dumps({"execution_plan": plan, "config": {"pipeline_args": []}}), encoding="utf-8")
    summary = benchmark.aggregate(run, manifest, root=root, allow_network=False, n_resamples=99)
    report = M.check_goal(summary, manifest, expected_count=3)
    assert report["goal_met"], json.dumps(report, indent=2)
    health = summary["reps"][0]["health"]
    assert health["vpr_status"] == "disabled" and health["poi_retrieval_status"] == "ok"
    assert health["poi_source"] == "mapillary" and health["poi_device"] == "cpu"
    assert summary["clips"]["clipA"]["health"]["n_degraded"] == 0
    methods = summary["fleets"]["core"]["methods"]
    assert {"answer", "poi_only"} <= set(methods)
    assert not {"pipeline", "matcher_only", "vpr_track_only"} & set(methods)
    markdown = benchmark.render_markdown(summary)
    assert "Answer: symmetric" in markdown and "Experimental POI alignment" in markdown
    # Database overrides are method inputs, not ignored per-clip paths.
    result["provenance"]["config"]["poi_map_path"] = "/maps/override.graphml"
    (run / entry["id"] / "rep0/result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(SystemExit, match="different effective method configurations"):
        benchmark.aggregate(run, manifest, root=root, allow_network=False, n_resamples=99)


def test_partial_poi_health_does_not_silently_label_a_primary_answer_healthy():
    from tests.test_benchmark import M as benchmark

    cfg, inputs, audit = _poi_contract("City")
    audit["regions"][0]["status"] = "failed"
    result = {"provenance": {"config": cfg, "poi_inputs": inputs, "poi_localization_audit": audit},
              "answer": {"mode": "poi"}, "channels": {"poi_retrieval": {"status": "ok"}}}
    health = benchmark._health_of(result)
    assert not health["poi_only"]
    assert "POI localization is partial" in " ".join(benchmark.degraded_reasons(health, token_set=True))


def test_missing_region_audit_is_reportable_failure_metadata():
    from tests.test_benchmark import M as benchmark

    cfg, inputs, audit = _poi_contract("City")
    audit.update(status="inactive", regions=None)
    result = {"provenance": {"config": cfg, "poi_inputs": inputs, "poi_localization_audit": audit}}
    provenance = benchmark._provenance_of(result)
    health = benchmark._health_of(result)
    assert provenance["vpr_model"] is None and health["poi_complete"] is False
    assert "partial" in " ".join(benchmark.degraded_reasons(health, token_set=True))


def test_real_poi_stage_and_core_emit_a_goal_auditable_schema_with_mocked_acquisition(tmp_path, monkeypatch):
    """Exercise actual artifact/hash assembly, without OCR/retrieval/VO models."""
    import monocular_osm.pipeline as pipeline
    from monocular_osm import kartaview_vpr as kv, poi_clues, poi_localization
    from tests.test_pipeline_poi import _setup

    real_align = poi_localization.core.align_candidate_clusters
    cfg, _frames, _calls, _caches, _ocr = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(poi_localization.core, "align_candidate_clusters", real_align)
    original_inventory = poi_clues.fetch_city_retail
    def inventory(city, directory):
        value = original_inventory(city, directory)
        value["snapshot_sha256"] = poi_clues._digest(value["pois"])
        west, south, east, north = value["bbox_wsen"]
        value["query"] = f'[out:json];nwr["shop"]({south:.7f},{west:.7f},{north:.7f},{east:.7f});out center tags;'
        return value
    monkeypatch.setattr(poi_clues, "fetch_city_retail", inventory)
    original_evidence = kv.kartaview_vpr_evidence
    def evidence(*args, **kwargs):
        raw = original_evidence(*args, **kwargs)
        _cfg, _inputs, template = _poi_contract(cfg.city)
        prototype = template["regions"][0]["provenance"]
        raw["provenance"].update({key: deepcopy(prototype[key]) for key in
                                 ("backbone", "preprocessing", "reference_encoding_identity")})
        raw["provenance"]["reference_inventory"] = kv._reference_snapshot([
            {"id": key, "lat": ll[0], "lon": ll[1]} for key, ll in zip(raw["ref_ids"], raw["ref_latlon"])])
        return raw
    monkeypatch.setattr(kv, "kartaview_vpr_evidence", evidence)
    result = pipeline.run_pipeline(cfg)
    score = {"meta": {"mode": "poi", "method": "poi_only"}}
    issues = M._poi_answer_issues({"provenance": result["provenance"]}, score, cfg.city, "0:11")
    assert not issues, issues
    assert result["answer"]["mode"] == "poi"
