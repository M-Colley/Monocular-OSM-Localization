"""Lazy audit of frozen assisted evidence; no models, VO or labels are run.

Legacy goal checks stay dependency-free. Assisted checks additionally bind
JSON/NPZ evidence and replay the fixed policy. Public reference/POI positions
are permitted; they are not query-video GPS or ground truth.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_string(value):
    return isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relative(path):
    path = Path(str(path).replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts, "escaping relative artifact path")
    return path


def load_json(path):
    require(Path(path).stat().st_size <= 64 * 1024**2, "JSON artifact exceeds the audit budget")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    """Check NPY dimensions before allocating any array; pickle is forbidden."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 64, "NPZ contains too many arrays")
        require(len(archive.namelist()) == len(set(archive.namelist())), "NPZ contains duplicate arrays")
        require(sum(item.file_size for item in archive.infolist()) <= 64 * 1024**2,
                "NPZ exceeds its uncompressed audit budget")
        for item in archive.infolist():
            require(item.filename.endswith(".npy") and "/" not in item.filename, "unexpected NPZ member")
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY format")
                require(not dtype.hasobject and len(shape) <= 3 and all(x >= 0 for x in shape)
                        and math.prod(shape) * dtype.itemsize <= 64 * 1024**2, "unsafe NPY array")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


class Files:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)

    def path(self, path):
        value = Path(path).resolve(strict=True)
        require(value.is_relative_to(self.root), "artifact escapes the parent output")
        return value

    def record(self, record, ledger=None):
        require(isinstance(record, dict) and sha_string(record.get("sha256")), "artifact lacks a digest")
        path = self.path(record["path"])
        require(file_sha(path) == record["sha256"], "frozen artifact content changed")
        if ledger is not None:
            require(ledger.get(path) == record["sha256"], "artifact is absent from its child freeze")
        return path

    def freeze(self, path):
        path = self.path(path)
        value = load_json(path)
        require(value.get("scoring_has_not_run") is True, "artifact was not frozen before scoring")
        ledger = {}
        for name, expected in value["artifacts"].items():
            target = self.path(path.parent / relative(name))
            require(target.is_relative_to(path.parent) and sha_string(expected) and file_sha(target) == expected,
                    "child freeze contains an escaping or changed artifact")
            ledger[target] = expected
        return value, ledger


def sources(files, root, declared):
    require(isinstance(declared, dict) and bool(declared), "source identity is missing")
    for name, expected in declared.items():
        path = files.path(Path(root) / "source_snapshot" / relative(name))
        require(sha_string(expected) and file_sha(path) == expected, "source snapshot hash differs")


def comparison(files, value, ledger, window, *, require_all=False):
    """Bind decoded statistics to exact raw matrices and actual audit rows."""
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha, _valid_audit
    artifacts = value.get("audit_artifacts") or []
    if not artifacts and value.get("unavailable_reason"):
        require(not value.get("decoded_reference_cosines"), "unavailable comparison carries promotable scores")
        require(not require_all, "complete visual candidate lacks actual reference audits")
        return []
    require(artifacts, "retrieval comparison lacks frozen full-matrix audits")
    records = {files.record(record, ledger): record for record in artifacts}
    require(len(records) == len(artifacts), "retrieval audit repeats an artifact instead of an outcome")
    metas = sorted((p for p in records if p.suffix == ".json"), key=lambda p: int(p.stem))
    require(len(metas) * 2 == len(records), "retrieval metadata/matrix pairs are incomplete")
    audits = []
    for index, path in enumerate(metas):
        require(int(path.stem) == index, "retrieval audit ordinals are incomplete")
        meta = load_json(path)
        matrix_record = meta.pop("array_artifact")
        matrix_path = files.record(matrix_record, ledger)
        require(matrix_path in records and matrix_path.stem == path.stem, "audit matrix pair is stale")
        arrays = load_npz(matrix_path)
        declared = matrix_record["arrays"]
        require(set(declared) == set(arrays), "audit array inventory is incomplete")
        for name, array in arrays.items():
            require(declared[name] == {"sha256": _array_sha(array), "shape": list(array.shape), "dtype": array.dtype.str},
                    "audit array identity is stale")
        returned = {name.removeprefix("returned_"): array for name, array in arrays.items()
                    if name.startswith("returned_")}
        meta.update({name: array for name, array in arrays.items() if not name.startswith("returned_")})
        if returned:
            meta["returned_track"] = returned
        if _valid_audit(meta):
            require(len(meta["query_indices"]) <= 80 and meta["raw_similarities"].size <= 1_600_000,
                    "comparison exceeds its query/matrix budget")
            refs = meta["ref_latlon"]
            require(np.all(np.abs(refs[:, 0]) <= 90) and np.all(np.abs(refs[:, 1]) <= 180),
                    "reference positions are outside WGS84")
            require(meta["query_times"][0] >= window[0] - 1
                    and (window[1] is None or meta["query_times"][-1] <= window[1] + 1),
                    "retrieval queries are outside the registered video window")
        if require_all and not _valid_audit(meta):
            request = meta.get("request") or {}
            terminal_search({"status": "unavailable", "center": request.get("center"),
                "radius_m": request.get("radius_m"), "ref_cap": request.get("cap"),
                "track_decoding": {"terminal_search": meta.get("terminal_search")}})
        audits.append(meta)
    if require_all:
        successful = [a for a in audits if _valid_audit(a)]
        identities = {digest({key: a[key] for key in ("query_pixels_sha256", "query_descriptors_sha256",
            "actual_model_state_sha256", "backbone", "preprocessing", "query_mode")}
            | {"indices": _array_sha(a["query_indices"]), "times": _array_sha(a["query_times"])}) for a in successful}
        require(len(identities) <= 1, "planned pools used different actual query/model/preprocessing identities")
    if value.get("unavailable_reason"):
        require(not value.get("decoded_reference_cosines"), "unavailable comparison carries promotable scores")
        return audits
    selected = value.get("selected_audit_rows") or []
    require(selected, "comparison has no decoded-reference row selection")
    used, cosines, coords, ids = [], [], [], []
    for position, pair in enumerate(selected):
        require(isinstance(pair, list) and len(pair) == 2 and all(type(i) is int and i >= 0 for i in pair),
                "invalid selected audit row")
        number, row = pair
        require(number < len(audits), "selected audit is missing")
        audit = audits[number]
        require(_valid_audit(audit) and row < len(audit["query_indices"]), "selected audit is inconsistent")
        require(int(audit["query_indices"][row]) == value["query"]["indices"][position],
                "selected row belongs to a different query grid")
        used.append(audit)
        chosen = int(audit["decoded_ref_indices"][row])
        cosines.append(float(audit["raw_similarities"][row, chosen]))
        coords.append(audit["ref_latlon"][chosen].tolist())
        ids.append(str(audit["ref_ids"][chosen]))
    require(len({a["comparison_identity_sha256"] for a in used}) == 1,
            "comparison combines different query/model/decoder identities")
    first = used[0]
    decoder = first["decoder"]
    expected = {"statistic": first["score_kind"],
        "model": {**first["backbone"], "weights_sha256": first["actual_model_state_sha256"]},
        "query": {"indices": first["query_indices"].tolist(), "times": first["query_times"].tolist(),
                  "mode": first["query_mode"], "preprocessing": first["preprocessing"],
                  "pixels_sha256": first["query_pixels_sha256"], "descriptors_sha256": first["query_descriptors_sha256"]},
        "inner_decoder": {"actual": decoder["actual"], "fallback": bool(decoder.get("fallback_reason")),
                          "settings": {k: v for k, v in decoder.items() if k not in {"actual", "fallback_reason"}}},
        "audited_identity": first["comparison_identity"], "audited_identity_sha256": first["comparison_identity_sha256"],
        "decoded_reference_cosines": cosines, "decoded_reference_latlon": coords, "decoded_reference_ids": ids}
    require(len(selected) == len(expected["query"]["indices"])
            and all(value.get(key) == expected_value for key, expected_value in expected.items()),
            "comparison differs from its exact frozen query/matrix evidence")
    sampling = value.get("source_sampling") or {}
    count = sampling.get("dense_count")
    require(sampling.get("mode") == "linspace" and type(count) is int and count >= len(selected)
            and sampling.get("requested_sample_count") == 80 and sampling.get("sample_count") == len(selected)
            and len(selected) == min(80, count)
            and sampling.get("dense_indices") == np.linspace(0, count - 1, min(80, count)).astype(int).tolist()
            and expected["query"]["indices"] == list(range(len(selected))), "comparison omitted part of the declared input grid")
    return audits


def verify_child(files, record, inputs, before):
    """Bind transport, nested files, source/video/window and selected route."""
    from monocular_osm.assisted_worker import _from_snapshot, child_config
    from monocular_osm.provenance import config_snapshot
    from monocular_osm.stages.assisted import child_record
    kind = record["kind"]
    require(kind in {"baseline", "cues", "poi_joint", "poi_image_track", "metric_pf", "expanded"}, "unknown reached stage")
    roles = {item["role"]: item for item in record.get("artifacts") or []}
    if not roles:
        require(record.get("status") == "failed" and bool(record.get("error")), "unfrozen child lacks a terminal failure")
        child = {"status": "failed", "error": record["error"]}
        require(child_record(kind, child, files.root, expected=inputs) == record, "failed child summary is stale")
        return child, {}
    require(set(roles) == {"result_artifact", "candidate_artifact", "frozen_artifact"}, "incomplete child artifact roles")
    base = files.root / "children" / inputs["execution_id"] / kind
    for role, name in (("candidate_artifact", "candidate.json"), ("result_artifact", "result.json"),
                       ("frozen_artifact", "child_frozen.json")):
        require(files.record(roles[role]) == base / name, "child belongs to a different execution")
    candidate = load_json(base / "candidate.json")
    child = {**candidate, **roles, "worker_returncode": record.get("worker_returncode"), "status": record.get("status")}
    if record.get("reported_status") is not None:
        child.update(reported_status=record["reported_status"], transport_error=record.get("error"))
    require(child_record(kind, child, files.root, expected=inputs) == record, "child record differs from frozen evidence")
    _, ledger = files.freeze(base / "child_frozen.json")
    request_path = base / "request.json"
    require(request_path in ledger, "child freeze omits its requested inputs")
    request = load_json(request_path)
    require(request["kind"] == kind and request["video_sha256"] == inputs["video"]["sha256"]
            and request["source_sha256"] == inputs["source_sha256"]
            and request["policy_evidence"] == before and Path(request["parent_output"]).resolve() == files.root,
            "child request differs from its parent video/source/decision evidence")
    require(candidate.get("kind") == kind, "candidate identifies another stage")
    sources(files, base, candidate["source_sha256"])
    cfg, parent_cfg = candidate["config"], inputs["config"]
    requested = deepcopy(request["config"])
    if requested.get("frame_stride") is None:
        requested["frame_stride"] = cfg.get("frame_stride")
    require(requested == cfg, "child execution changed its requested configuration")
    expected = child_config(_from_snapshot(parent_cfg), kind, parent_cfg.get("video_path") or inputs["video"]["path"])
    expected.output_dir = Path(cfg["output_dir"])
    require(expected.output_dir.resolve() == base, "child effective output identifies a different execution")
    if parent_cfg.get("frame_stride") is None or kind == "metric_pf":
        expected.frame_stride = cfg.get("frame_stride")
    require(config_snapshot(expected) == cfg, "child changed its component/baseline configuration")
    result = load_json(base / "result.json")
    if "candidate" in result:
        require(result["candidate"] == candidate, "component result differs from frozen candidate")
    elif candidate.get("route_latlon") is not None:
        require((result.get("answer") or {}).get("route_latlon") == candidate["route_latlon"], "child selected route is stale")
    if kind in {"baseline", "expanded"} and child.get("status") == "complete" and not (child.get("comparison") or {}).get("unavailable_reason"):
        retrieval_route(candidate, result)
    return child, ledger


def retrieval_route(child, physical):
    """Retain the ordinary saved six-decimal track and median-five semantics."""
    from monocular_osm.baselines import vpr_track_only
    value = child["comparison"]
    expected = [[round(float(t), 2), round(float(ll[0]), 6), round(float(ll[1]), 6), round(float(c), 4)]
                for t, ll, c in zip(value["query"]["times"], value["decoded_reference_latlon"],
                                    value["decoded_reference_cosines"])]
    require((physical.get("vpr_prior") or {}).get("track") == expected,
            "saved ordinary retrieval track differs from its raw decoded evidence")
    prediction = vpr_track_only(physical)
    require(prediction is not None and prediction.route_latlon == child["route_latlon"]
            and child.get("route_timestamps") == value["query"]["times"],
            "retrieval route differs from the ordinary median-five selected track")


def metric_package(files, prepared, ledger, inputs, auto_quality=None):
    """Recompute native validity/plane quality from frozen measured arrays."""
    import numpy as np
    from monocular_osm.assisted_motion import PROFILE, PROFILE_SHA256
    from monocular_osm.assisted_policy import CONFIG_SHA256, policy_config
    from monocular_osm.assisted_worker import _array_sha
    require(prepared.get("profile_sha256") == PROFILE_SHA256
            and prepared.get("policy_config_sha256") == CONFIG_SHA256, "native metric profile changed")
    freeze = files.record(prepared["artifacts"]["freeze"], ledger)
    frozen = load_json(freeze)
    require(frozen.get("scoring_has_not_run") is True and frozen.get("profile_sha256") == PROFILE_SHA256
            and frozen.get("policy_config_sha256") == CONFIG_SHA256, "native metric package lacks its fixed inference freeze")
    for record in frozen["artifacts"].values():
        files.record(record, ledger)
    registration = load_json(freeze.parent / "metric_registration.json")
    summary = load_json(freeze.parent / "metric_summary.json")
    require(summary == prepared["summary"] and registration["window"] == inputs["registered_window"]
            and registration["video"]["sha256"] == inputs["video"]["sha256"]
            and registration["profile"] == dict(PROFILE), "native metric identity or summary is stale")
    require(all(value is False for value in registration["input_policy"].values())
            and set(registration["input_policy"]) == {"ground_truth", "external_trajectory", "calibration",
                "gps_or_oxts", "position_prior", "filter_execution"}, "native preflight used forbidden inputs")
    require(frozen["source_sha256"] == registration["source_sha256"] == prepared["source_sha256"],
            "native metric source declaration is stale")
    sources(files, freeze.parent, registration["source_sha256"])
    parent_sources = {str(k).replace("\\", "/"): v for k, v in inputs["source_sha256"].items()}
    require(all(parent_sources.get("monocular_osm/" + name) == sha for name, sha in registration["source_sha256"].items()),
            "native metric source differs from the parent source")
    if auto_quality is not None:
        require(registration["auto_quality"] == summary["auto_quality"] == auto_quality,
                "native metric preflight belongs to a different auto-VO measurement")
    if summary.get("status") != "complete":
        require(summary.get("eligible") is False, "incomplete native metric package was admitted")
        return summary
    data = load_npz(freeze.parent / "native_metric_inputs.npz")
    valid, times = data["valid"], data["timestamps"]
    native = summary["native_quality"]
    require(str(data["profile_sha256"]) == PROFILE_SHA256 and str(data["input_signature"]) == summary["input_signature"]
            and summary["input_signature"] == digest({"registration_sha256": file_sha(freeze.parent / "metric_registration.json"),
                                                     "native_quality": native, "plane": summary["plane"]}), "native metric signature is stale")
    require(valid.dtype.kind == "b" and valid.ndim == 1 and times.shape == valid.shape
            and np.isfinite(times).all() and np.all(np.diff(times) > 0)
            and native["n_frames"] == len(valid) and native["n_valid"] == int(valid.sum())
            and native["valid_fraction"] == float(valid.mean())
            and native["duration_s"] == float(times[-1] - times[0])
            and native["timestamps_sha256"] == _array_sha(times)
            and native["K_sha256"] == _array_sha(data["K"]), "native VO measurements differ from policy scalars")
    grid = registration["native_grid"]
    shape = native["native_shape"]
    from monocular_osm.visual_odometry import default_intrinsics
    require(len(shape) == 3 and shape[2] == 3 and all(type(n) is int and n > 0 for n in shape)
            and np.array_equal(data["K"], default_intrinsics(shape[1], shape[0], hfov_deg=70.))
            and grid["stride"] == native["stride"] == max(1, round(grid["fps"] / 10.))
            and grid["fps"] == native["fps"] and sha_string(native["pixels_sha256"]), "native camera or 10Hz input grid changed")
    ordinals = grid["first_ordinal"] + np.arange(len(times), dtype="<i8") * grid["stride"]
    require(grid["expected_n_frames"] == len(times) and native["full_declared_grid"] is True
            and np.array_equal(data["frame_ordinals"], ordinals)
            and native["frame_ordinals_sha256"] == _array_sha(data["frame_ordinals"]), "native metric grid is incomplete")
    stop = min(grid["container_frame_count"], int(inputs["registered_window"][1] * grid["fps"])
               if inputs["registered_window"][1] is not None else grid["container_frame_count"])
    require(grid["first_ordinal"] == int(inputs["registered_window"][0] * grid["fps"])
            and grid["stop_ordinal_exclusive"] == stop
            and len(times) == max(0, (stop - grid["first_ordinal"] + grid["stride"] - 1) // grid["stride"])
            and native["first_s"] == float(times[0]) and native["last_s"] == float(times[-1])
            and times[0] >= inputs["registered_window"][0] - 1
            and (inputs["registered_window"][1] is None or times[-1] <= inputs["registered_window"][1] + 1),
            "native metric package truncated or changed the registered full input window")
    for name, shape in {"centers": (len(times), 3), "xz": (len(times), 2), "n_inliers": (len(times),),
                        "rotations": (len(times), 3, 3), "translations": (len(times), 3), "K": (3, 3)}.items():
        require(data[name].shape == shape and np.isfinite(data[name]).all(), "native VO arrays are incomplete")
    raw, lengths = data["plane_raw_m"], data["plane_lengths_m"]
    moving = valid[1:] & (np.linalg.norm(np.diff(data["xz"], axis=0), axis=1) > 1e-6)
    require(raw.shape == lengths.shape == moving.shape and np.isfinite(lengths).all()
            and np.all(lengths >= 0) and not np.isinf(raw).any(), "native plane arrays are invalid")
    measured = np.isfinite(raw) & moving & (lengths > 0)
    coverage = float(measured.sum() / max(1, moving.sum()))
    spread = float(np.median(np.abs(raw[measured] / lengths[measured] - 1))) if measured.sum() >= 10 else None
    require(summary["plane"]["coverage"] == coverage and summary["plane"]["rel_spread"] == spread
            and summary["plane"]["camera_height_m"] == 1.5, "plane admission scalars are not measured values")
    policy = policy_config()
    eligible = (auto_quality is not None and auto_quality["duration_s"] >= policy["pf_duration_min_s"]
                and auto_quality["valid_fraction"] >= policy["pf_vo_valid_min"]
                and native["valid_fraction"] >= policy["pf_vo_valid_min"] and coverage >= policy["pf_plane_coverage_min"]
                and spread is not None and spread <= policy["pf_plane_rel_spread_max"])
    require(summary["eligible"] is eligible, "native metric admission differs from fixed thresholds")
    return summary


def cues(files, child, ledger, inputs, reader_issues):
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha
    from monocular_osm.poi_clues import build_poi_hypotheses
    result = child["cues"]
    quality = child["auto_quality"]
    data = load_npz(files.record(quality["vo_artifact"], ledger))
    valid, times = data["valid"], data["timestamps"]
    require(valid.dtype.kind == "b" and valid.ndim == 1 and times.shape == valid.shape
            and len(times) >= 2 and np.isfinite(times).all() and np.all(np.diff(times) > 0)
            and quality["n_frames"] == len(valid) and quality["n_valid"] == int(valid.sum())
            and quality["valid_fraction"] == float(valid.mean())
            and quality["duration_s"] == float(times[-1] - times[0])
            and quality["timestamps_sha256"] == _array_sha(times)
            and quality["video_sha256"] == inputs["video"]["sha256"]
            and quality["window"] == inputs["registered_window"], "cue VO policy scalars lack native video measurements")
    require(result["vo"] == quality and result["duration_s"] == quality["duration_s"], "cue summary is stale")
    signed = load_npz(files.record(quality["signed_vo_cache"], ledger))
    signature = json.loads(str(signed["cache_signature"].item()))
    extraction, runtime = signature["extraction"], signature["odometry_runtime"]
    require(signature["schema"] == 2 and signature["vo_algo_version"] == int(signed["vo_algo_version"])
            and type(signature["vo_algo_version"]) is int and signature["vo_algo_version"] > 0
            and all(np.array_equal(signed[key], data[key]) for key in ("xz", "valid"))
            and runtime["workers"] == 2 and all(isinstance(runtime[k], str) and runtime[k] for k in ("numpy", "opencv"))
            and sha_string(signature["frames_sha256"])
            and Path(signature["video"]["path"]).resolve() == Path(inputs["video"]["path"]).resolve()
            and extraction["start_sec"] == inputs["registered_window"][0]
            and extraction["end_sec"] == inputs["registered_window"][1]
            and extraction["stride"] == quality["stride"] and extraction["fps"] == quality["fps"]
            and extraction["n_frames"] == len(times) and extraction["max_frames"] is None
            and extraction["frame_max_width"] is None and extraction["frame_memory_mb"] == 128.
            and extraction["timestamps_sha256"] == hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest(),
            "cue signed VO cache lacks the full native pixel/extraction/runtime identity")
    require(data["xz"].shape == (len(times), 2) and np.isfinite(data["xz"]).all()
            and data["K"].shape == (3, 3) and np.isfinite(data["K"]).all()
            and times[0] >= inputs["registered_window"][0] - 1
            and (inputs["registered_window"][1] is None or times[-1] <= inputs["registered_window"][1] + 1),
            "cue VO arrays are incomplete or outside the registered video")
    if result["ocr"]["status"] == "complete":
        base = Path(child["candidate_artifact"]["path"]).parent
        ocr = load_json(base / "poi_ocr.json")
        inventory = load_json(base / "poi_inventory.json")
        require(base / "poi_ocr.json" in ledger and base / "poi_inventory.json" in ledger,
                "native clues lack frozen OCR/inventory snapshots")
        inventory_meta = child["inventory"]
        bbox, query = inventory["bbox_wsen"], inventory["query"]
        require(inventory["city"] == inputs["city"] and inventory["scope"] == "named_city_bbox"
                and inventory["snapshot_sha256"] == digest(inventory["pois"])
                and sha_string(inventory["response_sha256"])
                and len(bbox) == 4 and all(finite(v) for v in bbox)
                and f"({bbox[1]:.7f},{bbox[0]:.7f},{bbox[3]:.7f},{bbox[2]:.7f})" in query
                and 'nwr["shop"]' in query and "out center tags" in query
                and files.record({"path": inventory_meta["path"], "sha256": inventory_meta["file_sha256"]}, ledger) == base / "poi_inventory.json"
                and all(inventory_meta[k] == inventory[k] for k in ("city", "scope", "snapshot_sha256", "response_sha256", "query", "bbox_wsen")),
                "cue inventory is not its fingerprinted full named-city public snapshot")
        meta = child["ocr_provenance"]
        from monocular_osm.scene_text import is_position_stamp
        from monocular_osm.visual_odometry import default_intrinsics
        shape = meta["frame_shape_hw"]
        require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)
                and np.array_equal(data["K"], default_intrinsics(shape[1], shape[0]))
                and all(not is_position_stamp(item["text"]) and times[0] <= item["t_sec"] <= times[-1]
                        for item in ocr["detections"]), "cue OCR contains position stamps or a different pixel/camera grid")
        require(ocr["provenance"] == meta and meta["video_sha256"] == inputs["video"]["sha256"]
                and meta.get("source") == "same_video_native_frames" and meta.get("native_bbox_grid") is True
                and meta.get("sample_count") == 24 and meta.get("min_len") == 2
                and meta.get("min_confidence") == .3 and meta.get("super_res") is False
                and meta.get("use_gpu") is False and meta.get("position_stamps_rejected") is True
                and meta.get("position_stamp_filter") == "scene_text.is_position_stamp"
                and meta["window"] == [float(times[0]), float(times[-1])]
                and meta["K_sha256"] == _array_sha(data["K"])
                and digest(ocr["detections"]) == meta["detections_sha256"]
                and not reader_issues(meta), "cue OCR lacks its fixed native/CPU/runtime identity")
        hypotheses = build_poi_hypotheses(ocr["detections"], inventory)
        require(hypotheses == child["clues"] and result["ocr"]["clue_branch_count"] == len(hypotheses["candidate_clusters"])
                and result["ocr"]["detections_sha256"] == meta["detections_sha256"]
                and result["ocr"]["inventory_sha256"] == inventory["snapshot_sha256"], "cue branches do not match literal frozen evidence")
    if child.get("metric_preflight"):
        measured = metric_package(files, child["metric_preflight"], ledger, inputs, quality)
        plane = {**measured["plane"], "native_quality": measured["native_quality"], "native_eligible": measured["eligible"]}
        require(result["plane"] == plane, "cue plane summary differs from its metric package")
    else:
        require(result["plane"].get("measured") is False, "cue claims an unmeasured road-plane signal")
        from monocular_osm.assisted_policy import policy_config
        policy = policy_config()
        require(quality["duration_s"] < policy["pf_duration_min_s"]
                or quality["valid_fraction"] < policy["pf_vo_valid_min"], "eligible auto-VO omitted native plane preflight")


def image_child(files, child, ledger, inputs, cue):
    import numpy as np
    from monocular_osm.poi_image_track import PROFILE, PROFILE_SHA256, _validated_evidence, plan_single_branch
    plan = child["image_plan"]
    require(plan == plan_single_branch(cue["clues"], cue["ocr_provenance"]), "image plan differs from the sole native clue branch")
    require(plan["profile"] == dict(PROFILE) and plan["profile_sha256"] == PROFILE_SHA256
            and child["uses_rigid_vo"] is False and child["retained_branch_count"] == child["successful_branch_count"] == 1
            and child["radius_m"] == 3000 and child["cap"] == 1500
            and child["query_mode"] == "crops" and child["score_normalization"] == "reference_mean",
            "image-only component changed its fixed bbox-v2/no-VO profile")
    records = child["evidence_artifacts"]
    require(len(records) == 1, "single-branch image component has an incomplete reference plan")
    base = Path(child["candidate_artifact"]["path"]).parent
    evidence = load_npz(files.record({**records[0], "path": base / relative(records[0]["path"])}, ledger))
    evidence["provenance"] = child["provenance"]["source"]
    indices, times, refs, scores, ids, frame, xy, _ = _validated_evidence(evidence, plan, inputs["video"]["sha256"])
    from monocular_osm.assisted_worker import _array_sha
    native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
    grid = child["acquisition"]
    require(grid["n_frames"] == len(native["timestamps"]) and grid["stride"] == cue["auto_quality"]["stride"]
            and grid["timestamps_sha256"] == _array_sha(native["timestamps"])
            and np.array_equal(indices, np.linspace(0, grid["n_frames"] - 1, min(80, grid["n_frames"])).astype(int))
            and np.array_equal(times, native["timestamps"][indices]), "image queries differ from the native video grid")
    chosen = np.asarray(child["chosen_ref_indices"])
    require(chosen.dtype.kind in "iu" and chosen.shape == indices.shape
            and np.all(chosen >= 0) and np.all(chosen < len(refs)), "image chosen reference indices are invalid")
    centered = scores - scores.mean(axis=0, keepdims=True)
    require(child["chosen_ref_ids"] == ids[chosen].tolist()
            and child["raw_route_latlon"] == refs[chosen].tolist()
            and child["selected_raw_cosines"] == scores[np.arange(len(times)), chosen].tolist()
            and child["selected_centered_scores"] == centered[np.arange(len(times)), chosen].tolist()
            and child["route_timestamps"] == times.tolist() and child["query_indices"] == indices.tolist(),
            "image output differs from its full reference/query matrix")
    selected_xy = xy[chosen]
    smoothed = np.array([np.median(selected_xy[max(0, i - 2):i + 3], axis=0) for i in range(len(times))])
    route = np.column_stack(frame.to_latlon(smoothed[:, 0], smoothed[:, 1]))
    require(np.allclose(child["route_latlon"], route, rtol=0, atol=1e-10), "image route is not its fixed metric median track")
    from monocular_osm.poi_image_track import _provenance
    require(child["provenance"] == _provenance(plan, inputs["video"]["sha256"], indices, times, refs, scores, ids,
                                             evidence["provenance"]), "image full-matrix/input provenance is stale")
    cpu = evidence["provenance"].get("cpu_runtime") or {}
    require(cpu.get("device") == "cpu" and cpu.get("intra_op_threads") == 4
            and type(cpu.get("inter_op_threads")) is int and cpu["inter_op_threads"] > 0,
            "image acquisition lacks bounded actual CPU runtime proof")


def poi_artifacts(files, child, ledger, cue):
    """Check the selected joint component's real matrices and reused native VO."""
    import numpy as np
    from monocular_osm.assisted_worker import _array_sha
    inputs, audit = child["poi_inputs"], child["poi_audit"]
    require(file_sha(inputs["map"]["cache_path"]) == inputs["map"]["sha256"], "joint whole-city map database changed")
    base = Path(child["candidate_artifact"]["path"]).parent
    native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
    vo = load_npz(files.path(base / "poi_vo.npz"))
    require(base / "poi_vo.npz" in ledger and all(np.array_equal(vo[k], native[k]) for k in ("xz", "valid", "timestamps", "K")),
            "joint component replaced its already acquired native VO/camera grid")
    require(child["route_timestamps"] == native["timestamps"].tolist(), "joint route does not cover its full native video timeline")
    for region, record in zip(audit["regions"], inputs["evidence_artifacts"]):
        evidence = load_npz(files.record({**record, "path": base / relative(record["path"])}, ledger))
        declared = region["provenance"]["evidence_identity"]
        expected = {"query_indices": evidence["query_indices"].tolist(), "query_times": evidence["query_times"].tolist(),
            "ref_ids": evidence["ref_ids"].astype(str).tolist(), "similarities_sha256": _array_sha(evidence["similarities"]),
            "ref_latlon_sha256": _array_sha(evidence["ref_latlon"])}
        require(expected == declared and np.array_equal(evidence["query_times"], native["timestamps"][evidence["query_indices"]]),
                "joint region matrix differs from its complete same-video query/reference identity")


def terminal_search(record):
    """A public completed empty/sparse listing is a result; errors are not."""
    if record.get("status") == "ok" and not record.get("error"):
        return True
    proof = (record.get("track_decoding") or {}).get("terminal_search") or {}
    count = proof.get("count")
    require(record.get("status") == "unavailable" and not record.get("error")
            and type(count) is int and 0 <= count < 30
            and proof.get("status") == ("complete_empty" if count == 0 else "complete_sparse")
            and proof.get("source") == "mapillary", "unavailable reference region has no completed listing proof")
    stats, snapshot = proof["listing"], proof["snapshot"]
    refs = proof["reference_inventory"]
    require(stats.get("listing_stats_known") is True and type(stats.get("lost")) is int and stats["lost"] == 0
            and stats["snapshot"] == snapshot and snapshot["n_refs"] == count and len(refs) == count,
            "sparse listing is partial or has stale counts")
    from monocular_osm.kartaview_vpr import _reference_snapshot, _fetch_signature
    require(_reference_snapshot(refs)["refs_sha256"] == snapshot["refs_sha256"], "sparse inventory digest is stale")
    require(proof["fetch_signature"] == _fetch_signature(record["center"], record["radius_m"], record["ref_cap"]),
            "sparse proof identifies another planned reference region")
    return True


def expansion(files, child, ledger, inputs, before):
    from monocular_osm.assisted_policy import decide_assisted_policy
    from monocular_osm.stages.assisted import summary
    selection = child["retrieval_selection"]
    multi = selection["multi_region_retrieval"]
    regions = multi["regions"]
    require(regions and len(regions) <= inputs["config"]["vpr_region_cap"], "metadata reference plans are incomplete")
    complete = True
    for region in regions:
        try:
            terminal_search(region)
        except (ValueError, KeyError, TypeError):
            complete = False
    gate = selection["assisted_expansion_gate"]
    if gate.get("error"):
        require(gate.get("municipality_skipped") is False and child["status"] != "complete",
                "failed metadata admission was promoted to a complete expanded candidate")
        return
    require(gate["ordered_subplans"] == ["metadata_and_corridors", "municipality_only_if_not_admitted"],
            "expanded search changed its metadata-first plan")
    comparison(files, gate["comparison"], ledger, inputs["registered_window"], require_all=complete)
    metadata_route = gate["comparison"].get("decoded_reference_latlon") or []
    status = "complete" if complete and len(metadata_route) >= 2 else "partial"
    evidence = {**deepcopy(before), "expanded": summary("expanded", {
        "status": status, "route_latlon": metadata_route, "comparison": gate["comparison"]})}
    decision = decide_assisted_policy(evidence)
    require(gate["evidence"] == evidence and gate["decision"] == decision
            and gate["municipality_skipped"] is (decision["selected_candidate"] == "expanded"),
            "metadata municipality-admission decision is stale or is not the fixed policy")
    city = selection.get("city_cover_retrieval")
    all_regions = list(regions)
    if gate["municipality_skipped"]:
        require(city is None, "municipality search ran after metadata was already admitted")
    else:
        require(isinstance(city, dict), "required municipality search is missing")
        coarse, fine = city.get("coarse_regions") or [], city.get("fine_regions") or []
        if city.get("status") not in {"ok", "unavailable"} or not coarse:
            complete = False
        else:
            n = city["n_cells"]
            require(type(n) is int and 0 < n <= 64 and len(coarse) == n
                    and [r["cell_index"] for r in coarse] == list(range(n))
                    and len(fine) == len(city.get("densified_cell_indices") or []) <= inputs["config"]["vpr_city_fine_regions"]
                    and [r["cell_index"] for r in fine] == (city.get("densified_cell_indices") or []), "municipality cell plan is incomplete")
            require(city["plan"]["boundary_source"]["query"] == inputs["city"]
                    and city["plan"]["boundary_validation"]["administrative_boundary_verified"] is True,
                    "municipality plan is not a verified named-city boundary")
            for region in coarse + fine:
                try:
                    terminal_search(region)
                except (ValueError, KeyError, TypeError):
                    complete = False
            require(all(r["ref_cap"] == inputs["config"]["vpr_city_coarse_cap"] for r in coarse)
                    and all(r["ref_cap"] == inputs["config"]["vpr_ref_cap"] for r in fine),
                    "municipality reference budgets differ from their fixed plan")
            all_regions.extend(coarse + fine)
    require(child["status"] != "complete" or complete, "partial/failed expanded reference plans were promoted")
    if child["status"] == "complete":
        audits = comparison(files, child["comparison"], ledger, inputs["registered_window"], require_all=True)
        for region in all_regions:
            if region["status"] != "ok":
                continue
            require(any((a.get("reference_pool") or {}).get("center") == region["center"]
                        and (a.get("reference_pool") or {}).get("radius_m") == region["radius_m"]
                        and (a.get("reference_pool") or {}).get("cap") == region["ref_cap"] for a in audits),
                    "successful planned region lacks its bound full-matrix audit")


def audit_assisted(rep, score, city, segment, *, poi_issues, filter_issues, reader_issues, ocr_identity):
    """Return JSON-safe issues and semantic identities for fleet consistency."""
    result = {"issues": [], "models": [], "weights": [], "runtimes": [], "ocr_models": [], "policy": None,
              "source": None, "configuration": None, "repeat_inputs": None}
    try:
        from monocular_osm.assisted_policy import CONFIG_SHA256, POLICY_VERSION, decide_assisted_policy, policy_config
        from monocular_osm.assisted_worker import _array_sha
        from monocular_osm.stages.assisted import REQUESTS, summary, valid_route
        provenance = rep["provenance"]
        cfg, inputs, audit = provenance["config"], provenance["assisted_inputs"], provenance["assisted_audit"]
        window = [float(value) if value else None for value in segment.split(":")]
        require(cfg.get("answer") == "assisted" and cfg.get("city") == city
                and (score.get("meta") or {}).get("mode") == "assisted"
                and (score.get("meta") or {}).get("method") == "assisted_policy", "selected assisted mode/method/config disagree")
        require(inputs.get("experimental") is True and inputs.get("validation_status") == "posthoc_dev_unvalidated"
                and inputs["city"] == city and inputs["registered_window"] == window and inputs["config"] == cfg,
                "assisted profile is not the registered experimental video/city input")
        require(Path(inputs["video"]["path"]).is_absolute() and sha_string(inputs["video"]["sha256"])
                and file_sha(inputs["video"]["path"]) == inputs["video"]["sha256"], "assisted encoded video content changed or is unavailable")
        require(re.fullmatch(r"[a-f0-9]{32}", inputs["execution_id"]) is not None, "missing unique assisted execution identity")
        forbidden = ("ground_truth", "calibration", "sensor_timestamps", "raw_images", "gps_or_oxts",
                     "manual_region_coordinates", "external_trajectory", "external_ocr_video")
        require(all(inputs["input_policy"].get(key) is False for key in forbidden), "assisted external/GT input policy is unproven")
        require(cfg.get("max_frames") is None and cfg.get("frame_max_width") is None
                and cfg.get("vpr_query_mode") == "warp" and cfg.get("ocr_engine") == "rapidocr"
                and cfg.get("vpr_source") in {"mapillary", "kartaview", "panoramax"}
                and cfg.get("poi_radius_m") == 1000 and cfg.get("poi_ref_cap") == 500 and cfg.get("camera_height_m") == 1.5,
                "assisted fixed component/input profile changed")
        require(not any(cfg.get(key) for key in ("ground_truth_waypoints", "ground_truth_streets", "openvo_trajectory_path",
                "vggt_long_trajectory_path", "use_da3_trajectory", "ocr_video_path", "use_ocr_super_res", "use_ground_flow_scale",
                "use_ipm_scale", "estimated_length_m", "graph_filter", "graph_filter_vpr", "graph_filter_map_path"))
                and (cfg.get("osm_around") is None or cfg.get("osm_around_source") == "video_title"), "assisted contains external/manual inputs")
        policy = inputs["policy"]
        require(policy["version"] == audit["policy_version"] == POLICY_VERSION
                and policy["config_sha256"] == audit["config_sha256"] == CONFIG_SHA256
                and policy["config"] == policy_config(), "assisted policy constants/version changed")
        declared = {str(key).replace("\\", "/"): value for key, value in inputs["source_sha256"].items()}
        required_sources = {"main.py", "monocular_osm/assisted_policy.py", "monocular_osm/assisted_worker.py",
            "monocular_osm/stages/assisted.py", "monocular_osm/stages/prior.py", "monocular_osm/pipeline.py",
            "monocular_osm/kartaview_vpr.py", "monocular_osm/poi_clues.py", "monocular_osm/poi_alignment.py",
            "monocular_osm/poi_localization.py", "monocular_osm/poi_image_track.py", "monocular_osm/assisted_motion.py"}
        required_sources |= {"monocular_osm/" + name for name in ("visual_odometry.py", "frame_extraction.py",
            "plane_speed.py", "graph_filter.py", "provenance.py", "baselines.py", "position.py",
            "stages/poi_runtime.py", "stages/poi_only.py", "stages/trajectory.py", "stages/acquire.py",
            "stages/graph_filter.py", "stages/report.py")}
        require(required_sources <= set(declared), "assisted source snapshot omits a required component")
        require(policy["source_sha256"] == declared["monocular_osm/assisted_policy.py"], "policy source is not bound to the parent snapshot")
        files = Files(cfg["output_dir"])
        sources(files, files.root, inputs["source_sha256"])
        physical = load_json(files.root / "result.json")
        require((physical.get("provenance") or {}).get("assisted_inputs") == inputs
                and (physical.get("provenance") or {}).get("assisted_audit") == audit
                and (physical.get("provenance") or {}).get("config") == cfg, "summary is stale relative to the actual parent result")
        frozen, parent_ledger = files.freeze(files.root / "frozen_prediction.json")
        require(frozen["inputs"] == inputs and frozen.get("prediction_available") is True, "parent freeze lacks its selected prediction/inputs")
        for name in ("result.json", "assisted_inputs.json", "assisted_audit.json"):
            require(files.root / name in parent_ledger, "parent freeze omits its result/input/audit")
        records, trace, trace_inputs = audit["reached_stages"], audit["decision_trace"], audit["decision_inputs"]
        require(0 < len(records) == len(trace) == len(trace_inputs) <= 6, "requested stage/decision transcript is incomplete")
        evidence, children, ledgers = {}, {}, {}
        next_kind, terminal = "baseline", None
        for index, record in enumerate(records):
            kind = record["kind"]
            require(kind == next_kind and kind not in children, "policy stages were skipped, repeated or reordered")
            before = deepcopy(evidence)
            for artifact in record.get("artifacts") or []:
                files.record(artifact, parent_ledger)
            child, ledger = verify_child(files, record, inputs, before)
            children[kind], ledgers[kind] = child, ledger
            if child.get("comparison") is not None:
                comparison(files, child["comparison"], ledger, window,
                           require_all=kind == "expanded" and child.get("status") == "complete")
            if child.get("status") == "complete":
                if kind in {"baseline", "expanded", "poi_joint", "poi_image_track"}:
                    runtime = child["cpu_runtime"]
                    require(runtime.get("device") == "cpu" and runtime.get("intra_op_threads") == 4
                            and type(runtime.get("inter_op_threads")) is int and runtime["inter_op_threads"] > 0
                            and isinstance(runtime.get("torch_version"), str) and bool(runtime["torch_version"])
                            and child.get("model_downloads_disabled") is True, "visual child lacks actual bounded CPU/model runtime proof")
                    result["runtimes"].append(digest({key: runtime[key] for key in
                        ("device", "intra_op_threads", "inter_op_threads", "torch_version")}))
                    from monocular_osm.kartaview_vpr import HUB_MEGALOC
                    backbone, weights = child["actual_backbone"], child["actual_model_state"]
                    require(backbone == {"name": "megaloc", "hub": HUB_MEGALOC, "device": "cpu"}
                            and sha_string(weights.get("sha256")), "visual child lacks its actual registered model-state identity")
                    if kind in {"baseline", "expanded"} and not child.get("comparison", {}).get("unavailable_reason"):
                        require(child["comparison"]["model"] == {**backbone, "weights_sha256": weights["sha256"]},
                                "decoded comparison belongs to different actual model weights")
                    result["weights"].append(digest({**backbone, "weights_sha256": weights["sha256"]}))
                if kind == "cues":
                    cues(files, child, ledger, inputs, reader_issues)
                    if child["cues"]["ocr"]["status"] == "complete":
                        result["ocr_models"].append(ocr_identity(child["ocr_provenance"]))
                elif kind == "expanded":
                    expansion(files, child, ledger, inputs, before)
                elif kind == "poi_joint":
                    issues = poi_issues({"provenance": {"config": child["config"], "poi_inputs": child["poi_inputs"],
                        "poi_localization_audit": child["poi_audit"]}}, {"meta": {"method": "poi_only"}}, city, segment)
                    require(not issues, "; ".join(issues))
                    require(child["poi_audit"]["clues"] == children["cues"]["clues"], "joint POI child changed native clues/branches")
                    poi_artifacts(files, child, ledger, children["cues"])
                elif kind == "poi_image_track":
                    image_child(files, child, ledger, inputs, children["cues"])
                elif kind == "metric_pf":
                    require(child["metric_preflight"] == children["cues"]["metric_preflight"], "PF child replaced its native preflight")
                    measured = metric_package(files, child["metric_preflight"], ledgers["cues"], inputs, children["cues"]["auto_quality"])
                    require(measured["eligible"] is True and child.get("metric_only") is True
                            and child.get("uses_user_spatial_prior") is False and child.get("position_observation_count") == 0,
                            "PF child lacks independently eligible native metric odometry")
                    require(child["config"]["frame_stride"] == measured["native_quality"]["stride"], "PF grid differs from native metric preflight")
                    native_times = load_npz(files.path(Path(child["metric_preflight"]["artifacts"]["freeze"]["path"]).parent
                                                      / "native_metric_inputs.npz"))["timestamps"].tolist()
                    route_times = child["route_timestamps"]
                    require(route_times[0] == native_times[0] and route_times[-1] == native_times[-1]
                            and set(route_times) <= set(native_times), "PF route times differ from the measured native video grid")
                    issues = filter_issues({"provenance": {"config": child["config"], "graph_filter_inputs": child["filter_inputs"]}},
                                           {"meta": {"method": "graph_filter_only"}}, city)
                    require(not issues, "; ".join(issues))
                    graph = child["filter_inputs"]["map"]
                    require(file_sha(graph["cache_path"]) == graph["sha256"], "PF whole-city map database changed")
            derived = summary(kind, child)
            require(record["summary"] == derived, "reached stage scalars are not derived from its frozen candidate")
            evidence[kind] = derived
            require(trace_inputs[index] == evidence and trace[index] == decide_assisted_policy(evidence), "assisted decision replay differs")
            decision = trace[index]
            if decision["next_action"] in REQUESTS:
                next_kind = REQUESTS[decision["next_action"]]
            else:
                require(index == len(records) - 1, "unrequested future stages were executed")
                terminal = decision
            if kind == "baseline" and not child.get("comparison", {}).get("unavailable_reason"):
                comp = child["comparison"]
                result["models"].append(digest({"model": comp["model"], "preprocessing": comp["query"]["preprocessing"]}))
        require(terminal is not None and audit["evidence"] == evidence, "assisted transcript did not reach a terminal decision")
        selected = terminal["selected_candidate"]
        require(selected == audit["selected_candidate"] and selected in children, "selected candidate disagrees with policy replay")
        child = children[selected]
        require(valid_route(child) is not None and child.get("worker_returncode") == 0, "selected assisted child is partial/failed")
        if selected == "baseline":
            require(not (child.get("comparison") or {}).get("unavailable_reason")
                    and bool((child.get("comparison") or {}).get("audit_artifacts")),
                    "selected baseline lacks actual decoded-reference evidence")
        route_sha = digest(child["route_latlon"])
        require(route_sha == audit["selected_route_sha256"]
                and audit["selected_child_artifact_sha256"] == child["candidate_artifact"]["sha256"], "selected child/route identity is stale")
        answer = physical["answer"]
        require(answer.get("mode") == "assisted" and answer.get("method") == "assisted_policy"
                and answer.get("source") == "result.assisted_position" and answer.get("experimental") is True
                and answer.get("confidence_calibrated") is False and digest(answer["route_latlon"]) == route_sha
                and provenance["assisted_selected_answer"]["route_sha256"] == route_sha, "primary Answer differs from its selected child")
        result["policy"] = digest(policy)
        from scripts.benchmark import _effective_config
        result["source"] = digest(inputs["source_sha256"])
        result["configuration"] = digest(_effective_config({"config": cfg}))
        repeat = {"video": inputs["video"]["sha256"], "source": digest(inputs["source_sha256"])}
        repeat["configuration"] = digest(_effective_config({"config": cfg}, within_clip=True))
        if "cues" in children and children["cues"].get("status") == "complete":
            cue = children["cues"]
            native = load_npz(files.record(cue["auto_quality"]["vo_artifact"]))
            repeat["vo"] = digest({key: _array_sha(native[key]) for key in ("xz", "valid", "timestamps", "K")})
            if cue["cues"]["ocr"]["status"] == "complete":
                repeat.update(ocr=cue["ocr_provenance"]["detections_sha256"], inventory=cue["inventory"]["snapshot_sha256"])
        for kind, name in (("poi_joint", "poi_inputs"), ("metric_pf", "filter_inputs")):
            if children.get(kind, {}).get("status") == "complete":
                repeat[kind + "_map"] = children[kind][name]["map"]["sha256"]
        result["repeat_inputs"] = repeat
    except Exception as error:
        result["issues"].append(f"assisted frozen-input audit: {type(error).__name__}: {error}")
    return result
