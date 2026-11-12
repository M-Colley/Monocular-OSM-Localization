"""Finite native-DVLT protocol: historical pixels are DATA, never NN authority.

STUDY_REF is deliberately unset. No real execution is permitted until Root
prospectively registers the final sources and replaces this placeholder.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import zipfile
from monocular_osm import multiview_motion_inputs as old

ROOT = Path(__file__).resolve().parents[1]
STUDY_REF = None
STUDY_SCHEMA = "root-prospective-dvlt-native-fixed32-local-geometry-study-v1"
SOURCE_PATHS = (
    "monocular_osm/multiview_motion_inputs.py", "monocular_osm/native_pixel_mask.py",
    "monocular_osm/native_text_detection.py", "monocular_osm/dvlt_native_geometry.py",
    "monocular_osm/dvlt_native_protocol.py", "monocular_osm/dvlt_native_runtime.py",
    "scripts/prepare_multiview_motion_inputs.py", "scripts/prepare_dvlt_native_masks.py",
    "scripts/dvlt_native_window_probe.py",
)
MAX_JSON = 16 << 20
MAX_NPZ = 256 << 20
MAX_TYPED_OUTPUT = 128 << 20
MAX_NATIVE_PIXELS = 8500000
MASK_MEMBERS = ("masked_bgr", "invalid_mask", "video_ordinals", "timestamps",
                "native_read_rows", "quad_coordinates", "mask_quad_coordinates", "quad_offsets")
require, safe, sha, ref = old.require, old.safe, old.sha, old.ref
typed, typed_header = old.typed, old.typed_header
json_bytes, write_json = old.json_bytes, old.write_json


def same_ref(a, b):
    """Compare historical names without stat/resolve/open."""
    return (isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) == {"path", "sha256"}
            and old.digest_ok(a["sha256"]) and a["sha256"] == b["sha256"]
            and old.opaque_path(a["path"]) == old.opaque_path(b["path"]))


def same_path(a, b):
    """Canonical path identity only; no file/binary hash claim or IO."""
    return old.opaque_path(a) == old.opaque_path(b)


def observed_header(array):
    """Detector observer's compact-JSON typed convention, distinct from old packets."""
    import json
    import numpy as np
    require(isinstance(array, np.ndarray) and array.dtype.kind in "biuf" and array.dtype.itemsize <= 8,
            "bounded observer numeric dtype required")
    value = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str], sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode())
    if value.size:
        h.update(memoryview(value).cast("B"))
    return {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": h.hexdigest()}


class Ledger(old.Ledger):
    """Inherited shared256MiB JSON accounting, strict keys/NaN/SHA checks."""


def load_config(path, expected, phase):
    require(phase in {"mask", "run"}, "unknown phase")
    require(STUDY_REF is not None, "native DVLT study unregistered; placeholder is not authority")
    require(Path(__file__).resolve() == ROOT/"monocular_osm/dvlt_native_protocol.py"
            and Path(old.__file__).resolve() == ROOT/"monocular_osm/multiview_motion_inputs.py",
            "genuine protocol/dependency origins required")
    a = Ledger()
    cfg = a.json({"path": str(safe(path)), "sha256": expected})
    common = {"schema", "source_hashes", "study_registration_ref", "data_acceptance_ref", "window_id", "output_dir"}
    key = "detector_control_ref" if phase == "mask" else "model_control_ref"
    keys = common | ({key} if phase == "mask" else {key, "mask_acceptance_ref"})
    require(set(cfg) == keys and cfg["schema"] == ("dvlt-native-mask-window-run-v1" if phase == "mask"
            else "dvlt-native-local-geometry-window-run-v1"), "exact phase config required")
    require(type(cfg["window_id"]) is int and 0 <= cfg["window_id"] < 32, "fixed window0..31 required")
    require(same_ref(cfg["study_registration_ref"], STUDY_REF), "prospective study differs")
    sources = cfg["source_hashes"]
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_PATHS), "complete executing source closure required")
    for name, digest in sources.items():
        a.add({"path": str(ROOT/name), "sha256": digest})
    study = a.json(cfg["study_registration_ref"])
    core = {k: v for k, v in sources.items() if k != "monocular_osm/dvlt_native_protocol.py"}
    require(study.get("schema") == STUDY_SCHEMA and study.get("core_source_hashes") == core,
            "registered exact eight-core source/study differs")
    require(study.get("window_ids") == list(range(32)) and study.get("views_per_window") == 7
            and study.get("one_window_per_actor") is True and study.get("require_224_distinct_read_rows") is True,
            "fixed full32/224 distinct-frame one-window study required")
    for name in ("original_histories_opaque", "old_no_neural_authority_not_upgraded",
                 "new_neural_study_separate_from_original_numerical_authority"):
        require(study.get(name) is True, "separate DATA/NN authority required")
    require(study.get("ground_truth_used") is False and study.get("policy_admission") is False
            and study.get("selected_answer") is None, "diagnostic-only study required")
    require(study.get("resources") == {"cpu": 2, "rss_bytes": 3 << 30, "headroom_bytes": 8 << 30,
            "inner_seconds": 1800, "outer_seconds": 1830, "parent_seconds": 14400}, "fixed resource envelope required")
    require(same_ref(cfg[key], study[key]), "study control ref differs")
    profiles = study.get("profiles")
    profile_keys = {"mask_profile_sha256", "detection_profile_sha256", "mask_operand_profile_sha256",
                    "geometry_profile_sha256", "model_profile_sha256"}
    require(isinstance(profiles, dict) and set(profiles) == profile_keys
            and all(old.digest_ok(x) for x in profiles.values()), "all five registered profiles required")
    if phase == "mask":
        from monocular_osm import native_pixel_mask as masking, native_text_detection as detection
        require(profiles["mask_profile_sha256"] == masking.PROFILE_SHA256
                and profiles["detection_profile_sha256"] == detection.PROFILE_SHA256
                and profiles["mask_operand_profile_sha256"] == detection.MASK_OPERAND_PROFILE_SHA256,
                "registered exact detection/mask profiles differ")
    else:
        from monocular_osm import dvlt_native_geometry as geometry, dvlt_native_runtime as runtime
        require(profiles["geometry_profile_sha256"] == geometry.PROFILE_SHA256
                and profiles["model_profile_sha256"] == runtime.PROFILE_SHA256, "registered geometry/model profiles differ")
    a.study, a.control = study, a.json(cfg[key])
    safe(cfg["output_dir"])
    return cfg, a


def admit_data(cfg, a):
    """New study DATA admission before any original window NPZ member."""
    qa = a.json(cfg["data_acceptance_ref"])
    require(qa.get("schema") == "root-closed-dvlt-native-data-provenance-v1"
            and qa.get("status") == "ACCEPTED_FIXED32_NATIVE_PIXEL_CLOCK_DATA_FOR_DVLT_STUDY"
            and same_ref(qa.get("study_ref"), cfg["study_registration_ref"]), "new Root pixel DATA admission required")
    require(set(a.study["input_data"]) == {"original_prepared_manifest_ref", "original_prepared_input_acceptance_ref",
            "native_read_ledger_ref"}, "exact pixel DATA authority refs required")
    for key, value in a.study["input_data"].items():
        require(same_ref(qa.get(key), value), "declared original DATA ref differs")
    for key in ("original_data_root_closed", "fixed32_all224_native_frames_verified", "historical_paths_remain_opaque",
                "old_no_neural_or_numerical_authority_not_upgraded"):
        require(qa.get(key) is True, "new Root DATA proof incomplete")
    require(qa.get("window_count") == 32 and qa.get("geometry_or_DVLT_quality_admitted") is False,
            "DATA is not model/geometry authority")
    oq = a.json(qa["original_prepared_input_acceptance_ref"])
    require(oq.get("schema") == "root-closed-native-multiview-prepared-inputs-v1"
            and oq.get("status") == "ACCEPTED_NATIVE_PIXEL_CLOCK_WINDOWS_NO_GEOMETRY"
            and oq.get("window_count") == 32
            and same_ref(oq.get("prepared_manifest_ref"), qa["original_prepared_manifest_ref"])
            and same_ref(oq.get("native_read_ledger_ref"), qa["native_read_ledger_ref"]),
            "genuine original pixel/clock acceptance required")
    for key in ("all_observed_worker_births_closed", "all_protected_input_hashes_unchanged",
                "full_dense_native_pixels_clock_verified", "no_geometry_computed"):
        require(oq.get(key) is True, "original pixel DATA not closed")
    read = a.json(qa["native_read_ledger_ref"])
    manifest = a.json(qa["original_prepared_manifest_ref"])
    require(read.get("schema") == "root-fixed32-native-read-window-lineage-v1"
            and manifest.get("schema") == old.PREPARED_SCHEMA and manifest.get("status") == old.PREPARED_STATUS,
            "original packet schemas differ")
    wanted = old.windows(read["native_dense_count"], read["first_native_video_ordinal"], read["original_video_ordinal_stride"])
    records, rows = manifest.get("windows"), read.get("windows")
    require(isinstance(records, list) and isinstance(rows, list) and len(records) == len(rows) == 32,
            "entire original32 inventory required")
    folder = safe(qa["original_prepared_manifest_ref"]["path"]).parent
    total, selected = 0, []
    for record, row, expected in zip(records, rows, wanted):
        require(all(type(row.get(k)) is type(v) and row.get(k) == v for k, v in expected.items()),
                "fixed original READ-window schedule differs")
        require(all(record.get(k) == v for k, v in row.items()), "original window lineage differs")
        times = row.get("signed_decoder_times_s")
        require(isinstance(times, list) and len(times) == 7
                and all(type(t) is float and math.isfinite(t) and t > 0 for t in times)
                and all(x < y for x, y in zip(times, times[1:])), "positive signed cached decoder timing required")
        require(type(record.get("bytes")) is int and 0 < record["bytes"] <= MAX_NPZ
                and same_ref(record["array_ref"], {"path": str(folder/f"window_{row['window_id']:02d}.npz"),
                "sha256": record["array_ref"]["sha256"]}), "original window physical role differs")
        require(set(record.get("typed_headers", {})) == set(old.WINDOW_MEMBERS)
                and len(record.get("frame_typed_sha256", [])) == 7
                and all(old.digest_ok(x) for x in record["frame_typed_sha256"]), "original typed/frame inventory incomplete")
        selected.extend(row["native_read_rows"])
        total += record["bytes"]
    require(len(set(selected)) == 224 and total == manifest.get("total_npz_bytes") and total <= 8 << 30,
            "fixed224 distinct frames/whole packet budget differs")
    proof = manifest.get("full_native_proof", {})
    for pk, rk in (("frames_sha256", "full_signed_native_frames_sha256"),
                   ("clock_typed_sha256", "full_signed_native_clock_typed_sha256"),
                   ("clock_raw_F64_sha256", "full_signed_native_clock_raw_F64_sha256")):
        require(old.digest_ok(read.get(rk)) and proof.get(pk) == read[rk], "whole native proof differs")
    require(proof.get("positive_CAP_PROP_POS_MSEC_exact_cached_F64") is True
            and proof.get("independent_container_PTS_verified") is False
            and proof.get("K_is_measured_calibration") is False, "original timing/K limitations changed")
    history = manifest.get("protected_hashes")
    require(isinstance(history, dict) and all(old.digest_ok(v) for v in history.values())
            and len({old.opaque_path(k) for k in history}) == len(history), "unambiguous historical map required")
    a.read, a.data_manifest, a.data_qa = read, manifest, qa
    a.historical_protected_refs = dict(history)
    return records


def read_npz(role, ledger, specs, typed_headers, *, maximum=MAX_NPZ, check=lambda: None):
    """ALL archive bounds, then bounded headers, then exact finite typed bodies."""
    import numpy as np
    check()
    path = safe(role["path"])
    require(path.stat().st_size <= maximum, "archive storage cap")
    ledger.add(role)
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        require(len(entries) == len(specs) and len({x.filename for x in entries}) == len(entries)
                and {x.filename for x in entries} == {k+".npy" for k in specs}, "exact unique NPZ members required")
        require(all(not x.is_dir() and not x.flag_bits & 1 and 0 <= x.file_size <= maximum for x in entries)
                and sum(x.file_size for x in entries) <= maximum, "all ZIP budgets must precede headers")
        for item in entries:
            with z.open(item) as f:
                prefix = f.read(8)
                require(prefix[:6] == b"\x93NUMPY" and prefix[6:] in (b"\x01\x00", b"\x02\x00"), "NPY version")
                n = 2 if prefix[6] == 1 else 4
                length_bytes = f.read(n)
                require(len(length_bytes) == n, "short NPY header")
                size = int.from_bytes(length_bytes, "little")
                require(size <= 4096 and 8+n+size <= item.file_size, "NPY header length cap before parse")
            with z.open(item) as f:
                version = np.lib.format.read_magic(f)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(f)
                name = item.filename[:-4]
                require(not fortran and not dtype.hasobject and (shape, dtype) == specs[name]
                        and math.prod(shape)*dtype.itemsize == item.file_size-f.tell(), "exact typed NPY header/body length")
        arrays = {}
        for item in entries:
            check()
            with z.open(item) as f:
                value = np.lib.format.read_array(f, allow_pickle=False)
                require(f.read(1) == b"", "NPY payload EOF/CRC required")
            arrays[item.filename[:-4]] = np.ascontiguousarray(value)
    require({k: typed_header(v) for k, v in arrays.items()} == typed_headers, "typed array values differ")
    check()
    require(sha(path) == role["sha256"], "window bytes changed while reading")
    return arrays


def read_original_window(record, ledger, check=lambda: None):
    import numpy as np
    shape = tuple(record["typed_headers"]["frames_bgr"]["shape"])
    require(len(shape) == 4 and shape[0] == 7 and shape[3] == 3
            and all(type(x) is int and x > 0 for x in shape) and shape[1]*shape[2] <= MAX_NATIVE_PIXELS,
            "native shape cap")
    specs = {"frames_bgr": (shape, np.dtype("u1")), "video_ordinals": ((7,), np.dtype("i8")),
             "timestamps": ((7,), np.dtype("f8")), "K": ((3, 3), np.dtype("f8")),
             "native_read_rows": ((7,), np.dtype("i8"))}
    body = read_npz(record["array_ref"], ledger, specs, record["typed_headers"], check=check)
    require(list(shape[1:]) == ledger.read["expected_native_shape"]
            and typed(body["K"]) == ledger.read["native70_K_typed_sha256"], "original native shape/K history differs")
    require([typed(v) for v in body["frames_bgr"]] == record["frame_typed_sha256"], "native per-frame pixels differ")
    for name, key in (("video_ordinals", "video_ordinals"), ("timestamps", "signed_decoder_times_s"),
                      ("native_read_rows", "native_read_rows")):
        require(np.array_equal(body[name], record[key]), "native clock/row identity differs")
    return body


def write_npz(path, arrays, *, maximum=MAX_NPZ, payload_maximum=MAX_NPZ):
    import numpy as np
    require(isinstance(arrays, dict) and 0 < len(arrays) <= 128
            and all(isinstance(k, str) and k and k.replace("_", "").isalnum() for k in arrays), "bounded safe array roles required")
    require(all(isinstance(v, np.ndarray) and not v.dtype.hasobject and v.flags.c_contiguous for v in arrays.values())
            and sum(v.nbytes for v in arrays.values()) <= payload_maximum, "typed payload cap")
    with Path(path).open("xb") as f:
        np.savez(f, **arrays)
    require(Path(path).stat().st_size <= maximum, "derived archive cap")
    return {"array_ref": ref(path), "typed_headers": {k: typed_header(v) for k, v in arrays.items()},
            "payload_bytes": sum(v.nbytes for v in arrays.values()), "archive_bytes": Path(path).stat().st_size}


def publish(path, manifest, ledger, outputs, check=lambda: None):
    """Final callback first; no subsequent callback may mutate checked files."""
    check()
    for name, expected in outputs.items():
        require(sha(name) == expected, "final generated bytes differ")
    old.output_membership(Path(path).parent, [Path(p).name for p in outputs])
    ledger.verify()
    write_json(path, manifest)


def closed_actor(qa, ledger, *, subject_key, subject_ref, schema):
    """Authenticate Root's external closure; producer never closes itself."""
    require(qa.get("all_observed_worker_births_closed") is True
            and qa.get("all_protected_input_hashes_unchanged") is True, "Root actor/protected closure required")
    ledger.json(qa["registration_ref"])
    result = ledger.json(qa["result_ref"])
    require(result.get("status") == "complete" and type(result.get("returncode")) is int
            and result["returncode"] == 0 and result.get("reason") is None, "closed actor rc0 required")
    closure = ledger.json(qa["external_closure_ref"])
    require(closure.get("schema") == schema and same_ref(closure.get(subject_key), subject_ref)
            and same_ref(closure.get("registration_ref"), qa["registration_ref"])
            and same_ref(closure.get("result_ref"), qa["result_ref"])
            and closure.get("worker_births_closed") is True and closure.get("protected_inputs_unchanged") is True,
            "exact external closure binding required")


def admit_mask(cfg, ledger, original):
    """Root-closed completed detector/mask evidence precedes DVLT/NPZ bodies."""
    qa = ledger.json(cfg["mask_acceptance_ref"])
    require(qa.get("schema") == "root-closed-dvlt-native-mask-window-v1"
            and qa.get("status") == "ACCEPTED_NATIVE_MASKED_WINDOW_NO_DVLT_GEOMETRY"
            and same_ref(qa.get("study_ref"), cfg["study_registration_ref"])
            and same_ref(qa.get("data_acceptance_ref"), cfg["data_acceptance_ref"])
            and type(qa.get("window_id")) is int and qa["window_id"] == cfg["window_id"],
            "closed matching native mask acceptance required")
    require(qa.get("all_seven_detector_forwards_and_postprocessing_completed") is True
            and qa.get("no_DVLT_forward_or_geometry_computed") is True, "all seven positive detector completions required")
    role = qa["mask_manifest_ref"]
    closed_actor(qa, ledger, subject_key="mask_manifest_ref", subject_ref=role,
                 schema="root-dvlt-native-mask-window-external-closure-v1")
    manifest = ledger.json(role)
    require(manifest.get("schema") == "frozen-dvlt-native-mask-window-v1"
            and manifest.get("status") == "PREPARED_NATIVE_MASKED_WINDOW_NO_DVLT_GEOMETRY"
            and manifest.get("window_id") == cfg["window_id"]
            and manifest.get("source_hashes") == cfg["source_hashes"]
            and same_ref(manifest.get("study_registration_ref"), cfg["study_registration_ref"])
            and same_ref(manifest.get("data_acceptance_ref"), cfg["data_acceptance_ref"])
            and same_ref(manifest.get("detector_control_ref"), ledger.study["detector_control_ref"])
            and same_ref(manifest.get("original_array_ref"), original["array_ref"]), "mask manifest authority differs")
    require(manifest.get("mask_profile_sha256") == ledger.study["profiles"]["mask_profile_sha256"], "mask policy profile differs")
    require(manifest.get("neural_forward_run") is True and manifest.get("DVLT_forward_run") is False
            and manifest.get("geometry_computed") is False and manifest.get("policy_admission") is False,
            "detector-only mask scope required")
    require(all(manifest.get(k) == original[k] for k in ("native_read_rows", "video_ordinals", "signed_decoder_times_s")),
            "mask original row/clock lineage differs")
    frames = manifest.get("frames")
    require(isinstance(frames, list) and len(frames) == 7, "all seven native mask receipts required")
    for i, frame in enumerate(frames):
        receipt = frame["detection_receipt"]
        require(frame.get("view_index") == i and frame.get("original_typed_sha256") == original["frame_typed_sha256"][i]
                and receipt.get("schema") == "observed-native-detection-only-receipt-v1"
                and receipt.get("profile_sha256") == ledger.study["profiles"]["detection_profile_sha256"]
                and receipt.get("mask_operand_conversion", {}).get("profile_sha256") == ledger.study["profiles"]["mask_operand_profile_sha256"]
                and receipt.get("session", {}).get("completed") == 1
                and receipt.get("postprocess", {}).get("completed") == 1
                and receipt.get("public_native_output_completed") is True
                and receipt.get("all_returned_quads_retained") is True
                and receipt.get("observers_restored") is True, "detector completion/frame identity differs")
    ledger.add(manifest["packet"]["array_ref"])
    ledger.mask_manifest_ref = dict(role)
    return manifest


def read_masked_window(manifest, ledger, check=lambda: None):
    import numpy as np
    packet = manifest["packet"]
    shape = tuple(packet["typed_headers"]["masked_bgr"]["shape"])
    require(len(shape) == 4 and shape[0] == 7 and shape[-1] == 3
            and all(type(n) is int and n > 0 for n in shape) and shape[1]*shape[2] <= MAX_NATIVE_PIXELS,
            "bounded native masked grid required")
    qshape = tuple(packet["typed_headers"]["quad_coordinates"]["shape"])
    require(len(qshape) == 3 and qshape[1:] == (4, 2) and type(qshape[0]) is int and 0 <= qshape[0] <= 7000,
            "all quads within seven frame cap required")
    specs = {"masked_bgr": (shape, np.dtype("u1")), "invalid_mask": (shape[:3], np.dtype("bool")),
             "video_ordinals": ((7,), np.dtype("i8")), "timestamps": ((7,), np.dtype("f8")),
             "native_read_rows": ((7,), np.dtype("i8")), "quad_coordinates": (qshape, np.dtype("f8")),
             "mask_quad_coordinates": (qshape, np.dtype("f8")),
             "quad_offsets": ((8,), np.dtype("i8"))}
    body = read_npz(packet["array_ref"], ledger, specs, packet["typed_headers"], check=check)
    offsets = body["quad_offsets"]
    require(offsets[0] == 0 and offsets[-1] == qshape[0] and np.all(np.diff(offsets) >= 0)
            and np.all(np.diff(offsets) <= 1000), "all native quad partitions required")
    raw, operand = body["quad_coordinates"], body["mask_quad_coordinates"]
    require(np.isfinite(raw).all() and np.isfinite(operand).all()
            and np.all((raw[..., 0] >= 0) & (raw[..., 0] <= shape[2]))
            and np.all((raw[..., 1] >= 0) & (raw[..., 1] <= shape[1])), "original public native quad bounds")
    expected_operand = raw.copy()
    expected_operand[..., 0] = np.minimum(expected_operand[..., 0], shape[2]-1)
    expected_operand[..., 1] = np.minimum(expected_operand[..., 1], shape[1]-1)
    require(np.array_equal(operand, expected_operand), "only declared outer-boundary mask operands allowed")
    require(np.all(body["masked_bgr"][body["invalid_mask"]] == 0), "masked invalid pixels must be black")
    for key, name in (("native_read_rows", "native_read_rows"), ("video_ordinals", "video_ordinals"),
                      ("timestamps", "signed_decoder_times_s")):
        require(np.array_equal(body[key], manifest[name]), "masked packet original row/clock differs")
    for i, frame in enumerate(manifest["frames"]):
        require(typed(body["masked_bgr"][i]) == frame["masked_typed_sha256"]
                and typed(body["invalid_mask"][i]) == frame["mask_typed_sha256"]
                and typed(body["quad_coordinates"][offsets[i]:offsets[i+1]]) == frame["quads_typed_sha256"]
                and typed(body["mask_quad_coordinates"][offsets[i]:offsets[i+1]]) == frame["mask_quads_typed_sha256"],
                "native mask/quad typed identities differ")
        for name, receipt_key in (("quad_coordinates", "native_quads"), ("mask_quad_coordinates", "mask_quads")):
            original = frame["detection_receipt"][receipt_key]
            dtype = np.dtype(original["dtype"])
            require(dtype.kind in "iuf" and dtype.itemsize <= 8 and dtype.isnative, "original native quad dtype required")
            restored = body[name][offsets[i]:offsets[i+1]].astype(dtype)
            require(observed_header(restored) == original
                    and np.array_equal(restored.astype(np.float64), body[name][offsets[i]:offsets[i+1]]),
                    "packed quad cannot exactly reconstruct original observer values/bytes")
    return body


def validate_cohort(rows):
    """Root controller accounting helper, never a numerical/quality aggregate."""
    allowed = {"complete_local_diagnostic", "input_unavailable", "detector_failed", "model_failed",
               "resource_failed", "check_unavailable", "actor_failed", "not_started"}
    require(isinstance(rows, list) and len(rows) == 32, "all32 outcomes including unrun windows required")
    for i, row in enumerate(rows):
        require(type(row.get("window_id")) is int and row["window_id"] == i and row.get("status") in allowed,
                "fixed ordered window outcome required")
        require(row.get("metric_scale_or_global_gauge_claimed") is False and row.get("policy_admission") is False,
                "local diagnostic scope required")
        require(row.get("status") != "not_started" or row.get("reason") == "parent_deadline_exhausted",
                "unrun deadline windows must remain operational unavailable")
    return {"window_count": 32, "counts": {s: sum(r["status"] == s for r in rows) for s in sorted(allowed)},
            "fixed_window_denominator": 32, "creator": "Root/controller; not DVLT fleet master"}


def validate_public_prediction(arrays):
    import numpy as np
    require(isinstance(arrays, dict) and set(arrays) == {"depths", "depths_conf", "world_points", "world_points_direct",
            "world_points_direct_conf", "camera_to_worlds", "intrinsics", "input_images", "gradio_valid_pixels"},
            "all and only public prediction/input arrays required")
    image = arrays["input_images"]
    require(isinstance(image, np.ndarray) and image.ndim == 5 and image.shape[:3] == (1, 7, 3), "input batch shape")
    h, w = image.shape[-2:]
    require(14 <= min(h, w) <= max(h, w) <= 504 and h % 14 == w % 14 == 0, "official processed grid")
    specs = {"depths": (1, 7, h, w), "depths_conf": (1, 7, h, w), "world_points": (1, 7, h, w, 3),
             "world_points_direct": (1, 7, h, w, 3), "world_points_direct_conf": (1, 7, h, w),
             "camera_to_worlds": (7, 3, 4), "intrinsics": (7, 3, 3), "input_images": (1, 7, 3, h, w),
             "gradio_valid_pixels": (1, 7, h, w)}
    require(all(isinstance(a, np.ndarray) and a.flags.c_contiguous and a.shape == specs[k]
                and a.dtype == (np.bool_ if k == "gradio_valid_pixels" else np.float32) for k, a in arrays.items()),
            "exact actual F32 public output/bool validity shapes required")
    return h, w
