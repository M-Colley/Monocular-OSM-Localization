"""Private tiny packets only; no model import, weights, video or actual refs."""
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
import pytest
from monocular_osm import dvlt_native_protocol as p


def write(tmp, name, value):
    path = (tmp/name).resolve()
    assert path.is_relative_to(tmp.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        p.write_json(path, value)
    return p.ref(path)


def packet_fixture(tmp, monkeypatch):
    """All32 real tiny NPZ files, fabricated closed Root metadata, private paths."""
    folder = tmp/"original"; folder.mkdir()
    K = np.eye(3, dtype=np.float64)
    records = []
    windows = p.old.windows(224, 11, 4)
    for row in windows:
        row["signed_decoder_times_s"] = [.1+float(r)*.1 for r in row["native_read_rows"]]
        frames = np.full((7, 64, 96, 3), row["window_id"]+1, dtype=np.uint8)
        arrays = {"frames_bgr": frames, "K": K, "timestamps": np.asarray(row["signed_decoder_times_s"]),
                  "video_ordinals": np.asarray(row["video_ordinals"], dtype=np.int64),
                  "native_read_rows": np.asarray(row["native_read_rows"], dtype=np.int64)}
        result = p.write_npz(folder/f"window_{row['window_id']:02d}.npz", arrays)
        records.append({**row, "array_ref": result["array_ref"], "bytes": result["archive_bytes"],
                        "typed_headers": result["typed_headers"], "frame_typed_sha256": [p.typed(x) for x in frames]})
    read = {"schema": "root-fixed32-native-read-window-lineage-v1", "native_dense_count": 224,
            "first_native_video_ordinal": 11, "original_video_ordinal_stride": 4,
            "expected_native_shape": [64, 96, 3], "native70_K_typed_sha256": p.typed(K),
            "windows": windows, "full_signed_native_frames_sha256": "1"*64,
            "full_signed_native_clock_typed_sha256": "2"*64, "full_signed_native_clock_raw_F64_sha256": "3"*64}
    read_ref = write(tmp, "read.json", read)
    history = {str((tmp/"NEVER_OPEN_original_video.mp4").resolve()): "4"*64}
    manifest = {"schema": p.old.PREPARED_SCHEMA, "status": p.old.PREPARED_STATUS, "windows": records,
        "total_npz_bytes": sum(r["bytes"] for r in records), "protected_hashes": history,
        "full_native_proof": {"frames_sha256": "1"*64, "clock_typed_sha256": "2"*64,
            "clock_raw_F64_sha256": "3"*64, "positive_CAP_PROP_POS_MSEC_exact_cached_F64": True,
            "independent_container_PTS_verified": False, "K_is_measured_calibration": False}}
    manifest_ref = write(tmp, "original/manifest.json", manifest)
    oldqa = {"schema": "root-closed-native-multiview-prepared-inputs-v1",
        "status": "ACCEPTED_NATIVE_PIXEL_CLOCK_WINDOWS_NO_GEOMETRY", "window_count": 32,
        "prepared_manifest_ref": manifest_ref, "native_read_ledger_ref": read_ref,
        **{k: True for k in ("all_observed_worker_births_closed", "all_protected_input_hashes_unchanged",
                             "full_dense_native_pixels_clock_verified", "no_geometry_computed")}}
    oldqa_ref = write(tmp, "oldqa.json", oldqa)
    detector_ref = write(tmp, "detector_control.json", {"fake_detector_only": True})
    model_ref = write(tmp, "model_control.json", {"fake_model_only": True})
    from monocular_osm import native_pixel_mask as masking, native_text_detection as detection
    from monocular_osm import dvlt_native_geometry as geometry, dvlt_native_runtime as runtime
    # Actual current code is read/hash-only. No mutation fixture points at it.
    sources = {name: p.sha(p.ROOT/name) for name in p.SOURCE_PATHS}
    study = {"schema": p.STUDY_SCHEMA, "core_source_hashes": {k: v for k, v in sources.items()
             if k != "monocular_osm/dvlt_native_protocol.py"}, "window_ids": list(range(32)),
        "views_per_window": 7, "one_window_per_actor": True, "require_224_distinct_read_rows": True,
        "original_histories_opaque": True, "old_no_neural_authority_not_upgraded": True,
        "new_neural_study_separate_from_original_numerical_authority": True,
        "ground_truth_used": False, "policy_admission": False, "selected_answer": None,
        "resources": {"cpu": 2, "rss_bytes": 3 << 30, "headroom_bytes": 8 << 30,
                      "inner_seconds": 1800, "outer_seconds": 1830, "parent_seconds": 14400},
        "detector_control_ref": detector_ref, "model_control_ref": model_ref,
        "profiles": {"mask_profile_sha256": masking.PROFILE_SHA256,
                     "detection_profile_sha256": detection.PROFILE_SHA256,
                     "mask_operand_profile_sha256": detection.MASK_OPERAND_PROFILE_SHA256,
                     "geometry_profile_sha256": geometry.PROFILE_SHA256, "model_profile_sha256": runtime.PROFILE_SHA256},
        "input_data": {"original_prepared_manifest_ref": manifest_ref,
                       "original_prepared_input_acceptance_ref": oldqa_ref, "native_read_ledger_ref": read_ref}}
    study_ref = write(tmp, "study.json", study)
    monkeypatch.setattr(p, "STUDY_REF", study_ref)
    qa = {"schema": "root-closed-dvlt-native-data-provenance-v1",
        "status": "ACCEPTED_FIXED32_NATIVE_PIXEL_CLOCK_DATA_FOR_DVLT_STUDY", "study_ref": study_ref,
        **study["input_data"], "window_count": 32, "geometry_or_DVLT_quality_admitted": False,
        **{k: True for k in ("original_data_root_closed", "fixed32_all224_native_frames_verified",
              "historical_paths_remain_opaque", "old_no_neural_or_numerical_authority_not_upgraded")}}
    qa_ref = write(tmp, "new_data_qa.json", qa)
    return {"study": study, "study_ref": study_ref, "sources": sources, "qa": qa, "qa_ref": qa_ref,
            "records": records, "manifest": manifest, "manifest_ref": manifest_ref,
            "detector_ref": detector_ref, "model_ref": model_ref}


def config(tmp, case, phase="mask"):
    value = {"schema": "dvlt-native-mask-window-run-v1" if phase == "mask" else "dvlt-native-local-geometry-window-run-v1",
             "source_hashes": case["sources"], "study_registration_ref": case["study_ref"],
             "data_acceptance_ref": case["qa_ref"], "window_id": 0, "output_dir": str(tmp/("out_"+phase)),
             ("detector_control_ref" if phase == "mask" else "model_control_ref"):
             case["detector_ref" if phase == "mask" else "model_ref"]}
    if phase == "run": value["mask_acceptance_ref"] = case["mask_qa_ref"]
    return write(tmp, phase+"_config.json", value)


def test_placeholder_refuses_before_any_path_access(monkeypatch):
    monkeypatch.setattr(p, "STUDY_REF", None)
    with pytest.raises(ValueError, match="unregistered"):
        p.load_config("DO_NOT_OPEN", "0"*64, "run")


def test_npz_roundtrip_exact_and_late_mutation(tmp_path):
    a = {"x": np.arange(12, dtype=np.float64).reshape(3, 4)}
    packet = p.write_npz(tmp_path/"a.npz", a)
    specs = {"x": ((3, 4), np.dtype("f8"))}
    got = p.read_npz(packet["array_ref"], p.Ledger(), specs, packet["typed_headers"])
    assert np.array_equal(got["x"], a["x"])
    count = 0
    def change():
        nonlocal count
        count += 1
        if count == 3:
            with (tmp_path/"a.npz").open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        p.read_npz(packet["array_ref"], p.Ledger(), specs, packet["typed_headers"], check=change)


def test_zip_all_members_refused_before_first_header(tmp_path, monkeypatch):
    path = tmp_path/"bad.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.npy", b"x")
        z.writestr("b.npy", b"0"*10000)
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *a: pytest.fail("header read before all ZIP caps"))
    with pytest.raises(ValueError, match="budgets"):
        p.read_npz(p.ref(path), p.Ledger(), {"a": ((1,), np.dtype("u1")), "b": ((1,), np.dtype("u1"))}, {}, maximum=1000)


def test_header_length_rejected_before_header_parser(tmp_path, monkeypatch):
    path = tmp_path/"bad.npz"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.npy", b"\x93NUMPY\x01\x00"+(4097).to_bytes(2,"little")+b" "*4097)
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *a: pytest.fail("oversize header parser"))
    with pytest.raises(ValueError, match="header length"):
        p.read_npz(p.ref(path), p.Ledger(), {"x": ((1,), np.dtype("u1"))}, {})


def test_whole_cohort_denominator_and_operational_unrun():
    rows = [{"window_id": i, "status": "not_started", "reason": "parent_deadline_exhausted",
             "metric_scale_or_global_gauge_claimed": False, "policy_admission": False} for i in range(32)]
    assert p.validate_cohort(rows)["counts"]["not_started"] == 32
    with pytest.raises(ValueError, match="all32"):
        p.validate_cohort(rows[:-1])
    rows[0]["reason"] = "geometry_failed"
    with pytest.raises(ValueError, match="operational"):
        p.validate_cohort(rows)


def environment_fixture(monkeypatch):
    import sys
    from types import SimpleNamespace
    from monocular_osm import dvlt_native_runtime as runtime
    versions = {"Torch": "2.1.0+cpu", "Num_Py": "1.2.3"}
    distributions = [SimpleNamespace(metadata={"Name": k}, version=v) for k, v in versions.items()]
    monkeypatch.setattr(runtime.importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: {"torch": "2.1.0+cpu"}[name])
    env = {"schema": "root-isolated-dvlt-exact-core-environment-v1", "python": sys.version,
           "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
           "versions": versions.copy(), "all_official_core_exact_pins_satisfied": True,
           "shared_project_venv_mutated": False, "model_imported_or_forwarded": False}
    return runtime, env, distributions


def test_original_full_environment_with_core_subset_and_historical_scope(monkeypatch):
    runtime, env, _ = environment_fixture(monkeypatch)
    assert runtime.validate_environment(env, ["torch==2.1.0"]) == env["versions"]
    env["model_imported_or_forwarded"] = True
    # This constructor-time flag is provenance, never current neural permission.
    assert runtime.validate_environment(env, ["torch==2.1.0"]) == env["versions"]


@pytest.mark.parametrize("fault", ["python", "executable", "changed", "extra", "duplicate", "core"])
def test_environment_drift_refused_without_model_io(monkeypatch, fault):
    from types import SimpleNamespace
    runtime, env, distributions = environment_fixture(monkeypatch)
    dependencies = ["torch==2.1.0"]
    if fault == "python": env["python"] += " drift"
    if fault == "executable": env["executable"] = str((Path(runtime.__file__).parent/"fake_python.exe").resolve())
    if fault == "changed": env["versions"]["Torch"] = "0.0.0"
    if fault == "extra": distributions.append(SimpleNamespace(metadata={"Name": "new-package"}, version="1"))
    if fault == "duplicate": distributions.append(SimpleNamespace(metadata={"Name": "num-py"}, version="1.2.3"))
    if fault == "core": dependencies = ["torch==2.2.0"]
    with pytest.raises(ValueError): runtime.validate_environment(env, dependencies)
