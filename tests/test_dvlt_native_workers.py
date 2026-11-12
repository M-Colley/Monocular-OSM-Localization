"""Owned tiny fake dependencies only: no OCR/DVLT/model/weights/actual packets."""
from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import numpy as np
import pytest
from monocular_osm import dvlt_native_protocol as p
from scripts import prepare_dvlt_native_masks as mask_worker
from scripts import dvlt_native_window_probe as worker
from tests.test_dvlt_native_protocol import write, packet_fixture, config


class Detector:
    def __init__(self, fail_at=None): self.calls = 0; self.fail_at = fail_at
    def detect(self, image, *, check):
        from monocular_osm import native_text_detection as detection
        check(); i = self.calls; self.calls += 1
        if i == self.fail_at:
            return {"status": "unavailable", "quads": None, "receipt": {"status": "unavailable"}, "reason": "private session failure"}
        h, w = image.shape[:2]
        raw = np.asarray([[[w, 20.], [w, 30.], [w-4., 30.], [w-4., 20.]]], dtype=np.float64)
        operand = raw.copy(); operand[..., 0] = np.minimum(operand[..., 0], w-1)
        return {"status": "complete_native_detection_only", "quads": raw, "mask_quads": operand,
                "receipt": {"schema": "observed-native-detection-only-receipt-v1", "session": {"completed": 1},
                    "profile_sha256": detection.PROFILE_SHA256,
                    "mask_operand_conversion": {"profile_sha256": detection.MASK_OPERAND_PROFILE_SHA256},
                    "postprocess": {"completed": 1}, "public_native_output_completed": True,
                    "all_returned_quads_retained": True, "observers_restored": True,
                    "native_quads": p.observed_header(raw), "mask_quads": p.observed_header(operand)}}


class FakeTransform:
    def __init__(self, hw):
        self.native_hw = tuple(hw); self.resized_hw = (14, 14)
        self.crop_top_left = self.pad_top_left = (0, 0)
    def receipt(self): return {"native_hw": list(self.native_hw), "processed_hw": [14, 14]}


class Geometry:
    PROFILE_SHA256 = "9"*64
    @staticmethod
    def make_transform(hw, processed_hw):
        assert tuple(processed_hw) == (14, 14)
        return FakeTransform(hw)
    @staticmethod
    def track_masked_window(images, masks, ordinals, times, *, check):
        check(); assert images.shape[0] == masks.shape[0] == 7
        assert np.all(images[masks] == 0)
        return {"seed_ids": np.arange(4, dtype=np.int32), "check_seed": np.asarray([0, 1, 0, 1], dtype=np.bool_),
                "points_px": np.zeros((7, 4, 2), dtype=np.float64), "failure_reason": np.full((7, 4), "", dtype="U48")}
    @staticmethod
    def validate_geometry(tracks, c2w, K, transform, *, world_points, check):
        check(); assert c2w.shape == (7, 3, 4) and K.shape == (7, 3, 3)
        assert world_points.shape == (7, 14, 14, 3)
        return {"pose_only": {"projection_valid": np.ones((7, 4), dtype=np.bool_),
                              "errors": np.full((7, 4), np.inf)},
                "summary": {"fixed_check_ids": np.asarray([1, 3], dtype=np.int32), "metric_scale": False,
                            "nonfinite_scalar": float("nan")}}


class Model:
    def __init__(self, *args, check=lambda: None, **kwargs): self.calls = 0
    def infer(self, images, *, check):
        check(); self.calls += 1
        arrays = {name: np.ones((1, 7, 14, 14), dtype=np.float32)
                  for name in ("depths", "depths_conf", "world_points_direct_conf")}
        arrays.update({name: np.ones((1, 7, 14, 14, 3), dtype=np.float32)
                       for name in ("world_points", "world_points_direct")})
        arrays.update(camera_to_worlds=np.repeat(np.eye(4, dtype=np.float32)[None, :3], 7, axis=0),
                      intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 7, axis=0),
                      input_images=np.zeros((1, 7, 3, 14, 14), dtype=np.float32),
                      gradio_valid_pixels=np.ones((1, 7, 14, 14), dtype=np.bool_))
        self.last_input_arrays = {k: arrays[k] for k in ("input_images", "gradio_valid_pixels")}
        return arrays, {"fake_private_model": True, "native_K_supplied_to_model": False}
    def verify(self, *, check):
        check(); return {"fake_private_state": True, "all_state_and_buffers_unchanged": True}


def close_mask(tmp, case, result):
    folder = Path(result["packet"]["array_ref"]["path"]).parent
    subject = p.ref(folder/"mask_window.json")
    registration = write(tmp, "private_mask_registration.json", {"fake_registration": True})
    status = write(tmp, "private_mask_result.json", {"status": "complete", "returncode": 0, "reason": None})
    closure = write(tmp, "private_mask_closure.json", {"schema": "root-dvlt-native-mask-window-external-closure-v1",
        "mask_manifest_ref": subject, "registration_ref": registration, "result_ref": status,
        "worker_births_closed": True, "protected_inputs_unchanged": True})
    case["mask_qa_ref"] = write(tmp, "private_mask_qa.json", {"schema": "root-closed-dvlt-native-mask-window-v1",
        "status": "ACCEPTED_NATIVE_MASKED_WINDOW_NO_DVLT_GEOMETRY", "window_id": 0,
        "study_ref": case["study_ref"], "data_acceptance_ref": case["qa_ref"], "mask_manifest_ref": subject,
        "registration_ref": registration, "result_ref": status, "external_closure_ref": closure,
        "all_observed_worker_births_closed": True, "all_protected_input_hashes_unchanged": True,
        "all_seven_detector_forwards_and_postprocessing_completed": True, "no_DVLT_forward_or_geometry_computed": True})


def prepare(tmp, case):
    cfg = config(tmp, case)
    return mask_worker.execute(cfg["path"], cfg["sha256"], _detector_factory=lambda *a, **k: Detector(),
                               _guard_factory=lambda: lambda: None)


def test_both_phases_closed_mask_before_fake_model(tmp_path, monkeypatch):
    case = packet_fixture(tmp_path, monkeypatch)
    masked = prepare(tmp_path, case)
    close_mask(tmp_path, case, masked)
    cfg = config(tmp_path, case, "run")
    result = worker.execute(cfg["path"], cfg["sha256"], _model_factory=Model, _model_preflight=lambda *a: None,
                            _guard_factory=lambda: lambda: None, _geometry=Geometry)
    assert result["status"] == "complete_local_diagnostic"
    assert result["external_actor_closure_verified"] is False
    details = p.old.strict_json(Path(result["result_ref"]["path"]).read_bytes())
    assert details["DVLT_forward_completed"] and details["CHECK_not_unseen_model_images"]
    assert details["native_K_supplied_to_model"] is False
    assert details["geometry"]["summary"]["nonfinite_scalar"] is None
    assert {"path": "/geometry/summary/nonfinite_scalar", "kind": "nan"} in details["json_nonfinite_path_inventory"]
    with np.load(result["result_array_ref"]["path"], allow_pickle=False) as z:
        assert np.isinf(z["geometry_pose_only_errors"]).all()
        assert np.array_equal(z["geometry_summary_fixed_check_ids"], [1, 3])
        assert z["native_to_processed_affine"].shape == (7, 3, 3)
    with np.load(masked["packet"]["array_ref"]["path"], allow_pickle=False) as z:
        assert z["quad_coordinates"][0, 0, 0] == 96.
        assert z["mask_quad_coordinates"][0, 0, 0] == 95.


def test_unavailable_detector_preserves_only_genuine_prefix(tmp_path, monkeypatch):
    case = packet_fixture(tmp_path, monkeypatch)
    cfg = config(tmp_path, case)
    result = mask_worker.execute(cfg["path"], cfg["sha256"],
        _detector_factory=lambda *a, **k: Detector(fail_at=2), _guard_factory=lambda: lambda: None)
    assert result["status"] == "detector_failed" and result["completed_detector_frames"] == 2
    assert not result["all_seven_detector_forwards_and_postprocessing_completed"]
    with np.load(result["packet"]["array_ref"]["path"], allow_pickle=False) as z:
        assert not any(k.startswith("frame_02") for k in z.files)
        assert "frame_01_mask_quads" in z.files


@pytest.mark.parametrize("field", ["original_data_root_closed", "fixed32_all224_native_frames_verified",
    "historical_paths_remain_opaque", "old_no_neural_or_numerical_authority_not_upgraded"])
def test_data_refusal_precedes_npz_members(tmp_path, monkeypatch, field):
    case = packet_fixture(tmp_path, monkeypatch)
    case["qa"][field] = False
    case["qa_ref"] = write(tmp_path, "negative_data_qa.json", case["qa"])
    cfg = config(tmp_path, case)
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *a: pytest.fail("NPZ before DATA admission"))
    with pytest.raises(ValueError, match="DATA proof"):
        mask_worker.execute(cfg["path"], cfg["sha256"], _detector_factory=lambda *a, **k: pytest.fail("detector before DATA"))


def test_all32_metadata_validated_before_first_window(tmp_path, monkeypatch):
    case = packet_fixture(tmp_path, monkeypatch)
    case["manifest"]["windows"][31]["video_ordinals"] = [999]*7
    new = write(tmp_path, "private_bad_manifest.json", case["manifest"])
    # Direct admission uses fake metadata already bound by the private study/QA.
    case["qa"]["original_prepared_manifest_ref"] = new
    ledger = p.Ledger(); ledger.study = dict(case["study"])
    ledger.study["input_data"] = {**ledger.study["input_data"], "original_prepared_manifest_ref": new}
    case["qa_ref"] = write(tmp_path, "bad_manifest_qa.json", case["qa"])
    with pytest.raises(ValueError):
        p.admit_data({"data_acceptance_ref": case["qa_ref"], "study_registration_ref": case["study_ref"]}, ledger)


def test_raw_loaded_witness_source_functions_exact_s7():
    from monocular_osm import dvlt_native_runtime as runtime
    source = p.ROOT/"output/dvlt_seven_view_feasibility_design_20261111/feasibility_runner_s7_draft.py"
    prior = ast.parse(source.read_text()); current = ast.parse(Path(runtime.__file__).read_text())
    a = {x.name: ast.dump(x) for x in prior.body if isinstance(x, ast.FunctionDef)}
    b = {x.name: ast.dump(x) for x in current.body if isinstance(x, ast.FunctionDef)}
    for key in ("state_digest", "buffer_digest", "checkpoint_state_witness", "require_cpu_eval", "source_check"):
        assert a[key] == b[key]


def test_callback_before_final_hash_and_no_manifest_on_mutation(tmp_path):
    protected = write(tmp_path, "protected.json", {"private": True})
    ledger = p.Ledger(); ledger.add(protected)
    generated = write(tmp_path, "out/result.json", {"private": True})
    def mutate():
        target = Path(generated["path"]).resolve()
        assert target.is_relative_to(tmp_path.resolve())
        target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="generated bytes"):
        p.publish(tmp_path/"out/frozen_run.json", {}, ledger, {generated["path"]: generated["sha256"]}, check=mutate)
    assert not (tmp_path/"out/frozen_run.json").exists()


def test_numerical_phase_never_opens_stats_or_hashes_original_npz(tmp_path, monkeypatch):
    case = packet_fixture(tmp_path, monkeypatch)
    masked = prepare(tmp_path, case); close_mask(tmp_path, case, masked)
    cfg = config(tmp_path, case, "run")
    forbidden = {str(Path(r["array_ref"]["path"])) for r in case["records"]}
    original_open, original_stat = Path.open, Path.stat
    def guarded_open(path, *a, **k):
        assert str(path) not in forbidden and "NEVER_OPEN" not in str(path)
        return original_open(path, *a, **k)
    def guarded_stat(path, *a, **k):
        assert str(path) not in forbidden and "NEVER_OPEN" not in str(path)
        return original_stat(path, *a, **k)
    monkeypatch.setattr(Path, "open", guarded_open); monkeypatch.setattr(Path, "stat", guarded_stat)
    result = worker.execute(cfg["path"], cfg["sha256"], _model_factory=Model, _model_preflight=lambda *a: None,
                            _guard_factory=lambda: lambda: None, _geometry=Geometry)
    assert result["historical_original_refs_opaque"]
    assert all(name not in forbidden for name in result["protected_hashes"])


def tiny_masked_body():
    return {"masked_bgr": np.zeros((7, 64, 96, 3), dtype=np.uint8),
            "invalid_mask": np.ones((7, 64, 96), dtype=np.bool_),
            "video_ordinals": np.arange(7, dtype=np.int64)*4,
            "timestamps": np.arange(1, 8, dtype=np.float64)*.1,
            "native_read_rows": np.arange(7, dtype=np.int64)}


def test_completed_predictions_retained_when_state_identity_fails():
    class ChangedState(Model):
        def verify(self, *, check): raise ValueError("private changed weights")
    arrays, report = worker.process_window(tiny_masked_body(), ChangedState(), _geometry=Geometry)
    assert "world_points" in arrays and "geometry_summary_fixed_check_ids" in arrays
    assert report["status"] == "model_failed"
    assert report["model_state"]["all_state_and_buffers_unchanged"] is False
    assert report["failures"][-1]["stage"] == "model_state"


def test_original_observation_exception_not_overwritten():
    class NoTracks(Geometry):
        @staticmethod
        def track_masked_window(*args, **kwargs): raise ValueError("private original observation error")
    arrays, report = worker.process_window(tiny_masked_body(), Model(), _geometry=NoTracks)
    assert "world_points" in arrays and report["status"] == "check_unavailable"
    assert report["failure"]["message"] == "private original observation error"
    assert [row["stage"] for row in report["failures"]] == ["observations", "geometry"]


def test_caught_observation_deadline_does_not_enter_model():
    stopped = False
    class Deadline(Geometry):
        @staticmethod
        def track_masked_window(*args, **kwargs):
            nonlocal stopped
            stopped = True
            raise RuntimeError("private deadline")
    def check():
        if stopped: raise RuntimeError("private deadline remains exhausted")
    class NeverModel(Model):
        def infer(self, *args, **kwargs): pytest.fail("unguarded model preprocessing after deadline")
    with pytest.raises(RuntimeError, match="exhausted"):
        worker.process_window(tiny_masked_body(), NeverModel(), check=check, _geometry=Deadline)


def test_input_mutation_during_final_output_hash_refused(tmp_path, monkeypatch):
    protected = write(tmp_path, "protected.json", {"private": True})
    generated = write(tmp_path, "out/result.json", {"private": True})
    ledger = p.Ledger(); ledger.add(protected)
    original_sha = p.sha
    def hash_and_change(path):
        value = original_sha(path)
        if str(path) == generated["path"]:
            target = Path(protected["path"]).resolve()
            assert target.is_relative_to(tmp_path.resolve())
            target.write_bytes(b"changed after output hash")
        return value
    monkeypatch.setattr(p, "sha", hash_and_change)
    with pytest.raises(ValueError, match="protected bytes"):
        p.publish(tmp_path/"out/frozen_run.json", {}, ledger, {generated["path"]: generated["sha256"]})
    assert not (tmp_path/"out/frozen_run.json").exists()


def test_genuine_geometry_lost_tracks_keeps_invalid_full_seed_arrays():
    import cv2
    from monocular_osm import dvlt_native_geometry as geometry
    rng = np.random.default_rng(12)
    frames = rng.integers(0, 256, (7, 512, 768, 3), dtype=np.uint8)
    masks = np.zeros((7, 512, 768), dtype=np.bool_)
    masks[:, :128] = True; masks[:, -128:] = True; masks[1] = True
    frames[masks] = 0
    body = {**tiny_masked_body(), "masked_bgr": frames, "invalid_mask": masks}
    class GridModel(Model):
        def infer(self, images, *, check):
            pred, receipt = super().infer(images, check=check)
            h, w = 336, 504
            for key in ("depths", "depths_conf", "world_points_direct_conf"):
                pred[key] = np.ones((1, 7, h, w), dtype=np.float32)
            for key in ("world_points", "world_points_direct"):
                pred[key] = np.ones((1, 7, h, w, 3), dtype=np.float32)
            pred["input_images"] = np.zeros((1, 7, 3, h, w), dtype=np.float32)
            pred["gradio_valid_pixels"] = np.ones((1, 7, h, w), dtype=np.bool_)
            return pred, receipt
    old_threads = cv2.getNumThreads()
    try:
        cv2.setNumThreads(2)
        arrays, report = worker.process_window(body, GridModel(), _geometry=geometry)
    finally:
        cv2.setNumThreads(old_threads)
    n = len(arrays["tracks_seed_ids"])
    assert n > 0 and arrays["tracks_valid"].shape == (7, n)
    assert not arrays["tracks_valid"][1].any()
    assert arrays["tracks_rejection_reason"].shape == (7, n)
    assert report["status"] == "check_unavailable" and report["DVLT_forward_completed"]
    assert np.isinf(arrays["geometry_pose_only_errors_with_invalid_inf"]).any()
    assert arrays["geometry_pose_only_rejection_reason"].shape == (7, n)


def test_model_environment_preflight_refuses_before_mask_npz_members(tmp_path, monkeypatch):
    case = packet_fixture(tmp_path, monkeypatch)
    masked = prepare(tmp_path, case); close_mask(tmp_path, case, masked)
    cfg = config(tmp_path, case, "run")
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *a: pytest.fail("masked NPZ before model/source/env gate"))
    def refuse(*args): raise ValueError("private model environment refused")
    with pytest.raises(ValueError, match="environment refused"):
        worker.execute(cfg["path"], cfg["sha256"], _model_preflight=refuse,
            _model_factory=lambda *a, **k: pytest.fail("model constructed"), _guard_factory=lambda: lambda: None)


@pytest.mark.parametrize("flag", ["all_observed_worker_births_closed", "all_protected_input_hashes_unchanged",
    "all_seven_detector_forwards_and_postprocessing_completed", "no_DVLT_forward_or_geometry_computed"])
def test_mask_root_closure_refused_before_model_or_npz(tmp_path, monkeypatch, flag):
    case = packet_fixture(tmp_path, monkeypatch)
    masked = prepare(tmp_path, case); close_mask(tmp_path, case, masked)
    qa = p.old.strict_json(Path(case["mask_qa_ref"]["path"]).read_bytes()); qa[flag] = False
    case["mask_qa_ref"] = write(tmp_path, "negative_mask_qa.json", qa)
    cfg = config(tmp_path, case, "run")
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *a: pytest.fail("NPZ before closed mask"))
    with pytest.raises(ValueError):
        worker.execute(cfg["path"], cfg["sha256"], _model_factory=lambda *a, **k: pytest.fail("model before closed mask"),
                       _guard_factory=lambda: lambda: None)


def test_source_guard_runs_before_detector_or_data_payload(tmp_path, monkeypatch):
    case = packet_fixture(tmp_path, monkeypatch)
    role = config(tmp_path, case)
    value = p.old.strict_json(Path(role["path"]).read_bytes())
    value["source_hashes"]["monocular_osm/dvlt_native_runtime.py"] = "a"*64
    role = write(tmp_path, "wrong_source_config.json", value)
    monkeypatch.setattr(np.lib.format, "read_magic", lambda *a: pytest.fail("NPZ before executing source guard"))
    with pytest.raises(ValueError, match="protected bytes"):
        mask_worker.execute(role["path"], role["sha256"], _detector_factory=lambda *a, **k: pytest.fail("detector before sources"),
                            _guard_factory=lambda: lambda: None)


def test_mask_worker_with_genuine_adapter_public_pipeline_and_fake_session(tmp_path, monkeypatch):
    from monocular_osm import native_text_detection as detection
    from tests.test_native_text_detection import engine
    case = packet_fixture(tmp_path, monkeypatch)
    fake_engine, binding = engine(tmp_path)
    adapter = detection.NativeDetectionAdapter(fake_engine, model_binding=binding)
    cfg = config(tmp_path, case)
    result = mask_worker.execute(cfg["path"], cfg["sha256"], _detector_factory=lambda *a, **k: adapter,
                                 _guard_factory=lambda: lambda: None)
    assert result["completed_detector_frames"] == 7
    assert fake_engine.text_det.session.calls == 7
    assert fake_engine.text_cls is fake_engine.text_rec is None
    for frame in result["frames"]:
        assert frame["detection_receipt"]["session"]["completed"] == 1
        assert frame["detection_receipt"]["native_quads"]["dtype"] == "<f4"
    close_mask(tmp_path, case, result)
    numerical = config(tmp_path, case, "run")
    cfg_body, ledger = p.load_config(numerical["path"], numerical["sha256"], "run")
    records = p.admit_data(cfg_body, ledger)
    accepted = p.admit_mask(cfg_body, ledger, records[0])
    body = p.read_masked_window(accepted, ledger)
    assert body["quad_coordinates"].dtype == np.float64
    assert p.observed_header(body["quad_coordinates"][:1].astype(np.float32)) == result["frames"][0]["detection_receipt"]["native_quads"]


@pytest.mark.parametrize("session,expected", [({"attempted": 1, "completed": 1}, True),
    ({"attempted": 0, "completed": 0}, False), ({"attempted": 1, "completed": 0}, None)])
def test_first_frame_failure_retains_completed_forward_scope(session, expected):
    class FailedDetection:
        def detect(self, native, *, check):
            return {"status": "unavailable", "quads": None, "reason": "private failed postprocessing",
                    "receipt": {"session": session}}
    body = {**tiny_masked_body(), "frames_bgr": tiny_masked_body()["masked_bgr"]}
    arrays, report = mask_worker.process_window(body, FailedDetection())
    assert not arrays and not report["frames"]
    assert report["neural_forward_run"] is expected and report["failure"]["view_index"] == 0
