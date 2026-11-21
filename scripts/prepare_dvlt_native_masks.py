"""One fixed native window: detector-only actor, never imports DVLT/Torch.

Root owns the ordered32 cohort and its14400s parent deadline. A failed window
is retained as an operational outcome; it is not a geometry negative.
"""
from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from monocular_osm import dvlt_native_protocol as p


def open_detector(control, ledger, *, check=lambda: None):
    """Finite local detector control; no download/recognition or model fallback."""
    from monocular_osm import native_text_detection as detection
    p.require(set(control) == {"schema", "model_binding", "params", "package_versions", "dependency_sources",
                              "runtime_identity_sha256"}
              and control["schema"] == "root-frozen-native-text-detector-control-v1", "fixed detector control required")
    binding = control["model_binding"]
    p.require(binding.get("schema") == "externally-frozen-detector-model-binding-v1", "explicit detector model identity required")
    model = binding["model"]
    path = p.safe(model["path"])
    p.require(type(model["size"]) is int and 0 < model["size"] <= 1 << 30
              and path.stat().st_size == model["size"], "bounded detector model size differs")
    ledger.add({"path": model["path"], "sha256": model["sha256"]})
    for name, version in control["package_versions"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        p.require(actual == version, "detector environment version differs")
    sources = control["dependency_sources"]
    p.require(isinstance(sources, dict) and 0 < len(sources) <= 1000, "bounded detector source closure required")
    p.require(sum(p.safe(name).stat().st_size for name in sources) <= 64 << 20, "detector dependency source byte cap")
    for name, digest in sources.items():
        ledger.add({"path": name, "sha256": digest})
    params = control["params"]
    p.require(p.same_ref({"path": params.get("Det.model_path"), "sha256": model["sha256"]},
                        {"path": str(path), "sha256": model["sha256"]})
              and params.get("Global.use_det") is True and params.get("Global.use_cls") is False
              and params.get("Global.use_rec") is False
              and params.get("EngineConfig.onnxruntime.intra_op_num_threads") == 2
              and params.get("EngineConfig.onnxruntime.inter_op_num_threads") == 2,
              "exact Root-frozen local model/ORT2-by-2 detector configuration required")
    check()
    from rapidocr import RapidOCR
    engine = RapidOCR(params=params)
    engine._load_det_model()  # Exact pinned library's own lazy constructor; no forward.
    check()
    adapter = detection.NativeDetectionAdapter(engine, model_binding=binding)
    identity = adapter.identity
    p.require(identity["providers"] == ["CPUExecutionProvider"], "genuine CPU-only detector session required")
    import hashlib, json
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    p.require(identity_sha == control["runtime_identity_sha256"], "actual complete detector runtime identity differs")
    for entry in [*identity["classes"].values(), *identity["public_functions"].values()]:
        role = entry["source"]
        p.require(role is not None and sources.get(role["path"]) == role["sha256"], "actual detector source outside frozen closure")
    return adapter


def process_window(body, detector, *, check=lambda: None):
    """No recognizer: retain every quad and every actual completed prefix."""
    import numpy as np
    from monocular_osm import native_pixel_mask as masking
    before = {k: p.typed(v) for k, v in body.items()}
    masked, masks, quads, mask_quads, offsets, frames = [], [], [], [], [0], []
    failure = None
    for i, native in enumerate(body["frames_bgr"]):
        check()
        result = detector.detect(native, check=check)
        if result.get("status") != "complete_native_detection_only":
            failure = {"view_index": i, "stage": "detector", "result": result}
            break
        raw, operand = result["quads"], result["mask_quads"]
        p.require(p.observed_header(raw) == result["receipt"]["native_quads"]
                  and p.observed_header(operand) == result["receipt"]["mask_quads"], "original observer quad bytes differ")
        boxes = np.ascontiguousarray(raw, dtype=np.float64)
        mask_boxes = np.ascontiguousarray(operand, dtype=np.float64)
        p.require(p.observed_header(boxes.astype(raw.dtype)) == result["receipt"]["native_quads"]
                  and p.observed_header(mask_boxes.astype(operand.dtype)) == result["receipt"]["mask_quads"],
                  "packed F64 must exactly reconstruct original quad dtype/bytes")
        p.require(mask_boxes.shape == boxes.shape, "all raw quads need distinct masking operands")
        value, invalid, receipt = masking.mask_native_bgr(native, mask_boxes)
        masked.append(value); masks.append(invalid); quads.append(boxes); mask_quads.append(mask_boxes)
        offsets.append(offsets[-1]+len(boxes))
        frames.append({"view_index": i, "original_typed_sha256": p.typed(native),
            "masked_typed_sha256": p.typed(value), "mask_typed_sha256": p.typed(invalid),
            "quads_typed_sha256": p.typed(boxes), "mask_quads_typed_sha256": p.typed(mask_boxes),
            "detection_receipt": result["receipt"], "mask_receipt": receipt})
    p.require(all(p.typed(body[k]) == v for k, v in before.items()), "detector/mask changed original input")
    if failure is None:
        arrays = {"masked_bgr": np.stack(masked), "invalid_mask": np.stack(masks),
            "quad_coordinates": np.concatenate(quads), "quad_offsets": np.asarray(offsets, dtype=np.int64),
            "mask_quad_coordinates": np.concatenate(mask_quads),
            **{k: body[k].copy() for k in ("video_ordinals", "timestamps", "native_read_rows")}}
    else:
        # Actual prefix only; never fabricate absent frames as zeros or empty success.
        arrays = {f"frame_{i:02d}_{name}": value for i, values in enumerate(zip(masked, masks, quads, mask_quads))
                  for name, value in zip(("masked_bgr", "invalid_mask", "quads", "mask_quads"), values)}
    receipts = [f["detection_receipt"] for f in frames]
    if failure is not None:
        receipts.append(failure["result"].get("receipt", {}))
    sessions = [r.get("session", {}) for r in receipts]
    neural_forward = (True if any(type(s.get("completed")) is int and s["completed"] > 0 for s in sessions)
                      else False if sessions and all(type(s.get("attempted")) is int and s["attempted"] == 0
                            and type(s.get("completed")) is int and s["completed"] == 0 for s in sessions)
                      else None)
    return arrays, {"status": "PREPARED_NATIVE_MASKED_WINDOW_NO_DVLT_GEOMETRY" if failure is None else "detector_failed",
                    "frames": frames, "failure": failure, "completed_detector_frames": len(frames),
                    "neural_forward_run": neural_forward,
                    "mask_profile_sha256": masking.PROFILE_SHA256,
                    "all_seven_detector_forwards_and_postprocessing_completed": failure is None}


def execute(config_path, expected, *, _detector_factory=None, _guard_factory=None):
    if _guard_factory is None:
        from scripts.prepare_multiview_motion_inputs import RuntimeGuard
        _guard_factory = RuntimeGuard
    check = _guard_factory()
    check()
    cfg, ledger = p.load_config(config_path, expected, "mask")
    records = p.admit_data(cfg, ledger)
    output = p.safe(cfg["output_dir"])
    p.require(all(not Path(name).is_relative_to(output) for name in ledger.protected), "output overlaps protected input")
    output.mkdir(parents=True, exist_ok=False)
    record = records[cfg["window_id"]]
    detector = (_detector_factory or open_detector)(ledger.control, ledger, check=check)
    body = p.read_original_window(record, ledger, check=check)
    arrays, result = process_window(body, detector, check=check)
    del body, detector
    files = {}
    packet = None
    if arrays:
        packet = p.write_npz(output/"masked_window.npz", arrays)
        files[str(output/"masked_window.npz")] = packet["array_ref"]["sha256"]
    manifest = {"schema": "frozen-dvlt-native-mask-window-v1", **result,
        "window_id": cfg["window_id"], "study_registration_ref": cfg["study_registration_ref"],
        "data_acceptance_ref": cfg["data_acceptance_ref"], "detector_control_ref": cfg["detector_control_ref"],
        "source_hashes": cfg["source_hashes"], "original_array_ref": record["array_ref"], "packet": packet,
        **{k: record[k] for k in ("native_read_rows", "video_ordinals", "signed_decoder_times_s")},
        "DVLT_forward_run": False, "geometry_computed": False,
        "policy_admission": False, "selected_answer": None, "external_actor_closure_verified": False,
        "protected_hashes": dict(ledger.protected), "original_historical_refs_are_opaque": True}
    manifest["quad_packing"] = "lossless F64; original raw/operand dtype and typed SHA retained in every detector receipt"
    p.publish(output/"mask_window.json", manifest, ledger, files, check=check)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    execute(args.config, args.config_sha256)
