"""One Root-admitted masked seven-view public DVLT diagnostic; no raw history."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from monocular_osm import dvlt_native_protocol as p


def pack_report(value, arrays, prefix, *, nonfinite=None, json_path=None):
    """Retain every observer ndarray, including invalids, IDs and reasons."""
    import numpy as np
    json_path = "/"+prefix if json_path is None else json_path
    if isinstance(value, np.ndarray):
        name = prefix.replace("/", "_")
        p.require(name not in arrays and not value.dtype.hasobject, "unique numeric observer role required")
        arrays[name] = np.ascontiguousarray(value)
        return {"array_role": name, "typed_header": p.typed_header(arrays[name])}
    if isinstance(value, np.generic):
        return pack_report(value.item(), arrays, prefix, nonfinite=nonfinite, json_path=json_path)
    if isinstance(value, dict):
        return {str(k): pack_report(v, arrays, prefix+"/"+str(k), nonfinite=nonfinite,
                json_path=json_path+"/"+str(k).replace("~", "~0").replace("/", "~1")) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack_report(v, arrays, prefix+"/"+str(i), nonfinite=nonfinite,
                            json_path=json_path+"/"+str(i)) for i, v in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        if nonfinite is not None:
            nonfinite.append({"path": json_path, "kind": "nan" if np.isnan(value) else "+inf" if value > 0 else "-inf"})
        return None
    p.require(value is None or isinstance(value, (str, bool, int, float)), "plain observer metadata required")
    return value


def process_window(body, model, *, check=lambda: None, _geometry=None):
    import numpy as np
    if _geometry is None:
        from monocular_osm import dvlt_native_geometry as _geometry
    before = {k: p.typed(v) for k, v in body.items()}
    arrays = {k: body[k].copy() for k in ("video_ordinals", "timestamps", "native_read_rows")}
    result = {"status": "check_unavailable", "failure": None, "DVLT_forward_completed": False,
              "geometry_profile_sha256": _geometry.PROFILE_SHA256,
              "native_K_supplied_to_model": False, "metric_scale_recovered": False,
              "shared_cross_window_gauge": False, "ground_truth_used": False,
              "policy_admission": False, "selected_answer": None, "json_nonfinite_path_inventory": [], "failures": []}
    def failure(stage, error):
        row = {"stage": stage, "exception_type": type(error).__name__, "message": str(error)}
        result["failures"].append(row)
        if result["failure"] is None:
            result["failure"] = row
    check()
    try:
        tracks = _geometry.track_masked_window(body["masked_bgr"], body["invalid_mask"],
                                              body["video_ordinals"], body["timestamps"], check=check)
    except Exception as error:
        tracks = None
        failure("observations", error)
    else:
        result["observations"] = pack_report(tracks, arrays, "tracks", nonfinite=result["json_nonfinite_path_inventory"],
                                              json_path="/observations")
    check()  # Do not turn a caught observation deadline into unguarded preprocessing.
    try:
        prediction, receipt = model.infer(body["masked_bgr"], check=check)
    except Exception as error:
        result["status"] = "model_failed"
        failure("model", error)
        for key, value in getattr(model, "last_input_arrays", {}).items():
            arrays[key] = value
        completed = getattr(model, "last_prediction_arrays", None)
        if completed is not None:
            p.validate_public_prediction(completed)
            arrays.update(completed)
        result["DVLT_forward_completed"] = bool(getattr(model, "forward_completed", False))
    else:
        p.validate_public_prediction(prediction)
        arrays.update(prediction)
        result.update(DVLT_forward_completed=True, model_receipt=receipt)
        transform = _geometry.make_transform(body["masked_bgr"].shape[1:3], prediction["input_images"].shape[-2:])
        result["preprocessing"] = transform.receipt()
        h, w = transform.native_hw; rh, rw = transform.resized_hw
        top, left = transform.crop_top_left; pt, pl = transform.pad_top_left
        affine = np.asarray([[rw/w, 0., .5*(rw/w)-left+pl],
                             [0., rh/h, .5*(rh/h)-top+pt], [0., 0., 1.]])
        arrays["native_to_processed_affine"] = np.repeat(affine[None], 7, axis=0)
        try:
            if tracks is None:
                raise ValueError("observed track extraction unavailable; no geometry result fabricated")
            geometry = _geometry.validate_geometry(tracks, prediction["camera_to_worlds"], prediction["intrinsics"],
                transform, world_points=prediction["world_points"][0], check=check)
        except Exception as error:
            failure("geometry", error)
        else:
            result["geometry"] = pack_report(geometry, arrays, "geometry", nonfinite=result["json_nonfinite_path_inventory"])
            result["status"] = ("check_unavailable" if geometry.get("pose_only", {}).get("status") == "unavailable"
                                and geometry.get("model_depth_pose", {}).get("status") == "unavailable"
                                else "complete_local_diagnostic")
    check()
    p.require(all(p.typed(body[k]) == value for k, value in before.items()), "observer/model changed native sanitized inputs")
    try:
        result["model_state"] = model.verify(check=check)
    except Exception as error:
        failure("model_state", error)
        result["model_state"] = {"all_state_and_buffers_unchanged": False,
                                 "state_identity_verification_unavailable": True}
        result["status"] = "model_failed"
    p.require(sum(v.nbytes for v in arrays.values()) <= p.MAX_TYPED_OUTPUT, "retained numerical outputs exceed payload budget")
    result["NPZ_nonfinite_values_preserved"] = True
    result["original_pixels_and_masks_retained_by_input_packet_ref"] = True
    result["CHECK_not_unseen_model_images"] = True
    return arrays, result


def execute(config_path, expected, *, _model_factory=None, _model_preflight=None, _guard_factory=None, _geometry=None):
    from monocular_osm import dvlt_native_runtime as runtime
    if _guard_factory is None:
        from scripts.prepare_multiview_motion_inputs import RuntimeGuard
        _guard_factory = RuntimeGuard
    check = _guard_factory()
    check()
    cfg, ledger = p.load_config(config_path, expected, "run")
    records = p.admit_data(cfg, ledger)
    mask = p.admit_mask(cfg, ledger, records[cfg["window_id"]])
    (_model_preflight or runtime.preflight)(ledger.control, ledger)
    check()
    output = p.safe(cfg["output_dir"])
    p.require(all(not Path(name).is_relative_to(output) for name in ledger.protected), "output overlaps protected role")
    output.mkdir(parents=True, exist_ok=False)
    body = p.read_masked_window(mask, ledger, check=check)
    model = (_model_factory or runtime.PublicModel)(ledger.control, ledger, check=check)
    arrays, result = process_window(body, model, check=check, _geometry=_geometry)
    del body, model
    packet = p.write_npz(output/"result.npz", arrays, payload_maximum=p.MAX_TYPED_OUTPUT)
    files = {str(output/"result.npz"): packet["array_ref"]["sha256"]}
    result.update(schema="dvlt-native-local-geometry-window-result-v1", window_id=cfg["window_id"], packet=packet)
    p.require(len(p.json_bytes(result)) <= p.MAX_JSON, "bounded complete diagnostic metadata required")
    p.write_json(output/"result.json", result)
    files[str(output/"result.json")] = p.sha(output/"result.json")
    manifest = {"schema": "frozen-dvlt-native-local-geometry-window-v1", "status": result["status"],
        "window_id": cfg["window_id"], "source_hashes": cfg["source_hashes"],
        "study_registration_ref": cfg["study_registration_ref"], "data_acceptance_ref": cfg["data_acceptance_ref"],
        "mask_acceptance_ref": cfg["mask_acceptance_ref"], "mask_manifest_ref": ledger.mask_manifest_ref,
        "model_control_ref": cfg["model_control_ref"], "result_ref": p.ref(output/"result.json"),
        "result_array_ref": packet["array_ref"], "protected_hashes": dict(ledger.protected),
        "historical_original_refs_opaque": True, "new_neural_study_authority_only": True,
        "external_actor_closure_verified": False, "metric_scale_or_global_gauge_claimed": False,
        "ground_truth_used": False, "policy_admission": False, "selected_answer": None}
    p.publish(output/"frozen_run.json", manifest, ledger, files, check=check)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    execute(args.config, args.config_sha256)
