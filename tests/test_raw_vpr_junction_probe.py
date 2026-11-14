"""Owned tiny disk fixtures and synthetic roads only; no real inputs/NN/GT."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import raw_vpr_junction_probe as w
from monocular_osm import junction_inputs as ji
from monocular_osm import road_sign_clues as clues
from tests.test_junction_sequence_probe import roads
from tests.test_raw_vpr_appearance import case as appearance_case


def write(root, path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(root.resolve()), "private fixture writes only"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(w.p.json_bytes(value))
    return {"path": str(path), "sha256": w.p.sha(path)}


def ref(path):
    return {"path": str(path.resolve()), "sha256": w.p.sha(path)}


def fixture(tmp, monkeypatch):
    _, times, _, _, _, bank, facts = appearance_case()
    sources = {name: w.p.sha(w.ROOT / name) for name in w.SOURCE_PATHS}
    graph, net = roads()
    ll = net.local_to_latlon(np.array([[200., 0.]]))[0]
    plan = ji.plan_inputs("Private synthetic", [1., 2., 1.2, 2.2], bbox_source={
        "city": "Private synthetic", "scope": "named_city_bbox", "source_kind": "public_geocode", "source_sha256": "a" * 64})
    junctions = [{"id": "node/3", "lat": float(ll[0]), "lon": float(ll[1]), "ref": "7A",
                  "normalized_refs": ["7A"], "tags": {"highway": "motorway_junction", "ref": "7A"}}]
    inventory = {"status": "complete", "city": "Private synthetic", "scope": "named_city_bbox",
                 "plan": plan["inventory_plan"], "junctions": junctions, "snapshot_sha256": clues._digest(junctions)}
    literal = clues.extract_road_sign_clues([{"text": "EXIT 7A", "confidence": .99, "t_sec": 4.3,
                                            "bbox": [10., 10., 80., 30.]}], [480, 640])
    raw = tmp / "private.graphml"
    raw.write_bytes(b"private synthetic graph-loader input")
    import osmnx as ox
    monkeypatch.setattr(ox, "load_graphml", lambda _: graph)
    monkeypatch.setattr(ji, "prepare_graph", lambda *a, **k: (net, {"schema": "private-topology"}))
    prep_config = {"schema": "junction-sequence-probe-prepare-v1", "source_hashes": sources,
        "inputs": {"raw_graphml": ref(raw), "junction_inventory": write(tmp, tmp / "inventory.json", inventory),
                   "temporal_ocr": write(tmp, tmp / "ocr.json", {"clues": literal, "origin_video_sha256": "d" * 64})},
        "output_dir": str(tmp / "prepared")}
    pc = write(tmp, tmp / "prepare_config.json", prep_config)
    w.p.prepare(pc["path"], pc["sha256"])
    n = len(times)
    centers = np.column_stack([np.arange(n, dtype=float), np.zeros((n, 2))])
    video = {"path": str((tmp / "never_opened_video.mp4").resolve()), "size": 123, "sha256": "d" * 64}
    signature = {"schema": 2, "vo_algo_version": 3,
        "video": {"path": os.path.normcase(video["path"]), "size": 123},
        "extraction": {"n_frames": n, "stride": 1, "fps": 100.,
                       "timestamps_sha256": hashlib.sha256(times.astype("<f8").tobytes()).hexdigest()}}
    motion = {"centers": centers, "xz": centers[:, [0, 2]], "valid": np.ones(n, bool),
              "n_inliers": np.full(n, 20, np.int64), "rotations": np.repeat(np.eye(3)[None], n, axis=0),
              "translations": -centers, "cache_signature": np.array(json.dumps(signature))}
    clock = {"xz": motion["xz"].copy(), "valid": motion["valid"].copy(), "timestamps": times.copy(), "K": np.eye(3)}
    w.p.save_npz(tmp / "motion.npz", motion)
    w.p.save_npz(tmp / "clock.npz", clock)
    facts.update(signed_vo_cache=ref(tmp / "motion.npz"), cues_vo=ref(tmp / "clock.npz"),
                 native70_K_sha256=w.native.typed(clock["K"]), protected_hashes={video["path"]: video["sha256"]})
    facts_ref = write(tmp, tmp / "native_facts.json", facts)
    native_qa = write(tmp, tmp / "native_root_qa.json", {
        "schema": "root-closed-raw-vpr-native-acceptance-v1", "status": "ACCEPTED_INPUT_PIXELS_CLOCK_ONLY_NO_NN",
        "native_facts_ref": facts_ref, "all_observed_worker_births_closed": True,
        "all_protected_input_hashes_unchanged": True, "native_input_pixels_clock_verified": True})
    bank_dir = tmp / "admitted"
    bank_dir.mkdir()
    arrays = {name: bank.pop(name) for name in w.UNION_SPECS}
    w.p.save_npz(bank_dir / "bank_00.npz", arrays)
    bank["array_artifact"] = {**ref(bank_dir / "bank_00.npz"), "arrays": {
        name: {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": w.native.typed(value)}
        for name, value in arrays.items()}}
    negative = {"bank_key": "expanded", "status": "UNAVAILABLE", "profile_sha256": w.bank_core.PROFILE_SHA256,
                "reason": "private complete bank unavailable; retain outcome"}
    report = {"schema": "root-native-accepted-raw-vpr-banks-v1", "profile": w.admission.PROFILE,
        "profile_sha256": w.admission.PROFILE_SHA256, "native_facts_ref": facts_ref,
        "native_root_acceptance_ref": native_qa, "banks": [bank, negative],
        "external_actor_closure_verified": False, "reference_encoding_identity": "UNVERIFIED",
        **{k: False for k in ("field_built", "decoder_run", "new_neural_forwards", "network", "ground_truth_used", "policy_admission")},
        "selected_answer": None}
    report_ref = write(tmp, bank_dir / "bank_admission.json", report)
    registration = write(tmp, tmp / "bank_guard_registration.json", {"schema": "private-guard-registration"})
    result = write(tmp, tmp / "bank_guard_result.json", {"status": "complete", "returncode": 0, "reason": None})
    closure = write(tmp, tmp / "bank_closure.json", {"schema": "root-raw-vpr-bank-external-closure-v1",
        "bank_report_ref": report_ref, "registration_ref": registration, "result_ref": result,
        "worker_births_closed": True, "protected_inputs_unchanged": True})
    qa = {"schema": "root-closed-raw-vpr-bank-acceptance-v1",
        "status": "ACCEPTED_COMPLETE_BANKS_OR_EXPLICIT_UNAVAILABLE_NO_POLICY", "bank_report_ref": report_ref,
        "native_facts_ref": facts_ref, "native_root_acceptance_ref": native_qa,
        "external_guard_registration_ref": registration, "external_guard_result_ref": result, "external_closure_ref": closure,
        "all_observed_worker_births_closed": True, "all_protected_input_hashes_unchanged": True,
        "banks": [{"bank_key": bank["bank_key"], "status": bank["status"], "array_artifact_ref": ref(bank_dir / "bank_00.npz")},
                  {"bank_key": negative["bank_key"], "status": negative["status"]}],
        "reference_encoding_identity": "UNVERIFIED", "model_inference_run": False,
        "ground_truth_used": False, "policy_admission": False, "selected_answer": None}
    config = {"schema": "raw-vpr-junction-probe-run-v1", "source_hashes": sources,
        "inputs": {"prepared": ref(tmp / "prepared/frozen_prepare.json"), "native_vo": facts["signed_vo_cache"],
                   "native_clock": facts["cues_vo"], "bank_report": report_ref, "native_facts": facts_ref,
                   "bank_root_acceptance": write(tmp, tmp / "bank_root_qa.json", qa)},
        "bank_key": "baseline", "video_binding": video, "output_dir": str(tmp / "run")}
    return config, report, qa


def rebind(tmp, config, report, qa):
    config["inputs"]["bank_report"] = write(tmp, config["inputs"]["bank_report"]["path"], report)
    qa["bank_report_ref"] = config["inputs"]["bank_report"]
    closure = w.p.read_json(qa["external_closure_ref"]["path"])
    closure["bank_report_ref"] = config["inputs"]["bank_report"]
    qa["external_closure_ref"] = write(tmp, qa["external_closure_ref"]["path"], closure)
    config["inputs"]["bank_root_acceptance"] = write(tmp, config["inputs"]["bank_root_acceptance"]["path"], qa)


def execute(tmp, config, **kwargs):
    record = write(tmp, tmp / "run_config.json", config)
    return w.execute(record["path"], record["sha256"], **kwargs)


def forbid_values(monkeypatch):
    monkeypatch.setattr(w.p, "load_npz", lambda *a: pytest.fail("NPZ member before authority"))
    monkeypatch.setattr(w.component, "_arrays_bound", lambda *a: pytest.fail("union values before authority"))


def test_fixed_genuine_entry_on_tiny_owned_inputs_no_descriptors_or_video(tmp_path, monkeypatch):
    config, report, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(w.component, "load_descriptor_rows", lambda *a: pytest.fail("descriptor body reloaded"))
    result = execute(tmp_path, config)
    assert result["numerical_status"] == "complete_diagnostic_only"
    assert result["config"] == {"particles": 50000, "seed": 0, "genealogy_bytes": 1 << 30,
                                 "inner_s": 1800., "junction_prior": "conditional-uniform-v3"}
    assert result["accuracy_admitted"] is False and result["selected_answer"] is None
    assert result["external_actor_closure_verified"] is False
    output = Path(config["output_dir"])
    saved = w.p.read_json(output / "result.json")
    assert saved["accepted_bank_metadata"] == report["banks"][0]
    assert saved["descriptor_bodies_reloaded"] is False and saved["historical_encoder_verified"] is False
    assert [r["query_row"] for r in saved["appearance_events"]] == list(range(80))
    assert len(saved["threadpools"]) >= 2 and all(x["num_threads"] == 2 for x in saved["threadpools"])
    assert set(x.name for x in output.iterdir()) == {"result.npz", "result.json", "frozen_run.json"}
    for name, digest in result["artifacts"].items():
        assert w.p.sha(output / name) == digest
    assert not Path(config["video_binding"]["path"]).exists()


@pytest.mark.parametrize("fault", ["unaccepted", "root_false", "guard_failed", "closure_ref", "outcome",
                                    "array_ref", "native_ref", "unavailable", "pure_profile"])
def test_bank_root_status_refs_and_full_outcomes_precede_any_npz(tmp_path, monkeypatch, fault):
    config, report, qa = fixture(tmp_path, monkeypatch)
    if fault == "unaccepted": qa["status"] = "not accepted"
    if fault == "root_false": qa["all_observed_worker_births_closed"] = False
    if fault == "guard_failed":
        result = w.p.read_json(qa["external_guard_result_ref"]["path"])
        result["returncode"] = 1
        qa["external_guard_result_ref"] = write(tmp_path, qa["external_guard_result_ref"]["path"], result)
    if fault == "outcome": qa["banks"].pop()  # Must preserve even the unchosen negative bank.
    if fault == "array_ref": qa["banks"][0]["array_artifact_ref"]["sha256"] = "f" * 64
    if fault == "native_ref": qa["native_facts_ref"] = {**qa["native_facts_ref"], "sha256": "f" * 64}
    if fault == "unavailable": config["bank_key"] = "expanded"
    if fault == "pure_profile": report["banks"][0]["profile_sha256"] = "f" * 64
    rebind(tmp_path, config, report, qa)
    if fault == "closure_ref":
        closure = w.p.read_json(qa["external_closure_ref"]["path"])
        closure["registration_ref"] = {**closure["registration_ref"], "sha256": "f" * 64}
        qa["external_closure_ref"] = write(tmp_path, qa["external_closure_ref"]["path"], closure)
        config["inputs"]["bank_root_acceptance"] = write(tmp_path, config["inputs"]["bank_root_acceptance"]["path"], qa)
    forbid_values(monkeypatch)
    with pytest.raises(ValueError): execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("fault", ["source_pin", "old_prepare", "motion_ref", "video", "extra_knob"])
def test_same_source_fresh_prepare_and_native_identity_before_npz(tmp_path, monkeypatch, fault):
    config, _, _ = fixture(tmp_path, monkeypatch)
    if fault == "source_pin": config["source_hashes"]["scripts/raw_vpr_junction_probe.py"] = "f" * 64
    if fault == "old_prepare":
        record = config["inputs"]["prepared"]
        prepared = w.p.read_json(record["path"])
        source = str((w.ROOT / "scripts/raw_vpr_junction_probe.py").resolve())
        prepared["protected_hashes"].pop(source)
        config["inputs"]["prepared"] = write(tmp_path, record["path"], prepared)
    if fault == "motion_ref":
        original = Path(config["inputs"]["native_vo"]["path"])
        alias = tmp_path / "different_motion_alias.npz"
        alias.write_bytes(original.read_bytes())
        config["inputs"]["native_vo"] = ref(alias)
    if fault == "video": config["video_binding"]["sha256"] = "f" * 64
    if fault == "extra_knob": config["particles"] = 1
    forbid_values(monkeypatch)
    with pytest.raises(ValueError): execute(tmp_path, config)
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("fault", ["byte", "typed_header", "actual_dtype"])
def test_union_body_typed_headers_and_exact_dtypes_refuse_before_fusion(tmp_path, monkeypatch, fault):
    config, report, qa = fixture(tmp_path, monkeypatch)
    row = report["banks"][0]
    artifact = row["array_artifact"]
    if fault == "byte":
        target = Path(artifact["path"])
        assert target.resolve().is_relative_to(tmp_path.resolve())
        target.write_bytes(b"private corrupt body")
    if fault == "typed_header": artifact["arrays"]["raw_similarities"]["sha256"] = "f" * 64
    if fault == "actual_dtype":
        arrays = w.component._arrays_bound(Path(artifact["path"]), {Path(artifact["path"]): artifact["sha256"]}, w.UNION_SPECS)
        arrays["query_times"] = arrays["query_times"].astype(np.float32)
        target = tmp_path / "different_typed_union.npz"
        w.p.save_npz(target, arrays)
        row["array_artifact"] = {**ref(target), "arrays": {name: {
            "shape": list(value.shape), "dtype": value.dtype.str, "sha256": w.native.typed(value)}
            for name, value in arrays.items()}}
        qa["banks"][0]["array_artifact_ref"] = ref(target)
    rebind(tmp_path, config, report, qa)
    from monocular_osm import junction_sequence
    monkeypatch.setattr(junction_sequence, "align_with_appearance", lambda *a, **k: pytest.fail("fusion before typed union guards"))
    with pytest.raises(ValueError): execute(tmp_path, config)
    assert not (Path(config["output_dir"]) / "frozen_run.json").exists()


@pytest.mark.parametrize("target", ["input", "output", "membership"])
def test_final_callback_drift_cannot_publish_master(tmp_path, monkeypatch, target):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"])
    fired = []
    def check():
        if fired or not (output / "result.json").exists(): return
        path = (Path(config["inputs"]["native_clock"]["path"]) if target == "input" else
                output / ("result.npz" if target == "output" else "late.bin"))
        assert path.resolve().is_relative_to(tmp_path.resolve())
        path.write_bytes(b"private late callback mutation")
        fired.append(path)
    with pytest.raises(ValueError): execute(tmp_path, config, check=check)
    assert fired and not (output / "frozen_run.json").exists()


def test_existing_output_is_refused_before_any_numeric_read(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"])
    output.mkdir()
    (output / "preserve.bin").write_bytes(b"previous attempt")
    forbid_values(monkeypatch)
    with pytest.raises(FileExistsError): execute(tmp_path, config)
    assert (output / "preserve.bin").read_bytes() == b"previous attempt"


def test_source_hash_is_rechecked_after_last_callback_without_editing_real_source(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    output = Path(config["output_dir"])
    target = (w.ROOT / "scripts/raw_vpr_junction_probe.py").resolve()
    original = w.p.sha
    final = []
    def check():
        if (output / "result.json").exists(): final.append(True)
    def changed_hash(path):
        # Simulate a changed source read; never mutate the shared checkout.
        return "f" * 64 if final and Path(path).resolve() == target else original(path)
    monkeypatch.setattr(w.p, "sha", changed_hash)
    with pytest.raises(ValueError, match="protected/generated bytes changed"):
        execute(tmp_path, config, check=check)
    assert final and not (output / "frozen_run.json").exists()
