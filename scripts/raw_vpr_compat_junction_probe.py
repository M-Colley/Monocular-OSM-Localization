"""Separate numeric-compatible diagnostic worker; strict worker remains unchanged."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import raw_vpr_junction_probe as original
from scripts import admit_raw_vpr_compat_bank as admission
from monocular_osm import raw_vpr_compat_bank as bank_core
from monocular_osm import raw_vpr_compat_appearance as appearance
p, component, native = original.p, original.component, original.native
role, Ledger, equal_ref = original.role, original.Ledger, original.equal_ref
prepared_inputs = original.prepared_inputs
INPUTS, PREPARED_MEMBERS = original.INPUTS, original.PREPARED_MEMBERS
SOURCE_PATHS = original.SOURCE_PATHS | admission.SOURCES | {
    "scripts/raw_vpr_compat_junction_probe.py", "monocular_osm/raw_vpr_compat_appearance.py"}
UNION_SPECS = {"ref_ids": (30000, "US", 512), "ref_latlon": (60000, "f", 8),
    "raw_similarities": (2400000, "f", 4), "query_indices": (80, "i", 8), "query_times": (80, "f", 8)}


def sources(config, ledger):
    p.require(SOURCE_PATHS <= set(config["source_hashes"]), "ALL compatible source pins required")
    expected = original.sources(config, ledger)
    for module in (admission, bank_core, appearance):
        p.require(Path(module.__file__).resolve() == ROOT / (module.__name__.replace(".", "/") + ".py"), "foreign compatible module origin")
    return expected


def bank_authority(config, ledger):
    """All metadata/closed-actor gates precede any NPZ member loading."""
    inputs, base = config["inputs"], ledger.base
    report = admission.read_compatible_report(ledger, role(inputs["bank_report"], base))
    facts = ledger.json(inputs["native_facts"])
    qa = ledger.json(inputs["bank_root_acceptance"])
    p.require(qa.get("schema") == "root-closed-raw-vpr-compatible-bank-acceptance-v1"
              and qa.get("status") == "ACCEPTED_COMPLETE_COMPATIBLE_BANKS_OR_EXPLICIT_UNAVAILABLE_NO_POLICY", "Root closed bank acceptance required")
    for name, expected in (("bank_report_ref", inputs["bank_report"]), ("native_facts_ref", inputs["native_facts"]),
                           ("native_root_acceptance_ref", report["native_root_acceptance_ref"])):
        equal_ref(qa[name], expected, base)
    equal_ref(qa["study_registration_ref"], config["study_registration_ref"], base)
    equal_ref(report["study_registration_ref"], config["study_registration_ref"], base)
    p.require(report.get("compatibility_not_bitwise_identity") is True and report.get("historical_encoder_verified") is False, "compatible report claims differ")
    equal_ref(report["native_facts_ref"], inputs["native_facts"], base)
    p.require(all(qa.get(k) is True for k in ("all_observed_worker_births_closed", "all_protected_input_hashes_unchanged"))
              and qa.get("reference_encoding_identity") == "UNVERIFIED"
              and all(qa.get(k) is False for k in ("model_inference_run", "ground_truth_used", "policy_admission"))
              and "selected_answer" in qa and qa["selected_answer"] is None, "bank Root authority must remain closed/input-only")
    registration = ledger.json(qa["external_guard_registration_ref"])
    result = ledger.json(qa["external_guard_result_ref"])
    closure = ledger.json(qa["external_closure_ref"])
    p.require(isinstance(registration, dict) and registration and result.get("status") == "complete"
              and type(result.get("returncode")) is int and result["returncode"] == 0
              and "reason" in result and result["reason"] is None, "bank guard must close rc0")
    p.require(closure.get("schema") == "root-raw-vpr-compatible-bank-external-closure-v1"
              and closure.get("worker_births_closed") is True and closure.get("protected_inputs_unchanged") is True,
              "independent bank worker closure required")
    for key, ref in (("bank_report_ref", inputs["bank_report"]), ("registration_ref", qa["external_guard_registration_ref"]),
                     ("result_ref", qa["external_guard_result_ref"])):
        equal_ref(closure[key], ref, base)
    p.require(report.get("schema") == "root-native-accepted-compatible-raw-vpr-banks-v1"
              and report.get("profile") == admission.PROFILE and report.get("profile_sha256") == admission.PROFILE_SHA256
              and report.get("reference_encoding_identity") == "UNVERIFIED"
              and all(report.get(k) is False for k in ("field_built", "decoder_run", "new_neural_forwards", "network", "ground_truth_used", "policy_admission"))
              and "selected_answer" in report and report["selected_answer"] is None, "genuine research bank report required")
    rows, accepted = report["banks"], qa["banks"]
    p.require(isinstance(rows, list) and 1 <= len(rows) <= 2 and len({r["bank_key"] for r in rows}) == len(rows)
              and isinstance(accepted, list) and len(accepted) == len(rows), "complete bank outcome inventory required")
    for row, bound in zip(rows, accepted):
        complete = row.get("status") == "complete_compatible_research_bank"
        p.require(row.get("status") in {"complete_compatible_research_bank", "UNAVAILABLE"}
                  and bound.get("bank_key") == row["bank_key"] and bound.get("status") == row["status"]
                  and set(bound) == ({"bank_key", "status", "array_artifact_ref"} if complete else {"bank_key", "status"}), "Root/producer bank outcomes differ")
        if complete:
            equal_ref(bound["array_artifact_ref"], {k: row["array_artifact"][k] for k in ("path", "sha256")}, base)
        else:
            p.require("array_artifact" not in row, "unavailable bank carries partial arrays")
    chosen = next((r for r in rows if r["bank_key"] == config["bank_key"]), None)
    p.require(chosen is not None and chosen["status"] == "complete_compatible_research_bank", "chosen bank unavailable; no fallback")
    p.require(chosen.get("profile") == dict(bank_core.PROFILE) and chosen.get("profile_sha256") == bank_core.PROFILE_SHA256,
              "chosen genuine pure-bank profile differs")
    p.require(chosen.get("compatibility_not_bitwise_identity") is True and chosen.get("historical_encoder_verified") is False, "compatible bank claims differ")
    records = [v["source"] for v in chosen["inputs"].values()]
    native.validate_native_facts(facts, records, chosen["bank_key"])
    return report, facts, chosen



def load_bank(chosen, ledger):
    record = chosen["array_artifact"]
    path = ledger.protect({k: record[k] for k in ("path", "sha256")})
    arrays = component._arrays_bound(path, {path: record["sha256"]}, UNION_SPECS)
    headers = record["arrays"]
    p.require(set(headers) == set(arrays) == set(UNION_SPECS), "five exact union array roles required")
    for name, value in arrays.items():
        p.require(headers[name] == {"shape": list(value.shape), "dtype": value.dtype.str, "sha256": native.typed(value)},
                  "union actual typed bytes differ: " + name)
    ids, coords, score, indices, times = (arrays[k] for k in
        ("ref_ids", "ref_latlon", "raw_similarities", "query_indices", "query_times"))
    n = len(ids)
    p.require(ids.ndim == 1 and ids.dtype.kind in "US" and 1 <= n <= 30000
              and len(set(ids.tolist())) == n and coords.shape == (n, 2) and coords.dtype == np.float64
              and np.isfinite(coords).all() and np.all(abs(coords[:, 0]) <= 90) and np.all(abs(coords[:, 1]) <= 180)
              and score.shape == (80, n) and score.dtype == np.float32 and np.isfinite(score).all()
              and np.max(abs(score)) <= 1.0001 and indices.shape == (80,) and indices.dtype == np.int64
              and indices[0] >= 0 and np.all(np.diff(indices) > 0) and times.shape == (80,)
              and times.dtype == np.float64 and np.isfinite(times).all() and np.all(np.diff(times) > 0),
              "bounded complete union shape/dtype/value identity required")
    p.require([r["ref_id"] for r in chosen["mapping"]] == ids.tolist(), "full union physical mapping differs from array columns")
    return {**deepcopy(chosen), **arrays}, path



def execute(config_path, expected, *, check=lambda: None):
    began = time.monotonic()
    def poll():
        check()
        p.require(time.monotonic() - began <= 1800., "complete fused worker exceeded inner deadline")
    config_path = Path(config_path).resolve(strict=True)
    ledger = Ledger(config_path.parent, poll)
    config_ref = {"path": str(config_path), "sha256": expected}
    config = ledger.json(config_ref)
    p.require(set(config) == {"schema", "source_hashes", "inputs", "bank_key", "video_binding", "output_dir", "study_registration_ref"}
              and config["schema"] == "raw-vpr-compatible-junction-probe-run-v1" and set(config["inputs"]) == INPUTS
              and isinstance(config["bank_key"], str) and config["bank_key"], "fixed fused worker configuration required; no extra knobs")
    expected_sources = sources(config, ledger)
    _, _, sequence, graph_filter, visual_odometry, _ = p.dependencies()
    study = admission.admit_study(config, ledger)
    report, facts, chosen = bank_authority(config, ledger)
    native_qa = ledger.json(report["native_root_acceptance_ref"])
    p.require(native_qa.get("schema") == "root-closed-raw-vpr-native-acceptance-v1"
              and native_qa.get("status") == "ACCEPTED_INPUT_PIXELS_CLOCK_ONLY_NO_NN"
              and all(native_qa.get(k) is True for k in ("all_observed_worker_births_closed",
                  "all_protected_input_hashes_unchanged", "native_input_pixels_clock_verified")), "accepted native input chain required")
    equal_ref(native_qa["native_facts_ref"], config["inputs"]["native_facts"], ledger.base)
    equal_ref(config["inputs"]["native_vo"], facts["signed_vo_cache"], ledger.base)
    equal_ref(config["inputs"]["native_clock"], facts["cues_vo"], ledger.base)
    video = config["video_binding"]
    p.require(isinstance(video, dict) and set(video) == {"path", "size", "sha256"}
              and native.is_sha(video["sha256"]) and type(video["size"]) is int and video["size"] > 0
              and Path(video["path"]).is_absolute() and video["sha256"] == facts["video_sha256"]
              and native.protected_map(facts["protected_hashes"]).get(native.key(video["path"])) == video["sha256"],
              "accepted full encoded-video identity required; no video reopened")
    prepared_path, members = prepared_inputs(config, ledger, expected_sources, facts)
    output = Path(config["output_dir"])
    output = (ledger.base / output if not output.is_absolute() else output).resolve()
    union_location = Path(role({k: chosen["array_artifact"][k] for k in ("path", "sha256")}, ledger.base)["path"])
    p.require(not output.is_relative_to(prepared_path.parent) and not output.is_relative_to(union_location.parent)
              and not any(Path(path).is_relative_to(output) for path in ledger.protected), "new output must be separate from protected inputs")
    output.mkdir(parents=True, exist_ok=False)  # Reserve once, before any numeric values or computation.
    # Every Root acceptance, outcome and same-source gate above precedes NPZ values.
    bank, union_path = load_bank(chosen, ledger)
    net = p.load_network(p.load_npz(members["network.npz"]), ledger.json({"path": str(members["network.json"]),
        "sha256": ledger.protected[native.key(members["network.json"])]}), graph_filter)
    matched = ledger.json({"path": str(members["matched.json"]), "sha256": ledger.protected[native.key(members["matched.json"])]})
    motion_path = ledger.protect(config["inputs"]["native_vo"])
    clock_path = ledger.protect(config["inputs"]["native_clock"])
    signed, clock_arrays = p.load_npz(motion_path), p.load_npz(clock_path)
    odometry, status, times = p.load_motion(signed, clock_arrays, "timestamps", video, graph_filter, visual_odometry)
    p.require(native.typed(times) == facts["signed_vo_clock_sha256"] and clock_arrays["K"].shape == (3, 3)
              and clock_arrays["K"].dtype == np.float64 and native.typed(clock_arrays["K"]) == facts["native70_K_sha256"],
              "accepted native motion clock/K identity differs")
    del signed, clock_arrays  # Release full cached poses before particle genealogy allocation.
    runtime_config = sequence.Config(particles=50_000, seed=0, inner_s=1800., junction_prior="conditional-uniform-v3")
    with component.cpu_two_threads() as threads:
        result = sequence.align_with_compatible_appearance(net, odometry, status, times, matched, bank, facts,
                                                config=runtime_config, check=poll)
        thread_proof = deepcopy(threads)
    p.require(result.get("status") in {"complete_diagnostic_only", "unavailable"}
              and result.get("accuracy_admitted") is False and result.get("selection") is None
              and result.get("confidence_calibrated") is False and result.get("config") == asdict(runtime_config)
              and result["profile"].get("appearance_profile_sha256") == appearance.PROFILE_SHA256,
              "fixed genuine diagnostic fusion result required")
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update(native_times=times, motion_status=status, native_step=odometry.step,
                  native_heading=odometry.heading, native_heading_var=odometry.heading_var, native_dt=odometry.dt)
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    metadata.update(study_registration_ref=config["study_registration_ref"], declaration_ref=study["declaration_ref"],
        compatibility_not_bitwise_identity=True, config_ref=config_ref, source_hashes=config["source_hashes"], input_refs=config["inputs"],
        bank_key=config["bank_key"], accepted_bank_metadata=chosen, bank_profile_sha256=bank_core.PROFILE_SHA256,
        union_array_ref={"path": str(union_path), "sha256": chosen["array_artifact"]["sha256"]},
        threadpools=thread_proof, array_keys=sorted(arrays), original_native_clock_sha256=p.typed_sha(times),
        full_video_binding=video, video_bytes_opened_or_verified_by_probe=False,
        descriptor_bodies_reloaded=False, new_neural_forwards=False, historical_encoder_verified=False,
        reference_encoding_identity="UNVERIFIED", ground_truth_used=False, selected_answer=None,
        external_actor_closure_verified=False, clock_semantics="exact original query PTS; verified next composed endpoint with explicit <=1s lag")
    p.save_npz(output / "result.npz", arrays)
    p.require(len(p.json_bytes(metadata)) <= admission.MAX_COMPAT_REPORT_JSON,
              "complete compatible derived report exceeds64MiB; no truncation")
    p.write_json(output / "result.json", metadata)
    generated = {name: p.sha(output / name) for name in ("result.npz", "result.json")}
    master = {"schema": "raw-vpr-compatible-junction-probe-run-freeze-v1", "status": "CLOSED_COMPATIBLE_RAW_VPR_JUNCTION_DIAGNOSTIC_NO_ACCURACY",
        "config_ref": config_ref, "source_hashes": config["source_hashes"], "input_refs": config["inputs"],
        "study_registration_ref": config["study_registration_ref"], "declaration_ref": study["declaration_ref"],
        "compatibility_not_bitwise_identity": True, "bank_key": config["bank_key"], "artifacts": generated, "protected_hashes": dict(ledger.protected),
        "numerical_status": result["status"], "profile": result["profile"], "config": asdict(runtime_config),
        "reference_encoding_identity": "UNVERIFIED", "ground_truth_used": False, "selected_answer": None,
        "accuracy_admitted": False, "external_actor_closure_verified": False,
        "resource_enforcement": "Root external CPU2/RSS3GiB/headroom8GiB/outer1830s guard required"}
    poll()  # Final callback; membership and expected-byte checks below are callback-free.
    p.require({path.name for path in output.iterdir()} == set(generated), "unexpected numerical output membership")
    p.verify_files({**ledger.protected, **{str(output / name): digest for name, digest in generated.items()}})
    p.write_json(output / "frozen_run.json", master)
    return master


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    execute(args.config, args.config_sha256)


if __name__ == "__main__":
    main()
