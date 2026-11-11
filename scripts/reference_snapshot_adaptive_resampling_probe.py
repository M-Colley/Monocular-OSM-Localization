"""Explicit target-ESS adaptive auxiliary resampling over the original run9 target.

All target factors and native heading variance remain unchanged.
No model, labels, re-encoding, policy admission or fallback.
"""
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
from scripts import reference_snapshot_future_guide_probe as original
from monocular_osm import snapshot_future_guide as future_core
from monocular_osm import adaptive_auxiliary_resampling as resampling_core
from monocular_osm import junction_heading_likelihood as heading_core
p, component, native = original.p, original.component, original.native
admission, bank_core, appearance = original.admission, original.bank_core, original.appearance
role, Ledger, equal_ref = original.role, original.Ledger, original.equal_ref
prepared_inputs, bank_authority, load_bank = original.prepared_inputs, original.bank_authority, original.load_bank
INPUTS, PREPARED_MEMBERS, UNION_SPECS = original.INPUTS, original.PREPARED_MEMBERS, original.UNION_SPECS
SOURCE_PATHS = original.SOURCE_PATHS | {
    "scripts/reference_snapshot_adaptive_resampling_probe.py", "monocular_osm/adaptive_auxiliary_resampling.py"}
HEADING_MODE = "normalized-junction-v1"
HEADING_STUDY_REF = {"path": "E:/monocular-osm-joint-vpr-proof/run8/heading_study_contract.json",
    "sha256": "e7e6188c3c01909f8f089ddda70b4fc947b81f5f130a567c1101db5d2fd980a3"}


def sources(config, ledger):
    p.require(SOURCE_PATHS <= set(config["source_hashes"]), "ALL adaptive-resampling sources required")
    expected = original.sources(config, ledger)
    for module in (heading_core, future_core, resampling_core, original):
        p.require(Path(module.__file__).resolve() == ROOT / (module.__name__.replace(".", "/") + ".py"),
                  "foreign adaptive-resampling helper/worker origin")
    return expected


def admit_heading_study(config, ledger):
    equal_ref(config["heading_study_registration_ref"], HEADING_STUDY_REF, ledger.base)
    study = ledger.json(config["heading_study_registration_ref"])
    p.require(study.get("schema") == "root-prospective-normalized-junction-heading-snapshot-study-v1",
              "prospective heading-control declaration required")
    return study

FUTURE_STUDY_REF = {"path": "E:/monocular-osm-joint-vpr-proof/run9/future_guide_study_contract_v2.json",
    "sha256": "da673b3f0316417c53cccb9e8f8868971c236910e3b8ad759ad41c5c58dd87fd"}
TARGETS = frozenset(("legacy", "normalized-junction-v1"))


def admit_future_study(config, ledger):
    equal_ref(config["future_guide_study_registration_ref"], FUTURE_STUDY_REF, ledger.base)
    study = ledger.json(config["future_guide_study_registration_ref"])
    p.require(study.get("schema") == "root-prospective-snapshot-future-appearance-proposal-study-v1",
              "prospective four-control future-proposal study required")
    return study


ADAPTIVE_RESAMPLING_STUDY_REF = {
    "path": "E:/monocular-osm-joint-vpr-proof/run12/adaptive_resampling_study_contract.json",
    "sha256": "e38f59c9c682955a2de4e5a3874860bb00b710b7979333b4df7323ffe61fa990"}
RESAMPLING_MODE = "target-ess-half-v1"


def admit_adaptive_resampling_study(config, ledger):
    equal_ref(config["adaptive_resampling_study_registration_ref"], ADAPTIVE_RESAMPLING_STUDY_REF, ledger.base)
    study = ledger.json(config["adaptive_resampling_study_registration_ref"])
    p.require(study.get("schema") == "root-prospective-adaptive-ESS-future-snapshot-study-v1",
              "prospective four-control adaptive resampling study required")
    return study


def resampling_diagnostics(result, count):
    """Verify the declared timing/types without recomputing any particle path."""
    complete = result["status"] == "complete_diagnostic_only"
    names = ("resampled_after_step", "initial_ancestor_count")
    present = [name in result for name in names]
    p.require(all(present) if complete else not any(present),
              "complete adaptive outcome needs both diagnostics; unavailable must not invent them")
    if not complete:
        return False
    flags, ancestors = (result[name] for name in names)
    times = result["times"]
    p.require(isinstance(times, np.ndarray) and times.dtype == np.float64 and times.ndim == 1
              and 2 <= len(times) <= 1201 and np.isfinite(times).all() and np.all(np.diff(times) > 0),
              "complete adaptive diagnostics need the exact ordered composed clock")
    p.require(isinstance(flags, np.ndarray) and flags.dtype == np.bool_ and flags.shape == times.shape
              and isinstance(ancestors, np.ndarray) and ancestors.dtype == np.int32
              and ancestors.shape == times.shape and 2 <= len(times) <= 1201
              and not bool(flags[-1]) and int(ancestors[0]) == count
              and np.all(ancestors >= 1) and np.all(ancestors <= count)
              and np.all(ancestors[1:] <= ancestors[:-1]),
              "typed complete resampling/initial-label diagnostics required")
    p.require(np.all(ancestors[1:][~flags[:-1]] == ancestors[:-1][~flags[:-1]]),
              "a skipped after-step decision must retain the same initial labels")
    return True


def execute(config_path, expected, *, check=lambda: None):
    began = time.monotonic()
    def poll():
        check()
        p.require(time.monotonic() - began <= 1800., "complete fused worker exceeded inner deadline")
    config_path = Path(config_path).resolve(strict=True)
    ledger = Ledger(config_path.parent, poll)
    config_ref = {"path": str(config_path), "sha256": expected}
    config = ledger.json(config_ref)
    p.require(set(config) == {"schema", "source_hashes", "inputs", "bank_key", "video_binding", "output_dir", "study_registration_ref", "heading_study_registration_ref", "future_guide_study_registration_ref", "adaptive_resampling_study_registration_ref", "heading_target"}
              and config["schema"] == "reference-snapshot-adaptive-resampling-probe-run-v1" and set(config["inputs"]) == INPUTS
              and isinstance(config["bank_key"], str) and config["bank_key"]
              and type(config["heading_target"]) is str and config["heading_target"] in TARGETS, "fixed fused worker configuration required; no extra knobs")
    expected_sources = sources(config, ledger)
    admit_heading_study(config, ledger)
    admit_future_study(config, ledger)
    admit_adaptive_resampling_study(config, ledger)  # All4 declarations precede ANY numeric member.
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
        entry = (sequence.align_with_snapshot_future_guide_adaptive_resampling if config["heading_target"] == "legacy" else
                 sequence.align_with_snapshot_normalized_heading_future_guide_adaptive_resampling)
        result = entry(net, odometry, status, times, matched, bank, facts,
                                                config=runtime_config, check=poll)
        thread_proof = deepcopy(threads)
    p.require(result.get("status") in {"complete_diagnostic_only", "unavailable"}
              and result.get("accuracy_admitted") is False and result.get("selection") is None
              and result.get("confidence_calibrated") is False and result.get("config") == asdict(runtime_config)
              and result["profile"].get("appearance_profile_sha256") == appearance.PROFILE_SHA256
              and result["profile"].get("base_heading_target") == config["heading_target"]
              and result["profile"].get("future_guide_profile_sha256") == future_core.PROFILE_SHA256
              and result["profile"].get("resampling_mode") == RESAMPLING_MODE
              and result["profile"].get("adaptive_resampling_profile_sha256") == resampling_core.PROFILE_SHA256
              and result["profile"].get("target_ESS_precedes_guide") is True
              and result["profile"].get("native_heading_variance_unchanged") is True
              and result["profile"].get("proposal_only") is True
              and result["profile"].get("appearance_target_unchanged") is True,
              "fixed genuine diagnostic fusion result required")
    if config["heading_target"] == "normalized-junction-v1":
        p.require(result["profile"].get("heading_mode") == HEADING_MODE
                  and result["profile"].get("heading_profile_sha256") == heading_core.PROFILE_SHA256
                  and result["profile"].get("exact_marginalized_arc_convolution") is False
                  and result["profile"].get("exact_bias_posterior_moments") is False,
                  "normalized heading target differs")
    else:
        p.require("heading_mode" not in result["profile"], "legacy target silently changed")
    diagnostics_exported = resampling_diagnostics(result, runtime_config.particles)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update(native_times=times, motion_status=status, native_step=odometry.step,
                  native_heading=odometry.heading, native_heading_var=odometry.heading_var, native_dt=odometry.dt)
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    metadata.update(heading_study_registration_ref=config["heading_study_registration_ref"],
        heading_target=config["heading_target"],
        future_guide_study_registration_ref=config["future_guide_study_registration_ref"],
        future_guide_profile_sha256=future_core.PROFILE_SHA256, proposal_only=True,
        adaptive_resampling_study_registration_ref=config["adaptive_resampling_study_registration_ref"],
        resampling_mode=RESAMPLING_MODE, adaptive_resampling_profile_sha256=resampling_core.PROFILE_SHA256,
        resampling_diagnostics_exported=diagnostics_exported, native_heading_variance_unchanged=True,
        heading_profile_sha256=(heading_core.PROFILE_SHA256 if config["heading_target"] != "legacy" else None),
        study_registration_ref=config["study_registration_ref"], declaration_ref=study["declaration_ref"],
        descriptor_equivalence_assumed=False, old_strict_or_numeric_status_upgraded=False, config_ref=config_ref, source_hashes=config["source_hashes"], input_refs=config["inputs"],
        bank_key=config["bank_key"], accepted_bank_metadata=chosen, bank_profile_sha256=bank_core.PROFILE_SHA256,
        union_array_ref={"path": str(union_path), "sha256": chosen["array_artifact"]["sha256"]},
        threadpools=thread_proof, array_keys=sorted(arrays), original_native_clock_sha256=p.typed_sha(times),
        full_video_binding=video, video_bytes_opened_or_verified_by_probe=False,
        descriptor_bodies_reloaded=False, new_neural_forwards=False, historical_encoder_verified=False,
        reference_encoding_identity="UNVERIFIED", ground_truth_used=False, selected_answer=None,
        external_actor_closure_verified=False, clock_semantics="exact original query PTS; verified next composed endpoint with explicit <=1s lag")
    p.save_npz(output / "result.npz", arrays)
    p.require(len(p.json_bytes(metadata)) <= admission.MAX_REPORT_JSON,
              "complete snapshot derived report exceeds64MiB; no truncation")
    p.write_json(output / "result.json", metadata)
    generated = {name: p.sha(output / name) for name in ("result.npz", "result.json")}
    master = {"schema": "reference-snapshot-adaptive-resampling-probe-run-freeze-v1", "status": "CLOSED_REFERENCE_SNAPSHOT_ADAPTIVE_RESAMPLING_DIAGNOSTIC_NO_ACCURACY",
        "config_ref": config_ref, "source_hashes": config["source_hashes"], "input_refs": config["inputs"],
        "heading_study_registration_ref": config["heading_study_registration_ref"],
        "heading_target": config["heading_target"],
        "future_guide_study_registration_ref": config["future_guide_study_registration_ref"],
        "future_guide_profile_sha256": future_core.PROFILE_SHA256, "proposal_only": True,
        "adaptive_resampling_study_registration_ref": config["adaptive_resampling_study_registration_ref"],
        "resampling_mode": RESAMPLING_MODE, "adaptive_resampling_profile_sha256": resampling_core.PROFILE_SHA256,
        "resampling_diagnostics_exported": diagnostics_exported, "native_heading_variance_unchanged": True,
        "heading_profile_sha256": (heading_core.PROFILE_SHA256 if config["heading_target"] != "legacy" else None),
        "heading_predictive_kernel_normalized": config["heading_target"] != "legacy", "exact_bias_posterior_moments": False,
        "exact_marginalized_arc_convolution": False,
        "study_registration_ref": config["study_registration_ref"], "declaration_ref": study["declaration_ref"],
        "descriptor_equivalence_assumed": False, "historical_encoder_verified": False, "old_strict_or_numeric_status_upgraded": False, "bank_key": config["bank_key"], "artifacts": generated, "protected_hashes": dict(ledger.protected),
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
