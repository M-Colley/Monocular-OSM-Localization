"""Fixed road-context epsilon target study; no policy, GT or new neural inference.

Only the declared epsilon transition changes. Native heading variance, all80
appearance factors, EXIT observations and proposal corrections remain unchanged.
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
from scripts import reference_snapshot_normalized_heading_probe as original
from monocular_osm import snapshot_future_guide as future_core
from monocular_osm import junction_heading_likelihood as heading_core
from monocular_osm import road_speed_kernel as kernel_core
from monocular_osm import road_speed_context as context_core
from monocular_osm import road_speed_io as semantic_io
from monocular_osm import road_segment_semantics, road_segment_semantics_io
p, component, native = original.p, original.component, original.native
admission, bank_core, appearance = original.admission, original.bank_core, original.appearance
role, Ledger, equal_ref = original.role, original.Ledger, original.equal_ref
prepared_inputs, bank_authority, load_bank = semantic_io.prepared_inputs, original.bank_authority, original.load_bank
INPUTS, PREPARED_MEMBERS, UNION_SPECS = original.INPUTS, original.PREPARED_MEMBERS, original.UNION_SPECS
SOURCE_PATHS = original.SOURCE_PATHS | {
    "scripts/reference_snapshot_road_speed_probe.py", "monocular_osm/snapshot_future_guide.py",
    "monocular_osm/road_speed_kernel.py", "monocular_osm/road_speed_context.py",
    "monocular_osm/road_speed_io.py", "monocular_osm/road_segment_semantics.py", "monocular_osm/road_segment_semantics_io.py"}
HEADING_MODE = "normalized-junction-v1"
HEADING_STUDY_REF = {"path": "E:/monocular-osm-joint-vpr-proof/run8/heading_study_contract.json",
    "sha256": "e7e6188c3c01909f8f089ddda70b4fc947b81f5f130a567c1101db5d2fd980a3"}


def sources(config, ledger):
    p.require(SOURCE_PATHS <= set(config["source_hashes"]), "ALL future-proposal sources required")
    expected = original.sources(config, ledger)
    for module in (heading_core, future_core, kernel_core, context_core, semantic_io, road_segment_semantics, road_segment_semantics_io, original):
        p.require(Path(module.__file__).resolve() == ROOT / (module.__name__.replace(".", "/") + ".py"),
                  "foreign future-proposal helper/worker origin")
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


ROAD_SPEED_STUDY_REF = {"path": "E:/monocular-osm-joint-vpr-proof/run19/road_speed_study_contract.json",
    "sha256": "1a6bd1833b8c591001bee5b37d8dd7d603df5705bfeebee6d9eb8a26cb774ba1"}


def admit_road_study(config, ledger):
    equal_ref(config["road_speed_study_registration_ref"], ROAD_SPEED_STUDY_REF, ledger.base)
    study=ledger.json(config["road_speed_study_registration_ref"])
    p.require(study.get("schema")=="root-prospective-road-speed-ou-snapshot-study-v1"
        and study.get("kernel_mode")==kernel_core.MODE, "prospective fixed road-speed target required")
    controls=study.get("controls")
    p.require(isinstance(controls,list)and len(controls)==4 and all(isinstance(c,dict)
        and isinstance(c.get("bank_key"),str)and c.get("heading_target")in TARGETS for c in controls),"four fixed declared controls required")
    pairs={(c["bank_key"],c["heading_target"])for c in controls};banks={c["bank_key"]for c in controls}
    p.require(len(banks)==2 and pairs=={(b,h)for b in banks for h in TARGETS}
        and (config["bank_key"],config["heading_target"])in pairs,"undeclared bank/heading control refused")
    for key,expected in (("snapshot_DATA_study_ref",config["study_registration_ref"]),
        ("normalized_heading_study_ref",config["heading_study_registration_ref"]),
        ("future_proposal_study_ref",config["future_guide_study_registration_ref"])):
        equal_ref(study[key],expected,ledger.base)
    return study


WITNESS_FLOATS=("previous_offset_m","logk_before","eps_before","logk_after","eps_after","innovation","uniform")


def validate_witness(result,net=None):
    """Bounded export checks; Root independently replays the actual PCG64 draws."""
    mask=result.get("road_model_eligible")
    if mask is None:
        p.require(result["status"]=="unavailable" and not any(k.startswith("road_witness_")or k in
            ("road_context_count","road_component_count","road_speed_witness_rng_before")for k in result),"uncomputed road witness must be absent")
        return
    p.require(isinstance(mask,np.ndarray)and mask.dtype==np.bool_ and mask.ndim==1 and 2<=len(mask)<=1201
        and not mask[0],"full typed moving interval mask required")
    total=len(mask)-1;planned=kernel_core.witness_steps(total)
    done=result.get("road_speed_completed_prediction_steps");eps_done=result.get("road_speed_epsilon_steps_computed")
    p.require(type(done)is int and type(eps_done)is int and 0<=done<=eps_done<=min(total,done+1)
        and result.get("road_speed_witness_planned_steps")==planned.tolist(),"honest full prediction/epsilon prefixes required")
    for key in ("road_context_count","road_component_count"):
        a=result.get(key)
        p.require(isinstance(a,np.ndarray)and a.dtype==np.int32 and a.shape==mask.shape and a[0]==0
            and np.all((a>=0)&(a<=50000)) and np.all(a[eps_done+1:]==0),"typed count prefix/sentinel required")
    context=result["road_context_count"];component=result["road_component_count"]
    p.require(np.all(component<=context)and np.all(context[~mask]==0),"road component/context eligibility counts differ")
    if result["status"]=="complete_diagnostic_only":
        p.require(done==eps_done==total and np.array_equal(result["times"].shape,mask.shape),"complete full-window counts required")
    completed=planned[planned<=eps_done]
    if not len(completed):
        p.require("road_witness_steps"not in result and "road_speed_witness_rng_before"not in result,"uncomputed witness rows must be absent")
        return
    steps=result.get("road_witness_steps");edge=result.get("road_witness_previous_edge")
    p.require(isinstance(steps,np.ndarray)and steps.dtype==np.int32 and np.array_equal(steps,completed)
        and isinstance(edge,np.ndarray)and edge.dtype==np.int32 and edge.shape==(len(completed),50000)
        and np.all(edge>=0),"fixed typed complete witness prefix required")
    if net is not None:
        p.require(np.all(edge<len(net.edge_len)),"previous-edge witness escaped accepted geometry")
    payload=steps.nbytes+edge.nbytes
    for name in WITNESS_FLOATS:
        a=result.get("road_witness_"+name)
        p.require(isinstance(a,np.ndarray)and a.dtype==np.float64 and a.shape==edge.shape and np.isfinite(a).all(),"finite complete Float64 road witness required: "+name)
        payload+=a.nbytes
    uniform=result["road_witness_uniform"]
    p.require(np.all((uniform==-1)|((uniform>=0)&(uniform<1)))and np.all(result["road_witness_previous_offset_m"]>=0)
        and payload<=64<<20,"road witness support/byte cap required")
    states=result.get("road_speed_witness_rng_before")
    p.require(isinstance(states,list)and len(states)==len(steps),"actual PCG64 pre-predict witness states required")
    for row,(step,state)in enumerate(zip(steps,states)):
        p.require(isinstance(state,dict)and set(state)=={"bit_generator","state","has_uint32","uinteger"}
            and state["bit_generator"]=="PCG64"and isinstance(state["state"],dict)
            and set(state["state"])=={"state","inc"}and all(type(v)is int and 0<=v<1<<128 for v in state["state"].values())
            and type(state["has_uint32"])is int and state["has_uint32"]in(0,1)
            and type(state["uinteger"])is int and 0<=state["uinteger"]<1<<32,"typed actual PCG64 state required")
        p.require(np.count_nonzero(uniform[row]>=.25)==int(component[step])
            and np.count_nonzero(uniform[row]>=0)<=int(context[step]),"witness U/component counts differ")


def execute(config_path, expected, *, check=lambda: None):
    began = time.monotonic()
    def poll():
        check()
        p.require(time.monotonic() - began <= 1800., "complete fused worker exceeded inner deadline")
    config_path = Path(config_path).resolve(strict=True)
    ledger = Ledger(config_path.parent, poll)
    config_ref = {"path": str(config_path), "sha256": expected}
    config = ledger.json(config_ref)
    p.require(set(config) == {"schema", "source_hashes", "inputs", "bank_key", "video_binding", "output_dir", "study_registration_ref", "heading_study_registration_ref", "future_guide_study_registration_ref", "heading_target", "road_speed_study_registration_ref", "semantic_inputs"}
              and config["schema"] == "reference-snapshot-road-speed-probe-run-v1" and set(config["inputs"]) == INPUTS
              and isinstance(config["bank_key"], str) and config["bank_key"]
              and type(config["heading_target"]) is str and config["heading_target"] in TARGETS, "fixed fused worker configuration required; no extra knobs")
    expected_sources = sources(config, ledger)
    admit_heading_study(config, ledger)
    admit_future_study(config, ledger)
    road_study = admit_road_study(config, ledger)  # All four declarations before ANY numeric body.
    semantic = semantic_io.admit_semantics(config, ledger, road_study, equal_ref)
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
    net, road_context, equality = semantic_io.load_context(semantic, members, ledger, graph_filter, context_core, check=poll)
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
        entry = (sequence.align_with_snapshot_future_guide_road_speed if config["heading_target"] == "legacy" else
                 sequence.align_with_snapshot_normalized_heading_future_guide_road_speed)
        result = entry(net, odometry, status, times, matched, bank, facts,
                                                road_context=road_context, config=runtime_config, check=poll)
        thread_proof = deepcopy(threads)
    p.require(result.get("status") in {"complete_diagnostic_only", "unavailable"}
              and result.get("accuracy_admitted") is False and result.get("selection") is None
              and result.get("confidence_calibrated") is False and result.get("config") == asdict(runtime_config)
              and result["profile"].get("appearance_profile_sha256") == appearance.PROFILE_SHA256
              and result["profile"].get("base_heading_target") == config["heading_target"]
              and result["profile"].get("future_guide_profile_sha256") == future_core.PROFILE_SHA256
              and result["profile"].get("proposal_only") is False
              and result["profile"].get("future_guide_is_proposal_only") is True
              and result["profile"].get("road_speed_mode") == kernel_core.MODE
              and result["profile"].get("road_speed_kernel_profile_sha256") == kernel_core.PROFILE_SHA256
              and result["profile"].get("road_speed_context_profile_sha256") == context_core.PROFILE_SHA256
              and result["profile"].get("motion_target_changed") is True
              and result["profile"].get("target_changed") is True
              and result["profile"].get("native_heading_variance_unchanged") is True
              and result["profile"].get("epsilon_transition_only") is True
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
    receipt=result.get("road_speed_kernel_receipt",{})
    p.require(receipt.get("schema")=="road-speed-epsilon-transition-receipt-v1"
        and receipt.get("mode")==kernel_core.MODE and receipt.get("profile_sha256")==kernel_core.PROFILE_SHA256
        and receipt.get("direct_q_equals_p")is True and receipt.get("new_PDF_weight_applied")is False,
        "genuine direct kernel receipt required; no invented density correction")
    validate_witness(result,net)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays.update(native_times=times, motion_status=status, native_step=odometry.step,
                  native_heading=odometry.heading, native_heading_var=odometry.heading_var, native_dt=odometry.dt)
    metadata = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    metadata.update(heading_study_registration_ref=config["heading_study_registration_ref"],
        heading_target=config["heading_target"],
        future_guide_study_registration_ref=config["future_guide_study_registration_ref"],
        future_guide_profile_sha256=future_core.PROFILE_SHA256, proposal_only=False,
        future_guide_is_proposal_only=True, road_speed_mode=kernel_core.MODE,
        motion_target_changed=True, target_changed=True, appearance_target_unchanged=True, native_heading_variance_unchanged=True,
        road_speed_kernel_profile_sha256=kernel_core.PROFILE_SHA256, road_speed_context_profile_sha256=context_core.PROFILE_SHA256,
        road_speed_study_registration_ref=config["road_speed_study_registration_ref"], semantic_inputs=config["semantic_inputs"],
        semantic_equality=equality, semantic_history=semantic["historical_binding"],
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
    master = {"schema": "reference-snapshot-road-speed-probe-run-freeze-v1", "status": "CLOSED_REFERENCE_SNAPSHOT_ROAD_SPEED_DIAGNOSTIC_NO_ACCURACY",
        "config_ref": config_ref, "source_hashes": config["source_hashes"], "input_refs": config["inputs"],
        "heading_study_registration_ref": config["heading_study_registration_ref"],
        "heading_target": config["heading_target"],
        "future_guide_study_registration_ref": config["future_guide_study_registration_ref"],
        "future_guide_profile_sha256": future_core.PROFILE_SHA256, "proposal_only": False,
        "future_guide_is_proposal_only": True, "road_speed_mode": kernel_core.MODE,
        "motion_target_changed": True, "target_changed": True, "appearance_target_unchanged": True, "native_heading_variance_unchanged": True,
        "road_speed_kernel_profile_sha256": kernel_core.PROFILE_SHA256, "road_speed_context_profile_sha256": context_core.PROFILE_SHA256,
        "road_speed_study_registration_ref": config["road_speed_study_registration_ref"], "semantic_inputs": config["semantic_inputs"],
        "semantic_equality": equality, "semantic_history": semantic["historical_binding"],
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
    semantic_io.verify_memberships(semantic, prepared_path)
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
