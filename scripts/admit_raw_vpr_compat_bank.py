"""Separate Root-run numerical-compatibility study; strict admission unchanged.

Original IO/native/OWN/cache and per-pool consumed-tensor gates are reused.
No model, field, decoder, selected Answer or historical identity upgrade.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts import admit_raw_vpr_bank as original
from monocular_osm import raw_vpr_compat_bank as bank

component, native = original.component, original.native
Ledger = original.Ledger
source_and_plan, native_authority = original.source_and_plan, original.native_authority
payload_inventory, verify_directories = original.payload_inventory, original.verify_directories
own_config, common_comparison = original.own_config, original.common_comparison
SOURCES = original.SOURCES | {"scripts/admit_raw_vpr_compat_bank.py", "monocular_osm/raw_vpr_compat_bank.py"}
STUDY_REF = {"path": "E:/monocular-osm-joint-vpr-proof/run5/study_contract.json",
             "sha256": "a6f8cd9a25e7a0fe576a2eabd8b6e7c88bd56ba5b46b545c4b86952e19379d11"}
DECLARATION_REF = {"path": "E:/monocular-osm-joint-vpr-proof/run5/compatibility_profile_declaration.json",
                   "sha256": "789c231517b0ddc496074bce5aff9abcf844c70dd643884f8ae8aa054e68c742"}
STUDY_VALUES = {"data_contract_kind": "whole-bank-frozen-feature-numerical-compatibility-v1",
    "controls": ["baseline-full-recorded-bank", "expanded-full-recorded-bank"],
    "descriptor_l2_max_f64": 1e-6, "raw_score_max_full_envelope": 1e-6,
    "require_exact_coordinates": True, "require_all_physical_calls_and_aliases": True,
    "require_original_per_pool_gates_and_consumed_tensor_sha": True,
    "maximum_calls_per_bank": 64, "query_count": 80, "maximum_pool_references": 1500,
    "maximum_unique_references_per_bank": 30000, "canonical_descriptor_F32_byte_limit": 1013760000,
    "maximum_raw_F32_matrix_values": 2400000, "maximum_field_F32_values": 16000000,
    "reference_encoding_identity": "UNVERIFIED", "bitwise_descriptor_identity_claimed": False,
    "old_strict_status_or_data_authority_upgraded": False, "policy_admission": False,
    "neural_forward_run": False, "test4_labels_used": False}
PROFILE = {**original.PROFILE, "method": "root-accepted-native80-numeric-compatible-bank-admission-v1",
    "bank_profile_sha256": bank.PROFILE_SHA256, "maximum_references_per_bank": 30000,
    "maximum_raw_F32_matrix_values": 2400000, "compatibility_not_bitwise_identity": True,
    "historical_encoder_verified": False, "study_registration_ref": STUDY_REF}
ARRAY_NAMES = original.ARRAY_NAMES
MAX_COMPAT_REPORT_JSON = 64 << 20
PROFILE["maximum_derived_report_JSON_bytes"] = MAX_COMPAT_REPORT_JSON
PROFILE_SHA256 = native.digest(PROFILE)


def read_compatible_report(ledger, record):
    """Only derived compatible reports get 64MiB; share the 256MiB counter."""
    path, expected = native.ref(record)
    component.safe(path)
    size = Path(path).stat().st_size
    native.need(size <= MAX_COMPAT_REPORT_JSON and ledger.total_json + size <= 256 << 20,
                "compatible report budget before body/hash read")
    ledger.total_json += size
    path = ledger.protect(record)
    with path.open("rb") as stream:
        body = stream.read(size + 1)
    native.need(len(body) == size and hashlib.sha256(body).hexdigest() == expected,
                "compatible report changed during bounded body read")
    def pairs(items):
        out = {}
        for key, value in items:
            native.need(key not in out, "duplicate compatible report JSON key")
            out[key] = value
        return out
    return json.loads(body.decode("utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def admit_study(config, ledger):
    """Real prospective Root declarations precede every numerical member."""
    original.equal_ref(config["study_registration_ref"], STUDY_REF)
    study = ledger.json(config["study_registration_ref"])
    native.need(study.get("schema") == "root-raw-vpr-compatible-study-contract-v1", "explicit compatible study required")
    original.equal_ref(study["declaration_ref"], DECLARATION_REF)
    declaration = ledger.json(study["declaration_ref"])
    native.need(declaration.get("schema") == "root-prospective-full-bank-numerical-compatibility-study-v1", "prospective declaration required")
    for key, expected in STUDY_VALUES.items():
        native.need(type(study.get(key)) is type(expected) and study[key] == expected, "study constant/claim differs: " + key)
    numerical = ledger.json(study["numerical_fusion_profile_ref"])
    native.need(isinstance(numerical, dict) and numerical, "original numerical fusion declaration required")
    return study


def execute(config_path, config_sha256, *, check=lambda: None, descriptor_loader=None,
            numerical_context=None, resource_gate=None):
    began = time.monotonic()
    resource_gate = component.headroom if resource_gate is None else resource_gate
    def poll():
        check()
        native.need(time.monotonic() - began <= 1800, "inner admission deadline exceeded")
        resource_gate()
    ledger = Ledger(poll)
    config_ref = {"path": str(Path(config_path).resolve()), "sha256": config_sha256}
    config = ledger.json(config_ref)
    native.need(config.get("schema") == "raw-vpr-compatible-bank-admission-config-v1", "wrong bank configuration")
    native.need(SOURCES <= set(config["source_hashes"]), "complete compatible source pins required")
    study = admit_study(config, ledger)
    plan = source_and_plan(config, ledger)
    facts = native_authority(config, ledger)
    for declaration in config["banks"]:
        records = [r for r in config["calls"] if r["key"] in declaration["call_keys"]]
        native.need(native.validate_native_facts(facts, records, declaration["key"]) == plan["appearance_identity"],
                    "native proof and original metadata identity differ")
    payloads, frozen_files, groups = payload_inventory(config, ledger)
    root, output = Path(config["request_execution_root"]).resolve(), component.safe(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    descriptor_loader = component.load_descriptor_rows if descriptor_loader is None else descriptor_loader
    receipts = {}
    def loader(record):
        audit = ledger.json(record["metadata_ref"])
        directory = native.key(audit["public_cache_request"]["cache_dir"])
        native.need(directory == native.key(payloads["calls"][record["key"]]), "call cache directory differs")
        verify_directories({directory: groups[directory]})
        path = ledger.protect(record["array_ref"])
        arrays = component._arrays_bound(path, {path: record["array_ref"]["sha256"]}, component.ARRAYS)
        region = {"key": record["key"], "audit": audit, "arrays": arrays}
        request, saved = ledger.json(record["request_ref"]), deepcopy(audit)
        evidence = descriptor_loader(region, own_config(request, root))
        native.need(isinstance(evidence, component.DescriptorEvidence), "genuine descriptor evidence required")
        evidence.verify()
        native.need(audit == saved, "descriptor loader altered original metadata")
        receipt = evidence.receipt
        native.need(native.key(receipt["request"]["cache_dir"]) == directory, "current receipt cache differs")
        expected = set()
        for item in receipt["files"]:
            path = native.key(Path(directory) / item["name"]); bound = frozen_files.get(path)
            native.need(bound is not None and bound["sha256"] == item["sha256"] and bound["bytes"] == item["bytes"],
                        "current receipt payload outside pre-value frozen inventory")
            expected.add((path, item["sha256"]))
        native.need({(native.key(p), h) for p, h in evidence.protected_hashes.items()} == expected,
                    "actual evidence does not cover exact receipt files")
        receipts[record["key"]] = {"current_descriptor_receipt": deepcopy(receipt),
            "original_persisted_receipt_present": "public_reference_cache" in audit,
            "original_persisted_receipt": deepcopy(audit.get("public_reference_cache")),
            "claim": "current frozen payload matches original consumed tensor; historical encoding UNVERIFIED"}
        region["descriptor_rows"] = evidence.rows
        return region
    results = []
    context = component.cpu_two_threads if numerical_context is None else numerical_context
    with context() as threads:
        for index, declaration in enumerate(config["banks"]):
            records = [r for r in config["calls"] if r["key"] in declaration["call_keys"]]
            try:
                grid = ledger.members(records[0]["array_ref"], ("query_indices", "query_times"))
                result = bank.admit_bank(declaration["key"], records,
                    common_comparison(facts["appearance_identity"], grid), facts, loader)
            except Exception as exc:
                result = {"status": "UNAVAILABLE", "bank_key": declaration["key"],
                    "profile_sha256": bank.PROFILE_SHA256, "reason": type(exc).__name__ + ": " + str(exc)}
            if result["status"] == "complete_compatible_research_bank":
                arrays = {name: result.pop(name) for name in ARRAY_NAMES}; target = output / f"bank_{index:02d}.npz"
                with target.open("xb") as stream: np.savez_compressed(stream, **arrays)
                expected = native.sha(target); ledger.protected[str(target)] = expected
                result["array_artifact"] = {"path": str(target), "sha256": expected, "arrays": {
                    name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": native.typed(a)} for name, a in arrays.items()}}
                del arrays
            results.append(result)
        thread_proof = deepcopy(threads)
    report = {"schema": "root-native-accepted-compatible-raw-vpr-banks-v1", "profile": PROFILE, "profile_sha256": PROFILE_SHA256,
        "config_ref": config_ref, "study_registration_ref": config["study_registration_ref"],
        "declaration_ref": study["declaration_ref"], "compatibility_not_bitwise_identity": True,
        "historical_encoder_verified": False, "native_root_acceptance_ref": config["native_root_acceptance_ref"],
        "native_facts_ref": config["native_facts_ref"], "reference_payload_inventory": config["reference_payload_inventory"],
        "banks": results, "descriptor_receipts": receipts, "threadpools": thread_proof,
        "protected_hashes": dict(ledger.protected), "elapsed_before_final_hashes_s": time.monotonic() - began,
        "external_actor_closure_verified": False, "reference_encoding_identity": "UNVERIFIED",
        "field_built": False, "decoder_run": False, "new_neural_forwards": False, "network": False,
        "ground_truth_used": False, "policy_admission": False, "selected_answer": None}
    serialized = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    native.need(len(serialized.encode("utf-8")) <= MAX_COMPAT_REPORT_JSON, "complete compatible report exceeds64MiB; no truncation")
    expected_outputs = {Path(v["array_artifact"]["path"]).name for v in results if "array_artifact" in v}
    poll()  # Last injectable/resource callback precedes final membership AND bytes.
    native.need({p.name for p in output.iterdir()} == expected_outputs, "unexpected admission output membership")
    verify_directories(groups)
    for path, expected in ledger.protected.items():
        native.need(native.sha(path) == expected, "protected source/input/output changed before publication: " + path)
    with (output / "bank_admission.json").open("x", encoding="utf-8") as stream: stream.write(serialized)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    execute(args.config, args.config_sha256)


if __name__ == "__main__": main()
