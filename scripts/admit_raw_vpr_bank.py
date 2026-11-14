"""Root-run frozen bank admission; no models, fields, decoding or selection.

External CPU2/3GiB-RSS/8GiB-both/1830s guard required; inner budget 1800s.
This worker never certifies its own actor closure.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from monocular_osm import assisted_metadata_union as component
from monocular_osm import public_reference_cache as public
from monocular_osm import raw_vpr_bank as bank, raw_vpr_native as native

SOURCES = frozenset(("scripts/admit_raw_vpr_bank.py", "monocular_osm/raw_vpr_bank.py",
    "monocular_osm/raw_vpr_native.py", "monocular_osm/metadata_union.py", "monocular_osm/full_sequence.py",
    "monocular_osm/sparse_sequence.py", "monocular_osm/assisted_metadata_union.py",
    "monocular_osm/assisted_comparison.py", "monocular_osm/assisted_policy.py",
    "monocular_osm/reference_inventory.py", "monocular_osm/public_reference_cache.py",
    "monocular_osm/kartaview_vpr.py", "monocular_osm/provenance.py", "monocular_osm/geo.py"))
PROFILE = {"method": "root-accepted-native80-frozen-raw-bank-admission-v1",
    "bank_profile_sha256": bank.PROFILE_SHA256, "maximum_banks": 2, "maximum_calls_per_bank": 64,
    "maximum_cache_directories": 128, "maximum_public_payload_files": 4096,
    "maximum_public_payload_bytes": 256 * 1024**3, "inner_wall_budget_s": 1800,
    "external_wall_budget_s": 1830, "cpu_threads": 2, "external_RSS_limit_bytes": 3 * 1024**3,
    "minimum_free_physical_and_true_commit_bytes": 8 * 1024**3,
    "reference_encoding_identity": "UNVERIFIED", "research_only": True,
    "new_neural_forwards": False, "network": False, "policy_admission": False}
PROFILE_SHA256 = native.digest(PROFILE)
ARRAY_NAMES = ("ref_ids", "ref_latlon", "raw_similarities", "query_indices", "query_times")

class Ledger(native.Ledger):
    def protect(self, record):
        component.safe(record["path"])
        return super().protect(record)
    def json(self, record):
        component.safe(record["path"])
        return super().json(record)

def equal_ref(a, b):
    native.need(native.ref(a) == native.ref(b), "authority reference differs")


def native_authority(config, ledger):
    """Authenticate explicit Root/closed-worker authority before NPZ loading."""
    facts, qa = ledger.json(config["native_facts_ref"]), ledger.json(config["native_root_acceptance_ref"])
    native.need(qa.get("schema") == "root-closed-raw-vpr-native-acceptance-v1"
        and qa.get("status") == "ACCEPTED_INPUT_PIXELS_CLOCK_ONLY_NO_NN", "explicit Root native acceptance required")
    for name in ("native_facts_ref", "inventory_ref", "own_request_bindings_ref"):
        equal_ref(qa[name], config[name])
    native.need(all(qa.get(k) is True for k in ("all_observed_worker_births_closed",
        "all_protected_input_hashes_unchanged", "native_input_pixels_clock_verified")), "Root native gates incomplete")
    native.need(qa.get("reference_encoding_identity") == "UNVERIFIED"
        and all(qa.get(k) is False for k in ("model_inference_run", "reference_encoding_upgraded",
            "original_child_completion_certified", "ground_truth_used"))
        and "selected_answer" in qa and qa["selected_answer"] is None, "Root native authority must remain input-only")
    registration = ledger.json(qa["external_guard_registration_ref"])
    result, closure = ledger.json(qa["external_guard_result_ref"]), ledger.json(qa["external_closure_ref"])
    native.need(isinstance(registration, dict) and registration and result.get("status") == "complete"
        and type(result.get("returncode")) is int and result["returncode"] == 0
        and "reason" in result and result["reason"] is None, "native external guard did not close successfully")
    native.need(closure.get("schema") == "root-native-replay-external-closure-v1"
        and closure.get("worker_births_closed") is True and closure.get("protected_inputs_unchanged") is True,
        "separate Root native worker closure required")
    for left, right in ((closure["native_facts_ref"], config["native_facts_ref"]),
        (closure["registration_ref"], qa["external_guard_registration_ref"]),
        (closure["result_ref"], qa["external_guard_result_ref"]),
        (facts["metadata_inventory"], config["inventory_ref"]),
        (facts["own_request_bindings"], config["own_request_bindings_ref"]),
        (facts["request_execution_origin_ref"], config["request_execution_origin_ref"])):
        equal_ref(left, right)
    native.need(native.key(facts["request_execution_root"]) == native.key(config["request_execution_root"]),
                "native replay historical root differs")
    for path, expected in facts["protected_hashes"].items(): ledger.protect({"path": path, "sha256": expected})
    return facts


def source_and_plan(config, ledger):
    sources = config["source_hashes"]
    native.need(isinstance(sources, dict) and SOURCES <= set(sources), "complete explicit executed source pins required")
    for name, expected in sources.items():
        path = (ROOT / name).resolve(strict=True)
        native.need(not Path(name).is_absolute() and path.is_relative_to(ROOT), "source pin escaped repository")
        ledger.protect({"path": str(path), "sha256": expected})
    root = Path(config["request_execution_root"])
    native.need(root.is_absolute(), "explicit absolute historical execution root required")
    origin = ledger.json(config["request_execution_origin_ref"]); commit = origin.get("source_commit")
    native.need(native.key(origin["source"]) == native.key(root) and origin.get("source_clean") is True
        and isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
        "historical execution origin registration differs")
    inventory, bindings = ledger.json(config["inventory_ref"]), ledger.json(config["own_request_bindings_ref"])
    equal_ref(bindings["inventory_ref"], config["inventory_ref"])
    plan = native.make_plan(inventory, config["calls"], bindings, ledger.json, ledger.json,
                            request_execution_root=str(root))
    declared = config["banks"]
    native.need(isinstance(declared, list) and 1 <= len(declared) <= 2
        and all(set(v) == {"key", "call_keys"} for v in declared)
        and len({v["key"] for v in declared}) == len(declared), "explicit unique bank registrations required")
    native.need({v["key"]: v["call_keys"] for v in declared} == plan["bank_membership"],
                "all predeclared bank/physical-call memberships required")
    return plan


def verify_directories(groups):
    for directory, expected_names in groups.items():
        native.need({p.name for p in Path(directory).iterdir() if public._PAYLOAD.fullmatch(p.name)} == expected_names,
                    "public cache filename inventory changed")


def payload_inventory(config, ledger):
    inventory = ledger.json(config["reference_payload_inventory"])
    native.need(inventory.get("schema") == "root-raw-vpr-reference-payload-inventory-v1", "pre-value public inventory required")
    equal_ref(inventory["metadata_inventory_ref"], config["inventory_ref"])
    native.need(set(inventory["calls"]) == {r["key"] for r in config["calls"]}, "ALL calls need frozen cache identity")
    directories = inventory["directories"]
    native.need(isinstance(directories, list) and 1 <= len(directories) <= 128, "public directory cap")
    files, groups, total = {}, {}, 0
    for group in directories:
        directory = component.safe(group["path"]).resolve(strict=True); directory_key = native.key(directory)
        native.need(directory.is_dir() and directory_key not in groups, "duplicate/invalid public directory")
        names = set()
        for item in group["files"]:
            name = item["name"]
            native.need(isinstance(name, str) and public._PAYLOAD.fullmatch(name) and name not in names
                and type(item["bytes"]) is int and item["bytes"] >= 0, "invalid public payload role/size")
            names.add(name)
            path = component.safe(item["path"]).resolve(strict=True)
            native.need(path == directory / name and native.key(path) not in files, "public payload escaped/repeated")
            total += item["bytes"]
            native.need(len(files) < 4096 and total <= PROFILE["maximum_public_payload_bytes"], "public payload budget")
            files[native.key(path)] = {"path": str(path), "sha256": item["sha256"], "bytes": item["bytes"]}
        native.need(names, "public payload directory empty")
        groups[directory_key] = names
    native.need({native.key(v) for v in inventory["calls"].values()} == set(groups), "unused/missing public directory")
    verify_directories(groups)
    for item in files.values():
        native.need(Path(item["path"]).stat().st_size == item["bytes"], "pre-frozen public payload size differs")
        ledger.protect({k: item[k] for k in ("path", "sha256")})
    return inventory, files, groups


def own_config(request, execution_root):
    values = deepcopy(request["config"])
    for name in ("data_dir", "output_dir", "vpr_region_cache_dir"):
        value = values.get(name)
        if value is not None:
            path = Path(value)
            values[name] = str(path.resolve() if path.is_absolute() else (execution_root / path).resolve())
    return SimpleNamespace(**values)


def common_comparison(flat, arrays):
    return {"statistic": bank.legacy.STATISTIC,
        "model": {**deepcopy(flat["backbone"]), "weights_sha256": flat["actual_model_state_sha256"]},
        "query": {"indices": arrays["query_indices"], "times": arrays["query_times"], "mode": flat["query_mode"],
            "preprocessing": deepcopy(flat["preprocessing"]), "pixels_sha256": flat["query_pixels_sha256"],
            "descriptors_sha256": flat["query_descriptors_sha256"]}}


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
    native.need(config.get("schema") == "raw-vpr-bank-admission-config-v1", "wrong bank configuration")
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
            if result["status"] == "complete_research_bank":
                arrays = {name: result.pop(name) for name in ARRAY_NAMES}; target = output / f"bank_{index:02d}.npz"
                with target.open("xb") as stream: np.savez_compressed(stream, **arrays)
                expected = native.sha(target); ledger.protected[str(target)] = expected
                result["array_artifact"] = {"path": str(target), "sha256": expected, "arrays": {
                    name: {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": native.typed(a)} for name, a in arrays.items()}}
                del arrays
            results.append(result)
        thread_proof = deepcopy(threads)
    report = {"schema": "root-native-accepted-raw-vpr-banks-v1", "profile": PROFILE, "profile_sha256": PROFILE_SHA256,
        "config_ref": config_ref, "native_root_acceptance_ref": config["native_root_acceptance_ref"],
        "native_facts_ref": config["native_facts_ref"], "reference_payload_inventory": config["reference_payload_inventory"],
        "banks": results, "descriptor_receipts": receipts, "threadpools": thread_proof,
        "protected_hashes": dict(ledger.protected), "elapsed_before_final_hashes_s": time.monotonic() - began,
        "external_actor_closure_verified": False, "reference_encoding_identity": "UNVERIFIED",
        "field_built": False, "decoder_run": False, "new_neural_forwards": False, "network": False,
        "ground_truth_used": False, "policy_admission": False, "selected_answer": None}
    serialized = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
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
