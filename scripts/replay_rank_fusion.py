"""Model-free frozen rank controls/equal-three fusion: sixteen new +24 controls.

Posthoc DEV, not a validated policy. Prepare and infer are separate; inference
requires a clean pin, an explicit input SHA and an external 8GiB watchdog.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import sys
import time

if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from monocular_osm import rank_fusion as core, full_sequence as full, heading_sequence as heading_core
from scripts import replay_heading_sequence as utility

require, sha, load, write = utility.require, utility.sha, utility.load, utility.write
SHAPE = (80, 9735)
MATRIX_KEYS = {m: m+"_raw_cosines" for m in core.MODELS}
SHARED = ("ref_ids", "ref_latlon", "query_times", "query_indices", "frame_ordinals")
VARIANTS = ("raw", "reference_mean")
DECODERS = ("simple", "heading_erased")
APPEARANCES = tuple(core.PROFILE["outputs"])
NEW_OUTPUTS = [a+"/"+d+"/"+v for a in APPEARANCES for d in DECODERS for v in VARIANTS]
OLD_OUTPUTS = ["selavpr/simple/"+v for v in VARIANTS] + [
    "selavpr/heading/"+v+"/"+m for v in VARIANTS for m in heading_core.PROFILE["modes"]] + [
    "controls/simple/"+m+"/"+v for m in ("frozen_megaloc", "ImAge_GSV") for v in VARIANTS] + [
    "controls/heading/"+m+"/"+v+"/"+mode for m in ("megaloc", "image")
    for v in VARIANTS for mode in heading_core.PROFILE["modes"]]
ALL_OUTPUTS = NEW_OUTPUTS + ["cosine/"+n for n in OLD_OUTPUTS]
SOURCE_PROFILE = "e0908035b1ef4b6f0039a39226d9e70d7c5b40b25d78944406d93d0b9f532ad2"
PROFILE = {"version": "frozen-rank-calibration-equal-three-replay-v1", "posthoc_DEV": True,
    "validation_status": "after_encoder_outcomes_not_blind_not_policy", "core_profile_sha256": core.PROFILE_SHA256,
    "simple_profile_sha256": full.PROFILE_SHA256, "heading_profile_sha256": heading_core.PROFILE_SHA256,
    "original_encoder_profile_sha256": SOURCE_PROFILE, "queries": 80, "references": 9735,
    "outputs": ALL_OUTPUTS, "new_outputs": 16, "retained_controls": 24,
    "percentile_dtype": "float64", "mean_correction": "AFTER rank/fusion using unchanged decoder",
    "cpu_threads": 2, "runtime_wall_budget_s": 10800,
    "minimum_free_physical_and_true_commit_GiB": 8, "external_watchdog_required": True,
    "all40_before_new_scoring": True, "new_forwards": False, "network": False,
    "candidate_pruning": False, "labels_or_scores": False, "selected_variant": None,
    "confidence": None, "default_adopted": False,
    "legacy_megaloc_reference_encoding_identity": "UNVERIFIED_no_causal_encoder_only_claim"}
PROFILE_SHA256 = heading_core._json_sha(PROFILE)


@contextmanager
def numerical_threads():
    """Control installed NumPy/SciPy OpenBLAS without an extra dependency."""
    controlled, records = [], []
    try:
        for module in (np, scipy):
            directory = Path(module.__file__).parent.parent/(module.__name__+".libs")
            for path in sorted(directory.glob("*openblas*")):
                library = ctypes.CDLL(str(path))
                get = setter = None
                for symbol in ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                               "openblas_get_num_threads64_", "openblas_get_num_threads"):
                    try:
                        get = getattr(library, symbol)
                        setter = getattr(library, symbol.replace("get_num_threads", "set_num_threads"))
                        break
                    except AttributeError: continue
                require(get is not None and setter is not None, "native BLAS thread control unavailable")
                get.argtypes, get.restype = [], ctypes.c_int
                setter.argtypes, setter.restype = [ctypes.c_int], None
                previous = int(get())
                controlled.append((library, setter, previous))
                setter(2)
                require(int(get()) == 2, "actual native thread ceiling differs")
                records.append({"path": str(path), "num_threads": int(get()),
                                "previous_num_threads": previous, "query_symbol": symbol})
        require(len(records) >= 2, "NumPy/SciPy actual thread proof unavailable")
        yield records
    finally:
        for library, setter, previous in reversed(controlled): setter(previous)


def specification(path):
    value = load(path)
    require(isinstance(value, dict) and set(value) == {"source", "audit"}, "exact source/audit roles required")
    for role, path_key in (("source", "freeze"), ("audit", "path")):
        row = value[role]
        require(isinstance(row, dict) and set(row) == {path_key, "sha256"}
            and isinstance(row[path_key], str) and utility.digest(row["sha256"]), "explicit role path/SHA required")
        utility.allowed(row[path_key])
    return value


def source_identity(expected, previous=None):
    require(Path(__file__).resolve() == ROOT/"scripts/replay_rank_fusion.py" and utility.ROOT == ROOT,
            "executed runner/utilities escaped source pin")
    sources = utility.source_identity(expected)
    for name, module in list(sys.modules.items()):
        if name.startswith(("scripts.", "monocular_osm.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(ROOT), "loaded module escaped source pin")
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    sources["scripts/replay_rank_fusion.py"] = sha(__file__)
    require(previous is None or sources == previous, "loaded source bytes changed")
    return sources


def copy(source, target, expected):
    source, target = utility.allowed(source).resolve(strict=True), utility.allowed(target)
    require(sha(source) == expected and not target.exists(), "copy source differs or destination exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha(target) == expected, "copied bytes differ")


def read_evidence(path):
    q, n = SHAPE
    specs = {"ref_ids": (n, "US", 512), "ref_latlon": (n*2, "f", 8),
        "query_times": (q, "f", 8), "query_indices": (q, "iu", 8), "frame_ordinals": (q, "iu", 8),
        **{key: (q*n, "f", 4) for key in MATRIX_KEYS.values()}}
    arrays = utility.npz_arrays(path, specs)
    require(arrays["ref_ids"].shape == (n,) and arrays["ref_latlon"].shape == (n, 2)
        and all(arrays[key].shape == (q,) for key in SHARED[2:]), "fixed complete ID/query grid required")
    require(np.array_equal(arrays["query_indices"], np.arange(q))
        and np.all(np.diff(arrays["frame_ordinals"].astype(np.int64)) > 0)
        and arrays["frame_ordinals"][0] >= 0, "complete native query indices/ordinals required")
    evidence = {model: {"scores": arrays[key], **{k: arrays[k] for k in SHARED[:3]}}
                for model, key in MATRIX_KEYS.items()}
    # Input validation only: no ranks, graph construction or route decoding.
    identities = {}
    for model, item in evidence.items():
        scores = core._matrix(item["scores"])
        require(scores.shape == (q, n), "fixed complete matrix required")
        heading_core._inputs(scores[:, None, :], item["ref_ids"], item["ref_latlon"], item["query_times"],
            [None]*n, np.full(n, np.nan), np.zeros(1), None, None,
            heading_core.PROFILE["default_temp_bytes"], heading_core.PROFILE["max_joint_backpointer_bytes"])
        identities[model] = heading_core._array_sha(scores)
    identity = {k: heading_core._array_sha(arrays[k]) for k in SHARED}
    identity["matrices"] = identities
    return arrays, evidence, identity


def canonical_artifacts(artifacts):
    require(isinstance(artifacts, dict), "artifact digest map required")
    result, aliases = {}, set()
    for name, digest in artifacts.items():
        utility.allowed(name)
        key = utility.relative(name).as_posix()
        require(key not in ("", ".") and key.casefold() not in aliases and utility.digest(digest),
                "duplicate/unsafe artifact path alias or digest")
        aliases.add(key.casefold())
        result[key] = digest
    return result


def verified_source(spec):
    source = utility.allowed(spec["source"]["freeze"]).resolve(strict=True)
    frozen = utility.freeze(source, spec["source"]["sha256"])
    # Existing Windows producers used both slash spellings in index/freezes.
    # Canonicalize only in memory; preserve every original byte on disk.
    frozen = {**frozen, "artifacts": canonical_artifacts(frozen["artifacts"])}
    require(frozen.get("status") == "complete" and frozen.get("scoring_has_not_run") is True
        and frozen.get("profile_sha256") == SOURCE_PROFILE and frozen.get("selected_variant") is None
        and frozen.get("all_requested_outputs") == OLD_OUTPUTS
        and frozen.get("new_outputs") == 8 and frozen.get("retained_controls") == 16,
        "complete original24 encoder freeze required")
    audit_path = utility.allowed(spec["audit"]["path"]).resolve(strict=True)
    require(sha(audit_path) == spec["audit"]["sha256"], "accepted audit bytes differ")
    audit = load(audit_path)
    require(audit.get("status") == "PASS" and audit.get("input_only") is False
        and audit.get("ground_truth_or_official_scores_opened") is False
        and audit.get("new_neural_forwards") is False and audit.get("model_constructors") is False
        and utility.digest(audit.get("auditor_sha256")) and utility.digest(audit.get("dependency_sha256"))
        and audit.get("freeze_sha256") == spec["source"]["sha256"]
        and audit.get("input_freeze_sha256") == frozen["input_freeze_sha256"]
        and audit.get("source_commit") == frozen["source_commit"]
        and Path(audit["family"]).resolve() == source.parent.parent,
        "accepted audit does not bind exact completed source/input/family and input-only scope")
    protected = audit.get("protected_sha256")
    require(isinstance(protected, dict) and 1 <= len(protected) <= 20000
        and audit.get("protected_files") == len(protected)
        and all(utility.digest(d) for d in protected.values()), "bounded complete protected audit ledger required")
    paths = [(utility.allowed(p).resolve(strict=True), d) for p, d in protected.items()]
    for path, digest in paths: require(sha(path) == digest, "audited protected input/source changed")
    required = ("full_evidence.npz", "comparison_index.json", "runtime.json")
    require(all(n in frozen["artifacts"] for n in required), "audited matrix/control/native identity roles absent")
    arrays, evidence, identity = read_evidence(source.parent/"full_evidence.npz")
    runtime = load(source.parent/"runtime.json")
    require(runtime["source_commit"] == frozen["source_commit"]
        and runtime["input_freeze_sha256"] == frozen["input_freeze_sha256"]
        and all(runtime["input_identity"][k] == identity[k] for k in SHARED)
        and all(runtime["input_identity"]["matrices"][m] == identity["matrices"][m] for m in ("megaloc", "image")),
        "audited native query/gallery/model matrix identity differs")
    index = load(source.parent/"comparison_index.json")
    controls = index["outputs"]
    require(index["profile_sha256"] == SOURCE_PROFILE and len(controls) == 24
        and [r["name"] for r in controls] == OLD_OUTPUTS, "all original24 control records required")
    records = []
    for i, row in enumerate(controls):
        relative = utility.relative(row["file"])
        path = utility.allowed(source.parent/relative).resolve(strict=True)
        require(path.is_relative_to(source.parent) and frozen["artifacts"].get(relative.as_posix()) == row["sha256"]
            and sha(path) == row["sha256"], "control is not a protected original prediction")
        records.append({"name": "cosine/"+row["name"], "source": str(path), "sha256": row["sha256"],
                        "file": f"controls/{i:02d}.json"})
    return {"source": source, "frozen": frozen, "arrays": arrays, "evidence": evidence,
        "identity": identity, "native_identity": runtime["input_identity"], "controls": records}


def prepare(spec_path, family, expected_commit):
    sources = source_identity(expected_commit)
    utility.check_headroom()
    spec_path = utility.allowed(spec_path).resolve(strict=True)
    spec, family = specification(spec_path), utility.allowed(family).resolve()
    roots = [ROOT, Path(spec["source"]["freeze"]).resolve().parent.parent,
             Path(spec["audit"]["path"]).resolve().parent]
    require(not family.exists() and not any(family.is_relative_to(p) or p.is_relative_to(family) for p in roots),
            "new isolated output family required")
    bundle = verified_source(spec)
    family.mkdir(parents=True)
    try:
        copy(spec_path, family/"inputs/spec.json", sha(spec_path))
        copy(bundle["source"], family/"inputs/original_freeze.json", spec["source"]["sha256"])
        copy(spec["audit"]["path"], family/"inputs/accepted_audit.json", spec["audit"]["sha256"])
        copy(bundle["source"].parent/"full_evidence.npz", family/"inputs/full_evidence.npz",
             bundle["frozen"]["artifacts"]["full_evidence.npz"])
        for row in bundle["controls"]: copy(row["source"], family/"inputs"/row["file"], row["sha256"])
        for name, digest in sources.items(): copy(ROOT/name, family/"inputs/source_snapshot"/name, digest)
        write(family/"plan.json", {"profile": PROFILE, "profile_sha256": PROFILE_SHA256,
            "source_commit": expected_commit, "source_sha256": sources, "declared": spec,
            "input_identity": bundle["identity"], "native_identity": bundle["native_identity"],
            "controls": bundle["controls"], "all_requested_outputs": ALL_OUTPUTS, "selected_variant": None})
        require(verified_source(spec)["identity"] == bundle["identity"], "source inputs changed during prepare")
        source_identity(expected_commit, sources)
        write(family/"inputs_frozen.json", {"status": "prepared_only", "scoring_has_not_run": True,
            "models_have_not_run": True, "all_requested_outputs": ALL_OUTPUTS,
            "artifacts": {p.relative_to(family).as_posix(): sha(p) for p in family.rglob("*") if p.is_file()}})
        return {"status": "prepared_only", "input_freeze_sha256": sha(family/"inputs_frozen.json"),
                "profile_sha256": PROFILE_SHA256, "new_outputs": 16, "retained_controls": 24}
    except Exception as error:
        write(family/"preparation_failed.json", {"status": "failed", "error": str(error), "labels_opened": False})
        raise


def verify_prepared(family, expected_commit, expected_inputs):
    require(utility.digest(expected_inputs), "explicit prepared input-freeze SHA required")
    utility.freeze(family/"inputs_frozen.json", expected_inputs)
    plan = load(family/"plan.json")
    require(plan["source_commit"] == expected_commit and plan["profile"] == PROFILE
        and plan["profile_sha256"] == PROFILE_SHA256 and plan["all_requested_outputs"] == ALL_OUTPUTS
        and plan["selected_variant"] is None, "registered profile/source/output list differs")
    source_identity(expected_commit, plan["source_sha256"])
    spec = specification(family/"inputs/spec.json")
    require(spec == plan["declared"], "input source/audit spec changed")
    bundle = verified_source(spec)
    require(bundle["identity"] == plan["input_identity"] and bundle["native_identity"] == plan["native_identity"]
        and bundle["controls"] == plan["controls"], "audited original identity/controls changed")
    require(sha(family/"inputs/full_evidence.npz") == bundle["frozen"]["artifacts"]["full_evidence.npz"],
            "prepared matrix differs from original source")
    return plan, bundle


def resource_gate(started):
    require(time.monotonic()-started <= PROFILE["runtime_wall_budget_s"], "registered wall budget exceeded")
    return utility.check_headroom()


def decode_all(output, calibrated, shared, started):
    records = []
    for appearance in APPEARANCES:
        matrix = calibrated["matrices"][appearance]
        for decoder in DECODERS:
            resource_gate(started)
            result = full.decode_full_tracks(matrix, *[shared[k] for k in SHARED[:3]]) if decoder == "simple" else (
                core.decode_erased_tracks(matrix, *[shared[k] for k in SHARED[:3]],
                    progress=lambda row, count: resource_gate(started)))
            require(set(result["outputs"]) == set(VARIANTS), "both normalization outputs required")
            write(output/f"bundle_{appearance}_{decoder}.json", result)
            for variant in VARIANTS:
                name = appearance+"/"+decoder+"/"+variant
                filename = "prediction_"+name.replace("/", "_")+".json"
                prediction = {**result["outputs"][variant], "appearance": appearance,
                    "score_semantics": "percentile rank or equal-three percentile mean, not raw cosine"}
                write(output/filename, prediction)
                records.append({"name": name, "file": filename, "sha256": sha(output/filename), "new": True})
            print(f"rank replay {appearance}/{decoder} both variants complete", flush=True)
    require([r["name"] for r in records] == NEW_OUTPUTS, "all sixteen new outputs required")
    return records


def infer(family, expected_commit, input_freeze_sha256):
    family = utility.allowed(family).resolve(strict=True)
    plan, bundle = verify_prepared(family, expected_commit, input_freeze_sha256)
    output = family/"predictions"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        before = resource_gate(started)
        with utility.offline(), numerical_threads() as pools:
            calibrated = core.calibrate_evidence(bundle["evidence"])
            write(output/"calibration.json", {k: v for k, v in calibrated.items() if k != "matrices"})
            np.savez(output/"calibrated_evidence.npz", **{k: bundle["arrays"][k] for k in SHARED}, **calibrated["matrices"])
            records = decode_all(output, calibrated, bundle["arrays"], started)
            for row in bundle["controls"]:
                copy(family/"inputs"/row["file"], output/row["file"], row["sha256"])
                records.append({**row, "new": False})
            require([r["name"] for r in records] == ALL_OUTPUTS
                and all(sha(output/r["file"]) == r["sha256"] for r in records), "complete forty-output comparison required")
            write(output/"comparison_index.json", {"outputs": records, "selected_variant": None,
                "new_outputs": 16, "retained_controls": 24, "profile_sha256": PROFILE_SHA256})
            write(output/"runtime.json", {"source_commit": expected_commit, "profile_sha256": PROFILE_SHA256,
                "input_freeze_sha256": input_freeze_sha256, "source_sha256": plan["source_sha256"],
                "input_identity": bundle["identity"], "elapsed_s": time.monotonic()-started,
                "memory_before": before, "memory_end": resource_gate(started), "threadpools": pools,
                "libraries": {"numpy": np.__version__, "scipy": scipy.__version__},
                "torch_imported": "torch" in sys.modules, "labels_opened": False, "new_forwards": False,
                "old_controls_redecoded": False, "selected_variant": None})
            verify_prepared(family, expected_commit, input_freeze_sha256)
            require(all(sha(output/r["file"]) == r["sha256"] for r in records), "prediction changed before master freeze")
            write(output/"frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
                "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256,
                "input_freeze_sha256": input_freeze_sha256, "all_requested_outputs": ALL_OUTPUTS,
                "new_outputs": 16, "retained_controls": 24, "selected_variant": None,
                "artifacts": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}})
        return {"status": "complete", "outputs": 40, "freeze_sha256": sha(output/"frozen_predictions.json")}
    except Exception as error:
        write(output/"failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "elapsed_s": time.monotonic()-started, "labels_opened": False, "master_written": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup, run = sub.add_parser("prepare"), sub.add_parser("infer")
    setup.add_argument("--spec", type=Path, required=True)
    run.add_argument("--input-freeze-sha256", required=True)
    for command in (setup, run):
        command.add_argument("--family", type=Path, required=True)
        command.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare": args["spec_path"] = args.pop("spec")
    with utility.offline(): result = prepare(**args) if command == "prepare" else infer(**args)
    print(utility.json.dumps(result))
    return 0


if __name__ == "__main__": raise SystemExit(main())
