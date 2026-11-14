"""Standalone frozen metadata-union replay; no CLI policy/default wiring.

Prepare is model-free and never decodes. Infer requires a clean source pin,
accepted input SHA and external8GiB physical/true-commit CPU2 watchdog.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[key] = "2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from monocular_osm import metadata_union as core
from scripts import replay_heading_sequence as utility
from scripts.replay_rank_fusion import numerical_threads

require, sha, load, write = utility.require, utility.sha, utility.load, utility.write
ROLES = {"parent": "freeze", "baseline": "freeze", "expanded": "freeze", "audit": "path", "proposal": "path"}
PROFILE = {"version": "frozen-metadata-union-replay-v1", "posthoc_DEV": True,
    "core_profile_sha256": core.PROFILE_SHA256, "proposal_sha256": dict(core.PROFILE)["proposal_file_sha256"],
    "proposal_profile_sha256": dict(core.PROFILE)["proposal_profile_sha256"],
    "regions": 6, "reference_occurrences": 8700, "queries": 80,
    "research_outputs": ["raw", "reference_mean"], "sole_new_primary": "reference_mean",
    "original_controls": "unaltered source JSON/NPZ artifacts; no reformatted old prediction claim",
    "reference_identity": "UNVERIFIED legacy reference encodings retained",
    "comparison": "selected global reference maps to original pool/column/query-row RAW cosine; true full-DP profile",
    "source_scope": "metadata admission before original municipality failure; original partial expanded child remains a control",
    "cpu_threads": 2, "minimum_free_physical_and_true_commit_GiB": 8,
    "runtime_wall_budget_s": 3600, "external_watchdog_required": True,
    "selected_research_variant": None, "policy_adopted": False, "models_or_network_or_GT": False}
PROFILE_SHA256 = core.json_sha(PROFILE)


def specification(path):
    value = load(path)
    require(isinstance(value, dict) and set(value) == {*ROLES, "original_source_commit"}
        and isinstance(value["original_source_commit"], str) and re.fullmatch("[0-9a-f]{40}", value["original_source_commit"]) is not None,
        "exact accepted source/audit/proposal roles required")
    for role, key in ROLES.items():
        row = value[role]
        require(isinstance(row, dict) and set(row) == {key, "sha256"} and isinstance(row[key], str)
            and utility.digest(row["sha256"]), "explicit role path and SHA required")
        utility.allowed(row[key])
    return value


def source_identity(expected, previous=None):
    require(Path(__file__).resolve() == ROOT/"scripts/replay_metadata_union.py" and utility.ROOT == ROOT, "executed driver escaped source pin")
    sources = utility.source_identity(expected)
    for name, module in list(sys.modules.items()):
        if name.startswith(("scripts.", "monocular_osm.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve(); require(path.is_relative_to(ROOT), "foreign project source loaded")
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    sources["scripts/replay_metadata_union.py"] = sha(__file__)
    require(previous is None or previous == sources, "executed source bytes changed")
    return sources


def artifacts(value):
    result, seen = {}, set()
    for path, digest in value["artifacts"].items():
        utility.allowed(path); name = utility.relative(path).as_posix()
        require(name.casefold() not in seen and utility.digest(digest), "duplicate artifact path alias")
        seen.add(name.casefold()); result[name] = digest
    return result


def bound_member(freeze, frozen, relative):
    root = Path(freeze).resolve().parent; name = utility.relative(relative).as_posix()
    path = utility.allowed(root/name).resolve(strict=True)
    require(path.is_relative_to(root) and artifacts(frozen).get(name) == sha(path), "required source artifact not in accepted freeze")
    return path


def read_region(meta_path, array_path):
    meta = load(meta_path)
    require(meta["array_artifact"]["sha256"] == sha(array_path), "regional array source changed")
    specs = {"query_indices": (80, "iu", 8), "query_times": (80, "f", 8), "ref_ids": (1500, "US", 512),
        "ref_latlon": (3000, "f", 8), "raw_similarities": (120000, "f", 4),
        "decoded_ref_indices": (80, "iu", 8), "decoded_cosines": (80, "f", 8),
        "returned_indices": (80, "iu", 8), "returned_latlon": (160, "f", 8), "returned_cosines": (80, "f", 8)}
    arrays = utility.npz_arrays(array_path, specs)
    require(set(meta["array_artifact"]["arrays"]) == set(arrays), "regional typed roles differ")
    for key, value in arrays.items():
        record = meta["array_artifact"]["arrays"][key]
        require(record == {"sha256": core.full._array_sha(value), "shape": list(value.shape), "dtype": value.dtype.str}, "regional typed array proof differs")
    return meta, arrays


def read_baseline_pools(freeze_path, frozen, comparison):
    """Bind every named AND frozen baseline audit, preserving numeric ordering."""
    root = freeze_path.parent
    records = comparison.get("audit_artifacts")
    require(isinstance(records, list) and 2 <= len(records) <= 2*core.PROFILE["maximum_baseline_audits"], "all baseline audit artifacts required")
    paths = {}
    for record in records:
        require(isinstance(record, dict) and set(record) == {"path", "sha256"} and utility.digest(record["sha256"]), "invalid baseline audit artifact record")
        path = utility.allowed(record["path"]).resolve(strict=True)
        require(path.is_relative_to(root), "baseline audit artifact escaped child")
        name = path.relative_to(root).as_posix()
        require(re.fullmatch(r"retrieval_audit/[0-9]{4}\.(json|npz)", name) is not None and name not in paths
            and bound_member(freeze_path, frozen, name) == path and sha(path) == record["sha256"], "baseline audit artifact is stale or repeated")
        paths[name] = path
    frozen_names = {n for n in artifacts(frozen) if n.startswith("retrieval_audit/") and n.endswith((".json", ".npz"))}
    directory_names = {p.relative_to(root).as_posix() for p in (root/"retrieval_audit").iterdir() if p.suffix in {".json", ".npz"}}
    require(set(paths) == frozen_names == directory_names, "baseline audit inventory omitted an original pool")
    n = len(paths)//2
    require(set(paths) == {f"retrieval_audit/{i:04d}.{suffix}" for i in range(n) for suffix in ("json", "npz")}, "baseline audit pairs/ordinals incomplete")
    regions, originals = [], []
    for i in range(n):
        meta_path, array_path = (paths[f"retrieval_audit/{i:04d}.{suffix}"] for suffix in ("json", "npz"))
        meta, arrays = read_region(meta_path, array_path)
        require(utility.allowed(meta["array_artifact"]["path"]).resolve(strict=True) == array_path, "baseline metadata names a different matrix")
        regions.append({"key": f"original_baseline_{i:04d}", "audit": meta, "arrays": arrays})
        originals += [{"source": str(p), "copy": f"controls/baseline_retrieval_audit/{i:04d}.{suffix}", "sha256": sha(p)}
            for p, suffix in ((meta_path,"json"),(array_path,"npz"))]
    selected = core.prove_baseline_selection(regions, comparison)
    return regions, originals, selected


def verified_source(spec):
    paths = {role: utility.allowed(spec[role][key]).resolve(strict=True) for role, key in ROLES.items()}
    freezes = {role: utility.freeze(paths[role], spec[role]["sha256"]) for role in ("parent", "baseline", "expanded")}
    parent_root = paths["parent"].parent
    require(freezes["parent"].get("scoring_has_not_run") is True, "original parent was not frozen before scoring")
    for role in ("baseline", "expanded"):
        require(paths[role].is_relative_to(parent_root)
            and bound_member(paths["parent"], freezes["parent"], paths[role].relative_to(parent_root)) == paths[role]
            and freezes[role].get("kind") == role, "child freeze is not bound to accepted parent")
    require(sha(paths["audit"]) == spec["audit"]["sha256"], "accepted audit SHA differs")
    audit = load(paths["audit"])
    require(audit.get("status") == "PASS" and audit.get("freeze_sha256") == spec["parent"]["sha256"]
        and audit.get("source_commit") == spec["original_source_commit"] and audit.get("ground_truth_read") is False
        and audit.get("numeric_accuracy_scores_read") is False and audit.get("new_models_or_forwards") is False
        and audit.get("protected_unchanged") is True and audit.get("source_clean") is True,
        "complete source-bound pre-GT input audit required")
    protected = audit["protected_hashes"]
    require(isinstance(protected, dict) and 1 <= len(protected) <= 20000, "bounded original mutable/static hash ledger required")
    validated = [(utility.allowed(p).resolve(strict=True), d) for p, d in protected.items()]
    for path, digest in validated: require(utility.digest(digest) and sha(path) == digest, "audited original destination/source changed")
    old_source = utility.allowed(audit["source"]).resolve(strict=True)
    git = lambda *args: subprocess.check_output(["git", "-C", str(old_source), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == spec["original_source_commit"] and not git("status", "--porcelain", "--untracked-files=all"), "original source pin changed")
    require(sha(paths["proposal"]) == spec["proposal"]["sha256"] == PROFILE["proposal_sha256"], "fixed proposal changed")
    proposal = load(paths["proposal"])
    require(proposal.get("proposal_sha256") == PROFILE["proposal_profile_sha256"], "fixed proposal profile differs")
    baseline_path = bound_member(paths["baseline"], freezes["baseline"], "candidate.json")
    admission_path = bound_member(paths["expanded"], freezes["expanded"], "metadata_admission/admission.json")
    baseline, admission = load(baseline_path), load(admission_path)
    require(baseline.get("status") == "complete" and baseline.get("kind") == "baseline"
        and admission["decision"] == core.policy.decide_assisted_policy(admission["evidence"]), "original baseline/admission proof differs")
    require(admission["evidence"]["expanded"]["status"] == "complete", "metadata subplan must be complete despite final municipality partial status")
    directory = paths["expanded"].parent/"metadata_admission/retrieval_audit"
    require(sorted(p.name for p in directory.glob("*.json")) == [f"{i:04d}.json" for i in range(6)]
        and sorted(p.name for p in directory.glob("*.npz")) == [f"{i:04d}.npz" for i in range(6)], "all six metadata matrix/control roles required")
    regions, originals = [], []
    for i in range(6):
        meta_path = bound_member(paths["expanded"], freezes["expanded"], f"metadata_admission/retrieval_audit/{i:04d}.json")
        array_path = bound_member(paths["expanded"], freezes["expanded"], f"metadata_admission/retrieval_audit/{i:04d}.npz")
        meta, arrays = read_region(meta_path, array_path)
        require(arrays["raw_similarities"].shape == (80, 1200 if i == 0 else 1500), "fixed complete six-pool shape differs")
        regions.append({"key": f"region_{i:04d}", "audit": meta, "arrays": arrays})
        originals += [{"source": str(meta_path), "copy": f"regions/{i:04d}/audit.json", "sha256": sha(meta_path)},
                      {"source": str(array_path), "copy": f"regions/{i:04d}/data.npz", "sha256": sha(array_path)}]
    base_summary, meta_summary = admission["evidence"]["baseline"], admission["evidence"]["expanded"]
    core.legacy_summary(baseline["comparison"], base_summary)
    core.legacy_summary(admission["comparison"], meta_summary)
    identity, indices, times = core.appearance(baseline["comparison"])
    for region in regions: core.validate_region(region, identity, indices, times)
    originals += [{"source": str(baseline_path), "copy": "controls/baseline_candidate.json", "sha256": sha(baseline_path)},
                  {"source": str(admission_path), "copy": "controls/metadata_admission.json", "sha256": sha(admission_path)}]
    # The baseline can choose a fine pool or mix pools per query; no 0000 assumption.
    baseline_regions, baseline_originals, baseline_selection = read_baseline_pools(paths["baseline"], freezes["baseline"], baseline["comparison"])
    originals += baseline_originals
    identity_record = {"appearance": identity, "regions": [{"key": r["key"], "arrays": {k: core.full._array_sha(v) for k, v in r["arrays"].items()},
        "audit_sha256": originals[i*2]["sha256"]} for i, r in enumerate(regions)],
        "baseline_comparison_sha256": core.json_sha(baseline["comparison"]), "metadata_comparison_sha256": core.json_sha(admission["comparison"]),
        "baseline_audits": [{"key": region["key"], "arrays": {k: core.full._array_sha(v) for k,v in region["arrays"].items()}}
            for region in baseline_regions], "baseline_selected_sources": baseline_selection}
    return {"regions": regions, "baseline": baseline, "admission": admission, "identity": identity_record,
            "baseline_regions": baseline_regions, "originals": originals, "paths": paths}


def copy(source, target, expected):
    source, target = utility.allowed(source).resolve(strict=True), utility.allowed(target)
    require(sha(source) == expected and not target.exists(), "copy changed or destination exists")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    require(sha(target) == expected, "copied input bytes changed")


def prepare(spec_path, family, expected_commit):
    sources = source_identity(expected_commit); utility.check_headroom()
    spec_path = utility.allowed(spec_path).resolve(strict=True); spec = specification(spec_path)
    family = utility.allowed(family).resolve()
    roots = [ROOT, Path(spec["parent"]["freeze"]).resolve().parent, Path(spec["audit"]["path"]).resolve().parent,
             Path(spec["proposal"]["path"]).resolve().parent]
    require(not family.exists() and not any(family.is_relative_to(p) or p.is_relative_to(family) for p in roots), "new isolated replay family required")
    bundle = verified_source(spec); family.mkdir(parents=True)
    try:
        copy(spec_path, family/"inputs/spec.json", sha(spec_path))
        for role, key in ROLES.items(): copy(spec[role][key], family/f"inputs/original_{role}.json", spec[role]["sha256"])
        for row in bundle["originals"]: copy(row["source"], family/"inputs"/row["copy"], row["sha256"])
        for name, digest in sources.items(): copy(ROOT/name, family/"inputs/source_snapshot"/name, digest)
        write(family/"plan.json", {"profile": PROFILE, "profile_sha256": PROFILE_SHA256, "source_commit": expected_commit,
            "source_sha256": sources, "declared": spec, "input_identity": bundle["identity"], "originals": bundle["originals"],
            "sole_new_primary": "reference_mean", "research_outputs": ["raw", "reference_mean"]})
        require(verified_source(spec)["identity"] == bundle["identity"], "original source changed during preparation")
        source_identity(expected_commit, sources)
        write(family/"inputs_frozen.json", {"status": "prepared_only", "new_DP_has_not_run": True, "models_or_GT": False,
            "artifacts": {p.relative_to(family).as_posix(): sha(p) for p in family.rglob("*") if p.is_file()}})
        return {"status": "prepared_only", "input_freeze_sha256": sha(family/"inputs_frozen.json"), "profile_sha256": PROFILE_SHA256}
    except Exception as error:
        write(family/"preparation_failed.json", {"status": "failed", "error": str(error)}); raise


def verify_prepared(family, expected_commit, expected_inputs):
    require(utility.digest(expected_inputs), "explicit accepted input SHA required")
    utility.freeze(family/"inputs_frozen.json", expected_inputs)
    plan = load(family/"plan.json")
    require(plan["source_commit"] == expected_commit and plan["profile"] == PROFILE and plan["profile_sha256"] == PROFILE_SHA256
        and plan["sole_new_primary"] == "reference_mean" and plan["research_outputs"] == ["raw", "reference_mean"], "registered source/profile/primary changed")
    source_identity(expected_commit, plan["source_sha256"])
    spec = specification(family/"inputs/spec.json"); require(spec == plan["declared"], "original input roles changed")
    bundle = verified_source(spec)
    require(bundle["identity"] == plan["input_identity"] and bundle["originals"] == plan["originals"], "original input identity changed")
    for row in bundle["originals"]: require(sha(family/"inputs"/row["copy"]) == row["sha256"], "prepared source-control copy changed")
    return plan, bundle


def resource_gate(started):
    require(time.monotonic()-started <= 3600, "registered3600s wall budget exceeded")
    return utility.check_headroom()


def infer(family, expected_commit, input_freeze_sha256):
    family = utility.allowed(family).resolve(strict=True)
    plan, bundle = verify_prepared(family, expected_commit, input_freeze_sha256)
    output = family/"predictions"; output.mkdir(exist_ok=False); started = time.monotonic()
    try:
        before = resource_gate(started)
        with utility.offline(), numerical_threads() as pools:
            admission = bundle["admission"]
            result = core.replay_metadata_union(bundle["regions"], bundle["baseline"]["comparison"], admission["evidence"]["baseline"],
                admission["comparison"], admission["evidence"]["expanded"], bundle["baseline_regions"])
            if result["status"] == "complete":
                require(set(result["outputs"]) == {"raw", "reference_mean"}, "both registered research routes required")
                union = result["union"]
                np.savez(output/"union_evidence.npz", **{k: union[k] for k in ("ref_ids", "ref_latlon", "raw_similarities", "query_indices", "query_times")})
                write(output/"union_proof.json", {k: v for k, v in union.items() if not isinstance(v, np.ndarray)})
                for variant in ("raw", "reference_mean"):
                    write(output/f"prediction_{variant}.json", result["outputs"][variant])
                    write(output/f"comparison_{variant}.json", result["comparisons"][variant])
                write(output/"decoder_evidence.json", result["full_decoder_evidence"])
                write(output/"admission.json", result["admission"])
            else:
                require(result["status"] == "preserved" and result["selection"] == "legacy_metadata" and result["outputs"] == {}, "unsupported incomplete result")
                write(output/"admission.json", result)
            write(output/"selection.json", {"selection": result["selection"], "sole_new_primary": "reference_mean",
                "raw_is_diagnostic_only": True, "research_only_not_policy_adopted": True, "status": result["status"]})
            write(output/"runtime.json", {"source_commit": expected_commit, "source_sha256": plan["source_sha256"],
                "profile_sha256": PROFILE_SHA256, "input_freeze_sha256": input_freeze_sha256, "elapsed_s": time.monotonic()-started,
                "input_identity": bundle["identity"], "memory_before": before, "memory_end": resource_gate(started), "threadpools": pools,
                "torch_imported": "torch" in sys.modules, "models_or_forwards_or_GT": False})
            verify_prepared(family, expected_commit, input_freeze_sha256)
            write(output/"frozen_predictions.json", {"status": "complete", "decoding_status": result["status"],
                "scoring_has_not_run": True, "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256,
                "input_freeze_sha256": input_freeze_sha256, "sole_new_primary": "reference_mean", "selection": result["selection"],
                "both_research_outputs": result["status"] == "complete", "original_controls_preserved_as_source_artifacts": True,
                "artifacts": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}})
        return {"status": "complete", "freeze_sha256": sha(output/"frozen_predictions.json")}
    except Exception as error:
        write(output/"failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "models_or_GT": False}); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare"); setup.add_argument("--spec", type=Path, required=True)
    run = sub.add_parser("infer"); run.add_argument("--input-freeze-sha256", required=True)
    for command in (setup, run):
        command.add_argument("--family", type=Path, required=True); command.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    if command == "prepare": args["spec_path"] = args.pop("spec")
    with utility.offline(): result = prepare(**args) if command == "prepare" else infer(**args)
    print(json.dumps(result)); return 0


if __name__ == "__main__": raise SystemExit(main())
