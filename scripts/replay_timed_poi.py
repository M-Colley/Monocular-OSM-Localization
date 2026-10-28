"""Freeze a cached timed-POI/erased-heading diagnostic, without model or GT IO."""
from __future__ import annotations
import argparse
import json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from monocular_osm import timed_poi_heading as core, poi_image_union as producer
from scripts import replay_heading_sequence as frozen

require, sha, load, write = core.require, frozen.sha, frozen.load, frozen.write
MEMBERS = {"cues": ("cues.json", "poi_ocr.json", "poi_inventory.json", "cues_vo.npz"),
    "union": ("image_union_plan.json", "image_reference_union.npz", "image_union_prediction.json")}
PROFILE = {"version": "cached-timed-poi-replay-v1", "posthoc_DEV": True,
    "core_profile_sha256": core.PROFILE_SHA256, "cpu_threads": 2,
    "wall_budget_s": 3600, "minimum_free_physical_and_true_commit_GiB": 8,
    "max_queries": 80, "max_references": 9000,
    "outputs": "both_normalizations_times_all_branches_plus_null_plus_joint",
    "all_declared_outputs_before_master": True, "network_models_or_GT": False,
    "rigid_VO_or_metric_translation": False, "selected_variant": None,
    "confidence": None, "default_policy_adopted": False}
PROFILE_SHA256 = core.base._json_sha(PROFILE)


def source_identity(expected, previous=None):
    require(Path(__file__).resolve() == ROOT / "scripts/replay_timed_poi.py" and frozen.ROOT == ROOT,
            "executed timed runner escaped source pin")
    result = frozen.source_identity(expected)
    result["scripts/replay_timed_poi.py"] = sha(Path(__file__))
    require(previous is None or result == previous, "timed replay source bytes changed")
    return result


def specification(path):
    value = load(path)
    require(isinstance(value, dict) and set(value) == {"cues", "union", "parent"}, "three exact frozen input roles required")
    for row in value.values():
        require(isinstance(row, dict) and set(row) == {"freeze", "freeze_sha256"}
                and isinstance(row["freeze"], str) and frozen.digest(row["freeze_sha256"]), "fixed freeze path/SHA required")
        frozen.allowed(row["freeze"])
    return value


def original_inputs(spec):
    records, paths, identities, manifests = [], {}, {}, {}
    for role in ("cues", "union", "parent"):
        row = spec[role]
        path = frozen.allowed(row["freeze"]).resolve(strict=True)
        manifest = frozen.freeze(path, row["freeze_sha256"])
        manifests[role] = (path, manifest)
        require(manifest.get("scoring_has_not_run") is True, "input did not freeze before its historical scoring")
        records.append((role, "original_freeze.json", path, row["freeze_sha256"]))
        identities[role] = manifest.get("video_sha256")
        if role in MEMBERS:
            require(set(MEMBERS[role]) <= set(manifest["artifacts"]), "frozen matrix/native cue roles are incomplete")
            paths[role] = {}
            for name in MEMBERS[role]:
                source = path.parent / frozen.relative(name)
                paths[role][name] = source
                records.append((role, name, source, manifest["artifacts"][name]))
    require(identities["cues"] == identities["union"] and frozen.digest(identities["cues"]), "cue/appearance source videos differ")
    parent_path, parent = manifests["parent"]
    members = {str(frozen.relative(k)).replace("\\", "/"): v for k, v in parent["artifacts"].items()}
    for role in ("cues", "union"):
        path = manifests[role][0]
        require(path.is_relative_to(parent_path.parent), "child evidence is outside parent freeze")
        key = str(path.relative_to(parent_path.parent)).replace("\\", "/")
        require(members.get(key) == spec[role]["freeze_sha256"], "parent does not bind the exact child freeze")
    return records, paths, identities["cues"]


def typed_inputs(paths, video_sha256):
    cues, ocr, inventory = [load(paths["cues"][name]) for name in MEMBERS["cues"][:3]]
    declared = load(paths["union"]["image_union_plan.json"])
    proof = load(paths["union"]["image_union_prediction.json"])
    path = paths["union"]["image_reference_union.npz"]
    specs = {"ref_ids": (9000, "U", 512), "ref_latlon": (18000, "f", 8),
        "query_times": (80, "f", 8), "query_indices": (80, "iu", 8), "similarities": (720000, "f", 4)}
    arrays = frozen.npz_arrays(path, specs)
    ids, refs, times, indices, scores = [arrays[key] for key in specs]
    require(ids.ndim == 1 and 1 <= len(ids) <= PROFILE["max_references"]
            and len(set(ids.tolist())) == len(ids) and all(str(v).startswith("mapillary:") for v in ids)
            and refs.shape == (len(ids), 2) and times.shape == indices.shape == (80,)
            and scores.shape == (80, len(ids)) and np.isfinite(scores).all() and np.max(np.abs(scores)) <= 1.00001,
            "complete typed shared80/full-gallery appearance required")
    native = paths["cues"]["cues_vo.npz"]
    frozen.bounded_header(native, "K", 9, "f", 8, (3, 3))
    frozen.bounded_header(native, "timestamps", 100000, "f", 8)
    with np.load(native, allow_pickle=False) as saved:
        K, native_times = saved["K"], saved["timestamps"]
    require(native_times.ndim == 1 and len(native_times) >= 80 and np.isfinite(native_times).all() and np.all(np.diff(native_times) > 0)
            and indices[0] >= 0 and indices[-1] < len(native_times) and np.all(np.diff(indices) > 0)
            and np.array_equal(indices, np.linspace(0, len(native_times)-1, 80).astype(np.int64))
            and np.array_equal(native_times[indices], times), "shared query times/indices differ from native camera grid")
    auto = cues["auto_quality"]
    window = np.asarray(auto["window"])
    require(auto["status"] == "complete" and auto["n_frames"] == len(native_times)
            and auto["timestamps_sha256"] == core.base._array_sha(native_times)
            and auto["video_sha256"] == video_sha256 and window.shape == (2,) and window.dtype.kind in "fiu"
            and np.isfinite(window).all() and 0 <= window[0] <= native_times[0] < native_times[-1] <= window[1],
            "native timestamp/input identity differs")
    provenance = ocr["provenance"]
    require(cues["video_sha256"] == provenance["video_sha256"] == video_sha256
            and provenance["window"] == [float(native_times[0]), float(native_times[-1])]
            and declared["ocr_provenance"] == provenance and declared["clues"] == cues["clues"],
            "native OCR source/branch plan differs from acquisition")
    common = proof["provenance"]["common_query_encoder_identity"]
    identity = common["query_identity"]
    require(common["backbone"]["name"] == "megaloc" and common["backbone"]["device"] == "cpu"
            and common["backbone"]["hub"] == "gmberton/MegaLoc:5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
            and identity["query_video_sha256"] == video_sha256 and identity["query_mode"] == "crops"
            and identity["views"] == 3 and identity["query_indices_sha256"] == producer._array_sha(indices)
            and identity["aggregation"] == producer.PROFILE["crop_aggregation"]
            and identity["query_times_sha256"] == producer._array_sha(times)
            and all(frozen.digest(identity[k]) for k in ("query_pixels_sha256", "query_descriptors_sha256", "actual_model_state_sha256")),
            "actual common query/encoder identity is incomplete or mismatched")
    producer._validate_plan(declared)
    require(proof["status"] == "complete" and proof["route_valid"] is True and proof["failed_regions"] == []
            and proof["profile"] == dict(producer.PROFILE)
            and proof["successful_branch_count"] == declared["branch_count"], "source all-branch acquisition was partial")
    hashes = {"query_indices_sha256": producer._array_sha(indices), "query_times_sha256": producer._array_sha(times),
        "ref_ids_sha256": producer._json_sha(ids.tolist()), "ref_latlon_sha256": producer._array_sha(refs),
        "similarities_sha256": producer._array_sha(scores)}
    require(proof["reference_union"]["hashes"] == hashes and proof["reference_union"]["ref_ids"] == ids.tolist(),
            "union full-column/row identity differs from producer evidence")
    plan = core.build_plan(cues["clues"], ocr["detections"], inventory, provenance, K, times)
    return arrays, plan, {"video_sha256": video_sha256, "query_identity": identity,
        "union_hashes": hashes, "registered_window": cues["auto_quality"]["window"],
        "K_sha256": core.base._array_sha(K), "native_times_sha256": core.base._array_sha(native_times)}


def requested_outputs(plan):
    return [variant+"/"+component for variant in core.PROFILE["variants"]
            for component in ["null", *[b["id"] for b in plan["branches"]], "joint"]]


def prepare(spec, family, expected_commit):
    sources = source_identity(expected_commit)
    frozen.check_headroom()
    spec, family = frozen.allowed(spec).resolve(strict=True), frozen.allowed(family).resolve()
    declared = specification(spec)
    roots = [Path(row["freeze"]).resolve().parent for row in declared.values()]+[ROOT]
    require(not family.exists() and not any(family.is_relative_to(root) or root.is_relative_to(family) for root in roots),
            "new isolated timed replay family required")
    records, paths, video = original_inputs(declared)
    _, plan, identity = typed_inputs(paths, video)
    family.mkdir(parents=True)
    copied = []
    try:
        frozen.base.copy_verified(spec, family / "inputs/spec.json", sha(spec), copied)
        for role, name, path, digest in records:
            frozen.base.copy_verified(path, family / "inputs" / role / frozen.relative(name), digest, copied)
        for relative, digest in sources.items():
            target = family / "inputs/source_snapshot" / relative
            frozen.base.copy_verified(ROOT / relative, target, digest, copied)
        write(family / "timed_plan.json", plan)
        write(family / "registration.json", {"source_commit": expected_commit, "source_sha256": sources,
            "profile": PROFILE, "profile_sha256": PROFILE_SHA256, "input_identity": identity,
            "all_requested_outputs": requested_outputs(plan), "selected_variant": None,
            "models_or_predictions_run": False, "labels_opened": False})
        write(family / "input_lineage.json", {"copied": copied, "original_spec": declared})
        source_identity(expected_commit, sources)
        original_inputs(declared)
        frozen.base.verify_originals(copied)
        artifacts = {str(Path(row["path"]).relative_to(family)): row["sha256"] for row in copied}
        artifacts.update({name: sha(family / name) for name in ("timed_plan.json", "registration.json", "input_lineage.json")})
        write(family / "inputs_frozen.json", {"scoring_has_not_run": True, "models_or_predictions_run": False,
            "artifacts": artifacts, "all_requested_outputs": requested_outputs(plan)})
        return {"status": "prepared_only", "output_count": len(requested_outputs(plan)),
                "input_freeze_sha256": sha(family / "inputs_frozen.json"), "profile_sha256": PROFILE_SHA256}
    except Exception as error:
        write(family / "preparation_failed.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "models_or_predictions_run": False, "labels_opened": False})
        raise


def prepared_inputs(family, expected, commit):
    frozen.freeze(family / "inputs_frozen.json", expected)
    registration = load(family / "registration.json")
    require(registration["source_commit"] == commit and registration["profile"] == PROFILE
            and registration["profile_sha256"] == PROFILE_SHA256 and registration["selected_variant"] is None,
            "registered timed source/profile differs")
    source_identity(commit, registration["source_sha256"])
    declared = specification(family / "inputs/spec.json")
    _, _, video = original_inputs(declared)
    lineage = load(family / "input_lineage.json")
    require(lineage["original_spec"] == declared, "original input lineage differs")
    frozen.base.verify_originals(lineage["copied"])
    paths = {role: {name: family / "inputs" / role / name for name in names} for role, names in MEMBERS.items()}
    arrays, plan, identity = typed_inputs(paths, video)
    require(plan == load(family / "timed_plan.json") and identity == registration["input_identity"]
            and requested_outputs(plan) == registration["all_requested_outputs"], "frozen timed event/branch/query plan differs")
    return arrays, plan, registration


def infer(family, expected_commit, input_freeze_sha256):
    require(frozen.digest(input_freeze_sha256), "explicit prepared input-freeze SHA required")
    family = frozen.allowed(family).resolve(strict=True)
    arrays, plan, registration = prepared_inputs(family, input_freeze_sha256, expected_commit)
    output = family / "predictions"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    try:
        before = frozen.check_headroom()
        require("torch" not in sys.modules, "cached diagnostic refuses a loaded neural runtime")
        print(f"timed POI: all{len(arrays['ref_ids'])}refs/all{len(plan['branches'])}branches/null; complete decode starting", flush=True)
        result = core.decode_timed_tracks(arrays["similarities"], arrays["ref_ids"], arrays["ref_latlon"], arrays["query_times"], plan)
        wanted = registration["all_requested_outputs"]
        require(result["profile_sha256"] == core.PROFILE_SHA256 and set(result["outputs"]) == set(wanted)
                and result["confidence"] is None and result["selected_variant"] is None, "all fixed outputs are required")
        write(output / "predictions.json", result)
        records = []
        for name in wanted:
            filename = "prediction_"+name.replace("/", "_")+".json"
            write(output / filename, result["outputs"][name])
            records.append({"name": name, "file": filename, "sha256": sha(output / filename)})
        require(time.monotonic()-started <= PROFILE["wall_budget_s"], "registered wall budget exceeded")
        end = frozen.check_headroom()
        prepared_inputs(family, input_freeze_sha256, expected_commit)
        require("torch" not in sys.modules and all(sha(output / r["file"]) == r["sha256"] for r in records),
                "source/runtime/prediction bytes changed before freeze")
        write(output / "runtime.json", {"status": "complete", "source_commit": expected_commit,
            "source_sha256": registration["source_sha256"], "profile_sha256": PROFILE_SHA256,
            "input_freeze_sha256": input_freeze_sha256, "input_identity": registration["input_identity"],
            "elapsed_s": time.monotonic()-started, "memory_before": before, "memory_end": end,
            "outputs": records, "labels_opened": False, "models_or_network_run": False,
            "selected_variant": None, "confidence": None})
        write(output / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
            "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256,
            "input_freeze_sha256": input_freeze_sha256, "all_requested_outputs": wanted,
            "selected_variant": None, "confidence": None,
            "artifacts": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
        return {"status": "complete", "outputs": len(wanted), "freeze_sha256": sha(output / "frozen_predictions.json")}
    except Exception as error:
        write(output / "failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "labels_opened": False, "scoring_has_not_run": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare");setup.add_argument("--spec", type=Path, required=True)
    run = commands.add_parser("infer");run.add_argument("--input-freeze-sha256", required=True)
    for command in (setup, run):
        command.add_argument("--family", type=Path, required=True)
        command.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv));command = args.pop("command")
    with frozen.offline():
        result = prepare(**args) if command == "prepare" else infer(**args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
