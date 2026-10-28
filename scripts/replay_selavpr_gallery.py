"""Frozen full-gallery SelaVPR++ comparison; eight new routes, sixteen controls.

Prepare is model-free. Infer requires a reviewed clean pin and an external
8 GiB physical/actual-commit watchdog. Neither command accepts labels/scores.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
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
from monocular_osm import full_sequence as full, heading_sequence as heading_core
from scripts import replay_heading_sequence as heading, smoke_selavpr as smoke
from scripts import selavpr_backend as backend

gallery = smoke.gallery
require, sha, load, write = backend.require, heading.sha, heading.load, heading.write
NEW_OUTPUTS = ["selavpr/simple/" + v for v in ("raw", "reference_mean")] + [
    "selavpr/heading/" + v + "/" + m for v in heading_core.PROFILE["variants"] for m in heading_core.PROFILE["modes"]]
CONTROL_FILES = {
    **{f"controls/simple/{model}/{variant}": ("simple", f"prediction_{model}_{variant}.json")
       for model in ("frozen_megaloc", "ImAge_GSV") for variant in ("raw", "reference_mean")},
    **{f"controls/heading/{model}/{variant}/{mode}": ("heading", f"prediction_{model}_{variant}_{mode}.json")
       for model in ("megaloc", "image") for variant in heading_core.PROFILE["variants"] for mode in heading_core.PROFILE["modes"]},
}
PROFILE = {"version": "selavpr-full-gallery-simple-heading-v1", "posthoc_DEV": True,
    "validation_status": "unvalidated_development_comparison", "selected_variant": None,
    "backend_profile_sha256": backend.PROFILE_SHA256, "simple_profile_sha256": full.PROFILE_SHA256,
    "heading_profile_sha256": heading_core.PROFILE_SHA256, "motion_profile_sha256": heading.motion.PROFILE_SHA256,
    "new_outputs": NEW_OUTPUTS, "retained_control_outputs": list(CONTROL_FILES), "comparison_count": 24,
    "all_eight_new_outputs_required_before_master_freeze": True, "control_routes": "byte_identical_copies_no_redecode",
    "queries": 80, "query_views": 3, "references": 9735, "dimensions": 2048,
    "fresh_forward_count": 9975, "batch_size": 1, "cpu_threads": 2, "interop_threads": 1,
    "smoke_descriptor_replay_atol": 2e-6,
    "reference_block": 256, "query_pooling": "maximum_three_actual_native_crop_cosines",
    "canonical_pixels": "existing_frozen_first_pool_occurrence_for_every_ID",
    "raw_reference_storage": "BGR_512_square", "neural_input": "official_RGB_tensor_normalize_then_322_resize",
    "topK": False, "new_images_or_weights": False, "network": False, "labels_or_scores": False,
    "runtime_wall_budget_s": 10800, "minimum_free_physical_and_true_commit_GiB": 8,
    "extra_tree_RSS_target_GiB": 3, "external_watchdog_required": True,
    "primary_routes": "unchanged_simple_median5_and_unsmoothed_joint_heading_MAP_both_predeclared",
    "confidence": None, "default_adopted": False,
    "legacy_megaloc_reference_encoding_identity": "UNVERIFIED_no_causal_encoder_only_claim"}
PROFILE_SHA256 = heading.core._json_sha(PROFILE)
ROLES = ("gallery", "simple", "heading", "smoke")


def specification(path):
    value = load(path)
    require(isinstance(value, dict) and set(value) == {*ROLES, "assets"}, "exact experiment input roles required")
    for role in ROLES:
        row = value[role]
        require(isinstance(row, dict) and set(row) == {"freeze", "freeze_sha256"}
                and isinstance(row["freeze"], str) and heading.digest(row["freeze_sha256"]), "fixed freeze path/SHA required")
        heading.allowed(row["freeze"])
    require(isinstance(value["assets"], str), "local inspected asset root required")
    heading.allowed(value["assets"])
    return value


def smoke_proof(path, expected):
    """Independent arithmetic/identity audit of the accepted runtime-only smoke."""
    path = Path(path).resolve(strict=True)
    frozen = heading.freeze(path, expected)
    require(frozen.get("status") == "complete" and frozen.get("forward_count") == 44
            and frozen.get("descriptors_shape") == [44, 2048] and frozen.get("no_accuracy_result") is True
            and frozen.get("labels_opened") is False and frozen.get("route_decoding") is False
            and frozen.get("profile_sha256") == smoke.PROFILE_SHA256, "complete runtime-only smoke required")
    root = path.parent.parent
    heading.freeze(root / "inputs_frozen.json", frozen["input_freeze_sha256"])
    registration, runtime = load(root / "registration.json"), load(path.parent / "runtime.json")
    require(registration["profile"] == smoke.PROFILE and registration["profile_sha256"] == smoke.PROFILE_SHA256
            and registration["source_commit"] == frozen["source_commit"], "smoke source/profile differs")
    heading.bounded_header(path.parent / "descriptors.npy", None, 44 * 2048, "f", 4, (44, 2048))
    descriptors = np.load(path.parent / "descriptors.npy", allow_pickle=False)
    norms = np.linalg.norm(descriptors, axis=1)
    actual = runtime["runtime"]
    require(np.isfinite(descriptors).all() and np.max(np.abs(norms - 1)) <= 1e-4
            and actual["forward_calls"] == 44 and actual["cpu_threads"] == 2 and actual["interop_threads"] == 1
            and actual["device"] == "cpu" and actual["strict_state_load"] is True
            and not any(actual["xformers_available"].values())
            and runtime["state"]["elements"] == 97909629 and heading.digest(runtime["state"]["sha256"]),
            "smoke normalization/runtime/state proof differs")
    require(len(runtime["per_image"]) == 44 and all(row["preprocessing"]["canonical_bitwise_equal"] is True
            and heading.digest(row["preprocessing"]["input_tensor_sha256"])
            and np.isfinite(row["seconds"]) and row["seconds"] >= 0
            and abs(row["descriptor_norm"] - float(norm)) <= 1e-6 for row, norm in zip(runtime["per_image"], norms)),
            "smoke per-image preprocessing/timing/norm proof differs")
    return {"freeze_sha256": expected, "input_freeze_sha256": frozen["input_freeze_sha256"],
        "source_commit": frozen["source_commit"], "profile_sha256": smoke.PROFILE_SHA256,
        "state": runtime["state"], "torch": actual["torch"], "torchvision": actual["torchvision"],
        "assets": registration["assets"], "descriptors_sha256": sha(path.parent / "descriptors.npy"),
        "reference_columns": registration["reference_columns"],
        "constructor_s": runtime["constructor_s"], "all44_norms_and_preprocessing_verified": True}


def heading_inputs(root):
    paths = {role: {name: root / "inputs" / role / heading.relative(name)
                   for name in heading.MEMBERS[role]} for role in heading.ROLES}
    frozen = load(root / "inputs/metadata/original_freeze.json")
    paths["metadata"].update({name: root / "inputs/metadata" / heading.relative(name) for name in frozen["artifacts"]})
    plans = {role: load(root / "inputs" / role / "upstream_plan.json") for role in ("megaloc", "image")}
    return paths, plans


def control_linkage(heading_root, gallery_plan, spec):
    require(sha(heading_root / "inputs/image/original_freeze.json") == spec["simple"]["freeze_sha256"]
            and sha(heading_root / "inputs/megaloc/original_freeze.json") == gallery_plan["parent_prediction_freeze_sha256"],
            "simple and heading controls do not originate from the same frozen encoder evidence")
    return {role: sha(heading_root / "inputs" / role / "original_freeze.json") for role in ("metadata", "motion")}


def verified_inputs(spec):
    roots = {role: Path(spec[role]["freeze"]).resolve(strict=True) for role in ROLES}
    originals = {role: heading.freeze(roots[role], spec[role]["freeze_sha256"]) for role in ROLES}
    gallery_root, heading_root = roots["gallery"].parent, roots["heading"].parent.parent
    plan, times = smoke.verify_gallery(gallery_root, spec["gallery"]["freeze_sha256"])
    require(roots["simple"].parent.parent == gallery_root
            and originals["simple"].get("status") == "complete"
            and originals["simple"].get("profile_sha256") == gallery.PROFILE_SHA256
            and originals["simple"].get("scoring_has_not_run") is True
            and originals["heading"].get("status") == "complete"
            and originals["heading"].get("scoring_has_not_run") is True
            and originals["heading"].get("selected_variant") is None
            and originals["heading"].get("profile_sha256") == heading.PROFILE_SHA256
            and originals["heading"].get("all_requested_outputs") == heading.PROFILE["outputs"],
            "complete fixed simple/heading controls required")
    heading.freeze(heading_root / "inputs_frozen.json", originals["heading"]["input_freeze_sha256"])
    hp = load(heading_root / "plan.json")
    cue_freezes = control_linkage(heading_root, plan, spec)
    paths, plans = heading_inputs(heading_root)
    shared, _, matrices, cameras, angles, offsets, yaw, identity = heading.inspect_bundle(paths, plans)
    require(identity == hp["input_identity"] and hp["profile"] == heading.PROFILE
            and shared["ref_ids"].tolist() == [row["id"] for row in plan["canonical_rows"]]
            and np.array_equal(shared["ref_latlon"], [row["latlon"] for row in plan["canonical_rows"]])
            and np.array_equal(shared["query_times"], times)
            and np.array_equal(shared["frame_ordinals"], plan["query"]["planned_frame_ordinals"])
            and identity["native_pixels_sha256"] == heading.core._json_sha(plan["per_query_native_pixel_sha256"]),
            "Sela gallery and frozen controls/cues do not share every ID/query/native pixel")
    require(identity["matrices"]["megaloc"] == plan["control"]["raw_cosines_sha256"], "MegaLoc control matrix differs")
    controls = []
    for name, (role, filename) in CONTROL_FILES.items():
        require(filename in originals[role]["artifacts"], "one of sixteen fixed control routes is absent")
        source = roots[role].parent / filename
        controls.append({"name": name, "source": str(source), "sha256": originals[role]["artifacts"][filename],
                         "copy": "controls/" + role + "_" + filename})
    proof, assets = smoke_proof(roots["smoke"], spec["smoke"]["freeze_sha256"]), backend.inspect_assets(spec["assets"])
    require(proof["assets"] == assets, "full experiment assets differ from successful smoke")
    return {"gallery_root": gallery_root, "plan": plan, "shared": shared, "matrices": matrices,
        "cameras": cameras, "angles": angles, "offsets": offsets, "yaw": yaw, "identity": identity,
        "controls": controls, "smoke": proof, "assets": assets, "cue_source_freezes": cue_freezes}


def source_binding(expected):
    require(Path(__file__).resolve() == ROOT / "scripts/replay_selavpr_gallery.py"
            and smoke.ROOT == heading.ROOT == ROOT, "executed source utilities escaped clean pin")
    return smoke.source_binding(expected)


def cue_arrays(bundle):
    return {**bundle["shared"], "reference_heading_deg": bundle["angles"], "crop_offsets_deg": bundle["offsets"],
        "vo_yaw_delta_deg": bundle["yaw"]["vo_yaw_delta_deg"], "vo_quality": bundle["yaw"]["vo_quality"]}


def copy_file(source, target, expected):
    source, target = heading.allowed(source).resolve(strict=True), heading.allowed(target)
    require(sha(source) == expected and not target.exists(), "copy source differs or destination exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha(target) == expected, "copy changed bytes")


def prepare(spec_path, family, expected_commit):
    sources = source_binding(expected_commit)
    gallery.sparse.check_headroom()
    spec_path = heading.allowed(spec_path).resolve(strict=True)
    declared, family = specification(spec_path), heading.allowed(family).resolve()
    roots = [Path(declared[role]["freeze"]).resolve().parent.parent for role in ROLES]
    roots += [Path(declared["assets"]).resolve(), ROOT]
    require(not family.exists() and not any(family.is_relative_to(p) or p.is_relative_to(family) for p in roots),
            "new isolated experiment family required")
    bundle = verified_inputs(declared)
    family.mkdir(parents=True)
    try:
        copy_file(spec_path, family / "inputs/spec.json", sha(spec_path))
        for row in bundle["controls"]:
            copy_file(row["source"], family / "inputs" / row["copy"], row["sha256"])
        arrays = cue_arrays(bundle)
        np.savez(family / "inputs/cue_arrays.npz", **arrays)
        write(family / "inputs/cue_identity.json", {"identity": bundle["identity"], "camera_types": bundle["cameras"],
            "typed_arrays_sha256": {key: heading.core._array_sha(value) for key, value in arrays.items()},
            "motion_diagnostics": {key: value for key, value in bundle["yaw"].items()
                if key not in ("vo_yaw_delta_deg", "vo_quality", "query_native_indices")}})
        snapshot = family / "inputs/source_snapshot"
        for relative, digest in sources.items():
            copy_file(ROOT / relative, snapshot / relative, digest)
        proposal = Path(declared["assets"]) / "paired_full_gallery_plan.json"
        write(family / "plan.json", {"profile": PROFILE, "profile_sha256": PROFILE_SHA256,
            "source_commit": expected_commit, "source_sha256": sources, "declared": declared,
            "input_identity": bundle["identity"], "smoke_proof": bundle["smoke"], "assets": bundle["assets"],
            "cue_source_freezes": bundle["cue_source_freezes"],
            "controls": bundle["controls"], "all_requested_outputs": NEW_OUTPUTS + list(CONTROL_FILES),
            "superseded_six_output_proposal": {"path": str(proposal), "sha256": sha(proposal),
                "amendment": "all_eight_new_Sela_simple_and_heading_routes_plus16_fixed_controls_predeclared_before_new_forwards"},
            "selected_variant": None, "models_have_not_run": True, "labels_opened": False})
        final = verified_inputs(declared)
        require(final["identity"] == bundle["identity"] and final["smoke"] == bundle["smoke"]
                and source_binding(expected_commit) == sources, "inputs/source changed during preparation")
        artifacts = {str(path.relative_to(family)): sha(path) for path in family.rglob("*") if path.is_file()}
        write(family / "inputs_frozen.json", {"status": "prepared_only", "scoring_has_not_run": True,
            "models_have_not_run": True, "artifacts": artifacts, "all_requested_outputs": NEW_OUTPUTS + list(CONTROL_FILES)})
        return {"status": "prepared_only", "input_freeze_sha256": sha(family / "inputs_frozen.json"),
                "profile_sha256": PROFILE_SHA256, "new_outputs": 8, "retained_controls": 16}
    except Exception as error:
        write(family / "preparation_failed.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "models_have_not_run": True, "labels_opened": False})
        raise


def verify_prepared(family, expected_inputs, expected_commit):
    heading.freeze(family / "inputs_frozen.json", expected_inputs)
    plan = load(family / "plan.json")
    require(plan["profile"] == PROFILE and plan["profile_sha256"] == PROFILE_SHA256
            and plan["source_commit"] == expected_commit and source_binding(expected_commit) == plan["source_sha256"]
            and plan["all_requested_outputs"] == NEW_OUTPUTS + list(CONTROL_FILES)
            and plan["selected_variant"] is None, "registered full experiment source/profile differs")
    declared = specification(family / "inputs/spec.json")
    require(declared == plan["declared"], "frozen input spec differs")
    bundle = verified_inputs(declared)
    require(bundle["identity"] == plan["input_identity"] and bundle["smoke"] == plan["smoke_proof"]
            and bundle["assets"] == plan["assets"] and bundle["controls"] == plan["controls"]
            and bundle["cue_source_freezes"] == plan["cue_source_freezes"], "original bound inputs changed")
    identity = load(family / "inputs/cue_identity.json")
    require(identity["identity"] == bundle["identity"] and identity["camera_types"] == bundle["cameras"], "frozen cue identity differs")
    arrays = cue_arrays(bundle)
    for key, value in arrays.items():
        heading.bounded_header(family / "inputs/cue_arrays.npz", key, max(value.size, 1), value.dtype.kind,
                               value.dtype.itemsize, value.shape)
    with np.load(family / "inputs/cue_arrays.npz", allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays) and all(heading.core._array_sha(saved[key]) == identity["typed_arrays_sha256"][key]
                == heading.core._array_sha(value) for key, value in arrays.items()), "frozen cue typed arrays differ")
    proposal = plan["superseded_six_output_proposal"]
    require(sha(proposal["path"]) == proposal["sha256"], "original proposal changed")
    return plan, bundle


def resource_gate(started):
    require(time.monotonic() - started <= PROFILE["runtime_wall_budget_s"], "registered10800s wall budget exceeded")
    return gallery.sparse.check_headroom()


def encode_one(model, pixels, record, started, rows):
    memory = resource_gate(started)
    preprocessing = smoke.verify_preprocessing(model, pixels)
    begin = time.monotonic()
    values = model.encode(pixels)
    require(values.shape == (2048,) and values.dtype == np.float32 and np.isfinite(values).all()
            and abs(float(np.linalg.norm(values)) - 1) <= 1e-4, "invalid normalized Sela descriptor")
    rows.append({**record, "forward_s": time.monotonic() - begin,
        "preprocessing": preprocessing, "descriptor_norm": float(np.linalg.norm(values)), "headroom": memory})
    return values


def encode_references(root, plan, model, output, started, rows):
    require(not output.exists(), "reference descriptors cannot overwrite a partial run")
    canonical = plan["canonical_rows"]
    descriptors = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(canonical), 2048))
    staging = output.parent / "raw_staging"
    staging.mkdir(exist_ok=False)
    seen = np.zeros(len(canonical), bool)
    try:
        for number, source in enumerate(plan["raw_sources"]):
            resource_gate(started)
            archive = (root / heading.relative(source["copy"])).resolve(strict=True)
            require(archive.is_relative_to(root) and sha(archive) == source["sha256"], "frozen raw pool differs")
            chosen = [row for row in canonical if row["source"] == source["source"]]
            gallery.raw_header(archive, source["raw_shape"])
            with np.load(archive, allow_pickle=False) as saved:
                ids, xy = saved["ids"], saved["xy"]
            require(ids.shape == (source["raw_shape"][0],) and xy.shape == (len(ids), 2)
                    and len(set(ids.astype(str).tolist())) == len(ids), "raw pool ID/position inventory invalid")
            raw_path, raw = staging / f"pool_{number:02d}.npy", None
            try:
                extracted = gallery.image.extract_raw(archive, raw_path)
                require(extracted["shape"] == source["raw_shape"], "streamed raw shape differs")
                raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
                for row in chosen:
                    index, column = row["raw_row"], row["column"]
                    require(type(index) is int and 0 <= index < len(ids) and 0 <= column < len(canonical)
                            and not seen[column] and "mapillary:" + str(ids[index]) == row["id"]
                            and xy[index].tolist() == row["latlon"], "canonical raw row identity differs")
                    pixels = raw[index]
                    require(hashlib.sha256(memoryview(pixels).cast("B")).hexdigest() == row["pixels_sha256"],
                            "canonical plain pixel hash differs")
                    descriptors[column] = encode_one(model, pixels, {"kind": "reference", "column": column,
                        "id": row["id"], "pool": number, "raw_row": index, "pixels_sha256": row["pixels_sha256"]}, started, rows)
                    seen[column] = True
                descriptors.flush()
                print(f"Sela gallery pool {number+1}/{len(plan['raw_sources'])}: {int(seen.sum())}/{len(canonical)}", flush=True)
            finally:
                if raw is not None:
                    gallery.close_map(raw)
                raw_path.unlink(missing_ok=True)
        require(seen.all(), "one or more canonical references were omitted")
        descriptors.flush()
    finally:
        gallery.close_map(descriptors)
        require(staging.resolve().parent == output.parent.resolve(), "raw cleanup escaped owned output")
        shutil.rmtree(staging)


def encode_queries(root, plan, times, model, output, started, rows):
    frames, previous = None, tempfile.tempdir
    temporary = output.parent / "frame_tmp"
    temporary.mkdir(exist_ok=False)
    tempfile.tempdir = str(temporary)
    descriptors = np.empty((240, 2048), np.float32)
    try:
        frames = gallery.native_queries(root, plan, times)
        from monocular_osm.kartaview_vpr import _square_query_crop
        require(len(frames.frames) == 80, "all80 native queries required")
        for index, frame in enumerate(frames.frames):
            for view in range(3):
                descriptors[index * 3 + view] = encode_one(model, _square_query_crop(frame, view),
                    {"kind": "query", "position": index, "view": view,
                     "frame_ordinal": plan["query"]["planned_frame_ordinals"][index],
                     "timestamp_s": float(times[index]), "native_pixels_sha256": plan["per_query_native_pixel_sha256"][index]}, started, rows)
            print(f"Sela native query {index+1}/80: all3 views", flush=True)
        np.save(output, descriptors, allow_pickle=False)
    finally:
        if frames is not None:
            frames.close()
        tempfile.tempdir = previous
        require(temporary.resolve().parent == output.parent.resolve(), "frame cleanup escaped owned output")
        shutil.rmtree(temporary)
    return descriptors


def cosine_views(queries, references):
    require(queries.shape == (240, 2048) and references.ndim == 2 and references.shape[1] == 2048
            and 0 < len(references) <= 10000 and queries.dtype == references.dtype == np.float32,
            "complete three-view float32 descriptor grid required")
    for array in (queries, references):
        for start in range(0, len(array), 256):
            values = array[start:start+256]
            require(np.isfinite(values).all() and np.max(np.abs(np.linalg.norm(values, axis=1)-1)) <= 1e-4,
                    "cosines require finite normalized descriptors")
    views = np.empty((80, 3, len(references)), np.float32)
    for start in range(0, len(references), 256):
        stop = min(start+256, len(references))
        views[:, :, start:stop] = (queries @ references[start:stop].T).reshape(80, 3, stop-start)
    require(np.isfinite(views).all() and np.max(np.abs(views)) <= 1.00001, "invalid full cosine cube")
    return views


def write_new_routes(output, views, bundle):
    shared = bundle["shared"]
    result = full.decode_full_tracks(views.max(axis=1), shared["ref_ids"], shared["ref_latlon"], shared["query_times"])
    require(result["profile_sha256"] == full.PROFILE_SHA256 and set(result["outputs"]) == {"raw", "reference_mean"},
            "fixed full-sequence decoder omitted a variant")
    write(output / "predictions_simple.json", result)
    records = []
    for variant in ("raw", "reference_mean"):
        filename = "prediction_selavpr_simple_" + variant + ".json"
        write(output / filename, result["outputs"][variant])
        records.append({"name": "selavpr/simple/" + variant, "file": filename,
                        "sha256": sha(output / filename), "profile_sha256": full.PROFILE_SHA256, "new": True})
    del result
    print("Sela both fixed simple routes complete; all6 heading routes starting", flush=True)
    result = heading_core.decode_heading_tracks(views, shared["ref_ids"], shared["ref_latlon"], shared["query_times"],
        bundle["cameras"], bundle["angles"], bundle["offsets"],
        vo_yaw_delta_deg=bundle["yaw"]["vo_yaw_delta_deg"], vo_quality=bundle["yaw"]["vo_quality"])
    expected = [variant + "/" + mode for variant in heading_core.PROFILE["variants"] for mode in heading_core.PROFILE["modes"]]
    require(result["profile_sha256"] == heading_core.PROFILE_SHA256 and set(result["outputs"]) == set(expected),
            "fixed heading decoder omitted a variant/mode")
    write(output / "predictions_heading.json", result)
    for key in expected:
        filename = "prediction_selavpr_heading_" + key.replace("/", "_") + ".json"
        write(output / filename, result["outputs"][key])
        records.append({"name": "selavpr/heading/" + key, "file": filename,
                        "sha256": sha(output / filename), "profile_sha256": heading_core.PROFILE_SHA256, "new": True})
    require([row["name"] for row in records] == NEW_OUTPUTS, "all eight predeclared new routes required")
    return records


def verify_smoke_replay(spec, bundle, queries, references, records):
    root = Path(spec["smoke"]["freeze"]).resolve().parent
    saved = np.load(root / "descriptors.npy", allow_pickle=False)
    columns = bundle["smoke"]["reference_columns"]
    replay = np.concatenate([queries[:12], references[columns]], axis=0)
    require(saved.shape == replay.shape == (44, 2048), "smoke replay descriptor rows differ")
    delta = float(np.max(np.abs(replay - saved)))
    require(delta <= PROFILE["smoke_descriptor_replay_atol"], "new full encoding differs from the fixed runtime smoke")
    old = load(root / "runtime.json")["per_image"]
    queries_by_key = {(row["position"], row["view"]): row for row in records if row["kind"] == "query"}
    refs_by_column = {row["column"]: row for row in records if row["kind"] == "reference"}
    selected = [queries_by_key[(position, view)] for position in range(4) for view in range(3)]
    selected += [refs_by_column[column] for column in columns]
    require(all(new["preprocessing"]["input_tensor_sha256"] == prior["preprocessing"]["input_tensor_sha256"]
                for new, prior in zip(selected, old)), "full encoding input tensors differ from successful smoke")
    return {"max_abs_delta": delta, "bitwise_equal": bool(np.array_equal(replay, saved)),
            "all44_input_tensor_hashes_equal": True, "reference_columns": columns}


def infer(family, expected_commit, input_freeze_sha256):
    require(heading.digest(input_freeze_sha256), "explicit prepared input-freeze SHA required")
    family = heading.allowed(family).resolve(strict=True)
    plan, bundle = verify_prepared(family, input_freeze_sha256, expected_commit)
    target = family / "inference"
    target.mkdir(exist_ok=False)
    started, completed = time.monotonic(), []
    try:
        before = resource_gate(started)
        for row in bundle["controls"]:
            copy_file(family / "inputs" / row["copy"], target / row["copy"], row["sha256"])
        constructor_start = time.monotonic()
        model = backend.ReleasedSelaBase(bundle["assets"]["root"])
        constructor_s = time.monotonic() - constructor_start
        proof = bundle["smoke"]
        require(model.state == proof["state"] and model.runtime["torch"] == proof["torch"]
                and model.runtime["torchvision"] == proof["torchvision"], "model state/runtime differ from accepted smoke")
        after_constructor = resource_gate(started)
        rows = []
        encode_references(bundle["gallery_root"], bundle["plan"], model, target / "reference_descriptors.npy", started, rows)
        queries = encode_queries(bundle["gallery_root"], bundle["plan"], bundle["shared"]["query_times"],
                                 model, target / "query_descriptors.npy", started, rows)
        require(model.runtime["forward_calls"] == PROFILE["fresh_forward_count"] and len(rows) == PROFILE["fresh_forward_count"],
                "complete9975 unique forwards required")
        model.verify_unchanged()
        model_state, model_runtime = dict(model.state), dict(model.runtime)
        references = np.load(target / "reference_descriptors.npy", mmap_mode="r", allow_pickle=False)
        try:
            smoke_replay = verify_smoke_replay(plan["declared"], bundle, queries, references, rows)
            views = cosine_views(queries, references)
        finally:
            gallery.close_map(references)
        del model
        gc.collect()
        resource_gate(started)
        np.save(target / "view_cosines.npy", views, allow_pickle=False)
        shared = bundle["shared"]
        np.savez(target / "full_evidence.npz", selavpr_raw_cosines=views.max(axis=1),
            megaloc_raw_cosines=bundle["matrices"]["megaloc"], image_raw_cosines=bundle["matrices"]["image"], **shared)
        new_records = write_new_routes(target, views, bundle)
        completed = [row["name"] for row in new_records]
        _finalize(family, target, plan, bundle, expected_commit, input_freeze_sha256,
                  started, before, after_constructor, constructor_s, model_state, model_runtime,
                  rows, views, smoke_replay, new_records)
        return {"status": "complete", "new_outputs": 8, "comparison_outputs": 24,
                "freeze_sha256": sha(target / "frozen_predictions.json")}
    except Exception as error:
        write(target / "failure.json", {"status": "failed", "completed_new_outputs": completed,
            "error": f"{type(error).__name__}: {error}", "labels_opened": False,
            "scoring_has_not_run": True, "elapsed_s": time.monotonic()-started})
        raise


def _finalize(family, target, plan, bundle, expected_commit, expected_inputs,
              started, before, after_constructor, constructor_s, model_state, model_runtime,
              rows, views, smoke_replay, new_records):
    controls = [{"name": row["name"], "file": row["copy"], "sha256": row["sha256"],
                 "original_source": row["source"], "new": False} for row in bundle["controls"]]
    index = new_records + controls
    require([row["name"] for row in index] == NEW_OUTPUTS + list(CONTROL_FILES)
            and all(sha(target / row["file"]) == row["sha256"] for row in index), "complete24-route comparison required")
    write(target / "comparison_index.json", {"profile_sha256": PROFILE_SHA256, "outputs": index,
        "new_outputs": 8, "retained_controls": 16, "selected_variant": None, "labels_opened": False})
    write(target / "runtime.json", {"status": "complete", "source_commit": expected_commit,
        "profile_sha256": PROFILE_SHA256, "input_freeze_sha256": expected_inputs,
        "input_identity": bundle["identity"], "state": model_state, "runtime": model_runtime,
        "constructor_s": constructor_s, "elapsed_s": time.monotonic()-started, "per_image": rows,
        "smoke_replay": smoke_replay, "memory_before": before, "memory_after_constructor": after_constructor,
        "memory_end": resource_gate(started), "full_view_cosines_sha256": heading.core._array_sha(views),
        "all8_new_and16_old_controls": True, "labels_opened": False, "scoring_called": False,
        "selected_variant": None, "old_controls_redecoded": False})
    # Expected-SHA validation is repeated after all input maps/temp frames close.
    verify_prepared(family, expected_inputs, expected_commit)
    require(all(sha(target / row["file"]) == row["sha256"] for row in index), "prediction bytes changed before freeze")
    write(target / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
        "new_outputs_frozen_before_new_scoring": True, "historical_controls_preserved": True,
        "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256,
        "input_freeze_sha256": expected_inputs, "all_requested_outputs": NEW_OUTPUTS + list(CONTROL_FILES),
        "new_outputs": 8, "retained_controls": 16, "selected_variant": None,
        "artifacts": {str(path.relative_to(target)): sha(path) for path in target.rglob("*") if path.is_file()}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--spec", type=Path, required=True)
    run = commands.add_parser("infer")
    run.add_argument("--input-freeze-sha256", required=True)
    for command in (setup, run):
        command.add_argument("--family", type=Path, required=True)
        command.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        args["spec_path"] = args.pop("spec")
    with heading.offline():
        result = prepare(**args) if command == "prepare" else infer(**args)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
