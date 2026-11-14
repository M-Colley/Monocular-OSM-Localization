"""Fixed 44-image SelaVPR++ runtime probe; no route or scoring interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scripts import selavpr_backend as backend
from scripts import pilot_image_gallery as gallery
from scripts.panorama_gallery_inputs import stage_candidate_pixels

require, sha, load, write = backend.require, gallery.sha, gallery.load, gallery.write
PROFILE = {"version": "selavpr-runtime-only-44-v1", "backend_profile_sha256": backend.PROFILE_SHA256,
    "query_positions": [0, 1, 2, 3], "query_views": [0, 1, 2], "reference_count": 32,
    "reference_selection": "lowest_SHA256_UTF8_full_ID_then_ID_take32_then_global_column_order",
    "queries": 12, "references": 32, "forward_count": 44,
    "input_color": "native_Bgr_and_frozen_Bgr_then_backend_RGB_once",
    "CPU_threads": 2, "interop_threads": 1, "batch_size": 1,
    "route_decoding": False, "accuracy_or_label_reads": False, "network": False,
    "all_descriptors_retained": True, "resource_floor_GiB": 8,
    "source_and_full_original_input_rechecks_after_cleanup": True}
PROFILE_SHA256 = hashlib.sha256(json.dumps(PROFILE, sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_binding(expected):
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT), *a], text=True).strip()
    require(git("rev-parse", "HEAD") == expected
            and not git("status", "--porcelain", "--untracked-files=all"), "exact clean smoke source required")
    require(Path(__file__).resolve() == ROOT / "scripts/smoke_selavpr.py", "foreign executed smoke driver")
    for name, module in list(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts.") or name == "monocular_osm" or name.startswith("monocular_osm."):
            filename = getattr(module, "__file__", None)
            paths = [filename] if filename else list(getattr(module, "__path__", []))
            require(paths and all(Path(p).resolve(strict=True).is_relative_to(ROOT) for p in paths),
                    "foreign loaded project module")
    paths = [*sorted((ROOT / "scripts").rglob("*.py")), *sorted((ROOT / "monocular_osm").rglob("*.py"))]
    return {str(path.relative_to(ROOT)): backend.sha(path) for path in paths}


def selected_columns(rows, count=32):
    require(type(count) is int and 0 < count <= len(rows) <= 10000, "bounded reference sample required")
    require([row["column"] for row in rows] == list(range(len(rows)))
            and len({row["id"] for row in rows}) == len(rows)
            and all(isinstance(row["id"], str) and row["id"] for row in rows), "canonical ID order is incomplete")
    chosen = sorted(rows, key=lambda row: (hashlib.sha256(row["id"].encode()).hexdigest(), row["id"]))[:count]
    return sorted(row["column"] for row in chosen)


def verify_gallery(family, expected):
    family = Path(family).resolve(strict=True)
    require(sha(family / "inputs_frozen.json") == expected, "original gallery freeze identity differs")
    gallery.verify_freeze(family / "inputs_frozen.json")
    plan = load(family / "plan.json")
    require(len(plan["canonical_rows"]) == 9735 and len(plan["raw_sources"]) == 32,
            "complete registered full gallery required")
    with np.load(family / "inputs/full_evidence.npz", allow_pickle=False) as archive:
        ids, xy, times, ordinals, indices = [archive[name] for name in (
            "ref_ids", "ref_latlon", "query_times", "frame_ordinals", "query_indices")]
    require(ids.dtype.kind == "U" and ids.shape == (9735,) and xy.shape == (9735, 2)
            and times.shape == ordinals.shape == indices.shape == (80,)
            and np.isfinite(xy).all() and np.isfinite(times).all() and np.all(np.diff(times) > 0)
            and ids.tolist() == [row["id"] for row in plan["canonical_rows"]]
            and np.array_equal(xy, [row["latlon"] for row in plan["canonical_rows"]])
            and np.array_equal(ordinals, plan["query"]["planned_frame_ordinals"])
            and np.array_equal(indices, np.arange(80)), "full input ID/geotag/query grid differs")
    return plan, times


def _sample_members(family, plan, times, output, columns):
    """Stage score-independent pixels with existing audited row/native proofs."""
    pixels = stage_candidate_pixels(family, plan, np.asarray(columns, np.int64), output / "references.npy")
    frames, old_temp = None, tempfile.tempdir
    temporary = output / "frame_tmp"
    temporary.mkdir()
    tempfile.tempdir = str(temporary)
    records = []
    try:
        frames = gallery.native_queries(family, plan, times)
        from monocular_osm.kartaview_vpr import _square_query_crop
        for position in PROFILE["query_positions"]:
            native = np.asarray(frames.frames[position])
            for view in PROFILE["query_views"]:
                crop = np.ascontiguousarray(_square_query_crop(native, view))
                name = f"query_{position:02d}_view_{view}.npy"
                np.save(output / name, crop, allow_pickle=False)
                records.append({"file": name, "position": position, "view": view,
                    "frame_ordinal": plan["query"]["planned_frame_ordinals"][position],
                    "timestamp_s": float(times[position]), "shape": list(crop.shape),
                    "native_sha256": plan["per_query_native_pixel_sha256"][position],
                    "crop_plain_pixel_sha256": hashlib.sha256(memoryview(crop).cast("B")).hexdigest(),
                    "npy_sha256": sha(output / name)})
        return {"references": pixels, "queries": records,
                "all80_native_queries_verified": True, "storage_color": "BGR"}
    finally:
        if frames is not None:
            frames.close()
        tempfile.tempdir = old_temp
        require(temporary.resolve().parent == output.resolve(), "temporary cleanup escaped smoke family")
        shutil.rmtree(temporary)


def prepare(gallery_root, expected_gallery, assets_root, output, expected_source):
    source = source_binding(expected_source)
    gallery_root, assets_root, output = [Path(p).resolve() for p in (gallery_root, assets_root, output)]
    require(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output)
            for p in (gallery_root, assets_root, ROOT)), "new isolated smoke family required")
    gallery.sparse.check_headroom()
    plan, times = verify_gallery(gallery_root, expected_gallery)
    assets = backend.inspect_assets(assets_root)
    columns = selected_columns(plan["canonical_rows"])
    output.mkdir(parents=True)
    registration = {"profile": PROFILE, "profile_sha256": PROFILE_SHA256,
        "source_commit": expected_source, "source_sha256": source,
        "gallery": str(gallery_root), "gallery_freeze_sha256": expected_gallery,
        "assets": assets, "reference_columns": columns,
        "reference_ids": [plan["canonical_rows"][c]["id"] for c in columns],
        "query_positions": PROFILE["query_positions"], "models_have_not_run": True,
        "labels_opened": False, "route_decoding": False}
    write(output / "registration.json", registration)
    try:
        samples = _sample_members(gallery_root, plan, times, output, columns)
        write(output / "pixel_proof.json", samples)
        snapshot = output / "source_snapshot"
        for relative, expected in source.items():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            require(backend.sha(target) == expected, "source snapshot copy differs")
        verify_gallery(gallery_root, expected_gallery)
        require(backend.inspect_assets(assets_root) == assets
                and source_binding(expected_source) == source, "source/assets changed during pixel preparation")
        artifacts = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}
        write(output / "inputs_frozen.json", {"status": "prepared_only", "models_have_not_run": True,
            "labels_opened": False, "scoring_has_not_run": True, "artifacts": artifacts})
        return {"status": "prepared_only", "input_freeze_sha256": sha(output / "inputs_frozen.json"),
                "queries": 12, "references": 32, "profile_sha256": PROFILE_SHA256}
    except Exception as error:
        write(output / "prepare_failure.json", {"status": "failed", "models_have_not_run": True,
            "error": f"{type(error).__name__}: {error}", "labels_opened": False})
        raise


def verify_samples(output, registration, plan, times):
    proof = load(output / "pixel_proof.json")
    columns = selected_columns(plan["canonical_rows"])
    require(columns == registration["reference_columns"]
            and [plan["canonical_rows"][c]["id"] for c in columns] == registration["reference_ids"],
            "reference sample is not the fixed ID-only rule")
    _validate_query_proof(output, plan, proof, times)
    refs = np.load(output / "references.npy", mmap_mode="r", allow_pickle=False)
    try:
        require(refs.shape == (32, 512, 512, 3) and refs.dtype == np.uint8 and refs.flags.c_contiguous,
                "fixed reference pixel tensor differs")
        require(proof["references"]["columns"] == columns and proof["all80_native_queries_verified"] is True,
                "complete original pixel proof required")
        for index, column in enumerate(columns):
            require(hashlib.sha256(memoryview(refs[index]).cast("B")).hexdigest()
                    == plan["canonical_rows"][column]["pixels_sha256"], "reference plain pixels differ from original")
    except Exception:
        gallery.close_map(refs)
        raise
    return refs, proof


def _validate_query_proof(output, plan, proof, times):
    expected = [(position, view) for position in PROFILE["query_positions"] for view in PROFILE["query_views"]]
    require([(r["position"], r["view"]) for r in proof["queries"]] == expected, "all12 fixed query views required")
    for row in proof["queries"]:
        position = row["position"]
        name = gallery.sparse.relative(row["file"])
        require(name == Path(f"query_{position:02d}_view_{row['view']}.npy")
                and row["frame_ordinal"] == plan["query"]["planned_frame_ordinals"][position]
                and row["timestamp_s"] == float(times[position])
                and row["native_sha256"] == plan["per_query_native_pixel_sha256"][position]
                and sha(output / name) == row["npy_sha256"], "query source/pixel proof differs")
        pixels = np.load(output / name, allow_pickle=False)
        edge = min(plan["query"]["native_shape_hw"])
        require(pixels.dtype == np.uint8 and list(pixels.shape) == row["shape"] == [edge, edge, 3]
                and pixels.flags.c_contiguous
                and hashlib.sha256(memoryview(pixels).cast("B")).hexdigest() == row["crop_plain_pixel_sha256"],
                "query crop plain pixels differ")


def verify_preprocessing(model, bgr):
    """Bitwise runtime check against the independently expressed canonical path."""
    actual = model.input_tensor(bgr)
    # This existing helper is PIL RGB -> ToTensor -> Normalize -> tensor resize.
    # Its fixed ImageNet values/322/bilinear/antialias match inspected Sela source.
    expected = gallery.image.canonical_tensor(backend.rgb_pixels(bgr)).unsqueeze(0)
    require(model.torch.equal(actual, expected), "actual Sela preprocessing differs from canonical official order/layout")
    values = actual.detach().cpu().numpy()
    return {"canonical_bitwise_equal": True, "shape": list(values.shape), "dtype": values.dtype.str,
            "input_tensor_sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest()}


def infer(output, expected_inputs, expected_source):
    output = Path(output).resolve(strict=True)
    require(sha(output / "inputs_frozen.json") == expected_inputs, "prepared smoke freeze differs")
    gallery.verify_freeze(output / "inputs_frozen.json")
    registration = load(output / "registration.json")
    require(registration["profile"] == PROFILE and registration["profile_sha256"] == PROFILE_SHA256
            and registration["source_commit"] == expected_source
            and source_binding(expected_source) == registration["source_sha256"], "registered smoke source/profile differs")
    plan, times = verify_gallery(registration["gallery"], registration["gallery_freeze_sha256"])
    require(backend.inspect_assets(registration["assets"]["root"]) == registration["assets"], "registered assets differ")
    target = output / "inference"
    target.mkdir(exist_ok=False)
    started, refs = time.monotonic(), None
    try:
        refs, proof = verify_samples(output, registration, plan, times)
        memory = [gallery.sparse.check_headroom()]
        constructor_start = time.monotonic()
        model = backend.ReleasedSelaBase(registration["assets"]["root"])
        constructor_s = time.monotonic() - constructor_start
        memory.append(gallery.sparse.check_headroom())
        import psutil
        process = psutil.Process()
        samples, descriptors = [], np.empty((44, 2048), np.float32)
        records = [dict(row, kind="query") for row in proof["queries"]]
        records += [{"kind": "reference", "sample_index": i, "id": registration["reference_ids"][i],
                     "column": column} for i, column in enumerate(registration["reference_columns"])]
        for index, row in enumerate(records):
            memory.append(gallery.sparse.check_headroom())
            pixels = (np.load(output / row["file"], allow_pickle=False) if row["kind"] == "query"
                      else refs[row["sample_index"]])
            preprocessing = verify_preprocessing(model, pixels)
            begin = time.monotonic()
            descriptors[index] = model.encode(pixels)
            usage = process.memory_info()
            samples.append({**row, "seconds": time.monotonic() - begin,
                "descriptor_norm": float(np.linalg.norm(descriptors[index])),
                "preprocessing": preprocessing,
                "rss_bytes": usage.rss, "private_bytes": getattr(usage, "private", usage.vms)})
            print(f"Sela runtime image {index+1}/44 ({row['kind']})", flush=True)
        require(model.runtime["forward_calls"] == 44, "unexpected model forward count")
        model.verify_unchanged()
        np.save(target / "descriptors.npy", descriptors, allow_pickle=False)
        write(target / "runtime.json", {"status": "complete", "constructor_s": constructor_s,
            "elapsed_s": time.monotonic() - started, "state": model.state, "runtime": model.runtime,
            "per_image": samples, "minimum_headroom": gallery.minimum_memory(memory),
            "sampled_self_peak_rss_bytes": max(row["rss_bytes"] for row in samples),
            "sampled_self_peak_private_bytes": max(row["private_bytes"] for row in samples),
            "external_process_tree_watchdog_required": True, "no_accuracy_result": True,
            "labels_opened": False, "route_decoding": False, "network": False})
    except Exception as error:
        write(target / "failure.json", {"status": "failed", "labels_opened": False,
            "route_decoding": False, "error": f"{type(error).__name__}: {error}",
            "elapsed_s": time.monotonic() - started})
        raise
    finally:
        if refs is not None:
            gallery.close_map(refs)
    # Final checks occur after releasing input maps and preserve the entry SHA.
    require(sha(output / "inputs_frozen.json") == expected_inputs, "prepared smoke freeze changed during execution")
    gallery.verify_freeze(output / "inputs_frozen.json")
    verify_gallery(registration["gallery"], registration["gallery_freeze_sha256"])
    require(source_binding(expected_source) == registration["source_sha256"]
            and backend.inspect_assets(registration["assets"]["root"]) == registration["assets"],
            "source/assets changed after cleanup")
    artifacts = {str(path.relative_to(target)): sha(path) for path in target.rglob("*") if path.is_file()}
    write(target / "runtime_frozen.json", {"status": "complete", "input_freeze_sha256": expected_inputs,
        "source_commit": expected_source, "profile_sha256": PROFILE_SHA256, "artifacts": artifacts,
        "forward_count": 44, "descriptors_shape": [44, 2048], "labels_opened": False,
        "route_decoding": False, "network": False, "no_accuracy_result": True})
    return {"status": "complete", "runtime_freeze_sha256": sha(target / "runtime_frozen.json"), "forwards": 44}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--gallery", type=Path, required=True)
    prepare_parser.add_argument("--gallery-freeze-sha256", required=True)
    prepare_parser.add_argument("--assets", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--input-freeze-sha256", required=True)
    infer_parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    with gallery.sparse.offline():
        if args.command == "prepare":
            result = prepare(args.gallery, args.gallery_freeze_sha256, args.assets, args.output, args.source_commit)
        else:
            result = infer(args.output, args.input_freeze_sha256, args.source_commit)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
