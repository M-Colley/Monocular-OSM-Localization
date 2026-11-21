"""Freeze both full-support spatial routes from complete saved VPR evidence.

No video decoding, model construction, metadata fetch or label/scoring access.
This is posthoc DEV research: earlier diagnostic scores may have been seen.
Exact source/config/input identities and every posterior freeze before any
separate evaluation. Existing output families are never resumed or modified.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from monocular_osm.spatial_sequence import (SpatialSequenceConfig, VARIANTS, array_sha, decode_spatial_tracks, json_sha)
from scripts.prepare_road_target_motion import read, sha, write, verify_family, windows_memory

SOURCES = ("scripts/replay_spatial_sequence.py", "scripts/prepare_road_target_motion.py",
    "monocular_osm/spatial_sequence.py", "monocular_osm/poi_alignment.py", "monocular_osm/geo.py",
    "monocular_osm/__init__.py")
NPZ_SPECS = {"raw_cosines": (8000000, "f", 8), "query_times": (80, "f", 8),
    "query_indices": (80, "iu", 8), "frame_ordinals": (80, "iu", 8),
    "ref_ids": (100000, "US", 512), "ref_latlon": (200000, "f", 8)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes(commit, expected=None):
    require(Path(__file__).resolve() == (ROOT/"scripts/replay_spatial_sequence.py").resolve(), "executed runner escaped declared source")
    require(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == commit
        and not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip(),
        "exact clean source pin required")
    values = {relative: sha(ROOT/relative) for relative in SOURCES}
    require(expected is None or values == expected, "execution source changed before freeze")
    for name, module in list(sys.modules.items()):
        if name == "monocular_osm" or name.startswith("monocular_osm.") or name == "scripts.prepare_road_target_motion":
            path = getattr(module, "__file__", None)
            require(path is not None and Path(path).resolve().is_relative_to(ROOT), "loaded module escaped source pin")
    return values


def bounded_evidence(path):
    path = Path(path)
    require(path.stat().st_size <= 128*1024**2, "evidence NPZ exceeds fixed disk budget")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name, (maximum, kinds, itemsize) in NPZ_SPECS.items():
            require(names.count(name+".npy") == 1, "missing/duplicate evidence member: "+name)
            with archive.open(name+".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported evidence NPY version")
            count = math.prod(shape)
            require(count <= maximum and dtype.kind in kinds and dtype.itemsize <= itemsize,
                    "evidence member exceeds fixed typed budget: "+name)
            require(archive.getinfo(name+".npy").file_size <= maximum*itemsize+65536,
                    "evidence member exceeds uncompressed allocation budget")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in NPZ_SPECS}


def require_frozen_members(manifest, root, paths):
    recorded = {(Path(root)/name).resolve(): digest for name, digest in manifest["artifacts"].items()}
    for path in paths:
        path = Path(path).resolve(strict=True)
        require(path in recorded and sha(path) == recorded[path], "required input was not covered by its freeze: "+path.name)


def load_frozen_evidence(family, expected):
    family = Path(family).resolve(strict=True)
    frozen = verify_family(family/"frozen_predictions.json", expected)
    require(frozen["status"] == "complete" and frozen["scoring_has_not_run"] is True,
            "input predictions must have frozen completely before their separate score")
    require_frozen_members(frozen, family, [family/"full_evidence.npz", family/"runtime_and_lineage.json"])
    runtime = read(family/"runtime_and_lineage.json")
    require(runtime["labels_opened"] is False and runtime["scoring_called"] is False,
            "input generation must have been label-free")
    inputs = family.parent/"inputs_frozen.json"
    input_manifest = verify_family(inputs, runtime["input_freeze_sha256"])
    require_frozen_members(input_manifest, family.parent, [family.parent/"inputs/replay_plan.json"])
    plan = read(family.parent/"inputs/replay_plan.json")
    full = bounded_evidence(family/"full_evidence.npz")
    n, r = full["raw_cosines"].shape if full["raw_cosines"].ndim == 2 else (0, 0)
    require(2 <= n <= 80 and 1 <= r <= 100000 and full["raw_cosines"].size <= 8000000
        and full["raw_cosines"].dtype.itemsize in (4, 8) and np.isfinite(full["raw_cosines"]).all()
        and np.all(np.abs(full["raw_cosines"]) <= 1.00001), "complete raw cosine matrix required")
    ids, refs, times, ordinals = full["ref_ids"], full["ref_latlon"], full["query_times"], full["frame_ordinals"]
    require(ids.shape == (r,) and len(set(ids.astype(str))) == r and all(ids.astype(str))
        and refs.shape == (r, 2) and np.isfinite(refs).all() and np.all(np.abs(refs[:, 0]) <= 90)
        and np.all(np.abs(refs[:, 1]) <= 180), "unique complete public reference gallery required")
    require(ids.astype(str).tolist() == plan["union"]["ref_ids"] and r == plan["union"]["count"]
        and array_sha(refs) == plan["union"]["ref_latlon_sha256"], "full union differs from frozen input plan")
    require(times.shape == (n,) and np.isfinite(times).all() and times[0] >= 0 and np.all(np.diff(times) > 0)
        and ordinals.shape == (n,) and np.all(np.diff(ordinals.astype(float)) > 0)
        and np.array_equal(full["query_indices"], np.arange(n))
        and np.array_equal(ordinals, plan["query"]["planned_frame_ordinals"])
        and array_sha(times) == runtime["actual_query_times_sha256"], "actual query row/ordinal/PTS identity differs")
    pixels = runtime["per_query_native_pixel_sha256"]
    video_sha = plan["query"]["video"]["sha256"]
    require(len(pixels) == n and all(isinstance(value, str) and len(value) == 64 for value in pixels)
        and isinstance(video_sha, str) and len(video_sha) == 64, "frozen native pixel/video identities required")
    record = {"evidence_family": str(family), "evidence_freeze_sha256": expected,
        "inputs_freeze_sha256": runtime["input_freeze_sha256"], "matrix_file_sha256": sha(family/"full_evidence.npz"),
        "runtime_file_sha256": sha(family/"runtime_and_lineage.json"), "plan_file_sha256": sha(family.parent/"inputs/replay_plan.json"),
        "video_sha256": video_sha, "registered_window": plan["query"]["registered_window"],
        "native_pixel_hashes_sha256": json_sha(pixels), "query_count": n, "reference_count": r,
        "typed_arrays": {name: array_sha(value) for name, value in full.items()}}
    return full, record


def new_output(path, evidence):
    output, family = Path(path).resolve(), Path(evidence).resolve()
    require(not output.exists() and not output.is_relative_to(ROOT) and not output.is_relative_to(family.parent)
        and not ROOT.is_relative_to(output) and not family.is_relative_to(output),
        "output must be new and isolated from source/input families; no resume")
    return output


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_predictions(output, result, full):
    """Persist both complete posterior cubes and all diagnostics before freeze."""
    require(set(result["variants"]) == set(VARIANTS) and result["selected_variant"] is None,
            "both registered variants and no oracle selection required")
    posterior_path = output/"spatial_posteriors.npz"
    with posterior_path.open("xb") as stream:
        np.savez(stream, **{variant+"_posterior": result["variants"][variant]["posterior"] for variant in VARIANTS},
            query_times=full["query_times"], frame_ordinals=full["frame_ordinals"],
            ref_ids=full["ref_ids"], ref_latlon=full["ref_latlon"],
            origin_xy=np.array(result["origin_xy"]), cell_m=np.array(result["cell_m"]))
    posterior_hash = sha(posterior_path)
    diagnostics = {key: value for key, value in result.items() if key != "variants"}
    diagnostics["variants"] = {}
    for variant in VARIANTS:
        values = result["variants"][variant]
        diagnostics["variants"][variant] = {key: value for key, value in values.items() if key != "posterior"}
        write(output/f"prediction_{variant}.json", jsonable({"status": "complete", "variant": variant,
            "kind": "spatial_posterior_experimental", "primary_route": values["primary_route"],
            "route_latlon": values["route_latlon"], "route_timestamps": values["route_timestamps"],
            "confidence": None, "confidence_calibrated": False, "selected_variant": None,
            "profile_sha256": result["profile_sha256"], "posterior_file_sha256": posterior_hash,
            "posterior_sha256": values["posterior_sha256"]}))
    write(output/"all_spatial_diagnostics.json", jsonable(diagnostics))


def replay(args):
    sources = source_hashes(args.expected_commit)
    memory = windows_memory()
    require(all(np.isfinite(memory[key]) and memory[key] >= 8. for key in ("available_physical_gib", "available_commit_gib")),
            "spatial replay requires physical and actual free commit >=8GiB")
    full, identity = load_frozen_evidence(args.evidence_family, args.evidence_freeze_sha256)
    output = new_output(args.output, args.evidence_family)
    output.mkdir(parents=True, exist_ok=False)
    for relative, digest in sources.items():
        dest = output/"source_snapshot"/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, dest)
        require(sha(dest) == digest, "source changed during snapshot")
    config = SpatialSequenceConfig()
    registration = {"source_commit": args.expected_commit, "sources": sources, "input_identity": identity,
        "profile": config.record(), "profile_sha256": json_sha(config.record()), "variants": list(VARIANTS),
        "selected_variant": None, "development_context": "posthoc_DEV; earlier offline diagnostics disclosed",
        "labels_opened_by_this_run": False, "ground_truth_used": False, "models_or_network_or_vo": False,
        "preflight_memory": memory}
    write(output/"spatial_registration.json", registration)
    started = time.monotonic()
    try:
        result = decode_spatial_tracks(full["raw_cosines"], full["ref_ids"], full["ref_latlon"], full["query_times"],
            config=config, provenance={"registration_sha256": sha(output/"spatial_registration.json"), **identity})
        save_predictions(output, result, full)
        _, final_identity = load_frozen_evidence(args.evidence_family, args.evidence_freeze_sha256)
        require(final_identity == identity, "frozen input identity changed during inference")
        source_hashes(args.expected_commit, sources)
        import psutil
        process = psutil.Process().memory_info()
        write(output/"runtime.json", {"status": "complete", "elapsed_s": time.monotonic()-started,
            "torch_imported": "torch" in sys.modules, "labels_opened": False, "scoring_called": False,
            "peak_rss_bytes": getattr(process, "peak_wset", process.rss),
            "peak_private_bytes": getattr(process, "peak_pagefile", process.vms)})
        write(output/"frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
            "source_commit": args.expected_commit, "ground_truth_used": False, "variants": list(VARIANTS),
            "selected_variant": None, "artifacts": {str(path.relative_to(output)): sha(path)
                for path in output.rglob("*") if path.is_file()}})
        print(jsonable({"status": "complete", "variants": list(VARIANTS), "output": str(output),
            "freeze_sha256": sha(output/"frozen_predictions.json")}))
    except Exception as error:
        write(output/"failed.json", {"status": "unknown", "all_predictions_frozen": False,
            "ground_truth_used": False, "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--evidence-family", type=Path, required=True)
    parser.add_argument("--evidence-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    replay(parser.parse_args())
