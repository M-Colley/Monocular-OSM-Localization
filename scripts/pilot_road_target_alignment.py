"""Frozen-input cue-only/raw/reference-mean target fits; no models or labels.

Every variant and assignment freezes before separate evaluation. The selected
route is an anchored experimental diagnostic, not an assisted-policy decision.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
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
from monocular_osm.geo import LocalFrame
from monocular_osm.poi_alignment import POIAlignmentConfig, MAX_FIELD_VALUES, build_appearance_field
from monocular_osm.road_target_alignment import RoadTargetConfig, align_road_targets, _array_sha, _json_sha
from scripts.prepare_road_target_motion import read, sha, write, verify_family, verify_source, windows_memory, require_headroom

VARIANTS = ("cue_only", "raw", "reference_mean")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_npz(path, specs):
    """Validate uncompressed member cardinalities before loading arrays."""
    require(path.stat().st_size <= 128*1024**2, "NPZ artifact exceeds fixed disk budget")
    with zipfile.ZipFile(path) as archive:
        for name, (max_values, kinds, max_itemsize) in specs.items():
            with archive.open(name+".npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("unsupported evidence NPY header")
            count = 1
            for value in shape:
                count *= value
            require(count <= max_values and dtype.kind in kinds and dtype.itemsize <= max_itemsize,
                    "NPZ member exceeds typed allocation budget: "+name)
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in specs}


def load_frozen_inputs(pbf, pbf_sha, motion, motion_sha, evidence, evidence_sha):
    verify_family(pbf/"frozen_pbf_probe.json", pbf_sha)
    verify_family(motion/"motion_frozen.json", motion_sha)
    frozen = verify_family(evidence/"frozen_predictions.json", evidence_sha)
    require(frozen["scoring_has_not_run"] is True, "appearance must have frozen before scoring")
    runtime = read(evidence/"runtime_and_lineage.json")
    require(runtime["labels_opened"] is False and runtime["scoring_called"] is False, "appearance label-free lineage required")
    verify_family(evidence.parent/"inputs_frozen.json", runtime["input_freeze_sha256"])
    plan = read(evidence.parent/"inputs/replay_plan.json")
    registration, summary = read(motion/"motion_registration.json"), read(motion/"motion_summary.json")
    require(summary["status"] == "complete" and not summary["ground_truth_used"]
        and registration["legacy_cache_used"] is False and registration["ground_truth_used"] is False,
        "fresh complete video-only motion required")
    require(registration["inputs"]["pbf_freeze_sha256"] == pbf_sha
        and registration["inputs"]["video_sha256"] == plan["query"]["video"]["sha256"]
        and registration["inputs"]["window"] == plan["query"]["registered_window"]
        and registration["stride"] == plan["query"]["stride"] and registration["fps"] == plan["query"]["fps"],
        "same-video/window/native motion-query grid required")
    native = bounded_npz(motion/"native_motion.npz", {"xz": (200000, "f", 8), "valid": (100000, "b", 1),
        "timestamps": (100000, "f", 8), "frame_ordinals": (100000, "iu", 8),
        "native_frame_pixels_sha256": (100000, "US", 256), "K": (9, "f", 8)})
    full = bounded_npz(evidence/"full_evidence.npz", {"raw_cosines": (8000000, "f", 8),
        "query_times": (80, "f", 8), "query_indices": (80, "iu", 8), "frame_ordinals": (80, "iu", 8),
        "ref_ids": (100000, "US", 512), "ref_latlon": (200000, "f", 8)})
    scores, ids, refs, times = full["raw_cosines"], full["ref_ids"], full["ref_latlon"], full["query_times"]
    require(scores.ndim == 2 and scores.dtype.itemsize in (4, 8) and 2 <= scores.shape[0] <= 80
        and 1 <= scores.shape[1] <= 100000 and np.isfinite(scores).all() and np.abs(scores).max() <= 1.00001,
        "bounded complete raw-cosine matrix required")
    require(ids.shape == (scores.shape[1],) and len(set(ids.astype(str))) == len(ids)
        and all(ids.astype(str)) and refs.shape == (len(ids), 2) and np.isfinite(refs).all()
        and np.abs(refs[:, 0]).max() <= 90 and np.abs(refs[:, 1]).max() <= 180,
        "complete unique public references required")
    require(ids.astype(str).tolist() == plan["union"]["ref_ids"] and len(ids) == plan["union"]["count"]
        and _array_sha(refs) == plan["union"]["ref_latlon_sha256"], "full reference union differs from frozen acquisition plan")
    ordinals = full["frame_ordinals"]
    require(times.shape == (len(scores),) and np.array_equal(full["query_indices"], np.arange(len(scores)))
        and np.array_equal(ordinals, plan["query"]["planned_frame_ordinals"])
        and _array_sha(times) == runtime["actual_query_times_sha256"], "frozen actual query identity required")
    native_ordinals = native["frame_ordinals"]
    n = len(native_ordinals)
    require(native_ordinals.ndim == 1 and 2 <= n <= 100000 and native["xz"].shape == (n, 2)
        and native["valid"].shape == (n,) and native["timestamps"].shape == (n,)
        and native["native_frame_pixels_sha256"].shape == (n,) and native["K"].shape == (3, 3)
        and np.all(np.diff(native_ordinals.astype(float)) > 0), "native ordinal/grid shapes must agree and increase")
    indexes = np.searchsorted(native_ordinals, ordinals)
    require(np.all(indexes < len(native["timestamps"])) and np.array_equal(native_ordinals[indexes], ordinals)
        and np.array_equal(native["timestamps"][indexes], times), "query ordinals/actual PTS differ from native motion")
    require(native["native_frame_pixels_sha256"][indexes].astype(str).tolist() == runtime["per_query_native_pixel_sha256"],
        "appearance query pixels differ from independently verified native motion pixels")
    require(_array_sha(native["timestamps"]) == summary["timestamps_sha256"]
        and _array_sha(native_ordinals) == summary["frame_ordinals_sha256"]
        and _array_sha(native["native_frame_pixels_sha256"]) == summary["native_frame_pixels_sha256"]
        and sha(motion/"native_motion.npz") == summary["motion_npz_sha256"], "typed motion identity changed")
    targets = read(pbf/"all_junction_targets.json")
    require(targets["selected_answer"] is None and targets["camera_fix"] is None, "target alternatives must remain unselected")
    return native, full, targets, registration, runtime


def metric_inputs(full, targets):
    refs = np.asarray(full["ref_latlon"], float)
    ordered = np.argsort(full["ref_ids"].astype(str), kind="stable")
    angles = np.radians(refs[ordered, 1])
    sin, cos = float(np.sin(angles).mean()), float(np.cos(angles).mean())
    require(np.hypot(sin, cos) > 1e-8, "public references have no stable local metric origin")
    frame = LocalFrame(float(refs[ordered, 0].mean()), float(np.degrees(np.arctan2(sin, cos))))
    ref_xy = np.column_stack(frame.to_xy(refs[:, 0], refs[:, 1]))
    observations = []
    for entry in targets["matches"]:
        row = entry["observation"]
        candidates = [{"id": candidate["id"], "xy": list(frame.to_xy(candidate["lat"], candidate["lon"]))}
                      for candidate in entry["candidates"]]
        observations.append({"id": "sign_"+_json_sha(row)[:20], "ref": row["ref"], "t_sec": row["t_sec"],
                             "confidence": row["confidence"], "candidates": candidates})
    points = np.vstack([ref_xy, *[np.asarray([target["xy"] for target in row["candidates"]]).reshape(-1, 2) for row in observations]])
    require(np.isfinite(points).all(), "public geometry must project into a finite metric frame")
    cell = 100.
    while True:
        low = np.floor((points.min(axis=0)-2000.)/cell)*cell
        high = np.ceil((points.max(axis=0)+2000.)/cell)*cell
        width, height = np.rint((high-low)/cell).astype(int)+1
        if len(full["query_times"])*int(width)*int(height) <= MAX_FIELD_VALUES:
            break
        cell *= 2
        require(cell <= 6400., "all-reference extent cannot fit fixed appearance budget")
    return frame, ref_xy, observations, low, (int(height), int(width)), cell


def add_latlon(candidate, frame):
    xy = np.asarray(candidate["route_xy"])
    candidate["route_latlon"] = np.column_stack(frame.to_latlon(xy[:, 0], xy[:, 1])).tolist()


def run_variants(native, full, frame, ref_xy, observations, low, shape, cell, evidence_sha, output):
    """Write every declared variant; only the caller may freeze after rechecks."""
    for variant in VARIANTS:
        field = None if variant == "cue_only" else build_appearance_field(ref_xy, np.ascontiguousarray(full["raw_cosines"]), full["query_times"],
            origin_xy=low, shape=shape, reference_mean=variant == "reference_mean",
            config=replace(POIAlignmentConfig(), field_cell_m=cell), provenance={"evidence_freeze_sha256": evidence_sha})
        result = align_road_targets(native["xz"], native["timestamps"], native["valid"], observations,
            query_times=full["query_times"], appearance_field=field,
            provenance={"fit_registration_sha256": sha(output/"fit_registration.json")})
        add_latlon(result["selected"], frame)
        for branch in result["branches"]:
            add_latlon(branch["best"], frame)
            for alternative in branch["alternatives"]:
                add_latlon(alternative, frame)
        write(output/f"all_candidates_{variant}.json", result)
        write(output/f"prediction_{variant}.json", {"status": "complete", "kind": "road_target_diagnostic",
            "variant": variant, "route_latlon": result["selected"]["route_latlon"],
            "route_timestamps": result["selected"]["route_timestamps"], "branch_id": result["selected"]["branch_id"],
            "objective": result["selected"]["objective"], "selection_scope": result["selection_scope"]})


def fit(args):
    require(Path(__file__).resolve() == (ROOT/"scripts/pilot_road_target_alignment.py").resolve(), "executed fit script differs from declared source")
    verify_source(ROOT, args.expected_commit)
    actual = args.expected_commit
    require_headroom(windows_memory())
    native, full, targets, motion_registration, runtime = load_frozen_inputs(args.pbf_family, args.pbf_freeze_sha256,
        args.motion_family, args.motion_freeze_sha256, args.evidence_family, args.evidence_freeze_sha256)
    frame, ref_xy, observations, low, shape, cell = metric_inputs(full, targets)
    output = args.output.resolve()
    require(output.drive.upper() == "E:" and not any(output.is_relative_to(root.resolve()) for root in
        (ROOT, args.pbf_family, args.motion_family, args.evidence_family.parent)), "new isolated E fit family required")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for relative in ("scripts/pilot_road_target_alignment.py", "scripts/prepare_road_target_motion.py",
        "monocular_osm/road_target_alignment.py", "monocular_osm/poi_alignment.py", "monocular_osm/geo.py"):
        path = output/"source_snapshot"/relative; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, path); sources[relative] = sha(path)
    registration = {"source_commit": actual, "sources": sources, "variants": list(VARIANTS), "selected_variant": None,
        "ground_truth_used": False, "new_models_ocr_vo_or_downloads": False,
        "pbf_family": str(args.pbf_family.resolve()), "pbf_freeze_sha256": args.pbf_freeze_sha256,
        "motion_family": str(args.motion_family.resolve()), "motion_freeze_sha256": args.motion_freeze_sha256,
        "evidence_family": str(args.evidence_family.resolve()), "evidence_freeze_sha256": args.evidence_freeze_sha256,
        "full_matrix_sha256": sha(args.evidence_family/"full_evidence.npz"),
        "query_times_sha256": _array_sha(full["query_times"]), "query_ordinals_sha256": _array_sha(full["frame_ordinals"]),
        "reference_ids_sha256": _json_sha(full["ref_ids"].astype(str).tolist()),
        "profile": asdict(RoadTargetConfig()), "map_crs": frame.crs.to_string(),
        "appearance": {"all_references": len(ref_xy), "origin_xy": low.tolist(), "shape": list(shape), "cell_m": cell,
            "bound_rule": "all reference/target extent+2km;100m cells doubled only to fit16M values",
            "field_config": asdict(replace(POIAlignmentConfig(), field_cell_m=cell)), "max_field_values": MAX_FIELD_VALUES},
        "selection_scope": "anchored diagnostic only; all-null unresolved, no baseline admission",
        "target_inventory_scope": targets["scope"], "target_bbox_complete": targets["complete_parent_bbox_coverage"],
        "reference_encoding_identity": runtime["runtime"]["reference_encoding_identity"]}
    write(output/"fit_registration.json", registration)
    started = time.monotonic()
    try:
        run_variants(native, full, frame, ref_xy, observations, low, shape, cell, args.evidence_freeze_sha256, output)
        load_frozen_inputs(args.pbf_family, args.pbf_freeze_sha256, args.motion_family, args.motion_freeze_sha256,
                           args.evidence_family, args.evidence_freeze_sha256)
        verify_source(ROOT, actual, sources)
        import psutil
        memory = psutil.Process().memory_info()
        write(output/"fit_summary.json", {"status": "complete", "elapsed_s": time.monotonic()-started,
            "variants": list(VARIANTS), "selected_variant": None, "labels_opened": False, "torch_imported": "torch" in sys.modules,
            "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss), "peak_private_bytes": getattr(memory, "peak_pagefile", memory.vms)})
        write(output/"frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True, "ground_truth_used": False,
            "source_commit": actual, "variants": list(VARIANTS), "selected_variant": None,
            "artifacts": {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}})
        print(json.dumps({"status": "complete", "freeze_sha256": sha(output/"frozen_predictions.json"), "variants": list(VARIANTS)}))
    except Exception as error:
        write(output/"failed.json", {"status": "unknown", "all_predictions_frozen": False, "ground_truth_used": False,
            "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for name in ("pbf", "motion", "evidence"):
        parser.add_argument(f"--{name}-family", type=Path, required=True)
        parser.add_argument(f"--{name}-freeze-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    fit(parser.parse_args())
