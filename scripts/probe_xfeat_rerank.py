"""Pinned XFeat replacement of frozen ORB extraction/matching; no policy change."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from orb_rerank import PROFILE as ORB_PROFILE, file_sha256, rank_candidates, stable_topk, stream_raw_store, verify_geometry
from probe_orb_rerank import ROOT, digest, headroom, inspect_inputs, read, square_view, stream_queries, write
from xfeat_backend import LOCAL_PROFILE, ReleasedXFeat, inspect_assets

PROFILE = deepcopy(ORB_PROFILE)
PROFILE.update(version="xfeat-fundamental-coverage-v1", orb=None, xfeat=LOCAL_PROFILE,
    matching={"type": "official_XFeat_sparse_reciprocal_cosine", "min_cossim": -1},
    paired_control="unmodified_reference_mean_centered_argmax",
    parent_geometry_profile_sha256=digest(ORB_PROFILE))
SOURCES = ("scripts/probe_xfeat_rerank.py", "scripts/xfeat_backend.py",
           "scripts/probe_orb_rerank.py", "scripts/orb_rerank.py")


def points_sha(first, second):
    array = np.ascontiguousarray(np.hstack((first, second)), dtype=np.float32)
    value = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    if array.size:
        value.update(memoryview(array).cast("B"))
    return value.hexdigest()


def execute(output, registration, evidence, query, assets):
    began = time.perf_counter()
    backend = ReleasedXFeat(assets["root"])
    write(output / "backend_identity.json", {"assets": backend.assets, "runtime": backend.runtime,
                                            "actual_numerical_state": backend.state})
    temp = output / "staging"
    temp.mkdir()
    raw, staging = stream_raw_store(registration["raw_store"]["path"], temp / "raw.npy",
        expected_sha256=registration["raw_store"]["sha256"],
        expected_ids=[str(value).split(":", 1)[1] for value in evidence["ref_ids"]], allowed_temp_root=temp)
    write(output / "raw_staging.json", staging)
    def features(pixels):
        return backend.features(cv2.resize(pixels, (512, 512), interpolation=cv2.INTER_LINEAR))
    scores = evidence["similarities"]
    centered = scores-scores.mean(axis=0, keepdims=True)
    candidates = stable_topk(centered, 8)
    references = OrderedDict()
    selections, all_records = [], []
    pixel_hash = hashlib.sha256()
    original_threads = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        for row, pixels in stream_queries(registration["video"]["path"], query):
            if row != len(selections):
                raise ValueError("native query rows are not the complete ordered frozen grid")
            pixels = np.ascontiguousarray(pixels)
            pixel_hash.update(str((pixels.shape, pixels.dtype.str)).encode())
            pixel_hash.update(memoryview(pixels).cast("B"))
            views = [features(square_view(pixels, view)) for view in range(3)]
            records = []
            for index in candidates[row]:
                index = int(index)
                if index not in references:
                    references[index] = features(raw[index])
                    if len(references) > 64:
                        references.popitem(last=False)
                references.move_to_end(index)
                ref_features = references[index]
                for view, query_features in enumerate(views):
                    pa, pb = backend.correspondences(query_features, ref_features)
                    metrics = verify_geometry(pa, pb, (512, 512), (512, 512))
                    records.append({"query_index": row, "ref_index": index,
                        "ref_id": str(evidence["ref_ids"][index]), "view": view,
                        "raw_cosine": float(scores[row, index]), "centered_cosine": float(centered[row, index]),
                        "query_feature_count": len(query_features[0]), "reference_feature_count": len(ref_features[0]),
                        "correspondences_sha256": points_sha(pa, pb), **metrics})
            if len(records) != len(candidates[row])*3:
                raise ValueError("incomplete fixed candidate/view search")
            selections.append(rank_candidates(centered[row], candidates[row], records))
            all_records.extend(records)
            print(f"XFeat query {row+1}/{len(candidates)} complete", flush=True)
        if len(selections) != len(candidates) or len(all_records) != len(candidates)*len(candidates[0])*3:
            raise ValueError("incomplete fixed query/candidate/view search")
        if pixel_hash.hexdigest() != query["pixels_sha256"]:
            raise ValueError("native query pixels differ from original frozen forward")
        chosen = np.asarray([item["selected_ref_index"] for item in selections], int)
        control = np.asarray([item["original_ref_index"] for item in selections], int)
        for name, indices, method in (("diagnostic_prediction.json", chosen, "xfeat_geometry_rerank"),
                ("centered_top1_control.json", control, "unmodified_reference_mean_centered_argmax")):
            write(output / name, {"status": "complete", "experimental": True, "selected_answer": None,
                "method": method, "uses_rigid_vo": False, "temporal_decoder": None,
                "route_latlon": evidence["ref_latlon"][indices].tolist(),
                "route_timestamps": evidence["query_times"].tolist(),
                "chosen_ref_ids": evidence["ref_ids"][indices].tolist(), "point_count": len(indices),
                "query_pixels_sha256": pixel_hash.hexdigest(), "profile_sha256": digest(PROFILE),
                "pair_view_count": len(all_records), "elapsed_s": time.perf_counter()-began,
                **({"selections": selections} if name == "diagnostic_prediction.json" else {})})
        write(output / "pair_metrics.json", all_records)
        backend.verify_unchanged()
        write(output / "backend_identity.json", {"assets": backend.assets, "runtime": backend.runtime,
                                                "actual_numerical_state": backend.state})
    finally:
        cv2.setNumThreads(original_threads)
        references.clear()
        del raw
        (temp / "raw.npy").unlink(missing_ok=True)
    for name in ("frozen_prediction", "matrix", "recovery_lineage", "raw_store", "video"):
        item = registration[name]
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("original frozen input changed during XFeat execution")
    for name in SOURCES:
        if file_sha256(ROOT/name) != file_sha256(output/"source_snapshot"/name):
            raise ValueError("diagnostic source changed during execution")
    if subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() != read(output/"preregistration.json")["source_commit"]:
        raise ValueError("diagnostic source HEAD changed during execution")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise ValueError("diagnostic source checkout is no longer clean")
    write(output / "frozen_prediction.json", {"ground_truth_opened": False,
        "new_global_descriptor_forwards": False, "local_feature_model": "official_XFeat_sparse",
        "original_inputs_unchanged": True, "policy_adopted": False,
        "artifacts": {str(path.relative_to(output)): file_sha256(path)
            for path in output.rglob("*") if path.is_file()}})


def main():
    if Path(__file__).resolve(strict=True) != (ROOT/"scripts/probe_xfeat_rerank.py").resolve(strict=True):
        raise ValueError("executed XFeat driver is outside its recorded source checkout")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-run", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--source-commit")
    p.add_argument("--profile-sha256")
    args = p.parse_args()
    registration, evidence, query = inspect_inputs(args.input_run)
    assets = inspect_assets(args.assets)
    proposal = {"profile": PROFILE, "profile_sha256": digest(PROFILE), "inputs": registration,
        "assets": assets, "input_policy": {"ground_truth": False, "query_gps": False, "vo": False,
            "new_global_descriptors": False, "public_reference_gps_for_output": True}}
    if not args.execute:
        import json
        print(json.dumps(proposal, indent=2))
        return 0
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    if dirty or commit != args.source_commit or args.profile_sha256 != digest(PROFILE):
        raise ValueError("actual execution requires clean explicit source/profile pins")
    # Set before the first Torch import/constructor; this is a CPU-only command.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.output is None:
        raise ValueError("new isolated output required")
    output = args.output.resolve()
    if output.drive.upper() != "E:" or output.is_relative_to(args.input_run.resolve()) or output.is_relative_to(args.assets.resolve()):
        raise ValueError("output must be a new independent owned E-drive root")
    proposal.update(source_commit=commit, source_sha256={name: file_sha256(ROOT/name) for name in SOURCES},
        runtime={"opencv_version": cv2.__version__, "numpy_version": np.__version__, "minimum_headroom": headroom()})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "preregistration.json", proposal)
    for name in SOURCES:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    for name in assets["code"]:
        target = output / "upstream_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(assets["root"])/name, target)
    target = output / "weights/xfeat.pt"
    target.parent.mkdir()
    shutil.copyfile(assets["weight"]["path"], target)
    if file_sha256(target) != assets["weight"]["sha256"]:
        raise ValueError("frozen XFeat asset copy mismatch")
    execute(output, registration, evidence, query, assets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
