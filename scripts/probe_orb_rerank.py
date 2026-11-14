"""Frozen-matrix ORB diagnostic; default validates inputs without executing pixels.

Actual execution requires a clean explicit source pin and preregistered profile.
No models, query GPS, geocoder, VO, labels, or original scores.json are used.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from orb_rerank import (PROFILE, file_sha256, mutual_ratio_matches, rank_candidates,
                        stable_topk, stream_raw_store, verify_geometry)

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def checked_file(root, relative):
    path = (root / relative.replace("\\", "/")).resolve(strict=True)
    if not path.is_relative_to(root) or path.name.casefold() == "scores.json":
        raise ValueError("artifact escaped frozen inputs or requested labels/scores")
    return path


def inspect_inputs(run):
    run = Path(run).resolve(strict=True)
    frozen_path = run / "frozen_prediction.json"
    freeze = read(frozen_path)
    if freeze.get("ground_truth_opened") is not False:
        raise ValueError("source prediction was not frozen before labels")
    for relative, expected in freeze["artifacts"].items():
        if file_sha256(checked_file(run, relative)) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    for required in ("experiment.json", "query_input.json", "reference_evidence.npz",
                     "reference_snapshot.json", "recovery_lineage.json"):
        if required not in freeze["artifacts"]:
            raise ValueError(f"required identity is not frozen: {required}")
    experiment, query = read(run / "experiment.json"), read(run / "query_input.json")
    if (experiment["window"][0] != 0 or experiment["runtime"]["query_stride"] != 1
            or experiment["profile"]["query_mode"] != "crops"
            or experiment["profile"]["score_normalization"] != "reference_mean"):
        raise ValueError("requires a proven zero-start stride1 crops control")
    with np.load(run / "reference_evidence.npz", allow_pickle=False) as saved:
        evidence = {name: saved[name] for name in saved.files if name != "metadata"}
        metadata = json.loads(str(saved["metadata"]))
    scores = evidence["similarities"]
    times = np.asarray(query["timestamps"], float)
    indices = np.asarray(query["sampling"]["dense_indices"])
    count = len(times)
    if (scores.ndim != 2 or scores.shape[0] != count or not 2 <= count <= 80
            or not 1 <= scores.shape[1] <= 1500 or not np.isfinite(scores).all()
            or not np.array_equal(evidence["query_indices"], np.arange(count))
            or not np.array_equal(evidence["query_times"], times)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer)
            or indices[0] != 0 or np.any(np.diff(indices) <= 0)
            or indices[-1] >= query["sampling"]["dense_count"]
            or metadata["query_mode"] != "crops" or metadata["source"] != "mapillary"):
        raise ValueError("unsupported or inconsistent frozen matrix/query registration")
    if (not np.isfinite(query["fps"]) or query["fps"] <= 0 or times[0] != 0
            or not np.isfinite(experiment["window"][1]) or times[-1] >= experiment["window"][1]
            or not np.allclose(times, indices/query["fps"], rtol=0, atol=1e-6)):
        raise ValueError("source ordinal map is not a proven zero-start constant-FPS registration")
    identifiers = np.asarray(evidence["ref_ids"])
    if (identifiers.shape != (scores.shape[1],) or len(set(identifiers.tolist())) != len(identifiers)
            or any(not str(value).startswith("mapillary:") for value in identifiers)
            or np.asarray(evidence["ref_latlon"]).shape != (scores.shape[1], 2)
            or not np.isfinite(evidence["ref_latlon"]).all()):
        raise ValueError("finite uniquely identified public reference coordinates required")
    raw_record = next(item for item in read(run / "reference_snapshot.json")
        if Path(item["relative"]).name == "ref_img_store.npz")
    raw_path = checked_file(run, raw_record["relative"])
    if raw_path.stat().st_size != raw_record["bytes"] or file_sha256(raw_path) != raw_record["sha256"]:
        raise ValueError("raw pixels differ from the frozen reference snapshot")
    video = run / "inputs/video.mp4"
    if file_sha256(video) != experiment["video"]["sha256"]:
        raise ValueError("copied encoded video differs from source prediction")
    with np.load(raw_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["xy"], evidence["ref_latlon"]):
            raise ValueError("raw-pixel coordinates do not match frozen matrix columns")
        if [str(value) for value in saved["ids"]] != [str(value).split(":", 1)[1] for value in identifiers]:
            raise ValueError("raw-pixel IDs/order do not match frozen matrix columns")
    registration = {"frozen_prediction": {"path": str(frozen_path), "sha256": file_sha256(frozen_path)},
        "matrix": {"path": str(run / "reference_evidence.npz"), "sha256": file_sha256(run / "reference_evidence.npz")},
        "recovery_lineage": {"path": str(run / "recovery_lineage.json"), "sha256": file_sha256(run / "recovery_lineage.json")},
        "raw_store": {"path": str(raw_path), "sha256": raw_record["sha256"]},
        "video": {"path": str(video), "sha256": experiment["video"]["sha256"]},
        "query_input_sha256": file_sha256(run / "query_input.json"),
        "query_pixels_sha256": query["pixels_sha256"], "window": experiment["window"],
        "n_queries": count, "n_refs": scores.shape[1], "ground_truth_opened": False}
    return registration, evidence, query


def stream_queries(video, query):
    """Decode sequentially, holding one frame, proving every selected source ordinal."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("cannot open frozen encoded video")
    wanted = np.asarray(query["sampling"]["dense_indices"], int)
    times = np.asarray(query["timestamps"], float)
    try:
        if not np.isclose(capture.get(cv2.CAP_PROP_FPS), query["fps"], rtol=0, atol=1e-9):
            raise ValueError("video FPS differs from frozen query registration")
        selected = 0
        for ordinal in range(int(query["sampling"]["dense_count"])):
            if not capture.grab():
                raise ValueError("video truncated before registered dense window ended")
            if not np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), ordinal+1, rtol=0, atol=1e-6):
                raise ValueError("decoder native frame ordinal differs from the frozen registration")
            if selected >= len(wanted) or ordinal != wanted[selected]:
                continue
            ok, pixels = capture.retrieve()
            if not ok or list(pixels.shape) != query["native_frame_shape"]:
                raise ValueError("decoded query shape differs from freeze")
            if not np.isclose(capture.get(cv2.CAP_PROP_POS_MSEC)/1000., times[selected], rtol=0, atol=1e-6):
                raise ValueError("actual video PTS differ from frozen query row")
            yield selected, pixels
            selected += 1
        if selected != len(wanted):
            raise ValueError("not every frozen query frame was decoded")
    finally:
        capture.release()


def square_view(pixels, view):
    if view not in (0, 1, 2) or pixels.ndim < 2 or not all(pixels.shape[:2]):
        raise ValueError("nonempty native image and crop view0/1/2 required")
    h, w = pixels.shape[:2]
    edge = min(h, w)
    offset = round((max(h, w)-edge)*view/2.)
    return pixels[:, offset:offset+edge] if w >= h else pixels[offset:offset+edge, :]


def headroom():
    import psutil
    available = {"physical_bytes": psutil.virtual_memory().available}
    if os.name == "nt":
        class Performance(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                (name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
        item = Performance()
        item.cb = ctypes.sizeof(item)
        if not ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo(ctypes.byref(item), item.cb):
            raise OSError("cannot verify actual Windows commit headroom")
        available["commit_bytes"] = (item.CommitLimit-item.CommitTotal)*item.PageSize
    if min(available.values()) < 8*1024**3:
        raise RuntimeError(f"queued: need8GiB physical/commit headroom; actual={available}")
    return available


def execute(output, registration, evidence, query):
    started = time.perf_counter()
    temp = output / "staging"
    temp.mkdir()
    ids = [str(value).split(":", 1)[1] for value in evidence["ref_ids"]]
    raw, staging = stream_raw_store(registration["raw_store"]["path"], temp / "raw.npy",
        expected_sha256=registration["raw_store"]["sha256"], expected_ids=ids, allowed_temp_root=temp)
    write(output / "raw_staging.json", staging)
    original_threads = cv2.getNumThreads()
    cv2.setNumThreads(2)
    detector = cv2.ORB_create(**dict(PROFILE["orb"], scoreType=cv2.ORB_HARRIS_SCORE))
    def features(pixels):
        pixels = cv2.resize(pixels, tuple(PROFILE["image_size_wh"]), interpolation=cv2.INTER_LINEAR)
        kp, des = detector.detectAndCompute(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), None)
        return np.asarray([point.pt for point in kp], np.float32).reshape(-1, 2), des
    scores = evidence["similarities"]
    centered = scores-scores.mean(axis=0, keepdims=True)
    candidates = stable_topk(centered, PROFILE["top_k"])
    references = OrderedDict()
    selections, all_records = [], []
    pixel_hash = hashlib.sha256()
    try:
        for row, pixels in stream_queries(registration["video"]["path"], query):
            pixels = np.ascontiguousarray(pixels)
            pixel_hash.update(str((pixels.shape, pixels.dtype.str)).encode())
            pixel_hash.update(memoryview(pixels).cast("B"))
            views = [features(square_view(pixels, view)) for view in range(3)]
            records = []
            for ref_index in candidates[row]:
                ref_index = int(ref_index)
                if ref_index not in references:
                    references[ref_index] = features(raw[ref_index])
                    if len(references) > 64:
                        references.popitem(last=False)
                references.move_to_end(ref_index)
                points_b, descriptors_b = references[ref_index]
                for view, (points_a, descriptors_a) in enumerate(views):
                    matches = mutual_ratio_matches(descriptors_a, descriptors_b)
                    metrics = verify_geometry(points_a[matches[:, 0]], points_b[matches[:, 1]], (512, 512), (512, 512))
                    records.append({"query_index": row, "ref_index": ref_index, "ref_id": str(evidence["ref_ids"][ref_index]),
                        "view": view, "raw_cosine": float(scores[row, ref_index]),
                        "centered_cosine": float(centered[row, ref_index]), **metrics})
            if len(records) != len(candidates[row])*3:
                raise ValueError("incomplete fixed candidate/view search")
            selections.append(rank_candidates(centered[row], candidates[row], records))
            all_records.extend(records)
            print(f"ORB query {row+1}/{len(candidates)} complete", flush=True)
        if pixel_hash.hexdigest() != query["pixels_sha256"]:
            raise ValueError("actual native query pixel aggregate differs from original forwards")
        chosen = np.asarray([item["selected_ref_index"] for item in selections], int)
        write(output / "pair_metrics.json", all_records)
        write(output / "diagnostic_prediction.json", {"status": "complete", "experimental": True,
            "selected_answer": None, "uses_rigid_vo": False, "temporal_decoder": None,
            "route_latlon": evidence["ref_latlon"][chosen].tolist(),
            "route_timestamps": evidence["query_times"].tolist(),
            "chosen_ref_ids": evidence["ref_ids"][chosen].tolist(), "selections": selections,
            "query_pixels_sha256": pixel_hash.hexdigest(), "profile_sha256": digest(PROFILE),
            "point_count": len(chosen), "pair_view_count": len(all_records), "elapsed_s": time.perf_counter()-started})
    finally:
        cv2.setNumThreads(original_threads)
        references.clear()
        del raw
        (temp / "raw.npy").unlink(missing_ok=True)
    for value in (registration["frozen_prediction"], registration["matrix"],
                  registration["recovery_lineage"], registration["raw_store"], registration["video"]):
        if file_sha256(value["path"]) != value["sha256"]:
            raise ValueError("a frozen source input changed during diagnostic")
    write(output / "frozen_prediction.json", {"ground_truth_opened": False,
        "model_forwards": False, "original_inputs_unchanged": True,
        "artifacts": {str(path.relative_to(output)): file_sha256(path)
            for path in output.rglob("*") if path.is_file()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--source-commit")
    p.add_argument("--profile-sha256")
    a = p.parse_args()
    registration, evidence, query = inspect_inputs(a.input_run)
    proposal = {"profile": PROFILE, "profile_sha256": digest(PROFILE), "inputs": registration,
        "input_policy": {"ground_truth": False, "query_gps": False, "vo": False,
                         "new_model_forwards": False, "public_reference_gps_for_route_output": True}}
    if not a.execute:
        print(json.dumps(proposal, indent=2))
        return 0
    current = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    if current != a.source_commit or dirty or a.profile_sha256 != digest(PROFILE):
        raise ValueError("actual execution requires clean source and preregistered profile pins")
    if a.output is None:
        raise ValueError("new isolated output root required")
    output = a.output.resolve()
    if output.drive.upper() != "E:" or output.is_relative_to(a.input_run.resolve()):
        raise ValueError("output must be an independent owned E-drive root")
    proposal.update(source_commit=current, source_sha256={name: file_sha256(ROOT / name)
        for name in ("scripts/orb_rerank.py", "scripts/probe_orb_rerank.py")},
        runtime={"opencv_version": cv2.__version__, "numpy_version": np.__version__,
                 "opencv_threads": 2, "minimum_headroom": headroom()})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "preregistration.json", proposal)
    for name in proposal["source_sha256"]:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    execute(output, registration, evidence, query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
