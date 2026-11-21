"""Paired input-frozen panorama/XFeat diagnostic; no acquisition or Answer policy."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path: sys.path.insert(0, str(_path))
sys.dont_write_bytecode = True
import cv2
import numpy as np
import probe_xfeat_rerank as original
import probe_orb_rerank as inputs
import orb_rerank as orb
import xfeat_backend as xbackend
from orb_rerank import rank_candidates, stable_topk, stream_raw_store, verify_geometry
from xfeat_backend import ReleasedXFeat, inspect_assets
from monocular_osm import panorama_views as panorama
from monocular_osm import reference_metadata as metadata

PROFILE = deepcopy(original.PROFILE)
PROFILE.update(version="panorama-xfeat-paired-v1", variants=["original_single", "panorama_aware"],
    panorama=dict(panorama.PROFILE), parent_xfeat_profile_sha256=inputs.digest(original.PROFILE),
    reference_view_pooling="maximum across all reference and query views",
    reference_view_ties="lowest query-view index, then lowest panorama-view index",
    reference_feature_cache={"policy": "LRU", "bundles": 64, "max_features_per_bundle": 5,
        "max_feature_payload_bytes": 64 * 5 * 2500 * (64 * 4 + 2 * 4)},
    correspondence_evidence={"format": "per-query NPZ; float32 matched points, returned F and valid binary mask",
        "max_chunk_bytes": 8 * 1024**2, "max_total_bytes": 100 * 1024**2,
        "max_galleries": 2, "max_unique_pairs": 2 * 80 * 8 * 3 * 5, "max_matches_per_pair": 2500,
        "over_budget": "refuse completion; preserve partial evidence; never trim"},
    minimum_free_disk_bytes=3 * 1024**3, max_final_output_bytes=512 * 1024**2,
    max_metadata_bytes=96 * 1024**2,
    source_contract="known panorama metadata AND complete equirectangular cached raster required",
    limitations=["stitched panoramas are approximately central cameras", "yaw offsets are not world headings",
        "four reference views enlarge a maximum pool; geometry scores are uncalibrated",
        "municipal reference global-encoding revision/preprocessing remains unverified",
        "posthoc DEV diagnostic; no default or policy adoption"], selected_variant=None)
PROFILE = json.loads(json.dumps(PROFILE))
PROFILE_SHA256 = inputs.digest(PROFILE)
SOURCES = tuple(dict.fromkeys(("scripts/probe_panorama_xfeat.py", *original.SOURCES,
    "scripts/panorama_gallery_inputs.py", "scripts/pilot_image_gallery.py", "scripts/pilot_image_paired.py",
    "scripts/pilot_sparse_city_replay.py", "scripts/plan_sparse_city_replay.py",
    "monocular_osm/panorama_views.py", "monocular_osm/reference_metadata.py",
    "monocular_osm/geo.py", "monocular_osm/__init__.py", "monocular_osm/frame_extraction.py",
    "monocular_osm/provenance.py", "monocular_osm/assisted_worker.py", "monocular_osm/stages/poi_runtime.py")))
SOURCES = tuple(dict.fromkeys((*SOURCES, *sorted(str(p.relative_to(ROOT)).replace("\\", "/")
    for p in (ROOT / "monocular_osm").rglob("*.py")))))


def require(value, reason):
    if not value: raise ValueError(reason)


def allowed(path):
    value = Path(path)
    for p in (value, value.resolve()):
        text, name = str(p).replace("\\", "/").casefold(), p.name.casefold()
        score = name.endswith((".json", ".jsonl")) and re.search(r"(?:^|[_.-])(?:scores?|scoring)(?:[_.-]|$)", name)
        require("ground_truth" not in text and not score, "labels and saved scores are outside diagnostic inputs")
    return value


def sha(path): return original.file_sha256(allowed(path))


def read(path):
    path = allowed(path)
    require(path.stat().st_size <= 64 * 1024**2, "JSON input exceeds diagnostic bound")
    return inputs.read(path)


def write(path, value):
    with allowed(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def relative(name):
    path = Path(str(name).replace("\\", "/"))
    require(not path.is_absolute() and not path.drive and ".." not in path.parts, "escaping artifact path")
    return path


def verify_freeze(path, expected=None):
    path = allowed(path).resolve(strict=True)
    require(expected is None or sha(path) == expected, "registered freeze SHA changed")
    frozen, targets = read(path), []
    for name, digest in frozen["artifacts"].items():
        allowed(name)
        target = allowed(path.parent / relative(name)).resolve(strict=True)
        require(target.is_relative_to(path.parent), "artifact escaped frozen root")
        targets.append((target, digest))
    for target, digest in targets: require(sha(target) == digest, "frozen member changed")
    return frozen


def artifact_hashes(root):
    paths = [allowed(p) for p in Path(root).rglob("*") if p.is_file()]
    require(all(p.resolve().is_relative_to(Path(root).resolve()) for p in paths), "output artifact escaped owned root")
    return {str(p.relative_to(root)): sha(p) for p in paths}


def inspect_metadata(freeze_path, expected, evidence):
    """Rebuild every camera field from the frozen successful public responses."""
    path = allowed(freeze_path).resolve(strict=True)
    frozen = verify_freeze(path, expected)
    require(frozen.get("status") == "complete" and frozen.get("all_ids_retained") is True
        and frozen.get("models_or_labels") is False and frozen.get("selected_answer") is None,
        "complete all-ID input-only metadata acquisition required")
    require({"metadata_plan.json", "all_reference_metadata.json", "request_ledger.json"} <= set(frozen["artifacts"]), "metadata roles missing")
    require(sum((path.parent / relative(n)).stat().st_size for n in frozen["artifacts"]) <= PROFILE["max_metadata_bytes"], "metadata copy budget exceeded")
    plan = read(path.parent / "metadata_plan.json")
    rebuilt = metadata.plan_requests(evidence["ref_ids"], evidence["ref_latlon"], provenance=plan["provenance"])
    require(all(plan[k] == v for k, v in rebuilt.items()), "metadata membership, geotags or requests differ from gallery")
    ledger = read(path.parent / "request_ledger.json")
    require(ledger["all_batches_successful"] is True and len(ledger["batches"]) == len(plan["batches"])
        and ledger["requests"] <= metadata.PROFILE["max_requests"]
        and ledger["decoded_body_bytes"] <= metadata.PROFILE["max_total_decoded_body_bytes"], "partial or over-budget metadata")
    records = {}
    for batch, outcome in zip(plan["batches"], ledger["batches"]):
        require(outcome["index"] == batch["index"] and outcome["success"] is True
            and 1 <= len(outcome["attempts"]) <= metadata.PROFILE["max_attempts_per_batch"], "incomplete metadata batch")
        attempt = outcome["attempts"][-1]
        require(attempt == f"batch_{batch['index']:04d}_attempt_{len(outcome['attempts'])}", "metadata attempt order changed")
        prefix = "attempts/" + attempt
        names = {str(relative(n)).replace("\\", "/") for n in frozen["artifacts"]}
        require({prefix + "/request.json", prefix + "/response.json", prefix + "/response.body"} <= names, "metadata response not frozen")
        request, response = read(path.parent / prefix / "request.json"), read(path.parent / prefix / "response.json")
        require(request["endpoint"] == metadata.PROFILE["endpoint"] and request["params"] == batch["params"]
            and request["batch_index"] == batch["index"], "metadata request changed")
        body_path = allowed(path.parent / prefix / "response.body")
        require(body_path.stat().st_size <= metadata.PROFILE["max_decoded_body_bytes"], "metadata response body exceeds bound")
        body = body_path.read_bytes()
        require(response["status"] == 200 and response["error_type"] is None and response["parsed_successfully"] is True
            and response["body_redacted_or_reencoded"] is False
            and hashlib.sha256(body).hexdigest() == response["stored_body_sha256"] == response["decoded_body_sha256"], "metadata response not a complete unmodified success")
        records.update(metadata.parse_response(body, batch["reference_ids"]))
    require(records == read(path.parent / "all_reference_metadata.json"), "metadata differs from public response bytes")
    cameras = {k: r["fields"]["camera_type"] if r["field_status"]["camera_type"] == "present" else None for k, r in records.items()}
    return {"path": str(path), "sha256": sha(path), "reference_count": len(records),
        "panorama_count": sum(v in panorama.PROFILE["panorama_camera_types"] for v in cameras.values())}, cameras


def inspect_prior(path, expected, registration):
    path = allowed(path).resolve(strict=True)
    frozen = verify_freeze(path / "frozen_prediction.json", expected)
    require(frozen.get("ground_truth_opened") is False and frozen.get("policy_adopted") is False
        and frozen.get("original_inputs_unchanged") is True, "prior XFeat was not input-frozen")
    require({"preregistration.json", "backend_identity.json", "pair_metrics.json", "diagnostic_prediction.json"} <= set(frozen["artifacts"]), "prior XFeat roles missing")
    reg, backend = read(path / "preregistration.json"), read(path / "backend_identity.json")
    require(reg["profile_sha256"] == inputs.digest(original.PROFILE) and reg["profile"] == original.PROFILE, "prior XFeat profile changed")
    for key in ("frozen_prediction", "matrix", "recovery_lineage", "raw_store", "video"):
        require(reg["inputs"][key]["sha256"] == registration[key]["sha256"], "prior XFeat used another input")
    for key in ("query_input_sha256", "query_pixels_sha256", "window", "n_queries", "n_refs"):
        require(reg["inputs"][key] == registration[key], "prior XFeat grid changed")
    require(reg["runtime"]["opencv_version"] == cv2.__version__ and reg["runtime"]["numpy_version"] == np.__version__, "prior control library identity differs")
    return {"path": str(path), "freeze_sha256": sha(path / "frozen_prediction.json"), "backend": backend,
        "prediction": read(path / "diagnostic_prediction.json"), "records": read(path / "pair_metrics.json")}


@contextmanager
def offline():
    old, old_ex = socket.socket.connect, socket.socket.connect_ex
    def refused(*args, **kwargs): raise RuntimeError("network is forbidden in the paired diagnostic")
    socket.socket.connect = socket.socket.connect_ex = refused
    try: yield
    finally: socket.socket.connect, socket.socket.connect_ex = old, old_ex


def resources(output):
    require(os.name == "nt", "actual physical AND commit verification requires Windows")
    memory = inputs.headroom()
    require({"physical_bytes", "commit_bytes"} <= set(memory), "actual physical and commit proof required")
    require(all(type(memory[k]) is int and memory[k] >= 8 * 1024**3 for k in ("physical_bytes", "commit_bytes")), "8GiB physical AND actual commit required")
    free = shutil.disk_usage(output).free
    require(free >= PROFILE["minimum_free_disk_bytes"], "insufficient owned-disk headroom")
    return dict(memory, free_disk_bytes=free)


def captured_geometry(pa, pb):
    """Unchanged geometry helper, retaining safely returned fitting data."""
    fit = {"status": "not_called", "F": np.empty((0,), np.float64), "mask": np.empty((0,), np.uint8)}
    def capture(a, b, **kwargs):
        try: F, mask = cv2.findFundamentalMat(a, b, **kwargs)
        except cv2.error:
            fit["status"] = "opencv_error"
            raise
        fit["status"] = "none" if F is None else "returned"
        if F is not None:
            matrix = np.asarray(F)
            require(matrix.size <= 64 and matrix.dtype.kind in "fiu" and matrix.dtype.itemsize <= 8, "unbounded fitting matrix")
            fit["F"] = np.array(matrix, copy=True)  # Rank tolerances depend on the actual returned dtype.
            # F=None can accompany an uninitialised mask. Never touch it.
            if (matrix.shape == (3, 3) and np.isfinite(matrix).all() and np.linalg.norm(matrix) != 0
                    and np.linalg.matrix_rank(matrix) == 2 and mask is not None):
                candidate = np.asarray(mask).reshape(-1)
                if candidate.shape == (len(a),) and np.isin(candidate, [0, 1]).all():
                    fit["mask"] = np.array(candidate, dtype=np.uint8, copy=True)
        return F, mask
    metrics = verify_geometry(pa, pb, (512, 512), (512, 512), find_fn=capture)
    return metrics, fit


class CorrespondenceWriter:
    """One bounded query chunk at a time; no all-run point buffer or trimming."""
    def __init__(self, root, budget=None):
        self.root = Path(root); self.root.mkdir(exist_ok=False)
        self.total_bytes = self.pairs = 0
        self.chunks, self.arrays, self.query = [], {}, None
        self.budget = {"bytes": 0, "pairs": 0} if budget is None else budget

    def start(self, query):
        require(self.query is None and not self.arrays, "previous correspondence chunk unfinished")
        self.query = int(query)

    def add(self, pa, pb, fit):
        require(self.query is not None and self.budget["pairs"] < PROFILE["correspondence_evidence"]["max_unique_pairs"], "pair budget exceeded")
        first, second = np.asarray(pa), np.asarray(pb)
        require(first.shape == second.shape and first.ndim == 2 and first.shape[1] == 2
            and len(first) <= 2500 and np.isfinite(first).all() and np.isfinite(second).all(), "bounded finite correspondences required")
        points = np.ascontiguousarray(np.hstack((first, second)), dtype=np.float32)
        prefix = f"p{self.pairs:05d}"
        self.arrays[prefix + "_points"], self.arrays[prefix + "_F"], self.arrays[prefix + "_mask"] = points, fit["F"], fit["mask"]
        self.pairs += 1
        self.budget["pairs"] += 1
        require(sum(a.nbytes for a in self.arrays.values()) <= PROFILE["correspondence_evidence"]["max_chunk_bytes"], "correspondence chunk payload exceeded")
        return {"chunk": f"correspondences/query_{self.query:03d}.npz", "prefix": prefix,
            "fit_status": fit["status"], "correspondences_sha256": original.points_sha(first, second)}

    def finish(self):
        require(self.query is not None, "no correspondence query started")
        buffer = io.BytesIO(); np.savez(buffer, **self.arrays); payload = buffer.getvalue()
        limits = PROFILE["correspondence_evidence"]
        require(len(payload) <= limits["max_chunk_bytes"] and self.budget["bytes"] + len(payload) <= limits["max_total_bytes"], "correspondence disk budget exceeded; no trimming allowed")
        path = self.root / f"query_{self.query:03d}.npz"
        with path.open("xb") as stream: stream.write(payload)
        self.total_bytes += len(payload)
        self.budget["bytes"] += len(payload)
        self.chunks.append({"path": path.name, "sha256": sha(path), "bytes": len(payload)})
        self.arrays.clear(); self.query = None


def replay_geometry(record, chunk):
    """Decision replay from retained points/F/mask, without fitting or a model."""
    key = record["evidence"]["prefix"]
    points, F, mask = [chunk[key + suffix] for suffix in ("_points", "_F", "_mask")]
    require(original.points_sha(points[:, :2], points[:, 2:]) == record["correspondences_sha256"], "saved correspondence identity changed")
    return verify_geometry(np.ascontiguousarray(points[:, :2]), np.ascontiguousarray(points[:, 2:]), (512, 512), (512, 512),
        find_fn=lambda *a, **k: (F if F.size else None, mask if mask.size else None))


def feature_identity(features):
    points, descriptors = features
    if hasattr(descriptors, "detach"): descriptors = descriptors.detach().cpu().numpy()
    arrays = [np.ascontiguousarray(a) for a in (points, descriptors)]
    require(arrays[0].shape == (len(arrays[1]), 2) and arrays[1].shape == (len(arrays[0]), 64)
        and len(arrays[0]) <= 2500 and all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays), "invalid sparse features")
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.shape, a.dtype.str)).encode())
        if a.size: h.update(memoryview(a).cast("B"))
    return {"sha256": h.hexdigest(), "count": len(arrays[0])}


def paired_search(output, evidence, query, camera_types, raw, backend, *, stream, check=resources, geometry=captured_geometry, correspondence_budget=None):
    """One query extraction; LRU64 original/virtual reference-feature bundles."""
    scores = evidence["similarities"]
    require(scores.ndim == 2 and scores.dtype == np.float32 and 2 <= len(scores) <= 80
        and 8 <= scores.shape[1] <= 10000 and np.isfinite(scores).all(), "bounded cosine matrix required")
    require(set(camera_types) == set(evidence["ref_ids"].tolist()) and len(raw) == scores.shape[1], "complete camera and raw membership required")
    centered = scores - scores.mean(axis=0, keepdims=True)
    candidates = stable_topk(centered, 8)
    writer = CorrespondenceWriter(output / "correspondences", correspondence_budget)
    cache, events, memory = OrderedDict(), [], []
    records = {name: [] for name in PROFILE["variants"]}
    selections = {name: [] for name in PROFILE["variants"]}
    pixels_hash = hashlib.sha256()
    def features(pixels):
        return backend.features(cv2.resize(pixels, (512, 512), interpolation=cv2.INTER_LINEAR))
    def pair(row, index, view, qfeatures, rfeatures, ref_view, yaw):
        pa, pb = backend.correspondences(qfeatures, rfeatures)
        metrics, fit = geometry(pa, pb)
        saved = writer.add(pa, pb, fit)
        return {"query_index": row, "ref_index": index, "ref_id": str(evidence["ref_ids"][index]),
            "view": view, "raw_cosine": float(scores[row, index]), "centered_cosine": float(centered[row, index]),
            "query_feature_count": len(qfeatures[0]), "reference_feature_count": len(rfeatures[0]),
            "correspondences_sha256": saved["correspondences_sha256"], **metrics,
            "reference_view_index": ref_view, "reference_yaw_deg": yaw, "evidence": saved}
    try:
        for row, pixels in stream:
            require(row == len(selections["original_single"]) and row < len(scores), "query rows are not the complete ordered grid")
            memory.append(check(output)); writer.start(row)
            pixels = np.ascontiguousarray(pixels)
            pixels_hash.update(str((pixels.shape, pixels.dtype.str)).encode()); pixels_hash.update(memoryview(pixels).cast("B"))
            queries = [features(inputs.square_view(pixels, view)) for view in range(3)]
            events.append({"kind": "query", "query_index": row, "views": [feature_identity(f) for f in queries]})
            local = {name: [] for name in PROFILE["variants"]}
            for index in candidates[row]:
                index = int(index); identifier = str(evidence["ref_ids"][index]); camera = camera_types[identifier]
                if index not in cache:
                    memory.append(check(output))
                    unmodified = features(raw[index])
                    if camera in panorama.PROFILE["panorama_camera_types"]:
                        views = [(yaw, features(pixels)) for yaw, pixels in panorama.reference_views(raw[index], camera)]
                    else: views = [(None, unmodified)]
                    cache[index] = {"original": unmodified, "views": views}
                    if len(cache) > 64: cache.popitem(last=False)
                    events.append({"kind": "reference_cache_miss", "ref_index": index, "ref_id": identifier,
                        "camera_type": camera, "original": feature_identity(unmodified),
                        "views": [{"yaw_deg": yaw, **feature_identity(f)} for yaw, f in views]})
                cache.move_to_end(index); bundle = cache[index]
                for view, qfeatures in enumerate(queries):
                    baseline = pair(row, index, view, qfeatures, bundle["original"], None, None)
                    local["original_single"].append(baseline)
                    if camera not in panorama.PROFILE["panorama_camera_types"]:
                        local["panorama_aware"].append(dict(baseline, reference_view_index=0))
                if camera in panorama.PROFILE["panorama_camera_types"]:
                    for ref_view, (yaw, rfeatures) in enumerate(bundle["views"]):
                        for view, qfeatures in enumerate(queries):
                            local["panorama_aware"].append(pair(row, index, view, qfeatures, rfeatures, ref_view, yaw))
            expected = sum(12 if camera_types[str(evidence["ref_ids"][int(i)])] in panorama.PROFILE["panorama_camera_types"] else 3 for i in candidates[row])
            require(len(local["original_single"]) == 24 and len(local["panorama_aware"]) == expected, "candidate/view search incomplete")
            for variant in PROFILE["variants"]:
                selection = rank_candidates(centered[row], candidates[row], local[variant])
                if variant == "panorama_aware":
                    winning = [r for r in local[variant] if r["ref_index"] == selection["selected_ref_index"]
                        and r["view"] == selection["selected_view"] and r["score"] == selection["geometry_score"]]
                    chosen = min(winning, key=lambda r: r["reference_view_index"]) if winning else None
                    selection.update(selected_reference_view_index=chosen["reference_view_index"] if chosen else None,
                        selected_reference_yaw_deg=chosen["reference_yaw_deg"] if chosen else None)
                selections[variant].append(selection); records[variant].extend(local[variant])
            writer.finish()
            print(f"paired XFeat query {row+1}/{len(scores)} complete", flush=True)
        require(len(selections["original_single"]) == len(scores) and pixels_hash.hexdigest() == query["pixels_sha256"], "native query grid or pixels differ from control")
        return {"records": records, "selections": selections, "feature_events": events, "memory": memory,
            "chunks": writer.chunks, "unique_pairs": writer.pairs, "correspondence_bytes": writer.total_bytes,
            "query_pixels_sha256": pixels_hash.hexdigest()}
    finally: cache.clear()


def compare_control(result, prior, evidence):
    current, previous = result["records"]["original_single"], prior["records"]
    require(len(current) == len(previous) and all({k: a[k] for k in b} == b for a, b in zip(current, previous)), "original XFeat pair evidence differs from prior control")
    selected = result["selections"]["original_single"]
    require(selected == prior["prediction"]["selections"], "original XFeat ranking differs from frozen control")
    indices = np.asarray([s["selected_ref_index"] for s in selected])
    for key, value in (("chosen_ref_ids", evidence["ref_ids"][indices].tolist()),
            ("route_latlon", evidence["ref_latlon"][indices].tolist()), ("route_timestamps", evidence["query_times"].tolist())):
        require(prior["prediction"][key] == value, "prior XFeat route identity changed")


def run_component(output, evidence, query, cameras, raw, backend, *, stream, prior=None,
                  check=resources, geometry=captured_geometry, correspondence_budget=None):
    """Write both complete routes/evidence; caller alone freezes the whole group."""
    output = Path(output); output.mkdir(exist_ok=False)
    started = time.perf_counter()
    try:
        result = paired_search(output, evidence, query, cameras, raw, backend, stream=stream, check=check, geometry=geometry, correspondence_budget=correspondence_budget)
        if prior is not None: compare_control(result, prior, evidence)
        for variant in PROFILE["variants"]:
            selected = result["selections"][variant]
            indices = np.asarray([s["selected_ref_index"] for s in selected])
            write(output / f"prediction_{variant}.json", {"status": "complete", "experimental": True,
                "selected_answer": None, "selected_variant": None, "method": "panorama_xfeat_" + variant,
                "uses_rigid_vo": False, "temporal_decoder": None, "heading_claim": False,
                "profile_sha256": PROFILE_SHA256, "route_latlon": evidence["ref_latlon"][indices].tolist(),
                "route_timestamps": evidence["query_times"].tolist(), "chosen_ref_ids": evidence["ref_ids"][indices].tolist(),
                "point_count": len(indices), "selections": selected, "query_pixels_sha256": result["query_pixels_sha256"]})
            write(output / f"pair_metrics_{variant}.json", result["records"][variant])
        write(output / "feature_events.json", result["feature_events"])
        write(output / "component_runtime.json", {"elapsed_s": time.perf_counter()-started,
            "unique_pair_count": result["unique_pairs"], "pair_records": {v: len(result["records"][v]) for v in PROFILE["variants"]},
            "correspondence_bytes": result["correspondence_bytes"], "chunks": result["chunks"],
            "headroom": result["memory"], "memory_end": check(output),
            "prior_original_control_exact": prior is not None, "source_contract": PROFILE["source_contract"],
            "panorama_reference_ids": sorted(k for k, v in cameras.items() if v in panorama.PROFILE["panorama_camera_types"]),
            "ground_truth_opened": False, "scoring_called": False, "policy_adopted": False})
        return {"status": "complete", "artifacts": artifact_hashes(output), "variants": list(PROFILE["variants"])}
    except Exception as error:
        write(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
            "ground_truth_opened": False, "scoring_called": False})
        raise


def source_identity(expected):
    require(Path(__file__).resolve() == ROOT / "scripts/probe_panorama_xfeat.py"
        and inputs.ROOT == original.ROOT == ROOT
        and Path(panorama.__file__).resolve() == ROOT / "monocular_osm/panorama_views.py"
        and Path(metadata.__file__).resolve() == ROOT / "monocular_osm/reference_metadata.py", "foreign diagnostic source")
    for module, name in ((inputs, "scripts/probe_orb_rerank.py"), (original, "scripts/probe_xfeat_rerank.py"),
                         (orb, "scripts/orb_rerank.py"), (xbackend, "scripts/xfeat_backend.py")):
        require(getattr(module, "__file__", None) and Path(module.__file__).resolve() == ROOT / name, "foreign original/backend helper source")
    for name, module in list(sys.modules.items()):
        if name.startswith("scripts.") or name == "monocular_osm" or name.startswith("monocular_osm."):
            require(getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT), "loaded module escaped complete source pin")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
    require(commit == expected and not dirty, "exact clean committed source required")
    return {name: sha(ROOT / name) for name in SOURCES}


def inspect_night(run, prior_run, prior_sha, metadata_freeze, metadata_sha):
    run = allowed(run).resolve(strict=True)
    verify_freeze(run / "frozen_prediction.json")  # all paths checked before old helper hashes them
    with np.load(allowed(run / "reference_evidence.npz"), allow_pickle=False) as archive:
        require(set(archive.files) == {"ref_ids", "ref_latlon", "query_indices", "query_times", "similarities", "metadata"}, "unexpected matrix members")
    registration, evidence, query = inputs.inspect_inputs(run)
    require((registration["n_queries"], registration["n_refs"]) == (80, 1498), "fixed complete Night control gallery required")
    prior = inspect_prior(prior_run, prior_sha, registration)
    meta, cameras = inspect_metadata(metadata_freeze, metadata_sha, evidence)
    return registration, evidence, query, prior, meta, cameras


def copied(source, target, expected):
    require(sha(source) == expected and not target.exists(), "copy source changed or target exists")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    require(sha(target) == expected, "frozen copy differs")


def copy_metadata(freeze_path, expected, output):
    frozen = verify_freeze(freeze_path, expected); root = Path(freeze_path).resolve().parent
    for name, h in frozen["artifacts"].items(): copied(root / relative(name), output / relative(name), h)
    copied(freeze_path, output / "acquisition_frozen.json", expected)


@contextmanager
def night_pixels(registration, evidence, staging):
    staging.mkdir(exist_ok=False); raw = None
    try:
        raw, record = stream_raw_store(registration["raw_store"]["path"], staging / "raw.npy",
            expected_sha256=registration["raw_store"]["sha256"],
            expected_ids=[str(v).split(":",1)[1] for v in evidence["ref_ids"]], allowed_temp_root=staging)
        yield raw, record
    finally:
        if raw is not None: raw._mmap.close()
        (staging / "raw.npy").unlink(missing_ok=True)


def compatible_backend(backend, prior):
    require(backend.state == prior["backend"]["actual_numerical_state"], "actual XFeat state differs from prior control")
    def identity(assets):
        return {k: v for k, v in assets.items() if k not in ("root", "weight")} | {
            "weight": {k: v for k, v in assets["weight"].items() if k != "path"}}
    require(identity(backend.assets) == identity(prior["backend"]["assets"]), "official XFeat assets differ from prior control")
    for name in ("torch_version", "cpu_threads", "interop_threads", "device", "weight_load", "lighterglue_constructed"):
        require(backend.runtime[name] == prior["backend"]["runtime"][name], "actual XFeat runtime differs from prior control")


def run_group(output, jobs, backend, *, check=resources):
    """One backend, both galleries and paths; no success freeze after one job."""
    require([j["name"] for j in jobs] == ["night", "municipal"], "fixed two-gallery group required")
    budget, completed = {"bytes": 0, "pairs": 0}, {}
    for job in jobs:
        completed[job["name"]] = run_component(output / job["name"], job["evidence"], job["query"],
            job["cameras"], job["raw"], backend, stream=job["stream"], prior=job.get("prior"), check=check,
            correspondence_budget=budget)
    return {"galleries": completed, "correspondence_budget_used": budget}


def inspect_group(args):
    require(args.profile_sha256 == PROFILE_SHA256, "explicit preregistered profile SHA required")
    sources = source_identity(args.source_commit)
    reg, evidence, query, prior, meta, cameras = inspect_night(args.input_run, args.prior_run,
        args.prior_freeze_sha256, args.metadata_freeze, args.metadata_freeze_sha256)
    assets = inspect_assets(args.assets)
    gallery = allowed(args.gallery_family).resolve(strict=True)
    verify_freeze(gallery / "inputs_frozen.json", args.gallery_input_freeze_sha256)
    with np.load(allowed(gallery / "inputs/full_evidence.npz"), allow_pickle=False) as z:
        membership = {name: z[name] for name in ("ref_ids", "ref_latlon")}
    gallery_meta, _ = inspect_metadata(args.gallery_metadata_freeze, args.gallery_metadata_freeze_sha256, membership)
    output = allowed(args.output).resolve()
    forbidden = [ROOT, Path(args.input_run).resolve(), Path(args.prior_run).resolve(), Path(args.assets).resolve(),
        gallery, Path(args.metadata_freeze).resolve().parent, Path(args.gallery_metadata_freeze).resolve().parent]
    require(output.drive.upper() == "E:" and not output.exists() and not any(output.is_relative_to(p) for p in forbidden), "new isolated owned E-drive output required")
    return {"sources": sources, "reg": reg, "evidence": evidence, "query": query, "prior": prior,
        "meta": meta, "cameras": cameras, "assets": assets, "gallery": gallery, "gallery_meta": gallery_meta, "output": output}


def prepare_output(args, job):
    output, assets, prior = job["output"], job["assets"], job["prior"]
    output.mkdir(parents=True)
    job["_owned_output"] = True
    for name, h in job["sources"].items(): copied(ROOT / name, output / "source_snapshot" / name, h)
    for name, item in assets["code"].items(): copied(Path(assets["root"]) / name, output / "upstream_source_snapshot" / name, item["sha256"])
    copied(assets["weight"]["path"], output / "weights/xfeat.pt", assets["weight"]["sha256"])
    copy_metadata(args.metadata_freeze, args.metadata_freeze_sha256, output / "inputs/metadata_night")
    copy_metadata(args.gallery_metadata_freeze, args.gallery_metadata_freeze_sha256, output / "inputs/metadata_municipal")
    for name in ("preregistration.json", "backend_identity.json", "pair_metrics.json", "diagnostic_prediction.json", "frozen_prediction.json"):
        copied(Path(prior["path"]) / name, output / "inputs/prior_xfeat" / name, sha(Path(prior["path"]) / name))
    write(output / "preregistration.json", {"profile": PROFILE, "profile_sha256": PROFILE_SHA256,
        "source_commit": args.source_commit, "source_sha256": job["sources"], "night": job["reg"], "night_metadata": job["meta"],
        "municipal_input_freeze": {"path": str(job["gallery"] / "inputs_frozen.json"), "sha256": args.gallery_input_freeze_sha256},
        "municipal_metadata": job["gallery_meta"], "prior_original_control_freeze_sha256": prior["freeze_sha256"], "assets": assets,
        "input_policy": {"ground_truth": False, "query_gps": False, "vo": False, "new_global_descriptor_forwards": False},
        "thumbnail_full_fov_verified": False,
        "source_geometry_status": "conditional complete-equirectangular assumption; original thumbnail dimensions were not retained"})


def final_inputs(args, job):
    reg = job["reg"]
    for name in ("frozen_prediction", "matrix", "recovery_lineage", "raw_store", "video"):
        require(sha(reg[name]["path"]) == reg[name]["sha256"], "original Night input changed")
    require(sha(Path(args.input_run) / "query_input.json") == reg["query_input_sha256"], "Night query registration changed")
    verify_freeze(reg["frozen_prediction"]["path"], reg["frozen_prediction"]["sha256"])
    verify_freeze(job["gallery"] / "inputs_frozen.json", args.gallery_input_freeze_sha256)
    verify_freeze(Path(args.prior_run) / "frozen_prediction.json", job["prior"]["freeze_sha256"])
    verify_freeze(args.metadata_freeze, args.metadata_freeze_sha256)
    verify_freeze(args.gallery_metadata_freeze, args.gallery_metadata_freeze_sha256)
    verify_freeze(job["output"] / "inputs_frozen.json")
    require(source_identity(args.source_commit) == job["sources"] and inspect_assets(args.assets) == job["assets"], "source or official assets changed")


def execute(args):
    job = inspect_group(args)
    output, reg, evidence, query, prior = [job[n] for n in ("output", "reg", "evidence", "query", "prior")]
    started = time.perf_counter()
    try:
        prepare_output(args, job)
        before = resources(output)
        # Deferred: Night-only unit tests need no adapter or prepared real gallery.
        from scripts.panorama_gallery_inputs import prepared_gallery_inputs
        require(source_identity(args.source_commit) == job["sources"], "deferred helper source differs before staging")
        with offline(), ExitStack() as stack:
            raw, staging = stack.enter_context(night_pixels(reg, evidence, output / "staging_night"))
            municipal = stack.enter_context(prepared_gallery_inputs(job["gallery"], output / "staging_municipal",
                expected_input_freeze_sha256=args.gallery_input_freeze_sha256))
            gm, gc = inspect_metadata(args.gallery_metadata_freeze, args.gallery_metadata_freeze_sha256, municipal["evidence"])
            require(gm == job["gallery_meta"] and len(municipal["evidence"]["similarities"]) == 80, "municipal membership or query grid changed")
            write(output / "staged_inputs.json", {"night": staging, "municipal": municipal["registration"]})
            members = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
                and not str(p.relative_to(output)).replace("\\", "/").startswith("staging_")}
            write(output / "inputs_frozen.json", {"models_have_not_run": True, "scoring_has_not_run": True, "artifacts": members})
            require(source_identity(args.source_commit) == job["sources"], "deferred helper source differs before model construction")
            before_model = resources(output)
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            backend = ReleasedXFeat(args.assets)
            compatible_backend(backend, prior)
            after_model = resources(output)
            old_threads = cv2.getNumThreads(); cv2.setNumThreads(2)
            opencv_threads = cv2.getNumThreads()
            try:
                require(opencv_threads == 2, "actual OpenCV thread count differs from paired control")
                completed = run_group(output, [
                    {"name": "night", "evidence": evidence, "query": query, "cameras": job["cameras"], "raw": raw,
                     "stream": inputs.stream_queries(reg["video"]["path"], query), "prior": prior},
                    {"name": "municipal", "evidence": municipal["evidence"], "query": municipal["query"], "cameras": gc,
                     "raw": municipal["raw"], "stream": municipal["stream"]}], backend)
                backend.verify_unchanged(); compatible_backend(backend, prior)
                write(output / "backend_identity.json", {"assets": backend.assets, "runtime": backend.runtime, "actual_numerical_state": backend.state})
                municipal["verify_inputs"]()
            finally: cv2.setNumThreads(old_threads)
        final_inputs(args, job)
        write(output / "group_runtime.json", {**completed, "elapsed_s": time.perf_counter()-started,
            "opencv_version": cv2.__version__, "opencv_threads_actual": opencv_threads,
            "memory_before": before, "memory_before_model": before_model, "memory_after_model": after_model,
            "memory_end": resources(output), "new_global_descriptor_forwards": False, "ground_truth_opened": False,
            "scoring_called": False, "selected_variant": None})
        members = artifact_hashes(output)
        require(sum((output / n).stat().st_size for n in members) <= PROFILE["max_final_output_bytes"], "final owned output disk bound exceeded")
        write(output / "frozen_predictions.json", {"status": "complete", "galleries": ["night", "municipal"],
            "variants": PROFILE["variants"], "profile_sha256": PROFILE_SHA256, "source_commit": args.source_commit,
            "ground_truth_opened": False, "scoring_has_not_run": True, "selected_variant": None, "policy_adopted": False,
            "all_four_outputs_complete": True, "artifacts": members})
        return {"status": "complete", "freeze_sha256": sha(output / "frozen_predictions.json")}
    except Exception as error:
        if job.get("_owned_output"):
            write(output / "failure.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                "ground_truth_opened": False, "scoring_called": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-run", "prior-run", "metadata-freeze", "gallery-family", "gallery-metadata-freeze", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prior-freeze-sha256", "metadata-freeze-sha256", "gallery-input-freeze-sha256",
                 "gallery-metadata-freeze-sha256", "source-commit", "profile-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__": main()
