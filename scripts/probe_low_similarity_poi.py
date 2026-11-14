"""Fresh native OCR/public-retail cues for input-only low-similarity selection.

No VO, reference imagery, Torch, predictions, labels or scoring are used.
Every baseline repeat is examined through a structural JSON allowlist. The
retained C2F pass stores its unrounded median cosine at decoded references;
rounded track values only check that binding or prove exclusion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_retrieval_health import _project

POLICY = {"cutoff": .162, "comparison": "strictly_less_than",
          "statistic": "median raw cosine at the decoded reference",
          "ocr_samples": 24, "ocr_min_len": 2, "ocr_min_confidence": .3,
          "native_pixels": True, "super_res": False, "ort_threads": [2, 1],
          "rec_batch": 4, "cls_batch": 4, "clue_config": "unmodified PoiClueConfig defaults"}
FIELDS = {"vpr_prior": {"track": True},
          "vpr_coarse_to_fine": {"kept": True, "pass1_sim": True, "pass2_sim": True},
          "provenance": {"config": {key: True for key in
              ("city", "video_path", "url", "data_dir", "vo_start_sec", "vo_end_sec")}}}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def project_inputs(raw):
    return _project(raw.decode("utf-8-sig"), 0, FIELDS)[0]


def select_from_inputs(value, cutoff=POLICY["cutoff"]):
    rows = (value.get("vpr_prior") or {}).get("track") or []
    if not rows or any(not isinstance(r, list) or len(r) != 4
                       or isinstance(r[3], bool) or not isinstance(r[3], (int, float))
                       or not math.isfinite(r[3]) or not -1 <= r[3] <= 1 for r in rows):
        raise ValueError("missing finite saved decoded-reference cosine vector")
    median = statistics.median(row[3] for row in rows)
    interval = [median - .00005 - 1e-12, median + .00005 + 1e-12]
    c2f = value.get("vpr_coarse_to_fine") or {}
    key = {"pass-1": "pass1_sim", "pass-2": "pass2_sim"}.get(c2f.get("kept"))
    exact = c2f.get(key) if key else None
    if exact is not None:
        if (isinstance(exact, bool) or not isinstance(exact, (int, float))
                or not math.isfinite(exact) or not interval[0] <= exact <= interval[1]):
            raise ValueError("retained-pass raw median is stale or inconsistent with saved track")
        return {"selected": exact < cutoff, "median_unrounded": exact,
                "median_source": "vpr_coarse_to_fine." + key,
                "rounded_track_median_interval": interval}
    if interval[0] >= cutoff:
        return {"selected": False, "median_unrounded": None,
                "median_source": "rounded track interval proves exclusion; raw median unavailable",
                "rounded_track_median_interval": interval}
    raise ValueError("raw median unavailable for a potentially eligible clip; refuse silent omission")


def checked_copy(source, target, expected_sha=None, expected_bytes=None):
    source, target = Path(source), Path(target)
    if not source.is_file() or target.exists():
        raise ValueError("copy requires an existing file and a new target")
    digest = sha(source)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("verified source hash changed")
    if expected_bytes is not None and source.stat().st_size != expected_bytes:
        raise ValueError("verified source size changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(target) != digest or sha(source) != digest:
        raise ValueError("copy changed or source was modified during the copy")
    return {"source": str(source.resolve()), "path": str(target.resolve()),
            "bytes": target.stat().st_size, "sha256": digest}


def validate_inventory(value, city):
    bbox = value.get("bbox_wsen") or []
    if (value.get("schema") != 1 or value.get("city") != city or value.get("scope") != "named_city_bbox"
            or len(bbox) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox)
            or not -180 <= bbox[0] < bbox[2] <= 180 or not -90 <= bbox[1] < bbox[3] <= 90
            or not isinstance(value.get("query"), str) or not value["query"]
            or not re.fullmatch(r"[0-9a-f]{64}", value.get("response_sha256", ""))
            or value.get("snapshot_sha256") != json_sha(value.get("pois"))):
        raise ValueError("public city inventory identity/scope/content is invalid")


@contextmanager
def offline_network():
    previous_connect, previous_create = socket.socket.connect, socket.create_connection
    def refuse(*args, **kwargs):
        raise RuntimeError("network is disabled during fresh native OCR")
    socket.socket.connect, socket.create_connection = refuse, refuse
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = previous_connect, previous_create


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("Torch is forbidden in the input-cue probe")
        return None


def pinned_source(source, commit):
    source = Path(source).resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if head != commit or dirty:
        raise ValueError("probe libraries must come from the declared clean immutable checkout")
    sys.path.insert(0, str(source))
    return source


def resolve_video(config, baseline_root):
    if config.get("video_path"):
        path = Path(config["video_path"])
        path = path if path.is_absolute() else baseline_root / path
    else:
        from monocular_osm.stages.acquire import pick_cached_input
        path = pick_cached_input(Path(config["data_dir"]))
    if path is None or not path.is_file():
        raise FileNotFoundError("the registered baseline video is not locally available")
    return path.resolve()


def fresh_ocr(video, start, end, output, model_dir):
    import numpy as np
    import psutil
    from monocular_osm import scene_text
    from monocular_osm.stages.poi_runtime import pinned_reader, reader_spec, validate_reader_record

    samples = []
    process = psutil.Process()
    def frames(path, first, last, interval):
        for timestamp, pixels in scene_text._sample_frames(path, first, last, interval, sample_count=24):
            if process.memory_info().rss > 2 * 1024**3:
                raise MemoryError("streaming cue OCR exceeded 2 GiB resident-memory limit")
            value = np.ascontiguousarray(pixels)
            digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
            digest.update(memoryview(value).cast("B"))
            samples.append({"t_sec": timestamp, "shape": list(value.shape), "pixels_sha256": digest.hexdigest()})
            print(f"  native OCR {len(samples)}/24", flush=True)
            yield timestamp, pixels

    spec = reader_spec("rapidocr", model_dir=model_dir)
    began = time.perf_counter()
    with offline_network(), pinned_reader(spec) as (reader, runtime):
        detections = scene_text.extract_scene_text(video, start_sec=start, end_sec=end,
            sample_count=24, min_len=2, min_confidence=.3, use_gpu=False, super_res=False,
            engine="rapidocr", frame_reader=frames, ocr_reader=reader, cache_path=None)
    validate_reader_record(runtime)
    if runtime.get("read_calls") != 24 or len(samples) != 24:
        raise ValueError("fresh native OCR did not process exactly 24 sampled frames")
    record = {"status": "complete", "source": "same_video_native_frames", "window": [start, end],
              "sample_count": 24, "sampled_frames": samples, "reader": runtime,
              "native_bbox_grid": True, "min_len": 2, "min_confidence": .3,
              "super_res": False, "use_gpu": False, "position_stamps_rejected": True,
              "detections": [asdict(x) for x in detections], "elapsed_s": time.perf_counter() - began,
              "runtime": {name: importlib.metadata.version(name) for name in
                          ("rapidocr", "onnxruntime", "opencv-python", "numpy")},
              "memory": {"rss_bytes": process.memory_info().rss,
                         "native_peak_rss_bytes": getattr(process.memory_info(), "peak_wset", None)}}
    write(output / "ocr.json", record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--asset-plan", type=Path, required=True)
    p.add_argument("--inventory", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offline-inventory", action="store_true")
    args = p.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PROJ_NETWORK="OFF")
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, NoTorch())
    if "torch" in sys.modules:
        raise RuntimeError("Torch was already imported")
    source = pinned_source(args.source_root, args.source_commit)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    copies = [checked_copy(path, output / "source_snapshot" / path.relative_to(source))
              for path in sorted((source / "monocular_osm").rglob("*.py"))]
    copies += [checked_copy(Path(__file__), output / "source_snapshot/probe_low_similarity_poi.py"),
               checked_copy(ROOT / "scripts/probe_retrieval_health.py", output / "source_snapshot/probe_retrieval_health.py")]
    assets = read(args.asset_plan)["rapidocr"]
    asset_root = Path(assets["source"]).resolve(strict=True)
    for asset in assets["files"]:
        original = (asset_root / asset["relative"]).resolve(strict=True)
        target = (output / "runtime_packages/rapidocr" / asset["relative"]).resolve()
        if not original.is_relative_to(asset_root) or not target.is_relative_to(output):
            raise ValueError("OCR asset plan escaped its declared root")
        copies.append(checked_copy(original, target, asset["sha256"], asset["bytes"]))
    sys.path.insert(0, str(output / "runtime_packages"))
    plan = {"experimental": True, "scoring_has_not_run": True, "policy": POLICY,
            "source_root": str(source), "source_commit": args.source_commit,
            "asset_plan_sha256": sha(args.asset_plan), "copies": copies, "clips": []}
    for path in sorted(args.baseline_run.glob("*/rep0/result.json")):
        raw = path.read_bytes(); projected = project_inputs(raw)
        selection = select_from_inputs(projected)
        cfg = projected["provenance"]["config"]
        entry = {"clip": path.parent.parent.name, "source_result": str(path.resolve()),
                 "source_result_sha256": hashlib.sha256(raw).hexdigest(),
                 "projected_inputs_sha256": json_sha(projected), "selection": selection,
                 "city": cfg["city"], "window": [cfg["vo_start_sec"], cfg["vo_end_sec"]]}
        if selection["selected"]:
            video = resolve_video(cfg, args.baseline_root.resolve())
            entry["video"] = checked_copy(video, output / "videos" / (entry["clip"] + video.suffix))
        plan["clips"].append(entry)
    if len(plan["clips"]) != 15:
        raise ValueError("the requested complete baseline must contain all 15 repeat-zero inputs")
    inventory_dir = output / "inventories"; inventory_dir.mkdir()
    for original in args.inventory:
        value = read(original); validate_inventory(value, value["city"])
        copies.append(checked_copy(original, inventory_dir / original.name))
    write(output / "input_plan.json", plan)
    from monocular_osm.poi_clues import fetch_city_retail, build_poi_hypotheses
    import cv2
    cv2.setNumThreads(1)
    records = []
    for entry in plan["clips"]:
        if not entry["selection"]["selected"]:
            continue
        clip_out = output / entry["clip"]; clip_out.mkdir()
        record = {"clip": entry["clip"], "city": entry["city"], "selection": entry["selection"],
                  "window": entry["window"], "video": entry["video"], "scoring_has_not_run": True}
        began = time.perf_counter()
        try:
            print(f"{entry['clip']}: fresh native cues, {entry['city']}", flush=True)
            ocr = fresh_ocr(Path(entry["video"]["path"]), *entry["window"], clip_out,
                            output / "runtime_packages/rapidocr/models")
            record.update(ocr_status="complete", n_detections=len(ocr["detections"]))
            import osmnx as ox
            ox.settings.cache_folder = str(output / "public_http_cache")
            ox.settings.requests_timeout = 60
            inventory = fetch_city_retail(entry["city"], inventory_dir, offline=args.offline_inventory)
            validate_inventory(inventory, entry["city"])
            inventory_path = inventory_dir / (re.sub(r"[^a-z0-9]+", "-", entry["city"].casefold()).strip("-") + ".json")
            record["inventory"] = {"path": str(inventory_path), "file_sha256": sha(inventory_path),
                                   "snapshot_sha256": inventory["snapshot_sha256"],
                                   "response_sha256": inventory["response_sha256"], "n_pois": len(inventory["pois"])}
            hypotheses = build_poi_hypotheses(ocr["detections"], inventory)
            write(clip_out / "hypotheses.json", hypotheses)
            record.update(status="complete", matched_observations=len(hypotheses["matched_observations"]),
                          branch_count=len(hypotheses["candidate_clusters"]), selected_answer=hypotheses["selected_answer"])
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["elapsed_s"] = time.perf_counter() - began
        write(clip_out / "attempt.json", record)
        write(clip_out / "frozen_cues.json", {"scoring_has_not_run": True, "status": record["status"],
            "artifacts": {str(f.relative_to(output)): sha(f) for f in clip_out.iterdir() if f.is_file()}})
        records.append(record)
        print(json.dumps({k: record.get(k) for k in ["clip", "status", "n_detections", "branch_count", "elapsed_s", "error"]}), flush=True)
        write(output / "progress.json", {"attempts": records, "scoring_has_not_run": True})
    if "torch" in sys.modules:
        raise RuntimeError("Torch was unexpectedly loaded")
    assert all(sha(record["source"]) == record["sha256"] for record in copies)
    pinned_source(source, args.source_commit)
    summary = {"purpose": "input-only native POI cue inventory; no VO/references/GT or accuracy claim",
               "scoring_has_not_run": True, "input_plan_sha256": sha(output / "input_plan.json"),
               "source_commit": args.source_commit, "policy": POLICY, "n_baseline_clips": len(plan["clips"]),
               "n_selected": len(records), "attempts": records, "torch_imported": False,
               "elapsed_s": time.perf_counter() - started, "frozen_utc": datetime.now(timezone.utc).isoformat()}
    write(output / "summary.json", summary)
    write(output / "frozen_probe.json", {"scoring_has_not_run": True, "input_plan_sha256": sha(output / "input_plan.json"),
          "artifacts": {str(f.relative_to(output)): sha(f) for f in sorted(output.rglob("*"))
                        if f.is_file() and f.name != "frozen_probe.json"}})
    print(json.dumps({"output": str(output), "n_selected": len(records), "freeze_sha256": sha(output / "frozen_probe.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
