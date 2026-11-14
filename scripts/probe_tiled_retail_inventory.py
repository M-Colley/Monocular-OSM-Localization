"""Metadata-only full-city tile fallback using existing frozen native OCR.

No video/OCR/VO/reference/model/GT work. The supplied city-wide public bbox,
prior failed inventory attempts and native detections must all hash-verify.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
sys.dont_write_bytecode = True


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024**2): digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".part.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_family(record):
    path = Path(record["path"]).resolve(strict=True)
    if sha(path) != record["sha256"]: raise ValueError("frozen prior family changed")
    frozen = read(path)
    if frozen.get("scoring_has_not_run") is not True: raise ValueError("prior family has no input-only freeze")
    anchors = {}
    for relative, expected in frozen["artifacts"].items():
        artifact = (path.parent / relative).resolve(strict=True)
        if not artifact.is_relative_to(path.parent) or sha(artifact) != expected:
            raise ValueError("a frozen prior-family input changed or escaped its root")
        anchors[str(artifact)] = expected
    return anchors


def verified_inputs(manifest):
    anchors = {}
    for record in manifest["prior_families"]: anchors.update(verify_family(record))
    for role in ("ocr", "bbox_response", "initial_attempt", "previous_summary"):
        record = manifest[role]
        if (sha(record["path"]) != record["sha256"]
                or anchors.get(str(Path(record["path"]).resolve())) != record["sha256"]):
            raise ValueError("metadata/OCR input changed or was not in a preserved prior freeze")
    city, initial = manifest["city"], read(manifest["initial_attempt"]["path"])
    previous = read(manifest["previous_summary"]["path"])
    failed = [record for record in previous["records"] if record.get("city") == city]
    if (initial.get("city") != city or initial.get("status") != "failed" or initial.get("ocr_status") != "complete"
            or len(failed) != 1 or failed[0].get("status") != "failed"):
        raise ValueError("fallback is only for city inventory failures after complete native OCR")
    ocr = read(manifest["ocr"]["path"])
    if (ocr.get("status") != "complete" or ocr.get("source") != "same_video_native_frames"
            or ocr.get("native_bbox_grid") is not True or ocr.get("sample_count") != 24
            or len(ocr.get("detections", [])) != initial.get("n_detections")):
        raise ValueError("preserved OCR is not the complete original native input")
    geocodes = read(manifest["bbox_response"]["path"])
    matches = [item for item in geocodes if item.get("osm_type") == manifest["bbox_response"]["osm_type"]
               and item.get("osm_id") == manifest["bbox_response"]["osm_id"]]
    if len(matches) != 1 or matches[0].get("class") != "boundary" or matches[0].get("type") != "administrative":
        raise ValueError("parent box must identify the frozen public named-city boundary")
    south, north, west, east = map(float, matches[0]["boundingbox"])
    return ocr, [west, south, east, north]


def infer(args):
    source = args.source_root.resolve(strict=True)
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if pin != args.source_commit or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("metadata probe requires its declared clean source pin")
    sys.path.insert(0, str(source))
    from monocular_osm import retail_inventory as retail
    from monocular_osm.poi_clues import build_poi_hypotheses
    if not Path(retail.__file__).resolve().is_relative_to(source): raise ValueError("fallback import escaped source pin")
    manifest = read(args.inputs)
    ocr, box = verified_inputs(manifest)
    plan = retail.plan_city_retail_tiles(manifest["city"], box, bbox_source={
        "city": manifest["city"], "scope": "named_city_bbox", "source_kind": "public_geocode",
        "source_sha256": manifest["bbox_response"]["sha256"], "osm_type": manifest["bbox_response"]["osm_type"],
        "osm_id": manifest["bbox_response"]["osm_id"]})
    output = args.output.resolve()
    if output.drive.upper() != "E:" or output.is_relative_to(source): raise ValueError("use a new isolated E metadata output")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    copies = {}
    for role in ("ocr", "bbox_response", "initial_attempt", "previous_summary"):
        target = output / "inputs" / f"{role}.json"
        shutil.copyfile(manifest[role]["path"], target)
        if sha(target) != manifest[role]["sha256"]: raise ValueError("frozen input copy changed")
        copies[role] = {"path": str(target), "sha256": sha(target)}
    sources = {}
    for name in ("retail_inventory.py", "poi_clues.py", "geo.py", "osm_gazetteer.py", "scene_text.py", "net.py"):
        target = output / "source_snapshot" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source / "monocular_osm" / name, target)
        sources[name] = sha(target)
    shutil.copyfile(__file__, output / "probe_source.py")
    write(output / "acquisition_plan.json", {"source_commit": pin, "source_sha256": sources,
        "runner_sha256": sha(__file__), "input_manifest": manifest, "input_manifest_sha256": sha(args.inputs),
        "copies": copies, "all_four_tile_plan": plan, "ground_truth_used": False,
        "new_ocr": False, "new_vo": False, "new_references": False, "new_models": False})
    result = {"city": manifest["city"], "status": "unknown", "selected_answer": None, "ground_truth_used": False}
    try:
        inventory = retail.fetch_tiled_city_retail(plan, output / "tiled_inventory")
        hypotheses = build_poi_hypotheses(ocr["detections"], inventory)
        write(output / "hypotheses.json", hypotheses)
        result.update(status="complete", n_pois=len(inventory["pois"]), n_detections=len(ocr["detections"]),
            matched_observations=len(hypotheses["matched_observations"]), branches=len(hypotheses["candidate_clusters"]),
            inventory_sha256=inventory["snapshot_sha256"], response_sha256=inventory["response_sha256"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write(output / "summary.json", result)
    write(output / "frozen_inventory_probe.json", {"scoring_has_not_run": True, "ground_truth_used": False,
        "artifacts": {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()},
        "torch_imported": "torch" in sys.modules})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    infer(parser.parse_args())
