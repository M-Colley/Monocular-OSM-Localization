"""Decode a hash-frozen failed image pilot after an acquisition-contract fix.

No forwards, downloads, reference filtering, new OCR or VO. Copy and verify
the immutable input family into a NEW output; freeze prediction before score.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import shutil
import time

spec = importlib.util.spec_from_file_location("image_pilot_recovery_base", Path(__file__).with_name("pilot_poi_image_track.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
np = base.np


def verify_parent(manifest_path):
    manifest = base.read(manifest_path)
    parent = Path(manifest["parent_run"]).resolve(strict=True)
    if (manifest.get("ground_truth_opened") is not False
            or manifest.get("status") != "failed_after_forwards_before_decode"):
        raise ValueError("recovery requires preserved label-free failed forward inputs")
    records = manifest.get("files", [])
    required = {"experiment.json", "query_input.json", "plan.json", "reference_evidence.npz", "failed.json"}
    if not required.issubset({record["relative"] for record in records}):
        raise ValueError("failed-input preservation manifest is incomplete")
    seen = set()
    for record in records:
        path = (parent / record["relative"]).resolve(strict=True)
        if (record["relative"] in seen or not path.is_relative_to(parent)
                or path.stat().st_size != record["bytes"] or base.sha(path) != record["sha256"]):
            raise ValueError("preserved failed input changed or escaped its family")
        seen.add(record["relative"])
    if seen != {str(path.relative_to(parent)) for path in parent.rglob("*") if path.is_file()}:
        raise ValueError("failed-input preservation manifest must cover every source file")
    return parent, manifest


def unchanged_accuracy(old, new):
    """Only extent-contract declaration/version can differ from the first run."""
    ignored = {"version", "reference_extent"}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        raise ValueError("decoder recovery cannot change registered accuracy constants")


def recover(args):
    source, image = base.pinned_source(args)
    parent, manifest = verify_parent(args.failed_manifest)
    args.run = args.run.resolve()
    if args.run.drive.upper() != "E:" or args.run.is_relative_to(parent) or args.run.is_relative_to(source):
        raise ValueError("decoder recovery needs a separate fresh E output")
    if args.run.exists(): raise FileExistsError("decoder recovery output must be entirely new")
    return _recover_verified(args, source, image, parent, manifest)


def _recover_verified(args, source, image, parent, manifest):
    old = base.read(parent / "experiment.json")
    unchanged_accuracy(old["profile"], dict(image.PROFILE))
    started = time.perf_counter()
    shutil.copytree(parent, args.run)
    for record in manifest["files"]:
        if base.sha(args.run / record["relative"]) != record["sha256"]:
            raise ValueError("recovery input copy changed")
    shutil.copyfile(args.failed_manifest, args.run / "parent_failed_inputs_frozen.json")
    shutil.copyfile(__file__, args.run / "decoder_recovery_source.py")
    for name in ("experiment.json", "plan.json", "failed.json"):
        (args.run / name).rename(args.run / ("original_" + name))
    registration = dict(old)
    registration.update(source_commit=args.source_commit, source_root=str(source),
        profile=dict(image.PROFILE), profile_sha256=image.PROFILE_SHA256,
        runner_sha256=base.sha(Path(base.__file__)), recovery_runner_sha256=base.sha(__file__))
    registration["execution_source_roles"] = {
        "completed_forwards": {"source_commit": old["source_commit"],
            "source_files": dict(old["source_files"]), "runner_sha256": old["runner_sha256"],
            "preserved_parent_manifest_sha256": base.sha(args.failed_manifest)},
        "decoder_only_recovery": {"source_commit": args.source_commit,
            "new_forwards": False, "new_downloads": False,
            "core_sha256": base.sha(source / "monocular_osm/poi_image_track.py")}}
    for name in ("video", "inventory"):
        registration[name] = {**old[name], "path": str(args.run / "inputs" / Path(old[name]["path"]).name)}
    for name in registration["source_files"]:
        path = args.run / "source_snapshot/monocular_osm" / name
        shutil.copyfile(source / "monocular_osm" / name, path)
        registration["source_files"][name] = base.sha(path)
    shutil.copyfile(base.__file__, args.run / "source_snapshot" / Path(base.__file__).name)
    base.write(args.run / "experiment.json", registration)
    base.write(args.run / "recovery_lineage.json", {
        "parent": str(parent), "parent_manifest_sha256": base.sha(args.failed_manifest),
        "original_source_commit": old["source_commit"], "new_source_commit": args.source_commit,
        "original_profile_sha256": old["profile_sha256"], "new_profile_sha256": image.PROFILE_SHA256,
        "correction": "validate the existing provider geodesic-disc bounding box, retaining every reference",
        "new_forwards": False, "new_downloads": False, "reference_filtering": False,
        "ground_truth_opened": False, "recovery_runner_sha256": base.sha(__file__)})
    plan_old = base.read(args.run / "original_plan.json")
    plan = image.plan_single_branch(plan_old["clues"], plan_old["ocr_provenance"])
    base.write(args.run / "plan.json", plan)
    with np.load(args.run / "reference_evidence.npz", allow_pickle=False) as data:
        evidence = {key: data[key].copy() for key in ("query_indices", "query_times", "ref_latlon", "ref_ids", "similarities")}
        evidence["provenance"] = base.json.loads(str(data["metadata"]))
    base.validate_query_identity(evidence, base.read(args.run / "query_input.json")["timestamps"])
    prediction = image.decode_single_branch_image_track(evidence, plan, query_video_sha256=registration["video"]["sha256"])
    base.write(args.run / "prediction.json", prediction)
    base.write(args.run / "recovery_runtime.json", {"seconds": time.perf_counter() - started,
        "reference_count": len(evidence["ref_ids"]), "query_count": len(evidence["query_times"]),
        "new_forwards": False, "ground_truth_opened": False})
    base.freeze_outputs(args.run)
    base.verify_frozen(args.run)
    print(f"Decoder-only recovery frozen: {prediction['point_count']} queries, {len(evidence['ref_ids'])} references; no labels opened")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inf = commands.add_parser("recover")
    inf.add_argument("--source-root", type=Path, required=True)
    inf.add_argument("--source-commit", required=True)
    inf.add_argument("--failed-manifest", type=Path, required=True)
    inf.add_argument("--run", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args(argv)
    (recover if args.command == "recover" else base.score)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
