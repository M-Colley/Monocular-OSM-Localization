"""Prepare/extract a whole public OSM road-geometry index; no route inputs.

Actual extraction is separate from preparation and requires the accepted input
freeze and a clean source pin. This program never performs route projection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from monocular_osm import road_geometry as core
from monocular_osm.pbf_junction_inventory import parse_extract_poly


def write(path, value):
    require = core.require
    require(not Path(path).exists(), "artifact already exists")
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def source_identity(expected):
    core.require(Path(__file__).resolve() == ROOT/"scripts/extract_road_geometry.py", "foreign executed driver")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    core.require(head == expected and len(expected) == 40, "source pin differs")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    core.require(not status.strip(), "tracked source must be clean")
    result = {"commit": head, "files": {"scripts/extract_road_geometry.py": core.file_sha(__file__)}}
    for name, module in list(sys.modules.items()):
        if name.startswith("monocular_osm") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            core.require(path.is_relative_to(ROOT), "foreign project module")
            result["files"][path.relative_to(ROOT).as_posix()] = core.file_sha(path)
    return result


def outside_inputs(output, inputs):
    target = core.safe_path(output).resolve()
    core.require(not target.is_relative_to(ROOT), "output must be outside source checkout")
    for path in inputs:
        source = core.safe_path(path).resolve(strict=True)
        core.require(not target.is_relative_to(source.parent) and not source.is_relative_to(target),
            "output aliases an original input family")
    return target


def prepare(pbf, pbf_sha256, polygon, polygon_sha256, ledger, family, expected_commit):
    source = source_identity(expected_commit)
    pbf, polygon, ledger = [core.safe_path(p).resolve(strict=True) for p in (pbf, polygon, ledger)]
    target = outside_inputs(family, (pbf, polygon, ledger))
    core.require(not target.exists(), "family must be new")
    core.require(pbf.suffix.lower() in {".pbf", ".osm"} and pbf.stat().st_size <= core.PROFILE["max_source_bytes"], "source type/size bound refused")
    core.require(polygon.stat().st_size <= 2*1024**2 and ledger.stat().st_size <= 1024**2,
        "public polygon/ledger size bound exceeded")
    hashes = {str(p): core.file_sha(p) for p in (pbf, polygon, ledger)}
    core.require(hashes[str(pbf)] == pbf_sha256 and hashes[str(polygon)] == polygon_sha256, "public source/polygon hash differs")
    identity = json.loads(ledger.read_text(encoding="utf-8"))
    core.require(identity.get("verified") is True and identity.get("sha256") == pbf_sha256
        and identity.get("bytes") == pbf.stat().st_size and identity.get("license")
        and identity.get("source_url"), "verified public source ledger required")
    footprint = parse_extract_poly(polygon.read_bytes())
    reader = core.GdalWays()
    target.mkdir(parents=True)
    temporary = target/"temporary"; temporary.mkdir()
    guard = core.ResourceGuard(target, temporary); guard.check(force=True)
    config = target/"osmconf.ini"; config.write_text(core.OSM_CONFIG, encoding="utf-8", newline="\n")
    write(target/"profile.json", {"profile": core.PROFILE, "profile_sha256": core.PROFILE_SHA256})
    for role, path in (("extract_ledger.json", ledger), ("extract.poly", polygon)):
        (target/role).write_bytes(path.read_bytes())
    registration = {"schema": 1, "scope": "whole_public_extract_physical_way_geometry",
        "source": source, "source_inputs": hashes, "pbf": str(pbf), "polygon": str(polygon),
        "extract_source": identity, "footprint": footprint, "reader": reader.identity,
        "profile_sha256": core.PROFILE_SHA256, "osm_config_sha256": core.file_sha(config),
        "bbox": None, "routes_labels_or_query_data_used": False,
        "footprint_completeness_limit": "extract selection polygon; missing source OSM data cannot be inferred; no coverage beyond it",
        "prepared_memory": guard.record()}
    write(target/"registration.json", registration)
    artifacts = {p.name: core.file_sha(p) for p in target.iterdir() if p.is_file()}
    core.require(source_identity(expected_commit) == source and all(core.file_sha(p) == h for p, h in hashes.items()), "source/input changed during prepare")
    frozen = {"status": "prepared", "artifacts": artifacts, "source_commit": expected_commit,
        "profile_sha256": core.PROFILE_SHA256, "registration_sha256": core.file_sha(target/"registration.json"),
        "roads_extracted": False, "projection_run": False, "ground_truth_read": False}
    write(target/"inputs_frozen.json", frozen)
    return frozen


def verified_prepared(family, expected_commit, expected_input_sha256):
    family = core.safe_path(family).resolve(strict=True)
    path = family/"inputs_frozen.json"
    core.require(core.file_sha(path) == expected_input_sha256, "accepted input freeze differs")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    core.require(frozen.get("status") == "prepared" and frozen.get("source_commit") == expected_commit
        and frozen.get("profile_sha256") == core.PROFILE_SHA256, "input freeze profile/source differs")
    core.require(set(frozen["artifacts"]) == {"registration.json", "profile.json", "osmconf.ini", "extract_ledger.json", "extract.poly"},
        "exact prepared artifact membership required")
    for name, expected in frozen["artifacts"].items():
        p = core.safe_path(family/name).resolve(strict=True)
        core.require(p.is_relative_to(family) and core.file_sha(p) == expected, "prepared artifact changed/escaped")
    registration = json.loads((family/"registration.json").read_text(encoding="utf-8"))
    core.require(registration["source"] == source_identity(expected_commit), "prepared source bytes differ")
    for p, expected in registration["source_inputs"].items():
        core.require(core.file_sha(p) == expected, "original input changed")
    core.require(registration["profile_sha256"] == core.PROFILE_SHA256
        and registration["osm_config_sha256"] == hashlib.sha256(core.OSM_CONFIG.encode()).hexdigest()
        and registration["bbox"] is None, "whole-extract parser profile differs")
    for name, expected in frozen["artifacts"].items():
        core.require(core.file_sha(family/name) == expected, "prepared artifact changed during verification")
    core.require(core.file_sha(path) == expected_input_sha256, "accepted input freeze changed during verification")
    return family, registration


def extract(family, expected_commit, expected_input_sha256):
    family, registration = verified_prepared(family, expected_commit, expected_input_sha256)
    for name in ("roads.sqlite", "source_ways.sqlite", "summary.json", "frozen_road_geometry.json", "failure.json"):
        core.require(not (family/name).exists(), "extraction output exists; no implicit resume/overwrite")
    reader = core.GdalWays()
    core.require(reader.identity == registration["reader"], "GDAL runtime changed")
    guard = core.ResourceGuard(family, family/"temporary")
    guard.check(force=True)
    write(family/"execution.json", {"accepted_inputs_sha256": expected_input_sha256,
        "source": registration["source"], "profile_sha256": core.PROFILE_SHA256,
        "argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    try:
        source_counts = core.source_way_ledger(registration["pbf"], family/"source_ways.sqlite", guard)
        rows = reader.rows(registration["pbf"], family/"osmconf.ini", family/"temporary", guard)
        summary = core.build_index(rows, family/"roads.sqlite", guard, source_ledger=family/"source_ways.sqlite")
        core.require(summary["source_way_counts_and_tags_verified"], "source-way assembly guard absent")
        summary.update(reader=reader.identity, driver_stats=dict(reader.stats), profile_sha256=core.PROFILE_SHA256,
            source_sha256=registration["source_inputs"], runtime=guard.record(), projection_run=False,
            source_way_ledger=source_counts, complete_within_predicate_and_source=True, public_access_or_connected_route_claim=False)
        guard.check(force=True)
        verified_prepared(family, expected_commit, expected_input_sha256)
        write(family/"summary.json", summary)
        artifacts = {name: core.file_sha(family/name) for name in ("roads.sqlite", "source_ways.sqlite", "summary.json", "execution.json", "inputs_frozen.json")}
        core.require(source_identity(expected_commit) == registration["source"], "source changed before complete freeze")
        guard.check(force=True)
        # The last resource callback and large output hashes may take time.
        # Rebind all outputs, then originals and the EXTERNAL input SHA last;
        # a rewritten/reblessed registration must not authorize new inputs.
        for name, expected in artifacts.items():
            core.require(core.file_sha(family/name) == expected, "output artifact changed before complete freeze")
        verified_prepared(family, expected_commit, expected_input_sha256)
        core.require(core.file_sha(family/"inputs_frozen.json") == expected_input_sha256,
            "external accepted input freeze changed before completion")
        core.require(source_identity(expected_commit) == registration["source"], "source changed at completion")
        frozen = {"status": "complete", "artifacts": artifacts, "source_commit": expected_commit,
            "profile_sha256": core.PROFILE_SHA256, "accepted_inputs_sha256": expected_input_sha256,
            "scope": "whole_extract_physical_road_way_geometry", "bbox": None,
            "topology": False, "projection_run": False, "ground_truth_read": False,
            "new_models_API_downloads": False, "runtime_end": guard.record()}
        write(family/"frozen_road_geometry.json", frozen)
        return frozen
    except Exception as error:
        write(family/"failure.json", {"status": "failed", "error": type(error).__name__+": "+str(error),
            "partial_artifacts_retained": True, "projection_run": False, "ground_truth_read": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    for name in ("pbf", "pbf-sha256", "polygon", "polygon-sha256", "ledger", "family", "expected-commit"):
        p.add_argument("--"+name, required=True)
    p = sub.add_parser("extract")
    for name in ("family", "expected-commit", "input-freeze-sha256"): p.add_argument("--"+name, required=True)
    args = vars(parser.parse_args(argv)); action = args.pop("action")
    result = prepare(**args) if action == "prepare" else extract(args["family"], args["expected_commit"], args["input_freeze_sha256"])
    print(json.dumps({"status": result["status"], "profile_sha256": core.PROFILE_SHA256}))


if __name__ == "__main__":
    main()
