"""Replay frozen raw matrices with one exact all-state profile; no models/GT."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import sys
import time
if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[key] = "2"
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PROJ_NETWORK="OFF", CUDA_VISIBLE_DEVICES="-1")
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from monocular_osm import full_sequence as full
from scripts.pilot_sparse_city_replay import (write, load, relative, require, sha, verify_freeze,
    copy_verified, verify_originals, source_identity, check_headroom, offline)
from scripts.plan_sparse_city_replay import header

SOURCES = ("scripts/replay_full_sequence.py", "scripts/pilot_sparse_city_replay.py", "scripts/plan_sparse_city_replay.py",
    "monocular_osm/full_sequence.py", "monocular_osm/sparse_sequence.py", "monocular_osm/geo.py", "monocular_osm/provenance.py")


def name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", value), "unsafe group/model name")
    return value


def specification(path):
    spec = load(path)
    require(set(spec) == {"groups"} and isinstance(spec["groups"], list) and 0 < len(spec["groups"]) <= 8, "bounded complete group list required")
    names, count = set(), 0
    for group in spec["groups"]:
        require(set(group) == {"name", "freeze", "freeze_sha256", "evidence", "models"}, "unexpected per-group configuration")
        group_name = name(group["name"]); require(group_name not in names, "duplicate group"); names.add(group_name)
        require(isinstance(group["models"], dict) and 0 < len(group["models"]) <= 4, "bounded model map required")
        require(len(set(group["models"].values())) == len(group["models"]), "duplicate raw matrix role")
        for model, key in group["models"].items():
            name(model); require(isinstance(key, str) and (key == "raw_cosines" or re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*_raw_cosines", key)), "raw-cosine role required")
        count += len(group["models"])
        require(count <= 8, "replay exceeds eight model matrices")
    return spec


def arrays(path, models):
    """Validate uncompressed headers before loading the requested arrays."""
    shapes = {}
    for key in ["ref_ids", "ref_latlon", "query_times", *models.values()]:
        h = header(path, key); dtype = np.dtype(h["dtype"]); size = 1
        for v in h["shape"]: size *= v
        if key == "ref_ids":
            require(dtype.kind in "US" and dtype.itemsize <= 512 and len(h["shape"]) == 1 and size <= 10000, "ID header exceeds budget")
        elif key == "ref_latlon":
            require(dtype.kind in "fiu" and h["shape"][-1:] == [2] and len(h["shape"]) == 2 and size <= 20000, "coordinate header exceeds budget")
        elif key == "query_times":
            require(dtype.kind in "fiu" and len(h["shape"]) == 1 and size <= 80, "time header exceeds budget")
        else:
            require(dtype.kind == "f" and dtype.itemsize in (4, 8) and len(h["shape"]) == 2
                and 2 <= h["shape"][0] <= 80 and 1 <= h["shape"][1] <= 10000 and size <= 800000, "matrix header exceeds budget")
        shapes[key] = h
    with np.load(path, allow_pickle=False) as z:
        shared = {key: z[key] for key in ("ref_ids", "ref_latlon", "query_times")}
        matrices = {model: z[key] for model, key in models.items()}
    for matrix in matrices.values(): full._inputs(matrix, shared["ref_ids"], shared["ref_latlon"], shared["query_times"])
    return shared, matrices, {"headers": shapes, "ref_ids_sha256": full._json_sha(shared["ref_ids"].astype(str).tolist()),
        "ref_latlon_sha256": full._array_sha(shared["ref_latlon"]), "query_times_sha256": full._array_sha(shared["query_times"]),
        "raw_matrix_sha256": {model: full._array_sha(matrix) for model, matrix in matrices.items()}}


def original_freezes(groups):
    for group in groups:
        path = Path(group["original_freeze"]).resolve(strict=True)
        require(sha(path) == group["original_freeze_sha256"], "source prediction freeze changed")
        frozen = verify_freeze(path)
        require(frozen.get("status") == "complete", "source prediction was incomplete")


def prepare(spec, family, expected_commit):
    source_identity(expected_commit)
    spec, family = Path(spec).resolve(strict=True), Path(family).resolve()
    declared = specification(spec)
    require(not family.exists() and not family.is_relative_to(ROOT), "new isolated family required")
    plans = []
    for group in declared["groups"]:
        freeze = Path(group["freeze"]).resolve(strict=True)
        require(sha(freeze) == group["freeze_sha256"], "source freeze differs from declared digest")
        frozen = verify_freeze(freeze)
        evidence = (freeze.parent / relative(group["evidence"])).resolve(strict=True)
        require(evidence.is_relative_to(freeze.parent) and group["evidence"] in frozen["artifacts"]
            and sha(evidence) == frozen["artifacts"][group["evidence"]] and frozen.get("status") == "complete", "matrix is not in complete prediction freeze")
        require(not family.is_relative_to(freeze.parent), "replay cannot write inside original family")
        _, _, identity = arrays(evidence, group["models"])
        plans.append({"name": group["name"], "models": group["models"], "input_identity": identity,
            "original_freeze": str(freeze), "original_freeze_sha256": group["freeze_sha256"],
            "original_evidence": str(evidence), "evidence_sha256": sha(evidence)})
    family.mkdir(parents=True)
    copied = []
    try: return prepare_copies(spec, family, expected_commit, plans, copied)
    except Exception as error:
        write(family / "preparation_failed.json", {"status": "failed", "scoring_has_not_run": True, "error": str(error), "copied": copied})
        raise


def prepare_copies(spec, family, expected_commit, plans, copied):
    copy_verified(spec, family / "inputs/spec.json", sha(spec), copied)
    for group in plans:
        prefix = family / "inputs" / group["name"]
        copy_verified(group["original_freeze"], prefix / "original_freeze.json", group["original_freeze_sha256"], copied)
        copy_verified(group["original_evidence"], prefix / "full_evidence.npz", group["evidence_sha256"], copied)
        group["evidence"] = str((prefix / "full_evidence.npz").resolve())
    original_freezes(plans); verify_originals(copied); source_identity(expected_commit)
    write(family / "input_lineage.json", {"copied": copied, "scoring_has_not_run": True})
    write(family / "plan.json", {"source_commit": expected_commit, "profile": dict(full.PROFILE), "profile_sha256": full.PROFILE_SHA256,
        "groups": plans, "selected_variant": None, "all_requested_outputs": [g["name"] + "/" + model + "/" + variant
            for g in plans for model in g["models"] for variant in full.PROFILE["variants"]]})
    artifacts = {str(Path(row["path"]).relative_to(family)): row["sha256"] for row in copied}
    artifacts.update({n: sha(family / n) for n in ("input_lineage.json", "plan.json")})
    write(family / "inputs_frozen.json", {"scoring_has_not_run": True, "models_have_not_run": True, "artifacts": artifacts})
    return {"status": "planned_only", "groups": len(plans), "outputs": sum(2*len(g["models"]) for g in plans), "profile_sha256": full.PROFILE_SHA256}


def profile_matches(plan):
    require(plan["profile_sha256"] == full.PROFILE_SHA256 and full._json_sha(plan["profile"]) == full.PROFILE_SHA256
            and plan["selected_variant"] is None, "full replay profile changed")
    requested = [g["name"] + "/" + model + "/" + variant for g in plan["groups"] for model in g["models"] for variant in full.PROFILE["variants"]]
    require(requested == plan["all_requested_outputs"], "requested group/model/variant list changed")


def replay_group(group, output, family, expected_commit):
    output.mkdir()
    before = check_headroom()
    from monocular_osm.provenance import snapshot_sources
    sources = snapshot_sources([ROOT / path for path in SOURCES], output)
    require(sha(group["evidence"]) == group["evidence_sha256"], "prepared matrix changed")
    shared, matrices, identity = arrays(group["evidence"], group["models"])
    require(identity == group["input_identity"], "complete matrix/grid/ID identity changed")
    stats, artifacts = {}, []
    for model, scores in matrices.items():
        started = time.monotonic()
        result = full.decode_full_tracks(scores, shared["ref_ids"], shared["ref_latlon"], shared["query_times"])
        write(output / ("predictions_" + model + ".json"), result)
        for variant in full.PROFILE["variants"]:
            name = "prediction_" + model + "_" + variant + ".json"
            write(output / name, result["outputs"][variant]); artifacts.append(name)
        stats[model] = {"elapsed_s": time.monotonic()-started, "allocation": result["allocation"],
            "source_matrix_key": group["models"][model], "raw_matrix_sha256": identity["raw_matrix_sha256"][model]}
        print(f"full sequence: {group['name']}/{model} complete", flush=True)
    source_identity(expected_commit); original_freezes([group])
    require(all(sha(ROOT / path) == digest for path, digest in sources.items()), "loaded source bytes changed")
    require(sha(group["evidence"]) == group["evidence_sha256"], "matrix changed during decoding")
    import psutil
    memory = psutil.Process().memory_info()
    write(output / "runtime.json", {"source_commit": expected_commit, "source_sha256": sources,
        "profile_sha256": full.PROFILE_SHA256, "input_freeze_sha256": sha(family / "inputs_frozen.json"),
        "group": group, "models": stats, "output_files": artifacts, "memory_before": before,
        "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss), "peak_private_bytes": getattr(memory, "peak_pagefile", memory.vms),
        "torch_imported": "torch" in sys.modules, "labels_opened": False, "new_forwards": False, "selected_variant": None})
    write(output / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
        "source_commit": expected_commit, "profile_sha256": full.PROFILE_SHA256, "selected_variant": None,
        "artifacts": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    return {"name": group["name"], "freeze_sha256": sha(output / "frozen_predictions.json"), "outputs": artifacts}


@offline()
def infer(family, expected_commit):
    family = Path(family).resolve(strict=True)
    require(Path(full.__file__).resolve() == ROOT / "monocular_osm/full_sequence.py", "decoder imported from another source")
    source_identity(expected_commit); verify_freeze(family / "inputs_frozen.json")
    plan, lineage = load(family / "plan.json"), load(family / "input_lineage.json")
    profile_matches(plan)
    require(plan["source_commit"] == expected_commit, "plan was registered under another source pin")
    declared = specification(family / "inputs/spec.json")["groups"]
    require(len(declared) == len(plan["groups"]) and all(
        g["name"] == s["name"] and g["models"] == s["models"] and g["original_freeze_sha256"] == s["freeze_sha256"]
        and Path(g["original_freeze"]).resolve() == Path(s["freeze"]).resolve()
        and Path(g["original_evidence"]).resolve() == (Path(s["freeze"]).parent / relative(s["evidence"])).resolve()
        for g, s in zip(plan["groups"], declared)), "plan omitted or changed declared source/model/group")
    verify_originals(lineage["copied"]); original_freezes(plan["groups"])
    output = family / "predictions"
    require(not output.exists(), "refuse to rerun or overwrite full replay")
    output.mkdir()
    completed, started = [], time.monotonic()
    try: return replay_all(plan, lineage, family, output, expected_commit, completed, started)
    except Exception as error:
        write(output / "failure.json", {"status": "failed", "scoring_has_not_run": True,
            "completed_groups": completed, "error": f"{type(error).__name__}: {error}"})
        raise


def replay_all(plan, lineage, family, output, expected_commit, completed, started):
    for group in plan["groups"]:
        completed.append(replay_group(group, output / group["name"], family, expected_commit))
    source_identity(expected_commit); verify_freeze(family / "inputs_frozen.json")
    original_freezes(plan["groups"]); verify_originals(lineage["copied"])
    write(output / "runtime.json", {"status": "complete", "groups": completed, "elapsed_s": time.monotonic()-started,
        "source_commit": expected_commit, "profile_sha256": full.PROFILE_SHA256,
        "all_requested_outputs": plan["all_requested_outputs"], "scoring_has_not_run": True,
        "new_forwards": False, "selected_variant": None, "torch_imported": "torch" in sys.modules})
    write(output / "frozen_predictions.json", {"status": "complete", "scoring_has_not_run": True,
        "profile_sha256": full.PROFILE_SHA256, "source_commit": expected_commit,
        "all_requested_outputs": plan["all_requested_outputs"], "selected_variant": None,
        "input_freeze_sha256": sha(family / "inputs_frozen.json"),
        "artifacts": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
    return {"status": "complete", "outputs": len(plan["all_requested_outputs"]), "freeze_sha256": sha(output / "frozen_predictions.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare"); setup.add_argument("--spec", type=Path, required=True)
    run = sub.add_parser("infer")
    for p in (setup, run):
        p.add_argument("--family", type=Path, required=True); p.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    print(__import__("json").dumps(prepare(**args) if command == "prepare" else infer(**args)))


if __name__ == "__main__": main()
