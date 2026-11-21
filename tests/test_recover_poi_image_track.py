"""Synthetic decoder recovery, immutable failed inputs and no model/label work."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from monocular_osm import poi_image_track as image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/recover_poi_image_track.py"
spec = importlib.util.spec_from_file_location("image_recovery_test", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
base = recovery.base


def _parent(tmp_path):
    parent = tmp_path / "failed"
    (parent / "inputs").mkdir(parents=True)
    video, inventory = parent / "inputs/video.mp4", parent / "inputs/inventory.json"
    video.write_bytes(b"same native pixels")
    inventory.write_text("public inventory")
    native = {"source": "same_video_native_frames", "native_bbox_grid": True, "sample_count": 24,
              "video_sha256": base.sha(video), "window": [0., 100.]}
    clues = {"selected_answer": None, "detections_sha256": "a" * 64,
        "inventory": {"city": "Synthetic City", "scope": "named_city_bbox", "snapshot_sha256": "c" * 64},
        "candidate_clusters": [{"pois": [{"id": "one", "lat": 0., "lon": 0.}], "conjunctions": []}]}
    plan = image.plan_single_branch(clues, native)
    old_profile = {key: value for key, value in image.PROFILE.items() if key != "reference_extent"}
    old_profile["version"] = "single-poi-image-track-v1"
    plan["profile"], plan["profile_sha256"] = old_profile, "d" * 64
    base.write(parent / "plan.json", plan)
    metadata = {"source": "mapillary", "center": plan["center"], "radius_m": 3000., "cap": 1500,
        "query_mode": "crops", "backbone": {"name": "megaloc", "hub": "fixed"},
        "preprocessing": {"descriptor_normalization": "L2"}}
    np.savez(parent / "reference_evidence.npz", query_indices=np.arange(3), query_times=np.arange(3.),
        ref_latlon=np.array([[0., 0.], [0., .001], [.001, .001], [.026, .026]]),
        ref_ids=np.array(["mapillary:a", "mapillary:b", "mapillary:c", "mapillary:corner"]),
        similarities=np.array([[.8,.1,.1,.9],[.1,.8,.1,.9],[.1,.1,.8,.9]],np.float32), metadata=np.array(base.json.dumps(metadata)))
    snapshot = parent / "source_snapshot"
    snapshot.mkdir()
    (snapshot / Path(base.__file__).name).write_bytes(Path(base.__file__).read_bytes())
    source_files = {}
    for name in ("poi_image_track.py", "geo.py", "metrics.py"):
        target = snapshot / "monocular_osm" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((ROOT / "monocular_osm" / name).read_bytes())
        source_files[name] = base.sha(target)
    registration = {"source_commit": "oldpin", "profile": old_profile, "profile_sha256": "d" * 64,
        "source_root": str(ROOT), "source_files": source_files, "model_assets": [],
        "runner_sha256": base.sha(base.__file__), "window": [0.,100.],
        **{key: {"path":str(path),"sha256":base.sha(path)} for key,path in (("video",video),("inventory",inventory))}}
    base.write(parent / "experiment.json", registration)
    for name in ("failed", "clues", "native_ocr", "resources"):
        base.write(parent / f"{name}.json", {})
    base.write(parent / "query_input.json", {"timestamps": [0.,1.,2.]})
    files = [{"relative":str(p.relative_to(parent)),"bytes":p.stat().st_size,"sha256":base.sha(p)}
             for p in parent.rglob("*") if p.is_file()]
    manifest = tmp_path / "failed_inputs.json"
    base.write(manifest, {"parent_run": str(parent), "ground_truth_opened": False,
        "status": "failed_after_forwards_before_decode", "files": files})
    return parent, manifest


def test_saved_matrix_recovery_retains_every_reference_and_preserves_parent(tmp_path, monkeypatch):
    parent, manifest_path = _parent(tmp_path)
    parent, manifest = recovery.verify_parent(manifest_path)
    def forbidden(*args, **kwargs): pytest.fail("decoder-only recovery attempted network/model work")
    monkeypatch.setattr("socket.create_connection", forbidden)
    args = SimpleNamespace(run=tmp_path / "recovery", failed_manifest=manifest_path, source_commit="newpin")
    recovery._recover_verified(args, ROOT, image, parent, manifest)
    prediction = base.read(args.run / "prediction.json")
    assert prediction["status"] == "complete" and prediction["provenance"]["reference_count"] == 4
    lineage = base.read(args.run / "recovery_lineage.json")
    assert not lineage["new_forwards"] and not lineage["new_downloads"] and not lineage["reference_filtering"]
    assert not lineage["ground_truth_opened"] and (args.run / "frozen_prediction.json").exists()
    base.verify_frozen(args.run)
    for record in manifest["files"]:
        assert base.sha(parent / record["relative"]) == record["sha256"]
    with pytest.raises(FileExistsError):
        recovery._recover_verified(args, ROOT, image, parent, manifest)


@pytest.mark.parametrize("target", ["reference_evidence.npz", "query_input.json", "inputs/video.mp4", "source_snapshot/monocular_osm/geo.py"])
def test_failed_inputs_must_verify_before_any_recovery(tmp_path, target):
    parent, manifest = _parent(tmp_path)
    (parent / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        recovery.verify_parent(manifest)


@pytest.mark.parametrize("key,value", [("cap", 1600), ("radius_m", 4000), ("query_mode", "warp"), ("transition_free_speed_mps", 50.)])
def test_contract_recovery_cannot_change_accuracy_constants(key, value):
    old = dict(image.PROFILE)
    old[key] = value
    with pytest.raises(ValueError, match="accuracy constants"):
        recovery.unchanged_accuracy(old, dict(image.PROFILE))
