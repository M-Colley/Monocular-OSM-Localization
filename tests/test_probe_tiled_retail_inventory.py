"""No-network frozen-input barriers for city metadata recovery."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_tiled_retail_inventory.py"
spec = importlib.util.spec_from_file_location("tiled_retail_probe_test", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _inputs(tmp_path):
    original, recovery = tmp_path / "original", tmp_path / "recovery"
    original.mkdir(); recovery.mkdir()
    objects = {"ocr": {"status": "complete", "source": "same_video_native_frames", "native_bbox_grid": True,
        "sample_count": 24, "detections": [{"text": "Aster"}]},
        "initial_attempt": {"city": "Synthetic City", "status": "failed", "ocr_status": "complete", "n_detections": 1},
        "bbox_response": [{"osm_type": "relation", "osm_id": 12, "class": "boundary", "type": "administrative",
                           "boundingbox": ["2", "2.4", "1", "1.2"]}]}
    manifest = {"city": "Synthetic City", "prior_families": []}
    for role, value in objects.items():
        path = original / f"{role}.json"
        probe.write(path, value)
        manifest[role] = {"path": str(path), "sha256": probe.sha(path)}
    manifest["bbox_response"].update(osm_type="relation", osm_id=12)
    summary = recovery / "previous_summary.json"
    probe.write(summary, {"records": [{"city": "Synthetic City", "status": "failed"}]})
    manifest["previous_summary"] = {"path": str(summary), "sha256": probe.sha(summary)}
    for directory in (original, recovery):
        frozen = directory / "freeze.json"
        probe.write(frozen, {"scoring_has_not_run": True,
            "artifacts": {p.name: probe.sha(p) for p in directory.iterdir() if p.is_file()}})
        manifest["prior_families"].append({"path": str(frozen), "sha256": probe.sha(frozen)})
    return manifest


def test_metadata_probe_reuses_exact_frozen_ocr_and_full_public_city_box(tmp_path):
    inputs = _inputs(tmp_path)
    ocr, box = probe.verified_inputs(inputs)
    assert box == [1.,2.,1.2,2.4] and ocr["detections"] == [{"text": "Aster"}]


@pytest.mark.parametrize("role", ["ocr", "bbox_response", "initial_attempt", "previous_summary"])
def test_changed_original_inputs_refuse_before_any_request(tmp_path, role):
    inputs = _inputs(tmp_path)
    Path(inputs[role]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        probe.verified_inputs(inputs)


def test_new_replacement_ocr_cannot_be_blessed_by_merely_updating_its_hash(tmp_path):
    inputs = _inputs(tmp_path)
    fresh = tmp_path / "unfrozen_ocr.json"
    probe.write(fresh, {"status": "complete", "detections": []})
    inputs["ocr"] = {"path": str(fresh), "sha256": probe.sha(fresh)}
    with pytest.raises(ValueError, match="prior freeze"):
        probe.verified_inputs(inputs)
