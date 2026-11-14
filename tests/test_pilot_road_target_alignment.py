import io
import json
import zipfile
import numpy as np
import pytest
from scripts import pilot_road_target_alignment as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def freeze(root, filename, **extra):
    path = root/filename
    write(path, {**extra, "artifacts": {str(p.relative_to(root)): runner.sha(p)
        for p in root.rglob("*") if p.is_file() and p != path}})
    return runner.sha(path)


@pytest.fixture
def packages(tmp_path):
    pbf, motion, base = [tmp_path/name for name in ("pbf", "motion", "appearance")]
    evidence = base/"inference"
    for root in (pbf, motion, evidence): root.mkdir(parents=True)
    targets = {"scope": "named_city_admin_polygon_junction_nodes", "complete_parent_bbox_coverage": False,
        "selected_answer": None, "camera_fix": None, "matches": [{"observation": {
            "ref": "12B", "t_sec": .7, "confidence": .98}, "candidates": [
            {"id": "node/1", "lat": 10., "lon": 20.}, {"id": "node/2", "lat": 10.01, "lon": 20.01}]}]}
    write(pbf/"all_junction_targets.json", targets)
    pbf_sha = freeze(pbf, "frozen_pbf_probe.json")
    native_times = np.array([0., .24, .7, 1.4, 1.9])
    native_ordinals = np.arange(5, dtype=np.int64)
    pixels = np.array([str(i)*64 for i in range(5)])
    np.savez(motion/"native_motion.npz", xz=np.column_stack([np.arange(5.), np.zeros(5)]), valid=np.ones(5, bool),
        timestamps=native_times, frame_ordinals=native_ordinals, native_frame_pixels_sha256=pixels, K=np.eye(3))
    write(motion/"motion_registration.json", {"inputs": {"video_sha256": "video-identity", "pbf_freeze_sha256": pbf_sha,
        "window": [0., 2.]}, "stride": 1, "fps": 2., "legacy_cache_used": False, "ground_truth_used": False})
    write(motion/"motion_summary.json", {"status": "complete", "ground_truth_used": False,
        "timestamps_sha256": runner._array_sha(native_times), "frame_ordinals_sha256": runner._array_sha(native_ordinals),
        "native_frame_pixels_sha256": runner._array_sha(pixels), "motion_npz_sha256": runner.sha(motion/"native_motion.npz")})
    motion_sha = freeze(motion, "motion_frozen.json")
    query = [0, 2, 4]
    refs = np.array([[10., 20.], [10.01, 20.01], [10.02, 20.02]])
    ids = ["mapillary:a", "mapillary:b", "mapillary:c"]
    write(base/"inputs/replay_plan.json", {"query": {"video": {"sha256": "video-identity"}, "registered_window": [0., 2.],
        "stride": 1, "fps": 2., "planned_frame_ordinals": query},
        "union": {"ref_ids": ids, "count": len(ids), "ref_latlon_sha256": runner._array_sha(refs)}})
    inputs_sha = freeze(base, "inputs_frozen.json", scoring_has_not_run=True)
    times = native_times[query]
    np.savez(evidence/"full_evidence.npz", raw_cosines=np.full((3, 3), .2, np.float32), query_times=times,
        query_indices=np.arange(3), frame_ordinals=np.array(query), ref_ids=np.array(ids), ref_latlon=refs)
    write(evidence/"runtime_and_lineage.json", {"labels_opened": False, "scoring_called": False,
        "input_freeze_sha256": inputs_sha, "actual_query_times_sha256": runner._array_sha(times),
        "per_query_native_pixel_sha256": pixels[query].tolist(), "runtime": {"reference_encoding_identity": "unverified legacy"}})
    evidence_sha = freeze(evidence, "frozen_predictions.json", scoring_has_not_run=True)
    return pbf, pbf_sha, motion, motion_sha, evidence, evidence_sha


def test_frozen_ordinals_actual_vfr_times_and_pixels_bind_across_copy_paths(packages):
    native, full, targets, registration, runtime = runner.load_frozen_inputs(*packages)
    assert full["query_times"].tolist() == [0., .7, 1.9]
    assert full["frame_ordinals"].tolist() == [0, 2, 4]
    assert len(targets["matches"][0]["candidates"]) == 2
    assert not np.array_equal(np.rint(full["query_times"]*2), full["frame_ordinals"])


@pytest.mark.parametrize("fault", ["pixels", "times", "ordinals", "video", "window", "selected_target"])
def test_mismatched_or_selected_evidence_refused_without_labels(packages, fault):
    pbf, pbf_sha, motion, motion_sha, evidence, evidence_sha = packages
    if fault == "pixels":
        path = evidence/"runtime_and_lineage.json"; value = runner.read(path)
        value["per_query_native_pixel_sha256"][1] = "f"*64; write(path, value)
    elif fault in {"times", "ordinals"}:
        path = evidence/"full_evidence.npz"
        with np.load(path) as saved: value = {name: saved[name] for name in saved.files}
        if fault == "times":
            value["query_times"][1] += .01
            runtime_path = evidence/"runtime_and_lineage.json"; runtime = runner.read(runtime_path)
            runtime["actual_query_times_sha256"] = runner._array_sha(value["query_times"]); write(runtime_path, runtime)
        else: value["frame_ordinals"][1] = 1
        np.savez(path, **value)
    elif fault in {"video", "window"}:
        path = motion/"motion_registration.json"; value = runner.read(path)
        value["inputs"]["video_sha256" if fault == "video" else "window"] = "different"; write(path, value)
        motion_sha = freeze(motion, "motion_frozen.json")
    else:
        path = pbf/"all_junction_targets.json"; value = runner.read(path); value["camera_fix"] = [10., 20.]; write(path, value)
    if fault != "selected_target": evidence_sha = freeze(evidence, "frozen_predictions.json", scoring_has_not_run=True)
    with pytest.raises(ValueError):
        runner.load_frozen_inputs(pbf, pbf_sha, motion, motion_sha, evidence, evidence_sha)


def test_all_references_and_target_alternatives_determine_metric_bounds(packages):
    _, full, targets, _, _ = runner.load_frozen_inputs(*packages)
    frame, refs, observations, low, shape, cell = runner.metric_inputs(full, targets)
    assert len(refs) == 3 and len(observations[0]["candidates"]) == 2 and cell == 100.
    assert len(full["query_times"])*np.prod(shape) <= runner.MAX_FIELD_VALUES and np.all(refs >= low)


def test_field_spacing_doubles_only_to_fit_declared_budget(packages, monkeypatch):
    _, full, targets, _, _ = runner.load_frozen_inputs(*packages)
    monkeypatch.setattr(runner, "MAX_FIELD_VALUES", 3000)
    _, _, _, _, shape, cell = runner.metric_inputs(full, targets)
    assert cell > 100 and np.log2(cell/100).is_integer()
    assert len(full["query_times"])*np.prod(shape) <= 3000


def test_npz_header_budget_refuses_compressed_allocation_bomb_before_load(tmp_path):
    path = tmp_path/"oversized.npz"; header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)})
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scores.npy", header.getvalue())
    with pytest.raises(ValueError, match="allocation budget"):
        runner.bounded_npz(path, {"scores": (8, "f", 8)})


@pytest.mark.parametrize("failure", [False, True])
def test_all_declared_variants_write_candidates_before_caller_may_freeze(packages, tmp_path, monkeypatch, failure):
    import copy
    native, full, targets, _, _ = runner.load_frozen_inputs(*packages)
    frame, refs, observations, low, shape, cell = runner.metric_inputs(full, targets)
    output = tmp_path/"fit"; output.mkdir(); write(output/"fit_registration.json", {"variants": list(runner.VARIANTS)})
    calls = []
    monkeypatch.setattr(runner, "build_appearance_field", lambda *args, **kwargs: kwargs["reference_mean"])
    def alignment(*args, **kwargs):
        calls.append(kwargs["appearance_field"])
        assert np.array_equal(kwargs["query_times"], full["query_times"])
        if failure and len(calls) == 2:
            raise ValueError("mocked second-variant failure")
        candidate = {"route_xy": [[0., 0.], [100., 0.], [200., 0.]], "route_timestamps": full["query_times"].tolist(),
                     "objective": 1., "branch_id": "all-preserved"}
        return {"selected": copy.deepcopy(candidate), "selection_scope": "anchored diagnostic hypotheses only",
            "branches": [{"best": copy.deepcopy(candidate), "alternatives": [copy.deepcopy(candidate)]}],
            "unlocalized_alternative": {"route_xy": None}}
    monkeypatch.setattr(runner, "align_road_targets", alignment)
    args = (native, full, frame, refs, observations, low, shape, cell, "frozen-evidence", output)
    if failure:
        with pytest.raises(ValueError, match="second-variant"):
            runner.run_variants(*args)
        assert (output/"prediction_cue_only.json").exists() and not (output/"prediction_raw.json").exists()
    else:
        runner.run_variants(*args)
        assert calls == [None, False, True]
        for variant in runner.VARIANTS:
            result = runner.read(output/f"all_candidates_{variant}.json")
            assert result["unlocalized_alternative"]["route_xy"] is None
            assert len(result["branches"][0]["best"]["route_latlon"]) == 3
            assert runner.read(output/f"prediction_{variant}.json")["variant"] == variant
    assert not (output/"frozen_predictions.json").exists()
