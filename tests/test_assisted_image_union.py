from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import assisted_image_union as adapter, assisted_worker as worker
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.provenance import file_sha256
from monocular_osm.stages import acquire, poi_only


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = replace(PipelineConfig(url="https://example.test/video", data_dir=tmp_path / "data", output_dir=tmp_path / "out", city="Example"),
        answer="poi", vpr_query_mode="crops", poi_radius_m=3000., poi_ref_cap=1500,
        vo_start_sec=0., vo_end_sec=10., frame_stride=3)
    cfg.output_dir.mkdir()
    times = np.arange(100, dtype=np.float64)/10.
    closed, calls = [], []
    frames = [np.zeros((4, 8, 3), np.uint8) for _ in times]
    bundle = SimpleNamespace(frames=frames, timestamps=times, fps=10., close=lambda: closed.append(True))
    cue = {"window": [0., 10.], "clues": {}, "inventory": {"snapshot_sha256": "a"*64},
        "ocr_provenance": {"frame_shape_hw": [4, 8]}, "auto_quality": {
            "window": [0., 10.], "stride": 3, "fps": 10., "n_frames": 100,
            "timestamps_sha256": worker._array_sha(times)}}
    monkeypatch.setattr(worker, "_load_cues", lambda request: cue)
    monkeypatch.setattr(acquire, "extract_video_frames", lambda ctx: setattr(ctx, "frames", bundle))
    plan = {"status": "ready", "branch_count": 2, "region_plans": [
        {"center": [50., 8.], "radius_m": 3000, "cap": 1500},
        {"center": [50.1, 8.1], "radius_m": 3000, "cap": 1500}]}
    monkeypatch.setattr(adapter, "plan_image_union", lambda clues, native: deepcopy(plan))
    caches = []
    def retrieval(ctx, inputs, *, audit_query=False):
        assert audit_query is True and inputs["video"]["sha256"] == "b"*64
        def retrieve(actual_frames, actual_times, center, **kwargs):
            assert actual_frames is frames and np.array_equal(actual_times, times)
            assert kwargs["radius_m"] == 3000 and kwargs["cap"] == 1500 and kwargs["n_query"] == 80
            caches.append(kwargs["query_cache"])
            kwargs["query_cache"]["synthetic_query_descriptor"] = True
            calls.append(center)
            return {"status": "complete"}
        return retrieve
    monkeypatch.setattr(poi_only, "_retrieval", retrieval)
    def localize(actual_plan, *, timestamps, query_video_sha256, retrieve_evidence, query_cache):
        assert actual_plan == plan and query_video_sha256 == "b"*64
        for region in actual_plan["region_plans"]:
            retrieve_evidence(region, n_query=80, query_mode="crops", query_cache=query_cache)
        return {"status": "complete", "reference_union": {"ref_ids": ["mapillary:1", "mapillary:2"],
            "query_indices": np.linspace(0, 99, 80).astype(np.int64), "query_times": times[np.linspace(0, 99, 80).astype(int)],
            "ref_latlon": np.array([[50., 8.], [50.1, 8.1]]), "similarities": np.ones((80, 2), np.float32)*.2,
            "hashes": {"synthetic": "c"*64}, "reference_count": 2}}
    monkeypatch.setattr(adapter, "localize_image_union", localize)
    return SimpleNamespace(cfg=cfg, cue=cue, frames=bundle, plan=plan, times=times, closed=closed,
        calls=calls, caches=caches, request={"video_sha256": "b"*64})


def test_adapter_uses_audited_shared_query_and_freezes_union_arrays_once(setup):
    s = setup
    candidate = adapter.run_image_union_child(s.cfg, s.cfg.data_dir / "video.mp4", s.request)
    assert len(s.calls) == 2 and s.caches[0] is s.caches[1] and s.caches[0] == {}
    assert s.closed == [True]
    record = candidate["reference_union"]["matrix_artifact"]
    path = s.cfg.output_dir / record["path"]
    assert file_sha256(path) == record["sha256"]
    with np.load(path, allow_pickle=False) as z:
        assert z["similarities"].shape == (80, 2)
        assert z["ref_ids"].tolist() == ["mapillary:1", "mapillary:2"]
        np.testing.assert_array_equal(z["query_times"], s.times[z["query_indices"]])
    assert "similarities" not in candidate["reference_union"]
    assert candidate["evidence_artifacts"] == [record]
    assert (s.cfg.output_dir / "image_union_prediction.json").is_file()


@pytest.mark.parametrize("field", ["stride", "fps", "n_frames", "timestamps_sha256", "window"])
def test_native_cue_grid_drift_declines_before_retrieval_and_closes_frames(setup, field):
    s = setup
    s.cue["auto_quality"][field] = None
    with pytest.raises(ValueError, match="frozen native cue grid"):
        adapter.run_image_union_child(s.cfg, s.cfg.data_dir / "video.mp4", s.request)
    assert s.calls == [] and s.closed == [True]


def test_failed_localization_closes_frames_and_clears_query_cache(setup, monkeypatch):
    s = setup
    def failed(plan, **kwargs):
        kwargs["retrieve_evidence"](plan["region_plans"][0], n_query=80, query_mode="crops", query_cache=kwargs["query_cache"])
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(adapter, "localize_image_union", failed)
    with pytest.raises(RuntimeError): adapter.run_image_union_child(s.cfg, s.cfg.data_dir / "video.mp4", s.request)
    assert s.closed == [True] and s.caches[0] == {}


def test_inactive_plan_is_retained_without_extracting_frames(setup):
    s = setup
    s.plan.update(status="inactive", reason="branch_count_outside_fixed_budget", branch_count=7)
    candidate = adapter.run_image_union_child(s.cfg, s.cfg.data_dir / "video.mp4", s.request)
    assert candidate["status"] == "unavailable" and candidate["retained_branch_count"] == 7
    assert s.closed == [] and s.calls == []


def test_worker_kind_uses_native_data_scope_and_fixed_crop_profile(tmp_path):
    parent = PipelineConfig(url="", city="Example", data_dir=tmp_path / "data" / "clip",
        output_dir=tmp_path / "out", answer="assisted", assisted_profile="v2")
    cfg = worker.child_config(parent, "poi_image_union", tmp_path / "video.mp4")
    assert cfg.data_dir == parent.data_dir / "assisted" / "native" / "clip"
    assert cfg.output_dir == parent.output_dir / "children" / "poi_image_union"
    assert cfg.answer == "poi" and cfg.assisted_profile == "v2"
    assert cfg.vpr_query_mode == "crops" and cfg.poi_radius_m == 3000 and cfg.poi_ref_cap == 1500
    assert cfg.max_frames is None and cfg.frame_max_width is None and cfg.frame_memory_mb == 128.


def test_visual_worker_dispatches_union_adapter_without_legacy_pipeline(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(worker, "_pipeline_child", lambda *a: pytest.fail("legacy child dispatched"))
    monkeypatch.setattr(worker, "_image_child", lambda *a: pytest.fail("single-branch child dispatched"))
    calls = []
    def union(cfg, video, request):
        calls.append((cfg, video, request))
        return {"status": "unavailable"}
    monkeypatch.setattr(adapter, "run_image_union_child", union)
    result = worker._visual_child(s.cfg, s.cfg.data_dir / "video.mp4", "poi_image_union", s.request)
    assert len(calls) == 1 and result["model_downloads_disabled"] is True
    assert result["cpu_runtime"]["intra_op_threads"] == 4
