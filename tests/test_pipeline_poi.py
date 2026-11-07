"""Mocked POI integration; no OCR/retrieval models or network requests."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.pipeline as pipeline
from monocular_osm import baselines
from monocular_osm.cli import build_arg_parser
from monocular_osm.frame_extraction import ExtractedFrames, FrameSequence
from monocular_osm.provenance import config_snapshot, file_sha256
from monocular_osm.stages import poi_only as stage


def _cfg(tmp_path, **kwargs):
    return pipeline.PipelineConfig(url="local", city="Example city", data_dir=tmp_path / "d",
                                  output_dir=tmp_path / "o", vo_end_sec=11., **kwargs)


def _fake_runtimes(tmp_path, monkeypatch):
    from monocular_osm.stages import poi_runtime
    directory = tmp_path / "model-assets"
    directory.mkdir(exist_ok=True)
    for name in poi_runtime._ASSETS.values():
        (directory / name).write_bytes(b"synthetic weights; never loaded")
    monkeypatch.setattr(stage, "reader_spec", lambda engine: poi_runtime.reader_spec(engine, model_dir=directory))
    @contextmanager
    def reader(spec):
        proof = {"providers": ["CPUExecutionProvider"], "intra_op_num_threads": 2, "inter_op_num_threads": 1}
        yield object(), {**deepcopy(spec), "actual_sessions": {name: dict(proof) for name in
                        ("text_det", "text_rec", "text_cls")},
                        "actual_batches": {"rec_batch_num": 4, "cls_batch_num": 4}, "cache_hit": False, "read_calls": 24}
    @contextmanager
    def threads(count):
        assert count == 4
        yield {"device": "cpu", "intra_op_threads": 4, "inter_op_threads": 1, "torch_version": "synthetic"}
    monkeypatch.setattr(stage, "pinned_reader", reader)
    monkeypatch.setattr(stage, "torch_cpu_threads", threads)


def test_mode_is_explicit_and_uses_an_uncapped_window(tmp_path):
    assert _cfg(tmp_path).answer == "track" and _cfg(tmp_path).frame_stride == 6
    cfg = _cfg(tmp_path, answer="poi")
    assert cfg.frame_stride is None and cfg.max_frames is None and cfg.vo_workers == 2
    assert cfg.poi_radius_m == 1000 and cfg.poi_ref_cap == 500 and not cfg.graph_filter
    assert build_arg_parser().parse_args(["--poi-only"]).answer == "poi"
    assert build_arg_parser().parse_args([]).answer == "track"


@pytest.mark.parametrize("override", [{"max_frames": 256}, {"frame_max_width": 1280},
    {"use_ocr_super_res": True}, {"ocr_video_path": Path("other.mp4")}, {"osm_around": (1., 2., 1000.)},
    {"estimated_length_m": 2000.}, {"ground_truth_waypoints": Path("labels.json")},
    {"ground_truth_streets": ("road",)}, {"openvo_trajectory_path": Path("poses.npz")},
    {"vggt_long_trajectory_path": Path("poses.txt")}, {"use_da3_trajectory": True},
    {"use_ground_flow_scale": True}, {"graph_filter": True}, {"graph_filter_vpr": True},
    {"graph_filter_map_path": Path("map.graphml")}, {"vpr_source": "union"}, {"vpr_source": "unknown"},
    {"poi_radius_m": True}, {"poi_ref_cap": True}, {"ocr_engine": "easyocr"}, {"retrieval_only": True}])
def test_external_or_incompatible_inputs_are_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        _cfg(tmp_path, answer="poi", **override)


def _setup(tmp_path, monkeypatch, *, missing=False, failed=False, bad_time=False, external=False, real_alignment=False):
    _fake_runtimes(tmp_path, monkeypatch)
    from tests.test_poi_localization import _case
    from monocular_osm import kartaview_vpr as kv, poi_clues, poi_localization, provenance, position, scene_text
    from monocular_osm.osm_data import _build_polyline_view
    import networkx as nx

    cfg = _cfg(tmp_path, answer="poi", frame_stride=3,
               vpr_region_cache_dir=tmp_path / "external" if external else None)
    cfg.data_dir.mkdir()
    cfg.output_dir.mkdir()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"same video")
    frames = FrameSequence(1)
    for _ in range(11):
        frames.append(np.zeros((12, 16, 3), np.uint8))
    times = np.arange(11, dtype=float)
    args, evidence = _case()
    detections = [] if missing else deepcopy(args["detections"])
    for item in detections:
        item["bbox"] = [10., 4., 15., 8.]
    if bad_time:
        detections[0]["t_sec"] = 10.01
    inventory = deepcopy(args["inventory"])
    inventory["snapshot_sha256"] = poi_clues._digest(inventory["pois"])
    inventory.update(scope="named_city_bbox", response_sha256="a" * 64,
                     query='[out:json];nwr["shop"](0.0000000,0.0000000,0.0200000,0.0200000);out center tags;',
                     bbox_wsen=[0., 0., .02, .02])
    monkeypatch.setattr(pipeline, "acquire_video", lambda ctx: setattr(ctx, "video_path", video))
    extracted = ExtractedFrames(frames, times.tolist(), 3., 33)
    monkeypatch.setattr(pipeline, "extract_video_frames", lambda ctx: setattr(ctx, "frames", extracted))
    def vo(ctx):
        ctx.traj = SimpleNamespace(xz=np.c_[times, np.zeros(11)], valid=np.ones(11, bool))
        ctx.n_valid = 11
        ctx.vo_cache = cfg.data_dir / "signed_vo.npz"
        signature = json.dumps({"schema": 2, "video": {"path": str(video.resolve())}, "extraction": {"stride": 3}})
        np.savez(ctx.vo_cache, cache_signature=signature)
    monkeypatch.setattr(pipeline, "run_visual_odometry", vo)
    for name in ("vpr_coarse_prior", "coarse_seed_from_frames", "plate_district_prior", "vlm_district_prior",
                 "estimate_length_prior", "load_road_graph", "_run_geometry_stages", "evaluate_waypoints"):
        monkeypatch.setattr(pipeline, name, lambda *_: pytest.fail("unrelated or label-reading stage ran"))
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    graph.add_node(0, x=0., y=0.)
    graph.add_node(1, x=1200., y=0.)
    graph.add_edge(0, 1, length=1200.)
    road = _build_polyline_view(graph)
    map_path = tmp_path / "city.graphml"
    map_path.write_bytes(b"whole named-city snapshot")
    metadata = {"scope": "whole_city", "city": cfg.city, "query": cfg.city, "source": "city_cache",
                "cache_path": str(map_path.resolve()), "configured_path": None,
                "sha256": file_sha256(map_path), "around": None, "bbox": None}
    monkeypatch.setattr(stage, "_whole_city_map", lambda ctx: (road, metadata))
    monkeypatch.setattr(poi_clues, "fetch_city_retail", lambda city, directory: deepcopy(inventory))
    ocr_calls = []
    def ocr(path, **kwargs):
        ocr_calls.append(kwargs)
        Path(kwargs["cache_path"]).write_text(json.dumps({"detections": detections}), encoding="utf-8")
        return deepcopy(detections)
    monkeypatch.setattr(scene_text, "extract_scene_text", ocr)
    calls, caches = [], []
    monkeypatch.setenv("MLY_TOKEN", "unit-test-private-token")
    def retrieve(frames_arg, timestamps, center, **kwargs):
        assert frames_arg is frames
        assert kwargs["cap"] == 500 and kwargs["radius_m"] == 1000 and kwargs["n_query"] == 80
        assert kwargs["device"] == "cpu"
        calls.append((center, kwargs))
        cache = kwargs["query_cache"]
        caches.append(cache)
        if len(calls) > 1:
            assert cache["synthetic descriptor"] == "shared"
        cache["synthetic descriptor"] = "shared"
        if failed and len(calls) == 2:
            kv.LAST_FAILURE = "temporary acquisition failure"
            return None
        raw = deepcopy(evidence[len(calls) - 1])
        raw["provenance"]["backbone"]["device"] = "cpu"
        raw["provenance"]["preprocessing"] = {"version": 1, "reference_color": "RGB", "reference_size": [512, 512],
            "reference_resize": "opencv INTER_LINEAR", "scale": "divide by255", "mean": [.485, .456, .406],
            "std": [.229, .224, .225], "descriptor_normalization": "L2", "dimensions": 2}
        raw["provenance"].update(source="mapillary", fetch={"diagnostic": "token=unit-test-private-token"},
            reference_encoding_identity="unverified synthetic test encoding",
            reference_inventory=kv._reference_snapshot([{"id": key, "lat": xy[0], "lon": xy[1]}
                for key, xy in zip(raw["ref_ids"], raw["ref_latlon"])]))
        return raw
    monkeypatch.setattr(kv, "kartaview_vpr_evidence", retrieve)
    def align(vo_xy, ts, K, clues, **kwargs):
        ids = [str(branch.get("id", f"hypothesis_{i}")) for i, branch in enumerate(clues["candidate_clusters"])]
        return {"selected": {"hypothesis_id": ids[-1], "route_latlon": [[0., 0.], [0., .001]],
                             "route_timestamps": [float(ts[0]), float(ts[-1])]},
                "hypotheses": [{"id": identifier} for identifier in ids], "search": {"hypothesis_count": len(ids)},
                "provenance": {"vo_sha256": stage._array_sha(vo_xy), "times_sha256": stage._array_sha(ts),
                               "intrinsics_sha256": stage._array_sha(K)}}
    if not real_alignment:
        monkeypatch.setattr(poi_localization.core, "align_candidate_clusters", align)
    monkeypatch.setattr(provenance, "collect_provenance", lambda conf, extra=None: {"config": config_snapshot(conf), **(extra or {})})
    monkeypatch.setattr(position, "plot_answer", lambda answer, path, **kw: Path(path).write_bytes(b"selected POI map"))
    return cfg, frames, calls, caches, ocr_calls


@pytest.mark.parametrize("external", [False, True])
def test_all_branches_select_and_freeze_without_models(tmp_path, monkeypatch, capsys, external):
    cfg, frames, calls, caches, ocr = _setup(tmp_path, monkeypatch, external=external)
    result = pipeline.run_pipeline(cfg)
    assert frames._closed and len(calls) == 2 and caches[0] is caches[1] and caches[0] == {}
    assert result["answer"]["mode"] == "poi" and result["answer"]["method"] == "poi_only"
    assert result["answer"]["experimental"] and result["answer"]["confidence_calibrated"] is False
    assert baselines.answer(result).meta == {"mode": "poi", "method": "poi_only"}
    assert "EXPERIMENTAL POI component" in capsys.readouterr().out
    assert result["channels"]["vpr"]["status"] == "disabled" and "vpr_prior" not in result
    assert result["channels"]["poi_retrieval"]["status"] == "ok"
    audit = result["provenance"]["poi_localization_audit"]
    assert len(audit["clues"]["candidate_clusters"]) == len(audit["regions"]) == 2
    assert audit["alignment"]["hypothesis_ids"] == ["hypothesis_0", "hypothesis_1"]
    assert audit["reference_union"]["count"] == 3
    assert all("evidence_identity" in record["provenance"] for record in audit["regions"])
    inputs = result["provenance"]["poi_inputs"]
    assert inputs["input_policy"]["retrieval"] and not inputs["input_policy"]["gps_or_oxts"]
    assert inputs["extraction"]["max_frames"] is None and inputs["ocr"]["native_bbox_grid"]
    assert ocr[0]["sample_count"] == 24 and ocr[0]["min_len"] == 2 and ocr[0]["end_sec"] == 10.
    for _, kwargs in calls:
        path = Path(kwargs["cache_dir"])
        if external:
            assert path.parent == cfg.vpr_region_cache_dir / cfg.data_dir.name
            assert path.name.startswith("mapillary_region_poi_")
        else:
            assert path.parent.name == "mapillary_regions"
    frozen = json.loads((cfg.output_dir / "frozen_prediction.json").read_text())
    assert frozen["prediction_available"] and frozen["scoring_has_not_run"]
    for relative, expected in frozen["artifacts"].items():
        assert file_sha256(cfg.output_dir / relative) == expected
    assert "unit-test-private-token" not in json.dumps(result)


@pytest.mark.parametrize("missing,failed", [(True, False), (False, True)])
def test_no_clues_or_partial_retrieval_remain_unanswered(tmp_path, monkeypatch, missing, failed):
    cfg, frames, calls, *_ = _setup(tmp_path, monkeypatch, missing=missing, failed=failed)
    result = pipeline.run_pipeline(cfg)
    assert "answer" not in result and "poi_position" not in result and baselines.poi_only(result) is None
    assert result["poi_localization"]["status"] == "inactive" and frames._closed
    assert len(calls) == (0 if missing else 2)
    if failed:
        assert result["poi_localization"]["diagnostic_position"] is not None
        assert result["poi_localization"]["failed_regions"] == ["hypothesis_1"]
    assert not json.loads((cfg.output_dir / "frozen_prediction.json").read_text())["prediction_available"]


def test_cached_ocr_outside_vo_horizon_fails_without_pruning(tmp_path, monkeypatch):
    cfg, frames, calls, *_ = _setup(tmp_path, monkeypatch, bad_time=True)
    with pytest.raises(ValueError, match="outside the actual VO horizon"):
        pipeline.run_pipeline(cfg)
    assert frames._closed and not calls
    saved = json.loads((cfg.output_dir / "result.json").read_text())
    assert saved["execution"]["stage"] == "failed" and "answer" not in saved


def test_road_sampling_is_bounded_without_cropping(monkeypatch):
    road = SimpleNamespace(polylines=[np.array([[0., 0.], [100., 0.]])])
    assert np.array_equal(stage._road_samples(road), np.c_[np.arange(0, 101, 12.5), np.zeros(9)])
    monkeypatch.setattr(stage, "MAX_ROAD_SAMPLES", 4)
    with pytest.raises(ValueError, match="no spatial crop"):
        stage._road_samples(road)


@pytest.mark.parametrize("fault", [None, "boundary", "shape", "time"])
def test_native_ocr_sampling_binds_pixels_and_vo_horizon(tmp_path, monkeypatch, fault):
    _fake_runtimes(tmp_path, monkeypatch)
    from monocular_osm import scene_text
    from monocular_osm.stages.context import RunContext

    cfg = _cfg(tmp_path, answer="poi")
    cfg.data_dir.mkdir()
    cfg.output_dir.mkdir()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"native video")
    ctx = RunContext(cfg=cfg, fusion_w=None, video_path=video,
                     frames=SimpleNamespace(timestamps=[3., 3.5, 4.], fps=30.))
    seen = []
    def samples(path, start, end, interval, *, sample_count):
        seen.append((start, end, sample_count))
        image = np.zeros((8, 8, 3) if fault == "shape" else (12, 16, 3), np.uint8)
        # "boundary": the last VO frame on the container's millisecond clock
        yield {"time": 4.1, "boundary": 4.0004}.get(fault, 3.5), image
    monkeypatch.setattr(scene_text, "_sample_frames", samples)
    def extract(path, **kwargs):
        assert kwargs["sample_count"] == 24 and kwargs["min_len"] == 2
        list(kwargs["frame_reader"](path, kwargs["start_sec"], kwargs["end_sec"], 6.))
        Path(kwargs["cache_path"]).write_text("{}")
        return []
    monkeypatch.setattr(scene_text, "extract_scene_text", extract)
    if fault in ("shape", "time"):
        with pytest.raises(ValueError, match="pixels/times"):
            stage._ocr(ctx, np.eye(3), file_sha256(video), (12, 16))
    else:
        _, metadata = stage._ocr(ctx, np.eye(3), file_sha256(video), (12, 16))
        assert seen == [(3., 4., 24)] and metadata["window"] == [3., 4.]
        assert metadata["native_bbox_grid"] and metadata["position_stamps_rejected"]


def test_explicit_poi_map_missing_never_refetches(tmp_path, monkeypatch):
    import monocular_osm.graph_filter as gf
    from monocular_osm.stages.context import RunContext
    monkeypatch.setattr(gf, "fetch_drive_graph", lambda **kwargs: pytest.fail("explicit map refetched"))
    ctx = RunContext(cfg=_cfg(tmp_path, answer="poi", poi_map_path=tmp_path / "missing.graphml"), fusion_w=None)
    with pytest.raises(FileNotFoundError):
        stage._whole_city_map(ctx)


def test_poi_sampling_uses_existing_auto_stride_for_direct_api(tmp_path, monkeypatch):
    import cv2
    from monocular_osm.stages import acquire
    from monocular_osm.stages.context import RunContext
    class Capture:
        def __init__(self, *_): pass
        def isOpened(self): return True
        def get(self, _): return 10.
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    calls = []
    def extract(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=[None] * 4, fps=10.)
    monkeypatch.setattr(acquire, "extract_frames", extract)
    ctx = RunContext(cfg=_cfg(tmp_path, answer="poi"), fusion_w=None, video_path=tmp_path / "video.mp4")
    acquire.extract_video_frames(ctx)
    assert ctx.cfg.frame_stride == calls[0]["stride"] == 3
    assert calls[0]["max_frames"] is None and "sample_count" not in calls[0]


def test_mocked_acquisition_and_real_geometry_emit_a_goal_auditable_contract(tmp_path, monkeypatch):
    from scripts.benchmark import _provenance_of
    from scripts.check_dev_goal import _poi_answer_issues
    cfg, frames, *_ = _setup(tmp_path, monkeypatch, real_alignment=True)
    result = pipeline.run_pipeline(cfg)
    assert result["answer"]["mode"] == "poi" and frames._closed
    score = {"meta": {"mode": "poi", "method": "poi_only"}}
    issues = _poi_answer_issues({"provenance": _provenance_of(result)}, score, cfg.city, "0:11")
    assert not issues, issues


def test_component_rejects_a_pruned_successful_helper_result(tmp_path, monkeypatch):
    cfg, frames, *_ = _setup(tmp_path, monkeypatch)
    real = stage.localize_with_poi
    def prune(*args, **kwargs):
        result = real(*args, **kwargs)
        result["regions"] = result["regions"][:1]
        return result
    monkeypatch.setattr(stage, "localize_with_poi", prune)
    with pytest.raises(ValueError, match="preserve every literal branch"):
        pipeline.run_pipeline(cfg)
    assert frames._closed
    assert not json.loads((cfg.output_dir / "frozen_prediction.json").read_text())["prediction_available"]


def test_api_error_redaction_covers_url_encoded_credentials(monkeypatch):
    from urllib.parse import quote
    token = "private|token/with+characters"
    monkeypatch.setenv("MLY_TOKEN", token)
    assert stage._safe({"url": f"https://example.invalid/?access_token={quote(token, safe='')}", "text": token}) == {
        "url": "https://example.invalid/?access_token=[REDACTED]", "text": "[REDACTED]"}


def test_warm_ocr_cache_retains_checked_actual_runtime_identity(tmp_path, monkeypatch):
    from monocular_osm import scene_text
    from monocular_osm.stages.context import RunContext
    _fake_runtimes(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, answer="poi")
    cfg.data_dir.mkdir()
    cfg.output_dir.mkdir()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"native video")
    ctx = RunContext(cfg=cfg, fusion_w=None, video_path=video, frames=SimpleNamespace(timestamps=[3., 3.5, 4.]))
    def cached_extract(path, **kwargs):
        Path(kwargs["cache_path"]).write_text("{}")
        assert kwargs["ocr_reader"] is not None
        return []
    monkeypatch.setattr(scene_text, "extract_scene_text", cached_extract)
    _, cold = stage._ocr(ctx, np.eye(3), file_sha256(video), (12, 16))
    @contextmanager
    def unused_reader(spec):
        yield object(), {**deepcopy(spec), "actual_sessions": {}, "actual_batches": {}, "read_calls": 0}
    monkeypatch.setattr(stage, "pinned_reader", unused_reader)
    _, warm = stage._ocr(ctx, np.eye(3), file_sha256(video), (12, 16))
    assert warm["reader"]["cache_hit"] and not cold["reader"]["cache_hit"]
    assert warm["reader"]["actual_sessions"] == cold["reader"]["actual_sessions"]
    assert warm["reader"]["spec_sha256"] == cold["reader"]["spec_sha256"]
    sidecar = Path(warm["reader_sidecar_path"])
    data = json.loads(sidecar.read_text())
    data["reader"]["spec_sha256"] = "f" * 64
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no verified CPU provider"):
        stage._ocr(ctx, np.eye(3), file_sha256(video), (12, 16))
