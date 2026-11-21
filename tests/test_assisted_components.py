"""Child-component boundaries with synthetic arrays; no acquisition/models/PF."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import assisted_worker as worker
from monocular_osm import assisted_motion as motion
from monocular_osm import poi_image_track as image
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.plane_speed import PlaneSpeed
from monocular_osm.visual_odometry import Trajectory
from monocular_osm.stages import acquire, graph_filter, trajectory, report, filter_only, poi_only


def _case(tmp_path, kind):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"same encoded native video")
    parent = PipelineConfig(url="", city="Synthetic City", data_dir=tmp_path / "data/input",
        output_dir=tmp_path / "out", vo_start_sec=4., vo_end_sec=164., answer="assisted")
    cfg = worker.child_config(parent, kind, video)
    cfg.output_dir.mkdir(parents=True)
    request = {"video_sha256": worker.file_sha256(video)}
    return cfg, video, request


def _forbidden(*args, **kwargs):
    pytest.fail("unreached heavy acquisition/VO/filter stage ran")


def _motion_case(tmp_path, monkeypatch):
    cfg, video, request = _case(tmp_path, "metric_pf")
    n = 1600
    times = 4. + np.arange(n) / 10
    centers = np.c_[np.zeros((n,2)), np.arange(n)]
    traj = Trajectory(centers, centers[:,[0,2]], np.ones(n,bool), [30]*n,
                      np.repeat(np.eye(3)[None],n,axis=0), -centers)
    speed = PlaneSpeed(np.full(n-1,.6), np.full(n-1,.6), np.full(n-1,30), .6, .1, 1.5)
    K = np.array([[8.,0.,6.],[0.,8.,4.],[0.,0.,1.]])
    prepared = {"summary": {"status": "complete", "eligible": True,
        "native_quality": {"status":"complete","stride":3,"fps":30.,"valid_fraction":1.},
        "plane": {"status":"complete","measured":True,"coverage":.6,"rel_spread":.1}}}
    monkeypatch.setattr(worker, "_load_cues", lambda request: {"metric_preflight": prepared})
    loaded, graphed, finalized = [], [], []
    def load(package, path, *, window):
        assert package is prepared and path == video and window == [4.,164.]
        loaded.append(True)
        return traj, speed, times, K
    monkeypatch.setattr(motion,"load_metric_inputs",load)
    monkeypatch.setattr(acquire,"extract_video_frames",_forbidden)
    monkeypatch.setattr(trajectory,"run_visual_odometry",_forbidden)
    monkeypatch.setattr(motion,"extract_frames",_forbidden)
    monkeypatch.setattr(motion,"estimate_trajectory",_forbidden)
    monkeypatch.setattr(graph_filter,"_plane_speed",_forbidden)
    monkeypatch.setattr(worker,"_sources",lambda output: {"synthetic.py":"a"*64})
    route = [[50.,10.],[50.001,10.001]]
    def graph(ctx):
        graphed.append(ctx)
        assert ctx.traj is traj and ctx.validated_metric_speed is speed and ctx.require_metric_filter is True
        assert ctx.frames.frames is None and ctx.frames.timestamps is times and ctx.frames.fps == 30.
        assert ctx.cfg.frame_stride == 3 and ctx.n_valid == int(traj.valid.sum())
        assert ctx.cfg.answer == "filter" and ctx.cfg.graph_filter_speed == "plane"
        assert ctx.cfg.graph_filter_particles == 600_000 and ctx.cfg.graph_filter_seed == 0
        assert ctx.cfg.graph_filter_vpr is False and ctx.cfg.osm_around is None and ctx.vpr_track is None
        assert ctx.cfg.graph_filter_map_path is None and ctx.cfg.camera_height_m == 1.5
        ctx.result.update(filter_position={"route_latlon":route,"route_t_sec":[times[0],times[-1]]},
            graph_filter={"metric_odometry":True,"n_position_observations":0},
            provenance={"graph_filter_inputs":{"metric_only":True}})
    monkeypatch.setattr(graph_filter,"graph_filter_channel",graph)
    def finalize(ctx):
        finalized.append(ctx)
        if "filter_position" in ctx.result: ctx.result["answer"] = ctx.result["filter_position"]
    monkeypatch.setattr(report,"finalize_result",finalize)
    monkeypatch.setattr(filter_only,"_freeze",lambda ctx: None)
    return SimpleNamespace(cfg=cfg,video=video,request=request,prepared=prepared,traj=traj,speed=speed,
        times=times,K=K,loaded=loaded,graphed=graphed,finalized=finalized)


def test_metric_child_reuses_verified_native_arrays_without_frames_or_recomputation(tmp_path,monkeypatch):
    case = _motion_case(tmp_path,monkeypatch)
    result = worker._metric_child(case.cfg,case.video,case.request)
    assert result["status"] == "complete" and result["metric_only"] is True
    assert result["uses_user_spatial_prior"] is False and result["position_observation_count"] == 0
    assert result["route_timestamps"] == [case.times[0],case.times[-1]]
    assert len(case.loaded) == len(case.graphed) == len(case.finalized) == 1


@pytest.mark.parametrize("reason",["native_weak","plane_weak","missing"])
def test_metric_ineligible_preflight_declines_before_loader_graph_or_particles(tmp_path,monkeypatch,reason):
    case = _motion_case(tmp_path,monkeypatch)
    if reason == "missing": case.prepared.clear()
    else:
        case.prepared["summary"]["eligible"] = False
        if reason == "native_weak": case.prepared["summary"]["native_quality"]["valid_fraction"] = .94
        else: case.prepared["summary"]["plane"]["coverage"] = .49
    monkeypatch.setattr(motion,"load_metric_inputs",_forbidden)
    monkeypatch.setattr(graph_filter,"graph_filter_channel",_forbidden)
    result = worker._metric_child(case.cfg,case.video,case.request)
    assert result["status"] == "unavailable"
    assert not case.loaded and not case.graphed and not case.finalized


def test_metric_changed_frozen_package_refuses_before_graph_or_particles(tmp_path,monkeypatch):
    case = _motion_case(tmp_path,monkeypatch)
    def reject(*args,**kwargs): raise ValueError("frozen source or input hash changed")
    monkeypatch.setattr(motion,"load_metric_inputs",reject)
    monkeypatch.setattr(graph_filter,"graph_filter_channel",_forbidden)
    with pytest.raises(ValueError,match="hash changed"):
        worker._metric_child(case.cfg,case.video,case.request)
    assert not case.graphed and not case.finalized


@pytest.mark.parametrize("fault",["native_weak","coverage","spread","unknown_spread"])
def test_metric_actual_loaded_arrays_are_gated_even_if_summary_says_eligible(tmp_path,monkeypatch,fault):
    case = _motion_case(tmp_path,monkeypatch)
    if fault == "native_weak": case.traj.valid[1500:] = False
    elif fault == "coverage": case.speed.coverage = .49
    elif fault == "spread": case.speed.rel_spread = .151
    else: case.speed.rel_spread = np.nan
    monkeypatch.setattr(graph_filter,"graph_filter_channel",_forbidden)
    result = worker._metric_child(case.cfg,case.video,case.request)
    assert result["status"] == "unavailable" and "actual loaded" in result["error"]
    assert case.loaded == [True] and not case.graphed and not case.finalized


def test_metric_actual_admission_boundaries_are_inclusive(tmp_path,monkeypatch):
    case = _motion_case(tmp_path,monkeypatch)
    case.traj.valid[1520:] = False
    case.speed.coverage,case.speed.rel_spread = .5,.15
    assert worker._metric_child(case.cfg,case.video,case.request)["status"] == "complete"


@pytest.mark.parametrize("failure",["unit_steps","no_route"])
def test_metric_child_cannot_promote_nonmetric_or_absent_filter_output(tmp_path,monkeypatch,failure):
    case = _motion_case(tmp_path,monkeypatch)
    original = graph_filter.graph_filter_channel
    def faulty_graph(ctx):
        original(ctx)
        if failure == "unit_steps": ctx.result["graph_filter"]["metric_odometry"] = False
        else: ctx.result.pop("filter_position")
    monkeypatch.setattr(graph_filter,"graph_filter_channel",faulty_graph)
    result = worker._metric_child(case.cfg,case.video,case.request)
    assert result["status"] == "unavailable"
    assert case.graphed[0].result["execution"]["stage"] == "failed"


def _image_case(tmp_path,monkeypatch):
    cfg,video,request = _case(tmp_path,"poi_image_track")
    times = 4. + np.arange(480)/3
    pixels = [np.zeros((8,12,3),np.uint8)] * len(times)
    closed,caches,calls = [],[],[]
    frame_package = SimpleNamespace(frames=pixels,timestamps=times,fps=30.,close=lambda:closed.append(True))
    native = {"source":"same_video_native_frames","native_bbox_grid":True,"sample_count":24,
        "video_sha256":request["video_sha256"],"frame_shape_hw":[8,12],"window":[times[0],times[-1]]}
    clues = {"selected_answer":None,"detections_sha256":"a"*64,
        "inventory":{"city":"Synthetic City","scope":"named_city_bbox","snapshot_sha256":"b"*64},
        "candidate_clusters":[{"pois":[{"id":"one","lat":0.,"lon":0.},{"id":"two","lat":0.,"lon":.0002}],
                               "conjunctions":[]}]}
    auto = {"status":"complete","valid_fraction":.3,"n_valid":144,"n_frames":480,"stride":10,"fps":30.,
        "timestamps_sha256":worker._array_sha(times),"video_sha256":request["video_sha256"],"window":[4.,164.]}
    cue = {"clues":clues,"ocr_provenance":native,"inventory":clues["inventory"],"auto_quality":auto,
           "video_sha256":request["video_sha256"],"window":[4.,164.],"vo":object()}
    indices = np.linspace(0,len(times)-1,80).astype(int)
    scores = np.full((80,3),.1,np.float32)
    scores[np.arange(80),(np.arange(80)//27)%3] = .8
    evidence = {"query_indices":indices,"query_times":times[indices],"ref_ids":np.array(["mapillary:a","mapillary:b","mapillary:c"]),
        "ref_latlon":np.array([[0.,0.],[0.,.0002],[0.,.0004]]),"similarities":scores,
        "provenance":{"source":"mapillary","center":[0.,.0001],"radius_m":3000.,"cap":1500,"query_mode":"crops",
            "backbone":{"name":"megaloc","hub":"declared-fixed-encoder"},"preprocessing":{"descriptor_normalization":"L2"}}}
    monkeypatch.setattr(worker,"_load_cues",lambda request:cue)
    def extract(ctx):
        assert ctx.cfg.frame_stride is None and ctx.cfg.max_frames is None and ctx.cfg.frame_max_width is None
        ctx.cfg.frame_stride = 10
        ctx.frames = frame_package
    monkeypatch.setattr(acquire,"extract_video_frames",extract)
    def retrieval(ctx,inputs):
        def retrieve(frames,stamps,center,*,radius_m,cap,n_query,query_cache):
            assert frames is pixels and stamps is times and ctx.cfg.vpr_query_mode == "crops"
            assert np.allclose(center,[0.,.0001]) and (radius_m,cap,n_query)==(3000.,1500,80)
            assert not query_cache
            query_cache["descriptors"] = object()
            caches.append(query_cache)
            calls.append(True)
            return evidence
        return retrieve
    monkeypatch.setattr(poi_only,"_retrieval",retrieval)
    monkeypatch.setattr(trajectory,"run_visual_odometry",_forbidden)
    monkeypatch.setattr(motion,"load_metric_inputs",_forbidden)
    monkeypatch.setattr(motion,"estimate_trajectory",_forbidden)
    return SimpleNamespace(cfg=cfg,video=video,request=request,times=times,evidence=evidence,cue=cue,
                           closed=closed,caches=caches,calls=calls)


def test_image_child_uses_same_auto_grid_and_actual_query_times_without_vo(tmp_path,monkeypatch):
    case = _image_case(tmp_path,monkeypatch)
    result = worker._image_child(case.cfg,case.video,case.request)
    assert result["status"] == "complete" and result["uses_rigid_vo"] is False
    assert result["retained_branch_count"] == result["successful_branch_count"] == 1
    np.testing.assert_array_equal(result["route_timestamps"],case.evidence["query_times"])
    assert result["acquisition"]["stride"] == 10 and result["acquisition"]["n_frames"] == 480
    assert result["acquisition"]["timestamps_sha256"] == worker._array_sha(case.times)
    assert case.calls == [True] and case.closed == [True] and case.caches == [{}]


@pytest.mark.parametrize("fault",["timestamps_sha256","stride","fps","n_frames","window"])
def test_image_changed_auto_grid_declines_before_reference_forward(tmp_path,monkeypatch,fault):
    case = _image_case(tmp_path,monkeypatch)
    auto = case.cue["auto_quality"]
    if fault == "timestamps_sha256": auto[fault] = "c"*64
    elif fault == "window": auto[fault] = [5.,164.]
    else: auto[fault] += 1
    monkeypatch.setattr(poi_only,"_retrieval",_forbidden)
    with pytest.raises(ValueError): worker._image_child(case.cfg,case.video,case.request)
    assert not case.calls and case.closed == [True]


def test_image_query_score_times_must_match_actual_native_rows_before_decoder(tmp_path,monkeypatch):
    case = _image_case(tmp_path,monkeypatch)
    case.evidence["query_times"][1] += .01
    monkeypatch.setattr(image,"decode_single_branch_image_track",_forbidden)
    with pytest.raises(ValueError,match="actual native video times"):
        worker._image_child(case.cfg,case.video,case.request)
    assert case.closed == [True] and case.caches == [{}]


def test_image_ambiguous_clues_decline_before_frame_or_reference_acquisition(tmp_path,monkeypatch):
    case = _image_case(tmp_path,monkeypatch)
    case.cue["clues"]["candidate_clusters"] *= 2
    monkeypatch.setattr(acquire,"extract_video_frames",_forbidden)
    monkeypatch.setattr(poi_only,"_retrieval",_forbidden)
    with pytest.raises(ValueError,match="exactly one"):
        worker._image_child(case.cfg,case.video,case.request)
    assert not case.calls
