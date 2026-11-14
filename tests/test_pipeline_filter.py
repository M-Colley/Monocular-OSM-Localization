"""The opt-in filter answer uses video and an independent named-city graph."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.pipeline as pipeline
from monocular_osm import baselines
from monocular_osm.cli import build_arg_parser
from monocular_osm.frame_extraction import ExtractedFrames, FrameSequence
from monocular_osm.provenance import file_sha256
from monocular_osm.stages.context import RunContext


def cfg(tmp_path, **kwargs):
    return pipeline.PipelineConfig(url="local", city="Testville, Country",
                                  data_dir=tmp_path / "data", output_dir=tmp_path / "output", **kwargs)


def test_profile_defaults_and_explicit_overrides(tmp_path):
    old = cfg(tmp_path)
    assert old.frame_stride == 6 and old.graph_filter_particles == 200_000
    assert old.graph_filter_path_budget_mb is None and not old.graph_filter
    selected = cfg(tmp_path, answer="filter")
    assert selected.graph_filter and selected.frame_stride is None
    assert selected.graph_filter_particles == 600_000
    assert selected.graph_filter_path_budget_mb == 256 and selected.vo_workers == 2
    explicit = cfg(tmp_path, answer="filter", graph_filter_particles=2000, frame_stride=4, vo_workers=3)
    assert explicit.graph_filter_particles == 2000 and explicit.frame_stride == 4 and explicit.vo_workers == 3
    assert build_arg_parser().parse_args(["--graph-filter-only"]).answer == "filter"


@pytest.mark.parametrize("settings", [{"osm_around": (1., 2., 100.)}, {"ground_truth_waypoints": Path("labels.json")},
    {"ground_truth_streets": ("Street",)}, {"graph_filter_vpr": True}, {"retrieval_only": True},
    {"use_ground_flow_scale": True}, {"estimated_length_m": 1000.}])
def test_filter_profile_rejects_conflicting_inputs(tmp_path, settings):
    with pytest.raises(ValueError):
        cfg(tmp_path, answer="filter", **settings)


@pytest.mark.parametrize("fps,stride", [(10., 1), (30., 3), (60., 6), (7.5, 1)])
def test_filter_auto_stride_uses_the_actual_video_fps(tmp_path, monkeypatch, fps, stride):
    from monocular_osm.stages import acquire
    import cv2

    class Capture:
        def __init__(self, *_): pass
        def isOpened(self): return True
        def get(self, _): return fps
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    calls = []
    def extract(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=[np.zeros((8, 8, 3), np.uint8)] * 3, fps=fps)
    monkeypatch.setattr(acquire, "extract_frames", extract)
    ctx = RunContext(cfg=cfg(tmp_path, answer="filter"), fusion_w=None, video_path=tmp_path / "video.mp4")
    acquire.extract_video_frames(ctx)
    assert ctx.cfg.frame_stride == stride == calls[0]["stride"]
    assert "sample_count" not in calls[0]


def _mock_profile(tmp_path, monkeypatch, *, failed=False):
    configuration = cfg(tmp_path, answer="filter")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic video input")
    frames = FrameSequence(1)
    for _ in range(3):
        frames.append(np.zeros((8, 8, 3), np.uint8))
    extracted = ExtractedFrames(frames, [12., 12.1, 12.3], 10., 3)
    monkeypatch.setattr(pipeline, "acquire_video", lambda ctx: setattr(ctx, "video_path", video))
    def extract(ctx):
        ctx.cfg.frame_stride = 1
        ctx.frames = extracted
    monkeypatch.setattr(pipeline, "extract_video_frames", extract)
    monkeypatch.setattr(pipeline, "run_visual_odometry", lambda ctx: None)
    for name in ("vpr_coarse_prior", "coarse_seed_from_frames", "plate_district_prior", "vlm_district_prior",
                 "estimate_length_prior", "load_road_graph", "_run_geometry_stages", "evaluate_waypoints"):
        monkeypatch.setattr(pipeline, name, lambda *_: pytest.fail("an unrelated stage ran"))
    def filter_stage(ctx):
        if failed:
            return
        assert (ctx.cfg.output_dir / "filter_inputs.json").exists()
        ctx.result["filter_position"] = {"source": "graph_filter", "route_latlon": [[1., 2.], [1.01, 2.01]],
                                         "route_t_sec": [12., 12.3]}
    monkeypatch.setattr(pipeline, "graph_filter_channel", filter_stage)
    import monocular_osm.provenance as provenance
    monkeypatch.setattr(provenance, "collect_provenance", lambda conf, extra=None: {"config": provenance.config_snapshot(conf), **(extra or {})})
    from monocular_osm import position
    monkeypatch.setattr(position, "plot_answer", lambda answer, path, **kwargs: Path(path).write_bytes(b"selected route map"))
    return configuration, frames


def test_normal_pipeline_filter_selects_maps_and_freezes_without_other_stages(tmp_path, monkeypatch, capsys):
    configuration, frames = _mock_profile(tmp_path, monkeypatch)
    result = pipeline.run_pipeline(configuration)
    answer = result["answer"]
    assert answer["mode"] == "filter" and answer["method"] == "graph_filter_only"
    assert answer["source"] == "result.filter_position" and answer["route_t_sec"] == [12., 12.3]
    assert "position" not in result and result["execution"]["retrieval"] == "skipped"
    assert result["channels"]["vpr"]["status"] == "disabled" and frames._closed
    assert "answer=filter, method=graph_filter_only" in capsys.readouterr().out
    assert answer["google_maps_url"] and answer["map_artifact"] == "answer.png"
    assert baselines.answer(result).meta == {"mode": "filter", "method": "graph_filter_only"}
    frozen = json.loads((configuration.output_dir / "frozen_prediction.json").read_text())
    assert frozen["prediction_available"] and frozen["scoring_has_not_run"]
    for name, expected in frozen["artifacts"].items():
        assert file_sha256(configuration.output_dir / name) == expected
    for relative, expected in frozen["inputs"]["source_sha256"].items():
        assert file_sha256(configuration.output_dir / "source_snapshot" / relative) == expected


def test_filter_failure_preserves_failure_checkpoint_and_releases_frames(tmp_path, monkeypatch):
    configuration, frames = _mock_profile(tmp_path, monkeypatch, failed=True)
    with pytest.raises(RuntimeError, match="produced no route"):
        pipeline.run_pipeline(configuration)
    saved = json.loads((configuration.output_dir / "result.json").read_text())
    assert saved["execution"]["stage"] == "failed" and "answer" not in saved
    assert frames._closed
    assert not json.loads((configuration.output_dir / "frozen_prediction.json").read_text())["prediction_available"]


def test_stage_uses_whole_city_graph_pts_and_explicit_budget(tmp_path, monkeypatch):
    import monocular_osm.graph_filter as gf
    import monocular_osm.stages.graph_filter as stage
    from tests.test_stages_graph_filter import _vo_like_trajectory, jittered_grid

    configuration = cfg(tmp_path, answer="filter", graph_filter_particles=100)
    configuration.output_dir.mkdir()
    ctx = RunContext(cfg=configuration, fusion_w=None)
    ctx.traj = _vo_like_trajectory(n=11)
    intervals = np.array([.1, .12, .09, .11, .1, .09, .1, .12, .1, .1])
    times = np.r_[17., 17. + np.cumsum(intervals)]
    ctx.frames = SimpleNamespace(timestamps=times)
    ctx.result["filter_inputs"] = {"input_policy": {"ground_truth": False}}
    ctx.road = SimpleNamespace(graph="unrelated cropped map")
    graph = jittered_grid()
    metadata = {"scope": "whole_city", "cache_path": str(tmp_path / "whole_city.graphml")}
    monkeypatch.setattr(stage, "_whole_city_map", lambda _: (graph, metadata))
    speed = SimpleNamespace(lengths_m=np.ones(10), rel_spread=.12, to_dict=lambda: {"coverage": 1.})
    monkeypatch.setattr(stage, "_plane_speed", lambda _: speed)
    seen = {}
    real = gf.run_filter
    def run(net, odom, config, **kwargs):
        seen.update(odom=odom, config=config, observations=kwargs["position_obs"])
        return real(net, odom, config, **kwargs)
    monkeypatch.setattr(gf, "run_filter", run)
    stage.graph_filter_channel(ctx)
    assert ctx.road.graph is graph
    assert np.allclose(seen["odom"].dt, [intervals[:5].sum(), intervals[5:].sum()])
    assert seen["config"].path_budget_mb == 256 * 1024**2 / 1e6
    assert seen["config"].init_disc is None and seen["observations"] == []
    assert ctx.result["filter_position"]["route_t_sec"][0] == 17.
    with np.load(configuration.output_dir / "filter_prediction.npz") as result:
        assert np.allclose(result["video_times_s"], times[result["frame_index"]])


def test_explicit_missing_map_never_refetches(tmp_path, monkeypatch):
    import monocular_osm.graph_filter as gf
    from monocular_osm.stages.graph_filter import _whole_city_map
    monkeypatch.setattr(gf, "fetch_drive_graph", lambda **kw: pytest.fail("explicit map refetched"))
    ctx = RunContext(cfg=cfg(tmp_path, answer="filter", graph_filter_map_path=tmp_path / "missing.graphml"), fusion_w=None)
    with pytest.raises(FileNotFoundError):
        _whole_city_map(ctx)


def test_named_city_map_loader_never_uses_the_context_disc(tmp_path, monkeypatch):
    import monocular_osm.graph_filter as gf
    from monocular_osm.stages.graph_filter import _whole_city_map
    calls = []
    graph = object()
    def fetch(**kwargs):
        calls.append(kwargs)
        kwargs["cache_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["cache_path"].write_bytes(b"named-city snapshot")
        return graph
    monkeypatch.setattr(gf, "fetch_drive_graph", fetch)
    ctx = RunContext(cfg=cfg(tmp_path, answer="filter"), fusion_w=None)
    ctx.osm_around = (1., 2., 100.)
    actual, metadata = _whole_city_map(ctx)
    assert actual is graph and set(calls[0]) == {"place", "cache_path"}
    assert calls[0]["place"] == ctx.cfg.city
    assert metadata["scope"] == "whole_city" and metadata["around"] is None and metadata["bbox"] is None
    assert metadata["source"] == "named_city_fetch"
    assert _whole_city_map(ctx)[1]["source"] == "city_cache"


def test_benchmark_health_keeps_filter_failures_without_requiring_retrieval():
    from tests.test_benchmark import M
    result = {"answer": {"mode": "filter"},
              "provenance": {"config": {"answer": "filter", "graph_filter": True}},
              "channels": {"vpr": {"status": "disabled"}, "graph_filter": {"status": "ok"}}}
    assert M.degraded_reasons(M._health_of(result), token_set=False) == []
    result["channels"]["graph_filter"]["status"] = "failed"
    assert any("graph_filter" in reason for reason in M.degraded_reasons(M._health_of(result), token_set=False))
