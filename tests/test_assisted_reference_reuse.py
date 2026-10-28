"""Assisted transport/freezing integration for public cache reuse; no models."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from monocular_osm import assisted_worker as worker
from monocular_osm import public_reference_cache as cache
from monocular_osm import kartaview_vpr as kv
from monocular_osm.pipeline import PipelineConfig


def parent(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"same encoded input")
    cfg = PipelineConfig(url="", city="Synthetic City", data_dir=tmp_path / "data/clip",
                         output_dir=tmp_path / "out", answer="assisted")
    return SimpleNamespace(cfg=cfg, video_path=video,
        assisted_child_root=cfg.output_dir / "children/execution", assisted_evidence={},
        result={"assisted_inputs": {"execution_id": "execution", "video": {"sha256": worker.file_sha256(video)},
                                   "source_sha256": {}}})


@pytest.mark.parametrize("baseline_rc", [0, 9])
def test_copy_receipt_precedes_spawn_and_source_transport_is_not_upgraded(tmp_path, monkeypatch, baseline_rc):
    ctx = parent(tmp_path)
    events = []
    monkeypatch.setattr(worker, "_commit_headroom", lambda: 9 * 1024**3)
    def prepare(source, cfg, **identity):
        assert source["worker_returncode"] == baseline_rc and source["execution_id"] == "execution"
        assert source["status"] == ("complete" if baseline_rc == 0 else "failed")
        assert "route_latlon" not in source and "comparison" not in source
        assert identity["execution_id"] == "execution" and identity["child_root"] == ctx.assisted_child_root
        path = cfg.output_dir / "public_reference_reuse.json"
        path.write_text(json.dumps({"before_child_launch": True, "status": "complete" if baseline_rc == 0 else "unavailable"}))
        events.append("prepare")
        return {"path": str(path), "sha256": worker.file_sha256(path)}
    monkeypatch.setattr(cache, "prepare_reuse", prepare)
    route = [[42., -83.], [42.001, -83.001]]
    def popen(command, **options):
        request_path = Path(command[-1]); request = json.loads(request_path.read_text())
        kind = request["kind"]
        assert request["execution_id"] == "execution"
        if kind == "expanded":
            item = request["public_reference_reuse"]
            assert Path(item["path"]).exists() and worker.file_sha256(item["path"]) == item["sha256"]
            assert events[-1] == "prepare"
        events.append("spawn_" + kind)
        frozen = request_path.with_name("child_frozen.json"); frozen.write_text("{}")
        request_path.with_name("response.json").write_text(json.dumps({"status": "complete", "route_latlon": route,
            "frozen_artifact": {"path": str(frozen), "sha256": worker.file_sha256(frozen)}}))
        return SimpleNamespace(wait=lambda: baseline_rc if kind == "baseline" else 0)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    base = worker.run_child("baseline", ctx)
    expanded = worker.run_child("expanded", ctx)
    assert base["route_latlon"] == expanded["route_latlon"] == route
    assert events == ["spawn_baseline", "prepare", "spawn_expanded"]


def test_absent_baseline_creates_honest_no_reuse_receipt_before_normal_expanded_spawn(tmp_path, monkeypatch):
    ctx = parent(tmp_path)
    monkeypatch.setattr(worker, "_commit_headroom", lambda: 9 * 1024**3)
    def popen(command, **options):
        path = Path(command[-1]); request = json.loads(path.read_text())
        receipt = json.loads(Path(request["public_reference_reuse"]["path"]).read_text())
        assert receipt["status"] == "unavailable" and not receipt["pools"]
        assert receipt["query_artifacts_copied"] is False
        path.with_name("response.json").write_text(json.dumps({"status": "unavailable"}))
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    assert worker.run_child("expanded", ctx)["status"] == "unavailable"


def test_baseline_seals_all_reference_requests_after_core_resets_audit(tmp_path, monkeypatch):
    from monocular_osm import pipeline, baselines
    cfg = PipelineConfig(url="", city="Synthetic City", data_dir=tmp_path / "data/clip", output_dir=tmp_path / "out")
    cfg.output_dir.mkdir()
    closed, sealed, compared = [], [], []
    ctx = SimpleNamespace(cfg=cfg, frames=SimpleNamespace(close=lambda: closed.append(True)), vpr_track=None)
    monkeypatch.setattr(pipeline.RunContext, "create", lambda cfg: ctx)
    def original(frames_bgr, center, radius_m=3000., *, cache_dir=None, source="mapillary", model_name="megaloc",
                 device="cpu", cap=2, query_mode="warp", query_batch_size=4, audit=None):
        audit.clear()  # actual core deliberately resets the opt-in audit sink
        audit.update(status="complete", marker="reference evidence")
        return np.arange(2), np.array([[42., -83.], [42.001, -83.001]]), np.ones(2)
    monkeypatch.setattr(kv, "kartaview_vpr_track", original)
    def run(context):
        context.retrieval_audits = []
        for directory, center in (("mapillary", [42.1234567, -83.]), ("mapillary_c2f", [42.2, -83.1])):
            audit = {}
            kv.kartaview_vpr_track([], center, radius_m=8000., cache_dir=str(cfg.data_dir / directory), audit=audit)
            context.retrieval_audits.append(audit)
        return {"answer": {"route_t_sec": [0., 1.]}}
    monkeypatch.setattr(pipeline, "_run_pipeline_stages", run)
    monkeypatch.setattr(baselines, "answer", lambda result: SimpleNamespace(route_latlon=[[42., -83.], [42.001, -83.001]]))
    def seal(audits, config):
        assert len(audits) == 2
        assert [Path(a["public_cache_request"]["cache_dir"]).name for a in audits] == ["mapillary", "mapillary_c2f"]
        assert audits[0]["public_cache_request"]["center"] == [42.1234567, -83.]
        for audit in audits:
            audit["public_reference_cache"] = {"status": "complete", "claim": "same snapshot"}
        sealed.append(True)
    monkeypatch.setattr(cache, "seal_pool_audits", seal)
    def comparison(context, output):
        assert sealed and all(a["public_reference_cache"]["status"] == "complete" for a in context.retrieval_audits)
        compared.append(True)
        return {"status": "synthetic"}
    monkeypatch.setattr(worker, "_comparison", comparison)
    candidate = worker._pipeline_child(cfg, "baseline", {})
    assert candidate["status"] == "complete" and sealed == compared == closed == [True]
    assert kv.kartaview_vpr_track is original


def test_child_freeze_binds_prelaunch_lineage_and_request(tmp_path, monkeypatch):
    ctx = parent(tmp_path)
    cfg = worker.child_config(ctx.cfg, "expanded", ctx.video_path)
    cfg.output_dir.mkdir(parents=True)
    receipt = cfg.output_dir / "public_reference_reuse.json"
    receipt.write_text(json.dumps({"schema": cache.SCHEMA, "status": "complete", "pools": []}))
    request = {"kind": "expanded", "video_sha256": worker.file_sha256(ctx.video_path), "source_sha256": {},
        "execution_id": "execution", "public_reference_reuse": {"path": str(receipt), "sha256": worker.file_sha256(receipt)}}
    (cfg.output_dir / "request.json").write_text(json.dumps(request))
    monkeypatch.setattr(worker, "_sources", lambda output: {})
    response = worker._finish(cfg, request, {"status": "unavailable"})
    frozen = json.loads(Path(response["frozen_artifact"]["path"]).read_text())
    assert frozen["artifacts"][receipt.name] == worker.file_sha256(receipt)
    assert frozen["artifacts"]["request.json"] == worker.file_sha256(cfg.output_dir / "request.json")


def test_tracker_restores_default_api_on_cancellation(tmp_path, monkeypatch):
    cfg = SimpleNamespace(data_dir=tmp_path / "data", output_dir=tmp_path / "out")
    def original(frames_bgr, center, *, cache_dir=None, source="mapillary", **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(kv, "kartaview_vpr_track", original)
    with pytest.raises(KeyboardInterrupt):
        with cache.track_scope(cfg, "baseline"):
            kv.kartaview_vpr_track([], [42., -83.], cache_dir=str(cfg.data_dir / "mapillary"))
    assert kv.kartaview_vpr_track is original
