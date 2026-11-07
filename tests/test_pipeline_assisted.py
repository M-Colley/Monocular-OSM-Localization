import json
import pytest
from monocular_osm import pipeline
from monocular_osm import assisted_worker as worker
from monocular_osm.cli import build_arg_parser
from monocular_osm.kartaview_vpr import HUB_MEGALOC


def config(tmp_path, **kwargs):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"mock encoded source")
    return pipeline.PipelineConfig(url="", city="Example City", data_dir=tmp_path / "data/clip",
        output_dir=tmp_path / "output", video_path=video, answer="assisted", vo_end_sec=3., **kwargs)


def baseline():
    route = [[49.001, 8.401], [49.002, 8.402], [49.003, 8.403], [49.004, 8.404]]
    return {"status": "complete", "route_latlon": route, "route_timestamps": [0., 1., 2., 3.],
        "comparison": {"statistic": "raw_cosine_at_decoded_reference",
            "model": {"name": "megaloc", "hub": HUB_MEGALOC, "weights_sha256": "1"*64},
            "query": {"indices": [0, 1, 2, 3], "times": [0., 1., 2., 3.], "mode": "warp",
                      "preprocessing": {"size": [512, 512]}, "pixels_sha256": "2"*64, "descriptors_sha256": "3"*64},
            "inner_decoder": {"actual": "viterbi", "fallback": False, "settings": {"free_speed": 40}},
            "decoded_reference_cosines": [.21]*4, "decoded_reference_latlon": route}}


def test_assisted_alias_selects_experimental_mode():
    args = build_arg_parser().parse_args(["--video", "video.mp4", "--city", "Example City", "--assisted-only"])
    assert args.answer == "assisted"


def test_normal_pipeline_preserves_baseline_and_emits_selected_map_and_freeze(tmp_path, monkeypatch):
    cfg, calls = config(tmp_path), []
    original = baseline()
    def child(kind, ctx):
        calls.append(kind)
        assert kind == "baseline", "strong baseline must not start future children"
        child_cfg = worker.child_config(cfg, kind, ctx.video_path)
        child_cfg.output_dir = ctx.assisted_child_root / kind
        child_cfg.output_dir.mkdir(parents=True)
        response = worker._finish(child_cfg, {"kind": kind,
            "video_sha256": ctx.result["assisted_inputs"]["video"]["sha256"],
            "source_sha256": ctx.result["assisted_inputs"]["source_sha256"]}, original)
        return {**response, "worker_returncode": 0}
    monkeypatch.setattr(worker, "run_child", child)
    monkeypatch.setattr(pipeline, "extract_video_frames", lambda *_a, **_k: pytest.fail("parent decoded future frames"))
    monkeypatch.setattr(pipeline, "vpr_coarse_prior", lambda *_a: pytest.fail("parent ran ordinary retrieval"))
    result = pipeline.run_pipeline(cfg)
    assert calls == ["baseline"]
    assert result["answer"]["mode"] == "assisted" and result["answer"]["method"] == "assisted_policy"
    assert result["answer"]["route_latlon"] == original["route_latlon"]
    assert result["answer"]["route_t_sec"] == original["route_timestamps"]
    assert result["answer"]["experimental"] is True and result["answer"]["confidence_calibrated"] is False
    assert (cfg.output_dir / "answer.png").is_file()
    frozen = json.loads((cfg.output_dir / "frozen_prediction.json").read_text())
    assert frozen["scoring_has_not_run"] and frozen["prediction_available"]
    assert result["provenance"]["assisted_audit"]["selected_candidate"] == "baseline"
    assert result["provenance"]["assisted_audit"]["selected_child_artifact_sha256"]


@pytest.mark.parametrize("kwargs", [{"max_frames": 80}, {"frame_max_width": 1280},
    {"osm_around": (49., 8., 1000.)}, {"vpr_query_mode": "crops"}, {"ocr_engine": "easyocr"},
    {"graph_filter": True}, {"use_ipm_scale": True}, {"poi_radius_m": 1500.}, {"vpr_source": "union"}])
def test_assisted_rejects_incompatible_inputs(tmp_path, kwargs):
    with pytest.raises(ValueError):
        config(tmp_path, **kwargs)


@pytest.mark.parametrize("region_status,admitted", [("ok", True), ("failed", False), ("unavailable", False)])
def test_metadata_admission_uses_fixed_policy_and_complete_search(tmp_path, monkeypatch, region_status, admitted):
    from types import SimpleNamespace
    import numpy as np
    from monocular_osm.stages.assisted import summary
    cfg = config(tmp_path)
    cfg.output_dir.mkdir()
    weak, expanded = baseline(), baseline()
    weak["comparison"]["decoded_reference_cosines"] = [.12]*4
    monkeypatch.setattr(worker, "_comparison", lambda *_: expanded["comparison"])
    ctx = SimpleNamespace(cfg=cfg, vpr_track=(np.arange(4), np.array(expanded["route_latlon"]), np.full(4, .21)),
        retrieval_audits=[], result={"multi_region_retrieval": {"regions": [{"status": region_status}]}})
    request = {"policy_evidence": {"baseline": summary("baseline", weak), "cues": {"status": "failed"}}}
    assert worker._metadata_admission(ctx, request) is admitted
    record = ctx.result["assisted_expansion_gate"]
    assert record["municipality_skipped"] is admitted
    assert (cfg.output_dir / "metadata_admission/admission.json").is_file()


@pytest.mark.parametrize("count", [0, 1])
def test_completed_sparse_listing_retains_plan_and_uncertain_none_declines(tmp_path, count):
    from monocular_osm import kartaview_vpr as kv
    from monocular_osm.stages.prior import _terminal_sparse_search
    center, radius, cap = [49., 8.], 1000., 500
    refs = [{"id": "public-photo", "lat": 49., "lon": 8.}] if count else []
    signature = kv._fetch_signature(center, radius, cap)
    if count:
        (tmp_path / f"mly_ref_meta_{kv._sig_tag(signature)}.json").write_text(json.dumps({"signature": signature, "refs": refs}))
    request = {"source": "mapillary", "radius_m": radius, "cap": cap, "cache_dir": str(tmp_path)}
    stats = {"listing": "tiles", "listing_stats_known": True, "lost": 0, "snapshot": kv._reference_snapshot(refs)}
    reason = f"only {count} mapillary reference images within 1000 m (need 30)"
    proof = _terminal_sparse_search(center, request, stats, reason)
    record = {"status": "unavailable", "center": center, "radius_m": radius, "ref_cap": cap,
              "track_decoding": {"terminal_search": proof}}
    assert worker._completed_region(record)
    assert not worker._completed_region({**record, "status": "failed"})
    for changes in ({"listing": "graph"}, {"listing": None}, {"truncated": True}):
        assert _terminal_sparse_search(center, request, {**stats, **changes}, reason) is None
        stale = {**proof, "listing": {**proof["listing"], **changes}}
        assert not worker._valid_terminal(stale, record)
    assert _terminal_sparse_search(center, request, {**stats, "lost": 1}, reason) is None
    assert not worker._completed_region({**record, "track_decoding": {}})


def test_sparse_listing_proof_counts_listed_members_so_url_less_ones_change_nothing(tmp_path):
    """The terminal proof binds LISTED rows (the decoder minimum is on listed refs);
    members without a thumbnail URL are listed, so they never create a shortfall."""
    from monocular_osm import kartaview_vpr as kv
    from monocular_osm.stages.prior import _terminal_sparse_search
    center, radius, cap = [49., 8.], 1000., 500
    refs = [{"id": f"photo{i}", "lat": 49., "lon": 8. + i * 1e-4} for i in range(3)]
    signature = kv._fetch_signature(center, radius, cap)
    (tmp_path / f"mly_ref_meta_{kv._sig_tag(signature)}.json").write_text(json.dumps({"signature": signature, "refs": refs}))
    request = {"source": "mapillary", "radius_m": radius, "cap": cap, "cache_dir": str(tmp_path)}
    stats = {"listing": "tiles", "listing_stats_known": True, "lost": 0, "snapshot": kv._reference_snapshot(refs),
             "url_lookups": 3, "url_missing": 2, "url_missing_refs": refs[1:]}
    proof = _terminal_sparse_search(center, request, stats, "only 3 mapillary reference images within 1000 m (need 30)")
    record = {"status": "unavailable", "center": center, "radius_m": radius, "ref_cap": cap,
              "track_decoding": {"terminal_search": proof}}
    assert proof["count"] == 3 and proof["listing"]["url_missing"] == 2
    assert worker._valid_terminal(proof, record) and worker._completed_region(record)
