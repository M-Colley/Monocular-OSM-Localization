"""The report stage: diagnostics, channel summary, provenance and result.json."""

from __future__ import annotations

import json

import numpy as np
import pytest

import monocular_osm.stages.report as report
from monocular_osm.stages.context import RunContext

from ._stage_fixtures import make_ctx


# ---------------------------------------------------------------------------
# finalize_result
# ---------------------------------------------------------------------------


def _report_ctx(tmp_path, **kw) -> RunContext:
    ctx = make_ctx(tmp_path, **kw)
    ctx.cache_path = tmp_path / "data" / "Ulm_Germany.graphml"
    ctx.vo_cache = tmp_path / "data" / "trajectory.npz"
    return ctx


def test_result_json_is_written_with_provenance_and_the_osm_snapshot(tmp_path):
    ctx = _report_ctx(tmp_path)
    ctx.road.graph.graph["created_date"] = "2026-10-01"
    report.finalize_result(ctx)
    on_disk = json.loads((ctx.cfg.output_dir / "result.json").read_text(encoding="utf-8"))
    assert on_disk["city"] == "Ulm, Germany"
    prov = on_disk["provenance"]
    assert prov["osm_graph"]["created_date"] == "2026-10-01"
    assert prov["osm_graph"]["nodes"] == ctx.road.graph.number_of_nodes()
    assert prov["vo_cache"].endswith("trajectory.npz") and "vo_algo_version" in prov
    assert prov["config"]["city"] == "Ulm, Germany"


def test_unrun_channels_are_reported_as_disabled(tmp_path):
    ctx = _report_ctx(tmp_path)
    report.finalize_result(ctx)
    ch = ctx.result["channels"]
    assert ch["vpr"] == {"status": "inactive"}                # enabled by default, nothing located
    assert ch["plate"] == {"status": "disabled"} and ch["sun"] == {"status": "disabled"}
    assert ch["ocr_anchor"] == {"status": "disabled"}


def test_enabled_channels_are_ok_inactive_or_failed(tmp_path):
    ctx = _report_ctx(tmp_path, use_plate_anchor=True, use_vlm_anchor=True)
    ctx.vpr_center, ctx.anchor_origin, ctx.vpr_source_used = (48.4, 9.99), "vpr", "kartaview"
    ctx.vpr_track = (np.arange(5), np.zeros((5, 2)), np.array([0.1, 0.2, 0.25, 0.3, 0.4]))
    ctx.channel_status["sun"] = {"status": "failed", "error": "RuntimeError: x"}   # pre-recorded
    ctx.cfg.use_sun_heading = True
    report.finalize_result(ctx)
    ch = ctx.result["channels"]
    assert ch["vpr"] == {"status": "ok"}
    assert ch["plate"] == {"status": "inactive"}              # enabled, produced nothing
    assert ch["sun"]["status"] == "failed"                    # a recorded failure is never overwritten
    vp = ctx.result["vpr_prior"]
    assert vp["origin"] == "vpr" and vp["source"] == "kartaview" and vp["track_frames"] == 5
    assert vp["track_sim_median"] == 0.25                     # retrieval confidence, recorded
    assert len(vp["track"]) == 5 and vp["track"][2][1:] == [0.0, 0.0, 0.25]   # [t, lat, lon, sim]


def test_diagnostics_are_attached_even_when_nothing_matched(tmp_path):
    ctx = _report_ctx(tmp_path)
    ctx.candidates, ctx.result = [], {"city": "Ulm, Germany", "matches": []}
    ctx.result_sun_heading = {"available": False, "reason": "no clock", "headings": np.zeros(3)}
    ctx.c2f_info = {"kept": "pass-2"}
    ctx.result_scale_recovery = {"status": "declined"}
    ctx.result_loop_closure = {"closed": False}
    ctx.result_vggt = {"poses": 4, "seed_nodes": 2}
    ctx.result_ipm_scale = {"status": "ok", "length_m": 900.0}
    report.finalize_result(ctx)
    r = ctx.result
    assert r["sun_heading"] == {"available": False, "reason": "no clock"}   # numpy array dropped
    assert r["vpr_coarse_to_fine"] == {"kept": "pass-2"} and r["vggt_gating"]["seed_nodes"] == 2
    assert r["scale_recovery"]["status"] == "declined" and r["ipm_scale"]["length_m"] == 900.0
    assert r["loop_closure"] == {"closed": False}
    assert (ctx.cfg.output_dir / "result.json").exists()


def test_an_enabled_plate_anchor_that_found_nothing_says_so(tmp_path):
    ctx = _report_ctx(tmp_path, use_plate_anchor=True)
    report.finalize_result(ctx)
    assert ctx.result["plate_anchor"] == {"applied": False, "status": "no_confident_district"}


def test_a_plate_anchor_that_ran_but_was_not_fused_is_marked_unapplied(tmp_path):
    ctx = _report_ctx(tmp_path, use_plate_anchor=True)
    ctx.result_plate_anchor = {"code": "UL"}
    report.finalize_result(ctx)
    assert ctx.result["plate_anchor"] == {"code": "UL", "applied": False}


def test_fusion_channels_are_summarised_by_their_effective_weight(tmp_path):
    ctx = _report_ctx(tmp_path)
    ctx.result["fusion"] = {"weights_effective": {"shape": 1.0, "turn": 0.0}}
    report.finalize_result(ctx)
    assert ctx.result["channels"]["fusion_shape"] == {"status": "ok", "weight": 1.0}
    assert ctx.result["channels"]["fusion_turn"] == {"status": "inactive", "weight": 0.0}


def _answer_ctx(tmp_path, answer, fit_lon):
    ctx = _report_ctx(tmp_path, answer=answer)
    ctx.result["position"] = {"route_latlon": [[48.4000, fit_lon], [48.4040, fit_lon]]}
    ctx.vpr_center, ctx.anchor_origin, ctx.vpr_source_used = (48.402, 9.98), "vpr", "mapillary"
    ctx.vpr_track = (np.arange(9), np.c_[48.400 + 0.0005 * np.arange(9), np.full(9, 9.98)],
                     np.full(9, 0.2))
    return ctx


def test_answer_follows_the_configured_mode(tmp_path):
    for mode, fit_lon, method in (("fit", 9.99, "pipeline"),
                                  ("track", 9.9801, "vpr_track_only"),
                                  ("checked", 9.9801, "pipeline"),      # agrees: the fit
                                  ("checked", 9.99, "vpr_track_only")):  # 740 m off: the track
        ctx = _answer_ctx(tmp_path / f"{mode}{fit_lon}", mode, fit_lon)
        report.finalize_result(ctx)
        a = ctx.result["answer"]
        assert a["mode"] == mode and a["method"] == method, (mode, fit_lon, a)
        assert ctx.result["position"]["route_latlon"][0][1] == fit_lon   # the fit is kept
    assert json.loads((ctx.cfg.output_dir / "result.json").read_text(encoding="utf-8"))["answer"]


def test_track_answer_without_a_track_is_the_fit(tmp_path):
    ctx = _answer_ctx(tmp_path, "track", 9.99)
    ctx.vpr_track = None
    report.finalize_result(ctx)
    assert ctx.result["answer"]["method"] == "pipeline"


@pytest.mark.parametrize("mode,fit_lon,chosen_lon", [
    ("track", 9.99, 9.98),
    ("fit", 9.99, 9.99),
    ("checked", 9.9801, 9.9801),
    ("checked", 9.99, 9.98),
])
def test_primary_console_links_and_map_use_the_selected_answer(
        tmp_path, capsys, monkeypatch, mode, fit_lon, chosen_lon):
    ctx = _answer_ctx(tmp_path, mode, fit_lon)
    plotted = []
    from monocular_osm import position

    real_plot = position.plot_answer

    def capture_plot(answer, path, **kw):
        plotted.append(answer["route_latlon"])
        return real_plot(answer, path, **kw)

    monkeypatch.setattr(position, "plot_answer", capture_plot)
    report.finalize_result(ctx)
    a = ctx.result["answer"]
    console = capsys.readouterr().out
    assert console.count("ESTIMATED POSITION") == 1
    assert f"answer={mode}" in console
    assert f"{chosen_lon:.6f}" in console
    assert f"q={a['latitude']:.6f},{chosen_lon:.6f}" in console
    assert f"mlon={chosen_lon:.6f}" in console
    assert a["start"]["longitude"] == chosen_lon
    assert plotted == [a["route_latlon"]]
    assert a["map_artifact"] == "answer.png"
    assert (ctx.cfg.output_dir / "answer.png").read_bytes().startswith(b"\x89PNG")
    assert ctx.result["position"]["route_latlon"][0][1] == fit_lon


def test_retrieval_answer_is_reported_without_candidates_or_a_road(tmp_path, capsys):
    ctx = _answer_ctx(tmp_path, "track", 9.99)
    del ctx.result["position"]
    ctx.candidates, ctx.road, ctx.cache_path = [], None, None
    report.finalize_result(ctx)
    saved = json.loads((ctx.cfg.output_dir / "result.json").read_text(encoding="utf-8"))
    assert saved["answer"]["method"] == "vpr_track_only"
    assert saved["answer"]["map_artifact"] == "answer.png"
    assert saved["provenance"]["osm_graph"] is None
    assert "ESTIMATED POSITION" in capsys.readouterr().out


def test_map_failure_keeps_the_route_and_removes_a_stale_map(tmp_path, monkeypatch):
    ctx = _answer_ctx(tmp_path, "track", 9.99)
    stale = ctx.cfg.output_dir / "answer.png"
    stale.write_bytes(b"old map")
    from monocular_osm import position

    def fail_plot(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(position, "plot_answer", fail_plot)
    report.finalize_result(ctx)
    a = ctx.result["answer"]
    assert a["route_latlon"] and "disk unavailable" in a["map_error"]
    assert "map_artifact" not in a and not stale.exists()
    assert json.loads((ctx.cfg.output_dir / "result.json").read_text(encoding="utf-8"))["answer"]


def test_no_selected_answer_removes_a_previous_map(tmp_path):
    ctx = _report_ctx(tmp_path)
    stale = ctx.cfg.output_dir / "answer.png"
    stale.write_bytes(b"old map")
    report.finalize_result(ctx)
    assert "answer" not in ctx.result and not stale.exists()


def test_checkpoint_does_not_freeze_derived_channel_statuses(tmp_path):
    ctx = _report_ctx(tmp_path, use_plate_anchor=True)
    report.finalize_result(ctx)
    assert ctx.result["channels"]["plate"]["status"] == "inactive"
    assert "plate" not in ctx.channel_status
    ctx.plate_center = (48.4, 9.99)
    report.finalize_result(ctx)
    assert ctx.result["channels"]["plate"]["status"] == "ok"


def test_failed_json_write_preserves_the_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"answer": "usable checkpoint"}', encoding="utf-8")

    def interrupted_dump(result, stream, **kwargs):
        stream.write('{"answer":')
        raise OSError("disk write failed")

    monkeypatch.setattr(report.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="disk write failed"):
        report._write_result_json(path, {"answer": "new"})
    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == "usable checkpoint"
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_does_not_print_a_second_primary_position(tmp_path, capsys):
    ctx = _answer_ctx(tmp_path, "track", 9.99)
    ctx.result["execution"] = {"stage": "retrieval", "geometry": "pending"}
    report.finalize_result(ctx)
    checkpoint_console = capsys.readouterr().out
    assert "checkpointed" in checkpoint_console and "ESTIMATED POSITION" not in checkpoint_console
    ctx.result["execution"]["stage"] = "complete"
    report.finalize_result(ctx)
    assert capsys.readouterr().out.count("ESTIMATED POSITION") == 1


def test_frame_sampling_provenance_survives_a_retrieval_only_report(tmp_path):
    ctx = _report_ctx(tmp_path)
    ctx.road = None
    ctx.frames.sampling = {"mode": "linspace", "dense_count": 300,
                           "dense_indices": [0, 149, 299], "sample_count": 3}
    report.finalize_result(ctx)
    saved = json.loads((ctx.cfg.output_dir / "result.json").read_text(encoding="utf-8"))
    assert saved["frame_sampling"] == ctx.frames.sampling
