"""The coarse-prior stages (frame seed, VPR, plate district, VLM) on a hand-built context.

Their model/network backends are replaced by small fakes, so what is under test
is the stage's own decision logic: when it runs, what it publishes on the
context, and how a missing/failed backend is recorded in ``channel_status``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import monocular_osm.kartaview_vpr as kv
import monocular_osm.plate_anchor as plate_anchor
import monocular_osm.stages.prior as prior
import monocular_osm.text_anchor as text_anchor
import monocular_osm.vlm_anchor as vlm_anchor
from monocular_osm.frame_extraction import ExtractedFrames
from monocular_osm.pipeline import PipelineConfig
from monocular_osm.stages.context import RunContext

CENTRE = (48.4, 9.99)


def _cfg(tmp_path: Path, **kw) -> PipelineConfig:
    return PipelineConfig(url="u", city="Ulm, Germany", data_dir=tmp_path / "data",
                          output_dir=tmp_path / "out", **kw)


def _ctx(tmp_path: Path, n: int = 40, **kw) -> RunContext:
    ctx = RunContext.create(_cfg(tmp_path, **kw))
    ctx.video_path = tmp_path / "v.mp4"
    ctx.frames = ExtractedFrames(
        frames=[np.zeros((8, 8, 3), np.uint8) for _ in range(n)],
        timestamps=[float(i) for i in range(n)], fps=30.0, total_frames=n * 3)
    ctx.estimated_length_m = 1000.0
    return ctx


def _track(n: int = 20, centre=CENTRE, spread_deg: float = 0.002):
    idx = np.arange(n) * 2
    ll = np.c_[centre[0] + np.linspace(-spread_deg, spread_deg, n),
               centre[1] + np.linspace(-spread_deg, spread_deg, n)]
    return idx, ll, np.full(n, 0.7)


# ---------------------------------------------------------------------------
# coarse_seed_from_frames
# ---------------------------------------------------------------------------


def test_the_frame_seed_becomes_the_search_disc(tmp_path, monkeypatch):
    calls = []

    def fake_seed(video, frames, city, **kw):
        calls.append((city, kw["cache_dir"], kw["ocr_engine"]))
        return (48.4, 9.99, 1500.0, "plate:UL")

    monkeypatch.setattr(prior, "_frame_coarse_seed", fake_seed)
    ctx = _ctx(tmp_path)
    prior.coarse_seed_from_frames(ctx)
    assert ctx.osm_around == (48.4, 9.99, 1500.0)
    assert ctx.frame_seed_source == "plate:UL"
    assert calls == [("Ulm, Germany", ctx.cfg.data_dir, "rapidocr")]


def test_an_explicit_disc_wins_over_the_frame_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(prior, "_frame_coarse_seed",
                        lambda *a, **k: pytest.fail("a disc was given"))
    ctx = _ctx(tmp_path, osm_around=(48.0, 9.0, 800.0))
    prior.coarse_seed_from_frames(ctx)
    assert ctx.osm_around == (48.0, 9.0, 800.0) and ctx.frame_seed_source is None


def test_frame_seeding_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(prior, "_frame_coarse_seed",
                        lambda *a, **k: pytest.fail("seeding is disabled"))
    ctx = _ctx(tmp_path, coarse_from_frames=False)
    prior.coarse_seed_from_frames(ctx)
    assert ctx.osm_around is None


def test_no_frame_seed_leaves_the_disc_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(prior, "_frame_coarse_seed", lambda *a, **k: None)
    ctx = _ctx(tmp_path)
    prior.coarse_seed_from_frames(ctx)
    assert ctx.osm_around is None and ctx.frame_seed_source is None


# ---------------------------------------------------------------------------
# vpr_coarse_prior
# ---------------------------------------------------------------------------


def _vpr_ctx(tmp_path, monkeypatch, tracks, **kw) -> tuple[RunContext, list]:
    """A context whose VPR backend yields ``tracks`` in call order."""
    calls: list = []
    it = iter(tracks)

    def fake_track(frames, center, radius_m=3000.0, **k):
        calls.append({"center": center, "radius_m": radius_m, "cap": k.get("cap"),
                      "source": k.get("source")})
        return next(it)

    monkeypatch.setattr(kv, "kartaview_vpr_track", fake_track)
    kw.setdefault("vpr_source", "kartaview")
    kw.setdefault("osm_around", (CENTRE[0], CENTRE[1], 900.0))
    ctx = _ctx(tmp_path, **kw)
    ctx.osm_around = ctx.cfg.osm_around
    return ctx, calls


def test_vpr_publishes_the_centre_track_and_origin(tmp_path, monkeypatch):
    ctx, calls = _vpr_ctx(tmp_path, monkeypatch, [_track()])
    prior.vpr_coarse_prior(ctx)
    assert ctx.anchor_origin == "vpr" and ctx.vpr_source_used == "kartaview"
    assert ctx.vpr_track is not None and len(ctx.vpr_track[0]) == 20
    assert ctx.vpr_center == pytest.approx(CENTRE, abs=0.003)
    # a seeded disc narrower than the default radius caps the search disc
    assert calls[0]["radius_m"] == 900.0
    assert "vpr" not in ctx.channel_status


def test_vpr_that_finds_nothing_is_recorded_as_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(kv, "LAST_FAILURE", "no reference track", raising=False)
    ctx, _ = _vpr_ctx(tmp_path, monkeypatch, [None])
    prior.vpr_coarse_prior(ctx)
    assert ctx.vpr_center is None and ctx.vpr_track is None
    assert ctx.channel_status["vpr"]["status"] == "inactive"


def test_a_vpr_backend_that_raises_is_recorded_as_failed(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(kv, "kartaview_vpr_track", boom)
    ctx = _ctx(tmp_path, vpr_source="kartaview", osm_around=(CENTRE[0], CENTRE[1], 900.0))
    ctx.osm_around = ctx.cfg.osm_around
    prior.vpr_coarse_prior(ctx)
    assert ctx.vpr_center is None
    assert ctx.channel_status["vpr"]["status"] == "failed"
    assert "model download failed" in ctx.channel_status["vpr"]["error"]


def test_vpr_can_be_disabled(tmp_path, monkeypatch):
    ctx, calls = _vpr_ctx(tmp_path, monkeypatch, [], use_vpr_prior=False)
    prior.vpr_coarse_prior(ctx)
    assert calls == [] and ctx.vpr_center is None and ctx.channel_status == {}


def test_coarse_to_fine_recentres_the_graph_on_the_refined_prior(tmp_path, monkeypatch):
    """A wide pass-1 disc is re-fetched tight around its own centre; the OSM
    graph disc follows the kept centre."""
    pass2_centre = (CENTRE[0] + 0.001, CENTRE[1] + 0.001)
    ctx, calls = _vpr_ctx(
        tmp_path, monkeypatch, [_track(), _track(centre=pass2_centre)],
        osm_around=(CENTRE[0], CENTRE[1], 5000.0), vpr_search_radius_m=5000.0,
        vpr_coarse_to_fine=True, vpr_c2f_radius_m=1500.0)
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 2 and calls[1]["radius_m"] == 1500.0
    assert ctx.c2f_info["kept"] == "pass-2"
    assert ctx.vpr_center == pytest.approx(pass2_centre, abs=0.003)
    la, lo, rad = ctx.osm_around
    assert (la, lo) == pytest.approx(pass2_centre, abs=0.003)
    assert rad == pytest.approx(1500.0 + prior._C2F_SNAP_TOL_M)


def test_coarse_to_fine_keeps_pass_one_when_pass_two_matches_worse(tmp_path, monkeypatch):
    weak = (np.arange(20) * 2, _track()[1], np.full(20, 0.3))
    ctx, _ = _vpr_ctx(
        tmp_path, monkeypatch, [_track(), weak],
        osm_around=(CENTRE[0], CENTRE[1], 5000.0), vpr_search_radius_m=5000.0,
        vpr_coarse_to_fine=True, vpr_c2f_radius_m=1500.0)
    prior.vpr_coarse_prior(ctx)
    assert ctx.c2f_info["kept"].startswith("pass-1")
    assert float(np.median(ctx.vpr_track[2])) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# plate_district_prior
# ---------------------------------------------------------------------------


def _plate(**kw):
    base = dict(code="UL", district="Ulm", lat=48.4, lon=9.99, votes=5, total_unique=6,
                margin=2.34567, tally={"UL": 5, "NU": 1})
    base.update(kw)
    return SimpleNamespace(**base)


def test_plate_prior_publishes_the_district_centre(tmp_path, monkeypatch):
    monkeypatch.setattr(plate_anchor, "plate_district_anchor", lambda path: _plate())
    ctx = _ctx(tmp_path, use_plate_anchor=True)
    prior.plate_district_prior(ctx)
    assert ctx.plate_center == (48.4, 9.99)
    assert ctx.result_plate_anchor["code"] == "UL"
    assert ctx.result_plate_anchor["margin"] == 2.35           # rounded for the report


def test_plate_prior_without_a_confident_district_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(plate_anchor, "plate_district_anchor", lambda path: None)
    ctx = _ctx(tmp_path, use_plate_anchor=True)
    prior.plate_district_prior(ctx)
    assert ctx.plate_center is None and ctx.result_plate_anchor is None
    assert ctx.channel_status == {}


def test_a_failing_plate_reader_is_recorded(tmp_path, monkeypatch):
    def boom(path):
        raise OSError("no video")

    monkeypatch.setattr(plate_anchor, "plate_district_anchor", boom)
    ctx = _ctx(tmp_path, use_plate_anchor=True)
    prior.plate_district_prior(ctx)
    assert ctx.channel_status["plate"]["status"] == "failed"


def test_plate_prior_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(plate_anchor, "plate_district_anchor",
                        lambda path: pytest.fail("not enabled"))
    prior.plate_district_prior(_ctx(tmp_path))


# ---------------------------------------------------------------------------
# vlm_district_prior
# ---------------------------------------------------------------------------


def _vlm(**kw):
    base = dict(lat=48.41, lon=9.98, label="Muenster", street_votes=2, district_votes=3,
                text_votes=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_vlm_prior_fills_in_when_vpr_found_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_anchor, "vlm_district_anchor", lambda *a, **k: _vlm())
    monkeypatch.setattr(text_anchor, "default_geocode_fn", lambda path: None)
    ctx = _ctx(tmp_path, use_vlm_anchor=True)
    prior.vlm_district_prior(ctx)
    assert ctx.vpr_center == (48.41, 9.98) and ctx.anchor_origin == "vlm"
    assert ctx.result_vlm_anchor["applied_as_prior"] is True


def test_vlm_prior_is_a_fallback_only(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_anchor, "vlm_district_anchor",
                        lambda *a, **k: pytest.fail("VPR already located the clip"))
    ctx = _ctx(tmp_path, use_vlm_anchor=True)
    ctx.vpr_center, ctx.anchor_origin = (48.4, 9.99), "vpr"
    prior.vlm_district_prior(ctx)
    assert ctx.anchor_origin == "vpr" and ctx.result_vlm_anchor is None


def test_vlm_prior_without_a_geocodable_place_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_anchor, "vlm_district_anchor", lambda *a, **k: None)
    monkeypatch.setattr(text_anchor, "default_geocode_fn", lambda path: None)
    ctx = _ctx(tmp_path, use_vlm_anchor=True)
    prior.vlm_district_prior(ctx)
    assert ctx.vpr_center is None and ctx.result_vlm_anchor is None


# ---------------------------------------------------------------------------
# retrieval device: a nearly-full GPU falls back to the CPU
# ---------------------------------------------------------------------------


def test_retrieval_runs_on_cpu_when_gpu_memory_is_short(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2e9, 24e9))
    assert prior.select_retrieval_device() == "cpu"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (12e9, 24e9))
    assert prior.select_retrieval_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert prior.select_retrieval_device() == "cpu"


def test_backbone_info_reports_the_device(monkeypatch) -> None:
    monkeypatch.setattr(kv, "_MODEL_NAME", "megaloc")
    monkeypatch.setattr(kv, "_MODEL_DEVICE", "cpu")
    assert kv.backbone_info()["device"] == "cpu"
