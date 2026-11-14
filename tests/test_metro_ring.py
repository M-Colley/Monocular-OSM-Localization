import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm.city_retrieval import metro_ring_regions
from monocular_osm.geo import LocalFrame, geodesic_distance_m
from monocular_osm.stages import prior

CITY = (45.5031824, -73.5698065)
ROOT = Path(__file__).resolve().parents[1]


def east_of(origin, metres):
    return tuple(map(float, LocalFrame(*origin).to_latlon(metres, 0.0)))


def test_ring_cells_cover_the_annulus_and_skip_the_searched_disc():
    regions, audit = metro_ring_regions("Testville", CITY, outer_radius_m=20000.0,
                                        searched_centre=CITY, searched_radius_m=8000.0,
                                        cell_radius_m=4000.0)
    assert audit["plan"] == "metro_ring" and audit["uncovered_polygon_area_m2"] <= 1.0
    assert 30 <= len(regions) <= 50
    assert abs(audit["ring_area_km2"] - np.pi * (20.0 ** 2 - 8.0 ** 2)) < 5.0
    d = np.array([geodesic_distance_m(*CITY, *r["center"]) for r in regions])
    # A cell whose hexagon lies wholly inside the searched disc is not planned.
    assert d.min() > 8000.0 - 4000.0 and d.max() < 20000.0 + 4000.0
    assert {r["origin"] for r in regions} == {"metro_ring"}


def test_seeded_run_excludes_the_seed_disc_not_the_city_disc():
    seed = east_of(CITY, 5000.0)
    regions, audit = metro_ring_regions("Testville", CITY, outer_radius_m=20000.0,
                                        searched_centre=seed, searched_radius_m=6000.0,
                                        cell_radius_m=4000.0)
    assert abs(audit["ring_area_km2"] - np.pi * (20.0 ** 2 - 6.0 ** 2)) < 5.0
    # The city centre itself is outside the seed disc, so it must be covered.
    d = [geodesic_distance_m(*CITY, *r["center"]) for r in regions]
    assert min(d) < 4000.0


def test_gate_is_the_preregistered_selective_threshold():
    rules = json.loads((ROOT / "benchmarks/preregistered_rules.json").read_text(encoding="utf-8"))
    assert prior._METRO_RING_GATE == rules["rules"]["pipeline_selective"]["tau"]
    assert rules["rules"]["pipeline_selective"]["confidence_signal"] == [
        "vpr_coarse_to_fine.pass2_sim", "vpr_prior.track_sim_median"]


def test_confidence_follows_the_selective_signal_order():
    track = (np.arange(3), np.zeros((3, 2)), np.array([0.1, 0.2, 0.3]))
    assert prior.metro_ring_confidence({"pass2_sim": 0.15}, track) == (0.15, "vpr_coarse_to_fine.pass2_sim")
    assert prior.metro_ring_confidence(None, track) == (pytest.approx(0.2), "vpr_prior.track_sim_median")
    assert prior.metro_ring_confidence(None, None) == (None, None)


def test_ring_cells_keep_the_pass1_city_density():
    assert prior.metro_ring_coarse_cap(True) == 300     # 1,200 refs per 8 km disc
    assert prior.metro_ring_coarse_cap(False) == 750    # 3,000 refs per 8 km disc


def _fake_run(tmp_path, monkeypatch, baseline_sims, true_point):
    """A world in which only references near ``true_point`` match the video."""
    import osmnx

    monkeypatch.setattr(osmnx, "geocode", lambda city: CITY)
    n = 12
    timestamps = np.arange(n) * 4.0
    calls = []

    def track_fn(frames, center, radius_m, **kwargs):
        calls.append((tuple(center), radius_m, kwargs.get("cap")))
        near = geodesic_distance_m(*center, *true_point) <= radius_m
        pos = np.repeat([true_point if near else center], n, axis=0)
        return np.arange(n), pos, np.full(n, 0.30 if near else 0.08)

    baseline = (np.arange(n), np.repeat([CITY], n, axis=0), np.asarray(baseline_sims, float))
    cfg = SimpleNamespace(city="Testville", vpr_metro_radius_m=20000.0, vpr_c2f_radius_m=2000.0,
                          vpr_ref_cap=1500, data_dir=tmp_path, vpr_viterbi=True,
                          vpr_query_mode="warp", vpr_query_batch_size=4, vpr_region_cache_dir=None,
                          vpr_city_coarse_cap=250, vpr_city_fine_regions=3,
                          vpr_search_radius_m=3000.0)
    ctx = SimpleNamespace(cfg=cfg, c2f_info={"pass2_sim": float(np.median(baseline_sims))},
                          vpr_track=baseline, vpr_center=CITY, result={}, vpr_listing=None,
                          channel_status={}, osm_around=(*CITY, 2050.0), anchor_origin="vpr",
                          vpr_source_used="mapillary",
                          frames=SimpleNamespace(frames=[None] * n, timestamps=timestamps))
    prior._metro_ring_vpr(ctx, center=CITY, searched_radius_m=8000.0, estimated_length_m=3000.0,
                          source="mapillary", token="set", device="cpu", query_dt_s=4.0,
                          track_fn=track_fn, query_cache={})
    return ctx, calls


def test_confident_track_is_left_alone(tmp_path, monkeypatch):
    ctx, calls = _fake_run(tmp_path, monkeypatch, [0.2] * 12, east_of(CITY, 13000.0))
    assert calls == []
    assert ctx.result["metro_ring_retrieval"]["status"] == "not_triggered"
    assert ctx.osm_around == (*CITY, 2050.0)


def test_weak_track_moves_to_the_ring_cell_that_matches(tmp_path, monkeypatch):
    true_point = east_of(CITY, 13000.0)
    ctx, calls = _fake_run(tmp_path, monkeypatch, [0.12] * 12, true_point)
    record = ctx.result["metro_ring_retrieval"]
    assert record["kept"] == "ring+refine"
    assert geodesic_distance_m(*ctx.osm_around[:2], *true_point) < 50.0
    assert {cap for _, _, cap in calls[:len(calls) - 4]} == {300}       # coarse cells
    assert calls[-1][1] == 2000.0                                       # the refining pass
    baseline = record["baseline_track"]
    assert len(baseline) == 12 and baseline[1] == [4.0, round(CITY[0], 6), round(CITY[1], 6), 0.12]


def test_weak_track_with_no_better_ring_evidence_is_kept(tmp_path, monkeypatch):
    far = east_of(CITY, 60000.0)          # outside every ring cell
    ctx, _ = _fake_run(tmp_path, monkeypatch, [0.12] * 12, far)
    assert ctx.result["metro_ring_retrieval"]["kept"] == "baseline"
    assert ctx.osm_around == (*CITY, 2050.0)
