"""Whole-city planning and hierarchical selection without models/network/GT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from monocular_osm.city_retrieval import (
    CityBoundaryTypeError, CityCoverageLimitError, MAX_CITY_CELLS, municipality_regions,
    plan_city_regions, select_city_tracks,
)
from monocular_osm.geo import geodesic_distance_m


def _regions(n=4):
    return [{"name": f"cell{i}", "center": [50.0, 8.0 + i * .01],
             "radius_m": 3000, "origin": "municipality_grid", "cell_axial": [i, 0]}
            for i in range(n)]


def _track(similarities, *, far=False, indices=None):
    sims = np.asarray(similarities, float)
    indices = np.arange(len(sims)) if indices is None else np.asarray(indices)
    coords = np.c_[np.full(len(sims), 50.0 + (1.0 if far else 0)),
                   8.0 + np.arange(len(sims)) * .0001]
    return indices, coords, sims


def test_city_grid_covers_concave_polygon_with_actual_geodesic_discs():
    city = Polygon([(-.12, -.1), (.12, -.1), (.12, 0), (0, 0), (0, .1), (-.12, .1)])
    regions, audit = municipality_regions(city, "Synthetic city")
    assert 1 < len(regions) <= MAX_CITY_CELLS
    assert audit["uncovered_polygon_area_m2"] <= 1
    samples = np.array([(lo, la) for lo in np.linspace(-.12, .12, 31)
                        for la in np.linspace(-.1, .1, 31) if city.covers(Point(lo, la))])
    distances = np.min([geodesic_distance_m(samples[:, 1], samples[:, 0], *r["center"])
                        for r in regions], axis=0)
    assert distances.max() <= 3000
    assert regions == municipality_regions(city, "Synthetic city")[0]


def test_full_city_limit_is_an_explicit_refusal_not_a_truncated_grid():
    with pytest.raises(CityCoverageLimitError, match="full-city plan refused") as exc:
        municipality_regions(box(-1, -1, 1, 1), "Too large")
    assert exc.value.required_cells > MAX_CITY_CELLS
    assert exc.value.max_cells == MAX_CITY_CELLS


def test_city_polygon_cache_is_reused_and_fingerprinted(tmp_path):
    calls = []

    def boundary(city):
        calls.append(city)
        return box(-.02, -.02, .02, .02), {"query": city, "osm_id": 123, "osm_type": "relation",
                                        "class": "boundary", "type": "administrative",
                                        "addresstype": "city", "place_rank": 16}

    first, info = plan_city_regions("Small city", tmp_path, boundary_fn=boundary)
    second, cached = plan_city_regions("Small city", tmp_path, boundary_fn=lambda *_: pytest.fail("cached"))
    assert first == second and calls == ["Small city"]
    assert not info["boundary_cache_hit"] and cached["boundary_cache_hit"]
    assert info["boundary_sha256"] == cached["boundary_sha256"]
    assert cached["boundary_source"]["osm_id"] == 123
    assert cached["boundary_validation"]["administrative_boundary_verified"] is True
    blob = json.loads(next(tmp_path.glob("*.geojson")).read_text())
    assert blob["geometry"]["type"] == "Polygon"


def test_invalid_city_boundary_is_not_published_as_a_cache(tmp_path):
    with pytest.raises(ValueError, match="municipality polygon"):
        plan_city_regions("Not a polygon", tmp_path, boundary_fn=lambda _: (
            Point(0, 0), {"class": "boundary", "type": "administrative"}))
    assert not list(tmp_path.glob("*.geojson"))


def test_nominatim_boundary_source_preserves_taxonomy(monkeypatch):
    import geopandas as gpd
    import osmnx as ox
    from monocular_osm.city_retrieval import _city_boundary

    row = gpd.GeoDataFrame({"osm_id": [42], "osm_type": ["relation"],
                           "class": ["boundary"], "type": ["administrative"],
                           "addresstype": ["city"], "place_rank": [16]},
                          geometry=[box(0, 0, .01, .01)], crs=4326)
    monkeypatch.setattr(ox, "geocode_to_gdf", lambda city, which_result: row)
    _geometry, source = _city_boundary("Named city")
    assert {k: source[k] for k in ("class", "type", "addresstype", "place_rank")} == {
        "class": "boundary", "type": "administrative", "addresstype": "city", "place_rank": 16}


@pytest.mark.parametrize("policy", ["refuse", "allow_unverified"])
def test_cached_nonmunicipal_polygon_is_refused_without_refetch_or_rewrite(tmp_path, policy):
    import hashlib
    from shapely.geometry import mapping

    city = "Named peninsula"
    path = tmp_path / f"city_{hashlib.sha256(city.encode()).hexdigest()[:16]}.geojson"
    blob = {"schema": 1, "city": city, "geometry": mapping(box(-.02, -.02, .02, .02)),
            "source": {"osm_id": 3944756, "class": "leisure", "type": "nature_reserve",
                       "addresstype": "nature_reserve", "place_rank": 24,
                       "display_name": "Synthetic watershed"}}
    original = json.dumps(blob).encode()
    path.write_bytes(original)
    with pytest.raises(CityBoundaryTypeError, match="not an administrative boundary") as exc:
        plan_city_regions(city, tmp_path, legacy_unknown_policy=policy,
                          boundary_fn=lambda _: pytest.fail("cached snapshot must not refetch"))
    assert exc.value.validation["status"] == "non_administrative"
    assert exc.value.source["place_rank"] == 24
    assert path.read_bytes() == original


def test_legacy_missing_taxonomy_requires_explicit_unverified_policy(tmp_path):
    boundary = lambda _: (box(-.02, -.02, .02, .02), {"osm_id": 123})
    with pytest.raises(CityBoundaryTypeError, match="classification is missing"):
        plan_city_regions("Legacy city", tmp_path, boundary_fn=boundary)
    assert not list(tmp_path.glob("*.geojson"))
    regions, audit = plan_city_regions("Legacy city", tmp_path, boundary_fn=boundary,
                                       legacy_unknown_policy="allow_unverified")
    assert regions and audit["boundary_validation"]["status"] == "classification_unknown"
    assert audit["boundary_validation"]["administrative_boundary_verified"] is False
    with pytest.raises(CityBoundaryTypeError, match="policy=refuse"):
        plan_city_regions("Legacy city", tmp_path, boundary_fn=lambda _: pytest.fail("cached"))


def test_searches_all_coarse_cells_and_densifies_only_three_supported_cells():
    similarities = [
        [.4, .4, .1, .1, .1, .1, .1, .1],
        [.1, .1, .5, .5, .5, .1, .1, .1],
        [.1, .1, .1, .1, .1, .6, .6, .1],
        [.1, .1, .1, .1, .1, .1, .1, .7],
    ]
    calls = []

    def retrieve(region, *, cap, phase, index):
        calls.append((index, phase, cap))
        sim = np.array(similarities[index]) + (.1 if phase == "fine" else 0)
        return _track(sim), {"cache": f"{phase}/{index}", "source": "fake-source"}

    baseline = _track(np.full(8, .05))
    merged, info = select_city_tracks(_regions(), baseline, np.arange(8) * 10, retrieve)
    assert calls[:4] == [(i, "coarse", 250) for i in range(4)]
    assert len(calls[4:]) == 3 and all(c[1:] == ("fine", 1500) for c in calls[4:])
    assert info["densified_cell_indices"] == [1, 0, 2]
    assert info["selected_city_frames"] == 8
    assert len(info["coarse_regions"]) == 4 and len(info["fine_regions"]) == 3
    # The fourth cell stays usable on the route although only three were dense.
    assert info["final_selection"]["selected_sources"][-1] == {"phase": "coarse", "cell_index": 3}
    assert merged[2][-1] == .7
    assert info["coarse_regions"][0]["cache"] == "coarse/0"


def test_good_baseline_track_is_preserved_exactly_after_fine_cell_comparison():
    baseline = _track(np.full(5, .9))
    calls = []

    def retrieve(region, *, cap, phase, index):
        calls.append(phase)
        return _track(np.full(5, .1)), {}

    merged, info = select_city_tracks(_regions(3), baseline, np.arange(5), retrieve)
    assert merged is baseline
    assert calls == ["coarse"] * 3 + ["fine"]
    assert info["densified_cell_indices"] == [0] and info["selected_city_frames"] == 0


def test_continuity_rejects_an_isolated_high_similarity_far_away():
    baseline = _track(np.full(5, .2))
    candidate = _track([.1, .1, .99, .1, .1])
    candidate[1][2, 0] += 1.0                # An isolated 100 km teleport.
    merged, info = select_city_tracks(_regions(1), baseline, np.arange(5),
                                      lambda *a, **k: (candidate, {}))
    assert merged is baseline and info["selected_city_frames"] == 0


def test_dense_baseline_cannot_starve_a_promising_thin_cell_of_its_fine_budget():
    baseline = _track(np.full(5, .2))

    def retrieve(region, *, phase, **kwargs):
        return _track(np.full(5, .15 if phase == "coarse" else .5)), {}

    merged, info = select_city_tracks(_regions(1), baseline, np.arange(5), retrieve)
    assert info["densified_cell_indices"] == [0]
    np.testing.assert_array_equal(merged[2], np.full(5, .5))


def test_coarse_failures_keep_baseline_and_record_every_attempt():
    baseline = _track(np.full(5, .2))

    def retrieve(region, *, index, **kwargs):
        if index == 0:
            raise RuntimeError("reference service failed")
        return None, {"reason": "no references", "cache": "empty"}

    merged, info = select_city_tracks(_regions(2), baseline, np.arange(5), retrieve)
    assert merged is baseline and info["status"] == "unavailable"
    assert [r["status"] for r in info["coarse_regions"]] == ["failed", "unavailable"]
    assert "reference service failed" in info["coarse_regions"][0]["error"]
    assert info["coarse_regions"][1]["reason"] == "no references"


def test_failed_fine_fetch_retains_the_supported_coarse_candidate():
    baseline = _track(np.full(5, .1))
    coarse = _track(np.full(5, .3))

    def retrieve(region, *, phase, **kwargs):
        return (coarse, {}) if phase == "coarse" else (None, {"reason": "fine unavailable"})

    merged, info = select_city_tracks(_regions(1), baseline, np.arange(5), retrieve)
    np.testing.assert_array_equal(merged[2], coarse[2])
    assert info["fine_regions"][0]["status"] == "unavailable"


def test_misaligned_query_frames_are_rejected_before_regional_selection():
    baseline = _track(np.full(5, .1))
    wrong = _track(np.full(5, .9), indices=np.arange(5) + 1)
    merged, info = select_city_tracks(_regions(1), baseline, np.arange(6),
                                      lambda *a, **k: (wrong, {}))
    assert merged is baseline
    assert info["coarse_regions"][0]["status"] == "failed"
    assert "different query frames" in info["coarse_regions"][0]["error"]


def test_city_retrieval_can_recover_when_the_ordinary_region_has_no_track():
    candidate = _track(np.full(5, .3))
    merged, info = select_city_tracks(_regions(1), None, np.arange(5),
                                      lambda *a, **k: (candidate, {"source": "fake"}))
    assert merged is not None and info["selected_city_frames"] == 5


@pytest.mark.parametrize("field,value", [("coarse_cap", 0), ("fine_regions", 0), ("fine_cap", -1)])
def test_hierarchy_budgets_must_be_positive_integers(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        select_city_tracks([], None, [], lambda *a, **k: pytest.fail("must not fetch"), **{field: value})


def test_prior_city_hierarchy_is_independent_of_metadata_budget_and_reuses_query_cache(tmp_path, monkeypatch):
    from tests.test_stages_prior import _ctx, _track as stage_track, CENTRE
    import monocular_osm.city_retrieval as city
    import monocular_osm.kartaview_vpr as kv
    import monocular_osm.stages.prior as prior

    other = (CENTRE[0] + .03, CENTRE[1])
    regions = [{"name": "cell0", "center": list(CENTRE), "radius_m": 3000},
               {"name": "cell1", "center": list(other), "radius_m": 3000}]
    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (
        regions, {"n_cells": 2, "boundary_sha256": "boundary", "max_cells": 64}))
    idx = stage_track()[0]
    baseline = (idx, stage_track()[1], np.full(20, .1))
    strong = (idx, stage_track(centre=other)[1], np.full(20, .3))
    fine = (idx, strong[1], np.full(20, .4))
    tracks = iter([baseline, (idx, baseline[1], np.full(20, .05)), strong, fine])
    calls = []

    def retrieve(*a, **kw):
        calls.append(kw)
        return next(tracks)

    monkeypatch.setattr(kv, "kartaview_vpr_track", retrieve)
    ctx = _ctx(tmp_path, vpr_source="kartaview", vpr_city_cover=True,
               vpr_region_cap=1, vpr_query_mode="crops", vpr_query_batch_size=2,
               osm_around=(*CENTRE, 900))
    ctx.osm_around = ctx.cfg.osm_around
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 4 and [c["cap"] for c in calls] == [1500, 250, 250, 1500]
    assert all(c["query_cache"] is calls[0]["query_cache"] for c in calls)
    assert all(c["query_mode"] == "crops" and c["query_batch_size"] == 2 for c in calls)
    assert calls[0]["cache_dir"] == str(ctx.cfg.data_dir / "kartaview")
    np.testing.assert_array_equal(ctx.vpr_track[2], fine[2])
    assert ctx.result["city_cover_retrieval"]["selected_city_frames"] == 20
    assert ctx.c2f_info is None             # No single-disc pass-2 overwrites the city track.
    assert ctx.channel_status["city_cover"]["status"] == "ok"


def test_prior_good_baseline_still_uses_ordinary_coarse_to_fine(tmp_path, monkeypatch):
    from tests.test_stages_prior import _vpr_ctx, _track as stage_track, CENTRE
    import monocular_osm.city_retrieval as city
    import monocular_osm.stages.prior as prior

    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (
        [{"name": "cell", "center": list(CENTRE), "radius_m": 5000}], {"n_cells": 1}))
    idx, coords, _ = stage_track()
    ctx, calls = _vpr_ctx(
        tmp_path, monkeypatch, [(idx, coords, np.full(20, .8)),
                                (idx, coords, np.full(20, .82)),
                                (idx, coords, np.full(20, .1)),
                                (idx, coords, np.full(20, .2))],
        vpr_city_cover=True, osm_around=(*CENTRE, 5000), vpr_search_radius_m=5000,
        vpr_c2f_radius_m=1500)
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 4 and calls[1]["radius_m"] == 1500
    assert ctx.c2f_info["kept"] == "pass-2"
    assert ctx.result["city_cover_retrieval"]["selected_city_frames"] == 0


def test_city_hierarchy_cannot_suppress_a_better_ordinary_second_pass(tmp_path, monkeypatch):
    from tests.test_stages_prior import _vpr_ctx, _track as stage_track, CENTRE
    import monocular_osm.city_retrieval as city
    import monocular_osm.stages.prior as prior

    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (
        [{"name": "cell", "center": list(CENTRE), "radius_m": 5000}], {"n_cells": 1}))
    idx, coords, _ = stage_track()
    ctx, calls = _vpr_ctx(
        tmp_path, monkeypatch, [(idx, coords, np.full(20, .1)),
                                (idx, coords, np.full(20, .9)),
                                (idx, coords, np.full(20, .6)),
                                (idx, coords, np.full(20, .7))],
        vpr_city_cover=True, osm_around=(*CENTRE, 5000), vpr_search_radius_m=5000,
        vpr_c2f_radius_m=1500)
    prior.vpr_coarse_prior(ctx)
    assert calls[1]["radius_m"] == 1500 and len(calls) == 4
    np.testing.assert_array_equal(ctx.vpr_track[2], np.full(20, .9))
    assert ctx.result["city_cover_retrieval"]["selected_city_frames"] == 0
    assert ctx.result["city_cover_retrieval"]["baseline"]["coarse_to_fine"]["kept"] == "pass-2"


def test_prior_city_cell_limit_preserves_baseline_with_explicit_diagnostics(tmp_path, monkeypatch):
    from tests.test_stages_prior import _vpr_ctx, _track as stage_track
    import monocular_osm.city_retrieval as city
    import monocular_osm.stages.prior as prior

    def refused(*a, **k):
        raise CityCoverageLimitError(65, 64, lower_bound=True)

    monkeypatch.setattr(city, "plan_city_regions", refused)
    baseline = stage_track()
    ctx, calls = _vpr_ctx(tmp_path, monkeypatch, [baseline], vpr_city_cover=True,
                          vpr_coarse_to_fine=False)
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 1 and ctx.vpr_track is baseline
    info = ctx.result["city_cover_retrieval"]
    assert info["status"] == "failed" and info["cell_limit"]["required_cells"] == 65
    assert ctx.channel_status["city_cover"]["status"] == "failed"
    assert ctx.anchor_origin == "vpr"


def test_prior_cached_nonadministrative_boundary_preserves_complete_baseline(tmp_path, monkeypatch):
    import hashlib
    from shapely.geometry import mapping
    from tests.test_stages_prior import _vpr_ctx, _track as stage_track
    import monocular_osm.stages.prior as prior

    baseline = stage_track()
    ctx, calls = _vpr_ctx(tmp_path, monkeypatch, [baseline], vpr_city_cover=True,
                          vpr_coarse_to_fine=False)
    directory = ctx.cfg.data_dir.parent / "_city_boundaries"
    directory.mkdir(parents=True)
    path = directory / f"city_{hashlib.sha256(ctx.cfg.city.encode()).hexdigest()[:16]}.geojson"
    original = json.dumps({"city": ctx.cfg.city, "geometry": mapping(box(0, 0, .01, .01)),
                           "source": {"class": "leisure", "type": "nature_reserve", "place_rank": 24}}).encode()
    path.write_bytes(original)
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 1 and ctx.vpr_track is baseline
    info = ctx.result["city_cover_retrieval"]
    assert info["status"] == "failed" and info["reason"] == "city_plan_refused"
    assert info["boundary_validation"]["status"] == "non_administrative"
    assert info["boundary_source"]["place_rank"] == 24
    assert ctx.anchor_origin == "vpr" and path.read_bytes() == original


def test_city_cache_paths_and_listing_provenance_are_isolated_per_phase(tmp_path, monkeypatch):
    from tests.test_stages_prior import _ctx, _track as stage_track, CENTRE
    import monocular_osm.city_retrieval as city
    import monocular_osm.kartaview_vpr as kv
    import monocular_osm.stages.prior as prior

    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (
        [{"name": "cell", "center": list(CENTRE), "radius_m": 3000}], {"n_cells": 1}))
    ctx = _ctx(tmp_path, vpr_region_cache_dir=tmp_path / "large_drive")
    idx, coords, _ = stage_track()
    baseline = (idx, coords, np.full(20, .1))
    paths = []

    def retrieve(*a, **kw):
        path = Path(kw["cache_dir"])
        path.mkdir(parents=True)
        paths.append(path)
        kv.LAST_FETCH_STATS = {"snapshot": {"refs_sha256": path.name}}
        return idx, coords, np.full(20, .3 if kw["cap"] == 250 else .4)

    cache = {}
    _track, info, listing = prior._city_cover_vpr(
        ctx.cfg, ctx.frames, baseline, {"center": list(CENTRE)},
        "mapillary", "not-recorded", "cpu", 1, retrieve, query_cache=cache)
    assert len(paths) == 2 and paths[0] != paths[1]
    assert paths[0].parent == tmp_path / "large_drive" / ctx.cfg.data_dir.name
    assert paths[0].name.startswith("mapillary_city_coarse_")
    assert paths[1].name.startswith("mapillary_city_fine_")
    assert info["coarse_regions"][0]["cache"] == str(paths[0])
    assert len(listing) == 2 and "not-recorded" not in json.dumps(info)
    assert set(prior._city_cache_directories(ctx.cfg, "mapillary")) == set(paths)


def test_city_cover_is_disabled_by_default_and_does_not_plan(tmp_path, monkeypatch):
    from tests.test_stages_prior import _vpr_ctx, _track as stage_track
    import monocular_osm.city_retrieval as city
    import monocular_osm.stages.prior as prior

    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: pytest.fail("city cover disabled"))
    ctx, calls = _vpr_ctx(tmp_path, monkeypatch, [stage_track()], vpr_coarse_to_fine=False)
    assert ctx.cfg.vpr_city_cover is False
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 1 and "city_cover_retrieval" not in ctx.result


def test_existing_city_references_avoid_tokenless_source_fallback(tmp_path, monkeypatch):
    from tests.test_stages_prior import _ctx, _track as stage_track, CENTRE
    import monocular_osm.city_retrieval as city
    import monocular_osm.kartaview_vpr as kv
    import monocular_osm.stages.prior as prior

    monkeypatch.delenv("MLY_TOKEN", raising=False)
    ctx = _ctx(tmp_path, vpr_source="mapillary", vpr_city_cover=True,
               osm_around=(*CENTRE, 900), vpr_coarse_to_fine=False)
    ctx.osm_around = ctx.cfg.osm_around
    warm = ctx.cfg.data_dir / "mapillary_city" / "coarse_existing"
    warm.mkdir(parents=True)
    monkeypatch.setattr(kv, "has_mapillary_cache", lambda path: Path(path) == warm)
    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (
        [{"name": "cell", "center": list(CENTRE), "radius_m": 3000}], {"n_cells": 1}))
    idx, coords, _ = stage_track()
    tracks = iter([None, (idx, coords, np.full(20, .3)), (idx, coords, np.full(20, .4))])
    calls = []

    def retrieve(*a, **kw):
        calls.append(kw)
        return next(tracks)

    monkeypatch.setattr(kv, "kartaview_vpr_track", retrieve)
    prior.vpr_coarse_prior(ctx)
    assert len(calls) == 3 and all(c["source"] == "mapillary" for c in calls)
    assert all(c["token"] is None for c in calls)
    assert ctx.vpr_source_used == "mapillary" and ctx.vpr_track is not None


def test_city_cells_with_a_different_descriptor_are_not_compared_against_baseline(tmp_path, monkeypatch):
    from tests.test_stages_prior import _ctx, _track as stage_track, CENTRE
    import monocular_osm.city_retrieval as city
    import monocular_osm.kartaview_vpr as kv
    import monocular_osm.stages.prior as prior

    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (
        [{"name": "cell", "center": list(CENTRE), "radius_m": 3000}], {"n_cells": 1}))
    model = {"name": "megaloc", "hub": "pinned-A"}
    monkeypatch.setattr(kv, "backbone_info", lambda: dict(model))
    ctx = _ctx(tmp_path)
    baseline = stage_track()

    def retrieve(*a, **k):
        model.update(name="different-descriptor", hub="pinned-B")
        idx, coords, _ = baseline
        return idx, coords, np.full(len(idx), .99)

    merged, info, _listing = prior._city_cover_vpr(
        ctx.cfg, ctx.frames, baseline, {}, "kartaview", None, "cpu", 1, retrieve)
    assert merged is baseline and info["selected_city_frames"] == 0
    assert info["coarse_regions"][0]["status"] == "failed"
    assert "not comparable" in info["coarse_regions"][0]["error"]
