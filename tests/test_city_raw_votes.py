"""Raw coarse-cell evidence through real stage/cache wrappers; tiny fakes only."""
from functools import wraps
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from monocular_osm import city_retrieval as city, kartaview_vpr as kv
from monocular_osm import public_reference_cache as cache
from monocular_osm.stages import prior


MAXIMA = np.array([[.9, .8, .7, .6, .5],
                   [.2, .9, .8, .7, .1],
                   [.1, .2, .9, .8, .7],
                   [.7, .1, .2, .9, .8]], np.float32)


def regions():
    return [{"name": f"cell{i}", "center": [50., 8. + i * .001], "radius_m": 3000.}
            for i in range(5)]


def evidence(monkeypatch, maxima, *, center=(50., 8.), chosen_score=.05, identity_change=None):
    q = len(maxima)
    raw = np.column_stack([np.full(q, chosen_score, np.float32), maxima])
    ix, times, chosen = np.arange(q), np.arange(q, dtype=float), np.zeros(q, dtype=int)
    refs = np.array([center, (center[0], center[1] + .0001)], np.float64)
    audit = {"schema": "vpr-track-audit-v1", "issues": [], "query_indices": ix,
        "ref_ids": np.array(["mapillary:1", "mapillary:2"]), "ref_latlon": refs,
        "raw_similarities": raw, "ref_embeddings_sha256": "a" * 64,
        "query_pixels_sha256": "b" * 64, "query_descriptors_sha256": "c" * 64,
        "actual_model_state_sha256": "d" * 64, "backbone": {"name": "fixture", "hub": "fake"},
        "preprocessing": {"fixture": "warp"}, "query_mode": "warp"}
    audit.update(identity_change or {})
    monkeypatch.setattr(kv, "LAST_TRACK_STATS", {"decoder": "argmax"})
    cosines = raw[np.arange(q), chosen].astype(float)
    kv._finish_track_audit(audit, ix, raw, chosen, cosines, times, True, 1.)
    assert audit["status"] == "complete"
    return (ix, refs[chosen], cosines), audit


def hierarchy(monkeypatch, *, selector="raw-votes", fault=None):
    calls = []
    baseline = (np.arange(4), np.tile([50., 8.], (4, 1)), np.full(4, .99))

    def retrieve(region, *, cap, phase, index):
        calls.append((index, phase, cap))
        track, audit = evidence(monkeypatch, MAXIMA[:, index], center=region["center"])
        if phase == "coarse" and index == 1:
            if fault == "missing":
                return track, {}
            if fault == "nan":
                audit["raw_similarities"][0, 1] = np.nan
            elif fault == "clock":
                audit["query_times"][0] += .01
            elif fault == "unsigned_order":
                audit["query_indices"] = np.array([0, 2, 1, 3], np.uint64)
            elif fault == "track_grid":
                track = (np.array([0, 2, 1, 3]), track[1], track[2])
            elif fault in {"query", "model"}:
                changed = {"query_descriptors_sha256" if fault == "query" else "actual_model_state_sha256": "e" * 64}
                track, audit = evidence(monkeypatch, MAXIMA[:, index], center=region["center"], identity_change=changed)
        return track, {}, audit

    result, info = city.select_city_tracks(regions(), baseline, np.arange(4, dtype=float), retrieve,
                                           cell_selector=selector)
    return result, info, calls, baseline


def test_raw_votes_densifies_hidden_raw_support_without_changing_final_baseline_rule(monkeypatch):
    old, decoded, old_calls, _ = hierarchy(monkeypatch, selector="decoded")
    result, info, calls, baseline = hierarchy(monkeypatch)
    assert decoded["densified_cell_indices"] == [0]
    # Raw top3 votes: counts [2,2,3,3,2], independently of decoded column0.
    assert info["densified_cell_indices"] == [2, 3, 0]
    assert info["coarse_selection"]["region_frame_counts"] == [2, 2, 3, 3, 2]
    assert calls[:5] == old_calls[:5] == [(i, "coarse", 250) for i in range(5)]
    assert calls[5:] == [(i, "fine", 1500) for i in (2, 3, 0)]
    assert info["reference_budget"]["fine_max"] == decoded["reference_budget"]["fine_max"] == 4500
    assert result is baseline and info["selected_city_frames"] == 0
    assert info["raw_votes"]["status"] == "complete"


def test_raw_votes_uses_cell_order_for_score_and_vote_ties_not_decoded_medians(monkeypatch):
    def retrieve(region, *, phase, index, **kwargs):
        track, audit = evidence(monkeypatch, np.full(4, .9, np.float32),
                                chosen_score=.1 + index * .1, center=region["center"])
        return track, {}, audit

    _, info = city.select_city_tracks(regions(), None, np.arange(4, dtype=float), retrieve,
                                      cell_selector="raw-votes")
    assert info["densified_cell_indices"] == [0, 1, 2]
    assert info["coarse_selection"]["top_cell_indices_per_query"] == [[0, 1, 2]] * 4


@pytest.mark.parametrize("fault", ["missing", "nan", "clock", "unsigned_order", "track_grid", "query", "model"])
def test_missing_nonfinite_or_incomparable_raw_evidence_declines_without_decoded_fallback(monkeypatch, fault):
    result, info, calls, baseline = hierarchy(monkeypatch, fault=fault)
    assert result is baseline and info["status"] == "unavailable"
    assert info["raw_votes"]["unavailable_coarse_cell_indices"] == [1]
    assert info["densified_cell_indices"] == [] and len(calls) == 5
    assert info["coarse_regions"][1]["status"] == ("failed" if fault == "track_grid" else "ok")
    assert info["coarse_regions"][1]["raw_evidence"]["status"] == "unavailable"


def test_unknown_selector_refuses_before_retrieval():
    with pytest.raises(ValueError, match="cell selector"):
        city.select_city_tracks([], None, [], lambda *_: pytest.fail("retrieved"), cell_selector="best")


def test_decoded_default_needs_no_raw_audit_and_keeps_existing_selection(monkeypatch):
    result, info, calls, baseline = hierarchy(monkeypatch, selector="decoded", fault="missing")
    assert result is baseline and info["densified_cell_indices"] == [0]
    assert "raw_votes" not in info and len(calls) == 6


def test_incomparable_fine_pool_is_not_merged_but_coarse_evidence_is_retained(monkeypatch):
    def retrieve(region, *, phase, index, **kwargs):
        changed = {"query_descriptors_sha256": "e" * 64} if phase == "fine" else None
        track, audit = evidence(monkeypatch, MAXIMA[:, index], center=region["center"], identity_change=changed)
        return track, {}, audit

    result, info = city.select_city_tracks(regions(), None, np.arange(4, dtype=float), retrieve,
                                           cell_selector="raw-votes")
    assert info["raw_votes"]["status"] == "complete" and result is not None
    assert info["n_usable_fine"] == 0 and len(info["fine_regions"]) == 3
    assert all(row["status"] == "failed" for row in info["fine_regions"])
    assert all(row["phase"] == "coarse" for row in info["final_selection"]["selected_sources"])


def test_prior_and_cache_clear_populate_the_same_caller_audit_before_raw_voting(tmp_path, monkeypatch):
    from tests.test_stages_prior import _ctx

    monkeypatch.setitem(sys.modules, "osmnx", SimpleNamespace())
    monkeypatch.setattr(prior, "select_retrieval_device", lambda: "cpu")
    monkeypatch.setattr(kv, "backbone_info", lambda: {"name": "fixture", "hub": "fake"})
    monkeypatch.setattr(kv, "_robust_center", lambda ll, sims: tuple(ll.mean(axis=0)))
    monkeypatch.setattr(city, "plan_city_regions", lambda *a, **k: (regions(), {"n_cells": 5}))
    ctx = _ctx(tmp_path, n=4, vpr_source="kartaview", vpr_city_cover=True,
               vpr_city_cell_selector="raw-votes", vpr_coarse_to_fine=False,
               osm_around=(50., 8., 900.))
    ctx.osm_around = ctx.cfg.osm_around
    ctx.capture_retrieval_audit = True
    original = kv.kartaview_vpr_track
    sinks, calls = [], []

    @wraps(original)
    def actual_fake(frames, center, **kwargs):
        sink = kwargs["audit"]
        sinks.append(sink)
        calls.append(kwargs)
        sink.clear()  # The actual KV/cache path clears this shared dictionary.
        column = len(calls) - 2
        maxima = MAXIMA[:, column] if 0 <= column < 5 else np.full(4, .15, np.float32)
        track, populated = evidence(monkeypatch, maxima, center=center)
        sink.update(populated)  # Fake backend, inside the actual call, not posthoc.
        return track

    monkeypatch.setattr(kv, "kartaview_vpr_track", actual_fake)
    with cache.track_scope(ctx.cfg, "baseline") as public_calls:
        prior.vpr_coarse_prior(ctx)
    info = ctx.result["city_cover_retrieval"]
    assert info["cell_selector"] == "raw-votes" and info["raw_votes"]["status"] == "complete"
    assert info["densified_cell_indices"] == [2, 3, 0] and len(calls) == 9
    assert len(ctx.retrieval_audits) == len(public_calls) == len(sinks) == 9
    assert all(sink is saved for sink, saved in zip(sinks, ctx.retrieval_audits))
    assert all("call_index" not in sink and sink["public_cache_request"] == call["request"]
               for sink, call in zip(sinks, public_calls))
    assert all(call["query_cache"] is calls[0]["query_cache"] for call in calls)
    assert all("audit" not in (sink.get("request") or {}) for sink in sinks)
