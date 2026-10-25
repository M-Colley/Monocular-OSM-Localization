"""Baselines/ablations (monocular_osm/baselines.py). No network anywhere."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from monocular_osm import baselines as B
from monocular_osm.metrics import score_route

ROOT = Path(__file__).resolve().parents[1]


def _no_network(city):  # pragma: no cover - must never be called
    raise AssertionError(f"network geocode attempted for {city!r}")


# ---------------------------------------------------------------------------
# city centroid
# ---------------------------------------------------------------------------


def test_city_centroid_reads_snapshot_exact_key(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"Ulm, Germany": [48.3984968, 9.9912458],
                                "Toronto, Ontario, Canada": {"lat": 43.65, "lon": -79.38}}),
                    encoding="utf-8")
    p = B.city_centroid("Ulm, Germany", snapshots=[snap], geocoder=_no_network)
    assert p.route_latlon == [[48.3984968, 9.9912458]]
    assert p.kind == "point" and p.source.startswith("snapshot:")
    p2 = B.city_centroid("Toronto, Ontario, Canada", snapshots=[snap], geocoder=_no_network)
    assert p2.route_latlon == [[43.65, -79.38]]
    # Exact-string keys: a different spelling is a miss, not a fuzzy hit.
    assert B.city_centroid("Ulm", snapshots=[snap], allow_network=False) is None


def test_city_centroid_falls_back_to_geocoder_and_records_it(tmp_path: Path) -> None:
    p = B.city_centroid("Nowhere, XX", snapshots=[tmp_path / "missing.json"],
                        geocoder=lambda c: (10.0, 20.0))
    assert p.route_latlon == [[10.0, 20.0]]
    assert p.source == "osmnx.geocode"
    failing = B.city_centroid("Nowhere, XX", snapshots=[],
                              geocoder=lambda c: (_ for _ in ()).throw(RuntimeError()))
    assert failing is None


# The October 2026 review's city-centroid numbers (mean GT->pred, i.e. the
# legacy "mean route error"), with the city strings the README rows used.
REVIEW_CENTROID = [
    ("ulm_ULl8s4qydrk_0-300.json", "Ulm, Germany", 534),
    ("ulm_LC9Sa--u5KE.json", "Ulm, Germany", 558),
    ("malaga_extract07.json", "Málaga, Spain", 5352),
    ("london_T4wTL3LpLqU.json", "London, UK", 1751),
    ("comma_148.json", "Daly City, California, USA", 2067),
    ("kitti_drive_0009.json", "Karlsruhe, Germany", 2645),
    ("kitti_drive_0033.json", "Karlsruhe, Germany", 6843),
    ("boreas_glenshields.json", "Vaughan, Ontario, Canada", 3805),
]


@pytest.mark.parametrize("gt_file,city,expected", REVIEW_CENTROID)
def test_city_centroid_regression_against_review(gt_file, city, expected) -> None:
    gt = json.loads((ROOT / "ground_truth" / gt_file).read_text(encoding="utf-8"))
    gt_ll = np.array([[w["lat"], w["lon"]] for w in gt["waypoints"]])
    pred = B.city_centroid(city, allow_network=False)
    assert pred is not None and pred.source == "snapshot:data/geocode_cache.json"
    s = score_route(pred.route_latlon, gt_ll)
    assert s.gt_to_pred_mean_m == pytest.approx(expected, abs=5.0)


# ---------------------------------------------------------------------------
# result.json-derived ablations
# ---------------------------------------------------------------------------

ROUTE_A = [[48.40, 9.98], [48.41, 9.99]]
ROUTE_B = [[48.30, 9.90], [48.31, 9.91]]


def _result(**over) -> dict:
    r = {
        "position": {"source": "anchor_primary_vpr", "route_latlon": ROUTE_A,
                     "start": {"latitude": 48.40, "longitude": 9.98},
                     "hypotheses": [
                         {"rank": 1, "latitude": 48.30, "longitude": 9.90},
                         {"rank": 2, "latitude": 48.401, "longitude": 9.981},
                         {"rank": 3, "latitude": 48.50, "longitude": 9.70},
                     ]},
        "matcher_position": {"source": "matcher", "route_latlon": ROUTE_B},
        "pinned_position": {"source": "ocr_anchor_pin", "route_latlon": ROUTE_A},
        "vpr_prior": {"lat": 48.405, "lon": 9.985, "origin": "vpr", "track_frames": 80},
    }
    r.update(over)
    return r


def test_report_derived_baselines() -> None:
    r = _result()
    assert B.pipeline_headline(r).route_latlon == ROUTE_A
    m = B.matcher_only(r)
    assert m.route_latlon == ROUTE_B and m.meta["report_source"] == "matcher"
    p = B.pinned_only(r)
    assert p.route_latlon == ROUTE_A and p.meta["report_source"] == "ocr_anchor_pin"
    v = B.vpr_prior_only(r)
    assert v.route_latlon == [[48.405, 9.985]] and v.kind == "point"
    assert v.meta["origin"] == "vpr"


def test_vpr_track_only_is_the_time_ordered_track_with_outliers_damped() -> None:
    track = [[float(t), 48.40 + 0.001 * t, 9.98, 0.2] for t in range(9)]
    track[4][1] = 49.0                                  # one wrong retrieval
    track = track[5:] + track[:5]                       # stored out of order
    v = B.vpr_track_only(_result(vpr_prior={"lat": 48.4, "lon": 9.98, "track": track}))
    assert v.kind == "route" and len(v.route_latlon) == 9
    lats = [p[0] for p in v.route_latlon]
    assert max(lats) < 48.41                            # the spike is gone
    assert lats[0] < 48.402 and lats[-1] > 48.406       # time order, not storage order
    assert B.vpr_track_only(_result()) is None          # older runs: no track saved


def test_track_checked_keeps_an_agreeing_fit_and_replaces_a_far_one() -> None:
    near = [[48.4000 + 0.0005 * t, 9.9800, 0.2] for t in range(9)]
    track = [[float(t), *p] for t, p in enumerate(near)]
    fit_near = [[48.4000, 9.9801], [48.4040, 9.9801]]       # ~7 m off the track
    fit_far = [[48.4000, 9.9900], [48.4040, 9.9900]]        # ~740 m east
    kept = B.track_checked(_result(position={"route_latlon": fit_near},
                                   vpr_prior={"lat": 48.4, "lon": 9.98, "track": track}))
    assert kept.meta["chose"] == "pipeline" and kept.meta["disagreement_m"] < 20
    swapped = B.track_checked(_result(position={"route_latlon": fit_far},
                                      vpr_prior={"lat": 48.4, "lon": 9.98, "track": track}))
    assert swapped.meta["chose"] == "vpr_track_only" and swapped.meta["disagreement_m"] > 700
    assert swapped.route_latlon == B.vpr_track_only(
        _result(vpr_prior={"lat": 48.4, "lon": 9.98, "track": track})).route_latlon
    no_track = B.track_checked(_result())                 # older run: the fit, unchecked
    assert no_track.meta["chose"] == "pipeline" and no_track.meta["disagreement_m"] is None


def test_missing_channels_are_none_not_zero() -> None:
    r = _result()
    del r["pinned_position"], r["vpr_prior"]
    assert B.pinned_only(r) is None
    assert B.vpr_prior_only(r) is None
    assert B.matcher_only({}) is None
    assert B.pipeline_headline(None) is None


def test_report_route_prefers_orienternet_refinement_and_falls_back() -> None:
    rep = {"route_latlon": ROUTE_B, "orienternet_route_latlon": ROUTE_A}
    assert B.report_route(rep) == ROUTE_A
    assert B.report_route({"start": {"latitude": 1.0, "longitude": 2.0},
                           "end": {"latitude": 1.5, "longitude": 2.5}}) == [[1.0, 2.0], [1.5, 2.5]]
    assert B.report_route({"latitude": 1.0, "longitude": 2.0}) == [[1.0, 2.0]]
    assert B.report_route({"route_latlon": [[1.0, None]]}) is None


def test_shortlist_oracle_picks_by_gt_and_is_labelled() -> None:
    gt = [[48.401, 9.9811], [48.42, 9.99]]
    o = B.shortlist_oracle(_result(), gt, k=5)
    assert o.oracle is True and o.name == "shortlist_oracle@5"
    assert o.route_latlon == [[48.401, 9.981]]
    assert o.meta == {"k": 5, "k_available": 3, "chosen_rank": 2}
    # k limits the shortlist: with k=1 only hypothesis #1 is eligible.
    assert B.shortlist_oracle(_result(), gt, k=1).route_latlon == [[48.30, 9.90]]
    r = _result()
    r["position"].pop("hypotheses")
    assert B.shortlist_oracle(r, gt) is None


# ---------------------------------------------------------------------------
# selective prediction with a pre-registered threshold
# ---------------------------------------------------------------------------


def test_retrieval_confidence_prefers_pass2_then_track_median() -> None:
    assert B.retrieval_confidence(_result()) == (None, None)
    r = _result(vpr_coarse_to_fine={"pass1_sim": 0.1, "pass2_sim": 0.2})
    assert B.retrieval_confidence(r) == (0.2, "vpr_coarse_to_fine.pass2_sim")
    r = _result()
    r["vpr_prior"]["track_sim_median"] = 0.3
    assert B.retrieval_confidence(r) == (0.3, "vpr_prior.track_sim_median")


def test_pipeline_selective_answers_or_abstains_by_the_rule() -> None:
    centre = B.Prediction("city_centroid", [[48.0, 9.0]], "point", source="snap")
    rule = {"tau": 0.16}
    hi = _result(vpr_coarse_to_fine={"pass2_sim": 0.21})
    lo = _result(vpr_coarse_to_fine={"pass2_sim": 0.12})
    a = B.pipeline_selective(hi, centre, rule)
    assert a.route_latlon == ROUTE_A and a.meta["abstained"] is False
    b = B.pipeline_selective(lo, centre, rule)
    assert b.route_latlon == [[48.0, 9.0]] and b.meta["abstained"] is True
    c = B.pipeline_selective(_result(), centre, rule)       # no signal: abstain
    assert c.meta["abstained"] is True and c.meta["confidence"] is None
    assert B.pipeline_selective(lo, None, rule) is None     # no centre: no answer


def test_the_committed_rule_is_loadable() -> None:
    rule = B.load_preregistered_rule()
    assert rule is not None and 0.0 < rule["tau"] < 1.0
    assert rule["fallback"] == "city_centroid"


def test_answer_reads_the_recorded_answer() -> None:
    r = _result(answer={"mode": "track", "method": "vpr_track_only", "route_latlon": ROUTE_B})
    a = B.answer(r)
    assert a.route_latlon == ROUTE_B and a.meta == {"mode": "track", "method": "vpr_track_only"}
    assert B.answer(_result()) is None                  # runs before --answer existed
