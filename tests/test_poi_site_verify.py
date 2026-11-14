"""Sign-site verification (monocular_osm.poi_site_verify) and policy v4; no models."""

from __future__ import annotations

import numpy as np
import pytest

from monocular_osm import assisted_policy as base
from monocular_osm import assisted_policy_v3 as v3
from monocular_osm import assisted_policy_v4 as v4
from monocular_osm import poi_site_verify as S
from monocular_osm.assisted_profiles import V3_FAMILY, resolve_policy

TRUE_SITE, FALSE_SITE = (33.7800, -84.3684), (33.8801, -84.3806)


def _clues():
    def poi(i, lat, lon, name):
        return {"id": i, "lat": lat, "lon": lon, "tags": {"name": name}}
    branches = [{"pois": [poi("a1", TRUE_SITE[0], TRUE_SITE[1], "Shop A"),
                          poi("b1", TRUE_SITE[0] + 4e-4, TRUE_SITE[1] - 8e-4, "Shop B")]},
                {"pois": [poi("a2", FALSE_SITE[0], FALSE_SITE[1], "Shop A"),
                          poi("b2", FALSE_SITE[0] + 1e-4, FALSE_SITE[1], "Shop B")]}]
    obs = [{"poi_id": p, "t_sec": 102.5} for p in ("a1", "b1", "a2", "b2")] + [{"poi_id": "a1", "t_sec": 115.4}]
    return {"selected_answer": None, "candidate_clusters": branches, "matched_observations": obs}


OCR = {"window": [0.0, 294.9], "sample_count": 24}


def test_plan_has_one_site_per_branch_with_its_sign_times_and_the_ocr_interval():
    plan = S.plan_sites(_clues(), OCR)
    assert plan["profile_sha256"] == S.PROFILE_SHA256 and len(plan["sites"]) == 2
    assert plan["sign_margin_s"] == pytest.approx(294.9 / 23)
    assert plan["sites"][0]["sign_times"] == [102.5, 115.4] and plan["sites"][1]["sign_times"] == [102.5]
    assert np.allclose(plan["sites"][0]["center"], [TRUE_SITE[0] + 2e-4, TRUE_SITE[1] - 4e-4], atol=1e-6)


@pytest.mark.parametrize("bad", [{"candidate_clusters": []}, {"candidate_clusters": [{"pois": []}] * 7},
                                 {"candidate_clusters": [{"pois": [{"id": 1, "lat": 1, "lon": 1}]}],
                                  "selected_answer": "x"}])
def test_plan_refuses_no_too_many_or_selected_branches(bad):
    with pytest.raises(ValueError):
        S.plan_sites({"selected_answer": None, **bad}, OCR)


def test_sign_rows_fall_back_to_every_row_without_a_read_time():
    times = np.linspace(0, 295, 80)
    rows = S.sign_rows(times, [102.5], 294.9 / 23)
    assert np.all(np.abs(times[rows] - 102.5) <= 294.9 / 23) and len(rows) < 80
    assert np.array_equal(S.sign_rows(times, [], 10.0), np.arange(80))


def test_the_site_whose_imagery_matches_the_sign_frames_is_chosen_and_decoded():
    rng = np.random.default_rng(0)
    times = np.linspace(0, 295, 80)
    plan = S.plan_sites(_clues(), OCR)
    rows = S.sign_rows(times, plan["sites"][0]["sign_times"], plan["sign_margin_s"])
    true_sims = rng.uniform(0.05, 0.12, (80, 30)); true_sims[rows, 3] = 0.25        # the signed shop, seen
    false_sims = rng.uniform(0.05, 0.14, (80, 30))                                  # busier but never matching
    scores = [S.site_score(true_sims, rows), S.site_score(false_sims, rows)]
    assert S.select_site(scores) == 0
    ll = np.c_[TRUE_SITE[0] + rng.uniform(-2e-3, 2e-3, 30), TRUE_SITE[1] + rng.uniform(-2e-3, 2e-3, 30)]
    route_ll = np.c_[TRUE_SITE[0] + rng.uniform(-1e-2, 1e-2, 50), TRUE_SITE[1] + rng.uniform(-1e-2, 1e-2, 50)]
    ids, latlon, sims = S.merge_pools(([f"d{i}" for i in range(30)], ll, true_sims),
                                      ([f"d{i}" for i in range(5)] + [f"r{i}" for i in range(45)],
                                       route_ll, rng.uniform(0.05, 0.12, (80, 50))))
    assert len(ids) == 75 and sims.shape == (80, 75)                     # five shared ids kept once
    out = S.decode(sims, latlon, times)
    assert len(out["route_latlon"]) == 80 and out["route_timestamps"] == times.tolist()
    assert max(abs(p[0] - TRUE_SITE[0]) for p in out["route_latlon"]) < 0.02


def test_select_site_breaks_ties_to_the_first_and_refuses_non_finite():
    assert S.select_site([0.2, 0.2]) == 0
    with pytest.raises(ValueError):
        S.select_site([0.1, float("nan")])


# ---------------------------------------------------------------------------
# policy v4
# ---------------------------------------------------------------------------

def _evidence(*, median=0.14, valid=0.49, branches=2, site=None):
    ev = {"baseline": {"status": "complete", "route_valid": True, "point_count": 80, "score_kind": "raw_cosine",
                       "score_identity": "x" * 64, "similarity_median": median},
          "cues": {"status": "complete", "duration_s": 295.0, "vo": {"status": "complete", "valid_fraction": valid},
                   "ocr": {"status": "complete", "clue_branch_count": branches}}}
    if site is not None:
        ev["poi_site"] = site
    return ev


def _complete_site(branches=2, **over):
    site = {"status": "complete", "route_valid": True, "point_count": 80, "uses_rigid_vo": False,
            "profile_sha256": S.PROFILE_SHA256, "candidate_pruning": False, "score_normalization": "reference_mean",
            "query_mode": "warp", "radius_m": 3000, "cap": 1500, "retained_branch_count": branches,
            "successful_branch_count": branches}
    site.update(over)
    return site


def test_v4_resolves_and_shares_the_v3_family():
    assert resolve_policy("v4") is v4 and "v4" in V3_FAMILY and "v3" in V3_FAMILY
    assert v4.policy_config()["site_profile_sha256"] == S.PROFILE_SHA256


def test_weak_vo_with_sign_branches_requests_then_selects_the_site_route():
    first = v4.decide_assisted_policy(_evidence())
    assert first["next_action"] == "request_poi_site" and first["policy_version"] == v4.POLICY_VERSION
    done = v4.decide_assisted_policy(_evidence(site=_complete_site()))
    assert done["next_action"] == "select_candidate" and done["selected_candidate"] == "poi_site"


def test_an_incomplete_site_candidate_falls_back_to_the_unchanged_v3_decision():
    bad = v4.decide_assisted_policy(_evidence(site=_complete_site(query_mode="crops")))
    expected = v3.decide_assisted_policy({**_evidence(), "poi_image_track": {"status": "unavailable"}})
    assert bad["next_action"] == expected["next_action"] == "request_expanded_retrieval"
    assert bad["evidence_summary"]["poi_site"]["decline_reason"].startswith("incomplete")


@pytest.mark.parametrize("ev", [_evidence(median=0.20),            # strong baseline: v3 keeps it
                                _evidence(valid=0.90),             # good VO: the joint POI attempt
                                _evidence(branches=0)])            # no signs: v3's later stages
def test_outside_its_admission_v4_decides_exactly_like_v3(ev):
    a, b = v4.decide_assisted_policy(ev), v3.decide_assisted_policy(ev)
    assert {k: v for k, v in a.items() if k not in ("policy_version", "config_sha256")} == \
           {k: v for k, v in b.items() if k not in ("policy_version", "config_sha256")}


def test_base_thresholds_are_the_ones_v4_reads():
    cfg = base.policy_config()
    assert cfg["joint_vo_valid_min"] == 0.8 and cfg["baseline_median_min"] == 0.162
