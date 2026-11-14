"""Synthetic rank algebra and exact erased-only decoder equivalence."""
import copy

import numpy as np
import pytest

from monocular_osm import rank_fusion as r, heading_sequence as h


def evidence(dtype=np.float32):
    rows = np.array([[.3, .1, .3, .8], [.2, .6, .6, .1], [.4, .4, .4, .4]], dtype)
    return {model: {"scores": np.roll(rows, i, axis=1), "ref_ids": np.array(["z", "b", "a", "c"]),
        "ref_latlon": np.array([[51., .001], [51., .002], [51., 0.], [51., .003]]),
        "query_times": np.array([0., 3., 11.])} for i, model in enumerate(r.MODELS)}


def test_exact_average_ties_float64_and_complete_support():
    a = np.array([[.1, .1, .3, .7, .7, .7, .9], [.4]*7], np.float32)
    actual = r.average_rank_percentiles(a)
    assert actual.dtype == np.float64 and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual[0], np.array([1., 1., 2.5, 4.5, 4.5, 4.5, 6.5])/7)
    np.testing.assert_array_equal(actual[1], np.full(7, .5))
    assert (actual > 0).all() and (actual < 1).all()
    np.testing.assert_allclose(actual.mean(axis=1), .5, rtol=0, atol=1e-16)


def test_single_reference_is_half_and_signed_zero_ties():
    np.testing.assert_array_equal(r.average_rank_percentiles(np.ones((2, 1))), [[.5], [.5]])
    a = r.average_rank_percentiles(np.array([[0., -0., 1.], [1., 0., -0.]]))
    assert a[0, 0] == a[0, 1] == 1/3


def test_equal_mean_fixed_order_and_single_rank_controls_are_preserved():
    source = evidence()
    before = copy.deepcopy(source)
    result = r.calibrate_evidence(source)
    assert tuple(result["matrices"]) == r.PROFILE["outputs"]
    matrices = result["matrices"]
    expected = (matrices["rank_megaloc"]+matrices["rank_image"]+matrices["rank_selavpr"])/3
    np.testing.assert_array_equal(matrices["equal3"], expected)
    for model in r.MODELS:
        np.testing.assert_array_equal(matrices["rank_"+model], r.average_rank_percentiles(source[model]["scores"]))
        for key in source[model]: np.testing.assert_array_equal(source[model][key], before[model][key])
    assert result["selected_variant"] is result["confidence"] is None


def test_reference_permutation_and_fortran_layout_do_not_change_ranks():
    source = evidence()
    original = r.calibrate_evidence(source)
    order = [3, 0, 2, 1]
    for value in source.values():
        value["scores"] = np.asfortranarray(value["scores"][:, order])
        value["ref_ids"] = value["ref_ids"][order]
        value["ref_latlon"] = value["ref_latlon"][order]
    permuted = r.calibrate_evidence(source)
    for name in r.PROFILE["outputs"]:
        np.testing.assert_array_equal(original["matrices"][name][:, order], permuted["matrices"][name])


@pytest.mark.parametrize("field", ["ids", "coordinates", "times", "score_dtype", "coordinate_dtype", "time_dtype", "shape"])
def test_mismatched_encoder_identity_declines(field):
    source = evidence()
    item = source["image"]
    if field == "ids": item["ref_ids"] = item["ref_ids"][::-1]
    if field == "coordinates": item["ref_latlon"][0, 0] += .0001
    if field == "times": item["query_times"][1] += .1
    if field == "score_dtype": item["scores"] = item["scores"].astype(np.float64)
    if field == "coordinate_dtype": item["ref_latlon"] = item["ref_latlon"].astype(np.float32)
    if field == "time_dtype": item["query_times"] = item["query_times"].astype(np.float32)
    if field == "shape": item["scores"] = item["scores"][:, :-1]
    with pytest.raises(ValueError): r.calibrate_evidence(source)


@pytest.mark.parametrize("bad", [np.full((2, 3), np.nan), np.full((2, 3), np.inf), np.ones((1, 4)),
    np.ones((2, 10001)), np.ones((2, 3), np.int32), np.full((2, 3), 1.1), np.empty((2, 0))])
def test_bad_bounded_matrix_declines(bad):
    with pytest.raises(ValueError): r.average_rank_percentiles(bad)


def test_missing_encoder_duplicate_ids_and_oversize_ids_decline():
    source = evidence()
    del source["selavpr"]
    with pytest.raises(ValueError): r.calibrate_evidence(source)
    for ids in (np.array(["same"]*4), np.array(["a"*129, "b", "c", "d"])):
        source = evidence()
        for value in source.values(): value["ref_ids"] = ids
        with pytest.raises(ValueError): r.calibrate_evidence(source)


@pytest.mark.parametrize("geometry", ["regular", "ties", "disconnected", "dateline"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_erased_paths_objectives_match_original_six_exactly(geometry, dtype):
    rng = np.random.default_rng(7)
    scores = rng.uniform(.1, .9, (4, 3)).astype(dtype)
    refs = np.array([[52., 0.], [52.0006, .001], [52., .002]])
    times = np.array([0., .2, 8., 9.5])
    if geometry == "ties": scores.fill(.5); refs[:] = [52., 0.]
    if geometry == "disconnected": refs[:, 1] = [0., 1., 2.]
    if geometry == "dateline": refs[:, 1] = [179.999, -179.999, 179.998]
    ids = np.array(["z", "a", "m"])
    before = dict(h.PROFILE)
    old = h.decode_heading_tracks(scores[:, None, :], ids, refs, times, [None]*3, np.full(3, np.nan), np.zeros(1))
    new = r.decode_erased_tracks(scores, ids, refs, times)
    for variant in ("raw", "reference_mean"):
        expected = old["outputs"][variant+"/erased"]
        actual = new["outputs"][variant]
        for key in expected: assert actual[key] == expected[key], key
    assert dict(h.PROFILE) == before
    assert new["allocation"]["joint_backpointer_bytes"] == 2*4*3*16*4
    assert new["allocation"]["state_methods"] == 2


def test_helper_calls_only_two_neutral_transition_states_and_resource_progress(monkeypatch):
    original = h._advance
    calls, progress = [], []
    def advance(previous, tree, xy, dt, yaw, quality, budget):
        assert previous.shape[0] == 2 and np.isnan(yaw) and quality == 0
        calls.append(dt)
        return original(previous, tree, xy, dt, yaw, quality, budget)
    monkeypatch.setattr(h, "_advance", advance)
    item = evidence()["megaloc"]
    r.decode_erased_tracks(item["scores"], item["ref_ids"], item["ref_latlon"], item["query_times"],
        progress=lambda row, count: progress.append((row, count)))
    assert calls == [3., 8.] and progress == [(1, 3), (2, 3)]


def test_helper_allocation_and_temporary_bounds_refuse():
    item = evidence()["megaloc"]
    args = [item[k] for k in ("scores", "ref_ids", "ref_latlon", "query_times")]
    with pytest.raises(ValueError, match="backpointer"):
        r.decode_erased_tracks(*args, max_backpointer_bytes=2*3*4*16*4-1)
    with pytest.raises(ValueError, match="temporary"):
        r.decode_erased_tracks(*args, max_temp_bytes=1)


def test_profile_fixed_after_outcomes_and_no_policy_claim():
    assert r.PROFILE["validation_status"].startswith("posthoc_DEV_after")
    assert r.PROFILE["heading_profile_sha256"] == h.PROFILE_SHA256
    assert r.PROFILE["candidate_pruning"] is r.PROFILE["default_policy_adopted"] is False
    assert r.PROFILE["selected_variant"] is r.PROFILE["confidence"] is None
    assert h._json_sha(dict(r.PROFILE)) == r.PROFILE_SHA256
