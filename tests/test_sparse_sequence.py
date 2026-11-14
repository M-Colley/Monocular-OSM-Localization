"""Bounded retrieval-only sequence search; synthetic scores, no models/GT."""
from copy import deepcopy
import itertools
import json

import numpy as np
import pytest

from monocular_osm import sparse_sequence as sparse
from monocular_osm.geo import LocalFrame


def evidence(n=4, q=3):
    scores = np.full((q, n), .1, np.float32)
    refs = np.c_[np.zeros(n), np.arange(n) * .0001]
    ids = np.array([f"public:{i:06}" for i in range(n)])
    return scores, ids, refs, np.arange(q, dtype=float)


def test_reference_mean_and_raw_are_both_returned_without_selected_winner():
    scores, ids, refs, times = evidence()
    scores[:] = [[.8, .1, .1, .95], [.1, .8, .1, .95], [.1, .1, .8, .95]]
    before = [x.copy() for x in (scores, ids, refs, times)]
    result = sparse.decode_sparse_tracks(scores, ids, refs, times)
    assert result["selected_variant"] is None and not result["confidence_calibrated"]
    assert result["outputs"]["raw"]["chosen_ref_ids"] == [ids[3]] * 3
    assert result["outputs"]["reference_mean"]["chosen_ref_ids"] == ids[:3].tolist()
    assert all(np.array_equal(x, old) for x, old in zip((scores, ids, refs, times), before))
    assert result["input_identity"]["raw_cosines_sha256"] == sparse._array_sha(scores)
    assert result["profile_sha256"] == sparse._json_sha(result["profile"])
    json.dumps(result, allow_nan=False)


def test_each_actual_interval_changes_transition_budget():
    scores, ids, refs, times = evidence(2)
    scores[:] = [[.95, .1], [.7, .8], [.1, .95]]
    refs[1, 1] = .009
    result = sparse.decode_sparse_tracks(scores, ids, refs, np.array([0., 1., 100.]))
    assert result["outputs"]["raw"]["chosen_ref_ids"] == [ids[0], ids[0], ids[1]]
    shorter = sparse.decode_sparse_tracks(scores, ids, refs, np.array([0., 1., 2.]))
    assert shorter["outputs"]["raw"]["chosen_ref_ids"] != result["outputs"]["raw"]["chosen_ref_ids"]


def test_half_second_transition_floor_is_fixed():
    scores, ids, refs, _ = evidence(2, 2)
    scores[:] = [[.95, .1], [.1, .95]]
    refs[1, 1] = .0008
    short = sparse.decode_sparse_tracks(scores, ids, refs, np.array([0., .1]))
    floor = sparse.decode_sparse_tracks(scores, ids, refs, np.array([0., .5]))
    assert short["outputs"]["raw"]["objective"] == floor["outputs"]["raw"]["objective"]
    assert short["outputs"]["raw"]["chosen_ref_ids"] == floor["outputs"]["raw"]["chosen_ref_ids"]


def brute_force(scores, refs, times, origin, *, centered):
    frame = LocalFrame(*origin)
    xy = np.column_stack(frame.to_xy(refs[:, 0], refs[:, 1]))
    emissions = scores.astype(float) - (scores.mean(axis=0, dtype=np.float64) if centered else 0.)
    best = None
    for path in itertools.product(range(len(refs)), repeat=len(times)):
        value = float(emissions[0, path[0]])
        for row, interval in enumerate(np.diff(times), 1):
            distance = float(np.linalg.norm(xy[path[row]] - xy[path[row - 1]]))
            value += float(emissions[row, path[row]]) - .02 * max(distance - 30. - 40. * max(float(interval), .5), 0.)
        if best is None or value > best[0]: best = value, list(path)
    return best


@pytest.mark.parametrize("seed", range(4))
def test_unpruned_small_problem_matches_independent_full_path_enumeration(seed):
    rng = np.random.default_rng(seed)
    scores, ids, refs, _ = evidence(3, 4)
    scores[:] = rng.uniform(-.2, .8, scores.shape)
    refs[:, 1] *= 10
    times = np.array([2., 2.1, 4., 15.])
    result = sparse.decode_sparse_tracks(scores, ids, refs, times)
    for variant, output in result["outputs"].items():
        objective, path = brute_force(scores, refs, times, result["input_identity"]["metric_origin"],
                                     centered=variant == "reference_mean")
        assert output["chosen_ref_indices"] == path
        assert output["objective"] == pytest.approx(objective, abs=1e-12)
        assert not output["candidate_pruning"]["pruned"]


def test_id_ties_and_rank_boundary_are_independent_of_reference_column_order():
    scores, ids, refs, times = evidence(100)
    scores[:] = .5
    result = sparse.decode_sparse_tracks(scores, ids, refs, times)
    order = np.random.default_rng(17).permutation(len(ids))
    permuted = sparse.decode_sparse_tracks(scores[:, order], ids[order], refs[order], times)
    for variant, output in result["outputs"].items():
        assert output["candidate_pruning"]["candidate_ids"] == [ids[:64].tolist()] * len(times)
        assert output["chosen_ref_ids"] == [ids[0]] * len(times)
        assert permuted["outputs"][variant]["chosen_ref_ids"] == output["chosen_ref_ids"]
        assert permuted["outputs"][variant]["route_latlon"] == output["route_latlon"]


def test_reference_means_and_results_are_identical_for_c_and_fortran_input_layout():
    scores, ids, refs, times = evidence(129, 80)
    scores = np.random.default_rng(0).uniform(-1, 1, scores.shape)
    first = sparse.decode_sparse_tracks(scores, ids, refs, times)
    second = sparse.decode_sparse_tracks(np.asfortranarray(scores), ids, refs, times)
    assert first == second


def test_top64_pruning_can_miss_global_best_and_is_not_labeled_full_dp():
    scores, ids, refs, times = evidence(129, 2)
    refs[:64, 1] = 0.
    refs[64:128, 1] = .01
    refs[128, 1] = .005
    scores[0, :64], scores[1, :64] = .5, -.5
    scores[0, 64:128], scores[1, 64:128] = -.5, .5
    scores[:, 128] = .4
    result = sparse.decode_sparse_tracks(scores, ids, refs, times)
    raw = result["outputs"]["raw"]
    assert ids[128] not in set(sum(raw["candidate_pruning"]["candidate_ids"], []))
    assert raw["objective"] < .8  # Full-reference path128→128 scores.8 with no penalty.
    assert raw["candidate_pruning"]["pruned"] and "not full-reference DP" in raw["candidate_pruning"]["claim"]


def test_large_full_pool_never_constructs_global_quadratic_transition(monkeypatch):
    scores, ids, refs, times = evidence(7001, 80)
    refs[:, 1] /= 100
    shapes = []
    original = np.hypot
    def bounded(x, y, *args, **kwargs):
        shape = np.broadcast_shapes(np.shape(x), np.shape(y))
        if len(shape) == 2:
            shapes.append(shape)
            assert shape[0] <= 64 and shape[1] <= 64
        return original(x, y, *args, **kwargs)
    monkeypatch.setattr(sparse.np, "hypot", bounded)
    result = sparse.decode_sparse_tracks(scores, ids, refs, times)
    assert shapes and max(a * b for a, b in shapes) == 4096
    assert result["input_identity"]["raw_matrix_shape"] == [80, 7001]
    for output in result["outputs"].values():
        assert output["candidate_pruning"]["retained_counts"] == [64] * 80
        assert output["candidate_pruning"]["largest_transition_values"] == 4096


def test_metric_smoothing_uses_local_plane_and_handles_wrapped_longitudes():
    scores, ids, refs, times = evidence(3)
    scores[:] = np.eye(3) * .8
    refs[:] = [[0., 179.9999], [0., -179.9999], [0., 179.9998]]
    result = sparse.decode_sparse_tracks(scores, ids, refs, times)
    raw = result["outputs"]["raw"]
    assert max(abs(p[1]) for p in raw["route_latlon"]) > 179.
    frame = LocalFrame(*result["input_identity"]["metric_origin"])
    xy = np.column_stack(frame.to_xy(refs[:, 0], refs[:, 1]))
    smooth = np.repeat(np.median(xy, axis=0)[None], 3, axis=0)
    expected = np.column_stack(frame.to_latlon(smooth[:, 0], smooth[:, 1]))
    np.testing.assert_allclose(raw["route_latlon"], expected, atol=1e-10)


@pytest.mark.parametrize("fault", ["one_query", "too_many_queries", "empty_pool", "bad_dtype", "nan", "cosine",
    "ids", "duplicate_id", "refs", "latitude", "times", "backwards", "negative", "antipodal"])
def test_invalid_or_unbounded_inputs_decline_explicitly(fault):
    scores, ids, refs, times = evidence()
    if fault == "one_query": scores, times = scores[:1], times[:1]
    elif fault == "too_many_queries": scores, times = np.zeros((81, 4)), np.arange(81.)
    elif fault == "empty_pool": scores, ids, refs = scores[:, :0], ids[:0], refs[:0]
    elif fault == "bad_dtype": scores = scores.astype(np.float16)
    elif fault == "nan": scores[0, 0] = np.nan
    elif fault == "cosine": scores[0, 0] = 1.01
    elif fault == "ids": ids = ids[:2]
    elif fault == "duplicate_id": ids[1] = ids[0]
    elif fault == "refs": refs = refs[:2]
    elif fault == "latitude": refs[0, 0] = 91.
    elif fault == "times": times = times[:2]
    elif fault == "backwards": times = times[::-1]
    elif fault == "negative": times -= 1.
    else: refs[:, 1] = [0., 180., 0., 180.]
    with pytest.raises(ValueError): sparse.decode_sparse_tracks(scores, ids, refs, times)


def test_reference_limit_is_checked_before_projection_or_matrix_copy(monkeypatch):
    monkeypatch.setattr(sparse, "LocalFrame", lambda *a: pytest.fail("budget must refuse first"))
    matrix = np.broadcast_to(np.float32(0.), (2, 100001))
    with pytest.raises(ValueError, match="budget"):
        sparse.decode_sparse_tracks(matrix, [], [], np.array([0., 1.]))


def test_profile_is_immutable_and_cannot_select_one_variant_after_scoring():
    with pytest.raises(TypeError): sparse.PROFILE["candidates_per_query"] = 128
    assert sparse.PROFILE["variants"] == ("raw", "reference_mean")
