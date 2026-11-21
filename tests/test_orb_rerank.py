"""Synthetic descriptor/geometry/streaming contracts; no real pixels or labels."""
import gc
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import orb_rerank as orb


def test_stable_topk_finite_rows_ties_and_short_pool():
    scores = np.array([[.1, .8, .8, .2], [.4, .4, .3, .2]])
    assert orb.stable_topk(scores, 2).tolist() == [[1, 2], [0, 1]]
    assert orb.stable_topk(np.asfortranarray(scores), 8).tolist() == [[1, 2, 3, 0], [0, 1, 2, 3]]


def test_unsigned_numeric_scores_are_not_ranked_by_wrapped_negation():
    assert orb.stable_topk(np.array([[0, 1, 255]], np.uint8), 3).tolist() == [[2, 1, 0]]


@pytest.mark.parametrize("scores,k", [(np.array([[np.nan]]), 1), (np.ones(3), 1), (np.empty((0, 3)), 1),
                                       (np.ones((1, 2)), 0), (np.ones((1, 2)), True)])
def test_invalid_topk_inputs_refused(scores, k):
    with pytest.raises(ValueError): orb.stable_topk(scores, k)


def descriptors():
    return np.array([[0]*32, [255]*32, [85]*32], np.uint8)


def test_reciprocal_hamming_ratio_matches_actual_descriptor_permutation():
    first = descriptors(); second = first[[2, 0, 1]]
    assert orb.mutual_ratio_matches(first, second).tolist() == [[0, 1], [1, 2], [2, 0]]


def test_ambiguous_equal_zero_distance_neighbours_fail_lowe_ratio():
    first = np.array([[0]*32, [0]*32, [255]*32], np.uint8)
    assert orb.mutual_ratio_matches(first, first).tolist() == [[2, 2]]


@pytest.mark.parametrize("which", [None, np.empty((0, 32), np.uint8), np.zeros((1, 32), np.uint8)])
def test_absent_or_short_descriptors_give_typed_empty_matches(which):
    matches = orb.mutual_ratio_matches(which, descriptors())
    assert matches.shape == (0, 2) and matches.dtype.kind == "i"
    assert orb.mutual_ratio_matches(descriptors(), which).shape == (0, 2)


def test_convex_hull_coverage_is_normalized_and_degenerate_is_zero():
    points = np.array([[10., 10.], [60., 10.], [60., 60.], [10., 60.]])
    assert orb.coverage(points, (100, 100)) == pytest.approx(.25)
    assert orb.coverage(np.array([[0., 0.], [1., 1.], [2., 2.]]), (100, 100)) == 0


def correspondences():
    return np.array([[10, 10], [80, 10], [80, 80], [10, 80], [20, 20], [60, 20], [20, 60], [60, 60]], float)


def fundamental():
    return np.array([[0., -1., 0.], [1., 0., -1.], [0., 1., 0.]])


def test_fixed_magsac_parameters_and_spread_weighted_score():
    import cv2
    a = correspondences(); b = a*.5
    def find(pa, pb, **kwargs):
        assert kwargs == {"method": cv2.USAC_MAGSAC, "ransacReprojThreshold": 3., "confidence": .999, "maxIters": 10000}
        return fundamental(), np.ones((8, 1), np.uint8)
    result = orb.verify_geometry(a, b, (100, 100), (100, 100), find_fn=find)
    assert result["inlier_count"] == 8 and result["score"] == pytest.approx(8*.1225)
    assert result["query_coverage"] == pytest.approx(.49)
    assert result["reference_coverage"] == pytest.approx(.1225)


def test_none_fundamental_never_reads_an_uninitialized_mask():
    class Unreadable:
        def __array__(self, *args, **kwargs): pytest.fail("mask must not be read when F is None")
    result = orb.verify_geometry(correspondences(), correspondences(), (100, 100), (100, 100),
                                 find_fn=lambda *a, **k: (None, Unreadable()))
    assert result["score"] == 0 and result["inlier_count"] == 0


@pytest.mark.parametrize("F,mask", [(np.eye(2), np.ones(8)), (np.eye(3), np.ones(8)),
    (np.zeros((3, 3)), np.ones(8)), (np.full((3, 3), np.nan), np.ones(8)),
    (fundamental(), None), (fundamental(), np.ones(7)), (fundamental(), np.full(8, 2)),
    (fundamental(), np.full(8, np.nan)), (fundamental(), np.array([1]*7+[0]))])
def test_invalid_fundamental_or_inlier_masks_fail_closed(F, mask):
    result = orb.verify_geometry(correspondences(), correspondences(), (100, 100), (100, 100),
                                 find_fn=lambda *a, **k: (F, mask))
    assert result["score"] == 0 and result["inlier_count"] == 0


def test_insufficient_matches_never_call_the_geometry_solver():
    result = orb.verify_geometry(correspondences()[:7], correspondences()[:7], (100, 100), (100, 100),
                                 find_fn=lambda *a, **k: pytest.fail("not enough correspondences"))
    assert result["score"] == 0


def test_rank_uses_max_view_not_sum_then_cosine_then_index():
    row = np.array([.4, .5, .5])
    records = [{"ref_index": 0, "view": view, "score": 3., "inlier_count": 10} for view in range(3)]
    records += [{"ref_index": 1, "view": 1, "score": 4., "inlier_count": 12},
                {"ref_index": 2, "view": 2, "score": 4., "inlier_count": 12}]
    result = orb.rank_candidates(row, np.array([1, 2, 0]), records)
    assert result["selected_ref_index"] == 1
    assert orb.rank_candidates(row, np.array([1, 2, 0]), records[::-1])["selected_ref_index"] == 1


def test_zero_geometric_evidence_keeps_original_centered_cosine_winner():
    row = np.array([.1, .8, .2])
    records = [{"ref_index": index, "view": view, "score": 0., "inlier_count": 0}
               for index in range(3) for view in range(3)]
    result = orb.rank_candidates(row, np.array([1, 2, 0]), records)
    assert result["selected_ref_index"] == result["original_ref_index"] == 1


@pytest.fixture
def pixel_store(tmp_path):
    path = tmp_path/"references.npz"
    pixels = np.arange(3*8*9*3, dtype=np.uint8).reshape(3, 8, 9, 3)
    np.savez(path, ids=np.array(["11", "22", "33"]), raw=pixels, xy=np.zeros((3, 2)))
    root = tmp_path/"isolated_temp"; root.mkdir()
    return path, pixels, root


def test_raw_store_streams_without_materializing_raw_zip_member_and_maps_readonly(pixel_store, monkeypatch):
    path, pixels, root = pixel_store
    original = np.lib.npyio.NpzFile.__getitem__
    def item(saved, key):
        assert key != "raw", "raw member must be streamed, not loaded by NumPy"
        return original(saved, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", item)
    expected = orb.file_sha256(path)
    raw, record = orb.stream_raw_store(path, root/"pixels.npy", expected_sha256=expected,
        expected_ids=["11", "22", "33"], allowed_temp_root=root)
    try:
        assert isinstance(raw, np.memmap) and not raw.flags.writeable
        np.testing.assert_array_equal(raw, pixels)
        assert expected in json.dumps(record) and orb.file_sha256(path) == expected
        with pytest.raises(ValueError): raw[0, 0, 0, 0] = 7
    finally:
        raw._mmap.close(); del raw; gc.collect()


@pytest.mark.parametrize("fault", ["hash", "id_order", "prefixed_ids", "outside", "existing"])
def test_raw_stream_refuses_changed_identity_or_unsafe_destination(pixel_store, tmp_path, fault):
    path, pixels, root = pixel_store
    dest, expected, ids = root/"pixels.npy", orb.file_sha256(path), ["11", "22", "33"]
    if fault == "hash": expected = "0"*64
    elif fault == "id_order": ids.reverse()
    elif fault == "prefixed_ids": ids = ["mapillary:"+value for value in ids]
    elif fault == "outside": dest = tmp_path/"outside.npy"
    else: dest.write_bytes(b"preserve existing")
    with pytest.raises(ValueError):
        orb.stream_raw_store(path, dest, expected_sha256=expected, expected_ids=ids, allowed_temp_root=root)
    if fault == "existing": assert dest.read_bytes() == b"preserve existing"
