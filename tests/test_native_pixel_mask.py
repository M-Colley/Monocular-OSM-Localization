"""Synthetic pixels only. No video/image/model/detector execution."""
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


from monocular_osm import native_pixel_mask as m


def pixels(height=40, width=64):
    y, x = np.indices((height, width))
    return np.stack(((x + 2 * y + 1) % 251, (3 * x + y + 17) % 253,
                     (x + 5 * y + 83) % 255), axis=-1).astype(np.uint8)


def empty():
    return np.empty((0, 4, 2), dtype=np.float32)


def box(left=24, top=17, right=30, bottom=20):
    return np.array([[[left, top], [right, top], [right, bottom], [left, bottom]]], dtype=np.float64)


def test_fixed_quarter_bands_only_and_channel_preservation():
    original = pixels()
    before = original.copy()
    masked, mask, receipt = m.mask_native_bgr(original, empty())
    assert np.array_equal(original, before)
    assert masked.dtype == original.dtype and masked.shape == original.shape
    assert mask.dtype == np.bool_ and mask.shape == original.shape[:2]
    assert mask[:10].all() and mask[-10:].all() and not mask[10:30].any()
    assert (masked[mask] == 0).all()
    assert np.array_equal(masked[~mask], original[~mask])
    assert not np.shares_memory(masked, original)
    assert receipt["masked_pixel_count"] == 20 * 64
    assert receipt["native_BGR_plain_C_bytes_sha256"] == hashlib.sha256(before.tobytes()).hexdigest()
    assert receipt["detector_execution_and_recall_certified_by_this_function"] is False


def test_every_box_gets_exact_four_pixel_inclusive_padding():
    original = pixels()
    masked, mask, receipt = m.mask_native_bgr(original, box())
    expected = np.zeros((40, 64), dtype=bool)
    expected[:10] = expected[-10:] = True
    expected[13:25, 20:35] = True
    assert np.array_equal(mask, expected)
    assert receipt["expanded_inclusive_rectangles"] == [[20, 13, 34, 24]]
    assert np.array_equal(masked[~mask], original[~mask])


def test_fractional_rotated_quad_conservatively_enclosed():
    b = np.array([[[22.2, 16.7], [30.1, 17.2], [28.8, 21.3], [20.9, 19.4]]])
    _, mask, r = m.mask_native_bgr(pixels(), b)
    assert r["expanded_inclusive_rectangles"] == [[16, 12, 35, 26]]
    assert mask[12:27, 16:36].all()
    assert not mask[11, 16] and not mask[12, 15]


def test_independent_scalar_pixel_membership_and_outside_bytes():
    native = pixels(33, 47)
    b = np.array([[[2.2, 12.3], [8.7, 13.1], [7.1, 18.8], [1.4, 17.3]],
                  [[31.2, 15.4], [44.8, 15.2], [45.1, 20.1], [30.7, 21.2]]])
    masked, mask, _ = m.mask_native_bgr(native, b)
    for y in range(33):
        for x in range(47):
            expected = y < 9 or y >= 24
            for quad in b.tolist():
                expected |= (math.floor(min(p[0] for p in quad)) - 4 <= x <= math.ceil(max(p[0] for p in quad)) + 4
                             and math.floor(min(p[1] for p in quad)) - 4 <= y <= math.ceil(max(p[1] for p in quad)) + 4)
            assert bool(mask[y, x]) == expected
            assert np.array_equal(masked[y, x], np.zeros(3, dtype=np.uint8) if expected else native[y, x])


def test_box_order_duplicates_and_vertex_order_do_not_change_pixels():
    b = np.concatenate((box(), box(40, 13, 45, 16), box()))
    a, mask_a, receipt_a = m.mask_native_bgr(pixels(), b)
    c, mask_c, receipt_c = m.mask_native_bgr(pixels(), b[::-1, ::-1])
    assert np.array_equal(a, c) and np.array_equal(mask_a, mask_c)
    assert receipt_a == receipt_c and receipt_a["detector_box_count"] == 3


def test_clamp_only_expanded_rectangle_to_native_edges():
    _, _, r = m.mask_native_bgr(pixels(), box(0, 0, 63, 39))
    assert r["expanded_inclusive_rectangles"] == [[0, 0, 63, 39]]
    assert r["masked_pixel_count"] == 40 * 64


@pytest.mark.parametrize("height,expected", [(1, 1), (2, 1), (3, 1), (5, 2), (9, 3), (720, 180)])
def test_ceil_quarter_policy_on_odd_and_known_native_heights(height, expected):
    native = pixels(height, 12)
    masked, mask, r = m.mask_native_bgr(native, empty())
    assert r["band_rows_each"] == expected
    assert np.array_equal(masked[~mask], native[~mask])
    assert masked.shape == native.shape


def test_same_profile_for_all_four_registered_native_cases():
    # Labels/ordinals cannot influence the pure function: it has no such inputs.
    for _ordinal in (0, 8847, 119, 17879):
        _, mask, r = m.mask_native_bgr(pixels(720, 1280), empty())
        assert r["profile_sha256"] == m.PROFILE_SHA256
        assert mask[:180].all() and mask[-180:].all() and not mask[180:540].any()


@pytest.mark.parametrize("kind", ["readonly", "noncontiguous"])
def test_readonly_and_noncontiguous_native_input_never_mutated(kind):
    native = pixels()
    if kind == "readonly":
        native.setflags(write=False)
    else:
        native = native[:, ::-1]
    before = native.copy()
    masked, mask, _ = m.mask_native_bgr(native, box())
    assert np.array_equal(native, before) and np.array_equal(masked[~mask], before[~mask])


@pytest.mark.parametrize("bad", [None, [], np.empty((0, 4)), np.empty((1, 5, 2)),
                                  np.zeros((1, 4, 2), dtype=bool),
                                  np.zeros((1, 4, 2), dtype=complex),
                                  np.zeros((1, 4, 2), dtype=object),
                                  np.full((1, 4, 2), np.nan),
                                  np.full((1, 4, 2), np.inf),
                                  np.zeros((1001, 4, 2))])
def test_missing_malformed_or_over_budget_boxes_fail_closed(bad):
    with pytest.raises(ValueError):
        m.mask_native_bgr(pixels(), bad)


@pytest.mark.parametrize("axis,value", [(0, -0.01), (0, 64), (1, -1), (1, 40)])
def test_reject_scaled_or_out_of_frame_detector_coordinates(axis, value):
    b = box()
    b[0, 0, axis] = value
    with pytest.raises(ValueError, match="native grid"):
        m.mask_native_bgr(pixels(), b)


@pytest.mark.parametrize("bad", [None, [], np.zeros((4, 5), dtype=np.uint8),
                                  np.zeros((4, 5, 4), dtype=np.uint8),
                                  np.zeros((4, 5, 3), dtype=np.float32),
                                  np.zeros((0, 5, 3), dtype=np.uint8)])
def test_malformed_frames_refused(bad):
    with pytest.raises(ValueError):
        m.mask_native_bgr(bad, empty())


def test_large_frame_refused_before_mask_allocation():
    # Zero-stride view requires no large allocation to exercise the input bound.
    native = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (4001, 4000, 3))
    with pytest.raises(ValueError, match="pixel budget"):
        m.mask_native_bgr(native, empty())


def test_profile_returns_unshared_copy_and_no_neural_imports():
    first = m.profile()
    first["fill_BGR"][0] = 200
    assert m.profile()["fill_BGR"] == [0, 0, 0]
    assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() == "30ad5721fb0ced78f625807bfdb316c0056ec0234f5bff7021ef69d961fdae10"
