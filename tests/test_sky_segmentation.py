"""Offline tests for the learned sky-segmentation skyline extractor.

The SegFormer weights are never downloaded here: a fake segmenter returns a
hand-built sky mask so the row-extraction contract is tested deterministically.
"""

from __future__ import annotations

import numpy as np

from src.sky_segmentation import skyline_from_frame_seg


class _FakeSegmenter:
    """Returns a preset (h, w) boolean sky mask, ignoring the frame."""

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def sky_mask(self, frame_bgr, wh):
        return self._mask


WH = (8, 10)  # w=8, h=10


def _frame():
    return np.zeros((20, 16, 3), dtype=np.uint8)


def test_seg_skyline_reads_roofline_row() -> None:
    # columns 0-3: sky rows 0..4 then building -> roofline at row 5
    # columns 4-7: sky all the way down -> open (+inf)
    mask = np.zeros((10, 8), dtype=bool)
    mask[0:5, 0:4] = True
    mask[:, 4:8] = True
    rows = skyline_from_frame_seg(_frame(), WH, _FakeSegmenter(mask))
    assert np.allclose(rows[0:4], 5.0)
    assert np.isposinf(rows[4:8]).all()


def test_seg_skyline_top_not_sky_is_unusable() -> None:
    # a column whose very top row is NOT sky (occluder) -> NaN
    mask = np.ones((10, 8), dtype=bool)
    mask[0, 2] = False            # column 2 blocked at the top
    rows = skyline_from_frame_seg(_frame(), WH, _FakeSegmenter(mask))
    assert np.isnan(rows[2])
    assert np.isposinf(rows[0])   # other columns open


def test_seg_skyline_immediate_break_is_nan() -> None:
    # sky only in the top 1-2 rows then structure -> immediate break -> NaN
    mask = np.zeros((10, 8), dtype=bool)
    mask[0:2, :] = True
    rows = skyline_from_frame_seg(_frame(), WH, _FakeSegmenter(mask))
    assert np.isnan(rows).all()


def test_seg_skyline_format_matches_heuristic_contract() -> None:
    # finite / +inf / NaN only — never -inf or negatives
    mask = np.zeros((10, 8), dtype=bool)
    mask[0:6, 0:2] = True
    mask[:, 2:5] = True
    rows = skyline_from_frame_seg(_frame(), WH, _FakeSegmenter(mask))
    finite = rows[np.isfinite(rows)]
    assert (finite >= 0).all()
    assert not np.isneginf(rows).any()
