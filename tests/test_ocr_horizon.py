"""Fixed OCR samples on the VO horizon: two clocks for one boundary frame."""

import pytest

from monocular_osm.stages.poi_only import _snap_to_horizon

# Detroit g5lnpYCk1Ec: VO frame 119 at 119 / 29.854 s; the container PTS of the
# same frame reads 3.986 s, 0.07 ms earlier, which used to fail the whole OCR cue.
FIRST, LAST, FPS = 119 / 29.854, 17879 / 29.854, 29.854


def test_the_boundary_frame_read_on_the_container_clock_is_the_vo_boundary():
    assert _snap_to_horizon(3.986, FIRST, LAST, FPS) == FIRST
    assert _snap_to_horizon(LAST + 0.004, FIRST, LAST, FPS) == LAST


def test_interior_times_are_untouched():
    for t in (FIRST, 29.845, 300.0, LAST):
        assert _snap_to_horizon(t, FIRST, LAST, FPS) == pytest.approx(t, abs=0)


@pytest.mark.parametrize("t", [FIRST - 0.6 / FPS, LAST + 0.6 / FPS, 0.0, 700.0])
def test_a_different_frame_is_left_outside_for_the_caller_to_refuse(t):
    out = _snap_to_horizon(t, FIRST, LAST, FPS)
    assert out == t and not FIRST <= out <= LAST
