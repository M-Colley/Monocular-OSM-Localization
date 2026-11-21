"""Regional decoding tests use retrieval evidence and known synthetic motion."""

import numpy as np
import pytest

from monocular_osm.regional_retrieval import fill_region_corridors, merge_region_tracks
from monocular_osm.geo import LocalFrame, geodesic_distance_m


def _track(lon, similarities):
    similarities = np.asarray(similarities, dtype=float)
    n = len(similarities)
    coords = np.c_[np.full(n, 48.4), lon + np.arange(n) * 0.0001]
    return np.arange(n), coords, similarities


def test_a_consistently_better_region_wins_without_coordinate_averaging():
    weak = _track(9.99, [0.15] * 5)
    strong = _track(10.19, [0.30] * 5)
    track, diagnostics = merge_region_tracks([weak, strong], np.arange(5))
    np.testing.assert_array_equal(track[1], strong[1])
    assert diagnostics["region_frame_counts"] == [0, 5]


def test_one_confident_far_match_does_not_teleport_the_regional_track():
    stable = _track(9.99, [0.20] * 5)
    outlier = _track(10.19, [0.10, 0.10, 0.55, 0.10, 0.10])
    track, diagnostics = merge_region_tracks([stable, outlier], np.arange(5))
    np.testing.assert_array_equal(track[1], stable[1])
    assert diagnostics["selected_regions"] == [0] * 5


def test_the_route_can_move_between_overlapping_regions_when_evidence_changes():
    first = _track(9.99, [0.40, 0.40, 0.10, 0.10])
    second = _track(9.9901, [0.10, 0.10, 0.40, 0.40])
    track, diagnostics = merge_region_tracks([first, second], np.arange(4))
    np.testing.assert_array_equal(track[1], np.vstack([first[1][:2], second[1][2:]]))
    assert diagnostics["selected_regions"] == [0, 0, 1, 1]


def test_misaligned_query_grids_are_rejected():
    first = _track(9.99, [0.3] * 4)
    other = _track(10.19, [0.2] * 4)
    with pytest.raises(ValueError, match="same query frames"):
        merge_region_tracks([first, (other[0] + 1, other[1], other[2])], np.arange(6))


def test_empty_or_nonfinite_evidence_does_not_produce_an_answer():
    assert merge_region_tracks([], [0, 1])[0] is None
    broken = _track(9.99, [0.2, np.nan])
    assert merge_region_tracks([broken], [0, 1])[0] is None


def _regions(points, radii=3000):
    frame = LocalFrame(48.4, 9.99)
    if np.isscalar(radii):
        radii = [radii] * len(points)
    regions = []
    for i, (point, radius) in enumerate(zip(points, radii)):
        lat, lon = frame.to_latlon(*point)
        regions.append({"name": chr(ord("A") + i), "center": [lat, lon],
                        "radius_m": radius, "origin": "video_named"})
    return frame, regions


def _region_xy(frame, region):
    return np.array(frame.to_xy(*region["center"]))


def test_corridor_fills_the_largest_gap_and_preserves_named_regions():
    from copy import deepcopy
    frame, regions = _regions([(0, 0), (20_000, 0)])
    original = deepcopy(regions)
    result = fill_region_corridors(regions, max_regions=3)
    assert result[:2] == original and regions == original
    assert len(result) == 3
    np.testing.assert_allclose(_region_xy(frame, result[-1]), [10_000, 0], atol=0.01)
    assert result[-1]["origin"] == "named_corridor"
    assert result[-1]["corridor_between"] == ["A", "B"]


def test_corridor_clearance_accounts_for_the_original_eight_km_disc():
    frame, regions = _regions([(0, 0), (20_000, 0)], [8000, 3000])
    result = fill_region_corridors(regions, max_regions=3)
    # The uncovered span is 8..17 km, whose farthest point is 12.5 km.
    # Averaging the named centers would waste coverage nearer the larger disc.
    np.testing.assert_allclose(_region_xy(frame, result[-1]), [12_500, 0], atol=0.01)
    assert result[0]["radius_m"] == 8000


def test_corridor_stops_when_covered_and_does_not_spend_the_whole_budget():
    frame, regions = _regions([(0, 0), (20_000, 0)])
    result = fill_region_corridors(regions, max_regions=6)
    assert 2 < len(result) < 6
    samples = np.c_[np.linspace(0, 20_000, 2001), np.zeros(2001)]
    margins = np.min([np.linalg.norm(samples - _region_xy(frame, r), axis=1) - r["radius_m"]
                      for r in result], axis=0)
    assert margins.max() <= 0.01
    assert fill_region_corridors(result, max_regions=6) == result


def test_already_overlapping_discs_need_no_bridge():
    _frame, regions = _regions([(0, 0), (5000, 0)])
    assert fill_region_corridors(regions) == regions
    assert fill_region_corridors([]) == []
    assert fill_region_corridors(regions[:1]) == regions[:1]


def test_other_large_discs_affect_the_farthest_point_on_an_mst_edge():
    frame, regions = _regions([(0, 0), (20_000, 0), (10_000, 2000), (10_000, 20_000)],
                              [9000, 3000, 3000, 3000])
    result = fill_region_corridors(regions, max_regions=5)
    # MST edges are A-C, B-C, C-D. On C-D the larger A disc suppresses
    # its midpoint. Its boundary-distance tie with D is y=576000/52 m.
    np.testing.assert_allclose(_region_xy(frame, result[-1]), [10_000, 576_000 / 52], atol=0.02)
    assert result[-1]["corridor_between"] == ["C", "D"]


def test_added_centers_stay_on_mst_paths_instead_of_filling_a_city_bbox():
    frame, regions = _regions([(0, 0), (20_000, 0), (0, 20_000)])
    result = fill_region_corridors(regions, max_regions=6)
    assert len(result) == 6
    for region in result[3:]:
        x, y = _region_xy(frame, region)
        assert min(abs(x), abs(y)) < 0.02
        assert -0.02 <= x <= 20_000.02 and -0.02 <= y <= 20_000.02
    assert fill_region_corridors(regions, max_regions=6) == result


def test_corridor_crossing_the_dateline_uses_the_short_named_connection():
    regions = [{"name": "west", "center": [0, 179.9], "radius_m": 3000},
               {"name": "east", "center": [0, -179.9], "radius_m": 3000}]
    result = fill_region_corridors(regions, max_regions=3)
    center = result[-1]["center"]
    assert abs(abs(center[1]) - 180) < 1e-6
    via = sum(geodesic_distance_m(*r["center"], *center) for r in regions)
    assert via == pytest.approx(geodesic_distance_m(*regions[0]["center"],
                                                   *regions[1]["center"]), abs=0.01)


def test_duplicate_centers_and_exhausted_budgets_preserve_all_originals():
    _frame, regions = _regions([(0, 0), (0, 0), (20_000, 0)])
    assert fill_region_corridors(regions, max_regions=2) == regions
    assert fill_region_corridors(regions[:2]) == regions[:2]


@pytest.mark.parametrize("center,radius", [
    ([np.nan, 0], 3000), ([0, np.inf], 3000), ([91, 0], 3000), ([0, 181], 3000),
    ([0], 3000), ([0, 0], np.inf), ([0, 0], np.nan), ([0, 0], 0), ([0, 0], -1),
])
def test_corridor_rejects_invalid_region_geometry(center, radius):
    with pytest.raises(ValueError, match="finite WGS84|WGS84 center"):
        fill_region_corridors([{"center": center, "radius_m": radius}])


@pytest.mark.parametrize("budget", [0, -1, 2.5, np.inf, True])
def test_corridor_requires_a_positive_integer_budget(budget):
    with pytest.raises(ValueError, match="max_regions"):
        fill_region_corridors([], max_regions=budget)


@pytest.mark.parametrize("radius", [0, -1, np.inf, np.nan, None])
def test_corridor_requires_a_finite_positive_new_radius(radius):
    with pytest.raises(ValueError, match="radius_m"):
        fill_region_corridors([], radius_m=radius)
