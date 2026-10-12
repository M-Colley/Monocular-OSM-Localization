"""Route metrics (monocular_osm/metrics.py) on hand-computed geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyproj import CRS, Transformer

from monocular_osm.metrics import (
    THRESHOLDS_M,
    failed_score,
    local_transformer,
    project,
    score_route,
)

LAT0, LON0 = 48.40, 9.99


def _ll(xy) -> np.ndarray:
    """Local metres (east, north) around (LAT0, LON0) -> [lat, lon]."""
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +datum=WGS84 +units=m")
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xy = np.asarray(xy, dtype=float)
    lon, lat = inv.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lat, lon])


# GT: 1 km due east, a waypoint every 100 m.
GT_XY = np.array([[x, 0.0] for x in range(0, 1001, 100)], dtype=float)
GT = _ll(GT_XY)


def test_local_frame_is_metric() -> None:
    tf = local_transformer(LAT0, LON0)
    xy = project(GT, tf)
    # Re-centred on the same origin -> the input metres come back.
    np.testing.assert_allclose(xy, GT_XY, atol=1e-6)


def test_exact_route_scores_zero() -> None:
    s = score_route(GT, GT)
    for k in ("start_error_m", "end_error_m", "gt_to_pred_mean_m", "pred_to_gt_mean_m",
              "symmetric_mean_m", "hausdorff_m", "frechet_m"):
        assert getattr(s, k) == pytest.approx(0.0, abs=0.01), k
    assert s.pred_length_m == pytest.approx(1000.0, abs=0.01)
    assert not s.degenerate
    assert all(s.success["start_error_m"].values())


def test_parallel_offset_is_the_offset_everywhere() -> None:
    pred = _ll(GT_XY + [0.0, 100.0])
    s = score_route(pred, GT)
    for k in ("start_error_m", "end_error_m", "gt_to_pred_mean_m", "pred_to_gt_mean_m",
              "symmetric_mean_m", "hausdorff_m"):
        assert getattr(s, k) == pytest.approx(100.0, abs=0.05), k
    # Discrete Fréchet on 10 m-segmentized curves: exact up to the spacing.
    assert s.frechet_m == pytest.approx(100.0, abs=5.0)
    assert s.success["symmetric_mean_m"] == {
        "25": False, "50": False, "100": True, "250": True, "500": True, "1000": True}


def test_mid_segment_distance_is_exact_not_vertex_sampled() -> None:
    # A 2-vertex 1 km prediction 30 m north of a GT waypoint at its middle.
    pred = _ll([[0.0, 30.0], [1000.0, 30.0]])
    gt = _ll([[500.0, 0.0]])
    s = score_route(pred, gt)
    assert s.gt_to_pred_mean_m == pytest.approx(30.0, abs=0.01)


def test_one_directional_pathology_is_caught() -> None:
    """A prediction that follows the GT and then wanders 3 km away is
    perfect under the legacy GT->pred mean, terrible under pred->GT."""
    wander = [[1000.0, y] for y in range(100, 3001, 100)] + \
             [[x, 3000.0] for x in range(900, -1, -100)]
    pred = _ll(np.vstack([GT_XY, wander]))
    s = score_route(pred, GT)
    assert s.gt_to_pred_mean_m == pytest.approx(0.0, abs=0.01)      # legacy: "perfect"
    assert s.start_error_m == pytest.approx(0.0, abs=0.01)
    assert s.pred_to_gt_mean_m > 1000.0                            # most of it is far away
    assert s.symmetric_mean_m > 500.0
    assert s.hausdorff_m == pytest.approx(3000.0, abs=1.0)
    assert s.pred_length_m == pytest.approx(5000.0, abs=1.0)
    assert not s.success["symmetric_mean_m"]["250"]
    # Arc-length average of the distance to GT over the 5 km curve:
    # 1 km at 0 m, 3 km rising 0->3000 (mean 1500), 1 km at ~3000.
    expect = (1000 * 0 + 3000 * 1500 + 1000 * 3000) / 5000
    assert s.pred_to_gt_mean_m == pytest.approx(expect, rel=0.01)


def test_reversed_route_only_frechet_and_endpoints_object() -> None:
    pred = _ll(GT_XY[::-1])
    s = score_route(pred, GT)
    assert s.gt_to_pred_mean_m == pytest.approx(0.0, abs=0.01)
    assert s.pred_to_gt_mean_m == pytest.approx(0.0, abs=0.01)
    assert s.hausdorff_m == pytest.approx(0.0, abs=0.01)
    assert s.start_error_m == pytest.approx(1000.0, abs=0.05)
    assert s.frechet_m > 400.0     # order-aware: driving it backwards is wrong


def test_single_point_prediction_is_a_degenerate_route() -> None:
    pred = _ll([[0.0, 0.0]])
    s = score_route(pred, GT)
    assert s.degenerate
    assert s.pred_length_m == 0.0
    assert s.start_error_m == pytest.approx(0.0, abs=0.01)
    assert s.end_error_m == pytest.approx(1000.0, abs=0.05)
    assert s.gt_to_pred_mean_m == pytest.approx(500.0, abs=0.05)   # mean of 0..1000
    assert s.pred_to_gt_mean_m == pytest.approx(0.0, abs=0.01)     # point lies on GT
    assert s.symmetric_mean_m == pytest.approx(250.0, abs=0.05)
    assert s.hausdorff_m == pytest.approx(1000.0, abs=0.05)
    assert s.frechet_m == pytest.approx(1000.0, abs=0.05)
    # A route that collapses to one location (stationary) is a point too.
    s2 = score_route(_ll([[0.0, 0.0], [0.0, 0.0]]), GT)
    assert s2.degenerate


def test_success_flags_cover_all_thresholds() -> None:
    s = score_route(_ll(GT_XY + [0.0, 40.0]), GT)
    assert set(s.success) == {"start_error_m", "symmetric_mean_m"}
    assert list(s.success["start_error_m"]) == [str(t) for t in THRESHOLDS_M]
    assert s.success["start_error_m"]["25"] is False
    assert s.success["start_error_m"]["50"] is True


def test_failed_score_counts_against_the_method() -> None:
    f = failed_score(10)
    assert math.isinf(f["symmetric_mean_m"]) and math.isinf(f["start_error_m"])
    assert not any(f["success"]["start_error_m"].values())
    assert f["failed"] is True


@pytest.mark.parametrize("bad", [[], [[91.0, 0.0]], [[float("nan"), 1.0]], [[1, 2, 3]]])
def test_rejects_malformed_input(bad) -> None:
    with pytest.raises(ValueError):
        score_route(bad, GT)


def test_gt_to_pred_reproduces_the_legacy_evaluator() -> None:
    """metrics.gt_to_pred_mean_m == evaluator's mean route error (the README's
    headline number) on a fixture in the road graph's UTM CRS, within 1 m."""
    from monocular_osm.evaluator import evaluate_candidates_against_waypoints
    from monocular_osm.position import xy_to_latlon
    from monocular_osm.trajectory_matching import MatchCandidate

    crs = "EPSG:32632"
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = tf.transform(LON0, LAT0)
    # A wiggly 2.5 km camera path and GT waypoints scattered 0-300 m off it.
    t = np.linspace(0, 1, 60)
    traj = np.column_stack([x0 + 2000 * t, y0 + 400 * np.sin(6 * t) + 300 * t ** 2])
    rng = np.random.default_rng(3)
    idx = np.linspace(0, len(traj) - 1, 15).round().astype(int)
    wp_xy = traj[idx] + rng.normal(0, 120, size=(len(idx), 2))
    wp_xy[0] = traj[0] + [150.0, -80.0]

    class _Road:
        pass

    road = _Road()
    road.crs = crs
    cand = MatchCandidate(score=1.0, bearing_corr=0.5, start_node=0, walk=[],
                          walk_xy=traj, aligned_traj_xy=traj, walk_length_m=2500.0)
    gt_ll = xy_to_latlon(wp_xy, crs)
    legacy = evaluate_candidates_against_waypoints([cand], road, gt_ll)[0]
    s = score_route(xy_to_latlon(traj, crs), gt_ll)
    assert s.gt_to_pred_mean_m == pytest.approx(legacy.mean_route_error_m, abs=1.0)
    assert s.gt_to_pred_max_m == pytest.approx(legacy.max_route_error_m, abs=1.0)
    assert s.start_error_m == pytest.approx(legacy.start_error_m, abs=1.0)


# --- GEOS stack-overflow guard -------------------------------------------------------

def test_cap_vertices_keeps_endpoints_and_leaves_short_lines_alone() -> None:
    import numpy as np
    from shapely.geometry import LineString

    from monocular_osm.metrics import _cap_vertices

    short = LineString([(0, 0), (1, 0), (2, 0)])
    assert _cap_vertices(short) is short
    long_line = LineString(np.column_stack([np.arange(5000.0), np.zeros(5000)]))
    capped = _cap_vertices(long_line, 1500)
    coords = np.asarray(capped.coords)
    assert len(coords) == 1500
    assert coords[0].tolist() == [0.0, 0.0] and coords[-1].tolist() == [4999.0, 0.0]


def test_normal_routes_are_bit_identical_after_the_cap() -> None:
    """A 12 km route at 10 m spacing is 1,200 vertices: below the cap, so the
    Fréchet is computed on exactly the same geometry as before."""
    import numpy as np

    from monocular_osm.metrics import _cap_vertices, _segmentized, _geometry

    xy = np.column_stack([np.linspace(0, 12000, 40), np.sin(np.linspace(0, 9, 40)) * 300])
    seg = _segmentized(_geometry(xy), 10.0)
    assert _cap_vertices(seg) is seg


def test_a_very_long_prediction_does_not_kill_the_interpreter() -> None:
    """Run in a subprocess: before the cap this took the whole process down
    (native stack overflow in GEOS), which would kill pytest itself."""
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from pyproj import Transformer\n"
        "from monocular_osm.metrics import score_route\n"
        "rng = np.random.default_rng(0)\n"
        "gt_xy = np.column_stack([np.linspace(0, 2000, 30), np.zeros(30)])\n"
        "pred_xy = np.cumsum(rng.normal(0, 400, (400, 2)), axis=0)   # ~150 km of path\n"
        "inv = Transformer.from_crs('+proj=aeqd +lat_0=48.4 +lon_0=9.99 +datum=WGS84', 4326, always_xy=True)\n"
        "ll = lambda xy: np.column_stack(inv.transform(xy[:, 0], xy[:, 1])[::-1])\n"
        "s = score_route(ll(pred_xy), ll(gt_xy))\n"
        "assert s.pred_length_m > 100000 and np.isfinite(s.frechet_m), s\n"
        "print('OK', round(s.pred_length_m / 1000), 'km')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0 and "OK" in out.stdout, (out.returncode, out.stderr[-400:])
