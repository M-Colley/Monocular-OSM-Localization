import numpy as np
import pytest

from monocular_osm.heading_motion import crop_bearings_deg, local_yaw_evidence


def scene(degrees, dt=.1):
    # Clockwise world-camera optical bearing: camera->world Ry(theta).
    a = -np.radians(degrees)
    r = np.zeros((len(a), 3, 3), np.float64)
    r[:, 0, 0] = r[:, 2, 2] = np.cos(a)
    r[:, 0, 2] = np.sin(a)
    r[:, 2, 0] = -np.sin(a)
    r[:, 1, 1] = 1.
    n = len(a)
    return dict(rotations=r, valid=np.ones(n, bool), n_inliers=np.full(n, 20),
        timestamps=np.arange(n, dtype=np.float64)*dt, frame_ordinals=np.arange(n)*4)


def run(data, query=None):
    query = np.array([0, len(data["timestamps"])-1]) if query is None else np.asarray(query)
    return local_yaw_evidence(**data, query_times=data["timestamps"][query],
        query_ordinals=data["frame_ordinals"][query])


@pytest.mark.parametrize("direction", [1., -1.])
def test_clockwise_sign_comes_from_synthetic_camera_geometry(direction):
    data = scene(np.arange(11)*direction*2.)
    result = run(data)
    assert result["vo_yaw_delta_deg"][0] == pytest.approx(20.*direction)
    assert result["vo_quality"][0] == pytest.approx(1.)


def test_local_turn_wrap_and_nonzero_start_time():
    data = scene(np.arange(101)*2.+170)
    data["timestamps"] += 4.
    assert run(data)["vo_yaw_delta_deg"][0] == pytest.approx(-160.)


def test_invalid_held_rotation_is_missing_and_low_coverage_declines():
    data = scene(np.zeros(11))
    data["valid"][1:5] = False
    result = run(data)
    assert np.isnan(result["vo_yaw_delta_deg"][0])
    assert result["vo_quality"][0] == 0
    assert result["intervals"][0]["accepted_time_fraction"] == pytest.approx(.6)


def test_time_coverage_not_frame_fraction_and_long_gap_declines():
    data = scene(np.zeros(101), dt=.1)
    data["valid"][21:33] = False
    result = run(data)
    assert result["intervals"][0]["accepted_time_fraction"] == pytest.approx(.88)
    assert result["intervals"][0]["longest_missing_gap_s"] == pytest.approx(1.2)
    assert np.isnan(result["vo_yaw_delta_deg"][0])


def test_one_missing_step_is_excluded_from_sum_and_quality():
    data = scene(np.arange(11)*2.)
    data["valid"][3] = False
    result = run(data)
    assert result["vo_yaw_delta_deg"][0] == pytest.approx(18.)
    assert result["vo_quality"][0] == pytest.approx(.9)


@pytest.mark.parametrize("bad", ["inliers", "speed", "native_gap"])
def test_step_quality_gates(bad):
    data = scene(np.zeros(11))
    if bad == "inliers": data["n_inliers"][1:] = 11
    if bad == "speed": data = scene(np.arange(11)*7.)
    if bad == "native_gap": data["timestamps"] *= 11
    result = run(data)
    assert result["accepted_native_steps"] == 0
    assert np.isnan(result["vo_yaw_delta_deg"]).all()


@pytest.mark.parametrize("bad", ["time", "ordinal", "improper", "nonorthogonal", "nan", "valid_dtype"])
def test_malformed_or_mismatched_motion_declines(bad):
    data = scene(np.zeros(11))
    qt, qo = data["timestamps"][[0, 10]].copy(), data["frame_ordinals"][[0, 10]].copy()
    if bad == "time": qt[1] = np.nextafter(qt[1], np.inf)
    if bad == "ordinal": qo[1] += 1
    if bad == "improper": data["rotations"][3, 0, 0] = -1
    if bad == "nonorthogonal": data["rotations"][3, 0, 0] = .9
    if bad == "nan": data["rotations"][3, 0, 0] = np.nan
    if bad == "valid_dtype": data["valid"] = data["valid"].astype(int)
    with pytest.raises(ValueError): local_yaw_evidence(**data, query_times=qt, query_ordinals=qo)


def test_actual_crop_centers_use_native_intrinsics_and_rounding():
    k = np.array([[900., 0, 640.], [0, 900., 360.], [0, 0, 1.]])
    actual = crop_bearings_deg(k, (720, 1280))
    np.testing.assert_allclose(actual, np.degrees(np.arctan2(np.array([359.5, 639.5, 919.5])-640., 900.)))
    # Portrait crops move vertically; their horizontal bearing is identical.
    np.testing.assert_array_equal(crop_bearings_deg(k, (1280, 720)), np.repeat(actual[0], 3))


def test_skew_is_resolved_by_full_inverse_intrinsics():
    k = np.array([[900., 40, 640.], [0, 900., 360.], [0, 0, 1.]])
    center = np.array([639.5, 359.5, 1.])
    ray = np.linalg.solve(k, center)
    assert crop_bearings_deg(k, (720, 1280))[1] == pytest.approx(np.degrees(np.arctan2(ray[0], ray[2])))
