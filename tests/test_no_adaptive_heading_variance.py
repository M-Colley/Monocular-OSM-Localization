"""Pure copied odometry and synthetic genuine coarsening; no actual assets."""
import hashlib
import json
import math
import numpy as np
import pytest
from monocular_osm import no_adaptive_heading_variance as q, junction_sequence as j
from monocular_osm.graph_filter import Odometry, GraphParticleFilter, FilterConfig
from tests.test_junction_sequence import roads


def original(count=20, metric=False):
    return Odometry(np.ones(count), np.linspace(0., .2, count), np.full(count, .1),
        metric=metric, unit_seconds=None if metric else .1,
        frame_index=np.arange(10, 10 + count, dtype=np.int64),
        heading_var=np.full(count, math.radians(3.)**2))


def snapshots(odometry):
    return {name: (getattr(odometry, name).tobytes(), getattr(odometry, name).dtype.str,
                   getattr(odometry, name).flags.writeable, getattr(odometry, name).strides)
        for name in ("step", "heading", "dt", "frame_index", "heading_var")}


@pytest.mark.parametrize("metric", [False, True])
def test_independent_exact_copy_and_only_F64_positive_zero_variance(metric):
    source = original(metric=metric)
    source.step[2] = -0.
    source.heading.view(np.uint64)[3] = 0x7ff8000000000042  # retain actual NaN payload
    source.heading[4] = -0.
    source.original_stats = {"n_inliers": "external caller provenance, never admitted here"}
    for name in ("step", "heading", "dt", "frame_index", "heading_var"):
        getattr(source, name).flags.writeable = False
    before = snapshots(source)
    result = q.without_adaptive_heading_variance(source)
    assert type(result) is Odometry and result is not source
    assert result.metric is source.metric and result.unit_seconds is source.unit_seconds
    assert result.original_stats == source.original_stats
    for name in ("step", "heading", "dt", "frame_index"):
        actual, expected = getattr(result, name), getattr(source, name)
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        assert not np.shares_memory(actual, expected)
    assert result.heading_var.dtype == np.float64 and result.heading_var.shape == source.step.shape
    np.testing.assert_array_equal(result.heading_var.view(np.uint64), np.zeros(len(source), np.uint64))
    result.step[0] = 8.
    result.heading[0] = .7
    result.dt[0] = 2.
    result.frame_index[0] = 0
    result.heading_var[0] = 9.
    assert snapshots(source) == before


def test_missing_adaptive_variance_becomes_explicit_zeros_without_claiming_precision():
    source = original()
    source.heading_var = None
    result = q.without_adaptive_heading_variance(source)
    assert source.heading_var is None
    np.testing.assert_array_equal(result.heading_var, np.zeros(len(source)))
    assert q.PROFILE["measured_covariance"] is False and q.PROFILE["calibrated_covariance"] is False


def test_genuine_coarsen_preserves_all_means_SE2_status_clock_and_unknown_noise():
    source = original()
    status = np.full(len(source), "measured", dtype="U10")
    status[6:9] = "unknown"
    status[15] = "stationary"
    source.step[6:9] = 0.
    source.heading[6:9] = np.nan
    source.step[15] = 0.
    clock = 100. + np.arange(len(source) + 1, dtype=np.float64) * .1
    source.dt = np.diff(clock)
    before, clock_before, status_before = snapshots(source), clock.tobytes(), status.tobytes()
    result = q.without_adaptive_heading_variance(source)
    baseline = j.coarsen_motion(source, status, clock)
    control = j.coarsen_motion(result, status, clock)
    for name in ("times", "step", "heading", "unknown_seconds", "stationary_seconds",
                 "native_end_indices", "counts", "se2"):
        np.testing.assert_array_equal(getattr(control, name), getattr(baseline, name))
    assert control.unit_seconds == baseline.unit_seconds == source.unit_seconds
    expected_unknown = math.radians(5.)**2 * control.unknown_seconds
    np.testing.assert_array_equal(control.heading_var, expected_unknown)
    assert np.any(control.heading_var > 0) and np.all(control.heading_var[control.unknown_seconds == 0] == 0)
    lo = 0
    for row, hi in enumerate(control.native_end_indices):
        expected = float(source.heading_var[lo:hi].sum())
        assert baseline.heading_var[row] - control.heading_var[row] == pytest.approx(expected, rel=1e-14)
        lo = int(hi)
    assert snapshots(source) == before and clock.tobytes() == clock_before and status.tobytes() == status_before


def test_genuine_predict_still_adds_fixed2deg_drift_and_passed_unknown_contribution():
    cfg = FilterConfig(n_particles=4, seed=0, path_storage=False, delta_rw_deg=2., heading_sigma_deg=15.)
    pf = GraphParticleFilter(roads(), cfg, metric=False, unit_seconds=.1)
    pf.delta_ready = True
    pf.delta_var.fill(math.radians(10.)**2)
    old = pf.delta_var.copy()
    unknown_extra = math.radians(5.)**2 * .3
    pf.predict(1., 0., .5, heading_var=unknown_extra)
    np.testing.assert_allclose(pf.delta_var - old, math.radians(2.)**2 * .5 + unknown_extra,
                               rtol=1e-14, atol=0)
    assert pf.cfg.heading_sigma_deg == 15. and pf.cfg.delta_rw_deg == 2.


def test_full_native_bound_payload_and_exact_scalar_and_integer_types():
    source = original(100_000)
    source.metric = np.bool_(False)
    source.unit_seconds = np.float64(.1)
    source.frame_index = source.frame_index.astype(np.uint64)
    result = q.without_adaptive_heading_variance(source)
    assert type(result.metric) is type(source.metric) and result.metric is source.metric
    assert type(result.unit_seconds) is type(source.unit_seconds) and result.unit_seconds is source.unit_seconds
    assert result.frame_index.dtype == np.uint64
    assert sum(getattr(result, name).nbytes for name in
        ("step", "heading", "dt", "frame_index", "heading_var")) == 4_000_000
    np.testing.assert_array_equal(result.step, source.step)


@pytest.mark.parametrize("fault", ["empty", "overcap", "foreign", "step_type", "step_shape", "step_negative",
    "step_nan", "heading_inf", "heading_shape", "dt_type", "dt_zero", "dt_nan", "frame_type", "frame_negative",
    "frame_order", "variance_type", "variance_shape", "variance_negative", "variance_inf", "metric_alias",
    "unit_missing", "unit_bool", "unit_zero", "unit_inf", "unit_huge"])
def test_type_bounds_units_and_source_validity_before_any_output_allocation(monkeypatch, fault):
    source = original(0 if fault == "empty" else 100_001 if fault == "overcap" else 20)
    if fault == "foreign": source = object()
    if fault == "step_type": source.step = source.step.astype(np.float32)
    if fault == "step_shape": source.step = source.step[:, None]
    if fault == "step_negative": source.step[0] = -1.
    if fault == "step_nan": source.step[0] = np.nan
    if fault == "heading_inf": source.heading[0] = np.inf
    if fault == "heading_shape": source.heading = source.heading[:-1]
    if fault == "dt_type": source.dt = source.dt.astype(np.float32)
    if fault == "dt_zero": source.dt[0] = 0.
    if fault == "dt_nan": source.dt[0] = np.nan
    if fault == "frame_type": source.frame_index = source.frame_index.astype(np.float64)
    if fault == "frame_negative": source.frame_index[0] = -1
    if fault == "frame_order": source.frame_index[2] = source.frame_index[1]
    if fault == "variance_type": source.heading_var = source.heading_var.astype(np.float32)
    if fault == "variance_shape": source.heading_var = source.heading_var[:-1]
    if fault == "variance_negative": source.heading_var[0] = -1.
    if fault == "variance_inf": source.heading_var[0] = np.inf
    if fault == "metric_alias": source.metric = 1
    if fault == "unit_missing": source.unit_seconds = None
    if fault == "unit_bool": source.unit_seconds = True
    if fault == "unit_zero": source.unit_seconds = 0.
    if fault == "unit_inf": source.unit_seconds = np.inf
    if fault == "unit_huge": source.unit_seconds = 10**1000
    monkeypatch.setattr(q, "_copy", lambda *a: pytest.fail("object cloned before complete validation"))
    monkeypatch.setattr(q.np, "zeros", lambda *a, **k: pytest.fail("output allocated before validation"))
    with pytest.raises(ValueError):
        q.without_adaptive_heading_variance(source)


def test_source_profile_and_RNG_are_unchanged_and_no_admission_is_claimed():
    state = np.random.get_state()
    q.without_adaptive_heading_variance(original())
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    expected = hashlib.sha256(json.dumps(dict(q.PROFILE), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert q.PROFILE_SHA256 == expected
    assert q.PROFILE["fixed_bias_random_walk_deg_per_sqrt_s_untouched"] == 2.
    assert q.PROFILE["unknown_motion_sigma_deg_per_sqrt_s_untouched"] == 5.
    for key in ("measured_covariance", "calibrated_covariance", "native_clock_or_input_admission",
                "historical_source_upgraded", "policy_admission"):
        assert q.PROFILE[key] is False
    with pytest.raises(TypeError):
        q.PROFILE["measured_covariance"] = True
