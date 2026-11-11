"""Private rasters/clocks only; no actual cases, IO, models or GT."""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
from monocular_osm import snapshot_future_guide as g


def appearance(values=None, times=None, clock=None):
    times = np.arange(80, dtype=np.float64) * 4. if times is None else times
    clock = np.arange(0., times[-1] + .5, .5) if clock is None else clock
    if values is None:
        values = np.zeros((80, 3, 5), np.float32)
        values[-1] = np.array([[-6., -4., -2., 0., 3.]] * 3, np.float32)
    field = g.strict.poi.RasterField(values, (1_000_000., 2_000_000.), 100., -6., times,
        {"appearance_profile_sha256": g.snapshot.PROFILE_SHA256})
    groups, powers = g.strict.temporal_groups(times)
    steps = np.searchsorted(clock, times, side="left").astype(np.int64)
    maxima = values.max(axis=(1, 2)).astype(np.float64)
    receipt = {"profile": dict(g.snapshot.PROFILE), "profile_sha256": g.snapshot.PROFILE_SHA256,
        "all_queries_once": True, "historical_encoder_verified": False, "crs": "EPSG:3857",
        "field": field.source_record(), "query_times": times.tolist(), "group_ids": groups.tolist(),
        "group_powers": powers.tolist(), "scheduled_steps": steps.tolist(),
        "row_maxima_over_complete_grid": maxima.tolist()}
    return g.strict.Appearance(field, maxima, groups, powers, steps, receipt), clock


def net():
    # A source-private geometry stub exercises the genuine RasterField sampler.
    return SimpleNamespace(crs="EPSG:3857", n_edges=1, edge_len=np.array([1e8]),
        locate=lambda e, s: (np.asarray(s), np.full(np.shape(s), 100.), np.zeros(np.shape(s)), None),
        local_to_crs=lambda xy: xy + np.array([1_000_000., 2_000_000.]))


def gaussian_scalar(values, sigma):
    radius = int(3. * sigma + .5)
    kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= sum(kernel)
    out = values.astype(np.float64)
    for axis in (0, 1):
        result = np.empty_like(out)
        for index in np.ndindex(out.shape):
            result[index] = sum(float(w) * out[tuple(
                min(max(index[k] + (delta if k == axis else 0), 0), out.shape[k] - 1)
                for k in (0, 1))] for delta, w in zip(range(-radius, radius + 1), kernel))
        out = result
    return out


def factor(a, q):
    return (.1 + .9 * np.exp(a.field.values[q].astype(np.float64) - a.maxima[q])) ** a.powers[q]


def test_backward_recursion_matches_independent_nearest_scalar_oracle_and_terminal():
    a, clock = appearance()
    guide = g.build(a, clock)
    expected = np.ones(a.field.values.shape, np.float32)
    for q in range(78, -1, -1):
        row = gaussian_scalar(factor(a, q + 1) * expected[q + 1], .25 * (a.field.query_times[q + 1] - a.field.query_times[q]))
        expected[q] = row / row.max()
    np.testing.assert_allclose(guide.backward_H, expected, rtol=0, atol=6e-8)
    np.testing.assert_array_equal(guide.backward_H[79], np.ones((3, 5)))
    assert guide.backward_H.dtype == np.float32 and not guide.backward_H.flags.writeable
    assert guide.receipt["target_changed"] is False
    assert guide.receipt["profile_sha256"] == g.PROFILE_SHA256


def test_active_native_remaining_clock_full_crs_bilinear_and_outside_floor():
    a, clock = appearance()
    guide = g.build(a, clock)
    index = 621  # t310.5, next query q78 at native t312
    q = int(np.searchsorted(a.steps, index, side="right"))
    raw = factor(a, q) * guide.backward_H[q]
    wanted = gaussian_scalar(raw, .25 * (a.field.query_times[q] - clock[index]))
    wanted = np.maximum(wanted / wanted.max(), 1e-6).astype(np.float32)
    offsets = np.array([0., 50., 100., 400., 500.])
    actual = guide.sample(index, net(), np.zeros(5, np.int64), offsets)
    raster = g.strict.poi.RasterField(wanted, a.field.origin_xy, 100., 1e-6)
    oracle = raster.sample(net().local_to_crs(np.column_stack([offsets, np.full(5, 100.)])), np.empty(0))
    np.testing.assert_array_equal(actual, oracle)
    assert actual[-1] == 1e-6 and actual[1] == (actual[0] + actual[2]) / 2
    assert guide._next_query == q and guide.gaussian_calls == 1


def test_same_endpoint_all_rows_skipped_and_each_index_cached_for_fresh_positions(monkeypatch):
    times = np.arange(80, dtype=np.float64) * .1
    a, clock = appearance(times=times)
    guide = g.build(a, clock)
    calls = []
    real = g.gaussian_filter
    def record(*args, **kwargs):
        calls.append(kwargs["sigma"])
        return real(*args, **kwargs)
    monkeypatch.setattr(g, "gaussian_filter", record)
    first = guide.sample(1, net(), np.zeros((2, 3), np.int64), np.array([0., 100., 200.]))
    second = guide.sample(1, net(), np.array([0], np.int64), np.array([300.]))
    assert first.shape == (2, 3) and second.shape == (1,) and len(calls) == 1
    assert guide._next_query == 6 and np.all(a.steps[:6] <= 1) and a.steps[6] > 1
    guide.sample(2, net(), np.array([0], np.int64), np.array([0.]))
    assert guide._next_query == 11 and len(calls) == 2
    with pytest.raises(ValueError, match="monotone"):
        guide.sample(1, net(), np.array([0], np.int64), np.array([0.]))


def test_constant_fields_and_no_future_are_exactly_neutral_including_outside(monkeypatch):
    values = np.repeat(np.linspace(-6., 6., 80, dtype=np.float32)[:, None, None], 15, axis=1).reshape(80, 3, 5)
    a, clock = appearance(values)
    guide = g.build(a, clock)
    np.testing.assert_array_equal(guide.backward_H, np.ones_like(values))
    np.testing.assert_array_equal(guide.sample(0, net(), np.array([0, 0]), np.array([0., 400.])), [1., 1.])
    monkeypatch.setattr(g, "gaussian_filter", lambda *a, **k: pytest.fail("Gaussian without a future query"))
    np.testing.assert_array_equal(guide.sample(len(clock) - 1, net(), np.array([0, 0]), np.array([0., 1e7])), [1., 1.])


def test_q0_not_in_initial_future_guide_and_original_first_factor_is_untouched():
    a, clock = appearance()
    changed = a.field.values.copy()
    changed[0] = np.arange(15, dtype=np.float32).reshape(3, 5)
    b, _ = appearance(changed)
    left, right = g.build(a, clock), g.build(b, clock)
    offsets = np.array([0., 200., 400.])
    np.testing.assert_array_equal(left.sample(0, net(), np.zeros(3, np.int64), offsets),
                                  right.sample(0, net(), np.zeros(3, np.int64), offsets))
    assert not np.array_equal(a.initial_weight(net(), np.zeros(3, np.int64), offsets),
                             b.initial_weight(net(), np.zeros(3, np.int64), offsets))


def test_original80_fields_factors_powers_clocks_and_rng_immutable():
    a, clock = appearance()
    originals = [x.copy() for x in (a.field.values, a.field.query_times, a.maxima,
                                     a.powers, a.groups, a.steps, clock)]
    state = np.random.get_state()
    guide = g.build(a, clock)
    guide.sample(0, net(), np.array([0]), np.array([200.]))
    guide.sample(1, net(), np.array([0]), np.array([200.]))
    for old, actual in zip(originals, (a.field.values, a.field.query_times, a.maxima,
                                      a.powers, a.groups, a.steps, clock)):
        np.testing.assert_array_equal(old, actual)
    now = np.random.get_state()
    assert state[0] == now[0] and state[2:] == now[2:]
    np.testing.assert_array_equal(state[1], now[1])
    for group in np.unique(a.groups):
        assert sum(a.powers[a.groups == group]) == pytest.approx(1.)
    assert g.PROFILE_SHA256 == g.strict.poi._json_digest(dict(g.PROFILE))
    with pytest.raises(TypeError):
        g.PROFILE["active_floor"] = .1


def test_original_group_power_is_applied_after_robust_mixture_before_diffusion():
    a, clock = appearance()
    row = np.empty((3, 5), np.float64)
    g._factor(a, 79, row)
    np.testing.assert_array_equal(row, factor(a, 79))
    delta = a.field.values[79].astype(np.float64) - a.maxima[79]
    wrong = .1 + .9 * np.exp(delta * a.powers[79])
    assert np.max(abs(row - wrong)) > .01
    p = a.powers[a.groups == a.groups[79]]
    np.testing.assert_allclose(np.prod(np.repeat(row[None], len(p), axis=0), axis=0),
                               .1 + .9 * np.exp(delta), rtol=1e-14, atol=0)


def test_auxiliary_defense_exact_mixture_correction_and_four_bound():
    w = np.array([.7, .2, .1, 0.], np.float64)
    raw = np.array([1e-12, .01, 1., .3], np.float64)
    e = g.defensive_auxiliary_guide(w, raw)
    q = w * e
    expected = .25 * w + .75 * w * raw / (w @ raw)
    np.testing.assert_allclose(q, expected, rtol=1e-15, atol=0)
    np.testing.assert_allclose(q / e, w, rtol=1e-15, atol=0)
    assert q.sum() == pytest.approx(1., abs=2e-16) and np.max(1. / e) <= 4.
    np.testing.assert_allclose(g.defensive_auxiliary_guide(w, np.ones(4)), np.ones(4), rtol=0, atol=3e-16)
    np.testing.assert_allclose(g.defensive_auxiliary_guide(w, raw * 1e-90), e, rtol=1e-15)


@pytest.mark.parametrize("fault", ["sum", "negative", "nan", "float32", "zero_g", "inf_g", "shape", "count"])
def test_auxiliary_weights_and_guides_refuse_invalid_inputs(fault):
    w, raw = np.array([.5, .5]), np.array([.2, 1.])
    if fault == "sum": w *= 2
    if fault == "negative": w[:] = [-.5, 1.5]
    if fault == "nan": w[0] = np.nan
    if fault == "float32": w = w.astype(np.float32)
    if fault == "zero_g": raw[0] = 0.
    if fault == "inf_g": raw[0] = np.inf
    if fault == "shape": raw = raw[:1]
    if fault == "count": w, raw = np.full(50001, 1 / 50001), np.ones(50001)
    with pytest.raises(ValueError):
        g.defensive_auxiliary_guide(w, raw)


def test_enumerated_late_evidence_initial_and_turn_corrections_recover_same_target():
    values = np.zeros((80, 3, 60), np.float32)
    values[-1] = np.linspace(-6., 3., 60)
    a, clock = appearance(values)
    guide = g.build(a, clock)
    early = guide.sample(0, net(), np.zeros(3, np.int64), np.array([0., 3000., 5900.]))
    assert np.ptp(early) > 0  # q1 is flat; this information came backwards from late q79
    prior = np.array([.2, .5, .3])
    transition = np.array([[.6, .3, .1], [.2, .6, .2], [.1, .3, .6]])
    late = np.array([.1, .2, 1.])
    initial_q = .25 * prior + .75 * prior * early / (prior @ early)
    oracle, corrected, uncorrected = np.zeros(3), np.zeros(3), np.zeros(3)
    for i in range(3):
        # Genuine turn proposal shape and defense; no extra defense on raw guide.
        turn_q = .25 * transition[i] + .75 * transition[i] * early / (transition[i] @ early)
        for j in range(3):
            oracle[j] += prior[i] * transition[i, j] * late[j]
            proposal = initial_q[i] * turn_q[j]
            correction = prior[i] / initial_q[i] * transition[i, j] / turn_q[j]
            corrected[j] += proposal * correction * late[j]
            uncorrected[j] += proposal * late[j]
    np.testing.assert_allclose(corrected, oracle, rtol=2e-15, atol=0)
    assert np.max(abs(uncorrected - oracle)) > 1e-7


def test_delayed_q0_always_excluded_from_future_proposal_but_not_target():
    times = 1. + np.arange(80, dtype=np.float64) * .5
    clock = np.arange(0., times[-1] + .5, .5)
    a, _ = appearance(times=times, clock=clock)
    guide = g.build(a, clock)
    guide.sample(1, net(), np.array([0]), np.array([0.]))
    assert a.steps[0] > 1 and guide._next_query == 1 and g.PROFILE["eligible_start_query"] == 1
    assert a.steps[0] == 2 and a.field.query_times[0] == 1.  # original target schedule retained


@pytest.mark.parametrize("which", ["backward", "active"])
def test_all_diffusion_widths_preflight_before_any_volume_or_gaussian_allocation(monkeypatch, which):
    if which == "backward":
        times = np.arange(80, dtype=np.float64) * 33.
        a, clock = appearance(times=times, clock=times.copy())
    else:
        times = 32. + np.arange(80, dtype=np.float64) * .5
        a, clock = appearance(times=times)
    real_empty = np.empty
    def bounded_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) >= 2:
            pytest.fail("grid/volume allocation before all-width preflight")
        return real_empty(shape, *args, **kwargs)
    monkeypatch.setattr(g.np, "empty", bounded_empty)
    monkeypatch.setattr(g, "gaussian_filter", lambda *a, **k: pytest.fail("Gaussian before width refusal"))
    with pytest.raises(ValueError, match="8-cell"):
        g.build(a, clock)


def test_sigma8_boundary_accepted_without_clipping(monkeypatch):
    times = np.arange(80, dtype=np.float64) * 32.
    a, clock = appearance(times=times, clock=times.copy())
    real, widths = g.gaussian_filter, []
    def record(*args, **kwargs):
        widths.append(kwargs["sigma"])
        return real(*args, **kwargs)
    monkeypatch.setattr(g, "gaussian_filter", record)
    guide = g.build(a, clock)
    guide.sample(0, net(), np.array([0]), np.array([0.]))
    assert widths == [8.] * 80


@pytest.mark.parametrize("fault", ["profile", "old_profile", "field_nan", "field_dtype", "field_fortran",
    "field_cap", "clock_dtype", "clock_duplicate", "clock_count", "times", "steps", "powers", "maxima", "receipt"])
def test_complete_field_original_identity_and_clock_fail_closed(fault, monkeypatch):
    a, clock = appearance()
    if fault == "profile": a.receipt["profile_sha256"] = "a" * 64
    if fault == "old_profile": a.receipt["profile"] = dict(g.strict.PROFILE)
    if fault == "field_nan": a.field.values[-1, 0, 0] = np.nan
    if fault == "field_dtype": object.__setattr__(a.field, "values", a.field.values.astype(np.float64))
    if fault == "field_fortran": object.__setattr__(a.field, "values", np.asfortranarray(a.field.values))
    if fault == "field_cap": object.__setattr__(a.field, "values", np.broadcast_to(np.zeros((), np.float32), (80, 501, 400)))
    if fault == "clock_dtype": clock = clock.astype(np.float32)
    if fault == "clock_duplicate": clock[2] = clock[1]
    if fault == "clock_count": clock = np.linspace(0., 316., 1202)
    if fault == "times": a.field.query_times[2] = a.field.query_times[1]
    if fault == "steps": a.steps[2] += 1
    if fault == "powers": a.powers[2] *= 2
    if fault == "maxima": a.maxima[-1] += .01
    if fault == "receipt": a.receipt["field"]["values_sha256"] = "b" * 64
    monkeypatch.setattr(g, "gaussian_filter", lambda *a, **k: pytest.fail("Gaussian before identity refusal"))
    with pytest.raises(ValueError):
        g.build(a, clock)


@pytest.mark.parametrize("fault", ["index_bool", "index_negative", "index_end", "positions", "wrong_crs", "bad_edge", "bad_offset"])
def test_sample_bounds_and_types(fault):
    a, clock = appearance()
    guide, network = g.build(a, clock), net()
    index, edges, offsets = 0, np.array([0]), np.array([0.])
    if fault == "index_bool": index = True
    if fault == "index_negative": index = -1
    if fault == "index_end": index = len(clock)
    if fault == "positions": edges, offsets = np.zeros(200001, np.int64), np.zeros(200001)
    if fault == "wrong_crs": network.crs = "EPSG:4326"
    if fault == "bad_edge": edges[0] = 1
    if fault == "bad_offset": offsets[0] = np.nan
    with pytest.raises(ValueError):
        guide.sample(index, network, edges, offsets)


def test_genuine_snapshot_field_road_crs_and_200k_initial_candidates():
    from tests.test_reference_snapshot_junction_probe import case
    network, native_times, _, _, motion, evidence, facts = case(versions=True)
    a = g.snapshot.prepare_snapshot(evidence, network, facts, native_times, motion.times)
    guide = g.build(a, motion.times)
    count = 200_000
    edges = np.zeros(count, np.int64)
    offsets = np.full(count, 100.)
    actual = guide.sample(0, network, edges, offsets)
    x, y, _, _ = network.locate(edges[:1], offsets[:1])
    absolute = network.local_to_crs(np.column_stack([x, y]))
    expected = guide._raster.sample(absolute, np.empty(0))[0]
    np.testing.assert_array_equal(actual, np.full(count, expected))
    assert guide.gaussian_calls == 1 and np.linalg.norm(network.origin) > 1_000_000


def test_budget_check_can_stop_backward_work_without_returning_partial_guide():
    a, clock = appearance()
    calls = []
    def check():
        calls.append(None)
        if len(calls) == 8:
            raise TimeoutError("private cooperative budget")
    with pytest.raises(TimeoutError, match="cooperative"):
        g.build(a, clock, check=check)
    assert len(calls) == 8
