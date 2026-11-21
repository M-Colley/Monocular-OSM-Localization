"""Pure synthetic raw scores; no case/model/native evidence admission."""
import hashlib
import json

import numpy as np
import pytest

from monocular_osm import raw_query_veto as v


def test_original_float32_maxima_and_absolute_gate_scalar_oracle():
    raw = np.linspace(-.3, .4, 80 * 7, dtype=np.float32).reshape(80, 7)
    enabled, maxima, receipt = v.build_gate(raw)
    expected = np.array([max(float(x) for x in row) for row in raw])
    np.testing.assert_array_equal(maxima, expected)
    np.testing.assert_array_equal(enabled, [x >= .162 for x in expected])
    assert maxima.dtype == np.float64 and enabled.dtype == np.bool_
    assert receipt["enabled_count"] + receipt["disabled_count"] == 80
    assert receipt["snapshot_count"] == 7 and receipt["query_count"] == 80


def test_cutoff_is_not_silently_rounded_to_float32():
    near = np.float32(.162)
    assert float(near) > .162
    below = np.nextafter(near, np.float32(-np.inf))
    raw = np.full((80, 1), below, dtype=np.float32)
    raw[1, 0] = near
    enabled, maxima, _ = v.build_gate(raw)
    assert not enabled[0] and enabled[1]
    assert maxima[0] < .162 < maxima[1]


def test_absolute_shift_changes_gate_despite_same_relative_ranking():
    low = np.tile(np.array([.05, .06, .07, .08], dtype=np.float32), (80, 1))
    high = low + np.float32(.2)
    assert not v.build_gate(low)[0].any()
    assert v.build_gate(high)[0].all()
    np.testing.assert_array_equal(np.argsort(low, axis=1), np.argsort(high, axis=1))


def test_alias_and_column_permutation_invariance_no_version_selection():
    raw = np.tile(np.array([.12, .22, -.1, .2], dtype=np.float32), (80, 1))
    original = v.build_gate(raw)
    actual = v.build_gate(np.ascontiguousarray(raw[:, [3, 1, 0, 1, 2, 1]]))
    np.testing.assert_array_equal(actual[0], original[0])
    np.testing.assert_array_equal(actual[1], original[1])
    assert actual[2]["snapshot_count"] == 6


def test_readonly_input_rng_preserved_and_outputs_independent():
    raw = np.full((80, 2), .2, dtype=np.float32)
    raw.setflags(write=False)
    before = raw.tobytes(), raw.strides, raw.flags.writeable
    state = np.random.get_state()
    enabled, maxima, _ = v.build_gate(raw)
    enabled[:] = False
    maxima[:] = 0.
    assert (raw.tobytes(), raw.strides, raw.flags.writeable) == before
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize("position", [(0, 0), (37, 2), (79, 3)])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_any_nonfinite_refuses_whole_gate(position, value):
    raw = np.full((80, 4), .2, dtype=np.float32)
    raw[position] = value
    with pytest.raises(ValueError, match="finite"):
        v.build_gate(raw)


@pytest.mark.parametrize("raw", [
    [[.2]], np.zeros((80, 1), dtype=np.float64), np.zeros((80, 1), dtype=np.int32),
    np.zeros((80, 1), dtype=np.bool_), np.zeros((79, 1), dtype=np.float32),
    np.zeros((81, 1), dtype=np.float32), np.zeros((80, 0), dtype=np.float32),
    np.zeros((80, 1, 1), dtype=np.float32), np.zeros((80,), dtype=np.float32),
    np.zeros((80, 2), dtype=np.float32)[:, ::2],
    np.zeros((80, 2), dtype=np.float32, order="F"),
    np.zeros((80, 1), dtype=np.dtype(">f4")),
])
def test_abi_refused_before_numeric_work(raw, monkeypatch):
    monkeypatch.setattr(v.np, "isfinite", lambda _: pytest.fail("invalid ABI read"))
    with pytest.raises(ValueError, match="contiguous"):
        v.build_gate(raw)


def test_cap_refused_before_numerical_scratch(monkeypatch):
    raw = np.empty((80, 96_001), dtype=np.float32)
    monkeypatch.setattr(v.np, "empty", lambda *a, **kw: pytest.fail("oversize scratch"))
    with pytest.raises(ValueError, match="96000"):
        v.build_gate(raw)


def test_full_cap_row_bounded_validation_and_original_values(monkeypatch):
    raw = np.full((80, 96_000), -.2, dtype=np.float32)
    raw[42, -1] = .3
    real = np.isfinite
    shapes = []
    def finite(row):
        shapes.append(row.shape)
        return real(row)
    monkeypatch.setattr(v.np, "isfinite", finite)
    enabled, maxima, _ = v.build_gate(raw)
    assert np.flatnonzero(enabled).tolist() == [42]
    assert len(shapes) == 80 and set(shapes) == {(96_000,)}
    assert maxima[42] == float(np.float32(.3)) and raw[0, 0] == np.float32(-.2)


def test_finite_scores_not_clipped_or_normalized():
    raw = np.full((80, 1), -100., dtype=np.float32)
    raw[79, 0] = 100.
    enabled, maxima, _ = v.build_gate(raw)
    assert maxima[0] == -100. and maxima[79] == 100.
    assert np.flatnonzero(enabled).tolist() == [79]


def test_immutable_profile_exact_policy_origin_false_confidence_claims():
    from monocular_osm import assisted_policy
    assert type(assisted_policy._CONFIG["baseline_median_min"]) is float
    assert assisted_policy._CONFIG["baseline_median_min"] == v.THRESHOLD == .162
    digest = hashlib.sha256(json.dumps(dict(v.PROFILE), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert v.PROFILE_SHA256 == digest
    assert v.PROFILE["target_changed"] and not v.PROFILE["calibrated"]
    assert not v.PROFILE["query_inlier_probability"]
    assert not v.PROFILE["historical_reference_encoding_upgraded"]
    with pytest.raises(TypeError):
        v.PROFILE["threshold"] = .1
