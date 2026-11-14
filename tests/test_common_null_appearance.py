"""Private raw columns, scalar kernels and probability grids; no cases or NN."""
import gc
import math
import weakref
import numpy as np
import pytest
from monocular_osm import common_null_appearance as c


def data():
    refs = np.array([[100., 100.], [100., 100.], [500., 400.]], np.float64)
    raw = np.tile(np.array([[.2, .6, -.4], [-.1, .1, .4]], np.float32), (40, 1))
    weights = np.array([.5, .5, 1.], np.float64)
    times = np.linspace(0., 600., 80, dtype=np.float64)
    return refs, raw, weights, times


def build(args=None, **kwargs):
    return c.build_common_null_field(*(data() if args is None else args),
                                    origin_xy=(0., 0.), shape=(7, 8), **kwargs)


def scalar_gaussian(grid):
    radius, sigma = 5, 1.5
    kernel = [math.exp(-.5*(x/sigma)**2) for x in range(-radius, radius+1)]
    total = math.fsum(kernel); kernel = [x/total for x in kernel]
    result = grid.copy()
    for axis in (0, 1):
        out = np.zeros_like(result)
        for i, j in np.ndindex(result.shape):
            terms = []
            for delta, weight in zip(range(-radius, radius+1), kernel):
                a, b = (i+delta, j) if axis == 0 else (i, j+delta)
                if 0 <= a < result.shape[0] and 0 <= b < result.shape[1]:
                    terms.append(weight*result[a, b])
            out[i, j] = math.fsum(terms)
        result = out
    return result


def oracle(refs, scores, weights, shape):
    indices = [(round(x/100), round(y/100)) for x, y in refs]
    mass = math.fsum(weights)
    mean = math.fsum(float(w)*float(v) for w, v in zip(weights, scores))/mass
    std = math.sqrt(math.fsum(float(w)*(float(v)-mean)**2 for w, v in zip(weights, scores))/mass)
    density, numerator = np.zeros(shape), np.zeros(shape)
    for (x, y), value, weight in zip(indices, scores, weights):
        z = min(max((float(value)-mean)/max(std, .005), -6.), 6.)
        density[y, x] += weight; numerator[y, x] += weight*math.exp(z)
    d, n = scalar_gaussian(density), scalar_gaussian(numerator)
    result = np.log(.1/d.size+.9*n/math.fsum(n.ravel())) - np.log(.1/d.size+.9*d/math.fsum(d.ravel()))
    return result, float(d.sum()), float(n.sum())


@pytest.mark.parametrize("scale", [1., .125, 7., 1e-100, 1e100])
def test_proportional_mass_grids_are_neutral_including_zero_support(scale):
    d = np.array([[1., 2., 0.], [3., 0., 4.]])
    n = d*scale; old = (n.copy(), d.copy())
    n.flags.writeable = False; d.flags.writeable = False
    result = c.common_null_log_ratio(n, d)
    np.testing.assert_allclose(result, 0., rtol=0, atol=2e-15)
    np.testing.assert_array_equal(result[d == 0], np.zeros(2))
    for actual, expected in zip((n, d), old): np.testing.assert_array_equal(actual, expected)


def test_density_ratio_bounds_and_discrete_quadrature_are_exact():
    n, d = np.zeros((2, 3)), np.zeros((2, 3)); n[0, 0] = d[1, 2] = 1.
    ratio = np.exp(c.common_null_log_ratio(n, d)); bound = 1+9*n.size
    np.testing.assert_allclose([ratio.max(), ratio.min()], [bound, 1/bound], rtol=2e-15)
    baseline = .1/n.size + .9*d/d.sum()
    assert np.sum(baseline*ratio) == pytest.approx(1., abs=8*np.finfo(np.float64).eps)
    np.testing.assert_allclose(c.common_null_log_ratio(n*1e100, d*1e-100),
                               c.common_null_log_ratio(n, d), rtol=0, atol=1e-15)


def test_builder_matches_independent_constant_boundary_kernel_and_grid_sums():
    refs, raw, weights, times = data()
    field = build((refs, raw, weights, times))
    assert field.values.dtype == np.float32 and field.values.shape == (80, 7, 8)
    assert field.outside_value == 0. and field.cell_m == 100.
    for q in (0, 1, 79):
        expected, dsum, nsum = oracle(refs, raw[q], weights, (7, 8))
        np.testing.assert_allclose(field.values[q], expected, rtol=2e-6, atol=3e-7)
        assert field.provenance["actual_gaussian_density_sum"] == pytest.approx(dsum, rel=2e-15)
        assert field.provenance["actual_gaussian_numerator_sums"][q] == pytest.approx(nsum, rel=2e-15)
        assert dsum < weights.sum()  # normalize actual finite-grid Gaussian mass, not raw weights
    assert field.provenance["profile_sha256"] == c.PROFILE_SHA256
    assert field.provenance["imagelikelihood_calibrated"] is False


def test_flat_query_is_neutral_on_entire_grid_and_outside_with_readonly_inputs():
    refs, raw, weights, times = data(); raw.fill(.25)
    before = [a.copy() for a in (refs, raw, weights, times)]
    for a in (refs, raw, weights, times): a.flags.writeable = False
    field = c.build_common_null_field(refs, raw, weights, times, origin_xy=(0., 0.), shape=(30, 40))
    np.testing.assert_array_equal(field.values, np.zeros_like(field.values))
    positions = np.array([[[0., 0.]], [[1e9, 1e9]]])
    np.testing.assert_array_equal(field.sample(positions, times[:1]), np.zeros((2, 1)))
    for actual, expected in zip((refs, raw, weights, times), before): np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(field.query_times, times)


def test_hot_provider_evidence_weakens_to_neutral_in_tails():
    refs = np.array([[1000., 1000.], [4000., 1000.]], np.float64)
    raw = np.tile(np.array([.7, -.1], np.float32), (80, 1))
    times = np.linspace(0., 600., 80)
    field = c.build_common_null_field(refs, raw, np.ones(2), times, origin_xy=(0., 0.), shape=(21, 51))
    peak, near, corner, outside = (float(field.values[0, y, x]) for y, x in
                                  ((10, 10), (10, 5), (5, 5), (4, 4)))
    assert peak > near > corner > 0 and outside == 0.
    # Pure mass-grid limit independently verifies neutral evidence, not a hard floor.
    tails = [float(c.common_null_log_ratio(np.array([t, 1.]), np.array([2*t, 1.]))[0])
             for t in (1e-3, 1e-6, 1e-9)]
    assert abs(tails[0]) > abs(tails[1]) > abs(tails[2])


def test_split_duplicate_weight_and_column_permutation_preserve_field():
    refs = np.array([[100., 100.], [500., 400.]], np.float64)
    raw = np.tile(np.array([.6, -.3], np.float32), (80, 1)); times = np.linspace(0., 600., 80)
    original = build((refs, raw, np.ones(2), times))
    split = build((refs[[0, 0, 1]], raw[:, [0, 0, 1]].copy(), np.array([.5, .5, 1.]), times))
    np.testing.assert_allclose(split.values, original.values, rtol=0, atol=2e-7)
    scaled = build((refs, raw, np.full(2, .125), times))
    np.testing.assert_allclose(scaled.values, original.values, rtol=0, atol=2e-7)
    permuted = build((refs[::-1].copy(), raw[:, ::-1].copy(), np.ones(2), times))
    np.testing.assert_allclose(permuted.values, original.values, rtol=0, atol=2e-7)


def test_genuine_private_snapshot_columns_and_all_one_rows_use_new_F64_math(monkeypatch):
    from tests.test_reference_snapshot_junction_probe_v2 import case
    from monocular_osm.raw_vpr_appearance import full_extent
    net, times, _, _, _, evidence, _ = case(versions=True)
    refs = net.local_to_crs(net.latlon_to_local(evidence["ref_latlon"]))
    origin, shape, _ = full_extent(net, refs)
    originals = [x.copy() for x in (evidence["raw_similarities"], evidence["snapshot_weights"])]
    monkeypatch.setattr(c.poi, "build_appearance_field", lambda *a, **k: pytest.fail("oldF32 builder called"))
    field = c.build_common_null_field(refs, evidence["raw_similarities"], evidence["snapshot_weights"],
                                    evidence["query_times"], origin_xy=origin, shape=shape)
    assert field.values.shape[0] == 80 and np.isfinite(field.values).all()
    for actual, expected in zip((evidence["raw_similarities"], evidence["snapshot_weights"]), originals):
        np.testing.assert_array_equal(actual, expected)
    args = list(data()); args[2] = np.ones(3)
    allone = build(tuple(args))
    assert allone.provenance["all_one_weights_also_F64"] is True


def test_no_F64_cube_and_no_input_retention(monkeypatch):
    real = np.empty
    def bounded(shape, dtype=float, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 3:
            assert np.dtype(dtype) == np.dtype("float32")
        return real(shape, dtype, *args, **kwargs)
    monkeypatch.setattr(c.np, "empty", bounded)
    def owned():
        inputs = data(); refs = [weakref.ref(a) for a in inputs]
        field = build(inputs)
        return field, refs
    field, references = owned(); gc.collect()
    assert field.values.dtype == np.float32 and all(r() is None for r in references)


@pytest.mark.parametrize("bad", ["negative", "nan", "zero", "shape", "complex", "overcap"])
def test_invalid_mass_grids_refuse(bad):
    n, d = np.ones((2, 3)), np.ones((2, 3))
    if bad == "negative": n[0, 0] = -1
    if bad == "nan": n[0, 0] = np.nan
    if bad == "zero": n[:] = 0
    if bad == "shape": d = np.ones((2, 2))
    if bad == "complex": n = n.astype(complex)
    if bad == "overcap": n = d = np.zeros(200_001)
    with pytest.raises(ValueError): c.common_null_log_ratio(n, d)


@pytest.mark.parametrize("fault", ["rows", "slots", "score_type", "score_nan_late", "score_range",
    "weights_type", "weights_zero", "weights_inf", "coordinates", "times", "time_type",
    "shape_cap", "shape_bool", "shape_float", "origin", "outside", "fortran"])
def test_all_input_and_extent_caps_refuse_before_gaussian(fault, monkeypatch):
    refs, raw, weights, times = data(); shape, origin = (7, 8), (0., 0.)
    if fault == "rows": raw = raw[:79]
    if fault == "slots": raw = np.zeros((80, 96_001), np.float32)
    if fault == "score_type": raw = raw.astype(np.float64)
    if fault == "score_nan_late": raw[-1, -1] = np.nan
    if fault == "score_range": raw[-1, -1] = 1.1
    if fault == "weights_type": weights = weights.astype(np.float32)
    if fault == "weights_zero": weights[0] = 0
    if fault == "weights_inf": weights[0] = np.inf
    if fault == "coordinates": refs = refs.astype(np.float32)
    if fault == "times": times[-1] = times[-2]
    if fault == "time_type": times = times.astype(np.float32)
    if fault == "shape_cap": shape = (401, 500)
    if fault == "shape_bool": shape = (True, 4)
    if fault == "shape_float": shape = (7., 8)
    if fault == "origin": origin = (np.inf, 0.)
    if fault == "outside": refs[0] = [10000., 10000.]
    if fault == "fortran": raw = np.asfortranarray(raw)
    monkeypatch.setattr(c, "gaussian_filter", lambda *a, **k: pytest.fail("Gaussian before refusal"))
    with pytest.raises(ValueError):
        c.build_common_null_field(refs, raw, weights, times, origin_xy=origin, shape=shape)


def test_callback_failure_returns_no_partial_field_and_no_input_mutation():
    args = data(); originals = [a.copy() for a in args]; count = []
    def check():
        count.append(None)
        if len(count) == 90: raise TimeoutError("private bounded construction")
    with pytest.raises(TimeoutError, match="private bounded"):
        build(args, check=check)
    for actual, expected in zip(args, originals): np.testing.assert_array_equal(actual, expected)
    assert len(count) == 90


def test_profile_is_distinct_immutable_and_direct_log_sampling_only():
    assert c.PROFILE["new_target"] is True and not c.PROFILE["imagelikelihood_calibrated"]
    assert c.PROFILE["outside_log_factor"] == 0.
    assert c.PROFILE_SHA256 == c.poi._json_digest(dict(c.PROFILE))
    with pytest.raises(TypeError): c.PROFILE["row_std_floor"] = 1.
