from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from monocular_osm.spatial_sequence import (GaussianTransition, SpatialSequenceConfig,
    array_sha, decode_spatial_tracks, infer_spatial_posterior, json_sha)


def dense_transition(shape, cell, dt, config):
    """Independent direct formula, without convolution or the implementation."""
    points = np.array(list(np.ndindex(shape)))
    delta = points[None, :, :] - points[:, None, :]
    sigma = config.motion_three_sigma_speed_mps * dt / (3 * cell)
    radius = np.minimum(np.array(shape)-1, math.ceil(config.motion_truncate_sigma * sigma))
    kernel = np.exp(-np.sum(delta.astype(float)**2, axis=2) / (2*sigma**2))
    kernel[np.any(np.abs(delta) > radius, axis=2)] = 0
    kernel /= kernel.sum(axis=1, keepdims=True)
    recovery = -np.expm1(-dt/config.recovery_time_s)
    return (1-recovery)*kernel + recovery/len(points)


def brute_posterior(field, times, cell, config):
    emissions = np.exp(field.reshape(len(field), -1)*config.emission_log_scale)
    transitions = [dense_transition(field.shape[1:], cell, dt, config) for dt in np.diff(times)]
    result = np.zeros_like(emissions)
    total = 0.
    for states in itertools.product(range(emissions.shape[1]), repeat=len(field)):
        weight = emissions[0, states[0]]/emissions.shape[1]
        for q in range(1, len(field)):
            weight *= transitions[q-1][states[q-1], states[q]]*emissions[q, states[q]]
        total += weight
        for q, state in enumerate(states):
            result[q, state] += weight
    return (result/total).reshape(field.shape), math.log(total)


def test_forward_backward_matches_complete_path_enumeration_at_boundaries():
    config = replace(SpatialSequenceConfig(), motion_three_sigma_speed_mps=3., recovery_time_s=4.)
    field = np.array([[[2., 0., -1.], [0., .2, -.2]], [[0., 1., 2.], [.1, -.3, .8]],
                      [[-.2, .5, 0.], [2., 0., 1.]]])
    times = np.array([1., 1.4, 3.1])
    expected, evidence = brute_posterior(field, times, 1., config)
    actual = infer_spatial_posterior(field, times, 1., config=config)
    np.testing.assert_allclose(actual["posterior"], expected, atol=3e-8)
    assert actual["log_evidence"] == pytest.approx(evidence, abs=1e-12)
    assert actual["maximum_transition_mass_error"] < 1e-14
    assert [row["interval_s"] for row in actual["transitions"]] == pytest.approx([.4, 1.7])


@pytest.mark.parametrize("dt", [.001, .4, 3.])
def test_transition_is_stochastic_and_backward_is_adjoint(dt):
    config = replace(SpatialSequenceConfig(), motion_three_sigma_speed_mps=3.)
    shape = (3, 5)
    motion = GaussianTransition(shape, 1., dt, config)
    expected = dense_transition(shape, 1., dt, config)
    np.testing.assert_allclose(expected.sum(axis=1), 1., atol=1e-14)
    for origin in (0, 2, 7, 14):
        mass = np.zeros(shape); mass.flat[origin] = 1.
        np.testing.assert_allclose(motion.forward(mass).ravel(), expected[origin], atol=1e-14)
    rng = np.random.default_rng(3)
    mass, likelihood = rng.random(shape), rng.random(shape)
    assert motion.forward(mass).sum() == pytest.approx(mass.sum(), abs=1e-12)
    assert np.sum(motion.forward(mass)*likelihood) == pytest.approx(np.sum(mass*motion.backward(likelihood)), abs=1e-12)
    np.testing.assert_allclose(motion.backward(np.ones(shape)), 1., atol=1e-14)


def test_boundary_is_not_reflected_or_wrapped_and_source_normalization_matters():
    config = replace(SpatialSequenceConfig(), motion_three_sigma_speed_mps=3., recovery_time_s=1e12)
    motion = GaussianTransition((5, 6), 1., 1., config)
    mass = np.zeros((5, 6)); mass[0, 0] = 1
    result = motion.forward(mass)
    expected = dense_transition(mass.shape, 1., 1., config)[0].reshape(mass.shape)
    np.testing.assert_allclose(result, expected, atol=1e-14)
    assert result[0, 0] > result[0, 1] > result[0, 3]
    assert result[-1, -1] < 1e-10
    assert motion.source_normalizer[0, 0] < motion.source_normalizer[2, 2]


def test_future_evidence_changes_early_posterior_not_only_filter_endpoint():
    config = replace(SpatialSequenceConfig(), motion_three_sigma_speed_mps=1., recovery_time_s=1e8)
    field = np.zeros((3, 2, 3)); field[-1, 0, 2] = 6
    actual = infer_spatial_posterior(field, [0., .1, .2], 1., config=config)
    assert actual["posterior"][0, 0, 2] > .98
    assert actual["route_rc"].tolist() == [[0, 2]]*3


def test_multimodal_primary_is_peak_not_mean_and_complete_mass_is_retained():
    config = replace(SpatialSequenceConfig(), mode_separation_m=200.)
    field = np.full((3, 3, 7), -6.); field[:, 1, 1] = field[:, 1, 5] = 6.
    result = infer_spatial_posterior(field, [0., .001, .002], 100., config=config)
    assert result["route_rc"].tolist() == [[1, 1]]*3
    for q, modes in enumerate(result["modes"]):
        assert {(mode["row"], mode["col"]) for mode in modes[:2]} == {(1, 1), (1, 5)}
        assert result["posterior"][q, 1, 1] > .49
        assert result["posterior"][q, 1, 5] > .49
        assert result["posterior"][q, 1, 3] < 1e-8
    np.testing.assert_allclose(result["posterior"].sum(axis=(1, 2)), 1., atol=1e-7)
    assert result["confidence"] is None and result["unlocalized_probability"] is None


def test_actual_dt_changes_motion_and_recovery_and_stops_are_possible():
    config = SpatialSequenceConfig()
    short = GaussianTransition((9, 9), 100., .1, config)
    long = GaussianTransition((9, 9), 100., 10., config)
    assert long.sigma_cells == pytest.approx(100*short.sigma_cells)
    assert long.recovery > short.recovery
    mass = np.zeros((9, 9)); mass[4, 4] = 1
    assert short.forward(mass)[4, 4] > .99
    assert long.forward(mass)[4, 4] > 0


def case(n=70):
    ids = np.array([f"mapillary/{i:05}" for i in range(n)])
    refs = np.column_stack([np.full(n, 48.), 8.+np.arange(n)*.0001])
    scores = np.random.default_rng(8).uniform(-.1, .3, (4, n)).astype(np.float32)
    return scores, ids, refs, np.array([0., 1., 4., 6.])


def test_all_references_and_queries_both_variants_are_used_without_mutation():
    values = case()
    snapshots = [value.copy() for value in values]
    result = decode_spatial_tracks(*values, provenance={"freeze": "abc"})
    assert result["reference_count"] == 70 and result["query_count"] == 4
    assert result["selected_variant"] is None and set(result["variants"]) == {"raw", "reference_mean"}
    assert result["profile_sha256"] == json_sha(SpatialSequenceConfig().record())
    for variant, output in result["variants"].items():
        assert output["appearance"]["sources"]["reference_count"] == 70
        assert output["appearance"]["sources"]["reference_mean"] == (variant == "reference_mean")
        assert output["posterior"].shape[0] == len(values[-1])
        assert output["posterior_sha256"] == array_sha(output["posterior"])
        np.testing.assert_array_equal(output["route_rc"], np.column_stack(np.divmod(output["posterior"].reshape(4, -1).argmax(axis=1), result["shape"][1])))
        np.testing.assert_array_equal(output["route_timestamps"], values[-1])
    for value, before in zip(values, snapshots):
        np.testing.assert_array_equal(value, before)


def test_gallery_permutation_and_score_memory_layout_do_not_change_posterior():
    scores, ids, refs, times = case()
    permutation = np.random.default_rng(2).permutation(len(ids))
    original = decode_spatial_tracks(scores, ids, refs, times)
    shuffled = decode_spatial_tracks(np.asfortranarray(scores[:, permutation]), ids[permutation], refs[permutation], times)
    for variant in original["variants"]:
        np.testing.assert_array_equal(original["variants"][variant]["posterior"], shuffled["variants"][variant]["posterior"])


def test_input_only_coarsening_respects_cap_and_never_drops_references():
    config = replace(SpatialSequenceConfig(), max_field_values=1600)
    result = decode_spatial_tracks(*case(3), config=config)
    assert result["cell_m"] > 100 and 4*math.prod(result["shape"]) <= 1600
    assert result["reference_count"] == 3
    with pytest.raises(ValueError, match="extent"):
        decode_spatial_tracks(*case(3), config=replace(config, max_cell_m=100.))


@pytest.mark.parametrize("name,value", [("recovery_time_s", 0), ("field_cell_m", float("nan")),
    ("retained_modes", True), ("retained_modes", 33), ("max_field_values", 16000001),
    ("max_queries", 81), ("motion_three_sigma_speed_mps", -1), ("max_kernel_radius", 1.5)])
def test_bad_configs_decline(name, value):
    with pytest.raises(ValueError):
        replace(SpatialSequenceConfig(), **{name: value})


@pytest.mark.parametrize("which", ["nan", "duplicate", "times", "cosine", "object", "latitude"])
def test_bad_raw_inputs_decline(which):
    scores, ids, refs, times = case(3)
    if which == "nan": scores[0, 0] = np.nan
    if which == "duplicate": ids[1] = ids[0]
    if which == "times": times[2] = times[1]
    if which == "cosine": scores[1, 1] = 1.5
    if which == "object": scores = scores.astype(object)
    if which == "latitude": refs[1, 0] = 91
    with pytest.raises(ValueError):
        decode_spatial_tracks(scores, ids, refs, times)


def test_kernel_budget_checked_before_filter_allocation(monkeypatch):
    import scipy.ndimage
    monkeypatch.setattr(scipy.ndimage, "gaussian_filter", lambda *a, **k: pytest.fail("allocated before checking kernel"))
    with pytest.raises(ValueError, match="kernel"):
        GaussianTransition((20, 20), 1e-12, 1., replace(SpatialSequenceConfig(), max_kernel_radius=4))


def test_extreme_mode_radius_is_bounded_by_grid_without_squared_overflow():
    result = infer_spatial_posterior(np.zeros((2, 2, 3)), [0., 1.], 100.,
        config=replace(SpatialSequenceConfig(), mode_separation_m=1e300))
    assert all(len(modes) == 1 for modes in result["modes"])


def test_pure_id_storage_budget_precedes_cast_and_field_allocation(monkeypatch):
    import monocular_osm.spatial_sequence as module
    scores, ids, refs, times = case(3)
    ids = ids.astype("U129")  # 516 bytes per ID, beyond the runner's typed cap
    monkeypatch.setattr(module, "build_appearance_field", lambda *a, **k: pytest.fail("allocated field before ID guard"))
    with pytest.raises(ValueError, match="complete bounded"):
        decode_spatial_tracks(scores, ids, refs, times)


def test_kernel_ceiling_cannot_be_raised_by_configuration():
    with pytest.raises(ValueError, match="allocation"):
        replace(SpatialSequenceConfig(), max_kernel_radius=4097)


@pytest.mark.parametrize("times", [[0, 0], [-1, 1], [0, 4000]])
def test_bad_field_times_decline(times):
    with pytest.raises(ValueError):
        infer_spatial_posterior(np.zeros((2, 2, 2)), times, 100.)
